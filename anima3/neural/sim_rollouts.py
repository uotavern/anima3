"""Seeded simulator rollouts and simulator evaluation games. Needs torch.

Requires the ``neural`` extra and is never imported by ``anima3`` or
``anima3.neural``. There is no command-line entry, no schedule, no training loop
and no promotion; nothing here writes a file.

- :func:`collect_simulator` is ported from ``neural/train.py`` at the training tip
  ``819f4f4`` (``6b8d6f0``, ``94051a2``, ``25a169f``). The loop is unchanged. Local
  rules: the step budget, seed, episode budget and shaping discount have no
  defaults; the run is CPU only and seeded inside ``torch.random.fork_rng``, so the
  caller's global RNG is untouched and a fixed seed gives identical trajectories;
  every trajectory is labelled with the simulator domain and code identity; and
  teacher-labelled trajectories carry ``ppo_eligible: false``, so ``ppo_update``
  refuses them. Every episode seed is below ``EVALUATION_SEED_FLOOR``.
- :func:`_suite_rows` is the game loop of ``neural/league.py`` ``evaluate_suite``
  (``94051a2``, ``25a169f``, ``3dfe973``, ``a972d84``), unchanged except that the
  styles, budget, seed and sampling come from a :class:`SimulatorEvaluationPlan`.
- :func:`evaluate_in_simulator` loads a checkpoint by its SHA-256 and plays the
  plan's games; :func:`decide_in_simulator` evaluates both policies and returns the
  simulator-only gate decision of ``sim_evaluation.simulator_gate``.

Simulator trajectories are :class:`~anima3.neural.train.Trajectory` objects in
memory. They are never live episodes: the live PPO path accepts only a
``TrainingSelection`` of receipt-verified ServUO episodes, and live behaviour
cloning only a verified demonstration selection.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch

from .checkpoints import load_checkpoint
from .drills import DRILLS, SIM_DOMAIN, SIMULATOR_SHA256, setup_drill
from .model import RecurrentPolicy, masked_distribution
from .schema import ACTION_NAMES
from .sim import SIM_VERSION, DuelSim
from .sim_evaluation import (
    EVALUATION_SEED_FLOOR,
    SimulatorEvaluationPlan,
    SimulatorRefused,
    simulator_gate,
    simulator_record,
)
from .teacher import TEACHER_SHA256, TEACHER_STYLES, TacticalTeacher
from .train import MAX_TOTAL_TRANSITIONS, Trajectory


def _input(frame, device):
    return (
        torch.tensor([frame.features], dtype=torch.float32, device=device),
        torch.tensor([frame.mask], dtype=torch.bool, device=device),
    )


def _labels(**extra) -> dict:
    return {
        "domain": SIM_DOMAIN,
        "simulator_sha256": SIMULATOR_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "live_episode": False,
        **extra,
    }


def collect_simulator(
    model: RecurrentPolicy,
    *,
    steps: int,
    seed: int,
    max_episode_steps: int,
    shaping_gamma: float,
    opponents: list[RecurrentPolicy] | None = None,
    teacher_actions: bool = False,
    drill: str = "full",
    opponent_styles: tuple[str, ...] = ("balanced",),
    teacher_style: str = "balanced",
) -> tuple[list[Trajectory], dict]:
    """Bounded simulator rollouts for ``steps`` learner decisions; nothing is written.

    With ``teacher_actions`` the learner's recorded actions are the teacher's (labels
    for simulator behaviour cloning, ``ppo_eligible: false``); otherwise they are
    sampled from ``model``. Frozen ``opponents`` replace the scripted opponent in
    about half of the games, as in the source league.
    """
    if type(steps) is not int or not 1 <= steps <= MAX_TOTAL_TRANSITIONS:
        raise ValueError(f"steps must be an integer in 1..{MAX_TOTAL_TRANSITIONS}")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("an explicit non-negative integer seed is required")
    if type(max_episode_steps) is not int or max_episode_steps < 1:
        raise ValueError("max_episode_steps must be a positive integer")
    if type(shaping_gamma) not in (int, float) or not (
        math.isfinite(shaping_gamma) and 0 < shaping_gamma <= 1
    ):
        raise ValueError("shaping_gamma must be in (0, 1]")
    if type(teacher_actions) is not bool or drill not in DRILLS:
        raise ValueError("teacher_actions must be a bool and drill one of DRILLS")
    if (
        not isinstance(opponent_styles, tuple)
        or not opponent_styles
        or not all(s in TEACHER_STYLES for s in (*opponent_styles, teacher_style))
    ):
        raise ValueError("opponent and teacher styles must be teacher styles")
    for policy in (model, *(opponents or ())):
        if policy.device.type != "cpu":
            raise ValueError("simulator rollouts run on CPU only")
    # A private RNG scope: the explicit seed decides every sampled action.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        episodes, summary = _collect(
            model,
            steps=steps,
            seed=seed,
            opponents=opponents,
            teacher_actions=teacher_actions,
            max_episode_steps=max_episode_steps,
            shaping_gamma=shaping_gamma,
            drill=drill,
            opponent_styles=opponent_styles,
            teacher_style=teacher_style,
        )
    summary.update(
        _labels(
            seed=seed,
            drill=drill,
            teacher_actions=teacher_actions,
            max_episode_steps=max_episode_steps,
            shaping_gamma=shaping_gamma,
            real_server_strength_established=False,
        )
    )
    return episodes, summary


def _collect(
    model,
    *,
    steps,
    seed,
    opponents,
    teacher_actions,
    max_episode_steps,
    shaping_gamma,
    drill,
    opponent_styles,
    teacher_style,
):
    # The source loop (train.collect_simulator at 819f4f4), with labels added.
    rng = random.Random(seed)
    episodes = []
    total = completed = wins = losses = draws = 0
    source_counts: dict[str, int] = {}
    while total < steps:
        episode_seed = rng.randrange(1_000_000_000)
        # Held-out simulator evaluation seeds start at the floor; training never does.
        assert episode_seed < EVALUATION_SEED_FLOOR
        sim = DuelSim(episode_seed, max_episode_steps, shaping_gamma=shaping_gamma)
        side = len(episodes) % 2
        frames = setup_drill(sim, drill, side)
        learner_teacher = TacticalTeacher(teacher_style)
        if drill == "burst":
            learner_teacher.opened.update(("weaken", "clumsy"))
        opponent_teacher = TacticalTeacher(rng.choice(opponent_styles))
        # Retain a scripted anchor while broadening opponents with frozen snapshots.
        opponent = rng.choice(opponents) if opponents and rng.random() < 0.5 else None
        label = "frozen_policy" if opponent is not None else opponent_teacher.style
        source_counts[label] = source_counts.get(label, 0) + 1
        hidden = model.initial_state()
        opponent_hidden = opponent.initial_state() if opponent else None
        episode = Trajectory()
        done = False
        while total < steps and not done:
            frame = frames[side]
            features, mask = _input(frame, model.device)
            action, log_prob, value, next_hidden = model.act(features, mask, hidden)
            if teacher_actions:
                action.fill_(learner_teacher.choose(sim.view(side)))
                with torch.no_grad():
                    logits, _, _ = model(features.unsqueeze(1), hidden)
                    log_prob = masked_distribution(logits[:, 0], mask).log_prob(action)
            if opponent is None:
                opponent_action = opponent_teacher.choose(sim.view(1 - side))
            else:
                opponent_action_tensor, _, _, opponent_hidden = opponent.act(
                    *_input(frames[1 - side], opponent.device), opponent_hidden
                )
                opponent_action = int(opponent_action_tensor.item())
            actions = (
                (int(action.item()), opponent_action)
                if side == 0
                else (opponent_action, int(action.item()))
            )
            before = float(getattr(sim, "elapsed", 0.0))
            next_frames, rewards, done, info = sim.step(actions)
            elapsed = float(info.get("elapsed", before + 0.25))
            episode.features.append(frame.features)
            episode.masks.append(frame.mask)
            episode.actions.append(int(action.item()))
            episode.log_probs.append(float(log_prob.item()))
            episode.values.append(float(value.item()))
            episode.hidden.append(hidden[0].detach().cpu().tolist())
            episode.rewards.append(float(rewards[side]))
            episode.durations.append(max(0.001, elapsed - before))
            frames, hidden = next_frames, next_hidden
            total += 1
        episode.terminated = bool(done and info.get("terminated", not info.get("truncated", False)))
        episode.truncated = bool(not episode.terminated)
        episode.info = {**info, "learner_side": side, "opponent": label}
        episode.info.update(_labels(sim_seed=episode_seed, rollout_seed=seed, drill=drill))
        if teacher_actions:
            # Teacher labels are for simulator behaviour cloning, never PPO experience.
            episode.info["ppo_eligible"] = False
        if not episode.terminated:
            with torch.no_grad():
                _, value, _ = model(_input(frames[side], model.device)[0].unsqueeze(1), hidden)
            episode.bootstrap_value = float(value.item())
        if done:
            completed += 1
            winner = info.get("winner")
            wins += int(winner == side)
            losses += int(winner is not None and winner != side)
            draws += int(winner is None)
        episodes.append(episode)
    return episodes, {
        "steps": total,
        "completed_games": completed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "opponents": source_counts,
    }


@torch.no_grad()
def _suite_rows(model, *, styles, games_per_style, seed, max_steps, sampling) -> list[dict]:
    """The source ``evaluate_suite`` game loop for the given teacher styles."""
    rows = []
    model.eval()
    for style in styles:
        style_index = TEACHER_STYLES.index(style)
        for game in range(games_per_style):
            # Both sides of each seed; separate policy RNG from simulator RNG.
            fixture_seed = seed + style_index * 10000 + game // 2
            side = game % 2
            sim = DuelSim(fixture_seed, max_steps)
            teacher = TacticalTeacher(style)
            hidden = model.initial_state()
            generator = torch.Generator().manual_seed(fixture_seed + side)
            counts = {}
            while not sim.done:
                frames = sim.frames()
                action, _, _, hidden = model.act(
                    *_input(frames[side], model.device),
                    hidden,
                    generator=generator,
                    deterministic=sampling == "greedy",
                )
                action = int(action.item())
                name = ACTION_NAMES[action]
                counts[name] = counts.get(name, 0) + 1
                other = teacher.choose(sim.view(1 - side))
                _, _, _, info = sim.step((action, other) if side == 0 else (other, action))
            score = 0.5 if sim.winner is None else float(sim.winner == side)
            rows.append(
                {
                    "style": style,
                    "seed": fixture_seed,
                    "side": side,
                    "score": score,
                    "damage": info["damage"][side],
                    "self_damage": info["self_damage"][side],
                    "actions": counts,
                    "elapsed": sim.elapsed,
                    "terminated": info["terminated"],
                    "truncated": info["truncated"],
                }
            )
    return rows


def evaluate_in_simulator(checkpoint, *, expected_sha256: str, plan: SimulatorEvaluationPlan):
    """Load ``checkpoint`` by SHA-256 and play every game of the simulator plan."""
    if type(plan) is not SimulatorEvaluationPlan:
        raise SimulatorRefused("a SimulatorEvaluationPlan is required")
    plan.require_current()
    evaluation = plan.evaluation
    if expected_sha256 not in (evaluation.candidate_sha256, evaluation.incumbent_sha256):
        raise SimulatorRefused("policy is neither the plan's candidate nor its incumbent")
    if not Path(checkpoint).is_absolute():
        raise SimulatorRefused("name the checkpoint by an absolute path")
    # Building the model and playing must not advance the caller's global RNG.
    with torch.random.fork_rng(devices=[]):
        loaded = load_checkpoint(checkpoint, expected_sha256=expected_sha256)
        if loaded.model.device.type != "cpu":
            raise ValueError("simulator evaluation runs on CPU only")
        rows = _suite_rows(
            loaded.model,
            styles=[style for _, style in plan.styles()],
            games_per_style=evaluation.games_per_opponent,
            seed=plan.seed,
            max_steps=plan.max_steps,
            sampling=plan.sampling,
        )
    record = simulator_record(plan, loaded.policy_sha, rows)
    record["checkpoint"] = str(loaded.directory)
    record["simulator_version"] = SIM_VERSION
    return record


def decide_in_simulator(plan: SimulatorEvaluationPlan, *, incumbent, candidate) -> dict:
    """Evaluate the plan's incumbent and candidate checkpoints and gate them.

    The result is a simulator-only decision. It is re-derivable (the same plan and
    checkpoints give the identical decision), is never written by this function and
    can never be recorded as a live selection.
    """
    if type(plan) is not SimulatorEvaluationPlan:
        raise SimulatorRefused("a SimulatorEvaluationPlan is required")
    before = evaluate_in_simulator(
        incumbent, expected_sha256=plan.evaluation.incumbent_sha256, plan=plan
    )
    after = evaluate_in_simulator(
        candidate, expected_sha256=plan.evaluation.candidate_sha256, plan=plan
    )
    decision = simulator_gate(plan, before, after)
    decision["evidence"]["simulator"]["checkpoints"] = {
        "incumbent": before["checkpoint"],
        "candidate": after["checkpoint"],
    }
    return decision
