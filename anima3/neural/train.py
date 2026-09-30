"""Recurrent PPO, simulator self-play, and gated updates from live trajectories.

Simulator scores describe this approximate simulator only.  No model is
promoted to the public arena automatically.  Live updates require a trusted
receipt verifier and exact behavior-policy provenance.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .checkpoints import (
    LoadedCheckpoint,
    load_checkpoint,
    restore_optimizer,
    restore_training_rng,
    save_checkpoint,
    schema_fingerprint,
)
from .model import PolicyConfig, RecurrentPolicy, masked_distribution
from .schema import ACTION_NAMES, FEATURE_NAMES

LIVE_SCHEMA = "anima3-live-policy-v1"


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    epochs: int = 4
    sequence_length: int = 64
    burn_in: int = 8
    batch_sequences: int = 8
    max_grad_norm: float = 0.5
    target_kl: float = 0.04
    # 0 means one discount per decision; a positive value uses gamma**(dt/unit).
    discount_time_unit: float = 0.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.learning_rate,
                self.gamma,
                self.gae_lambda,
                self.clip_range,
                self.value_coef,
                self.entropy_coef,
                self.max_grad_norm,
                self.target_kl,
                self.discount_time_unit,
            )
        ):
            raise ValueError("training coefficients must be finite")
        if not (0 < self.learning_rate <= 0.1 and 0 < self.gamma <= 1):
            raise ValueError("invalid learning rate or discount")
        if not (0 <= self.gae_lambda <= 1 and 0 < self.clip_range < 1):
            raise ValueError("invalid GAE lambda or PPO clip range")
        if self.value_coef < 0 or self.entropy_coef < 0:
            raise ValueError("value and entropy coefficients must be nonnegative")
        if min(self.epochs, self.sequence_length, self.batch_sequences) < 1 or self.burn_in < 0:
            raise ValueError("invalid recurrent training budget")
        if self.discount_time_unit < 0 or self.max_grad_norm <= 0 or self.target_kl <= 0:
            raise ValueError("invalid discount time unit or optimization bounds")


@dataclass
class Trajectory:
    features: list[list[float]] = field(default_factory=list)
    masks: list[list[bool]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    hidden: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    bootstrap_value: float = 0.0
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)


def generalized_advantage(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    boundaries: Tensor,
    *,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    durations: Tensor | None = None,
    discount_time_unit: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """GAE bootstraps truncations but stops its trace at every episode boundary.

    ``terminated`` means a real terminal state, whose next value is zero.
    ``boundaries`` also includes time limits and collection truncations.  A
    duration represents the time between this decision and the next state.
    """
    if rewards.ndim != 1 or not all(
        tensor.shape == rewards.shape for tensor in (values, next_values, terminated, boundaries)
    ):
        raise ValueError("GAE arrays must have the same one-dimensional shape")
    if discount_time_unit:
        if durations is None or durations.shape != rewards.shape or bool((durations <= 0).any()):
            raise ValueError("positive transition durations are required")
        powers = durations / discount_time_unit
        discounts = torch.pow(torch.full_like(rewards, gamma), powers)
        traces = torch.pow(torch.full_like(rewards, gae_lambda), powers)
    else:
        discounts = torch.full_like(rewards, gamma)
        traces = torch.full_like(rewards, gae_lambda)
    deltas = rewards + discounts * (~terminated.bool()) * next_values - values
    advantage = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for index in range(len(rewards) - 1, -1, -1):
        running = (
            deltas[index] + discounts[index] * traces[index] * (~boundaries[index].bool()) * running
        )
        advantage[index] = running
    return advantage, advantage + values


def clipped_policy_loss(log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor, clip: float):
    ratio = (log_probs - old_log_probs).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)


def _targets(episodes: list[Trajectory], config: TrainConfig) -> list[tuple[Tensor, Tensor]]:
    targets = []
    for episode in episodes:
        values = torch.tensor(episode.values, dtype=torch.float32)
        terminals = torch.zeros(len(episode), dtype=torch.bool)
        boundaries = terminals.clone()
        terminals[-1] = episode.terminated
        boundaries[-1] = True
        targets.append(
            generalized_advantage(
                torch.tensor(episode.rewards, dtype=torch.float32),
                values,
                torch.cat((values[1:], torch.tensor([episode.bootstrap_value]))),
                terminals,
                boundaries,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                durations=torch.tensor(episode.durations, dtype=torch.float32),
                discount_time_unit=config.discount_time_unit,
            )
        )
    free_choices = [torch.tensor(episode.masks).sum(-1) > 1 for episode in episodes]
    # Forced holds remain in the recurrent trajectory and critic targets, but
    # their advantages must not set the scale of the actor's actual decisions.
    actor_advantages = torch.cat(
        [target[0][free] for target, free in zip(targets, free_choices, strict=True)]
    )
    if len(actor_advantages):
        mean = actor_advantages.mean()
        std = actor_advantages.std(unbiased=False).clamp_min(1e-8)
    else:
        mean, std = 0.0, 1.0
    return [
        (torch.where(free, (adv - mean) / std, 0.0), returns)
        for (adv, returns), free in zip(targets, free_choices, strict=True)
    ]


def _chunks(episodes: list[Trajectory], length: int):
    return [
        (episode, start)
        for episode, data in enumerate(episodes)
        for start in range(0, len(data), length)
    ]


def _batch(
    model: RecurrentPolicy,
    episodes: list[Trajectory],
    selections: list[tuple[int, int]],
    config: TrainConfig,
    targets: list[tuple[Tensor, Tensor]] | None,
) -> dict[str, Tensor]:
    """Recompute burn-in without gradients, then retain recurrence through the chunk.

    Truncated BPTT detaches only at chunk boundaries.  Padding has a hold-only
    mask and contributes neither policy nor value loss.
    """
    device = model.device
    count, length = len(selections), config.sequence_length
    batch = {
        "features": torch.zeros(count, length, model.config.feature_dim, device=device),
        "masks": torch.zeros(
            count, length, model.config.action_dim, device=device, dtype=torch.bool
        ),
        "actions": torch.zeros(count, length, device=device, dtype=torch.long),
        "old_log_probs": torch.zeros(count, length, device=device),
        "old_values": torch.zeros(count, length, device=device),
        "advantages": torch.zeros(count, length, device=device),
        "returns": torch.zeros(count, length, device=device),
        "valid": torch.zeros(count, length, device=device, dtype=torch.bool),
        "hidden": model.initial_state(count),
    }
    batch["masks"][:, :, 0] = True
    for row, (episode_index, start) in enumerate(selections):
        episode = episodes[episode_index]
        stop = min(start + length, len(episode))
        size = stop - start
        burn_start = max(0, start - config.burn_in)
        hidden = torch.tensor([episode.hidden[burn_start]], device=device)
        if burn_start < start:
            with torch.no_grad():
                _, _, hidden = model(
                    torch.tensor([episode.features[burn_start:start]], device=device), hidden
                )
        batch["hidden"][row] = hidden[0]
        for key, source in (
            ("features", episode.features),
            ("masks", episode.masks),
            ("actions", episode.actions),
            ("old_log_probs", episode.log_probs),
            ("old_values", episode.values),
        ):
            batch[key][row, :size] = torch.as_tensor(source[start:stop], device=device)
        if targets is not None:
            advantage, returns = targets[episode_index]
            batch["advantages"][row, :size] = advantage[start:stop].to(device)
            batch["returns"][row, :size] = returns[start:stop].to(device)
        batch["valid"][row, :size] = True
    return batch


def ppo_update(
    model: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    episodes: list[Trajectory],
    config: TrainConfig,
    *,
    seed: int = 0,
) -> dict:
    if not episodes or not all(len(episode) for episode in episodes):
        raise ValueError("PPO requires non-empty trajectories")
    targets = _targets(episodes, config)
    chunks = _chunks(episodes, config.sequence_length)
    rng = random.Random(seed)
    totals: dict[str, float] = {}
    count = 0
    actor_count = 0
    actor_metric_names = {"policy_loss", "entropy", "approximate_kl", "clip_fraction"}
    stopped = False
    model.train()
    for _ in range(config.epochs):
        rng.shuffle(chunks)
        for offset in range(0, len(chunks), config.batch_sequences):
            batch = _batch(
                model, episodes, chunks[offset : offset + config.batch_sequences], config, targets
            )
            log_probs, entropy, values, _ = model.evaluate_actions(
                batch["features"], batch["masks"], batch["actions"], batch["hidden"]
            )
            valid = batch["valid"]
            free = valid & (batch["masks"].sum(-1) > 1)
            actor_weight = int(free.sum())
            if actor_weight:
                log_ratio = log_probs[free] - batch["old_log_probs"][free]
                approximate_kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                if float(approximate_kl.detach()) > config.target_kl * 1.5:
                    stopped = True
                    break
                policy_loss = clipped_policy_loss(
                    log_probs[free],
                    batch["old_log_probs"][free],
                    batch["advantages"][free],
                    config.clip_range,
                ).mean()
                mean_entropy = entropy[free].mean()
                clip_fraction = (
                    ((log_ratio.exp() - 1).abs() > config.clip_range).float().mean()
                )
            else:
                # A chunk can be entirely waiting/casting.  Its critic and
                # recurrent state still learn without an empty actor mean.
                policy_loss = mean_entropy = approximate_kl = clip_fraction = values.new_zeros(())
            clipped_values = batch["old_values"] + (values - batch["old_values"]).clamp(
                -config.clip_range, config.clip_range
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (values[valid] - batch["returns"][valid]).square(),
                    (clipped_values[valid] - batch["returns"][valid]).square(),
                ).mean()
            )
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * mean_entropy
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not bool(torch.isfinite(gradient)):
                raise ValueError("non-finite PPO gradient")
            optimizer.step()
            weight = int(valid.sum())
            metrics = {
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(mean_entropy.detach()),
                "approximate_kl": float(approximate_kl.detach()),
                "clip_fraction": float(clip_fraction.detach()),
                "grad_norm": float(gradient),
            }
            for key, value in metrics.items():
                metric_weight = actor_weight if key in actor_metric_names else weight
                totals[key] = totals.get(key, 0.0) + value * metric_weight
            count += weight
            actor_count += actor_weight
        if stopped:
            break
    model.eval()
    return {
        **{
            key: value / max(actor_count if key in actor_metric_names else count, 1)
            for key, value in totals.items()
        },
        "optimized_steps": count,
        "policy_optimized_steps": actor_count,
        "kl_early_stop": stopped,
        "rollout_steps": sum(map(len, episodes)),
    }


def _input(frame, device):
    return (
        torch.tensor([frame.features], dtype=torch.float32, device=device),
        torch.tensor([frame.mask], dtype=torch.bool, device=device),
    )


def collect_simulator(
    model: RecurrentPolicy,
    *,
    steps: int,
    seed: int,
    opponents: list[RecurrentPolicy] | None = None,
    teacher_actions: bool = False,
    max_episode_steps: int = 1200,
    shaping_gamma: float = 0.995,
) -> tuple[list[Trajectory], dict]:
    from .sim import DuelSim

    rng = random.Random(seed)
    episodes = []
    total = completed = wins = losses = draws = 0
    source_counts: dict[str, int] = {}
    while total < steps:
        sim = DuelSim(max_steps=max_episode_steps, shaping_gamma=shaping_gamma)
        frames = sim.reset(seed=rng.randrange(1_000_000_000))
        side = len(episodes) % 2
        # Retain a scripted anchor while broadening opponents with frozen snapshots.
        opponent = rng.choice(opponents) if opponents and rng.random() < 0.5 else None
        label = "frozen_policy" if opponent is not None else "scripted_teacher"
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
                action.fill_(sim.teacher(side))
                with torch.no_grad():
                    logits, _, _ = model(features.unsqueeze(1), hidden)
                    log_prob = masked_distribution(logits[:, 0], mask).log_prob(action)
            if opponent is None:
                opponent_action = sim.teacher(1 - side)
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


def behavioral_clone(
    model: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    episodes: list[Trajectory],
    config: TrainConfig,
    *,
    epochs: int = 4,
    seed: int = 0,
) -> dict:
    chunks = _chunks(episodes, config.sequence_length)
    rng = random.Random(seed)
    metrics = []
    model.train()
    for _ in range(epochs):
        rng.shuffle(chunks)
        for offset in range(0, len(chunks), config.batch_sequences):
            batch = _batch(
                model, episodes, chunks[offset : offset + config.batch_sequences], config, None
            )
            log_prob, _, _, _ = model.evaluate_actions(
                batch["features"], batch["masks"], batch["actions"], batch["hidden"]
            )
            # Forced hold frames have zero policy loss and dilute the useful labels.
            valid = batch["valid"] & (batch["masks"].sum(-1) > 1)
            if not bool(valid.any()):
                continue
            loss = -log_prob[valid].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            metrics.append(float(loss.detach()))
    model.eval()
    return {
        "bc_updates": len(metrics),
        "bc_loss_first": metrics[0] if metrics else None,
        "bc_loss_last": metrics[-1] if metrics else None,
        "teacher_steps": sum(map(len, episodes)),
    }


def evaluate(
    model: RecurrentPolicy,
    *,
    games: int = 20,
    seed: int = 1_000_000,
    max_episode_steps: int = 1200,
    sampling: str = "greedy",
) -> dict:
    """Held-out seeds, alternating spawn sides, fixed scripted opponent."""
    from .sim import SIM_VERSION, DuelSim

    if sampling not in {"greedy", "categorical"}:
        raise ValueError("evaluation sampling must be greedy or categorical")

    rows = []
    model.eval()
    for index in range(games):
        side = index % 2
        policy_seed = seed + index
        generator = torch.Generator(device="cpu").manual_seed(policy_seed)
        sim = DuelSim(max_steps=max_episode_steps)
        frames = sim.reset(seed=seed + index // 2)
        hidden = model.initial_state()
        done = False
        steps = 0
        action_counts = dict.fromkeys(ACTION_NAMES, 0)
        free_action_counts = dict.fromkeys(ACTION_NAMES, 0)
        while not done:
            action, _, _, hidden = model.act(
                *_input(frames[side], model.device), hidden,
                deterministic=sampling == "greedy", generator=generator,
            )
            action_name = ACTION_NAMES[int(action.item())]
            action_counts[action_name] += 1
            if sum(frames[side].mask) > 1:
                free_action_counts[action_name] += 1
            opponent_action = sim.teacher(1 - side)
            pair = (
                (int(action.item()), opponent_action)
                if side == 0
                else (opponent_action, int(action.item()))
            )
            frames, _, done, info = sim.step(pair)
            steps += 1
        winner = info.get("winner")
        rows.append(
            {
                "seed": seed + index // 2,
                "side": side,
                "policy_seed": policy_seed if sampling == "categorical" else None,
                "outcome": "draw" if winner is None else "win" if winner == side else "loss",
                "steps": steps,
                "action_counts": action_counts,
                "free_action_counts": free_action_counts,
                "elapsed": info.get("elapsed"),
                "terminated": bool(info.get("terminated")),
                "truncated": bool(info.get("truncated")),
                "damage": info.get("damage"),
                "self_damage": info.get("self_damage"),
                "healing": info.get("healing"),
            }
        )
    wins = sum(row["outcome"] == "win" for row in rows)
    draws = sum(row["outcome"] == "draw" for row in rows)
    return {
        "domain": "approximate-pre-aos-simulator",
        "simulator_version": SIM_VERSION,
        "sampling": sampling,
        "games": games,
        "wins": wins,
        "losses": games - wins - draws,
        "draws": draws,
        "win_rate": wins / games if games else None,
        "rows": rows,
        "real_server_strength_established": False,
        "automatic_promotion": False,
    }


def _live_trajectory(loaded: LoadedCheckpoint, episode: dict) -> Trajectory:
    """Independently recompute the behavior likelihood and recurrent state chain."""
    if episode.get("schema") != LIVE_SCHEMA or episode.get("domain") != "servuo":
        raise ValueError("unsupported live trajectory schema/domain")
    if episode.get("sampling") != "categorical":
        raise ValueError("PPO requires stochastic categorical behavior, not argmax evaluation")
    if episode.get("policy_sha") != loaded.policy_sha:
        raise ValueError("off-policy trajectory: checkpoint SHA does not match")
    if episode.get("schema_fingerprint") != schema_fingerprint():
        raise ValueError("live observation schema mismatch")
    verification = episode.get("verification", {})
    if not (
        verification.get("verified") is True
        and verification.get("complete") is True
        and verification.get("training") is True
        and verification.get("aborted") is False
    ):
        raise ValueError("only complete verified training duels may update the policy")
    digest = verification.get("replay_sha256", "")
    if (
        not verification.get("match_id")
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("missing replay provenance")
    transitions = episode.get("transitions", [])
    if not isinstance(transitions, list) or not (1 <= len(transitions) <= 50_000):
        raise ValueError("invalid live trajectory length")
    model = loaded.model
    hidden = model.initial_state()
    result = Trajectory(terminated=True, info={"verification": verification})
    for index, step in enumerate(transitions):
        if step.get("accepted") is not True:
            raise ValueError("rejected or substituted actions invalidate an on-policy episode")
        if step.get("done") is not (index == len(transitions) - 1):
            raise ValueError("live trajectory must contain exactly one terminal boundary")
        features = step.get("features")
        mask = step.get("mask")
        action = step.get("action")
        if not isinstance(features, list) or len(features) != model.config.feature_dim:
            raise ValueError("live features have wrong shape")
        if not all(
            isinstance(x, (int, float)) and math.isfinite(x) and -1.001 <= x <= 1.001
            for x in features
        ):
            raise ValueError("invalid normalized live features")
        if (
            not isinstance(mask, list)
            or len(mask) != model.config.action_dim
            or not all(type(x) is bool for x in mask)
        ):
            raise ValueError("invalid live action mask")
        if type(action) is not int or not (0 <= action < len(mask)) or not mask[action]:
            raise ValueError("live action was not legal in its recorded mask")
        hidden_in = torch.tensor(
            step.get("hidden_in", []), dtype=torch.float32, device=model.device
        )
        if hidden_in.shape != hidden[0].shape or not torch.allclose(
            hidden_in, hidden[0], atol=2e-4, rtol=2e-4
        ):
            raise ValueError("live recurrent state chain mismatch")
        with torch.no_grad():
            logits, values, next_hidden = model(
                torch.tensor([[features]], device=model.device), hidden
            )
            distribution = masked_distribution(
                logits[:, 0], torch.tensor([mask], device=model.device)
            )
            log_prob = float(
                distribution.log_prob(torch.tensor([action], device=model.device)).item()
            )
            value = float(values.item())
        for key, expected in (("log_prob", log_prob), ("value", value)):
            actual = step.get(key)
            if (
                not isinstance(actual, (float, int))
                or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=2e-4, abs_tol=2e-4)
            ):
                raise ValueError(f"live behavior policy {key} mismatch")
        reward, duration = step.get("reward"), step.get("dt")
        if not isinstance(reward, (float, int)) or not math.isfinite(reward) or abs(reward) > 10:
            raise ValueError("invalid live reward")
        if (
            not isinstance(duration, (float, int))
            or not math.isfinite(duration)
            or not (0 < duration <= 600)
        ):
            raise ValueError("live dt must be elapsed seconds in (0, 600]")
        result.features.append(features)
        result.masks.append(mask)
        result.actions.append(action)
        result.hidden.append(hidden[0].cpu().tolist())
        result.values.append(value)
        result.log_probs.append(log_prob)
        result.rewards.append(float(reward))
        result.durations.append(float(duration))
        hidden = next_hidden
    return result


def _episode_actor(episode: dict) -> int:
    actor = episode.get("player_serial", episode.get("actor"))
    if type(actor) is not int or actor <= 0:
        raise ValueError("live episode requires a positive actor serial")
    if "actor" in episode and episode["actor"] != actor:
        raise ValueError("conflicting actor serials in live episode")
    return actor


def update_verified_episodes(
    loaded: LoadedCheckpoint,
    episodes: list[dict],
    *,
    verify_episode: Callable[[dict], bool],
    out: str | Path,
    config: TrainConfig | None = None,
    seed: int = 0,
) -> dict:
    """One PPO update using complete duels collected by this exact checkpoint.

    A JSON ``verified`` flag is insufficient: the caller must provide a trusted
    receipt/replay verifier.  This function additionally rejects changed model
    versions, masks, substituted actions and broken recurrent state histories.
    All episodes are validated before any parameter changes.
    """
    if not callable(verify_episode) or not (1 <= len(episodes) <= 200):
        raise ValueError("trusted verifier and 1..200 live episodes required")
    config = config or TrainConfig(discount_time_unit=1.0)
    trajectories = []
    consumed = list(loaded.manifest.get("training", {}).get("live_consumed", []))
    seen = {tuple(identity) for identity in consumed}
    for episode in episodes:
        identity = (episode.get("verification", {}).get("match_id"), _episode_actor(episode))
        if identity in seen:
            raise ValueError("duplicate live episode")
        seen.add(identity)
        consumed.append(list(identity))
        if verify_episode(episode) is not True:
            raise ValueError("trusted replay verification failed")
        trajectories.append(_live_trajectory(loaded, episode))
    candidate = copy.deepcopy(loaded.model)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=config.learning_rate, eps=1e-5)
    restore_optimizer(loaded, optimizer)
    # An explicit live configuration always wins over the saved optimizer LR.
    for group in optimizer.param_groups:
        group["lr"] = config.learning_rate
    metrics = ppo_update(candidate, optimizer, trajectories, config, seed=seed)
    training = {
        "source": "verified-servuo-on-policy",
        "parent_policy_sha": loaded.policy_sha,
        "config": asdict(config),
        "actor_normalization": "free-choice",
        "seed": seed,
        "metrics": metrics,
        "episodes": [
            {
                "match_id": row["verification"]["match_id"],
                "replay_sha256": row["verification"]["replay_sha256"],
                "player_serial": _episode_actor(row),
            }
            for row in episodes
        ],
        "discount": "gamma**dt_seconds" if config.discount_time_unit == 1 else "configured",
        "strength_improvement_established": False,
        "live_consumed": consumed,
    }
    manifest = save_checkpoint(out, candidate, training=training, optimizer=optimizer)
    result = {
        "policy_sha": manifest["weights"]["sha256"],
        "parent_policy_sha": loaded.policy_sha,
        "episodes": len(episodes),
        "metrics": metrics,
        "automatic_promotion": False,
    }
    Path(out, "live-update.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _append_metrics(root: Path, row: dict) -> None:
    with (root / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), **row}, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--steps", type=int, default=20_000, help="additional PPO environment steps"
    )
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--bc-steps", type=int, default=2048)
    parser.add_argument("--bc-epochs", type=int, default=6)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-seed", type=int, default=2_000_000_000)
    parser.add_argument("--eval-sampling", choices=("greedy", "categorical"), default="greedy")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--discount-time-unit",
        type=float,
        default=0.0,
        help="seconds per discount unit; 0 discounts once per decision",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--league-every", type=int, default=4)
    args = parser.parse_args(argv)
    if (
        min(args.steps, args.bc_steps, args.eval_games) < 0
        or min(
            args.rollout_steps,
            args.threads,
            args.max_episode_steps,
            args.league_every,
            args.bc_epochs,
        )
        < 1
    ):
        parser.error("budgets must be nonnegative and intervals/threads positive")
    if args.eval_seed < 1_000_000_000:
        parser.error("held-out eval seeds must be >= 1000000000; training seeds are below it")
    if args.evaluate_only and args.resume is None:
        parser.error("--evaluate-only requires --resume")
    try:
        config = TrainConfig(
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            entropy_coef=args.entropy_coef,
            discount_time_unit=args.discount_time_unit,
            epochs=args.epochs,
            sequence_length=args.sequence_length,
        )
    except ValueError as exc:
        parser.error(str(exc))
    simulator_dt = 0.25
    # Potential shaping must use the same transition discount as GAE.  With
    # elapsed-time discounting, gamma is per configured unit, not per sim tick.
    shaping_gamma = (
        config.gamma ** (simulator_dt / config.discount_time_unit)
        if config.discount_time_unit > 0
        else config.gamma
    )
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    loaded = load_checkpoint(args.resume, args.device) if args.resume else None
    model = (
        loaded.model
        if loaded
        else RecurrentPolicy(
            PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), args.hidden_size, args.hidden_size)
        ).to(args.device)
    )
    if args.evaluate_only:
        result = evaluate(
            model,
            games=args.eval_games,
            seed=args.eval_seed,
            max_episode_steps=args.max_episode_steps,
            sampling=args.eval_sampling,
        )
        result["policy_sha"] = loaded.policy_sha
        (args.out / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in result.items() if key != "rows"}), flush=True
        )
        return 0
    if not loaded and (args.out / "manifest.json").exists():
        parser.error("output already contains a policy; use --resume or a new output directory")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    prior = loaded.manifest.get("training", {}) if loaded else {}
    if loaded:
        restore_optimizer(loaded, optimizer)
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate
    updates = int(prior.get("updates", 0))
    total_steps = int(prior.get("environment_steps", 0))
    previous_sha = loaded.policy_sha if loaded else None
    before = evaluate(
        model, games=args.eval_games, seed=args.eval_seed,
        max_episode_steps=args.max_episode_steps, sampling=args.eval_sampling,
    )
    (args.out / "evaluation-before.json").write_text(json.dumps(before, indent=2) + "\n")
    _append_metrics(
        args.out, {"event": "evaluation_before", **{k: v for k, v in before.items() if k != "rows"}}
    )
    if args.bc_steps and not loaded:
        demonstrations, _ = collect_simulator(
            model,
            steps=args.bc_steps,
            seed=args.seed,
            teacher_actions=True,
            max_episode_steps=args.max_episode_steps,
            shaping_gamma=shaping_gamma,
        )
        bc = behavioral_clone(
            model, optimizer, demonstrations, config, epochs=args.bc_epochs, seed=args.seed
        )
        _append_metrics(args.out, {"event": "behavioral_cloning", **bc})
    league = [copy.deepcopy(model).eval()]
    if loaded:
        league_root = loaded.directory / "league"
        for artifact in sorted(league_root.glob("update-*/manifest.json"))[-3:]:
            league.append(load_checkpoint(artifact, args.device).model)
    for member in league:
        member.requires_grad_(False)
    from .sim import SIM_VERSION

    metadata = {
        "source": "approximate-pre-aos-simulator",
        "config": asdict(config),
        "actor_normalization": "free-choice",
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "eval_sampling": args.eval_sampling,
        "environment_steps": total_steps,
        "updates": updates,
        "parent_policy_sha": previous_sha,
        "teacher_steps": args.bc_steps if not loaded else prior.get("teacher_steps", 0),
        "bc_epochs": args.bc_epochs if not loaded else prior.get("bc_epochs", 0),
        "torch_version": str(torch.__version__),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "real_server_strength_established": False,
        "live_consumed": prior.get("live_consumed", []),
        "simulator": {
            "version": SIM_VERSION,
            "dt": simulator_dt,
            "max_episode_steps": args.max_episode_steps,
            "showdown_after": 180,
            "shaping_gamma": shaping_gamma,
        },
    }
    if loaded:
        restore_training_rng(loaded)
    else:
        save_checkpoint(
            args.out / "warmstart",
            model,
            training={**metadata, "phase": "before-PPO"},
            optimizer=optimizer,
        )
        after_bc = evaluate(
            model,
            games=args.eval_games,
            seed=args.eval_seed,
            max_episode_steps=args.max_episode_steps,
            sampling=args.eval_sampling,
        )
        (args.out / "evaluation-after-bc.json").write_text(json.dumps(after_bc, indent=2) + "\n")
        _append_metrics(
            args.out,
            {"event": "evaluation_after_bc", **{k: v for k, v in after_bc.items() if k != "rows"}},
        )
    save_checkpoint(args.out, model, training=metadata, optimizer=optimizer)
    remaining = args.steps
    started = time.monotonic()
    while remaining > 0:
        steps = min(args.rollout_steps, remaining)
        episodes, rollout = collect_simulator(
            model,
            steps=steps,
            seed=args.seed + updates * 100_003,
            opponents=league,
            max_episode_steps=args.max_episode_steps,
            shaping_gamma=shaping_gamma,
        )
        metrics = ppo_update(model, optimizer, episodes, config, seed=args.seed + updates)
        remaining -= steps
        total_steps += steps
        updates += 1
        _append_metrics(
            args.out,
            {
                "event": "ppo",
                "update": updates,
                "environment_steps": total_steps,
                "wall_seconds": round(time.monotonic() - started, 3),
                **rollout,
                **metrics,
            },
        )
        metadata.update(environment_steps=total_steps, updates=updates, latest_metrics=metrics)
        save_checkpoint(args.out, model, training=metadata, optimizer=optimizer)
        if updates % args.league_every == 0:
            frozen = copy.deepcopy(model).eval().requires_grad_(False)
            league.append(frozen)
            league = league[-4:]
            save_checkpoint(
                args.out / "league" / f"update-{updates:06d}", frozen, training=metadata
            )
    after = evaluate(
        model, games=args.eval_games, seed=args.eval_seed,
        max_episode_steps=args.max_episode_steps, sampling=args.eval_sampling,
    )
    current = load_checkpoint(args.out)
    after.update(
        policy_sha=current.policy_sha,
        before_win_rate=before["win_rate"],
        simulator_win_rate_delta=(after["win_rate"] - before["win_rate"])
        if args.eval_games
        else None,
        training_steps_this_run=args.steps,
        environment_steps=total_steps,
    )
    (args.out / "evaluation.json").write_text(json.dumps(after, indent=2) + "\n")
    _append_metrics(
        args.out, {"event": "evaluation_after", **{k: v for k, v in after.items() if k != "rows"}}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
