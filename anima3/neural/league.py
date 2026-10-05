"""Bounded curriculum/league training with paired, frozen evaluation.

Promotion here selects an experimental checkpoint in a local registry. It never
replaces the public Arena agent. Simulator and ServUO evidence stay distinct.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch

from ..arena_learning import atomic_json
from .checkpoints import load_checkpoint, save_checkpoint
from .curriculum import DRILLS, STYLES, TacticalTeacher
from .model import PolicyConfig, RecurrentPolicy
from .schema import ACTION_NAMES, FEATURE_NAMES
from .sim import DuelSim
from .train import TrainConfig, _input, behavioral_clone, collect_simulator, ppo_update


@torch.no_grad()
def evaluate_suite(model, *, games_per_style=20, seed=3_600_000_000, max_steps=800):
    if games_per_style < 2 or games_per_style % 2 or seed < 1_000_000_000:
        raise ValueError("even game count >=2 and held-out seed required")
    rows = []
    model.eval()
    for style_index, style in enumerate(STYLES):
        for game in range(games_per_style):
            # Both sides of each seed; separate policy RNG from simulator RNG.
            fixture_seed = seed + style_index * 10000 + game // 2
            side = game % 2
            sim = DuelSim(seed=fixture_seed, max_steps=max_steps)
            teacher = TacticalTeacher(style)
            hidden = model.initial_state()
            generator = torch.Generator().manual_seed(fixture_seed + side)
            counts = {}
            while not sim.done:
                frames = sim.frames()
                action, _, _, hidden = model.act(
                    *_input(frames[side], model.device), hidden, generator=generator)
                action = int(action.item())
                name = ACTION_NAMES[action]
                counts[name] = counts.get(name, 0) + 1
                other = teacher.choose(sim.view(1-side))
                _, _, _, info = sim.step((action, other) if side == 0 else (other, action))
            score = 0.5 if sim.winner is None else float(sim.winner == side)
            rows.append({"style": style, "seed": fixture_seed, "side": side,
                         "score": score, "damage": info["damage"][side],
                         "self_damage": info["self_damage"][side],
                         "actions": counts, "elapsed": sim.elapsed})
    return {"domain": "approximate-pre-aos-simulator", "rows": rows,
            "games": len(rows), "score": sum(r["score"] for r in rows)/len(rows),
            "real_server_strength_established": False}


def promotion_gate(before, after, *, min_games_per_style=20, max_regression=0.10):
    """Paired bootstrap over seed blocks (both sides stay together). Fail closed."""
    if before["domain"] != after["domain"]:
        raise ValueError("cannot compare simulator with live outcomes")
    key = lambda r: (r["style"], r["seed"], r["side"])
    a, b = {key(r): r for r in before["rows"]}, {key(r): r for r in after["rows"]}
    if not a or a.keys() != b.keys() or len(a) != len(before["rows"]) or len(b) != len(after["rows"]):
        raise ValueError("paired evaluation fixtures differ or duplicate")
    by_style, blocks = {}, {}
    for k, row in a.items():
        delta = b[k]["score"] - row["score"]
        by_style.setdefault(k[0], []).append(delta)
        blocks.setdefault(k[:2], []).append(delta)
    if any(len(v) != 2 for v in blocks.values()):
        raise ValueError("each evaluation seed requires both sides")
    deltas = [sum(v)/len(v) for v in blocks.values()]
    rng = random.Random(619)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(2000))
    lower = samples[49]  # two-sided 95% bootstrap lower bound
    enough = set(by_style) == set(STYLES) and all(len(v) >= min_games_per_style for v in by_style.values())
    regressions = {k: sum(v)/len(v) for k, v in by_style.items()}
    passed = enough and lower > 0 and min(regressions.values()) >= -max_regression
    return {"passed": passed, "enough_evidence": enough, "paired_delta": sum(deltas)/len(deltas),
            "delta_ci95": [lower, samples[1949]], "style_deltas": regressions,
            "reason": "improved" if passed else "retain incumbent: insufficient or regressed evidence",
            "domain": before["domain"], "public_deployment": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--bc-steps-per-drill", type=int, default=4096)
    p.add_argument("--bc-epochs", type=int, default=32)
    p.add_argument("--updates", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=4096)
    p.add_argument("--eval-games-per-style", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=3_600_000_000)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)
    if min(args.bc_steps_per_drill, args.updates, args.rollout_steps, args.max_steps, args.threads, args.bc_epochs) < 1:
        p.error("positive bounded budgets required")
    if args.eval_games_per_style < 2 or args.eval_games_per_style % 2:
        p.error("evaluation count must be positive and even")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), 64, 64))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    config = TrainConfig(learning_rate=1e-4, epochs=4)
    metadata = {"domain": "approximate-pre-aos-simulator", "source": "curriculum-league-v2",
                "seed": args.seed, "real_server_strength_established": False}
    save_checkpoint(root / "initial", model, training=metadata)
    log = []

    def record(row):
        log.append(row)
        atomic_json(root / "progress.json", {"events": log, "status": "running"})
        print(json.dumps(row), flush=True)

    # BC is deliberately separate from PPO: demonstrations are never PPO episodes.
    demos = []
    for i, drill in enumerate(DRILLS):
        trajectories, summary = collect_simulator(
            model, steps=args.bc_steps_per_drill, seed=args.seed + i,
            teacher_actions=True, drill=drill,
            teacher_style="pressure" if drill == "melee" else "balanced",
            opponent_styles=STYLES, max_episode_steps=min(args.max_steps, 400),
            shaping_gamma=config.gamma)
        demos.extend(trajectories)
        record({"stage": "demonstrations", "drill": drill, **summary})
    # Portable labels preserve exact public features/masks/actions for audit.
    atomic_json(root / "demonstrations.json", {
        "domain": metadata["domain"], "ppo_eligible": False,
        "episodes": [{"features": e.features, "masks": e.masks, "actions": e.actions}
                     for e in demos]})
    metrics = behavioral_clone(model, optimizer, demos, config, epochs=args.bc_epochs, seed=args.seed)
    save_checkpoint(root / "warmstart", model, training={**metadata, "bc": metrics})
    record({"stage": "imitation", **metrics})
    incumbent = copy.deepcopy(model).eval().requires_grad_(False)
    league = [incumbent]
    # New optimizer: supervised moments are not PPO optimizer state.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    for update in range(args.updates):
        drill = DRILLS[min(3, update * 4 // args.updates)]
        episodes, rollout = collect_simulator(
            model, steps=args.rollout_steps, seed=args.seed + 1000 + update,
            opponents=league, drill=drill, opponent_styles=STYLES,
            max_episode_steps=args.max_steps, shaping_gamma=config.gamma)
        metrics = ppo_update(model, optimizer, episodes, config, seed=args.seed + update)
        checkpoint = root / "snapshots" / f"update-{update+1:03d}"
        save_checkpoint(checkpoint, model, training={**metadata, "update": update+1,
                                                    "drill": drill, "metrics": metrics})
        record({"stage": "ppo", "update": update+1, "drill": drill, **rollout, **metrics})
        if (update+1) % 2 == 0:
            league.append(copy.deepcopy(model).eval().requires_grad_(False))
            league = [incumbent] + league[-3:]
    eval_kw = {"games_per_style": args.eval_games_per_style, "max_steps": args.max_steps, "seed": args.eval_seed}
    # Used once after training: acceptance seeds never feed gradients or scheduling.
    initial = load_checkpoint(root / "initial").model
    reports = {}
    for label, frozen in (("initial", initial), ("warmstart", incumbent), ("candidate", model)):
        reports[label] = evaluate_suite(frozen, **eval_kw)
        atomic_json(root / f"evaluation-{label}.json", reports[label])
        record({"stage": "evaluation", "model": label, "score": reports[label]["score"]})
    bc_gate = promotion_gate(reports["initial"], reports["warmstart"])
    gate = promotion_gate(reports["warmstart"], reports["candidate"])
    base_label = "warmstart" if bc_gate["passed"] else "initial"
    final_gate = promotion_gate(reports[base_label], reports["candidate"])
    final_path = checkpoint if final_gate["passed"] else root / base_label
    selected = load_checkpoint(final_path)
    report = {"status": "complete", "bc_gate": bc_gate, "ppo_gate": gate, "selection_gate": final_gate,
              "selected_checkpoint": str(final_path), "selected_sha": selected.policy_sha,
              "selection_scope": "experimental-training-only", "public_deployment": False,
              "real_server_strength_established": False,
              "training_steps": args.updates*args.rollout_steps,
              "demonstration_steps": args.bc_steps_per_drill*len(DRILLS)}
    atomic_json(root / "registry.json", report)
    atomic_json(root / "progress.json", {"events": log, **report})
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
