"""Run a bounded real-server experiment and retain its result and model lineage.

Credentials stay in a private file outside the checkout. A failed collection
stops the experiment: restarting blindly could create additional server duels.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from anima3.arena_learning import atomic_json


def summarize(root: Path) -> dict:
    experiment = json.loads((root / "experiment.json").read_text())
    cycle_file = root / "cycle.json"
    cycle = json.loads(cycle_file.read_text()) if cycle_file.exists() else {}
    rows = []
    for file in sorted(root.glob("round-*/collection/results.json")):
        collection = json.loads(file.read_text())
        for match in collection["matches"]:
            episode = json.loads(Path(match["episodes"][0]).read_text())
            rows.append({
                "number": len(rows) + 1,
                "match_id": match["match_id"],
                "outcome": episode["final_result"]["outcome"],
                "eligible": episode["eligible"],
                "invalid_reasons": episode["invalid_reasons"],
                "transitions": len(episode["transitions"]),
                "policy_sha": episode["policy_sha"],
                "replay": match["replay"],
            })
    if len({row["match_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate match IDs in experiment")
    if len(rows) > experiment["target_matches"]:
        raise ValueError("experiment exceeded its match budget")
    history = cycle.get("history", [])
    result = {
        "status": cycle.get("status", "not_started"),
        "target_matches": experiment["target_matches"],
        "verified_matches": len(rows),
        "wins": sum(row["outcome"] == 1 for row in rows),
        "losses": sum(row["outcome"] == -1 for row in rows),
        "draws": sum(row["outcome"] == 0 for row in rows),
        "updates": sum(not h["learning"].get("skipped", False) for h in history),
        "skipped_updates": sum(h["learning"].get("skipped", False) for h in history),
        "initial_policy_sha": experiment["initial_policy_sha"],
        "final_checkpoint": history[-1]["checkpoint"] if history else experiment["checkpoint"],
        "automatic_promotion": False,
        "real_server_strength_improvement_established": False,
        "matches": rows,
        "error": cycle.get("error"),
    }
    atomic_json(root / "report.json", result)
    lines = [
        "# Arena 100-match training experiment", "",
        f"Status: {result['status']}",
        f"Verified: {len(rows)}/{result['target_matches']}",
        f"W/L/D: {result['wins']}/{result['losses']}/{result['draws']}",
        f"Verified PPO updates: {result['updates']}", "",
        "7x + explosion potion, unranked training. Challenger order alternates.",
        "Opponent is a fixed scripted agent. Public agents are not replaced.",
        "Training outcomes alone do not establish improved playing strength.", "",
        "| Match | Result | Eligible | Replay |",
        "|---|---|---|---|",
    ]
    for row in rows:
        label = {1: "Win", -1: "Loss", 0: "Draw"}[row["outcome"]]
        lines.append(
            f"| {row['number']} | {label} | {row['eligible']} | "
            f"[Watch]({row['replay']}) |"
        )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accounts-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-baseline", type=Path)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--user-a", default="arena_train_a")
    parser.add_argument("--user-b", default="arena_train_b")
    parser.add_argument("--matches-total", type=int, default=100)
    parser.add_argument("--fixture-start", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.matches_total <= 100 or args.fixture_start < 0:
        parser.error("matches-total must be 1..100 and fixture-start nonnegative")
    root = args.log_dir.resolve()
    if root.exists():
        parser.error("choose a new log directory; automatic restart is disabled")
    import torch

    from anima3.neural.checkpoints import load_checkpoint
    from anima3.neural.online import cycle_main
    from anima3.neural.train import evaluate

    torch.set_num_threads(1)
    torch.manual_seed(20261002)
    checkpoint = args.checkpoint.resolve()
    initial = load_checkpoint(checkpoint)
    accounts = json.loads(args.accounts_file.read_text())
    os.environ["SPAR_PASSWORD_A"] = accounts[args.user_a]
    os.environ["SPAR_PASSWORD_B"] = accounts[args.user_b]
    root.mkdir(parents=True, mode=0o700)
    atomic_json(root / "experiment.json", {
        "started_at": datetime.now(UTC).isoformat(),
        "target_matches": args.matches_total,
        "fixture_start": args.fixture_start,
        "checkpoint": str(checkpoint),
        "initial_policy_sha": initial.policy_sha,
        "rules": "standard7-explosion-training",
        "opponent": "fixed-scripted",
        "matches_per_connection": 1,
        "deadline_ms": 1000,
        "alternate_challenger": True,
        "eval_seed": 3501000000,
        "eval_games_per_checkpoint": 200,
        "automatic_promotion": False,
    })
    try:
        code = cycle_main([
            "--checkpoint", str(checkpoint), "--host", args.host,
            "--bridge", args.bridge, "--data-dir", args.data_dir,
            "--user-a", args.user_a, "--user-b", args.user_b,
            "--opponent", "scripted", "--matches", "1", "--iterations", str(args.matches_total),
            "--fixture-start", str(args.fixture_start),
            "--deadline-ms", "1000", "--seed", "20261002", "--log-dir", str(root),
        ])
    finally:
        result = summarize(root)
    if result["status"] != "complete" or result["verified_matches"] != args.matches_total:
        return 2
    # Evaluate the frozen original and final policies using the same new seeds.
    # These are simulator fixtures, not additional Arena matches.
    baseline = args.evaluation_baseline or checkpoint
    for label, path in (("before", baseline), ("after", result["final_checkpoint"])):
        loaded = load_checkpoint(path)
        evaluation = evaluate(
            loaded.model, games=200, seed=3501000000,
            max_episode_steps=1200, sampling="categorical",
        )
        atomic_json(root / f"evaluation-{label}.json", {
            **evaluation, "policy_sha": loaded.policy_sha,
            "domain": "approximate_simulator",
            "real_server_strength_improvement_established": False,
        })
    atomic_json(root / "finished.json", {
        "finished_at": datetime.now(UTC).isoformat(),
        "verified_matches": result["verified_matches"], "updates": result["updates"],
        "evaluation_complete": True, "automatic_promotion": False,
    })
    return code


if __name__ == "__main__":
    raise SystemExit(main())
