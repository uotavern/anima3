"""Explicit, bounded real-server learning. Checkpoints remain candidates."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from ..arena_learning import atomic_json


def update_collection(checkpoint, collection, out, *, seed=0):
    from .checkpoints import load_checkpoint
    from .live import LiveReplayVerifier
    from .train import TrainConfig, update_verified_episodes

    collection = Path(collection).resolve()
    report = json.loads((collection / "results.json").read_text())
    if report.get("domain") != "servuo":
        raise ValueError("expected a completed ServUO collection")
    loaded = load_checkpoint(checkpoint)
    episodes, rejected = [], []
    for match in report["matches"]:
        for filename in match["episodes"]:
            file = Path(filename).resolve()
            if not file.is_relative_to(collection):
                raise ValueError("episode path escaped collection directory")
            episode = json.loads(file.read_text())
            if not episode.get("eligible"):
                rejected.append({"file": str(file), "reasons": episode.get("invalid_reasons")})
            elif episode.get("policy_sha") != loaded.policy_sha:
                rejected.append({"file": str(file), "reasons": ["other_policy"]})
            else:
                episodes.append(episode)
    if not episodes:
        raise ValueError(f"no eligible on-policy episodes; rejected={rejected}")
    result = update_verified_episodes(
        loaded,
        episodes,
        verify_episode=LiveReplayVerifier(collection),
        out=out,
        config=TrainConfig(discount_time_unit=1.0),
        seed=seed,
    )
    summary = {
        "source": str(collection),
        "accepted_episodes": len(episodes),
        "rejected": rejected,
        "update": result,
        "automatic_promotion": False,
    }
    atomic_json(Path(out) / "live-update.json", summary)
    return summary


def update_main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    print(
        json.dumps(update_collection(args.checkpoint, args.collection, args.out, seed=args.seed)),
        flush=True,
    )
    return 0


def cycle_main(argv=None):
    from .live import add_arguments, run

    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not 1 <= args.iterations <= 100 or not 1 <= args.matches <= 100:
        ap.error("iterations and matches must each be 1..100")
    root = args.log_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "cycle.json").exists():
        ap.error("choose a new cycle directory")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    history = []
    try:
        for iteration in range(args.iterations):
            if stop.is_set():
                break
            # Reconnecting after each game must not reset challenger/spawn
            # alternation for every collection in the learning cycle.
            args.fixture_offset = iteration * args.matches
            args.log_dir = root / f"round-{iteration + 1:03d}" / "collection"
            atomic_json(
                root / "cycle.json",
                {
                    "history": history,
                    "automatic_promotion": False,
                    "status": "collecting",
                    "iteration": iteration + 1,
                },
            )
            matches = run(args, stop)
            if stop.is_set():
                break
            # The verified previous fixture establishes the build; lingering
            # Weaken/Clumsy must expire before reuse instead of requesting balls.
            args.known_build = bool(matches)
            out = args.log_dir.parent / "candidate"
            update = update_collection(
                args.checkpoint, args.log_dir, out, seed=args.seed + iteration
            )
            history.append(
                {
                    "iteration": iteration + 1,
                    "matches": matches,
                    "learning": update,
                    "checkpoint": str(out),
                }
            )
            # Every next batch uses the newly saved behavior policy.
            args.checkpoint = out
    except BaseException as exc:
        atomic_json(
            root / "cycle.json",
            {
                "history": history,
                "automatic_promotion": False,
                "status": "failed",
                "error": str(exc),
            },
        )
        raise
    atomic_json(
        root / "cycle.json",
        {
            "history": history,
            "automatic_promotion": False,
            "status": "stopped" if stop.is_set() else "complete",
        },
    )
    return 0
