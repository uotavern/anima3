"""Explicit, bounded real-server learning. Checkpoints remain candidates."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from ..arena_learning import atomic_json


class NoEligibleEpisodes(ValueError):
    def __init__(self, rejected):
        self.rejected = rejected
        super().__init__(f"no eligible on-policy episodes; rejected={rejected}")


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
        raise NoEligibleEpisodes(rejected)
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
    from ..sparring import TrainingIdleTimeout
    from .live import add_arguments, run

    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixture-start", type=int, default=0)
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
            args.fixture_offset = args.fixture_start + iteration * args.matches
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
            for attempt in range(3):
                try:
                    matches = run(args, stop)
                    break
                except TrainingIdleTimeout:
                    # run() has closed both clients. This exception is only
                    # raised before issuing a challenge, never during a fight.
                    # Keep the failed handshake evidence and the same fixture.
                    if stop.is_set() or attempt == 2:
                        raise
                    args.log_dir.rename(args.log_dir.with_name(f"idle-retry-{attempt + 1}"))
                    if stop.wait(5):
                        raise RuntimeError("stopped before handshake retry") from None
            if stop.is_set():
                break
            # The verified previous fixture establishes the build; lingering
            # Weaken/Clumsy must expire before reuse instead of requesting balls.
            args.known_build = bool(matches)
            out = args.log_dir.parent / "candidate"
            try:
                update = update_collection(
                    args.checkpoint, args.log_dir, out, seed=args.seed + iteration
                )
            except NoEligibleEpisodes as exc:
                # A verified completed fixture still consumes the match budget.
                # Keep its rejected trajectory and use the unchanged policy.
                update = {"accepted_episodes": 0, "rejected": exc.rejected,
                          "skipped": True, "automatic_promotion": False}
                atomic_json(args.log_dir.parent / "learning-skipped.json", update)
                out = args.checkpoint
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


def update_collections(checkpoint, collections, out, *, seed=0, min_episodes=4):
    """One frozen behavior policy across independently verified league fixtures."""
    from .checkpoints import load_checkpoint
    from .live import LiveReplayVerifier
    from .train import TrainConfig, update_verified_episodes

    if min_episodes < 1:
        raise ValueError("positive minimum batch required")
    loaded = load_checkpoint(checkpoint)
    episodes, rejected, verifiers = [], [], {}
    for collection in collections:
        root = Path(collection).resolve()
        report = json.loads((root / "results.json").read_text())
        if report.get("domain") != "servuo" or report.get("collection_status") != "complete":
            raise ValueError("all collections must have completed normally")
        verifier = LiveReplayVerifier(root)
        for match in report["matches"]:
            for name in match["episodes"]:
                file = Path(name).resolve()
                if not file.is_relative_to(root):
                    raise ValueError("episode escaped collection")
                episode = json.loads(file.read_text())
                if episode.get("policy_sha") != loaded.policy_sha:
                    continue  # historical opponent data is off-policy
                if not episode.get("eligible"):
                    rejected.append({"file": str(file), "reasons": episode.get("invalid_reasons")})
                    continue
                key = (episode["verification"]["match_id"], episode.get("player_serial", episode.get("actor")))
                if key in verifiers:
                    raise ValueError("duplicate episode in batch")
                verifiers[key] = verifier
                episodes.append(episode)
    if len(episodes) < min_episodes:
        raise NoEligibleEpisodes(rejected + [{"reason": "below minimum batch", "count": len(episodes)}])
    def verify(episode):
        return verifiers[(episode["verification"]["match_id"], episode.get("player_serial", episode.get("actor")))](episode)
    result = update_verified_episodes(
        loaded, episodes, verify_episode=verify, out=out,
        config=TrainConfig(learning_rate=1e-4, discount_time_unit=1.0), seed=seed)
    summary = {"accepted_episodes": len(episodes), "rejected": rejected,
               "collections": [str(p) for p in collections], "update": result,
               "automatic_promotion": False}
    atomic_json(Path(out) / "batch-update.json", summary)
    return summary
