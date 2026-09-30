"""Prepare a neural ordinary-client challenger for human-friendly 7x + ex-pot duels.

Human matches are never fed into PPO implicitly. The dedicated ``live``/``cycle``
commands collect explicitly unrated training fixtures instead.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path

from ..arena import append_event
from ..arena_learning import atomic_json
from ..contract import say
from ..duel_wait import RULES, prepare, pump
from ..sparring import connect_training, poll
from .inference import InferenceClient
from .live import _actor


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--host", default="arena.uotavern.com")
    ap.add_argument("--port", type=int, default=2593)
    ap.add_argument("--user", required=True)
    ap.add_argument("--password-env", default="ARENA_BOT_PASSWORD")
    ap.add_argument("--bridge")
    ap.add_argument("--data-dir")
    ap.add_argument("--log-dir", type=Path, default=Path(".logs/neural-wait"))
    ap.add_argument("--matches", type=int, default=1)
    ap.add_argument("--wait-seconds", type=int, default=3600)
    ap.add_argument("--deadline-ms", type=float, default=500)
    ap.add_argument("--no-list", action="store_true")
    args = ap.parse_args(argv)
    if args.matches < 1 or args.wait_seconds < 1 or not os.environ.get(args.password_env):
        ap.error("positive matches/wait-seconds and password environment variable required")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    log = args.log_dir.resolve()
    log.mkdir(parents=True, exist_ok=True)
    body = None
    worker = InferenceClient(args.checkpoint, deadline_ms=args.deadline_ms).start()
    try:
        body = connect_training(args, args.user, os.environ[args.password_env], log, stop.is_set)
        pump(body, 2)
        until = time.monotonic() + 90
        while not worker.ready:
            if stop.is_set() or worker.startup_error or time.monotonic() > until:
                raise RuntimeError("inference startup failed")
            body.pump(50)
        if poll(body).get("phase") != "Idle":
            raise RuntimeError("account already in a duel; refusing takeover")
        for game in range(args.matches):
            player = prepare(body, previously_configured=game > 0)
            body.act(say(f"[ArenaAgent neural {worker.policy_sha[:12]} recurrent"))
            pump(body, 0.6)
            if not args.no_list:
                body.act(say("[Arena list 7"))
                pump(body, 0.6)
            ready = {
                "stage": "ready",
                "serial": player["serial"],
                "name": player["name"],
                "rules": RULES,
                "ranked": False,
                "rounds": 1,
                "policy_sha": worker.policy_sha,
            }
            atomic_json(log / "status.json", ready)
            print(json.dumps(ready), flush=True)
            deadline = time.monotonic() + args.wait_seconds
            invite = None
            seen = set()
            while not stop.is_set() and time.monotonic() < deadline:
                state = poll(body)
                if state.get("phase") != "Idle":
                    raise RuntimeError("unexpected duel assigned while waiting")
                candidate = state.get("challenge")
                if candidate and candidate["id"] not in seen:
                    seen.add(candidate["id"])
                    accepted = (
                        candidate["rules"] == RULES
                        and not candidate.get("ranked")
                        and candidate.get("rounds") == 1
                    )
                    append_event(
                        log / "progress.jsonl",
                        {"stage": "challenge", "accepted": accepted, **candidate},
                    )
                    if accepted:
                        invite = candidate
                        break
                    body.act(say("I accept one-round friendly 7x + explosion potion duels."))
                body.pump(250)
            if invite is None:
                atomic_json(log / "status.json", {"stage": "waiting_ended", "completed": game})
                break
            body.act(say("[DuelAccept " + invite["id"]))
            result = _actor(body, invite["opponent"], worker, log, stop, expected_rules=RULES)
            # Retain diagnostic actions, not a 'verified' or training episode.
            atomic_json(log / result["id"] / "human-match.json", {**result, "training": False})
            atomic_json(
                log / "status.json",
                {"stage": "match_released", "id": result["id"], "completed": game + 1},
            )
    finally:
        worker.close()
        if body is not None:
            body.close()
    return 0
