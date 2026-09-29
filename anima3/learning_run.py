"""Experiment identity, receipt integrity and observable online policy statistics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .arena_learning import atomic_json, eligible
from .magic import PLAYBOOKS
from .stats import wilson


def experiment(directory, args):
    """Resume only the same opponent, backend, rules and fixed evaluation protocol."""
    spec = {
        "schema": 1,
        "host": args.host,
        "port": args.port,
        "web": args.web,
        "users": [args.user_a, args.user_b],
        "backend": args.backend,
        "model": args.model if args.backend == "jev" else args.backend,
        "minimum": args.minimum,
        "sample": args.sample,
        "rules": "7x-magic-explosion-classic-training",
        "protocol": "playbook-v2",
    }
    if getattr(args, "brain", "direct") == "hybrid":
        spec.update(
            brain="hybrid-v1", llm_model=args.llm_model, promotion="disabled-adaptive-experience"
        )
    if getattr(args, "opponent", "same") != "same":
        spec["opponent"] = args.opponent
    if getattr(args, "opening", "none") != "none":
        spec["opening"] = args.opening
    if getattr(args, "burst_combo", False):
        spec["burstCombo"] = True
    path = directory / "experiment.json"
    if path.exists():
        if json.loads(path.read_text()) != spec:
            raise ValueError("Experiment settings changed; use a new log directory")
    elif (directory / "learning.jsonl").exists() and (directory / "learning.jsonl").stat().st_size:
        raise ValueError("Legacy logs need a separate new experiment directory")
    else:
        atomic_json(path, spec)
    events = directory / "learning.jsonl"
    if events.exists() and events.read_bytes() and not events.read_bytes().endswith(b"\n"):
        raise ValueError("Incomplete audit append: preserve and repair before resuming")


def verify_receipts(directory: Path, rows):
    """Recheck immutable server recordings before learning or promotion, including resumes."""
    for row in rows:
        mid = row["id"]
        if len(mid) != 32 or any(c not in "0123456789abcdef" for c in mid):
            raise ValueError("Invalid replay ID")
        raw = (directory / mid / "replay.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.get("sha256"):
            raise ValueError("Replay hash mismatch: " + mid)
        records = [json.loads(s) for s in raw.decode("utf-8-sig").splitlines()]
        h, end = records[0], records[-1]
        if (
            row.get("valid") is not True
            or row.get("training") is not True
            or row.get("aborted") is not False
            or row.get("source") != "verified-server-replay"
            or h.get("id") != mid
            or h.get("training") is not True
            or h.get("rules") != "7x-magic-explosion-classic-training"
            or [p["serial"] for p in h["players"]] != [row["a"], row["b"]]
            or end.get("type") != "end"
            or end.get("complete") is not True
            or end.get("aborted") is not None
            or end.get("winner") != row["winner"]
        ):
            raise ValueError("Replay outcome or training fixture mismatch: " + mid)


def progress(directory, rows, champion, minimum, clients=()):
    arms = {book: {"matches": 0, "wins": 0, "draws": 0} for book in PLAYBOOKS}
    for row in rows:
        if not eligible(row):
            continue
        for side, other in [("a", "b"), ("b", "a")]:
            book = row.get("playbook_" + side)
            if (
                book not in arms
                or row.get("policy_" + side) != "explore-" + book
                or row.get("policy_" + other) != champion["version"]
            ):
                continue
            a = arms[book]
            a["matches"] += 1
            a["wins"] += row["winner"] == row[side]
            a["draws"] += row["winner"] is None
    for a in arms.values():
        n, wins = a["matches"], a["wins"]
        a.update(
            posteriorWinMean=(wins + 1) / (n + 2),
            winInterval95=list(wilson(wins, n)),
            remaining=max(0, minimum - n),
        )
    result = {
        "schema": 1,
        "completed": len(rows),
        "champion": champion,
        "arms": arms,
        "providerCalls": [c.calls for c in clients],
        "providerErrors": [c.errors for c in clients],
        "learning": "Match-reward playbook selection; no foundation-model weight updates",
    }
    atomic_json(directory / "learning-progress.json", result)
    return result


def update(directory, minimum=10, sample=40, clients=()):
    from .arena import read_policy
    from .arena_learning import cycle, matches

    rows = matches(directory / "learning.jsonl")
    verify_receipts(directory, rows)
    policies = directory / "policies"
    champion = read_policy(policies / "champion.json")
    stats = progress(directory, rows, champion, minimum, clients)
    if (
        all(a["remaining"] == 0 for a in stats["arms"].values())
        or (policies / "candidate.json").exists()
    ):
        result = cycle(directory / "learning.jsonl", policies, minimum, sample, promote=True)
        progress(directory, rows, read_policy(policies / "champion.json"), minimum, clients)
        return result
    result = {
        "status": "collecting_training",
        "champion": champion["version"],
        "minimum_per_arm": minimum,
        "remaining": sum(a["remaining"] for a in stats["arms"].values()),
    }
    atomic_json(policies / "evaluation.json", result)
    return result
