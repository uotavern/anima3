"""Learn a mage playbook from authoritative matches, evaluate, and promote atomically.

Training uses exploration matches. A candidate is frozen and then evaluated on
new head-to-head matches against the current champion. Public players always
face the champion; their matches never enter the promotion test. This is a
finite-action policy learner, not a claim that every new policy is stronger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from .arena import read_policy
from .magic import PLAYBOOKS
from .stats import wilson

BASELINE = {"schema": 1, "build": "mage", "version": "baseline-v1", "playbook": "standard"}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def matches(path: Path) -> list[dict]:
    """Deduplicate IDs; fail on corruption, but ignore an incomplete final append."""
    found = {}
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                break
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ValueError(f"invalid audit JSON on line {number}") from e
            if row.get("event") != "match_end":
                continue
            if row.get("schema") != 1 or not isinstance(row.get("id"), str):
                raise ValueError(f"unsupported match on line {number}")
            if row["id"] in found and found[row["id"]] != row:
                raise ValueError("conflicting duplicate match: " + row["id"])
            found[row["id"]] = row
    return list(found.values())


def eligible(row: dict) -> bool:
    return (row.get("valid") is True and row.get("aborted") is False
            and not row.get("modelBudgetExhausted", False)
            and row.get("brain", "direct") == "direct"
            and row.get("training") is True and row.get("build") == "mage"
            and type(row.get("a")) is int and type(row.get("b")) is int and row["a"] != row["b"]
            and row.get("winner") in (row["a"], row["b"], None))


def train(rows: list[dict], champion: dict, minimum: int = 20) -> dict | None:
    if minimum < 10:
        raise ValueError("at least 10 training matches per arm are required")
    arms = {book: {"wins": 0, "games": 0, "ids": []} for book in PLAYBOOKS}
    for row in rows:
        if not eligible(row):
            continue
        for side, other in (("a", "b"), ("b", "a")):
            book = row.get("playbook_" + side)
            # Only exploration against this exact champion constitutes training data.
            if (book not in arms or row.get("policy_" + side) != "explore-" + book
                    or row.get("policy_" + other) != champion["version"]):
                continue
            arm = arms[book]
            arm["games"] += 1
            arm["wins"] += row["winner"] == row[side]  # draws conservatively count as non-wins
            arm["ids"].append(row["id"])
    tested = {book: arm for book, arm in arms.items() if arm["games"] >= minimum}
    if not tested:
        return None
    book = max(tested, key=lambda b: (wilson(tested[b]["wins"], tested[b]["games"])[0], b))
    if book == champion["playbook"]:
        return None
    evidence = sorted({mid for arm in arms.values() for mid in arm["ids"]})
    digest = hashlib.sha256(json.dumps([champion["version"], book, evidence]).encode()).hexdigest()[:16]
    return {"schema": 1, "build": "mage", "version": "candidate-" + digest, "playbook": book,
            "parent": champion["version"], "training_ids": evidence,
            **{k: champion[k] for k in ("backend", "model") if k in champion},
            "training": {b: {k: v for k, v in a.items() if k != "ids"} for b, a in arms.items()}}


def evaluate(rows: list[dict], candidate: dict, champion: dict, sample: int = 100) -> dict:
    if sample < 40:
        raise ValueError("evaluation requires a fixed sample of at least 40 matches")
    training_ids = set(candidate.get("training_ids", []))
    test = []
    for row in rows:
        if not eligible(row) or row["id"] in training_ids:
            continue
        policies = (row.get("policy_a"), row.get("policy_b"))
        if policies not in ((candidate["version"], champion["version"]), (champion["version"], candidate["version"])):
            continue
        side = "a" if policies[0] == candidate["version"] else "b"
        other = "b" if side == "a" else "a"
        if row.get("playbook_" + side) != candidate["playbook"] or row.get("playbook_" + other) != champion["playbook"]:
            continue
        test.append((row, side))
        if len(test) == sample:
            break  # fixed first N, never keep sampling the same candidate until a lucky pass
    wins = sum(row["winner"] == row[side] for row, side in test)
    draws = sum(row["winner"] is None for row, _ in test)
    sides = {s: sum(side == s for _, side in test) for s in ("a", "b")}
    lower, upper = wilson(wins, len(test), z=2.576)  # 99% interval; draws non-wins
    valid_parent = candidate.get("parent") == champion["version"]
    passed = valid_parent and len(test) == sample and min(sides.values()) >= sample // 3 and lower > 0.5
    return {"candidate": candidate["version"], "champion": champion["version"], "sample": sample,
            "matches": len(test), "wins": wins, "draws": draws, "sides": sides,
            "win_interval_99": [lower, upper], "passed": passed, "parent_matches": valid_parent,
            "evaluation_ids": [row["id"] for row, _ in test]}


def cycle(events: Path, directory: Path, minimum: int = 20, sample: int = 100, promote: bool = False) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    champion_path = directory / "champion.json"
    if not champion_path.exists():
        atomic_json(champion_path, BASELINE)
    champion = read_policy(champion_path)
    rows = matches(events)
    candidate_path = directory / "candidate.json"
    candidate = read_policy(candidate_path) if candidate_path.exists() else None
    replace_candidate = candidate is None or candidate.get("parent") != champion["version"]
    if not replace_candidate:
        previous = evaluate(rows, candidate, champion, candidate.get("evaluation_sample", sample))
        # After a failed frozen evaluation, gather a whole new training batch before
        # proposing another version. Old evaluation games never become training games.
        if previous["matches"] == previous["sample"] and not previous["passed"]:
            proposed = train(rows, champion, minimum)
            if proposed and len(set(proposed["training_ids"]) - set(candidate.get("training_ids", []))) >= minimum * len(PLAYBOOKS):
                atomic_json(directory / "history" / (candidate["version"] + "-evaluation.json"), previous)
                replace_candidate = True
    if replace_candidate:
        candidate = train(rows, champion, minimum)
        if candidate is None:
            report = {"status": "collecting_training", "champion": champion["version"], "minimum_per_arm": minimum}
            atomic_json(directory / "evaluation.json", report)
            return report
        candidate["evaluation_sample"] = sample
        atomic_json(candidate_path, candidate)
    report = evaluate(rows, candidate, champion, candidate.get("evaluation_sample", sample))
    report["status"] = "passed" if report["passed"] else "rejected" if report["matches"] == report["sample"] else "evaluating"
    atomic_json(directory / "evaluation.json", report)
    if report["passed"] and promote:
        # Preserve the whole policy and proof for review/rollback. A failed candidate
        # stays frozen so repeated watch cycles cannot fish for a different sample.
        atomic_json(directory / "history" / (champion["version"] + ".json"), champion)
        atomic_json(directory / "history" / (candidate["version"] + "-evaluation.json"), report)
        atomic_json(champion_path, {**candidate, "evaluation": report})
        report["status"] = "promoted"
        atomic_json(directory / "evaluation.json", report)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True, type=Path, help="trusted ServUO Logs/Arena/events.jsonl")
    ap.add_argument("--policies", type=Path, default=Path(".logs/arena-policies"))
    ap.add_argument("--minimum", type=int, default=20)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--promote", action="store_true", help="apply only a candidate that passed the fixed evaluation")
    ap.add_argument("--watch", action="store_true")
    args = ap.parse_args(argv)
    while True:
        print(json.dumps(cycle(args.events, args.policies, args.minimum, args.sample, args.promote)), flush=True)
        if not args.watch:
            break
        time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
