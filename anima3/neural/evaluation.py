"""Fixed-budget evaluation statistics and a fail-closed model-selection gate.

Stdlib only; importing this module never imports torch, trains or promotes.

- An :class:`EvaluationPlan` fixes, before any result is read, the candidate and
  incumbent weight SHA-256s, the opponents, the number of games against each one
  and every acceptance threshold. Its SHA-256 identifies the decision.
- Each policy's evaluation against one opponent is one held-out unit-6
  ``evaluation`` collection whose own plan has exactly that budget and whose
  ledger has spent it. A training collection, a partially spent (interim)
  collection, a second collection for the same opponent, or more games than
  planned are refused, so the match count can never grow after looking at scores.
- Failed, invalid and unresolved fixtures stay in the record as excluded rows.
  They count against the budget and, beyond the plan's allowance, fail the gate.
- The gate needs a paired-bootstrap lower bound above the plan's threshold *and*
  no opponent whose mean score drops by more than ``max_regression``. An average
  improvement is not superiority against every opponent.
- Every decision, passed or rejected, is appended to a selection record. A
  passed gate is evidence for a separate, explicit decision; it never promotes
  or deploys a checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import replay
from ..stats import wilson
from .inference import require_sha256
from .live import TRAINING_RULES, LiveReplayVerifier, _fsync_dir
from .splits import SplitRefused, read_collection, read_episode

RECORD_SCHEMA = "anima3-evaluation-record-v1"
DECISION_SCHEMA = "anima3-selection-decision-v1"
DOMAINS = ("servuo", "approximate-pre-aos-simulator")
# Two-sided 95%: the exact normal quantile, not the rounded 1.96.
Z95 = 1.959963984540054
# Ported from the training-league gate: 2000 resamples, fixed seed 619, and the
# 50th/1951st order statistics as the two-sided 95% percentile interval.
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 619
MAX_GAMES_PER_OPPONENT = 100
MAX_OPPONENTS = 16
OPPONENT = re.compile(r"(scripted|neural:[0-9a-f]{64})")


class EvaluationRefused(ValueError):
    """The evidence cannot be evaluated under this plan; nothing was decided."""


# --- win-rate confidence ---------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, *, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for ``successes`` wins in ``trials`` games (95% by default)."""
    if type(successes) is not int or type(trials) is not int:
        raise ValueError("wins and games must be integers")
    if trials < 0 or not 0 <= successes <= max(trials, 0):
        raise ValueError("wins must be between 0 and the number of games")
    if not (isinstance(z, (int, float)) and math.isfinite(z) and z > 0):
        raise ValueError("z must be a positive finite number")
    return wilson(successes, trials, z)


def bootstrap_mean_interval(
    values: list[float], *, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Deterministic two-sided 95% percentile bootstrap of a mean (2000 resamples)."""
    if not values:
        raise ValueError("bootstrap needs at least one value")
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(BOOTSTRAP_SAMPLES))
    return means[49], means[1949]


# --- the fixed plan --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationPlan:
    """Everything a selection decision depends on, fixed before results are read."""

    candidate_sha256: str
    incumbent_sha256: str
    opponents: tuple[str, ...]
    games_per_opponent: int
    max_regression: float = 0.10
    min_lower_bound: float = 0.0
    max_excluded_per_opponent: int = 0
    min_paired_games_per_opponent: int = 20
    domain: str = "servuo"

    def __post_init__(self):
        require_sha256(self.candidate_sha256)
        require_sha256(self.incumbent_sha256)
        if self.candidate_sha256 == self.incumbent_sha256:
            raise ValueError("candidate and incumbent must be different checkpoints")
        if (
            not isinstance(self.opponents, tuple)
            or not 1 <= len(self.opponents) <= MAX_OPPONENTS
            or len(set(self.opponents)) != len(self.opponents)
            or not all(isinstance(o, str) and OPPONENT.fullmatch(o) for o in self.opponents)
        ):
            raise ValueError("opponents must be 1..16 distinct 'scripted'/'neural:<sha>' labels")
        if f"neural:{self.candidate_sha256}" in self.opponents:
            raise ValueError("the candidate cannot be its own evaluation opponent")
        games = self.games_per_opponent
        if type(games) is not int or not 2 <= games <= MAX_GAMES_PER_OPPONENT or games % 2:
            raise ValueError("games per opponent must be an even integer in 2..100")
        for name in ("max_regression", "min_lower_bound"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 <= self.max_regression <= 1 or not -1 <= self.min_lower_bound < 1:
            raise ValueError("regression limit must be in [0, 1] and lower bound in [-1, 1)")
        excluded = self.max_excluded_per_opponent
        if type(excluded) is not int or not 0 <= excluded < games:
            raise ValueError("excluded-game allowance must be an integer below the budget")
        minimum = self.min_paired_games_per_opponent
        if type(minimum) is not int or not 2 <= minimum <= games:
            raise ValueError("minimum paired games must be an integer in 2..games per opponent")
        if self.domain not in DOMAINS:
            raise ValueError("unknown evaluation domain")

    def to_json(self) -> dict:
        value = asdict(self)
        value["opponents"] = list(self.opponents)
        return value

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


# --- evaluation records from held-out live collections ---------------------------------------


def _opponent_label(arm: dict) -> str:
    if arm.get("kind") == "scripted":
        return "scripted"
    return f"neural:{require_sha256(arm.get('policy_sha256'))}"


def _score(winner, actor: int) -> float:
    return 0.5 if winner is None else (1.0 if winner == actor else 0.0)


def record_from_collection(directory, policy_sha: str) -> dict:
    """One policy's complete evaluation against one opponent, from one held-out collection.

    Completed fixtures become scored rows whose outcome is re-read from the stored,
    SHA-checked replay; ``valid`` additionally needs the full unit-6 episode
    verification. Every other ledger outcome stays as an excluded row.
    """
    policy_sha = require_sha256(policy_sha)
    record = read_collection(directory)
    if record.purpose != "evaluation":
        raise SplitRefused("only a held-out evaluation collection can be evaluated")
    arms = record.plan.get("arms")
    if not isinstance(arms, list) or len(arms) != 2 or not all(isinstance(a, dict) for a in arms):
        raise EvaluationRefused("collection plan lacks two arms")
    subject = [a.get("kind") == "neural" and a.get("policy_sha256") == policy_sha for a in arms]
    if subject.count(True) != 1:
        raise EvaluationRefused("the policy must hold exactly one seat of the collection")
    opponent = _opponent_label(arms[subject.index(False)])
    budget = record.plan.get("matches")
    ledger = record.ledger
    if type(budget) is not int or ledger.issued != budget or ledger.open_fixtures:
        raise EvaluationRefused("collection has not spent its fixed budget; interim results")
    issued = {r["fixture"]: r for r in ledger.entries if r["event"] == "issued"}
    closed = {r["fixture"]: r for r in ledger.entries if r["event"] != "issued"}
    verifier = LiveReplayVerifier(record.root, policy_sha)
    rows = []
    for fixture in range(budget):
        outcome = closed[fixture]
        row = {
            "opponent": opponent,
            "fixture": fixture,
            "status": "completed" if outcome["event"] == "completed" else "excluded",
            "event": outcome["event"],
            "match_id": outcome.get("match_id"),
        }
        if outcome["event"] != "completed":
            rows.append({**row, "score": None, "valid": False, "reason": outcome.get("reason")})
            continue
        own = []
        for name in outcome["episodes"]:
            _, episode = read_episode(record, outcome, name)
            if episode.get("policy_sha") == policy_sha:
                own.append(episode)
        if len(own) != 1:
            raise EvaluationRefused("a completed fixture needs exactly one episode of the policy")
        episode = own[0]
        actor = episode.get("actor")
        match_dir = record.root / outcome["match_id"]
        try:
            meta = json.loads((match_dir / "replay.meta.json").read_text())
            replay_rows = replay.validate((match_dir / "replay.jsonl").read_bytes(), meta)
        except (OSError, ValueError) as exc:
            raise EvaluationRefused(f"stored receipt failed validation: {exc}") from exc
        seats = [p.get("serial") for p in replay_rows[0].get("players", [])]
        if (
            meta.get("id") != outcome["match_id"]
            or meta.get("training") is not True
            or meta.get("rules") != TRAINING_RULES
            or type(actor) is not int
            or actor not in seats
        ):
            raise EvaluationRefused("stored receipt does not match the fixture and policy seat")
        rows.append(
            {
                **row,
                "score": _score(meta.get("winner"), actor),
                "side": "challenger" if issued[fixture].get("challenger") == actor else "receiver",
                "valid": verifier(episode),
                "invalid_reasons": list(episode.get("invalid_reasons") or []),
            }
        )
    return {
        "schema": RECORD_SCHEMA,
        "domain": "servuo",
        "policy_sha": policy_sha,
        "opponent": opponent,
        "collection": str(record.root),
        "collection_plan_sha256": record.plan_sha256,
        "budget": budget,
        "rows": rows,
    }


def combine_records(plan: EvaluationPlan, policy_sha: str, records: list[dict]) -> dict:
    """Join one policy's per-opponent records into the evaluation the plan names."""
    policy_sha = require_sha256(policy_sha)
    if policy_sha not in (plan.candidate_sha256, plan.incumbent_sha256):
        raise EvaluationRefused("policy is neither the plan's candidate nor its incumbent")
    by_opponent = {}
    for record in records:
        if record.get("schema") != RECORD_SCHEMA or record.get("policy_sha") != policy_sha:
            raise EvaluationRefused("record belongs to another policy or format")
        if record.get("domain") != plan.domain:
            raise EvaluationRefused("record domain differs from the plan")
        opponent = record.get("opponent")
        if opponent in by_opponent:
            raise EvaluationRefused("two records for one opponent; the budget is fixed")
        if record.get("budget") != plan.games_per_opponent:
            raise EvaluationRefused("collection budget differs from the plan")
        by_opponent[opponent] = record
    if set(by_opponent) != set(plan.opponents):
        raise EvaluationRefused("records must cover exactly the planned opponents")
    return {
        "schema": RECORD_SCHEMA,
        "domain": plan.domain,
        "policy_sha": policy_sha,
        "collections": {o: by_opponent[o].get("collection") for o in plan.opponents},
        "rows": [row for o in plan.opponents for row in by_opponent[o]["rows"]],
    }


# --- the gate --------------------------------------------------------------------------------


def _check_rows(plan: EvaluationPlan, evaluation: dict, expected_sha: str) -> dict:
    if evaluation.get("domain") != plan.domain:
        raise EvaluationRefused("cannot compare evaluations from different domains")
    if evaluation.get("policy_sha") != expected_sha:
        raise EvaluationRefused("evaluation policy SHA-256 differs from the plan")
    rows = evaluation.get("rows")
    if not isinstance(rows, list):
        raise EvaluationRefused("evaluation rows missing")
    keyed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("opponent") not in plan.opponents:
            raise EvaluationRefused("row for an unplanned opponent")
        fixture = row.get("fixture")
        if type(fixture) is not int or not 0 <= fixture < plan.games_per_opponent:
            raise EvaluationRefused("row fixture outside the fixed budget")
        key = (row["opponent"], fixture)
        if key in keyed:
            raise EvaluationRefused("duplicate evaluation fixture")
        status = row.get("status")
        if status == "completed":
            if row.get("score") not in (0, 0.5, 1) or type(row["score"]) not in (int, float):
                raise EvaluationRefused("evaluation scores must be loss/draw/win")
            if type(row.get("valid")) is not bool:
                raise EvaluationRefused("completed row must state its validity")
        elif status != "excluded":
            raise EvaluationRefused("row status must be completed or excluded")
        keyed[key] = row
    counts = {o: sum(1 for k in keyed if k[0] == o) for o in plan.opponents}
    if any(n != plan.games_per_opponent for n in counts.values()):
        raise EvaluationRefused("every opponent needs exactly its planned games, no more or fewer")
    return keyed


def _arm_summary(rows: list[dict]) -> dict:
    completed = [r for r in rows if r["status"] == "completed"]
    wins = sum(1 for r in completed if r["score"] == 1)
    draws = sum(1 for r in completed if r["score"] == 0.5)
    n = len(completed)
    low, high = wilson_interval(wins, n)
    return {
        "games": n,
        "wins": wins,
        "draws": draws,
        "losses": n - wins - draws,
        "excluded": len(rows) - n,
        "invalid": sum(1 for r in completed if not r["valid"]),
        "win_rate": wins / n if n else None,
        "win_rate_ci95": [low, high],
        "mean_score": sum(r["score"] for r in completed) / n if n else None,
    }


def selection_gate(plan: EvaluationPlan, incumbent: dict, candidate: dict) -> dict:
    """Decide whether the candidate beat the incumbent under the fixed plan. Fail closed.

    Rows are paired by (opponent, fixture): both policies faced the same opponent
    at the same fixture position and seat. The bootstrap resamples blocks of two
    consecutive fixtures (both seats stay together), as the source gate did with
    its seed blocks.
    """
    before = _check_rows(plan, incumbent, plan.incumbent_sha256)
    after = _check_rows(plan, candidate, plan.candidate_sha256)
    reasons = []
    opponents = {}
    blocks: dict[tuple[str, int], list[float]] = {}
    for opponent in plan.opponents:
        keys = [(opponent, f) for f in range(plan.games_per_opponent)]
        paired = [
            k
            for k in keys
            if before[k]["status"] == "completed" and after[k]["status"] == "completed"
        ]
        deltas = [after[k]["score"] - before[k]["score"] for k in paired]
        for k, delta in zip(paired, deltas, strict=True):
            blocks.setdefault((opponent, k[1] // 2), []).append(delta)
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        summary = {
            "incumbent": _arm_summary([before[k] for k in keys]),
            "candidate": _arm_summary([after[k] for k in keys]),
            "paired_games": len(paired),
            "mean_score_delta": mean_delta,
            "max_regression": plan.max_regression,
        }
        excluded = max(summary["incumbent"]["excluded"], summary["candidate"]["excluded"])
        if excluded > plan.max_excluded_per_opponent:
            reasons.append(f"excluded games beyond allowance against {opponent}")
        if summary["incumbent"]["invalid"] or summary["candidate"]["invalid"]:
            reasons.append(f"invalid episode evidence against {opponent}")
        if len(paired) < plan.min_paired_games_per_opponent:
            reasons.append(f"insufficient paired games against {opponent}")
        regressed = mean_delta is not None and mean_delta < -plan.max_regression - 1e-12
        if regressed:
            reasons.append(f"regression beyond limit against {opponent}")
        summary["regressed"] = regressed
        opponents[opponent] = summary
    block_means = [sum(v) / len(v) for v in blocks.values()]
    if block_means:
        lower, upper = bootstrap_mean_interval(block_means)
        paired_delta = sum(block_means) / len(block_means)
        if not lower > plan.min_lower_bound:
            reasons.append("paired improvement lower bound not above threshold")
    else:
        lower = upper = paired_delta = None
        reasons.append("no paired evidence")
    passed = not reasons
    return {
        "schema": DECISION_SCHEMA,
        "plan": plan.to_json(),
        "plan_sha256": plan.sha256,
        "domain": plan.domain,
        "candidate_sha256": plan.candidate_sha256,
        "incumbent_sha256": plan.incumbent_sha256,
        "passed": passed,
        "decision": "candidate_passed_gate" if passed else "retain_incumbent",
        "reasons": reasons,
        "paired_delta": paired_delta,
        "paired_delta_ci95": None if lower is None else [lower, upper],
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "blocks": len(blocks)},
        "opponents": opponents,
        "excluded_rows": [
            {"policy": label, **row}
            for label, keyed in (("incumbent", before), ("candidate", after))
            for row in keyed.values()
            if row["status"] == "excluded"
        ],
        "collections": {
            "incumbent": incumbent.get("collections"),
            "candidate": candidate.get("collections"),
        },
        "automatic_promotion": False,
        "public_deployment": False,
        "real_server_strength_established": False,
    }


# --- the selection record --------------------------------------------------------------------


def record_decision(path, decision: dict) -> dict:
    """Append a gate decision, passed or rejected, to an append-only JSONL record.

    A plan is decided once: a second decision for the same plan SHA-256 is refused,
    so a rejected plan cannot be silently re-decided. Rows are never rewritten.
    """
    if not isinstance(decision, dict) or decision.get("schema") != DECISION_SCHEMA:
        raise EvaluationRefused("not a selection decision")
    if decision.get("automatic_promotion") is not False or type(decision.get("passed")) is not bool:
        raise EvaluationRefused("a decision never promotes automatically")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        text = path.read_text()
        if text and not text.endswith("\n"):
            raise EvaluationRefused("torn final selection-record line; repair by hand")
        try:
            existing = [json.loads(line) for line in text.splitlines()]
        except ValueError:
            raise EvaluationRefused("corrupt selection record") from None
    if any(row.get("plan_sha256") == decision.get("plan_sha256") for row in existing):
        raise EvaluationRefused("this plan already has a recorded decision")
    row = {"sequence": len(existing), "utc": time.time(), **decision}
    created = not path.exists()
    line = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a") as out:
        out.write(line)
        out.flush()
        os.fsync(out.fileno())
    if created:
        _fsync_dir(path.parent)
    return row
