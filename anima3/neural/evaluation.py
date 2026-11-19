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
  They count against the budget and are scored against the candidate: an
  excluded candidate game is a loss, an excluded incumbent game earns the
  candidate nothing. Beyond the plan's allowance (at most a tenth of the games)
  they fail the gate.
- The gate needs at least 20 paired games per opponent, at least ten complete
  two-seat blocks, a paired-bootstrap lower bound above the plan's threshold
  *and* no opponent whose mean score drops by more than ``max_regression``. An
  average improvement is not superiority against every opponent. Games are only
  paired from the same seat.
- Every decision, passed or rejected, is appended to the lineage's selection
  record after being re-derived from its collections under the lineage lock.
  Exclusive claim files in the collections, the candidate checkpoint directory
  and the lineage make evidence and the candidate/incumbent pair single-use
  across all lineages, and a trained candidate is bound to the lineage named in
  its manifest. A passed gate is
  evidence for a separate, explicit decision; it never promotes or deploys a
  checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass, fields
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
# sorted resample means at indices 49 and 1949 (the 50th and 1950th order
# statistics) as the two-sided 95% percentile interval.
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 619
# Evidence floors: a plan cannot be smaller than this, and the bootstrap needs at
# least MIN_BLOCKS complete two-seat blocks before any decision can pass.
MIN_GAMES_PER_OPPONENT = 20
MIN_BLOCKS = 10
THRESHOLD_STEP = 0.01
SIDES = ("challenger", "receiver")
SELECTION_RECORD = "selection.jsonl"
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_CLAIM_BYTES = 64 * 1024
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


def _grid(name: str, value) -> float:
    """Thresholds live on a 0.01 grid, so a tiny nudge cannot mint a new plan identity."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    steps = round(value / THRESHOLD_STEP)
    if abs(value / THRESHOLD_STEP - steps) > 1e-6:
        raise ValueError(f"{name} must be a multiple of {THRESHOLD_STEP}")
    return float(round(steps * THRESHOLD_STEP, 2)) + 0.0  # never -0.0


@dataclass(frozen=True)
class EvaluationPlan:
    """Everything a selection decision depends on, fixed before results are read.

    The plan is canonical: opponents are stored sorted and thresholds on a 0.01
    grid, so reordering or nudging them cannot produce a second identity for the
    same comparison. Every count has a floor: at least 20 games and 20 paired
    games per opponent, and an exclusion allowance of at most a tenth of them.
    """

    candidate_sha256: str
    incumbent_sha256: str
    opponents: tuple[str, ...]
    games_per_opponent: int
    max_regression: float = 0.10
    min_lower_bound: float = 0.0
    max_excluded_per_opponent: int = 0
    min_paired_games_per_opponent: int = MIN_GAMES_PER_OPPONENT
    domain: str = "servuo"

    def __post_init__(self):
        require_sha256(self.candidate_sha256)
        require_sha256(self.incumbent_sha256)
        if self.candidate_sha256 == self.incumbent_sha256:
            raise ValueError("candidate and incumbent must be different checkpoints")
        opponents = self.opponents
        if (
            not isinstance(opponents, (tuple, list))
            or not 1 <= len(opponents) <= MAX_OPPONENTS
            or not all(isinstance(o, str) and OPPONENT.fullmatch(o) for o in opponents)
            or len(set(opponents)) != len(opponents)
        ):
            raise ValueError("opponents must be 1..16 distinct 'scripted'/'neural:<sha>' labels")
        object.__setattr__(self, "opponents", tuple(sorted(opponents)))
        if f"neural:{self.candidate_sha256}" in self.opponents:
            raise ValueError("the candidate cannot be its own evaluation opponent")
        if f"neural:{self.incumbent_sha256}" in self.opponents:
            raise ValueError("the incumbent cannot also be an evaluation opponent")
        games = self.games_per_opponent
        if (
            type(games) is not int
            or not MIN_GAMES_PER_OPPONENT <= games <= MAX_GAMES_PER_OPPONENT
            or games % 2
        ):
            raise ValueError("games per opponent must be an even integer in 20..100")
        regression = _grid("max_regression", self.max_regression)
        lower = _grid("min_lower_bound", self.min_lower_bound)
        if not 0 <= regression <= 1 or not -1 <= lower < 1:
            raise ValueError("regression limit must be in [0, 1] and lower bound in [-1, 1)")
        object.__setattr__(self, "max_regression", regression)
        object.__setattr__(self, "min_lower_bound", lower)
        excluded = self.max_excluded_per_opponent
        if type(excluded) is not int or not 0 <= excluded <= games // 10:
            raise ValueError("excluded-game allowance must be an integer of at most games/10")
        minimum = self.min_paired_games_per_opponent
        if type(minimum) is not int or not MIN_GAMES_PER_OPPONENT <= minimum <= games:
            raise ValueError("minimum paired games must be an integer in 20..games per opponent")
        if self.domain not in DOMAINS:
            raise ValueError("unknown evaluation domain")

    def to_json(self) -> dict:
        value = asdict(self)
        value["opponents"] = list(self.opponents)
        return value

    @classmethod
    def from_json(cls, value) -> EvaluationPlan:
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError("not an evaluation plan")
        return cls(**{**value, "opponents": tuple(value["opponents"])})

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


# --- evaluation records from held-out live collections ---------------------------------------


def _opponent_label(arm: dict) -> str:
    if arm.get("kind") == "scripted":
        return "scripted"
    if arm.get("kind") != "neural":
        # A teacher or probe is a scripted controller: it is never scored under
        # a neural label, and the fixed plan has no label for it yet.
        raise EvaluationRefused(f"a {arm.get('kind')!r} arm is not an evaluation opponent")
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
        # Identical plans have identical SHA-256s, so the evidence itself (the
        # ledger and its server match identities) identifies a collection.
        "ledger_sha256": hashlib.sha256((record.root / "ledger.jsonl").read_bytes()).hexdigest(),
        "match_ids": sorted(ledger.match_ids),
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
        "collections": {
            o: {
                "path": by_opponent[o].get("collection"),
                "plan_sha256": by_opponent[o].get("collection_plan_sha256"),
                "ledger_sha256": by_opponent[o].get("ledger_sha256"),
                "match_ids": by_opponent[o].get("match_ids"),
            }
            for o in plan.opponents
        },
        "rows": [row for o in plan.opponents for row in by_opponent[o]["rows"]],
    }


def _check_rows(plan: EvaluationPlan, evaluation: dict, expected_sha: str) -> dict:
    if not isinstance(evaluation, dict) or evaluation.get("domain") != plan.domain:
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
            if row.get("side") not in SIDES:
                raise EvaluationRefused("completed row must state its seat")
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

    Rows pair by (opponent, fixture): both policies faced the same opponent at the
    same fixture position, and the pair must also share its seat (challenger or
    receiver), or the evidence is refused. Exclusions are scored against the
    candidate: an excluded candidate game counts as a candidate loss, and an
    excluded incumbent game drops that fixture (the candidate gets no credit for
    it). Each opponent's fixtures 2k and 2k+1 form a block that must hold both
    seats; as in the source gate only complete two-delta blocks enter the
    bootstrap, so every resampled block weighs the same. Fewer than ten complete
    blocks in total is too little evidence to decide, and the incumbent is kept.
    """
    before = _check_rows(plan, incumbent, plan.incumbent_sha256)
    after = _check_rows(plan, candidate, plan.candidate_sha256)
    reasons = []
    opponents = {}
    block_means: list[float] = []
    for opponent in plan.opponents:
        keys = [(opponent, f) for f in range(plan.games_per_opponent)]
        deltas: dict[tuple[str, int], float] = {}
        both_completed = 0
        excluded_fixtures = 0
        for key in keys:
            b, a = before[key], after[key]
            if b["status"] == "excluded" or a["status"] == "excluded":
                excluded_fixtures += 1
            if b["status"] == "excluded":
                continue  # no credit for the incumbent's failure
            if a["status"] == "completed":
                if a["side"] != b["side"]:
                    raise EvaluationRefused("paired games were played from different seats")
                both_completed += 1
                deltas[key] = a["score"] - b["score"]
            else:
                deltas[key] = 0.0 - b["score"]  # the candidate's failure is its loss
        for block in range(plan.games_per_opponent // 2):
            first, second = (opponent, 2 * block), (opponent, 2 * block + 1)
            if first in deltas and second in deltas:
                if before[first]["side"] == before[second]["side"]:
                    raise EvaluationRefused("a fixture block must hold both seats")
                block_means.append((deltas[first] + deltas[second]) / 2)
        mean_delta = sum(deltas.values()) / len(deltas) if deltas else None
        regressed = mean_delta is None or mean_delta < -plan.max_regression - 1e-12
        summary = {
            "incumbent": _arm_summary([before[k] for k in keys]),
            "candidate": _arm_summary([after[k] for k in keys]),
            "paired_games": both_completed,
            "scored_fixtures": len(deltas),
            "excluded_fixtures": excluded_fixtures,
            "mean_score_delta": mean_delta,
            "max_regression": plan.max_regression,
            "regressed": regressed,
        }
        if excluded_fixtures > plan.max_excluded_per_opponent:
            reasons.append(f"excluded games beyond allowance against {opponent}")
        if summary["incumbent"]["invalid"] or summary["candidate"]["invalid"]:
            reasons.append(f"invalid episode evidence against {opponent}")
        if both_completed < plan.min_paired_games_per_opponent:
            reasons.append(f"insufficient paired games against {opponent}")
        if regressed:
            reasons.append(f"regression beyond limit against {opponent}")
        opponents[opponent] = summary
    if len(block_means) >= MIN_BLOCKS:
        lower, upper = bootstrap_mean_interval(block_means)
        paired_delta = sum(block_means) / len(block_means)
        if not lower > plan.min_lower_bound:
            reasons.append("paired improvement lower bound not above threshold")
    else:
        lower = upper = paired_delta = None
        reasons.append("too few complete blocks to decide")
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
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "blocks": len(block_means),
            "min_blocks": MIN_BLOCKS,
        },
        "opponents": opponents,
        "excluded_rows": [
            {"policy": label, **row}
            for label, keyed in (("incumbent", before), ("candidate", after))
            for row in keyed.values()
            if row["status"] == "excluded"
        ],
        "evidence": {
            "incumbent": incumbent.get("collections"),
            "candidate": candidate.get("collections"),
        },
        "external_candidate": False,
        "automatic_promotion": False,
        "public_deployment": False,
        "real_server_strength_established": False,
    }


def decide_from_collections(
    plan: EvaluationPlan, *, incumbent, candidate, external_candidate: bool = False
) -> dict:
    """Re-derive both evaluations from their held-out collection directories and gate them.

    ``external_candidate`` declares that the candidate checkpoint was not produced
    by ``train_from_collections`` and so carries no lineage binding.
    """
    if type(external_candidate) is not bool:
        raise EvaluationRefused("external_candidate must be True or False")
    evaluations = {}
    for sha, directories in (
        (plan.incumbent_sha256, incumbent),
        (plan.candidate_sha256, candidate),
    ):
        directories = [Path(d) for d in directories]
        records = [record_from_collection(d, sha) for d in directories]
        evaluations[sha] = combine_records(plan, sha, records)
    ledgers = [c["ledger_sha256"] for e in evaluations.values() for c in e["collections"].values()]
    if len(set(ledgers)) != len(ledgers):
        raise EvaluationRefused("one collection cannot serve two evaluations")
    decision = selection_gate(
        plan, evaluations[plan.incumbent_sha256], evaluations[plan.candidate_sha256]
    )
    decision["external_candidate"] = external_candidate
    return decision


# --- the selection record --------------------------------------------------------------------


def _evidence_paths(decision: dict, plan: EvaluationPlan, arm: str) -> list[str]:
    evidence = decision.get("evidence")
    named = evidence.get(arm) if isinstance(evidence, dict) else None
    if not isinstance(named, dict) or set(named) != set(plan.opponents):
        raise EvaluationRefused("a recorded decision must name its collections")
    paths = []
    for opponent in plan.opponents:
        entry = named[opponent]
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise EvaluationRefused("a recorded decision must name its collections")
        paths.append(entry["path"])
    return paths


def _evidence_used(row: dict) -> set[tuple[str, str]]:
    """Every collection ledger and server match a decision's evidence rests on."""
    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        raise EvaluationRefused("selection record row lacks its evidence")
    used = set()
    for named in evidence.values():
        if not isinstance(named, dict):
            raise EvaluationRefused("selection record row lacks its evidence")
        for entry in named.values():
            ledger = entry.get("ledger_sha256") if isinstance(entry, dict) else None
            plan_sha = entry.get("plan_sha256") if isinstance(entry, dict) else None
            matches = entry.get("match_ids") if isinstance(entry, dict) else None
            if not (
                isinstance(ledger, str)
                and isinstance(plan_sha, str)
                and isinstance(matches, list)
                and all(isinstance(m, str) for m in matches)
            ):
                raise EvaluationRefused("selection record row lacks its evidence")
            used.add(("ledger", ledger))
            used |= {("match", m) for m in matches}
    return used


def _validate_decision(decision) -> EvaluationPlan:
    if not isinstance(decision, dict) or decision.get("schema") != DECISION_SCHEMA:
        raise EvaluationRefused("not a selection decision")
    if decision.get("automatic_promotion") is not False or type(decision.get("passed")) is not bool:
        raise EvaluationRefused("a decision never promotes automatically")
    try:
        plan = EvaluationPlan.from_json(decision.get("plan"))
    except (TypeError, ValueError) as exc:
        raise EvaluationRefused(f"decision plan is invalid: {exc}") from exc
    if (
        plan.to_json() != decision["plan"]
        or plan.sha256 != decision.get("plan_sha256")
        or decision.get("candidate_sha256") != plan.candidate_sha256
        or decision.get("incumbent_sha256") != plan.incumbent_sha256
    ):
        raise EvaluationRefused("decision does not match its re-hashed plan")
    if type(decision.get("external_candidate")) is not bool:
        raise EvaluationRefused("decision must state whether the candidate is external")
    reasons = decision.get("reasons")
    expected = "candidate_passed_gate" if decision["passed"] else "retain_incumbent"
    if (
        not isinstance(reasons, list)
        or decision["passed"] != (not reasons)
        or decision.get("decision") != expected
    ):
        raise EvaluationRefused("decision outcome is inconsistent with its reasons")
    return plan


def _candidate_checkpoints(
    plan: EvaluationPlan, directories: list[str]
) -> tuple[list[Path], str | None]:
    """Reach the candidate checkpoint by SHA-256 through each collection's arm."""
    found = []
    for directory in directories:
        arms = read_collection(directory).plan.get("arms") or []
        paths = [
            a.get("checkpoint")
            for a in arms
            if isinstance(a, dict) and a.get("policy_sha256") == plan.candidate_sha256
        ]
        if len(paths) != 1 or not isinstance(paths[0], str):
            raise EvaluationRefused("collection does not name the candidate checkpoint")
        if not Path(paths[0]).is_absolute():
            # A relative path would resolve against whatever the current directory is.
            raise EvaluationRefused("collection names the candidate checkpoint by a relative path")
        root = Path(paths[0]).resolve()
        root = root if root.is_dir() else root.parent
        try:
            with (root / "manifest.json").open("rb") as stream:
                payload = stream.read(MAX_MANIFEST_BYTES + 1)
            if len(payload) > MAX_MANIFEST_BYTES:
                raise ValueError("checkpoint manifest exceeds size limit")
            manifest = json.loads(payload)
            weights = manifest["weights"]
            name = weights["file"]
            if Path(name).name != name or weights["sha256"] != plan.candidate_sha256:
                raise ValueError("weights are not the candidate")
            digest = hashlib.sha256()
            with (root / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise EvaluationRefused(f"cannot reach the candidate checkpoint by SHA: {exc}") from exc
        if digest.hexdigest() != plan.candidate_sha256:
            raise EvaluationRefused("candidate checkpoint weights do not hash to its SHA-256")
        found.append((root, manifest))
    roots = {root for root, _ in found}
    lineages = {json.dumps((m.get("training") or {}).get("lineage_dir")) for _, m in found}
    if len(lineages) != 1:
        raise EvaluationRefused("candidate checkpoint copies disagree on their lineage")
    return sorted(roots), json.loads(lineages.pop())


def _claim(path: Path, content: dict, created: list[Path]) -> None:
    """Create ``path`` exclusively and durably, remembering it for rollback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise EvaluationRefused(f"evidence already claimed by a decision: {path}") from None
    created.append(path)
    with os.fdopen(fd, "w") as out:
        out.write(json.dumps(content, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())
    _fsync_dir(path.parent)


def _release(created: list[Path]) -> list[str]:
    """Remove claims created by a failed decision; return any that could not be removed."""
    remaining = []
    for path in reversed(created):
        try:
            path.unlink()
            _fsync_dir(path.parent)
        except FileNotFoundError:
            pass
        except OSError as exc:
            remaining.append(f"{path} ({type(exc).__name__})")
    return remaining


def _read_claim(path: Path):
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_CLAIM_BYTES + 1)
        return json.loads(payload) if len(payload) <= MAX_CLAIM_BYTES else None
    except (OSError, ValueError):
        return None


def record_decision(lineage_dir, decision: dict) -> dict:
    """Append a gate decision, passed or rejected, to the lineage's selection record.

    The record is ``<lineage_dir>/selection.jsonl``. Under an exclusive lock on
    the lineage, the decision's plan is re-hashed, its outcome checked against its
    reasons, and the whole decision re-derived from the named held-out
    collections; it must be identical. Then, all or nothing, exclusive claim
    files are created:

    - ``selection-claim.json`` in every evaluation collection directory used,
      so the same evidence can never be decided again in any lineage;
    - ``selection-claims/<incumbent>.json`` in the candidate checkpoint
      directory, so the same candidate/incumbent pair can never be decided
      again in any lineage, even with fresh collections;
    - ``claims/<sequence>.json`` in the lineage directory, one per record row,
      so a deleted or truncated ``selection.jsonl`` no longer matches its claims.

    The candidate checkpoint is reached by SHA-256 through the collections' arms.
    If its manifest names a ``lineage_dir`` (``train_from_collections`` records
    one), only that lineage may record it; a candidate without one must be
    declared ``external_candidate`` in the decision. A second decision for the
    same pair, or one reusing a collection ledger or server match, is also
    refused within the record itself. Rows are never rewritten.
    """
    import fcntl

    plan = _validate_decision(decision)
    root = Path(lineage_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / SELECTION_RECORD
    with (root / f"{SELECTION_RECORD}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise EvaluationRefused("another writer holds the selection record") from None
        incumbent_dirs = _evidence_paths(decision, plan, "incumbent")
        candidate_dirs = _evidence_paths(decision, plan, "candidate")
        recomputed = decide_from_collections(
            plan,
            incumbent=incumbent_dirs,
            candidate=candidate_dirs,
            external_candidate=decision["external_candidate"],
        )
        if recomputed != decision:
            raise EvaluationRefused("decision differs from the one its collections give")
        checkpoints, bound = _candidate_checkpoints(plan, candidate_dirs)
        if bound is None:
            if decision["external_candidate"] is not True:
                raise EvaluationRefused(
                    "candidate checkpoint names no lineage; declare external_candidate"
                )
        elif decision["external_candidate"] or Path(bound).resolve() != root:
            raise EvaluationRefused("candidate checkpoint is bound to another lineage")
        existing = []
        if path.exists():
            text = path.read_text()
            if text and not text.endswith("\n"):
                raise EvaluationRefused("torn final selection-record line; repair by hand")
            for line in text.splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    raise EvaluationRefused("corrupt selection record") from None
                if not isinstance(row, dict):
                    raise EvaluationRefused("selection record line is not an object")
                existing.append(row)
        claims_dir = root / "claims"
        claimed = sorted(p.name for p in claims_dir.glob("*.json")) if claims_dir.is_dir() else []
        sequence = len(existing)
        rows_claimed = [f"{i:06d}.json" for i in range(sequence)]
        if [r.get("sequence") for r in existing] != list(range(sequence)) or claimed not in (
            rows_claimed,
            [*rows_claimed, f"{sequence:06d}.json"],
        ):
            raise EvaluationRefused("selection record was deleted, truncated or reordered")
        used = set()
        for row in existing:
            if (row.get("candidate_sha256"), row.get("incumbent_sha256")) == (
                plan.candidate_sha256,
                plan.incumbent_sha256,
            ):
                raise EvaluationRefused("this candidate/incumbent pair was already decided")
            used |= _evidence_used(row)
        if used & _evidence_used(decision):
            raise EvaluationRefused("an evaluation collection was already used by a decision")
        claim = {
            # The exact decision, so a retry is recognised and any other refused.
            "decision_sha256": hashlib.sha256(
                json.dumps(decision, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            "plan_sha256": plan.sha256,
            "record": str(path),
            "sequence": sequence,
            "candidate_sha256": plan.candidate_sha256,
            "incumbent_sha256": plan.incumbent_sha256,
        }
        pairs = [c / "selection-claims" / f"{plan.incumbent_sha256}.json" for c in checkpoints]
        evidence = [
            Path(d) / "selection-claim.json"
            for d in sorted({str(Path(d).resolve()) for d in incumbent_dirs + candidate_dirs})
        ]
        own_row = claims_dir / f"{sequence:06d}.json"
        targets = [*pairs, *evidence, own_row]
        present = {t: _read_claim(t) for t in targets if t.exists() or t.is_symlink()}
        foreign = [t for t, content in present.items() if content != claim]
        if own_row in foreign:
            raise EvaluationRefused(
                f"interrupted decision at sequence {sequence} in this lineage holds its claims;"
                " retry that identical decision or see the repair procedure"
            )
        if any(t in foreign for t in pairs):
            raise EvaluationRefused("this candidate/incumbent pair was already decided")
        if foreign:
            raise EvaluationRefused(f"evidence already claimed by a decision: {foreign[0]}")
        if present and len(present) != len(targets):
            missing = [str(t) for t in targets if t not in present]
            raise EvaluationRefused(
                "interrupted decision: only some of its claims exist; retry the identical"
                f" decision or see the repair procedure (missing: {missing})"
            )
        # Either no claim exists yet, or every claim matches this exact decision
        # (a crash after claiming but before the row): adopt them and append.
        created: list[Path] = []
        try:
            if not present:
                for target in targets:
                    _claim(target, claim, created)
            row = {"sequence": sequence, "utc": time.time(), **decision}
            new_file = not path.exists()
            line = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            with path.open("a") as out:
                out.write(line)
                out.flush()
                os.fsync(out.fileno())
            if new_file:
                _fsync_dir(root)
        except BaseException as exc:
            remaining = _release(created)
            if remaining:
                raise EvaluationRefused(
                    f"decision failed and rollback left claims behind: {remaining}"
                ) from exc
            if isinstance(exc, OSError):
                raise EvaluationRefused(f"claim could not be written: {exc}") from exc
            raise
    return row
