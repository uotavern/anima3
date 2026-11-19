"""Unit-7 evaluation statistics, selection gate and train/evaluation split; no torch.

Collections are produced by the real unit-6 ``run_collection`` (ledger, staged
publication, receipt and episode verification) around the mocked participant
layer of ``test_neural_live.Harness``. No server, bridge, account or model runs.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_neural_live import OTHER_SHA, Harness, neural_arm, replay_fixture

from anima3.neural import live
from anima3.neural.evaluation import (
    BOOTSTRAP_SEED,
    EvaluationPlan,
    EvaluationRefused,
    bootstrap_mean_interval,
    combine_records,
    decide_from_collections,
    record_decision,
    record_from_collection,
    selection_gate,
    wilson_interval,
)
from anima3.neural.live import Arm, CollectionPlan
from anima3.neural.splits import (
    NO_HELD_OUT,
    SplitRefused,
    TrainingEpisodeVerifier,
    TrainingSelection,
    read_collection,
    select_training_episodes,
)

# Candidate checkpoints must be reachable by SHA-256, so their identities are real
# digests of small stand-in weight files (written by ``arms_vs``).
BLOBS = {
    hashlib.sha256(blob).hexdigest(): blob
    for blob in (b"candidate weights", b"incumbent weights", b"second candidate weights")
}
CANDIDATE, INCUMBENT, SECOND = BLOBS
RIVAL = OTHER_SHA
_IDS = itertools.count(1)


# --- win-rate confidence ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "wins,games,low,high",
    [
        # Newcombe (1998), Statistics in Medicine 17:857, Table I, method 3 (Wilson score).
        (81, 263, 0.2553, 0.3662),
        (15, 148, 0.0624, 0.1605),
        (0, 20, 0.0, 0.1611),
        (1, 29, 0.0061, 0.1718),
        (8, 10, 0.4902, 0.9433),
        (5, 10, 0.2366, 0.7634),
    ],
)
def test_wilson_interval_matches_published_values(wins, games, low, high):
    lo, hi = wilson_interval(wins, games)
    assert (round(lo, 4), round(hi, 4)) == (low, high)


def test_wilson_interval_edges_and_bad_counts():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(10, 10)
    assert round(hi, 12) == 1.0 and round(lo, 4) == 0.7225
    for wins, games in ((11, 10), (-1, 10), (True, 10), (1.0, 10), (0, -1)):
        with pytest.raises(ValueError):
            wilson_interval(wins, games)


def test_bootstrap_is_deterministic_and_degenerate_for_constants():
    values = [0.0, 1.0, 0.5, -0.5, 1.0, 0.0, 0.5, 1.0]
    first = bootstrap_mean_interval(values)
    assert first == bootstrap_mean_interval(values, seed=BOOTSTRAP_SEED)
    assert first != bootstrap_mean_interval(values, seed=BOOTSTRAP_SEED + 1)
    assert first[0] < sum(values) / len(values) < first[1]
    assert bootstrap_mean_interval([0.25] * 7) == (0.25, 0.25)
    with pytest.raises(ValueError):
        bootstrap_mean_interval([])


# --- the fixed plan --------------------------------------------------------------------------


def plan(**overrides):
    values = {
        "candidate_sha256": CANDIDATE,
        "incumbent_sha256": INCUMBENT,
        "opponents": ("scripted", f"neural:{RIVAL}"),
        "games_per_opponent": 20,
        **overrides,
    }
    return EvaluationPlan(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"games_per_opponent": 0},
        {"games_per_opponent": 2},
        {"games_per_opponent": 4},
        {"games_per_opponent": 18},
        {"games_per_opponent": 21},
        {"games_per_opponent": 102},
        {"games_per_opponent": True},
        {"incumbent_sha256": CANDIDATE},
        {"candidate_sha256": "AB" * 32},
        {"opponents": ()},
        {"opponents": ("scripted", "scripted")},
        {"opponents": ("rival",)},
        {"opponents": (f"neural:{CANDIDATE}",)},
        {"opponents": (f"neural:{INCUMBENT}", "scripted")},
        {"max_regression": float("nan")},
        {"max_regression": -0.1},
        {"max_regression": 0.105},
        {"min_lower_bound": float("inf")},
        {"min_lower_bound": 0.001},
        {"max_excluded_per_opponent": 3},
        {"max_excluded_per_opponent": 20},
        {"min_paired_games_per_opponent": 2},
        {"min_paired_games_per_opponent": 19},
        {"min_paired_games_per_opponent": 22},
        {"domain": "elsewhere"},
    ],
)
def test_evaluation_plan_is_bounded_and_explicit(overrides):
    with pytest.raises(ValueError):
        plan(**overrides)


def test_plan_floors_refuse_tiny_budgets_even_with_a_matching_minimum():
    for games in (2, 4, 18):
        with pytest.raises(ValueError, match="games per opponent must be an even integer in 20"):
            plan(games_per_opponent=games, min_paired_games_per_opponent=games)


def test_plan_identity_is_canonical_but_covers_every_threshold():
    assert plan().sha256 == plan().sha256
    # Reordering opponents or nudging a threshold below the 0.01 grid is the same plan.
    assert plan(opponents=(f"neural:{RIVAL}", "scripted")).sha256 == plan().sha256
    assert plan(opponents=[f"neural:{RIVAL}", "scripted"]).opponents == plan().opponents
    assert plan(min_lower_bound=1e-12).sha256 == plan().sha256
    assert plan(min_lower_bound=-1e-12).to_json()["min_lower_bound"] == 0.0
    assert plan(max_regression=0.1 + 1e-12).sha256 == plan().sha256
    assert plan(max_regression=0).sha256 == plan(max_regression=0.0).sha256
    assert plan().sha256 != plan(max_regression=0.2).sha256
    assert plan().sha256 != plan(games_per_opponent=22).sha256
    assert EvaluationPlan.from_json(plan().to_json()) == plan()


# --- the gate on direct rows -----------------------------------------------------------------


def side(fixture):
    return "challenger" if fixture % 2 == 0 else "receiver"


def evaluation(sha, scores_by_opponent, *, valid=True):
    rows = []
    for opponent, scores in scores_by_opponent.items():
        for fixture, score in enumerate(scores):
            if score is None:
                rows.append(
                    {"opponent": opponent, "fixture": fixture, "status": "excluded", "score": None}
                )
            else:
                rows.append(
                    {
                        "opponent": opponent,
                        "fixture": fixture,
                        "status": "completed",
                        "score": score,
                        "side": side(fixture),
                        "valid": valid,
                    }
                )
    return {"domain": "servuo", "policy_sha": sha, "rows": rows}


def scores(wins, n=20, *, offset=0):
    """``wins`` wins among ``n`` games, spread across fixture blocks."""
    out = [0.0] * n
    for i in range(wins):
        out[(i * 7 + offset) % n] = 1.0
    assert sum(out) == wins
    return out


RIVAL_LABEL = f"neural:{RIVAL}"


def test_clear_improvement_against_every_opponent_passes():
    p = plan()
    before = evaluation(INCUMBENT, {"scripted": scores(4), RIVAL_LABEL: scores(4)})
    after = evaluation(CANDIDATE, {"scripted": scores(16), RIVAL_LABEL: scores(15)})
    gate = selection_gate(p, before, after)
    assert gate["passed"] is True and gate["decision"] == "candidate_passed_gate"
    assert gate["paired_delta_ci95"][0] > 0 and gate["bootstrap"]["blocks"] == 20
    assert gate["opponents"]["scripted"]["candidate"]["win_rate"] == 0.8
    assert gate["opponents"]["scripted"]["candidate"]["win_rate_ci95"] == list(
        wilson_interval(16, 20)
    )
    assert gate["automatic_promotion"] is False and gate["public_deployment"] is False


def test_single_opponent_regression_fails_even_when_the_average_improves():
    p = plan(max_regression=0.10)
    before = evaluation(INCUMBENT, {"scripted": scores(2), RIVAL_LABEL: scores(10)})
    # +0.85 against scripted, -0.15 against the rival: the average improves by 0.35.
    after = evaluation(CANDIDATE, {"scripted": scores(19), RIVAL_LABEL: scores(7)})
    gate = selection_gate(p, before, after)
    assert gate["paired_delta"] > 0.3 and gate["paired_delta_ci95"][0] > 0
    assert gate["passed"] is False and gate["decision"] == "retain_incumbent"
    assert gate["opponents"][RIVAL_LABEL]["regressed"] is True
    assert gate["opponents"]["scripted"]["regressed"] is False
    assert gate["reasons"] == [f"regression beyond limit against {RIVAL_LABEL}"]
    # The same evidence passes only under an explicitly looser per-opponent limit.
    assert selection_gate(plan(max_regression=0.15), before, after)["passed"] is True


def test_equal_evidence_retains_the_incumbent():
    same = {"scripted": scores(10), RIVAL_LABEL: scores(10)}
    gate = selection_gate(plan(), evaluation(INCUMBENT, same), evaluation(CANDIDATE, same))
    assert gate["passed"] is False
    assert "paired improvement lower bound not above threshold" in gate["reasons"]


def test_too_few_complete_blocks_cannot_pass_even_when_every_game_improves():
    p = plan(opponents=("scripted",), max_excluded_per_opponent=2)
    incumbent = [0.0] * 20
    incumbent[1] = incumbent[3] = None  # breaks blocks 0 and 1: 8 complete blocks remain
    gate = selection_gate(
        p,
        evaluation(INCUMBENT, {"scripted": incumbent}),
        evaluation(CANDIDATE, {"scripted": [1.0] * 20}),
    )
    assert gate["bootstrap"]["blocks"] == 8
    assert gate["passed"] is False and gate["paired_delta_ci95"] is None
    assert gate["reasons"] == [
        "insufficient paired games against scripted",
        "too few complete blocks to decide",
    ]


def test_a_positive_average_without_a_positive_lower_bound_is_not_an_improvement():
    before = evaluation(INCUMBENT, {"scripted": scores(10), RIVAL_LABEL: scores(10)})
    after = evaluation(
        CANDIDATE, {"scripted": scores(12, offset=3), RIVAL_LABEL: scores(11, offset=5)}
    )
    gate = selection_gate(plan(), before, after)
    assert gate["paired_delta"] == pytest.approx(0.075)
    assert gate["paired_delta_ci95"][0] <= 0 < gate["paired_delta_ci95"][1]
    assert gate["passed"] is False
    assert gate["reasons"] == ["paired improvement lower bound not above threshold"]


def test_excluded_games_stay_in_the_record_and_fail_beyond_the_allowance():
    candidate_scores = scores(18)
    candidate_scores[3] = None  # a failed fixture: spent, kept, never rescored
    before = evaluation(INCUMBENT, {"scripted": scores(2), RIVAL_LABEL: scores(2)})
    after = evaluation(CANDIDATE, {"scripted": candidate_scores, RIVAL_LABEL: scores(18)})
    gate = selection_gate(plan(), before, after)
    assert gate["passed"] is False
    assert gate["reasons"] == [
        "excluded games beyond allowance against scripted",
        "insufficient paired games against scripted",
    ]
    assert gate["excluded_rows"] == [
        {
            "policy": "candidate",
            "opponent": "scripted",
            "fixture": 3,
            "status": "excluded",
            "score": None,
        }
    ]
    assert gate["opponents"]["scripted"]["paired_games"] == 19


def test_exclusions_are_scored_against_the_candidate():
    p = plan(opponents=("scripted",), games_per_opponent=40, max_excluded_per_opponent=4)
    # The candidate wins every game it finishes; its 4 failures count as losses.
    candidate = [1.0] * 40
    for fixture in (0, 2, 4, 6):
        candidate[fixture] = None
    before = evaluation(INCUMBENT, {"scripted": [1.0] * 40})
    gate = selection_gate(p, before, evaluation(CANDIDATE, {"scripted": candidate}))
    summary = gate["opponents"]["scripted"]
    assert summary["scored_fixtures"] == 40 and summary["paired_games"] == 36
    assert summary["mean_score_delta"] == pytest.approx(-0.1)
    assert gate["passed"] is False
    # An incumbent failure earns the candidate nothing: that fixture is dropped.
    incumbent = [0.0] * 40
    for fixture in (0, 2, 4, 6):
        incumbent[fixture] = None
    gate = selection_gate(
        p,
        evaluation(INCUMBENT, {"scripted": incumbent}),
        evaluation(CANDIDATE, {"scripted": [0.0] * 40}),
    )
    summary = gate["opponents"]["scripted"]
    assert summary["scored_fixtures"] == 36 and summary["mean_score_delta"] == 0.0
    assert gate["bootstrap"]["blocks"] == 16
    allowed = selection_gate(
        p,
        evaluation(INCUMBENT, {"scripted": incumbent}),
        evaluation(CANDIDATE, {"scripted": [1.0] * 40}),
    )
    assert allowed["passed"] is True and len(allowed["excluded_rows"]) == 4


def test_paired_games_must_share_their_seat_and_blocks_hold_both_seats():
    before = evaluation(INCUMBENT, {"scripted": [0.0] * 20, RIVAL_LABEL: [0.0] * 20})
    after = evaluation(CANDIDATE, {"scripted": [1.0] * 20, RIVAL_LABEL: [1.0] * 20})
    swapped = copy.deepcopy(after)
    for row in swapped["rows"]:
        row["side"] = "receiver" if row["side"] == "challenger" else "challenger"
    with pytest.raises(EvaluationRefused, match="different seats"):
        selection_gate(plan(), before, swapped)
    same_seat = copy.deepcopy(before), copy.deepcopy(after)
    for evaluation_ in same_seat:
        for row in evaluation_["rows"]:
            row["side"] = "challenger"
    with pytest.raises(EvaluationRefused, match="both seats"):
        selection_gate(plan(), *same_seat)
    unseated = copy.deepcopy(after)
    del unseated["rows"][0]["side"]
    with pytest.raises(EvaluationRefused, match="seat"):
        selection_gate(plan(), before, unseated)


def test_invalid_episode_evidence_fails_closed():
    before = evaluation(INCUMBENT, {"scripted": scores(2), RIVAL_LABEL: scores(2)})
    after = evaluation(CANDIDATE, {"scripted": scores(18), RIVAL_LABEL: scores(18)})
    after["rows"][0]["valid"] = False
    gate = selection_gate(plan(), before, after)
    assert gate["passed"] is False
    assert gate["reasons"] == ["invalid episode evidence against scripted"]


def test_match_count_cannot_grow_or_shrink_after_the_plan():
    p = plan()
    before = evaluation(INCUMBENT, {"scripted": scores(2), RIVAL_LABEL: scores(2)})
    after = evaluation(CANDIDATE, {"scripted": scores(18), RIVAL_LABEL: scores(18)})
    grown = copy.deepcopy(after)
    grown["rows"].append({**grown["rows"][0], "fixture": 20})
    with pytest.raises(EvaluationRefused, match="fixed budget"):
        selection_gate(p, before, grown)
    shrunk = copy.deepcopy(after)
    shrunk["rows"].pop()
    with pytest.raises(EvaluationRefused, match="exactly its planned games"):
        selection_gate(p, before, shrunk)
    duplicate = copy.deepcopy(after)
    duplicate["rows"][1] = dict(duplicate["rows"][0])
    with pytest.raises(EvaluationRefused, match="duplicate"):
        selection_gate(p, before, duplicate)


@pytest.mark.parametrize(
    "damage,error",
    [
        ("policy", "policy SHA"),
        ("domain", "domains"),
        ("score", "loss/draw/win"),
        ("bool", "loss/draw/win"),
        ("nan", "loss/draw/win"),
        ("status", "status"),
        ("opponent", "unplanned"),
        ("valid", "validity"),
    ],
)
def test_gate_refuses_malformed_or_mismatched_evidence(damage, error):
    before = evaluation(INCUMBENT, {"scripted": scores(2), RIVAL_LABEL: scores(2)})
    after = evaluation(CANDIDATE, {"scripted": scores(18), RIVAL_LABEL: scores(18)})
    row = after["rows"][0]
    if damage == "policy":
        after["policy_sha"] = INCUMBENT
    elif damage == "domain":
        after["domain"] = "approximate-pre-aos-simulator"
    elif damage == "score":
        row["score"] = 0.75
    elif damage == "bool":
        row["score"] = True
    elif damage == "nan":
        row["score"] = float("nan")
    elif damage == "status":
        row["status"] = "retried"
    elif damage == "opponent":
        row["opponent"] = "scripted-v2"
    elif damage == "valid":
        row.pop("valid")
    with pytest.raises(EvaluationRefused, match=error):
        selection_gate(plan(), before, after)


# --- records from real unit-6 collections ----------------------------------------------------


class Scored(Harness):
    """Each receipt reports the next winner; unique match IDs across a test."""

    def __init__(self, monkeypatch, winners):
        super().__init__(monkeypatch, [f"{next(_IDS):032x}" for _ in winners])
        self.winners = list(winners)

    def receipt(self, match_id, seats):
        return replay_fixture(match_id, tuple(seats), winner=self.winners.pop(0))


def collect(
    tmp_path, monkeypatch, name, arms, winners, *, purpose="evaluation", matches=None, resume=False
):
    harness = Scored(monkeypatch, winners)
    plan_ = CollectionPlan(
        endpoint="lab.invalid:2593",
        accounts=("alpha", "beta"),
        arms=arms,
        matches=matches or len(winners),
        purpose=purpose,
    )
    live.run_collection(
        plan_,
        tmp_path / name,
        connect=harness.connect,
        fetch_receipt=harness.receipt,
        lock_dir=tmp_path / "locks",
        client_factory=harness.factory,
        find_spec=lambda _: object(),
        resume=resume,
    )
    return tmp_path / name


def checkpoint_dir(tmp_path, sha):
    return tmp_path / sha[:4] / "checkpoint"


def write_checkpoint(tmp_path, sha, lineage=None):
    """A stand-in checkpoint whose weight file hashes to ``sha``."""
    root = checkpoint_dir(tmp_path, sha)
    root.mkdir(parents=True, exist_ok=True)
    (root / "weights.pt").write_bytes(BLOBS[sha])
    training = {} if lineage is None else {"lineage_dir": str(Path(lineage).resolve())}
    manifest = {"weights": {"file": "weights.pt", "sha256": sha}, "training": training}
    (root / "manifest.json").write_text(json.dumps(manifest))


def arms_vs(tmp_path, sha, opponent):
    other = Arm("scripted") if opponent == "scripted" else neural_arm(tmp_path / "o", RIVAL)
    (tmp_path / "o").mkdir(exist_ok=True)
    (tmp_path / sha[:4]).mkdir(exist_ok=True)
    manifest = checkpoint_dir(tmp_path, sha) / "manifest.json"
    kept = manifest.read_text() if manifest.exists() else None
    arm = neural_arm(tmp_path / sha[:4], sha)  # (re)writes a placeholder manifest
    if kept is not None and kept != "{}":
        manifest.write_text(kept)
    elif sha in BLOBS:
        write_checkpoint(tmp_path, sha)
    return (arm, other)


@pytest.fixture
def sha_dirs(tmp_path):
    for sha in (CANDIDATE, INCUMBENT, RIVAL):
        (tmp_path / sha[:4]).mkdir()
    (tmp_path / "o").mkdir()
    return tmp_path


def test_record_from_collection_scores_receipts_and_alternates_seats(sha_dirs, monkeypatch):
    tmp_path = sha_dirs
    # Serial 1 is the candidate's seat (alpha). Winners: win, loss, win, draw.
    root = collect(
        tmp_path, monkeypatch, "c-s", arms_vs(tmp_path, CANDIDATE, "scripted"), [1, 2, 1, None]
    )
    record = record_from_collection(root, CANDIDATE)
    assert record["opponent"] == "scripted" and record["budget"] == 4
    assert [r["score"] for r in record["rows"]] == [1.0, 0.0, 1.0, 0.5]
    assert [r["side"] for r in record["rows"]] == ["challenger", "receiver"] * 2
    assert all(r["valid"] is True for r in record["rows"])
    rival = collect(tmp_path, monkeypatch, "c-r", arms_vs(tmp_path, CANDIDATE, "neural"), [2, 2])
    assert record_from_collection(rival, CANDIDATE)["opponent"] == RIVAL_LABEL
    with pytest.raises(EvaluationRefused, match="exactly one seat"):
        record_from_collection(rival, INCUMBENT)


def test_failed_fixture_is_kept_as_an_excluded_row(sha_dirs, monkeypatch):
    tmp_path = sha_dirs
    arms = arms_vs(tmp_path, CANDIDATE, "scripted")
    with pytest.raises(RuntimeError, match="connection failed"):
        collect(tmp_path, monkeypatch, "c-s", arms, [1], matches=4)
    with pytest.raises(EvaluationRefused, match="interim"):
        record_from_collection(tmp_path / "c-s", CANDIDATE)
    collect(tmp_path, monkeypatch, "c-s", arms, [2, 1], matches=4, resume=True)
    record = record_from_collection(tmp_path / "c-s", CANDIDATE)
    assert [r["status"] for r in record["rows"]] == [
        "completed",
        "excluded",
        "completed",
        "completed",
    ]
    excluded = record["rows"][1]
    assert excluded["event"] == "failed" and excluded["score"] is None
    assert "connection failed" in excluded["reason"]


def test_training_collections_and_partial_budgets_are_not_evaluations(sha_dirs, monkeypatch):
    tmp_path = sha_dirs
    arms = arms_vs(tmp_path, CANDIDATE, "scripted")
    root = collect(tmp_path, monkeypatch, "t", arms, [1, 2], purpose="training")
    with pytest.raises(SplitRefused, match="held-out evaluation"):
        record_from_collection(root, CANDIDATE)


WIN, LOSS = 1, 2


def evaluation_set(tmp_path, monkeypatch, sha, prefix, scripted, rival):
    return [
        collect(tmp_path, monkeypatch, f"{prefix}-s", arms_vs(tmp_path, sha, "scripted"), scripted),
        collect(tmp_path, monkeypatch, f"{prefix}-r", arms_vs(tmp_path, sha, "neural"), rival),
    ]


def test_live_collections_feed_the_gate_end_to_end(sha_dirs, monkeypatch, tmp_path):
    p = plan(max_regression=0.25)
    incumbent = evaluation_set(
        tmp_path, monkeypatch, INCUMBENT, "i", [LOSS] * 15 + [WIN] * 5, [LOSS, WIN] * 10
    )
    candidate = evaluation_set(
        tmp_path, monkeypatch, CANDIDATE, "c", [WIN] * 20, [WIN] * 15 + [LOSS] * 5
    )
    gate = decide_from_collections(p, incumbent=incumbent, candidate=candidate)
    assert gate["opponents"]["scripted"]["mean_score_delta"] == 0.75
    assert gate["opponents"][RIVAL_LABEL]["mean_score_delta"] == 0.25
    assert gate["passed"] is True
    evidence = gate["evidence"]["candidate"]["scripted"]
    assert evidence["path"] == str(candidate[0].resolve())
    assert evidence["plan_sha256"] == read_collection(candidate[0]).plan_sha256
    first = record_from_collection(candidate[0], CANDIDATE)
    with pytest.raises(EvaluationRefused, match="one opponent"):
        combine_records(p, CANDIDATE, [first] * 2)
    with pytest.raises(EvaluationRefused, match="exactly the planned"):
        combine_records(p, CANDIDATE, [first])
    with pytest.raises(EvaluationRefused, match="budget differs"):
        combine_records(
            plan(games_per_opponent=22),
            CANDIDATE,
            [record_from_collection(r, CANDIDATE) for r in candidate],
        )
    with pytest.raises(EvaluationRefused, match="one opponent"):
        decide_from_collections(p, incumbent=incumbent, candidate=[candidate[0], candidate[0]])


def test_every_decision_is_recorded_once_and_rederived(sha_dirs, monkeypatch, tmp_path):
    lineage = tmp_path / "lineage"
    write_checkpoint(tmp_path, CANDIDATE, lineage)
    write_checkpoint(tmp_path, SECOND, lineage)
    p = plan()
    incumbent = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i", [LOSS] * 20, [LOSS] * 20)
    candidate = evaluation_set(
        tmp_path, monkeypatch, CANDIDATE, "c", [WIN] * 20, [WIN] * 10 + [LOSS] * 10
    )
    decision = decide_from_collections(p, incumbent=incumbent, candidate=candidate)
    # Forged or inconsistent decisions never reach the record.
    for forged, error in (
        ({**decision, "plan_sha256": "f" * 64}, "re-hashed plan"),
        ({**decision, "passed": False}, "inconsistent"),
        ({**decision, "passed": True, "reasons": ["x"]}, "inconsistent"),
        ({**decision, "paired_delta": 1.0}, "differs from the one its collections give"),
        ({**decision, "automatic_promotion": True}, "never promotes"),
        ({**decision, "evidence": {"incumbent": None, "candidate": None}}, "name its collections"),
        ({**decision, "plan": {**decision["plan"], "games_per_opponent": 2}}, "plan is invalid"),
    ):
        with pytest.raises(EvaluationRefused, match=error):
            record_decision(lineage, forged)
    assert not (lineage / "selection.jsonl").exists()
    row = record_decision(lineage, decision)
    assert row["sequence"] == 0 and row["passed"] is True
    # The same pair under a reordered, nudged or loosened plan is not decided again.
    for again in (
        plan(opponents=(f"neural:{RIVAL}", "scripted")),
        plan(min_lower_bound=1e-12),
        plan(max_regression=0.5),
    ):
        with pytest.raises(EvaluationRefused, match="pair was already decided"):
            record_decision(
                lineage, decide_from_collections(again, incumbent=incumbent, candidate=candidate)
            )
    # Another candidate may not reuse the incumbent's evaluation collections.
    other = SECOND
    second = evaluation_set(tmp_path, monkeypatch, other, "d", [WIN] * 20, [LOSS] * 20)
    reuse = decide_from_collections(
        plan(candidate_sha256=other), incumbent=incumbent, candidate=second
    )
    with pytest.raises(EvaluationRefused, match="already used"):
        record_decision(lineage, reuse)
    fresh = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "j", [LOSS] * 20, [WIN] * 20)
    late = decide_from_collections(plan(candidate_sha256=other), incumbent=fresh, candidate=second)
    assert late["passed"] is False  # regression against the rival
    assert record_decision(lineage, late)["sequence"] == 1
    rows = [json.loads(line) for line in (lineage / "selection.jsonl").read_text().splitlines()]
    assert [(r["sequence"], r["passed"]) for r in rows] == [(0, True), (1, False)]
    assert all(r["automatic_promotion"] is False for r in rows)
    # A lost final row is indistinguishable from a crash before the append: only
    # the identical decision restores it. Losing more than that is tampering.
    record = lineage / "selection.jsonl"
    record.write_text(json.dumps(rows[0], sort_keys=True) + "\n")
    assert record_decision(lineage, late)["sequence"] == 1
    record.unlink()
    with pytest.raises(EvaluationRefused, match="deleted, truncated or reordered"):
        record_decision(lineage, late)


def test_selection_record_is_locked_and_refuses_corruption(sha_dirs, monkeypatch, tmp_path):
    import fcntl

    lineage = tmp_path / "lineage"
    lineage.mkdir()
    write_checkpoint(tmp_path, CANDIDATE, lineage)
    incumbent = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i", [LOSS] * 20, [LOSS] * 20)
    candidate = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "c", [WIN] * 20, [WIN] * 20)
    decision = decide_from_collections(plan(), incumbent=incumbent, candidate=candidate)
    with (lineage / "selection.jsonl.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(EvaluationRefused, match="another writer"):
            record_decision(lineage, decision)
        # Re-derivation happens under the lock: a forged decision waits for it too.
        with pytest.raises(EvaluationRefused, match="another writer"):
            record_decision(lineage, {**decision, "paired_delta": 0.0})
    record = lineage / "selection.jsonl"
    record.write_text("[1, 2]\n")
    with pytest.raises(EvaluationRefused, match="not an object"):
        record_decision(lineage, decision)
    record.write_text('{"torn"')
    with pytest.raises(EvaluationRefused, match="torn"):
        record_decision(lineage, decision)
    record.write_text("")
    assert record_decision(lineage, decision)["sequence"] == 0


def test_evidence_pairs_and_lineages_hold_across_record_locations(sha_dirs, monkeypatch, tmp_path):
    """The reviewer's round-2 probe: every re-decision outside the record is refused."""
    l1, l2, l3 = (tmp_path / n for n in ("L1", "L2", "L3"))
    write_checkpoint(tmp_path, CANDIDATE, l1)
    p = plan()
    inc = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i", [LOSS] * 20, [LOSS] * 20)
    cand = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "c", [WIN] * 20, [WIN] * 20)
    d = decide_from_collections(p, incumbent=inc, candidate=cand)
    assert record_decision(l1, d)["sequence"] == 0
    for root in inc + cand:
        claim = json.loads((root / "selection-claim.json").read_text())
        assert claim["record"] == str((l1 / "selection.jsonl").resolve())
        assert claim["sequence"] == 0 and claim["plan_sha256"] == p.sha256
        assert (claim["candidate_sha256"], claim["incumbent_sha256"]) == (CANDIDATE, INCUMBENT)
    # Same evidence, another lineage directory.
    with pytest.raises(EvaluationRefused, match="bound to another lineage"):
        record_decision(l2, d)
    # Fresh evidence for the decided pair, a third directory.
    inc2 = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i2", [LOSS] * 20, [LOSS] * 20)
    cand2 = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "c2", [WIN] * 20, [WIN] * 20)
    d2 = decide_from_collections(p, incumbent=inc2, candidate=cand2)
    with pytest.raises(EvaluationRefused, match="pair was already decided"):
        record_decision(l1, d2)
    with pytest.raises(EvaluationRefused, match="bound to another lineage"):
        record_decision(l3, d2)
    # Deleting or emptying the record no longer resets the lineage. A lost final
    # row looks exactly like a crash before the append, so only the identical
    # decision may restore it; any other decision is refused.
    (l1 / "selection.jsonl").unlink()
    with pytest.raises(EvaluationRefused, match="interrupted decision at sequence 0"):
        record_decision(l1, d2)
    (l1 / "selection.jsonl").write_text("")
    with pytest.raises(EvaluationRefused, match="interrupted decision at sequence 0"):
        record_decision(l1, d2)
    # Even with record and lineage claims gone, the pair claim in the checkpoint holds.
    (l1 / "selection.jsonl").unlink()
    (l1 / "claims" / "000000.json").unlink()
    with pytest.raises(EvaluationRefused, match="pair was already decided"):
        record_decision(l1, d2)
    assert not (inc2[0] / "selection-claim.json").exists()


def test_external_candidates_must_say_so_and_claims_are_all_or_nothing(
    sha_dirs, monkeypatch, tmp_path
):
    l1, l2 = tmp_path / "L1", tmp_path / "L2"
    p = plan()
    inc = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i", [LOSS] * 20, [LOSS] * 20)
    cand = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "c", [WIN] * 20, [WIN] * 20)
    implicit = decide_from_collections(p, incumbent=inc, candidate=cand)
    with pytest.raises(EvaluationRefused, match="declare external_candidate"):
        record_decision(l1, implicit)
    with pytest.raises(EvaluationRefused, match="True or False"):
        decide_from_collections(p, incumbent=inc, candidate=cand, external_candidate=1)
    external = decide_from_collections(p, incumbent=inc, candidate=cand, external_candidate=True)
    with pytest.raises(EvaluationRefused, match="differs"):
        record_decision(l1, {**external, "paired_delta": 0.0})
    assert record_decision(l1, external)["external_candidate"] is True
    # The same external pair cannot be decided again anywhere, even with fresh evidence.
    inc2 = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i2", [LOSS] * 20, [LOSS] * 20)
    cand2 = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "c2", [WIN] * 20, [WIN] * 20)
    again = decide_from_collections(p, incumbent=inc2, candidate=cand2, external_candidate=True)
    with pytest.raises(EvaluationRefused, match="pair was already decided"):
        record_decision(l2, again)
    # A second candidate reusing the claimed incumbent evidence in a new lineage:
    # refused, and the pair claim it created first is rolled back.
    reuse = decide_from_collections(
        plan(candidate_sha256=SECOND),
        incumbent=inc,
        candidate=evaluation_set(tmp_path, monkeypatch, SECOND, "d", [WIN] * 20, [WIN] * 20),
        external_candidate=True,
    )
    with pytest.raises(EvaluationRefused, match="already claimed"):
        record_decision(l2, reuse)
    pair = checkpoint_dir(tmp_path, SECOND) / "selection-claims" / f"{INCUMBENT}.json"
    assert not pair.exists()
    assert not (l2 / "selection.jsonl").exists() and not list((l2 / "claims").glob("*"))
    # A copied collection carries its claim with it.
    import shutil

    copied = tmp_path / "copy-i-s"
    shutil.copytree(inc[0], copied)
    assert (copied / "selection-claim.json").exists()
    # A bound candidate cannot be declared external, and unverifiable weights are refused.
    write_checkpoint(tmp_path, SECOND, l2)
    fresh = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "j", [LOSS] * 20, [LOSS] * 20)
    second = [tmp_path / "d-s", tmp_path / "d-r"]
    bound = decide_from_collections(
        plan(candidate_sha256=SECOND), incumbent=fresh, candidate=second, external_candidate=True
    )
    with pytest.raises(EvaluationRefused, match="bound to another lineage"):
        record_decision(l2, bound)
    (checkpoint_dir(tmp_path, SECOND) / "weights.pt").write_bytes(b"tampered")
    with pytest.raises(EvaluationRefused, match="hash to its SHA-256"):
        record_decision(l2, {**bound, "external_candidate": False})


CRASH = """
import json, os, sys
import anima3.neural.evaluation as E
limit = int(sys.argv[3])
if limit < 0:
    E.time.time = lambda: os._exit(9)  # every claim written, the row not yet appended
else:
    real, calls = E._claim, []
    def claim(*args):
        if len(calls) == limit:
            os._exit(9)  # crash part-way through claiming
        calls.append(1)
        real(*args)
    E._claim = claim
E.record_decision(sys.argv[1], json.load(open(sys.argv[2])))
"""


def crash_recording(tmp_path, lineage, decision, after_claims=-1):
    (tmp_path / "decision.json").write_text(json.dumps(decision))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            CRASH,
            str(lineage),
            str(tmp_path / "decision.json"),
            str(after_claims),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    assert result.returncode == 9


def claim_files(tmp_path):
    return sorted(p for p in tmp_path.rglob("*.json") if "claim" in str(p.relative_to(tmp_path)))


def test_a_crash_after_claiming_is_finished_by_retrying_the_identical_decision(
    sha_dirs, monkeypatch, tmp_path
):
    l1 = tmp_path / "L1"
    write_checkpoint(tmp_path, CANDIDATE, l1)
    write_checkpoint(tmp_path, SECOND, l1)
    inc = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i", [LOSS] * 20, [LOSS] * 20)
    cand = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "c", [WIN] * 20, [WIN] * 20)
    d = decide_from_collections(plan(), incumbent=inc, candidate=cand)
    crash_recording(tmp_path, l1, d)
    assert not (l1 / "selection.jsonl").exists()
    orphaned = claim_files(tmp_path)
    assert len(orphaned) == 6  # pair, four collections, lineage row 0
    # A different decision against the orphaned claims is refused, distinctly.
    inc2 = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i2", [LOSS] * 20, [LOSS] * 20)
    cand2 = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "c2", [WIN] * 20, [WIN] * 20)
    d2 = decide_from_collections(plan(), incumbent=inc2, candidate=cand2)
    with pytest.raises(EvaluationRefused, match="interrupted decision at sequence 0"):
        record_decision(l1, d2)
    other = decide_from_collections(
        plan(candidate_sha256=SECOND),
        incumbent=inc,
        candidate=evaluation_set(tmp_path, monkeypatch, SECOND, "d", [WIN] * 20, [WIN] * 20),
    )
    with pytest.raises(EvaluationRefused, match="interrupted decision at sequence 0"):
        record_decision(l1, other)
    # The identical decision adopts its claims and is appended at the right sequence.
    row = record_decision(l1, d)
    assert row["sequence"] == 0
    assert claim_files(tmp_path) == orphaned
    with pytest.raises(EvaluationRefused, match="pair was already decided"):
        record_decision(l1, d)
    fresh = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "j", [LOSS] * 20, [LOSS] * 20)
    second = [tmp_path / "d-s", tmp_path / "d-r"]
    later = decide_from_collections(
        plan(candidate_sha256=SECOND), incumbent=fresh, candidate=second
    )
    assert record_decision(l1, later)["sequence"] == 1


def test_a_crash_part_way_through_claiming_is_reported_and_repairable(
    sha_dirs, monkeypatch, tmp_path
):
    l1 = tmp_path / "L1"
    write_checkpoint(tmp_path, CANDIDATE, l1)
    inc = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i", [LOSS] * 20, [LOSS] * 20)
    cand = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "c", [WIN] * 20, [WIN] * 20)
    d = decide_from_collections(plan(), incumbent=inc, candidate=cand)
    crash_recording(tmp_path, l1, d, after_claims=3)
    partial = claim_files(tmp_path)
    assert len(partial) == 3
    with pytest.raises(EvaluationRefused, match="only some of its claims exist") as refused:
        record_decision(l1, d)
    assert "deleted, truncated" not in str(refused.value)
    # Repair: every claim names its decision; remove that interrupted decision's claims.
    digest = json.loads(partial[0].read_text())["decision_sha256"]
    for path in partial:
        assert json.loads(path.read_text())["decision_sha256"] == digest
        path.unlink()
    assert record_decision(l1, d)["sequence"] == 0


def test_a_failed_rollback_names_the_claims_it_left(sha_dirs, monkeypatch, tmp_path):
    from anima3.neural import evaluation

    l1 = tmp_path / "L1"
    write_checkpoint(tmp_path, CANDIDATE, l1)
    inc = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i", [LOSS] * 20, [LOSS] * 20)
    cand = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "c", [WIN] * 20, [WIN] * 20)
    d = decide_from_collections(plan(), incumbent=inc, candidate=cand)
    real_claim, real_unlink = evaluation._claim, Path.unlink
    stuck = []

    def claim(path, content, created):
        if path.parent.name == "claims":
            raise OSError("disk full")
        real_claim(path, content, created)
        if not stuck:
            stuck.append(path)

    def unlink(self, *args, **kwargs):
        if stuck and self == stuck[0]:
            raise PermissionError("read-only")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(evaluation, "_claim", claim)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(EvaluationRefused, match="rollback left claims behind") as refused:
        record_decision(l1, d)
    assert str(stuck[0]) in str(refused.value)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert claim_files(tmp_path) == [stuck[0]]


def test_a_concurrent_claim_from_another_lineage_wins_the_race(sha_dirs, monkeypatch, tmp_path):
    """Lineage locks do not serialise different lineages; exclusive creation does."""
    from anima3.neural import evaluation

    l1 = tmp_path / "L1"
    inc = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i", [LOSS] * 20, [LOSS] * 20)
    cand = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "c", [WIN] * 20, [WIN] * 20)
    d = decide_from_collections(plan(), incumbent=inc, candidate=cand, external_candidate=True)
    real_claim = evaluation._claim
    rival = '{"decision_sha256": "another lineage"}\n'

    def racing(path, content, created):
        if path.name == "selection-claim.json" and not path.exists():
            path.write_text(rival)  # another lineage's writer got there first
        real_claim(path, content, created)

    monkeypatch.setattr(evaluation, "_claim", racing)
    with pytest.raises(EvaluationRefused, match="already claimed"):
        record_decision(l1, d)
    claimed = [p for p in claim_files(tmp_path) if p.name == "selection-claim.json"]
    assert [p.read_text() for p in claimed] == [rival]
    assert not (l1 / "selection.jsonl").exists()


def test_relative_or_oversized_candidate_checkpoints_are_refused(sha_dirs, monkeypatch, tmp_path):
    l1 = tmp_path / "L1"
    write_checkpoint(tmp_path, CANDIDATE, l1)
    inc = evaluation_set(tmp_path, monkeypatch, INCUMBENT, "i", [LOSS] * 20, [LOSS] * 20)
    monkeypatch.chdir(checkpoint_dir(tmp_path, CANDIDATE).parent)
    relative = Arm("neural", "checkpoint", CANDIDATE)
    rival = neural_arm(tmp_path / "o", RIVAL)
    cand = [
        collect(tmp_path, monkeypatch, "c-s", (relative, Arm("scripted")), [WIN] * 20),
        collect(tmp_path, monkeypatch, "c-r", (relative, rival), [WIN] * 20),
    ]
    d = decide_from_collections(plan(), incumbent=inc, candidate=cand)
    with pytest.raises(EvaluationRefused, match="relative path"):
        record_decision(l1, d)
    assert claim_files(tmp_path) == []
    absolute = evaluation_set(tmp_path, monkeypatch, CANDIDATE, "a", [WIN] * 20, [WIN] * 20)
    manifest = checkpoint_dir(tmp_path, CANDIDATE) / "manifest.json"
    content = json.loads(manifest.read_text())
    content["padding"] = "x" * (1024 * 1024)
    manifest.write_text(json.dumps(content))
    d = decide_from_collections(plan(), incumbent=inc, candidate=absolute)
    with pytest.raises(EvaluationRefused, match="size limit"):
        record_decision(l1, d)


# --- the training split ----------------------------------------------------------------------


def test_plans_are_held_out_unless_explicitly_training(tmp_path):
    base = {
        "endpoint": "lab.invalid:2593",
        "accounts": ("alpha", "beta"),
        "arms": (neural_arm(tmp_path), Arm("scripted")),
        "matches": 1,
    }
    assert CollectionPlan(**base).purpose == "evaluation"
    assert CollectionPlan(**base).sha256 != CollectionPlan(**base, purpose="training").sha256
    for purpose in ("", "train", None, ["training"]):
        with pytest.raises(ValueError, match="purpose"):
            CollectionPlan(**base, purpose=purpose)


def training_root(tmp_path, monkeypatch, name="t", winners=(1, 2), **kw):
    arms = arms_vs(tmp_path, CANDIDATE, "neural")
    return collect(tmp_path, monkeypatch, name, arms, list(winners), purpose="training", **kw)


def test_training_selection_takes_verified_completed_episodes_and_records_rejections(
    sha_dirs, monkeypatch
):
    tmp_path = sha_dirs
    root = training_root(tmp_path, monkeypatch, winners=(1, 2, 1))
    # Make the third fixture's candidate episode ineligible as the runner would.
    third = read_collection(root).ledger.completed[2]
    path = next(Path(e) for e in third["episodes"] if e.endswith("episode-1.json"))
    episode = json.loads(path.read_text())
    episode.update(eligible=False, invalid_reasons=["deadline"])
    path.write_text(json.dumps(episode))
    selection = select_training_episodes([root], CANDIDATE, held_out=NO_HELD_OUT)
    assert len(selection.episodes) == 2
    assert all(
        e["policy_sha"] == CANDIDATE and e["purpose"] == "training" for e in selection.episodes
    )
    assert all(selection.verify(e) for e in selection.episodes)
    reasons = sorted(tuple(r["reasons"]) for r in selection.rejected)
    # Three rival-policy episodes (off-policy) and one ineligible candidate episode.
    assert reasons == [("ineligible", "deadline")] + [("other_policy",)] * 3


def test_the_held_out_set_must_be_stated_explicitly(sha_dirs, monkeypatch):
    train = training_root(sha_dirs, monkeypatch)
    with pytest.raises(TypeError, match="held_out"):
        select_training_episodes([train], CANDIDATE)
    with pytest.raises(SplitRefused, match="NO_HELD_OUT"):
        select_training_episodes([train], CANDIDATE, held_out=str(train))
    assert len(select_training_episodes([train], CANDIDATE, held_out=NO_HELD_OUT).episodes) == 2


def test_selection_verify_trusts_only_its_own_training_verifiers(sha_dirs, monkeypatch):
    train = training_root(sha_dirs, monkeypatch)
    selection = select_training_episodes([train], CANDIDATE, held_out=NO_HELD_OUT)
    episode = selection.episodes[0]
    assert selection.verify(episode) is True
    identity = next(iter(selection.verifiers))
    forged = TrainingSelection(
        CANDIDATE, episodes=selection.episodes, verifiers={identity: lambda _: True}
    )
    assert forged.verify(episode) is False
    other = TrainingSelection(INCUMBENT, episodes=selection.episodes, verifiers=selection.verifiers)
    assert other.verify(episode) is False


def test_evaluation_collections_are_never_training_input(sha_dirs, monkeypatch):
    tmp_path = sha_dirs
    held = collect(tmp_path, monkeypatch, "e", arms_vs(tmp_path, CANDIDATE, "neural"), [1, 2])
    with pytest.raises(SplitRefused, match="held out"):
        select_training_episodes([held], CANDIDATE, held_out=NO_HELD_OUT)
    with pytest.raises(SplitRefused, match="held out"):
        TrainingEpisodeVerifier(held, CANDIDATE)
    # An evaluation episode relabelled as training is still refused by the verifier.
    train = training_root(tmp_path, monkeypatch)
    verifier = TrainingEpisodeVerifier(train, CANDIDATE)
    row = read_collection(held).ledger.completed[0]
    episode = json.loads(Path(row["episodes"][0]).read_text())
    assert episode["purpose"] == "evaluation"
    assert verifier({**episode, "purpose": "training"}) is False
    with pytest.raises(SplitRefused, match="both training input and held out"):
        select_training_episodes([train], CANDIDATE, held_out=[train])
    with pytest.raises(SplitRefused, match="must be an evaluation"):
        select_training_episodes([held], CANDIDATE, held_out=[train])


def test_a_training_match_found_in_a_held_out_collection_is_refused(sha_dirs, monkeypatch):
    tmp_path = sha_dirs
    train = training_root(tmp_path, monkeypatch)
    held = collect(tmp_path, monkeypatch, "e", arms_vs(tmp_path, CANDIDATE, "scripted"), [1, 2])
    assert len(select_training_episodes([train], CANDIDATE, held_out=[held]).episodes) == 2
    # Splice one of the held-out match identities into the training ledger.
    held_id = read_collection(held).ledger.completed[0]["match_id"]
    ledger = train / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[1]["match_id"] = held_id
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SplitRefused, match="held-out evaluation"):
        select_training_episodes([train], CANDIDATE, held_out=[held])


@pytest.mark.parametrize("damage", ["replay", "plan_sha", "purpose", "relabelled_plan", "ledger"])
def test_unverified_training_episodes_refuse_the_batch(sha_dirs, monkeypatch, damage):
    tmp_path = sha_dirs
    root = training_root(tmp_path, monkeypatch)
    row = read_collection(root).ledger.completed[0]
    path = next(Path(e) for e in row["episodes"] if e.endswith("episode-1.json"))
    episode = json.loads(path.read_text())
    if damage == "replay":
        with (root / row["match_id"] / "replay.jsonl").open("ab") as stream:
            stream.write(b" ")
    elif damage == "plan_sha":
        episode["plan_sha256"] = "0" * 64
    elif damage == "purpose":
        episode["purpose"] = "evaluation"
    elif damage == "relabelled_plan":
        manifest = json.loads((root / "collection.json").read_text())
        manifest["plan"]["purpose"] = "evaluation"
        (root / "collection.json").write_text(json.dumps(manifest))
    elif damage == "ledger":
        # Drop fixture 0's completed row: the fixture is open, as after a crash.
        ledger = root / "ledger.jsonl"
        kept = [
            line
            for line in ledger.read_text().splitlines()
            if not (json.loads(line)["event"] == "completed" and json.loads(line)["fixture"] == 0)
        ]
        ledger.write_text("".join(line + "\n" for line in kept))
    path.write_text(json.dumps(episode))
    if damage == "ledger":
        # Fixture 0 is now open: its episodes have no completed row and are never read.
        selection = select_training_episodes([root], CANDIDATE, held_out=NO_HELD_OUT)
        assert all(e["verification"]["match_id"] != row["match_id"] for e in selection.episodes)
        assert TrainingEpisodeVerifier(root, CANDIDATE)(episode) is False
        return
    with pytest.raises(SplitRefused):
        select_training_episodes([root], CANDIDATE, held_out=NO_HELD_OUT)


def test_ledger_episode_paths_cannot_escape_their_match(sha_dirs, monkeypatch):
    tmp_path = sha_dirs
    root = training_root(tmp_path, monkeypatch)
    ledger = root / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[1]["episodes"][0] = str(root / "other" / "episode-1.json")
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SplitRefused, match="escapes"):
        select_training_episodes([root], CANDIDATE, held_out=NO_HELD_OUT)


def test_evaluation_and_split_modules_do_not_import_torch():
    code = textwrap.dedent(
        """
        import sys
        class Refuse:
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("torch import refused")
        sys.meta_path.insert(0, Refuse())
        import anima3.neural.evaluation, anima3.neural.splits
        assert "torch" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)
