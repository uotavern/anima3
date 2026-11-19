"""Unit-7 evaluation statistics, selection gate and train/evaluation split; no torch.

Collections are produced by the real unit-6 ``run_collection`` (ledger, staged
publication, receipt and episode verification) around the mocked participant
layer of ``test_neural_live.Harness``. No server, bridge, account or model runs.
"""

from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from test_neural_live import OTHER_SHA, POLICY_SHA, Harness, neural_arm, replay_fixture

from anima3.neural import live
from anima3.neural.evaluation import (
    BOOTSTRAP_SEED,
    EvaluationPlan,
    EvaluationRefused,
    bootstrap_mean_interval,
    combine_records,
    record_decision,
    record_from_collection,
    selection_gate,
    wilson_interval,
)
from anima3.neural.live import Arm, CollectionPlan
from anima3.neural.splits import (
    SplitRefused,
    TrainingEpisodeVerifier,
    read_collection,
    select_training_episodes,
)

CANDIDATE = POLICY_SHA
INCUMBENT = "12" * 32
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
        {"games_per_opponent": 3},
        {"games_per_opponent": 102},
        {"games_per_opponent": True},
        {"incumbent_sha256": CANDIDATE},
        {"candidate_sha256": "AB" * 32},
        {"opponents": ()},
        {"opponents": ("scripted", "scripted")},
        {"opponents": ["scripted"]},
        {"opponents": ("rival",)},
        {"opponents": (f"neural:{CANDIDATE}",)},
        {"max_regression": float("nan")},
        {"max_regression": -0.1},
        {"min_lower_bound": float("inf")},
        {"max_excluded_per_opponent": 20},
        {"min_paired_games_per_opponent": 1},
        {"min_paired_games_per_opponent": 22},
        {"domain": "elsewhere"},
    ],
)
def test_evaluation_plan_is_bounded_and_explicit(overrides):
    with pytest.raises(ValueError):
        plan(**overrides)


def test_plan_identity_covers_every_threshold():
    assert plan().sha256 == plan().sha256
    assert plan().sha256 != plan(max_regression=0.2).sha256
    assert plan().sha256 != plan(games_per_opponent=22).sha256


# --- the gate on direct rows -----------------------------------------------------------------


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
    assert gate["paired_delta_ci95"][0] > 0
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


def test_equal_or_noisy_evidence_retains_the_incumbent():
    p = plan()
    same = {"scripted": scores(10), RIVAL_LABEL: scores(10)}
    gate = selection_gate(p, evaluation(INCUMBENT, same), evaluation(CANDIDATE, same))
    assert gate["passed"] is False
    assert "paired improvement lower bound not above threshold" in gate["reasons"]
    # A small budget needs an explicit, recorded minimum; the default is 20 per opponent.
    with pytest.raises(ValueError, match="minimum paired games"):
        plan(games_per_opponent=2)
    lenient = plan(max_excluded_per_opponent=4, min_paired_games_per_opponent=18)
    candidate_scores = scores(18)
    candidate_scores[0] = candidate_scores[1] = candidate_scores[2] = None
    gate = selection_gate(
        lenient,
        evaluation(INCUMBENT, {"scripted": scores(2), RIVAL_LABEL: scores(2)}),
        evaluation(CANDIDATE, {"scripted": candidate_scores, RIVAL_LABEL: scores(18)}),
    )
    assert gate["passed"] is False
    assert gate["reasons"] == ["insufficient paired games against scripted"]


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
    lenient = plan(max_excluded_per_opponent=1, min_paired_games_per_opponent=19)
    allowed = selection_gate(lenient, before, after)
    assert allowed["passed"] is True and len(allowed["excluded_rows"]) == 1


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


def test_every_decision_is_recorded_once_including_rejections(tmp_path):
    record = tmp_path / "selection.jsonl"
    before = evaluation(INCUMBENT, {"scripted": scores(2), RIVAL_LABEL: scores(10)})
    after = evaluation(CANDIDATE, {"scripted": scores(19), RIVAL_LABEL: scores(7)})
    rejected = selection_gate(plan(), before, after)
    record_decision(record, rejected)
    with pytest.raises(EvaluationRefused, match="already has a recorded decision"):
        record_decision(record, selection_gate(plan(), before, after))
    accepted = selection_gate(plan(max_regression=0.15), before, after)
    record_decision(record, accepted)
    rows = [json.loads(line) for line in record.read_text().splitlines()]
    assert [(r["sequence"], r["passed"]) for r in rows] == [(0, False), (1, True)]
    assert all(r["automatic_promotion"] is False for r in rows)
    forged = {**accepted, "plan_sha256": "0" * 64, "automatic_promotion": True}
    with pytest.raises(EvaluationRefused, match="never promotes"):
        record_decision(record, forged)
    with record.open("a") as stream:
        stream.write('{"torn"')
    with pytest.raises(EvaluationRefused, match="torn"):
        record_decision(record, {**accepted, "plan_sha256": "1" * 64})


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


def arms_vs(tmp_path, sha, opponent):
    other = Arm("scripted") if opponent == "scripted" else neural_arm(tmp_path / "o", RIVAL)
    (tmp_path / "o").mkdir(exist_ok=True)
    return (neural_arm(tmp_path / sha[:4], sha), other)


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


def test_live_collections_feed_the_gate_end_to_end(sha_dirs, monkeypatch, tmp_path):
    p = plan(games_per_opponent=4, max_regression=0.25, min_paired_games_per_opponent=4)
    win, loss = 1, 2
    records = {
        INCUMBENT: [
            collect(
                tmp_path,
                monkeypatch,
                "i-s",
                arms_vs(tmp_path, INCUMBENT, "scripted"),
                [loss, loss, loss, win],
            ),
            collect(
                tmp_path,
                monkeypatch,
                "i-r",
                arms_vs(tmp_path, INCUMBENT, "neural"),
                [loss, win, loss, loss],
            ),
        ],
        CANDIDATE: [
            collect(
                tmp_path,
                monkeypatch,
                "c-s",
                arms_vs(tmp_path, CANDIDATE, "scripted"),
                [win, win, win, win],
            ),
            collect(
                tmp_path,
                monkeypatch,
                "c-r",
                arms_vs(tmp_path, CANDIDATE, "neural"),
                [win, win, win, loss],
            ),
        ],
    }
    joined = {
        sha: combine_records(p, sha, [record_from_collection(r, sha) for r in roots])
        for sha, roots in records.items()
    }
    gate = selection_gate(p, joined[INCUMBENT], joined[CANDIDATE])
    assert gate["opponents"]["scripted"]["mean_score_delta"] == 0.75
    assert gate["opponents"][RIVAL_LABEL]["mean_score_delta"] == 0.5
    assert gate["passed"] is True
    assert gate["collections"]["candidate"]["scripted"] == str(records[CANDIDATE][0])
    with pytest.raises(EvaluationRefused, match="one opponent"):
        combine_records(
            p, CANDIDATE, [record_from_collection(records[CANDIDATE][0], CANDIDATE)] * 2
        )
    with pytest.raises(EvaluationRefused, match="exactly the planned"):
        combine_records(p, CANDIDATE, [record_from_collection(records[CANDIDATE][0], CANDIDATE)])
    with pytest.raises(EvaluationRefused, match="budget differs"):
        combine_records(
            plan(games_per_opponent=6, min_paired_games_per_opponent=4),
            CANDIDATE,
            [record_from_collection(r, CANDIDATE) for r in records[CANDIDATE]],
        )


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
    selection = select_training_episodes([root], CANDIDATE)
    assert len(selection.episodes) == 2
    assert all(
        e["policy_sha"] == CANDIDATE and e["purpose"] == "training" for e in selection.episodes
    )
    assert all(selection.verify(e) for e in selection.episodes)
    reasons = sorted(tuple(r["reasons"]) for r in selection.rejected)
    # Three rival-policy episodes (off-policy) and one ineligible candidate episode.
    assert reasons == [("ineligible", "deadline")] + [("other_policy",)] * 3


def test_evaluation_collections_are_never_training_input(sha_dirs, monkeypatch):
    tmp_path = sha_dirs
    held = collect(tmp_path, monkeypatch, "e", arms_vs(tmp_path, CANDIDATE, "neural"), [1, 2])
    with pytest.raises(SplitRefused, match="held out"):
        select_training_episodes([held], CANDIDATE)
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
        selection = select_training_episodes([root], CANDIDATE)
        assert all(e["verification"]["match_id"] != row["match_id"] for e in selection.episodes)
        assert TrainingEpisodeVerifier(root, CANDIDATE)(episode) is False
        return
    with pytest.raises(SplitRefused):
        select_training_episodes([root], CANDIDATE)


def test_ledger_episode_paths_cannot_escape_their_match(sha_dirs, monkeypatch):
    tmp_path = sha_dirs
    root = training_root(tmp_path, monkeypatch)
    ledger = root / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[1]["episodes"][0] = str(root / "other" / "episode-1.json")
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SplitRefused, match="escapes"):
        select_training_episodes([root], CANDIDATE)


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
