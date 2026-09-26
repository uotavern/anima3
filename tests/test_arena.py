import json

import pytest

from anima3.arena import ObservedBody, read_policy, server_state
from anima3.arena_learning import BASELINE, atomic_json, cycle, evaluate, matches, train
from anima3.contract import Journal, Observation


def message(state, serial=0xFFFFFFFF):
    return Journal(serial, "System", "[ArenaState] " + json.dumps(state), 6, 53, 0)


def test_assignment_ignores_player_spoof_and_bad_state():
    good = {"id": "a" * 32, "phase": "Fighting", "opponent": 123, "round": 1, "build": "mage"}
    assert server_state(Observation(new_journal=[message(good, 444)])) is None
    assert server_state(Observation(new_journal=[message({**good, "opponent": -1})])) is None
    assert server_state(Observation(new_journal=[message(good)])) == good
    assert server_state(Observation(new_journal=[message(good), message({"phase": "Idle"})])) == {"phase": "Idle"}


def test_observation_keeps_journal_for_agent():
    class Body:
        def observe(self):
            return Observation()
    body = ObservedBody(Body())
    obs = Observation(new_journal=[message({"phase": "Idle"})])
    body.pending = obs
    assert body.observe() is obs
    assert not body.observe().new_journal


def test_policy_validation_and_atomic_replace(tmp_path):
    path = tmp_path / "champion.json"
    atomic_json(path, BASELINE)
    assert read_policy(path) == BASELINE
    atomic_json(path, {**BASELINE, "version": "bad command"})
    with pytest.raises(ValueError):
        read_policy(path)
    atomic_json(path, {**BASELINE, "build": "warrior", "playbook": "poison"})
    with pytest.raises(ValueError):
        read_policy(path, "warrior")
    assert list(tmp_path.iterdir()) == [path]


def result(i, policy="candidate-1", book="poison", win=True, swap=False, **extra):
    a, b = (101, 102) if not swap else (102, 101)
    return {"schema": 1, "event": "match_end", "id": f"{i:032x}", "build": "mage", "training": True,
            "valid": True, "aborted": False, "a": a, "b": b,
            "policy_a": policy if not swap else "baseline-v1", "policy_b": "baseline-v1" if not swap else policy,
            "playbook_a": book if not swap else "standard", "playbook_b": "standard" if not swap else book,
            "winner": 101 if win else 102, **extra}


def candidate():
    return {**BASELINE, "version": "candidate-1", "parent": "baseline-v1", "playbook": "poison", "training_ids": []}


def test_learning_uses_only_valid_exploration_against_champion():
    rows = [result(i, "explore-poison", win=i % 10 != 0) for i in range(20)]
    rows += [result(100+i, "explore-sustain", training=False) for i in range(80)]
    rows += [result(200+i, "explore-control", valid=False) for i in range(80)]
    learned = train(rows, BASELINE)
    assert learned["playbook"] == "poison"
    assert learned["parent"] == BASELINE["version"]
    assert len(learned["training_ids"]) == 20
    assert train(rows, {**BASELINE, "version": "another"}) is None


def test_fixed_evaluation_does_not_fish_for_later_wins():
    rows = [result(i, win=i < 20, swap=i % 2 == 0) for i in range(40)]
    rows += [result(i, swap=i % 2 == 0) for i in range(40, 200)]
    report = evaluate(rows, candidate(), BASELINE, 40)
    assert report["matches"] == 40
    assert report["wins"] == 20
    assert not report["passed"]
    assert report["sides"] == {"a": 20, "b": 20}


def test_promote_needs_new_heldout_balanced_matches_and_matching_parent():
    rows = [result(i, swap=i % 2 == 0) for i in range(40)]
    assert evaluate(rows, candidate(), BASELINE, 40)["passed"]
    assert not evaluate(rows, {**candidate(), "training_ids": [rows[0]["id"]]}, BASELINE, 40)["passed"]
    assert not evaluate(rows, {**candidate(), "parent": "different"}, BASELINE, 40)["passed"]
    assert not evaluate([result(i) for i in range(40)], candidate(), BASELINE, 40)["passed"]
    assert not evaluate([result(i, winner=None, swap=i % 2 == 0) for i in range(40)], candidate(), BASELINE, 40)["passed"]
    assert not evaluate([{**r, "valid": False} for r in rows], candidate(), BASELINE, 40)["passed"]
    with pytest.raises(ValueError):
        evaluate(rows, candidate(), BASELINE, 2)


def test_audit_dedup_partial_tail_and_corruption(tmp_path):
    path = tmp_path / "events.jsonl"
    line = json.dumps(result(1))
    path.write_text('\ufeff' + line + '\n' + line + '\n{"partial"')
    assert len(matches(path)) == 1
    path.write_text(line + '\nnot json\n')
    with pytest.raises(ValueError):
        matches(path)
    path.write_text(line + '\n' + json.dumps(result(1, win=False)) + '\n')
    with pytest.raises(ValueError):
        matches(path)


def test_full_train_evaluate_promote_and_rollback_artifact(tmp_path):
    events = tmp_path / "events.jsonl"
    directory = tmp_path / "policies"
    rows = [result(i, "explore-poison", swap=i % 2 == 0) for i in range(20)]
    events.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert cycle(events, directory, sample=40)["status"] == "evaluating"
    new = read_policy(directory / "candidate.json")
    rows += [result(100+i, new["version"], swap=i % 2 == 0) for i in range(40)]
    events.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert cycle(events, directory, sample=40, promote=False)["status"] == "passed"
    assert read_policy(directory / "champion.json")["version"] == "baseline-v1"
    assert cycle(events, directory, sample=40, promote=True)["status"] == "promoted"
    assert read_policy(directory / "champion.json")["version"] == new["version"]
    assert read_policy(directory / "history" / "baseline-v1.json") == BASELINE


def test_curriculum_explores_evaluates_then_resumes_on_rejection(tmp_path):
    import random

    from anima3.arena import curriculum_policy
    atomic_json(tmp_path / "champion.json", BASELINE)
    assert curriculum_policy(tmp_path, random.Random(1))["version"].startswith("explore-")
    atomic_json(tmp_path / "candidate.json", candidate())
    assert curriculum_policy(tmp_path, random.Random(1))["version"] == "candidate-1"
    atomic_json(tmp_path / "evaluation.json", {"candidate": "candidate-1", "status": "rejected"})
    assert curriculum_policy(tmp_path, random.Random(1))["version"].startswith("explore-")


def test_frozen_candidate_keeps_original_evaluation_sample(tmp_path):
    directory = tmp_path / "policies"
    atomic_json(directory / "champion.json", BASELINE)
    atomic_json(directory / "candidate.json", {**candidate(), "evaluation_sample": 40})
    rows = [result(i, win=i < 10, swap=i % 2 == 0) for i in range(40)]
    events = tmp_path / "events.jsonl"
    events.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    verdict = cycle(events, directory, sample=100)
    assert verdict["status"] == "rejected"
    assert verdict["sample"] == 40
