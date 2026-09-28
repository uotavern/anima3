import json
from types import SimpleNamespace

import pytest

from anima3.arena_learning import BASELINE, atomic_json, eligible
from anima3.arena_policy import BoundedClient
from anima3.learning_run import experiment, progress, verify_receipts
from tests.test_arena_improvements import make_receipt


def test_experiment_refuses_changed_backend_and_partial_append(tmp_path):
    args = SimpleNamespace(
        host="localhost",
        port=2599,
        web="http://localhost:8097",
        user_a="a",
        user_b="b",
        backend="jev",
        model="jev-1.13.0",
        minimum=10,
        sample=40,
    )
    experiment(tmp_path, args)
    experiment(tmp_path, args)
    args.backend = "scripted"
    with pytest.raises(ValueError, match="settings changed"):
        experiment(tmp_path, args)
    args.backend = "jev"
    (tmp_path / "learning.jsonl").write_text("{")
    with pytest.raises(ValueError, match="Incomplete"):
        experiment(tmp_path, args)


def test_model_budget_is_bounded_and_secret_errors_are_not_logged():
    class Broken:
        name = "jev"

        def choose(self, *_):
            raise RuntimeError("secret credentials in HTTP exception")

    client = BoundedClient(Broken(), max_calls=1)
    d = client.choose("", "", {"a": "a"})
    assert d.error == "provider request failed" and client.errors == 1
    assert client.choose("", "", {"a": "a"}).error == "provider call budget exhausted"
    assert client.calls == 1 and client.exhausted


def test_receipt_rechecks_server_outcome_and_excludes_budget_fallback(tmp_path):
    row = make_receipt(tmp_path)
    row["build"] = "mage"
    p = tmp_path / row["id"] / "replay.jsonl"
    import hashlib

    records = [json.loads(s) for s in p.read_text().splitlines()]
    records[0]["rules"] = "7x-magic-explosion-classic-training"
    raw = "".join(json.dumps(r) + "\n" for r in records).encode()
    p.write_bytes(raw)
    row["sha256"] = hashlib.sha256(raw).hexdigest()
    verify_receipts(tmp_path, [row])
    row["winner"] = 2
    with pytest.raises(ValueError, match="outcome"):
        verify_receipts(tmp_path, [row])
    row.update(winner=1, modelBudgetExhausted=True)
    verify_receipts(tmp_path, [row])
    assert not eligible(row)


def test_reward_statistics_update_without_claiming_promotion(tmp_path):
    row = {
        "valid": True,
        "aborted": False,
        "training": True,
        "build": "mage",
        "a": 1,
        "b": 2,
        "winner": 1,
        "policy_a": "explore-poison",
        "playbook_a": "poison",
        "policy_b": "baseline-v1",
    }
    state = progress(tmp_path, [row], BASELINE, 10)
    assert state["arms"]["poison"]["posteriorWinMean"] == pytest.approx(2 / 3)
    assert state["arms"]["poison"]["remaining"] == 9
    assert state["champion"] == BASELINE
    state = progress(tmp_path, [row, {**row, "winner": 2}], BASELINE, 10)
    assert state["arms"]["poison"]["posteriorWinMean"] == 0.5


def test_full_receipt_learning_waits_for_all_arms_then_promotes_fixed_sample(tmp_path):
    import hashlib

    from anima3.learning_run import update
    from anima3.magic import PLAYBOOKS

    policies = tmp_path / "policies"
    atomic_json(policies / "champion.json", {**BASELINE, "backend": "jev", "model": "pinned"})
    events = tmp_path / "learning.jsonl"
    events.touch()
    serial = 0

    def record(policy, book, win, swap):
        nonlocal serial
        serial += 1
        mid = f"{serial:032x}"
        a, b = (2, 1) if swap else (1, 2)
        row = {
            "schema": 1,
            "event": "match_end",
            "id": mid,
            "valid": True,
            "training": True,
            "aborted": False,
            "build": "mage",
            "a": a,
            "b": b,
            "winner": 1 if win else 2,
            "policy_a": "baseline-v1" if swap else policy,
            "policy_b": policy if swap else "baseline-v1",
            "playbook_a": "standard" if swap else book,
            "playbook_b": book if swap else "standard",
            "source": "verified-server-replay",
        }
        data = [
            {
                "id": mid,
                "training": True,
                "rules": "7x-magic-explosion-classic-training",
                "players": [{"serial": a}, {"serial": b}],
            },
            {"type": "end", "complete": True, "aborted": None, "winner": row["winner"]},
        ]
        raw = "".join(json.dumps(r) + "\n" for r in data).encode()
        folder = tmp_path / mid
        folder.mkdir()
        (folder / "replay.jsonl").write_bytes(raw)
        row["sha256"] = hashlib.sha256(raw).hexdigest()
        with events.open("a") as f:
            f.write(json.dumps(row) + "\n")

    for book in PLAYBOOKS:
        for i in range(10):
            record("explore-" + book, book, book == "poison", i % 2)
        report = update(tmp_path)
        if book != list(PLAYBOOKS)[-1]:
            assert report["status"] == "collecting_training"
    assert report["status"] == "evaluating"
    candidate = json.loads((policies / "candidate.json").read_text())
    assert candidate["backend"] == "jev" and candidate["model"] == "pinned"
    for i in range(39):
        record(candidate["version"], "poison", True, i % 2)
    assert update(tmp_path)["status"] == "evaluating"
    record(candidate["version"], "poison", True, True)
    assert update(tmp_path)["status"] == "promoted"
    assert json.loads((policies / "champion.json").read_text())["version"] == candidate["version"]
