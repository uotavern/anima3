import hashlib
import json
from types import SimpleNamespace

import pytest

from anima3 import sparring
from anima3.learning_run import experiment
from anima3.ten_duels import report


def recording(root, n, winner=1):
    mid = f"{n:032x}"
    rules = sparring.TRAINING_RULES
    rows = [
        {
            "type": "header",
            "schema": 1,
            "seq": 0,
            "t": 0,
            "id": mid,
            "training": True,
            "rules": rules,
            "players": [{"serial": 1}, {"serial": 2}],
        },
        {"type": "damage", "seq": 1, "t": 10, "actor": 1, "amount": 20},
        {
            "type": "end",
            "seq": 2,
            "t": 100,
            "id": mid,
            "complete": True,
            "dropped": 0,
            "winner": winner,
            "score": [1, 0],
            "training": True,
            "aborted": None,
        },
    ]
    raw = b"".join((json.dumps(r) + "\n").encode() for r in rows)
    sha = hashlib.sha256(raw).hexdigest()
    folder = root / mid
    folder.mkdir()
    (folder / "replay.jsonl").write_bytes(raw)
    meta = {
        "id": mid,
        "complete": True,
        "sha256": sha,
        "winner": winner,
        "score": [1, 0],
        "training": True,
        "aborted": None,
        "durationMs": 100,
    }
    (folder / "replay.meta.json").write_text(json.dumps(meta))
    row = {
        "schema": 1,
        "id": mid,
        "event": "match_end",
        "a": 1,
        "b": 2,
        "learner": 1,
        "opponent": "fixed-scripted",
        "valid": True,
        "training": True,
        "aborted": False,
        "source": "verified-server-replay",
        "winner": winner,
        "sha256": sha,
    }
    with (root / "learning.jsonl").open("a") as out:
        out.write(json.dumps(row) + "\n")
    return mid


def test_report_checks_temporal_history_and_does_not_claim_superiority(tmp_path):
    first = recording(tmp_path, 1)
    second = recording(tmp_path, 2)
    folder = tmp_path / "brain-0"
    folder.mkdir()
    logs = [
        {"event": "plan_requested", "context": [first], "verifiedHistory": []},
        {
            "event": "plan_applied",
            "context": [first],
            "plan": {"primary": "standard", "responses": {}, "reason": "Start"},
        },
        {"event": "plan_requested", "context": [second], "verifiedHistory": [{"id": first}]},
        {
            "event": "plan_applied",
            "context": [second],
            "plan": {"primary": "poison", "responses": {}, "reason": "Change"},
        },
    ]
    (folder / "strategy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
    r = report(tmp_path)
    assert (
        r["completed"] == 2 and r["experienceUsedMatches"] == 1 and r["strategyChangedMatches"] == 1
    )
    assert r["performanceImprovement"] == "not-established"
    logs[0]["verifiedHistory"] = [{"id": second}]
    (folder / "strategy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
    with pytest.raises(ValueError, match="future"):
        report(tmp_path)


def test_fixed_opponent_is_part_of_resume_identity(tmp_path):
    args = SimpleNamespace(
        host="local",
        port=2593,
        web="local",
        user_a="a",
        user_b="b",
        backend="jev",
        model="jev",
        minimum=10,
        sample=40,
        opponent="fixed-scripted",
    )
    experiment(tmp_path, args)
    args.opponent = "same"
    with pytest.raises(ValueError, match="settings changed"):
        experiment(tmp_path, args)


def test_wait_idle_refuses_unrelated_match(monkeypatch, tmp_path):
    now = [10.0]
    monkeypatch.setattr(sparring.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(sparring, "server_state", lambda obs: obs.state)

    class Body:
        def __init__(self, serial):
            self.serial = serial

        def observe(self):
            return SimpleNamespace(
                player=SimpleNamespace(serial=self.serial),
                state={"phase": "Fighting", "rules": sparring.TRAINING_RULES, "opponent": 999},
            )

        def pump(self, ms):
            now[0] += 0.1

        def act(self, _):
            pass

    with pytest.raises(RuntimeError, match="unrelated duel"):
        sparring.wait_training_idle([Body(1), Body(2)], tmp_path, lambda: False)


def test_fixed_opponent_never_receives_learning_session(monkeypatch, tmp_path):
    monkeypatch.setenv("SPAR_PASSWORD_A", "local")
    monkeypatch.setenv("SPAR_PASSWORD_B", "local")
    bodies = [SimpleNamespace(serial=i, act=lambda _: None, close=lambda: None) for i in (1, 2)]
    calls = []
    monkeypatch.setattr(sparring, "connect_training", lambda *a: bodies.pop(0))
    monkeypatch.setattr(sparring, "pump", lambda *a: None)
    monkeypatch.setattr(sparring, "wait_training_idle", lambda *a: None)
    monkeypatch.setattr(sparring, "prepare", lambda b: {"serial": b.serial, "name": str(b.serial)})
    monkeypatch.setattr(
        sparring,
        "poll",
        lambda b: {"challenge": {"rules": sparring.TRAINING_RULES, "opponent": 1, "id": "a" * 32}},
    )
    monkeypatch.setattr(
        sparring,
        "load_policy",
        lambda backend, **kw: SimpleNamespace(
            name=backend, calls=0, errors=0, exhausted=False, max_calls=None
        ),
    )
    monkeypatch.setattr(
        sparring,
        "build_session",
        lambda *a: SimpleNamespace(remember=lambda *r: calls.append("remember")),
    )

    def fight(bodies, chosen, log, stop, clients, strategies, opening, burst_combo):
        assert clients[1].name == "scripted" and strategies[1] is None
        assert chosen[1]["version"] == "fixed-scripted-v1"
        return "a" * 32

    monkeypatch.setattr(sparring, "fight", fight)
    monkeypatch.setattr(
        sparring,
        "receipt_while_pumping",
        lambda *a: (
            {"winner": 1, "sha256": "x"},
            [{"players": [{"serial": 1}, {"serial": 2}], "type": "header"}],
        ),
    )
    sparring.main(
        [
            "--user-a",
            "a",
            "--user-b",
            "b",
            "--brain",
            "hybrid",
            "--backend",
            "jev",
            "--opponent",
            "fixed-scripted",
            "--matches",
            "1",
            "--log-dir",
            str(tmp_path),
        ]
    )
    assert calls == ["remember"]
