import hashlib
import json

import pytest

from anima3.neural.live import EPISODE_SCHEMA, TRAINING_RULES, LiveReplayVerifier


def test_option_failures_remain_experience_but_policy_substitution_is_rejected():
    from anima3.neural.live import policy_override

    assert not policy_override({"intervention": True, "scope": "option_execution"})
    assert not policy_override({"type": "action_finished", "success": False})
    assert policy_override({"scope": "policy_override"})
    assert policy_override({"policy_override": True, "scope": "option_execution"})


def test_bridge_transport_errors_are_not_silently_treated_as_empty_pumps():
    from anima3.body import BodyError, BridgeBody

    body = object.__new__(BridgeBody)
    body._rpc = lambda _: {"ok": False, "error": "game packet decode failed"}
    with pytest.raises(BodyError, match="game packet decode failed"):
        body.pump(50)
    body._rpc = lambda _: {"ok": True, "applied": 3}
    assert body.pump(50) == 3


def test_option_cancelled_by_changed_world_is_a_logged_noop_not_a_substitution():
    from types import SimpleNamespace

    from anima3.neural.live import dispatch_option

    calls = []
    executor = SimpleNamespace(start=lambda *args: calls.append(args) or True)
    frame = SimpleNamespace(mask=[True, False, True])
    cancelled = dispatch_option(executor, 1, frame, None, 5.0, True)
    assert cancelled == {"execution_started": False, "cancellation_reason": "mask_changed"}
    assert calls == []
    assert dispatch_option(executor, 0, frame, None, 5.0, False)["cancellation_reason"] is None
    assert calls == []
    assert (
        dispatch_option(executor, 2, frame, None, 5.0, False)["cancellation_reason"]
        == "context_stale"
    )
    assert dispatch_option(executor, 2, frame, None, 5.0, True)["execution_started"] is True
    assert len(calls) == 1 and calls[0][0] == 2


def fixture(tmp_path):
    match_id = "a" * 32
    rows = [
        {
            "seq": 0,
            "t": 0,
            "type": "header",
            "schema": 1,
            "id": match_id,
            "rules": TRAINING_RULES,
            "players": [{"serial": 1}, {"serial": 2}],
        },
        {"seq": 1, "t": 100, "type": "damage", "actor": 1, "target": 2, "amount": 10},
        {
            "seq": 2,
            "t": 200,
            "type": "end",
            "id": match_id,
            "complete": True,
            "training": True,
            "winner": 1,
            "score": [1, 0],
            "aborted": None,
        },
    ]
    data = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    meta = {**rows[-1], "sha256": hashlib.sha256(data).hexdigest(), "rules": TRAINING_RULES}
    root = tmp_path / match_id
    root.mkdir()
    (root / "replay.jsonl").write_bytes(data)
    (root / "replay.meta.json").write_text(json.dumps(meta))
    episode = {
        "schema": EPISODE_SCHEMA,
        "domain": "servuo",
        "sampling": "categorical",
        "eligible": True,
        "invalid_reasons": [],
        "actor": 1,
        "opponent": 2,
        "verification": {"match_id": match_id, "replay_sha256": meta["sha256"]},
        "transitions": [
            {"reward": 0.0, "done": False, "dt": 0.5, "accepted": True},
            {"reward": 1.0, "done": True, "dt": 1.3, "accepted": True},
        ],
        "final_result": {"winner": 1, "outcome": 1.0},
    }
    return episode, root


def test_real_outcome_comes_from_replay_not_declared_reward(tmp_path):
    episode, _ = fixture(tmp_path)
    verifier = LiveReplayVerifier(tmp_path)
    assert verifier(episode)
    episode["final_result"]["outcome"] = -1
    episode["transitions"][-1]["reward"] = -1
    assert not verifier(episode)


@pytest.mark.parametrize(
    "damage", ["replay", "opponent", "intervention", "shaping", "path", "accepted", "duration"]
)
def test_bad_live_evidence_rejected(tmp_path, damage):
    episode, root = fixture(tmp_path)
    if damage == "replay":
        with (root / "replay.jsonl").open("ab") as stream:
            stream.write(b" ")
    if damage == "opponent":
        episode["opponent"] = 999
    if damage == "intervention":
        episode["invalid_reasons"].append("deadline")
    if damage == "shaping":
        episode["transitions"][0]["reward"] = 0.1
    if damage == "path":
        episode["verification"]["match_id"] = "../escape"
    if damage == "accepted":
        episode["transitions"][0]["accepted"] = False
    if damage == "duration":
        episode["transitions"][0]["dt"] = 0
    assert not LiveReplayVerifier(tmp_path)(episode)


def test_second_actor_failure_stops_first_without_waiting_full_match(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    from anima3.neural import live

    stop = threading.Event()
    bodies = [
        SimpleNamespace(
            observe=lambda serial=i: SimpleNamespace(player=SimpleNamespace(serial=serial))
        )
        for i in (1, 2)
    ]

    def actor(body, opponent, worker, log, stop):
        if worker == "broken":
            raise RuntimeError("second actor failed")
        assert stop.wait(2), "first actor was not stopped when the other actor failed"
        raise RuntimeError("first actor stopped")

    monkeypatch.setattr(live, "_actor", actor)
    with pytest.raises(RuntimeError, match="second actor failed"):
        live.fight(bodies, ["waiting", "broken"], tmp_path, stop)
    assert stop.is_set()


@pytest.mark.parametrize(
    "known,hidden,expected", [(False, False, 1), (True, False, 0), (False, True, 0)]
)
def test_match_preparation_requests_only_visible_unknown_opponent_health(
    tmp_path, monkeypatch, known, hidden, expected
):
    import threading
    from types import SimpleNamespace

    from anima3.neural import live

    state = {
        "id": "a" * 32,
        "phase": "Countdown",
        "opponent": 2,
        "rules": TRAINING_RULES,
        "round": 1,
    }
    obs = SimpleNamespace(
        player=SimpleNamespace(serial=1, hits=100, pos=SimpleNamespace(x=1, y=1)),
        mobiles=[SimpleNamespace(serial=2, hidden=hidden, hits_max=100 if known else 0)],
        new_journal=[],
    )
    sent = []
    body = SimpleNamespace(
        observe=lambda: obs, act=sent.append, pump=lambda _: state.update(phase="Idle")
    )
    monkeypatch.setattr(live, "server_state", lambda _: dict(state))
    result = live._actor(body, 2, None, tmp_path, threading.Event())
    assert result["id"] == state["id"]
    assert len([p for p in sent if p["type"] == "StatusRequest"]) == expected


@pytest.mark.parametrize("ending", ["Idle", "RoundOver"])
def test_actor_rpc_timing_uses_actual_clock_but_retains_original_observation_age(
    tmp_path, monkeypatch, ending
):
    import threading
    from types import SimpleNamespace

    from anima3.neural import live

    clock = [100.0]
    state = {
        "id": "a" * 32,
        "phase": "Fighting",
        "opponent": 2,
        "rules": TRAINING_RULES,
        "round": 1,
    }
    obs = SimpleNamespace(
        player=SimpleNamespace(serial=1, hits=100, pos=SimpleNamespace(x=1, y=1)),
        mobiles=[SimpleNamespace(serial=2, hidden=False, hits_max=100)],
        new_journal=[],
    )
    taken, submitted = [], []

    def act(_):
        clock[0] += 0.2  # A blocking state request and a blocking executor RPC.

    def pump(_):
        clock[0] += 0.05
        state["phase"] = ending

    def submit(frame, epoch, now=None, *, observed_at=None):
        submitted.append((clock[0], observed_at))
        return 1

    def drain():
        if state["phase"] != "Fighting":
            return [
                {
                    "type": "inference_rejected",
                    "reason": "deadline",
                    "epoch": f"{state['id']}:1:1",
                }
            ]
        return []

    worker = SimpleNamespace(
        policy_sha="b" * 64,
        startup_error=None,
        busy=False,
        reset=lambda _: None,
        drain_events=drain,
        take=lambda epoch, now: taken.append(now),
        submit=submit,
    )
    executor = SimpleNamespace(
        reset=lambda *a: None,
        observe=lambda *a, **k: SimpleNamespace(mask=[True, True]),
        tick=lambda *a: [{"type": "CastSpell", "spell": 1}],
        drain_events=list,
    )
    body = SimpleNamespace(observe=lambda: obs, act=act, pump=pump)
    monkeypatch.setattr(live.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(live, "server_state", lambda _: dict(state))
    monkeypatch.setattr(live, "CombatExecutor", lambda _: executor)
    result = live._actor(body, 2, worker, tmp_path, threading.Event())
    assert taken == pytest.approx([100.4])
    assert submitted[0] == pytest.approx((100.4, 100.0))
    assert result["collection_end_phase"] == ending
    assert result["invalid_reasons"] == ["deadline"]
    # RoundOver stops collection, but does not manufacture a verified outcome.
    assert "verification" not in result and "final_result" not in result
    events = [
        json.loads(line)
        for line in (tmp_path / state["id"] / "actor-1.jsonl").read_text().splitlines()
    ]
    rpc = [row for row in events if row["type"] == "bridge_rpc"]
    assert [row["duration_ms"] for row in rpc if row["operation"].startswith("act:")] == (
        pytest.approx([200.0, 200.0])
    )
    assert [row["duration_ms"] for row in rpc if row["operation"] == "pump"] == (
        pytest.approx([50.0])
    )


def test_later_connection_failure_keeps_prior_verified_match_available(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from anima3.neural import live

    match_id = "a" * 32
    args = SimpleNamespace(
        user_a="a",
        user_b="b",
        matches=2,
        password_a_env="TEST_A",
        password_b_env="TEST_B",
        log_dir=tmp_path,
        checkpoint=tmp_path,
        deadline_ms=500,
        opponent="scripted",
        opponent_checkpoint=None,
        web="https://example.test",
    )
    monkeypatch.setenv("TEST_A", "local-test-only")
    monkeypatch.setenv("TEST_B", "local-test-only")
    worker = SimpleNamespace(
        ready=True, startup_error=None, policy_sha="b" * 64, close=lambda: None
    )
    monkeypatch.setattr(
        live, "InferenceClient", lambda *a, **k: SimpleNamespace(start=lambda: worker)
    )

    def connect(args, user, *rest):
        return SimpleNamespace(
            serial=1 if user == "a" else 2, act=lambda _: None, close=lambda: None
        )

    monkeypatch.setattr(live, "connect_training", connect)
    monkeypatch.setattr(live, "pump", lambda *a: None)
    monkeypatch.setattr(live, "wait_training_idle", lambda *a: None)
    monkeypatch.setattr(live, "prepare", lambda b, **k: {"serial": b.serial})
    monkeypatch.setattr(
        live,
        "poll",
        lambda b: {
            "challenge": {
                "id": match_id,
                "rules": TRAINING_RULES,
                "opponent": 3 - b.serial,
            }
        },
    )
    calls = []

    def fight(*args):
        calls.append(True)
        if len(calls) > 1:
            raise RuntimeError("connection failed in second match")
        return [
            {
                "id": match_id,
                "actor": serial,
                "opponent": 3 - serial,
                "transitions": [{"reward": 0, "done": False}],
                "invalid_reasons": [],
                "policy_sha": worker.policy_sha if serial == 1 else None,
            }
            for serial in (1, 2)
        ]

    monkeypatch.setattr(live, "fight", fight)
    monkeypatch.setattr(
        live, "receipt_while_pumping", lambda *a: ({"winner": 1, "sha256": "c" * 64}, [])
    )
    monkeypatch.setattr(live.replay, "metrics", lambda _: {})
    monkeypatch.setattr(live, "LiveReplayVerifier", lambda _: lambda ep: True)
    with pytest.raises(RuntimeError, match="second match"):
        live.run(args)
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["collection_status"] == "stopped"
    assert len(report["matches"]) == 1
    assert report["matches"][0]["match_id"] == match_id
    assert len(report["matches"][0]["episodes"]) == 1
    assert "second match" in report["error"]
