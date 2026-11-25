"""Unit-6 live runner boundaries under controlled time; no server, bridge, model or torch.

The first cases come from the training branch's ``test_neural_live.py``; the rest
cover the unit-6 boundaries: arm identity, torch/checkpoint absence, SHA refusal,
mask and stale-state handling reaching the runner, round resets, and the bounded
resumable collection ledger.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from anima3.neural import live
from anima3.neural.inference import InferenceClient
from anima3.neural.live import (
    EPISODE_SCHEMA,
    EXECUTION_FINGERPRINT,
    TRAINING_RULES,
    Arm,
    CollectionPlan,
    CollectionRefused,
    DuplicateMatch,
    Ledger,
    LiveReplayVerifier,
    NeuralArmUnavailable,
    await_workers,
    dispatch_option,
    open_collection,
    policy_override,
    preflight_arm,
)
from anima3.neural.schema import schema_fingerprint

POLICY_SHA = "ab" * 32
OTHER_SHA = "cd" * 32
MATCH = "a" * 32


def neural_arm(tmp_path, sha=POLICY_SHA):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "manifest.json").write_text("{}")
    return Arm("neural", str(checkpoint), sha)


def plan_for(tmp_path, matches=2, **overrides):
    values = {
        "endpoint": "lab.invalid:2593",
        "accounts": ("alpha", "beta"),
        "arms": (neural_arm(tmp_path), Arm("scripted")),
        "matches": matches,
        **overrides,
    }
    return CollectionPlan(**values)


# --- ported from the training branch ------------------------------------------------------


def test_option_failures_remain_experience_but_policy_substitution_is_rejected():
    assert not policy_override({"intervention": True, "scope": "option_execution"})
    assert not policy_override({"type": "action_finished", "success": False})
    assert policy_override({"scope": "policy_override"})
    assert policy_override({"policy_override": True, "scope": "option_execution"})


def test_option_cancelled_by_changed_world_is_a_logged_noop_not_a_substitution():
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


def replay_fixture(match_id=MATCH, seats=(1, 2), winner=1, **header):
    rows = [
        {
            "seq": 0,
            "t": 0,
            "type": "header",
            "schema": 1,
            "id": match_id,
            "rules": TRAINING_RULES,
            "ranked": False,
            "training": True,
            "players": [{"serial": s} for s in seats],
            **header,
        },
        {"seq": 1, "t": 100, "type": "damage", "actor": seats[0], "target": seats[1], "amount": 10},
        {
            "seq": 2,
            "t": 200,
            "type": "end",
            "id": match_id,
            "complete": True,
            "training": True,
            "winner": winner,
            "score": [1, 0],
            "aborted": None,
        },
    ]
    data = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    meta = {
        **rows[-1],
        "sha256": hashlib.sha256(data).hexdigest(),
        "rules": TRAINING_RULES,
        "ranked": False,
        "players": [{"serial": s} for s in seats],
    }
    return meta, data


def completed_row(fixture, match_id, episodes=()):
    return {
        "fixture": fixture,
        "event": "completed",
        "match_id": match_id,
        "episodes": list(episodes),
        "result": {"fixture": fixture, "match_id": match_id},
    }


def fixture(tmp_path, *, ledger=True):
    meta, data = replay_fixture()
    if ledger:
        rows = [{"fixture": 0, "event": "issued"}, completed_row(0, MATCH)]
        (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    root = tmp_path / MATCH
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
        "policy_sha": POLICY_SHA,
        "fixture": 0,
        "schema_fingerprint": schema_fingerprint(),
        "executor_sha256": EXECUTION_FINGERPRINT,
        "verification": {"match_id": MATCH, "replay_sha256": meta["sha256"]},
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
    "damage",
    ["replay", "opponent", "intervention", "shaping", "path", "accepted", "duration", "executor"],
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
    if damage == "executor":
        episode["executor_sha256"] = "0" * 64
    assert not LiveReplayVerifier(tmp_path)(episode)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_fingerprint", None),
        ("executor_sha256", None),
        ("policy_sha", None),
        ("policy_sha", "not-a-sha"),
        ("policy_sha", "AB" * 32),
        ("policy_sha", 7),
        ("schema_fingerprint", "0" * 64),
    ],
)
def test_verifier_requires_fingerprints_and_a_well_formed_policy(tmp_path, field, value):
    episode, _ = fixture(tmp_path)
    if value is None:
        del episode[field]
    else:
        episode[field] = value
    assert not LiveReplayVerifier(tmp_path)(episode)


@pytest.mark.parametrize("ledger", ["absent", "other_fixture", "failed", "corrupt"])
def test_verifier_requires_a_completed_ledger_row(tmp_path, ledger):
    episode, _ = fixture(tmp_path, ledger=ledger != "absent")
    path = tmp_path / "ledger.jsonl"
    if ledger == "other_fixture":
        episode["fixture"] = 1
    if ledger == "failed":
        rows = [{"fixture": 0, "event": "issued"}, {"fixture": 0, "event": "failed"}]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if ledger == "corrupt":
        path.write_text(path.read_text() + "{torn")
    assert not LiveReplayVerifier(tmp_path)(episode)
    if ledger == "absent":
        # The runner checks staged episodes before their ledger row exists.
        assert LiveReplayVerifier(tmp_path, require_ledger=False)(episode)


def test_verifier_binds_the_episode_to_the_declared_policy(tmp_path):
    episode, _ = fixture(tmp_path)
    assert LiveReplayVerifier(tmp_path, POLICY_SHA)(episode)
    assert not LiveReplayVerifier(tmp_path, OTHER_SHA)(episode)
    del episode["policy_sha"]
    assert not LiveReplayVerifier(tmp_path, POLICY_SHA)(episode)


def test_second_actor_failure_stops_first_without_waiting_full_match(tmp_path, monkeypatch):
    stop = threading.Event()
    bodies = [
        SimpleNamespace(
            observe=lambda serial=i: SimpleNamespace(player=SimpleNamespace(serial=serial))
        )
        for i in (1, 2)
    ]

    def actor(body, opponent, worker, log, stop, **_):
        if worker == "broken":
            raise RuntimeError("second actor failed")
        assert stop.wait(2), "first actor was not stopped when the other actor failed"
        raise RuntimeError("first actor stopped")

    monkeypatch.setattr(live, "_actor", actor)
    with pytest.raises(RuntimeError, match="second actor failed"):
        live.fight(bodies, ["waiting", "broken"], tmp_path, stop)
    assert stop.is_set()


def observation(serial=1, *, hidden=False, hits_max=100):
    return SimpleNamespace(
        player=SimpleNamespace(serial=serial, hits=100, pos=SimpleNamespace(x=1, y=1)),
        mobiles=[SimpleNamespace(serial=2, hidden=hidden, hits_max=hits_max)],
        new_journal=[],
    )


@pytest.mark.parametrize(
    "known,hidden,expected", [(False, False, 1), (True, False, 0), (False, True, 0)]
)
def test_match_preparation_requests_only_visible_unknown_opponent_health(
    tmp_path, monkeypatch, known, hidden, expected
):
    state = {"id": MATCH, "phase": "Countdown", "opponent": 2, "rules": TRAINING_RULES, "round": 1}
    obs = observation(hidden=hidden, hits_max=100 if known else 0)
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
    clock = [100.0]
    state = {"id": MATCH, "phase": "Fighting", "opponent": 2, "rules": TRAINING_RULES, "round": 1}
    obs = observation()
    taken, submitted = [], []

    def observe():
        clock[0] += 0.15  # The snapshot can age before the reply reaches the actor.
        return obs

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
            return [{"type": "inference_rejected", "reason": "deadline", "epoch": f"{MATCH}:1:1"}]
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
    body = SimpleNamespace(observe=observe, act=act, pump=pump)
    monkeypatch.setattr(live.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(live, "server_state", lambda _: dict(state))
    monkeypatch.setattr(live, "CombatExecutor", lambda _, **kw: executor)
    result = live._actor(body, 2, worker, tmp_path, threading.Event())
    assert taken == pytest.approx([100.7])
    assert submitted[0] == pytest.approx((100.7, 100.15))
    assert result["collection_end_phase"] == ending
    assert result["invalid_reasons"] == ["deadline"]
    # RoundOver stops collection, but does not manufacture a verified outcome.
    assert "verification" not in result and "final_result" not in result
    events = [
        json.loads(line) for line in (tmp_path / MATCH / "actor-1.jsonl").read_text().splitlines()
    ]
    rpc = [row for row in events if row["type"] == "bridge_rpc"]
    assert rpc[0]["operation"] == "observe"
    assert rpc[0]["duration_ms"] == pytest.approx(150.0)
    assert [row["duration_ms"] for row in rpc if row["operation"].startswith("act:")] == (
        pytest.approx([200.0, 200.0])
    )
    assert [row["duration_ms"] for row in rpc if row["operation"] == "pump"] == (
        pytest.approx([50.0])
    )


# --- the frozen worker reaching the executor through the actor --------------------------------


class FakeExecutor:
    """Records resets and starts; the mask/freshness of each frame is scripted."""

    def __init__(self, mask=(True, True)):
        self.mask = list(mask)
        self.resets, self.started = [], []

    def reset(self, opponent, epoch, now=None):
        self.resets.append(epoch)

    def observe(self, *a, **k):
        return SimpleNamespace(mask=list(self.mask), features=[0.0, 0.0])

    def tick(self, *a):
        return []

    def drain_events(self):
        return []

    def start(self, action, obs, now):
        self.started.append(action)
        return True


class FakeWorker:
    def __init__(self, decisions=(), policy_sha=POLICY_SHA):
        self.policy_sha = policy_sha
        self.startup_error = None
        self.busy = False
        self.decisions = list(decisions)
        self.resets, self.submitted, self.taken = [], [], []
        self.events = []

    def reset(self, epoch):
        self.resets.append(epoch)

    def drain_events(self):
        rows, self.events = self.events, []
        return rows

    def take(self, epoch, now):
        self.taken.append(epoch)
        return self.decisions.pop(0) if self.decisions else None

    def submit(self, frame, epoch, now=None, *, observed_at=None):
        self.submitted.append((epoch, observed_at))
        return 1


def decision(action=1, observed_at=100.0):
    return SimpleNamespace(
        action=action,
        features=(0.0, 0.0),
        mask=(True, True),
        log_prob=-0.5,
        value=0.1,
        hidden_in=(0.0,),
        request_id=1,
        infer_ms=2.0,
        observed_at=observed_at,
    )


def scripted_actor(tmp_path, monkeypatch, phases, *, worker, executor, state_every=1):
    """Drive ``_actor`` through ``phases`` (phase, round) with one loop per entry.

    A server update arrives only every ``state_every`` loops so freshness can age.
    """
    clock = [100.0]
    step = [0]

    def current():
        return phases[min(step[0], len(phases) - 1)]

    def state_for(_obs):
        if step[0] % state_every:
            return None
        phase, round_ = current()
        return {"id": MATCH, "phase": phase, "opponent": 2, "rules": TRAINING_RULES,
                "round": round_}  # fmt: skip

    def pump(_):
        clock[0] += 1.0
        step[0] += 1

    body = SimpleNamespace(observe=lambda: observation(), act=lambda _: None, pump=pump)
    monkeypatch.setattr(live.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(live, "server_state", state_for)
    monkeypatch.setattr(live, "CombatExecutor", lambda *_, **__: executor)
    return live._actor(body, 2, worker, tmp_path, threading.Event())


def test_stale_server_state_reaches_the_runner_as_a_cancelled_option(tmp_path, monkeypatch):
    executor, worker = FakeExecutor(), FakeWorker([None, None, None, None, decision()])
    phases = [("Fighting", 1)] * 5 + [("Idle", 1)]
    # Updates every fifth loop: by the fifth decision the last state is 4 s old.
    result = scripted_actor(
        tmp_path, monkeypatch, phases, worker=worker, executor=executor, state_every=5
    )
    (transition,) = result["transitions"]
    assert transition["cancellation_reason"] == "context_stale"
    assert transition["execution_started"] is False
    assert executor.started == []
    # Stale state also stops new submissions instead of relabelling old input.
    assert len(worker.submitted) == 3


def test_mask_change_reaches_the_runner_without_substituting_an_action(tmp_path, monkeypatch):
    executor, worker = FakeExecutor(mask=(True, False)), FakeWorker([decision(action=1)])
    result = scripted_actor(
        tmp_path, monkeypatch, [("Fighting", 1), ("Idle", 1)], worker=worker, executor=executor
    )
    (transition,) = result["transitions"]
    assert transition["action"] == 1
    assert transition["cancellation_reason"] == "mask_changed"
    assert executor.started == [] and result["invalid_reasons"] == []
    assert result["policy_sha"] == POLICY_SHA


def test_fresh_legal_decision_starts_the_sampled_option(tmp_path, monkeypatch):
    executor, worker = FakeExecutor(), FakeWorker([decision(action=1)])
    result = scripted_actor(
        tmp_path, monkeypatch, [("Fighting", 1), ("Idle", 1)], worker=worker, executor=executor
    )
    assert executor.started == [1]
    assert result["transitions"][0]["execution_started"] is True


def test_round_boundary_resets_executor_and_recurrent_state(tmp_path, monkeypatch):
    executor, worker = FakeExecutor(), FakeWorker()
    phases = [("Fighting", 1), ("Countdown", 2), ("Fighting", 2), ("Fighting", 2), ("Idle", 2)]
    result = scripted_actor(tmp_path, monkeypatch, phases, worker=worker, executor=executor)
    epoch_1, epoch_2 = f"{MATCH}:1:1", f"{MATCH}:2:1"
    assert executor.resets == [epoch_1, "inactive", epoch_2]
    assert worker.resets == [epoch_1, "inactive", epoch_2]
    # No decision is taken outside Fighting and none crosses into the new round.
    assert worker.taken == [epoch_1, epoch_2, epoch_2]
    assert "multiple_rounds" in result["invalid_reasons"]


def test_rejected_decision_in_this_round_makes_the_episode_ineligible(tmp_path, monkeypatch):
    executor, worker = FakeExecutor(), FakeWorker()
    worker.events = [{"type": "inference_rejected", "reason": "deadline", "epoch": f"{MATCH}:1:1"}]
    result = scripted_actor(
        tmp_path, monkeypatch, [("Fighting", 1), ("Idle", 1)], worker=worker, executor=executor
    )
    assert result["invalid_reasons"] == ["deadline"]


def test_worker_failure_during_match_stops_the_actor(tmp_path, monkeypatch):
    executor, worker = FakeExecutor(), FakeWorker()
    worker.startup_error = "inference worker exited"
    with pytest.raises(RuntimeError, match="neural worker failed"):
        scripted_actor(tmp_path, monkeypatch, [("Fighting", 1)], worker=worker, executor=executor)


# --- arm identity, fallback refusal and SHA mismatch -----------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "neural", "checkpoint": "x"},
        {"kind": "neural", "checkpoint": "x", "policy_sha256": "AB" * 32},
        {"kind": "neural", "policy_sha256": POLICY_SHA},
        {"kind": "scripted", "checkpoint": "x"},
        {"kind": "scripted", "policy_sha256": POLICY_SHA},
        {"kind": "jev"},
    ],
)
def test_arm_identity_is_explicit_and_immutable(kwargs):
    with pytest.raises(ValueError):
        Arm(**kwargs)


def test_missing_checkpoint_refuses_the_neural_arm(tmp_path):
    with pytest.raises(NeuralArmUnavailable, match="checkpoint manifest not found"):
        preflight_arm(Arm("neural", str(tmp_path / "absent"), POLICY_SHA))
    preflight_arm(Arm("scripted"))  # no torch or checkpoint needed


def test_missing_torch_refuses_without_importing_it(tmp_path):
    with pytest.raises(NeuralArmUnavailable, match="neural' extra"):
        preflight_arm(neural_arm(tmp_path), find_spec=lambda name: None)


class Spy:
    def __init__(self):
        self.calls = []

    def connect(self, account):
        self.calls.append(("connect", account))
        raise AssertionError("no login may happen for an unavailable neural arm")

    def receipt(self, *a):
        raise AssertionError("no receipt may be fetched")


def test_absent_torch_refuses_the_run_before_any_login_or_challenge(tmp_path):
    spy = Spy()
    with pytest.raises(NeuralArmUnavailable):
        live.run_collection(
            plan_for(tmp_path),
            tmp_path / "run",
            connect=spy.connect,
            fetch_receipt=spy.receipt,
            lock_dir=tmp_path / "locks",
            find_spec=lambda name: None,
            client_factory=lambda *a, **k: pytest.fail("worker started without torch"),
        )
    assert spy.calls == []
    ledger = Ledger.load(tmp_path / "run" / "ledger.jsonl")
    assert ledger.issued == 0
    # Unit 8: the refusal precedes the fresh collection's manifest and results.
    assert not (tmp_path / "run" / "collection.json").exists()
    assert not (tmp_path / "run" / "results.json").exists()


@pytest.mark.parametrize(
    "worker,reason",
    [
        (SimpleNamespace(ready=False, startup_error="ModuleNotFoundError: torch"), "worker failed"),
        (SimpleNamespace(ready=True, startup_error=None, policy_sha=OTHER_SHA), "another policy"),
        (SimpleNamespace(ready=False, startup_error=None), "did not become ready"),
    ],
)
def test_workers_must_report_the_declared_policy(tmp_path, worker, reason):
    arms = (neural_arm(tmp_path), Arm("scripted"))
    with pytest.raises(NeuralArmUnavailable, match=reason):
        await_workers([worker, None], arms, timeout=0.05, sleep=lambda _: None)


WRONG_IDENTITY_WORKER = r"""
import json
print(json.dumps(dict(type="ready",policy_sha="cd"*32,hidden_size=2,feature_size=2,action_size=2)),flush=True)
import sys
sys.stdin.read()
"""


def test_checkpoint_sha_mismatch_refuses_the_run_end_to_end(tmp_path):
    """A real InferenceClient whose worker serves another policy never reaches a login."""
    spy = Spy()

    def client_factory(checkpoint, *, expected_sha256, deadline_ms):
        return InferenceClient(
            checkpoint,
            expected_sha256=expected_sha256,
            deadline_ms=deadline_ms,
            worker_command=[sys.executable, "-u", "-c", WRONG_IDENTITY_WORKER],
        )

    with pytest.raises(NeuralArmUnavailable, match="unexpected policy SHA256"):
        live.run_collection(
            plan_for(tmp_path),
            tmp_path / "run",
            connect=spy.connect,
            fetch_receipt=spy.receipt,
            lock_dir=tmp_path / "locks",
            find_spec=lambda name: object(),
            client_factory=client_factory,
            ready_timeout=30,
        )
    assert spy.calls == []


# --- bounded, resumable collections ----------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"matches": 0},
        {"matches": 101},
        {"matches": True},
        {"accounts": ("alpha", "ALPHA")},
        {"arms": (Arm("scripted"), Arm("scripted"))},
        {"endpoint": ""},
        {"deadline_ms": 0},
    ],
)
def test_collection_plan_is_bounded(tmp_path, overrides):
    with pytest.raises(ValueError):
        plan_for(tmp_path, **overrides)


def test_existing_collection_needs_explicit_resume_with_identical_plan(tmp_path):
    plan = plan_for(tmp_path)
    log = tmp_path / "run"
    with pytest.raises(CollectionRefused, match="nothing to resume"):
        open_collection(log, plan, resume=True)
    open_collection(log, plan)
    with pytest.raises(CollectionRefused, match="already exists"):
        open_collection(log, plan)
    other_policy = plan_for(tmp_path, arms=(neural_arm(tmp_path, OTHER_SHA), Arm("scripted")))
    with pytest.raises(CollectionRefused, match="identical plan"):
        open_collection(log, other_policy, resume=True)
    with pytest.raises(CollectionRefused, match="identical plan"):
        open_collection(log, plan_for(tmp_path, matches=3), resume=True)
    assert open_collection(log, plan, resume=True).issued == 0
    manifest_path = log / "collection.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["executor_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CollectionRefused, match="original combat executor"):
        open_collection(log, plan, resume=True)


def test_interrupted_fixture_is_spent_and_budget_cannot_be_exceeded(tmp_path):
    plan = plan_for(tmp_path, matches=2)
    log = tmp_path / "run"
    ledger = open_collection(log, plan)
    ledger.issue(0, challenger=1, opponent=2)  # crash before an outcome is recorded
    resumed = open_collection(log, plan, resume=True)
    assert resumed.issued == 1 and resumed.entries[-1]["event"] == "unresolved"
    resumed.issue(1, challenger=2, opponent=1)
    resumed.close(1, "completed", match_id=MATCH, episodes=[], result={})
    with pytest.raises(CollectionRefused, match="budget is exhausted"):
        open_collection(log, plan, resume=True)


@pytest.mark.parametrize(
    "lines,error",
    [
        (["not json"], "corrupt ledger"),
        ([{"fixture": "0", "event": "issued"}], "malformed"),
        ([{"fixture": 1, "event": "issued"}], "in order"),
        ([{"fixture": True, "event": "issued"}], "malformed"),
        ([{"fixture": -1, "event": "issued"}], "malformed"),
        ([completed_row(0, MATCH)], "unknown"),
        ([{"fixture": 0, "event": "issued"}, {"fixture": 0, "event": "completed"}], "lacks"),
        (
            [{"fixture": 0, "event": "issued"}, {"fixture": 0, "event": "failed", "match_id": [1]}],
            "invalid match_id",
        ),
        (
            [{"fixture": 0, "event": "issued"}, {**completed_row(0, MATCH), "match_id": "A" * 32}],
            "invalid match_id",
        ),
        (
            [{"fixture": 0, "event": "issued"}, {**completed_row(0, MATCH), "episodes": [3]}],
            "episode list",
        ),
        (
            [
                {"fixture": 0, "event": "issued"},
                completed_row(0, MATCH),
                {"fixture": 1, "event": "issued"},
                completed_row(1, MATCH),
            ],
            "duplicate match",
        ),
    ],
)
def test_tampered_or_duplicate_ledgers_are_refused(tmp_path, lines, error):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines) + "\n"
    )
    with pytest.raises(CollectionRefused, match=error):
        Ledger.load(path)


class Harness:
    """Mock participant layer around the real ledger, receipts and verifier."""

    def __init__(self, monkeypatch, match_ids, *, seats_override=None, policy_sha=None):
        self.match_ids = list(match_ids)
        self.challenges, self.connected = [], []
        self.seats_override = seats_override
        self.closed = []

        def factory(checkpoint, *, expected_sha256, deadline_ms):
            worker = SimpleNamespace(
                ready=True,
                startup_error=None,
                policy_sha=policy_sha or expected_sha256,
                close=lambda: self.closed.append("worker"),
            )
            return SimpleNamespace(start=lambda: worker)

        self.factory = factory
        monkeypatch.setattr(live, "wait_training_idle", lambda bodies, **k: [1, 2])
        monkeypatch.setattr(live, "prepare", lambda body, **k: {"serial": body.serial})
        monkeypatch.setattr(
            live,
            "request_training_challenge",
            lambda body, *, opponent, **k: self.challenges.append((body.serial, opponent)),
        )
        monkeypatch.setattr(live, "accept_invitation", lambda body, **k: "f" * 32)
        monkeypatch.setattr(live, "fight", self.fight)

    def connect(self, account):
        self.connected.append(account)
        serial = 1 if account == "alpha" else 2
        return SimpleNamespace(
            serial=serial,
            act=lambda _: None,
            pump=lambda _: 0,
            close=lambda: self.closed.append(f"body-{serial}"),
        )

    def fight(self, bodies, workers, log, stop, **_):
        if not self.match_ids:
            raise RuntimeError("connection failed in a later match")
        match_id = self.match_ids.pop(0)
        return [
            {
                "id": match_id,
                "actor": serial,
                "opponent": 3 - serial,
                "transitions": [{"reward": 0.0, "done": False, "dt": 0.5, "accepted": True}],
                "invalid_reasons": [],
                "policy_sha": workers[serial - 1].policy_sha if workers[serial - 1] else None,
                "policy_kind": "neural" if workers[serial - 1] else "scripted",
            }
            for serial in (1, 2)
        ]

    def receipt(self, match_id, seats):
        return replay_fixture(match_id, self.seats_override or tuple(seats))

    def run(self, tmp_path, plan, **kwargs):
        return live.run_collection(
            plan,
            tmp_path / "run",
            connect=self.connect,
            fetch_receipt=self.receipt,
            lock_dir=tmp_path / "locks",
            client_factory=self.factory,
            find_spec=lambda name: object(),
            **kwargs,
        )


def test_verified_fixtures_alternate_seats_and_write_bound_episodes(tmp_path, monkeypatch):
    harness = Harness(monkeypatch, ["1" * 32, "2" * 32])
    results = harness.run(tmp_path, plan_for(tmp_path, matches=2))
    assert [r["match_id"] for r in results] == ["1" * 32, "2" * 32]
    assert harness.challenges == [(1, 2), (2, 1)]
    episode = json.loads(Path(results[0]["episodes"][0]).read_text())
    assert episode["policy_sha"] == POLICY_SHA and episode["eligible"] is True
    assert episode["verification"]["match_id"] == "1" * 32
    assert len(results[0]["episodes"]) == 1  # the scripted arm records no episode
    assert LiveReplayVerifier(tmp_path / "run", POLICY_SHA)(episode)
    report = json.loads((tmp_path / "run" / "results.json").read_text())
    assert report["collection_status"] == "complete" and report["issued"] == 2
    assert report["automatic_promotion"] is False


def test_failure_is_recorded_spent_and_resume_continues_within_budget(tmp_path, monkeypatch):
    plan = plan_for(tmp_path, matches=3)
    harness = Harness(monkeypatch, ["1" * 32])
    with pytest.raises(RuntimeError, match="later match"):
        harness.run(tmp_path, plan)
    ledger = Ledger.load(tmp_path / "run" / "ledger.jsonl")
    assert [(r["fixture"], r["event"]) for r in ledger.entries] == [
        (0, "issued"), (0, "completed"), (1, "issued"), (1, "failed"),
    ]  # fmt: skip
    report = json.loads((tmp_path / "run" / "results.json").read_text())
    assert len(report["matches"]) == 1 and "later match" in report["error"]
    with pytest.raises(CollectionRefused, match="already exists"):
        harness.run(tmp_path, plan)
    harness.match_ids = ["3" * 32]
    results = harness.run(tmp_path, plan, resume=True)
    assert [r["fixture"] for r in results] == [2]
    assert harness.challenges[-1] == (1, 2)  # fixture 2 keeps the alternation
    with pytest.raises(CollectionRefused, match="budget is exhausted"):
        harness.run(tmp_path, plan, resume=True)


def test_repeated_server_match_identity_is_refused(tmp_path, monkeypatch):
    harness = Harness(monkeypatch, ["1" * 32, "1" * 32])
    with pytest.raises(CollectionRefused, match="already in this collection"):
        harness.run(tmp_path, plan_for(tmp_path, matches=2))
    entries = Ledger.load(tmp_path / "run" / "ledger.jsonl").entries
    assert [r["event"] for r in entries] == ["issued", "completed", "issued", "failed"]
    assert entries[-1]["duplicate_match_id"] == "1" * 32 and "match_id" not in entries[-1]


def test_receipt_for_other_seats_is_an_invalid_fixture(tmp_path, monkeypatch):
    harness = Harness(monkeypatch, ["1" * 32], seats_override=(2, 1))
    with pytest.raises(CollectionRefused, match="invalid fixture"):
        harness.run(tmp_path, plan_for(tmp_path, matches=2))
    entries = Ledger.load(tmp_path / "run" / "ledger.jsonl").entries
    assert entries[-1]["event"] == "invalid_fixture" and entries[-1]["match_id"] == "1" * 32
    assert not (tmp_path / "run" / ("1" * 32) / "episode-1.json").exists()


def test_actor_reporting_another_policy_is_refused(tmp_path, monkeypatch):
    harness = Harness(monkeypatch, ["1" * 32])
    harness.fight_inner = harness.fight

    def fight(*args, **kwargs):
        actors = harness.fight_inner(*args, **kwargs)
        actors[0]["policy_sha"] = OTHER_SHA
        return actors

    monkeypatch.setattr(live, "fight", fight)
    with pytest.raises(RuntimeError, match="policy identity differs"):
        harness.run(tmp_path, plan_for(tmp_path, matches=1))
    assert Ledger.load(tmp_path / "run" / "ledger.jsonl").entries[-1]["event"] == "failed"


def test_shared_account_lock_blocks_a_concurrent_runner(tmp_path, monkeypatch):
    from anima3.sparring import account_locks

    harness = Harness(monkeypatch, ["1" * 32])
    with (
        account_locks(tmp_path / "locks", "LAB.invalid:2593", ["Alpha", "other"]),
        pytest.raises(OSError),
    ):
        harness.run(tmp_path, plan_for(tmp_path, matches=1))
    assert harness.connected == [] and harness.challenges == []


def test_torn_final_ledger_line_is_refused_for_manual_repair(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps({"fixture": 0, "event": "issued"}) + "\n" + '{"fixture": 0, "ev')
    with pytest.raises(CollectionRefused, match="torn final ledger line"):
        Ledger.load(path)


def test_results_json_keeps_every_completed_fixture_across_resumes(tmp_path, monkeypatch):
    plan = plan_for(tmp_path, matches=3)
    harness = Harness(monkeypatch, ["1" * 32])
    with pytest.raises(RuntimeError, match="later match"):
        harness.run(tmp_path, plan)
    harness.match_ids = ["3" * 32]
    assert [r["fixture"] for r in harness.run(tmp_path, plan, resume=True)] == [2]
    report = json.loads((tmp_path / "run" / "results.json").read_text())
    assert [m["match_id"] for m in report["matches"]] == ["1" * 32, "3" * 32]
    assert report["collection_status"] == "complete" and report["issued"] == 3


def test_only_a_held_run_lock_is_reported_as_another_runner(tmp_path, monkeypatch):
    import fcntl

    harness = Harness(monkeypatch, ["1" * 32])
    log = tmp_path / "run"
    log.mkdir()
    with (log / "run.lock").open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(CollectionRefused, match="another runner"):
            harness.run(tmp_path, plan_for(tmp_path, matches=1))


def test_disk_failure_is_not_mislabelled_as_a_concurrent_runner(tmp_path, monkeypatch):
    harness = Harness(monkeypatch, ["1" * 32])

    def full(*_):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(live, "atomic_json", full)
    with pytest.raises(OSError, match="No space left") as raised:
        harness.run(tmp_path, plan_for(tmp_path, matches=1))
    assert not isinstance(raised.value, CollectionRefused)


def test_failed_close_still_closes_bodies_before_releasing_locks(tmp_path, monkeypatch):
    from anima3.sparring import account_locks

    harness = Harness(monkeypatch, ["1" * 32])
    factory = harness.factory

    def broken_close():
        raise RuntimeError("worker close failed")

    def broken(*a, **k):
        worker = factory(*a, **k).start()
        worker.close = broken_close
        return SimpleNamespace(start=lambda: worker)

    harness.factory = broken
    held_while_closing = []
    connect = harness.connect

    def connect_checked(account):
        body = connect(account)
        inner = body.close

        def close():
            # The account locks must still be held while a body closes.
            try:
                with account_locks(tmp_path / "locks", "lab.invalid:2593", ["alpha", "x"]):
                    held_while_closing.append(False)
            except OSError:
                held_while_closing.append(True)
            inner()

        body.close = close
        return body

    harness.connect = connect_checked
    with pytest.raises(RuntimeError, match="worker close failed"):
        harness.run(tmp_path, plan_for(tmp_path, matches=1))
    assert harness.closed == ["body-2", "body-1"]
    assert held_while_closing == [True, True]
    # The run lock and account locks were released despite the failed close.
    with account_locks(tmp_path / "locks", "lab.invalid:2593", ["alpha", "beta"]):
        pass
    harness.factory = factory
    harness.connect = connect
    with pytest.raises(CollectionRefused, match="budget is exhausted"):
        harness.run(tmp_path, plan_for(tmp_path, matches=1), resume=True)


def two_neural(tmp_path):
    first = neural_arm(tmp_path)
    second = Arm("neural", first.checkpoint, OTHER_SHA)
    return plan_for(tmp_path, matches=1, arms=(first, second))


def test_second_arm_refusal_publishes_no_episode_for_the_first(tmp_path, monkeypatch):
    harness = Harness(monkeypatch, ["1" * 32])
    real = live.LiveReplayVerifier
    monkeypatch.setattr(
        live,
        "LiveReplayVerifier",
        lambda d, sha, **k: (lambda ep: False) if sha == OTHER_SHA else real(d, sha, **k),
    )
    with pytest.raises(CollectionRefused, match="failed receipt verification"):
        harness.run(tmp_path, two_neural(tmp_path))
    root = tmp_path / "run" / ("1" * 32)
    assert not list(root.glob("episode-*.json"))
    assert Ledger.load(tmp_path / "run" / "ledger.jsonl").entries[-1]["event"] == "invalid_fixture"


def test_crash_after_completed_row_publishes_staged_episodes_on_resume(tmp_path, monkeypatch):
    harness = Harness(monkeypatch, ["1" * 32])
    publish = live._publish

    def crash(paths):
        raise RuntimeError("process died before publication")

    monkeypatch.setattr(live, "_publish", crash)
    plan = plan_for(tmp_path, matches=2)
    with pytest.raises(RuntimeError, match="before publication"):
        harness.run(tmp_path, plan)
    root = tmp_path / "run" / ("1" * 32)
    assert not (root / "episode-1.json").exists() and (root / ".staged" / "episode-1.json").exists()
    monkeypatch.setattr(live, "_publish", publish)
    open_collection(tmp_path / "run", plan, resume=True)
    episode = json.loads((root / "episode-1.json").read_text())
    assert LiveReplayVerifier(tmp_path / "run", POLICY_SHA)(episode)


def test_episode_claiming_a_fixture_without_completed_row_is_refused(tmp_path, monkeypatch):
    harness = Harness(monkeypatch, ["1" * 32])
    with pytest.raises(RuntimeError, match="later match"):
        harness.run(tmp_path, plan_for(tmp_path, matches=2))  # fixture 1 fails
    log = tmp_path / "run"
    episode = json.loads((log / ("1" * 32) / "episode-1.json").read_text())
    assert LiveReplayVerifier(log, POLICY_SHA)(episode)
    # The same bytes claiming the failed fixture 1 are refused.
    assert not LiveReplayVerifier(log, POLICY_SHA)({**episode, "fixture": 1})


def test_failed_rows_record_the_known_match_identity(tmp_path, monkeypatch):
    harness = Harness(monkeypatch, ["1" * 32])

    def unavailable(match_id, seats):
        raise TimeoutError("server receipt missing")

    harness.receipt = unavailable
    with pytest.raises(TimeoutError):
        harness.run(tmp_path, plan_for(tmp_path, matches=1))
    row = Ledger.load(tmp_path / "run" / "ledger.jsonl").entries[-1]
    assert row["event"] == "failed" and row["match_id"] == "1" * 32


def test_actor_refuses_a_known_match_before_writing_its_directory(tmp_path, monkeypatch):
    state = {"id": MATCH, "phase": "Countdown", "opponent": 2, "rules": TRAINING_RULES, "round": 1}
    body = SimpleNamespace(observe=observation, act=lambda _: None, pump=lambda _: None)
    monkeypatch.setattr(live, "server_state", lambda _: dict(state))
    with pytest.raises(DuplicateMatch) as raised:
        live._actor(body, 2, None, tmp_path, threading.Event(), deadline=5, known_match_ids={MATCH})
    assert raised.value.match_id == MATCH
    assert not (tmp_path / MATCH).exists()


def test_live_runner_import_does_not_import_torch():
    import subprocess

    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, *a):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch import refused')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import anima3.neural.live\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
