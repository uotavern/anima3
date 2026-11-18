"""InferenceClient IPC boundary under controlled time; no torch, model or server needed.

The first six cases come from the training branch's ``test_neural_runtime.py``
(deferred from unit 4).  The fake workers now report a full SHA-256 policy
identity because the client requires one that matches the caller's expectation.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from io import StringIO
from types import SimpleNamespace

import pytest

from anima3.neural.inference import InferenceClient, require_sha256
from anima3.neural.schema import Frame

POLICY_SHA = "ab" * 32
OTHER_SHA = "cd" * 32

FAKE_WORKER = r"""
import json,sys,time
SHA="ab"*32
print(json.dumps(dict(type="ready",policy_sha=SHA,hidden_size=2,feature_size=2,action_size=2)),flush=True)
for line in sys.stdin:
    row=json.loads(line)
    time.sleep(0.08 if row["features"][0] else 0.002)
    print(json.dumps(dict(type="decision",id=row["id"],epoch=row["epoch"],action=1,log_prob=-0.5,value=0.2,hidden_out=[x+1 for x in row["hidden"]],policy_sha=SHA,infer_ms=2)),flush=True)
"""


def fake_client(**kwargs):
    return InferenceClient(
        "unused",
        expected_sha256=POLICY_SHA,
        worker_command=[sys.executable, "-u", "-c", FAKE_WORKER],
        **kwargs,
    )


def wait_for(callable_, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = callable_()
        if result:
            return result
        time.sleep(0.002)
    raise AssertionError("timed out waiting for inference test")


def startup_failure(client):
    """Poll the worker (``ready`` drains the inbox) and return any startup error."""
    _ = client.ready
    return client.startup_error


def test_inference_one_inflight_deadline_and_epoch_reset():
    # A generous real deadline keeps the subprocess round trip from flaking on slow
    # runners; expiry is then exercised by consuming at an explicitly later clock.
    client = fake_client(deadline_ms=1000).start()
    try:
        wait_for(lambda: client.ready)
        client.reset("one")
        assert client.submit(Frame([0, 0], [True, True]), "one")
        assert client.submit(Frame([0, 0], [True, True]), "one") is None
        decision = wait_for(lambda: client.take("one"))
        assert decision.hidden_in == (0, 0) and client.hidden == (1, 1)
        assert decision.policy_sha == POLICY_SHA
        assert client.submit(Frame([1, 0], [True, True]), "one")
        wait_for(lambda: not client._inbox.empty())
        assert client.take("one", time.monotonic() + 2.0) is None
        assert client.hidden == (1, 1)
        assert any(row.get("reason") == "deadline" for row in client.drain_events())
        assert client.submit(Frame([1, 0], [True, True]), "one")
        client.reset("two")
        wait_for(lambda: not client.busy)
        assert client.take("two") is None
        assert client.hidden == (0, 0)
        assert any(row.get("reason") == "context_changed" for row in client.drain_events())
        assert client.submit(Frame([0, 0], [True, True]), "two")
        decision = wait_for(lambda: client.take("two"))
        assert decision.hidden_in == (0, 0)
        assert decision.features == (0, 0)
    finally:
        client.close()


def test_inference_rechecks_sampled_mask():
    client = fake_client().start()
    try:
        wait_for(lambda: client.ready)
        client.submit(Frame([0, 0], [True, False]), "one")
        events = []

        def illegal_action_seen():
            events.extend(client.drain_events())
            return any(row.get("reason") == "illegal_action" for row in events)

        wait_for(illegal_action_seen)
        assert client.take("one") is None
        assert client.hidden == (0, 0)
    finally:
        client.close()


def ready_row(policy_sha=POLICY_SHA, **overrides):
    row = {
        "type": "ready",
        "policy_sha": policy_sha,
        "hidden_size": 2,
        "feature_size": 2,
        "action_size": 2,
    }
    row.update(overrides)
    return row


def queued_inference():
    """Controlled IPC timing without sleeping or invoking model computation."""
    client = InferenceClient("unused", expected_sha256=POLICY_SHA, deadline_ms=500)
    client.process = SimpleNamespace(stdin=StringIO())
    client._inbox.put((ready_row(), 99.0))
    client._poll(100.0)
    client.reset("one")
    return client


def queue_policy_response(client, received_at, policy_sha=POLICY_SHA):
    pending = client._pending
    client._inbox.put(
        (
            {
                "type": "decision",
                "id": pending["id"],
                "epoch": "one",
                "action": 1,
                "log_prob": -0.5,
                "value": 0.2,
                "hidden_out": [1, 1],
                "policy_sha": policy_sha,
                "infer_ms": 2,
            },
            received_at,
        )
    )


def test_inference_separates_observation_submission_arrival_and_consumption():
    client = queued_inference()
    assert client.submit(Frame([0, 0], [True, True]), "one", 100.2, observed_at=100.0)
    queue_policy_response(client, 100.203)
    decision = client.take("one", 100.49)
    assert decision is not None
    assert decision.observed_at == 100.0 and decision.submitted_at == 100.2
    assert decision.received_at == 100.203
    assert decision.age_ms == pytest.approx(3.0)
    assert client.hidden == (1, 1)


@pytest.mark.parametrize("poll_before_take", [False, True])
def test_fast_response_consumed_late_keeps_observation_deadline_and_hidden_uncommitted(
    poll_before_take,
):
    client = queued_inference()
    client.submit(Frame([0, 0], [True, True]), "one", 100.2, observed_at=100.0)
    queue_policy_response(client, 100.203)
    if poll_before_take:
        client._poll(100.3)
    # Request age is only 310 ms, but the source observation is already 510 ms old.
    assert client.take("one", 100.51) is None
    assert client.hidden == (0, 0)
    rejected = [row for row in client._events if row.get("reason") == "deadline"]
    assert len(rejected) == 1
    assert rejected[0]["observation_age_ms"] == pytest.approx(510)
    assert rejected[0]["request_age_ms"] == pytest.approx(310)
    assert rejected[0]["response_wait_ms"] == pytest.approx(307)
    assert rejected[0]["received_at"] == 100.203


def test_stale_input_is_skipped_without_new_sample_or_fake_freshness():
    client = queued_inference()
    assert client.submit(Frame([0, 0], [True, True]), "one", 100.6, observed_at=100.0) is None
    assert client.process.stdin.getvalue() == ""
    assert client.hidden == (0, 0) and client._pending is None
    assert client._events[-1]["type"] == "inference_skipped"
    with pytest.raises(ValueError, match="observation time"):
        client.submit(Frame([0, 0], [True, True]), "one", 100.6, observed_at=100.7)


def test_stdout_reader_records_arrival_before_actor_consumption():
    client = fake_client(deadline_ms=500).start()
    try:
        wait_for(lambda: client.ready)
        client.submit(Frame([0, 0], [True, True]), "one")
        wait_for(lambda: not client._inbox.empty())
        inbox_ready_at = time.monotonic()
        time.sleep(0.02)
        decision = client.take("one")
        assert decision is not None
        assert decision.received_at <= inbox_ready_at
        assert time.monotonic() - decision.received_at >= 0.02
    finally:
        client.close()


# --- Unit 5 policy-identity boundary ---------------------------------------------------------


@pytest.mark.parametrize(
    "value", [None, "", "ab" * 31, "AB" * 32, "zz" * 32, "ab" * 32 + "\n", b"ab" * 32]
)
def test_expected_policy_sha_is_required_and_well_formed(value):
    with pytest.raises(ValueError, match="64 lowercase"):
        require_sha256(value)
    with pytest.raises(ValueError, match="64 lowercase"):
        InferenceClient("unused", expected_sha256=value)


def test_expected_policy_sha_has_no_default():
    with pytest.raises(TypeError):
        InferenceClient("unused")  # type: ignore[call-arg]


def test_default_worker_command_passes_expected_sha_to_worker(monkeypatch):
    client = InferenceClient("checkpoint-dir", expected_sha256=POLICY_SHA)
    captured = {}

    def fake_popen(command, **_kwargs):
        captured["command"] = command
        raise OSError("not started in tests")

    monkeypatch.setattr("anima3.neural.inference.subprocess.Popen", fake_popen)
    with pytest.raises(OSError):
        client.start()
    command = captured["command"]
    assert command[command.index("--expected-sha256") + 1] == POLICY_SHA
    assert command[command.index("-m") + 1] == "anima3.neural.inference"


@pytest.mark.parametrize(
    "row,reason",
    [
        (ready_row(OTHER_SHA), "unexpected policy SHA256"),
        (ready_row(None), "unexpected policy SHA256"),
        (ready_row(POLICY_SHA.upper()), "unexpected policy SHA256"),
        (ready_row(hidden_size=0), "invalid policy dimensions"),
        (ready_row(feature_size=True), "invalid policy dimensions"),
        (ready_row(action_size="2"), "invalid policy dimensions"),
    ],
)
def test_worker_reporting_another_policy_or_bad_dimensions_never_becomes_ready(row, reason):
    client = InferenceClient("unused", expected_sha256=POLICY_SHA, deadline_ms=500)
    client.process = SimpleNamespace(stdin=StringIO())
    client._inbox.put((row, 99.0))
    client._poll(100.0)
    assert not client._ready and reason in client.startup_error
    assert client.policy_sha is None
    assert client.submit(Frame([0, 0], [True, True]), "one", 100.0) is None
    assert client.process.stdin.getvalue() == ""
    # A later, matching ready line cannot rescue a worker that already lied.
    client._inbox.put((ready_row(), 100.1))
    client._poll(100.2)
    assert not client._ready


def test_decision_from_another_policy_is_rejected_without_committing_hidden():
    client = queued_inference()
    assert client.submit(Frame([0, 0], [True, True]), "one", 100.0)
    queue_policy_response(client, 100.01, policy_sha=OTHER_SHA)
    assert client.take("one", 100.02) is None
    assert client.hidden == (0, 0)
    assert any(row.get("reason") == "policy_changed" for row in client._events)


def test_fake_worker_with_wrong_identity_is_refused_end_to_end():
    worker = FAKE_WORKER.replace('SHA="ab"*32', 'SHA="cd"*32')
    client = InferenceClient(
        "unused", expected_sha256=POLICY_SHA, worker_command=[sys.executable, "-u", "-c", worker]
    ).start()
    try:
        wait_for(lambda: startup_failure(client))
        assert not client.ready
        assert "unexpected policy SHA256" in client.startup_error
    finally:
        client.close()


def test_oversized_worker_line_is_fatal_without_unbounded_read(monkeypatch):
    monkeypatch.setattr("anima3.neural.inference.MAX_WORKER_LINE_CHARS", 1000)
    worker = (
        "import sys; sys.stdout.write('x' * 5000 + '\\n'); sys.stdout.flush(); sys.stdin.read()"
    )
    client = InferenceClient(
        "unused", expected_sha256=POLICY_SHA, worker_command=[sys.executable, "-u", "-c", worker]
    ).start()
    try:
        assert "oversized" in wait_for(lambda: startup_failure(client))
        assert not client.ready
    finally:
        client.close()


# --- Optional dependency boundary ------------------------------------------------------------

NO_TORCH_IMPORT = textwrap.dedent(
    """
    import importlib.abc, sys

    class RefuseTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch must not be imported by the base package")
            return None

    sys.meta_path.insert(0, RefuseTorch())
    import anima3
    import anima3.neural
    import anima3.neural.executor
    import anima3.neural.inference
    import anima3.neural.schema
    loaded = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch."))
    assert not loaded, loaded
    print("ok")
    """
)


def test_base_neural_imports_never_import_torch():
    result = subprocess.run(
        [sys.executable, "-c", NO_TORCH_IMPORT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_real_worker_reports_missing_or_unloadable_checkpoint_as_startup_failure(tmp_path):
    # Runs without torch (missing optional dependency) and with torch (missing manifest).
    client = InferenceClient(tmp_path / "missing", expected_sha256=POLICY_SHA).start()
    try:
        wait_for(lambda: startup_failure(client), timeout=60)
        assert not client.ready
    finally:
        client.close()
