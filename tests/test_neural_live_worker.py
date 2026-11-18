"""The real frozen worker and real executor joined by the live actor; needs the neural extra.

A temporary checkpoint is served by ``python -m anima3.neural.inference``. The game
body is a scripted fake, so no server, bridge or account is involved.
"""

from __future__ import annotations

import hashlib
import threading
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from test_neural_runtime import observation

from anima3.neural import live
from anima3.neural.checkpoints import save_checkpoint
from anima3.neural.inference import InferenceClient
from anima3.neural.live import Arm, NeuralArmUnavailable, await_workers, preflight_arm
from anima3.neural.model import PolicyConfig, RecurrentPolicy
from anima3.neural.schema import ACTION_NAMES, FEATURE_NAMES

MATCH = "e" * 32


def checkpoint(tmp_path):
    torch.manual_seed(5)
    model = RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), 16, 16))
    manifest = save_checkpoint(tmp_path, model, training={"environment_steps": 1})
    return manifest["weights"]["sha256"]


class Recording:
    """Delegate to the real client and remember each committed decision."""

    def __init__(self, client):
        self.client, self.decisions = client, []

    def __getattr__(self, name):
        return getattr(self.client, name)

    def take(self, epoch, now=None):
        decision = self.client.take(epoch, now)
        if decision is not None:
            self.decisions.append(decision)
        return decision


def test_real_worker_decision_reaches_the_real_executor_through_the_actor(tmp_path, monkeypatch):
    sha = checkpoint(tmp_path)
    arm = Arm("neural", str(tmp_path), sha)
    preflight_arm(arm)
    client = InferenceClient(tmp_path, expected_sha256=sha, deadline_ms=10_000).start()
    try:
        await_workers([client], [arm], timeout=60)
        worker = Recording(client)
        give_up = time.monotonic() + 60

        def state(_obs):
            done = len(worker.decisions) >= 2 or time.monotonic() > give_up
            return {
                "id": MATCH,
                "phase": "Idle" if done else "Fighting",
                "opponent": 2,
                "rules": live.TRAINING_RULES,
                "round": 1,
            }

        body = SimpleNamespace(
            observe=observation, act=lambda _: None, pump=lambda _: time.sleep(0.005)
        )
        monkeypatch.setattr(live, "server_state", state)
        result = live._actor(body, 2, worker, tmp_path / "log", threading.Event())
    finally:
        client.close()
    assert len(worker.decisions) >= 2, "no frozen-policy decision reached the actor"
    assert result["policy_sha"] == sha and result["invalid_reasons"] == []
    first = result["transitions"][0]
    assert len(first["features"]) == len(FEATURE_NAMES) and first["mask"][first["action"]]
    assert first["cancellation_reason"] in (None, "mask_changed", "executor_refused")
    # The first decision of a round starts from the reset recurrent state.
    assert first["hidden_in"] == [0.0] * 16


def test_real_worker_with_another_identity_never_becomes_an_arm(tmp_path):
    checkpoint(tmp_path)
    other = hashlib.sha256(b"not this policy").hexdigest()
    arm = Arm("neural", str(tmp_path), other)
    preflight_arm(arm)
    client = InferenceClient(tmp_path, expected_sha256=other).start()
    try:
        with pytest.raises(NeuralArmUnavailable, match="not the expected policy"):
            await_workers([client], [arm], timeout=60)
    finally:
        client.close()
