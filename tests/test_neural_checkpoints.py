"""Frozen policy model, checkpoint validation and real inference worker (needs anima3[neural]).

Model and round-trip cases come from the training branch's ``test_neural_training.py``;
training, PPO, live-update and optimizer/RNG resume cases belong to later units.
"""

from __future__ import annotations

import hashlib
import io
import json
import time

import pytest

torch = pytest.importorskip("torch")

from anima3.neural.checkpoints import (
    MAX_ARTIFACT_BYTES,
    load_checkpoint,
    save_checkpoint,
    schema_fingerprint,
)
from anima3.neural.inference import InferenceClient
from anima3.neural.model import PolicyConfig, RecurrentPolicy, masked_distribution
from anima3.neural.schema import ACTION_NAMES, FEATURE_NAMES, Frame

# Audited Reference BC selection and its rollback, recorded in the workspace's
# docs/model-evidence.md.  The tests never read those artifacts; they only pin the
# schema the artifacts were trained against.
AUDITED_SCHEMA_FINGERPRINT = "cae99f11897deff05dd799c56025cc8c6aeb8d7f0737fd0b67fb287df3680bcb"


def policy(hidden=16, encoder=16):
    torch.set_num_threads(1)
    torch.manual_seed(23)
    return RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), hidden, encoder))


def saved(tmp_path, model=None):
    manifest = save_checkpoint(tmp_path, model or policy(), training={"environment_steps": 12})
    return manifest, manifest["weights"]["sha256"]


def rewrite_weights(tmp_path, state) -> str:
    """Replace the weight blob with self-consistent bytes, so only content checks can refuse."""
    stream = io.BytesIO()
    torch.save(state, stream)
    payload = stream.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    name = f"weights-{digest}.pt"
    (tmp_path / name).write_bytes(payload)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["weights"] = {"file": name, "sha256": digest, "bytes": len(payload)}
    path.write_text(json.dumps(manifest))
    return digest


def edit_manifest(tmp_path, change) -> None:
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))


# --- model (ported) ---------------------------------------------------------------------------


def test_illegal_actions_have_exactly_zero_probability():
    logits = torch.tensor([[100.0, -20.0, 400.0]])
    distribution = masked_distribution(logits, torch.tensor([[False, True, False]]))
    assert distribution.probs.tolist() == [[0.0, 1.0, 0.0]]
    assert set(distribution.sample((50,)).flatten().tolist()) == {1}
    with pytest.raises(ValueError, match="at least one"):
        masked_distribution(logits, torch.zeros_like(logits, dtype=torch.bool))


def test_recurrent_reset_matches_new_episode():
    model = policy()
    features = torch.randn(2, 3, len(FEATURE_NAMES))
    hidden = torch.randn(2, model.config.hidden_size)
    reset = torch.zeros(2, 3, dtype=torch.bool)
    reset[:, 0] = True
    actual = model(features, hidden, reset)
    expected = model(features)
    for left, right in zip(actual, expected, strict=True):
        assert torch.allclose(left, right, atol=1e-6)


def test_gradient_crosses_recurrent_time_steps():
    model = policy()
    features = torch.randn(1, 5, len(FEATURE_NAMES), requires_grad=True)
    _, values, _ = model(features)
    values[0, -1].backward()
    assert float(features.grad[0, 0].abs().sum()) > 0


def test_act_never_samples_masked_action_and_rejects_nonfinite_input():
    model = policy()
    mask = torch.zeros(1, len(ACTION_NAMES), dtype=torch.bool)
    mask[0, 4] = True
    for deterministic in (True, False):
        action, log_prob, _, hidden = model.act(
            torch.zeros(1, len(FEATURE_NAMES)), mask, deterministic=deterministic
        )
        assert int(action) == 4 and float(log_prob) == 0.0
        assert hidden.shape == (1, model.config.hidden_size)
    with pytest.raises(ValueError, match="non-finite"):
        model.act(torch.full((1, len(FEATURE_NAMES)), float("nan")), mask)


# --- checkpoint round trip and schema (ported, now SHA-bound) ---------------------------------


def test_current_schema_is_the_audited_policy_schema():
    assert schema_fingerprint() == AUDITED_SCHEMA_FINGERPRINT
    assert (len(FEATURE_NAMES), len(ACTION_NAMES)) == (76, 27)


def test_checkpoint_roundtrip_is_hash_and_schema_checked(tmp_path):
    model = policy()
    manifest, sha = saved(tmp_path, model)
    assert manifest["automatic_promotion"] is False
    assert set(manifest) == {
        "format", "created_at", "config", "schema", "schema_fingerprint",
        "weights", "training", "automatic_promotion",
    }  # fmt: skip
    loaded = load_checkpoint(tmp_path, expected_sha256=sha)
    assert loaded.policy_sha == sha
    assert not loaded.model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[name])
    assert load_checkpoint(tmp_path / "manifest.json", expected_sha256=sha).policy_sha == sha
    weight_path = tmp_path / manifest["weights"]["file"]
    data = bytearray(weight_path.read_bytes())
    data[-20] ^= 1
    weight_path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(tmp_path, expected_sha256=sha)


def test_checkpoint_rejects_schema_drift(tmp_path):
    _, sha = saved(tmp_path)
    edit_manifest(tmp_path, lambda m: m["schema"]["actions"].__setitem__(0, "unknown-new-action"))
    with pytest.raises(ValueError, match="schema mismatch"):
        load_checkpoint(tmp_path, expected_sha256=sha)


def test_checkpoint_rejects_fingerprint_drift_even_with_matching_names(tmp_path):
    _, sha = saved(tmp_path)
    edit_manifest(tmp_path, lambda m: m.__setitem__("schema_fingerprint", "0" * 64))
    with pytest.raises(ValueError, match="schema mismatch"):
        load_checkpoint(tmp_path, expected_sha256=sha)


# --- caller-supplied identity -----------------------------------------------------------------


def test_expected_sha_is_mandatory_and_well_formed(tmp_path):
    _, sha = saved(tmp_path)
    with pytest.raises(TypeError):
        load_checkpoint(tmp_path)  # type: ignore[call-arg]
    for value in (None, "", sha[:-1], sha.upper(), sha + "0"):
        with pytest.raises(ValueError, match="64 lowercase"):
            load_checkpoint(tmp_path, expected_sha256=value)


def test_valid_checkpoint_with_another_identity_is_refused(tmp_path):
    saved(tmp_path)
    other = hashlib.sha256(b"rollback").hexdigest()
    with pytest.raises(ValueError, match="not the expected policy"):
        load_checkpoint(tmp_path, expected_sha256=other)


def test_substituted_weights_with_consistent_manifest_are_refused(tmp_path):
    # An attacker who rewrites both blob and manifest still cannot satisfy the caller's SHA.
    _, sha = saved(tmp_path)
    replacement = {k: v + 1 for k, v in policy().state_dict().items()}
    new_sha = rewrite_weights(tmp_path, replacement)
    assert new_sha != sha
    with pytest.raises(ValueError, match="not the expected policy"):
        load_checkpoint(tmp_path, expected_sha256=sha)
    assert load_checkpoint(tmp_path, expected_sha256=new_sha).policy_sha == new_sha


def test_manifest_naming_expected_sha_over_other_bytes_is_refused(tmp_path):
    manifest, sha = saved(tmp_path)
    other = policy(hidden=16, encoder=16)
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.add_(0.5)
    stream = io.BytesIO()
    torch.save(other.state_dict(), stream)
    payload = stream.getvalue()
    (tmp_path / manifest["weights"]["file"]).write_bytes(payload)
    edit_manifest(tmp_path, lambda m: m["weights"].__setitem__("bytes", len(payload)))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_checkpoint(tmp_path, expected_sha256=sha)


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        "weights.pt",
        {"file": "../weights.pt", "sha256": "0" * 64, "bytes": 10},
        {"file": "", "sha256": "0" * 64, "bytes": 10},
        {"file": "weights.pt", "sha256": "0" * 64, "bytes": True},
        {"file": "weights.pt", "sha256": "0" * 64, "bytes": MAX_ARTIFACT_BYTES + 1},
    ],
)
def test_malformed_weight_descriptor_is_refused(tmp_path, descriptor):
    _, sha = saved(tmp_path)
    edit_manifest(tmp_path, lambda m: m.__setitem__("weights", descriptor))
    with pytest.raises(ValueError, match="invalid checkpoint artifact"):
        load_checkpoint(tmp_path, expected_sha256=sha)


# --- dimensions ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("feature_dim", len(FEATURE_NAMES) - 1, "dimensions do not match schema"),
        ("action_dim", len(ACTION_NAMES) + 1, "dimensions do not match schema"),
        ("feature_dim", float(len(FEATURE_NAMES)), "must be integers"),
        ("hidden_size", True, "must be integers"),
        ("hidden_size", 4, "invalid recurrent policy size"),
    ],
)
def test_declared_dimensions_must_match_schema(tmp_path, field, value, error):
    _, sha = saved(tmp_path)
    edit_manifest(tmp_path, lambda m: m["config"].__setitem__(field, value))
    with pytest.raises(ValueError, match=error):
        load_checkpoint(tmp_path, expected_sha256=sha)


def test_unknown_config_field_is_refused(tmp_path):
    _, sha = saved(tmp_path)
    edit_manifest(tmp_path, lambda m: m["config"].__setitem__("dropout", 0))
    with pytest.raises(ValueError, match="exactly the policy dimensions"):
        load_checkpoint(tmp_path, expected_sha256=sha)


def test_tensors_must_match_declared_hidden_size(tmp_path):
    _, sha = saved(tmp_path)
    edit_manifest(tmp_path, lambda m: m["config"].__setitem__("hidden_size", 32))
    with pytest.raises(ValueError, match="do not match the declared policy dimensions"):
        load_checkpoint(tmp_path, expected_sha256=sha)


def test_missing_or_extra_tensor_is_refused(tmp_path):
    saved(tmp_path)
    state = policy().state_dict()
    missing = dict(state)
    missing.pop("value_head.bias")
    sha = rewrite_weights(tmp_path, missing)
    with pytest.raises(ValueError, match="declared policy dimensions"):
        load_checkpoint(tmp_path, expected_sha256=sha)
    extra = {**state, "backdoor.weight": torch.zeros(2)}
    sha = rewrite_weights(tmp_path, extra)
    with pytest.raises(ValueError, match="declared policy dimensions"):
        load_checkpoint(tmp_path, expected_sha256=sha)


# --- finite, typed tensors and safe deserialization ------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weights_are_refused_even_when_hash_matches(tmp_path, bad):
    saved(tmp_path)
    state = policy().state_dict()
    state["policy_head.weight"][0, 0] = bad
    sha = rewrite_weights(tmp_path, state)
    with pytest.raises(ValueError, match="non-finite"):
        load_checkpoint(tmp_path, expected_sha256=sha)


def test_save_refuses_non_finite_weights(tmp_path):
    model = policy()
    with torch.no_grad():
        model.value_head.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        save_checkpoint(tmp_path, model)
    assert not (tmp_path / "manifest.json").exists()


def test_integer_tensors_are_refused(tmp_path):
    saved(tmp_path)
    state = {k: v.to(torch.int64) for k, v in policy().state_dict().items()}
    sha = rewrite_weights(tmp_path, state)
    with pytest.raises(ValueError, match="floating-point"):
        load_checkpoint(tmp_path, expected_sha256=sha)


def test_non_tensor_payload_is_refused(tmp_path):
    saved(tmp_path)
    sha = rewrite_weights(tmp_path, {"encoder.0.weight": [1.0, 2.0]})
    with pytest.raises(ValueError, match="tensor state dictionary"):
        load_checkpoint(tmp_path, expected_sha256=sha)


class _Exploit:
    executed = False

    def __reduce__(self):
        return (_mark_executed, ())


def _mark_executed():
    _Exploit.executed = True
    return {}


def test_pickle_code_in_weights_is_never_executed(tmp_path):
    saved(tmp_path)
    sha = rewrite_weights(tmp_path, {"encoder.0.weight": _Exploit()})
    _Exploit.executed = False
    with pytest.raises(ValueError, match="could not be read safely"):
        load_checkpoint(tmp_path, expected_sha256=sha)
    assert _Exploit.executed is False


def test_manifest_with_nan_or_non_object_is_refused(tmp_path):
    _, sha = saved(tmp_path)
    path = tmp_path / "manifest.json"
    original = path.read_text()
    path.write_text(original.replace('"environment_steps": 12', '"environment_steps": NaN'))
    with pytest.raises(ValueError, match="non-finite number"):
        load_checkpoint(tmp_path, expected_sha256=sha)
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_checkpoint(tmp_path, expected_sha256=sha)
    path.write_text("{")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_checkpoint(tmp_path, expected_sha256=sha)


def test_oversized_manifest_is_refused(tmp_path):
    _, sha = saved(tmp_path)
    (tmp_path / "manifest.json").write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="manifest exceeds size limit"):
        load_checkpoint(tmp_path, expected_sha256=sha)


def test_unsupported_format_is_refused(tmp_path):
    _, sha = saved(tmp_path)
    edit_manifest(tmp_path, lambda m: m.__setitem__("format", "anima3-recurrent-policy-v0"))
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(tmp_path, expected_sha256=sha)


# --- real worker subprocess ------------------------------------------------------------------


def wait_for(callable_, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = callable_()
        if result:
            return result
        time.sleep(0.005)
    raise AssertionError("timed out waiting for inference worker")


def test_real_worker_serves_only_the_expected_checkpoint(tmp_path):
    _, sha = saved(tmp_path)
    client = InferenceClient(tmp_path, expected_sha256=sha, deadline_ms=10_000, deterministic=True)
    client.start()
    try:
        wait_for(lambda: client.ready or client.startup_error)
        assert client.startup_error is None
        assert client.policy_sha == sha
        assert (client.feature_size, client.action_size, client.hidden_size) == (
            len(FEATURE_NAMES), len(ACTION_NAMES), 16,
        )  # fmt: skip
        client.reset("round-1")
        mask = [False] * len(ACTION_NAMES)
        mask[3] = True
        assert client.submit(Frame([0.0] * len(FEATURE_NAMES), mask), "round-1")
        decision = wait_for(lambda: client.take("round-1"))
        assert decision.action == 3 and decision.policy_sha == sha
        assert len(decision.hidden_out) == 16 and client.hidden == decision.hidden_out
    finally:
        client.close()


def test_real_worker_refuses_checkpoint_with_another_identity(tmp_path):
    saved(tmp_path)
    other = hashlib.sha256(b"not this policy").hexdigest()
    client = InferenceClient(tmp_path, expected_sha256=other).start()
    try:
        wait_for(lambda: (client.ready, client.startup_error)[1])
        assert "not the expected policy" in client.startup_error
        assert not client.ready and client.policy_sha is None
    finally:
        client.close()
