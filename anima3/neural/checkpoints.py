"""Hash checked, schema bound policy artifacts; never load arbitrary pickle code.

Loading is deliberately strict.  The caller supplies the SHA-256 of the weight
file it intends to run (for example an audited registry entry); a checkpoint is
refused unless its manifest names that exact file, the file bytes hash to it,
the observation/action schema and fingerprint match ``schema.py``, the declared
dimensions match the schema, and every tensor is a finite floating-point value
of the shape the policy expects.  Tensors are read with ``weights_only=True``.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .inference import require_sha256
from .model import PolicyConfig, RecurrentPolicy

FORMAT_VERSION = "anima3-recurrent-policy-v1"
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
CONFIG_FIELDS = ("feature_dim", "action_dim", "hidden_size", "encoder_size")


class CheckpointError(ValueError):
    """A policy artifact failed identity, schema, dimension or content validation."""


def schema_spec() -> dict:
    from .schema import ACTION_NAMES, FEATURE_NAMES, SCHEMA_VERSION

    return {
        "version": SCHEMA_VERSION,
        "features": list(FEATURE_NAMES),
        "actions": list(ACTION_NAMES),
    }


def schema_fingerprint() -> str:
    from .schema import schema_fingerprint as fingerprint

    return fingerprint()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _serialize(value: Any) -> bytes:
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.getvalue()


def _write_blob(root: Path, prefix: str, payload: bytes) -> dict:
    digest = hashlib.sha256(payload).hexdigest()
    name = f"{prefix}-{digest}.pt"
    _atomic_write(root / name, payload)
    return {"file": name, "sha256": digest, "bytes": len(payload)}


def save_checkpoint(
    directory: str | Path, model: RecurrentPolicy, *, training: dict | None = None
) -> dict:
    """Write an immutable weight blob before atomically publishing the manifest.

    A concurrent reader always sees either the complete previous artifact or
    the complete new one.  Training metadata is JSON, never arbitrary objects.
    Optimizer and RNG resume state belong to the training unit and are not
    written here.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    spec = schema_spec()
    if model.config.feature_dim != len(spec["features"]) or model.config.action_dim != len(
        spec["actions"]
    ):
        raise CheckpointError("model dimensions do not match current observation schema")
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise CheckpointError("refusing to save non-finite policy weights")
    # Validate JSON before creating any referenced artifacts.
    metadata = json.loads(json.dumps(training or {}, allow_nan=False))
    manifest = {
        "format": FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "config": model.config.to_dict(),
        "schema": spec,
        "schema_fingerprint": schema_fingerprint(),
        "weights": _write_blob(root, "weights", _serialize(state)),
        "training": metadata,
        "automatic_promotion": False,
    }
    _atomic_write(
        root / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    return manifest


@dataclass
class LoadedCheckpoint:
    model: RecurrentPolicy
    manifest: dict
    policy_sha: str
    directory: Path


def _read_bounded(path: Path, limit: int, what: str) -> bytes:
    # Read at most limit + 1 bytes so a file growing after stat cannot exceed the bound.
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise CheckpointError(f"checkpoint {what} exceeds size limit")
    return payload


def _reject_constant(name: str) -> None:
    raise CheckpointError(f"checkpoint manifest contains non-finite number {name}")


def _read_manifest(manifest_path: Path) -> dict:
    payload = _read_bounded(manifest_path, MAX_MANIFEST_BYTES, "manifest")
    try:
        manifest = json.loads(payload, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("checkpoint manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise CheckpointError("checkpoint manifest must be a JSON object")
    return manifest


def _policy_config(raw: Any) -> PolicyConfig:
    if not isinstance(raw, dict) or set(raw) != set(CONFIG_FIELDS):
        raise CheckpointError("checkpoint config must contain exactly the policy dimensions")
    if not all(type(raw[name]) is int for name in CONFIG_FIELDS):
        raise CheckpointError("checkpoint dimensions must be integers")
    return PolicyConfig(**{name: raw[name] for name in CONFIG_FIELDS})


def _read_blob(root: Path, descriptor: Any, expected_sha256: str | None = None) -> bytes:
    if not isinstance(descriptor, dict):
        raise CheckpointError("invalid checkpoint artifact descriptor")
    name = descriptor.get("file", "")
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise CheckpointError("invalid checkpoint artifact path")
    declared = descriptor.get("sha256")
    size = descriptor.get("bytes")
    if not isinstance(declared, str) or type(size) is not int or not 0 < size <= MAX_ARTIFACT_BYTES:
        raise CheckpointError("invalid checkpoint artifact descriptor")
    if expected_sha256 is not None and not hmac.compare_digest(declared, expected_sha256):
        raise CheckpointError("checkpoint weights are not the expected policy SHA256")
    payload = _read_bounded(root / name, MAX_ARTIFACT_BYTES, "artifact")
    if len(payload) != size:
        raise CheckpointError("checkpoint artifact length mismatch")
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, declared):
        raise CheckpointError("checkpoint artifact SHA256 mismatch")
    if expected_sha256 is not None and not hmac.compare_digest(actual, expected_sha256):
        raise CheckpointError("checkpoint weights are not the expected policy SHA256")
    return payload


def load_checkpoint(
    directory: str | Path, *, expected_sha256: str, device: str | torch.device = "cpu"
) -> LoadedCheckpoint:
    """Load a frozen policy only if its weight file hashes to ``expected_sha256``.

    ``directory`` is a checkpoint directory or its ``manifest.json``.  Every
    check runs before any tensor is materialized into the model.
    """
    expected_sha256 = require_sha256(expected_sha256)
    path = Path(directory)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = _read_manifest(manifest_path)
    if manifest.get("format") != FORMAT_VERSION:
        raise CheckpointError("unsupported checkpoint format")
    spec = schema_spec()
    if manifest.get("schema_fingerprint") != schema_fingerprint() or manifest.get("schema") != spec:
        raise CheckpointError("checkpoint observation/action schema mismatch")
    config = _policy_config(manifest.get("config"))
    if config.feature_dim != len(spec["features"]) or config.action_dim != len(spec["actions"]):
        raise CheckpointError("checkpoint dimensions do not match schema")
    payload = _read_blob(manifest_path.parent, manifest.get("weights"), expected_sha256)
    try:
        state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:  # any unpickling failure means an invalid artifact
        reason = type(exc).__name__
        raise CheckpointError(f"checkpoint weights could not be read safely: {reason}") from exc
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise CheckpointError("checkpoint weights must be a tensor state dictionary")
    if not all(value.is_floating_point() for value in state.values()):
        raise CheckpointError("checkpoint weights must be floating-point tensors")
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise CheckpointError("non-finite checkpoint weights")
    model = RecurrentPolicy(config)
    expected_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    actual_shapes = {key: tuple(value.shape) for key, value in state.items()}
    if actual_shapes != expected_shapes:
        raise CheckpointError("checkpoint tensors do not match the declared policy dimensions")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return LoadedCheckpoint(model, manifest, expected_sha256, manifest_path.parent)
