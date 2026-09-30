"""Hash checked, schema bound policy artifacts; never load arbitrary pickle code."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .model import PolicyConfig, RecurrentPolicy

FORMAT_VERSION = "anima3-recurrent-policy-v1"
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


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
    directory: str | Path,
    model: RecurrentPolicy,
    *,
    training: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Write immutable weight blobs before atomically publishing the manifest.

    A concurrent reader always sees either the complete previous artifact or
    the complete new one.  Training metadata is JSON, never arbitrary objects.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    spec = schema_spec()
    if model.config.feature_dim != len(spec["features"]) or model.config.action_dim != len(
        spec["actions"]
    ):
        raise ValueError("model dimensions do not match current observation schema")
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("refusing to save non-finite policy weights")
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
    if optimizer is not None:
        manifest["optimizer"] = _write_blob(root, "optimizer", _serialize(optimizer.state_dict()))
        rng = {"cpu": torch.get_rng_state()}
        if model.device.type == "mps":
            rng["mps"] = torch.mps.get_rng_state()
        elif model.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state_all()
        manifest["rng"] = _write_blob(root, "rng", _serialize(rng))
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


def _read_blob(root: Path, descriptor: dict) -> bytes:
    name = descriptor.get("file", "")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("invalid checkpoint artifact path")
    path = root / name
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("checkpoint artifact exceeds size limit")
    payload = path.read_bytes()
    if len(payload) != descriptor.get("bytes"):
        raise ValueError("checkpoint artifact length mismatch")
    if hashlib.sha256(payload).hexdigest() != descriptor.get("sha256"):
        raise ValueError("checkpoint artifact SHA256 mismatch")
    return payload


def load_checkpoint(directory: str | Path, device: str | torch.device = "cpu") -> LoadedCheckpoint:
    path = Path(directory)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("checkpoint manifest exceeds size limit")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    if (
        manifest.get("schema_fingerprint") != schema_fingerprint()
        or manifest.get("schema") != schema_spec()
    ):
        raise ValueError("checkpoint observation/action schema mismatch")
    config = PolicyConfig(**manifest["config"])
    spec = schema_spec()
    if config.feature_dim != len(spec["features"]) or config.action_dim != len(spec["actions"]):
        raise ValueError("checkpoint dimensions do not match schema")
    payload = _read_blob(manifest_path.parent, manifest["weights"])
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("checkpoint weights must be a tensor state dictionary")
    if not all(bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("non-finite checkpoint weights")
    model = RecurrentPolicy(config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return LoadedCheckpoint(model, manifest, manifest["weights"]["sha256"], manifest_path.parent)


def restore_optimizer(loaded: LoadedCheckpoint, optimizer: torch.optim.Optimizer) -> bool:
    descriptor = loaded.manifest.get("optimizer")
    if descriptor is None:
        return False
    payload = _read_blob(loaded.directory, descriptor)
    state = torch.load(io.BytesIO(payload), map_location=loaded.model.device, weights_only=True)
    optimizer.load_state_dict(state)
    return True


def restore_training_rng(loaded: LoadedCheckpoint) -> bool:
    """Restore sampling RNG after model construction when resuming training."""
    descriptor = loaded.manifest.get("rng")
    if descriptor is None:
        return False
    payload = _read_blob(loaded.directory, descriptor)
    states = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    torch.set_rng_state(states["cpu"])
    if loaded.model.device.type == "mps" and "mps" in states:
        torch.mps.set_rng_state(states["mps"])
    elif loaded.model.device.type == "cuda" and "cuda" in states:
        torch.cuda.set_rng_state_all(states["cuda"])
    return True
