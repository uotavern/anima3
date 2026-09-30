"""Bounded asynchronous inference, isolated from the time-critical UO executor.

The parent never imports torch.  Recurrent state travels with each immutable
request: a late response can neither change a new round nor advance its memory.
The worker reads one verified checkpoint at startup and never hot-reloads it.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InferenceDecision:
    request_id: int
    epoch: str
    action: int
    log_prob: float
    value: float
    features: tuple[float, ...]
    mask: tuple[bool, ...]
    hidden_in: tuple[float, ...]
    hidden_out: tuple[float, ...]
    policy_sha: str
    infer_ms: float
    submitted_at: float
    received_at: float
    observed_at: float

    @property
    def inference_ms(self) -> float:
        return self.infer_ms

    @property
    def age_ms(self) -> float:
        return (self.received_at - self.submitted_at) * 1000


class InferenceClient:
    """Nonblocking, one-inflight NDJSON client with explicit round epochs.

    ``start`` returns immediately.  Poll ``ready`` while pumping the game bridge.
    ``take`` is the only operation that commits a recurrent state transition.
    An expired request still occupies the wire until its answer is consumed;
    this prevents unbounded queued work after a slow inference.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        deadline_ms: float = 250.0,
        deterministic: bool = False,
        device: str = "cpu",
        worker_command: list[str] | None = None,
    ):
        if not 1 <= deadline_ms <= 10_000:
            raise ValueError("inference deadline must be between 1 and 10000 ms")
        self.checkpoint = str(Path(checkpoint).resolve())
        self.deadline_ms = float(deadline_ms)
        self.deterministic = deterministic
        self.device = device
        self.worker_command = worker_command
        self.process: subprocess.Popen | None = None
        self._inbox: queue.Queue = queue.Queue(maxsize=4)
        self._events: list[dict] = []
        self._ready = False
        self.startup_error: str | None = None
        self.policy_sha: str | None = None
        self.manifest: dict = {}
        self.hidden_size = 0
        self.feature_size = 0
        self.action_size = 0
        self._epoch = ""
        self._hidden: tuple[float, ...] = ()
        self._pending: dict | None = None
        self._decision: InferenceDecision | None = None
        self._next_id = 1
        self._stderr_tail: list[str] = []
        self._closed = False

    def start(self) -> InferenceClient:
        if self.process is not None:
            return self
        command = self.worker_command or [
            sys.executable,
            "-u",
            "-m",
            "anima3.neural.inference",
            "--worker",
            "--checkpoint",
            self.checkpoint,
            "--device",
            self.device,
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        return self

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                # Responses are one compact hidden vector, never weights.
                if len(line) > 2_000_000:
                    self._inbox.put(
                        ({"type": "fatal", "error": "oversized worker response"}, time.monotonic())
                    )
                    return
                try:
                    received_at = time.monotonic()
                    self._inbox.put((json.loads(line), received_at))
                except (ValueError, TypeError):
                    self._inbox.put(
                        ({"type": "fatal", "error": "invalid worker JSON"}, time.monotonic())
                    )
                    return
        finally:
            self._inbox.put(({"type": "eof"}, time.monotonic()))

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr_tail.append(line.rstrip()[:1000])
            del self._stderr_tail[:-10]

    @property
    def ready(self) -> bool:
        self._poll(time.monotonic())
        return self._ready and self.startup_error is None and not self._closed

    @property
    def busy(self) -> bool:
        self._poll(time.monotonic())
        return self._pending is not None or self._decision is not None

    @property
    def hidden(self) -> tuple[float, ...]:
        return self._hidden

    def reset(self, epoch: str) -> None:
        self._epoch = str(epoch)
        self._hidden = (0.0,) * self.hidden_size
        self._decision = None
        if self._pending is not None:
            self._pending["invalid"] = "context_changed"
        self._events.append({"type": "context_reset", "epoch": self._epoch})

    def submit(
        self, frame: Any, epoch: str, now: float | None = None, *, observed_at: float | None = None
    ) -> int | None:
        now = time.monotonic() if now is None else now
        observed_at = now if observed_at is None else observed_at
        if not math.isfinite(observed_at) or observed_at > now:
            raise ValueError("observation time must be finite and no later than submission")
        self._poll(now)
        if not self._ready or self.startup_error or self._closed or self._pending or self._decision:
            return None
        if str(epoch) != self._epoch:
            self.reset(str(epoch))
        if (now - observed_at) * 1000 > self.deadline_ms:
            # No policy sample exists yet. Skip stale input without committing
            # recurrence or relabelling an old observation as freshly acquired.
            self._events.append(
                {
                    "type": "inference_skipped",
                    "reason": "observation_stale",
                    "epoch": self._epoch,
                    "observed_at": observed_at,
                    "checked_at": now,
                    "observation_age_ms": (now - observed_at) * 1000,
                }
            )
            return None
        features = tuple(float(x) for x in frame.features)
        mask = tuple(bool(x) for x in frame.mask)
        if (
            len(features) != self.feature_size
            or len(mask) != self.action_size
            or not all(math.isfinite(x) for x in features)
            or not any(mask)
        ):
            raise ValueError("invalid inference frame dimensions, values or legal-action mask")
        request_id = self._next_id
        self._next_id += 1
        payload = {
            "id": request_id,
            "epoch": self._epoch,
            "features": features,
            "mask": mask,
            "hidden": self._hidden,
            "deterministic": self.deterministic,
        }
        self._pending = {
            "id": request_id,
            "epoch": self._epoch,
            "features": features,
            "mask": mask,
            "hidden": self._hidden,
            "submitted_at": now,
            "observed_at": observed_at,
            "invalid": None,
        }
        try:
            assert self.process is not None and self.process.stdin is not None
            # A single observation is only a few KB; one-inflight bounds pipe use.
            self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            self.process.stdin.flush()
        except (OSError, BrokenPipeError) as exc:
            self._pending = None
            self.startup_error = f"worker write failed: {type(exc).__name__}"
            self._ready = False
            return None
        self._events.append(
            {
                "type": "inference_submitted",
                "id": request_id,
                "epoch": self._epoch,
                "submitted_at": now,
                "observed_at": observed_at,
            }
        )
        return request_id

    @staticmethod
    def _timing(pending: dict, now: float, received_at: float | None = None) -> dict:
        return {
            "observed_at": pending["observed_at"],
            "submitted_at": pending["submitted_at"],
            "received_at": received_at,
            "checked_at": now,
            "request_age_ms": (now - pending["submitted_at"]) * 1000,
            "observation_age_ms": (now - pending["observed_at"]) * 1000,
            "response_wait_ms": (now - received_at) * 1000 if received_at is not None else None,
        }

    def _poll(self, now: float) -> None:
        while True:
            try:
                row, received_at = self._inbox.get_nowait()
            except queue.Empty:
                break
            kind = row.get("type")
            if kind == "ready":
                self.policy_sha = str(row["policy_sha"])
                self.manifest = row.get("manifest", {})
                self.hidden_size = int(row["hidden_size"])
                self.feature_size = int(row["feature_size"])
                self.action_size = int(row["action_size"])
                self._hidden = (0.0,) * self.hidden_size
                self._ready = True
                self._events.append({"type": "inference_ready", "policy_sha": self.policy_sha})
                continue
            if kind in ("fatal", "eof"):
                if not self._closed:
                    self.startup_error = (
                        row.get("error") or self.startup_error or "inference worker exited"
                    )
                    self._events.append({"type": "inference_error", "reason": self.startup_error})
                self._ready = False
                self._pending = None
                continue
            pending, self._pending = self._pending, None
            if pending is None:
                self._events.append(
                    {
                        "type": "inference_rejected",
                        "reason": "unsolicited_response",
                        "epoch": row.get("epoch"),
                    }
                )
                continue
            reason = pending["invalid"]
            if (
                row.get("id") != pending["id"]
                or row.get("epoch") != pending["epoch"]
                or row.get("epoch") != self._epoch
            ):
                reason = "context_changed"
            elif row.get("policy_sha") != self.policy_sha:
                reason = "policy_changed"
            elif row.get("error"):
                reason = "worker_error"
            elif (now - pending["observed_at"]) * 1000 > self.deadline_ms:
                reason = "deadline"
            try:
                action = int(row["action"])
                hidden_out = tuple(float(x) for x in row["hidden_out"])
                values = [
                    float(row["log_prob"]),
                    float(row["value"]),
                    float(row["infer_ms"]),
                    *hidden_out,
                ]
                if not 0 <= action < len(pending["mask"]) or not pending["mask"][action]:
                    reason = "illegal_action"
                if len(hidden_out) != self.hidden_size or not all(math.isfinite(x) for x in values):
                    reason = "invalid_values"
            except (KeyError, TypeError, ValueError):
                reason = reason or "invalid_response"
            if reason:
                if pending["invalid"] != "deadline":
                    self._events.append(
                        {
                            "type": "inference_rejected",
                            "reason": reason,
                            "id": pending["id"],
                            "epoch": pending["epoch"],
                            **self._timing(pending, now, received_at),
                        }
                    )
                continue
            self._decision = InferenceDecision(
                pending["id"],
                pending["epoch"],
                action,
                float(row["log_prob"]),
                float(row["value"]),
                pending["features"],
                pending["mask"],
                pending["hidden"],
                hidden_out,
                self.policy_sha or "",
                float(row["infer_ms"]),
                pending["submitted_at"],
                received_at,
                pending["observed_at"],
            )
        pending = self._pending
        if (
            pending
            and not pending["invalid"]
            and (now - pending["observed_at"]) * 1000 > self.deadline_ms
        ):
            pending["invalid"] = "deadline"
            self._events.append(
                {
                    "type": "inference_rejected",
                    "reason": "deadline",
                    "id": pending["id"],
                    "epoch": pending["epoch"],
                    **self._timing(pending, now),
                }
            )

    def take(self, epoch: str, now: float | None = None) -> InferenceDecision | None:
        now = time.monotonic() if now is None else now
        self._poll(now)
        decision, self._decision = self._decision, None
        if decision is None:
            return None
        if decision.epoch != str(epoch) or decision.epoch != self._epoch:
            self._events.append(
                {
                    "type": "inference_rejected",
                    "reason": "context_changed",
                    "id": decision.request_id,
                    "epoch": decision.epoch,
                }
            )
            return None
        if (now - decision.observed_at) * 1000 > self.deadline_ms:
            self._events.append(
                {
                    "type": "inference_rejected",
                    "reason": "deadline",
                    "id": decision.request_id,
                    "epoch": decision.epoch,
                    **self._timing(
                        {
                            "observed_at": decision.observed_at,
                            "submitted_at": decision.submitted_at,
                        },
                        now,
                        decision.received_at,
                    ),
                }
            )
            return None
        self._hidden = decision.hidden_out
        return decision

    def drain_events(self) -> list[dict]:
        self._poll(time.monotonic())
        result, self._events = self._events, []
        return result

    def close(self) -> None:
        self._closed = True
        self._ready = False
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)


def worker(checkpoint: str, device: str = "cpu") -> int:
    """Only this subprocess imports PyTorch or reads model tensors."""
    import torch

    from .checkpoints import load_checkpoint

    torch.set_num_threads(1)
    try:
        loaded = load_checkpoint(checkpoint, device=device)
        model = loaded.model
        model.eval()
        config = model.config
        feature_size = int(config.feature_dim)
        action_size = int(config.action_dim)
        hidden_size = int(config.hidden_size)
        # Exclude framework kernel initialization from the first combat request.
        with torch.inference_mode():
            model.act(
                torch.zeros((1, feature_size), device=device),
                torch.ones((1, action_size), dtype=torch.bool, device=device),
                deterministic=True,
            )
        print(
            json.dumps(
                {
                    "type": "ready",
                    "policy_sha": loaded.policy_sha,
                    "feature_size": feature_size,
                    "action_size": action_size,
                    "hidden_size": hidden_size,
                    "manifest": loaded.manifest,
                }
            ),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 -- process boundary reports startup failure
        print(json.dumps({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 1
    for line in sys.stdin:
        request: dict = {}
        try:
            request = json.loads(line)
            features, mask, hidden = request["features"], request["mask"], request["hidden"]
            if (
                len(features) != feature_size
                or len(mask) != action_size
                or len(hidden) != hidden_size
            ):
                raise ValueError("request dimension mismatch")
            if not any(mask) or not all(math.isfinite(float(x)) for x in [*features, *hidden]):
                raise ValueError("invalid request mask or values")
            started = time.monotonic()
            with torch.inference_mode():
                action, log_prob, value, hidden_out = model.act(
                    torch.tensor([features], dtype=torch.float32, device=device),
                    torch.tensor([mask], dtype=torch.bool, device=device),
                    torch.tensor([hidden], dtype=torch.float32, device=device),
                    deterministic=bool(request.get("deterministic", False)),
                )
            response = {
                "type": "decision",
                "id": request["id"],
                "epoch": request["epoch"],
                "action": int(action.item()),
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
                "hidden_in": hidden,
                "hidden_out": hidden_out.squeeze(0).cpu().tolist(),
                "policy_sha": loaded.policy_sha,
                "infer_ms": (time.monotonic() - started) * 1000,
            }
            print(json.dumps(response, allow_nan=False), flush=True)
        except Exception as exc:  # noqa: BLE001 -- malformed request cannot kill worker silently
            print(
                json.dumps(
                    {
                        "type": "decision",
                        "id": request.get("id"),
                        "epoch": request.get("epoch"),
                        "policy_sha": loaded.policy_sha,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    raise SystemExit(worker(args.checkpoint, args.device))
