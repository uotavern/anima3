"""Decision clients: pick one option id from a closed menu, with probabilities.

No backend generates text. `QwenLogprob` reads next-token log-probabilities of the
option letters (MLX, local). `JeffChoice` calls a TypeSafe-compatible `/v1/systemone`
endpoint (cloud Jev or the self-hosted `jeff`). `Scripted` returns the rule's first
choice. `gate()` turns a Decision into an admitted choice or a fallback.
"""

from __future__ import annotations

import os
import math
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

_SYSTEM = "You are a decision function for a character in Ultima Online. Reply with exactly one letter."
#: One MLX model serves every agent in the process; forward passes are serialized.
import threading

MLX_LOCK = threading.Lock()


@dataclass
class Decision:
    choice: str
    probs: dict[str, float]
    confidence: float          # top probability minus runner-up
    ms: float
    backend: str
    error: str | None = None
    meta: dict = field(default_factory=dict)


class DecisionClient(Protocol):
    name: str

    def choose(self, scene: str, question: str, options: dict[str, str]) -> Decision: ...


def _finish(probs: dict[str, float], t0: float, backend: str, **meta) -> Decision:
    ranked = sorted(probs.values(), reverse=True)
    top = max(probs, key=probs.get)
    conf = ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)
    return Decision(top, probs, conf, (time.perf_counter() - t0) * 1000, backend, meta=meta)


class Scripted:
    """The rule itself: always the first (rule-ordered) option, full confidence."""

    name = "scripted"

    def choose(self, scene: str, question: str, options: dict[str, str]) -> Decision:
        first = next(iter(options))
        return Decision(first, {k: (1.0 if k == first else 0.0) for k in options}, 1.0, 0.0, self.name)


class QwenLogprob:
    """Local MLX model in logprob mode. Loads lazily on first use (~1s)."""

    name = "qwen"
    DEFAULT_MODEL = os.environ.get("ANIMA3_MLX_MODEL", os.path.expanduser("~/dev/jev/models/Qwen3-4B-4bit"))

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or self.DEFAULT_MODEL
        self._model = self._tok = None

    def _load(self) -> None:
        if self._model is None:
            from mlx_lm import load
            self._model, self._tok = load(self.model_path)

    def warmup(self) -> float:
        """Load the model and run one throwaway decision; returns ms spent."""
        t0 = time.perf_counter()
        self.choose("warmup", "pick", {"a": "first", "b": "second"})
        return (time.perf_counter() - t0) * 1000

    def _letter_ids(self, letters: str) -> list[int]:
        ids = []
        for L in letters:
            t = self._tok.encode(L, add_special_tokens=False)
            if len(t) != 1:
                raise ValueError(f"option letter {L!r} is not a single token")
            ids.append(t[0])
        return ids

    def choose(self, scene: str, question: str, options: dict[str, str]) -> Decision:
        import mlx.core as mx
        t0 = time.perf_counter()
        self._load()
        keys = list(options)
        letters = string.ascii_uppercase[:len(keys)]
        menu = "\n".join(f"{L}. {options[k]}" for L, k in zip(letters, keys))
        msgs = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Situation:\n{scene}\n\nQuestion: {question}\nOptions:\n{menu}\n\nAnswer:"}]
        ids = self._tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        with MLX_LOCK:
            logits = self._model(mx.array([ids]))[:, -1, :]
            lp = (logits - mx.logsumexp(logits, keepdims=True)).squeeze(0)
            vals = mx.array([lp[i].item() for i in self._letter_ids(letters)])
            vals = mx.exp(vals - mx.logsumexp(vals)).tolist()
        return _finish({k: float(v) for k, v in zip(keys, vals)}, t0, self.name, prompt_tokens=len(ids))


#: Where a TypeSafe API key is kept on this machine (gitignored; never commit it).
_KEY_FILE = Path.home() / "dev" / "jev" / ".env"


def _typesafe_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    if _KEY_FILE.exists():
        for line in _KEY_FILE.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


class JeffChoice:
    """A TypeSafe System One `choice` question.

    `cloud=True` is TypeSafe's own Jev (measured here: 14/14 on the in-character
    probe and 3/3 on numeric state, where the open imitations scored 6/14 and 1/3);
    `cloud=False` is the self-hosted `jeff`/GLiFormer at TYPESAFE_BASE_URL."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None, cloud: bool = False) -> None:
        from typesafe_sdk import TypeSafeClient
        self.name = "jev" if cloud else "jeff"
        kw = {}
        key = _typesafe_key(api_key)
        if key:
            kw["api_key"] = key
        url = None if cloud else (base_url or os.environ.get("TYPESAFE_BASE_URL"))
        if url:
            kw["base_url"] = url
        if cloud and not key:
            raise ValueError("cloud Jev needs a TYPESAFE_API_KEY (env or ~/dev/jev/.env)")
        self._client = TypeSafeClient(**kw)

    def choose(self, scene: str, question: str, options: dict[str, str]) -> Decision:
        from typesafe_sdk import Choice
        t0 = time.perf_counter()
        a = self._client.system_one(state=scene, questions={"pick": Choice(instructions=question, criteria=options)}).answers["pick"]
        return _finish({k: float(v) for k, v in a.probabilities.items()}, t0, self.name, jev_confidence=a.confidence)


def build_client(kind: str) -> DecisionClient:
    kind = kind.lower()
    if kind in ("scripted", "rule", "none"):
        return Scripted()
    if kind in ("qwen", "mlx", "local"):
        return QwenLogprob()
    if kind in ("jev", "typesafe", "cloud"):
        return JeffChoice(cloud=True)
    if kind == "jeff":
        return JeffChoice()
    raise ValueError(f"unknown decision backend {kind!r}")


@dataclass(frozen=True)
class Admitted:
    choice: str
    used_model: bool
    reason: str


def gate(decision: Decision | None, options: dict[str, str], threshold: float) -> Admitted:
    """Admit the model's pick only if it is a real option and confident enough."""
    first = next(iter(options))
    if decision is None:
        return Admitted(first, False, "no decision")
    if decision.error:
        return Admitted(first, False, f"error: {decision.error}")
    if decision.choice not in options:
        return Admitted(first, False, "choice outside menu")
    if not math.isfinite(decision.confidence) or not 0 <= decision.confidence <= 1:
        return Admitted(first, False, "invalid confidence")
    if decision.confidence < threshold:
        return Admitted(first, False, f"confidence {decision.confidence:.2f} < {threshold:.2f}")
    return Admitted(decision.choice, True, "admitted")
