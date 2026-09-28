"""Participant policy plug-ins use DecisionClient: choose only from the supplied menu."""

from __future__ import annotations

import importlib
import re
import threading

from .decision import Decision, build_client

LABEL = re.compile(r"^[A-Za-z0-9_.+-]{1,48}$")


class BoundedClient:
    """One outstanding provider call across rounds; a stuck API cannot spawn more calls."""

    def __init__(self, client, max_calls=None):
        self.calls = 0
        self.errors = 0
        self.exhausted = False
        self.max_calls = max_calls
        self.client = client
        self.name = client.name
        self._busy = threading.Lock()

    def choose(self, scene, question, options):
        if not self._busy.acquire(blocking=False):
            return Decision("", {}, 0, 0, self.name, error="previous provider call still running")
        try:
            if self.max_calls is not None and self.calls >= self.max_calls:
                self.exhausted = True
                return Decision("", {}, 0, 0, self.name, error="provider call budget exhausted")
            self.calls += 1
            decision = self.client.choose(scene, question, options)
            if not isinstance(decision, Decision):
                raise TypeError("provider must return Decision")
            return decision
        except Exception:  # noqa: BLE001 -- provider errors must not stop combat or expose credentials
            self.errors += 1
            return Decision("", {}, 0, 0, self.name, error="provider request failed")
        finally:
            self._busy.release()


def load_policy(backend="scripted", factory=None, *, model=None, max_calls=None):
    if factory:
        if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", factory):
            raise ValueError("factory must be package.module:function")
        module, function = factory.split(":")
        client = getattr(importlib.import_module(module), function)()
    else:
        client = build_client(backend, model=model)
    if not callable(getattr(client, "choose", None)) or not isinstance(
        getattr(client, "name", None), str
    ):
        raise TypeError("policy must provide name and choose(scene, question, options)")
    return BoundedClient(client, max_calls if backend != "scripted" else None)
