"""Participant policy plug-ins use DecisionClient: choose only from the supplied menu."""
from __future__ import annotations

import importlib
import re
import threading
from .decision import Decision, build_client

LABEL = re.compile(r'^[A-Za-z0-9_.+-]{1,48}$')


class BoundedClient:
    """One outstanding provider call across rounds; a stuck API cannot spawn more calls."""
    def __init__(self, client):
        self.client = client
        self.name = client.name
        self._busy = threading.Lock()

    def choose(self, scene, question, options):
        if not self._busy.acquire(blocking=False):
            return Decision('', {}, 0, 0, self.name, error='previous provider call still running')
        try:
            decision=self.client.choose(scene, question, options)
            if not isinstance(decision,Decision):raise ValueError("provider must return Decision")
            return decision
        finally:
            self._busy.release()


def load_policy(backend='scripted', factory=None):
    if factory:
        if not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', factory):
            raise ValueError('factory must be package.module:function')
        module, function = factory.split(':')
        client = getattr(importlib.import_module(module), function)()
    else:
        client = build_client(backend)
    if not callable(getattr(client, 'choose', None)) or not isinstance(getattr(client, 'name', None), str):
        raise ValueError('policy must provide name and choose(scene, question, options)')
    return BoundedClient(client)
