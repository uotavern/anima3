"""Bounded participant handshakes and receipts, without a fight or training loop."""
from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from pathlib import Path

from . import replay
from .contract import say
from .duel_wait import RULES, BoundedBody, server_state

TRAINING_RULES = RULES + '-training'


def poll(body, *, timeout=30, stopped=lambda: False):
    """Request fresh server state; discard previously buffered journal first."""
    body = BoundedBody(body, timeout, stopped)
    body.observe()
    next_request = float('-inf')
    while True:
        body.check()
        if time.monotonic() >= next_request:
            body.act(say('[DuelState'))
            next_request = time.monotonic() + 2
        body.pump(100)
        state = server_state(body.observe())
        if state is not None:
            return state


def _pair(ids):
    if (len(ids) != 2 or any(type(s) is not int or s <= 0 for s in ids)
            or ids[0] == ids[1]):
        raise ValueError('two distinct positive participant serials required')


def wait_training_idle(bodies, *, rules=TRAINING_RULES, timeout=360, stopped=lambda: False):
    """Wait only for this pair's existing unrated fixture, refusing other duels."""
    if len(bodies) != 2 or not isinstance(rules, str) or not rules.endswith('-training'):
        raise ValueError('two bodies and explicit training rules required')
    guarded = [BoundedBody(b, timeout, stopped) for b in bodies]
    ids = [b.observe().player.serial for b in guarded]
    _pair(ids)
    states, fresh = [None, None], [float('-inf'), float('-inf')]
    while True:
        for i, body in enumerate(guarded):
            state = poll(body, timeout=min(timeout, 30), stopped=stopped)
            states[i], fresh[i] = state, time.monotonic()
            if state['phase'] != 'Idle' and (
                    state.get('rules') != rules or state.get('opponent') != ids[1-i]):
                raise RuntimeError('account is in an unrelated duel; refusing takeover')
            if state.get('challenge'):
                raise RuntimeError('pending challenge prevents idle preparation')
        if all(s['phase'] == 'Idle' for s in states) and all(
                time.monotonic() - t < 5 for t in fresh):
            return ids



def request_training_challenge(body, *, opponent, rules=TRAINING_RULES, rounds=1,
                               timeout=30, stopped=lambda: False):
    """Issue one explicit unrated challenge after a fresh idle preflight."""
    commands = {TRAINING_RULES: 'standard7-explosion-training',
                'noexplosion-training': 'training'}
    if (type(opponent) is not int or opponent <= 0 or type(rounds) is not int
            or not 1 <= rounds <= 15 or not isinstance(rules, str) or rules not in commands):
        raise ValueError('invalid training challenge')
    bounded = BoundedBody(body, timeout, stopped)
    state = poll(bounded, timeout=timeout, stopped=stopped)
    if state['phase'] != 'Idle' or state.get('challenge'):
        raise RuntimeError('challenger must be idle without a pending invitation')
    bounded.act(say(f'[Challenge 0x{opponent:X} {rounds} {commands[rules]}'))

def accept_invitation(body, *, opponent, rules=TRAINING_RULES, rounds=1,
                      timeout=30, stopped=lambda: False):
    """Accept exactly one fresh peer/rules/rounds-bound server challenge token."""
    if (type(opponent) is not int or opponent <= 0 or type(rounds) is not int
            or not 1 <= rounds <= 15 or not isinstance(rules, str) or not rules):
        raise ValueError('invalid expected invitation')
    bounded = BoundedBody(body, timeout, stopped)
    state = poll(bounded, timeout=timeout, stopped=stopped)
    invite = state.get('challenge')
    if (state['phase'] != 'Idle' or not invite or invite['opponent'] != opponent
            or invite['rules'] != rules or invite['rounds'] != rounds):
        raise RuntimeError('invitation does not match requested peer/rules/rounds')
    bounded.act(say('[DuelAccept ' + invite['id']))
    return invite['id']


def verify_receipt(data, metadata, ids, *, expected_match_id, rules=TRAINING_RULES):
    """Bind complete replay bytes to the expected training seats and rules."""
    _pair(ids)
    if not isinstance(rules, str) or not rules.endswith('-training'):
        raise ValueError('explicit training rules required')
    rows = replay.validate(data, metadata)
    players = metadata.get('players')
    if (not isinstance(players, list) or len(players) != 2 or not all(
            isinstance(p, dict) and type(p.get('serial')) is int and p['serial'] > 0
            for p in players)):
        raise ValueError('invalid manifest participants')
    if (metadata['id'] != expected_match_id
            or metadata.get('ranked') is not False or rows[0].get('ranked') is not False
            or rows[0].get('training') is not True
            or metadata['training'] is not True or metadata['aborted'] is not None
            or metadata.get('rules') != rules or rows[0].get('rules') != rules
            or [p['serial'] for p in rows[0]['players']] != list(ids)
            or [p.get('serial') for p in players] != list(ids)):
        raise ValueError('recording does not match requested training fixture')
    return rows


@contextmanager
def account_locks(directory, endpoint, accounts):
    """Nonblocking POSIX locks per account; all local runners must share directory.

    Locks coordinate this machine only. They do not prove remote account lease
    ownership; callers still perform fresh state checks before preparing.
    """
    import fcntl

    if len(accounts) != 2 or len(set(accounts)) != 2 or not all(
            isinstance(a, str) and a for a in accounts) or not isinstance(endpoint, str) or not endpoint:
        raise ValueError('two distinct account names and endpoint required')
    accounts = [a.casefold() for a in accounts]
    if len(set(accounts)) != 2:
        raise ValueError('account names must differ ignoring case')
    endpoint = endpoint.casefold()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    handles = []
    try:
        for account in sorted(accounts):
            key = hashlib.sha256((endpoint + '\0' + account).encode()).hexdigest()
            handle = (directory / (key + '.lock')).open('a')
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        for handle in reversed(handles):
            handle.close()
