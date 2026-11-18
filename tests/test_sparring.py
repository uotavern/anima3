import json
from types import SimpleNamespace

import pytest
from test_replay import encode, fixture

from anima3 import duel_wait, sparring
from anima3.contract import Journal, Observation


def observed(state, speaker=0):
    return Observation(new_journal=[Journal(speaker, 'System', '[DuelState] '+json.dumps(state), 6, 53, 0)])


class Body:
    def __init__(self, states, clock, serial=1):
        self.states = iter(states)
        self.clock, self.serial, self.actions = clock, serial, []

    def observe(self):
        result = next(self.states, Observation())
        result.player = SimpleNamespace(serial=self.serial)
        return result

    def pump(self, ms):
        self.clock[0] += ms / 1000

    def act(self, action):
        self.actions.append(action)


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(sparring.time, 'monotonic', lambda: now[0])
    return now


def invite(**patch):
    return {'id': 'a'*32, 'opponent': 2, 'rules': sparring.TRAINING_RULES, 'rounds': 1, **patch}


def test_accept_uses_fresh_server_token_and_exact_command(clock):
    body = Body([observed({'phase':'Idle','challenge':invite(id='b'*32)}),
                 observed({'phase':'Idle','challenge':invite()})], clock)
    assert sparring.accept_invitation(body, opponent=2) == 'a'*32
    assert body.actions == [{'type':'Say','text':'[DuelState'},
                            {'type':'Say','text':'[DuelAccept '+'a'*32}]


@pytest.mark.parametrize('patch', [{'opponent':3}, {'rules':'ranked'}, {'rounds':2}])
def test_mismatched_invitation_never_accepted(clock, patch):
    body = Body([Observation(), observed({'phase':'Idle','challenge':invite(**patch)})], clock)
    with pytest.raises(RuntimeError, match='requested'):
        sparring.accept_invitation(body, opponent=2)
    assert len(body.actions) == 1


def test_spoofed_state_and_silence_timeout(clock):
    body = Body([Observation(), observed({'phase':'Idle'}, speaker=99)], clock)
    with pytest.raises(TimeoutError):
        sparring.poll(body, timeout=.3)
    assert all(a['text'] == '[DuelState' for a in body.actions)


def test_stop_before_observe_or_action(clock):
    body = Body([], clock)
    with pytest.raises(RuntimeError, match='stopped'):
        sparring.poll(body, stopped=lambda: True)
    assert not body.actions


def test_stop_after_response_before_accept(clock):
    body = Body([Observation(), observed({'phase':'Idle','challenge':invite()})], clock)
    with pytest.raises(RuntimeError, match='stopped'):
        sparring.accept_invitation(body, opponent=2, stopped=lambda: clock[0] >= .1)
    assert len(body.actions) == 1


@pytest.mark.parametrize('state', [{'phase':'Fighting','id':'a'*32,'opponent':2,
                                   'rules':sparring.TRAINING_RULES,'round':1},
                                  {'phase':'Idle','challenge':invite()}])
def test_prepare_refuses_active_or_pending_before_mutation(clock, state):
    body = Body([Observation(), observed(state)], clock)
    with pytest.raises(RuntimeError, match='must be idle'):
        duel_wait.prepare(body)
    assert body.actions == [{'type':'Say','text':'[DuelState'}]


def test_prepare_calls_public_template_only_after_idle(clock, monkeypatch):
    body = Body([Observation(), observed({'phase':'Idle'})], clock)
    seen = []
    monkeypatch.setattr(duel_wait, '_prepare', lambda b, *a: seen.append(b) or {'serial':1})
    assert duel_wait.prepare(body) == {'serial':1}
    assert len(seen) == 1


def test_idle_wait_distinct_peers_and_unrelated_duel(clock):
    active = {'phase':'Fighting','id':'a'*32,'opponent':999,'rules':sparring.TRAINING_RULES,'round':1}
    bodies = [Body([Observation(), Observation(), observed(active)], clock, 1), Body([], clock, 2)]
    with pytest.raises(RuntimeError, match='unrelated'):
        sparring.wait_training_idle(bodies)


def test_idle_pair_success(clock):
    bodies = [Body([Observation(), Observation(), observed({'phase':'Idle'})], clock, s) for s in (1,2)]
    assert sparring.wait_training_idle(bodies) == [1,2]


def test_duplicate_serials_fail_before_commands(clock):
    bodies = [Body([], clock), Body([], clock)]
    with pytest.raises(ValueError, match='distinct'):
        sparring.wait_training_idle(bodies)
    assert all(not b.actions for b in bodies)


def test_receipt_binds_rules_seats_and_training():
    rows, meta = fixture()
    rows[0]['ranked'] = meta['ranked'] = False
    rows[0]['training'] = True
    rows[0]['rules'] = meta['rules'] = sparring.TRAINING_RULES
    meta['players'] = rows[0]['players']
    data, meta = encode(rows, meta)
    assert sparring.verify_receipt(data, meta, [1,2], expected_match_id='a'*32) == rows
    with pytest.raises(ValueError):
        sparring.verify_receipt(data, meta, [2,1], expected_match_id='a'*32)
    with pytest.raises(ValueError):
        sparring.verify_receipt(data, {**meta, 'players':[{'serial':2},{'serial':1}]}, [1,2], expected_match_id='a'*32)
    with pytest.raises(ValueError):
        sparring.verify_receipt(data, meta, [1,2], expected_match_id='a'*32, rules='other-training')


def test_account_lock_overlapping_pair_and_release(tmp_path):
    with (sparring.account_locks(tmp_path, 'local:2593', ['a','b']),
          pytest.raises(BlockingIOError),
          sparring.account_locks(tmp_path, 'local:2593', ['b','c'])):
        pytest.fail('overlapping account lock acquired')
    with sparring.account_locks(tmp_path, 'local:2593', ['b','c']):
        pass
    assert all('a' != p.stem and len(p.stem)==64 for p in tmp_path.iterdir())


@pytest.mark.parametrize('timeout',[0,-1,float('nan'),float('inf'),361])
def test_invalid_deadlines_rejected(timeout):
    with pytest.raises(ValueError):
        duel_wait.BoundedBody(None, timeout, lambda:False)


def test_training_challenge_is_explicit_and_idle_gated(clock):
    body = Body([Observation(), observed({'phase':'Idle'})], clock)
    sparring.request_training_challenge(body, opponent=57, rules='noexplosion-training')
    assert body.actions[-1]['text'] == '[Challenge 0x39 1 training'
    with pytest.raises(ValueError):
        sparring.request_training_challenge(body, opponent=57, rules='ranked')
    assert len(body.actions) == 2


def test_training_challenge_refuses_pending_invite(clock):
    body = Body([Observation(), observed({'phase':'Idle','challenge':invite()})], clock)
    with pytest.raises(RuntimeError):
        sparring.request_training_challenge(body, opponent=2)
    assert len(body.actions) == 1


@pytest.mark.parametrize('changed', ['historical_id','ranked_meta','ranked_header','boolean_seat'])
def test_receipt_rejects_historical_or_ranked_fixture(changed):
    rows, meta = fixture()
    rows[0].update(rules=sparring.TRAINING_RULES, training=True, ranked=False)
    meta.update(rules=sparring.TRAINING_RULES, ranked=False, players=rows[0]['players'])
    expected = 'b'*32 if changed == 'historical_id' else 'a'*32
    if changed == 'ranked_meta':
        meta['ranked'] = True
    if changed == 'ranked_header':
        rows[0]['ranked'] = True
    if changed == 'boolean_seat':
        meta['players'] = [{'serial':True}, {'serial':2}]
    data, meta = encode(rows, meta)
    with pytest.raises(ValueError):
        sparring.verify_receipt(data, meta, [1,2], expected_match_id=expected)


def test_account_locks_are_case_insensitive(tmp_path):
    with (sparring.account_locks(tmp_path, 'LOCAL:2593', ['Alice','Bob']),
          pytest.raises(BlockingIOError),
          sparring.account_locks(tmp_path, 'local:2593', ['alice','Charlie'])):
        pytest.fail('case variant bypassed account lock')
    with pytest.raises(ValueError), sparring.account_locks(tmp_path, 'local', ['Alice','alice']):
        pytest.fail('duplicate case variant accepted')
