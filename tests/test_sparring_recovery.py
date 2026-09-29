from types import SimpleNamespace

from anima3 import sparring


def test_short_state_gap_preserves_round_agent(monkeypatch, tmp_path):
    now = [0.0]
    monkeypatch.setattr(sparring.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(sparring, 'server_state', lambda obs: obs)
    created = []

    class Body:
        def observe(self):
            if 2 <= now[0] < 7:
                return None
            if now[0] >= 9:
                return {'phase': 'Idle'}
            return {'phase': 'Fighting', 'id': 'a' * 32, 'rules': sparring.TRAINING_RULES,
                    'round': 1, 'opponent': 2}

        def act(self, action):
            pass

        def pump(self, ms):
            now[0] += ms / 1000

    class Agent:
        def __init__(self, body, *args, **kwargs):
            self.body = body
            self.memory = {}
            self.reports, self.proc_log = [], []
            created.append(self)

        def tick(self):
            self.body.pump(250)

    monkeypatch.setattr(sparring, 'Agent', Agent)
    result = sparring.fight([Body(), Body()], [{'playbook': 'standard'}] * 2,
                           tmp_path, lambda: False,
                           [SimpleNamespace(name='scripted')] * 2)
    assert result == 'a' * 32
    assert len(created) == 2


def test_login_retries_are_bounded_and_do_not_log_credentials(monkeypatch, tmp_path):
    import pytest
    args = SimpleNamespace(host='host', port=2593, bridge=None, data_dir=None)
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise sparring.BodyError('private-password')

    monkeypatch.setattr(sparring.BridgeBody, 'spawn', fail)
    monkeypatch.setattr(sparring.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        sparring.connect_training(args, 'training', 'private-password', tmp_path, lambda: False)
    assert len(calls) == 3
    assert 'private-password' not in (tmp_path / 'progress.jsonl').read_text()
