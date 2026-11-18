import hashlib
import json

import pytest

from anima3.replay import metrics, validate


def fixture():
    rows = [
        {
            "seq": 0,
            "t": 0,
            "type": "header",
            "schema": 1,
            "id": "a" * 32,
            "players": [{"serial": 1}, {"serial": 2}],
        },
        {"seq": 1, "t": 100, "type": "damage", "actor": 1, "target": 2, "amount": 15},
        {
            "seq": 2,
            "t": 200,
            "type": "end",
            "id": "a" * 32,
            "complete": True,
            "training": True,
            "winner": 1,
            "score": [1, 0],
            "aborted": None,
        },
    ]
    meta = {k: v for k, v in rows[-1].items() if k not in ("seq", "t", "type")}
    return rows, meta


def encode(rows, meta):
    data = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    meta = {**meta, "sha256": hashlib.sha256(data).hexdigest()}
    return data, meta


def test_verified_replay_and_metrics():
    rows, meta = fixture()
    assert validate(*encode(rows, meta)) == rows
    assert metrics(rows)["1"]["damage"] == 15


@pytest.mark.parametrize(
    "change", ["gap", "backwards", "truncated", "wrong_id", "wrong_result", "wrong_hash"]
)
def test_corrupt_or_incomplete_evidence_rejected(change):
    rows, meta = fixture()
    if change == "gap":
        rows[1]["seq"] = 3
    if change == "backwards":
        rows[2]["t"] = 50
    if change == "truncated":
        rows[-1]["complete"] = False
    if change == "wrong_id":
        rows[0]["id"] = "b" * 32
    if change == "wrong_result":
        rows[-1]["winner"] = 2
    data, meta = encode(rows, meta)
    if change == "wrong_hash":
        data += b" "
    with pytest.raises(ValueError):
        validate(data, meta)


def test_download_retries_transient_transport_but_not_bad_evidence(monkeypatch):
    import urllib.error

    from anima3 import replay
    attempts, waits = [], []

    def unstable(url):
        attempts.append(url)
        if len(attempts) == 1:
            raise urllib.error.URLError('temporary TLS failure')
        return b'{}'

    monkeypatch.setattr(replay, '_fetch_once', unstable)
    monkeypatch.setattr(replay.time, 'sleep', waits.append)
    assert replay.fetch('https://arena.example/duel/replays/') == b'{}'
    assert len(attempts) == 2 and waits == [1]

    def missing(url):
        raise urllib.error.HTTPError(url, 404, 'expired', {}, None)

    monkeypatch.setattr(replay, '_fetch_once', missing)
    with pytest.raises(urllib.error.HTTPError):
        replay.fetch('https://arena.example/duel/replays/expired.jsonl')
    assert waits == [1]


def test_self_inflicted_potion_damage_is_not_offensive_reward():
    from anima3.replay import burst_metrics
    rows, _ = fixture()
    rows.insert(2, {"type": "damage", "actor": 1, "target": 1, "amount": 39, "t": 110})
    rows.insert(3, {"type": "damage", "actor": 1, "target": 999, "amount": 99, "t": 120})
    assert metrics(rows)["1"]["damage"] == 15
    assert metrics(rows)["1"]["self_damage"] == 39
    assert burst_metrics(rows)["1"]["peakOneSecondDamage"] == 15


@pytest.mark.parametrize('change', [
    'row_array', 'bool_sequence', 'bool_clock', 'negative_clock', 'missing_end',
    'extra_header', 'string_complete', 'dropped_frame', 'duplicate_players',
    'nonparticipant_winner', 'missing_result', 'bool_score', 'end_bool_score',
    'end_bool_winner', 'header_training_mismatch', 'malformed_json',
])
def test_malformed_receipts_fail_closed(change):
    rows, meta = fixture()
    if change == 'row_array':
        rows[1] = []
    elif change == 'bool_sequence':
        rows[0]['seq'] = False
    elif change == 'bool_clock':
        rows[0]['t'] = False
    elif change == 'negative_clock':
        rows[0]['t'] = -1
    elif change == 'missing_end':
        rows[-1]['type'] = 'damage'
    elif change == 'extra_header':
        rows[1]['type'] = 'header'
    elif change == 'string_complete':
        rows[-1]['complete'] = meta['complete'] = 'true'
    elif change == 'dropped_frame':
        rows[-1]['dropped'] = meta['dropped'] = 1
    elif change == 'duplicate_players':
        rows[0]['players'][1]['serial'] = 1
    elif change == 'nonparticipant_winner':
        rows[-1]['winner'] = meta['winner'] = 999
    elif change == 'missing_result':
        del rows[-1]['winner']
        del meta['winner']
    elif change == 'bool_score':
        rows[-1]['score'] = meta['score'] = [True, False]
    elif change == 'end_bool_score':
        rows[-1]['score'] = [True, False]
    elif change == 'end_bool_winner':
        rows[-1]['winner'] = True
    elif change == 'header_training_mismatch':
        rows[0]['training'] = False
    data, meta = encode(rows, meta)
    if change == 'malformed_json':
        data = b'{broken json}\n'
        meta['sha256'] = hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError):
        validate(data, meta)


def test_metadata_shape_and_direct_validation_size_limit(monkeypatch):
    from anima3 import replay
    with pytest.raises(ValueError):
        replay.validate(b'', [])
    rows, meta = fixture()
    data, meta = encode(rows, meta)
    monkeypatch.setattr(replay, 'MAX_BYTES', len(data) - 1)
    with pytest.raises(ValueError, match='too large'):
        replay.validate(data, meta)


def test_download_builds_id_url_and_verifies_bytes(monkeypatch):
    from anima3 import replay
    rows, meta = fixture()
    data, meta = encode(rows, meta)
    seen = []
    monkeypatch.setattr(replay, 'fetch', lambda url: seen.append(url) or data)
    assert replay.download('https://arena.example/', meta) == (data, rows)
    assert seen == ['https://arena.example/duel/replays/' + 'a' * 32 + '.jsonl']
    with pytest.raises(ValueError, match='invalid replay id'):
        replay.download('https://arena.example', {**meta, 'id': '../private'})
    assert len(seen) == 1


@pytest.mark.parametrize('value', [[], {}, {'schema': True, 'replays': []},
                                   {'schema': 1, 'replays': None}])
def test_invalid_index_shape_is_rejected(monkeypatch, value):
    from anima3 import replay
    monkeypatch.setattr(replay, 'fetch', lambda url: json.dumps(value).encode())
    with pytest.raises(ValueError, match='unsupported replay index'):
        replay.index('https://arena.example')


def test_fetch_has_exact_bounded_transport_retries(monkeypatch):
    import urllib.error

    from anima3 import replay
    attempts, waits = [], []
    def unavailable(url):
        attempts.append(url)
        raise urllib.error.HTTPError(url, 503, 'unavailable', {}, None)
    monkeypatch.setattr(replay, '_fetch_once', unavailable)
    monkeypatch.setattr(replay.time, 'sleep', waits.append)
    with pytest.raises(urllib.error.HTTPError):
        replay.fetch('https://arena.example')
    assert len(attempts) == 3 and waits == [1, 2]


def test_bad_evidence_is_not_retried(monkeypatch):
    from anima3 import replay
    attempts, waits = [], []
    def invalid(url):
        attempts.append(url)
        raise ValueError('bad archive')
    monkeypatch.setattr(replay, '_fetch_once', invalid)
    monkeypatch.setattr(replay.time, 'sleep', waits.append)
    with pytest.raises(ValueError):
        replay.fetch('https://arena.example')
    assert len(attempts) == 1 and waits == []


def response(data, headers=None):
    import io
    result = io.BytesIO(data)
    result.headers = headers or {}
    return result


def test_wire_and_expanded_sizes_are_bounded(monkeypatch):
    import gzip

    from anima3 import replay
    monkeypatch.setattr(replay, 'MAX_BYTES', 100)
    monkeypatch.setattr(replay.urllib.request, 'urlopen',
                        lambda *a, **k: response(b'x' * 101))
    with pytest.raises(ValueError, match='response too large'):
        replay._fetch_once('https://arena.example')
    compressed = gzip.compress(b'x' * 500)
    monkeypatch.setattr(replay.urllib.request, 'urlopen',
                        lambda *a, **k: response(compressed, {'Content-Encoding': 'gzip'}))
    with pytest.raises(ValueError, match='expanded replay too large'):
        replay._fetch_once('https://arena.example')


def test_gzip_and_declared_length(monkeypatch):
    import gzip

    from anima3 import replay
    payload = gzip.compress(b'valid response')
    seen = []
    def open_gzip(request, timeout):
        seen.append((request.get_method(), timeout))
        return response(payload, {'Content-Encoding': 'gzip', 'Content-Length': str(len(payload))})
    monkeypatch.setattr(replay.urllib.request, 'urlopen', open_gzip)
    assert replay._fetch_once('https://arena.example') == b'valid response'
    assert seen == [('GET', 15)]
    monkeypatch.setattr(replay.urllib.request, 'urlopen',
                        lambda *a, **k: response(b'{}', {'Content-Length': '100'}))
    with pytest.raises(ValueError, match='length mismatch'):
        replay._fetch_once('https://arena.example')


@pytest.mark.parametrize('metadata', [[], {'id': int('1' * 32)}])
def test_download_rejects_invalid_metadata_before_network(monkeypatch, metadata):
    from anima3 import replay
    def unexpected(url):
        pytest.fail('invalid metadata must not access network')
    monkeypatch.setattr(replay, 'fetch', unexpected)
    with pytest.raises(ValueError):
        replay.download('https://arena.example', metadata)


def test_burst_metrics_observe_window_and_potion_cycle():
    from anima3.replay import burst_metrics
    rows, _ = fixture()
    rows[1:1] = [
        {'type': 'cast', 'actor': 1, 'target': 2, 'name': 'Fireball', 't': 0},
        {'type': 'damage', 'actor': 1, 'target': 2, 'amount': 20, 't': 1000},
        {'type': 'damage', 'actor': 1, 'target': 2, 'amount': 30, 't': 2001},
        {'type': 'potion_state', 'actor': 1, 'item': 3, 'phase': 'prime', 't': 10},
        {'type': 'potion_state', 'actor': 1, 'item': 3, 'phase': 'throw', 't': 20},
        {'type': 'potion_state', 'actor': 1, 'item': 3, 'phase': 'explode', 't': 30},
    ]
    result = burst_metrics(rows)['1']
    assert result['peakOneSecondDamage'] == 35
    assert result['firstSpells'] == ['Fireball']
    assert result['potionCycles'] == [
        {'item': 3, 'primeMs': 10, 'throwMs': 20, 'explodeMs': 30}]


@pytest.mark.parametrize('event', [
    {}, {'type': ''}, {'type': 'damage', 'amount': 'bad'},
    {'type': 'damage', 'amount': True}, {'type': 'damage', 'amount': float('nan')},
    {'type': 'heal', 'amount': -1}, {'type': 'damage', 'amount': 1, 'actor': True},
    {'type': 'cast', 'actor': 1, 'target': True},
    {'type': 'potion_state', 'actor': 1},
    {'type': 'potion_state', 'actor': 1, 'item': True, 'phase': 'prime'},
    {'type': 'potion_state', 'actor': 1, 'item': 3, 'phase': []},
    {'type': 'cast', 'name': {}},
])
def test_invalid_metric_operands_fail_before_metrics(event):
    rows, meta = fixture()
    rows[1] = {'seq': 1, 't': 100, **event}
    with pytest.raises(ValueError):
        validate(*encode(rows, meta))


def test_unknown_event_is_safe_for_validated_metrics():
    from anima3.replay import burst_metrics
    rows, meta = fixture()
    rows[1] = {'seq': 1, 't': 100, 'type': 'future_event', 'payload': {}}
    verified = validate(*encode(rows, meta))
    assert metrics(verified)['1']['damage'] == 0
    assert burst_metrics(verified)['1']['peakOneSecondDamage'] == 0
