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
