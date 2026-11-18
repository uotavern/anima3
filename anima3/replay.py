"""Read versioned server replay data without executing its contents.

Checksums detect corruption, not an authenticated server signature. The configured
server and stored receipt directory remain the trust boundary. Integrity alone
is not proof of a valid training episode or improved playing strength.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.request

MAX_BYTES = 34 * 1024 * 1024


def _fetch_once(url):
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    with urllib.request.urlopen(request, timeout=15) as response:
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("replay response too large")
        length = response.headers.get("Content-Length")
        if length is not None and (not length.isdecimal() or int(length) != len(data)):
            raise ValueError("replay response length mismatch")
        if response.headers.get("Content-Encoding") == "gzip":
            import io

            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                data = stream.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError("expanded replay too large")
        return data


def fetch(url):
    # Read-only retries cover a proxy reload or transient TLS failure; corrupt
    # data and permanent HTTP errors remain fatal evidence failures.
    for attempt in range(3):
        try:
            return _fetch_once(url)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def index(base):
    value = json.loads(fetch(base.rstrip("/") + "/duel/replays/"))
    if (not isinstance(value, dict) or type(value.get("schema")) is not int
            or value["schema"] != 1 or not isinstance(value.get("replays"), list)):
        raise ValueError("unsupported replay index")
    return value["replays"]


def validate(data, metadata):
    """Return complete schema-1 rows or fail closed with ValueError."""
    if not isinstance(metadata, dict):
        raise ValueError("invalid replay metadata")  # noqa: TRY004 -- invalid decoded evidence
    if not isinstance(metadata.get("id"), str) or not re.fullmatch(r"[0-9a-f]{32}", metadata["id"]):
        raise ValueError("invalid replay id")
    if len(data) > MAX_BYTES:
        raise ValueError("replay data too large")
    if hashlib.sha256(data).hexdigest() != metadata.get("sha256"):
        raise ValueError("replay checksum mismatch")
    try:
        rows = [json.loads(line) for line in data.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid replay JSON") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("replay rows must be JSON objects")
    if len(rows) < 3 or rows[0].get("type") != "header" or rows[-1].get("type") != "end":
        raise ValueError("unfinished replay")
    header, end = rows[0], rows[-1]
    if (type(header.get("schema")) is not int or header["schema"] != 1
            or header.get("id") != metadata["id"] or end.get("id") != header["id"]):
        raise ValueError("replay identity mismatch")
    players = header.get("players")
    if (not isinstance(players, list) or len(players) != 2
            or not all(isinstance(p, dict) and type(p.get("serial")) is int
                       and p["serial"] > 0 for p in players)
            or players[0]["serial"] == players[1]["serial"]):
        raise ValueError("invalid replay participants")
    if (metadata.get("complete") is not True or end.get("complete") is not True
            or any(type(part.get("dropped", 0)) is not int or part.get("dropped", 0) != 0
                   for part in (metadata, end))):
        raise ValueError("incomplete recording cannot be evidence")
    for n, row in enumerate(rows):
        if (type(row.get("seq")) is not int or row["seq"] != n
                or type(row.get("t")) is not int or row["t"] < 0):
            raise ValueError("missing or invalid replay sequence")
        if n and row["t"] < rows[n - 1]["t"]:
            raise ValueError("replay clock moved backwards")
        if 0 < n < len(rows) - 1 and row.get("type") in ("header", "end"):
            raise ValueError("unexpected replay boundary")
    # Every field consumed by the metrics helpers is checked here. Unknown
    # event types remain extensible, but cannot smuggle unsafe metric operands.
    for row in rows:
        kind = row.get("type")
        if not isinstance(kind, str) or not kind:
            raise ValueError("invalid replay event type")
        for key in ("actor", "target"):
            if key in row and (type(row[key]) is not int or row[key] < 0):
                raise ValueError("invalid replay event serial")
        if kind in ("damage", "heal") and (type(row.get("amount")) is not int or row["amount"] < 0):
            raise ValueError("invalid replay amount")
        if kind == "potion_state" and (
                type(row.get("item")) is not int or row["item"] <= 0
                or not isinstance(row.get("phase"), str) or not row["phase"]):
            raise ValueError("invalid replay potion state")
        if kind == "cast" and "name" in row and not isinstance(row["name"], str):
            raise ValueError("invalid replay spell name")
    score = metadata.get("score")
    if (not isinstance(score, list) or len(score) != 2
            or not all(type(value) is int and value >= 0 for value in score)
            or type(metadata.get("training")) is not bool):
        raise ValueError("invalid replay result")
    winner = metadata.get("winner")
    if winner is not None and (type(winner) is not int or winner not in {p["serial"] for p in players}):
        raise ValueError("invalid replay winner")
    for key in ("score", "winner", "aborted", "training"):
        if (key not in metadata or key not in end or type(end[key]) is not type(metadata[key])
                or end[key] != metadata[key]
                or (key == "score" and any(type(value) is not int for value in end[key]))):
            raise ValueError("replay result disagrees with manifest")
    for key in ("rules", "training"):
        if (key in header and key in metadata
                and (type(header[key]) is not type(metadata[key]) or header[key] != metadata[key])):
            raise ValueError("replay header disagrees with manifest")
    return rows


def download(base, metadata):
    if not isinstance(metadata, dict):
        raise ValueError("invalid replay metadata")  # noqa: TRY004 -- invalid decoded evidence
    if not isinstance(metadata.get("id"), str) or not re.fullmatch(r"[0-9a-f]{32}", metadata["id"]):
        raise ValueError("invalid replay id")
    data = fetch(base.rstrip("/") + "/duel/replays/" + metadata["id"] + ".jsonl")
    return data, validate(data, metadata)


def metrics(rows):
    out = {
        str(p["serial"]): {"damage": 0, "self_damage": 0, "healing": 0, "casts": 0, "fizzles": 0, "potion_throws": 0}
        for p in rows[0]["players"]
    }
    for row in rows:
        player = out.get(str(row.get("actor")))
        key = {
            "damage": "damage",
            "heal": "healing",
            "cast": "casts",
            "fizzle": "fizzles",
            "potion_throw": "potion_throws",
        }.get(row["type"])
        if player is not None and key == "damage":
            if row.get("target") == row.get("actor"):
                key = "self_damage"
            elif str(row.get("target")) not in out:
                continue
        if player is not None and key:
            player[key] += max(0, row.get("amount", 0)) if key in ("damage", "self_damage", "healing") else 1
    return out


def burst_metrics(rows):
    """Observed damage clustering; does not infer which unnamed hit came from a spell."""
    result = {}
    for player in rows[0]['players']:
        serial = player['serial']
        hits = [r for r in rows if r['type'] == 'damage' and r.get('actor') == serial and r.get('target') != serial and r.get('target') in {p['serial'] for p in rows[0]['players']}]
        peak = max((sum(max(0, h.get('amount', 0)) for h in hits if r['t'] <= h['t'] <= r['t'] + 1000)
                    for r in hits), default=0)
        potions = [r for r in rows if r['type'] == 'potion_state' and r.get('actor') == serial]
        result[str(serial)] = {
            'peakOneSecondDamage': peak,
            'firstSpells': [r.get('name') for r in rows if r['type'] == 'cast' and r.get('actor') == serial and r.get('target') != serial and r.get('target') in {p['serial'] for p in rows[0]['players']}][:4],
            'potionCycles': [{'item': r['item'], 'primeMs': r['t'],
                              'throwMs': next((p['t'] for p in potions if p['item'] == r['item'] and p['phase'] == 'throw' and p['t'] >= r['t']), None),
                              'explodeMs': next((p['t'] for p in potions if p['item'] == r['item'] and p['phase'] == 'explode' and p['t'] >= r['t']), None)}
                             for r in potions if r['phase'] == 'prime'],
        }
    return result
