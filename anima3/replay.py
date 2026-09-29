"""Read versioned server-owned visual and combat replays (never execute content from them)."""

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
    if value.get("schema") != 1 or not isinstance(value.get("replays"), list):
        raise ValueError("unsupported replay index")
    return value["replays"]


def validate(data, metadata):
    if hashlib.sha256(data).hexdigest() != metadata["sha256"]:
        raise ValueError("replay checksum mismatch")
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) < 3 or rows[0].get("type") != "header" or rows[-1].get("type") != "end":
        raise ValueError("unfinished replay")
    header, end = rows[0], rows[-1]
    if header.get("schema") != 1 or header["id"] != metadata["id"] or end["id"] != header["id"]:
        raise ValueError("replay identity mismatch")
    if (not metadata.get("complete") or not end.get("complete")
            or metadata.get("dropped", 0) != 0 or end.get("dropped", 0) != 0):
        raise ValueError("incomplete recording cannot be learning evidence")
    for n, row in enumerate(rows):
        if row.get("seq") != n or type(row.get("t")) is not int or row["t"] < 0:
            raise ValueError("missing or invalid replay sequence")
        if n and row["t"] < rows[n - 1]["t"]:
            raise ValueError("replay clock moved backwards")
    for key in ("score", "winner", "aborted", "training"):
        if end.get(key) != metadata.get(key):
            raise ValueError("replay result disagrees with manifest")
    return rows


def download(base, metadata):
    if not re.fullmatch(r"[0-9a-f]{32}", str(metadata.get("id", ""))):
        raise ValueError("invalid replay id")
    data = fetch(base.rstrip("/") + "/duel/replays/" + metadata["id"] + ".jsonl")
    return data, validate(data, metadata)


def metrics(rows):
    out = {
        str(p["serial"]): {"damage": 0, "healing": 0, "casts": 0, "fizzles": 0, "potion_throws": 0}
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
        if player is not None and key:
            player[key] += max(0, row.get("amount", 0)) if key in ("damage", "healing") else 1
    return out


def burst_metrics(rows):
    """Observed damage clustering; does not infer which unnamed hit came from a spell."""
    result = {}
    for player in rows[0]['players']:
        serial = player['serial']
        hits = [r for r in rows if r['type'] == 'damage' and r.get('actor') == serial]
        peak = max((sum(max(0, h.get('amount', 0)) for h in hits if r['t'] <= h['t'] <= r['t'] + 1000)
                    for r in hits), default=0)
        potions = [r for r in rows if r['type'] == 'potion_state' and r.get('actor') == serial]
        result[str(serial)] = {
            'peakOneSecondDamage': peak,
            'firstSpells': [r.get('name') for r in rows if r['type'] == 'cast' and r.get('actor') == serial][:4],
            'potionCycles': [{'item': r['item'], 'primeMs': r['t'],
                              'throwMs': next((p['t'] for p in potions if p['item'] == r['item'] and p['phase'] == 'throw' and p['t'] >= r['t']), None),
                              'explodeMs': next((p['t'] for p in potions if p['item'] == r['item'] and p['phase'] == 'explode' and p['t'] >= r['t']), None)}
                             for r in potions if r['phase'] == 'prime'],
        }
    return result
