"""Bounded offline Jev probe; never controls a client or trains on evaluation games."""

import argparse
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anima3.decision import _typesafe_key
from anima3.neural.schema import FEATURE_NAMES, schema_fingerprint

TACTICS = {
    "pressure": "Sustain offensive pressure while retaining a recovery reserve.",
    "interrupt": "Prioritize disrupting a publicly observed enemy cast when feasible.",
    "recover": "Prioritize legal recovery or resource conservation; healing is forbidden in showdown.",
    "disengage": "Create distance to reduce immediate incoming danger.",
    "maintain": "Keep current behavior because evidence for switching is insufficient.",
}


def snapshot(step):
    f = dict(zip(FEATURE_NAMES, step["features"], strict=True))
    return {
        "self_health_fraction": round(f["self_hp"], 3),
        "self_health_band": "critical"
        if f["self_hp"] < 0.35
        else "wounded"
        if f["self_hp"] < 0.7
        else "healthy",
        "self_mana_band": "low" if f["self_mana"] < 0.3 else "adequate",
        "opponent_health_fraction": round(f["opponent_hp"], 3)
        if f["opponent_hp_known"]
        else "unknown",
        "opponent_mana": "unknown",
        "distance_band": "melee"
        if f["distance"] <= 1 / 18
        else "close"
        if f["distance"] <= 3 / 18
        else "ranged",
        "self_poisoned": bool(f["self_poisoned"]),
        "self_paralyzed": bool(f["self_paralyzed"]),
        "showdown": bool(f["showdown"]),
        "opponent_visible": bool(f["opponent_visible"]),
        "line_of_sight": bool(f["line_of_sight"]),
        "recent_damage_received": round(f["recent_damage_received"] * 100, 1),
        "incoming_explosion_observed": bool(f["incoming_explosion"]),
        "held_explosion_potion": f["potion_fuse"] > 0,
        "observed_opponent_spell": next(
            (
                k.removeprefix("opponent_cast_")
                for k, v in f.items()
                if k.startswith("opponent_cast_") and v
            ),
            "unknown",
        ),
        "opponent_spell_age_seconds": round(f["opponent_spell_age"] * 10, 2),
    }


def build_scenes(demos, allowed, excluded):
    rng = random.Random(20261011)
    scenes = []
    reserves = []
    for game, d in enumerate(demos):
        assert d["match_id"] in allowed and d["match_id"] not in excluded
        assert d["eligible"] and d["schema_fingerprint"] == schema_fingerprint()
        steps = d["steps"]
        candidates = []
        for i, s in enumerate(steps):
            future = next((x for x in steps[i + 1 :] if x["at"] >= s["at"] + 5), None)
            if sum(s["mask"]) > 1 and future and future["at"] <= s["at"] + 6:
                candidates.append((i, future))
        # Target 8/7 per match; backfill short matches from unused eligible scenes.
        chosen = rng.sample(candidates, min(len(candidates), 8 if game < 20 else 7))
        for i, future in candidates:
            s = steps[i]
            state = snapshot(s)
            prefix = [x for x in steps[: i + 1] if x["at"] >= s["at"] - 8]
            history = [
                {"seconds_before": round(s["at"] - x["at"], 2), **snapshot(x)} for x in prefix[-6:]
            ]
            label = future["features"][0] <= s["features"][0] - 0.2
            baseline = state["incoming_explosion_observed"] or state["recent_damage_received"] >= 15
            row = {
                "id": len(scenes),
                "match_id": d["match_id"],
                "step": i,
                "style": d["style"],
                "state": {"current": state, "recent_observations": history},
                "label_health_drop_5s": label,
                "baseline_health_drop": baseline,
            }
            (scenes if (i, future) in chosen else reserves).append(row)
    scenes.extend(rng.sample(reserves, 300 - len(scenes)))
    for i, row in enumerate(scenes):
        row["id"] = i
    assert len(scenes) == 300
    return scenes


def questions(reverse=False):
    return {
        "tactic": {
            "type": "choice",
            "instructions": "Choose a tactical priority for the next few seconds of a pre-AOS UO duel. Use observed evidence only. Do not infer private mana. Healing is forbidden in showdown. Return maintain if no change is justified.",
            "criteria": dict(reversed(list(TACTICS.items()))) if reverse else TACTICS,
        },
        "health_drop": {
            "type": "noul",
            "instructions": "Will self health fraction be at least 0.20 lower five seconds from now than it is now, under continuation of existing behavior? Healing may offset damage. Predict using only the observations supplied.",
        },
        "burst_evidence": {
            "type": "noul",
            "instructions": "Do the recent public spell observations support a hypothesis of an enemy delayed Explosion followed by another attack spell? Unknown or old observations are insufficient evidence.",
        },
    }


def call(scene, variant, key):
    body = {
        "model": "jev-1.13.0",
        "state": scene["state"],
        "questions": questions(variant == "reverse"),
    }
    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    start = time.monotonic()
    row = {
        "id": scene["id"],
        "variant": variant,
        "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
    }
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.load(response)
        assert data["model"] == "jev-1.13.0"
        answers = data["answers"]
        p = answers["tactic"]["probabilities"]
        assert set(p) == set(TACTICS) and all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
        assert abs(sum(p.values()) - 1) < 0.02 and answers["tactic"]["choice"] in TACTICS
        for k in ("health_drop", "burst_evidence"):
            assert math.isfinite(answers[k]["noul"]) and 0 <= answers[k]["noul"] <= 1
        row.update(response=data)
    except (
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        KeyError,
        TypeError,
        AssertionError,
        OSError,
    ) as exc:
        # No exception body, request headers or credentials in logs.
        row["error"] = type(exc).__name__
        if hasattr(exc, "code"):
            row["http_status"] = exc.code
    row["latency_ms"] = (time.monotonic() - start) * 1000
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    p = args.out
    fit = json.loads(
        Path(".logs/reference-bc-final-audit/reference-bc/demonstration-training.json").read_text()
    )
    allowed = {Path(f).parent.name for f in fit["files"]}
    excluded = {Path(f).parent.name for f in fit["validation_files"]}
    for f in Path(".logs/reference-live-final-audit").glob("evaluation-*.json"):
        excluded.update(r["match_id"] for r in json.loads(f.read_text())["rows"])
    source = p / "source-demonstrations.json"
    scenes = build_scenes(json.loads(source.read_text()), allowed, excluded)
    (p / "scenes.json").write_text(json.dumps(scenes, indent=2))
    plan = {
        "seed": 20261011,
        "scenes": 300,
        "source_matches": 40,
        "max_requests": 420,
        "model": "jev-1.13.0",
        "questions": questions(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prediction_target": "Net health fraction drop >= .20 at first observation in [5,6] seconds",
        "baseline": "Observed incoming Explosion OR recent two-second damage >=15",
        "limitations": [
            "Correlated scenes; observational prediction, not causal tactic value.",
            "Teacher-side observations; not candidate on-policy states.",
            "No training or live control.",
        ],
        "gate": "Require positive held-out match-level predictive value before training; no live promotion from this probe.",
    }
    (p / "plan.json").write_text(json.dumps(plan, indent=2))
    if not args.run:
        print("Prepared 300 scenes; 420 request maximum")
        return
    key = _typesafe_key()
    if not key:
        raise RuntimeError("Missing configured TypeSafe credential")
    dest = p / "responses.jsonl"
    done = (
        {
            (r["id"], r["variant"])
            for line in dest.read_text().splitlines()
            if (r := json.loads(line))
        }
        if dest.exists()
        else set()
    )
    jobs = [
        (s, v)
        for s in scenes
        for v in (("normal", "reverse", "repeat") if s["id"] % 5 == 0 else ("normal",))
        if (s["id"], v) not in done
    ]
    # One request preflight before the bounded concurrent batch; halt on error.
    if jobs:
        s, v = jobs.pop(0)
        row = call(s, v, key)
        with dest.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if "error" in row:
            print(json.dumps(row))
            return
    with ThreadPoolExecutor(max_workers=4) as pool, dest.open("a") as f:
        for i, future in enumerate(as_completed([pool.submit(call, s, v, key) for s, v in jobs])):
            row = future.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 25 == 0:
                print("Completed requests", len(done) + i + 2, flush=True)


if __name__ == "__main__":
    main()
