"""Describe a completed shadow probe without claiming causal combat improvement."""

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def report(root):
    scenes = json.loads((root / "scenes.json").read_text())
    rows = [json.loads(x) for x in (root / "responses.jsonl").read_text().splitlines()]
    assert len({(r["id"], r["variant"]) for r in rows}) == len(rows)
    good = {(r["id"], r["variant"]): r for r in rows if "response" in r}
    normals = [
        (s, good[s["id"], "normal"]["response"]["answers"])
        for s in scenes
        if (s["id"], "normal") in good
    ]
    assert normals

    def metrics(predictions):
        ys = [int(s["label_health_drop_5s"]) for s, _ in normals]
        return {
            "brier": statistics.mean((p - y) ** 2 for p, y in zip(predictions, ys)),
            "accuracy": statistics.mean((p >= 0.5) == y for p, y in zip(predictions, ys)),
            "predicted_positive": sum(p >= 0.5 for p in predictions),
        }

    probabilities = [a["health_drop"]["noul"] for _, a in normals]
    baseline = [float(s["baseline_health_drop"]) for s, _ in normals]
    prevalence = [
        statistics.mean(
            t["label_health_drop_5s"] for t, _ in normals if t["match_id"] != s["match_id"]
        )
        for s, _ in normals
    ]
    grouped = defaultdict(list)
    for (s, _), p, b in zip(normals, probabilities, prevalence):
        y = s["label_health_drop_5s"]
        grouped[s["match_id"]].append((p - y) ** 2 - (b - y) ** 2)
    blocks = list(grouped.values())
    rng = random.Random(20261011)
    samples = []
    for _ in range(2000):
        draw = [v for block in rng.choices(blocks, k=len(blocks)) for v in block]
        samples.append(statistics.mean(draw))
    samples.sort()
    variants = {}
    for variant in ("reverse", "repeat"):
        pairs = [
            (
                good[s["id"], "normal"]["response"]["answers"]["tactic"],
                good[s["id"], variant]["response"]["answers"]["tactic"],
            )
            for s in scenes
            if (s["id"], "normal") in good and (s["id"], variant) in good
        ]
        variants[variant] = {
            "pairs": len(pairs),
            "choice_agreement": statistics.mean(a["choice"] == b["choice"] for a, b in pairs)
            if pairs
            else None,
        }
    lat = sorted(r["latency_ms"] for r in rows)
    return {
        "requests": len(rows),
        "successful_requests": len(good),
        "errors": Counter(r.get("error") for r in rows if "error" in r),
        "scenes_scored": len(normals),
        "source_matches": len(grouped),
        "positive_health_drop_labels": sum(s["label_health_drop_5s"] for s, _ in normals),
        "jev": metrics(probabilities),
        "rule_baseline": metrics(baseline),
        "always_no_drop": metrics([0] * len(normals)),
        "leave_one_match_out_prevalence": metrics(prevalence),
        "jev_minus_prevalence_brier_ci95_match_bootstrap": [samples[49], samples[1949]],
        "choice_consistency": variants,
        "tactic_counts": Counter(a["tactic"]["choice"] for _, a in normals),
        "latency_ms": {
            "p50": statistics.median(lat),
            "p95": lat[int(0.95 * (len(lat) - 1))],
            "over_1500ms": sum(x > 1500 for x in lat),
        },
        "input_tokens": sum(
            r["response"].get("usage", {}).get("input_tokens", 0) for r in rows if "response" in r
        ),
        "limitations": [
            "Observational net-health prediction; does not measure value of unexecuted tactics.",
            "Scenes come from previous training matches, not new held-out live performance.",
            "Five-second outcome requirement excludes late terminal states and can introduce survivorship bias.",
            "Baseline prevalence uses other matches only; bootstrap is descriptive with shared fitted prevalence.",
            "Burst-evidence answers lack independent labels; not scored as validated judgments.",
        ],
        "live_control": False,
        "weights_updated": False,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    result = report(args.root)
    (args.root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
