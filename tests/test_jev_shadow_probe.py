import copy
import json

import pytest

from anima3.neural.schema import FEATURE_NAMES, schema_fingerprint
from scripts.jev_shadow_probe import build_scenes, questions
from scripts.report_jev_shadow import report


def demonstrations():
    return [
        {
            "match_id": str(i),
            "eligible": True,
            "style": "potions",
            "schema_fingerprint": schema_fingerprint(),
            "steps": [
                {"at": t, "features": [0.5] * len(FEATURE_NAMES), "mask": [True, True]}
                for t in range(20)
            ],
        }
        for i in range(40)
    ]


def test_scene_inputs_only_use_causal_observations():
    demos = demonstrations()
    allowed = {d["match_id"] for d in demos}
    scenes = build_scenes(demos, allowed, set())
    altered = copy.deepcopy(demos)
    # Last observation can supply a target, but is never a five-second-ahead input.
    for d in altered:
        d["steps"][-1]["features"][0] = 0
    after = build_scenes(altered, allowed, set())
    assert len(scenes) == 300
    assert [s["state"] for s in scenes] == [s["state"] for s in after]
    assert any(
        a["label_health_drop_5s"] != b["label_health_drop_5s"] for a, b in zip(scenes, after)
    )
    assert all(s["state"]["current"]["opponent_mana"] == "unknown" for s in scenes)


def test_evaluation_match_is_rejected():
    demos = demonstrations()
    with pytest.raises(AssertionError):
        build_scenes(demos, {d["match_id"] for d in demos}, {"0"})


def test_reversal_changes_only_option_order():
    a, b = questions(), questions(True)
    assert a == b
    assert list(a["tactic"]["criteria"]) == list(reversed(b["tactic"]["criteria"]))


def test_report_counts_always_negative_failure(tmp_path):
    scenes = [
        {
            "id": i,
            "match_id": str(i),
            "label_health_drop_5s": bool(i % 2),
            "baseline_health_drop": False,
        }
        for i in range(4)
    ]
    rows = [
        {
            "id": i,
            "variant": "normal",
            "latency_ms": 1,
            "response": {
                "answers": {"health_drop": {"noul": 0.1}, "tactic": {"choice": "maintain"}}
            },
        }
        for i in range(4)
    ]
    (tmp_path / "scenes.json").write_text(json.dumps(scenes))
    (tmp_path / "responses.jsonl").write_text("\n".join(map(json.dumps, rows)))
    r = report(tmp_path)
    assert r["jev"]["predicted_positive"] == 0
    assert r["jev"]["accuracy"] == 0.5
    assert r["jev"]["brier"] == pytest.approx(0.41)
