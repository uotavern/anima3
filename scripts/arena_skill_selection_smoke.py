"""Verify the pre-AOS combat/support skill dialog and server-side allowlist."""

import argparse
import json
import re
from pathlib import Path

from anima3.body import BridgeBody
from anima3.contract import use
from anima3.gm import Gm

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=2599)
ap.add_argument("--credentials", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--inspect-only", action="store_true")
a = ap.parse_args()
c = json.loads(a.credentials.read_text())
b = BridgeBody.spawn(a.host, a.port, "arena_train_a", c["arena_train_a"])
checks = {}


def pump(n=12):
    for _ in range(n):
        b.pump(250)
    return b.observe_raw()


def dialog():
    for _ in range(6):
        o = pump(4)
        gs = [g for g in o["gumps"] if "PRE-AOS COMBAT / SUPPORT SKILLS" in str(g)]
        if gs:
            return gs[0]
    raise AssertionError("Skill dialog missing")


def check(k, v):
    checks[k] = bool(v)
    print(k, bool(v), flush=True)
    assert v, k


try:
    pump(28)
    gm = Gm(b)
    gm.journal_after("[Arena leave", pumps=4)
    gm.journal_after("[Arena enter", pumps=6)
    gm.journal_after("[Arena skills", pumps=6)
    o = pump()
    pack = next(
        i for i in o["items"] if i.get("layer") == 21 and i["container"] == o["player"]["serial"]
    )
    b.act(use(pack["serial"]))
    o = pump()
    ball = next(i for i in o["items"] if i["graphic"] == 0xE2D and i["hue"] == 1153)
    b.act(use(ball["serial"]))
    g = dialog()
    ids = set(map(int, re.findall(r"\{ checkbox [^}]* (\d+) \}", g["layout"])))
    allowed = {40, 42, 41, 31, 43, 27, 1, 5, 17, 25, 16, 26, 46, 30, 21, 47, 14, 4, 0, 23, 44}
    check("exact_21_combat_support_skills", ids == allowed)
    check("poisoning_arms_lore_and_support", {30, 4, 0, 23, 44} <= ids)
    check(
        "trade_and_post_aos_excluded",
        not ids.intersection({7, 18, 34, 35, 38, 49, 50, 51, 52, 53, 54, 55, 56, 57}),
    )
    if not a.inspect_only:
        b.act({"type": "SkillsRequest"})
        before = pump()["skills"]
        b.act(
            {
                "type": "GumpResponse",
                "serial": g["serial"],
                "gump_id": g["gump_id"],
                "button": 1,
                "switches": [25, 16, 26, 46, 43, 30, 49],
            }
        )
        o = pump()
        check(
            "forged_post_aos_rejected",
            any(i["serial"] == ball["serial"] for i in o["items"]) and o["skills"] == before,
        )
        g = dialog()
        selected = [25, 16, 26, 46, 43, 30, 4]
        b.act(
            {
                "type": "GumpResponse",
                "serial": g["serial"],
                "gump_id": g["gump_id"],
                "button": 1,
                "switches": selected,
            }
        )
        b.act({"type": "SkillsRequest"})
        o = pump()
        check(
            "combat_support_7gm_applied",
            sum(s["base"] for s in o["skills"]) == 700
            and sum(s["base"] == 100 for s in o["skills"]) == 7,
        )
        check("ball_consumed", not any(i["serial"] == ball["serial"] for i in o["items"]))
finally:
    b.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(checks, indent=2))
