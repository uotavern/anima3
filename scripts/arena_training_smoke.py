"""Exercise training NPC, balls, validation and match guards through real UO packets.
Only use designated test accounts: this replaces their skills/stats.
"""

import argparse
import json
from pathlib import Path

from anima3.body import BridgeBody
from anima3.contract import use
from anima3.gm import Gm

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=2599)
ap.add_argument("--credentials", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument(
    "--supplies-only", action="store_true", help="Verify supplies without entering a public queue"
)
a = ap.parse_args()
c = json.loads(a.credentials.read_text())
bodies = []
checks = {}


def pump(b, n=4):
    for _ in range(n):
        b.pump(250 if a.host != "127.0.0.1" else 150)
    return b.observe_raw()


def gump(b, title):
    for _ in range(8):
        o = pump(b)
        found = [g for g in o["gumps"] if any(title in e.get("s", "") for e in g["elements"])]
        if found:
            return found[0]
    raise AssertionError("Gump not received: " + title)


def reply(b, g, button=1, **kw):
    b.act(dict(type="GumpResponse", serial=g["serial"], gump_id=g["gump_id"], button=button, **kw))
    return pump(b, 16 if a.host != "127.0.0.1" else 4)


def open_pack(b):
    o = pump(b)
    pack = next(
        i for i in o["items"] if i.get("layer") == 21 and i["container"] == o["player"]["serial"]
    )
    b.act(use(pack["serial"]))
    return pump(b, 8)


def open_ball(b, hue, title):
    o = open_pack(b)
    ball = next(i for i in o["items"] if i["graphic"] == 0xE2D and i["hue"] == hue)
    b.act(use(ball["serial"]))
    return ball, gump(b, title)


def check(name, value):
    checks[name] = bool(value)
    print(name, bool(value), flush=True)
    assert value, name


try:
    b = BridgeBody.spawn(a.host, a.port, "arena_train_a", c["arena_train_a"])
    bodies.append(b)
    pump(b, 40)
    gm = Gm(b)
    gm.journal_after("[Arena leave", pumps=3)
    gm.journal_after("[Arena enter", pumps=4)
    o = open_pack(b)
    npc = next(m for m in o["mobiles"] if m["pos"]["x"] == 5183 and m["pos"]["y"] == 332)
    sign = next(i for i in o["items"] if i["graphic"] == 0xBD2 and i["pos"]["x"] == 5184)
    b.act(use(sign["serial"]))
    check("guide_visible", bool(gump(b, "ARENA TRAINING GUIDE")))
    b.act(use(npc["serial"]))
    menu = gump(b, "ROWAN / FREE")
    reply(b, menu, 1)
    reply(b, gump(b, "ROWAN / FREE"), 2)
    check(
        "npc_gives_both_balls",
        len([i for i in pump(b)["items"] if i["graphic"] == 0xE2D and i["hue"] in [53, 1153]]) == 2,
    )
    gm.journal_after("[Arena skills", pumps=2)
    gm.journal_after("[Arena stats", pumps=2)
    check(
        "duplicate_supply_bounded",
        len([i for i in pump(b)["items"] if i["graphic"] == 0xE2D and i["hue"] in [53, 1153]]) == 2,
    )
    ball, g = open_ball(b, 1153, "6GM SKILL BALL")
    reply(b, g, switches=[16, 25, 26, 43, 46])
    check(
        "five_skills_rejected_ball_kept",
        any(i["serial"] == ball["serial"] for i in pump(b)["items"]),
    )
    g = gump(b, "6GM SKILL BALL")
    selected = [16, 25, 26, 43, 46, 1]
    reply(b, g, switches=selected)
    b.act({"type": "SkillsRequest"})
    o = pump(b)
    check("skill_ball_consumed", not any(i["serial"] == ball["serial"] for i in o["items"]))
    skills = o["skills"]
    check(
        "six_gm_applied",
        sum(s["base"] for s in skills) == 600 and sum(s["base"] == 100 for s in skills) == 6,
    )
    ball, g = open_ball(b, 53, "ARENA STATS")
    reply(b, g, entries=[[0, "100"], [1, "100"], [2, "100"]])
    check(
        "overcap_stats_rejected_ball_kept",
        any(i["serial"] == ball["serial"] for i in pump(b)["items"]),
    )
    reply(b, gump(b, "ARENA STATS"), entries=[[0, "100"], [1, "50"], [2, "75"]])
    o = pump(b)
    check(
        "stats_applied",
        [o["player"][k] for k in ["strength", "dexterity", "intelligence"]] == [100, 50, 75],
    )
    check("stat_ball_consumed", not any(i["serial"] == ball["serial"] for i in o["items"]))
    if a.supplies_only:
        raise SystemExit(0)
    gm.journal_after("[Arena stats", pumps=2)
    ball, g = open_ball(b, 53, "ARENA STATS")
    gm.journal_after("[Arena join mage practice", pumps=3)
    o = reply(b, g, entries=[[0, "90"], [1, "35"], [2, "100"]])
    check(
        "queued_stale_gump_blocked",
        [o["player"][k] for k in ["strength", "dexterity", "intelligence"]] == [100, 50, 75]
        and any(i["serial"] == ball["serial"] for i in o["items"]),
    )
    other = BridgeBody.spawn(a.host, a.port, "arena_train_b", c["arena_train_b"])
    bodies.append(other)
    pump(other, 40)
    Gm(other).journal_after("[Arena enter", pumps=2)
    Gm(other).journal_after("[Arena join mage practice", pumps=3)
    state = gm.journal_after("[ArenaState", pumps=3)
    check(
        "practice_match_started", any('"phase":"' in t and '"phase":"Idle"' not in t for t in state)
    )
    b.act({"type": "SkillsRequest"})
    o = pump(b)
    check(
        "practice_preserves_build",
        sum(s["base"] for s in o["skills"]) == 600
        and [o["player"][k] for k in ["strength", "dexterity", "intelligence"]] == [100, 50, 75],
    )
    b.act(use(ball["serial"]))
    o = pump(b)
    check(
        "match_ball_blocked", any("Customize while" in j.get("text", "") for j in o["new_journal"])
    )
finally:
    for body in bodies:
        body.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(checks, indent=2))
