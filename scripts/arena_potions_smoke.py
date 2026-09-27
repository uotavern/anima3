"""Local-only integration check of duel preset invitations, gates and 5/7GM balls."""

import json
from pathlib import Path

from anima3.body import BridgeBody
from anima3.contract import target_object, use
from anima3.gm import Gm

c = json.loads(Path("../runtime-training/test-accounts.json").read_text())
bodies = []
checks = {}


def pump(b, n=4):
    for _ in range(n):
        b.pump(150)
    return b.observe_raw()


def gump(b, title):
    return next(g for g in pump(b)["gumps"] if any(title in e.get("s", "") for e in g["elements"]))


def reply(b, g, button=1, **kw):
    b.act(dict(type="GumpResponse", serial=g["serial"], gump_id=g["gump_id"], button=button, **kw))
    return pump(b)


def check(k, v):
    checks[k] = bool(v)
    print(k, bool(v), flush=True)
    assert v, k


def train(b, n):
    gm = Gm(b)
    gm.journal_after("[Arena enter", pumps=2)
    gm.journal_after(f"[Arena skills {n}", pumps=2)
    ball = next(i for i in pump(b)["items"] if i["graphic"] == 0xE2D and i["hue"] == 1153)
    b.act(use(ball["serial"]))
    reply(b, gump(b, f"{n}GM SKILL BALL"), switches=[16, 25, 26, 43, 46, 1, 40][:n])
    b.act({"type": "SkillsRequest"})
    check(
        f"{n}gm_{b.observe_raw()['player']['serial']}",
        sum(s["base"] for s in pump(b)["skills"]) == 100 * n,
    )


try:
    for user in ["arena_admin", "arena_train_a", "arena_train_b"]:
        b = BridgeBody.spawn("127.0.0.1", 2599, user, c[user])
        bodies.append(b)
        pump(b, 20)
    admin, a, b = bodies
    staff = Gm(admin)
    ga = Gm(a)
    gb = Gm(b)
    staff.journal_after("[DuelReset 1", pumps=3)
    for p in [a, b]:
        Gm(p).journal_after("[Arena leave", pumps=2)
        Gm(p).journal_after("[Arena enter", pumps=2)
        o = pump(p)
        pack = next(
            i
            for i in o["items"]
            if i.get("layer") == 21 and i["container"] == o["player"]["serial"]
        )
        p.act(use(pack["serial"]))
        pump(p, 8)
        Gm(p).journal_after("[Arena skills", pumps=2)
        train(p, 7)
        Gm(p).journal_after("[Arena supplies", pumps=3)
    ga.journal_after("[Arena duel", pumps=2)
    menu = gump(a, "DUEL MODES")
    check("explosion_option_visible", "Allow Explosion Potions" in str(menu))
    # Default preset allows ordinary potions, but no explosion potions.
    reply(a, menu, 3)
    a.act(target_object(pump(b)["player"]["serial"]))
    invite = gump(b, "DUEL INVITATION")
    check("default_no_explosion", "noexplosion" in str(invite) and "nopotions" not in str(invite))
    reply(b, invite)
    for _ in range(22):
        a.pump(250)
        b.pump(250)
    o = pump(a)
    pack = next(
        i for i in o["items"] if i.get("layer") == 21 and i["container"] == o["player"]["serial"]
    )
    a.act(use(pack["serial"]))
    o = pump(a, 8)
    refresh = next(i for i in o["items"] if i["graphic"] == 0xF0B)
    explosion = next(i for i in o["items"] if i["graphic"] == 0xF0D)
    staff.command_on("[Set Stam 1", o["player"]["serial"])
    a.act(use(refresh["serial"]))
    o = pump(a, 8)
    check("regular_potion_restores_stamina", o["player"]["stam"] == o["player"]["stam_max"])
    a.act(use(explosion["serial"]))
    o = pump(a, 8)
    check(
        "explosion_disabled",
        any("Explosion potions require" in j.get("text", "") for j in o["new_journal"])
        and not o["pending_target"],
    )
    staff.journal_after("[DuelReset 1", pumps=3)
    for p in [a, b]:
        Gm(p).journal_after("[Arena enter", pumps=2)
    ga.journal_after("[Arena duel", pumps=2)
    reply(a, gump(a, "DUEL MODES"), 3, switches=[300])
    a.act(target_object(pump(b)["player"]["serial"]))
    invite = gump(b, "DUEL INVITATION")
    check(
        "enabled_option_in_invitation",
        "explosion" in str(invite) and "noexplosion" not in str(invite),
    )
    reply(b, invite)
    for _ in range(22):
        a.pump(250)
        b.pump(250)
    a.act(use(explosion["serial"]))
    o = pump(a, 4)
    check("enabled_explosion_target", bool(o["pending_target"]))
    target = pump(b)["player"]
    before = target["hits"]
    a.act(target_object(target["serial"]))
    for _ in range(12):
        a.pump(250)
        b.pump(250)
    check("enabled_explosion_damages_opponent", pump(b)["player"]["hits"] < before)
    staff.journal_after("[DuelReset 1", pumps=3)
    for p in [a, b]:
        Gm(p).journal_after("[Arena enter", pumps=2)
    ga.journal_after("[Arena join mage practice", pumps=3)
    gb.journal_after("[Arena join mage practice", pumps=3)
    status = staff.journal_after("[Duel status", pumps=3)
    check("practice_accepts_7gm", any("7x" in x and "idle" not in x for x in status))
finally:
    if bodies:
        Gm(bodies[0]).journal_after("[DuelReset 1", pumps=2)
    for p in bodies:
        p.close()
    Path(".logs/potions-local.json").write_text(json.dumps(checks, indent=2))
