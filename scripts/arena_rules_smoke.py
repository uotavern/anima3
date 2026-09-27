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
        pump(b, 12)
    admin, a, b = bodies
    ga = Gm(a)
    gb = Gm(b)
    staff = Gm(admin)
    staff.journal_after("[DuelReset 1", pumps=3)
    check(
        "idle_before",
        any("0 of 14 arenas busy" in s for s in staff.journal_after("[Duel status", pumps=2)),
    )
    for p in [a, b]:
        Gm(p).journal_after("[Arena leave", pumps=2)
        train(p, 5)
        Gm(p).journal_after("[Arena supplies", pumps=2)
    ga.journal_after("[Arena duel", pumps=2)
    reply(a, gump(a, "DUEL MODES"), 1)
    a.act(target_object(pump(b)["player"]["serial"]))
    invite = gump(b, "DUEL INVITATION")
    check("mage5_invitation", "mageonly" in str(invite) and "nopotions" in str(invite))
    reply(b, invite)
    check(
        "mage5_started",
        any(
            "5x-fists" in s and "idle" not in s
            for s in staff.journal_after("[Duel status", pumps=2)
        ),
    )
    # Countdown lasts ten seconds; pump both clients while it advances.
    for _ in range(22):
        a.pump(250)
        b.pump(250)
    a.act({"type": "CastSpell", "spell": 38})
    o = pump(a)
    check("paralyze_blocked", any("not allowed" in j.get("text", "") for j in o["new_journal"]))
    pack = next(
        i for i in o["items"] if i.get("layer") == 21 and i["container"] == o["player"]["serial"]
    )
    a.act(use(pack["serial"]))
    o = pump(a)
    potion = next(i for i in o["items"] if i["graphic"] == 0xF0C)
    a.act(use(potion["serial"]))
    o = pump(a)
    check(
        "potions_blocked",
        any("Potions are not allowed" in j.get("text", "") for j in o["new_journal"]),
    )
    staff.journal_after("[DuelReset 1", pumps=3)
    for p in [a, b]:
        train(p, 7)
    rejected = ga.journal_after(
        f"[Challenge 0x{pump(b)['player']['serial']:X} 3 mage5 arena:1", pumps=3
    )
    check("mage5_rejects_7gm", any("Cannot start" in s for s in rejected))
    for preset in [
        "mage7",
        "standard7",
        "dexxer7",
        "open7",
        "6x-classic-magic-nopot ions".replace(" ", ""),
    ]:
        if preset.startswith("6x"):
            for p in [a, b]:
                train(p, 6)
        ga.journal_after(
            f"[Challenge 0x{pump(b)['player']['serial']:X} 3 {preset} arena:1", pumps=2
        )
        reply(b, gump(b, "DUEL INVITATION"))
        status = staff.journal_after("[Duel status", pumps=2)
        check(preset + "_started", any("Arena 1:" in s and "idle" not in s for s in status))
        staff.journal_after("[DuelReset 1", pumps=2)
    check("saved", any("World save done" in s for s in staff.journal_after("[Save", pumps=5)))
finally:
    if bodies:
        Gm(bodies[0]).journal_after("[DuelReset 1", pumps=2)
    for b in bodies:
        b.close()
    Path(".logs/rules-local.json").write_text(json.dumps(checks, indent=2))
