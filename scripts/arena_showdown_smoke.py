"""Local packet integration test; runtime-training must use showdown=30s, limit=65s."""

import json
import time
from pathlib import Path

from anima3.body import BridgeBody
from anima3.contract import say, target_object, use
from anima3.gm import Gm

credentials = json.loads(Path("../runtime-training/test-accounts.json").read_text())
bodies = []
checks = {}


def check(name, value):
    checks[name] = bool(value)
    print(name, bool(value), flush=True)
    assert value, name


def pump(b, seconds=0.5):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        b.pump(100)
    return b.observe_raw()


def state(b):
    b.act(say("[DuelState"))
    for _ in range(20):
        o = pump(b, 0.1)
        for j in o["new_journal"]:
            if j.get("text", "").startswith("[DuelState] "):
                return json.loads(j["text"][12:])
    raise AssertionError("No DuelState")


def wait_state(b, predicate, timeout=75):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        s = state(b)
        if predicate(s):
            return s
        pump(b, 0.2)
    raise AssertionError(("state timeout", s))


try:
    for user in ["arena_admin", "arena_train_a", "arena_train_b"]:
        body = BridgeBody.spawn("127.0.0.1", 2599, user, credentials[user])
        bodies.append(body)
        pump(body, 3)
    admin, a, b = bodies
    staff = Gm(admin, 100)
    staff.journal_after("[DuelReset 1", pumps=3)
    ids = []
    for body in [a, b]:
        for command in ["[Arena leave", "[Arena enter", "[Arena supplies"]:
            body.act(say(command))
            pump(body)
        o = body.observe_raw()
        ids.append(o["player"]["serial"])
        pack = next(i for i in o["items"] if i.get("layer") == 21 and i["container"] == ids[-1])
        body.act(use(pack["serial"]))
        pump(body)
    items = a.observe_raw()["items"]
    heal = next(i for i in items if i["graphic"] == 0xF0C)
    bandage = next(i for i in items if i["graphic"] == 0xE21)
    refresh = next(i for i in items if i["graphic"] == 0xF0B)
    staff.journal_after(
        f"[Duel start 0x{ids[0]:X} 0x{ids[1]:X} 3 standard7-explosion arena:1", pumps=3
    )
    s = wait_state(a, lambda s: s.get("phase") == "Fighting")
    check("normal_phase_first", s["showdown"] is False and s["showdownRemaining"] > 0)
    staff.command_on("[Set Hits 30", ids[0])
    pump(a)
    a.act(use(heal["serial"]))
    o = pump(a)
    check("heal_potion_before_showdown", o["player"]["hits"] > 30)
    wait_state(a, lambda s: 0 < s.get("showdownRemaining", 999) <= 5)
    # Hold a pre-transition Heal cursor; it must not heal after the threshold.
    a.act({"type": "CastSpell", "spell": 4})
    o = pump(a, 2)
    check("pre_showdown_heal_cursor", o["pending_target"])
    wait_state(a, lambda s: s.get("showdown") is True)
    staff.command_on("[Set Hits 30", ids[0])
    pump(a)
    a.act(target_object(ids[0]))
    o = pump(a, 2)
    check("held_heal_blocked_at_resolution", o["player"]["hits"] == 30)

    # Potion and bandage should not be consumed; no cursor should open.
    def amount(o, item):
        return sum(i.get("amount", 1) for i in o["items"] if i["graphic"] == item["graphic"])

    before = a.observe_raw()
    heal = next(i for i in before["items"] if i["graphic"] == 0xF0C)
    a.act(use(heal["serial"]))
    o = pump(a)
    check(
        "heal_potion_blocked_unconsumed",
        o["player"]["hits"] == 30
        and amount(o, heal) == amount(before, heal)
        and any("SHOWDOWN" in j.get("text", "") for j in o["new_journal"]),
    )
    a.act(use(bandage["serial"]))
    o = pump(a)
    check(
        "bandage_blocked", not o["pending_target"] and amount(o, bandage) == amount(before, bandage)
    )
    for spell in [4, 29]:
        a.act({"type": "CastSpell", "spell": spell})
        o = pump(a, 2)
        check(
            f"healing_spell_{spell}_blocked", not o["pending_target"] and o["player"]["hits"] == 30
        )
    staff.command_on("[Set Stam 1", ids[0])
    a.act(use(refresh["serial"]))
    o = pump(a)
    check("refresh_still_allowed", o["player"]["stam"] == o["player"]["stam_max"])
    o = pump(a, 12)
    check("natural_hp_regen_blocked", o["player"]["hits"] == 30)
    s = wait_state(a, lambda s: s.get("round") == 2 and s.get("phase") == "Fighting")
    check("next_round_resets_showdown", s["showdown"] is False and s["showdownRemaining"] > 0)
    staff.command_on("[Set Hits 30", ids[0])
    a.act(use(heal["serial"]))
    o = pump(a)
    check("next_round_healing_restored", o["player"]["hits"] > 30)
finally:
    if bodies:
        Gm(bodies[0]).journal_after("[DuelReset 1", pumps=2)
    for body in bodies:
        body.close()
    Path(".logs/showdown-smoke.json").write_text(json.dumps(checks, indent=2))
