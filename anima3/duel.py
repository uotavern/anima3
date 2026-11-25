"""A refereed duel between two characters, under the old pit rules.

    python -m anima3.duel --a anima3d1:duelist_a:qwen --b anima3d2:duelist_b:scripted \\
        --rules 5x --weapon katana --armor leather --rounds 3

ServUO's own PVP Arena System is High-Seas-only (`Enabled => Core.HS`) and this shard
plays the T2A era, so the GM referees: it stages both fighters to the template, sets
them five tiles apart on the open ground, and calls the round when one falls
(Felucca: "anything goes"). 5x / 7x and the weapon rule were never enforced by the
engine in the old days either — players agreed them and the staging enforces them here.

Each side names its own decision backend, which makes a duel the cleanest benchmark
this project has: rule against model, model against model, jeff against Qwen.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field

from .agent import Agent
from .body import DEFAULT_HOST, DEFAULT_PORT, BridgeBody, ResilientBody
from .contract import Pos
from .decision import build_client
from .gm import Gm
from .persona import Persona

ARENA = (Pos(2602, 488, 20), Pos(2607, 488, 20))     # GM-refereed marks on the open ground (pre-arena)
#: The shard's own arena (Scripts/Services/Dueling): marks, lobby exits, spectator seat.
SERVER_MARKS = (Pos(2599, 491, 20), Pos(2605, 491, 20))
SERVER_LOBBY = (Pos(2599, 496, 20), Pos(2605, 496, 20))
SERVER_SEAT = Pos(2602, 495, 20)
#: The shard's four rings: {index: (mark A, mark B, exit A, exit B, spectator seat)}. A match
#: takes the lowest-numbered free ring, so an arm's `--arena` only decides where it waits and
#: where its spectator sits; the Start line reports the ring actually used.
def _ring(x: int, y: int, z: int = 15, gap: int = 6) -> tuple[Pos, Pos, Pos, Pos, Pos]:
    """A standard ring from its west mark: marks `gap` apart, exits and seat on the lobby row."""
    ex = y + 5
    return (Pos(x, y, z), Pos(x + gap, y, z), Pos(x, ex, z), Pos(x + gap, ex, z), Pos(x + gap // 2, ex + 1, z))


ARENAS: dict[int, tuple[Pos, Pos, Pos, Pos, Pos]] = {
    1: (Pos(2599, 491, 20), Pos(2605, 491, 20), Pos(2599, 496, 20), Pos(2605, 496, 20), Pos(2602, 495, 20)),
    2: _ring(5177, 320), 3: _ring(5257, 320), 4: _ring(5337, 320),
    5: _ring(5177, 378), 6: _ring(5257, 378), 7: _ring(5337, 378),
    8: _ring(5177, 436), 9: _ring(5257, 436), 10: _ring(5337, 436),
    11: _ring(5177, 494), 12: _ring(5257, 494),
    13: _ring(5309, 490, gap=14),   # LARGE 21x13 — kiting and meditation become viable
    14: _ring(5136, 327, gap=14),   # CORRIDOR 25x3 — no kiting at all
}
RULE_SET_SKILLS = ("Swords", "Tactics", "Anatomy", "Healing", "MagicResist", "Parry", "Hiding", "Wrestling")
#: Reagents each mage starts every match with (≈20 stones total; a 5-round match spends ~40).
REAGENT_TARGET = 120

TEMPLATES = {
    "5x": {"Swords": 100, "Tactics": 100, "Anatomy": 100, "Healing": 100, "MagicResist": 100},
    "7x": {"Swords": 100, "Tactics": 100, "Anatomy": 100, "Healing": 100, "MagicResist": 100, "Parry": 100, "Hiding": 100},
    "5x-mage": {"Magery": 100, "EvalInt": 100, "Meditation": 100, "MagicResist": 100, "Wrestling": 100},
    "7x-mage": {"Magery": 100, "EvalInt": 100, "Meditation": 100, "MagicResist": 100, "Wrestling": 100, "Anatomy": 100, "Healing": 100},
}
STATS = {"Str": 100, "Dex": 100, "Int": 25}
MAGE_STATS = {"Str": 90, "Dex": 35, "Int": 100}
#: Everything the shard's 5x/7x check sums (after the magic extension): zero what the template leaves out.
ALL_RULE_SKILLS = ("Swords", "Tactics", "Anatomy", "Healing", "MagicResist", "Parry", "Hiding", "Wrestling",
                   "Magery", "EvalInt", "Meditation", "Fencing", "Macing", "Archery")
WEAPONS = {"katana": ("Katana", 0x13FF, 1), "broadsword": ("Broadsword", 0x0F5E, 1), "vikingsword": ("VikingSword", 0x13B9, 1),
           "halberd": ("Halberd", 0x143E, 2), "bardiche": ("Bardiche", 0x0F4D, 2), "fists": (None, None, 0)}
ARMOR = {"leather": ["LeatherChest", "LeatherLegs", "LeatherArms", "LeatherGloves", "LeatherGorget", "LeatherCap"],
         "plate": ["PlateChest", "PlateLegs", "PlateArms", "PlateGloves", "PlateGorget", "PlateHelm"], "none": []}


@dataclass
class Fighter:
    account: str
    persona: Persona
    backend: str
    body: BridgeBody | None = None
    serial: int = 0
    agent: Agent | None = None
    wins: int = 0
    rounds: list[dict] = field(default_factory=list)


def parse(spec: str) -> Fighter:
    acct, pname, backend = (spec.split(":") + ["scripted"])[:3]
    return Fighter(acct, Persona.load(pname), backend)


def stage(gm: Gm, fx: Fighter, spot: Pos, rules: str, weapon: str, armor: str, rename: bool = True) -> dict:
    rep = {"rename": gm.command_on(f"[Set Name {fx.persona.name}", fx.serial)} if rename else {}
    template = dict(TEMPLATES[rules])
    wname, wgraphic, _ = WEAPONS[weapon]
    if wname is None:
        template.pop("Swords"); template["Wrestling"] = 100
    for sk, v in template.items():
        gm.command_on(f"[Set Skills.{sk}.Base {v}", fx.serial)
    for sk in ALL_RULE_SKILLS:      # the 5x/7x check sums every duelling skill: zero what the template leaves out
        if sk not in template:
            gm.command_on(f"[Set Skills.{sk}.Base 0", fx.serial)
    mage = rules.endswith("-mage")
    for st, v in (MAGE_STATS if mage else STATS).items():
        gm.command_on(f"[Set {st} {v}", fx.serial)
    if mage:
        from .contract import use
        from .magic import REAGENT_GRAPHICS, REAGENT_NAMES, SPELLBOOK_GRAPHIC
        obs = fx.body.observe()
        for _ in range(20):                         # a fresh login has not been told about its backpack yet
            if obs.backpack_serial() is not None:
                break
            fx.body.pump(250)
            obs = fx.body.observe()
        bp = obs.backpack_serial()
        if bp is not None and not obs.own_pack():   # the pack must be opened before it can be counted
            fx.body.act(use(bp))
            for _ in range(4):
                fx.body.pump(250)
                obs = fx.body.observe()
                if obs.own_pack():
                    break

        def count() -> dict[int, int]:
            have: dict[int, int] = {}
            for i in fx.body.observe().own_pack():
                have[i.graphic] = have.get(i.graphic, 0) + i.amount
            return have

        # Top every reagent back to the same level before each match: a learning experiment
        # needs the resource state held constant (a 63-round run drifted to 5 black pearls
        # and the loser simply switched spell, which looked like strategy). Verify by counting
        # again: under load the GM's target cursor can come too late, the command is silently
        # dropped, and a mage with no black pearl or mandrake stood frozen for whole matches.
        have = count()
        low = min(have.get(g, 0) for g in REAGENT_GRAPHICS)
        for attempt in range(4):
            if SPELLBOOK_GRAPHIC not in have:
                rep["spellbook"] = gm.command_on("[AddToPack Spellbook 18446744073709551615", fx.serial)   # every spell
            short = {g: REAGENT_TARGET - have.get(g, 0) for g in REAGENT_GRAPHICS if have.get(g, 0) < REAGENT_TARGET}
            if not short and SPELLBOOK_GRAPHIC in have:
                break
            for g, n in short.items():
                gm.command_on(f"[AddToPack {REAGENT_NAMES[g]} {n}", fx.serial)
            for _ in range(4):
                fx.body.pump(250)
            have = count()
        missing = [REAGENT_NAMES[g] for g in REAGENT_GRAPHICS if have.get(g, 0) < REAGENT_TARGET]
        if SPELLBOOK_GRAPHIC not in have:
            missing.append("Spellbook")
        rep["reagents"] = f"{low}->{min(have.get(g, 0) for g in REAGENT_GRAPHICS)}"
        if missing:
            rep["MISSING"] = missing
        gm.command_on("[Set Hits 90", fx.serial); gm.command_on("[Set Mana 100", fx.serial)
        # a mage duels unarmed: strip anything wielded
        for i in obs.items:
            if i.container == fx.serial and i.layer in (1, 2):
                gm.command_on("[Delete", i.serial)
        return rep
    obs = fx.body.observe()
    have = {i.graphic for i in obs.items if i.container in (fx.serial, obs.backpack_serial())}
    if wname and wgraphic not in have:
        rep["weapon"] = gm.command_on(f"[AddToPack {wname}", fx.serial)
    if wname is None:
        for i in obs.items:   # fists: no blade in hand
            if i.container == fx.serial and i.layer in (1, 2):
                gm.command_on("[Delete", i.serial)
    if rules == "7x" and weapon != "fists" and WEAPONS[weapon][2] == 1 and 0x1B73 not in have:
        rep["shield"] = gm.command_on("[AddToPack Buckler", fx.serial)
    from .affordances import GEAR_GRAPHICS
    worn_layers = {i.layer for i in obs.items if i.container == fx.serial}
    pack_layers = {GEAR_GRAPHICS[i.graphic][1] for i in obs.own_pack() if i.graphic in GEAR_GRAPHICS}
    armor_layer = {"LeatherChest": 0x0D, "LeatherLegs": 0x04, "LeatherArms": 0x13, "LeatherGloves": 0x07, "LeatherGorget": 0x0A, "LeatherCap": 0x06,
                   "PlateChest": 0x0D, "PlateLegs": 0x04, "PlateArms": 0x13, "PlateGloves": 0x07, "PlateGorget": 0x0A, "PlateHelm": 0x06}
    for piece in ARMOR[armor]:
        if armor_layer[piece] not in worn_layers | pack_layers:
            gm.command_on(f"[AddToPack {piece}", fx.serial)
    if not any(i.graphic == 0x0E21 for i in obs.own_pack()):
        gm.command_on("[AddToPack Bandage 100", fx.serial)
    gm.command_on("[Set CantWalk false", fx.serial)
    return rep


def reset(gm: Gm, fx: Fighter, spot: Pos, rules: str, weapon: str, armor: str) -> None:
    """Resurrect, restore, and re-kit: a fallen duelist's gear lies on the corpse (Felucca)."""
    for attempt in range(4):                       # wait until the body agrees the ghost is gone
        gm.command_on("[Resurrect", fx.serial)
        alive = False
        for _ in range(6):
            fx.body.pump(200)
            if not fx.body.observe().player.dead:
                alive = True
                break
        if alive:
            break
    gm.command_on(f"[Set X {spot.x} Y {spot.y} Z {spot.z}", fx.serial)
    gm.command_on("[Set Hits 100", fx.serial)
    gm.command_on("[Set Stam 100", fx.serial)
    gm.command_on("[Set Criminal false", fx.serial)
    gm.command_on("[Set Kills 0", fx.serial)
    for _ in range(3):
        fx.body.pump(200)
    stage(gm, fx, spot, rules, weapon, armor, rename=False)


def prep(a: Fighter, b: Fighter, clients: dict, pump_ms: int, max_ticks: int = 60) -> int:
    """Both fighters put their gear on before the round is called (no opponent set yet)."""
    for fx in (a, b):
        fx.agent = Agent(fx.body, fx.persona, clients["scripted"], decide_every=1, pump_ms=pump_ms, triage=None, reflect_every=0)
        fx.agent.memory["duel"] = True
    quiet = 0
    for t in range(max_ticks):
        for fx in (a, b):
            fx.agent.tick()
        dressed = all(not any(x.startswith("equip:") for x in fx.agent.reports[-1].options) for fx in (a, b))
        quiet = quiet + 1 if dressed else 0
        if t >= 8 and quiet >= 3:      # the re-issued kit takes a few ticks to show up in the pack
            return t + 1
    return max_ticks


def run_round(a: Fighter, b: Fighter, clients: dict, max_ticks: int, pump_ms: int, log_dir: str, n: int, speech) -> dict:
    for fx, other in ((a, b), (b, a)):
        fx.agent = Agent(fx.body, fx.persona, clients[fx.backend], decide_every=2, pump_ms=pump_ms,
                         log_path=f"{log_dir}/round{n}-{fx.persona.name.lower()}.jsonl", speech=speech,
                         triage=None, reflect_every=0)
        fx.agent.memory["duel_opponent"] = other.serial
        fx.agent.memory["duel"] = True
    stop = threading.Event()

    def loop(fx: Fighter) -> None:
        while not stop.is_set() and fx.agent.tick_no < max_ticks:
            r = fx.agent.tick()
            if r.dead:
                stop.set()
                break

    threads = [threading.Thread(target=loop, args=(fx,), daemon=True, name=fx.persona.name) for fx in (a, b)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    res = {"round": n, "seconds": round(time.time() - t0, 1)}
    for fx in (a, b):
        r = fx.agent.reports[-1]
        res[fx.persona.name] = {"dead": r.dead, "hp": round(r.hp_pct, 2), "ticks": r.tick,
                                "bandages": sum(1 for _, pid, v in fx.agent.proc_log if pid == "bandage" and v == "ok"),
                                "slipped": sum(1 for _, pid, v in fx.agent.proc_log if pid == "bandage" and v == "slipped"),
                                "attacks": sum(1 for x in fx.agent.reports if str(x.chosen).startswith("attack:")),
                                "model": (fx.agent.summary()["model_calls"], fx.agent.summary()["model_admitted"])}
    dead = [fx for fx in (a, b) if fx.agent.reports[-1].dead]
    if len(dead) == 1:
        winner = a if dead[0] is b else b
        winner.wins += 1
        res["winner"] = winner.persona.name
    else:
        res["winner"] = None
    return res


# --- Server-refereed mode: ServUO's duel system (Scripts/Services/Dueling) owns the rounds ---
import json
import re

DUEL_LINE = re.compile(r"^\[Duel\]\s*(.*)$")


class ServerDuel:
    """Drive a match through the shard's own duel commands and read its `[Duel]` journal
    lines. The brain only speaks: `[Challenge <name> <rounds> <rules>`, `[Accept`."""

    def __init__(self, a: Fighter, b: Fighter, rounds: int, rules_token: str, arena: int = 0, a_challenges: bool = True) -> None:
        self.a, self.b, self.rounds, self.rules, self.arena = a, b, rounds, rules_token, arena
        # Who speaks the challenge also decides the starting marks; alternating it cancels any side advantage.
        self.challenger, self.accepter = (a, b) if a_challenges else (b, a)
        self.pos: dict[int, int] = {a.serial: 0, b.serial: 0}   # lines consumed per fighter, by SEQUENCE
        self.lost: int = 0                                       # lines that scrolled out before we read them
        self.recent: list[str] = []                              # both fighters receive every line: skip the twin
        self.tries = 0
        self.error: str | None = None
        self.state = "idle"
        self.rounds_done: list[str] = []
        self.match: str | None = None
        self.log: list[str] = []

    def lines(self) -> list[str]:
        out = []
        for fx in (self.a, self.b):
            log, seq = fx.agent.journal_log, fx.agent.journal_seq
            first_held = seq - len(log)                 # sequence number of log[0]
            start = self.pos[fx.serial] - first_held
            if start < 0:                               # the trim ate lines we never read
                self.lost += -start
                start = 0
            fresh = log[start:]
            self.pos[fx.serial] = seq
            for t, ser, text in fresh:
                m = DUEL_LINE.match(text.strip())
                if m and m.group(1) not in self.recent:
                    self.recent.append(m.group(1))
                    del self.recent[:-8]
                    out.append(m.group(1))
        return out

    def step(self) -> None:
        mine = (self.a.persona.name.lower(), self.b.persona.name.lower())
        for line in self.lines():
            low = line.lower()
            # Four rings run at once and a fighter waiting in a lobby hears that ring's
            # announcements, which may belong to another arm: only our own names count.
            if any(w in low for w in ("defeats", "start:", "match:", "score:", "has challenged", "accepted", "detail:")) \
                    and not all(n in low for n in mine):
                continue
            self.log.append(line)
            if "has challenged" in low and self.state == "challenged":
                self.accepter.body.act({"type": "Say", "text": "[Accept"})
                self.state = "accepted"
            elif low.startswith("fight"):
                for fx, other in ((self.a, self.b), (self.b, self.a)):
                    won = lambda who: sum(1 for r in self.rounds_done if re.match(rf"^Round \d+: {re.escape(who)} defeats", r))
                    fx.agent.memory["duel_round"] = len(self.rounds_done) + 1
                    fx.agent.memory["duel_score"] = (won(fx.persona.name), won(other.persona.name))
                    fx.agent.memory["duel_opponent"] = other.serial
                self.state = "fighting"
            elif re.match(r"^round \d+: ", low):                       # "Round 1: Kael defeats Rook (...)"
                self.rounds_done.append(line)
                for fx in (self.a, self.b):
                    fx.agent.memory.pop("duel_opponent", None)
                    fx.agent.memory.pop("mana_recovery", None)
                    fx.agent.memory.pop("meditate_after", None)
                self.state = "between"
            elif low.startswith(("match:", "match aborted:")):
                self.match = line
                for fx in (self.a, self.b):
                    fx.agent.memory.pop("duel_opponent", None)
                self.state = "done"
            elif ("is busy" in low or low.startswith(("all arenas are busy", "no pending")) or "is already in a match" in low) and self.tries < 60:
                self.state = "retry"          # the ring has not finished clearing: challenge again shortly
                self.error = line
            elif low.startswith(("cannot start", "unknown rule", "rounds must", "staff only")):
                self.match = "ERROR: " + line
                self.state = "done"

    def challenge(self) -> None:
        self.tries += 1
        ring = f" arena:{self.arena}" if self.arena else ""
        self.challenger.body.act({"type": "Say", "text": f"[Challenge {self.accepter.persona.name} {self.rounds} {self.rules}{ring}"})
        self.state = "challenged"


def _median(xs: list[float]) -> float | None:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def validate_match(res: dict, reconnects: dict[str, int], mage: bool = False, missing: list[str] | None = None,
                   max_model_ms: float = 1000.0) -> list[str]:
    """Reasons a finished match cannot be counted; empty when it ran clean.

    Each rule is a failure that once passed silently into a tally: a frozen fighter (a round
    with no attacks from one side), a bridge that reconnected mid-match, a mage that never
    cast (no reagents on its menu), staging that left a fighter short, a match that did not end."""
    problems = [f"staging short of {', '.join(missing)}"] if missing else []
    if res.get("state") != "done":
        problems.append(f"match ended in state {res.get('state')}")
    elif not res.get("rounds"):
        problems.append("no rounds played")
    secs: dict[str, int] = {}
    for line in res.get("duel_lines", []):
        m = re.match(r"Round (\d+): .*?(\d+) seconds\)", line)
        if m:
            secs[m.group(1)] = int(m.group(2))
    for line in res.get("duel_lines", []):
        m = re.match(r"Round (\d+) detail: .*attacks (\d+)/(\d+)", line)
        if not m:
            continue
        rnd, x, y = m.group(1), int(m.group(2)), int(m.group(3))
        # A side that landed nothing in a long round was frozen; one that landed nothing in an
        # eight-second wipe simply lost fast, and voiding that would bias the tally toward it.
        if (x == 0 and y == 0) or ((x == 0 or y == 0) and secs.get(rnd, 0) > 30):
            problems.append(f"round {rnd}: attacks {x}/{y} in {secs.get(rnd, '?')} s")
    for name, k in reconnects.items():
        if k:
            problems.append(f"{name}'s bridge reconnected {k}x")
    for name, per in res.get("per", {}).items():
        # A decision head measured at ~250 ms is a different arm at 2 s: in experiment 3 Jev's API
        # slowed to 2-3 s for half an hour and both Jev arms lost 0-30 rounds in that window.
        if (per.get("model_ms") or 0) > max_model_ms:
            problems.append(f"{name}'s model took {per['model_ms']:.0f} ms per decision (limit {max_model_ms:.0f})")
    if mage:
        for name, per in res.get("per", {}).items():
            # Frozen means never even trying (no reagents on the menu). A mage that tried nine
            # times and was interrupted every time lost for real; voiding that biased the tally.
            if not sum(per.get("casts_ok", {}).values()) and not sum(per.get("cast_fail", {}).values()):
                problems.append(f"{name} never tried to cast")
    return problems


def run_server_match(a: Fighter, b: Fighter, clients: dict, rounds: int, rules_token: str, pump_ms: int, log_dir: str, max_ticks: int,
                     aims: tuple[str | None, str | None] = (None, None), mage: bool = False, tag: str = "server", arena: int = 0,
                     start_timeout_s: float = 120.0, sync_a: bool = False, a_challenges: bool = True,
                     keepalive=None, judge_a=None, playbook_a: str | None = None, wording: str = "vivid") -> dict:
    for fx, aim in zip((a, b), aims):
        sync = True if (sync_a and fx is a) else None      # None: the agent's default (async for a model)
        tactician = None
        if fx is a and judge_a is not None:
            from .judge import Asker
            from .tactics import Tactician
            tactician = Tactician(Asker(judge_a, log_path=f"{log_dir}/{tag}-{fx.persona.name.lower()}-judge.jsonl"),
                                  wording=wording)
        fx.agent = Agent(fx.body, fx.persona, clients[fx.backend], decide_every=2, pump_ms=pump_ms, sync=sync,
                         log_path=f"{log_dir}/{tag}-{fx.persona.name.lower()}.jsonl", triage=None, reflect_every=0,
                         tactician=tactician)
        fx.agent.memory["duel"] = True
        fx.agent.memory["mage"] = mage
        if fx is a and playbook_a:
            fx.agent.memory["playbook"] = playbook_a     # a fixed playbook arm: no judge, the rule runs this order throughout
        fx.agent.aim = aim
    ref = ServerDuel(a, b, rounds, rules_token, arena, a_challenges)
    stop = threading.Event()

    def loop(fx: Fighter) -> None:
        try:
            while not stop.is_set() and fx.agent.tick_no < max_ticks:
                fx.agent.tick()
        except Exception:  # noqa: BLE001 — a fighter thread must never die silently
            import traceback
            print(f"[{fx.persona.name}] fighter thread crashed:\n{traceback.format_exc()}", flush=True)
            stop.set()

    threads = [threading.Thread(target=loop, args=(fx,), daemon=True, name=fx.persona.name) for fx in (a, b)]
    for t in threads:
        t.start()
    time.sleep(1.0)
    ref.challenge()
    t0 = time.time()
    last_try = time.time()
    last_ping = time.time()
    while ref.state != "done" and any(t.is_alive() for t in threads) and time.time() - t0 < max_ticks * pump_ms / 1000:
        time.sleep(0.3)
        ref.step()
        if keepalive is not None and time.time() - last_ping > 10:
            last_ping = time.time()     # an unpumped staff bridge is dropped by the shard mid-match
            keepalive.pump(0)
        if ref.state == "retry" and time.time() - last_try > 8:
            last_try = time.time()
            ref.challenge()
        elif ref.state == "challenged" and time.time() - last_try > 20 and ref.tries < 4:
            last_try = time.time()      # the challenge was never seen at all (line lost): say it again
            ref.challenge()
        elif ref.state in ("challenged", "accepted", "retry") and not ref.rounds_done and time.time() - t0 > start_timeout_s:
            ref.state = "no-start"      # the bell never rang: give the match back instead of idling out the clock
            break
    stop.set()
    for t in threads:
        t.join(timeout=5)
    per = {}
    for fx in (a, b):
        import collections
        casts = collections.Counter((pid.split(":", 1)[1], v) for _, pid, v in fx.agent.proc_log if pid.startswith("cast:"))
        per[fx.persona.name] = {"bandages": sum(1 for _, pid, v in fx.agent.proc_log if pid == "bandage" and v == "ok"),
                                "bandage_verdicts": dict(collections.Counter(v for _, pid, v in fx.agent.proc_log if pid == "bandage")),
                                "casts_ok": dict(collections.Counter(k for (k, v) in casts.elements() if v == "ok")),
                                "cast_fail": dict(collections.Counter(v for (k, v) in casts.elements() if v != "ok")),
                                "meditations": sum(1 for _, pid, v in fx.agent.proc_log if pid == "meditate" and v == "ok"),
                                "model": (fx.agent.summary()["model_calls"], fx.agent.summary()["model_admitted"]),
                                "model_ms": _median([r.ms for r in fx.agent.reports if r.ms])}
        if fx.agent.tactician is not None:
            per[fx.persona.name]["tactics"] = fx.agent.tactician.summary()
    return {"state": ref.state, "rounds": ref.rounds_done, "match": ref.match or (f"(no match; last error: {ref.error})" if ref.error else None),
            "lost_lines": ref.lost, "seconds": round(time.time() - t0, 1),
            "duel_lines": ref.log, "per": per}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="anima3.duel")
    ap.add_argument("--a", required=True, help="account:persona:backend"); ap.add_argument("--b", required=True)
    ap.add_argument("--rules", choices=list(TEMPLATES), default="5x"); ap.add_argument("--weapon", choices=list(WEAPONS), default="katana")
    ap.add_argument("--matches", type=int, default=1, help="server mode: play this many matches back to back")
    ap.add_argument("--arena", type=int, default=0, help="server mode: ring to fight in (0 = let the shard choose a free one)")
    ap.add_argument("--learn", action="store_true", help="server mode: after each match the slow layer rewrites fighter A's aim from the playbook")
    ap.add_argument("--armor", choices=list(ARMOR), default="leather")
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--max-ticks", type=int, default=400)
    ap.add_argument("--host", default=DEFAULT_HOST); ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--gm-user", default="anima3"); ap.add_argument("--gm-pass", default=None,
                    help="defaults to the same string as --gm-user")
    ap.add_argument("--pump-ms", type=int, default=250); ap.add_argument("--speech", action="store_true")
    ap.add_argument("--log-dir", default=".logs/duel")
    ap.add_argument("--monitor-base", type=int, default=8811, help="anima-client spectator views: A on this port, B on the next (0 = off)")
    ap.add_argument("--open", action="store_true", help="open both spectator views in the browser")
    ap.add_argument("--referee", choices=["gm", "server"], default="gm", help="gm: this script referees; server: the shard's duel system does")
    ap.add_argument("--suffix", default="", help="appended to both fighters' names — every arm needs unique names because [Challenge resolves by name")
    ap.add_argument("--no-aim", action="store_true", help="ignore --aim-a: fighter A runs with no standing aim (the raw decision head)")
    ap.add_argument("--replays", type=int, default=2, help="server mode: replay a match that fails validation up to this many times")
    ap.add_argument("--alternate", action="store_true", help="server mode: B challenges in even matches, so neither fighter keeps the challenger's side")
    ap.add_argument("--sync-a", action="store_true", help="fighter A waits for its model at every decision instead of letting the rule act while it thinks")
    ap.add_argument("--rule-vs-rule", action="store_true", help="both sides use the rule backend (a symmetry baseline)")
    ap.add_argument("--aim-a", default=None, help="a standing aim placed in fighter A's scene (the slow layer's steering, held fixed)")
    ap.add_argument("--aim-b", default=None, help="same for fighter B")
    ap.add_argument("--tactics-a", choices=["none", "jev", "qwen", "random"], default="none",
                    help="server mode, mages: a judge picks fighter A's playbook at phase boundaries while the rule casts")
    ap.add_argument("--playbook-a", default=None, help="server mode, mages: fighter A runs this playbook throughout (no judge)")
    ap.add_argument("--tactics-wording", choices=["vivid", "neutral"], default="vivid",
                    help="how the playbooks are described to the judge: experiment 4's descriptions, or one plain shape in a shuffled order")
    args = ap.parse_args(argv)

    a, b = parse(args.a), parse(args.b)
    if args.suffix:
        for fx in (a, b):
            fx.persona.name = f"{fx.persona.name}{args.suffix}"
    if args.rule_vs_rule:
        a.backend = b.backend = "scripted"
    if args.no_aim:
        args.aim_a = None
    clients = {k: build_client(k) for k in {a.backend, b.backend}}
    for c in clients.values():
        if hasattr(c, "warmup"):
            c.warmup()
    judge_a = None
    if args.tactics_a != "none":
        from .judge import QwenJudge, build_judge
        judge_a = QwenJudge(clients["qwen"]) if args.tactics_a == "qwen" and "qwen" in clients else build_judge(args.tactics_a)
        if hasattr(judge_a, "warmup"):
            judge_a.warmup()
    if args.playbook_a:
        from .magic import PLAYBOOKS
        if args.playbook_a not in PLAYBOOKS:
            ap.error(f"--playbook-a must be one of {', '.join(PLAYBOOKS)}")
    speech = None
    if args.speech:
        from .speech import QwenSpeech
        qc = clients.get("qwen") or build_client("qwen")
        speech = QwenSpeech(qc, None)
    import os
    os.makedirs(args.log_dir, exist_ok=True)
    for i, fx in enumerate((a, b)):
        port = (args.monitor_base + i) if args.monitor_base else None
        fx.body = ResilientBody({"host": args.host, "port": args.port, "user": fx.account,
                                 "password": fx.account, "monitor_port": port})
        fx.serial = int(fx.body.ready["player"]["serial"])
        for _ in range(3):
            fx.body.pump(args.pump_ms)
        if port:
            time.sleep(0.5)
            print(f"watch {fx.persona.name}: {fx.body.monitor_url or f'http://127.0.0.1:{port}/'}", flush=True)
            if args.open:
                import subprocess
                subprocess.Popen(["open", fx.body.monitor_url or f"http://127.0.0.1:{port}/"])
    gm_body = ResilientBody({"host": args.host, "port": args.port, "user": args.gm_user,
                             "password": args.gm_pass or args.gm_user})
    gm = Gm(gm_body)
    try:
        if not any(line.startswith("You are at") for line in gm.journal_after("[Where", pumps=8)):
            print(f"preflight: {args.gm_user} gets no answer to [Where — not staff (any more?); refusing to run", flush=True)
            return 2
        gm.go(ARENA[0].x + 2, ARENA[0].y + 3, ARENA[0].z)
        for fx, spot in ((a, ARENA[0]), (b, ARENA[1])):
            rep = stage(gm, fx, spot, args.rules, args.weapon, args.armor)
            print(f"staged {fx.persona.name} ({fx.backend}, {args.rules}, {args.weapon}, {args.armor}): {rep}", flush=True)
        gm.journal_after("[Hide", pumps=2)
        print(f"\n== {a.persona.name} ({a.backend}) vs {b.persona.name} ({b.backend}) — {args.rules}, {args.weapon}, {args.armor}, best of {args.rounds} ==", flush=True)
        if args.referee == "server":
            mage = args.rules.endswith("-mage")
            token = f"{args.rules.split('-')[0]}-{'fists-magic' if mage else args.weapon}"
            ring = ARENAS.get(args.arena, ARENAS[1])
            lobby_a, lobby_b = ring[2], ring[3]
            aim_a = args.aim_a
            learner = None
            if args.learn:
                from .speech import QwenSpeech
                learner = QwenSpeech(clients.get("qwen") or build_client("qwen"), None)
            playbook = f"{args.log_dir}/playbook.md"
            tally = {a.persona.name: 0, b.persona.name: 0, "draw": 0}
            curve = []
            voided = 0
            for n in range(1, args.matches + 1):
                if n > 1:
                    time.sleep(4)        # let the ring clear before the next challenge
                for replay in range(args.replays + 1):
                    gm.command_on(f"[Set X {lobby_a.x} Y {lobby_a.y} Z {lobby_a.z}", a.serial)
                    gm.command_on(f"[Set X {lobby_b.x} Y {lobby_b.y} Z {lobby_b.z}", b.serial)
                    missing: list[str] = []
                    if mage:
                        for fx, spot in ((a, SERVER_MARKS[0]), (b, SERVER_MARKS[1])):
                            gm.command_on("[Set Mana 100", fx.serial)
                            rep = stage(gm, fx, spot, args.rules, args.weapon, args.armor, rename=False)
                            missing += [f"{fx.persona.name}:{m}" for m in rep.get("MISSING", [])]
                    before = {fx.persona.name: getattr(fx.body, "reconnects", 0) for fx in (a, b)}
                    for attempt in range(3):
                        res = run_server_match(a, b, clients, args.rounds, token, args.pump_ms, args.log_dir,
                                               max_ticks=args.max_ticks * args.rounds + 200, aims=(aim_a, args.aim_b), mage=mage,
                                               tag=f"m{n:03d}" + (f"r{replay}" if replay else ""), arena=args.arena, sync_a=args.sync_a,
                                               a_challenges=not (args.alternate and n % 2 == 0), keepalive=gm_body,
                                               judge_a=judge_a, playbook_a=args.playbook_a, wording=args.tactics_wording)
                        if res["state"] != "no-start":
                            break
                        print(f"   (match {n} never started, attempt {attempt + 1}; last line: {res['duel_lines'][-1:]}) — retrying", flush=True)
                        gm.journal_after(f"[DuelReset {args.arena}" if args.arena else "[Duel cancel", pumps=4)   # only our own ring: other arms are mid-match
                        time.sleep(6)
                    reconnects = {fx.persona.name: getattr(fx.body, "reconnects", 0) - before[fx.persona.name] for fx in (a, b)}
                    problems = validate_match(res, reconnects, mage=mage, missing=missing)
                    wins = {a.persona.name: 0, b.persona.name: 0, "draw": 0}
                    for line in res["rounds"]:
                        m = re.match(r"Round \d+: (\w+) defeats", line)
                        key = m.group(1) if m else "draw"
                        wins[key if key in wins else "draw"] += 1
                    with open(f"{args.log_dir}/matches.jsonl", "a") as fh:
                        fh.write(json.dumps({"match": n, "replay": replay, "valid": not problems, "problems": problems,
                                             "a_challenged": not (args.alternate and n % 2 == 0), "a": a.persona.name, "a_backend": a.backend,
                                             "b": b.persona.name, "b_backend": b.backend, "aim_a": aim_a, "tactics_a": args.tactics_a, "tactics_wording": args.tactics_wording,
                                             "playbook_a": args.playbook_a, "state": res["state"],
                                             "seconds": res["seconds"], "rounds": res["rounds"], "duel_lines": res["duel_lines"],
                                             "wins_a": wins[a.persona.name], "wins_b": wins[b.persona.name], "draws": wins["draw"],
                                             "per": res["per"]}) + "\n")
                    pa, pb = res["per"][a.persona.name], res["per"][b.persona.name]
                    print(f"{'match' if not problems else 'VOID '} {n:2d}/{args.matches}: {a.persona.name} {wins[a.persona.name]} - {wins[b.persona.name]} {b.persona.name} (draws {wins['draw']}) "
                          f"[{res['seconds']:.0f}s] | {a.persona.name} casts={pa['casts_ok']} fail={pa['cast_fail']} med={pa['meditations']} "
                          f"| {b.persona.name} casts={pb['casts_ok']} fail={pb['cast_fail']}", flush=True)
                    if pa.get("tactics"):
                        print(f"   tactics: {pa['tactics']}", flush=True)
                    if res.get("lost_lines"):
                        print(f"   (referee missed {res['lost_lines']} journal lines)", flush=True)
                    if not problems:
                        break
                    voided += 1
                    print(f"   void: {'; '.join(problems)}" + (" — replaying" if replay < args.replays else " — giving up on this match"), flush=True)
                    time.sleep(6)
                if problems:
                    continue                     # a match that never ran clean is left out of the tally and the learner
                for k in tally:
                    tally[k] += wins[k]
                curve.append((n, wins[a.persona.name], wins[b.persona.name], wins["draw"]))
                if learner is not None:
                    from .learn import next_aim
                    aim_a = next_aim(learner, a.persona, aim_a, res, wins, a.persona.name, b.persona.name, playbook, n)
                    print(f"   aim -> {aim_a}", flush=True)
            recon = sum(getattr(fx.body, "reconnects", 0) for fx in (a, b)) + getattr(gm_body, "reconnects", 0)
            from .stats import describe
            print(describe(tally[a.persona.name], tally[b.persona.name], a.persona.name) + (f"; {voided} void attempts replayed or dropped" if voided else ""))
            print(f"\nrounds: {a.persona.name} {tally[a.persona.name]} - {tally[b.persona.name]} {b.persona.name}, draws {tally['draw']}"
                  + (f" (bridges reconnected {recon}x)" if recon else ""))
            if len(curve) >= 4:
                q = max(1, len(curve) // 4)
                for i in range(0, len(curve), q):
                    chunk = curve[i:i + q]
                    print(f"  matches {chunk[0][0]}-{chunk[-1][0]}: {a.persona.name} won {sum(c[1] for c in chunk)} of {sum(c[1] + c[2] + c[3] for c in chunk)} rounds")
            return 0
        results = []
        clients.setdefault("scripted", build_client("scripted"))
        for n in range(1, args.rounds + 1):
            for fx, spot in ((a, ARENA[0]), (b, ARENA[1])):
                reset(gm, fx, spot, args.rules, args.weapon, args.armor)
            ready_in = prep(a, b, clients, args.pump_ms)
            for fx, spot in ((a, ARENA[0]), (b, ARENA[1])):
                gm.command_on(f"[Set X {spot.x} Y {spot.y} Z {spot.z}", fx.serial)   # back to the marks after dressing
            res = run_round(a, b, clients, args.max_ticks, args.pump_ms, args.log_dir, n, speech)
            res["prep_ticks"] = ready_in
            results.append(res)
            ra, rb = res[a.persona.name], res[b.persona.name]
            print(f"round {n}: winner={res['winner'] or 'draw'} in {max(ra['ticks'], rb['ticks'])} ticks ({res['seconds']}s, dressed in {res['prep_ticks']}) | "
                  f"{a.persona.name} hp={ra['hp']:.0%} bandages={ra['bandages']}(+{ra['slipped']} slipped) model={ra['model']} | "
                  f"{b.persona.name} hp={rb['hp']:.0%} bandages={rb['bandages']}(+{rb['slipped']} slipped) model={rb['model']}", flush=True)
        print(f"\nfinal: {a.persona.name} ({a.backend}) {a.wins} — {b.wins} {b.persona.name} ({b.backend})")
        for fx in (a, b):
            reset(gm, fx, ARENA[0] if fx is a else ARENA[1], args.rules, args.weapon, args.armor)
    finally:
        gm_body.close()
        for fx in (a, b):
            if fx.body:
                fx.body.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
