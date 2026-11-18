"""Ordinary-account template preparation; no policy loop or training runner."""
from __future__ import annotations

import json
import re
import time

from .contract import say, use

MATCH_ID = re.compile(r"^[0-9a-f]{32}$")
RULES = "7x-magic-explosion-classic"
SKILLS = [25, 16, 46, 26, 43, 1, 0]
TRAINING_SUPPLIES = {g: 100 for g in (0xF7A,0xF7B,0xF84,0xF85,0xF86,0xF88,0xF8D,0xF8C)}
TRAINING_SUPPLIES.update({g: 5 for g in (0xF0C,0xF07,0xF0D,0xF0B,0xF08)})


def server_state(obs):
    result = None
    for j in obs.new_journal:
        if j.serial not in (0, -1, 0xFFFFFFFF) or not j.text.startswith("[DuelState] "):
            continue
        try:
            s = json.loads(j.text[12:])
            if not isinstance(s, dict):
                continue

            def valid(v):
                return (
                    isinstance(v, dict)
                    and isinstance(v.get("id"), str)
                    and MATCH_ID.fullmatch(v["id"])
                    and type(v.get("opponent")) is int
                    and v["opponent"] > 0
                    and isinstance(v.get("rules"), str)
                )

            if s.get("phase") == "Idle":
                c = s.get("challenge")
                if c is None or (
                    valid(c) and type(c.get("rounds")) is int and 1 <= c["rounds"] <= 15
                ):
                    result = s
            elif (
                s.get("phase") in ("Countdown", "Fighting", "RoundOver")
                and valid(s)
                and type(s.get("round")) is int
                and 1 <= s["round"] <= 15
            ):
                result = s
        except (ValueError, TypeError):
            pass
    return result


def pump(body, seconds=2):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        body.pump(250)
    return body.observe_raw()


def configured(o):
    bases = {s.get("id", n): s["base"] for n, s in enumerate(o.get("skills", []))}
    return (
        all(bases.get(n) == 100 for n in SKILLS)
        and sum(bases.values()) == 700
        and [o["player"].get(k) for k in ("strength", "dexterity", "intelligence")]
        == [100, 25, 100]
    )


def read_base_stats(body, observation, timeout=35):
    """Read RawStr/RawDex/RawInt through the ordinary Arena stat-ball dialog.

    Status packets contain modified stats, so waiting for curses/buffs is not
    proof of the base template. The server dialog explicitly shows raw values.
    Cancel rather than apply it; no privileges or stat edits are used here.
    """
    deadline = time.monotonic() + timeout
    for gump in observation.get("gumps", []):
        if "ARENA STATS" in str(gump):
            body.act({"type": "GumpResponse", "serial": gump["serial"],
                      "gump_id": gump["gump_id"], "button": 0})
    requested = float("-inf")
    while time.monotonic() < deadline:
        packs = {i["serial"] for i in observation.get("items", [])
                 if i.get("layer") == 21 and i.get("container") == observation["player"]["serial"]}
        ball = next((i for i in observation.get("items", [])
                     if i.get("graphic") == 0xE2D and i.get("hue") == 53
                     and i.get("container") in packs), None)
        if ball:
            body.act(use(ball["serial"]))
            observation = pump(body, 1)
            gump = next((g for g in observation.get("gumps", []) if "ARENA STATS" in str(g)), None)
            if gump:
                entries = {e.get("id"): e.get("s") for e in gump.get("elements", [])
                           if e.get("type") == "entry"}
                body.act({"type": "GumpResponse", "serial": gump["serial"],
                          "gump_id": gump["gump_id"], "button": 0})
                pump(body, .5)
                try:
                    return [int(entries[i]) for i in range(3)]
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError("Arena stat dialog lacks three base stats") from exc
        elif time.monotonic() - requested >= 5:
            body.act(say("[Arena stats"))
            requested = time.monotonic()
            for pack in packs:
                body.act(use(pack))
        observation = pump(body, 1)
    raise RuntimeError("Could not verify base stats from Arena stat dialog")


def wait_prepared_stats(body, observation, timeout=125):
    """A previous pre-AOS Weaken/Clumsy can outlive the match by up to 120s."""
    deadline = time.monotonic() + timeout
    while [observation["player"].get(k) for k in ("strength", "dexterity", "intelligence")] != [100, 25, 100]:
        if time.monotonic() >= deadline:
            stats = [observation["player"].get(k) for k in ("strength", "dexterity", "intelligence")]
            raise RuntimeError(f"Server has not confirmed 100/25/100 stats after debuff recovery: {stats}")
        body.act({"type": "StatusRequest", "serial": observation["player"]["serial"]})
        observation = pump(body, 1)
    return observation



def training_supply_audit(observation):
    """Exact experimental supplies visible in the participant's open backpack."""
    player = observation['player']
    pack = next((i['serial'] for i in observation['items']
                 if i.get('layer') == 21 and i.get('container') == player['serial']), None)
    required = TRAINING_SUPPLIES
    counts = {g: sum(i.get('amount', 0) for i in observation['items']
                     if pack is not None and i.get('container') == pack and i['graphic'] == g)
              for g in required}
    valid = (pack is not None and player.get('dead') is False
             and player.get('weight_max', 0) > 0
             and 0 <= player.get('weight', -1) <= player['weight_max']
             and counts == required)
    return {'serial': player['serial'], 'verified': bool(valid),
            'weight': player.get('weight'), 'weight_max': player.get('weight_max'),
            'counts': {hex(g): counts[g] for g in required},
            'required': {hex(g): required[g] for g in required}}


def verify_training_supplies(body):
    o = body.observe_raw()
    pack = next((i['serial'] for i in o['items'] if i.get('layer') == 21
                 and i.get('container') == o['player']['serial']), None)
    if pack is not None:
        body.act(use(pack))
    body.act({'type': 'StatusRequest', 'serial': o['player']['serial']})
    return training_supply_audit(pump(body, 1))

def _prepare(body, previously_configured=False, verify_base_stats=False):
    """Use the same public commands and consumable dialogs as a human player."""
    for command in (
        "[Arena leave",
        "[Arena enter",
        "[Arena supplies",
    ):
        body.act(say(command))
        pump(body)
    o = body.observe_raw()
    pack = next(
        i for i in o["items"] if i.get("layer") == 21 and i["container"] == o["player"]["serial"]
    )
    body.act(use(pack["serial"]))
    o = pump(body)
    body.act({"type": "SkillsRequest"})
    o = pump(body, 1)
    if configured(o) and not verify_base_stats:
        return o["player"]
    if verify_base_stats:
        bases = {s.get("id", n): s["base"] for n, s in enumerate(o.get("skills", []))}
        skills_ok = all(bases.get(n) == 100 for n in SKILLS) and sum(bases.values()) == 700
        if skills_ok:
            raw_stats = read_base_stats(body, o)
            if raw_stats == [100, 25, 100]:
                return {**o["player"], "base_stat_verification": {
                    "source": "ArenaStatsGump", "values": raw_stats}}
            if previously_configured:
                raise RuntimeError(f"Previously verified base stats changed: {raw_stats}")
    if previously_configured:
        # A verified earlier match establishes the base build. Stat curses are
        # temporary; requesting fresh balls can be refused while combat clears.
        o = wait_prepared_stats(body, o)
        body.act({"type": "SkillsRequest"})
        o = pump(body, 1)
        if not configured(o):
            raise RuntimeError("Previously verified training build changed")
        return o["player"]
    for command in ("[Arena skills", "[Arena stats"):
        body.act(say(command))
        o = pump(body)
    body.act(use(pack["serial"]))
    o = pump(body)
    for hue, title, values in (
        (1153, "7GM SKILL BALL", {"switches": SKILLS}),
        (53, "ARENA STATS", {"entries": [[0, "100"], [1, "25"], [2, "100"]]}),
    ):
        ball = next(
            i
            for i in o["items"]
            if i["graphic"] == 0xE2D and i["hue"] == hue and i["container"] == pack["serial"]
        )
        body.act(use(ball["serial"]))
        g = None
        for attempt in range(16):
            if attempt == 8:
                body.act(use(ball["serial"]))
            o = pump(body, 0.5)
            g = next((g for g in o["gumps"] if title in str(g)), None)
            if g:
                break
        if g is None:
            raise RuntimeError("Preparation dialog missing: " + title)
        body.act(
            dict(type="GumpResponse", serial=g["serial"], gump_id=g["gump_id"], button=1, **values)
        )
        o = pump(body, 3)
    body.act({"type": "SkillsRequest"})
    o = pump(body, 3)
    bases = {s.get("id", n): s["base"] for n, s in enumerate(o["skills"])}
    if any(bases.get(n) != 100 for n in SKILLS) or sum(bases.values()) != 700:
        raise RuntimeError("Server has not confirmed requested seven GM skills")
    o = wait_prepared_stats(body, o)
    if verify_base_stats:
        raw_stats = read_base_stats(body, o)
        if raw_stats != [100, 25, 100]:
            raise RuntimeError("Server has not confirmed requested base stats")
        return {**o["player"], "base_stat_verification": {
            "source": "ArenaStatsGump", "values": raw_stats}}
    return o["player"]



class BoundedBody:
    """Cooperative deadline/cancellation before every bridge operation.

    A bridge RPC already in flight uses the bridge's own timeout. This is not a
    process watchdog and cannot interrupt a blocked third-party body method.
    """
    def __init__(self, body, timeout, stopped):
        if not 0 < timeout <= 360:
            raise ValueError("timeout must be within (0, 360] seconds")
        self.body, self.deadline, self.stopped = body, time.monotonic() + timeout, stopped

    def check(self):
        if self.stopped():
            raise RuntimeError("participant operation stopped")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("participant operation timed out")

    def act(self, action):
        self.check()
        return self.body.act(action)

    def pump(self, ms):
        self.check()
        return self.body.pump(ms)

    def observe(self):
        self.check()
        return self.body.observe()

    def observe_raw(self):
        self.check()
        return self.body.observe_raw()


def prepare(body, previously_configured=False, verify_base_stats=False, *,
            timeout=180, stopped=lambda: False):
    """Prepare only after a fresh idle response; never leave an active duel.

    Account ownership/locking belongs to the caller. Optional raw stat proof
    uses a public dialog and cancels it without applying modified status stats.
    """
    from .sparring import poll

    bounded = BoundedBody(body, timeout, stopped)
    state = poll(bounded, timeout=min(timeout, 30), stopped=stopped)
    if state['phase'] != 'Idle' or state.get('challenge'):
        raise RuntimeError("participant must be idle without a pending challenge before preparation")
    return _prepare(bounded, previously_configured, verify_base_stats)
