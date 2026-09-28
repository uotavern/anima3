"""Magery for the ring: a spell table, the cast procedure, and the mage's verb menu.

Casting is the flow anima2 proved live: `CastSpell{id}` → the server opens a target
cursor → `TargetObject{serial}` → the incantation runs for (4 + circle) × 0.25 s and can
fizzle, be disrupted by damage, or be refused for mana/reagents/recovery. The verdict is
read from the journal (fixed clilocs) or from the mana actually spent.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contract import Observation, target_object, use_skill

# --- reagents (art ids) ---------------------------------------------------------
BLACK_PEARL, BLOODMOSS, GARLIC, GINSENG, MANDRAKE, NIGHTSHADE, SPIDERS_SILK, SULFUROUS_ASH = (
    0x0F7A, 0x0F7B, 0x0F84, 0x0F85, 0x0F86, 0x0F88, 0x0F8D, 0x0F8C)
REAGENT_GRAPHICS = frozenset({BLACK_PEARL, BLOODMOSS, GARLIC, GINSENG, MANDRAKE, NIGHTSHADE, SPIDERS_SILK, SULFUROUS_ASH})
REAGENT_NAMES = {BLACK_PEARL: "BlackPearl", BLOODMOSS: "Bloodmoss", GARLIC: "Garlic", GINSENG: "Ginseng",
                 MANDRAKE: "MandrakeRoot", NIGHTSHADE: "Nightshade", SPIDERS_SILK: "SpidersSilk", SULFUROUS_ASH: "SulfurousAsh"}
SPELLBOOK_GRAPHIC = 0x0EFA
#: Pre-AOS mana by circle.
MANA = {1: 4, 2: 6, 3: 9, 4: 11, 5: 14, 6: 20, 7: 40, 8: 50}


@dataclass(frozen=True)
class Spell:
    key: str
    id: int          # wire id (1-based SpellRegistry index)
    circle: int
    reagents: frozenset[int]
    kind: str        # attack | heal | cure | buff | control
    blurb: str

    @property
    def mana(self) -> int:
        return MANA[self.circle]

    @property
    def cast_ticks(self) -> int:          # (4 + circle) * 0.25 s at ~0.3 s a tick, plus recovery
        return 2 + (4 + self.circle)


SPELLS: dict[str, Spell] = {s.key: s for s in [
    Spell("magic_arrow", 5, 1, frozenset({SULFUROUS_ASH}), "attack", "Magic Arrow: a quick, weak bolt (4 mana)."),
    Spell("harm", 12, 2, frozenset({NIGHTSHADE, SPIDERS_SILK}), "attack", "Harm: instant, hurts more up close (6 mana)."),
    Spell("fireball", 18, 3, frozenset({BLACK_PEARL}), "attack", "Fireball (9 mana)."),
    Spell("lightning", 30, 4, frozenset({MANDRAKE, SULFUROUS_ASH}), "attack", "Lightning: instant, solid damage (11 mana)."),
    Spell("energy_bolt", 42, 6, frozenset({BLACK_PEARL, NIGHTSHADE}), "attack", "Energy Bolt: heavy damage, slow cast (20 mana)."),
    Spell("explosion", 43, 6, frozenset({BLOODMOSS, MANDRAKE}), "attack", "Explosion: heavy damage that lands a moment later (20 mana)."),
    Spell("flamestrike", 51, 7, frozenset({SPIDERS_SILK, SULFUROUS_ASH}), "attack", "Flamestrike: the heaviest blow, very slow (40 mana)."),
    Spell("heal", 4, 1, frozenset({GARLIC, GINSENG, SPIDERS_SILK}), "heal", "Heal: a small quick heal on yourself (4 mana)."),
    Spell("greater_heal", 29, 4, frozenset({GARLIC, GINSENG, MANDRAKE, SPIDERS_SILK}), "heal", "Greater Heal: a large heal on yourself (11 mana)."),
    Spell("cure", 11, 2, frozenset({GARLIC, GINSENG}), "cure", "Cure: remove poison from yourself (6 mana)."),
    Spell("poison", 20, 3, frozenset({NIGHTSHADE}), "control", "Poison: poison the opponent, it keeps hurting (9 mana)."),
    Spell("paralyze", 38, 5, frozenset({GARLIC, MANDRAKE, SPIDERS_SILK}), "control", "Paralyze: freeze the opponent for a few seconds (14 mana)."),
    Spell("magic_reflection", 36, 5, frozenset({GARLIC, MANDRAKE, SPIDERS_SILK}), "buff", "Magic Reflection: bounce the next spell cast at you (14 mana)."),
    Spell("reactive_armor", 7, 1, frozenset({GARLIC, SPIDERS_SILK, SULFUROUS_ASH}), "buff", "Reactive Armor: reflect some melee damage (4 mana)."),
]}
SELF_TARGET = {"heal", "cure", "buff"}
#: Every cast is spoken aloud: the opponent's power words are how a mage reads the other's casting.
WORDS = {"in por ylem": "Magic Arrow", "an mani": "Harm", "vas flam": "Fireball", "por ort grav": "Lightning",
         "corp por": "Energy Bolt", "vas ort flam": "Explosion", "kal vas flam": "Flamestrike", "in mani": "Heal",
         "in vas mani": "Greater Heal", "an nox": "Cure", "in nox": "Poison", "an ex por": "Paralyze",
         "in jux sanct": "Magic Reflection", "flam sanct": "Reactive Armor"}

FIZZLE, NO_MANA, NO_REAGENTS, RECOVERING, ALREADY, FROZEN, NO_LOS = 502632, 502625, 502630, 502644, 502642, 502643, 500947
VERDICTS = {FIZZLE: "fizzle", NO_MANA: "no mana", NO_REAGENTS: "no reagents", RECOVERING: "recovering",
            502645: "already casting", ALREADY: "already casting", FROZEN: "frozen", 502646: "frozen",
            NO_LOS: "no line of sight", 502647: "no line of sight", 502629: "cannot cast here", 502628: "hands not free"}


def has_reagents(obs: Observation, spell: Spell) -> bool:
    have = {i.graphic for i in obs.own_pack() if i.graphic in REAGENT_GRAPHICS}
    return spell.reagents <= have


def cast_proc(spell: Spell, target_serial: int):
    def proc(obs0, memory):
        mana0 = obs0.player.mana
        obs = yield {"type": "CastSpell", "spell": spell.id}
        for _ in range(8):
            cl = {j.cliloc for j in obs.new_journal}
            for c, v in VERDICTS.items():
                if c in cl:
                    return v
            if obs.pending_target:
                break
            obs = yield None
        if not obs.pending_target:
            return "no cursor"
        obs = yield target_object(target_serial)
        low = mana0
        for _ in range(spell.cast_ticks + 4):
            cl = {j.cliloc for j in obs.new_journal}
            for c, v in VERDICTS.items():
                if c in cl:
                    return v
            low = min(low, obs.player.mana)
            if low <= mana0 - spell.mana // 2:            # the mana went: the spell went off (regen can mask the full cost)
                memory["last_cast"] = (memory.get("tick", 0), spell.key)
                if spell.key == "magic_reflection":           # nothing else set it: reflection was always "due"
                    memory["reflect_at"] = memory.get("tick", 0)
                memory["recover_until"] = memory.get("tick", 0) + 3
                return "ok"
            obs = yield None
        return "ok" if low < mana0 else "timeout"
    return proc


def meditate_proc():
    def proc(obs0, memory):
        mana0 = obs0.player.mana
        obs = yield use_skill(46)
        for _ in range(12):
            if obs.player.mana > mana0 + 2:
                return "ok"
            if any(j.cliloc in (500118, 501846, 502628) for j in obs.new_journal):
                return "refused"
            obs = yield None
        return "timeout"
    return proc


#: The rule's alternatives: each is a spell order the hand rule executes tick by tick. Which one
#: fits the moment has no obvious answer — that is where a judge can add something the rule cannot.
PLAYBOOKS: dict[str, str] = {
    "standard": "Heal below half, cure poison, then the heaviest bolt the mana allows (the hand rule).",
    "control": "Paralyze the opponent, then land Explosion and Energy Bolt while it cannot move or cast.",
    "poison": "Poison the opponent, then keep it busy with quick Harm and Magic Arrow so it cannot cure.",
    "interrupt": "Chain quick, cheap spells (Magic Arrow, Harm, Lightning) to disrupt the opponent's casting; mana-thrifty.",
    "sustain": "Stay healthy: heal below four fifths, cure at once, keep Magic Reflection up; attack in between.",
}
#: The same five spell orders in one neutral shape: each says only what `spell_order` casts, in
#: the same words. Experiment 4's vivid descriptions drew Jev to `control` 87% of the time
#: ("paralyze, then land Explosion…" reads as the decisive option) — the worst playbook there.
PLAYBOOKS_NEUTRAL: dict[str, str] = {
    "standard": "Order: Greater Heal below 50% health, Cure when poisoned, then Energy Bolt, Explosion, Lightning, Fireball, Harm, Magic Arrow.",
    "control": "Order: Greater Heal below 40% health, Paralyze, and while the opponent is paralyzed Explosion and Energy Bolt, then Lightning, Fireball, Harm, Magic Arrow.",
    "poison": "Order: Greater Heal below 40% health, Poison while the opponent is not poisoned, then Harm, Magic Arrow, Lightning, Energy Bolt, Explosion.",
    "interrupt": "Order: Greater Heal below 40% health, then Magic Arrow, Harm, Lightning, Fireball, Energy Bolt, Explosion.",
    "sustain": "Order: Greater Heal below 80% health, Cure when poisoned, Magic Reflection when due, then Energy Bolt, Explosion, Lightning, Fireball, Harm, Magic Arrow.",
}
_HEAVY = ("energy_bolt", "explosion", "lightning", "fireball", "harm", "magic_arrow")
_QUICK = ("magic_arrow", "harm", "lightning", "fireball", "energy_bolt", "explosion")


def spell_order(playbook: str, hp: float, poisoned: bool, opp_paralyzed: bool, opp_poisoned: bool, reflect_due: bool) -> list[str]:
    """The spell keys to offer, first = the rule's pick, for one playbook. `standard` is the
    hand rule of experiments 1-3, unchanged; every other playbook keeps a survival floor
    (Greater Heal under 40%, Cure when poisoned and under 60%) ahead of its own order."""
    if playbook not in PLAYBOOKS or playbook == "standard":
        order = (["greater_heal"] if hp < 0.5 else []) + (["cure"] if poisoned else []) + list(_HEAVY)
        order += (["greater_heal"] if hp < 0.8 else []) + ["paralyze", "poison"] + (["magic_reflection"] if reflect_due else [])
    else:
        order = (["greater_heal"] if hp < 0.4 else []) + (["cure"] if poisoned and hp < 0.6 else [])
        if playbook == "control":
            order += (["explosion", "energy_bolt"] if opp_paralyzed else ["paralyze"]) + list(_HEAVY)
        elif playbook == "poison":
            order += ([] if opp_poisoned else ["poison"]) + ["harm", "magic_arrow", "lightning"] + list(_HEAVY)
        elif playbook == "interrupt":
            order += list(_QUICK)
        elif playbook == "sustain":
            order += (["greater_heal"] if hp < 0.8 else []) + (["cure"] if poisoned else [])
            order += (["magic_reflection"] if reflect_due else []) + list(_HEAVY)
        order += (["cure"] if poisoned else []) + (["greater_heal"] if hp < 0.8 else []) + ["paralyze", "poison"]
    seen: set[str] = set()
    return [k for k in order if not (k in seen or seen.add(k))]


def explosion_potion_proc(serial: int, opponent: int):
    """Own the cursor until this potion has been thrown or the request times out."""
    from .contract import use
    def proc(obs0, memory):
        memory["potion_after"] = memory.get("tick", 0) + 40
        obs = yield use(serial)
        for _ in range(12):
            if obs.pending_target:
                yield target_object(opponent)
                return "thrown"
            obs = yield None
        return "no potion cursor"
    return proc


def mage_verbs(obs: Observation, f, memory: dict, threat) -> list:
    """The mage's closed menu in a duel, rule-ordered by the current playbook (`memory["playbook"]`,
    `standard` when none is set): heal when low, cure when poisoned, attacks in the playbook's
    order, meditate when dry and clear, wrestle when cornered."""
    from .affordances import Affordance, _step_options
    from .contract import attack, walk
    p = obs.player
    mana = p.mana
    out: list[Affordance] = []

    def cast(key: str, why: str = "") -> None:
        s = SPELLS[key]
        if memory.get("showdown") and s.kind == "heal":
            return
        if mana >= s.mana and has_reagents(obs, s) and not any(a.id == f"cast:{key}" for a in out):
            tgt = p.serial if s.kind in SELF_TARGET else threat.serial
            out.append(Affordance(f"cast:{key}", s.blurb + why, procedure=cast_proc(s, tgt)))

    if memory.get("tick", 0) < memory.get("recover_until", 0):
        return [Affordance("recover", "Catch your breath for a moment; the last spell still echoes.")]
    hp = f.hp_pct
    if (memory.get("explosion_potions") and not obs.pending_target and hp >= 0.5
            and 2 <= threat.distance <= 10
            and memory.get("tick", 0) >= memory.get("potion_after", 0)):
        potion = next((i for i in obs.own_pack() if i.graphic == 0xF0D), None)
        if potion:
            out.append(Affordance("potion:explosion", "Throw an explosion potion at the duel opponent.",
                                  procedure=explosion_potion_proc(potion.serial, threat.serial)))
    playbook = memory.get("playbook", "standard")
    reflect_due = memory.get("tick", 0) - memory.get("reflect_at", -999) > 60
    why = {"greater_heal": " You are below half." if hp < 0.5 else "", "cure": " You are poisoned."}
    for key in spell_order(playbook, hp, p.poisoned, getattr(threat, "paralyzed", False), getattr(threat, "poisoned", False), reflect_due):
        cast(key, why.get(key, ""))
    meditate_below = 0.6 if playbook in ("sustain", "interrupt") else 0.4
    if mana < meditate_below * max(1, p.mana_max) and threat.distance >= 3:
        out.append(Affordance("meditate", f"Meditate to recover mana ({mana}/{p.mana_max}); the opponent is {threat.distance} tiles off.",
                              procedure=meditate_proc()))
    if threat.distance <= 1:
        out.append(Affordance(f"attack:{threat.serial}", f"Wrestle {threat.name or 'the opponent'} hand to hand.", (attack(threat.serial),)))
    steps = _step_options(obs, away_from=threat.pos)
    if steps and threat.distance <= 2:
        out.append(Affordance("kite", f"Step {steps[0][1]}, out of arm's reach, to cast.", (walk(steps[0][0], run=True),)))
    return out
