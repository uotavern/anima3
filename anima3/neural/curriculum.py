"""Public-view teachers and randomized simulator drills; no production mutation."""
from __future__ import annotations

from .executor import cast_seconds
from .schema import ACTION_INDEX, CombatView, legal_mask

STYLES = ("balanced", "pressure", "burst", "defensive", "potions")
DRILLS = ("melee", "survival", "burst", "full")


class TacticalTeacher:
    def __init__(self, style="balanced"):
        if style not in (*STYLES, "tactical"):
            raise ValueError(style)
        self.style = style
        self.opened = set()
        self.followup = False
        self.recovering_mana = False

    def choose(self, view: CombatView, mask=None) -> int:
        mask = legal_mask(view) if mask is None else mask
        choices = []
        hp = view.hp / max(1, view.hp_max)
        if view.poisoned:
            choices += ["cure_potion", "cure"]
        if hp < (0.8 if self.style == "defensive" else 0.65):
            choices += ["heal_potion", "greater_heal", "heal"]
        if view.stamina < 15:
            choices += ["refresh_potion", "agility_potion"]
        if hp < 0.3 or (self.style == "defensive" and view.distance < 4):
            choices += ["retreat"]
        if self.style == "tactical":
            if view.mana < 23:
                self.recovering_mana = True
            elif view.mana >= min(60, view.mana_max * 0.75):
                self.recovering_mana = False
            if self.recovering_mana:
                # No chip attacks or opening debuffs while rebuilding mana.
                # Hysteresis prevents spending each newly regenerated point.
                choices += ["retreat"] if view.distance < 6 else []
                choices += ["meditate", "hold"]
                return next(ACTION_INDEX[n] for n in choices if mask[ACTION_INDEX[n]])
            # Public spell words indicate a possible cast, not guaranteed success.
            remaining = (cast_seconds(view.opponent_cast) - view.opponent_spell_age
                         if view.opponent_cast in ("greater_heal", "explosion", "energy_bolt",
                                                   "flamestrike", "poison", "paralyze") else 0)
            if remaining > 0:
                if view.distance <= 1:
                    choices += ["stun", "punch"]
                elif view.distance == 2 and remaining > 0.5 and hp > 0.65:
                    choices += ["approach"]
                # First-circle cast + pre-AOS delayed damage + polling margin.
                if remaining > cast_seconds("magic_arrow") + 0.5 + 0.15:
                    choices += ["magic_arrow"]
            if view.distance <= 1:
                choices += ["stun", "punch"]
        if self.style == "pressure" and view.distance > 1:
            choices += ["approach"]
        if view.distance > 8:
            choices += ["approach"]
        if view.elapsed < 12:
            choices += [n for n in ("weaken", "clumsy") if n not in self.opened]
        if view.distance <= 1:
            choices += ["agility_potion", "stun", "punch", "harm"]
        if self.followup:
            choices += ["energy_bolt", "flamestrike"]
        if view.mana < 23:
            choices += ["retreat"] if view.distance < 6 else []
            choices += ["meditate"]
        if self.style == "tactical":
            # Flamestrike consumes more mana; retain a heal/cure reserve through
            # the existing legal mask. Prefer its larger dump on a weakened foe.
            hurt = view.opponent_health_known and view.opponent_hp / max(1, view.opponent_hp_max) < 0.6
            choices += (["burst_flame", "burst_bolt"] if hurt else ["burst_bolt", "burst_flame"])
            choices += ["explosion_potion"]
        if self.style == "burst":
            choices += ["burst_bolt", "explosion"]
        elif self.style == "potions":
            choices += ["explosion_potion", "energy_bolt"]
        elif self.style == "balanced":
            choices += ["burst_bolt", "explosion_potion"]
        if self.style == "defensive" and not view.opponent_poisoned:
            choices += ["poison"]
        if view.opponent_cast and view.opponent_spell_age < 0.6:
            choices += ["magic_arrow"]
        choices += ["explosion", "energy_bolt", "magic_arrow", "meditate", "hold"]
        name = next(n for n in choices if mask[ACTION_INDEX[n]])
        if name in ("weaken", "clumsy"):
            self.opened.add(name)
        if name == "explosion":
            self.followup = True
        elif name in ("energy_bolt", "flamestrike"):
            self.followup = False
        return ACTION_INDEX[name]


def setup_drill(sim, name, side):
    """Set only isolated simulator initial state and reset potential afterwards."""
    if name not in DRILLS:
        raise ValueError(name)
    own, other = sim.fighters[side], sim.fighters[1-side]
    if name != "full":
        distance = sim.rng.choice((1, 2, 3)) if name == "melee" else sim.rng.randint(3, 7)
        own.x, own.y, other.x, other.y = 0, 0, distance, 0
        own.mana = sim.rng.uniform(45, 100)
        if name == "survival":
            own.hp = sim.rng.uniform(25, 60)
            if sim.rng.random() < 0.5:
                own.poison_until = 20
                own.poison_generation += 1
                sim._schedule(3, "poison_tick", 1-side, (side, own.poison_generation))
        elif name == "burst":
            own.mana = 100
            other.hp = sim.rng.uniform(45, 90)
        elif name == "melee":
            own.stamina = sim.rng.uniform(5, 25)
    sim._potential_previous = sim._potential()
    return sim.frames()
