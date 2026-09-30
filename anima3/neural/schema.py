"""Versioned, public-observation contract shared by simulation and live play.

No opponent mana, inventory, cooldown or private policy state belongs here. A
checkpoint is compatible only when this schema's fingerprint matches exactly.
All durations are seconds, resource values are absolute, and positions are tiles.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

SCHEMA_VERSION = 1
ACTION_NAMES = (
    "hold", "approach", "retreat", "strafe_left", "strafe_right",
    "weaken", "clumsy", "magic_arrow", "harm", "explosion", "energy_bolt",
    "flamestrike", "greater_heal", "heal", "cure", "poison", "paralyze",
    "meditate", "heal_potion", "cure_potion", "explosion_potion",
    "burst_bolt", "burst_flame",
)
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
SPELL_NAMES = ACTION_NAMES[5:17]
REAGENT_NAMES = (
    "black_pearl", "bloodmoss", "garlic", "ginseng", "mandrake",
    "nightshade", "spiders_silk", "sulfurous_ash",
)
REAGENT_GRAPHICS = (0x0F7A, 0x0F7B, 0x0F84, 0x0F85, 0x0F86, 0x0F88, 0x0F8D, 0x0F8C)
MANA_COST = {
    "weaken": 4, "clumsy": 4, "magic_arrow": 4, "harm": 6,
    "explosion": 20, "energy_bolt": 20, "flamestrike": 40,
    "greater_heal": 11, "heal": 4, "cure": 6, "poison": 9, "paralyze": 14,
}
SPELL_CIRCLE = {
    "weaken": 1, "clumsy": 1, "magic_arrow": 1, "harm": 2,
    "explosion": 6, "energy_bolt": 6, "flamestrike": 7,
    "greater_heal": 4, "heal": 1, "cure": 2, "poison": 3, "paralyze": 5,
}
# Indices into REAGENT_NAMES. Combined options consume each constituent's reagents.
SPELL_REAGENTS = {
    "weaken": (2, 5), "clumsy": (1, 5), "magic_arrow": (7,), "harm": (5, 6),
    "explosion": (1, 4), "energy_bolt": (0, 5), "flamestrike": (6, 7),
    "greater_heal": (2, 3, 4, 6), "heal": (2, 3, 6), "cure": (2, 3),
    "poison": (5,), "paralyze": (2, 4, 6),
}

FEATURE_NAMES = (
    "self_hp", "self_mana", "self_stamina", "opponent_hp", "opponent_hp_known",
    "relative_x", "relative_y", "distance", "self_poisoned", "opponent_poisoned",
    "self_paralyzed", "opponent_paralyzed", "cast_remaining", "busy_remaining",
    "recovery_remaining", "heal_potion_cooldown", "cure_potion_cooldown",
    "explosion_potion_cooldown", "burst_cooldown", "heal_potions", "cure_potions",
    "explosion_potions",
) + tuple("reagent_" + name for name in REAGENT_NAMES) + (
    "elapsed", "showdown", "time_to_showdown", "recent_damage_dealt",
    "recent_damage_received", "recent_self_damage", "opponent_spell_age",
    "opponent_visible", "line_of_sight", "fresh", "ready", "move_approach",
    "move_retreat", "move_strafe_left", "move_strafe_right", "potion_fuse",
    "incoming_explosion", "incoming_explosion_delay",
) + tuple("opponent_cast_" + name for name in SPELL_NAMES) + (
    "last_action_attack", "last_action_heal", "last_action_control",
    "last_action_move", "last_action_potion",
)


@dataclass(frozen=True)
class CombatView:
    """A single player's knowledge, not simulator/server omniscient state.

    ``recent_damage_*`` are damage observed over the last two seconds. A missing
    opponent health bar is represented by ``opponent_health_known=False``.
    ``move_legal`` is approach/retreat/left/right after terrain checks. Unknown
    incoming Explosion is -1; only one's own observed opponent cast/target may
    supply this estimate. ``opponent_cast`` is a public power-word spell key.
    """

    hp: float = 100.0
    hp_max: float = 100.0
    mana: float = 100.0
    mana_max: float = 100.0
    stamina: float = 100.0
    stamina_max: float = 100.0
    opponent_hp: float = 0.0
    opponent_hp_max: float = 100.0
    opponent_health_known: bool = False
    dx: float = 0.0
    dy: float = 0.0
    distance: float = 0.0
    poisoned: bool = False
    opponent_poisoned: bool = False
    paralyzed: bool = False
    opponent_paralyzed: bool = False
    cast_remaining: float = 0.0
    busy_remaining: float = 0.0
    recovery_remaining: float = 0.0
    heal_potion_cooldown: float = 0.0
    cure_potion_cooldown: float = 0.0
    explosion_potion_cooldown: float = 0.0
    burst_cooldown: float = 0.0
    heal_potions: int = 0
    cure_potions: int = 0
    explosion_potions: int = 0
    reagent_counts: tuple[int, ...] = (0,) * 8
    elapsed: float = 0.0
    showdown: bool = False
    showdown_after: float = 180.0
    recent_damage_dealt: float = 0.0
    recent_damage_received: float = 0.0
    recent_self_damage: float = 0.0
    opponent_cast: str = ""
    opponent_spell_age: float = 10.0
    opponent_visible: bool = False
    line_of_sight: bool = False
    fresh: bool = True
    ready: bool = True
    move_legal: tuple[bool, bool, bool, bool] = (False,) * 4
    potion_fuse: float = 0.0
    incoming_explosion_seconds: float = -1.0
    last_action: str = "hold"

    def __post_init__(self) -> None:
        if len(self.reagent_counts) != len(REAGENT_NAMES):
            raise ValueError("reagent_counts must have eight entries in REAGENT_NAMES order")
        if len(self.move_legal) != 4:
            raise ValueError("move_legal must be approach/retreat/strafe_left/strafe_right")


@dataclass(frozen=True)
class Frame:
    features: list[float]
    mask: list[bool]


def _unit(value: float, scale: float = 1.0, *, signed: bool = False) -> float:
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0:
        return 0.0
    return max(-1.0 if signed else 0.0, min(1.0, value / scale))


def _reagents(view: CombatView, *spells: str) -> bool:
    needed = [0] * len(REAGENT_NAMES)
    for spell in spells:
        for reagent in SPELL_REAGENTS[spell]:
            needed[reagent] += 1
    return all(have >= need for have, need in zip(view.reagent_counts, needed, strict=True))


def legal_mask(view: CombatView) -> list[bool]:
    """Conservative option mask used unchanged by simulator and executor.

    An option owns its cursor until completion: neural decisions cannot replace
    a pending cast or primed-potion sequence. The executor remains responsible
    for emergency cleanup inside the active option.
    """
    mask = [False] * len(ACTION_NAMES)
    mask[0] = True
    if (not view.fresh or not view.ready or view.hp <= 0
            or max(view.cast_remaining, view.busy_remaining, view.potion_fuse) > 0):
        return mask
    if not view.paralyzed and view.stamina > 0:
        mask[1:5] = list(view.move_legal)
    can_cast = not view.paralyzed and view.recovery_remaining <= 0
    targeted = view.opponent_visible and view.line_of_sight and view.distance <= 12
    for name in SPELL_NAMES:
        possible = can_cast and view.mana >= MANA_COST[name] and _reagents(view, name)
        if name in ("heal", "greater_heal"):
            possible = possible and not view.poisoned and not view.showdown and view.hp < view.hp_max
        elif name == "cure":
            possible = possible and view.poisoned
        else:
            possible = possible and targeted
            if name == "poison":
                possible = possible and not view.opponent_poisoned
            elif name == "paralyze":
                possible = possible and not view.opponent_paralyzed
        mask[ACTION_INDEX[name]] = bool(possible)
    mask[ACTION_INDEX["meditate"]] = (
        can_cast and not view.poisoned and view.mana < view.mana_max - 2
    )
    mask[ACTION_INDEX["heal_potion"]] = (
        view.heal_potions > 0 and view.heal_potion_cooldown <= 0
        and not view.poisoned and not view.showdown and view.hp < view.hp_max
    )
    mask[ACTION_INDEX["cure_potion"]] = (
        view.cure_potions > 0 and view.cure_potion_cooldown <= 0 and view.poisoned
    )
    potion_ok = (view.explosion_potions > 0 and view.explosion_potion_cooldown <= 0
                 and targeted and view.distance >= 3 and not view.paralyzed)
    mask[ACTION_INDEX["explosion_potion"]] = potion_ok
    for action, finisher in (("burst_bolt", "energy_bolt"), ("burst_flame", "flamestrike")):
        mask[ACTION_INDEX[action]] = (
            potion_ok and can_cast and 3 <= view.distance <= 10 and not view.poisoned
            and view.hp >= view.hp_max * 0.6 and view.burst_cooldown <= 0
            and view.mana >= 20 + MANA_COST[finisher] + 11
            and _reagents(view, "explosion", finisher)
        )
    return mask


def encode(view: CombatView) -> Frame:
    """Encode 65 bounded features; unknown health never leaks through its value."""
    values = [
        _unit(view.hp, view.hp_max), _unit(view.mana, view.mana_max),
        _unit(view.stamina, view.stamina_max),
        _unit(view.opponent_hp, view.opponent_hp_max) if view.opponent_health_known else 0.0,
        float(view.opponent_health_known), _unit(view.dx, 18, signed=True),
        _unit(view.dy, 18, signed=True), _unit(view.distance, 18),
        float(view.poisoned), float(view.opponent_poisoned), float(view.paralyzed),
        float(view.opponent_paralyzed), _unit(view.cast_remaining, 3),
        _unit(view.busy_remaining, 8), _unit(view.recovery_remaining, 1),
        _unit(view.heal_potion_cooldown, 10), _unit(view.cure_potion_cooldown, 2),
        _unit(view.explosion_potion_cooldown, 10), _unit(view.burst_cooldown, 20),
        _unit(view.heal_potions, 20), _unit(view.cure_potions, 20),
        _unit(view.explosion_potions, 20),
    ]
    values.extend(_unit(count, 100) for count in view.reagent_counts)
    values.extend((
        _unit(view.elapsed, 300), float(view.showdown),
        _unit(max(0.0, view.showdown_after - view.elapsed), 180),
        _unit(view.recent_damage_dealt, 100), _unit(view.recent_damage_received, 100),
        _unit(view.recent_self_damage, 100), _unit(view.opponent_spell_age, 10),
        float(view.opponent_visible), float(view.line_of_sight), float(view.fresh),
        float(view.ready), *(float(value) for value in view.move_legal),
        _unit(view.potion_fuse, 3.75), float(view.incoming_explosion_seconds >= 0),
        _unit(view.incoming_explosion_seconds, 2.5),
    ))
    values.extend(float(view.opponent_cast == name) for name in SPELL_NAMES)
    values.extend((
        float(view.last_action in ("magic_arrow", "harm", "explosion", "energy_bolt",
                                   "flamestrike", "burst_bolt", "burst_flame")),
        float(view.last_action in ("heal", "greater_heal", "cure", "meditate")),
        float(view.last_action in ("weaken", "clumsy", "poison", "paralyze")),
        float(view.last_action in ACTION_NAMES[1:5]),
        float(view.last_action in ACTION_NAMES[18:21]),
    ))
    assert len(values) == len(FEATURE_NAMES)
    return Frame(features=values, mask=legal_mask(view))


SCHEMA_FINGERPRINT = hashlib.sha256(json.dumps({
    "version": SCHEMA_VERSION, "features": FEATURE_NAMES, "actions": ACTION_NAMES,
    "normalization": "neural-v1-pre-aos-public-observation",
}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def schema_fingerprint() -> str:
    return SCHEMA_FINGERPRINT
