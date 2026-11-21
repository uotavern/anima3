"""Public-view tactical teacher: a scripted controller, never a neural policy.

``TacticalTeacher`` is ported unchanged from ``neural/curriculum.py`` at the
training tip ``819f4f4`` (``6b8d6f0`` to ``5672268``). The simulator drill helper
``setup_drill`` from the same source file needs ``sim.py`` and stays with the
simulator (row 7b), so this module holds only the teacher and its identity.

``TEACHER_SHA256`` identifies the teacher's behaviour, not just this file. It
is the SHA-256 of the canonical JSON map ``TEACHER_SOURCES`` from each module
the choice depends on to that module's SHA-256: this file, ``schema.py``
(``legal_mask``, ``ACTION_INDEX``, ``CombatView``), ``executor.py``
(``cast_seconds``), ``magic.py`` (spell circles behind ``cast_seconds``) and
``contract.py`` (imported by both). The simulator names it. A live teacher arm
names ``live.LIVE_TEACHER_SHA256``, this map extended by the live labelling
modules (``live.py``, the decision cadence and freshness rules, and
``duel_wait.py``, the DuelState interpretation), and every demonstration records
that map, so a demonstration says exactly which code chose and labelled its
actions. An edit to any of those modules changes the identity and refuses
resumption of a collection planned with the old one. The schema
fingerprint covers only feature and action names, not this logic. A teacher's
choices are demonstrations for supervised imitation only: they are never PPO
experience and never reported under a neural policy label.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .executor import cast_seconds
from .schema import ACTION_INDEX, CombatView, legal_mask

STYLES = ("balanced", "pressure", "burst", "defensive", "potions")
TEACHER_STYLES = (*STYLES, "tactical")
TEACHER_VERSION = "tactical-teacher-v1"
PACKAGE = Path(__file__).resolve().parents[1]
# Every module whose code decides a teacher (or probe) choice, relative to anima3/.
DEPENDENCIES = ("neural/schema.py", "neural/executor.py", "magic.py", "contract.py")


def source_hashes(modules, root=PACKAGE) -> dict[str, str]:
    """``{module: SHA-256}`` for the given modules under ``root`` (the anima3 package)."""
    return {m: hashlib.sha256((Path(root) / m).read_bytes()).hexdigest() for m in sorted(modules)}


def source_identity(hashes: dict[str, str]) -> str:
    """The SHA-256 of a canonical module-hash map."""
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


TEACHER_MODULES = ("neural/teacher.py", *DEPENDENCIES)
TEACHER_SOURCES = source_hashes(TEACHER_MODULES)
TEACHER_SHA256 = source_identity(TEACHER_SOURCES)


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
