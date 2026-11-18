"""Unit-4 integration boundaries: stun state, bounded cursors and the reflection fix.

These cases pin behavior the lineage plan names for this unit that the imported
training tests do not cover directly. No server, model or bridge is involved.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_combat_combo import world
from test_neural_runtime import observation

from anima3 import magic
from anima3.contract import Journal, Skill
from anima3.magic import (
    NO_CURSOR,
    SPELLS,
    cast_proc,
    explosion_potion_proc,
    mage_verbs,
    opening_proc,
)
from anima3.neural.executor import CombatExecutor
from anima3.neural.schema import ACTION_INDEX, SPELL_NAMES

STUN_SKILLS = [Skill(1, 90.0, 90.0, 100.0, 0), Skill(43, 90.0, 90.0, 100.0, 0)]


def melee_ready(now=0.0, journal=()):
    obs = observation(enemy_x=101, journal=journal)
    obs.skills = list(STUN_SKILLS)
    return obs


def fighting_executor(now=0.0):
    ex = CombatExecutor(2)
    ex.reset(2, "match-1/round-1", now)
    ex.observe(melee_ready(now), {"phase": "Fighting", "opponent": 2}, now)
    return ex


def test_stun_requests_then_attacks_and_server_feedback_drives_cooldown():
    ex = fighting_executor()
    assert ex.observe(melee_ready(), now=0.0).mask[ACTION_INDEX["stun"]]
    assert ex.start("stun", melee_ready(), 0.0)
    assert ex.tick(melee_ready(), 0.0) == [
        {"type": "StunRequest"},
        {"type": "Attack", "serial": 2},
    ]
    assert ex.observe(melee_ready(), now=0.1).mask[ACTION_INDEX["stun"]] is False
    assert ex.view.stun_pending
    ex.tick(melee_ready(), 0.3)
    assert not ex.busy
    # While pending, a second stun cannot be issued even after the swing timer.
    assert not ex.observe(melee_ready(), now=2.1).mask[ACTION_INDEX["stun"]]
    # 1019011: the server armed the stun; 1004013: it landed and starts the cooldown.
    ex.observe(melee_ready(journal=[Journal(1, "", "", 0, 0, 1019011)]), now=2.2)
    assert ex.view.stun_armed and not ex.view.stun_pending
    ex.observe(melee_ready(journal=[Journal(1, "", "", 0, 0, 1004013)]), now=2.4)
    assert not ex.view.stun_armed and ex.view.stun_cooldown == pytest.approx(10.0)
    assert not ex.observe(melee_ready(), fight(12.3), now=12.3).mask[ACTION_INDEX["stun"]]
    assert ex.observe(melee_ready(), fight(12.5), now=12.5).mask[ACTION_INDEX["stun"]]
    # Without a server state for three seconds every option is masked again.
    assert not ex.observe(melee_ready(), now=15.6).mask[ACTION_INDEX["stun"]]


def fight(at):
    return {"phase": "Fighting", "opponent": 2, "_received_at": at}


def test_stun_requires_both_skills_and_reach():
    ex = fighting_executor()
    unskilled = observation(enemy_x=101)
    assert not ex.observe(unskilled, now=0.0).mask[ACTION_INDEX["stun"]]
    distant = observation(enemy_x=103)
    distant.skills = list(STUN_SKILLS)
    assert not ex.observe(distant, now=0.0).mask[ACTION_INDEX["stun"]]


def test_round_boundary_clears_melee_and_stun_state():
    ex = fighting_executor()
    assert ex.start("stun", melee_ready(), 0.0)
    ex.tick(melee_ready(), 0.0)
    ex.reset(2, "match-1/round-2", 1.0)
    ex.observe(melee_ready(), {"phase": "Fighting", "opponent": 2}, 1.0)
    view = ex.view
    assert not (view.stun_pending or view.stun_armed or view.melee_active)


def test_executor_never_owns_a_cursorless_buff():
    # Magic Reflection's fix depends on buffs never waiting for a target cursor.
    # The neural option set must not reintroduce a buff through its own cast path.
    assert "buff" in NO_CURSOR
    assert all(SPELLS[name].kind not in NO_CURSOR for name in SPELL_NAMES)


def test_reflection_still_casts_without_cursor_and_restarts_clock_on_attempt():
    memory = {"tick": 40}
    gen = cast_proc(SPELLS["magic_reflection"], 7)(world(), memory)
    assert next(gen) == {"type": "CastSpell", "spell": SPELLS["magic_reflection"].id}
    assert memory["reflect_at"] == 40          # the attempt, not the success, restarts it
    with pytest.raises(StopIteration) as done:
        gen.send(world(mana=86))
    assert done.value.value == "ok"


def test_cast_cursor_wait_outlasts_eight_observations_within_wall_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(magic.time, "monotonic", lambda: now[0])
    gen = cast_proc(SPELLS["energy_bolt"], 2)(world(), {"tick": 1})
    next(gen)
    for _ in range(12):                         # 12 fast frames exceed the old 8-tick budget
        now[0] += 0.1
        assert gen.send(world()) is None
    cursor = world(target={"cursor_id": 4, "cursor_flag": 1, "target_type": 0})
    assert gen.send(cursor) == {"type": "TargetObject", "serial": 2}


def test_cast_cursor_wait_stops_at_wall_clock_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(magic.time, "monotonic", lambda: now[0])
    gen = cast_proc(SPELLS["energy_bolt"], 2)(world(), {"tick": 1})
    next(gen)
    # The bound starts when the first post-cast observation arrives:
    # (3 + circle 6) * 0.25 + 2 = 4.25 s.
    now[0] = 0.1
    assert gen.send(world()) is None
    now[0] = 4.3
    assert gen.send(world()) is None
    now[0] = 4.4
    with pytest.raises(StopIteration) as done:
        gen.send(world())
    assert done.value.value == "no cursor"


def test_showdown_offers_no_healing_spell():
    def menu(memory):
        return [a.id for a in mage_verbs(world(hp=30), SimpleNamespace(hp_pct=0.3), memory,
                                          world().mobiles[0])]
    assert menu({"tick": 1})[0] == "cast:greater_heal"          # control: healing offered first
    showdown = menu({"tick": 1, "showdown": True})
    assert not any(a in ("cast:greater_heal", "cast:heal") for a in showdown)
    assert any(a.startswith("cast:") for a in showdown)


def test_opening_abandoned_for_survival_cancels_its_cursor():
    memory = {"tick": 1}
    gen = opening_proc("weaken", 2)(world(), memory)
    assert next(gen)["type"] == "CastSpell"
    hurt = world(hp=20, target={"cursor_id": 3, "cursor_flag": 1, "target_type": 0})
    assert gen.send(hurt) == {"type": "TargetCancel"}
    with pytest.raises(StopIteration) as done:
        gen.send(world(hp=20))
    assert done.value.value == "opening abandoned for survival"
    assert memory["opening_abandoned"] and memory["opening_attempts"] == {"weaken": 1}


def test_explosion_potion_waits_past_a_stale_spell_cursor_for_its_throw_cursor():
    memory = {"tick": 3}
    gen = explosion_potion_proc(101, 2)(world(), memory)
    assert next(gen) == {"type": "Use", "serial": 101}
    # The frame after lighting the potion can still show an old spell cursor.
    stale = world(target={"cursor_id": 5, "cursor_flag": 1, "target_type": 0})
    assert gen.send(stale) is None
    assert gen.send(stale) is None
    throw = world(target={"cursor_id": 6, "cursor_flag": 0, "target_type": 1})
    assert gen.send(throw) == {"type": "TargetObject", "serial": 2}


def test_explosion_potion_never_targets_a_persistent_foreign_cursor():
    gen = explosion_potion_proc(101, 2)(world(), {"tick": 3})
    next(gen)
    stale = world(target={"cursor_id": 5, "cursor_flag": 1, "target_type": 0})
    for _ in range(12):                          # the whole bounded wait, never targeting
        assert gen.send(stale) is None
    with pytest.raises(StopIteration) as done:
        gen.send(stale)
    assert done.value.value == "no potion cursor"


def test_explosion_potion_accepts_boolean_only_cursor_from_older_bridges():
    gen = explosion_potion_proc(101, 2)(world(), {"tick": 3})
    next(gen)
    assert gen.send(world(target=True)) == {"type": "TargetObject", "serial": 2}
