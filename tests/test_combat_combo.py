from types import SimpleNamespace

import pytest

from anima3 import combat_combo
from anima3.contract import Observation
from anima3.magic import REAGENT_GRAPHICS, mage_verbs


def world(distance=6, hp=100, mana=100, target=None):
    return Observation.from_json(
        {
            "player": {
                "serial": 1,
                "hits": hp,
                "hits_max": 100,
                "mana": mana,
                "mana_max": 100,
                "pos": {"x": 0, "y": 0, "z": 0},
            },
            "mobiles": [
                {"serial": 2, "distance": distance, "pos": {"x": distance, "y": 0, "z": 0}}
            ],
            "pending_target": target,
            "items": [
                {"serial": 100, "graphic": 0xE75, "container": 1, "layer": 21},
                {"serial": 101, "graphic": 0xF0D, "container": 100},
            ]
            + [
                {"serial": 200 + i, "graphic": g, "container": 100}
                for i, g in enumerate(REAGENT_GRAPHICS)
            ],
        }
    )


def menu(obs, memory):
    return mage_verbs(obs, SimpleNamespace(hp_pct=obs.player.hp_pct), memory, obs.mobiles[0])


def test_weaken_then_clumsy_before_potions_and_no_repeating_opening():
    obs = world()
    memory = {"tick": 1, "opening": "weaken-clumsy", "explosion_potions": True}
    a = menu(obs, memory)[0]
    assert a.id == "cast:weaken"
    gen = a.procedure(obs, memory)
    assert next(gen) == {"type": "CastSpell", "spell": 8}
    obs = world(target={"cursor_id": 1, "cursor_flag": 1, "target_type": 0})
    assert gen.send(obs) == {"type": "TargetObject", "serial": 2}
    with pytest.raises(StopIteration):
        gen.send(world(mana=96))
    memory["tick"] = 10
    assert menu(world(), memory)[0].id == "cast:clumsy"
    memory["opening_finished"].append("clumsy")
    assert menu(world(), memory)[0].id == "potion:explosion"


def test_low_health_skips_opener_and_large_arena_approaches_first():
    memory = {"opening": "weaken-clumsy", "explosion_potions": True}
    assert menu(world(hp=25), memory)[0].id == "cast:greater_heal"
    assert memory["opening_abandoned"]
    assert menu(world(distance=19), {"opening": "weaken-clumsy"})[0].id == "approach:duel"


def test_cursor_metadata_is_preserved_and_boolean_only_is_not_trusted():
    obs = world(target={"cursor_id": 9, "cursor_flag": 1, "target_type": 0})
    assert obs.target_cursor_id == 9 and combat_combo.cursor(obs, 1, 0)
    assert not combat_combo.cursor(world(target=True), 1, 0)


def test_burst_orders_distinct_cursors_and_finisher(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(combat_combo.time, "monotonic", lambda: now[0])
    memory = {"tick": 1}
    gen = combat_combo.burst_proc(101, 2)(world(), memory)
    action = next(gen)
    mana = 100
    pending = None
    explosion_at = None
    finisher_at = None
    released = False
    thrown = False
    actions = []
    spell_id = 10
    for _ in range(120):
        if action:
            actions.append((round(now[0], 2), action))
            if action["type"] == "CastSpell":
                if action["spell"] == 43:
                    explosion_at = now[0]
                else:
                    finisher_at = now[0]
            elif action["type"] == "Use":
                pending = {"cursor_id": 20, "cursor_flag": 0, "target_type": 1}
            elif action["type"] == "TargetObject":
                assert pending is not None
                if pending["cursor_flag"] == 0:
                    thrown = True
                elif not released:
                    released = True
                    mana -= 20
                else:
                    mana -= 20
                pending = None
        now[0] += 0.1
        memory["tick"] += 1
        if explosion_at is not None and now[0] - explosion_at >= 2.15 and not released:
            pending = {"cursor_id": spell_id, "cursor_flag": 1, "target_type": 0}
        if finisher_at is not None and now[0] - finisher_at >= 2.15 and mana == 80:
            pending = {"cursor_id": 30, "cursor_flag": 1, "target_type": 0}
        try:
            action = gen.send(world(mana=mana, target=pending))
        except StopIteration as result:
            assert result.value == "ok"
            break
    else:
        pytest.fail("burst did not terminate")
    assert released and thrown
    stages = [e["stage"] for e in memory["combo_events"]]
    assert (
        stages.index("potion_prime")
        < stages.index("explosion_target")
        < stages.index("potion_throw_target")
        < stages.index("finisher_energy_bolt")
    )
    spell_actions = [a["spell"] for _, a in actions if a["type"] == "CastSpell"]
    assert spell_actions == [43, 42]


@pytest.mark.parametrize("reason", ["danger", "fizzle"])
def test_combo_aborts_before_priming_on_emergency_or_fizzle(monkeypatch, reason):
    from anima3.contract import Journal

    monkeypatch.setattr(combat_combo.time, "monotonic", lambda: 0.0)
    memory = {}
    gen = combat_combo.burst_proc(101, 2)(world(), memory)
    assert next(gen)["type"] == "CastSpell"
    obs = world(hp=30 if reason == "danger" else 100)
    if reason == "fizzle":
        obs.new_journal = [Journal(1, "System", "fizzle", 0, 0, 502632)]
    with pytest.raises(StopIteration, match="burst not primed"):
        gen.send(obs)
    assert [e["stage"] for e in memory["combo_events"]] == ["explosion_cast", "abort_before_prime"]


def test_combo_missing_spell_cursor_throws_potion_without_finisher(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(combat_combo.time, "monotonic", lambda: now[0])
    memory = {}
    gen = combat_combo.burst_proc(101, 2)(world(), memory)
    action = next(gen)
    actions = []
    pending = None
    for _ in range(60):
        if action:
            actions.append(action)
            if action["type"] == "Use":
                pending = {"cursor_id": 20, "cursor_flag": 0, "target_type": 1}
            if action["type"] == "TargetObject":
                pending = None
        now[0] += 0.1
        try:
            action = gen.send(world(target=pending))
        except StopIteration:
            break
    else:
        pytest.fail("missing cursor did not terminate")
    assert [a["spell"] for a in actions if a["type"] == "CastSpell"] == [43]
    assert {"type": "TargetObject", "serial": 2} in actions


def test_burst_metrics_measure_observed_hits_and_potion_timeline():
    from anima3.replay import burst_metrics
    rows = [
        {"type": "header", "players": [{"serial": 1}]},
        {"type": "cast", "actor": 1, "t": 0, "name": "ExplosionSpell"},
        {"type": "potion_state", "actor": 1, "item": 10, "phase": "prime", "t": 100},
        {"type": "potion_state", "actor": 1, "item": 10, "phase": "throw", "t": 3000},
        {"type": "potion_state", "actor": 1, "item": 10, "phase": "explode", "t": 3800},
        {"type": "damage", "actor": 1, "amount": 30, "t": 3800},
        {"type": "damage", "actor": 1, "amount": 25, "t": 4000},
        {"type": "damage", "actor": 1, "amount": 40, "t": 6000},
    ]
    result = burst_metrics(rows)["1"]
    assert result["peakOneSecondDamage"] == 55
    assert result["potionCycles"] == [{"item": 10, "primeMs": 100, "throwMs": 3000, "explodeMs": 3800}]
