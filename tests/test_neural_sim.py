"""Observable-contract and environment invariants, not a ServUO parity claim."""

from dataclasses import fields, replace

import pytest

from anima3.neural.schema import (
    ACTION_INDEX,
    ACTION_NAMES,
    FEATURE_NAMES,
    SCHEMA_FINGERPRINT,
    CombatView,
    encode,
    legal_mask,
)
from anima3.neural.sim import DuelSim, cast_seconds


def supplied(**changes):
    return replace(CombatView(
        opponent_visible=True, opponent_health_known=True, opponent_hp=100,
        line_of_sight=True, distance=8, dx=8, heal_potions=20,
        cure_potions=20, explosion_potions=20, reagent_counts=(100,) * 8,
        move_legal=(True,) * 4,
    ), **changes)


def action(name):
    return ACTION_INDEX[name]


def hold_until(sim, seconds):
    result = None
    while sim.elapsed < seconds and not sim.done:
        result = sim.step((0, 0))
    return result


def test_schema_is_public_bounded_and_unknown_health_does_not_leak():
    view = supplied(opponent_health_known=False, opponent_hp=10, dx=-100, dy=100)
    frame = encode(view)
    assert len(frame.features) == len(FEATURE_NAMES) == 65
    assert len(frame.mask) == len(ACTION_NAMES) == 23
    assert len(SCHEMA_FINGERPRINT) == 64
    assert all(-1 <= value <= 1 for value in frame.features)
    assert frame == encode(replace(view, opponent_hp=90, opponent_hp_max=150))
    assert not any("opponent_mana" in field.name or "opponent_reagent" in field.name
                   for field in fields(CombatView))


def test_masks_share_resource_cursor_showdown_and_blast_safety_rules():
    view = supplied(hp=60, poisoned=True)
    mask = legal_mask(view)
    assert mask[action("cure")] and mask[action("cure_potion")]
    assert not mask[action("heal")] and not mask[action("heal_potion")]
    mask = legal_mask(replace(view, poisoned=False, showdown=True))
    assert not mask[action("greater_heal")] and not mask[action("heal_potion")]
    assert not legal_mask(supplied(distance=2))[action("explosion_potion")]
    assert not legal_mask(supplied(mana=50))[action("burst_bolt")]
    assert legal_mask(supplied(mana=51))[action("burst_bolt")]
    for blocking in ({"busy_remaining": 1}, {"cast_remaining": 0.1}, {"potion_fuse": 0.1},
                     {"fresh": False}, {"ready": False}, {"hp": 0}):
        assert legal_mask(supplied(**blocking)) == [True] + [False] * 22
    empty = legal_mask(supplied(reagent_counts=(0,) * 8, mana=0))
    assert not any(empty[5:17])
    assert not empty[action("burst_bolt")]


def test_simulator_seeded_teacher_trajectory_is_reproducible():
    a, b = DuelSim(seed=71, max_steps=140), DuelSim(seed=71, max_steps=140)
    assert a.reset() == b.reset()
    while not a.done:
        actions_a = a.teacher(0), a.teacher(1)
        actions_b = b.teacher(0), b.teacher(1)
        assert actions_a == actions_b
        assert a.step(actions_a) == b.step(actions_b)
    assert a.trace == b.trace


def test_opponent_private_mana_and_supplies_never_change_our_frame_or_teacher():
    sim = DuelSim(seed=6)
    frame, teacher = sim.frames()[0], sim.teacher(0)
    sim.fighters[1].mana = 0
    sim.fighters[1].reagents = [0] * 8
    sim.fighters[1].explosion_potions = 0
    sim.fighters[1].heal_potion_until = 900
    assert sim.frames()[0] == frame
    assert sim.teacher(0) == teacher


def test_pre_aos_cast_and_damage_delay_are_separate():
    assert cast_seconds("weaken") == pytest.approx(0.4)
    assert cast_seconds("energy_bolt") == pytest.approx(1.65)
    assert cast_seconds("flamestrike") == pytest.approx(1.9)
    sim = DuelSim(domain_randomization=False)
    sim.step((action("energy_bolt"), 0))
    hold_until(sim, 1.5)
    assert sim.fighters[0].mana == 100
    assert sim.fighters[1].hp == 100
    hold_until(sim, 2.0)
    assert sim.fighters[0].mana < 81
    assert sim.fighters[1].hp == 100
    hold_until(sim, 2.25)
    assert sim.fighters[1].hp < 85


def test_quick_damage_interrupts_long_cast_without_spending_its_mana():
    sim = DuelSim(domain_randomization=False)
    sim.step((action("explosion"), action("magic_arrow")))
    hold_until(sim, 5.0)
    assert sim.fighters[0].interrupted == 1
    assert sim.fighters[0].mana == 100
    assert sim.fighters[1].hp == 100
    assert not any(event["type"] == "damage" and event["side"] == 0 for event in sim.trace)


def test_damage_fatigue_matches_recorded_89hp_14stamina_40damage_before_hp_reduction():
    # Authoritative replay b470dc944875413194d96fcc97b8e079, t=13.923s.
    sim = DuelSim(domain_randomization=False)
    victim = sim.fighters[1]
    victim.hp = victim.hp_max = 89
    victim.stamina = victim.stamina_max = 14
    sim._hurt(0, 1, 40, "energy_bolt")
    assert victim.hp == 49
    assert victim.stamina == 0
    assert not any(sim.frames()[1].mask[1:5])


@pytest.mark.parametrize(
    "hp, hp_max, stamina, damage, expected_stamina",
    [
        (100, 100, 25, 10, 20),  # Direct subtraction above the low-stamina boundary.
        (60, 100, 25, 20, 16),  # 100/60 is integer 1; fatigue 15 * prehit .6 = 9.
        (59, 100, 25, 20, 17),  # 15 * .59 = 8.85 truncates to 8, not 9.
        (100, 100, 5, 10, 5),  # Current/max stamina ratio gives negative fatigue.
        (100, 100, 0, 40, 0),  # No division by current stamina and no negative loss.
    ],
)
def test_damage_fatigue_uses_source_integer_ratio_and_truncation(
    hp, hp_max, stamina, damage, expected_stamina,
):
    sim = DuelSim(domain_randomization=False)
    victim = sim.fighters[1]
    victim.hp, victim.hp_max, victim.stamina = hp, hp_max, stamina
    sim._hurt(0, 1, damage, "energy_bolt")
    assert victim.hp == hp - damage
    assert victim.stamina == expected_stamina


def test_lethal_overkill_skips_fatigue_but_exact_zero_hp_preserves_source_branch():
    sim = DuelSim(domain_randomization=False)
    victim = sim.fighters[1]
    victim.hp = 40
    sim._hurt(0, 1, 41, "energy_bolt")
    assert victim.hp == 0
    assert victim.stamina == 25

    sim = DuelSim(domain_randomization=False)
    victim = sim.fighters[1]
    victim.hp = 40
    sim._hurt(0, 1, 40, "energy_bolt")
    assert victim.hp == 0
    assert victim.stamina == 0


@pytest.mark.parametrize("spell", ["weaken", "clumsy"])
def test_stat_debuff_interrupts_burst_before_prime_without_counting_as_damage(spell):
    sim = DuelSim(domain_randomization=False)
    sim.step((action("burst_bolt"), action(spell)))
    hold_until(sim, 2.0)
    assert sim.fighters[0].interrupted == 1
    assert sim.fighters[0].mana == 100
    assert sim.fighters[0].active_potion is None
    assert sim.fighters[0].explosion_potions == 5
    assert sim.fighters[0].damage_received == 0
    assert sim.fighters[1].damage == 0
    interrupts = [event for event in sim.trace if event["type"] == "interrupt"]
    assert len(interrupts) == 1 and interrupts[0]["cause"] == spell
    assert interrupts[0]["t"] == pytest.approx(0.45)


@pytest.mark.parametrize("spell", ["weaken", "clumsy", "poison"])
def test_control_target_breaks_existing_paralysis_and_cast_without_hp_damage(spell):
    sim = DuelSim(domain_randomization=False)
    sim.step((action("explosion"), 0))
    # A cast may already be in progress when another spell paralyzes its caster.
    sim.fighters[0].paralyzed_until = 20
    sim.step((0, action(spell)))
    hold_until(sim, 1.5)
    assert sim.fighters[0].paralyzed_until == 0
    assert sim.fighters[0].interrupted == 1
    assert sim.fighters[0].damage_received == 0
    assert sim.fighters[1].damage == 0


@pytest.mark.parametrize("spell", ["weaken", "clumsy"])
def test_equal_strength_reapplied_stat_curse_still_interrupts(spell):
    sim = DuelSim(domain_randomization=False)
    sim.step((0, action(spell)))
    hold_until(sim, 1.25)
    sim.step((action("energy_bolt"), action(spell)))
    hold_until(sim, 3.5)
    assert sim.fighters[0].interrupted == 1
    assert sim.fighters[0].damage == 0
    assert sim.fighters[1].damage == 0


def test_poison_interrupts_primed_burst_immediately_before_first_poison_tick():
    sim = DuelSim(domain_randomization=False)
    sim.step((action("burst_bolt"), action("poison")))
    hold_until(sim, 1.25)
    assert sim.fighters[0].poison_until > sim.elapsed
    assert sim.fighters[0].interrupted == 1
    assert sim.fighters[0].damage_received == 0
    assert sim.fighters[0].thrown == 1
    assert sim.fighters[0].active_potion is None
    assert any(event["type"] == "interrupt" and event["cause"] == "poison" for event in sim.trace)


@pytest.mark.parametrize("spell", ["paralyze", "cure"])
def test_paralyze_and_beneficial_cure_do_not_interrupt_existing_enemy_cast(spell):
    sim = DuelSim(domain_randomization=False)
    if spell == "cure":
        sim.fighters[1].poison_until = 30
    sim.step((action("explosion"), action(spell)))
    hold_until(sim, 4.5)
    assert sim.fighters[0].interrupted == 0
    assert sim.fighters[0].damage > 0
    if spell == "paralyze":
        assert sim.fighters[0].paralyzed_until > sim.elapsed
    else:
        assert sim.fighters[1].poison_until == 0


def test_potion_fuse_landing_and_damage_are_not_instant():
    sim = DuelSim(domain_randomization=False)
    sim.step((action("explosion_potion"), 0))
    hold_until(sim, 3.5)
    assert sim.fighters[1].hp == 100
    assert sim.fighters[0].thrown == 1
    hold_until(sim, 3.75)
    assert sim.fighters[1].hp <= 85
    assert sim.fighters[0].self_damage == 0
    assert sim.fighters[0].explosion_potions == 4
    events = [event["type"] for event in sim.trace if event["type"].startswith("potion_")]
    assert events == ["potion_prime", "potion_throw", "potion_explode"]


@pytest.mark.parametrize("name", ["heal_potion", "cure_potion", "explosion_potion"])
def test_arena_stock_is_five_each_and_exhaustion_disables_an_otherwise_legal_potion(name):
    sim = DuelSim(domain_randomization=False)
    assert all(
        (fighter.heal_potions, fighter.cure_potions, fighter.explosion_potions) == (5, 5, 5)
        for fighter in sim.fighters
    )
    # Keep the target alive while independently exercising five explosion uses.
    sim.fighters[1].hp = sim.fighters[1].hp_max = 1000
    stock_field = name + "s"
    for remaining in range(4, -1, -1):
        sim.fighters[0].hp = 30
        sim.fighters[0].poison_until = sim.elapsed + 30 if name == "cure_potion" else 0
        assert sim.frames()[0].mask[action(name)]
        _, _, _, info = sim.step((action(name), 0))
        assert not info["invalid_actions"][0]
        assert getattr(sim.fighters[0], stock_field) == remaining
        hold_until(sim, sim.elapsed + 10)
    sim.fighters[0].hp = 30
    sim.fighters[0].poison_until = sim.elapsed + 30 if name == "cure_potion" else 0
    assert not sim.frames()[0].mask[action(name)]
    _, _, _, info = sim.step((action(name), 0))
    assert info["invalid_actions"][0]
    assert getattr(sim.fighters[0], stock_field) == 0


def test_burst_option_clusters_delayed_spell_potion_and_finisher_damage():
    sim = DuelSim(domain_randomization=False)
    sim.step((action("burst_bolt"), 0))
    hold_until(sim, 6)
    hits = [event for event in sim.trace if event["type"] == "damage"]
    assert {hit["damage_kind"] for hit in hits} == {"explosion", "energy_bolt", "explosion_potion"}
    assert max(hit["t"] for hit in hits) - min(hit["t"] for hit in hits) <= 0.3
    assert sim.fighters[0].mana < 70
    assert sim.fighters[0].self_damage == 0
    assert sim.fighters[0].reagents[0] == 99
    assert sim.fighters[0].reagents[1] == 99


def test_interrupted_burst_disposes_primed_potion_and_abandons_finisher():
    sim = DuelSim(domain_randomization=False)
    sim.step((action("burst_bolt"), action("magic_arrow")))
    hold_until(sim, 6)
    assert sim.fighters[0].interrupted == 1
    assert sim.fighters[0].active_potion is None
    assert sim.fighters[0].self_damage == 0
    casts = [event["spell"] for event in sim.trace if event["type"] == "cast" and event["side"] == 0]
    assert casts == ["explosion"]
    assert any(event["type"] == "potion_throw" and event["t"] < 1 for event in sim.trace)


def test_walking_into_own_potion_blast_is_penalized_not_rewarded():
    sim = DuelSim(domain_randomization=False)
    sim.fighters[0].x, sim.fighters[1].x = 0, 3
    sim.step((action("explosion_potion"), 0))
    hold_until(sim, 2.25)
    sim.step((action("approach"), 0))
    hold_until(sim, 3.5)
    _, reward, _, info = sim.step((0, 0))
    assert info["self_damage"][0] > 0
    assert reward[0] < 0


def test_showdown_blocks_pending_heals_and_timeout_is_draw():
    sim = DuelSim(domain_randomization=False, max_steps=4, showdown_after=0.25)
    sim.fighters[0].hp = sim.fighters[1].hp = 50
    sim.step((action("heal"), action("heal")))
    assert not sim.frames()[0].mask[action("heal")]
    _, _, done, info = hold_until(sim, 1)
    assert done and info["terminated"] and not info["truncated"]
    assert info["time_limit"]
    assert info["winner"] is None
    assert info["healing"] == [0, 0]
    with pytest.raises(RuntimeError, match="reset"):
        sim.step((0, 0))


def test_health_shaping_cannot_farm_discounted_reward_by_repeated_healing():
    sim = DuelSim(domain_randomization=False, max_steps=60)
    sim.fighters[0].hp = sim.fighters[1].hp = 30
    discounted = 0.0
    while not sim.done:
        choice = action("heal") if sim.frames()[0].mask[action("heal")] else 0
        _, reward, _, _ = sim.step((choice, 0))
        discounted += sim.shaping_gamma ** (sim.steps - 1) * reward[0]
    assert sim.fighters[0].healing > 0
    assert discounted == pytest.approx(0, abs=1e-10)


def test_teacher_reaches_terminal_outcome_with_only_legal_actions():
    sim = DuelSim(seed=12, max_steps=600, showdown_after=60)
    while not sim.done:
        choices = sim.teacher(0), sim.teacher(1)
        for side in range(2):
            assert sim.frames()[side].mask[choices[side]]
        _, rewards, _, info = sim.step(choices)
    assert info["terminated"]
    assert all(value > 0 for value in info["damage"])
    if info["winner"] is not None:
        assert rewards[info["winner"]] >= 0.95
        assert rewards[1 - info["winner"]] <= -0.95
    assert info["source"] == "approximate-pre-aos-simulator"
