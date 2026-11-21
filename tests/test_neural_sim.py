"""Row 7b simulator, drills and simulator evaluation through the unit-7 gate; no torch.

The simulator tests are ported from ``tests/test_neural_sim.py`` and
``tests/test_training_league.py`` at the training tip ``819f4f4``, with each
``DuelSim`` given an explicit seed and episode budget. Source equivalence is
checked three ways: every ported definition is AST-identical to the source except
the constructor, a seeded driver gives byte-identical output from this port and from
the source tree exported with ``git archive`` into a temporary directory, and that
driver's digest is pinned so a clone without the source commit (CI's shallow
checkout) still checks it.
"""

from __future__ import annotations

import ast
import copy
import json
import random
import subprocess
import sys
import textwrap
from dataclasses import fields, replace
from pathlib import Path

import pytest
from test_neural_runtime import executor, observation

from anima3.contract import Journal, Skill
from anima3.neural import drills, evaluation, sim_evaluation
from anima3.neural.drills import DRILLS, SIM_DOMAIN, SIMULATOR_SHA256, setup_drill
from anima3.neural.evaluation import (
    DOMAINS,
    EvaluationPlan,
    EvaluationRefused,
    decide_from_collections,
    record_decision,
    selection_gate,
)
from anima3.neural.schema import (
    ACTION_INDEX,
    ACTION_NAMES,
    FEATURE_NAMES,
    SCHEMA_FINGERPRINT,
    CombatView,
    encode,
    legal_mask,
)
from anima3.neural.sim import SHAPING_GAMMA, DuelSim, cast_seconds
from anima3.neural.sim_evaluation import (
    EVALUATION_SEED_FLOOR,
    SIM_DECISION_SCHEMA,
    SimulatorEvaluationPlan,
    SimulatorRefused,
    simulator_gate,
    simulator_record,
    teacher_opponent,
)
from anima3.neural.teacher import STYLES, TEACHER_SHA256, TEACHER_STYLES, TacticalTeacher

REPO = Path(__file__).resolve().parents[1]
SOURCE = "819f4f426bfaf72e1a132abff9354480250f1444"
CANDIDATE, INCUMBENT = "ca" * 32, "1c" * 32

# --- ported source cases ---------------------------------------------------------------------


def supplied(**changes):
    return replace(
        CombatView(
            opponent_visible=True,
            opponent_health_known=True,
            opponent_hp=100,
            line_of_sight=True,
            distance=8,
            dx=8,
            heal_potions=20,
            cure_potions=20,
            explosion_potions=20,
            reagent_counts=(100,) * 8,
            move_legal=(True,) * 4,
        ),
        **changes,
    )


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
    assert len(frame.features) == len(FEATURE_NAMES) == 76
    assert len(frame.mask) == len(ACTION_NAMES) == 27
    assert len(SCHEMA_FINGERPRINT) == 64
    assert all(-1 <= value <= 1 for value in frame.features)
    assert frame == encode(replace(view, opponent_hp=90, opponent_hp_max=150))
    assert not any(
        "opponent_mana" in field.name or "opponent_reagent" in field.name
        for field in fields(CombatView)
    )


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
    for blocking in (
        {"busy_remaining": 1},
        {"cast_remaining": 0.1},
        {"potion_fuse": 0.1},
        {"fresh": False},
        {"ready": False},
        {"hp": 0},
    ):
        assert legal_mask(supplied(**blocking)) == [True] + [False] * (len(ACTION_NAMES) - 1)
    empty = legal_mask(supplied(reagent_counts=(0,) * 8, mana=0))
    assert not any(empty[5:17])
    assert not empty[action("burst_bolt")]


def test_simulator_seeded_teacher_trajectory_is_reproducible():
    a, b = (
        DuelSim(seed=71, max_steps=140, shaping_gamma=SHAPING_GAMMA),
        DuelSim(seed=71, max_steps=140, shaping_gamma=SHAPING_GAMMA),
    )
    assert a.reset() == b.reset()
    while not a.done:
        actions_a = a.teacher(0), a.teacher(1)
        actions_b = b.teacher(0), b.teacher(1)
        assert actions_a == actions_b
        assert a.step(actions_a) == b.step(actions_b)
    assert a.trace == b.trace


def test_opponent_private_mana_and_supplies_never_change_our_frame_or_teacher():
    sim = DuelSim(max_steps=1200, seed=6, shaping_gamma=SHAPING_GAMMA)
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
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
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
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    sim.step((action("explosion"), action("magic_arrow")))
    hold_until(sim, 5.0)
    assert sim.fighters[0].interrupted == 1
    assert sim.fighters[0].mana == 100
    assert sim.fighters[1].hp == 100
    assert not any(event["type"] == "damage" and event["side"] == 0 for event in sim.trace)


def test_damage_fatigue_matches_recorded_89hp_14stamina_40damage_before_hp_reduction():
    # Authoritative replay b470dc944875413194d96fcc97b8e079, t=13.923s.
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
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
    hp,
    hp_max,
    stamina,
    damage,
    expected_stamina,
):
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    victim = sim.fighters[1]
    victim.hp, victim.hp_max, victim.stamina = hp, hp_max, stamina
    sim._hurt(0, 1, damage, "energy_bolt")
    assert victim.hp == hp - damage
    assert victim.stamina == expected_stamina


def test_lethal_overkill_skips_fatigue_but_exact_zero_hp_preserves_source_branch():
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    victim = sim.fighters[1]
    victim.hp = 40
    sim._hurt(0, 1, 41, "energy_bolt")
    assert victim.hp == 0
    assert victim.stamina == 25

    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    victim = sim.fighters[1]
    victim.hp = 40
    sim._hurt(0, 1, 40, "energy_bolt")
    assert victim.hp == 0
    assert victim.stamina == 0


@pytest.mark.parametrize("spell", ["weaken", "clumsy"])
def test_stat_debuff_interrupts_burst_before_prime_without_counting_as_damage(spell):
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
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
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
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
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    sim.step((0, action(spell)))
    hold_until(sim, 1.25)
    sim.step((action("energy_bolt"), action(spell)))
    hold_until(sim, 3.5)
    assert sim.fighters[0].interrupted == 1
    assert sim.fighters[0].damage == 0
    assert sim.fighters[1].damage == 0


def test_poison_interrupts_primed_burst_immediately_before_first_poison_tick():
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
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
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    if spell == "cure":
        sim.fighters[1].poison_until = 30
    sim.step((action("explosion"), action(spell)))
    hold_until(sim, 5.5)
    assert sim.fighters[0].interrupted == 0
    assert sim.fighters[0].damage > 0
    if spell == "paralyze":
        assert sim.fighters[0].paralyzed_until > sim.elapsed
    else:
        assert sim.fighters[1].poison_until == 0


def test_potion_fuse_landing_and_damage_are_not_instant():
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
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
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
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
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
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
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    sim.step((action("burst_bolt"), action("magic_arrow")))
    hold_until(sim, 6)
    assert sim.fighters[0].interrupted == 1
    assert sim.fighters[0].active_potion is None
    assert sim.fighters[0].self_damage == 0
    casts = [
        event["spell"] for event in sim.trace if event["type"] == "cast" and event["side"] == 0
    ]
    assert casts == ["explosion"]
    assert any(event["type"] == "potion_throw" and event["t"] < 1 for event in sim.trace)


def test_walking_into_own_potion_blast_is_penalized_not_rewarded():
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    sim.fighters[0].x, sim.fighters[1].x = 0, 3
    sim.step((action("explosion_potion"), 0))
    hold_until(sim, 2.25)
    sim.step((action("approach"), 0))
    hold_until(sim, 3.5)
    _, reward, _, info = sim.step((0, 0))
    assert info["self_damage"][0] > 0
    assert reward[0] < 0


def test_showdown_blocks_pending_heals_and_timeout_is_draw():
    sim = DuelSim(
        seed=0,
        domain_randomization=False,
        max_steps=4,
        showdown_after=0.25,
        shaping_gamma=SHAPING_GAMMA,
    )
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
    sim = DuelSim(seed=0, domain_randomization=False, max_steps=60, shaping_gamma=SHAPING_GAMMA)
    sim.fighters[0].hp = sim.fighters[1].hp = 30
    discounted = 0.0
    while not sim.done:
        choice = action("heal") if sim.frames()[0].mask[action("heal")] else 0
        _, reward, _, _ = sim.step((choice, 0))
        discounted += sim.shaping_gamma ** (sim.steps - 1) * reward[0]
    assert sim.fighters[0].healing > 0
    assert discounted == pytest.approx(0, abs=1e-10)


def test_teacher_reaches_terminal_outcome_with_only_legal_actions():
    sim = DuelSim(seed=12, max_steps=600, showdown_after=60, shaping_gamma=SHAPING_GAMMA)
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


def test_meditation_obeys_server_skill_clock_and_continues_past_eight_seconds():
    sim = DuelSim(max_steps=1200, seed=7, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    sim.fighters[0].mana = 60
    sim.step((action("meditate"), 0))
    assert sim.fighters[0].meditation_until > 10
    hold_until(sim, 9)
    assert not sim.frames()[0].mask[action("meditate")]
    assert sim.fighters[0].mana >= 78
    hold_until(sim, 10)
    assert sim.frames()[0].mask[action("meditate")]
    sim.step((action("approach"), 0))
    assert sim.fighters[0].meditation_until == 0


def test_meditation_can_fail_at_empty_mana_and_still_has_ten_second_cooldown():
    sim = DuelSim(max_steps=1200, seed=0, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    sim.fighters[0].mana = 0
    sim.rng.random = lambda: 0.99
    sim.step((action("meditate"), 0))
    assert sim.fighters[0].meditation_until == 0
    hold_until(sim, 3)
    assert not sim.frames()[0].mask[action("meditate")]


def test_public_explosion_estimate_survives_followup_and_expires():
    sim = DuelSim(max_steps=1200, seed=1, shaping_gamma=SHAPING_GAMMA)
    sim._start_cast(1, "explosion")
    expected = cast_seconds("explosion") + 2.5
    sim.elapsed = 1.0
    sim._start_cast(1, "energy_bolt")
    assert sim.view(0).opponent_cast == "energy_bolt"
    assert sim.view(0).incoming_explosion_seconds == pytest.approx(expected - 1)
    # It remains an audible-word estimate, even with no actual queued damage.
    sim.events.clear()
    assert sim.view(0).incoming_explosion_seconds == pytest.approx(expected - 1)
    sim.elapsed = expected + 0.01
    assert sim.view(0).incoming_explosion_seconds == -1
    sim.reset()
    assert sim.view(0).incoming_explosion_seconds == -1


def test_stun_needs_resources_skills_and_ack_before_next_toggle():
    obs = observation(enemy_x=101)
    obs.skills = [Skill(1, 100, 100, 100, 2), Skill(43, 100, 100, 100, 2)]
    ex = executor(obs)
    assert ex.start("stun", obs, 0)
    assert ex.tick(obs, 0) == [{"type": "StunRequest"}, {"type": "Attack", "serial": 2}]
    ex.tick(obs, 0.3)
    obs2 = replace(obs, new_journal=[Journal(1, "", "", 0, 0, 1019011)])
    ex.observe(obs2, {"phase": "Fighting", "opponent": 2}, 2.1)
    assert ex.view.stun_armed and not ex.view.stun_pending
    assert not encode(ex.view).mask[ACTION_INDEX["stun"]]
    obs3 = replace(obs, new_journal=[Journal(1, "", "", 0, 0, 1004010)])
    ex.observe(obs3, {"phase": "Fighting", "opponent": 2}, 3)
    assert ex.view.stun_armed and ex.view.stun_cooldown == 10
    obs4 = replace(obs, new_journal=[Journal(1, "", "", 0, 0, 1004013)])
    ex.observe(obs4, {"phase": "Fighting", "opponent": 2}, 13)
    assert not ex.view.stun_armed
    assert not encode(ex.view).mask[ACTION_INDEX["stun"]]
    frame = ex.observe(obs, {"phase": "Fighting", "opponent": 2}, 23.1)
    assert frame.mask[ACTION_INDEX["stun"]]


def test_unknown_stun_skills_and_stale_state_fail_closed():
    v = CombatView(
        hands_free=True,
        opponent_visible=True,
        line_of_sight=True,
        distance=1,
        stamina=14,
        stun_skilled=True,
    )
    assert not encode(v).mask[ACTION_INDEX["stun"]]
    assert not encode(replace(v, stamina=25, stun_skilled=False)).mask[ACTION_INDEX["stun"]]
    assert encode(replace(v, stamina=25)).mask[ACTION_INDEX["stun"]]
    assert sum(encode(replace(v, fresh=False)).mask) == 1


def test_sim_stun_attempt_costs_and_freeze_survives_damage():
    sim = DuelSim(max_steps=1200, seed=4, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    sim.fighters[0].x, sim.fighters[1].x = 0, 1
    sim._perform(0, "stun")
    sim.step((0, 0))
    event = next(r for r in sim.trace if r["type"] == "stun_attempt")
    assert event["success"]
    assert 10 <= sim.fighters[0].stamina < 10.1
    assert sim.fighters[0].stun_until == pytest.approx(10.05)
    assert sim.view(1).paralyzed
    sim._hurt(0, 1, 1, "harm")
    assert sim.view(1).paralyzed  # Freeze differs from Paralyze spell


def test_first_circle_damage_does_not_disturb_but_second_does():
    sim = DuelSim(seed=0, max_steps=1200, domain_randomization=False, shaping_gamma=SHAPING_GAMMA)
    sim._start_cast(0, "magic_arrow")
    sim._hurt(1, 0, 5, "punch")
    assert sim.fighters[0].interrupted == 0
    sim._start_cast(0, "harm")
    sim._hurt(1, 0, 5, "punch")
    assert sim.fighters[0].interrupted == 1


@pytest.mark.parametrize("drill", DRILLS)
def test_drills_reseed_and_teachers_only_choose_legal_actions(drill):
    sim = DuelSim(seed=17, max_steps=100, shaping_gamma=SHAPING_GAMMA)
    frames = setup_drill(sim, drill, 0)
    assert sim._potential_previous == sim._potential()
    teachers = [TacticalTeacher("pressure"), TacticalTeacher("defensive")]
    for _ in range(100):
        actions = [teachers[i].choose(sim.view(i)) for i in (0, 1)]
        assert all(frames[i].mask[actions[i]] for i in (0, 1))
        frames, _, done, info = sim.step(actions)
        assert not any(info["invalid_actions"])
        if done:
            break


# --- equivalence with the source tree ---------------------------------------------------------


def source_available() -> bool:
    found = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "-e", f"{SOURCE}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return found.returncode == 0


def source_text(path: str) -> str:
    if not source_available():
        pytest.skip(f"{SOURCE[:7]} is not in this clone; the pinned digest is checked instead")
    shown = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{SOURCE}:{path}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return shown.stdout


def export_source(tmp_path_factory) -> Path:
    """The source ``anima3`` package at ``819f4f4``, exported into a temporary directory."""
    if not source_available():
        pytest.skip(f"{SOURCE[:7]} is not in this clone; the pinned digest is checked instead")
    root = tmp_path_factory.mktemp("source-819f4f4")
    archive = subprocess.run(
        ["git", "-C", str(REPO), "archive", SOURCE, "anima3"], capture_output=True, check=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
    return root


@pytest.fixture(scope="session")
def source_tree(tmp_path_factory):
    return export_source(tmp_path_factory)


def run_tree(root: Path, prelude: str, body: str):
    """Run ``body`` in a fresh, isolated interpreter that imports ``anima3`` from ``root``."""
    code = f"import sys\nsys.path.insert(0, {str(root)!r})\n{prelude}\n{body}"
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=root, check=False
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


PORT_IMPORTS = """
from anima3.neural.sim import SHAPING_GAMMA, DuelSim
from anima3.neural.drills import DRILLS, setup_drill
from anima3.neural.teacher import TacticalTeacher
"""
SOURCE_IMPORTS = """
from anima3.neural.sim import SHAPING_GAMMA, DuelSim
from anima3.neural.curriculum import DRILLS, setup_drill, TacticalTeacher
"""
# Teacher-driven full games with and without domain randomisation, then every drill
# from both seats against all six teacher styles, with some random legal (and a few
# illegal) learner actions. Every frame, reward, info dictionary and trace is hashed.
DRIVER = textwrap.dedent(
    """
    import hashlib, json, random
    from dataclasses import asdict, is_dataclass

    STYLES6 = ("balanced", "pressure", "burst", "defensive", "potions", "tactical")

    def _encode(value):
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(type(value).__name__)

    def drive():
        log = []
        for seed in (0, 1, 6, 71, 2026, 999_999_937):
            for randomized in (True, False):
                sim = DuelSim(seed=seed, max_steps=1200, domain_randomization=randomized, shaping_gamma=SHAPING_GAMMA)
                log.append(sim.reset())
                while not sim.done:
                    log.append(sim.step((sim.teacher(0), sim.teacher(1))))
                log.append(sim.trace)
        picker = random.Random(5)
        for index, drill in enumerate(DRILLS):
            for side in (0, 1):
                for number, style in enumerate(STYLES6):
                    sim = DuelSim(seed=10_000 + index * 7 + side, max_steps=480, shaping_gamma=SHAPING_GAMMA)
                    log.append(setup_drill(sim, drill, side))
                    other = STYLES6[(number + 1) % 6]
                    teachers = [TacticalTeacher(style), TacticalTeacher(other)]
                    while not sim.done:
                        actions = [teachers[k].choose(sim.view(k)) for k in (0, 1)]
                        if picker.random() < 0.2:
                            mask = sim.frames()[0].mask
                            legal = [a for a, ok in enumerate(mask) if ok]
                            if picker.random() < 0.9:
                                actions[0] = picker.choice(legal)
                            else:
                                actions[0] = picker.randrange(27)
                        log.append(sim.step(tuple(actions)))
                    log.append(sim.trace)
        blob = json.dumps(log, sort_keys=True, default=_encode).encode()
        return {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}

    print(json.dumps(drive()))
    """
)
# Recorded from the source tree at 819f4f4 and reproduced by this port.
PINNED_DRIVE = {
    "sha256": "8cd6f64907d06dad52fac9efc1852f6201c8af7e576e9e4e8bca5bef6c4dfb77",
    "bytes": 55413378,
}


@pytest.fixture(scope="session")
def port_drive():
    return run_tree(REPO, PORT_IMPORTS, DRIVER)


def test_seeded_simulator_output_matches_the_pinned_source_digest(port_drive):
    assert port_drive == PINNED_DRIVE


def test_seeded_simulator_output_equals_the_source_tree(source_tree, port_drive):
    assert run_tree(source_tree, SOURCE_IMPORTS, DRIVER) == port_drive


def _definitions(tree: ast.Module) -> dict[str, str]:
    """Top-level statements (and ``DuelSim`` members) by name, as AST dumps."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # the module docstring
        if isinstance(node, ast.ClassDef) and node.name == "DuelSim":
            for member in node.body:
                name = getattr(member, "name", None) or ast.dump(member)
                found[f"DuelSim.{name}"] = ast.dump(member)
            continue
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign):
            name = ",".join(ast.unparse(t) for t in node.targets)
        found[name or ast.dump(node)] = ast.dump(node)
    return found


def test_ported_simulator_is_ast_identical_to_source_except_the_constructor():
    source = _definitions(ast.parse(source_text("anima3/neural/sim.py")))
    port = _definitions(ast.parse((REPO / "anima3/neural/sim.py").read_text()))
    assert source.keys() == port.keys()
    differing = {name for name in source if source[name] != port[name]}
    assert differing == {"DuelSim.__init__"}
    old = next(
        n
        for n in ast.walk(ast.parse(source_text("anima3/neural/sim.py")))
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    new = next(
        n
        for n in ast.walk(ast.parse((REPO / "anima3/neural/sim.py").read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    # The same parameters, minus the seed, max_steps and shaping_gamma defaults, plus
    # two checks before the unchanged source body.
    assert [a.arg for a in old.args.args] == [a.arg for a in new.args.args]
    assert [ast.dump(a) for a in old.args.kwonlyargs] == [ast.dump(a) for a in new.args.kwonlyargs]
    assert [ast.unparse(d) for d in old.args.defaults] == ["0", "1200"]
    assert new.args.defaults == []
    old_kw = [None if d is None else ast.unparse(d) for d in old.args.kw_defaults]
    new_kw = [None if d is None else ast.unparse(d) for d in new.args.kw_defaults]
    assert [a.arg for a in new.args.kwonlyargs][-1] == "shaping_gamma"
    assert old_kw == ["0.25", "180.0", "True", "SHAPING_GAMMA"]
    assert new_kw == ["0.25", "180.0", "True", None]
    assert [ast.dump(s) for s in old.body] == [ast.dump(s) for s in new.body[2:]]
    assert "must be stated as integers" in ast.unparse(new.body[0])
    assert "shaping_gamma must be stated" in ast.unparse(new.body[1])


def test_ported_drills_are_ast_identical_to_source():
    source = ast.parse(source_text("anima3/neural/curriculum.py"))
    port = ast.parse((REPO / "anima3/neural/drills.py").read_text())

    def named(tree, name):
        for node in tree.body:
            if getattr(node, "name", None) == name or (
                isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == name
            ):
                return ast.dump(node)
        raise AssertionError(name)

    for name in ("setup_drill", "DRILLS"):
        assert named(source, name) == named(port, name)


def test_simulator_constructor_requires_an_explicit_seed_budget_and_discount():
    with pytest.raises(TypeError):
        DuelSim()
    with pytest.raises(TypeError):
        DuelSim(seed=3, shaping_gamma=SHAPING_GAMMA)
    with pytest.raises(TypeError):
        DuelSim(max_steps=10, shaping_gamma=SHAPING_GAMMA)
    with pytest.raises(TypeError):
        DuelSim(3, 10)  # no default shaping discount
    for seed, steps in ((True, 10), ("3", 10), (3.0, 10), (3, 10.0), (3, None), (3, False)):
        with pytest.raises(ValueError, match="stated as integers"):
            DuelSim(seed, steps, shaping_gamma=SHAPING_GAMMA)
    for gamma in (True, None, "0.99", float("nan"), float("inf")):
        with pytest.raises(ValueError, match="shaping_gamma must be stated"):
            DuelSim(3, 10, shaping_gamma=gamma)
    for gamma in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="shaping_gamma must be in"):
            DuelSim(3, 10, shaping_gamma=gamma)
    assert DuelSim(3, 10, shaping_gamma=1).max_steps == 10


def test_simulator_identity_covers_the_simulator_drills_and_schema(tmp_path):
    from anima3.neural.teacher import source_hashes, source_identity

    assert set(drills.SIMULATOR_SOURCES) == {
        "neural/sim.py",
        "neural/drills.py",
        "neural/schema.py",
    }
    assert source_identity(drills.SIMULATOR_SOURCES) == SIMULATOR_SHA256
    package = REPO / "anima3"
    for module in drills.SIMULATOR_MODULES:
        copy_root = tmp_path / module.replace("/", "_")
        for other in drills.SIMULATOR_MODULES:
            target = copy_root / other
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((package / other).read_bytes())
        (copy_root / module).write_bytes((package / module).read_bytes() + b"\n# edited\n")
        changed = source_identity(source_hashes(drills.SIMULATOR_MODULES, root=copy_root))
        assert changed != SIMULATOR_SHA256, module


def test_simulator_results_name_the_simulator_domain():
    assert SIM_DOMAIN == "approximate-pre-aos-simulator" and SIM_DOMAIN in DOMAINS
    sim = DuelSim(5, 4, shaping_gamma=SHAPING_GAMMA)
    while not sim.done:
        *_, info = sim.step((0, 0))
    assert info["source"] == SIM_DOMAIN and info["time_limit"] and info["winner"] is None


# --- the simulator evaluation plan -----------------------------------------------------------


def live_plan(**overrides) -> EvaluationPlan:
    values = {
        "candidate_sha256": CANDIDATE,
        "incumbent_sha256": INCUMBENT,
        "opponents": ("scripted", "neural:" + "cd" * 32),
        "games_per_opponent": 20,
    }
    return EvaluationPlan(**{**values, **overrides})


def sim_plan(
    styles=STYLES,
    games=20,
    evaluation_overrides=None,
    candidate=CANDIDATE,
    incumbent=INCUMBENT,
    **overrides,
):
    evaluation_ = EvaluationPlan(
        candidate_sha256=candidate,
        incumbent_sha256=incumbent,
        opponents=tuple(teacher_opponent(s) for s in styles),
        games_per_opponent=games,
        domain=SIM_DOMAIN,
        **(evaluation_overrides or {}),
    )
    values = {
        "evaluation": evaluation_,
        "seed": 3_600_000_000,
        "max_steps": 1200,
        "sampling": "categorical",
        "simulator_sha256": SIMULATOR_SHA256,
        "teacher_sha256": TEACHER_SHA256,
    }
    return SimulatorEvaluationPlan(**{**values, **overrides})


def test_live_plan_identity_is_unchanged_by_the_simulator_labels():
    # Values computed with the unit-9 main (f281555) before this change.
    assert live_plan().sha256 == (
        "b91239f7db9cf17f1e297012ee42ee305d3591ada1ee17ea5695830bfd72c61c"
    )
    assert live_plan(
        opponents=("scripted",),
        games_per_opponent=40,
        max_regression=0.05,
        min_lower_bound=-0.1,
        max_excluded_per_opponent=2,
        min_paired_games_per_opponent=30,
    ).sha256 == ("e126a52beb880d97fcd4fd6bfde64a6c80d470375485710933cab32133ed7c26")


def test_teacher_opponents_belong_only_to_simulator_plans():
    label = teacher_opponent("balanced")
    assert label == f"teacher:balanced:{TEACHER_SHA256}"
    with pytest.raises(ValueError, match="simulator plan names 'teacher"):
        live_plan(opponents=(label,))
    with pytest.raises(ValueError, match="simulator plan names 'teacher"):
        live_plan(opponents=("scripted", label))
    for live_label in ("scripted", "neural:" + "cd" * 32):
        with pytest.raises(ValueError, match="live plan names"):
            sim_plan().evaluation.__class__(
                **{**sim_plan().evaluation.to_json(), "opponents": (label, live_label)}
            )
    for bad in ("teacher:berserk:" + "ab" * 32, "teacher:balanced:" + "AB" * 32, "teacher:x"):
        with pytest.raises(ValueError):
            sim_plan().evaluation.__class__(
                **{**sim_plan().evaluation.to_json(), "opponents": (bad,)}
            )
    with pytest.raises(ValueError):
        teacher_opponent("berserk")


def test_simulator_plan_has_no_defaults_and_fixed_bounds():
    with pytest.raises(TypeError):
        SimulatorEvaluationPlan(evaluation=sim_plan().evaluation)
    with pytest.raises(ValueError, match="domain"):
        sim_plan(evaluation=live_plan())
    with pytest.raises(ValueError, match="EvaluationPlan"):
        sim_plan(evaluation=sim_plan().evaluation.to_json())
    for seed in (EVALUATION_SEED_FLOOR - 1, 0, True, float(EVALUATION_SEED_FLOOR), 2**62 + 1):
        with pytest.raises(ValueError, match="held-out seed"):
            sim_plan(seed=seed)
    for steps in (0, 1201, 4.0, True):
        with pytest.raises(ValueError, match="max_steps"):
            sim_plan(max_steps=steps)
    for sampling in (None, "argmax", ""):
        with pytest.raises(ValueError, match="sampling"):
            sim_plan(sampling=sampling)
    for name in ("simulator_sha256", "teacher_sha256"):
        with pytest.raises(ValueError):
            sim_plan(**{name: "00"})
    with pytest.raises(ValueError, match="teacher SHA-256"):
        sim_plan(teacher_sha256="ef" * 32)
    # The unit-7 floors still apply: at least 20 even games per opponent.
    with pytest.raises(ValueError, match="even integer"):
        sim_plan(games=2)


def test_simulator_plan_identity_is_canonical_and_complete():
    plan = sim_plan()
    assert sim_plan(styles=tuple(reversed(STYLES))).sha256 == plan.sha256
    assert SimulatorEvaluationPlan.from_json(json.loads(json.dumps(plan.to_json()))) == plan
    for change in (
        {"seed": plan.seed + 1},
        {"max_steps": 600},
        {"sampling": "greedy"},
        {"simulator_sha256": "ee" * 32},
    ):
        assert sim_plan(**change).sha256 != plan.sha256
    with pytest.raises(ValueError):
        SimulatorEvaluationPlan.from_json({**plan.to_json(), "extra": 1})
    with pytest.raises(ValueError):
        SimulatorEvaluationPlan.from_json({**plan.to_json(), "schema": "other"})


def test_simulator_plan_names_exactly_the_source_suite_styles():
    # The source gate required all five STYLES with at least 20 games each; the source
    # suite never played the tactical teacher.
    for styles in (STYLES[:4], ("defensive",), (*STYLES, "tactical"), (*STYLES[1:], "tactical")):
        with pytest.raises(ValueError, match="exactly the five source suite styles"):
            sim_plan(styles=styles)
    with pytest.raises(ValueError, match="even integer"):
        sim_plan(games=18)
    assert TEACHER_STYLES[: len(STYLES)] == STYLES
    # Played and recorded in the source suite's STYLES order, not sorted-label order.
    assert [style for _, style in sim_plan().styles()] == list(STYLES)
    assert [style for _, style in sim_plan(styles=tuple(reversed(STYLES))).styles()] == list(STYLES)
    rows = simulator_record(sim_plan(), CANDIDATE, games(sim_plan(), lambda s, f: 1))["rows"]
    assert [r["opponent"].split(":")[1] for r in rows[::20]] == list(STYLES)


def test_fixtures_follow_the_source_suite_and_form_two_seat_blocks():
    plan = sim_plan()
    for opponent, style in plan.styles():
        index = STYLES.index(style)
        fixtures = plan.fixtures(opponent)
        assert [f for f, _, _ in fixtures] == list(range(20))
        for fixture, seed, side in fixtures:
            # Source evaluate_suite: seed + style_index * 10000 + game // 2, side game % 2.
            assert seed == plan.seed + index * 10000 + fixture // 2
            assert side == fixture % 2
    seeds = [seed for o, _ in plan.styles() for _, seed, _ in plan.fixtures(o)]
    assert min(seeds) >= EVALUATION_SEED_FLOOR
    assert len(set(seeds)) == len(seeds) // 2


# --- simulator records through the unit-7 gate -----------------------------------------------


def games(plan, score):
    """Rollout rows for every planned game; ``score(style, fixture)`` picks the outcome."""
    rows = []
    for opponent, style in plan.styles():
        for fixture, seed, side in plan.fixtures(opponent):
            rows.append(
                {
                    "style": style,
                    "seed": seed,
                    "side": side,
                    "score": float(score(style, fixture)),
                    "damage": 1.0,
                    "self_damage": 0.0,
                    "actions": {"hold": 1},
                    "elapsed": 1.0,
                    "terminated": True,
                    "truncated": False,
                }
            )
    return rows


def records(plan, before, after):
    return (
        simulator_record(plan, INCUMBENT, games(plan, before)),
        simulator_record(plan, CANDIDATE, games(plan, after)),
    )


def test_simulator_gate_passes_only_as_a_simulator_decision():
    plan = sim_plan()
    incumbent, candidate = records(plan, lambda s, f: f % 3 == 0, lambda s, f: f % 3 != 0)
    decision = simulator_gate(plan, incumbent, candidate)
    assert decision["passed"] is True
    assert decision["schema"] == SIM_DECISION_SCHEMA
    assert decision["decision"] == "candidate_passed_simulator_gate"
    assert decision["domain"] == SIM_DOMAIN
    assert decision["simulator_only"] is True
    assert decision["live_gain_established"] is False
    assert decision["real_server_strength_established"] is False
    assert decision["automatic_promotion"] is False and decision["public_deployment"] is False
    assert decision["selection_scope"] == "simulator-experimental-only"
    assert decision["simulator_plan_sha256"] == plan.sha256
    assert decision["evidence"] == {
        "simulator": {
            "simulator_sha256": SIMULATOR_SHA256,
            "teacher_sha256": TEACHER_SHA256,
            "seed": plan.seed,
            "max_steps": 1200,
            "sampling": "categorical",
        }
    }
    assert decision["bootstrap"]["blocks"] == 50
    # The unit-7 rows: one per planned game, seat 0 as challenger and 1 as receiver.
    rows = candidate["rows"]
    assert [r["side"] for r in rows[:2]] == ["challenger", "receiver"]
    assert all(r["status"] == "completed" and r["valid"] for r in rows)


def test_simulator_gate_keeps_the_unit7_rules():
    plan = sim_plan()
    # An average gain that regresses against one style by more than 0.10 fails.
    incumbent, candidate = records(
        plan,
        lambda s, f: 0.5,
        lambda s, f: 0.0 if s == "potions" and f < 8 else (0.5 if s == "potions" else 1.0),
    )
    decision = simulator_gate(plan, incumbent, candidate)
    assert decision["passed"] is False and decision["decision"] == "retain_incumbent"
    assert decision["reasons"] == [f"regression beyond limit against {teacher_opponent('potions')}"]
    # Equal evidence never passes.
    same = records(plan, lambda s, f: f % 2, lambda s, f: f % 2)
    assert simulator_gate(plan, *same)["reasons"] == [
        "paired improvement lower bound not above threshold"
    ]
    # An unfinished (truncated or non-terminal) game is invalid evidence and fails closed.
    for unfinished in ({"truncated": True}, {"terminated": False}):
        rows = games(plan, lambda s, f: 1)
        rows[0].update(unfinished)
        candidate = simulator_record(plan, CANDIDATE, rows)
        assert candidate["rows"][0]["valid"] is False
        incumbent = simulator_record(plan, INCUMBENT, games(plan, lambda s, f: 0))
        decision = simulator_gate(plan, incumbent, candidate)
        assert decision["passed"] is False
        assert decision["reasons"] == [
            f"invalid episode evidence against {teacher_opponent('balanced')}"
        ]


def test_simulator_gate_equals_the_source_promotion_gate():
    tree = ast.parse(source_text("anima3/neural/league.py"))
    function = next(n for n in tree.body if getattr(n, "name", None) == "promotion_gate")
    namespace = {"random": random, "STYLES": STYLES}
    # The source gate itself, compiled from the source commit's text (stdlib only).
    exec(compile(ast.Module([function], []), "league.promotion_gate", "exec"), namespace)  # noqa: S102
    promotion_gate = namespace["promotion_gate"]
    plan = sim_plan()
    picker = random.Random(41)
    outcomes = {"passed": 0, "failed": 0, "sorted_order_differs": 0}
    for trial in range(300):
        bias = {s: picker.uniform(-0.3, 0.5) for s in STYLES}
        table = {
            (label, s, f): (picker.random(), picker.random())
            for label in ("before", "after")
            for s in STYLES
            for f in range(20)
        }

        def outcome(label, s, f, shift=0.0, table=table):
            draw, tie = table[label, s, f]
            if tie < 0.15:
                return 0.5
            return 1.0 if draw < 0.4 + shift else 0.0

        def before(s, f):
            return outcome("before", s, f)

        def after(s, f, bias=bias, trial=trial):
            return outcome("after", s, f, bias[s] * (trial % 4) / 3)

        incumbent, candidate = records(plan, before, after)
        ours = simulator_gate(plan, incumbent, candidate)

        def source_rows(score):
            # Built as the source evaluate_suite orders its rows: STYLES, then game.
            return [
                {
                    "style": style,
                    "seed": plan.seed + index * 10000 + game // 2,
                    "side": game % 2,
                    "score": float(score(style, game)),
                }
                for index, style in enumerate(STYLES)
                for game in range(20)
            ]

        theirs = promotion_gate(
            {"domain": SIM_DOMAIN, "rows": source_rows(before)},
            {"domain": SIM_DOMAIN, "rows": source_rows(after)},
        )
        assert ours["passed"] == theirs["passed"], trial
        assert ours["paired_delta"] == pytest.approx(theirs["paired_delta"], abs=1e-12)
        assert ours["paired_delta_ci95"] == theirs["delta_ci95"], trial
        assert {
            o.split(":")[1]: v["mean_score_delta"] for o, v in ours["opponents"].items()
        } == pytest.approx(theirs["style_deltas"], abs=1e-12)
        outcomes["passed" if ours["passed"] else "failed"] += 1
        # The plain unit-7 gate bootstraps in sorted-label order and can differ.
        plain = selection_gate(plan.evaluation, incumbent, candidate)
        outcomes["sorted_order_differs"] += plain["paired_delta_ci95"] != theirs["delta_ci95"]
    assert outcomes["passed"] and outcomes["failed"] and outcomes["sorted_order_differs"]


@pytest.mark.parametrize(
    "damage,error",
    [
        ("missing", "exactly the planned"),
        ("extra", "exactly the planned"),
        ("reordered", "planned fixture"),
        ("seed", "planned fixture"),
        ("side", "planned fixture"),
        ("style", "planned fixture"),
        ("score", "loss/draw/win"),
        ("int_score", "loss/draw/win"),
        ("policy", "neither"),
    ],
)
def test_simulator_record_refuses_games_outside_the_plan(damage, error):
    plan = sim_plan()
    rows = games(plan, lambda s, f: 1)
    policy = CANDIDATE
    if damage == "missing":
        rows.pop()
    elif damage == "extra":
        rows.append(dict(rows[-1]))
    elif damage == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif damage == "seed":
        rows[3]["seed"] += 1
    elif damage == "side":
        rows[3]["side"] = 1 - rows[3]["side"]
    elif damage == "style":
        rows[3]["style"] = "tactical"
    elif damage == "score":
        rows[3]["score"] = 0.75
    elif damage == "int_score":
        rows[3]["score"] = 1
    elif damage == "policy":
        policy = "ef" * 32
    with pytest.raises(SimulatorRefused, match=error):
        simulator_record(plan, policy, rows)


def test_simulator_gate_refuses_other_plans_code_or_policies(monkeypatch):
    plan = sim_plan()
    incumbent, candidate = records(plan, lambda s, f: 0, lambda s, f: 1)
    other = sim_plan(seed=plan.seed + 2)
    _, foreign = records(other, lambda s, f: 0, lambda s, f: 1)
    with pytest.raises(SimulatorRefused, match="another plan"):
        simulator_gate(plan, incumbent, foreign)
    swapped = copy.deepcopy(candidate)
    swapped["policy_sha"] = INCUMBENT
    with pytest.raises(SimulatorRefused, match="policy"):
        simulator_gate(plan, incumbent, swapped)
    relabelled = copy.deepcopy(candidate)
    relabelled["rows"][4]["seed"] += 10_000
    with pytest.raises(SimulatorRefused, match="planned seed"):
        simulator_gate(plan, incumbent, relabelled)
    renamed = copy.deepcopy(candidate)
    renamed["simulator_sha256"] = "ee" * 32
    with pytest.raises(SimulatorRefused, match="other simulator"):
        simulator_gate(plan, incumbent, renamed)
    live = copy.deepcopy(candidate)
    live["schema"] = evaluation.RECORD_SCHEMA
    with pytest.raises(SimulatorRefused, match="not a simulator"):
        simulator_gate(plan, incumbent, live)
    # A plan made for other simulator or teacher code is refused before anything else.
    monkeypatch.setattr(sim_evaluation, "SIMULATOR_SHA256", "ee" * 32)
    with pytest.raises(SimulatorRefused, match="simulator code changed"):
        simulator_gate(plan, incumbent, candidate)
    monkeypatch.undo()
    monkeypatch.setattr(sim_evaluation, "TEACHER_SHA256", "ee" * 32)
    with pytest.raises(SimulatorRefused, match="teacher code changed"):
        simulator_gate(plan, incumbent, candidate)


def test_simulator_evidence_never_meets_the_live_gate():
    plan = sim_plan()
    incumbent, candidate = records(plan, lambda s, f: 0, lambda s, f: 1)
    # The unit-7 gate under a live plan refuses simulator-domain evidence.
    live = live_plan()
    for record in (incumbent, candidate):
        record_rows = copy.deepcopy(record)
        with pytest.raises(EvaluationRefused, match="different domains"):
            selection_gate(live, record_rows, record_rows)
    # Live collections are never decided under a simulator plan.
    with pytest.raises(EvaluationRefused, match="only under a live plan"):
        decide_from_collections(plan.evaluation, incumbent=[], candidate=[])


def record_decision_refuses(lineage: Path, decision: dict) -> None:
    """A simulator decision, as is or relabelled as live, never enters a selection record."""
    with pytest.raises(EvaluationRefused, match="not a selection decision"):
        record_decision(lineage, decision)
    forged = {**decision, "schema": evaluation.DECISION_SCHEMA}
    forged["decision"] = "candidate_passed_gate" if decision["passed"] else "retain_incumbent"
    with pytest.raises(EvaluationRefused, match="only a live evaluation decision"):
        record_decision(lineage, forged)
    forged["domain"] = "servuo"
    with pytest.raises(EvaluationRefused, match="only a live evaluation decision"):
        record_decision(lineage, forged)
    assert not (lineage / evaluation.SELECTION_RECORD).exists()
    assert not (lineage / "claims").exists()


def test_a_simulator_decision_is_never_recorded_as_a_live_selection(tmp_path):
    plan = sim_plan()
    decision = simulator_gate(plan, *records(plan, lambda s, f: 0, lambda s, f: 1))
    assert decision["passed"]
    lineage = tmp_path / "lineage"
    record_decision_refuses(lineage, decision)


def test_simulator_modules_import_without_torch():
    code = textwrap.dedent(
        """
        import sys
        class Refuse:
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("torch import refused")
        sys.meta_path.insert(0, Refuse())
        import anima3, anima3.neural
        import anima3.neural.sim, anima3.neural.drills, anima3.neural.sim_evaluation
        assert "torch" not in sys.modules
        assert "anima3.neural.sim_rollouts" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_gate_opponent_order_must_name_exactly_the_plan_opponents():
    plan = sim_plan()
    incumbent, candidate = records(plan, lambda s, f: 0, lambda s, f: 1)
    labels = [o for o, _ in plan.styles()]
    for order in (labels[:4], [*labels, labels[0]], [*labels[:4], "scripted"]):
        with pytest.raises(EvaluationRefused, match="opponent order"):
            selection_gate(plan.evaluation, incumbent, candidate, opponent_order=order)
    ordered = selection_gate(plan.evaluation, incumbent, candidate, opponent_order=labels)
    assert list(ordered["opponents"]) == labels
    assert list(selection_gate(plan.evaluation, incumbent, candidate)["opponents"]) == sorted(
        labels
    )
