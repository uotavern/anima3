"""Unit-8 tactical teacher, probes and receipt-bound demonstrations; no server, model or torch.

The teacher and probe decision cases come from the training branch's
``test_neural_sim.py``, ``test_training_league.py`` and ``test_melee_probe.py``
(the simulator-driven ones stay with the simulator). The rest cover the unit-8
boundaries: teacher and probe arm identities that are never neural, teacher
choices recorded as demonstrations and never as PPO episodes, demonstration
provenance against the plan, ledger, receipt and supply audit, and training-only
selection with explicit held-out collections and kept rejections.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_neural_live import POLICY_SHA, Harness, Spy, neural_arm, observation

from anima3.duel_wait import TRAINING_SUPPLIES
from anima3.neural import live
from anima3.neural.demonstrations import (
    DemonstrationReceiptVerifier,
    DemonstrationRefused,
    select_demonstrations,
)
from anima3.neural.evaluation import EvaluationRefused, record_from_collection
from anima3.neural.live import (
    DEMONSTRATION_SCHEMA,
    PROBE_SHA256,
    TRAINING_RULES,
    Arm,
    ArmUnavailable,
    CollectionPlan,
    Ledger,
    LiveReplayVerifier,
    preflight_arm,
)
from anima3.neural.melee import CastDumpProbe, MeleeProbe
from anima3.neural.schema import ACTION_INDEX, ACTION_NAMES, FEATURE_NAMES, CombatView, legal_mask
from anima3.neural.splits import NO_HELD_OUT, SplitRefused, select_training_episodes
from anima3.neural.teacher import STYLES, TEACHER_SHA256, TEACHER_STYLES, TacticalTeacher

OTHER = "ee" * 32


def action(name):
    return ACTION_INDEX[name]


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


def teacher_arm(style="tactical", sha=TEACHER_SHA256):
    return Arm("teacher", variant=style, source_sha256=sha)


# --- ported teacher decisions ------------------------------------------------------------------


def test_tactical_teacher_prioritizes_interrupt_before_opening_or_burst():
    view = supplied(opponent_cast="explosion", opponent_spell_age=0.1)
    assert TacticalTeacher("tactical").choose(view) == action("magic_arrow")
    # Physical attack needs no cast, mana, or reagent but requires contact.
    close = replace(view, distance=1, hands_free=True, mana=30)
    assert TacticalTeacher("tactical").choose(close) == action("punch")
    assert TacticalTeacher("tactical").choose(replace(close, stun_skilled=True)) == action("stun")


def test_tactical_teacher_selects_both_dump_finishers_with_reserves():
    view = supplied(elapsed=20)
    assert TacticalTeacher("tactical").choose(view) == action("burst_bolt")
    assert TacticalTeacher("tactical").choose(replace(view, opponent_hp=50)) == action(
        "burst_flame"
    )
    assert TacticalTeacher("tactical").choose(replace(view, opponent_hp=50, mana=51)) == action(
        "burst_bolt"
    )
    # Healing remains more urgent than an interrupt opportunity.
    hurt = replace(view, hp=30, opponent_cast="explosion", opponent_spell_age=0.1)
    assert TacticalTeacher("tactical").choose(hurt) == action("heal_potion")


def test_tactical_low_mana_waits_until_reserve_and_never_chips():
    teacher = TacticalTeacher("tactical")
    view = supplied(
        elapsed=20, mana=10, hands_free=True, opponent_cast="explosion", opponent_spell_age=0.1
    )
    assert teacher.choose(view) == action("meditate")
    assert teacher.choose(replace(view, mana=40, meditation_cooldown=5)) == action("hold")
    assert teacher.choose(replace(view, mana=59)) == action("meditate")
    assert teacher.choose(replace(view, mana=60)) == action("magic_arrow")
    assert teacher.choose(replace(view, hp=25)) == action("heal_potion")


def test_burst_and_potion_opponents_have_distinct_priorities():
    v = CombatView(
        opponent_visible=True,
        line_of_sight=True,
        distance=5,
        reagent_counts=(100,) * 8,
        elapsed=20,
        explosion_potions=5,
    )
    assert TacticalTeacher("burst").choose(v) == ACTION_INDEX["burst_bolt"]
    assert TacticalTeacher("potions").choose(v) == ACTION_INDEX["explosion_potion"]


@pytest.mark.parametrize("style", TEACHER_STYLES)
def test_every_teacher_style_only_chooses_legal_actions(style):
    teacher = TacticalTeacher(style)
    views = [
        supplied(),
        supplied(distance=1, hands_free=True),
        supplied(hp=20, poisoned=True),
        supplied(mana=5, stamina=5),
        supplied(opponent_cast="flamestrike", opponent_spell_age=0.2, distance=2),
        CombatView(),
    ]
    for view in views:
        mask = legal_mask(view)
        assert mask[teacher.choose(view, mask)]
    with pytest.raises(ValueError):
        TacticalTeacher("neural")


def test_teacher_identity_is_the_sha256_of_its_source():
    from anima3.neural import teacher

    source = Path(teacher.__file__).read_bytes()
    assert TEACHER_SHA256 == hashlib.sha256(source).hexdigest()
    assert b"def setup_drill" not in source  # the simulator drill stays with row 7b
    assert STYLES == ("balanced", "pressure", "burst", "defensive", "potions")


# --- ported probes -----------------------------------------------------------------------------


def probe_view(**kw):
    return replace(
        CombatView(
            ready=True,
            fresh=True,
            opponent_visible=True,
            line_of_sight=True,
            reagent_counts=(100,) * 8,
            move_legal=(True,) * 4,
        ),
        **kw,
    )


def test_harm_requires_close_range_and_approaches_first():
    p = MeleeProbe("punch")
    p.opening = {"weaken", "clumsy"}
    v = probe_view(distance=2)
    assert p.choose(v, legal_mask(v)) == "approach"
    v = probe_view(distance=1)
    assert p.choose(v, legal_mask(v)) == "harm"


def test_stun_toggle_waits_for_confirmation_and_success_cooldown():
    p = MeleeProbe("stun")

    def packets(now):
        return p.packets(probe_view(distance=1), now, 58, busy=False, hands_free=True)

    assert [x["type"] for x in packets(0)] == ["StunRequest", "Attack"]
    assert not any(x["type"] == "StunRequest" for x in packets(3))
    p.feedback([SimpleNamespace(cliloc=1019011)], 3)
    p.feedback([SimpleNamespace(cliloc=1004010)], 4)
    assert p.stun_ready
    assert not any(x["type"] == "StunRequest" for x in packets(15))
    p.feedback([SimpleNamespace(cliloc=1004013)], 16)
    assert not any(x["type"] == "StunRequest" for x in packets(25))
    assert any(x["type"] == "StunRequest" for x in packets(26))


def test_no_punch_or_stun_with_unsafe_state_or_owned_cursor():
    for v in (
        probe_view(fresh=False),
        probe_view(ready=False),
        probe_view(paralyzed=True),
        probe_view(distance=2),
        probe_view(opponent_visible=False),
    ):
        assert MeleeProbe("stun").packets(v, 0, 58, busy=False, hands_free=True) == []
    for busy, hands in ((True, True), (False, False)):
        assert MeleeProbe("stun").packets(probe_view(), 0, 58, busy=busy, hands_free=hands) == []
    assert MeleeProbe("magic").packets(probe_view(), 0, 58, busy=False, hands_free=True) == []


def test_emergency_heal_precedes_closing_distance():
    v = probe_view(distance=6, hp=40)
    assert MeleeProbe("punch").choose(v, legal_mask(v)) == "greater_heal"


def test_agility_before_refresh_when_clumsy_lowers_maximum():
    p = MeleeProbe("stun", resources=True)
    items = [
        SimpleNamespace(graphic=0xF08, amount=5, serial=100),
        SimpleNamespace(graphic=0xF0B, amount=5, serial=200),
    ]

    def packet(v, now, busy=False):
        return p.resource_packet(v, now, items, busy=busy, hands_free=True)

    assert packet(probe_view(stamina_max=14, stamina=14), 0) == [{"type": "Use", "serial": 100}]
    assert packet(probe_view(stamina_max=34, stamina=14), 1) == []
    assert packet(probe_view(stamina_max=34, stamina=14), 5) == [{"type": "Use", "serial": 200}]
    assert packet(probe_view(stamina_max=34, stamina=34), 10) == []
    assert packet(probe_view(stamina_max=14, stamina=14), 10, busy=True) == []


def test_cast_dump_probe_repeats_one_spell_sequence_across_timing_arms():
    for mode in ("dump-baseline", "dump-fast", "dump-held"):
        p = CastDumpProbe(mode)
        v = probe_view(distance=4, mana=100)
        assert [p.choose(v, legal_mask(v)) for _ in range(3)] == [
            "explosion",
            "energy_bolt",
            "flamestrike",
        ]
    primed = CastDumpProbe("dump-primed")
    v = probe_view(distance=4)
    assert primed.choose(v, legal_mask(v)) in ("weaken", "clumsy")


# --- arm identities ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "teacher"},
        {"kind": "teacher", "variant": "tactical"},
        {"kind": "teacher", "variant": "neural", "source_sha256": TEACHER_SHA256},
        {"kind": "teacher", "variant": "tactical", "source_sha256": "AB" * 32},
        {"kind": "teacher", "variant": "tactical", "source_sha256": TEACHER_SHA256,
         "policy_sha256": POLICY_SHA},
        {"kind": "teacher", "variant": "tactical", "source_sha256": TEACHER_SHA256,
         "checkpoint": "x"},
        {"kind": "probe", "variant": "kick", "source_sha256": PROBE_SHA256},
        {"kind": "probe", "variant": "stun"},
        {"kind": "scripted", "variant": "tactical"},
        {"kind": "neural", "checkpoint": "x", "policy_sha256": POLICY_SHA, "variant": "stun"},
        {"kind": "neural", "checkpoint": "x", "policy_sha256": POLICY_SHA,
         "source_sha256": TEACHER_SHA256},
    ],
)  # fmt: skip
def test_teacher_and_probe_arms_need_a_variant_and_source_identity(kwargs):
    with pytest.raises(ValueError):
        Arm(**kwargs)


def test_teacher_and_probe_labels_are_never_neural(tmp_path):
    for arm in (teacher_arm(), Arm("probe", variant="stun", source_sha256=PROBE_SHA256)):
        assert not arm.neural
        assert not arm.label.startswith("neural")
        assert arm.agent_policy == "scripted"
        assert len(f"{arm.label} {arm.agent_policy}".split()) == 3
    assert teacher_arm().label == f"teacher tactical-{TEACHER_SHA256[:12]}"
    assert Arm("scripted").label == "scripted fixed-scripted"


def test_existing_plan_sha256_is_unchanged_by_the_new_arm_fields(tmp_path):
    arm = neural_arm(tmp_path)
    plan = CollectionPlan("lab.invalid:2593", ("alpha", "beta"), (arm, Arm("scripted")), 2)
    unit7 = {
        "endpoint": "lab.invalid:2593",
        "accounts": ["alpha", "beta"],
        "arms": [
            {"kind": "neural", "checkpoint": arm.checkpoint, "policy_sha256": POLICY_SHA},
            {"kind": "scripted", "checkpoint": None, "policy_sha256": None},
        ],
        "matches": 2,
        "deadline_ms": 500.0,
        "require_base_template": False,
        "match_deadline_s": 340.0,
        "purpose": "evaluation",
    }
    canonical = json.dumps(unit7, sort_keys=True, separators=(",", ":"))
    assert plan.sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    teacher_plan = CollectionPlan(
        "lab.invalid:2593", ("alpha", "beta"), (arm, teacher_arm()), 2, require_base_template=True
    )
    assert teacher_plan.to_json()["arms"][1] == {
        "kind": "teacher",
        "checkpoint": None,
        "policy_sha256": None,
        "variant": "tactical",
        "source_sha256": TEACHER_SHA256,
    }


def test_teacher_plan_needs_verified_base_template_and_allows_scripted_opponent():
    with pytest.raises(ValueError, match="require_base_template"):
        CollectionPlan("lab.invalid:2593", ("a", "b"), (teacher_arm(), Arm("scripted")), 2)
    CollectionPlan(
        "lab.invalid:2593", ("a", "b"), (teacher_arm(), Arm("scripted")), 2,
        require_base_template=True,
    )  # fmt: skip
    with pytest.raises(ValueError):
        CollectionPlan("lab.invalid:2593", ("a", "b"), (Arm("scripted"), Arm("scripted")), 2)


def test_a_changed_teacher_source_refuses_the_run_before_any_login(tmp_path):
    with pytest.raises(ArmUnavailable, match="teacher source differs"):
        preflight_arm(teacher_arm(sha=OTHER))
    with pytest.raises(ArmUnavailable, match="probe source differs"):
        preflight_arm(Arm("probe", variant="stun", source_sha256=OTHER))
    preflight_arm(teacher_arm())
    spy = Spy()
    plan = CollectionPlan(
        "lab.invalid:2593", ("alpha", "beta"), (teacher_arm(sha=OTHER), Arm("scripted")), 2,
        require_base_template=True,
    )  # fmt: skip
    with pytest.raises(ArmUnavailable):
        live.run_collection(
            plan,
            tmp_path / "run",
            connect=spy.connect,
            fetch_receipt=spy.receipt,
            lock_dir=tmp_path / "locks",
            find_spec=lambda name: object(),
        )
    assert spy.calls == []
    assert Ledger.load(tmp_path / "run" / "ledger.jsonl").issued == 0


# --- the teacher and probe seats inside the actor ----------------------------------------------


class TeacherExecutor:
    def __init__(self, view=None, *, start=True, defend=False, events=()):
        self.view = view or supplied(elapsed=20)
        self.busy = False
        self.started, self.resets = [], []
        self.accept = start
        self.defend = defend
        self.events = list(events)

    def reset(self, opponent, epoch, now=None):
        self.resets.append(epoch)

    def observe(self, *a, **k):
        return SimpleNamespace(mask=legal_mask(self.view), features=[0.0] * len(FEATURE_NAMES))

    def tick(self, *a):
        return []

    def drain_events(self):
        rows, self.events = self.events, []
        return rows

    def defend_potions(self, obs, now):
        return (True, []) if self.defend else (False, [])

    def start(self, act, obs, now):
        self.started.append(act)
        return self.accept


def run_actor(tmp_path, monkeypatch, executor, phases, **kwargs):
    clock = [100.0]
    step = [0]

    def state_for(_obs):
        phase, round_ = phases[min(step[0], len(phases) - 1)]
        return {"id": "a" * 32, "phase": phase, "opponent": 2, "rules": TRAINING_RULES,
                "round": round_}  # fmt: skip

    def pump(_):
        clock[0] += 1.0
        step[0] += 1

    obs = observation()
    obs.items = []
    obs.own_pack = list
    acted = []
    body = SimpleNamespace(observe=lambda: obs, act=acted.append, pump=pump)
    monkeypatch.setattr(live.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(live, "server_state", state_for)
    monkeypatch.setattr(live, "CombatExecutor", lambda *_, **__: executor)
    monkeypatch.setattr(
        live, "_scripted_baseline", lambda *a: pytest.fail("teacher replaced by the scripted agent")
    )
    result = live._actor(body, 2, None, tmp_path, threading.Event(), **kwargs)
    return result, acted


def test_teacher_choices_become_labelled_steps_never_transitions(tmp_path, monkeypatch):
    executor = TeacherExecutor()
    phases = [("Fighting", 1)] * 3 + [("Idle", 1)]
    result, _ = run_actor(tmp_path, monkeypatch, executor, phases, teacher="tactical")
    assert result["policy_kind"] == "teacher" and result["policy_sha"] is None
    assert result["transitions"] == [] and result["invalid_reasons"] == []
    steps = result["teacher_steps"]
    assert len(steps) == 3 and all(s["accepted"] is True for s in steps)
    assert [s["action"] for s in steps] == executor.started
    assert all(steps[i]["at"] < steps[i + 1]["at"] for i in range(2))
    assert all(s["mask"][s["action"]] for s in steps)


def test_rejected_teacher_actions_stay_recorded_and_make_the_round_ineligible(
    tmp_path, monkeypatch
):
    executor = TeacherExecutor(start=False)
    result, _ = run_actor(
        tmp_path, monkeypatch, executor, [("Fighting", 1)] * 2 + [("Idle", 1)], teacher="burst"
    )
    assert result["invalid_reasons"] == ["teacher_action_rejected"]
    assert [s["accepted"] for s in result["teacher_steps"]] == [False, False]


def test_potion_defense_and_substituted_procedures_are_outside_the_teacher_labels(
    tmp_path, monkeypatch
):
    defended, _ = run_actor(
        tmp_path, monkeypatch, TeacherExecutor(defend=True), [("Fighting", 1), ("Idle", 1)],
        teacher="tactical",
    )  # fmt: skip
    assert defended["invalid_reasons"] == ["potion_defense_outside_policy_schema"]
    assert defended["teacher_steps"] == []
    override = TeacherExecutor(events=[{"type": "x", "policy_override": True}])
    result, _ = run_actor(
        tmp_path, monkeypatch, override, [("Fighting", 1), ("Idle", 1)], teacher="balanced"
    )
    assert result["invalid_reasons"] == ["policy_override"]


def test_teacher_labels_nothing_from_stale_or_unready_state(tmp_path, monkeypatch):
    executor = TeacherExecutor(view=supplied(ready=False))
    result, _ = run_actor(
        tmp_path, monkeypatch, executor, [("Fighting", 1)] * 2 + [("Idle", 1)], teacher="tactical"
    )
    assert result["teacher_steps"] == [] and executor.started == []


def test_probe_seat_records_neither_steps_nor_transitions(tmp_path, monkeypatch):
    executor = TeacherExecutor(view=probe_view(distance=1, hands_free=True))
    executor.view = replace(executor.view, opponent_hp=100)
    result, acted = run_actor(
        tmp_path, monkeypatch, executor, [("Fighting", 1)] * 2 + [("Idle", 1)], probe="stun"
    )
    assert result["policy_kind"] == "probe" and "teacher_steps" not in result
    assert result["transitions"] == []
    assert {"type": "StunRequest"} in acted


def test_one_seat_runs_exactly_one_policy(tmp_path):
    with pytest.raises(ValueError, match="exactly one policy"):
        live._actor(None, 2, object(), tmp_path, threading.Event(), teacher="tactical")
    with pytest.raises(ValueError, match="teacher style"):
        live._actor(None, 2, None, tmp_path, threading.Event(), teacher="neural")
    with pytest.raises(ValueError, match="probe variant"):
        live._actor(None, 2, None, tmp_path, threading.Event(), probe="kick")


# --- collections with a teacher seat -----------------------------------------------------------

_IDS = iter(range(0x500, 0x10000))


def template_replay(match_id, seats, winner=1):
    rows = [
        {"seq": 0, "t": 0, "type": "header", "schema": 1, "id": match_id,
         "rules": TRAINING_RULES, "ranked": False, "training": True,
         "players": [{"serial": s, "stats": [100, 25, 100]} for s in seats]},
        {"seq": 1, "t": 50, "type": "frame", "phase": "Countdown",
         "players": [{"hitsMax": 100, "manaMax": 100, "stamMax": 25, "alive": True,
                      "poisoned": False} for _ in seats]},
        {"seq": 2, "t": 200, "type": "end", "id": match_id, "complete": True, "training": True,
         "winner": winner, "score": [1, 0], "aborted": None},
    ]  # fmt: skip
    data = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
    meta = {**rows[-1], "sha256": hashlib.sha256(data).hexdigest(), "rules": TRAINING_RULES,
            "ranked": False, "players": [{"serial": s} for s in seats]}  # fmt: skip
    return meta, data


def audit(serial, **changes):
    return {
        "serial": serial,
        "verified": True,
        "weight": 200,
        "weight_max": 390,
        "counts": {hex(k): v for k, v in TRAINING_SUPPLIES.items()},
        "required": {hex(k): v for k, v in TRAINING_SUPPLIES.items()},
        **changes,
    }


def teacher_step(at, act=1):
    mask = [True] + [False] * (len(ACTION_NAMES) - 1)
    mask[act] = True
    return {"at": at, "features": [0.0] * len(FEATURE_NAMES), "mask": mask, "action": act,
            "accepted": True}  # fmt: skip


class TeacherHarness(Harness):
    """The real runner, ledger and receipts with a teacher seat; fights are scripted."""

    def __init__(self, monkeypatch, games, *, steps=3, invalid=None, winners=None):
        super().__init__(monkeypatch, [f"{next(_IDS):032x}" for _ in range(games)])
        self.steps = steps
        self.invalid = invalid or {}
        self.winners = list(winners or [1] * games)
        self.games = 0
        monkeypatch.setattr(live, "verify_training_supplies", lambda body: audit(body.serial))

    def fight(self, bodies, workers, log, stop, *, arms=(None, None), **_):
        actors = super().fight(bodies, workers, log, stop)
        for i, arm in enumerate(arms):
            actors[i]["policy_kind"] = arm.kind
            if arm.kind == "teacher":
                actors[i]["transitions"] = []
                actors[i]["teacher_steps"] = [
                    teacher_step(float(n) + 0.5, 1 + n % 2) for n in range(self.steps)
                ]
                actors[i]["invalid_reasons"] = list(self.invalid.get(self.games, []))
        self.games += 1
        return actors

    def receipt(self, match_id, seats):
        return template_replay(match_id, tuple(seats), winner=self.winners.pop(0))


def teacher_plan(tmp_path, matches=2, *, purpose="training", arms=None):
    return CollectionPlan(
        endpoint="lab.invalid:2593",
        accounts=("alpha", "beta"),
        arms=arms or (neural_arm(tmp_path), teacher_arm()),
        matches=matches,
        require_base_template=True,
        purpose=purpose,
    )


def collect(tmp_path, monkeypatch, name, matches=2, *, purpose="training", arms=None, **kw):
    harness = TeacherHarness(monkeypatch, matches, **kw)
    plan = teacher_plan(tmp_path, matches, purpose=purpose, arms=arms)
    live.run_collection(
        plan,
        tmp_path / name,
        connect=harness.connect,
        fetch_receipt=harness.receipt,
        lock_dir=tmp_path / "locks",
        client_factory=harness.factory,
        find_spec=lambda n: object(),
    )
    return tmp_path / name


def demo_files(root):
    return sorted(root.glob("*/demonstration-*.json"))


def test_teacher_seat_writes_receipt_bound_demonstrations_beside_episodes(tmp_path, monkeypatch):
    root = collect(tmp_path, monkeypatch, "demos")
    ledger = Ledger.load(root / "ledger.jsonl")
    issued = [r for r in ledger.entries if r["event"] == "issued"]
    assert all([a["serial"] for a in r["supplies"]] == [1, 2] for r in issued)
    for row in ledger.completed:
        assert [Path(f).name for f in row["episodes"]] == ["episode-1.json"]
        assert [Path(f).name for f in row["demonstrations"]] == ["demonstration-2.json"]
        digest = hashlib.sha256(Path(row["demonstrations"][0]).read_bytes()).hexdigest()
        assert row["demonstration_sha256"] == [digest]
        assert row["result"]["arms"][1].startswith("teacher tactical-")
    files = demo_files(root)
    assert len(files) == 2 and not list(root.glob("*/.staged/*"))
    demo = json.loads(files[0].read_text())
    assert demo["schema"] == DEMONSTRATION_SCHEMA and demo["policy_kind"] == "teacher"
    assert demo["label"] == "teacher:tactical" and "policy_sha" not in demo
    assert demo["teacher"]["sha256"] == TEACHER_SHA256
    assert demo["ppo_eligible"] is False and demo["eligible"] is True
    assert demo["opponent_arm"]["kind"] == "neural"
    assert DemonstrationReceiptVerifier(root)(demo)
    # A demonstration is never a live policy episode, whatever SHA is asked for.
    assert not LiveReplayVerifier(root)(demo)
    assert not LiveReplayVerifier(root, POLICY_SHA)(demo)


def test_ppo_selection_reads_episodes_only_and_refuses_a_listed_demonstration(
    tmp_path, monkeypatch
):
    root = collect(tmp_path, monkeypatch, "demos")
    selection = select_training_episodes([root], POLICY_SHA, held_out=NO_HELD_OUT)
    assert len(selection.episodes) == 2
    assert all(e["schema"] == live.EPISODE_SCHEMA for e in selection.episodes)
    lines = (root / "ledger.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    for row in rows:
        if row["event"] == "completed":
            row["episodes"] = row["episodes"] + row["demonstrations"]
    (root / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SplitRefused, match="escapes"):
        select_training_episodes([root], POLICY_SHA, held_out=NO_HELD_OUT)


def valid_demo(tmp_path, monkeypatch):
    root = collect(tmp_path, monkeypatch, "demos", matches=2)
    path = demo_files(root)[0]
    return root, path, json.loads(path.read_text())


@pytest.mark.parametrize(
    "damage",
    [
        "teacher_sha", "style", "label", "policy_sha", "ppo_eligible", "kind", "plan", "purpose",
        "replay_sha", "executor", "fingerprint", "illegal", "rejected_step", "clock", "fixture",
        "fixture_bool", "seat", "result", "empty", "feature_range",
    ],
)  # fmt: skip
def test_demonstration_provenance_is_checked_not_trusted(tmp_path, monkeypatch, damage):
    root, _, demo = valid_demo(tmp_path, monkeypatch)
    step = demo["steps"][0]
    if damage == "teacher_sha":
        demo["teacher"]["sha256"] = OTHER
    elif damage == "style":
        demo["teacher"]["style"], demo["label"] = "burst", "teacher:burst"
    elif damage == "label":
        demo["label"] = f"neural {POLICY_SHA[:12]}"
    elif damage == "policy_sha":
        demo["policy_sha"] = POLICY_SHA
    elif damage == "ppo_eligible":
        demo["ppo_eligible"] = True
    elif damage == "kind":
        demo["policy_kind"] = "neural"
    elif damage == "plan":
        demo["plan_sha256"] = OTHER
    elif damage == "purpose":
        demo["purpose"] = "evaluation"
    elif damage == "replay_sha":
        demo["verification"]["replay_sha256"] = OTHER
    elif damage == "executor":
        demo["executor_sha256"] = OTHER
    elif damage == "fingerprint":
        demo["schema_fingerprint"] = "bad"
    elif damage == "illegal":
        step["action"] = len(ACTION_NAMES) - 1
    elif damage == "rejected_step":
        step["accepted"] = False
    elif damage == "clock":
        demo["steps"][1]["at"] = step["at"]
    elif damage == "fixture":
        demo["fixture"] = 1
    elif damage == "fixture_bool":
        demo["fixture"] = demo["fixture"] != 0  # a bool, equal to the int index
    elif damage == "seat":
        demo["actor"] = 1
    elif damage == "result":
        demo["final_result"]["outcome"] = 1.0
    elif damage == "empty":
        demo["steps"] = []
    else:
        step["features"][0] = 1.5
    verifier = DemonstrationReceiptVerifier(root)
    assert not verifier(demo)
    with pytest.raises(DemonstrationRefused):
        verifier.check(demo)


@pytest.mark.parametrize(
    "fault", ["missing", "potions", "weight", "participant", "unverified", "nonfinite"]
)
def test_demonstrations_need_the_pre_challenge_supply_audit(tmp_path, monkeypatch, fault):
    root, _, demo = valid_demo(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in (root / "ledger.jsonl").read_text().splitlines()]
    issued = next(r for r in rows if r["event"] == "issued" and r["fixture"] == demo["fixture"])
    if fault == "missing":
        del issued["supplies"]
    elif fault == "potions":
        issued["supplies"][0]["counts"]["0xf0d"] = 0
    elif fault == "weight":
        issued["supplies"][0]["weight"] = 415
    elif fault == "participant":
        issued["supplies"][0]["serial"] = 99
    elif fault == "unverified":
        issued["supplies"][1]["verified"] = False
    else:
        issued["supplies"][1]["weight"] = "NaN"
    (root / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(DemonstrationRefused, match="supply verification"):
        DemonstrationReceiptVerifier(root).check(demo)


def test_demonstrations_need_the_server_starting_template(tmp_path, monkeypatch):
    root, _, demo = valid_demo(tmp_path, monkeypatch)
    match = root / demo["verification"]["match_id"]
    lines = (match / "replay.jsonl").read_text().splitlines()
    frame = json.loads(lines[1])
    frame["players"][0]["hitsMax"] = 120
    lines[1] = json.dumps(frame)
    data = ("\n".join(lines) + "\n").encode()
    (match / "replay.jsonl").write_bytes(data)
    meta = json.loads((match / "replay.meta.json").read_text())
    meta["sha256"] = hashlib.sha256(data).hexdigest()
    (match / "replay.meta.json").write_text(json.dumps(meta))
    demo["verification"]["replay_sha256"] = meta["sha256"]
    with pytest.raises(DemonstrationRefused, match="receipt differs"):
        DemonstrationReceiptVerifier(root).check(demo)


def test_a_demonstration_needs_its_completed_row(tmp_path, monkeypatch):
    root, _, demo = valid_demo(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in (root / "ledger.jsonl").read_text().splitlines()]
    for row in rows:
        if row["event"] == "completed" and row["fixture"] == demo["fixture"]:
            row["demonstrations"], row["demonstration_sha256"] = [], []
    (root / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(DemonstrationRefused, match="completed ledger row"):
        DemonstrationReceiptVerifier(root).check(demo)
    assert DemonstrationReceiptVerifier(root, require_ledger=False)(demo)


def test_a_teacher_rejected_round_is_kept_but_never_selected(tmp_path, monkeypatch):
    root = collect(tmp_path, monkeypatch, "demos", invalid={1: ["teacher_action_rejected"]})
    selection = select_demonstrations([root], teacher_sha256=TEACHER_SHA256, held_out=NO_HELD_OUT)
    assert len(selection.demonstrations) == 1
    assert [r["reasons"] for r in selection.rejected] == [["ineligible", "teacher_action_rejected"]]
    record = selection.provenance[0]
    path = Path(record["file"])
    assert record["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["teacher"]["sha256"] == TEACHER_SHA256 and record["collection"] == str(root)
    assert record["replay_sha256"] == selection.demonstrations[0]["verification"]["replay_sha256"]


def test_other_teacher_and_style_demonstrations_are_rejected_on_record(tmp_path, monkeypatch):
    root = collect(tmp_path, monkeypatch, "demos")
    other = select_demonstrations([root], teacher_sha256=OTHER, held_out=NO_HELD_OUT)
    assert other.demonstrations == [] and all(
        r["reasons"] == ["other_teacher"] for r in other.rejected
    )
    styled = select_demonstrations(
        [root], teacher_sha256=TEACHER_SHA256, held_out=NO_HELD_OUT, styles=["burst"]
    )
    assert styled.demonstrations == [] and len(styled.rejected) == 2


def test_evaluation_collections_never_supply_demonstrations(tmp_path, monkeypatch):
    root = collect(tmp_path, monkeypatch, "held", purpose="evaluation")
    with pytest.raises(DemonstrationRefused, match="held out"):
        select_demonstrations([root], teacher_sha256=TEACHER_SHA256, held_out=NO_HELD_OUT)
    # Nor is a teacher ever scored as an evaluation opponent under any label.
    with pytest.raises(EvaluationRefused, match="'teacher' arm"):
        record_from_collection(root, POLICY_SHA)


def test_selection_needs_explicit_absolute_held_out_and_refuses_overlap(tmp_path, monkeypatch):
    train = collect(tmp_path, monkeypatch, "train")
    held = collect(
        tmp_path, monkeypatch, "held", purpose="evaluation",
        arms=(neural_arm(tmp_path), Arm("scripted")),
    )  # fmt: skip
    for bad in (None, str(held)):
        with pytest.raises(DemonstrationRefused, match="held_out"):
            select_demonstrations([train], teacher_sha256=TEACHER_SHA256, held_out=bad)
    with pytest.raises(DemonstrationRefused, match="both"):
        select_demonstrations([train], teacher_sha256=TEACHER_SHA256, held_out=[train])
    with pytest.raises(DemonstrationRefused, match="absolute"):
        select_demonstrations(
            [Path("relative/train")], teacher_sha256=TEACHER_SHA256, held_out=NO_HELD_OUT
        )
    with pytest.raises(DemonstrationRefused, match="listed twice"):
        select_demonstrations([train, train], teacher_sha256=TEACHER_SHA256, held_out=NO_HELD_OUT)
    with pytest.raises(DemonstrationRefused, match="teacher SHA-256"):
        select_demonstrations([train], teacher_sha256=None, held_out=NO_HELD_OUT)
    assert select_demonstrations([train], teacher_sha256=TEACHER_SHA256, held_out=[held])
    # A held-out collection sharing a match identity with the input refuses the selection.
    held_ledger = held / "ledger.jsonl"
    rows = [json.loads(x) for x in held_ledger.read_text().splitlines()]
    train_match = Ledger.load(train / "ledger.jsonl").completed[0]["match_id"]
    next(r for r in rows if r["event"] == "completed")["match_id"] = train_match
    held_ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(DemonstrationRefused, match="held-out"):
        select_demonstrations([train], teacher_sha256=TEACHER_SHA256, held_out=[held])


def test_a_tampered_eligible_demonstration_refuses_the_whole_selection(tmp_path, monkeypatch):
    root, path, demo = valid_demo(tmp_path, monkeypatch)
    demo["steps"][0]["action"] = 0  # still legal: only the ledger digest notices
    path.write_text(json.dumps(demo))
    with pytest.raises(DemonstrationRefused, match="SHA-256 in its ledger row"):
        select_demonstrations([root], teacher_sha256=TEACHER_SHA256, held_out=NO_HELD_OUT)
    demo["steps"][0]["action"] = len(ACTION_NAMES) - 1  # illegal in its recorded mask
    path.write_text(json.dumps(demo))
    rows = [json.loads(x) for x in (root / "ledger.jsonl").read_text().splitlines()]
    for row in rows:
        if str(path) in row.get("demonstrations", []):
            row["demonstration_sha256"] = [hashlib.sha256(path.read_bytes()).hexdigest()]
    (root / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(DemonstrationRefused, match="failed verification"):
        select_demonstrations([root], teacher_sha256=TEACHER_SHA256, held_out=NO_HELD_OUT)


def test_demonstration_paths_and_identities_must_match_their_row(tmp_path, monkeypatch):
    root, _, _ = valid_demo(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in (root / "ledger.jsonl").read_text().splitlines()]
    completed = next(r for r in rows if r["event"] == "completed")
    completed["demonstrations"] = [str(root / "elsewhere" / "demonstration-2.json")]
    (root / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(DemonstrationRefused, match="escapes"):
        select_demonstrations([root], teacher_sha256=TEACHER_SHA256, held_out=NO_HELD_OUT)


def test_crash_after_completed_row_publishes_staged_demonstrations(tmp_path, monkeypatch):
    harness = TeacherHarness(monkeypatch, 2)
    plan = teacher_plan(tmp_path, 2)
    real_publish = live._publish
    monkeypatch.setattr(
        live, "_publish", lambda paths: (_ for _ in ()).throw(RuntimeError("crash"))
    )
    with pytest.raises(RuntimeError, match="crash"):
        harness.run(tmp_path, plan)
    root = tmp_path / "run"
    assert demo_files(root) == [] and list(root.glob("*/.staged/demonstration-*.json"))
    monkeypatch.setattr(live, "_publish", real_publish)
    live.open_collection(root, plan, resume=True)
    assert len(demo_files(root)) == 1
    assert DemonstrationReceiptVerifier(root)(json.loads(demo_files(root)[0].read_text()))


def test_a_refused_staged_demonstration_closes_the_fixture_invalid(tmp_path, monkeypatch):
    harness = TeacherHarness(monkeypatch, 1)
    from anima3.neural import demonstrations

    monkeypatch.setattr(demonstrations, "_supplies_verified", lambda *a: False)
    with pytest.raises(live.CollectionRefused, match="demonstration failed"):
        harness.run(tmp_path, teacher_plan(tmp_path, 1))
    ledger = Ledger.load(tmp_path / "run" / "ledger.jsonl")
    assert [r["event"] for r in ledger.entries] == ["issued", "invalid_fixture"]
    assert demo_files(tmp_path / "run") == []


def test_actor_kind_must_match_the_planned_arm(tmp_path, monkeypatch):
    harness = TeacherHarness(monkeypatch, 1)
    original = harness.fight

    def wrong(*a, **k):
        actors = original(*a, **k)
        actors[1]["policy_kind"] = "neural"
        return actors

    monkeypatch.setattr(live, "fight", wrong)
    with pytest.raises(RuntimeError, match="policy kind"):
        harness.run(tmp_path, teacher_plan(tmp_path, 1))
    assert [r["event"] for r in Ledger.load(tmp_path / "run" / "ledger.jsonl").entries] == [
        "issued",
        "failed",
    ]


def test_teacher_modules_import_without_torch():
    import subprocess

    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, *a):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('torch import refused')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import anima3.neural.teacher, anima3.neural.melee, anima3.neural.demonstrations\n"
        "import anima3.neural.live\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "change",
    [
        {"demonstration_sha256": None},
        {"demonstration_sha256": ["0" * 63]},
        {"demonstration_sha256": []},
        {"demonstrations": "demonstration-2.json"},
    ],
)
def test_ledger_rows_bind_each_demonstration_to_a_digest(tmp_path, monkeypatch, change):
    root = collect(tmp_path, monkeypatch, "demos", matches=1)
    rows = [json.loads(x) for x in (root / "ledger.jsonl").read_text().splitlines()]
    completed = next(r for r in rows if r["event"] == "completed")
    for key, value in change.items():
        if value is None:
            del completed[key]
        else:
            completed[key] = value
    (root / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(live.CollectionRefused, match="malformed ledger line"):
        Ledger.load(root / "ledger.jsonl")
