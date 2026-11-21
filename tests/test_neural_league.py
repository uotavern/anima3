"""Row 7b-2 simulator league, simulator children and exact resume; tiny CPU budgets.

The league is compared with the source ``league.main`` at ``819f4f4`` (tag
``archive/training-league-819f4f4``), exported with ``git archive`` and run in a
separate interpreter with the same arguments. Crashes are simulated by making a
step raise; no server, client or long training run is involved.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_neural_sim import export_source, run_tree
from test_neural_training import LINEAGE, SMALL, live_episode, saved

from anima3.neural import evaluation, league, train
from anima3.neural.checkpoints import load_checkpoint, save_checkpoint
from anima3.neural.drills import SIM_DOMAIN, SIMULATOR_SHA256
from anima3.neural.imitation import ImitationConfig
from anima3.neural.league import LeaguePlan, LeagueRefused, run_league
from anima3.neural.model import PolicyConfig, RecurrentPolicy
from anima3.neural.schema import ACTION_NAMES, FEATURE_NAMES
from anima3.neural.sim_evaluation import EVALUATION_SEED_FLOOR
from anima3.neural.teacher import TEACHER_SHA256
from anima3.neural.train import TrainConfig, ancestry_simulator

LABELS = ("initial", "warmstart")


def plan(**overrides) -> LeaguePlan:
    values = {
        "seed": 20261006,
        "bc_steps_per_drill": 8,
        "bc": ImitationConfig(
            epochs=1, sequence_length=64, batch_sequences=8, balance_actions=False
        ),
        "bc_learning_rate": 3e-4,
        "rehearsal_epochs": 1,
        "rehearsal_learning_rate": 1e-4,
        "ppo": TrainConfig(
            epochs=4,
            sequence_length=64,
            burn_in=8,
            batch_sequences=8,
            discount_time_unit=1.0,
            learning_rate=1e-4,
        ),
        "updates": 4,
        "rollout_steps": 160,
        "max_steps": 40,
        "evaluation_seed": 3_600_000_000,
        "evaluation_games_per_style": 20,
        "evaluation_sampling": "categorical",
        "threads": 1,
        "start_checkpoint": None,
        "start_sha256": None,
    }
    return LeaguePlan(**{**values, **overrides})


def weights(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    return torch.load(directory / manifest["weights"]["file"], weights_only=True)


def same_weights(a: Path, b: Path) -> bool:
    x, y = weights(a), weights(b)
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def checkpoint_shas(root: Path) -> dict:
    found = {}
    for manifest in sorted(root.rglob("manifest.json")):
        found[str(manifest.parent.relative_to(root))] = json.loads(manifest.read_text())["weights"][
            "sha256"
        ]
    return found


def outcome(root: Path) -> dict:
    """Everything a run decides, without paths or timestamps."""
    progress = json.loads((root / "progress.json").read_text())
    return {
        "checkpoints": checkpoint_shas(root),
        "registry": json.loads((root / "registry.json").read_text()),
        "evaluations": {
            label: json.loads((root / f"evaluation-{label}.json").read_text())
            for label in ("initial", "warmstart", "candidate")
        },
        "events": progress["events"],
        "demonstrations": hashlib.sha256((root / "demonstrations.json").read_bytes()).hexdigest(),
    }


@pytest.fixture(scope="module")
def source_tree(tmp_path_factory):
    return export_source(tmp_path_factory)


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory):
    root = tmp_path_factory.mktemp("league") / "run"
    run_league(plan(), root)
    return root


# --- equivalence with the source league ------------------------------------------------------


SOURCE_MAIN = """
from anima3.neural.league import main
main({argv!r})
print("{{}}")
"""


def source_argv(p: LeaguePlan, out: Path, start=None) -> list[str]:
    argv = [
        "--out", str(out), "--seed", str(p.seed),
        "--bc-steps-per-drill", str(p.bc_steps_per_drill), "--bc-epochs", str(p.bc.epochs),
        "--bc-rehearsal-epochs", str(p.rehearsal_epochs),
        "--discount-time-unit", str(p.ppo.discount_time_unit),
        "--updates", str(p.updates), "--rollout-steps", str(p.rollout_steps),
        "--max-steps", str(p.max_steps),
        "--eval-games-per-style", str(p.evaluation_games_per_style),
        "--eval-seed", str(p.evaluation_seed), "--threads", str(p.threads),
    ]  # fmt: skip
    return argv + (["--start-checkpoint", str(start)] if start else [])


def assert_matches_source(ours: Path, theirs: Path, updates: int) -> None:
    for label in (*LABELS, *(f"snapshots/update-{u:03d}" for u in range(1, updates + 1))):
        assert same_weights(ours / label, theirs / label), label
    for label in ("initial", "warmstart", "candidate"):
        mine = json.loads((ours / f"evaluation-{label}.json").read_text())
        source = json.loads((theirs / f"evaluation-{label}.json").read_text())
        assert mine["rows"] == source["rows"], label
        assert mine["score"] == source["score"]
    registry = json.loads((ours / "registry.json").read_text())
    source = json.loads((theirs / "registry.json").read_text())
    for name, gate in (("bc_gate", "bc_gate"), ("ppo_gate", "ppo_gate")):
        assert registry[name]["passed"] == source[gate]["passed"], name
        if "paired_delta_ci95" in registry[name]:
            assert registry[name]["paired_delta_ci95"] == source[gate]["delta_ci95"]
            assert registry[name]["paired_delta"] == pytest.approx(source[gate]["paired_delta"])
    assert registry["selection_gate"]["passed"] == source["selection_gate"]["passed"]
    selected = Path(source["selected_checkpoint"]).name
    expected = {"initial": "initial", "warmstart": "warmstart"}.get(selected, "candidate")
    assert registry["selected"] == expected
    assert (
        registry["selected_sha"] == checkpoint_shas(ours)[registry["checkpoints"][expected]["path"]]
    )
    # Per-update PPO and rehearsal metrics, as the source logged them.
    mine = [e for e in json.loads((ours / "progress.json").read_text())["events"]]
    source = json.loads((theirs / "progress.json").read_text())["events"]
    assert len(mine) == len(source)
    for a, b in zip(mine, source, strict=True):
        assert {k: a[k] for k in b} == b


def test_league_equals_the_source_main(source_tree, uninterrupted, tmp_path):
    p = plan()
    run_tree(source_tree, "", SOURCE_MAIN.format(argv=source_argv(p, tmp_path / "source")))
    assert_matches_source(uninterrupted, tmp_path / "source", p.updates)
    scores = {
        r["score"]
        for label in ("initial", "warmstart", "candidate")
        for r in json.loads((uninterrupted / f"evaluation-{label}.json").read_text())["rows"]
    }
    assert len(scores) > 1  # the comparison includes decided games, not only draws
    # ...and rollouts against frozen league snapshots, not only scripted teachers.
    events = json.loads((uninterrupted / "progress.json").read_text())["events"]
    frozen = [e["opponents"].get("frozen_policy", 0) for e in events if e["stage"] == "ppo"]
    assert all(frozen) and len(frozen) == 4


@pytest.mark.parametrize(
    "bc,final,cloned,selected,labels",
    [
        (True, True, True, "candidate", ("not_gated", "rollback", "registry_selected")),
        (True, False, True, "warmstart", ("not_gated", "registry_selected", "rejected")),
        (False, True, True, "candidate", ("rollback", "rejected", "registry_selected")),
        (False, False, True, "initial", ("registry_selected", "rejected", "rejected")),
        (False, False, False, "initial", ("registry_selected", "not_gated", "rejected")),
    ],
)
def test_selection_follows_the_source_gates(bc, final, cloned, selected, labels):
    chosen, selection = league._select(bc, final, cloned=cloned)
    assert chosen == selected
    assert tuple(selection[k] for k in ("initial", "warmstart", "candidate")) == labels


def test_league_from_a_parent_equals_the_source_main(source_tree, tmp_path):
    torch.manual_seed(5)
    parent = RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), 64, 64))
    sha = save_checkpoint(tmp_path / "parent", parent)["weights"]["sha256"]
    p = plan(
        updates=6,
        rollout_steps=80,
        rehearsal_epochs=0,
        ppo=dataclasses.replace(plan().ppo, discount_time_unit=0.0),
        start_checkpoint=str(tmp_path / "parent"),
        start_sha256=sha,
    )
    run_league(p, tmp_path / "ours")
    argv = source_argv(p, tmp_path / "source", start=tmp_path / "parent")
    run_tree(source_tree, "", SOURCE_MAIN.format(argv=argv))
    assert_matches_source(tmp_path / "ours", tmp_path / "source", p.updates)
    registry = json.loads((tmp_path / "ours" / "registry.json").read_text())
    assert registry["bc_gate"]["reasons"] == ["candidate and incumbent are the same weights"]
    warm = load_checkpoint(tmp_path / "ours" / "warmstart", expected_sha256=sha)
    assert warm.manifest["training"]["bc"]["skipped"] is True
    assert warm.manifest["training"]["parent_policy_sha"] == sha


# --- the plan --------------------------------------------------------------------------------


def test_plan_has_no_defaults_and_fixed_bounds():
    with pytest.raises(TypeError):
        LeaguePlan()
    fields = {f.name for f in dataclasses.fields(LeaguePlan)}
    for name in fields:
        values = {f: getattr(plan(), f) for f in fields if f != name}
        with pytest.raises(TypeError):
            LeaguePlan(**values)
    for change in (
        {"seed": -1},
        {"seed": True},
        {"seed": EVALUATION_SEED_FLOOR - 10_000},
        {"updates": 0},
        {"updates": 65},
        {"bc_steps_per_drill": 0},
        {"bc": plan().ppo},
        {"bc": dataclasses.replace(plan().bc, reference_kl_coef=0.1)},
        {"bc_learning_rate": 0.0},
        {"rehearsal_learning_rate": float("nan")},
        {"rehearsal_epochs": 9},
        {"ppo": plan().bc},
        {"rollout_steps": 0},
        {"max_steps": 1201},
        {"evaluation_seed": EVALUATION_SEED_FLOOR - 1},
        {"evaluation_games_per_style": 18},
        {"evaluation_games_per_style": 21},
        {"evaluation_sampling": "argmax"},
        {"threads": 0},
        {"start_checkpoint": "/tmp/x"},
        {"start_sha256": "ab" * 32},
        {"start_checkpoint": "relative", "start_sha256": "ab" * 32},
    ):
        with pytest.raises(ValueError):
            plan(**change)
    assert plan().sha256 == plan().sha256 != plan(seed=1).sha256
    assert plan(threads=2).sha256 != plan().sha256


# --- simulator children ----------------------------------------------------------------------


def test_every_child_is_marked_as_simulator_trained(uninterrupted):
    p = plan()
    initial = None
    previous = None
    for label in (*LABELS, *(f"snapshots/update-{u:03d}" for u in range(1, p.updates + 1))):
        sha = checkpoint_shas(uninterrupted)[label]
        loaded = load_checkpoint(uninterrupted / label, expected_sha256=sha)
        training = loaded.manifest["training"]
        assert training["source"] == "simulator-curriculum-league"
        assert training["domain"] == SIM_DOMAIN
        assert training["simulator"] == {
            "simulator_sha256": SIMULATOR_SHA256,
            "teacher_sha256": TEACHER_SHA256,
            "plan_sha256": p.sha256,
        }
        assert training["live_gain_established"] is False
        assert training["live_consumed"] == [] and training["demonstrations_consumed"] == []
        assert loaded.manifest["automatic_promotion"] is False
        assert "lineage_dir" not in training  # never bound to a live selection lineage
        if label == "initial":
            assert training["simulator_trained"] is False and "parent_policy_sha" not in training
            initial = sha
        else:
            assert training["simulator_trained"] is True
            assert training["simulator_ancestry"] is True
            assert training["parent_policy_sha"] == (previous or initial)
            assert ancestry_simulator(loaded) == (True, training["simulator_consumed"])
        previous = sha if label != "initial" else None
    # The source league rule, duplicated incumbent included: [incumbent] + league[-3:].
    stages = uninterrupted / "resume"
    leagues = [json.loads((stages / f"stage-{k:03d}.json").read_text())["league"] for k in range(5)]
    assert leagues == [
        ["warmstart"],
        ["warmstart"],
        ["warmstart", "warmstart", "update-002"],
        ["warmstart", "warmstart", "update-002"],
        ["warmstart", "warmstart", "update-002", "update-004"],
    ]
    consumed = training["simulator_consumed"]
    assert [c[0] for c in consumed] == ["demonstrations"] * 4 + ["rollout"] * p.updates
    assert all(seed < EVALUATION_SEED_FLOOR for _, seed, _, _ in consumed)


def test_live_children_of_a_simulator_policy_stay_marked(tmp_path):
    record = [["rollout", 1001, "melee", 16]]
    parent = saved(
        tmp_path,
        training={
            "parent_policy_sha": "f" * 64,
            "live_consumed": [],
            "demonstrations_consumed": [],
            "simulator_trained": True,
            "simulator_ancestry": True,
            "simulator_consumed": record,
        },
    )
    result = train._unsafe_update_from_episodes(
        parent,
        [live_episode(parent, match=3)],
        verify_episode=lambda _: True,
        out=tmp_path / "child",
        config=SMALL,
        seed=1,
        lineage_dir=LINEAGE,
    )
    child = load_checkpoint(tmp_path / "child", expected_sha256=result["policy_sha"])
    assert child.manifest["training"]["simulator_ancestry"] is True
    assert child.manifest["training"]["simulator_consumed"] == record
    assert ancestry_simulator(child) == (True, record)
    # A plain live child is recorded as such.
    plain = saved(tmp_path, "plain")
    result = train._unsafe_update_from_episodes(
        plain,
        [live_episode(plain, match=4)],
        verify_episode=lambda _: True,
        out=tmp_path / "plain-child",
        config=SMALL,
        seed=1,
        lineage_dir=LINEAGE,
    )
    training = json.loads((tmp_path / "plain-child" / "manifest.json").read_text())["training"]
    assert training["simulator_ancestry"] is False and training["simulator_consumed"] == []
    # A marked checkpoint without its simulator record is refused, nothing written.
    for stripped in (
        {"simulator_trained": True},
        {"simulator_ancestry": True},
        {"simulator_trained": True, "simulator_consumed": [["rollout", 1]]},
    ):
        name = f"stripped-{len(list(tmp_path.iterdir()))}"
        bad = saved(tmp_path, name, training={"parent_policy_sha": "f" * 64, "live_consumed": [],
                                              "demonstrations_consumed": [], **stripped})  # fmt: skip
        with pytest.raises(ValueError, match="simulator record"):
            train._unsafe_update_from_episodes(
                bad, [live_episode(bad, match=9)], verify_episode=lambda _: True,
                out=tmp_path / "never", config=SMALL, seed=1, lineage_dir=LINEAGE,
            )  # fmt: skip
        assert not (tmp_path / "never").exists()


def test_registry_is_run_local_and_never_a_live_selection(uninterrupted, tmp_path):
    registry = json.loads((uninterrupted / "registry.json").read_text())
    assert registry["registry_kind"] == "run-local simulator registry"
    assert registry["selection_scope"] == "simulator-experimental-only"
    assert registry["simulator_only"] is True and registry["live_gain_established"] is False
    assert registry["public_deployment"] is False and registry["automatic_promotion"] is False
    vocabulary = {"registry_selected", "rollback", "rejected", "not_gated"}
    entries = registry["checkpoints"].values()
    assert {e["selection"] for e in entries} <= vocabulary
    assert [e["selection"] for e in entries].count("registry_selected") == 1
    assert {e["deployment"] for e in entries} == {"not_deployed"}
    assert registry["checkpoints"][registry["selected"]]["sha256"] == registry["selected_sha"]
    for name in ("bc_gate", "ppo_gate", "selection_gate"):
        gate = registry[name]
        assert gate["simulator_only"] is True and gate["live_gain_established"] is False
        if "schema" in gate:
            with pytest.raises(evaluation.EvaluationRefused):
                evaluation.record_decision(tmp_path / "lineage", gate)
    assert not (tmp_path / "lineage" / evaluation.SELECTION_RECORD).exists()


def test_league_leaves_the_callers_rng_and_threads_untouched(tmp_path):
    torch.manual_seed(31)
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    run_league(plan(updates=1, threads=2), tmp_path / "run")
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.get_num_threads() == threads


# --- exact resume ----------------------------------------------------------------------------


class Crash(RuntimeError):
    pass


def crash_on_call(monkeypatch, target, name, number):
    original = getattr(target, name)
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(1)
        if len(calls) == number:
            raise Crash(f"{name} call {number}")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, wrapped)


def test_resume_after_a_crash_is_bit_identical_and_idempotent(uninterrupted, tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        crash_on_call(patch, league, "ppo_update", 3)  # during update 3 of 4
        with pytest.raises(Crash):
            run_league(plan(), tmp_path / "crashed")
    assert not (tmp_path / "crashed" / "registry.json").exists()
    assert (tmp_path / "crashed" / "resume" / "stage-002.json").exists()
    expected = outcome(uninterrupted)
    before = {p: p.read_bytes() for p in (tmp_path / "crashed").rglob("*") if p.is_file()}
    run_league(plan(), tmp_path / "resumed", resume_from=tmp_path / "crashed")
    assert outcome(tmp_path / "resumed") == expected
    # The previous run is read only, and resuming it again gives the same result.
    assert {p: p.read_bytes() for p in (tmp_path / "crashed").rglob("*") if p.is_file()} == before
    run_league(plan(), tmp_path / "again", resume_from=tmp_path / "crashed")
    assert outcome(tmp_path / "again") == expected
    # Resuming a finished run only re-runs the evaluation, with the same result.
    run_league(plan(), tmp_path / "finished", resume_from=tmp_path / "resumed")
    assert outcome(tmp_path / "finished") == expected
    progress = json.loads((tmp_path / "resumed" / "progress.json").read_text())
    assert progress["resumed_from"] == str((tmp_path / "crashed").resolve())


def test_resume_chained_through_two_crashes_is_bit_identical(uninterrupted, tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        crash_on_call(patch, league, "ppo_update", 2)
        with pytest.raises(Crash):
            run_league(plan(), tmp_path / "first")
    with monkeypatch.context() as patch:
        crash_on_call(patch, league, "ppo_update", 2)  # update 3, the second after resume
        with pytest.raises(Crash):
            run_league(plan(), tmp_path / "second", resume_from=tmp_path / "first")
    run_league(plan(), tmp_path / "third", resume_from=tmp_path / "second")
    assert outcome(tmp_path / "third") == outcome(uninterrupted)


@pytest.mark.parametrize(
    "where",
    ["stage manifest", "optimizer blob", "snapshot checkpoint", "demonstrations", "evaluation"],
)
def test_a_crash_mid_write_resumes_from_the_last_complete_stage(
    uninterrupted, tmp_path, monkeypatch, where
):
    with monkeypatch.context() as patch:
        if where == "stage manifest":
            original = league.atomic_json

            def atomic_json(path, value):
                if path.name == "stage-002.json":
                    raise Crash("before the stage manifest")
                return original(path, value)

            patch.setattr(league, "atomic_json", atomic_json)
        elif where == "optimizer blob":
            crash_on_call(patch, league, "_blob", 5)  # a blob of stage 1
        elif where == "snapshot checkpoint":
            crash_on_call(patch, league, "save_checkpoint", 5)  # update-003
        elif where == "demonstrations":
            original = league.atomic_json

            def atomic_json(path, value):
                if path.name == "demonstrations.json":
                    raise Crash("before demonstrations")
                return original(path, value)

            patch.setattr(league, "atomic_json", atomic_json)
        else:
            crash_on_call(patch, league, "_suite_rows", 2)
        with pytest.raises(Crash):
            run_league(plan(), tmp_path / "crashed")
    run_league(plan(), tmp_path / "resumed", resume_from=tmp_path / "crashed")
    assert outcome(tmp_path / "resumed") == outcome(uninterrupted)


def tamper(root: Path, where: str) -> None:
    if where == "optimizer":
        blob = next((root / "resume" / "stage-004").glob("ppo-optimizer-*.pt"))
        blob.write_bytes(blob.read_bytes() + b"\0")
    elif where == "rng":
        blob = next((root / "resume" / "stage-002").glob("rng-*.pt"))
        blob.write_bytes(blob.read_bytes()[:-1] + b"\1")
    elif where == "snapshot":
        snapshot = root / "snapshots" / "update-002"
        manifest = json.loads((snapshot / "manifest.json").read_text())
        target = snapshot / manifest["weights"]["file"]
        target.write_bytes(target.read_bytes()[:-1] + b"\1")
    elif where == "demonstrations":
        path = root / "demonstrations.json"
        path.write_text(path.read_text().replace("true", "false", 1))
    elif where == "stage plan":
        path = root / "resume" / "stage-001.json"
        state = json.loads(path.read_text())
        state["plan_sha256"] = "0" * 64
        path.write_text(json.dumps(state))
    elif where == "missing middle stage":
        (root / "resume" / "stage-001.json").unlink()


@pytest.mark.parametrize(
    "where",
    ["optimizer", "rng", "snapshot", "demonstrations", "stage plan", "missing middle stage"],
)
def test_resume_refuses_tampered_state(uninterrupted, tmp_path, where):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(uninterrupted, copy)
    tamper(copy, where)
    with pytest.raises((LeagueRefused, ValueError)):
        run_league(plan(), tmp_path / "resumed", resume_from=copy)
    assert not (tmp_path / "resumed").exists()


def test_resume_refuses_another_plan_and_overlapping_outputs(uninterrupted, tmp_path):
    for other in (plan(seed=7), plan(rehearsal_epochs=0), plan(threads=2), plan(updates=5)):
        with pytest.raises(LeagueRefused, match="another plan or configuration"):
            run_league(other, tmp_path / "resumed", resume_from=uninterrupted)
    for out in (uninterrupted, uninterrupted / "inner", uninterrupted.parent):
        with pytest.raises(LeagueRefused):
            run_league(plan(), out, resume_from=uninterrupted)
    (tmp_path / "taken").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "missing")
    for taken in (tmp_path / "taken", tmp_path / "link"):
        with pytest.raises(LeagueRefused, match="already exists"):
            run_league(plan(), taken)
    assert not (tmp_path / "missing").exists() and not list((tmp_path / "taken").iterdir())
    with pytest.raises(LeagueRefused, match="absolute"):
        run_league(plan(), Path("relative-run"))
    with pytest.raises(LeagueRefused, match="absolute"):
        run_league(plan(), tmp_path / "x", resume_from=Path("relative-run"))
    with pytest.raises(LeagueRefused, match="LeaguePlan"):
        run_league(plan().to_json(), tmp_path / "x")
    assert not (tmp_path / "resumed").exists() and not (tmp_path / "x").exists()


def test_a_run_that_crashed_before_its_first_stage_restarts_from_the_seed(
    uninterrupted, tmp_path, monkeypatch
):
    with monkeypatch.context() as patch:
        crash_on_call(patch, league, "behavioral_clone", 1)
        with pytest.raises(Crash):
            run_league(plan(), tmp_path / "crashed")
    assert not list((tmp_path / "crashed").glob("resume/stage-*.json"))
    run_league(plan(), tmp_path / "resumed", resume_from=tmp_path / "crashed")
    assert outcome(tmp_path / "resumed") == outcome(uninterrupted)


def test_league_module_is_not_imported_implicitly():
    import subprocess
    import sys

    code = (
        "import sys, anima3, anima3.neural, anima3.neural.sim_evaluation;"
        "assert 'anima3.neural.league' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[1])
