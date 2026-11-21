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
from anima3.neural.league import LeaguePlan, LeagueRefused, run_league
from anima3.neural.model import PolicyConfig, RecurrentPolicy
from anima3.neural.schema import ACTION_NAMES, FEATURE_NAMES
from anima3.neural.sim_evaluation import EVALUATION_SEED_FLOOR
from anima3.neural.teacher import TEACHER_SHA256
from anima3.neural.train import ancestry_simulator

LABELS = ("initial", "warmstart")


def plan(**overrides) -> LeaguePlan:
    values = {
        "seed": 20261006,
        "bc_steps_per_drill": 8,
        # The source league's settings, every one stated.
        "bc_epochs": 1,
        "bc_sequence_length": 64,
        "bc_batch_sequences": 8,
        "bc_balance_actions": False,
        "bc_learning_rate": 3e-4,
        "bc_max_grad_norm": 0.5,
        "rehearsal_epochs": 1,
        "rehearsal_learning_rate": 1e-4,
        "ppo_epochs": 4,
        "ppo_sequence_length": 64,
        "ppo_burn_in": 8,
        "ppo_batch_sequences": 8,
        "ppo_learning_rate": 1e-4,
        "discount_time_unit": 1.0,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "ppo_max_grad_norm": 0.5,
        "target_kl": 0.04,
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
        "--bc-steps-per-drill", str(p.bc_steps_per_drill), "--bc-epochs", str(p.bc_epochs),
        "--bc-rehearsal-epochs", str(p.rehearsal_epochs),
        "--discount-time-unit", str(p.discount_time_unit),
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
        discount_time_unit=0.0,
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
        {"bc_epochs": 0},
        {"bc_sequence_length": 4097},
        {"bc_balance_actions": None},
        {"bc_learning_rate": 0.0},
        {"bc_max_grad_norm": 0.0},
        {"rehearsal_learning_rate": float("nan")},
        {"rehearsal_epochs": 9},
        {"ppo_epochs": 65},
        {"ppo_burn_in": -1},
        {"ppo_learning_rate": 0.5},
        {"discount_time_unit": -1.0},
        {"discount_time_unit": None},
        {"gamma": 0.0},
        {"gae_lambda": 1.5},
        {"clip_range": 1.0},
        {"value_coef": -0.1},
        {"entropy_coef": float("inf")},
        {"ppo_max_grad_norm": 0.0},
        {"target_kl": 0.0},
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
    # Every value the run uses is a plan field: the derived configurations have no
    # setting of their own that the plan does not state.
    p = plan()
    assert p.ppo.learning_rate == 1e-4 and p.ppo.gamma == 0.995
    assert p.bc.learning_rate == 3e-4 and p.rehearsal.learning_rate == 1e-4
    assert p.rehearsal.epochs == 1 and p.bc.reference_kl_coef == 0
    assert {"bc", "ppo"}.isdisjoint(p.to_json())


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
        {
            "simulator_trained": True,
            "simulator_ancestry": True,
            "simulator_consumed": [["r", 1, "d"]],
        },
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


def _stage_path(root: Path, stage: int) -> Path:
    return root / "resume" / f"stage-{stage:03d}.json"


def rebind(root: Path) -> None:
    """Re-derive every hash a stage records, as a coordinated forger would.

    Used to isolate one check: after an edit, the checkpoint-manifest hashes and the
    stage chain are recomputed, so only the content check under test can refuse.
    """
    previous = None
    for stage in range(5):
        path = _stage_path(root, stage)
        state = json.loads(path.read_text())
        state["checkpoint_manifests"] = {
            label: hashlib.sha256(
                (root / ("snapshots/" + label if label.startswith("update-") else label)
                 / "manifest.json").read_bytes()
            ).hexdigest()
            for label in state["checkpoint_manifests"]
        }  # fmt: skip
        state["previous_stage_sha256"] = previous
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        previous = hashlib.sha256(path.read_bytes()).hexdigest()


def edit_stage(root: Path, stage: int, change, *, chain: bool = True) -> None:
    path = _stage_path(root, stage)
    state = json.loads(path.read_text())
    change(state, root, stage)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    if chain:
        rebind(root)


def _swap(state, root, stage):
    optimizers = state["optimizers"]
    optimizers["ppo"], optimizers["rehearsal"] = optimizers["rehearsal"], optimizers["ppo"]


def _older_rng(state, root, stage):
    # The reviewer's probe: stage 0's RNG blob copied into this stage and named there.
    older = json.loads(_stage_path(root, 0).read_text())["rng"]
    source = root / "resume" / "stage-000" / older["file"]
    (root / "resume" / f"stage-{stage:03d}" / older["file"]).write_bytes(source.read_bytes())
    state["rng"] = older


def _outside_rng(state, root, stage):
    outside = root.parent / "outside"
    outside.mkdir(exist_ok=True)
    blob = root / "resume" / f"stage-{stage:03d}" / state["rng"]["file"]
    (outside / state["rng"]["file"]).write_bytes(blob.read_bytes())
    state["rng"] = {**state["rng"], "file": "../../../outside/" + state["rng"]["file"]}


def _renamed_blob(state, root, stage):
    descriptor = state["optimizers"]["ppo"]
    directory = root / "resume" / f"stage-{stage:03d}"
    renamed = descriptor["file"].replace("ppo-optimizer", "rehearsal-optimizer")
    (directory / renamed).write_bytes((directory / descriptor["file"]).read_bytes())
    state["optimizers"]["rehearsal"] = {**descriptor, "file": renamed}


def _model_initial(state, root, stage):
    state["model"] = {"label": "initial", "sha256": state["checkpoints"]["initial"]}


def _other_sha(state, root, stage):
    state["checkpoints"]["initial"] = state["checkpoints"]["warmstart"]


def _drop_label(state, root, stage):
    del state["checkpoints"]["update-001"]


def _manifest(root: Path, label: str, change) -> None:
    path = root / label / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _flip(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-1] + bytes([path.read_bytes()[-1] ^ 1]))


def _weights_file(directory: Path) -> Path:
    return directory / json.loads((directory / "manifest.json").read_text())["weights"]["file"]


def _plan_code(root: Path, *, drop: bool = False, **values) -> None:
    path = root / "plan.json"
    recorded = json.loads(path.read_text())
    if drop:
        del recorded["code"]
    else:
        recorded["code"].update(values)
    path.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")


def _set(**values):
    return lambda state, root, stage: state.update(values)


# Each edit is refused by the check named in its expected message. Stage edits are
# re-chained and re-bound (``rebind``) unless the hash chain itself is under test,
# so no other check can catch them first.
TAMPER = {
    "optimizer bytes": (
        lambda r: _flip(next((r / "resume/stage-004").glob("ppo-optimizer-*"))),
        "does not hash",
    ),
    "rng bytes": (
        lambda r: _flip(next((r / "resume/stage-002").glob("rng-*.pt"))),
        "does not hash",
    ),
    "snapshot weights": (
        lambda r: _flip(_weights_file(r / "snapshots/update-002")),
        "SHA256|SHA-256",
    ),
    "demonstrations": (
        lambda r: (r / "demonstrations.json").write_text(
            (r / "demonstrations.json").read_text().replace("true", "false", 1)
        ),
        "does not hash",
    ),
    "stage plan": (
        lambda r: edit_stage(r, 1, _set(plan_sha256="0" * 64)),
        "belongs to another plan",
    ),
    "missing middle stage": (lambda r: _stage_path(r, 1).unlink(), "not contiguous"),
    "stage chain broken": (
        lambda r: (
            edit_stage(r, 1, _set(), chain=False)
            or _stage_path(r, 1).write_text(json.dumps(json.loads(_stage_path(r, 1).read_text())))
        ),
        "does not follow the previous stage",
    ),
    "extra stage key": (lambda r: edit_stage(r, 2, _set(note=1)), "belongs to another plan"),
    "swapped optimizer descriptors": (lambda r: edit_stage(r, 1, _swap), "foreign"),
    "renamed optimizer blob": (lambda r: edit_stage(r, 1, _renamed_blob), "another stage or plan"),
    "older stage rng blob": (lambda r: edit_stage(r, 1, _older_rng), "another stage or plan"),
    "rng outside the run": (lambda r: edit_stage(r, 1, _outside_rng), "foreign"),
    "league labels": (
        lambda r: edit_stage(r, 2, _set(league=["warmstart", "warmstart"])),
        "league does not match",
    ),
    "model set to initial": (lambda r: edit_stage(r, 1, _model_initial), "model does not match"),
    "head model set to initial": (lambda r: edit_stage(r, 4, _model_initial), "model does not"),
    "events emptied": (lambda r: edit_stage(r, 1, _set(events=[])), "events does not match"),
    "head events emptied": (lambda r: edit_stage(r, 4, _set(events=[])), "events does not"),
    "simulator record emptied": (
        lambda r: edit_stage(r, 1, _set(simulator_consumed=[])),
        "simulator_consumed does not match",
    ),
    "stage checkpoint SHA replaced": (
        lambda r: edit_stage(r, 1, _other_sha),
        "checkpoints does not match",
    ),
    "checkpoint label dropped": (
        lambda r: edit_stage(r, 4, _drop_label),
        "does not name the run's checkpoints",
    ),
    "unbound checkpoint manifest": (
        lambda r: _manifest(r, "warmstart", lambda m: m.update(created_at="1970")),
        "checkpoint_manifests does not match",
    ),
    "head blob size edited": (
        lambda r: edit_stage(
            r, 4, lambda state, root, stage: state["rng"].update(bytes=state["rng"]["bytes"] + 1)
        ),
        "rng blob size differs",
    ),
    "middle blob size not an integer": (
        lambda r: edit_stage(
            r,
            2,
            lambda state, root, stage: state["optimizers"]["ppo"].update(
                bytes=float(state["optimizers"]["ppo"]["bytes"])
            ),
        ),
        "ppo-optimizer blob size differs",
    ),
    "stage code identity": (
        lambda r: edit_stage(
            r, 2, lambda state, root, stage: state["code"].update(league_sha256="0" * 64)
        ),
        "stage 2 was written by other code",
    ),
    "head stage teacher identity": (
        lambda r: edit_stage(
            r, 4, lambda state, root, stage: state["code"].update(teacher_sha256="0" * 64)
        ),
        "stage 4 was written by other code",
    ),
    "plan code identity": (
        lambda r: _plan_code(r, simulator_sha256="0" * 64),
        "other league, simulator or teacher code",
    ),
    "plan code identity dropped": (
        lambda r: _plan_code(r, drop=True),
        "other league, simulator or teacher code",
    ),
    "checkpoint teacher identity re-bound": (
        lambda r: (
            _manifest(
                r,
                "snapshots/update-002",
                lambda m: m["training"]["simulator"].update(teacher_sha256="0" * 64),
            )
            or rebind(r)
        ),
        "update-002 was made with another simulator, teacher or plan",
    ),
    "checkpoint simulator identity re-bound": (
        lambda r: (
            _manifest(
                r, "initial", lambda m: m["training"]["simulator"].update(simulator_sha256="0" * 64)
            )
            or rebind(r)
        ),
        "initial was made with another simulator, teacher or plan",
    ),
    "checkpoint record re-bound": (
        lambda r: (
            _manifest(
                r,
                "snapshots/update-001",
                lambda m: m["training"].update(live_consumed=[["a" * 32, 1]]),
            )
            or rebind(r)
        ),
        "does not carry this run's record",
    ),
}


@pytest.mark.parametrize("where", sorted(TAMPER))
def test_resume_refuses_tampered_state(uninterrupted, tmp_path, where):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(uninterrupted, copy)
    edit, error = TAMPER[where]
    edit(copy)
    with pytest.raises((LeagueRefused, ValueError), match=error):
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


def test_the_league_code_identity_covers_the_modules_that_decide_a_result(tmp_path):
    import shutil

    from anima3.neural import teacher

    package = Path(league.__file__).resolve().parents[1]
    assert set(league.LEAGUE_SOURCES) == {
        "neural/league.py", "neural/train.py", "neural/imitation.py", "neural/sim_rollouts.py",
        "neural/sim_evaluation.py", "neural/evaluation.py", "neural/model.py",
        "neural/checkpoints.py", "stats.py",
    }  # fmt: skip
    assert league.LEAGUE_SOURCES == teacher.source_hashes(league.LEAGUE_MODULES, package)
    assert league.LEAGUE_SHA256 == teacher.source_identity(league.LEAGUE_SOURCES)
    assert league.code_identity() == {
        "league_sha256": league.LEAGUE_SHA256,
        "league_sources": league.LEAGUE_SOURCES,
        "simulator_sha256": SIMULATOR_SHA256,
        "teacher_sha256": TEACHER_SHA256,
    }
    copy = tmp_path / "anima3"
    for module in league.LEAGUE_MODULES:
        (copy / module).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(package / module, copy / module)
    assert teacher.source_identity(teacher.source_hashes(league.LEAGUE_MODULES, copy)) == (
        league.LEAGUE_SHA256
    )
    for module in league.LEAGUE_MODULES:
        original = (copy / module).read_bytes()
        (copy / module).write_bytes(original + b"\n")
        changed = teacher.source_identity(teacher.source_hashes(league.LEAGUE_MODULES, copy))
        assert changed != league.LEAGUE_SHA256, module
        (copy / module).write_bytes(original)


def test_plan_stages_and_registry_record_the_code_identity(uninterrupted):
    code = league.code_identity()
    assert json.loads((uninterrupted / "plan.json").read_text())["code"] == code
    for stage in range(5):
        assert json.loads(_stage_path(uninterrupted, stage).read_text())["code"] == code
    assert json.loads((uninterrupted / "registry.json").read_text())["code"] == code


@pytest.mark.parametrize("name", ["LEAGUE_SHA256", "SIMULATOR_SHA256", "TEACHER_SHA256"])
def test_resume_refuses_a_run_made_with_other_code(uninterrupted, tmp_path, monkeypatch, name):
    # As if league.py, train.py, the simulator or the teacher had been edited since.
    monkeypatch.setattr(league, name, "e" * 64)
    with pytest.raises(LeagueRefused, match="other league, simulator or teacher code"):
        run_league(plan(), tmp_path / "resumed", resume_from=uninterrupted)
    assert not (tmp_path / "resumed").exists()


def test_reformatting_the_last_stage_file_is_the_one_accepted_edit(
    uninterrupted, tmp_path, monkeypatch
):
    """The documented limit: nothing hashes the last stage file, so a re-serialisation of
    the same JSON is accepted; every value in it is still checked, so the result is equal."""
    with monkeypatch.context() as patch:
        crash_on_call(patch, league, "ppo_update", 3)
        with pytest.raises(Crash):
            run_league(plan(), tmp_path / "crashed")
    head = _stage_path(tmp_path / "crashed", 2)
    head.write_text(json.dumps(json.loads(head.read_text()), separators=(",", ":")))
    run_league(plan(), tmp_path / "resumed", resume_from=tmp_path / "crashed")
    assert outcome(tmp_path / "resumed") == outcome(uninterrupted)


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaves(item, (*path, index))
    else:
        yield path, value


def _type_edits(state):
    """One edit per kind that Python's == hides: bool to int, int to float, float to int."""
    found = {}
    for path, value in _leaves(state):
        if path[0] in ("plan_sha256", "previous_stage_sha256"):
            continue
        if type(value) is bool:
            found.setdefault("bool", (path, int(value)))
        elif type(value) is int:
            found.setdefault(f"int {path[0]}", (path, float(value)))
        elif type(value) is float and value.is_integer():
            found.setdefault("float", (path, int(value)))
    return found


def test_type_edits_of_the_last_stage_and_plan_are_refused(tmp_path, monkeypatch):
    """Only a reformatting of the last stage is accepted: 0, 0.0 and false are distinct."""
    import shutil

    with monkeypatch.context() as patch:
        crash_on_call(patch, league, "ppo_update", 3)
        with pytest.raises(Crash):
            run_league(plan(), tmp_path / "crashed")
    head = _stage_path(tmp_path / "crashed", 2)
    original = head.read_text()
    state = json.loads(original)
    edits = _type_edits(state)
    assert {"bool", "float", "int stage", "int events", "int simulator_consumed"} <= set(edits)
    for number, (kind, (path, value)) in enumerate(sorted(edits.items())):
        edited = json.loads(original)
        target = edited
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        head.write_text(json.dumps(edited, indent=2, sort_keys=True) + "\n")
        expected = {"int stage": "another plan", "int optimizers": "size", "int rng": "size"}
        with pytest.raises(LeagueRefused, match=expected.get(kind, "does not match")):
            run_league(plan(), tmp_path / f"resumed-{number}", resume_from=tmp_path / "crashed")
        assert not (tmp_path / f"resumed-{number}").exists(), kind
    head.write_text(original)
    for field, value in (("updates", 4.0), ("bc_balance_actions", 0), ("rehearsal_epochs", True)):
        copy = tmp_path / f"plan-{field}"
        shutil.copytree(tmp_path / "crashed", copy)
        recorded = json.loads((copy / "plan.json").read_text())
        recorded["plan"][field] = value
        (copy / "plan.json").write_text(json.dumps(recorded))
        with pytest.raises(LeagueRefused, match="another plan or configuration"):
            run_league(plan(), tmp_path / "never", resume_from=copy)
    assert not (tmp_path / "never").exists()


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


def test_league_children_carry_the_parent_records(tmp_path):
    """Simulator data consumes no live evidence; the parent's records carry over."""
    live, demos = [["a" * 32, 5]], [["b" * 32, 6]]
    inherited = [["rollout", 7, "full", 3]]
    torch.manual_seed(9)
    parent = RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), 64, 64))
    training = {
        "parent_policy_sha": "f" * 64,
        "live_consumed": live,
        "demonstrations_consumed": demos,
        "simulator_trained": True,
        "simulator_ancestry": True,
        "simulator_consumed": inherited,
    }
    sha = save_checkpoint(tmp_path / "parent", parent, training=training)["weights"]["sha256"]
    p = plan(updates=2, start_checkpoint=str(tmp_path / "parent"), start_sha256=sha)
    with pytest.MonkeyPatch.context() as patch:
        crash_on_call(patch, league, "ppo_update", 2)
        with pytest.raises(Crash):
            run_league(p, tmp_path / "crashed")
    run_league(p, tmp_path / "run", resume_from=tmp_path / "crashed")
    shas = checkpoint_shas(tmp_path / "run")
    for label, sha_ in shas.items():
        child = load_checkpoint(tmp_path / "run" / label, expected_sha256=sha_)
        record = child.manifest["training"]
        assert record["live_consumed"] == live and record["demonstrations_consumed"] == demos
        assert record["simulator_ancestry"] is True
        assert record["simulator_consumed"][: len(inherited)] == inherited
        assert ancestry_simulator(child)[0] is True
    assert shas["initial"] == sha
    final = load_checkpoint(tmp_path / "run" / "snapshots/update-002", expected_sha256=shas[
        "snapshots/update-002"
    ]).manifest["training"]  # fmt: skip
    assert len(final["simulator_consumed"]) == len(inherited) + 4 + 2


@pytest.mark.parametrize(
    "training,error",
    [
        ({"simulator_trained": "true", "simulator_ancestry": True, "simulator_consumed": []},
         "malformed simulator marker"),
        ({"simulator_trained": False, "simulator_ancestry": 1, "simulator_consumed": []},
         "malformed simulator marker"),
        ({"source": "simulator-curriculum-league"}, "lacks its simulator record"),
        ({"domain": SIM_DOMAIN, "simulator_trained": False, "simulator_ancestry": False},
         "lacks its simulator record"),
        ({"simulator_trained": True, "simulator_ancestry": False, "simulator_consumed": []},
         "ancestry marker"),
        ({"simulator_consumed": [["rollout", 1, "full", 2]]}, "unmarked"),
        ({"simulator_trained": True}, "simulator record"),
    ],
)  # fmt: skip
def test_malformed_simulator_markers_are_refused_not_dropped(tmp_path, training, error):
    bad = saved(tmp_path, training={"parent_policy_sha": "f" * 64, "live_consumed": [],
                                    "demonstrations_consumed": [], **training})  # fmt: skip
    with pytest.raises(ValueError, match=error):
        ancestry_simulator(bad)
    with pytest.raises(ValueError, match=error):
        train._unsafe_update_from_episodes(
            bad, [live_episode(bad, match=11)], verify_episode=lambda _: True,
            out=tmp_path / "never", config=SMALL, seed=1, lineage_dir=LINEAGE,
        )  # fmt: skip
    assert not (tmp_path / "never").exists()


def test_the_train_module_names_the_simulator_domain():
    assert train.SIMULATOR_DOMAIN == SIM_DOMAIN
    assert train.SIMULATOR_SOURCE == league.SOURCE
