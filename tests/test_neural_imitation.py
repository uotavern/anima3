"""Unit-8 behaviour cloning on verified teacher demonstrations; tiny CPU loops under the neural extra.

The imitation-metric, action-balancing and reference-regularisation cases come
from the training branch's ``test_live_demonstrations.py``; the rest cover the
unit-8 boundaries: explicit budgets and seeds, demonstrations that are never
PPO input, SHA-bound output in a fresh directory with full provenance, training
collections only, validation disjoint from the fitted input, and lineage
consumption.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_neural_teacher import TeacherHarness, demo_files, teacher_plan, teacher_step
from test_neural_training import SMALL, policy, saved, tree_digest

from anima3.neural import live
from anima3.neural.checkpoints import load_checkpoint
from anima3.neural.demonstrations import DemonstrationRefused, select_demonstrations
from anima3.neural.imitation import (
    NO_VALIDATION,
    ImitationConfig,
    _clone_selection,
    behavioral_clone,
    clone_from_demonstrations,
    demonstration_trajectory,
    imitation_metrics,
)
from anima3.neural.live import LIVE_TEACHER_SHA256, Arm, CollectionPlan
from anima3.neural.model import PolicyConfig, RecurrentPolicy
from anima3.neural.schema import ACTION_NAMES, FEATURE_NAMES
from anima3.neural.splits import NO_HELD_OUT
from anima3.neural.train import ppo_update

BC = ImitationConfig(epochs=2, sequence_length=4, batch_sequences=2, balance_actions=False)
LINEAGE = Path("/nonexistent/anima3-imitation-lineage")


def demo(steps):
    return {"verification": {"match_id": "a" * 32}, "actor": 2, "steps": steps}


def two_choice(at, act):
    step = teacher_step(at, 1)
    step["mask"] = [True, True] + [False] * (len(ACTION_NAMES) - 2)
    step["action"] = act
    return step


def hold_only(at):
    step = teacher_step(at, 1)
    step["mask"] = [True] + [False] * (len(ACTION_NAMES) - 1)
    step["action"] = 0
    return step


def model(hidden=8):
    return RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), hidden, hidden))


# --- ported imitation core ---------------------------------------------------------------------


def test_demonstrations_are_bc_only_and_never_ppo_trajectories():
    episode = demonstration_trajectory(demo([hold_only(1.0)]), 64)
    assert len(episode) == 1 and len(episode.hidden[0]) == 64
    assert episode.info["ppo_eligible"] is False
    with pytest.raises(ValueError, match="not PPO"):
        ppo_update(model(), None, [episode], SMALL, seed=0)


def test_imitation_metrics_excludes_forced_holds():
    m = model()
    with torch.no_grad():
        m.policy_head.weight.zero_()
        m.policy_head.bias.zero_()
        m.policy_head.bias[0] = 5
    metrics = imitation_metrics(
        m, [demonstration_trajectory(demo([hold_only(1.0), two_choice(2.0, 1)]), 8)]
    )
    assert metrics["free_choices"] == 1
    assert metrics["accuracy"] == 0
    assert metrics["cross_entropy"] > 5
    assert "hold" not in metrics["per_action"]
    distribution = metrics["action_distribution"]
    assert distribution["hold"]["label_share"] == 0
    assert distribution["hold"]["argmax_share"] == 1
    assert distribution["hold"]["mean_probability"] == pytest.approx(1 / (1 + math.exp(-5)))
    assert distribution[ACTION_NAMES[1]]["label_share"] == 1
    assert distribution[ACTION_NAMES[2]]["mean_probability"] == 0
    assert distribution[ACTION_NAMES[2]]["legal_share"] == 0
    assert sum(x["mean_probability"] for x in distribution.values()) == pytest.approx(1)
    assert metrics["unobserved_label_probability"] == pytest.approx(
        distribution["hold"]["mean_probability"]
    )
    assert imitation_metrics(m, [])["accuracy"] is None
    assert imitation_metrics(m, [])["unobserved_label_probability"] is None


def test_balanced_imitation_does_not_count_forced_holds():
    steps = [hold_only(float(i)) for i in range(20)]
    steps += [two_choice(float(i), a) for i, a in enumerate((0, 0, 1), 20)]
    episode = demonstration_trajectory(demo(steps), 8)
    m = model()
    config = ImitationConfig(epochs=1, sequence_length=4, batch_sequences=2, balance_actions=True)
    result = behavioral_clone(m, torch.optim.Adam(m.parameters()), [episode], config, seed=0)
    weights = result["action_weights"]
    assert weights[1] / weights[0] == pytest.approx(2**0.5)
    assert result["bc_updates"] > 0


def test_reference_regularization_preserves_distribution_and_reference():
    torch.manual_seed(19)
    steps = [hold_only(-1.0)] + [two_choice(float(i), 1) for i in range(7)]
    episode = demonstration_trajectory(demo(steps), 8)
    reference = model()
    before = copy.deepcopy(reference.state_dict())
    probabilities = []
    for coefficient in (0.0, 10.0):
        student = copy.deepcopy(reference)
        config = ImitationConfig(
            epochs=20, sequence_length=3, batch_sequences=8, balance_actions=False,
            learning_rate=0.01, reference_kl_coef=coefficient,
        )  # fmt: skip
        metrics = behavioral_clone(
            student, torch.optim.Adam(student.parameters(), lr=0.01), [episode], config,
            seed=5, reference_model=reference,
        )  # fmt: skip
        probabilities.append(
            imitation_metrics(student, [episode])["action_distribution"][ACTION_NAMES[1]][
                "mean_probability"
            ]
        )
        if coefficient:
            assert math.isfinite(metrics["reference_kl_mean"]) and metrics["reference_kl_mean"] >= 0
        assert all(torch.equal(before[k], v) for k, v in reference.state_dict().items())
        assert all(p.grad is None for p in reference.parameters())
    assert probabilities[0] > probabilities[1] + 0.1
    assert probabilities[1] > 0.5  # Still learns the demonstrated action.
    for coefficient in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            ImitationConfig(
                epochs=1, sequence_length=1, batch_sequences=1, balance_actions=False,
                reference_kl_coef=coefficient,
            )  # fmt: skip
    config = ImitationConfig(
        epochs=1, sequence_length=1, batch_sequences=1, balance_actions=False, reference_kl_coef=1.0
    )
    with pytest.raises(ValueError, match="reference model required"):
        behavioral_clone(reference, None, [episode], config, seed=0)


# --- explicit budgets --------------------------------------------------------------------------


def test_imitation_config_has_no_default_budget():
    with pytest.raises(TypeError):
        ImitationConfig()
    with pytest.raises(TypeError):
        ImitationConfig(epochs=1, sequence_length=1, batch_sequences=1)
    for field, value in (
        ("epochs", 0), ("epochs", 65), ("epochs", True), ("sequence_length", 4097),
        ("batch_sequences", 0), ("balance_actions", 1), ("learning_rate", 0.0),
        ("max_grad_norm", 0.0),
    ):  # fmt: skip
        values = {"epochs": 1, "sequence_length": 1, "batch_sequences": 1,
                  "balance_actions": False, field: value}  # fmt: skip
        with pytest.raises(ValueError):
            ImitationConfig(**values)


def test_behavioral_clone_needs_a_seed_config_and_demonstrations():
    episode = demonstration_trajectory(demo([two_choice(1.0, 1)]), 8)
    m = model()
    with pytest.raises(TypeError):
        behavioral_clone(m, torch.optim.Adam(m.parameters()), [episode], BC)
    with pytest.raises(ValueError, match="seed"):
        behavioral_clone(m, torch.optim.Adam(m.parameters()), [episode], BC, seed=-1)
    with pytest.raises(TypeError, match="ImitationConfig"):
        behavioral_clone(m, torch.optim.Adam(m.parameters()), [episode], SMALL, seed=0)
    episode.info["ppo_eligible"] = True
    with pytest.raises(ValueError, match="not PPO"):
        behavioral_clone(m, torch.optim.Adam(m.parameters()), [episode], BC, seed=0)


# --- end to end on real unit-6 teacher collections ---------------------------------------------


def collect(tmp_path, monkeypatch, name, games=2, *, purpose="training", **kw):
    harness = TeacherHarness(monkeypatch, games, steps=6, **kw)
    plan = CollectionPlan(
        endpoint="lab.invalid:2593",
        accounts=("alpha", "beta"),
        arms=(Arm("scripted"), teacher_plan(tmp_path).arms[1]),
        matches=games,
        require_base_template=True,
        purpose=purpose,
    )
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


def clone(loaded, collections, out, **overrides):
    values = {
        "expected_sha256": loaded.policy_sha,
        "collections": collections,
        "held_out": NO_HELD_OUT,
        "validation": NO_VALIDATION,
        "teacher_sha256": LIVE_TEACHER_SHA256,
        "out": out,
        "config": BC,
        "seed": 11,
        "lineage_dir": LINEAGE,
        "min_demonstrations": 1,
        **overrides,
    }
    return clone_from_demonstrations(loaded.directory, **values)


def test_clone_writes_a_sha_bound_checkpoint_with_full_provenance(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect(tmp_path, monkeypatch, "train", invalid={1: ["teacher_action_rejected"]})
    checked = collect(tmp_path, monkeypatch, "validation")
    before = tree_digest(loaded.directory)
    inputs = {p: tree_digest(p) for p in (train, checked)}
    result = clone(loaded, [train], tmp_path / "student", validation=[checked])
    assert tree_digest(loaded.directory) == before
    assert {p: tree_digest(p) for p in (train, checked)} == inputs
    child = load_checkpoint(result["checkpoint"], expected_sha256=result["policy_sha"])
    training = child.manifest["training"]
    assert training["source"] == "servuo-teacher-imitation"
    assert training["parent_policy_sha"] == loaded.policy_sha
    assert training["teacher"] == {
        "name": "TacticalTeacher", "sha256": LIVE_TEACHER_SHA256, "styles": ["tactical"],
        "policy_kind": "teacher",
    }  # fmt: skip
    assert training["lineage_dir"] == str(LINEAGE)
    assert training["ppo_input"] is False and training["seed"] == 11
    assert training["budget"] == {"epochs": 2, "demonstrations": 1, "steps": 6,
                                  "bc_updates": training["metrics"]["bc_updates"]}  # fmt: skip
    (record,) = training["demonstrations"]
    assert record["collection"] == str(train) and record["plan_sha256"]
    assert record["file_sha256"] and record["replay_sha256"] and record["steps"] == 6
    assert [r["reasons"] for r in training["rejected"]] == [
        ["ineligible", "teacher_action_rejected"]
    ]
    assert training["validation"]["collections"] == [str(checked)]
    assert len(training["validation"]["demonstrations"]) == 2
    assert training["before"]["validation"]["free_choices"] == 12
    assert training["demonstrations_consumed"] == [[record["match_id"], record["actor"]]]
    assert training["live_consumed"] == []
    assert result["automatic_promotion"] is False
    update = json.loads((Path(result["checkpoint"]) / "imitation-update.json").read_text())
    assert update["policy_sha"] == result["policy_sha"]


def test_fixed_seed_gives_identical_weights_and_leaves_global_rng_alone(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect(tmp_path, monkeypatch, "train")
    torch.manual_seed(99)
    expected = torch.rand(3)
    torch.manual_seed(99)
    shas = [
        clone(loaded, [train], tmp_path / name, seed=seed)["policy_sha"]
        for name, seed in (("a", 7), ("b", 7), ("c", 8))
    ]
    assert torch.equal(torch.rand(3), expected)
    assert shas[0] == shas[1] != shas[2]


def test_clone_refuses_evaluation_input_overlap_and_relative_paths(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect(tmp_path, monkeypatch, "train")
    held = collect(tmp_path, monkeypatch, "held", purpose="evaluation")
    with pytest.raises(DemonstrationRefused, match="held out"):
        clone(loaded, [held], tmp_path / "x")
    with pytest.raises(DemonstrationRefused, match="held out"):
        clone(loaded, [train], tmp_path / "x", validation=[held])
    other = collect(tmp_path, monkeypatch, "other")
    with pytest.raises(DemonstrationRefused, match="share"):
        clone(loaded, [train], tmp_path / "x", validation=[train, other])
    with pytest.raises(ValueError, match="absolute"):
        clone(loaded, [train], Path("relative-out"))
    with pytest.raises(ValueError, match="absolute"):
        clone(loaded, [train], tmp_path / "x", lineage_dir=Path("relative"))
    with pytest.raises(ValueError, match="absolute"):
        clone_from_demonstrations(
            Path("relative-checkpoint"), expected_sha256=loaded.policy_sha, collections=[train],
            held_out=NO_HELD_OUT, validation=NO_VALIDATION, teacher_sha256=LIVE_TEACHER_SHA256,
            out=tmp_path / "x", config=BC, seed=1, lineage_dir=LINEAGE, min_demonstrations=1,
        )  # fmt: skip
    for missing in ("held_out", "validation"):
        with pytest.raises(Exception, match=missing):
            clone(loaded, [train], tmp_path / "x", **{missing: None})
    assert not (tmp_path / "x").exists()


def test_clone_refuses_existing_output_unmet_minimum_and_untied_reference(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect(tmp_path, monkeypatch, "train")
    (tmp_path / "exists").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        clone(loaded, [train], tmp_path / "exists")
    with pytest.raises(ValueError, match="output checkpoint must be outside"):
        clone(loaded, [train], loaded.directory / "child")
    with pytest.raises(DemonstrationRefused, match="need 3"):
        clone(loaded, [train], tmp_path / "x", min_demonstrations=3)
    with pytest.raises(ValueError, match="minimum"):
        clone(loaded, [train], tmp_path / "x", min_demonstrations=0)
    with pytest.raises(ValueError, match="go together"):
        clone(loaded, [train], tmp_path / "x", reference_checkpoint=loaded.directory,
              reference_sha256=loaded.policy_sha)  # fmt: skip
    with pytest.raises(TypeError):
        clone(loaded, [train], tmp_path / "x", config=SMALL)
    with pytest.raises(ValueError, match="seed"):
        clone(loaded, [train], tmp_path / "x", seed=None)
    with pytest.raises(ValueError, match="expected"):
        clone(loaded, [train], tmp_path / "x", expected_sha256="0" * 64)
    assert not (tmp_path / "x").exists()


def test_reference_regularised_clone_records_the_reference_identity(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    reference = saved(tmp_path, "reference", seed=5)
    train = collect(tmp_path, monkeypatch, "train")
    config = ImitationConfig(
        epochs=1, sequence_length=4, batch_sequences=2, balance_actions=True, reference_kl_coef=0.5
    )
    result = clone(
        loaded, [train], tmp_path / "student", config=config,
        reference_checkpoint=reference.directory, reference_sha256=reference.policy_sha,
    )  # fmt: skip
    training = load_checkpoint(result["checkpoint"], expected_sha256=result["policy_sha"]).manifest[
        "training"
    ]
    assert training["reference_policy_sha"] == reference.policy_sha
    assert training["metrics"]["reference_kl_mean"] >= 0


def test_a_lineage_never_imitates_the_same_demonstration_twice(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect(tmp_path, monkeypatch, "train")
    first = clone(loaded, [train], tmp_path / "first")
    child = load_checkpoint(first["checkpoint"], expected_sha256=first["policy_sha"])
    with pytest.raises(DemonstrationRefused, match="already imitated"):
        clone(child, [train], tmp_path / "second")
    assert not (tmp_path / "second").exists()


def test_an_altered_demonstration_after_selection_is_refused(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect(tmp_path, monkeypatch, "train")
    selection = select_demonstrations(
        [train], teacher_sha256=LIVE_TEACHER_SHA256, held_out=NO_HELD_OUT
    )
    selection.demonstrations[0]["steps"][0]["action"] = 0
    with pytest.raises(DemonstrationRefused, match="re-verification"):
        _clone_selection(
            loaded, selection, None, out=tmp_path / "x", config=BC, seed=1,
            lineage=str(LINEAGE), reference=None,
        )  # fmt: skip
    selection = select_demonstrations(
        [train], teacher_sha256=LIVE_TEACHER_SHA256, held_out=NO_HELD_OUT
    )
    path, _ = next(iter(selection.files.values()))
    original = Path(path).read_bytes()
    edited = json.loads(original)
    edited["steps"][0]["action"] = 0
    Path(path).write_text(json.dumps(edited))
    with pytest.raises(DemonstrationRefused, match="re-verification"):
        _clone_selection(
            loaded, selection, None, out=tmp_path / "x", config=BC, seed=1,
            lineage=str(LINEAGE), reference=None,
        )  # fmt: skip
    Path(path).write_bytes(original)
    selection = select_demonstrations(
        [train], teacher_sha256=LIVE_TEACHER_SHA256, held_out=NO_HELD_OUT
    )
    selection.verifiers = {k: (lambda d: True) for k in selection.verifiers}
    with pytest.raises(ValueError, match="own collections"):
        _clone_selection(
            loaded, selection, None, out=tmp_path / "x", config=BC, seed=1,
            lineage=str(LINEAGE), reference=None,
        )  # fmt: skip
    assert not (tmp_path / "x").exists()
    assert demo_files(train)


def test_imitation_runs_on_cpu_only(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect(tmp_path, monkeypatch, "train")
    from anima3.neural import imitation

    real = imitation.load_checkpoint

    def on_meta(*a, **k):
        result = real(*a, **k)
        monkeypatch.setattr(
            type(result.model), "device", property(lambda self: torch.device("meta"))
        )
        return result

    monkeypatch.setattr(imitation, "load_checkpoint", on_meta)
    with pytest.raises(ValueError, match="CPU only"):
        clone(loaded, [train], tmp_path / "x")
    assert policy  # the shared tiny policy factory stays importable


# --- review round 1 ----------------------------------------------------------------------------


def test_bc_then_ppo_then_bc_never_reimitates_a_demonstration(tmp_path, monkeypatch):
    from test_neural_training import live_episode

    from anima3.neural.train import _unsafe_update_from_episodes

    loaded = saved(tmp_path)
    train = collect(tmp_path, monkeypatch, "train")
    first = clone(loaded, [train], tmp_path / "bc1")
    bc1 = load_checkpoint(first["checkpoint"], expected_sha256=first["policy_sha"])
    ppo = _unsafe_update_from_episodes(
        bc1, [live_episode(bc1, match=99)], verify_episode=lambda _: True,
        out=tmp_path / "ppo", config=SMALL, seed=1, lineage_dir=LINEAGE,
    )  # fmt: skip
    child = load_checkpoint(tmp_path / "ppo", expected_sha256=ppo["policy_sha"])
    consumed = bc1.manifest["training"]["demonstrations_consumed"]
    assert child.manifest["training"]["demonstrations_consumed"] == consumed
    with pytest.raises(DemonstrationRefused, match="already imitated"):
        clone(child, [train], tmp_path / "bc2")
    assert not (tmp_path / "bc2").exists()


def test_a_trained_parent_without_its_consumed_record_is_refused(tmp_path, monkeypatch):
    stripped = saved(tmp_path, training={"parent_policy_sha": "f" * 64, "live_consumed": []})
    train = collect(tmp_path, monkeypatch, "train")
    with pytest.raises(ValueError, match="ancestry's consumed record"):
        clone(stripped, [train], tmp_path / "x")
    assert not (tmp_path / "x").exists()


def test_validation_demonstrations_are_checked_against_the_ancestry(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect(tmp_path, monkeypatch, "train")
    other = collect(tmp_path, monkeypatch, "other")
    first = clone(loaded, [train], tmp_path / "first")
    child = load_checkpoint(first["checkpoint"], expected_sha256=first["policy_sha"])
    with pytest.raises(DemonstrationRefused, match="validation demonstration was already"):
        clone(child, [other], tmp_path / "second", validation=[train])
    fit = select_demonstrations([other], teacher_sha256=LIVE_TEACHER_SHA256, held_out=NO_HELD_OUT)
    scored = select_demonstrations(
        [train], teacher_sha256=LIVE_TEACHER_SHA256, held_out=NO_HELD_OUT
    )
    scored.demonstrations[0]["steps"][0]["action"] = 0
    with pytest.raises(DemonstrationRefused, match="validation demonstration failed"):
        _clone_selection(
            loaded, fit, scored, out=tmp_path / "x", config=BC, seed=1,
            lineage=str(LINEAGE), reference=None,
        )  # fmt: skip
    assert not (tmp_path / "second").exists() and not (tmp_path / "x").exists()


def test_output_may_not_overlap_any_input_even_through_a_symlink(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    reference = saved(tmp_path, "reference", seed=5)
    train = collect(tmp_path, monkeypatch, "train")
    checked = collect(tmp_path, monkeypatch, "validation")
    held = collect(tmp_path, monkeypatch, "held", purpose="evaluation")
    link = tmp_path / "link"
    link.symlink_to(train, target_is_directory=True)
    config = ImitationConfig(
        epochs=1, sequence_length=4, batch_sequences=2, balance_actions=False,
        reference_kl_coef=0.5,
    )  # fmt: skip
    for out in (
        train / "student", link / "student", checked / "student", held / "student",
        reference.directory / "student",
    ):  # fmt: skip
        with pytest.raises(ValueError, match="overlaps an input"):
            clone(
                loaded, [train], out, validation=[checked], held_out=[held], config=config,
                reference_checkpoint=reference.directory, reference_sha256=reference.policy_sha,
            )  # fmt: skip
        assert not out.exists()
    assert clone(loaded, [train], tmp_path / "fine", validation=[checked], held_out=[held])


def test_imitation_children_of_a_simulator_policy_stay_marked(tmp_path, monkeypatch):
    """Row 7b-2: a clone keeps the ancestry's simulator marker and record."""
    record = [["demonstrations", 20261006, "melee", 8]]
    parent = saved(tmp_path, training={
        "parent_policy_sha": "f" * 64, "live_consumed": [], "demonstrations_consumed": [],
        "simulator_trained": True, "simulator_ancestry": True, "simulator_consumed": record,
    })  # fmt: skip
    train = collect(tmp_path, monkeypatch, "train")
    result = clone(parent, [train], tmp_path / "student")
    training = load_checkpoint(result["checkpoint"], expected_sha256=result["policy_sha"]).manifest[
        "training"
    ]
    assert training["simulator_ancestry"] is True and training["simulator_consumed"] == record
    plain = clone(saved(tmp_path, "plain"), [train], tmp_path / "plain-student")
    training = json.loads((Path(plain["checkpoint"]) / "manifest.json").read_text())["training"]
    assert training["simulator_ancestry"] is False and training["simulator_consumed"] == []
    stripped = saved(tmp_path, "stripped", training={
        "parent_policy_sha": "f" * 64, "live_consumed": [], "demonstrations_consumed": [],
        "simulator_trained": True,
    })  # fmt: skip
    with pytest.raises(ValueError, match="simulator record"):
        clone(stripped, [train], tmp_path / "x")
    malformed = saved(tmp_path, "malformed", training={
        "parent_policy_sha": "f" * 64, "live_consumed": [], "demonstrations_consumed": [],
        "simulator_trained": "true", "simulator_ancestry": True, "simulator_consumed": [],
    })  # fmt: skip
    with pytest.raises(ValueError, match="malformed simulator marker"):
        clone(malformed, [train], tmp_path / "x")
    assert not (tmp_path / "x").exists()
