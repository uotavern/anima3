"""Row 7b simulator rollouts and simulator evaluation games; tiny CPU models only.

``collect_simulator`` and the ``evaluate_suite`` game loop are compared with the
source tree at ``819f4f4``, exported with ``git archive`` and run in a separate
interpreter on the same weights. No server, client or training run is involved.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_neural_sim import (
    export_source,
    record_decision_refuses,
    run_tree,
    sim_plan,
)

from anima3.neural import sim_evaluation, train
from anima3.neural.checkpoints import CheckpointError, save_checkpoint
from anima3.neural.drills import SIM_DOMAIN, SIMULATOR_SHA256
from anima3.neural.live import LiveReplayVerifier
from anima3.neural.model import PolicyConfig, RecurrentPolicy
from anima3.neural.schema import ACTION_NAMES, FEATURE_NAMES
from anima3.neural.sim_evaluation import (
    EVALUATION_SEED_FLOOR,
    SIM_DECISION_SCHEMA,
    SimulatorRefused,
)
from anima3.neural.sim_rollouts import (
    _suite_rows,
    collect_simulator,
    decide_in_simulator,
    evaluate_in_simulator,
)
from anima3.neural.teacher import STYLES, TEACHER_SHA256, TacticalTeacher

SHAPING = 0.995**0.25
ADDED_LABELS = {
    "domain",
    "simulator_sha256",
    "teacher_sha256",
    "live_episode",
    "sim_seed",
    "rollout_seed",
    "drill",
    "ppo_eligible",
}


def policy(seed: int) -> RecurrentPolicy:
    torch.manual_seed(seed)
    return RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), 64, 64)).eval()


@pytest.fixture(scope="module")
def source_tree(tmp_path_factory):
    return export_source(tmp_path_factory)


@pytest.fixture
def weights(tmp_path):
    """Learner and opponent weights in a file both interpreters can load."""
    path = tmp_path / "weights.pt"
    learner, opponent = policy(11), policy(12)
    torch.save({"learner": learner.state_dict(), "opponent": opponent.state_dict()}, path)
    return path, learner, opponent


LOAD = """
import json, torch
from anima3.neural.model import PolicyConfig, RecurrentPolicy
state = torch.load({path!r}, weights_only=True)
def build(name):
    model = RecurrentPolicy(PolicyConfig(76, 27, 64, 64))
    model.load_state_dict(state[name])
    return model.eval()
learner, opponent = build("learner"), build("opponent")
"""
CASES = [
    {"steps": 300, "seed": 5, "drill": "full", "opponent_styles": STYLES},
    {"steps": 200, "seed": 6, "drill": "melee", "teacher_actions": True},
    {"steps": 250, "seed": 7, "drill": "survival", "opponent_styles": ("burst", "potions")},
    {
        "steps": 150,
        "seed": 8,
        "drill": "burst",
        "teacher_actions": True,
        "teacher_style": "pressure",
    },
    {"steps": 260, "seed": 9, "drill": "full", "frozen_opponents": True},
]
SOURCE_COLLECT = """
from anima3.neural.train import collect_simulator
out = []
for case in {cases!r}:
    case = dict(case)
    frozen = case.pop("frozen_opponents", False)
    torch.manual_seed(case["seed"])
    episodes, summary = collect_simulator(
        learner, max_episode_steps=60, shaping_gamma={shaping!r},
        opponents=[opponent] if frozen else None, **case)
    out.append({{"episodes": [e.__dict__ for e in episodes], "summary": summary}})
print(json.dumps(out))
"""


def _port_collect(learner, opponent):
    out = []
    for case in CASES:
        case = dict(case)
        frozen = case.pop("frozen_opponents", False)
        episodes, summary = collect_simulator(
            learner,
            max_episode_steps=60,
            shaping_gamma=SHAPING,
            opponents=[opponent] if frozen else None,
            **case,
        )
        out.append({"episodes": [asdict(e) for e in episodes], "summary": summary})
    return json.loads(json.dumps(out))


def test_collect_simulator_equals_the_source(source_tree, weights):
    path, learner, opponent = weights
    source = run_tree(
        source_tree,
        LOAD.format(path=str(path)),
        SOURCE_COLLECT.format(cases=CASES, shaping=SHAPING),
    )
    port = _port_collect(learner, opponent)
    assert len(source) == len(port) == len(CASES)
    for theirs, ours in zip(source, port, strict=True):
        assert {k: ours["summary"][k] for k in theirs["summary"]} == theirs["summary"]
        assert len(theirs["episodes"]) == len(ours["episodes"]) > 1
        for a, b in zip(theirs["episodes"], ours["episodes"], strict=True):
            labels = {k: b["info"].pop(k) for k in ADDED_LABELS if k in b["info"]}
            assert a == b
            assert labels["domain"] == SIM_DOMAIN and labels["live_episode"] is False


SOURCE_SUITE = """
from anima3.neural.league import evaluate_suite
out = {{}}
for sampling in ("categorical", "greedy"):
    out[sampling] = evaluate_suite(
        learner, games_per_style=4, seed=3_600_000_000, max_steps=80, sampling=sampling)["rows"]
print(json.dumps(out))
"""


def test_simulator_evaluation_games_equal_the_source_suite(source_tree, weights):
    path, learner, _ = weights
    source = run_tree(source_tree, LOAD.format(path=str(path)), SOURCE_SUITE.format())
    for sampling in ("categorical", "greedy"):
        ours = _suite_rows(
            learner,
            styles=STYLES,
            games_per_style=4,
            seed=3_600_000_000,
            max_steps=80,
            sampling=sampling,
        )
        assert json.loads(json.dumps(ours)) == source[sampling]
    assert {r["score"] for r in source["categorical"]} >= {0.5}


def test_evaluation_covers_the_full_server_round_and_showdown(monkeypatch):
    # Ported from the source test_default_evaluation_covers_full_server_showdown_window.
    class HoldPolicy:
        device = "cpu"

        def eval(self):
            return self

        def initial_state(self):
            return None

        def act(self, *args, **kwargs):
            return torch.tensor(0), None, None, None

    monkeypatch.setattr(TacticalTeacher, "choose", lambda *args: 0)
    rows = _suite_rows(
        HoldPolicy(),
        styles=STYLES,
        games_per_style=2,
        seed=3_600_000_000,
        max_steps=1200,
        sampling="categorical",
    )
    assert len(rows) == 10 and sum(r["score"] == 0.5 for r in rows) == 10
    assert all(r["elapsed"] == 300 and r["terminated"] for r in rows)


def test_collect_simulator_has_no_default_budget_or_seed():
    model = policy(1)
    for missing in ("steps", "seed", "max_episode_steps", "shaping_gamma"):
        kwargs = {"steps": 4, "seed": 1, "max_episode_steps": 4, "shaping_gamma": SHAPING}
        del kwargs[missing]
        with pytest.raises(TypeError):
            collect_simulator(model, **kwargs)
    good = {"steps": 4, "seed": 1, "max_episode_steps": 4, "shaping_gamma": SHAPING}
    for change in (
        {"steps": 0},
        {"steps": 1_000_001},
        {"steps": 4.0},
        {"seed": -1},
        {"seed": True},
        {"max_episode_steps": 0},
        {"shaping_gamma": 0.0},
        {"shaping_gamma": float("nan")},
        {"drill": "sparring"},
        {"teacher_actions": 1},
        {"opponent_styles": ()},
        {"opponent_styles": ["balanced"]},
        {"teacher_style": "berserk"},
    ):
        with pytest.raises(ValueError):
            collect_simulator(model, **{**good, **change})
    with pytest.raises(ValueError, match="CPU"):
        collect_simulator(policy(1).to("meta"), **good)


def test_collect_simulator_is_seeded_privately_and_reproducibly():
    model = policy(2)
    kwargs = {"steps": 120, "max_episode_steps": 40, "shaping_gamma": SHAPING}
    torch.manual_seed(1234)
    state, python_state = torch.get_rng_state(), random.getstate()
    first, _ = collect_simulator(model, seed=3, **kwargs)
    assert torch.equal(torch.get_rng_state(), state) and random.getstate() == python_state
    torch.manual_seed(999)  # the caller's global RNG does not decide the rollout
    again, _ = collect_simulator(model, seed=3, **kwargs)
    other, _ = collect_simulator(model, seed=4, **kwargs)
    assert [asdict(e) for e in first] == [asdict(e) for e in again]
    assert [asdict(e) for e in first] != [asdict(e) for e in other]


def test_simulator_trajectories_are_labelled_and_seeded_below_evaluation():
    episodes, summary = collect_simulator(
        policy(3), steps=400, seed=10, max_episode_steps=30, shaping_gamma=SHAPING
    )
    assert summary["domain"] == SIM_DOMAIN and summary["live_episode"] is False
    assert summary["simulator_sha256"] == SIMULATOR_SHA256
    assert summary["real_server_strength_established"] is False
    assert len(episodes) > 5
    for episode in episodes:
        info = episode.info
        assert info["source"] == info["domain"] == SIM_DOMAIN
        assert info["simulator_sha256"] == SIMULATOR_SHA256
        assert info["teacher_sha256"] == TEACHER_SHA256
        assert info["rollout_seed"] == 10 and info["live_episode"] is False
        assert 0 <= info["sim_seed"] < EVALUATION_SEED_FLOOR
        assert "ppo_eligible" not in info
    assert summary["steps"] == sum(map(len, episodes)) == 400


def test_teacher_labelled_rollouts_are_never_ppo_experience():
    model = policy(4)
    labels, _ = collect_simulator(
        model,
        steps=60,
        seed=11,
        max_episode_steps=30,
        shaping_gamma=SHAPING,
        teacher_actions=True,
        drill="melee",
    )
    assert all(e.info["ppo_eligible"] is False for e in labels)
    config = train.TrainConfig(
        epochs=1, sequence_length=16, burn_in=0, batch_sequences=4, discount_time_unit=1.0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="not PPO trajectories"):
        train.ppo_update(model, optimizer, labels, config, seed=1)


def test_simulator_rollouts_are_never_live_ppo_episodes(tmp_path):
    model = policy(5)
    rollouts, _ = collect_simulator(
        model, steps=40, seed=12, max_episode_steps=20, shaping_gamma=SHAPING
    )
    root = tmp_path / "parent"
    sha = save_checkpoint(root, model)["weights"]["sha256"]
    from anima3.neural.checkpoints import load_checkpoint

    loaded = load_checkpoint(root, expected_sha256=sha)
    out = tmp_path / "child"
    # The public live update takes only a receipt-verified TrainingSelection.
    with pytest.raises(TypeError, match="TrainingSelection"):
        train.update_verified_episodes(
            loaded, rollouts, out=out, config=None, seed=1, lineage_dir=tmp_path / "lineage"
        )
    config = train.TrainConfig(
        epochs=1, sequence_length=16, burn_in=0, batch_sequences=4, discount_time_unit=1.0
    )
    # Even a trusting verifier cannot turn a simulator rollout into a live episode.
    for episode in (
        rollouts[0],
        asdict(rollouts[0]),
        {**asdict(rollouts[0]), "purpose": "training"},
    ):
        with pytest.raises((ValueError, KeyError, TypeError)):
            train._unsafe_update_from_episodes(
                loaded,
                [episode],
                verify_episode=lambda e: True,
                out=out,
                config=config,
                seed=1,
                lineage_dir=tmp_path / "lineage",
            )
    assert not out.exists()
    assert LiveReplayVerifier(tmp_path, sha)(asdict(rollouts[0])) is False


# --- simulator evaluation through the unit-7 gate --------------------------------------------


def _tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def checkpoints(tmp_path):
    made = {}
    for name, seed in (("incumbent", 21), ("candidate", 22)):
        root = tmp_path / name
        made[name] = (root, save_checkpoint(root, policy(seed))["weights"]["sha256"])
    return made


def plan_for(checkpoints, **overrides):
    return sim_plan(
        styles=("balanced", "potions"),
        candidate=checkpoints["candidate"][1],
        incumbent=checkpoints["incumbent"][1],
        max_steps=30,
        **overrides,
    )


def test_evaluate_in_simulator_loads_by_sha_and_is_deterministic(checkpoints):
    plan = plan_for(checkpoints)
    root, sha = checkpoints["candidate"]
    before = _tree(root.parent)
    torch.manual_seed(77)
    state = torch.get_rng_state()
    first = evaluate_in_simulator(root, expected_sha256=sha, plan=plan)
    assert torch.equal(torch.get_rng_state(), state)
    assert evaluate_in_simulator(root, expected_sha256=sha, plan=plan) == first
    assert first["domain"] == SIM_DOMAIN and first["simulator_only"] is True
    assert first["simulator_plan_sha256"] == plan.sha256 and len(first["rows"]) == 40
    assert _tree(root.parent) == before  # nothing is written
    with pytest.raises(SimulatorRefused, match="neither"):
        evaluate_in_simulator(root, expected_sha256="ef" * 32, plan=plan)
    with pytest.raises(CheckpointError):
        evaluate_in_simulator(checkpoints["incumbent"][0], expected_sha256=sha, plan=plan)
    with pytest.raises(SimulatorRefused, match="absolute"):
        evaluate_in_simulator(Path(root.name), expected_sha256=sha, plan=plan)
    with pytest.raises(SimulatorRefused, match="SimulatorEvaluationPlan"):
        evaluate_in_simulator(root, expected_sha256=sha, plan=plan.evaluation)


def test_evaluate_in_simulator_refuses_a_plan_for_other_code(checkpoints, monkeypatch):
    plan = plan_for(checkpoints)
    root, sha = checkpoints["candidate"]
    monkeypatch.setattr(sim_evaluation, "SIMULATOR_SHA256", "ee" * 32)
    with pytest.raises(SimulatorRefused, match="simulator code changed"):
        evaluate_in_simulator(root, expected_sha256=sha, plan=plan)


def test_decide_in_simulator_is_re_derivable_and_simulator_only(checkpoints, tmp_path):
    plan = plan_for(checkpoints)
    before = _tree(tmp_path)
    decision = decide_in_simulator(
        plan, incumbent=checkpoints["incumbent"][0], candidate=checkpoints["candidate"][0]
    )
    assert (
        decide_in_simulator(
            plan, incumbent=checkpoints["incumbent"][0], candidate=checkpoints["candidate"][0]
        )
        == decision
    )
    assert _tree(tmp_path) == before
    assert decision["schema"] == SIM_DECISION_SCHEMA and decision["domain"] == SIM_DOMAIN
    assert decision["decision"] in ("candidate_passed_simulator_gate", "retain_incumbent")
    assert decision["live_gain_established"] is False and decision["simulator_only"] is True
    assert decision["evidence"]["simulator"]["checkpoints"] == {
        "incumbent": str(checkpoints["incumbent"][0]),
        "candidate": str(checkpoints["candidate"][0]),
    }
    record_decision_refuses(tmp_path / "lineage", decision)
    # Swapped checkpoints do not hash to the planned roles.
    with pytest.raises(CheckpointError):
        decide_in_simulator(
            plan, incumbent=checkpoints["candidate"][0], candidate=checkpoints["incumbent"][0]
        )
