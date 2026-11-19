"""Unit-7 bounded PPO on verified training episodes; tiny synthetic loops under the neural extra.

The PPO, GAE and live-update cases come from the training branch's
``test_neural_training.py``; the rest cover the unit-7 boundaries: explicit
budgets and seeds, determinism, SHA-recorded output that never overwrites its
input, and training input only from verified ``training`` collections.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_neural_evaluation import Scored

from anima3.neural import live
from anima3.neural.checkpoints import load_checkpoint, save_checkpoint, schema_fingerprint
from anima3.neural.live import EPISODE_SCHEMA, Arm, CollectionPlan
from anima3.neural.model import PolicyConfig, RecurrentPolicy
from anima3.neural.schema import ACTION_NAMES, FEATURE_NAMES
from anima3.neural.splits import (
    NO_HELD_OUT,
    SplitRefused,
    TrainingSelection,
    read_collection,
    select_training_episodes,
)
from anima3.neural.train import (
    MAX_EPOCHS,
    NoEligibleEpisodes,
    TrainConfig,
    Trajectory,
    _targets,
    _unsafe_update_from_episodes,
    clipped_policy_loss,
    generalized_advantage,
    ppo_update,
    train_from_collections,
    update_verified_episodes,
)

LINEAGE = Path("/nonexistent/anima3-test-lineage")
BUDGET = {
    "epochs": 2,
    "sequence_length": 4,
    "burn_in": 2,
    "batch_sequences": 2,
    "discount_time_unit": 1.0,
}
SMALL = TrainConfig(
    epochs=2, sequence_length=4, burn_in=2, batch_sequences=2, discount_time_unit=1.0
)


def policy(seed=23):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    return RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), 16, 16))


def trajectory(model, length=12, *, seed=0):
    episode = Trajectory(terminated=True)
    hidden = model.initial_state()
    generator = torch.Generator().manual_seed(seed)
    for index in range(length):
        features = torch.zeros(1, len(FEATURE_NAMES))
        features[0, (index + seed) % len(FEATURE_NAMES)] = 0.5
        mask = torch.zeros(1, len(ACTION_NAMES), dtype=torch.bool)
        mask[0, :3] = True
        action, log_prob, value, next_hidden = model.act(
            features, mask, hidden, generator=generator
        )
        episode.features.append(features[0].tolist())
        episode.masks.append(mask[0].tolist())
        episode.actions.append(int(action.item()))
        episode.log_probs.append(float(log_prob.item()))
        episode.values.append(float(value.item()))
        episode.hidden.append(hidden[0].tolist())
        episode.rewards.append(1.0 if index == length - 1 else 0.0)
        episode.durations.append(0.25)
        hidden = next_hidden
    return episode


def transitions(model, length=12, *, seed=0):
    trace = trajectory(model, length, seed=seed)
    return [
        {
            "features": trace.features[i],
            "mask": trace.masks[i],
            "action": trace.actions[i],
            "log_prob": trace.log_probs[i],
            "value": trace.values[i],
            "hidden_in": trace.hidden[i],
            "reward": trace.rewards[i],
            "dt": trace.durations[i],
            "done": i == len(trace) - 1,
            "accepted": True,
            "policy_accepted": True,
        }
        for i in range(len(trace))
    ]


def saved(tmp_path, name="input", seed=23, training=None):
    manifest = save_checkpoint(tmp_path / name, policy(seed), training=training)
    sha = manifest["weights"]["sha256"]
    return load_checkpoint(tmp_path / name, expected_sha256=sha)


def live_episode(loaded, *, match=1, actor=57, seed=0):
    return {
        "schema": EPISODE_SCHEMA,
        "domain": "servuo",
        "sampling": "categorical",
        "purpose": "training",
        "plan_sha256": "f" * 64,
        "policy_sha": loaded.policy_sha,
        "schema_fingerprint": schema_fingerprint(),
        "actor": actor,
        "verification": {
            "verified": True,
            "complete": True,
            "training": True,
            "aborted": False,
            "match_id": f"{match:032x}",
            "replay_sha256": "a" * 64,
        },
        "transitions": transitions(loaded.model, seed=seed),
    }


def tree_digest(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --- ported PPO core -------------------------------------------------------------------------


def test_gae_bootstraps_truncations_but_not_terminal_or_cross_episode():
    advantage, returns = generalized_advantage(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 10.0, 9.0]),
        torch.tensor([False, False, True]),
        torch.tensor([False, True, True]),
        gamma=0.5,
        gae_lambda=1.0,
    )
    assert advantage.tolist() == pytest.approx([4.5, 7.0, 3.0])
    assert returns.tolist() == pytest.approx([4.5, 7.0, 3.0])


def test_gae_uses_elapsed_seconds_for_semi_markov_actions():
    advantage, _ = generalized_advantage(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([4.0]),
        torch.tensor([False]),
        torch.tensor([True]),
        gamma=0.5,
        durations=torch.tensor([2.0]),
        discount_time_unit=1.0,
    )
    assert advantage.item() == pytest.approx(1.0)


def test_ppo_clipping_both_advantage_signs():
    loss = clipped_policy_loss(
        torch.tensor([2.0, 0.5]).log(), torch.zeros(2), torch.tensor([1.0, -1.0]), 0.2
    )
    assert loss.tolist() == pytest.approx([-1.2, 0.8])


@pytest.mark.parametrize("field", ["gamma", "gae_lambda", "entropy_coef", "discount_time_unit"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_training_coefficients_reject_nonfinite(field, value):
    with pytest.raises(ValueError, match="finite"):
        TrainConfig(**{**BUDGET, field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("gamma", 0.0),
        ("gamma", 1.01),
        ("gae_lambda", -0.01),
        ("gae_lambda", 1.01),
        ("entropy_coef", -0.01),
        ("discount_time_unit", -0.01),
        ("epochs", 0),
        ("epochs", MAX_EPOCHS + 1),
        ("epochs", True),
        ("epochs", 2.0),
        ("sequence_length", 4097),
        ("batch_sequences", 0),
        ("burn_in", -1),
    ],
)
def test_training_budget_and_ranges_are_bounded(field, value):
    with pytest.raises(ValueError):
        TrainConfig(**{**BUDGET, field: value})


def force_hold(episode, indices):
    for index in indices:
        episode.masks[index] = [True] + [False] * (len(ACTION_NAMES) - 1)
        episode.actions[index] = 0
        episode.log_probs[index] = 0.0


def test_actor_advantage_normalization_ignores_forced_holds_but_retains_critic_targets():
    episode = trajectory(policy(), length=4)
    episode.values = [0.0] * 4
    episode.rewards = [1.0, 100.0, 3.0, -100.0]
    force_hold(episode, [1, 3])
    advantages, returns = _targets([episode], TrainConfig(**BUDGET, gamma=1.0, gae_lambda=0.0))[0]
    assert advantages.tolist() == pytest.approx([-1.0, 0.0, 1.0, 0.0])
    assert returns.tolist() == pytest.approx(episode.rewards)


def test_all_forced_ppo_rollout_still_trains_critic_without_nan_or_actor_update():
    model = policy()
    episode = trajectory(model)
    force_hold(episode, range(len(episode)))
    policy_before = copy.deepcopy(model.policy_head.state_dict())
    value_before = copy.deepcopy(model.value_head.state_dict())
    metrics = ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        [episode],
        TrainConfig(**{**BUDGET, "epochs": 1, "batch_sequences": 1}),
        seed=0,
    )
    assert metrics["optimized_steps"] == len(episode)
    assert metrics["policy_optimized_steps"] == 0
    for key in ("policy_loss", "entropy", "approximate_kl", "clip_fraction"):
        assert metrics[key] == 0.0
    assert all(math.isfinite(value) for value in metrics.values())
    assert all(
        torch.equal(policy_before[name], value)
        for name, value in model.policy_head.state_dict().items()
    )
    assert any(
        not torch.equal(value_before[name], value)
        for name, value in model.value_head.state_dict().items()
    )


def test_forced_holds_do_not_dilute_actor_kl_early_stop():
    model = policy()
    episode = trajectory(model)
    force_hold(episode, range(2, len(episode)))
    episode.log_probs[0] -= 0.4
    episode.log_probs[1] -= 0.4
    before = copy.deepcopy(model.state_dict())
    metrics = ppo_update(
        model,
        torch.optim.Adam(model.parameters(), lr=1e-3),
        [episode],
        TrainConfig(**{**BUDGET, "epochs": 1, "sequence_length": len(episode)}),
        seed=0,
    )
    assert metrics["kl_early_stop"] is True
    assert metrics["optimized_steps"] == metrics["policy_optimized_steps"] == 0
    assert metrics["optimizer_steps"] == 0
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_ppo_optimizer_steps_never_exceed_the_epoch_and_batch_budget():
    model = policy()
    episodes = [trajectory(model, seed=s) for s in range(3)]  # 3 x 12 steps -> 9 chunks of 4
    config = TrainConfig(**{**BUDGET, "epochs": 3})
    metrics = ppo_update(
        model, torch.optim.Adam(model.parameters(), lr=1e-4), episodes, config, seed=0
    )
    assert metrics["optimizer_steps"] <= config.epochs * math.ceil(9 / config.batch_sequences)
    if not metrics["kl_early_stop"]:
        assert metrics["optimizer_steps"] == 15
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())


# --- live update boundaries ------------------------------------------------------------------


@pytest.mark.parametrize(
    "corruption,error",
    [
        ("version", "off-policy"),
        ("rejected", "rejected or substituted"),
        ("log_prob", "log_prob mismatch"),
        ("hidden", "state chain mismatch"),
        ("unfinished", "terminal boundary"),
        ("argmax", "stochastic categorical"),
        ("evaluation", "held-out evaluation"),
        ("unlabelled", "held-out evaluation"),
    ],
)
def test_live_updates_fail_closed_before_any_output(tmp_path, corruption, error):
    loaded = saved(tmp_path)
    episode = live_episode(loaded)
    if corruption == "version":
        episode["policy_sha"] = "b" * 64
    elif corruption == "rejected":
        episode["transitions"][0]["accepted"] = False
    elif corruption == "log_prob":
        episode["transitions"][0]["log_prob"] += 0.1
    elif corruption == "hidden":
        episode["transitions"][2]["hidden_in"][0] += 0.1
    elif corruption == "unfinished":
        episode["transitions"][-1]["done"] = False
    elif corruption == "argmax":
        episode["sampling"] = "argmax"
    elif corruption == "evaluation":
        episode["purpose"] = "evaluation"
    elif corruption == "unlabelled":
        del episode["purpose"]
    before = tree_digest(tmp_path / "input")
    with pytest.raises(ValueError, match=error):
        _unsafe_update_from_episodes(
            loaded,
            [episode],
            verify_episode=lambda _: True,
            out=tmp_path / "output",
            config=SMALL,
            seed=1,
            lineage_dir=LINEAGE,
        )
    assert not (tmp_path / "output").exists()
    assert tree_digest(tmp_path / "input") == before


def test_update_requires_explicit_budget_seed_and_trusted_verifier(tmp_path):
    loaded = saved(tmp_path)
    episode = live_episode(loaded)
    kwargs = {
        "out": tmp_path / "output",
        "verify_episode": lambda _: True,
        "lineage_dir": LINEAGE,
    }
    with pytest.raises(TypeError):
        _unsafe_update_from_episodes(loaded, [episode], **kwargs)  # no config/seed
    with pytest.raises(TypeError, match="TrainConfig"):
        _unsafe_update_from_episodes(loaded, [episode], config={"epochs": 1}, seed=1, **kwargs)
    for seed in (-1, True, 1.0, None):
        with pytest.raises(ValueError, match="seed"):
            _unsafe_update_from_episodes(loaded, [episode], config=SMALL, seed=seed, **kwargs)
    with pytest.raises(ValueError, match="verification failed"):
        _unsafe_update_from_episodes(
            loaded,
            [episode],
            verify_episode=lambda _: False,
            out=tmp_path / "output",
            config=SMALL,
            seed=1,
            lineage_dir=LINEAGE,
        )
    assert not (tmp_path / "output").exists()


def test_output_is_a_new_sha_recorded_checkpoint_that_never_overwrites_its_input(tmp_path):
    loaded = saved(tmp_path)
    before = tree_digest(tmp_path / "input")
    episode = live_episode(loaded)
    for out in (tmp_path / "input", tmp_path / "input" / "child"):
        with pytest.raises(ValueError, match="outside the input"):
            _unsafe_update_from_episodes(
                loaded,
                [episode],
                verify_episode=lambda _: True,
                out=out,
                config=SMALL,
                seed=1,
                lineage_dir=LINEAGE,
            )
    (tmp_path / "taken").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _unsafe_update_from_episodes(
            loaded,
            [episode],
            verify_episode=lambda _: True,
            out=tmp_path / "taken",
            config=SMALL,
            seed=1,
            lineage_dir=LINEAGE,
        )
    assert list((tmp_path / "taken").iterdir()) == []
    result = _unsafe_update_from_episodes(
        loaded,
        [episode],
        verify_episode=lambda _: True,
        out=tmp_path / "output",
        config=SMALL,
        seed=1,
        lineage_dir=LINEAGE,
    )
    assert tree_digest(tmp_path / "input") == before
    assert result["policy_sha"] != loaded.policy_sha
    assert result["parent_policy_sha"] == loaded.policy_sha
    assert result["automatic_promotion"] is False
    updated = load_checkpoint(tmp_path / "output", expected_sha256=result["policy_sha"])
    weights = updated.manifest["weights"]
    blob = (tmp_path / "output" / weights["file"]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == weights["sha256"] == result["policy_sha"]
    training = updated.manifest["training"]
    assert training["parent_policy_sha"] == loaded.policy_sha and training["seed"] == 1
    assert training["budget"]["epochs"] == SMALL.epochs and training["budget"]["episodes"] == 1
    assert training["live_consumed"] == [[f"{1:032x}", 57]]
    record = json.loads((tmp_path / "output" / "training-update.json").read_text())
    assert record["policy_sha"] == result["policy_sha"]
    # The parent's in-memory weights are untouched too.
    assert load_checkpoint(tmp_path / "input", expected_sha256=loaded.policy_sha)
    with pytest.raises(ValueError, match="already exists"):
        _unsafe_update_from_episodes(
            loaded,
            [episode],
            verify_episode=lambda _: True,
            out=tmp_path / "output",
            config=SMALL,
            seed=1,
            lineage_dir=LINEAGE,
        )


def test_fixed_seed_gives_identical_weights_and_leaves_global_rng_alone(tmp_path):
    loaded = saved(tmp_path)
    episodes = [live_episode(loaded, match=m, seed=m) for m in (1, 2, 3)]
    torch.manual_seed(99)
    expected_next = torch.rand(3)
    torch.manual_seed(99)
    shas = []
    for name, seed in (("a", 7), ("b", 7), ("c", 8)):
        result = _unsafe_update_from_episodes(
            loaded,
            episodes,
            verify_episode=lambda _: True,
            out=tmp_path / name,
            config=SMALL,
            seed=seed,
            lineage_dir=LINEAGE,
        )
        shas.append(result["policy_sha"])
    assert torch.equal(torch.rand(3), expected_next)
    assert shas[0] == shas[1]
    assert shas[2] != shas[0]


def test_both_sides_of_a_match_are_distinct_and_consumed_identities_are_refused(tmp_path):
    loaded = saved(tmp_path)
    first = live_episode(loaded, actor=57)
    second = live_episode(loaded, actor=58, seed=1)
    result = _unsafe_update_from_episodes(
        loaded,
        [first, second],
        verify_episode=lambda _: True,
        out=tmp_path / "child",
        config=SMALL,
        seed=1,
        lineage_dir=LINEAGE,
    )
    child = load_checkpoint(tmp_path / "child", expected_sha256=result["policy_sha"])
    assert child.manifest["training"]["live_consumed"] == [[f"{1:032x}", 57], [f"{1:032x}", 58]]
    with pytest.raises(ValueError, match="duplicate"):
        _unsafe_update_from_episodes(
            loaded,
            [first, copy.deepcopy(first)],
            verify_episode=lambda _: True,
            out=tmp_path / "dup",
            config=SMALL,
            seed=1,
            lineage_dir=LINEAGE,
        )
    lineage = saved(tmp_path, "lineage", training={"live_consumed": [[f"{1:032x}", 57]]})
    with pytest.raises(ValueError, match="duplicate"):
        _unsafe_update_from_episodes(
            lineage,
            [live_episode(lineage)],
            verify_episode=lambda _: True,
            out=tmp_path / "again",
            config=SMALL,
            seed=1,
            lineage_dir=LINEAGE,
        )


def test_batch_size_is_capped(tmp_path):
    loaded = saved(tmp_path)
    template = live_episode(loaded)
    template["transitions"] = template["transitions"][:1]
    template["transitions"][0].update(done=True, reward=1.0)
    episodes = []
    for match in range(100):
        for actor in (57, 58):
            episode = copy.deepcopy(template)
            episode["verification"]["match_id"] = f"{match:032x}"
            episode["actor"] = actor
            episodes.append(episode)
    config = TrainConfig(
        epochs=1, sequence_length=1, burn_in=0, batch_sequences=200, discount_time_unit=1.0
    )
    with pytest.raises(ValueError, match="1..200"):
        _unsafe_update_from_episodes(
            loaded,
            episodes + [copy.deepcopy(template)],
            verify_episode=lambda _: True,
            out=tmp_path / "oversized",
            config=config,
            seed=0,
            lineage_dir=LINEAGE,
        )
    assert not (tmp_path / "oversized").exists()
    result = _unsafe_update_from_episodes(
        loaded,
        episodes,
        verify_episode=lambda _: True,
        out=tmp_path / "output",
        config=config,
        seed=0,
        lineage_dir=LINEAGE,
    )
    assert result["episodes"] == 200 and result["metrics"]["rollout_steps"] == 200


# --- end to end from real unit-6 collections -------------------------------------------------


class PolicyHarness(Scored):
    """The candidate seat records transitions sampled from the real checkpoint."""

    def __init__(self, monkeypatch, winners, model):
        super().__init__(monkeypatch, winners)
        self.model = model
        self.games = 0

    def fight(self, bodies, workers, log, stop, **kw):
        actors = super().fight(bodies, workers, log, stop, **kw)
        self.games += 1
        steps = transitions(self.model, length=6, seed=self.games)
        for step in steps:
            step.update(reward=0.0, done=False)
        actors[0]["transitions"] = steps
        return actors


def collect_policy(tmp_path, monkeypatch, name, loaded, winners, purpose):
    harness = PolicyHarness(monkeypatch, winners, loaded.model)
    plan = CollectionPlan(
        endpoint="lab.invalid:2593",
        accounts=("alpha", "beta"),
        arms=(Arm("neural", str(loaded.directory), loaded.policy_sha), Arm("scripted")),
        matches=len(winners),
        purpose=purpose,
    )
    live.run_collection(
        plan,
        tmp_path / name,
        connect=harness.connect,
        fetch_receipt=harness.receipt,
        lock_dir=tmp_path / "locks",
        client_factory=harness.factory,
        find_spec=lambda _: object(),
    )
    return tmp_path / name


def test_train_from_verified_training_collections_end_to_end(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect_policy(tmp_path, monkeypatch, "train", loaded, [1, 2, 1], "training")
    held = collect_policy(tmp_path, monkeypatch, "held", loaded, [2, 1], "evaluation")
    before = {p: tree_digest(p) for p in (tmp_path / "input", train, held)}
    kwargs = {
        "expected_sha256": loaded.policy_sha,
        "config": SMALL,
        "seed": 3,
        "lineage_dir": LINEAGE,
    }
    with pytest.raises(TypeError, match="held_out"):
        train_from_collections(
            tmp_path / "input", collections=[train], out=tmp_path / "x", **kwargs
        )
    kwargs["held_out"] = NO_HELD_OUT
    with pytest.raises(SplitRefused, match="held out"):
        train_from_collections(tmp_path / "input", collections=[held], out=tmp_path / "x", **kwargs)
    with pytest.raises(NoEligibleEpisodes):
        train_from_collections(
            tmp_path / "input", collections=[train], out=tmp_path / "x", min_episodes=4, **kwargs
        )
    with pytest.raises(ValueError, match="expected policy"):
        train_from_collections(
            tmp_path / "input",
            collections=[train],
            out=tmp_path / "x",
            **{**kwargs, "expected_sha256": "0" * 64},
        )
    assert not (tmp_path / "x").exists()
    summary = train_from_collections(
        tmp_path / "input",
        collections=[train],
        out=tmp_path / "out",
        **{**kwargs, "held_out": [held]},
    )
    assert summary["accepted_episodes"] == 3 and summary["rejected"] == []
    assert summary["held_out"] == [str(held.resolve())]
    updated = load_checkpoint(tmp_path / "out", expected_sha256=summary["policy_sha"])
    assert updated.manifest["training"]["lineage_dir"] == str(LINEAGE.resolve())
    assert summary["lineage_dir"] == str(LINEAGE.resolve())
    consumed = updated.manifest["training"]["live_consumed"]
    assert [c[0] for c in consumed] == [
        r["match_id"] for r in read_collection(train).ledger.completed
    ]
    assert {p: tree_digest(p) for p in before} == before
    # The trained child is another policy: the same collection is off-policy for it.
    with pytest.raises(NoEligibleEpisodes):
        train_from_collections(
            tmp_path / "out",
            collections=[train],
            out=tmp_path / "grandchild",
            **{**kwargs, "expected_sha256": summary["policy_sha"]},
        )


def test_tampered_training_receipt_refuses_the_whole_update(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect_policy(tmp_path, monkeypatch, "train", loaded, [1, 2], "training")
    match_id = read_collection(train).ledger.completed[1]["match_id"]
    with (train / match_id / "replay.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(SplitRefused, match="failed verification"):
        train_from_collections(
            tmp_path / "input",
            expected_sha256=loaded.policy_sha,
            collections=[train],
            out=tmp_path / "out",
            config=SMALL,
            seed=3,
            held_out=NO_HELD_OUT,
            lineage_dir=LINEAGE,
        )
    assert not (tmp_path / "out").exists()


def test_training_config_has_no_default_budget_or_discount_clock(tmp_path):
    with pytest.raises(TypeError):
        TrainConfig()
    for missing in BUDGET:
        with pytest.raises(TypeError):
            TrainConfig(**{k: v for k, v in BUDGET.items() if k != missing})
    with pytest.raises(ValueError, match="discount_time_unit"):
        TrainConfig(**{**BUDGET, "discount_time_unit": True})
    with pytest.raises(TypeError):
        ppo_update(policy(), None, [], SMALL)  # no seed
    loaded = saved(tmp_path)
    per_decision = TrainConfig(**{**BUDGET, "discount_time_unit": 0.0})
    for name, config in (("seconds", SMALL), ("decisions", per_decision)):
        result = _unsafe_update_from_episodes(
            loaded,
            [live_episode(loaded)],
            verify_episode=lambda _: True,
            out=tmp_path / name,
            config=config,
            seed=1,
            lineage_dir=LINEAGE,
        )
        manifest = load_checkpoint(tmp_path / name, expected_sha256=result["policy_sha"]).manifest
        discount = manifest["training"]["discount"]
        assert discount == result["discount"]
        assert discount["gamma"] == config.gamma and discount["gae_lambda"] == config.gae_lambda
        assert discount["discount_time_unit"] == config.discount_time_unit
    assert result["discount"]["rule"] == "gamma per decision"
    assert manifest["training"]["config"]["discount_time_unit"] == 0.0


def test_output_symlinks_are_refused_even_when_dangling(tmp_path):
    loaded = saved(tmp_path)
    episode = live_episode(loaded)
    (tmp_path / "into").symlink_to(tmp_path / "input" / "child")
    (tmp_path / "dangling").symlink_to(tmp_path / "elsewhere")
    for out, error in (("into", "outside the input"), ("dangling", "already exists")):
        with pytest.raises(ValueError, match=error):
            _unsafe_update_from_episodes(
                loaded,
                [episode],
                verify_episode=lambda _: True,
                out=tmp_path / out,
                config=SMALL,
                seed=1,
                lineage_dir=LINEAGE,
            )
    assert not (tmp_path / "elsewhere").exists()


def test_training_on_another_device_is_refused(tmp_path):
    from types import SimpleNamespace

    loaded = saved(tmp_path)
    episode = live_episode(loaded)
    loaded.model = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    with pytest.raises(ValueError, match="CPU only"):
        _unsafe_update_from_episodes(
            loaded,
            [episode],
            verify_episode=lambda _: True,
            out=tmp_path / "out",
            config=SMALL,
            seed=1,
            lineage_dir=LINEAGE,
        )


def test_public_update_only_trusts_a_real_training_selection(tmp_path, monkeypatch):
    loaded = saved(tmp_path)
    train = collect_policy(tmp_path, monkeypatch, "train", loaded, [1, 2], "training")
    selection = select_training_episodes([train], loaded.policy_sha, held_out=NO_HELD_OUT)
    kwargs = {"config": SMALL, "seed": 1, "lineage_dir": LINEAGE}
    with pytest.raises(TypeError, match="TrainingSelection"):
        update_verified_episodes(loaded, selection.episodes, out=tmp_path / "a", **kwargs)
    forged = TrainingSelection(
        loaded.policy_sha,
        episodes=selection.episodes,
        verifiers={k: (lambda _: True) for k in selection.verifiers},
    )
    with pytest.raises(ValueError, match="another policy"):
        update_verified_episodes(loaded, forged, out=tmp_path / "b", **kwargs)
    other = saved(tmp_path, "other", seed=5)
    with pytest.raises(ValueError, match="another policy"):
        update_verified_episodes(other, selection, out=tmp_path / "c", **kwargs)
    # An episode altered after selection fails the selection's own re-verification.
    tampered = TrainingSelection(
        loaded.policy_sha,
        episodes=copy.deepcopy(selection.episodes),
        verifiers=selection.verifiers,
    )
    outcome = tampered.episodes[0]["final_result"]["outcome"]
    tampered.episodes[0]["final_result"]["outcome"] = -outcome
    with pytest.raises(ValueError, match="verification failed"):
        update_verified_episodes(loaded, tampered, out=tmp_path / "d", **kwargs)
    assert not any((tmp_path / n).exists() for n in "abcd")
    result = update_verified_episodes(loaded, selection, out=tmp_path / "ok", **kwargs)
    assert result["episodes"] == 2


def test_a_trained_candidate_is_bound_to_a_named_lineage(tmp_path):
    loaded = saved(tmp_path)
    episode = live_episode(loaded)
    common = {"verify_episode": lambda _: True, "config": SMALL, "seed": 1}
    with pytest.raises(TypeError, match="lineage_dir"):
        _unsafe_update_from_episodes(loaded, [episode], out=tmp_path / "a", **common)
    with pytest.raises(ValueError, match="lineage"):
        _unsafe_update_from_episodes(
            loaded, [episode], out=tmp_path / "a", lineage_dir="", **common
        )
    result = _unsafe_update_from_episodes(
        loaded, [episode], out=tmp_path / "b", lineage_dir=tmp_path / "L1", **common
    )
    manifest = load_checkpoint(tmp_path / "b", expected_sha256=result["policy_sha"]).manifest
    assert manifest["training"]["lineage_dir"] == str((tmp_path / "L1").resolve())


def test_train_from_collections_leaves_the_global_rng_alone(tmp_path, monkeypatch):
    """Unit 8: the behaviour checkpoint load runs in fork_rng as well."""
    loaded = saved(tmp_path)
    train = collect_policy(tmp_path, monkeypatch, "train", loaded, [1, 2], "training")
    torch.manual_seed(99)
    expected = torch.rand(3)
    torch.manual_seed(99)
    train_from_collections(
        tmp_path / "input", expected_sha256=loaded.policy_sha, collections=[train],
        out=tmp_path / "out", config=SMALL, seed=3, held_out=NO_HELD_OUT, lineage_dir=LINEAGE,
    )  # fmt: skip
    assert torch.equal(torch.rand(3), expected)


def test_ppo_children_carry_the_ancestry_demonstration_record(tmp_path):
    """Unit 8: a PPO step neither resets nor drops the demonstration ledger."""
    record = [["a" * 32, 2]]
    parent = saved(
        tmp_path,
        training={"parent_policy_sha": "f" * 64, "live_consumed": [],
                  "demonstrations_consumed": record},
    )  # fmt: skip
    result = _unsafe_update_from_episodes(
        parent, [live_episode(parent, match=7)], verify_episode=lambda _: True,
        out=tmp_path / "child", config=SMALL, seed=1, lineage_dir=LINEAGE,
    )  # fmt: skip
    child = load_checkpoint(tmp_path / "child", expected_sha256=result["policy_sha"])
    assert child.manifest["training"]["demonstrations_consumed"] == record
    stripped = saved(tmp_path, "stripped", training={"parent_policy_sha": "f" * 64,
                                                     "live_consumed": []})  # fmt: skip
    with pytest.raises(ValueError, match="ancestry's consumed record"):
        _unsafe_update_from_episodes(
            stripped, [live_episode(stripped, match=8)], verify_episode=lambda _: True,
            out=tmp_path / "x", config=SMALL, seed=1, lineage_dir=LINEAGE,
        )  # fmt: skip
    assert not (tmp_path / "x").exists()
