from __future__ import annotations

import copy
import json

import pytest
import torch

from anima3.neural.checkpoints import load_checkpoint, save_checkpoint, schema_fingerprint
from anima3.neural.model import PolicyConfig, RecurrentPolicy, masked_distribution
from anima3.neural.schema import ACTION_NAMES, FEATURE_NAMES
from anima3.neural.train import (
    LIVE_SCHEMA,
    TrainConfig,
    Trajectory,
    clipped_policy_loss,
    generalized_advantage,
    ppo_update,
    update_verified_episodes,
)


def policy():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    return RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), 16, 16))


def trajectory(model, length=12):
    episode = Trajectory(terminated=True)
    hidden = model.initial_state()
    for index in range(length):
        features = torch.zeros(1, len(FEATURE_NAMES))
        features[0, index % len(FEATURE_NAMES)] = 0.5
        mask = torch.zeros(1, len(ACTION_NAMES), dtype=torch.bool)
        mask[0, :3] = True
        action, log_prob, value, next_hidden = model.act(features, mask, hidden)
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


def live_episode(loaded):
    trace = trajectory(loaded.model)
    return {
        "schema": LIVE_SCHEMA,
        "domain": "servuo",
        "sampling": "categorical",
        "policy_sha": loaded.policy_sha,
        "schema_fingerprint": schema_fingerprint(),
        "player_serial": 57,
        "verification": {
            "verified": True,
            "complete": True,
            "training": True,
            "aborted": False,
            "match_id": "match-1",
            "replay_sha256": "a" * 64,
        },
        "transitions": [
            {
                "features": trace.features[index],
                "mask": trace.masks[index],
                "action": trace.actions[index],
                "log_prob": trace.log_probs[index],
                "value": trace.values[index],
                "hidden_in": trace.hidden[index],
                "reward": trace.rewards[index],
                "dt": trace.durations[index],
                "done": index == len(trace) - 1,
                "accepted": True,
            }
            for index in range(len(trace))
        ],
    }


def test_illegal_actions_have_exactly_zero_probability():
    logits = torch.tensor([[100.0, -20.0, 400.0]])
    distribution = masked_distribution(logits, torch.tensor([[False, True, False]]))
    assert distribution.probs.tolist() == [[0.0, 1.0, 0.0]]
    assert set(distribution.sample((50,)).flatten().tolist()) == {1}
    with pytest.raises(ValueError, match="at least one"):
        masked_distribution(logits, torch.zeros_like(logits, dtype=torch.bool))


def test_recurrent_reset_matches_new_episode():
    model = policy()
    features = torch.randn(2, 3, len(FEATURE_NAMES))
    hidden = torch.randn(2, model.config.hidden_size)
    reset = torch.zeros(2, 3, dtype=torch.bool)
    reset[:, 0] = True
    actual = model(features, hidden, reset)
    expected = model(features)
    for left, right in zip(actual, expected, strict=True):
        assert torch.allclose(left, right, atol=1e-6)


def test_gradient_crosses_recurrent_time_steps():
    model = policy()
    features = torch.randn(1, 5, len(FEATURE_NAMES), requires_grad=True)
    _, values, _ = model(features)
    values[0, -1].backward()
    assert float(features.grad[0, 0].abs().sum()) > 0


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


def test_ppo_changes_real_weights_and_keeps_finite_losses():
    model = policy()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = ppo_update(
        model,
        optimizer,
        [trajectory(model)],
        TrainConfig(epochs=2, sequence_length=6, burn_in=2, batch_sequences=2),
    )
    assert metrics["optimized_steps"] > 0
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    assert metrics["approximate_kl"] >= -1e-7


def test_checkpoint_roundtrip_is_hash_and_schema_checked(tmp_path):
    model = policy()
    manifest = save_checkpoint(tmp_path, model, training={"environment_steps": 12})
    loaded = load_checkpoint(tmp_path)
    assert loaded.policy_sha == manifest["weights"]["sha256"]
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[name])
    weight_path = tmp_path / manifest["weights"]["file"]
    data = bytearray(weight_path.read_bytes())
    data[-20] ^= 1
    weight_path.write_bytes(data)
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(tmp_path)


def test_checkpoint_rejects_schema_drift(tmp_path):
    save_checkpoint(tmp_path, policy())
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema"]["actions"][0] = "unknown-new-action"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema mismatch"):
        load_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "corruption,error",
    [
        ("version", "off-policy"),
        ("rejected", "rejected or substituted"),
        ("log_prob", "log_prob mismatch"),
        ("hidden", "state chain mismatch"),
        ("unfinished", "terminal boundary"),
    ],
)
def test_live_updates_fail_closed_before_weight_change(tmp_path, corruption, error):
    save_checkpoint(tmp_path / "input", policy())
    loaded = load_checkpoint(tmp_path / "input")
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
    before = copy.deepcopy(loaded.model.state_dict())
    with pytest.raises(ValueError, match=error):
        update_verified_episodes(
            loaded, [episode], verify_episode=lambda _: True, out=tmp_path / "output"
        )
    assert all(
        torch.equal(before[name], value) for name, value in loaded.model.state_dict().items()
    )
    assert not (tmp_path / "output" / "manifest.json").exists()


def test_live_update_requires_trusted_verifier_and_saves_actual_weights(tmp_path):
    save_checkpoint(tmp_path / "input", policy())
    loaded = load_checkpoint(tmp_path / "input")
    episode = live_episode(loaded)
    with pytest.raises(ValueError, match="verification failed"):
        update_verified_episodes(
            loaded, [episode], verify_episode=lambda _: False, out=tmp_path / "bad"
        )
    result = update_verified_episodes(
        loaded,
        [episode],
        verify_episode=lambda _: True,
        out=tmp_path / "output",
        config=TrainConfig(epochs=1, sequence_length=8, discount_time_unit=1.0),
    )
    assert result["policy_sha"] != result["parent_policy_sha"]
    assert result["metrics"]["optimized_steps"] > 0
    updated = load_checkpoint(tmp_path / "output")
    assert updated.manifest["training"]["source"] == "verified-servuo-on-policy"
    assert updated.manifest["automatic_promotion"] is False


def test_cancelled_option_and_losing_outcome_keep_complete_recurrent_experience(tmp_path):
    save_checkpoint(tmp_path / "input", policy())
    loaded = load_checkpoint(tmp_path / "input")
    episode = live_episode(loaded)
    episode["transitions"][0].update(
        policy_accepted=True, execution_started=False, cancellation_reason="mask_changed"
    )
    episode["transitions"][-1]["reward"] = -1.0
    result = update_verified_episodes(
        loaded,
        [episode],
        verify_episode=lambda _: True,
        out=tmp_path / "output",
        config=TrainConfig(epochs=1, sequence_length=8, discount_time_unit=1.0),
    )
    assert result["metrics"]["rollout_steps"] == len(episode["transitions"])
    assert result["policy_sha"] != loaded.policy_sha


def test_checkpoint_restores_sampling_rng_and_optimizer(tmp_path):
    from anima3.neural.checkpoints import restore_optimizer, restore_training_rng

    model = policy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ppo_update(model, optimizer, [trajectory(model)], TrainConfig(epochs=1, sequence_length=12))
    save_checkpoint(tmp_path, model, optimizer=optimizer)
    expected_random = torch.rand(5)
    loaded = load_checkpoint(tmp_path)
    restored = torch.optim.Adam(loaded.model.parameters(), lr=0.002)
    assert restore_optimizer(loaded, restored)
    assert restore_training_rng(loaded)
    assert torch.equal(torch.rand(5), expected_random)
    assert restored.param_groups[0]["lr"] == 0.001
    assert len(restored.state) == len(optimizer.state)


def test_live_actor_alias_and_both_sides_same_match_are_distinct(tmp_path):
    save_checkpoint(tmp_path / "input", policy())
    loaded = load_checkpoint(tmp_path / "input")
    first = live_episode(loaded)
    first["actor"] = first.pop("player_serial")
    second = copy.deepcopy(first)
    second["actor"] = 58
    original = copy.deepcopy(loaded.model.state_dict())
    result = update_verified_episodes(
        loaded,
        [first, second],
        verify_episode=lambda _: True,
        out=tmp_path / "output",
        config=TrainConfig(epochs=1, sequence_length=12),
    )
    assert result["episodes"] == 2
    assert all(
        torch.equal(original[name], value) for name, value in loaded.model.state_dict().items()
    )
    updated = load_checkpoint(tmp_path / "output")
    assert updated.manifest["training"]["live_consumed"] == [["match-1", 57], ["match-1", 58]]


def test_live_rejects_already_consumed_episode_identity(tmp_path):
    save_checkpoint(tmp_path / "input", policy(), training={"live_consumed": [["match-1", 57]]})
    loaded = load_checkpoint(tmp_path / "input")
    with pytest.raises(ValueError, match="duplicate"):
        update_verified_episodes(
            loaded, [live_episode(loaded)], verify_episode=lambda _: True, out=tmp_path / "output"
        )


def test_collect_distinguishes_game_draw_from_budget_truncation():
    from anima3.neural.train import collect_simulator

    model = policy()
    episodes, metrics = collect_simulator(model, steps=13, seed=4, max_episode_steps=8)
    assert metrics["completed_games"] == 1
    assert episodes[0].terminated and not episodes[0].truncated
    assert episodes[0].bootstrap_value == 0
    assert episodes[0].info["time_limit"] is True
    assert not episodes[1].terminated and episodes[1].truncated


def test_live_rejects_deterministic_action_selection(tmp_path):
    save_checkpoint(tmp_path / "input", policy())
    loaded = load_checkpoint(tmp_path / "input")
    episode = live_episode(loaded)
    episode["sampling"] = "argmax"
    with pytest.raises(ValueError, match="stochastic categorical"):
        update_verified_episodes(
            loaded, [episode], verify_episode=lambda _: True, out=tmp_path / "output"
        )


def test_live_update_accepts_both_sides_of_100_matches_but_caps_larger_batches(tmp_path):
    save_checkpoint(tmp_path / "input", policy())
    loaded = load_checkpoint(tmp_path / "input")
    template = live_episode(loaded)
    # A one-decision complete fixture keeps this batch-boundary test inexpensive.
    template["transitions"] = template["transitions"][:1]
    template["transitions"][0].update(done=True, reward=1.0)
    episodes = []
    for match in range(100):
        for serial in (57, 58):
            episode = copy.deepcopy(template)
            episode["verification"]["match_id"] = f"{match:032x}"
            episode["player_serial"] = serial
            episodes.append(episode)
    result = update_verified_episodes(
        loaded,
        episodes,
        verify_episode=lambda _: True,
        out=tmp_path / "output",
        config=TrainConfig(epochs=1, sequence_length=1, burn_in=0, batch_sequences=200),
    )
    assert result["episodes"] == 200
    assert result["metrics"]["rollout_steps"] == 200
    with pytest.raises(ValueError, match="1..200"):
        update_verified_episodes(
            loaded,
            episodes + [copy.deepcopy(template)],
            verify_episode=lambda _: True,
            out=tmp_path / "oversized",
        )
    assert not (tmp_path / "oversized").exists()
