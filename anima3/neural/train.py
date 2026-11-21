"""Explicit, bounded recurrent PPO on verified unit-6 training episodes.

Requires the ``neural`` extra and is never imported by ``anima3`` or
``anima3.neural``. Nothing here runs on its own: there is no command-line entry,
no schedule and no promotion. A caller invokes :func:`train_from_collections`
(or :func:`update_verified_episodes`) once, with an explicit configuration whose
epoch, sequence and batch budgets are fixed and bounded, and an explicit seed.

- The behaviour checkpoint is loaded by the caller's expected SHA-256.
- Training input comes only from explicitly named ``training`` collections, and
  only episodes that ``LiveReplayVerifier`` accepts with a ``completed`` ledger
  row (see ``splits.py``). Held-out evaluation episodes are refused.
- Every episode is re-derived against the exact behaviour checkpoint (action
  likelihood, value and recurrent state chain) before any parameter changes.
- The result is a new checkpoint in a new directory, identified by the SHA-256
  of its weights and re-loaded by that SHA-256. The input is never written.
- A fixed seed gives bit-identical weights on the same platform and torch build.

The recurrent PPO core (``TrainConfig`` through ``ppo_update``) and the live
trajectory re-derivation are ported from the training lineage ``819f4f4``.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .checkpoints import LoadedCheckpoint, load_checkpoint, save_checkpoint, schema_fingerprint
from .live import EPISODE_SCHEMA, atomic_json
from .model import RecurrentPolicy, masked_distribution
from .splits import TrainingEpisodeVerifier, TrainingSelection, episode_identity

MAX_EPOCHS = 64
MAX_EPISODES = 200
MAX_TRANSITIONS = 50_000
MAX_TOTAL_TRANSITIONS = 1_000_000


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    """PPO settings. The budget and the discount clock have no defaults.

    ``epochs``, ``sequence_length``, ``burn_in``, ``batch_sequences`` and
    ``discount_time_unit`` must be stated by the caller. The source live update
    used ``discount_time_unit=1.0`` (``gamma ** elapsed_seconds``); ``0`` means one
    discount per decision.
    """

    epochs: int
    sequence_length: int
    burn_in: int
    batch_sequences: int
    discount_time_unit: float
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.04

    def __post_init__(self) -> None:
        if type(self.discount_time_unit) not in (int, float):
            raise ValueError("discount_time_unit must be a number")
        if not all(
            math.isfinite(value)
            for value in (
                self.learning_rate,
                self.gamma,
                self.gae_lambda,
                self.clip_range,
                self.value_coef,
                self.entropy_coef,
                self.max_grad_norm,
                self.target_kl,
                self.discount_time_unit,
            )
        ):
            raise ValueError("training coefficients must be finite")
        if not (0 < self.learning_rate <= 0.1 and 0 < self.gamma <= 1):
            raise ValueError("invalid learning rate or discount")
        if not (0 <= self.gae_lambda <= 1 and 0 < self.clip_range < 1):
            raise ValueError("invalid GAE lambda or PPO clip range")
        if self.value_coef < 0 or self.entropy_coef < 0:
            raise ValueError("value and entropy coefficients must be nonnegative")
        if not all(
            type(value) is int
            for value in (self.epochs, self.sequence_length, self.burn_in, self.batch_sequences)
        ):
            raise ValueError("training budget fields must be integers")
        if min(self.epochs, self.sequence_length, self.batch_sequences) < 1 or self.burn_in < 0:
            raise ValueError("invalid recurrent training budget")
        if self.epochs > MAX_EPOCHS or self.sequence_length > 4096 or self.batch_sequences > 1024:
            raise ValueError("training budget exceeds the fixed upper bounds")
        if self.discount_time_unit < 0 or self.max_grad_norm <= 0 or self.target_kl <= 0:
            raise ValueError("invalid discount time unit or optimization bounds")


@dataclass
class Trajectory:
    features: list[list[float]] = field(default_factory=list)
    masks: list[list[bool]] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    hidden: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    bootstrap_value: float = 0.0
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)


def generalized_advantage(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    boundaries: Tensor,
    *,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    durations: Tensor | None = None,
    discount_time_unit: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """GAE bootstraps truncations but stops its trace at every episode boundary.

    ``terminated`` means a real terminal state, whose next value is zero.
    ``boundaries`` also includes time limits and collection truncations.  A
    duration represents the time between this decision and the next state.
    """
    if rewards.ndim != 1 or not all(
        tensor.shape == rewards.shape for tensor in (values, next_values, terminated, boundaries)
    ):
        raise ValueError("GAE arrays must have the same one-dimensional shape")
    if discount_time_unit:
        if durations is None or durations.shape != rewards.shape or bool((durations <= 0).any()):
            raise ValueError("positive transition durations are required")
        powers = durations / discount_time_unit
        discounts = torch.pow(torch.full_like(rewards, gamma), powers)
        traces = torch.pow(torch.full_like(rewards, gae_lambda), powers)
    else:
        discounts = torch.full_like(rewards, gamma)
        traces = torch.full_like(rewards, gae_lambda)
    deltas = rewards + discounts * (~terminated.bool()) * next_values - values
    advantage = torch.zeros_like(rewards)
    running = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for index in range(len(rewards) - 1, -1, -1):
        running = (
            deltas[index] + discounts[index] * traces[index] * (~boundaries[index].bool()) * running
        )
        advantage[index] = running
    return advantage, advantage + values


def clipped_policy_loss(log_probs: Tensor, old_log_probs: Tensor, advantages: Tensor, clip: float):
    ratio = (log_probs - old_log_probs).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)


def _targets(episodes: list[Trajectory], config: TrainConfig) -> list[tuple[Tensor, Tensor]]:
    targets = []
    for episode in episodes:
        values = torch.tensor(episode.values, dtype=torch.float32)
        terminals = torch.zeros(len(episode), dtype=torch.bool)
        boundaries = terminals.clone()
        terminals[-1] = episode.terminated
        boundaries[-1] = True
        targets.append(
            generalized_advantage(
                torch.tensor(episode.rewards, dtype=torch.float32),
                values,
                torch.cat((values[1:], torch.tensor([episode.bootstrap_value]))),
                terminals,
                boundaries,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                durations=torch.tensor(episode.durations, dtype=torch.float32),
                discount_time_unit=config.discount_time_unit,
            )
        )
    free_choices = [torch.tensor(episode.masks).sum(-1) > 1 for episode in episodes]
    # Forced holds remain in the recurrent trajectory and critic targets, but
    # their advantages must not set the scale of the actor's actual decisions.
    actor_advantages = torch.cat(
        [target[0][free] for target, free in zip(targets, free_choices, strict=True)]
    )
    if len(actor_advantages):
        mean = actor_advantages.mean()
        std = actor_advantages.std(unbiased=False).clamp_min(1e-8)
    else:
        mean, std = 0.0, 1.0
    return [
        (torch.where(free, (adv - mean) / std, 0.0), returns)
        for (adv, returns), free in zip(targets, free_choices, strict=True)
    ]


def _chunks(episodes: list[Trajectory], length: int):
    return [
        (episode, start)
        for episode, data in enumerate(episodes)
        for start in range(0, len(data), length)
    ]


def _batch(
    model: RecurrentPolicy,
    episodes: list[Trajectory],
    selections: list[tuple[int, int]],
    config: TrainConfig,
    targets: list[tuple[Tensor, Tensor]] | None,
) -> dict[str, Tensor]:
    """Recompute burn-in without gradients, then retain recurrence through the chunk.

    Truncated BPTT detaches only at chunk boundaries.  Padding has a hold-only
    mask and contributes neither policy nor value loss.
    """
    device = model.device
    count, length = len(selections), config.sequence_length
    batch = {
        "features": torch.zeros(count, length, model.config.feature_dim, device=device),
        "masks": torch.zeros(
            count, length, model.config.action_dim, device=device, dtype=torch.bool
        ),
        "actions": torch.zeros(count, length, device=device, dtype=torch.long),
        "old_log_probs": torch.zeros(count, length, device=device),
        "old_values": torch.zeros(count, length, device=device),
        "advantages": torch.zeros(count, length, device=device),
        "returns": torch.zeros(count, length, device=device),
        "valid": torch.zeros(count, length, device=device, dtype=torch.bool),
        "hidden": model.initial_state(count),
    }
    batch["masks"][:, :, 0] = True
    for row, (episode_index, start) in enumerate(selections):
        episode = episodes[episode_index]
        stop = min(start + length, len(episode))
        size = stop - start
        burn_start = 0 if targets is None else max(0, start - config.burn_in)
        hidden = torch.tensor([episode.hidden[burn_start]], device=device)
        if burn_start < start:
            with torch.no_grad():
                _, _, hidden = model(
                    torch.tensor([episode.features[burn_start:start]], device=device), hidden
                )
        batch["hidden"][row] = hidden[0]
        for key, source in (
            ("features", episode.features),
            ("masks", episode.masks),
            ("actions", episode.actions),
            ("old_log_probs", episode.log_probs),
            ("old_values", episode.values),
        ):
            batch[key][row, :size] = torch.as_tensor(source[start:stop], device=device)
        if targets is not None:
            advantage, returns = targets[episode_index]
            batch["advantages"][row, :size] = advantage[start:stop].to(device)
            batch["returns"][row, :size] = returns[start:stop].to(device)
        batch["valid"][row, :size] = True
    return batch


def ppo_update(
    model: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    episodes: list[Trajectory],
    config: TrainConfig,
    *,
    seed: int,
) -> dict:
    if not episodes or not all(len(episode) for episode in episodes):
        raise ValueError("PPO requires non-empty trajectories")
    if any(e.info.get("ppo_eligible") is False for e in episodes):
        raise ValueError("Supervised demonstrations are not PPO trajectories")
    targets = _targets(episodes, config)
    chunks = _chunks(episodes, config.sequence_length)
    rng = random.Random(seed)
    totals: dict[str, float] = {}
    count = 0
    actor_count = 0
    actor_metric_names = {"policy_loss", "entropy", "approximate_kl", "clip_fraction"}
    stopped = False
    optimizer_steps = 0
    model.train()
    for _ in range(config.epochs):
        rng.shuffle(chunks)
        for offset in range(0, len(chunks), config.batch_sequences):
            batch = _batch(
                model, episodes, chunks[offset : offset + config.batch_sequences], config, targets
            )
            log_probs, entropy, values, _ = model.evaluate_actions(
                batch["features"], batch["masks"], batch["actions"], batch["hidden"]
            )
            valid = batch["valid"]
            free = valid & (batch["masks"].sum(-1) > 1)
            actor_weight = int(free.sum())
            if actor_weight:
                log_ratio = log_probs[free] - batch["old_log_probs"][free]
                approximate_kl = ((log_ratio.exp() - 1) - log_ratio).mean()
                if float(approximate_kl.detach()) > config.target_kl * 1.5:
                    stopped = True
                    break
                policy_loss = clipped_policy_loss(
                    log_probs[free],
                    batch["old_log_probs"][free],
                    batch["advantages"][free],
                    config.clip_range,
                ).mean()
                mean_entropy = entropy[free].mean()
                clip_fraction = ((log_ratio.exp() - 1).abs() > config.clip_range).float().mean()
            else:
                # A chunk can be entirely waiting/casting.  Its critic and
                # recurrent state still learn without an empty actor mean.
                policy_loss = mean_entropy = approximate_kl = clip_fraction = values.new_zeros(())
            clipped_values = batch["old_values"] + (values - batch["old_values"]).clamp(
                -config.clip_range, config.clip_range
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (values[valid] - batch["returns"][valid]).square(),
                    (clipped_values[valid] - batch["returns"][valid]).square(),
                ).mean()
            )
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * mean_entropy
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not bool(torch.isfinite(gradient)):
                raise ValueError("non-finite PPO gradient")
            optimizer.step()
            optimizer_steps += 1
            weight = int(valid.sum())
            metrics = {
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(mean_entropy.detach()),
                "approximate_kl": float(approximate_kl.detach()),
                "clip_fraction": float(clip_fraction.detach()),
                "grad_norm": float(gradient),
            }
            for key, value in metrics.items():
                metric_weight = actor_weight if key in actor_metric_names else weight
                totals[key] = totals.get(key, 0.0) + value * metric_weight
            count += weight
            actor_count += actor_weight
        if stopped:
            break
    model.eval()
    return {
        **{
            key: value / max(actor_count if key in actor_metric_names else count, 1)
            for key, value in totals.items()
        },
        "optimized_steps": count,
        "policy_optimized_steps": actor_count,
        "optimizer_steps": optimizer_steps,
        "kl_early_stop": stopped,
        "rollout_steps": sum(map(len, episodes)),
    }


def _live_trajectory(loaded: LoadedCheckpoint, episode: dict) -> Trajectory:
    """Independently recompute the behavior likelihood and recurrent state chain."""
    if episode.get("schema") != EPISODE_SCHEMA or episode.get("domain") != "servuo":
        raise ValueError("unsupported live trajectory schema/domain")
    if episode.get("sampling") != "categorical":
        raise ValueError("PPO requires stochastic categorical behavior, not argmax evaluation")
    if episode.get("policy_sha") != loaded.policy_sha:
        raise ValueError("off-policy trajectory: checkpoint SHA does not match")
    if episode.get("schema_fingerprint") != schema_fingerprint():
        raise ValueError("live observation schema mismatch")
    verification = episode.get("verification", {})
    if not (
        verification.get("verified") is True
        and verification.get("complete") is True
        and verification.get("training") is True
        and verification.get("aborted") is False
    ):
        raise ValueError("only complete verified training duels may update the policy")
    digest = verification.get("replay_sha256", "")
    if (
        not verification.get("match_id")
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("missing replay provenance")
    transitions = episode.get("transitions", [])
    if not isinstance(transitions, list) or not (1 <= len(transitions) <= MAX_TRANSITIONS):
        raise ValueError("invalid live trajectory length")
    model = loaded.model
    hidden = model.initial_state()
    result = Trajectory(terminated=True, info={"verification": verification})
    for index, step in enumerate(transitions):
        if step.get("accepted") is not True:
            raise ValueError("rejected or substituted actions invalidate an on-policy episode")
        if step.get("done") is not (index == len(transitions) - 1):
            raise ValueError("live trajectory must contain exactly one terminal boundary")
        features = step.get("features")
        mask = step.get("mask")
        action = step.get("action")
        if not isinstance(features, list) or len(features) != model.config.feature_dim:
            raise ValueError("live features have wrong shape")
        if not all(
            isinstance(x, (int, float)) and math.isfinite(x) and -1.001 <= x <= 1.001
            for x in features
        ):
            raise ValueError("invalid normalized live features")
        if (
            not isinstance(mask, list)
            or len(mask) != model.config.action_dim
            or not all(type(x) is bool for x in mask)
        ):
            raise ValueError("invalid live action mask")
        if type(action) is not int or not (0 <= action < len(mask)) or not mask[action]:
            raise ValueError("live action was not legal in its recorded mask")
        hidden_in = torch.tensor(
            step.get("hidden_in", []), dtype=torch.float32, device=model.device
        )
        if hidden_in.shape != hidden[0].shape or not torch.allclose(
            hidden_in, hidden[0], atol=2e-4, rtol=2e-4
        ):
            raise ValueError("live recurrent state chain mismatch")
        with torch.no_grad():
            logits, values, next_hidden = model(
                torch.tensor([[features]], device=model.device), hidden
            )
            distribution = masked_distribution(
                logits[:, 0], torch.tensor([mask], device=model.device)
            )
            log_prob = float(
                distribution.log_prob(torch.tensor([action], device=model.device)).item()
            )
            value = float(values.item())
        for key, expected in (("log_prob", log_prob), ("value", value)):
            actual = step.get(key)
            if (
                not isinstance(actual, (float, int))
                or not math.isfinite(actual)
                or not math.isclose(actual, expected, rel_tol=2e-4, abs_tol=2e-4)
            ):
                raise ValueError(f"live behavior policy {key} mismatch")
        reward, duration = step.get("reward"), step.get("dt")
        if not isinstance(reward, (float, int)) or not math.isfinite(reward) or abs(reward) > 10:
            raise ValueError("invalid live reward")
        if (
            not isinstance(duration, (float, int))
            or not math.isfinite(duration)
            or not (0 < duration <= 600)
        ):
            raise ValueError("live dt must be elapsed seconds in (0, 600]")
        result.features.append(features)
        result.masks.append(mask)
        result.actions.append(action)
        result.hidden.append(hidden[0].cpu().tolist())
        result.values.append(value)
        result.log_probs.append(log_prob)
        result.rewards.append(float(reward))
        result.durations.append(float(duration))
        hidden = next_hidden
    return result


def _episode_actor(episode: dict) -> int:
    actor = episode.get("player_serial", episode.get("actor"))
    if type(actor) is not int or actor <= 0:
        raise ValueError("live episode requires a positive actor serial")
    if "actor" in episode and episode["actor"] != actor:
        raise ValueError("conflicting actor serials in live episode")
    return actor


class NoEligibleEpisodes(ValueError):
    """Too few verified training episodes; no checkpoint was written."""

    def __init__(self, rejected, count, minimum):
        self.rejected = rejected
        super().__init__(f"{count} verified training episodes, need {minimum}")


def _consumed(loaded: LoadedCheckpoint) -> list[list]:
    consumed = loaded.manifest.get("training", {}).get("live_consumed", [])
    if not isinstance(consumed, list) or not all(
        isinstance(i, list) and len(i) == 2 and isinstance(i[0], str) and type(i[1]) is int
        for i in consumed
    ):
        raise ValueError("checkpoint lineage has a malformed consumed-episode list")
    return [list(i) for i in consumed]


def _demonstrations_consumed(loaded: LoadedCheckpoint) -> list[list]:
    """Teacher demonstrations this ancestry already imitated; carried by every child."""
    consumed = loaded.manifest.get("training", {}).get("demonstrations_consumed", [])
    if not isinstance(consumed, list) or not all(
        isinstance(i, list) and len(i) == 2 and isinstance(i[0], str) and type(i[1]) is int
        for i in consumed
    ):
        raise ValueError("checkpoint lineage has a malformed consumed-demonstration list")
    return [list(i) for i in consumed]


def ancestry_consumed(loaded: LoadedCheckpoint) -> tuple[list[list], list[list]]:
    """The live episodes and demonstrations consumed by the whole ancestry of ``loaded``.

    Every trained child (PPO or imitation) carries both lists forward from its
    parent, so a parent's lists already cover its full ancestry. A trained
    checkpoint (one naming ``parent_policy_sha``) that lacks either list would
    silently reset the record, and is refused.
    """
    training = loaded.manifest.get("training", {})
    if not isinstance(training, dict):
        raise TypeError("checkpoint training metadata is not an object")
    if "parent_policy_sha" in training and not (
        "live_consumed" in training and "demonstrations_consumed" in training
    ):
        raise ValueError("trained checkpoint lacks its ancestry's consumed record")
    return _consumed(loaded), _demonstrations_consumed(loaded)


SIMULATOR_SOURCE = "simulator-curriculum-league"
SIMULATOR_DOMAIN = "approximate-pre-aos-simulator"  # drills.SIM_DOMAIN


def ancestry_simulator(loaded: LoadedCheckpoint) -> tuple[bool, list[list]]:
    """Whether the ancestry of ``loaded`` was trained in the simulator, and on what.

    A simulator-trained checkpoint (``simulator_trained``) and every descendant
    (``simulator_ancestry``) stays marked: each trained child, live PPO and
    imitation alike, carries the marker and the ``simulator_consumed`` record
    forward, so no descendant can be presented as purely live-trained. Refused,
    rather than read as unmarked: a marker that is not a bool, a simulator-labelled
    checkpoint (league source or simulator domain) without its markers and record,
    a marked checkpoint without its record, ``simulator_trained`` without
    ``simulator_ancestry``, and an unmarked checkpoint that names simulator data.
    """
    training = loaded.manifest.get("training", {})
    if not isinstance(training, dict):
        raise TypeError("checkpoint training metadata is not an object")
    markers = ("simulator_trained", "simulator_ancestry")
    if any(k in training and type(training[k]) is not bool for k in markers):
        raise ValueError("checkpoint has a malformed simulator marker")
    labelled = (
        training.get("source") == SIMULATOR_SOURCE or training.get("domain") == SIMULATOR_DOMAIN
    )
    if labelled and not all(k in training for k in (*markers, "simulator_consumed")):
        raise ValueError("simulator-labelled checkpoint lacks its simulator record")
    trained = training.get("simulator_trained", False)
    marked = trained or training.get("simulator_ancestry", False)
    if marked and "simulator_consumed" not in training:
        raise ValueError("simulator-trained checkpoint lacks its simulator record")
    if trained and training.get("simulator_ancestry") is not True:
        raise ValueError("simulator-trained checkpoint lacks its simulator ancestry marker")
    consumed = training.get("simulator_consumed", [])
    if not isinstance(consumed, list) or not all(
        isinstance(i, list)
        and len(i) == 4
        and isinstance(i[0], str)
        and type(i[1]) is int
        and isinstance(i[2], str)
        and type(i[3]) is int
        for i in consumed
    ):
        raise ValueError("checkpoint lineage has a malformed simulator record")
    if consumed and not marked:
        raise ValueError("an unmarked checkpoint names simulator training data")
    return marked, [list(i) for i in consumed]


def _fresh_output(out, parent: Path) -> Path:
    raw = Path(out).absolute()
    resolved = raw.resolve()
    parent = Path(parent).resolve()
    if resolved == parent or resolved.is_relative_to(parent) or parent.is_relative_to(resolved):
        raise ValueError("output checkpoint must be outside the input checkpoint")
    # A symbolic link, even a dangling one, could redirect the write elsewhere.
    if raw.is_symlink() or raw.exists() or resolved.exists():
        raise ValueError("output checkpoint directory already exists; choose a new one")
    return resolved


def _discount(config: TrainConfig) -> dict:
    unit = config.discount_time_unit
    return {
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "discount_time_unit": unit,
        "rule": f"gamma**(dt_seconds/{unit})" if unit else "gamma per decision",
    }


def update_verified_episodes(
    loaded: LoadedCheckpoint,
    selection: TrainingSelection,
    *,
    out: str | Path,
    config: TrainConfig,
    seed: int,
    lineage_dir,
) -> dict:
    """One bounded PPO update from a :class:`TrainingSelection` for this exact checkpoint.

    Every episode is re-verified through the selection's own
    ``TrainingEpisodeVerifier`` (collection purpose, plan SHA-256,
    ``LiveReplayVerifier`` with a completed ledger row) for this policy; a
    caller-supplied verifier is never trusted.
    """
    if type(selection) is not TrainingSelection:
        raise TypeError("training input must be a TrainingSelection from select_training_episodes")
    if selection.policy_sha != loaded.policy_sha or not all(
        type(v) is TrainingEpisodeVerifier and v.policy_sha == loaded.policy_sha
        for v in selection.verifiers.values()
    ):
        raise ValueError("training selection was verified for another policy")
    return _unsafe_update_from_episodes(
        loaded,
        selection.episodes,
        verify_episode=selection.verify,
        out=out,
        config=config,
        seed=seed,
        lineage_dir=lineage_dir,
    )


def _unsafe_update_from_episodes(
    loaded: LoadedCheckpoint,
    episodes: list[dict],
    *,
    verify_episode: Callable[[dict], bool],
    out: str | Path,
    config: TrainConfig,
    seed: int,
    lineage_dir,
) -> dict:
    """Private and unsafe: trusts ``verify_episode``. Use :func:`update_verified_episodes`.

    Beyond the verifier it refuses other policies, held-out evaluation episodes,
    substituted actions, broken recurrent state histories and episodes this
    lineage already consumed. All episodes are validated before any parameter
    changes, and nothing is written unless every check passes. Training runs on
    CPU: ``fork_rng`` isolates the CPU generator only, so a model on another
    device is refused.
    """
    if not isinstance(config, TrainConfig):
        raise TypeError("an explicit TrainConfig budget is required")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("an explicit non-negative integer seed is required")
    if not callable(verify_episode) or not (1 <= len(episodes) <= MAX_EPISODES):
        raise ValueError(f"trusted verifier and 1..{MAX_EPISODES} live episodes required")
    if not isinstance(lineage_dir, (str, Path)) or not str(lineage_dir):
        raise ValueError("name the selection lineage directory this candidate belongs to")
    lineage = str(Path(lineage_dir).resolve())
    if loaded.model.device.type != "cpu":
        raise ValueError("bounded training runs on CPU only")
    out = _fresh_output(out, loaded.directory)
    consumed, demonstrations_consumed = ancestry_consumed(loaded)
    simulator_ancestry, simulator_consumed = ancestry_simulator(loaded)
    seen = {tuple(identity) for identity in consumed}
    trajectories = []
    for episode in episodes:
        if not isinstance(episode, dict) or episode.get("purpose") != "training":
            raise ValueError("held-out evaluation episodes are never training input")
        identity = episode_identity(episode)
        if identity in seen:
            raise ValueError("duplicate live episode")
        seen.add(identity)
        consumed.append(list(identity))
        if verify_episode(episode) is not True:
            raise ValueError("trusted replay verification failed")
        trajectories.append(_live_trajectory(loaded, episode))
    if sum(map(len, trajectories)) > MAX_TOTAL_TRANSITIONS:
        raise ValueError("training batch exceeds the fixed transition budget")
    candidate = copy.deepcopy(loaded.model)
    # A private RNG scope: the fixed seed decides this update and nothing else.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=config.learning_rate, eps=1e-5)
        metrics = ppo_update(candidate, optimizer, trajectories, config, seed=seed)
    training = {
        "source": "verified-servuo-on-policy",
        "parent_policy_sha": loaded.policy_sha,
        # Only this lineage's selection record may decide on the new checkpoint.
        "lineage_dir": lineage,
        "config": asdict(config),
        "budget": {
            "epochs": config.epochs,
            "episodes": len(episodes),
            "transitions": metrics["rollout_steps"],
            "optimizer_steps": metrics["optimizer_steps"],
        },
        "optimizer_state": "fresh-adam-per-update",
        "actor_normalization": "free-choice",
        "seed": seed,
        "metrics": metrics,
        "episodes": [
            {
                "match_id": row["verification"]["match_id"],
                "replay_sha256": row["verification"]["replay_sha256"],
                "player_serial": _episode_actor(row),
                "plan_sha256": row.get("plan_sha256"),
            }
            for row in episodes
        ],
        "discount": _discount(config),
        "strength_improvement_established": False,
        "live_consumed": consumed,
        # PPO imitates nothing; the ancestry's demonstration record carries over.
        "demonstrations_consumed": demonstrations_consumed,
        # A live update of a simulator-trained policy stays marked as such.
        "simulator_ancestry": simulator_ancestry,
        "simulator_consumed": simulator_consumed,
    }
    out.mkdir(parents=True, exist_ok=False)
    manifest = save_checkpoint(out, candidate, training=training)
    policy_sha = manifest["weights"]["sha256"]
    # The new artifact must load by its own recorded identity. Building the
    # model for that check must not advance the caller's global RNG either.
    with torch.random.fork_rng(devices=[]):
        load_checkpoint(out, expected_sha256=policy_sha)
    result = {
        "policy_sha": policy_sha,
        "parent_policy_sha": loaded.policy_sha,
        "checkpoint": str(out),
        "lineage_dir": lineage,
        "episodes": len(episodes),
        "seed": seed,
        "budget": training["budget"],
        "discount": training["discount"],
        "metrics": metrics,
        "automatic_promotion": False,
    }
    atomic_json(out / "training-update.json", result)
    return result


def train_from_collections(
    checkpoint,
    *,
    expected_sha256: str,
    collections,
    out,
    config: TrainConfig,
    seed: int,
    held_out,
    lineage_dir,
    min_episodes: int = 1,
) -> dict:
    """Load the behaviour checkpoint by SHA, select verified training input, update once.

    ``held_out`` is required: name the evaluation collections whose matches must
    not be trained on, or pass ``splits.NO_HELD_OUT`` to state there are none.
    ``lineage_dir`` is required and recorded in the child manifest: only that
    lineage's ``record_decision`` may later decide on the child.
    """
    from .splits import select_training_episodes

    if type(min_episodes) is not int or not 1 <= min_episodes <= MAX_EPISODES:
        raise ValueError(f"minimum batch must be an integer in 1..{MAX_EPISODES}")
    # Building the model to load weights draws from the RNG; keep the caller's untouched.
    with torch.random.fork_rng(devices=[]):
        loaded = load_checkpoint(checkpoint, expected_sha256=expected_sha256)
    _fresh_output(out, loaded.directory)
    selection = select_training_episodes(collections, loaded.policy_sha, held_out=held_out)
    if len(selection.episodes) < min_episodes:
        raise NoEligibleEpisodes(selection.rejected, len(selection.episodes), min_episodes)
    result = update_verified_episodes(
        loaded, selection, out=out, config=config, seed=seed, lineage_dir=lineage_dir
    )
    summary = {
        **result,
        "collections": selection.collections,
        "held_out": [str(Path(h).resolve()) for h in held_out],
        "accepted_episodes": len(selection.episodes),
        "rejected": selection.rejected,
    }
    atomic_json(Path(result["checkpoint"]) / "training-update.json", summary)
    return summary
