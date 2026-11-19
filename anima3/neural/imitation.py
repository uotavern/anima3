"""Explicit, bounded behaviour cloning on verified live teacher demonstrations.

Requires the ``neural`` extra and is never imported by ``anima3`` or
``anima3.neural``. There is no command-line entry, no schedule and no
promotion. A caller invokes :func:`clone_from_demonstrations` once with an
explicit :class:`ImitationConfig` budget, seed, held-out set, validation set,
teacher SHA-256 and lineage directory.

- The starting checkpoint is loaded by the caller's expected SHA-256.
- Input comes only from :func:`~anima3.neural.demonstrations.select_demonstrations`
  over explicitly ``training`` collections; evaluation collections are refused.
  Each demonstration is re-verified through the selection's own verifiers.
- Demonstrations are never PPO trajectories: their placeholders for behaviour
  likelihood, value and reward are zero, and ``ppo_update`` refuses them.
- The result is a new checkpoint in a new directory, identified by the SHA-256
  of its weights and re-loaded by it. Its manifest names the teacher SHA-256,
  every demonstration file with its SHA-256, collection, plan and receipt, every
  rejection, the validation set, the budget and the seed. The student is a
  neural policy trained to imitate a scripted teacher; the teacher itself is
  never labelled neural.
- A fixed seed gives bit-identical weights on the same platform and torch build.

``behavioral_clone`` and ``imitation_metrics`` are ported from the training
lineage ``819f4f4`` (``neural/train.py`` and ``neural/demonstrations.py``).
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .checkpoints import LoadedCheckpoint, load_checkpoint, save_checkpoint
from .demonstrations import (
    DemonstrationRefused,
    DemonstrationSelection,
    TrainingDemonstrationVerifier,
    demonstration_identity,
    select_demonstrations,
)
from .live import atomic_json
from .model import RecurrentPolicy, masked_distribution
from .schema import ACTION_NAMES
from .train import (
    MAX_EPISODES,
    MAX_EPOCHS,
    MAX_TOTAL_TRANSITIONS,
    Trajectory,
    _batch,
    _chunks,
    _consumed,
    _fresh_output,
)

# Pass this as ``validation`` to state explicitly that nothing is scored on the side.
NO_VALIDATION: tuple = ()


@dataclass(frozen=True, kw_only=True)
class ImitationConfig:
    """Behaviour-cloning settings. The budget and action balancing have no defaults."""

    epochs: int
    sequence_length: int
    batch_sequences: int
    balance_actions: bool
    learning_rate: float = 1e-4
    reference_kl_coef: float = 0.0
    max_grad_norm: float = 0.5

    # ``_batch`` reads ``burn_in`` only for PPO targets; imitation replays the
    # full prefix from the initial state under current weights.
    burn_in = 0

    def __post_init__(self) -> None:
        if not all(
            type(v) is int for v in (self.epochs, self.sequence_length, self.batch_sequences)
        ):
            raise ValueError("imitation budget fields must be integers")
        if type(self.balance_actions) is not bool:
            raise ValueError("balance_actions must be stated as True or False")
        if not (
            1 <= self.epochs <= MAX_EPOCHS
            and 1 <= self.sequence_length <= 4096
            and 1 <= self.batch_sequences <= 1024
        ):
            raise ValueError("imitation budget is outside the fixed bounds")
        for value in (self.learning_rate, self.reference_kl_coef, self.max_grad_norm):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("imitation coefficients must be finite numbers")
        if not 0 < self.learning_rate <= 0.1 or self.max_grad_norm <= 0:
            raise ValueError("invalid learning rate or gradient bound")
        if self.reference_kl_coef < 0:
            raise ValueError("reference KL coefficient must be finite and nonnegative")


def demonstration_trajectory(demonstration: dict, hidden_size: int) -> Trajectory:
    """Labels only: zero placeholders that PPO must never read as behaviour likelihoods."""
    trajectory = Trajectory(
        info={
            "domain": "servuo-demonstration",
            "ppo_eligible": False,
            "match_id": demonstration["verification"]["match_id"],
            "actor": demonstration["actor"],
        }
    )
    for step in demonstration["steps"]:
        trajectory.features.append(list(step["features"]))
        trajectory.masks.append(list(step["mask"]))
        trajectory.actions.append(step["action"])
        # BC reconstructs the prefix from initial zero state under current weights.
        trajectory.hidden.append([0.0] * hidden_size)
        trajectory.log_probs.append(0.0)
        trajectory.values.append(0.0)
        trajectory.rewards.append(0.0)
    if not trajectory.actions:
        raise ValueError("empty demonstration")
    return trajectory


def behavioral_clone(
    model: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    episodes: list[Trajectory],
    config: ImitationConfig,
    *,
    seed: int,
    reference_model: RecurrentPolicy | None = None,
) -> dict:
    """Masked cross-entropy on free choices, optionally KL-regularised to a frozen reference."""
    if not isinstance(config, ImitationConfig):
        raise TypeError("an explicit ImitationConfig budget is required")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("an explicit non-negative integer seed is required")
    if any(e.info.get("ppo_eligible") is not False for e in episodes):
        raise ValueError("behaviour cloning takes demonstrations, not PPO trajectories")
    coefficient = config.reference_kl_coef
    if coefficient and reference_model is None:
        raise ValueError("reference model required for KL regularization")
    reference = None
    if coefficient:
        if reference_model.config != model.config:
            raise ValueError("reference model configuration differs")
        # Independent frozen weights and recurrent prefixes; never reuse student
        # hidden states for the reference or mutate the caller's reference model.
        reference = copy.deepcopy(reference_model).to(model.device).eval()
        reference.requires_grad_(False)
    chunks = _chunks(episodes, config.sequence_length)
    rng = random.Random(seed)
    metrics = []
    kl_metrics = []
    action_weights = torch.ones(model.config.action_dim, device=model.device)
    if config.balance_actions:
        counts = torch.zeros_like(action_weights)
        for episode in episodes:
            for action, mask in zip(episode.actions, episode.masks, strict=True):
                if sum(mask) > 1:
                    counts[action] += 1
        present = counts > 0
        if bool(present.any()):
            action_weights[present] = (
                (counts.sum() / (present.sum() * counts[present])).sqrt().clamp(0.25, 4)
            )
    model.train()
    for _ in range(config.epochs):
        rng.shuffle(chunks)
        for offset in range(0, len(chunks), config.batch_sequences):
            selections = chunks[offset : offset + config.batch_sequences]
            batch = _batch(model, episodes, selections, config, None)
            logits, _, _ = model(batch["features"], batch["hidden"])
            distribution = masked_distribution(logits, batch["masks"])
            log_prob = distribution.log_prob(batch["actions"])
            # Forced hold frames have zero policy loss and dilute the useful labels.
            valid = batch["valid"] & (batch["masks"].sum(-1) > 1)
            if not bool(valid.any()):
                continue
            weights = action_weights[batch["actions"][valid]]
            loss = (-log_prob[valid] * weights).sum() / weights.sum()
            if reference is not None:
                with torch.no_grad():
                    ref_batch = _batch(reference, episodes, selections, config, None)
                    ref_logits, _, _ = reference(ref_batch["features"], ref_batch["hidden"])
                    ref_distribution = masked_distribution(ref_logits, ref_batch["masks"])
                kl = torch.distributions.kl_divergence(ref_distribution, distribution)[valid].mean()
                loss = loss + coefficient * kl
                kl_metrics.append(float(kl.detach()))
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite imitation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            metrics.append(float(loss.detach()))
    model.eval()
    return {
        "balance_actions": config.balance_actions,
        "reference_kl_coef": coefficient,
        "reference_kl_mean": sum(kl_metrics) / len(kl_metrics) if kl_metrics else None,
        "action_weights": action_weights.cpu().tolist(),
        "bc_updates": len(metrics),
        "bc_loss_first": metrics[0] if metrics else None,
        "bc_loss_last": metrics[-1] if metrics else None,
        "teacher_steps": sum(map(len, episodes)),
    }


def imitation_metrics(model, episodes: list[Trajectory]) -> dict:
    """Causal full-prefix score on free choices; forced holds do not inflate accuracy."""
    counts, correct = {}, {}
    losses, total_correct = [], 0
    probability_mass = torch.zeros(len(ACTION_NAMES), dtype=torch.float64)
    predictions = torch.zeros(len(ACTION_NAMES), dtype=torch.int64)
    legal_counts = torch.zeros(len(ACTION_NAMES), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for episode in episodes:
            features = torch.tensor([episode.features], device=model.device)
            mask = torch.tensor([episode.masks], dtype=torch.bool, device=model.device)
            actions = torch.tensor([episode.actions], device=model.device)
            logits, _, _ = model(features, model.initial_state())
            distribution = masked_distribution(logits, mask)
            free = mask.sum(-1) > 1
            if not free.any():
                continue
            loss = -distribution.log_prob(actions)[free]
            predicted = distribution.probs.argmax(-1)[free]
            expected = actions[free]
            probability_mass += distribution.probs[free].double().sum(0).cpu()
            predictions += torch.bincount(predicted.cpu(), minlength=len(ACTION_NAMES))
            legal_counts += mask[free].sum(0).cpu()
            losses.extend(loss.cpu().tolist())
            hits = predicted == expected
            total_correct += int(hits.sum())
            for action, hit in zip(expected.cpu().tolist(), hits.cpu().tolist(), strict=True):
                name = ACTION_NAMES[action]
                counts[name] = counts.get(name, 0) + 1
                correct[name] = correct.get(name, 0) + int(hit)
    return {
        "episodes": len(episodes),
        "free_choices": len(losses),
        "cross_entropy": sum(losses) / len(losses) if losses else None,
        "accuracy": total_correct / len(losses) if losses else None,
        "per_action": {
            name: {"labels": n, "accuracy": correct[name] / n} for name, n in counts.items()
        },
        # These describe the policy on teacher states, not live action rates.
        # A missing teacher label is a coverage gap, not proof of illegality.
        "action_distribution": {
            name: {
                "labels": counts.get(name, 0),
                "label_share": counts.get(name, 0) / len(losses) if losses else None,
                "mean_probability": float(probability_mass[i]) / len(losses) if losses else None,
                "argmax_share": int(predictions[i]) / len(losses) if losses else None,
                "legal_share": int(legal_counts[i]) / len(losses) if losses else None,
            }
            for i, name in enumerate(ACTION_NAMES)
        },
        "unobserved_label_probability": (
            sum(
                float(probability_mass[i])
                for i, name in enumerate(ACTION_NAMES)
                if name not in counts
            )
            / len(losses)
            if losses
            else None
        ),
        "real_server_strength_established": False,
    }


def _demonstrations_consumed(loaded: LoadedCheckpoint) -> list[list]:
    consumed = loaded.manifest.get("training", {}).get("demonstrations_consumed", [])
    if not isinstance(consumed, list) or not all(
        isinstance(i, list) and len(i) == 2 and isinstance(i[0], str) and type(i[1]) is int
        for i in consumed
    ):
        raise ValueError("checkpoint lineage has a malformed consumed-demonstration list")
    return [list(i) for i in consumed]


def _absolute(path, what: str) -> Path:
    if not isinstance(path, (str, Path)) or not str(path) or not Path(path).is_absolute():
        raise ValueError(f"{what} must be an absolute path")
    return Path(path)


def clone_from_demonstrations(
    checkpoint,
    *,
    expected_sha256: str,
    collections,
    held_out,
    validation,
    teacher_sha256: str,
    out,
    config: ImitationConfig,
    seed: int,
    lineage_dir,
    min_demonstrations: int,
    styles=None,
    reference_checkpoint=None,
    reference_sha256: str | None = None,
) -> dict:
    """Load the start checkpoint by SHA, select verified demonstrations, clone once.

    ``held_out`` names the evaluation collections whose matches must not be
    imitated (or ``splits.NO_HELD_OUT``). ``validation`` names further training
    collections whose demonstrations are scored before and after but never
    fitted (or :data:`NO_VALIDATION`); they may share neither a collection nor a
    match with the fitted input. ``lineage_dir`` is recorded in the child
    manifest, as for PPO, so only that lineage may later decide on the child.
    A positive ``reference_kl_coef`` needs ``reference_checkpoint`` and its
    ``reference_sha256``; otherwise neither may be given.
    """
    if not isinstance(config, ImitationConfig):
        raise TypeError("an explicit ImitationConfig budget is required")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("an explicit non-negative integer seed is required")
    if type(min_demonstrations) is not int or not 1 <= min_demonstrations <= MAX_EPISODES:
        raise ValueError(f"minimum demonstrations must be an integer in 1..{MAX_EPISODES}")
    if isinstance(validation, (str, Path)) or validation is None:
        raise ValueError("validation must be a collection list or NO_VALIDATION")
    if (reference_checkpoint is None) != (reference_sha256 is None) or (
        reference_checkpoint is not None
    ) != (config.reference_kl_coef > 0):
        raise ValueError(
            "a reference checkpoint, its SHA-256 and a positive KL coefficient go together"
        )
    lineage = str(_absolute(lineage_dir, "lineage directory").resolve())
    # Building a model to load weights draws from the RNG; keep the caller's untouched.
    with torch.random.fork_rng(devices=[]):
        loaded = load_checkpoint(
            _absolute(checkpoint, "checkpoint"), expected_sha256=expected_sha256
        )
    if loaded.model.device.type != "cpu":
        raise ValueError("bounded imitation runs on CPU only")
    out = _fresh_output(_absolute(out, "output checkpoint"), loaded.directory)
    reference = None
    if reference_checkpoint is not None:
        with torch.random.fork_rng(devices=[]):
            reference = load_checkpoint(
                _absolute(reference_checkpoint, "reference checkpoint"),
                expected_sha256=reference_sha256,
            )
    selection = select_demonstrations(
        collections, teacher_sha256=teacher_sha256, held_out=held_out, styles=styles
    )
    held = selection.held_out
    scored = (
        select_demonstrations(
            validation, teacher_sha256=teacher_sha256, held_out=held, styles=styles
        )
        if validation
        else None
    )
    if scored is not None and (
        set(scored.collections) & set(selection.collections)
        or scored.match_ids & selection.match_ids
    ):
        raise DemonstrationRefused("imitation input and validation share a collection or match")
    if len(selection.demonstrations) < min_demonstrations:
        raise DemonstrationRefused(
            f"{len(selection.demonstrations)} verified demonstrations, need {min_demonstrations}"
        )
    if len(selection.demonstrations) > MAX_EPISODES:
        raise DemonstrationRefused(f"at most {MAX_EPISODES} demonstrations per update")
    return _clone_selection(
        loaded,
        selection,
        scored,
        out=out,
        config=config,
        seed=seed,
        lineage=lineage,
        reference=reference,
    )


def _clone_selection(
    loaded: LoadedCheckpoint,
    selection: DemonstrationSelection,
    scored: DemonstrationSelection | None,
    *,
    out: Path,
    config: ImitationConfig,
    seed: int,
    lineage: str,
    reference: LoadedCheckpoint | None,
) -> dict:
    if type(selection) is not DemonstrationSelection:
        raise TypeError("imitation input must be a DemonstrationSelection")
    if not all(type(v) is TrainingDemonstrationVerifier for v in selection.verifiers.values()):
        raise ValueError("imitation input was not verified by its own collections")
    consumed = _demonstrations_consumed(loaded)
    seen = {tuple(identity) for identity in consumed}
    hidden_size = loaded.model.config.hidden_size
    trajectories = []
    for demonstration in selection.demonstrations:
        identity = demonstration_identity(demonstration)
        if identity in seen:
            raise DemonstrationRefused("this lineage already imitated that demonstration")
        seen.add(identity)
        consumed.append(list(identity))
        # Re-verified through the selection's own collection verifiers, never a callable.
        if selection.verify(demonstration) is not True:
            raise DemonstrationRefused("demonstration failed re-verification")
        trajectories.append(demonstration_trajectory(demonstration, hidden_size))
    if sum(map(len, trajectories)) > MAX_TOTAL_TRANSITIONS:
        raise DemonstrationRefused("imitation batch exceeds the fixed step budget")
    validation = [
        demonstration_trajectory(d, hidden_size) for d in (scored.demonstrations if scored else [])
    ]
    candidate = copy.deepcopy(loaded.model)
    before = {
        "train": imitation_metrics(candidate, trajectories),
        "validation": imitation_metrics(candidate, validation),
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=config.learning_rate)
        metrics = behavioral_clone(
            candidate,
            optimizer,
            trajectories,
            config,
            seed=seed,
            reference_model=reference.model if reference else None,
        )
    after = {
        "train": imitation_metrics(candidate, trajectories),
        "validation": imitation_metrics(candidate, validation),
    }
    teachers = sorted({d["teacher"]["style"] for d in selection.demonstrations})
    training = {
        "source": "servuo-teacher-imitation",
        "parent_policy_sha": loaded.policy_sha,
        "lineage_dir": lineage,
        "teacher": {
            "name": "TacticalTeacher",
            "sha256": selection.teacher_sha256,
            "styles": teachers,
            "policy_kind": "teacher",
        },
        "config": asdict(config),
        "budget": {
            "epochs": config.epochs,
            "demonstrations": len(trajectories),
            "steps": metrics["teacher_steps"],
            "bc_updates": metrics["bc_updates"],
        },
        "optimizer_state": "fresh-adam-per-update",
        "seed": seed,
        "metrics": metrics,
        "before": before,
        "after": after,
        "reference_policy_sha": reference.policy_sha if reference else None,
        "collections": selection.collections,
        "held_out": selection.held_out,
        "demonstrations": selection.provenance,
        "rejected": selection.rejected,
        "validation": {
            "collections": scored.collections if scored else [],
            "demonstrations": scored.provenance if scored else [],
            "rejected": scored.rejected if scored else [],
        },
        "ppo_input": False,
        "strength_improvement_established": False,
        "real_server_strength_established": False,
        "demonstrations_consumed": consumed,
        # Imitation consumes no live PPO episode; the lineage's record carries over.
        "live_consumed": _consumed(loaded),
    }
    out.mkdir(parents=True, exist_ok=False)
    manifest = save_checkpoint(out, candidate, training=training)
    policy_sha = manifest["weights"]["sha256"]
    with torch.random.fork_rng(devices=[]):
        load_checkpoint(out, expected_sha256=policy_sha)
    result = {
        "policy_sha": policy_sha,
        "parent_policy_sha": loaded.policy_sha,
        "checkpoint": str(out),
        "lineage_dir": lineage,
        "teacher_sha256": selection.teacher_sha256,
        "demonstrations": len(trajectories),
        "rejected": selection.rejected,
        "seed": seed,
        "budget": training["budget"],
        "metrics": metrics,
        "before": before,
        "after": after,
        "automatic_promotion": False,
    }
    atomic_json(out / "imitation-update.json", result)
    return result
