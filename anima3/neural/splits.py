"""Training and held-out evaluation splits over unit-6 live collections.

Stdlib only; importing this module never imports torch. Every split decision
reads the collection's own recorded plan (``collection.json``) and append-only
ledger, never an episode's flags alone:

- A collection's ``purpose`` is fixed in its plan and bound into the plan
  SHA-256. Only an explicitly ``training`` collection can supply training input;
  an ``evaluation`` collection is held out and refused, whatever its episodes say.
- Training input is only an episode listed in a ``completed`` ledger row that
  :class:`~anima3.neural.live.LiveReplayVerifier` accepts for the behaviour
  policy SHA-256, and whose recorded purpose and plan SHA-256 match the
  collection. An eligible-looking episode that fails verification refuses the
  whole batch rather than being skipped.
- Ineligible and other-policy episodes are not training input, but each one is
  kept in the returned rejection record with its reason.
- Episodes never listed in a ``completed`` row (for example staged files from an
  interrupted fixture) are never read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .inference import require_sha256
from .live import COLLECTION_SCHEMA, PURPOSES, Ledger, LiveReplayVerifier

MAX_EPISODE_BYTES = 64 * 1024 * 1024
MAX_COLLECTION_MANIFEST_BYTES = 1024 * 1024


class SplitRefused(ValueError):
    """A collection or episode may not be used for the requested split."""


@dataclass(frozen=True)
class CollectionRecord:
    """A collection's recorded plan, purpose and ledger, read once."""

    root: Path
    plan: dict
    plan_sha256: str
    purpose: str
    ledger: Ledger


def _read_json(path: Path, limit: int, what: str):
    try:
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as exc:
        raise SplitRefused(f"cannot read {what}: {type(exc).__name__}") from exc
    if len(payload) > limit:
        raise SplitRefused(f"{what} exceeds size limit")
    try:
        return json.loads(payload, parse_constant=lambda name: _nonfinite(name, what))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitRefused(f"{what} is not valid JSON") from exc


def _nonfinite(name: str, what: str):
    raise SplitRefused(f"{what} contains non-finite number {name}")


def plan_sha256(plan: dict) -> str:
    """The canonical SHA-256 that ``CollectionPlan.sha256`` records for ``plan``."""
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def read_collection(directory) -> CollectionRecord:
    """Read and cross-check a collection manifest and its ledger."""
    from .live import CollectionRefused

    root = Path(directory).resolve()
    manifest = _read_json(root / "collection.json", MAX_COLLECTION_MANIFEST_BYTES, "collection")
    if not isinstance(manifest, dict) or manifest.get("schema") != COLLECTION_SCHEMA:
        raise SplitRefused("not a live collection manifest")
    plan = manifest.get("plan")
    if not isinstance(plan, dict) or manifest.get("plan_sha256") != plan_sha256(plan):
        raise SplitRefused("collection plan does not match its recorded SHA-256")
    purpose = plan.get("purpose")
    if purpose not in PURPOSES:
        raise SplitRefused("collection plan declares no evaluation/training purpose")
    try:
        ledger = Ledger.load(root / "ledger.jsonl")
    except CollectionRefused as exc:
        raise SplitRefused(f"collection ledger refused: {exc}") from exc
    return CollectionRecord(root, plan, manifest["plan_sha256"], purpose, ledger)


def read_episode(record: CollectionRecord, row: dict, name: str) -> tuple[Path, dict]:
    """Load one episode named by a ``completed`` ledger row, inside its match directory."""
    match_dir = (record.root / row["match_id"]).resolve()
    path = Path(name).resolve()
    if path.parent != match_dir or not (
        path.name.startswith("episode-") and path.suffix == ".json"
    ):
        raise SplitRefused("ledger episode path escapes its match directory")
    episode = _read_json(path, MAX_EPISODE_BYTES, "episode")
    if not isinstance(episode, dict):
        raise SplitRefused("episode is not a JSON object")
    return path, episode


def episode_identity(episode: dict) -> tuple[str, int]:
    """The (match, actor) identity an episode may be trained on at most once."""
    verification = episode.get("verification")
    match_id = verification.get("match_id") if isinstance(verification, dict) else None
    actor = episode.get("player_serial", episode.get("actor"))
    if not isinstance(match_id, str) or type(actor) is not int or actor <= 0:
        raise SplitRefused("episode lacks a match identity or positive actor serial")
    if "actor" in episode and episode["actor"] != actor:
        raise SplitRefused("conflicting actor serials in episode")
    return match_id, actor


class TrainingEpisodeVerifier:
    """Accept an episode only as training input from this training collection."""

    def __init__(self, directory, policy_sha: str):
        self.record = read_collection(directory)
        if self.record.purpose != "training":
            raise SplitRefused("evaluation collection is held out; it is never training input")
        self.policy_sha = require_sha256(policy_sha)
        self._live = LiveReplayVerifier(self.record.root, self.policy_sha)

    def __call__(self, episode) -> bool:
        return bool(
            isinstance(episode, dict)
            and episode.get("purpose") == "training"
            and episode.get("plan_sha256") == self.record.plan_sha256
            and self._live(episode)
        )


@dataclass
class TrainingSelection:
    """Accepted training episodes, their per-collection verifiers and every rejection."""

    episodes: list[dict] = field(default_factory=list)
    verifiers: dict[tuple[str, int], TrainingEpisodeVerifier] = field(default_factory=dict)
    rejected: list[dict] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)

    def verify(self, episode: dict) -> bool:
        verifier = self.verifiers.get(episode_identity(episode))
        return verifier is not None and verifier(episode)


def held_out_match_ids(directories) -> set[str]:
    """Every match identity recorded in the given held-out evaluation collections."""
    ids: set[str] = set()
    for directory in directories:
        record = read_collection(directory)
        if record.purpose != "evaluation":
            raise SplitRefused("a held-out collection must be an evaluation collection")
        ids |= record.ledger.match_ids
    return ids


def select_training_episodes(collections, policy_sha: str, *, held_out=()) -> TrainingSelection:
    """Collect training input for ``policy_sha`` from explicit training collections.

    ``held_out`` names evaluation collections whose matches must not occur in the
    training input; a collection may not appear on both sides.
    """
    policy_sha = require_sha256(policy_sha)
    roots = [Path(c).resolve() for c in collections]
    held_roots = {Path(c).resolve() for c in held_out}
    if not roots:
        raise SplitRefused("name at least one training collection explicitly")
    if len(set(roots)) != len(roots):
        raise SplitRefused("a training collection is listed twice")
    if held_roots & set(roots):
        raise SplitRefused("a collection cannot be both training input and held out")
    held_ids = held_out_match_ids(sorted(held_roots))
    selection = TrainingSelection()
    for root in roots:
        verifier = TrainingEpisodeVerifier(root, policy_sha)
        selection.collections.append(str(root))
        for row in verifier.record.ledger.completed:
            if row["match_id"] in held_ids:
                raise SplitRefused("a training match also occurs in a held-out evaluation")
            for name in row["episodes"]:
                path, episode = read_episode(verifier.record, row, name)
                identity = episode_identity(episode)
                if identity[0] != row["match_id"]:
                    raise SplitRefused("episode match differs from its ledger row")
                if episode.get("policy_sha") != policy_sha:
                    selection.rejected.append({"file": str(path), "reasons": ["other_policy"]})
                    continue
                if episode.get("eligible") is not True:
                    reasons = episode.get("invalid_reasons")
                    selection.rejected.append(
                        {
                            "file": str(path),
                            "reasons": [
                                "ineligible",
                                *(reasons if isinstance(reasons, list) else []),
                            ],
                        }
                    )
                    continue
                if not verifier(episode):
                    raise SplitRefused(f"eligible episode failed verification: {path}")
                if identity in selection.verifiers:
                    raise SplitRefused("duplicate episode identity across training collections")
                selection.verifiers[identity] = verifier
                selection.episodes.append(episode)
    return selection
