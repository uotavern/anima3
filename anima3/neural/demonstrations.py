"""Receipt-bound teacher demonstrations from unit-6 collections, for imitation only.

Stdlib only; importing this module never imports torch. A demonstration is the
labelled choice sequence of a :class:`~anima3.neural.teacher.TacticalTeacher`
seat in one receipt-verified live fixture. It is supervised imitation input and
never PPO experience:

- It has its own schema (``anima3-live-demonstration-v2``), ``policy_kind:
  teacher``, a ``teacher:<style>`` label, ``ppo_eligible: false`` and no
  ``policy_sha``, so :class:`~anima3.neural.live.LiveReplayVerifier` and the
  unit-7 PPO path refuse it. It is listed under a completed ledger row's
  ``demonstrations``, never under ``episodes``, and its file name starts with
  ``demonstration-``, which the PPO episode reader refuses.
- Its provenance is checked, not trusted: the teacher identity must equal the
  planned arm for that seat (style and teacher source SHA-256), the plan
  SHA-256 and purpose must match the collection's re-hashed plan, the stored
  server replay must validate and match its SHA-256, seats, rules, unranked
  training flag and starting template, and the issued ledger row must carry the
  pre-challenge supply audit of exactly those two participants.
- Only explicitly ``training`` collections supply demonstrations. Evaluation
  collections are held out and refused, whatever a demonstration says, and a
  demonstration whose match occurs in a named held-out collection refuses the
  whole selection.
- Ineligible, other-teacher and other-style demonstrations are not imitation
  input, but each one is kept in the selection's rejection record.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .. import replay
from ..duel_wait import TRAINING_SUPPLIES
from ..sparring import TRAINING_RULES
from .live import (
    COLLECTION_SCHEMA,
    COLLECTOR_VERSION,
    DEMONSTRATION_SCHEMA,
    EXECUTION_FINGERPRINT,
    MATCH_ID,
    SHA256,
    starting_template_verified,
)
from .schema import ACTION_NAMES, FEATURE_NAMES, schema_fingerprint
from .splits import (
    CollectionRecord,
    SplitRefused,
    _read_json,
    held_out_match_ids,
    read_collection,
)
from .teacher import TEACHER_MODULES, TEACHER_STYLES, TEACHER_VERSION, source_identity

MAX_DEMONSTRATION_BYTES = 64 * 1024 * 1024
MAX_STEPS = 50_000
# One selection accepts at most this many demonstrations (the imitation batch
# cap) and reads at most this many bytes in total, rejected files included.
MAX_SELECTED = 200
MAX_SELECTION_BYTES = 512 * 1024 * 1024
TEACHER_FIELDS = frozenset({"name", "style", "version", "sha256", "sources"})
END_PHASES = ("Idle", "RoundOver")
# The demonstration's own fields; anything else (for example a policy SHA) is refused.
FIELDS = frozenset(
    {
        "schema",
        "domain",
        "policy_kind",
        "label",
        "teacher",
        "ppo_eligible",
        "eligible",
        "invalid_reasons",
        "fixture",
        "plan_sha256",
        "purpose",
        "executor_sha256",
        "collector_version",
        "schema_fingerprint",
        "actor",
        "opponent",
        "opponent_arm",
        "verification",
        "final_result",
        "collection_end_phase",
        "steps",
    }
)
EXPECTED_SUPPLIES = {hex(k): v for k, v in TRAINING_SUPPLIES.items()}


class DemonstrationRefused(SplitRefused):
    """A demonstration or collection may not be used as imitation input."""


def _absolute(path, what: str) -> Path:
    if not isinstance(path, (str, Path)) or not str(path):
        raise DemonstrationRefused(f"{what} must be a path")
    if not Path(path).is_absolute():
        # A relative path would resolve against whatever the current directory is.
        raise DemonstrationRefused(f"{what} must be an absolute path: {path}")
    return Path(path).resolve()


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _supplies_verified(audits, participants: set[int]) -> bool:
    if not isinstance(audits, list) or len(audits) != 2:
        return False
    if {a.get("serial") for a in audits if isinstance(a, dict)} != participants:
        return False
    return all(
        isinstance(a, dict)
        and a.get("verified") is True
        and a.get("counts") == EXPECTED_SUPPLIES
        and _number(a.get("weight"))
        and _number(a.get("weight_max"))
        and a["weight_max"] > 0
        and 0 <= a["weight"] <= a["weight_max"]
        for a in audits
    )


def _steps_problem(steps) -> str | None:
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        return f"demonstration needs 1..{MAX_STEPS} steps"
    previous = None
    for step in steps:
        if not isinstance(step, dict) or set(step) != {
            "at",
            "features",
            "mask",
            "action",
            "accepted",
        }:
            return "malformed demonstration step"
        features, mask, action, at = (step[k] for k in ("features", "mask", "action", "at"))
        if (
            not isinstance(features, list)
            or len(features) != len(FEATURE_NAMES)
            or not all(_number(x) and -1 <= x <= 1 for x in features)
        ):
            return "invalid demonstration features"
        if (
            not isinstance(mask, list)
            or len(mask) != len(ACTION_NAMES)
            or not all(type(x) is bool for x in mask)
        ):
            return "invalid demonstration mask"
        if type(action) is not int or not 0 <= action < len(mask) or not mask[action]:
            return "demonstrated action was not legal in its recorded mask"
        if step["accepted"] is not True:
            return "rejected teacher action in an eligible demonstration"
        if not _number(at) or (previous is not None and at <= previous):
            return "demonstration step clock is not strictly increasing"
        previous = at
    return None


def _seat_arm(record: CollectionRecord, issued: dict, actor: int) -> dict | None:
    """The planned arm of ``actor``: fixture parity fixes which account challenged."""
    arms = record.plan.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        return None
    order = [0, 1] if issued["fixture"] % 2 == 0 else [1, 0]
    if actor == issued.get("challenger"):
        return arms[order[0]]
    if actor == issued.get("opponent"):
        return arms[order[1]]
    return None


class DemonstrationReceiptVerifier:
    """Revalidate a demonstration against its collection; never trust its own flags.

    ``directory`` is the collection directory. The collection manifest and
    ledger are re-read on every call. By default the demonstration must also be
    listed, under its own file name, in the ``completed`` row of its fixture;
    the runner checks staged demonstrations with ``require_ledger=False`` before
    writing that row. Hashes detect corruption; they are not a server signature.
    """

    def __init__(self, directory, *, require_ledger: bool = True):
        self.directory = Path(directory).resolve()
        self.require_ledger = require_ledger

    def check(self, demonstration) -> None:
        """Raise :class:`DemonstrationRefused` naming the first problem found."""
        d = demonstration
        if not isinstance(d, dict) or set(d) != FIELDS:
            raise DemonstrationRefused("demonstration has unknown or missing fields")
        if not (
            d.get("schema") == DEMONSTRATION_SCHEMA
            and d.get("domain") == "servuo"
            and d.get("policy_kind") == "teacher"
            and d.get("ppo_eligible") is False
        ):
            raise DemonstrationRefused("not a supervised-only teacher demonstration")
        if d.get("eligible") is not True or d.get("invalid_reasons") != []:
            raise DemonstrationRefused("demonstration is not eligible")
        teacher = d.get("teacher")
        if not (
            isinstance(teacher, dict)
            and set(teacher) == TEACHER_FIELDS
            and teacher.get("name") == "TacticalTeacher"
            and teacher.get("style") in TEACHER_STYLES
            and teacher.get("version") == TEACHER_VERSION
            and isinstance(teacher.get("sha256"), str)
            and SHA256.fullmatch(teacher["sha256"])
            and d.get("label") == f"teacher:{teacher['style']}"
        ):
            raise DemonstrationRefused("demonstration lacks a well-formed teacher identity")
        sources = teacher["sources"]
        if not (
            isinstance(sources, dict)
            and set(sources) == set(TEACHER_MODULES)
            and all(isinstance(v, str) and SHA256.fullmatch(v) for v in sources.values())
            and hmac.compare_digest(source_identity(sources), teacher["sha256"])
        ):
            raise DemonstrationRefused("teacher module hashes do not give its identity")
        if (
            d.get("collector_version") != COLLECTOR_VERSION
            or d.get("collection_end_phase") not in END_PHASES
        ):
            raise DemonstrationRefused("demonstration collector version or end phase differs")
        if (
            d.get("schema_fingerprint") != schema_fingerprint()
            or d.get("executor_sha256") != EXECUTION_FINGERPRINT
        ):
            raise DemonstrationRefused("demonstration schema or executor fingerprint differs")
        problem = _steps_problem(d.get("steps"))
        if problem:
            raise DemonstrationRefused(problem)
        record = read_collection(self.directory)
        if record.plan_sha256 != d.get("plan_sha256") or record.purpose != d.get("purpose"):
            raise DemonstrationRefused("demonstration plan or purpose differs from its collection")
        if record.plan.get("require_base_template") is not True:
            raise DemonstrationRefused("demonstrations need a base-template collection")
        fixture, actor, opponent = d.get("fixture"), d.get("actor"), d.get("opponent")
        verification = d.get("verification")
        if not isinstance(verification, dict) or set(verification) != {"match_id", "replay_sha256"}:
            raise DemonstrationRefused("demonstration verification has unknown or missing fields")
        if not isinstance(d.get("final_result"), dict) or set(d["final_result"]) != {
            "winner",
            "outcome",
        }:
            raise DemonstrationRefused("demonstration result has unknown or missing fields")
        match_id = verification.get("match_id")
        if not isinstance(match_id, str) or not MATCH_ID.fullmatch(match_id):
            raise DemonstrationRefused("demonstration lacks a match identity")
        if type(fixture) is not int:
            raise DemonstrationRefused("demonstration fixture is not an integer")
        if type(actor) is not int or type(opponent) is not int or actor <= 0 or actor == opponent:
            raise DemonstrationRefused("demonstration seats are malformed")
        issued = next(
            (
                r
                for r in record.ledger.entries
                if r["event"] == "issued" and r["fixture"] == fixture
            ),
            None,
        )
        if issued is None:
            raise DemonstrationRefused("no issued ledger row for the demonstration fixture")
        arm = _seat_arm(record, issued, actor)
        if not (
            isinstance(arm, dict)
            and arm.get("kind") == "teacher"
            and arm.get("variant") == teacher["style"]
            and isinstance(arm.get("source_sha256"), str)
            and hmac.compare_digest(arm["source_sha256"], teacher["sha256"])
        ):
            raise DemonstrationRefused("demonstration teacher differs from the planned seat arm")
        if d.get("opponent_arm") != _seat_arm(record, issued, opponent):
            raise DemonstrationRefused("demonstration opponent arm differs from the plan")
        if {issued.get("challenger"), issued.get("opponent")} != {actor, opponent}:
            raise DemonstrationRefused("demonstration seats differ from the issued fixture")
        if self.require_ledger:
            expected = str(self.directory / match_id / f"demonstration-{actor}.json")
            if not any(
                r["event"] == "completed"
                and r["fixture"] == fixture
                and r.get("match_id") == match_id
                and expected in r.get("demonstrations", [])
                for r in record.ledger.entries
            ):
                raise DemonstrationRefused("no completed ledger row lists this demonstration")
        root = self.directory / match_id
        meta = _read_json(root / "replay.meta.json", 1024 * 1024, "replay manifest")
        try:
            with (root / "replay.jsonl").open("rb") as stream:
                data = stream.read(replay.MAX_BYTES + 1)
            if len(data) > replay.MAX_BYTES:
                raise DemonstrationRefused("replay exceeds size limit")
            rows = replay.validate(data, meta)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DemonstrationRefused(f"demonstration receipt invalid: {exc}") from exc
        participants = {p.get("serial") for p in rows[0].get("players", [])}
        winner = meta.get("winner")
        outcome = 0.0 if winner is None else (1.0 if winner == actor else -1.0)
        if not (
            meta.get("id") == match_id
            and meta.get("training") is True
            and meta.get("ranked", False) is False
            and meta.get("aborted") is None
            and meta.get("rules") == TRAINING_RULES
            and rows[0].get("rules") == TRAINING_RULES
            and verification.get("replay_sha256") == meta.get("sha256")
            and len(rows[0].get("players", [])) == 2
            and participants == {actor, opponent}
            and starting_template_verified(rows)
            and d.get("final_result") == {"winner": winner, "outcome": outcome}
        ):
            raise DemonstrationRefused("demonstration receipt differs")
        if not _supplies_verified(issued.get("supplies"), participants):
            raise DemonstrationRefused("invalid pre-challenge supply verification")

    def __call__(self, demonstration) -> bool:
        try:
            self.check(demonstration)
        except (SplitRefused, OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
        return True


class TrainingDemonstrationVerifier:
    """Accept a demonstration only as imitation input from this training collection."""

    def __init__(self, directory):
        self.record = read_collection(directory)
        if self.record.purpose != "training":
            raise DemonstrationRefused(
                "evaluation collection is held out; its demonstrations are never imitation input"
            )
        self._receipts = DemonstrationReceiptVerifier(self.record.root)

    def __call__(self, demonstration) -> bool:
        return bool(
            isinstance(demonstration, dict)
            and demonstration.get("purpose") == "training"
            and demonstration.get("plan_sha256") == self.record.plan_sha256
            and self._receipts(demonstration)
        )


def demonstration_identity(demonstration: dict) -> tuple[str, int]:
    """The (match, teacher seat) identity a demonstration may be fitted on at most once."""
    verification = demonstration.get("verification")
    match_id = verification.get("match_id") if isinstance(verification, dict) else None
    actor = demonstration.get("actor")
    if not isinstance(match_id, str) or type(actor) is not int or actor <= 0:
        raise DemonstrationRefused("demonstration lacks a match identity or positive actor")
    return match_id, actor


@dataclass
class DemonstrationSelection:
    """Accepted demonstrations, their verifiers and provenance, and every rejection."""

    teacher_sha256: str
    demonstrations: list[dict] = field(default_factory=list)
    provenance: list[dict] = field(default_factory=list)
    verifiers: dict[tuple[str, int], TrainingDemonstrationVerifier] = field(default_factory=dict)
    rejected: list[dict] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    held_out: list[str] = field(default_factory=list)
    files: dict[tuple[str, int], tuple[str, str]] = field(default_factory=dict)

    @property
    def match_ids(self) -> set[str]:
        return {identity[0] for identity in self.verifiers}

    def verify(self, demonstration: dict) -> bool:
        """Re-verify through this selection's own verifier and its recorded file SHA-256.

        The demonstration must still equal the bytes read at selection time, so
        neither an in-memory edit nor a rewritten file can be fitted.
        """
        identity = demonstration_identity(demonstration)
        verifier = self.verifiers.get(identity)
        recorded = self.files.get(identity)
        if (
            type(verifier) is not TrainingDemonstrationVerifier
            or recorded is None
            or demonstration.get("teacher", {}).get("sha256") != self.teacher_sha256
        ):
            return False
        path, digest = recorded
        try:
            with open(path, "rb") as stream:
                payload = stream.read(MAX_DEMONSTRATION_BYTES + 1)
            current = _decode(payload, Path(path))
        except (OSError, DemonstrationRefused):
            return False
        return bool(
            hmac.compare_digest(hashlib.sha256(payload).hexdigest(), digest)
            and current == demonstration
            and verifier(demonstration) is True
        )


def _read_demonstration(record: CollectionRecord, row: dict, name: str) -> tuple[Path, bytes]:
    match_dir = (record.root / row["match_id"]).resolve()
    path = Path(name).resolve()
    if path.parent != match_dir or not (
        path.name.startswith("demonstration-") and path.suffix == ".json"
    ):
        raise DemonstrationRefused("ledger demonstration path escapes its match directory")
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_DEMONSTRATION_BYTES + 1)
    except OSError as exc:
        raise DemonstrationRefused(f"cannot read demonstration: {type(exc).__name__}") from exc
    if len(payload) > MAX_DEMONSTRATION_BYTES:
        raise DemonstrationRefused("demonstration exceeds size limit")
    return path, payload


def select_demonstrations(
    collections, *, teacher_sha256: str, held_out, styles=None
) -> DemonstrationSelection:
    """Collect imitation input recorded by teacher ``teacher_sha256`` from training collections.

    ``teacher_sha256`` is required: the teacher source SHA-256 the caller means
    to imitate (``teacher.TEACHER_SHA256`` for the current code). ``held_out``
    is required, as in :func:`~anima3.neural.splits.select_training_episodes`:
    the evaluation collections whose matches must not be imitated, or
    ``splits.NO_HELD_OUT``. ``styles`` optionally restricts the teacher styles.
    Every path must be absolute.
    """
    if not isinstance(teacher_sha256, str) or not SHA256.fullmatch(teacher_sha256):
        raise DemonstrationRefused("an explicit 64-hex teacher SHA-256 is required")
    if isinstance(held_out, (str, Path)) or held_out is None:
        raise DemonstrationRefused("held_out must be a collection list or NO_HELD_OUT")
    if styles is not None and (
        isinstance(styles, str) or not set(styles) <= set(TEACHER_STYLES) or not styles
    ):
        raise DemonstrationRefused("styles must be a non-empty list of teacher styles")
    roots = [_absolute(c, "training collection") for c in collections]
    held_roots = {_absolute(c, "held-out collection") for c in held_out}
    if not roots:
        raise DemonstrationRefused("name at least one training collection explicitly")
    if len(set(roots)) != len(roots):
        raise DemonstrationRefused("a training collection is listed twice")
    if held_roots & set(roots):
        raise DemonstrationRefused("a collection cannot be both imitation input and held out")
    held_ids = held_out_match_ids(sorted(held_roots))
    selection = DemonstrationSelection(teacher_sha256, held_out=sorted(map(str, held_roots)))
    total_bytes = 0
    for root in roots:
        verifier = TrainingDemonstrationVerifier(root)
        selection.collections.append(str(root))
        for row in verifier.record.ledger.completed:
            if row["match_id"] in held_ids:
                raise DemonstrationRefused(
                    "a demonstration match also occurs in a held-out evaluation"
                )
            for name, recorded in zip(
                row.get("demonstrations", []), row.get("demonstration_sha256", []), strict=True
            ):
                path, payload = _read_demonstration(verifier.record, row, name)
                total_bytes += len(payload)
                if total_bytes > MAX_SELECTION_BYTES:
                    raise DemonstrationRefused("selection exceeds its total read budget")
                if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), recorded):
                    raise DemonstrationRefused(
                        f"demonstration differs from the SHA-256 in its ledger row: {path}"
                    )
                demonstration = _decode(payload, path)
                identity = demonstration_identity(demonstration)
                if (
                    identity[0] != row["match_id"]
                    or path.name != f"demonstration-{identity[1]}.json"
                ):
                    raise DemonstrationRefused("demonstration identity differs from its ledger row")
                teacher = demonstration.get("teacher")
                teacher = teacher if isinstance(teacher, dict) else {}
                reasons = []
                if teacher.get("sha256") != teacher_sha256:
                    reasons.append("other_teacher")
                if styles is not None and teacher.get("style") not in styles:
                    reasons.append("other_style")
                if demonstration.get("eligible") is not True:
                    found = demonstration.get("invalid_reasons")
                    reasons += ["ineligible", *(found if isinstance(found, list) else [])]
                if reasons:
                    selection.rejected.append({"file": str(path), "reasons": reasons})
                    continue
                if not verifier(demonstration):
                    raise DemonstrationRefused(
                        f"eligible demonstration failed verification: {path}"
                    )
                if identity in selection.verifiers:
                    raise DemonstrationRefused(
                        "duplicate demonstration identity across collections"
                    )
                if len(selection.demonstrations) >= MAX_SELECTED:
                    raise DemonstrationRefused(
                        f"at most {MAX_SELECTED} demonstrations per selection"
                    )
                selection.verifiers[identity] = verifier
                digest = hashlib.sha256(payload).hexdigest()
                selection.files[identity] = (str(path), digest)
                selection.demonstrations.append(demonstration)
                selection.provenance.append(
                    {
                        "file": str(path),
                        "file_sha256": digest,
                        "collection": str(root),
                        "collection_schema": COLLECTION_SCHEMA,
                        "plan_sha256": verifier.record.plan_sha256,
                        "fixture": demonstration["fixture"],
                        "match_id": identity[0],
                        "actor": identity[1],
                        "replay_sha256": demonstration["verification"]["replay_sha256"],
                        "teacher": demonstration["teacher"],
                        "opponent_arm": demonstration.get("opponent_arm"),
                        "steps": len(demonstration["steps"]),
                    }
                )
    return selection


def _decode(payload: bytes, path: Path) -> dict:
    if len(payload) > MAX_DEMONSTRATION_BYTES:
        raise DemonstrationRefused("demonstration exceeds size limit")
    try:
        demonstration = json.loads(payload, parse_constant=_nonfinite)
    except (UnicodeDecodeError, ValueError) as exc:
        raise DemonstrationRefused(f"demonstration is not valid JSON: {path}") from exc
    if not isinstance(demonstration, dict):
        raise DemonstrationRefused("demonstration is not a JSON object")
    return demonstration


def _nonfinite(name: str):
    raise DemonstrationRefused(f"demonstration contains non-finite number {name}")
