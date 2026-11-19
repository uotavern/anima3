"""Frozen-policy live duel runner: SHA-bound inference wired to the option executor.

This module joins the unit-5 ``InferenceClient`` and the unit-4 ``CombatExecutor``
on one actor thread per game socket, and records only receipt-verified unrated
training fixtures. It has no command-line entry point and never trains, updates
or promotes a policy. The caller supplies body connections and the receipt
source, and must hold separate authorization before pointing it at a server.

Invariants:

- Every neural arm names an immutable checkpoint by the SHA-256 of its weights.
  A missing ``torch`` extra or checkpoint refuses the run before any login; it is
  never silently replaced by the scripted policy under the neural label.
- A collection directory records its plan once. Resuming requires the identical
  plan, and every issued challenge consumes the fixed budget whether or not it
  completed, so a crash or failure can never reissue more fixtures than planned.
- A match identity may appear in a collection at most once.
- A plan declares its purpose (held-out ``evaluation`` by default, or ``training``)
  once; it is bound into the plan SHA-256 and every episode.
- Rejected policy decisions make an episode ineligible. Deterministic option
  failure and recovery remain experience (no success-conditioned sampling).
- A ``teacher`` arm runs the scripted :class:`~anima3.neural.teacher.TacticalTeacher`
  and a ``probe`` arm a scripted melee/cast-dump ablation. Both are named by the
  SHA-256 of their source module, never by a neural label, and neither produces
  a PPO episode. A teacher's choices are written as receipt-bound demonstrations
  (supervised imitation input only) next to, never inside, the episode list.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .. import replay
from ..contract import say
from ..duel_wait import prepare, server_state, verify_training_supplies
from ..sparring import (
    TRAINING_RULES,
    accept_invitation,
    account_locks,
    request_training_challenge,
    verify_receipt,
    wait_training_idle,
)
from .executor import CombatExecutor
from .inference import InferenceClient, require_sha256
from .melee import CastDumpProbe, MeleeProbe
from .schema import ACTION_NAMES, schema_fingerprint
from .teacher import (
    DEPENDENCIES,
    TEACHER_SHA256,
    TEACHER_SOURCES,
    TEACHER_STYLES,
    TEACHER_VERSION,
    TacticalTeacher,
    source_hashes,
    source_identity,
)

EPISODE_SCHEMA = "anima3-live-policy-v1"
# The source branch wrote "anima3-live-demonstration-v1" without plan, purpose,
# teacher identity or ledger binding; those files are never accepted here.
DEMONSTRATION_SCHEMA = "anima3-live-demonstration-v2"
COLLECTION_SCHEMA = "anima3-live-collection-v1"
COLLECTOR_VERSION = "frozen-options-v1"
EXECUTION_FINGERPRINT = hashlib.sha256(
    Path(__file__).with_name("executor.py").read_bytes()
).hexdigest()
# Like the teacher, a probe is identified by its module and every module it uses.
PROBE_MODULES = ("neural/melee.py", *DEPENDENCIES)
PROBE_SOURCES = source_hashes(PROBE_MODULES)
PROBE_SHA256 = source_identity(PROBE_SOURCES)
PROBE_VARIANTS = (
    "magic",
    "punch",
    "stun",
    "magic-resources",
    "punch-resources",
    "stun-resources",
    "dump-baseline",
    "dump-fast",
    "dump-held",
    "dump-primed",
)
ARM_KINDS = ("neural", "scripted", "teacher", "probe")
MATCH_ID = re.compile(r"[0-9a-f]{32}")
SHA256 = re.compile(r"[0-9a-f]{64}")
STAGING = ".staged"
MAX_MATCHES = 100
# A collection is either a training source or a held-out evaluation; never both.
# Unless a plan explicitly says "training", its episodes are not training input.
PURPOSES = ("evaluation", "training")
DECISION_INTERVAL = 0.25
STATE_FRESH_SECONDS = 3.0
STATE_LOST_SECONDS = 20.0


class ArmUnavailable(RuntimeError):
    """An arm cannot run as declared; no fixture was issued for it."""


class NeuralArmUnavailable(ArmUnavailable):
    """A neural arm cannot run as declared; no fixture was issued for it."""


class CollectionRefused(RuntimeError):
    """The collection directory, plan or budget forbids issuing another fixture."""


class DuplicateMatch(CollectionRefused):
    """The server assigned a match identity this collection already recorded."""

    def __init__(self, match_id: str):
        super().__init__("server reported a match identity already in this collection")
        self.match_id = match_id


# --- immutable arm identities ---------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One seat's policy. A neural arm is bound to the SHA-256 of its weight file.

    ``teacher`` and ``probe`` arms are scripted controllers. They carry no
    checkpoint or policy SHA-256; instead ``variant`` names the teacher style or
    probe tactic and ``source_sha256`` the SHA-256 of the controller's source
    module (``TEACHER_SHA256`` or ``PROBE_SHA256``), checked before any login.
    """

    kind: str
    checkpoint: str | None = None
    policy_sha256: str | None = None
    variant: str | None = None
    source_sha256: str | None = None

    def __post_init__(self):
        if self.kind not in ARM_KINDS:
            raise ValueError("arm kind must be 'neural', 'scripted', 'teacher' or 'probe'")
        if self.kind == "neural":
            if not isinstance(self.checkpoint, str) or not self.checkpoint:
                raise ValueError("a neural arm needs an explicit checkpoint path")
            require_sha256(self.policy_sha256)
        elif self.checkpoint is not None or self.policy_sha256 is not None:
            raise ValueError("a scripted, teacher or probe arm carries no checkpoint identity")
        if self.kind in ("neural", "scripted"):
            if self.variant is not None or self.source_sha256 is not None:
                raise ValueError("only teacher and probe arms carry a variant and source SHA-256")
            return
        variants = TEACHER_STYLES if self.kind == "teacher" else PROBE_VARIANTS
        if self.variant not in variants:
            raise ValueError(f"{self.kind} variant must be one of {', '.join(variants)}")
        require_sha256(self.source_sha256)

    @property
    def neural(self) -> bool:
        return self.kind == "neural"

    @property
    def label(self) -> str:
        """Self-reported model and version labels; never 'neural' for a scripted arm."""
        if self.neural:
            return f"neural {self.policy_sha256[:12]}"
        if self.kind == "scripted":
            return "scripted fixed-scripted"
        return f"{self.kind} {self.variant}-{self.source_sha256[:12]}"

    @property
    def agent_policy(self) -> str:
        return "recurrent" if self.kind in ("neural", "scripted") else "scripted"

    def to_json(self) -> dict:
        # Unset teacher/probe fields are omitted so neural/scripted plans keep
        # the plan SHA-256 they had before these arm kinds existed.
        value = asdict(self)
        for key in ("variant", "source_sha256"):
            if value[key] is None:
                del value[key]
        return value


def preflight_arm(arm: Arm, *, find_spec=importlib.util.find_spec) -> None:
    """Refuse an unavailable neural arm before any account is touched.

    The worker subprocess would also report these failures, but only after a
    login. This check uses ``find_spec`` so the parent still never imports torch.
    """
    if arm.kind in ("teacher", "probe"):
        current = TEACHER_SHA256 if arm.kind == "teacher" else PROBE_SHA256
        if not hmac.compare_digest(arm.source_sha256, current):
            raise ArmUnavailable(
                f"{arm.kind} source differs from the planned identity {arm.source_sha256}"
            )
        return
    if not arm.neural:
        return
    path = Path(arm.checkpoint)
    manifest = path / "manifest.json" if path.is_dir() else path
    if not manifest.is_file():
        raise NeuralArmUnavailable(f"checkpoint manifest not found for policy {arm.policy_sha256}")
    if find_spec("torch") is None:
        raise NeuralArmUnavailable(
            "optional 'neural' extra (torch) is not installed; refusing to substitute a policy"
        )


def await_workers(workers, arms, *, timeout=90.0, stopped=lambda: False, sleep=time.sleep):
    """Wait for every neural worker to report the exact declared policy, or refuse."""
    deadline = time.monotonic() + timeout
    while True:
        for worker, arm in zip(workers, arms, strict=True):
            if worker is not None and worker.startup_error:
                raise NeuralArmUnavailable(f"inference worker failed: {worker.startup_error}")
        if all(w is None or w.ready for w in workers):
            break
        if stopped():
            raise NeuralArmUnavailable("stopped while waiting for inference workers")
        if time.monotonic() >= deadline:
            raise NeuralArmUnavailable("inference worker did not become ready in time")
        sleep(0.05)
    for worker, arm in zip(workers, arms, strict=True):
        if worker is not None and not (
            isinstance(worker.policy_sha, str)
            and hmac.compare_digest(worker.policy_sha, arm.policy_sha256)
        ):
            raise NeuralArmUnavailable("inference worker reports another policy identity")


# --- decisions reaching the executor ---------------------------------------------------------


def policy_override(event):
    """Internal option failure/recovery is experience, not a substituted action."""
    return event.get("policy_override") is True or event.get("scope") == "policy_override"


def dispatch_option(executor, action, frame, obs, now, fresh):
    """A sampled option may abort without substituting a different policy action.

    Mask changes during transport are an environmental execution outcome. Retain
    the original policy sample, including failures, instead of selecting only
    episodes whose actions happened to remain possible.
    """
    if action == 0:
        return {"execution_started": False, "cancellation_reason": None}
    if not fresh:
        return {"execution_started": False, "cancellation_reason": "context_stale"}
    if not frame.mask[action]:
        return {"execution_started": False, "cancellation_reason": "mask_changed"}
    started = executor.start(action, obs, now)
    return {
        "execution_started": started,
        "cancellation_reason": None if started else "executor_refused",
    }


# --- small file helpers ------------------------------------------------------------------------


def _fsync_dir(directory: Path) -> None:
    """Make a rename or file creation in ``directory`` durable."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps({"utc": time.time(), **event}, ensure_ascii=False) + "\n")


class ObservedBody:
    """Give the scripted Agent the already inspected observation, journal included."""

    def __init__(self, body):
        self.body = body
        self.pending = None

    def observe(self):
        if self.pending is not None:
            obs, self.pending = self.pending, None
            return obs
        return self.body.observe()

    def act(self, action):
        self.body.act(action)

    def pump(self, ms):
        return self.body.pump(ms)


# --- receipts ----------------------------------------------------------------------------------


def starting_template_verified(rows):
    if any(p.get("stats") != [100, 25, 100] for p in rows[0].get("players", [])):
        return False
    frame = next((r for r in rows if r.get("type") == "frame"), None)
    return bool(
        frame
        and frame.get("phase") == "Countdown"
        and len(frame.get("players", [])) == 2
        and all(
            (p.get("hitsMax"), p.get("manaMax"), p.get("stamMax")) == (100, 100, 25)
            and p.get("alive")
            and not p.get("poisoned")
            for p in frame["players"]
        )
    )


class LiveReplayVerifier:
    """Revalidate a stored receipt for an episode; never trust the episode's own flags.

    Hashes detect corruption; they are not a server signature. Keep the directory
    under the same trust boundary as the configured shard. Every episode must
    carry the current schema and executor fingerprints and a well-formed policy
    SHA-256; when ``expected_policy_sha`` is given it must be exactly that one.
    ``directory`` is the collection directory. By default the episode's match
    must also be closed ``completed`` for the same fixture in its ledger, so an
    episode left by an interrupted or refused fixture is never accepted. The
    runner itself checks staged episodes before writing that row.
    """

    def __init__(
        self, directory, expected_policy_sha: str | None = None, *, require_ledger: bool = True
    ):
        self.directory = Path(directory).resolve()
        self.expected_policy_sha = (
            None if expected_policy_sha is None else require_sha256(expected_policy_sha)
        )
        self.require_ledger = require_ledger

    def _ledger_completed(self, match_id, fixture) -> bool:
        try:
            ledger = Ledger.load(self.directory / "ledger.jsonl")
        except (CollectionRefused, OSError):
            return False
        return any(
            row["event"] == "completed"
            and row.get("match_id") == match_id
            and row["fixture"] == fixture
            for row in ledger.entries
        )

    def __call__(self, episode):
        try:
            match_id = episode["verification"]["match_id"]
            if not isinstance(match_id, str) or not MATCH_ID.fullmatch(match_id):
                return False
            root = self.directory / match_id
            meta = json.loads((root / "replay.meta.json").read_text())
            rows = replay.validate((root / "replay.jsonl").read_bytes(), meta)
            participants = [p["serial"] for p in rows[0]["players"]]
            result = episode["final_result"]
            actor = episode["actor"]
            outcome = 0.0 if meta["winner"] is None else (1.0 if meta["winner"] == actor else -1.0)
            transitions = episode["transitions"]
            policy = episode.get("policy_sha")
            if not isinstance(policy, str) or not SHA256.fullmatch(policy):
                return False
            if self.expected_policy_sha is not None and not hmac.compare_digest(
                policy, self.expected_policy_sha
            ):
                return False
            if self.require_ledger and not self._ledger_completed(match_id, episode.get("fixture")):
                return False
            return bool(
                episode.get("schema") == EPISODE_SCHEMA
                and episode.get("domain") == "servuo"
                and episode.get("sampling") == "categorical"
                and episode.get("eligible") is True
                and not episode.get("invalid_reasons")
                and episode.get("schema_fingerprint") == schema_fingerprint()
                and episode.get("executor_sha256") == EXECUTION_FINGERPRINT
                and meta["id"] == match_id
                and meta["training"] is True
                and meta.get("ranked", False) is False
                and meta["aborted"] is None
                and meta["rules"] == TRAINING_RULES
                and rows[0].get("rules") == TRAINING_RULES
                and episode["verification"]["replay_sha256"] == meta["sha256"]
                and len(participants) == 2
                and actor in participants
                and episode["opponent"] in participants
                and (not episode.get("require_base_template") or starting_template_verified(rows))
                and actor != episode["opponent"]
                and result["winner"] == meta["winner"]
                and result["outcome"] == outcome
                and transitions
                and transitions[-1]["done"] is True
                and transitions[-1]["reward"] == outcome
                and all(t["reward"] == 0 and t["done"] is False for t in transitions[:-1])
                and all(t["accepted"] is True and t["dt"] > 0 for t in transitions)
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
            return False


# --- one actor per game socket ----------------------------------------------------------------


def _scripted_baseline(view, opponent, round_number, log_path):
    from ..agent import Agent
    from ..decision import build_client
    from ..persona import Persona

    agent = Agent(
        view,
        Persona.load("mage_a"),
        build_client("scripted"),
        sync=True,
        pump_ms=50,
        reflect_every=0,
        log_path=log_path,
    )
    agent.memory.update(
        duel=True,
        mage=True,
        explosion_potions=True,
        opening="weaken-clumsy",
        burst_combo=False,
        duel_opponent=opponent,
        duel_round=round_number,
        duel_rules=TRAINING_RULES,
        playbook="standard",
    )
    return agent


def _actor(
    body,
    opponent,
    worker,
    log,
    stop,
    *,
    deadline=340.0,
    expected_rules=TRAINING_RULES,
    known_match_ids=frozenset(),
    teacher=None,
    probe=None,
):
    """All access to one bridge body belongs to this thread during a match.

    ``worker`` is a started ``InferenceClient`` for a neural arm, else ``None``.
    ``teacher`` names a ``TacticalTeacher`` style and ``probe`` a probe variant;
    with neither (and no worker) this seat is the fixed scripted ``Agent``. Each
    Fighting round starts a fresh epoch that resets both the executor and the
    worker's recurrent state; any other phase resets them to ``inactive`` so no
    option or late decision crosses a round. A match identity in
    ``known_match_ids`` is refused before anything is written under its directory.
    A teacher's labelled choices are returned as ``teacher_steps`` (never as
    transitions); a probe returns neither.
    """
    if sum(x is not None for x in (worker, teacher, probe)) > 1:
        raise ValueError("one seat runs exactly one policy")
    if teacher is not None and teacher not in TEACHER_STYLES:
        raise ValueError("unknown teacher style")
    if probe is not None and probe not in PROBE_VARIANTS:
        raise ValueError("unknown probe variant")
    kind = "neural" if worker else "teacher" if teacher else "probe" if probe else "scripted"
    state, last_state, last_ping = {}, time.monotonic(), 0.0
    match_id = epoch = None
    seen = False
    # Timing as on the source branch: neural and teacher seats use the fast
    # option clock and hold Explosion for the burst; dump arms vary only those.
    fast_timing = bool(worker or teacher) or probe in ("dump-fast", "dump-held", "dump-primed")
    hold_explosion = bool(worker or teacher) or probe in ("dump-held", "dump-primed")
    executor = CombatExecutor(
        opponent,
        **({"fast_timing": True, "hold_explosion": hold_explosion} if fast_timing else {}),
    )
    if fast_timing:
        # Adjacent movement and safe potion disposal fit within four tiles.
        body.terrain_radius = 4
    controller = None
    if probe is not None:
        tactic = probe.removesuffix("-resources")
        controller = (
            CastDumpProbe(tactic)
            if tactic.startswith("dump-")
            else MeleeProbe(tactic, resources=probe.endswith("-resources"))
        )
    tactical_teacher = TacticalTeacher(teacher) if teacher is not None else None
    teacher_steps = []
    transitions, invalid, latencies, events, start_times = [], set(), [], [], []
    end = time.monotonic() + deadline
    baseline = None
    view = ObservedBody(body)
    next_decision = 0.0
    seen_rounds = set()
    health_requests, last_health_request = 0, float("-inf")

    def emit(row):
        row = {"clock": time.monotonic(), **row}
        if match_id:
            append_event(log / match_id / f"actor-{body_id}.jsonl", row)
        elif len(events) < 1000:
            events.append(row)
            append_event(log / f"unassigned-actor-{body_id}.jsonl", row)

    def rpc(operation, call, *args):
        started_at = time.monotonic()
        try:
            return call(*args)
        finally:
            finished_at = time.monotonic()
            emit(
                {
                    "type": "bridge_rpc",
                    "operation": operation,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_ms": (finished_at - started_at) * 1000,
                }
            )

    def drain_worker_events():
        if worker:
            for event in worker.drain_events():
                emit(event)
                if event.get("type") == "inference_error" or (
                    event.get("type") == "inference_rejected" and event.get("epoch", epoch) == epoch
                ):
                    invalid.add(event.get("reason", event["type"]))

    body_id = body.observe().player.serial
    while not stop.is_set() and time.monotonic() < end:
        obs = rpc("observe", body.observe)
        observation_at = now = time.monotonic()
        enemy = next((m for m in obs.mobiles if m.serial == opponent), None)
        update = server_state(obs)
        if update is not None:
            state, last_state = update, now
            if state.get("id"):
                if (
                    (match_id is not None and state["id"] != match_id)
                    or state.get("opponent") != opponent
                    or state.get("rules") != expected_rules
                ):
                    raise RuntimeError("unexpected live match assignment")
                if match_id is None:
                    if state["id"] in known_match_ids:
                        raise DuplicateMatch(state["id"])
                    match_id = state["id"]
                    (log / match_id).mkdir(parents=True, exist_ok=True)
                    for row in events:
                        emit(row)
                    events.clear()
                seen = True
            emit({"type": "server_state", "state": state})
        if now - last_ping >= 0.8:
            rpc("act:duel_state", body.act, say("[DuelState"))
            last_ping = now
            emit(
                {
                    "type": "state_requested",
                    "last_state_age": now - last_state,
                    "hp": obs.player.hits,
                    "opponent_health_known": bool(enemy and enemy.hits_max > 0),
                    "bridge": getattr(body, "diagnostics", {}),
                }
            )
        if now - last_state > STATE_LOST_SECONDS:
            raise RuntimeError("lost authoritative live duel state")
        if (
            seen
            and state.get("phase") in ("Countdown", "Fighting")
            and now - last_state < STATE_FRESH_SECONDS
            and health_requests < 3
            and now - last_health_request >= 1
            and enemy is not None
            and not enemy.hidden
            and enemy.hits_max <= 0
        ):
            # The ordinary client requests the same public health bar.
            rpc("act:status_request", body.act, {"type": "StatusRequest", "serial": opponent})
            health_requests += 1
            last_health_request = now
            emit({"type": "public_health_requested", "opponent": opponent})
        final_training_round = (
            expected_rules == TRAINING_RULES
            and state.get("phase") == "RoundOver"
            and state.get("round") == 1
            and seen_rounds == {1}
        )
        if seen and (state.get("phase") == "Idle" or final_training_round):
            # The invitation requests one round. The caller still requires the
            # completed server receipt before assigning any terminal reward.
            drain_worker_events()
            if start_times:
                transitions[-1]["dt"] = max(0.001, now - start_times[-1])
            result = {
                "id": match_id,
                "actor": body_id,
                "opponent": opponent,
                "policy_kind": kind,
                "transitions": transitions,
                "invalid_reasons": sorted(invalid),
                "inference_ms": latencies,
                "policy_sha": worker.policy_sha if worker else None,
                "collection_end_phase": state.get("phase"),
            }
            if tactical_teacher is not None:
                result["teacher_steps"] = teacher_steps
            return result

        if state.get("phase") == "Fighting":
            current_epoch = f"{match_id}:{state['round']}:{body_id}"
            if current_epoch != epoch:
                seen_rounds.add(state["round"])
                if len(seen_rounds) > 1:
                    invalid.add("multiple_rounds")
                epoch = current_epoch
                executor.reset(opponent, epoch)
                if worker:
                    worker.reset(epoch)
                elif kind == "scripted":
                    baseline = _scripted_baseline(
                        view, opponent, state["round"], log / match_id / f"baseline-{body_id}.jsonl"
                    )
            if controller is not None:
                # Ported unchanged from the source probe branch. Raw Attack,
                # StunRequest and Use packets are this scripted ablation's own.
                frame = executor.observe(obs, server_state=update, now=now)
                controller.feedback(obs.new_journal, now)
                for j in obs.new_journal:
                    emit({"type": "probe_journal", "serial": j.serial,
                          "cliloc": j.cliloc, "text": j.text})  # fmt: skip
                hands_free = not any(
                    i.container == body_id and i.layer in (1, 2) for i in obs.items
                )
                if fast_timing:
                    for packet in executor.tick(obs, time.monotonic()):
                        rpc("act:" + packet["type"], body.act, packet)
                    frame = executor.observe(obs, now=time.monotonic())
                packets = controller.resource_packet(
                    executor.view, now, obs.own_pack(), busy=executor.busy, hands_free=hands_free
                )
                packets = packets or controller.packets(
                    executor.view, now, opponent, busy=executor.busy, hands_free=hands_free
                )
                for packet in packets:
                    rpc("probe:" + packet["type"], body.act, packet)
                    emit({"type": "probe_packet", "packet": packet,
                          "distance": executor.view.distance})  # fmt: skip
                if not executor.busy and not packets and now - last_state < STATE_FRESH_SECONDS:
                    name = controller.choose(executor.view, frame.mask)
                    executor.start(name, obs, time.monotonic())
                for packet in executor.tick(obs, time.monotonic()):
                    rpc("act:" + packet["type"], body.act, packet)
                for event in executor.drain_events():
                    emit({**event, "probe_distance": executor.view.distance})
                rpc("pump", body.pump, 10 if fast_timing else 50)
            elif tactical_teacher is not None:
                frame = executor.observe(obs, server_state=update, now=now)
                defending, defense_packets = (
                    executor.defend_potions(obs, time.monotonic())
                    if teacher == "tactical"
                    else (False, [])
                )
                if defending:
                    # Incoming-potion defense acts outside the 27-option schema.
                    invalid.add("potion_defense_outside_policy_schema")
                    for packet in defense_packets:
                        rpc("act:" + packet["type"], body.act, packet)
                    for event in executor.drain_events():
                        emit(event)
                    rpc("pump", body.pump, 10)
                    continue
                for packet in executor.tick(obs, time.monotonic()):
                    rpc("act:" + packet["type"], body.act, packet)
                decision_at = time.monotonic()
                # Death can arrive before the final DuelState receipt. There is
                # no decision to label outside a fresh, living combat context.
                if (
                    decision_at >= next_decision
                    and decision_at - last_state < STATE_FRESH_SECONDS
                    and executor.view.ready
                    and executor.view.fresh
                ):
                    frame = executor.observe(obs, now=decision_at)
                    action = tactical_teacher.choose(executor.view, frame.mask)
                    accepted = executor.busy and action == 0
                    if not executor.busy:
                        accepted = executor.start(action, obs, decision_at)
                    if not accepted:
                        invalid.add("teacher_action_rejected")
                    teacher_steps.append(
                        {
                            "at": decision_at,
                            "features": list(frame.features),
                            "mask": list(frame.mask),
                            "action": action,
                            "accepted": bool(accepted),
                        }
                    )
                    next_decision = decision_at + DECISION_INTERVAL
                for packet in executor.tick(obs, time.monotonic()):
                    rpc("act:" + packet["type"], body.act, packet)
                for event in executor.drain_events():
                    emit(event)
                    # Local hardening: any substituted procedure (for example a
                    # ground rethrow) is not a labelled teacher choice.
                    if policy_override(event):
                        invalid.add("policy_override")
                rpc("pump", body.pump, 10)
            elif worker:
                frame = executor.observe(obs, server_state=update, now=now)
                # Advance an already committed option even across brief state latency.
                for packet in executor.tick(obs, time.monotonic()):
                    rpc(f"act:{packet.get('type', 'unknown')}", body.act, packet)
                for event in executor.drain_events():
                    emit(event)
                    if policy_override(event):
                        invalid.add("policy_override")
                # RPCs above can block. Deadline checks and transition durations
                # use the actual consumption clock, never the old loop clock.
                now = time.monotonic()
                decision = worker.take(epoch, now)
                drain_worker_events()
                if worker.startup_error:
                    raise RuntimeError("neural worker failed during match")
                if decision is not None:
                    execution = dispatch_option(
                        executor,
                        decision.action,
                        frame,
                        obs,
                        now,
                        now - last_state < STATE_FRESH_SECONDS,
                    )
                    if transitions:
                        transitions[-1]["dt"] = max(0.001, now - start_times[-1])
                    transitions.append(
                        {
                            "features": list(decision.features),
                            "mask": list(decision.mask),
                            "action": decision.action,
                            "log_prob": decision.log_prob,
                            "value": decision.value,
                            "hidden_in": list(decision.hidden_in),
                            "reward": 0.0,
                            "done": False,
                            "dt": 0.001,
                            "accepted": True,
                            "policy_accepted": True,
                            **execution,
                        }
                    )
                    start_times.append(now)
                    latencies.append(decision.infer_ms)
                    emit(
                        {
                            "type": "policy_action",
                            "action": ACTION_NAMES[decision.action],
                            "request_id": decision.request_id,
                            "infer_ms": decision.infer_ms,
                            "observation_at": decision.observed_at,
                            "taken_at": now,
                            "observation_age_ms": (now - decision.observed_at) * 1000,
                            **execution,
                        }
                    )
                submit_now = time.monotonic()
                if (
                    submit_now >= next_decision
                    and not worker.busy
                    and submit_now - last_state < STATE_FRESH_SECONDS
                ):
                    worker.submit(
                        executor.observe(obs, now=submit_now), epoch, observed_at=observation_at
                    )
                    next_decision = time.monotonic() + DECISION_INTERVAL
                drain_worker_events()
                rpc("pump", body.pump, 10)
            elif now - last_state < STATE_FRESH_SECONDS:
                baseline.memory["showdown"] = state.get("showdown") is True
                view.pending = obs
                baseline.tick()
                del baseline.reports[:-1000]
                del baseline.proc_log[:-1000]
            else:
                rpc("pump", body.pump, 50)
        else:
            # Never execute offensive actions during countdown/round-over/lobby.
            if epoch is not None:
                executor.reset(opponent, "inactive")
                if worker:
                    worker.reset("inactive")
                epoch = None
            rpc("pump", body.pump, 50)
    raise RuntimeError("live match stopped or timed out; no unverified training result emitted")


def _seat_controller(arm: Arm | None) -> dict:
    if arm is None or arm.kind in ("neural", "scripted"):
        return {}
    return {arm.kind: arm.variant}


def fight(
    bodies, workers, log, stop, *, deadline=340.0, known_match_ids=frozenset(), arms=(None, None)
):
    """Run both actors; the first failure stops the other one immediately."""
    ids = [b.observe().player.serial for b in bodies]
    known = frozenset(known_match_ids)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                _actor,
                b,
                ids[1 - i],
                workers[i],
                log,
                stop,
                deadline=deadline,
                known_match_ids=known,
                **_seat_controller(arms[i]),
            )
            for i, b in enumerate(bodies)
        ]
        try:
            result = [None, None]
            for future in as_completed(futures):
                result[futures.index(future)] = future.result()
        except BaseException:
            stop.set()
            raise
    if result[0]["id"] != result[1]["id"]:
        raise RuntimeError("actors disagree on server match identity")
    return result


# --- bounded, resumable collections ----------------------------------------------------------


@dataclass(frozen=True)
class CollectionPlan:
    """The fixed, immutable description of one collection.

    ``purpose`` is part of the plan and its SHA-256: an ``evaluation`` collection
    (the default) is a held-out record and is never training input; only an
    explicitly ``training`` collection can feed the unit-7 learner or unit-8
    imitation. A plan with a teacher arm must verify the base template and
    supplies, because its demonstrations are bound to that evidence.
    """

    endpoint: str
    accounts: tuple[str, str]
    arms: tuple[Arm, Arm]
    matches: int
    deadline_ms: float = 500.0
    require_base_template: bool = False
    match_deadline_s: float = 340.0
    purpose: str = "evaluation"

    def __post_init__(self):
        if self.purpose not in PURPOSES:
            raise ValueError("collection purpose must be 'evaluation' or 'training'")
        if not isinstance(self.endpoint, str) or not self.endpoint:
            raise ValueError("explicit endpoint required")
        if (
            len(self.accounts) != 2
            or not all(isinstance(a, str) and a for a in self.accounts)
            or self.accounts[0].casefold() == self.accounts[1].casefold()
        ):
            raise ValueError("two distinct dedicated accounts required")
        if len(self.arms) != 2 or not all(isinstance(a, Arm) for a in self.arms):
            raise ValueError("exactly two arms required")
        if all(a.kind == "scripted" for a in self.arms):
            raise ValueError("a live collection needs a neural, teacher or probe arm")
        if any(a.kind == "teacher" for a in self.arms) and self.require_base_template is not True:
            raise ValueError("a teacher arm requires require_base_template=True")
        if type(self.matches) is not int or not 1 <= self.matches <= MAX_MATCHES:
            raise ValueError(f"matches must be an integer in 1..{MAX_MATCHES}")
        if not 1 <= self.deadline_ms <= 10_000:
            raise ValueError("inference deadline must be between 1 and 10000 ms")
        if not 30 <= self.match_deadline_s <= 900:
            raise ValueError("match deadline must be between 30 and 900 seconds")

    def to_json(self) -> dict:
        value = asdict(self)
        value["accounts"] = list(self.accounts)
        value["arms"] = [a.to_json() for a in self.arms]
        return value

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


TERMINAL = ("completed", "invalid_fixture", "failed", "unresolved")


def _row_problem(row) -> str | None:
    """Describe why a decoded ledger row is malformed, or return ``None``."""
    if not isinstance(row, dict):
        return "not an object"
    if type(row.get("fixture")) is not int or not 0 <= row["fixture"] < MAX_MATCHES:
        return "invalid fixture index"
    if row.get("event") not in ("issued", *TERMINAL):
        return "unknown event"
    for key in ("match_id", "duplicate_match_id"):
        value = row.get(key)
        if value is not None and not (isinstance(value, str) and MATCH_ID.fullmatch(value)):
            return f"invalid {key}"
    episodes = row.get("episodes")
    for key in ("episodes", "demonstrations"):
        files = row.get(key)
        if files is not None and not (
            isinstance(files, list) and all(isinstance(e, str) and e for e in files)
        ):
            return "invalid episode list" if key == "episodes" else "invalid demonstration list"
    digests = row.get("demonstration_sha256")
    if digests is not None and not (
        isinstance(digests, list)
        and isinstance(row.get("demonstrations"), list)
        and len(digests) == len(row["demonstrations"])
        and all(isinstance(d, str) and SHA256.fullmatch(d) for d in digests)
    ):
        return "invalid demonstration digests"
    if row.get("demonstrations") is not None and digests is None:
        return "demonstrations without digests"
    if "result" in row and not isinstance(row["result"], dict):
        return "invalid result"
    if row["event"] == "completed" and not (
        row.get("match_id") and isinstance(episodes, list) and isinstance(row.get("result"), dict)
    ):
        return "completed row lacks match, episodes or result"
    return None


class Ledger:
    """Append-only fixture ledger; ``issued`` is written before any challenge.

    Every row is a complete line written with one ``write`` and fsynced before the
    caller proceeds. A torn or otherwise undecodable line (including a final line
    cut short by a crash) is refused rather than guessed at: inspect and repair
    the file by hand. A torn ``issued`` line means its challenge was never sent;
    a torn terminal line leaves the fixture open, which resume closes as
    ``unresolved``.
    """

    def __init__(self, path: Path, entries: list[dict]):
        self.path = path
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> Ledger:
        entries = []
        if path.exists():
            text = path.read_text()
            if text and not text.endswith("\n"):
                raise CollectionRefused("torn final ledger line; repair the ledger by hand")
            for n, line in enumerate(text.splitlines()):
                try:
                    row = json.loads(line)
                except ValueError:
                    raise CollectionRefused(f"corrupt ledger line {n + 1}") from None
                problem = _row_problem(row)
                if problem:
                    raise CollectionRefused(f"malformed ledger line {n + 1}: {problem}")
                entries.append(row)
        ledger = cls(path, entries)
        ledger._check()
        return ledger

    def _check(self) -> None:
        issued, closed, seen_ids = [], set(), set()
        for row in self.entries:
            problem = _row_problem(row)
            if problem:
                raise CollectionRefused(f"malformed ledger row: {problem}")
            fixture = row["fixture"]
            if row["event"] == "issued":
                if fixture in issued or fixture != len(issued):
                    raise CollectionRefused("ledger fixtures are not issued in order")
                issued.append(fixture)
                continue
            if fixture not in issued or fixture in closed:
                raise CollectionRefused("ledger closes an unknown or already closed fixture")
            closed.add(fixture)
            match_id = row.get("match_id")
            if match_id is not None:
                if match_id in seen_ids:
                    raise CollectionRefused("duplicate match identity in collection ledger")
                seen_ids.add(match_id)

    def _append(self, row: dict) -> None:
        Ledger(self.path, [*self.entries, row])._check()
        created = not self.path.exists()
        line = json.dumps(row, sort_keys=True) + "\n"
        with self.path.open("a") as out:
            out.write(line)
            out.flush()
            os.fsync(out.fileno())
        if created:
            _fsync_dir(self.path.parent)
        self.entries.append(row)

    @property
    def issued(self) -> int:
        return sum(1 for r in self.entries if r["event"] == "issued")

    @property
    def open_fixtures(self) -> list[int]:
        closed = {r["fixture"] for r in self.entries if r["event"] != "issued"}
        return [
            r["fixture"]
            for r in self.entries
            if r["event"] == "issued" and r["fixture"] not in closed
        ]

    @property
    def match_ids(self) -> set[str]:
        return {r["match_id"] for r in self.entries if r.get("match_id")}

    @property
    def completed(self) -> list[dict]:
        return [r for r in self.entries if r["event"] == "completed"]

    def issue(self, fixture: int, **fields) -> None:
        self._append({"fixture": fixture, "event": "issued", "utc": time.time(), **fields})

    def close(self, fixture: int, event: str, **fields) -> None:
        if event not in TERMINAL:
            raise ValueError("unknown terminal ledger event")
        self._append({"fixture": fixture, "event": event, "utc": time.time(), **fields})


def _staged(path: Path) -> Path:
    return path.parent / STAGING / path.name


def _publish(paths) -> None:
    """Move staged episodes to their published names; only after ``completed``."""
    for path in map(Path, paths):
        staged = _staged(path)
        if staged.exists():
            os.replace(staged, path)
            _fsync_dir(path.parent)


def open_collection(log: Path, plan: CollectionPlan, *, resume: bool = False) -> Ledger:
    """Create a new collection or resume the identical plan within its fixed budget."""
    log.mkdir(parents=True, exist_ok=True)
    manifest_path = log / "collection.json"
    if manifest_path.exists():
        if not resume:
            raise CollectionRefused(
                "collection already exists; pass resume or choose a new directory"
            )
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError:
            raise CollectionRefused("corrupt collection manifest") from None
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != COLLECTION_SCHEMA
            or manifest.get("plan_sha256") != plan.sha256
            or manifest.get("plan") != plan.to_json()
        ):
            raise CollectionRefused("resume requires the identical plan and policy identities")
    elif resume:
        raise CollectionRefused("nothing to resume in this directory")
    else:
        atomic_json(
            manifest_path,
            {
                "schema": COLLECTION_SCHEMA,
                "plan": plan.to_json(),
                "plan_sha256": plan.sha256,
                "schema_fingerprint": schema_fingerprint(),
                "executor_sha256": EXECUTION_FINGERPRINT,
                "collector_version": COLLECTOR_VERSION,
                "automatic_promotion": False,
            },
        )
    ledger = Ledger.load(log / "ledger.jsonl")
    for row in ledger.completed:
        # A crash between the completed row and publication left staged files.
        _publish([*row["episodes"], *row.get("demonstrations", [])])
    for fixture in ledger.open_fixtures:
        # An issued fixture whose outcome was not recorded is spent, never reissued.
        ledger.close(fixture, "unresolved", reason="interrupted before a recorded outcome")
    if ledger.issued >= plan.matches:
        raise CollectionRefused("collection budget is exhausted")
    return ledger


def _pumping(bodies, call, *args):
    """Keep both game sockets serviced while a slow receipt download runs."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(call, *args)
        while not result.done():
            for body in bodies:
                body.pump(100)
        return result.result()


def run_collection(
    plan: CollectionPlan,
    log_dir,
    *,
    connect: Callable[[str], Any],
    fetch_receipt: Callable[[str, list[int]], tuple[dict, bytes]],
    lock_dir,
    stop: threading.Event | None = None,
    resume: bool = False,
    client_factory=InferenceClient,
    find_spec=importlib.util.find_spec,
    ready_timeout: float = 90.0,
) -> list[dict]:
    """Issue at most the plan's remaining fixtures and record verified episodes.

    ``connect(account)`` returns a logged-in body; ``fetch_receipt(match_id,
    seats)`` returns the server manifest and replay bytes. Neither is called
    before both arms pass preflight and every neural worker reports its
    declared policy SHA-256. Any failure after a challenge is issued is recorded
    in the ledger and stops the run; it is never retried automatically. Returns
    this run's fixtures; ``results.json`` lists every completed fixture of the
    collection, rebuilt from the ledger.
    """
    import fcntl

    stop = stop or threading.Event()
    log = Path(log_dir).resolve()
    log.mkdir(parents=True, exist_ok=True)
    results = []
    with ExitStack() as stack:
        # Callbacks run in reverse and independently: bodies and workers close
        # first, then the account locks are released, then the run lock.
        lock = stack.enter_context((log / "run.lock").open("a"))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CollectionRefused("another runner holds this collection") from None
        # Refuse an unavailable arm before a fresh run writes its collection.json.
        for arm in plan.arms:
            preflight_arm(arm, find_spec=find_spec)
        ledger = open_collection(log, plan, resume=resume)

        def save(status, error=None):
            atomic_json(
                log / "results.json",
                {
                    "domain": "servuo",
                    "plan_sha256": plan.sha256,
                    "purpose": plan.purpose,
                    "matches": [row["result"] for row in ledger.completed],
                    "issued": ledger.issued,
                    "budget": plan.matches,
                    "automatic_promotion": False,
                    "collection_status": status,
                    "error": error,
                },
            )

        try:
            stack.enter_context(account_locks(lock_dir, plan.endpoint, list(plan.accounts)))
            workers = []
            for arm in plan.arms:
                worker = None
                if arm.neural:
                    worker = client_factory(
                        arm.checkpoint,
                        expected_sha256=arm.policy_sha256,
                        deadline_ms=plan.deadline_ms,
                    ).start()
                    stack.callback(worker.close)
                workers.append(worker)
            await_workers(workers, plan.arms, timeout=ready_timeout, stopped=stop.is_set)
            bodies = []
            for account in plan.accounts:
                atomic_json(log / "status.json", {"stage": "connecting", "account": account})
                body = connect(account)
                stack.callback(body.close)
                bodies.append(body)
            ids = wait_training_idle(bodies, stopped=stop.is_set)
            for fixture in range(ledger.issued, plan.matches):
                if stop.is_set():
                    break
                results.append(
                    _one_fixture(
                        plan, log, ledger, fixture, bodies, ids, workers, stop, fetch_receipt
                    )
                )
                save("collecting")
            save("stopped" if stop.is_set() else "complete")
            return results
        except BaseException as exc:
            save("stopped", f"{type(exc).__name__}: {exc}")
            atomic_json(log / "status.json", {"stage": "stopped", "error": str(exc)})
            raise


def _demonstration(plan, fixture, actor, arm, other, match_id, meta, rows) -> dict:
    """A teacher seat's labelled choices with their full provenance; never a PPO episode."""
    steps = actor.pop("teacher_steps")
    actor.pop("transitions", None)
    invalid = list(actor["invalid_reasons"])
    if not starting_template_verified(rows):
        invalid.append("wrong_starting_template")
    outcome = 0.0 if meta["winner"] is None else (1.0 if meta["winner"] == actor["actor"] else -1.0)
    return {
        "schema": DEMONSTRATION_SCHEMA,
        "domain": "servuo",
        "policy_kind": "teacher",
        # A scripted teacher is never reported under a neural label or SHA.
        "label": f"teacher:{arm.variant}",
        "teacher": {
            "name": "TacticalTeacher",
            "style": arm.variant,
            "version": TEACHER_VERSION,
            "sha256": arm.source_sha256,
            "sources": dict(TEACHER_SOURCES),
        },
        "ppo_eligible": False,
        "eligible": bool(steps) and not invalid,
        "invalid_reasons": sorted(set(invalid)),
        "fixture": fixture,
        "plan_sha256": plan.sha256,
        "purpose": plan.purpose,
        "executor_sha256": EXECUTION_FINGERPRINT,
        "collector_version": COLLECTOR_VERSION,
        "schema_fingerprint": schema_fingerprint(),
        "actor": actor["actor"],
        "opponent": actor["opponent"],
        "opponent_arm": other.to_json(),
        "verification": {"match_id": match_id, "replay_sha256": meta["sha256"]},
        "final_result": {"winner": meta["winner"], "outcome": outcome},
        "collection_end_phase": actor.get("collection_end_phase"),
        "steps": steps,
    }


def _one_fixture(plan, log, ledger, fixture, bodies, ids, workers, stop, fetch_receipt):
    atomic_json(log / "status.json", {"stage": "preparing", "fixture": fixture + 1})
    with ThreadPoolExecutor(max_workers=2) as pool:
        players = list(
            pool.map(
                lambda b: prepare(
                    b,
                    previously_configured=fixture > 0,
                    verify_base_stats=plan.require_base_template,
                    stopped=stop.is_set,
                ),
                bodies,
            )
        )
    if [p.get("serial", ids[i]) for i, p in enumerate(players)] != ids:
        raise RuntimeError("prepared participants differ from the idle pair")
    supplies = None
    if plan.require_base_template:
        supplies = [verify_training_supplies(b) for b in bodies]
        if not all(audit["verified"] for audit in supplies):
            raise RuntimeError("unequal or incomplete training supplies; no challenge issued")
    for body, arm in zip(bodies, plan.arms, strict=True):
        body.act(say(f"[ArenaAgent {arm.label} {arm.agent_policy}"))
    order = [0, 1] if fixture % 2 == 0 else [1, 0]
    challenger, receiver = order
    # The pre-challenge supply audit is part of the durable issued row, so a
    # demonstration can later be checked against evidence written before play.
    ledger.issue(
        fixture,
        challenger=ids[challenger],
        opponent=ids[receiver],
        **({"supplies": supplies} if supplies is not None else {}),
    )
    match_id = None
    try:
        request_training_challenge(bodies[challenger], opponent=ids[receiver], stopped=stop.is_set)
        accept_invitation(bodies[receiver], opponent=ids[challenger], stopped=stop.is_set)
        atomic_json(log / "status.json", {"stage": "fighting", "fixture": fixture + 1})
        actors = fight(
            bodies,
            workers,
            log,
            stop,
            deadline=plan.match_deadline_s,
            known_match_ids=ledger.match_ids,
            arms=plan.arms,
        )
        reported = actors[0]["id"]
        if reported in ledger.match_ids:
            raise DuplicateMatch(reported)
        match_id = reported
        for actor, arm in zip(actors, plan.arms, strict=True):
            if actor["policy_sha"] != (arm.policy_sha256 if arm.neural else None):
                raise RuntimeError("actor policy identity differs from the declared arm")
            if actor.get("policy_kind") != arm.kind:
                raise RuntimeError("actor policy kind differs from the declared arm")
            if ("teacher_steps" in actor) != (arm.kind == "teacher"):
                raise RuntimeError("only a teacher arm may report teacher steps")
        seats = [ids[i] for i in order]
        meta, data = _pumping(bodies, fetch_receipt, match_id, seats)
    except BaseException as exc:
        fields = {"reason": f"{type(exc).__name__}: {exc}"[:500]}
        if isinstance(exc, DuplicateMatch):
            fields["duplicate_match_id"] = exc.match_id
        elif match_id is not None:
            fields["match_id"] = match_id
        ledger.close(fixture, "failed", **fields)
        raise
    try:
        rows = verify_receipt(data, meta, seats, expected_match_id=match_id)
    except (ValueError, KeyError, TypeError) as exc:
        ledger.close(fixture, "invalid_fixture", match_id=match_id, reason=str(exc)[:500])
        raise CollectionRefused(f"invalid fixture receipt: {exc}") from exc
    root = log / match_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "replay.jsonl").write_bytes(data)
    atomic_json(root / "replay.meta.json", meta)
    if plan.require_base_template and not starting_template_verified(rows):
        for actor in actors:
            actor["invalid_reasons"].append("wrong_starting_template")
    # Stage every episode and verify all of them before the ledger row. Only a
    # completed row publishes them, so no refused or interrupted fixture leaves
    # an episode under its published name.
    files, eligible, demonstrations, digests = [], [], [], []
    for i, (actor, arm) in enumerate(zip(actors, plan.arms, strict=True)):
        if arm.kind == "teacher":
            demonstration = _demonstration(
                plan, fixture, actor, arm, plan.arms[1 - i], match_id, meta, rows
            )
            from .demonstrations import DemonstrationReceiptVerifier

            verifier = DemonstrationReceiptVerifier(log, require_ledger=False)
            if demonstration["eligible"] and not verifier(demonstration):
                ledger.close(
                    fixture, "invalid_fixture", match_id=match_id, reason="demonstration receipt"
                )
                raise CollectionRefused("collected demonstration failed receipt verification")
            file = root / f"demonstration-{ids[i]}.json"
            atomic_json(_staged(file), demonstration)
            demonstrations.append(str(file))
            # The completed row binds each demonstration file's bytes.
            digests.append(hashlib.sha256(_staged(file).read_bytes()).hexdigest())
            continue
        if not arm.neural:
            continue
        transitions = actor.pop("transitions")
        outcome = 0.0 if meta["winner"] is None else (1.0 if meta["winner"] == ids[i] else -1.0)
        if transitions:
            transitions[-1].update(reward=outcome, done=True)
        episode = {
            **actor,
            "schema": EPISODE_SCHEMA,
            "domain": "servuo",
            "sampling": "categorical",
            "fixture": fixture,
            "plan_sha256": plan.sha256,
            "purpose": plan.purpose,
            "executor_sha256": EXECUTION_FINGERPRINT,
            "collector_version": COLLECTOR_VERSION,
            "schema_fingerprint": schema_fingerprint(),
            "require_base_template": plan.require_base_template,
            "eligible": bool(transitions) and not actor["invalid_reasons"],
            "transitions": transitions,
            "verification": {
                "verified": True,
                "match_id": match_id,
                "replay_sha256": meta["sha256"],
                "complete": True,
                "training": True,
                "aborted": False,
            },
            "final_result": {"winner": meta["winner"], "outcome": outcome},
        }
        verifier = LiveReplayVerifier(log, arm.policy_sha256, require_ledger=False)
        if episode["eligible"] and not verifier(episode):
            ledger.close(fixture, "invalid_fixture", match_id=match_id, reason="episode receipt")
            raise CollectionRefused("collected episode failed receipt verification")
        file = root / f"episode-{ids[i]}.json"
        atomic_json(_staged(file), episode)
        files.append(str(file))
        eligible.append(episode["eligible"])
    result = {
        "fixture": fixture,
        "challenger": ids[challenger],
        "match_id": match_id,
        "winner": meta["winner"],
        "metrics": replay.metrics(rows),
        "episodes": files,
        "eligible": all(eligible),
        "arms": [arm.label for arm in plan.arms],
    }
    if demonstrations:
        result["demonstrations"] = demonstrations
    ledger.close(
        fixture,
        "completed",
        match_id=match_id,
        episodes=files,
        result=result,
        **(
            {"demonstrations": demonstrations, "demonstration_sha256": digests}
            if demonstrations
            else {}
        ),
    )
    _publish([*files, *demonstrations])
    atomic_json(log / "status.json", {"stage": "verified", **result})
    return result


def server_receipt(web: str, *, timeout: float = 120.0, sleep=time.sleep):
    """Return a ``fetch_receipt`` that polls the read-only replay index within a bound."""
    if not 1 <= timeout <= 600:
        raise ValueError("receipt timeout must be between 1 and 600 seconds")

    def fetch(match_id, _seats):
        if not MATCH_ID.fullmatch(match_id):
            raise ValueError("invalid match identity")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            meta = next((m for m in replay.index(web) if m.get("id") == match_id), None)
            if meta is not None and meta.get("complete"):
                data, _ = replay.download(web, meta)
                return meta, data
            sleep(1.0)
        raise TimeoutError("server receipt missing; no result will be invented")

    return fetch
