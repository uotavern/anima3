"""Bounded simulator curriculum league with simulator-only gates and exact resume.

Requires the ``neural`` extra and is never imported by ``anima3`` or
``anima3.neural``. There is no command-line entry, no ``__main__`` and no
schedule. A caller invokes :func:`run_league` once with a :class:`LeaguePlan`
whose every budget, seed, learning rate and evaluation parameter is stated, and
a fresh output directory.

The loop is the source ``neural/league.py`` ``main`` at the training tip
``819f4f4`` (``94051a2``, ``25a169f``, ``366ec62``, ``95e5d0f``, ``e05ee1e``,
``a972d84``): an initial policy (or an exact parent loaded by SHA-256), teacher
demonstrations from the four drills, behaviour cloning to a warm start, PPO
updates against a league of the warm start and the last frozen snapshots (with
optional teacher rehearsal), then the simulator gates on initial, warm start and
final candidate. One global RNG stream, seeded once with ``plan.seed`` inside
``torch.random.fork_rng``, drives the whole run as in the source, so the same plan
gives the source's weights bit for bit.

- **Simulator provenance.** Every checkpoint the run writes says
  ``source: simulator-curriculum-league`` and the simulator domain and code
  identities. Every trained one (warm start, snapshots) is ``simulator_trained``
  and carries ``simulator_ancestry`` and a ``simulator_consumed`` record of the
  rollout seeds, drills and steps it was trained on, plus the parent's
  ``live_consumed`` and ``demonstrations_consumed`` unchanged (simulator data
  consumes no live episode or live demonstration). Live PPO and imitation
  children of a marked checkpoint stay marked (``train.ancestry_simulator``).
- **Run-local experimental registry.** ``registry.json`` names the checkpoint the
  simulator gates select (``registry_selected``) in the unit-9 vocabulary, with
  ``deployment: not_deployed`` for every checkpoint. Selection is not deployment:
  it is a simulator result, ``live_gain_established: false``, and it can never be
  recorded as a live selection.
- **Exact resume.** After the warm start and after every update the run writes,
  beside the snapshot, the Adam states of both optimizers, the RNG state and the
  league membership as SHA-256-bound blobs and then, atomically, a stage
  manifest. :func:`run_league` with ``resume_from`` refuses another plan or any
  blob, checkpoint or demonstration file whose SHA-256 differs, copies the run up
  to its last complete stage into a new directory and continues. A stage whose
  manifest was never written (a crash mid-write) is ignored. The result is
  bit-identical to an uninterrupted run, however often it is resumed.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from .checkpoints import _atomic_write, _serialize, load_checkpoint, save_checkpoint
from .drills import DRILLS, SIM_DOMAIN, SIMULATOR_SHA256
from .evaluation import EvaluationPlan
from .imitation import ImitationConfig, behavioral_clone
from .inference import require_sha256
from .live import atomic_json
from .model import PolicyConfig, RecurrentPolicy
from .schema import ACTION_NAMES, FEATURE_NAMES
from .sim import DuelSim
from .sim_evaluation import (
    EVALUATION_SEED_FLOOR,
    MAX_SEED,
    MAX_SIM_STEPS,
    SAMPLING,
    SimulatorEvaluationPlan,
    simulator_gate,
    simulator_record,
    teacher_opponent,
)
from .sim_rollouts import _collect, _suite_rows
from .teacher import STYLES, TEACHER_SHA256
from .train import (
    MAX_TOTAL_TRANSITIONS,
    TrainConfig,
    Trajectory,
    _discount,
    ancestry_consumed,
    ancestry_simulator,
    ppo_update,
)

PLAN_SCHEMA = "anima3-simulator-league-plan-v1"
STAGE_SCHEMA = "anima3-simulator-league-stage-v1"
REGISTRY_SCHEMA = "anima3-simulator-league-registry-v1"
SOURCE = "simulator-curriculum-league"
MAX_UPDATES = 64
MAX_REHEARSAL_EPOCHS = 8
# The source league's policy: 64 hidden and 64 encoder units.
HIDDEN_SIZE = ENCODER_SIZE = 64
MAX_BLOB_BYTES = 128 * 1024 * 1024
MAX_DEMONSTRATION_BYTES = 512 * 1024 * 1024
STAGE = re.compile(r"stage-(\d{3})\.json")


class LeagueRefused(ValueError):
    """The league cannot run or resume as asked; nothing was trained."""


def _rate(name: str, value) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 0.1:
        raise ValueError(f"{name} must be a finite learning rate in (0, 0.1]")


def _count(name: str, value, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


@dataclass(frozen=True, kw_only=True)
class LeaguePlan:
    """Every budget, seed and threshold of one league run; nothing has a default.

    ``seed`` seeds the run's single RNG stream; demonstrations use ``seed + i``,
    rollouts ``seed + 1000 + update``, PPO shuffles ``seed + update`` and
    rehearsal ``seed + 10000 + update``, as in the source. All of them stay below
    ``EVALUATION_SEED_FLOOR``; evaluation uses ``evaluation_seed`` at or above it.
    ``start_checkpoint`` and ``start_sha256`` are both ``None`` (a fresh initial
    policy) or name an exact parent by absolute path and weight SHA-256.
    """

    seed: int
    bc_steps_per_drill: int
    bc: ImitationConfig
    bc_learning_rate: float
    rehearsal_epochs: int
    rehearsal_learning_rate: float
    ppo: TrainConfig
    updates: int
    rollout_steps: int
    max_steps: int
    evaluation_seed: int
    evaluation_games_per_style: int
    evaluation_sampling: str
    threads: int
    start_checkpoint: str | None
    start_sha256: str | None

    def __post_init__(self):
        _count("updates", self.updates, 1, MAX_UPDATES)
        if type(self.seed) is not int or not 0 <= self.seed:
            raise ValueError("an explicit non-negative integer seed is required")
        if self.seed + 10_000 + self.updates >= EVALUATION_SEED_FLOOR:
            raise ValueError("training seeds must stay below the held-out evaluation range")
        _count("bc_steps_per_drill", self.bc_steps_per_drill, 1, MAX_TOTAL_TRANSITIONS // 4)
        if type(self.bc) is not ImitationConfig or self.bc.reference_kl_coef != 0:
            raise ValueError("bc must be an ImitationConfig without a reference model")
        _rate("bc_learning_rate", self.bc_learning_rate)
        _count("rehearsal_epochs", self.rehearsal_epochs, 0, MAX_REHEARSAL_EPOCHS)
        _rate("rehearsal_learning_rate", self.rehearsal_learning_rate)
        if type(self.ppo) is not TrainConfig:
            raise ValueError("ppo must be an explicit TrainConfig")
        _count("rollout_steps", self.rollout_steps, 1, MAX_TOTAL_TRANSITIONS)
        _count("max_steps", self.max_steps, 1, MAX_SIM_STEPS)
        if type(self.evaluation_seed) is not int or not (
            EVALUATION_SEED_FLOOR <= self.evaluation_seed <= MAX_SEED
        ):
            raise ValueError("evaluation needs an explicit held-out seed >= 10**9")
        games = self.evaluation_games_per_style
        if type(games) is not int or games % 2 or not 20 <= games <= 100:
            raise ValueError("evaluation games per style must be an even integer in 20..100")
        if self.evaluation_sampling not in SAMPLING:
            raise ValueError("evaluation_sampling must be 'categorical' or 'greedy'")
        _count("threads", self.threads, 1, 64)
        if (self.start_checkpoint is None) != (self.start_sha256 is None):
            raise ValueError("name a start checkpoint and its SHA-256 together, or neither")
        if self.start_checkpoint is not None:
            if (
                not isinstance(self.start_checkpoint, str)
                or not Path(self.start_checkpoint).is_absolute()
            ):
                raise ValueError("name the start checkpoint by an absolute path")
            require_sha256(self.start_sha256)

    def to_json(self) -> dict:
        return {"schema": PLAN_SCHEMA, **asdict(self)}

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def shaping_gamma(self) -> float:
        # Source: gamma ** (dt / unit) for an elapsed-time discount, else gamma per step.
        unit = self.ppo.discount_time_unit
        dt = DuelSim(0, 1, shaping_gamma=1.0).dt
        return self.ppo.gamma ** (dt / unit) if unit else self.ppo.gamma


def _evaluation_plan(plan: LeaguePlan, incumbent: str, candidate: str):
    return SimulatorEvaluationPlan(
        evaluation=EvaluationPlan(
            candidate_sha256=candidate,
            incumbent_sha256=incumbent,
            opponents=tuple(teacher_opponent(s) for s in STYLES),
            games_per_opponent=plan.evaluation_games_per_style,
            domain=SIM_DOMAIN,
        ),
        seed=plan.evaluation_seed,
        max_steps=plan.max_steps,
        sampling=plan.evaluation_sampling,
        simulator_sha256=SIMULATOR_SHA256,
        teacher_sha256=TEACHER_SHA256,
    )


# --- paths and SHA-bound blobs ---------------------------------------------------------------


def _absolute(path, what: str) -> Path:
    if not isinstance(path, (str, Path)) or not Path(path).is_absolute():
        raise LeagueRefused(f"name the {what} by an absolute path")
    return Path(path)


def _fresh_run(out, protected: list[Path]) -> Path:
    raw = _absolute(out, "output directory")
    resolved = raw.resolve()
    if raw.is_symlink() or raw.exists() or resolved.exists():
        raise LeagueRefused("league output directory already exists; choose a new one")
    for other in protected:
        other = other.resolve()
        if resolved == other or resolved.is_relative_to(other) or other.is_relative_to(resolved):
            raise LeagueRefused("league output must be outside every input")
    return resolved


def _blob(root: Path, prefix: str, value) -> dict:
    payload = _serialize(value)
    digest = hashlib.sha256(payload).hexdigest()
    name = f"{prefix}-{digest}.pt"
    _atomic_write(root / name, payload)
    return {"file": name, "sha256": digest, "bytes": len(payload)}


def _read_verified(path: Path, sha256: str, limit: int) -> bytes:
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit or hashlib.sha256(payload).hexdigest() != sha256:
        raise LeagueRefused(f"{path.name} does not hash to its recorded SHA-256")
    return payload


def _read_blob(root: Path, descriptor) -> object:
    if not isinstance(descriptor, dict) or Path(str(descriptor.get("file"))).name != str(
        descriptor.get("file")
    ):
        raise LeagueRefused("malformed resume blob descriptor")
    payload = _read_verified(root / descriptor["file"], descriptor.get("sha256"), MAX_BLOB_BYTES)
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def _load(path: Path, sha256: str):
    # Building a model draws from the RNG; the run's stream must not notice.
    with torch.random.fork_rng(devices=[]):
        return load_checkpoint(path, expected_sha256=sha256)


# --- the run ---------------------------------------------------------------------------------


class _Run:
    def __init__(self, plan: LeaguePlan, root: Path):
        self.plan = plan
        self.root = root
        self.events: list[dict] = []
        self.checkpoints: dict[str, str] = {}

    def path(self, label: str) -> Path:
        return self.root / ("snapshots/" + label if label.startswith("update-") else label)

    def record(self, row: dict) -> None:
        self.events.append(row)
        atomic_json(self.root / "progress.json", {"events": self.events, "status": "running"})

    def save(self, label: str, model, training: dict) -> str:
        manifest = save_checkpoint(self.path(label), model, training=training)
        sha = manifest["weights"]["sha256"]
        self.checkpoints[label] = sha
        return sha


def _demonstration_file(demos: list[Trajectory]) -> dict:
    return {
        "domain": SIM_DOMAIN,
        "ppo_eligible": False,
        "episodes": [
            {"features": e.features, "masks": e.masks, "actions": e.actions} for e in demos
        ],
    }


def _demonstrations_from(value: dict) -> list[Trajectory]:
    """BC reads only features, masks, actions and the episode's (zero) initial state."""
    episodes = []
    for row in value["episodes"]:
        trajectory = Trajectory(info={"domain": SIM_DOMAIN, "ppo_eligible": False})
        trajectory.features = row["features"]
        trajectory.masks = row["masks"]
        trajectory.actions = row["actions"]
        trajectory.hidden = [[0.0] * HIDDEN_SIZE for _ in row["actions"]]
        trajectory.log_probs = [0.0] * len(row["actions"])
        trajectory.values = [0.0] * len(row["actions"])
        trajectory.rewards = [0.0] * len(row["actions"])
        episodes.append(trajectory)
    return episodes


def _write_stage(run: _Run, stage: int, state: dict) -> None:
    blobs = run.root / "resume" / f"stage-{stage:03d}"
    manifest = {
        "schema": STAGE_SCHEMA,
        "plan_sha256": run.plan.sha256,
        "stage": stage,
        "model": state["model"],
        "checkpoints": dict(run.checkpoints),
        "league": state["league"],
        "simulator_consumed": state["simulator_consumed"],
        "demonstrations_sha256": state["demonstrations_sha256"],
        "events": list(run.events),
        "optimizers": {
            "ppo": _blob(blobs, "ppo-optimizer", state["optimizer"].state_dict()),
            "rehearsal": _blob(blobs, "rehearsal-optimizer", state["rehearsal"].state_dict()),
        },
        "rng": _blob(blobs, "rng", torch.get_rng_state()),
    }
    # The manifest is written last and atomically: without it the stage never happened.
    atomic_json(run.root / "resume" / f"stage-{stage:03d}.json", manifest)


def _latest_stage(previous: Path, plan: LeaguePlan) -> dict | None:
    """The last complete stage of a previous run of exactly this plan, fully verified."""
    try:
        recorded = json.loads((previous / "plan.json").read_text())
    except (OSError, ValueError) as exc:
        raise LeagueRefused(f"previous run has no readable plan: {exc}") from exc
    if recorded != {"schema": PLAN_SCHEMA, "plan": plan.to_json(), "plan_sha256": plan.sha256}:
        raise LeagueRefused("previous run was made with another plan or configuration")
    stages = sorted(
        int(m.group(1))
        for p in (previous / "resume").glob("stage-*.json")
        if (m := STAGE.fullmatch(p.name))
    )
    if not stages:
        return None
    if stages != list(range(len(stages))):
        raise LeagueRefused("previous run's resume stages are not contiguous")
    states = []
    for stage in stages:
        try:
            state = json.loads((previous / "resume" / f"stage-{stage:03d}.json").read_text())
        except (OSError, ValueError) as exc:
            raise LeagueRefused(f"resume stage {stage} is unreadable: {exc}") from exc
        if (
            not isinstance(state, dict)
            or state.get("schema") != STAGE_SCHEMA
            or state.get("plan_sha256") != plan.sha256
            or state.get("stage") != stage
        ):
            raise LeagueRefused(f"resume stage {stage} belongs to another plan")
        blobs = previous / "resume" / f"stage-{stage:03d}"
        for descriptor in (*state["optimizers"].values(), state["rng"]):
            _read_verified(blobs / descriptor["file"], descriptor["sha256"], MAX_BLOB_BYTES)
        states.append(state)
    final = states[-1]
    expected = ["initial", "warmstart", *(f"update-{u:03d}" for u in range(1, final["stage"] + 1))]
    if sorted(final["checkpoints"]) != sorted(expected):
        raise LeagueRefused("resume stage does not name the run's checkpoints")
    run = _Run(plan, previous)
    for label, sha in final["checkpoints"].items():
        _load(run.path(label), sha)  # refuses any checkpoint that does not hash to its SHA
    _read_verified(
        previous / "demonstrations.json", final["demonstrations_sha256"], MAX_DEMONSTRATION_BYTES
    )
    final["_stages"] = stages
    return final


def _copy_run(plan: LeaguePlan, previous: Path, root: Path, state: dict) -> None:
    source, target = _Run(plan, previous), _Run(plan, root)
    for label in state["checkpoints"]:
        shutil.copytree(source.path(label), target.path(label))
    shutil.copy2(previous / "demonstrations.json", root / "demonstrations.json")
    for stage in state["_stages"]:
        name = f"stage-{stage:03d}"
        shutil.copytree(previous / "resume" / name, root / "resume" / name)
        shutil.copy2(previous / "resume" / f"{name}.json", root / "resume" / f"{name}.json")


def run_league(plan: LeaguePlan, out, *, resume_from=None) -> dict:
    """Run (or resume) the simulator league once into the fresh directory ``out``.

    ``resume_from`` names a previous run directory of exactly this plan; its last
    complete stage is verified, copied into ``out`` and continued. Nothing outside
    ``out`` is written. Returns the run-local registry.
    """
    if type(plan) is not LeaguePlan:
        raise LeagueRefused("an explicit LeaguePlan is required")
    protected = []
    if plan.start_checkpoint is not None:
        protected.append(Path(plan.start_checkpoint))
    previous = None
    if resume_from is not None:
        previous = _absolute(resume_from, "previous run").resolve()
        protected.append(previous)
    root = _fresh_run(out, protected)
    state = _latest_stage(previous, plan) if previous is not None else None
    threads = torch.get_num_threads()
    torch.set_num_threads(plan.threads)
    try:
        with torch.random.fork_rng(devices=[]):
            root.mkdir(parents=True)
            atomic_json(
                root / "plan.json",
                {"schema": PLAN_SCHEMA, "plan": plan.to_json(), "plan_sha256": plan.sha256},
            )
            if state is not None:
                _copy_run(plan, previous, root, state)
            return _league(plan, root, state, previous)
    finally:
        torch.set_num_threads(threads)


def _league(plan: LeaguePlan, root: Path, state: dict | None, previous: Path | None) -> dict:
    run = _Run(plan, root)
    shaping_gamma = plan.shaping_gamma()
    simulator = {
        "simulator_sha256": SIMULATOR_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "plan_sha256": plan.sha256,
    }
    base = {
        "source": SOURCE,
        "domain": SIM_DOMAIN,
        "simulator": simulator,
        "seed": plan.seed,
        "discount": _discount(plan.ppo),
        "shaping_gamma": shaping_gamma,
        "bc_rehearsal_epochs": plan.rehearsal_epochs,
        "real_server_strength_established": False,
        "live_gain_established": False,
    }
    if state is None:
        # As the source: one seed for the whole run, then the initial policy.
        torch.manual_seed(plan.seed)
        if plan.start_checkpoint is not None:
            loaded = load_checkpoint(plan.start_checkpoint, expected_sha256=plan.start_sha256)
            model = loaded.model
            live, demonstrations = ancestry_consumed(loaded)
            ancestry, consumed = ancestry_simulator(loaded)
            parent = {"parent_policy_sha": loaded.policy_sha}
        else:
            model = RecurrentPolicy(
                PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), HIDDEN_SIZE, ENCODER_SIZE)
            )
            live, demonstrations, ancestry, consumed, parent = [], [], False, [], {}
        base.update(live_consumed=live, demonstrations_consumed=demonstrations)
        optimizer = torch.optim.Adam(model.parameters(), lr=plan.bc_learning_rate, eps=1e-5)
        initial_sha = run.save(
            "initial",
            model,
            {
                **base,
                **parent,
                "stage": "initial",
                "simulator_trained": False,
                "simulator_ancestry": ancestry,
                "simulator_consumed": consumed,
            },
        )
        demos = []
        consumed = list(consumed)
        for index, drill in enumerate(DRILLS):
            trajectories, summary = _collect(
                model,
                steps=plan.bc_steps_per_drill,
                seed=plan.seed + index,
                opponents=None,
                teacher_actions=True,
                max_episode_steps=min(plan.max_steps, 120 if drill != "full" else 400),
                shaping_gamma=shaping_gamma,
                drill=drill,
                opponent_styles=STYLES,
                teacher_style="pressure" if drill == "melee" else "balanced",
            )
            demos.extend(trajectories)
            consumed.append(["demonstrations", plan.seed + index, drill, plan.bc_steps_per_drill])
            run.record({"stage": "demonstrations", "drill": drill, **summary})
        demonstration_file = _demonstration_file(demos)
        atomic_json(root / "demonstrations.json", demonstration_file)
        demonstrations_sha = hashlib.sha256((root / "demonstrations.json").read_bytes()).hexdigest()
        if plan.start_checkpoint is not None:
            metrics = {"skipped": True, "reason": "preserve supplied policy before PPO"}
            trained = ancestry
        else:
            metrics = behavioral_clone(model, optimizer, demos, plan.bc, seed=plan.seed)
            trained = True
        previous_sha = run.save(
            "warmstart",
            model,
            {
                **base,
                "parent_policy_sha": initial_sha,
                "stage": "warmstart",
                "bc": metrics,
                "simulator_trained": trained,
                "simulator_ancestry": trained,
                "simulator_consumed": consumed,
            },
        )
        run.record({"stage": "imitation", **metrics})
        incumbent = copy.deepcopy(model).eval().requires_grad_(False)
        league, labels = [incumbent], ["warmstart"]
        # New optimizers: supervised moments are not PPO optimizer state.
        optimizer = torch.optim.Adam(model.parameters(), lr=plan.ppo.learning_rate, eps=1e-5)
        rehearsal = torch.optim.Adam(model.parameters(), lr=plan.rehearsal_learning_rate, eps=1e-5)
        first = 0
        _write_stage(
            run,
            0,
            {
                "model": {"label": "warmstart", "sha256": previous_sha},
                "league": labels,
                "simulator_consumed": consumed,
                "demonstrations_sha256": demonstrations_sha,
                "optimizer": optimizer,
                "rehearsal": rehearsal,
            },
        )
    else:
        run.events = list(state["events"])
        run.checkpoints = dict(state["checkpoints"])
        initial = _load(run.path("initial"), run.checkpoints["initial"])
        live, demonstrations = ancestry_consumed(initial)
        base.update(live_consumed=live, demonstrations_consumed=demonstrations)
        label, previous_sha = state["model"]["label"], state["model"]["sha256"]
        model = _load(run.path(label), previous_sha).model
        incumbent = _load(run.path("warmstart"), run.checkpoints["warmstart"]).model
        incumbent.requires_grad_(False)
        labels = list(state["league"])
        league = [
            incumbent
            if name == "warmstart"
            else _load(run.path(name), run.checkpoints[name]).model.requires_grad_(False)
            for name in labels
        ]
        optimizer = torch.optim.Adam(model.parameters(), lr=plan.ppo.learning_rate, eps=1e-5)
        rehearsal = torch.optim.Adam(model.parameters(), lr=plan.rehearsal_learning_rate, eps=1e-5)
        blobs = root / "resume" / f"stage-{state['stage']:03d}"
        optimizer.load_state_dict(_read_blob(blobs, state["optimizers"]["ppo"]))
        rehearsal.load_state_dict(_read_blob(blobs, state["optimizers"]["rehearsal"]))
        demonstrations_sha = state["demonstrations_sha256"]
        demos = _demonstrations_from(json.loads((root / "demonstrations.json").read_text()))
        consumed = list(state["simulator_consumed"])
        torch.set_rng_state(_read_blob(blobs, state["rng"]))
        first = state["stage"]
        atomic_json(root / "progress.json", {"events": run.events, "status": "running"})
    for update in range(first, plan.updates):
        drill = DRILLS[min(3, update * 4 // plan.updates)]
        episodes, rollout = _collect(
            model,
            steps=plan.rollout_steps,
            seed=plan.seed + 1000 + update,
            opponents=league,
            teacher_actions=False,
            max_episode_steps=plan.max_steps,
            shaping_gamma=shaping_gamma,
            drill=drill,
            opponent_styles=STYLES,
            teacher_style="balanced",
        )
        metrics = ppo_update(model, optimizer, episodes, plan.ppo, seed=plan.seed + update)
        if plan.rehearsal_epochs:
            # Original training demonstrations only; evaluation never feeds gradients.
            metrics["rehearsal"] = behavioral_clone(
                model,
                rehearsal,
                demos,
                replace(plan.bc, epochs=plan.rehearsal_epochs),
                seed=plan.seed + 10000 + update,
            )
        consumed = [*consumed, ["rollout", plan.seed + 1000 + update, drill, plan.rollout_steps]]
        label = f"update-{update + 1:03d}"
        previous_sha = run.save(
            label,
            model,
            {
                **base,
                "parent_policy_sha": previous_sha,
                "stage": "ppo",
                "update": update + 1,
                "drill": drill,
                "metrics": metrics,
                "optimizer_state": "resumable-adam-per-run",
                "simulator_trained": True,
                "simulator_ancestry": True,
                "simulator_consumed": consumed,
            },
        )
        run.record({"stage": "ppo", "update": update + 1, "drill": drill, **rollout, **metrics})
        if (update + 1) % 2 == 0:
            league.append(copy.deepcopy(model).eval().requires_grad_(False))
            labels.append(label)
            league, labels = [incumbent] + league[-3:], ["warmstart"] + labels[-3:]
        _write_stage(
            run,
            update + 1,
            {
                "model": {"label": label, "sha256": previous_sha},
                "league": labels,
                "simulator_consumed": consumed,
                "demonstrations_sha256": demonstrations_sha,
                "optimizer": optimizer,
                "rehearsal": rehearsal,
            },
        )
    return _evaluate(run, previous)


def _gate(plan: LeaguePlan, shas: dict, rows: dict, before: str, after: str) -> dict:
    incumbent, candidate = shas[before], shas[after]
    if incumbent == candidate:
        # Identical weights give identical evidence, which never passes the gate.
        return {
            "passed": False,
            "decision": "retain_incumbent",
            "reasons": ["candidate and incumbent are the same weights"],
            "simulator_only": True,
            "live_gain_established": False,
        }
    evaluation_plan = _evaluation_plan(plan, incumbent, candidate)
    return simulator_gate(
        evaluation_plan,
        simulator_record(evaluation_plan, incumbent, rows[before]),
        simulator_record(evaluation_plan, candidate, rows[after]),
    )


def _select(bc_passed: bool, final_passed: bool, *, cloned: bool) -> tuple[str, dict]:
    """The source selection, labelled in the unit-9 vocabulary.

    The base is the warm start if the BC gate passed, else the initial policy; the
    candidate is selected only if it passed the gate against that base.
    """
    base = "warmstart" if bc_passed else "initial"
    selected = "candidate" if final_passed else base
    selection = {}
    for label in ("initial", "warmstart", "candidate"):
        if label == selected:
            selection[label] = "registry_selected"
        elif label == base:
            selection[label] = "rollback"  # superseded by a passing candidate, kept for return
        elif (label == "warmstart" and cloned) or label == "candidate":
            selection[label] = "rejected"
        else:
            selection[label] = "not_gated"
    return selected, selection


def _evaluate(run: _Run, previous: Path | None) -> dict:
    plan = run.plan
    final = f"update-{plan.updates:03d}"
    shas = {
        "initial": run.checkpoints["initial"],
        "warmstart": run.checkpoints["warmstart"],
        "candidate": run.checkpoints[final],
    }
    paths = {"initial": "initial", "warmstart": "warmstart", "candidate": f"snapshots/{final}"}
    rows = {}
    # Used once after training: acceptance seeds never feed gradients or scheduling.
    for label in ("initial", "warmstart", "candidate"):
        model = _load(run.root / paths[label], shas[label]).model
        rows[label] = _suite_rows(
            model,
            styles=STYLES,
            games_per_style=plan.evaluation_games_per_style,
            seed=plan.evaluation_seed,
            max_steps=plan.max_steps,
            sampling=plan.evaluation_sampling,
        )
        report = {
            "domain": SIM_DOMAIN,
            "policy_sha": shas[label],
            "sampling": plan.evaluation_sampling,
            "rows": rows[label],
            "max_steps": plan.max_steps,
            "games": len(rows[label]),
            "score": sum(r["score"] for r in rows[label]) / len(rows[label]),
            "draws": sum(r["score"] == 0.5 for r in rows[label]),
            "real_server_strength_established": False,
        }
        atomic_json(run.root / f"evaluation-{label}.json", report)
        run.record({"stage": "evaluation", "model": label, "score": report["score"]})
    bc_gate = _gate(plan, shas, rows, "initial", "warmstart")
    ppo_gate = _gate(plan, shas, rows, "warmstart", "candidate")
    base = "warmstart" if bc_gate["passed"] else "initial"
    final_gate = _gate(plan, shas, rows, base, "candidate")
    selected, selection = _select(
        bc_gate["passed"], final_gate["passed"], cloned=plan.start_checkpoint is None
    )
    registry = {
        "schema": REGISTRY_SCHEMA,
        "status": "complete",
        "plan_sha256": plan.sha256,
        "bc_gate": bc_gate,
        "ppo_gate": ppo_gate,
        "selection_gate": final_gate,
        "selected": selected,
        "selected_sha": shas[selected],
        "checkpoints": {
            label: {
                "path": paths[label],
                "sha256": shas[label],
                "selection": selection[label],
                "deployment": "not_deployed",
            }
            for label in ("initial", "warmstart", "candidate")
        },
        "registry_kind": "run-local simulator registry",
        "selection_scope": "simulator-experimental-only",
        "simulator_only": True,
        "live_gain_established": False,
        "public_deployment": False,
        "automatic_promotion": False,
        "real_server_strength_established": False,
        "training_steps": plan.updates * plan.rollout_steps,
        "demonstration_steps": plan.bc_steps_per_drill * len(DRILLS),
    }
    atomic_json(run.root / "registry.json", registry)
    atomic_json(
        run.root / "progress.json",
        {"events": run.events, **registry, "resumed_from": str(previous) if previous else None},
    )
    return registry
