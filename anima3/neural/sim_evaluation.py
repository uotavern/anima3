"""Seeded simulator evaluation through the unit-7 gate, labelled as simulator only.

Stdlib only; importing this module never imports torch, trains or promotes. The
rollouts themselves are in ``sim_rollouts.py`` (which needs torch).

- A :class:`SimulatorEvaluationPlan` wraps a unit-7 :class:`EvaluationPlan` whose
  domain is ``approximate-pre-aos-simulator`` and fixes, before any game is played,
  the held-out seed base, the episode budget ``max_steps``, the policy sampling and
  the code identities of the simulator (``SIMULATOR_SHA256``) and of the teacher
  (``TEACHER_SHA256``). None of them has a default. Its SHA-256 identifies it.
- Opponents are teacher styles, labelled ``teacher:<style>:<teacher sha256>``. Only
  a simulator plan may name one; a live plan still refuses a teacher opponent.
- Fixtures follow the source ``league.evaluate_suite`` (``819f4f4``): game ``g``
  against the style with index ``i`` in ``TEACHER_STYLES`` uses simulator seed
  ``seed + 10000 * i + g // 2`` and seat ``g % 2``. Games 2k and 2k+1 share a seed
  and swap seats, which is exactly the unit-7 two-seat block. Seat 0 is recorded as
  the ``challenger`` side and seat 1 as ``receiver``.
- Evaluation seeds are at least ``EVALUATION_SEED_FLOOR`` (10**9), as in the
  source. Simulator rollouts for training draw every episode seed below it, so a
  simulator evaluation game is never a training game.
- :func:`simulator_gate` runs the unchanged unit-7 :func:`selection_gate` and then
  relabels the result as a simulator decision with its own schema. Its outcome is
  ``candidate_passed_simulator_gate`` or ``retain_incumbent`` and it carries
  ``simulator_only: true`` and ``live_gain_established: false``. Simulator gain is
  not live gain: ``evaluation.record_decision`` refuses any decision that is not a
  live one, and ``decide_from_collections`` refuses a simulator plan.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .drills import SIM_DOMAIN, SIMULATOR_SHA256
from .evaluation import SIDES, EvaluationPlan, EvaluationRefused, selection_gate
from .inference import require_sha256
from .teacher import TEACHER_SHA256, TEACHER_STYLES

SIM_PLAN_SCHEMA = "anima3-simulator-evaluation-plan-v1"
SIM_RECORD_SCHEMA = "anima3-simulator-evaluation-v1"
SIM_DECISION_SCHEMA = "anima3-simulator-gate-decision-v1"
# Source evaluate_suite refused evaluation seeds below 10**9; training rollouts
# draw episode seeds with ``randrange(10**9)``, so the two never meet.
EVALUATION_SEED_FLOOR = 1_000_000_000
STYLE_SEED_STRIDE = 10_000
MAX_SEED = 2**62
# 1200 steps of 0.25 s is the 300-second server round the source league matched.
MAX_SIM_STEPS = 1200
SAMPLING = ("categorical", "greedy")


class SimulatorRefused(EvaluationRefused):
    """The simulator evidence cannot be evaluated under this plan; nothing was decided."""


def teacher_opponent(style: str, teacher_sha256: str = TEACHER_SHA256) -> str:
    """The simulator plan label of one teacher style."""
    if style not in TEACHER_STYLES:
        raise ValueError(f"unknown teacher style {style!r}")
    return f"teacher:{style}:{require_sha256(teacher_sha256)}"


def _style(label: str) -> str:
    return label.split(":")[1]


@dataclass(frozen=True)
class SimulatorEvaluationPlan:
    """A simulator evaluation fixed before any game: unit-7 plan, seeds, budget, code."""

    evaluation: EvaluationPlan
    seed: int
    max_steps: int
    sampling: str
    simulator_sha256: str
    teacher_sha256: str

    def __post_init__(self):
        if type(self.evaluation) is not EvaluationPlan:
            raise ValueError("a simulator plan wraps a unit-7 EvaluationPlan")
        if self.evaluation.domain != SIM_DOMAIN:
            raise ValueError(f"a simulator plan needs the {SIM_DOMAIN!r} domain")
        seed = self.seed
        if type(seed) is not int or not EVALUATION_SEED_FLOOR <= seed <= MAX_SEED:
            raise ValueError("simulator evaluation needs an explicit held-out seed >= 10**9")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= MAX_SIM_STEPS:
            raise ValueError(f"max_steps must be an integer in 1..{MAX_SIM_STEPS}")
        if self.sampling not in SAMPLING:
            raise ValueError("sampling must be 'categorical' or 'greedy'")
        require_sha256(self.simulator_sha256)
        require_sha256(self.teacher_sha256)
        if any(o.split(":")[2] != self.teacher_sha256 for o in self.evaluation.opponents):
            raise ValueError("every teacher opponent must name the plan's teacher SHA-256")

    def to_json(self) -> dict:
        return {
            "schema": SIM_PLAN_SCHEMA,
            "evaluation": self.evaluation.to_json(),
            "seed": self.seed,
            "max_steps": self.max_steps,
            "sampling": self.sampling,
            "simulator_sha256": self.simulator_sha256,
            "teacher_sha256": self.teacher_sha256,
        }

    @classmethod
    def from_json(cls, value) -> SimulatorEvaluationPlan:
        keys = {"schema", "evaluation", "seed", "max_steps", "sampling"}
        keys |= {"simulator_sha256", "teacher_sha256"}
        if not isinstance(value, dict) or set(value) != keys or value["schema"] != SIM_PLAN_SCHEMA:
            raise ValueError("not a simulator evaluation plan")
        fields = {k: v for k, v in value.items() if k not in ("schema", "evaluation")}
        return cls(evaluation=EvaluationPlan.from_json(value["evaluation"]), **fields)

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def styles(self) -> list[tuple[str, str]]:
        """``(opponent label, teacher style)`` in the plan's canonical order."""
        return [(o, _style(o)) for o in self.evaluation.opponents]

    def fixtures(self, opponent: str) -> list[tuple[int, int, int]]:
        """``(fixture, simulator seed, seat)`` for every planned game against ``opponent``."""
        index = TEACHER_STYLES.index(_style(opponent))
        return [
            (game, self.seed + index * STYLE_SEED_STRIDE + game // 2, game % 2)
            for game in range(self.evaluation.games_per_opponent)
        ]

    def require_current(self) -> None:
        """Refuse a plan made for other simulator or teacher code than is running now."""
        if self.simulator_sha256 != SIMULATOR_SHA256:
            raise SimulatorRefused("the simulator code changed since this plan was made")
        if self.teacher_sha256 != TEACHER_SHA256:
            raise SimulatorRefused("the teacher code changed since this plan was made")


def simulator_record(plan: SimulatorEvaluationPlan, policy_sha: str, games: list[dict]) -> dict:
    """One policy's simulator evaluation, in the unit-7 row format and simulator domain.

    ``games`` are the rollout rows for every planned fixture, in plan order (as
    ``sim_rollouts`` produces them). Each must carry its planned style, seed and
    seat; a missing, extra or reordered game refuses the record.
    """
    if type(plan) is not SimulatorEvaluationPlan:
        raise SimulatorRefused("a SimulatorEvaluationPlan is required")
    policy_sha = require_sha256(policy_sha)
    if policy_sha not in (plan.evaluation.candidate_sha256, plan.evaluation.incumbent_sha256):
        raise SimulatorRefused("policy is neither the plan's candidate nor its incumbent")
    expected = [
        (opponent, style, fixture, seed, side)
        for opponent, style in plan.styles()
        for fixture, seed, side in plan.fixtures(opponent)
    ]
    if not isinstance(games, list) or len(games) != len(expected):
        raise SimulatorRefused("simulator games must cover exactly the planned fixtures")
    rows = []
    for game, (opponent, style, fixture, seed, side) in zip(games, expected, strict=True):
        planned = (style, seed, side)
        if (
            not isinstance(game, dict)
            or (game.get("style"), game.get("seed"), game.get("side")) != planned
        ):
            raise SimulatorRefused("simulator game does not match its planned fixture")
        score = game.get("score")
        if type(score) is not float or score not in (0.0, 0.5, 1.0):
            raise SimulatorRefused("simulator scores must be loss/draw/win")
        rows.append(
            {
                "opponent": opponent,
                "fixture": fixture,
                "status": "completed",
                "score": score,
                "side": SIDES[side],
                # A simulated game is complete evidence of itself once it reached
                # its terminal state (a time limit is a terminal draw).
                "valid": game.get("terminated") is True and game.get("truncated") is False,
                "seed": seed,
                "sim_side": side,
                **{k: game.get(k) for k in ("damage", "self_damage", "actions", "elapsed")},
            }
        )
    return {
        "schema": SIM_RECORD_SCHEMA,
        "domain": SIM_DOMAIN,
        "policy_sha": policy_sha,
        "simulator_plan_sha256": plan.sha256,
        "simulator_sha256": plan.simulator_sha256,
        "teacher_sha256": plan.teacher_sha256,
        "sampling": plan.sampling,
        "max_steps": plan.max_steps,
        "rows": rows,
        "simulator_only": True,
        "real_server_strength_established": False,
    }


def _check_record(plan: SimulatorEvaluationPlan, evaluation, expected_sha: str) -> None:
    if not isinstance(evaluation, dict) or evaluation.get("schema") != SIM_RECORD_SCHEMA:
        raise SimulatorRefused("not a simulator evaluation record")
    if evaluation.get("simulator_plan_sha256") != plan.sha256:
        raise SimulatorRefused("simulator evaluation was made under another plan")
    if (evaluation.get("simulator_sha256"), evaluation.get("teacher_sha256")) != (
        plan.simulator_sha256,
        plan.teacher_sha256,
    ):
        raise SimulatorRefused("simulator evaluation names other simulator or teacher code")
    if evaluation.get("policy_sha") != expected_sha:
        raise SimulatorRefused("simulator evaluation policy SHA-256 differs from the plan")
    planned = {
        (opponent, fixture): (seed, SIDES[side])
        for opponent, _ in plan.styles()
        for fixture, seed, side in plan.fixtures(opponent)
    }
    rows = evaluation.get("rows")
    if not isinstance(rows, list) or len(rows) != len(planned):
        raise SimulatorRefused("simulator evaluation must hold exactly the planned games")
    for row in rows:
        key = (row.get("opponent"), row.get("fixture")) if isinstance(row, dict) else None
        if planned.get(key) != (row.get("seed"), row.get("side")):
            raise SimulatorRefused("simulator row does not match its planned seed and seat")


def simulator_gate(plan: SimulatorEvaluationPlan, incumbent: dict, candidate: dict) -> dict:
    """The unit-7 gate on simulator evidence, returned as a simulator-only decision.

    The paired bootstrap, per-opponent regression limit, block and floor rules are
    those of :func:`~anima3.neural.evaluation.selection_gate`, unchanged. The result
    has its own schema and outcome vocabulary, so it can never be read or recorded
    as a live selection decision.
    """
    if type(plan) is not SimulatorEvaluationPlan:
        raise SimulatorRefused("a SimulatorEvaluationPlan is required")
    plan.require_current()
    _check_record(plan, incumbent, plan.evaluation.incumbent_sha256)
    _check_record(plan, candidate, plan.evaluation.candidate_sha256)
    decision = selection_gate(plan.evaluation, incumbent, candidate)
    passed = decision["passed"]
    decision.update(
        {
            "schema": SIM_DECISION_SCHEMA,
            "decision": "candidate_passed_simulator_gate" if passed else "retain_incumbent",
            "simulator_plan": plan.to_json(),
            "simulator_plan_sha256": plan.sha256,
            "evidence": {
                "simulator": {
                    "simulator_sha256": plan.simulator_sha256,
                    "teacher_sha256": plan.teacher_sha256,
                    "seed": plan.seed,
                    "max_steps": plan.max_steps,
                    "sampling": plan.sampling,
                }
            },
            "selection_scope": "simulator-experimental-only",
            "simulator_only": True,
            "live_gain_established": False,
            "automatic_promotion": False,
            "public_deployment": False,
            "real_server_strength_established": False,
        }
    )
    return decision
