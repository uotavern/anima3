"""Public UO observation tracker and clock-driven, cursor-safe option executor.

Neural inference chooses an option only while idle.  Once a spell or explosion
potion starts, this module owns its cursor and fuse without waiting for a model.
Every wire command and forced intervention is exposed for trajectory auditing.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from anima3.contract import (
    DIRECTION_DELTAS,
    Observation,
    Pos,
    chebyshev,
    direction_toward,
    target_ground,
    target_object,
    use,
    use_skill,
    walk,
)
from anima3.magic import SPELLS, VERDICTS, WORDS

from .schema import ACTION_INDEX, ACTION_NAMES, REAGENT_GRAPHICS, CombatView, Frame, encode

POTION_GRAPHICS = {"heal_potion": 0x0F0C, "cure_potion": 0x0F07, "explosion_potion": 0x0F0D}
FUSE_SECONDS = 3.75
SPELL_RECOVERY = 0.75
STATE_FRESH_SECONDS = 3.0
EXECUTOR_VERSION = "pre-aos-options-v4"
CAST_VERDICTS = {**VERDICTS, 500641: "concentration disturbed"}


def cast_seconds(name: str) -> float:
    """ServUO MagerySpell.GetCastDelay override, minus Spell.Cast's 100 ms."""
    return 0.5 + 0.25 * (SPELLS[name].circle - 1) - 0.1


def _cursor(obs: Observation, flag: int, kind: int) -> bool:
    return bool(
        obs.pending_target and obs.target_cursor_flag == flag and obs.target_cursor_type == kind
    )


@dataclass
class Procedure:
    action: int
    name: str
    started_at: float
    stage: str = "new"
    deadline: float = 0.0
    stage_at: float = 0.0
    spell: str = ""
    cast_at: float = 0.0
    cursor_at: float = 0.0
    target_at: float = 0.0
    mana_before: int = 0
    potion_serial: int = 0
    primed_at: float | None = None
    throw_at: float | None = None
    finisher: str = ""
    explosion_released: bool = False
    forced: bool = False
    failure: str = ""
    throw_target: dict | None = None
    last_cursor_id: int | None = None


class CombatExecutor:
    """One player's closed-loop executor. All supplied times are monotonic seconds.

    ``observe`` accepts only *new* DuelState updates (or one with an explicit
    ``_received_at`` monotonic timestamp). Cached server state need not be passed
    again. Opponent serial is pinned at a confirmed round boundary via ``reset``.
    ``start`` rechecks the latest mask and schedules an option; ``tick`` produces
    its wire packets. Always tick active options, including during a temporary
    DuelState gap, so a primed potion is not abandoned in the backpack.
    """

    def __init__(self, opponent_serial: int, emit: Callable[[dict], None] | None = None):
        self.opponent_serial = int(opponent_serial)
        self.emit = emit
        self.epoch = ""
        self._events: list[dict] = []
        self.procedure: Procedure | None = None
        self._state: dict = {}
        self._state_ref: dict | None = None
        self._state_at = -math.inf
        self._started_at: float | None = None
        self._last_observation: Observation | None = None
        self._last_cursor: tuple | None = None
        self._orphan_cursor: tuple | None = None
        self._orphan_attempts = 0
        self._orphan_sent_at = -math.inf
        self._observed_at = 0.0
        self._last_hp: float | None = None
        self._last_enemy_hp: float | None = None
        self._damage: deque[tuple[float, float, float]] = deque()
        self._opponent_cast = ""
        self._opponent_spell_at = -math.inf
        self._verdict_at = -math.inf
        self._verdict = ""
        self._los_refused_until = 0.0
        self._recovery_until = 0.0
        self._cooldowns = {name: 0.0 for name in (*POTION_GRAPHICS, "burst")}
        self._last_action = "hold"
        self._last_view = CombatView(ready=False, fresh=False)
        self._move_directions: dict[str, int] = {}

    @property
    def busy(self) -> bool:
        return self.procedure is not None

    @property
    def view(self) -> CombatView:
        return self._last_view

    def _event(self, kind: str, now: float, **fields: Any) -> None:
        row = {
            "type": kind,
            "at": now,
            "epoch": self.epoch,
            "executor_version": EXECUTOR_VERSION,
            **fields,
        }
        self._events.append(row)
        if self.emit:
            self.emit(row)

    def drain_events(self) -> list[dict]:
        rows, self._events = self._events, []
        return rows

    def reset(
        self, opponent_serial: int | None = None, epoch: str = "", now: float | None = None
    ) -> None:
        now = time.monotonic() if now is None else now
        if self.procedure:
            self._finish(now, False, "confirmed round boundary", forced=True)
        self.opponent_serial = (
            self.opponent_serial if opponent_serial is None else int(opponent_serial)
        )
        self.epoch = str(epoch)
        self._state, self._state_ref = {}, None
        self._state_at = -math.inf
        self._started_at = now
        self._last_observation = None
        self._last_cursor = None
        self._orphan_cursor = None
        self._orphan_attempts = 0
        self._orphan_sent_at = -math.inf
        self._last_hp = self._last_enemy_hp = None
        self._damage.clear()
        self._opponent_cast = ""
        self._opponent_spell_at = self._verdict_at = -math.inf
        self._los_refused_until = self._recovery_until = 0.0
        self._cooldowns = {name: 0.0 for name in (*POTION_GRAPHICS, "burst")}
        self._last_action = "hold"
        self._event("executor_reset", now, opponent=self.opponent_serial)

    def _opponent(self, obs: Observation):
        return next(
            (m for m in obs.mobiles if m.serial == self.opponent_serial and not m.hidden), None
        )

    def _ingest(self, obs: Observation, now: float) -> None:
        if obs is self._last_observation:
            return
        self._last_observation, self._observed_at = obs, now
        current_cursor = (
            obs.pending_target,
            obs.target_cursor_id,
            obs.target_cursor_flag,
            obs.target_cursor_type,
        )
        if current_cursor != self._last_cursor:
            self._last_cursor = current_cursor
            self._event(
                "cursor_observed",
                now,
                pending=current_cursor[0],
                cursor_id=current_cursor[1],
                flag=current_cursor[2],
                target_type=current_cursor[3],
            )
        enemy = self._opponent(obs)
        received = max(0.0, (self._last_hp or obs.player.hits) - obs.player.hits)
        dealt = 0.0
        if enemy is not None and enemy.hits_max > 0:
            # Health-bar units can be normalized by the game protocol. Preserve
            # observed deltas; authoritative attribution is replay-only.
            if self._last_enemy_hp is not None:
                dealt = max(0.0, self._last_enemy_hp - enemy.hits)
            self._last_enemy_hp = enemy.hits
        else:
            self._last_enemy_hp = None
        self._last_hp = obs.player.hits
        if received or dealt:
            self._damage.append((now, dealt, received))
        while self._damage and now - self._damage[0][0] > 2:
            self._damage.popleft()
        for journal in obs.new_journal:
            words = journal.text.casefold().strip()
            if journal.serial == self.opponent_serial and words in WORDS:
                self._opponent_cast = WORDS[words].casefold().replace(" ", "_")
                self._opponent_spell_at = now
            if journal.cliloc in CAST_VERDICTS:
                self._verdict_at, self._verdict = now, CAST_VERDICTS[journal.cliloc]
                self._event("spell_verdict", now, cliloc=journal.cliloc, verdict=self._verdict)
                if "line of sight" in self._verdict:
                    self._los_refused_until = now + 1.0

    def _movements(self, obs: Observation) -> dict[str, int]:
        enemy = self._opponent(obs)
        if enemy is None or obs.terrain is None or obs.player.paralyzed or obs.player.stam <= 0:
            return {}
        p = obs.player.pos
        preferred = direction_toward(p, enemy.pos)
        wants = {
            "approach": preferred,
            "retreat": (preferred + 4) % 8,
            "strafe_left": (preferred - 2) % 8,
            "strafe_right": (preferred + 2) % 8,
        }
        occupied = {(mobile.pos.x, mobile.pos.y) for mobile in obs.mobiles if not mobile.hidden}
        valid = []
        for direction, (dx, dy) in enumerate(DIRECTION_DELTAS):
            x, y = p.x + dx, p.y + dy
            if obs.terrain.walkable(x, y) is not True or (x, y) in occupied:
                continue
            # Conservative corner check avoids squeezing through two walls.
            if (
                dx
                and dy
                and (
                    obs.terrain.walkable(p.x + dx, p.y) is not True
                    or obs.terrain.walkable(p.x, p.y + dy) is not True
                )
            ):
                continue
            valid.append(direction)
        result = {}
        distance = chebyshev(p, enemy.pos)
        for name, want in wants.items():
            candidates = [
                direction
                for direction in valid
                if min((direction - want) % 8, (want - direction) % 8) <= 1
            ]
            if name in ("approach", "retreat"):
                candidates = [
                    direction
                    for direction in candidates
                    if (
                        chebyshev(
                            Pos(
                                p.x + DIRECTION_DELTAS[direction][0],
                                p.y + DIRECTION_DELTAS[direction][1],
                            ),
                            enemy.pos,
                        )
                        < distance
                        if name == "approach"
                        else chebyshev(
                            Pos(
                                p.x + DIRECTION_DELTAS[direction][0],
                                p.y + DIRECTION_DELTAS[direction][1],
                            ),
                            enemy.pos,
                        )
                        > distance
                    )
                ]
            if candidates:
                result[name] = min(
                    candidates,
                    key=lambda direction: min((direction - want) % 8, (want - direction) % 8),
                )
        return result

    def observe(
        self, obs: Observation, server_state: dict | None = None, now: float | None = None
    ) -> Frame:
        now = time.monotonic() if now is None else now
        if self._started_at is None:
            self._started_at = now
        self._ingest(obs, now)
        if server_state is not None and (
            server_state is not self._state_ref or "_received_at" in server_state
        ):
            self._state = dict(server_state)
            self._state_ref = server_state
            self._state_at = float(server_state.get("_received_at", now))
        enemy, p = self._opponent(obs), obs.player
        self._move_directions = self._movements(obs)
        counts = {
            graphic: sum(item.amount for item in obs.own_pack() if item.graphic == graphic)
            for graphic in REAGENT_GRAPHICS
        }
        potions = {
            name: sum(item.amount for item in obs.own_pack() if item.graphic == graphic)
            for name, graphic in POTION_GRAPHICS.items()
        }
        proc = self.procedure
        cast_remaining = 0.0
        if proc and proc.spell and proc.stage == "spell_wait":
            cast_remaining = max(0.01, proc.cast_at + cast_seconds(proc.spell) - now)
        fresh = 0 <= now - self._state_at < STATE_FRESH_SECONDS
        ready = (
            self._state.get("phase") == "Fighting"
            and self._state.get("opponent") == self.opponent_serial
            and not p.dead
            and not obs.pending_target
        )
        # Estimate only from public power words, matching the simulator's view.
        # A heard cast can still fizzle, be interrupted, or target somebody else;
        # this is deliberately not an authoritative incoming-damage queue.
        incoming_explosion = -1.0
        if self._opponent_cast == "explosion":
            estimate = cast_seconds("explosion") + 2.5 - (now - self._opponent_spell_at)
            if estimate >= 0:
                incoming_explosion = estimate
        self._last_view = CombatView(
            hp=p.hits,
            hp_max=p.hits_max or 100,
            mana=p.mana,
            mana_max=p.mana_max or 100,
            stamina=p.stam,
            stamina_max=p.stam_max or 100,
            opponent_hp=enemy.hits if enemy else 0,
            opponent_hp_max=(enemy.hits_max or 100) if enemy else 100,
            opponent_health_known=bool(enemy and enemy.hits_max > 0),
            dx=enemy.pos.x - p.pos.x if enemy else 0,
            dy=enemy.pos.y - p.pos.y if enemy else 0,
            distance=chebyshev(p.pos, enemy.pos) if enemy else 99,
            poisoned=p.poisoned,
            opponent_poisoned=bool(enemy and enemy.poisoned),
            paralyzed=p.paralyzed,
            opponent_paralyzed=bool(enemy and enemy.paralyzed),
            cast_remaining=cast_remaining,
            busy_remaining=max(0.01, proc.deadline - now) if proc else 0,
            recovery_remaining=max(0.0, self._recovery_until - now),
            heal_potion_cooldown=max(0.0, self._cooldowns["heal_potion"] - now),
            cure_potion_cooldown=max(0.0, self._cooldowns["cure_potion"] - now),
            explosion_potion_cooldown=max(0.0, self._cooldowns["explosion_potion"] - now),
            burst_cooldown=max(0.0, self._cooldowns["burst"] - now),
            heal_potions=potions["heal_potion"],
            cure_potions=potions["cure_potion"],
            explosion_potions=potions["explosion_potion"],
            reagent_counts=tuple(counts[graphic] for graphic in REAGENT_GRAPHICS),
            elapsed=max(0.0, now - self._started_at),
            showdown=bool(self._state.get("showdown")),
            recent_damage_dealt=sum(row[1] for row in self._damage if now - row[0] <= 2),
            recent_damage_received=sum(row[2] for row in self._damage if now - row[0] <= 2),
            # Damage from one's own potion cannot be reliably attributed by the
            # client health bar; do not invent this feature from a coincidence.
            recent_self_damage=0,
            opponent_cast=self._opponent_cast if now - self._opponent_spell_at <= 10 else "",
            opponent_spell_age=min(10.0, max(0.0, now - self._opponent_spell_at)),
            opponent_visible=enemy is not None,
            line_of_sight=bool(
                enemy and abs(enemy.pos.z - p.pos.z) <= 16 and now >= self._los_refused_until
            ),
            fresh=fresh,
            ready=ready,
            move_legal=tuple(name in self._move_directions for name in ACTION_NAMES[1:5]),
            potion_fuse=max(0.0, proc.primed_at + FUSE_SECONDS - now)
            if proc and proc.primed_at is not None and proc.throw_at is None
            else 0,
            incoming_explosion_seconds=incoming_explosion,
            last_action=self._last_action,
        )
        return encode(self._last_view)

    def start(self, action: int | str, obs: Observation, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        action = ACTION_INDEX[action] if isinstance(action, str) else int(action)
        if not 0 <= action < len(ACTION_NAMES):
            raise ValueError("unknown neural combat action")
        frame = self.observe(obs, now=now)
        if (
            self.busy
            or not frame.mask[action]
            or not self._last_view.fresh
            or not self._last_view.ready
        ):
            self._event(
                "action_rejected",
                now,
                action=action,
                name=ACTION_NAMES[action],
                reason="latest mask or combat context",
            )
            return False
        name = ACTION_NAMES[action]
        self.procedure = Procedure(action, name, now, deadline=now + 8.0, stage_at=now)
        self._last_action = name
        self._event(
            "action_started", now, action=action, name=name, observation_at=self._observed_at
        )
        return True

    def _wire(self, packet: dict, now: float) -> list[dict]:
        proc = self.procedure
        self._event(
            "wire_action",
            now,
            action=proc.action if proc else None,
            stage=proc.stage if proc else "idle",
            packet=packet,
            observation_at=self._observed_at,
            action_elapsed=now - proc.started_at if proc else 0,
        )
        return [packet]

    def _stage(self, stage: str, now: float, deadline: float | None = None) -> None:
        assert self.procedure is not None
        self.procedure.stage, self.procedure.stage_at = stage, now
        if deadline is not None:
            self.procedure.deadline = deadline
        self._event("executor_stage", now, action=self.procedure.action, stage=stage)

    def _intervene(self, now: float, reason: str) -> None:
        proc = self.procedure
        if proc:
            if not proc.forced:
                self._event(
                    "safety_intervention",
                    now,
                    action=proc.action,
                    reason=reason,
                    intervention=True,
                    scope="option_execution",
                    policy_override=False,
                )
            proc.forced = True
            proc.failure = proc.failure or reason

    def _finish(self, now: float, success: bool, reason: str, forced: bool = False) -> None:
        proc = self.procedure
        if not proc:
            return
        if forced:
            self._intervene(now, reason)
        self._event(
            "action_finished",
            now,
            action=proc.action,
            name=proc.name,
            success=success,
            reason=reason,
            duration=now - proc.started_at,
            intervention=proc.forced,
            scope="option_execution",
            policy_override=False,
            cast_at=proc.cast_at or None,
            target_at=proc.target_at or None,
            potion_prime_at=proc.primed_at,
            potion_throw_at=proc.throw_at,
        )
        self.procedure = None

    def _item(self, obs: Observation, name: str) -> int:
        return next(
            (
                item.serial
                for item in obs.own_pack()
                if item.graphic == POTION_GRAPHICS[name] and item.amount > 0
            ),
            0,
        )

    def _safe_throw_target(self, obs: Observation) -> dict | None:
        enemy = self._opponent(obs)
        p = obs.player.pos
        if (
            enemy
            and 3 <= chebyshev(p, enemy.pos) <= 12
            and abs(enemy.pos.z - p.z) <= 16
            and self._state.get("phase") == "Fighting"
            and self._state.get("opponent") == self.opponent_serial
        ):
            return target_object(self.opponent_serial)
        # Dispose a lit potion away from every currently visible character.
        # Use client-known walkable terrain; a guessed tile behind a wall is not
        # a reliable throw destination. Target refusal remains a visible failure.
        candidates: list[tuple[int, int]] = []
        if obs.terrain:
            for radius in (6, 7, 8, 9, 10):
                for dx, dy in DIRECTION_DELTAS:
                    x, y = p.x + dx * radius, p.y + dy * radius
                    if obs.terrain.walkable(x, y) is not True:
                        continue
                    if any(
                        obs.terrain.walkable(p.x + dx * step, p.y + dy * step) is not True
                        for step in range(1, radius)
                    ):
                        continue
                    if any(max(abs(m.pos.x - x), abs(m.pos.y - y)) <= 3 for m in obs.mobiles):
                        continue
                    candidates.append((x, y))
        if not candidates:
            return None
        x, y = candidates[0]
        return target_ground(x, y, p.z)

    def _begin_throw(self, obs: Observation, now: float, reason: str = "") -> list[dict]:
        proc = self.procedure
        assert proc is not None and proc.primed_at is not None
        if reason:
            self._intervene(now, reason)
        if _cursor(obs, 1, 0) or _cursor(obs, 2, 0):
            self._stage("throw_reopen", now, proc.primed_at + 2.7)
            return self._wire({"type": "TargetCancel"}, now)
        if _cursor(obs, 0, 1):
            self._stage("potion_cook", now, proc.primed_at + 2.7)
            return self._throw(obs, now)
        self._stage("potion_wait", now, proc.primed_at + 2.7)
        return self._wire(use(proc.potion_serial), now)

    def _throw(self, obs: Observation, now: float) -> list[dict]:
        proc = self.procedure
        assert proc is not None and proc.primed_at is not None
        if not _cursor(obs, 0, 1):
            return self._begin_throw(obs, now, "potion cursor replaced")
        target = self._safe_throw_target(obs)
        if target is None:
            self._intervene(now, "no safe potion destination")
            # Keep attempting disposal before fuse expiry rather than pretending
            # the potion was thrown. Never aim a lit potion at the player.
            if now >= proc.primed_at + FUSE_SECONDS + 0.25:
                self._finish(now, False, "potion expired without safe destination", forced=True)
            return []
        if target["type"] == "TargetGround":
            self._intervene(now, "opponent unavailable or inside blast radius")
        proc.throw_target, proc.throw_at = target, now
        proc.last_cursor_id = obs.target_cursor_id
        self._stage("throw_confirm", now, now + 0.8)
        return self._wire(target, now)

    def _begin_cast(self, name: str, obs: Observation, now: float) -> list[dict]:
        proc = self.procedure
        assert proc is not None
        proc.spell, proc.cast_at, proc.mana_before = name, now, obs.player.mana
        proc.cursor_at, proc.target_at = 0.0, 0.0
        self._stage("spell_wait", now, now + cast_seconds(name) + 2.0)
        return self._wire({"type": "CastSpell", "spell": SPELLS[name].id}, now)

    def _cleanup_orphan_spell_cursor(self, obs: Observation, now: float) -> list[dict]:
        """Cancel a late spell cursor after its owning procedure timed out.

        A server/network delay can deliver the target after the local deadline.
        Leaving that cursor open would mask every subsequent action until the
        server's 30-second target timeout. Neutral potion cursors are excluded:
        a lit potion must be thrown, never blindly dismissed by this cleanup.
        """
        if not obs.pending_target:
            self._orphan_cursor = None
            self._orphan_attempts = 0
            return []
        if not (_cursor(obs, 1, 0) or _cursor(obs, 2, 0)):
            return []
        if (
            obs.player.dead
            or self._state.get("phase") != "Fighting"
            or self._state.get("opponent") != self.opponent_serial
            or not 0 <= now - self._state_at < STATE_FRESH_SECONDS
        ):
            return []
        key = (obs.target_cursor_id, obs.target_cursor_flag, obs.target_cursor_type)
        if key != self._orphan_cursor:
            self._orphan_cursor, self._orphan_attempts = key, 0
        if self._orphan_attempts >= 2 or (
            self._orphan_attempts and now - self._orphan_sent_at < 0.5
        ):
            return []
        self._orphan_attempts += 1
        self._orphan_sent_at = now
        self._event(
            "orphan_cursor_cleanup",
            now,
            cursor_id=obs.target_cursor_id,
            flag=obs.target_cursor_flag,
            attempt=self._orphan_attempts,
            reason="late spell cursor without active procedure",
            intervention=True,
            scope="option_execution",
            policy_override=False,
        )
        return self._wire({"type": "TargetCancel"}, now)

    def tick(self, obs: Observation, now: float | None = None) -> list[dict]:
        now = time.monotonic() if now is None else now
        self._ingest(obs, now)
        proc = self.procedure
        if proc is None:
            return self._cleanup_orphan_spell_cursor(obs, now)
        if obs.player.dead:
            self._finish(now, False, "player died")
            return []
        # A state update announcing round end is authoritative; a missing update
        # is not. Continue owned cursors and fuses during a brief state gap.
        ended = (
            self._state.get("phase") != "Fighting"
            or self._state.get("opponent") != self.opponent_serial
        )
        if ended and proc.stage not in (
            "throw_reopen",
            "potion_wait",
            "potion_cook",
            "throw_confirm",
        ):
            if proc.primed_at is not None and proc.throw_at is None:
                return self._begin_throw(obs, now, "combat context ended with lit potion")
            packets = self._wire({"type": "TargetCancel"}, now) if obs.pending_target else []
            self._finish(now, False, "combat context ended", forced=True)
            return packets
        if proc.stage == "new":
            if proc.name == "hold":
                self._stage("simple_wait", now, now + 0.15)
            elif proc.name in ACTION_NAMES[1:5]:
                direction = self._movements(obs).get(proc.name)
                if direction is None:
                    self._finish(now, False, "movement blocked before send")
                    return []
                self._stage("simple_wait", now, now + 0.18)
                return self._wire(walk(direction, run=True), now)
            elif proc.name == "meditate":
                self._stage("simple_wait", now, now + 0.8)
                return self._wire(use_skill(46), now)
            elif proc.name in ("heal_potion", "cure_potion"):
                serial = self._item(obs, proc.name)
                if not serial:
                    self._finish(now, False, "potion missing")
                    return []
                self._cooldowns[proc.name] = now + (10 if proc.name == "heal_potion" else 1)
                self._stage("simple_wait", now, now + 0.25)
                return self._wire(use(serial), now)
            elif proc.name == "explosion_potion":
                proc.potion_serial = self._item(obs, "explosion_potion")
                if not proc.potion_serial:
                    self._finish(now, False, "potion missing")
                    return []
                proc.primed_at = now
                self._cooldowns["explosion_potion"] = now + 10
                self._stage("potion_wait", now, now + 1.0)
                return self._wire(use(proc.potion_serial), now)
            elif proc.name in ("burst_bolt", "burst_flame"):
                proc.potion_serial = self._item(obs, "explosion_potion")
                if not proc.potion_serial:
                    self._finish(now, False, "potion missing")
                    return []
                proc.finisher = "energy_bolt" if proc.name == "burst_bolt" else "flamestrike"
                self._cooldowns["burst"] = now + 20
                return self._begin_cast("explosion", obs, now)
            else:
                return self._begin_cast(proc.name, obs, now)
            return []
        if proc.stage == "simple_wait":
            if now >= proc.deadline:
                self._finish(now, True, "command completed")
            return []
        if proc.stage == "spell_wait":
            failure = self._verdict if self._verdict_at > proc.cast_at else ""
            if failure:
                if proc.primed_at is not None and proc.throw_at is None:
                    return self._begin_throw(obs, now, "interrupted spell: " + failure)
                self._finish(now, False, failure)
                return []
            self_target = SPELLS[proc.spell].kind in ("heal", "cure", "buff")
            if _cursor(obs, 2 if self_target else 1, 0):
                proc.cursor_at = now
                self._recovery_until = max(self._recovery_until, now + SPELL_RECOVERY)
                if proc.finisher and proc.spell == "explosion" and proc.primed_at is not None:
                    self._stage("spell_hold", now, min(now + 0.65, proc.primed_at + 2.0))
                    return []
                if proc.finisher and proc.spell == "explosion":
                    self._intervene(now, "spell cursor arrived before safe potion prime")
                    proc.finisher = ""
                proc.target_at = now
                self._stage("spell_target_wait", now, now + 1.2)
                return self._wire(
                    target_object(obs.player.serial if self_target else self.opponent_serial), now
                )
            if (
                proc.finisher
                and proc.spell == "explosion"
                and proc.primed_at is None
                and now - proc.cast_at >= 0.85
            ):
                if obs.pending_target or obs.player.hp_pct < 0.45 or obs.player.poisoned:
                    self._finish(now, False, "unsafe to prime burst", forced=True)
                    return self._wire({"type": "TargetCancel"}, now) if obs.pending_target else []
                proc.primed_at = now
                self._cooldowns["explosion_potion"] = now + 10
                self._event("potion_primed", now, action=proc.action, serial=proc.potion_serial)
                return self._wire(use(proc.potion_serial), now)
            if proc.primed_at is not None and proc.throw_at is None:
                enemy = self._opponent(obs)
                unsafe = (
                    obs.player.hp_pct < 0.45
                    or enemy is None
                    or chebyshev(obs.player.pos, enemy.pos) < 3
                )
                if unsafe or now - proc.primed_at >= 2.0:
                    return self._begin_throw(
                        obs,
                        now,
                        "burst safety release"
                        if unsafe
                        else "spell cursor timeout with lit potion",
                    )
            if now >= proc.deadline:
                packets = self._wire({"type": "TargetCancel"}, now) if obs.pending_target else []
                self._finish(now, False, "spell cursor timeout", forced=bool(obs.pending_target))
                return packets
            return []
        if proc.stage == "spell_hold":
            enemy = self._opponent(obs)
            if (
                self._verdict_at > proc.cast_at
                or obs.player.hp_pct < 0.45
                or enemy is None
                or chebyshev(obs.player.pos, enemy.pos) < 3
            ):
                return self._begin_throw(obs, now, "burst interrupted while holding spell")
            if now < proc.deadline:
                return []
            if not _cursor(obs, 1, 0):
                return self._begin_throw(obs, now, "spell cursor replaced before target")
            proc.target_at, proc.explosion_released = now, True
            self._stage("spell_target_wait", now, now + 0.45)
            return self._wire(target_object(self.opponent_serial), now)
        if proc.stage == "spell_target_wait":
            failure = self._verdict if self._verdict_at > proc.target_at else ""
            if proc.primed_at is not None and proc.throw_at is None:
                # Release Explosion's cursor before reusing the same potion.
                if obs.pending_target and now - proc.target_at < 0.2:
                    return []
                return self._begin_throw(obs, now, failure)
            if failure:
                self._finish(now, False, failure)
                return []
            if not obs.pending_target and obs.player.mana < proc.mana_before:
                self._stage("recovery", now, self._recovery_until)
            elif now >= proc.deadline:
                if obs.pending_target:
                    packets = self._wire({"type": "TargetCancel"}, now)
                    self._finish(now, False, "target not acknowledged", forced=True)
                    return packets
                self._finish(now, False, "spell expenditure unconfirmed")
            return []
        if proc.stage == "throw_reopen":
            if obs.pending_target and now - proc.stage_at < 0.1:
                return []
            self._stage(
                "potion_wait", now, (proc.primed_at if proc.primed_at is not None else now) + 2.7
            )
            return self._wire(use(proc.potion_serial), now)
        if proc.stage == "potion_wait":
            if _cursor(obs, 0, 1):
                self._stage(
                    "potion_cook",
                    now,
                    (proc.primed_at if proc.primed_at is not None else now) + 2.1,
                )
                if proc.finisher or proc.forced:
                    return self._throw(obs, now)
            elif now >= proc.deadline:
                # One retry still leaves a full second of fuse margin. A missing
                # cursor is not evidence that the potion was thrown.
                if proc.primed_at is not None and now - proc.primed_at < 2.3:
                    self._intervene(now, "potion cursor not observed; reopen")
                    self._stage("potion_wait", now, proc.primed_at + 2.6)
                    return self._wire(use(proc.potion_serial), now)
                if proc.primed_at is not None and now >= proc.primed_at + FUSE_SECONDS + 0.25:
                    self._finish(now, False, "potion cursor never confirmed", forced=True)
            return []
        if proc.stage == "potion_cook":
            enemy = self._opponent(obs)
            unsafe = (
                enemy is None
                or chebyshev(obs.player.pos, enemy.pos) < 3
                or obs.player.hp_pct < 0.4
                or ended
            )
            if unsafe:
                self._intervene(now, "potion emergency release")
            if unsafe or proc.forced or proc.finisher or now >= proc.deadline:
                return self._throw(obs, now)
            return []
        if proc.stage == "throw_confirm":
            if not obs.pending_target:
                if proc.finisher and proc.explosion_released and not proc.forced:
                    self._stage("finisher_recovery", now, self._recovery_until)
                else:
                    self._finish(now, True, "potion target acknowledged")
                return []
            if now >= proc.deadline:
                # Never report a still-open target as a successful throw.
                self._intervene(now, "potion throw unconfirmed")
                if proc.primed_at is not None and now - proc.primed_at < 2.8:
                    proc.throw_at = None
                    return self._begin_throw(obs, now)
                self._finish(now, False, "potion throw unconfirmed", forced=True)
            return []
        if proc.stage == "finisher_recovery":
            if now < proc.deadline:
                return []
            enemy = self._opponent(obs)
            if (
                obs.pending_target
                or enemy is None
                or chebyshev(obs.player.pos, enemy.pos) > 12
                or obs.player.hp_pct < 0.45
                or obs.player.poisoned
                or obs.player.mana < SPELLS[proc.finisher].mana
            ):
                self._finish(now, False, "finisher no longer safe or affordable", forced=True)
                return []
            return self._begin_cast(proc.finisher, obs, now)
        if proc.stage == "recovery":
            if now >= proc.deadline:
                self._finish(now, True, "spell target and mana acknowledged")
            return []
        self._finish(now, False, "unknown executor stage", forced=True)
        return []
