"""Fast *approximate* pre-AOS duel simulator for policy bootstrapping.

This is not ServUO and simulated wins are not evidence of live playing strength.
It models option execution, delayed damage, interruptible casts, finite supplies,
poison, cooldowns, public speech, a 2D bounded arena and a healing-free showdown.
Terrain, network cursors, skill/stat builds, armor and server RNG are simplified.

Reference: this workspace's ServUO MagerySpell.GetCastDelay (pre-AOS override),
Spell.GetCastRecovery, individual spell classes, and BaseExplosionPotion. Sixth
circle casting is 1.65 s before transport, NOT the AOS base formula's 2.15 s.
The tiny randomized latency/resist/regen terms are declared domain randomization,
not a calibrated model of the production server.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .schema import (
    ACTION_INDEX,
    ACTION_NAMES,
    MANA_COST,
    SPELL_CIRCLE,
    SPELL_REAGENTS,
    CombatView,
    Frame,
    encode,
    legal_mask,
)

SIM_VERSION = "approximate-pre-aos-options-v4"
POTION_FUSE = 3.75
POTION_FLIGHT = 1.0
POTION_RADIUS = 2
CAST_RECOVERY = 0.75
SHAPING_GAMMA = 0.99
_DAMAGE = {
    "magic_arrow": (4, 7), "harm": (1, 15), "explosion": (23, 44),
    "energy_bolt": (24, 41), "flamestrike": (27, 48),
}


def cast_seconds(spell: str) -> float:
    """Exact pre-AOS base cast duration before modeled transport/quantization."""
    return 0.5 + 0.25 * (SPELL_CIRCLE[spell] - 1) - 0.1


@dataclass
class Fighter:
    """Internal simulator state. Policies receive CombatView instead."""

    x: int
    y: int
    hp: float = 100.0
    hp_max: float = 100.0
    mana: float = 100.0
    mana_max: float = 100.0
    stamina: float = 25.0
    stamina_max: float = 25.0
    # ArenaSupplies.StockPotions replenishes five of each type for a duel.
    heal_potions: int = 5
    cure_potions: int = 5
    explosion_potions: int = 5
    reagents: list[int] = field(default_factory=lambda: [100] * 8)
    poison_until: float = 0.0
    poison_generation: int = 0
    paralyzed_until: float = 0.0
    weakened_until: float = 0.0
    clumsy_until: float = 0.0
    cast_spell: str = ""
    cast_until: float = 0.0
    cast_started: float = 0.0
    cast_generation: int = 0
    recovery_until: float = 0.0
    busy_until: float = 0.0
    heal_potion_until: float = 0.0
    cure_potion_until: float = 0.0
    explosion_potion_until: float = 0.0
    burst_until: float = 0.0
    meditation_until: float = 0.0
    last_action: str = "hold"
    last_spell: str = ""
    last_spell_at: float = -10.0
    completed: set[str] = field(default_factory=set)
    recent_dealt: deque = field(default_factory=deque)
    recent_received: deque = field(default_factory=deque)
    recent_self: deque = field(default_factory=deque)
    damage: float = 0.0
    damage_received: float = 0.0
    self_damage: float = 0.0
    healing: float = 0.0
    casts: int = 0
    interrupted: int = 0
    thrown: int = 0
    active_potion: int | None = None


@dataclass
class _Potion:
    owner: int
    explode_at: float
    x: int
    y: int
    held: bool = True
    thrown: bool = False


class DuelSim:
    """Symmetric, seeded two-player environment with fixed-time public frames.

    ``step`` accepts both actions at the same observation boundary. While an
    option owns the execution channel, its mask contains only hold. Invalid
    input becomes hold; info reports it. Match timeout is a terminal draw, not
    an invented victory for whichever bot happened to have more health.

    Terminal victory is +/-1. The only dense reward is discounted bounded
    health potential; total self-injury penalty is capped at 0.1 per episode.
    There is no reward per damage/heal action to farm with an opponent.
    """

    def __init__(
        self, seed: int = 0, max_steps: int = 1200, *, dt: float = 0.25,
        showdown_after: float = 180.0, domain_randomization: bool = True,
        shaping_gamma: float = SHAPING_GAMMA,
    ) -> None:
        if max_steps < 1 or not math.isfinite(dt) or not 0.05 <= dt <= 1.0:
            raise ValueError("max_steps must be positive and dt must be in [0.05, 1.0]")
        if not math.isfinite(showdown_after) or showdown_after < 0:
            raise ValueError("showdown_after must be finite and nonnegative")
        if not 0 < shaping_gamma <= 1:
            raise ValueError("shaping_gamma must be in (0, 1]")
        self.seed = seed
        self.max_steps = max_steps
        self.dt = dt
        self.showdown_after = showdown_after
        self.domain_randomization = domain_randomization
        self.shaping_gamma = shaping_gamma
        self.radius = 9
        self.reset(seed)

    def reset(self, seed: int | None = None) -> tuple[Frame, Frame]:
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.elapsed = 0.0
        self.steps = 0
        self.done = False
        self.winner: int | None = None
        self.events: list[tuple[float, int, str, int, Any]] = []
        self.event_sequence = 0
        self.potions: dict[int, _Potion] = {}
        self.potion_sequence = 0
        self.trace: list[dict[str, Any]] = []
        gap = self.rng.randint(4, 8) if self.domain_randomization else 4
        offset = self.rng.randint(-3, 3) if self.domain_randomization else 0
        self.fighters = [Fighter(-gap, -offset), Fighter(gap, offset)]
        self.latency = self.rng.uniform(0.03, 0.17) if self.domain_randomization else 0.05
        # Equal GM EvalInt/Resist, with a small distribution around the source's
        # circle-dependent resistance chance; not hidden skill observations.
        self.resist_offset = self.rng.uniform(-0.03, 0.03) if self.domain_randomization else 0.0
        # GM meditation+100int source rate is ~1 mana/s (active doubles it).
        self.mana_regen = self.rng.uniform(0.85, 1.05) if self.domain_randomization else 1.0
        self._potential_previous = self._potential()
        return self.frames()

    @property
    def showdown(self) -> bool:
        return self.elapsed >= self.showdown_after

    def _distance(self) -> int:
        a, b = self.fighters
        return max(abs(a.x - b.x), abs(a.y - b.y))

    def _destination(self, side: int, action: str) -> tuple[int, int]:
        own, other = self.fighters[side], self.fighters[1 - side]
        dx = (other.x > own.x) - (other.x < own.x)
        dy = (other.y > own.y) - (other.y < own.y)
        if dx == dy == 0:
            dx = 1 if side == 0 else -1
        if action == "retreat":
            dx, dy = -dx, -dy
        elif action == "strafe_left":
            dx, dy = dy, -dx
        elif action == "strafe_right":
            dx, dy = -dy, dx
        return own.x + dx, own.y + dy

    def _can_move(self, side: int, action: str) -> bool:
        x, y = self._destination(side, action)
        other = self.fighters[1 - side]
        return -self.radius <= x <= self.radius and -self.radius <= y <= self.radius and (x, y) != (other.x, other.y)

    @staticmethod
    def _recent(values: deque, now: float) -> float:
        while values and values[0][0] < now - 2.0:
            values.popleft()
        return sum(value for _, value in values)

    def view(self, side: int) -> CombatView:
        if side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        own, other = self.fighters[side], self.fighters[1 - side]
        age = self.elapsed - other.last_spell_at
        # This estimate comes from observed power words, not the pending damage
        # event queue. It may remain even when the actual spell fizzles.
        incoming = (-1.0 if other.last_spell != "explosion" else
                    cast_seconds("explosion") + 2.5 - age)
        if incoming < 0:
            incoming = -1.0
        potion = self.potions.get(own.active_potion)
        return CombatView(
            hp=own.hp, hp_max=own.hp_max, mana=own.mana, mana_max=own.mana_max,
            stamina=own.stamina, stamina_max=own.stamina_max,
            opponent_hp=other.hp, opponent_hp_max=other.hp_max,
            opponent_health_known=True, dx=other.x - own.x, dy=other.y - own.y,
            distance=self._distance(), poisoned=own.poison_until > self.elapsed,
            opponent_poisoned=other.poison_until > self.elapsed,
            paralyzed=own.paralyzed_until > self.elapsed,
            opponent_paralyzed=other.paralyzed_until > self.elapsed,
            cast_remaining=max(0.0, own.cast_until - self.elapsed),
            busy_remaining=max(0.0, own.busy_until - self.elapsed),
            recovery_remaining=max(0.0, own.recovery_until - self.elapsed),
            heal_potion_cooldown=max(0.0, own.heal_potion_until - self.elapsed),
            cure_potion_cooldown=max(0.0, own.cure_potion_until - self.elapsed),
            explosion_potion_cooldown=max(0.0, own.explosion_potion_until - self.elapsed),
            burst_cooldown=max(0.0, own.burst_until - self.elapsed),
            heal_potions=own.heal_potions, cure_potions=own.cure_potions,
            explosion_potions=own.explosion_potions, reagent_counts=tuple(own.reagents),
            elapsed=self.elapsed, showdown=self.showdown, showdown_after=self.showdown_after,
            recent_damage_dealt=self._recent(own.recent_dealt, self.elapsed),
            recent_damage_received=self._recent(own.recent_received, self.elapsed),
            recent_self_damage=self._recent(own.recent_self, self.elapsed),
            opponent_cast=other.last_spell if age <= 10.0 else "", opponent_spell_age=age,
            opponent_visible=True, line_of_sight=True, fresh=True, ready=not self.done,
            move_legal=tuple(self._can_move(side, name) for name in ACTION_NAMES[1:5]),
            potion_fuse=max(0.0, potion.explode_at - self.elapsed) if potion and potion.held else 0.0,
            incoming_explosion_seconds=incoming, last_action=own.last_action,
        )

    def frames(self) -> tuple[Frame, Frame]:
        return encode(self.view(0)), encode(self.view(1))

    def _schedule(self, when: float, kind: str, side: int, payload: Any = None) -> None:
        self.event_sequence += 1
        heapq.heappush(self.events, (when, self.event_sequence, kind, side, payload))

    def _record(self, kind: str, side: int, **values: Any) -> None:
        self.trace.append({"t": round(self.elapsed, 4), "type": kind, "side": side, **values})

    def _start_cast(self, side: int, spell: str, combo: str = "") -> None:
        own = self.fighters[side]
        own.cast_generation += 1
        own.cast_spell = spell
        own.cast_started = self.elapsed
        own.cast_until = self.elapsed + cast_seconds(spell) + self.latency
        own.busy_until = own.cast_until
        own.last_spell, own.last_spell_at = spell, self.elapsed
        own.meditation_until = 0.0
        own.casts += 1
        self._record("cast", side, spell=spell)
        token = own.cast_generation
        self._schedule(own.cast_until, "cast_complete", side, (spell, token, combo))
        if combo:
            own.burst_until = self.elapsed + 20.0
            own.busy_until = self.elapsed + 7.0  # released by completion/failure cleanup
            self._schedule(self.elapsed + 0.85, "combo_prime", side, token)

    def _consume_spell(self, side: int, spell: str) -> bool:
        own = self.fighters[side]
        if own.mana < MANA_COST[spell] or any(own.reagents[r] <= 0 for r in SPELL_REAGENTS[spell]):
            return False
        for reagent in SPELL_REAGENTS[spell]:
            own.reagents[reagent] -= 1
        # GM seventh circle: (100 - min65.714) / (max105.714 - min65.714)
        # succeeds 6/7 of the time (MagerySpell.GetCastSkills). Disrupted
        # casting does not spend mana, successful sequencing does.
        if spell == "flamestrike" and self.rng.random() < 1.0 / 7.0:
            self._record("fizzle", side, spell=spell)
            return False
        own.mana -= MANA_COST[spell]
        return True

    def _prime(self, side: int) -> int | None:
        own = self.fighters[side]
        if own.explosion_potions <= 0 or own.active_potion is not None:
            return None
        own.explosion_potions -= 1
        own.explosion_potion_until = self.elapsed + 10.0
        self.potion_sequence += 1
        potion_id = self.potion_sequence
        self.potions[potion_id] = _Potion(side, self.elapsed + POTION_FUSE, own.x, own.y)
        own.active_potion = potion_id
        self._schedule(self.elapsed + POTION_FUSE, "potion_explode", side, potion_id)
        self._record("potion_prime", side, potion=potion_id)
        return potion_id

    def _throw(self, side: int, potion_id: int) -> None:
        potion = self.potions.get(potion_id)
        own, other = self.fighters[side], self.fighters[1 - side]
        if not potion or not potion.held:
            return
        distance = self._distance()
        if not 3 <= distance <= 12 or own.hp <= 0:
            # The safety executor disposes to clear ground away from its owner.
            # Like any fixed destination, that ground can become dangerous later.
            dx = -1 if other.x >= own.x else 1
            x = max(-self.radius, min(self.radius, own.x + dx * 5))
            y = own.y
            if max(abs(x - own.x), abs(y - own.y)) <= 2:
                y = own.y + (5 if own.y <= 0 else -5)
            self._record("potion_dispose", side, potion=potion_id)
        else:
            x, y = other.x, other.y
        potion.held = False
        potion.thrown = True
        own.active_potion = None
        own.thrown += 1
        # While in flight, explosion location stays at the thrower's last tile.
        potion.x, potion.y = own.x, own.y
        self._schedule(self.elapsed + POTION_FLIGHT, "potion_land", side, (potion_id, x, y))
        self._record("potion_throw", side, potion=potion_id, x=x, y=y)

    def _perform(self, side: int, name: str) -> None:
        own = self.fighters[side]
        if name == "hold":
            return
        own.last_action = name
        if name in ACTION_NAMES[1:5]:
            if self._can_move(side, name):
                own.x, own.y = self._destination(side, name)
                own.stamina = max(0.0, own.stamina - 0.2)
            own.busy_until = self.elapsed + self.dt
            own.meditation_until = 0.0
        elif name in MANA_COST:
            self._start_cast(side, name)
        elif name.startswith("burst_"):
            self._start_cast(side, "explosion", "energy_bolt" if name == "burst_bolt" else "flamestrike")
        elif name == "meditate":
            own.meditation_until = self.elapsed + 8.0
            own.busy_until = self.elapsed + 2.0
        elif name == "heal_potion":
            own.heal_potions -= 1
            own.heal_potion_until = self.elapsed + 10.0
            self._heal(side, self.rng.randint(9, 30))
            own.busy_until = self.elapsed + self.dt
        elif name == "cure_potion":
            own.cure_potions -= 1
            own.cure_potion_until = self.elapsed + 1.0
            own.poison_until = 0.0
            own.poison_generation += 1
            own.busy_until = self.elapsed + self.dt
        elif name == "explosion_potion":
            potion = self._prime(side)
            if potion is not None:
                own.busy_until = self.elapsed + 2.1 + self.latency
                self._schedule(own.busy_until, "potion_throw", side, potion)

    def _heal(self, side: int, amount: float) -> None:
        own = self.fighters[side]
        if own.hp <= 0 or self.showdown or own.poison_until > self.elapsed:
            return
        amount = max(0.0, min(amount, own.hp_max - own.hp))
        own.hp += amount
        own.healing += amount
        self._record("heal", side, amount=amount)

    def _hurt(self, source: int, target: int, amount: float, kind: str) -> None:
        own, victim = self.fighters[source], self.fighters[target]
        actual = max(0.0, min(victim.hp, amount))
        if actual <= 0:
            return
        victim.hp -= actual
        victim.damage_received += actual
        victim.recent_received.append((self.elapsed, actual))
        victim.paralyzed_until = 0.0
        victim.meditation_until = 0.0
        if source == target:
            victim.self_damage += actual
            victim.recent_self.append((self.elapsed, actual))
        else:
            own.damage += actual
            own.recent_dealt.append((self.elapsed, actual))
        self._record("damage", source, target=target, amount=actual, damage_kind=kind)
        self._interrupt_cast(target, kind)

    def _interrupt_cast(self, target: int, cause: str) -> None:
        """ServUO OnCasterHurt without inventing HP damage for debuff hits.

        Only the Casting phase can be disturbed; an already-open target cursor
        is Sequencing. Damage and successful Weaken/Clumsy/Poison target effects
        reach this path, while Paralyze and beneficial Cure do not.
        """
        victim = self.fighters[target]
        if victim.cast_until > self.elapsed:
            # GM caster without Protection. Spell.GetDisturbRecovery imposes a
            # 0.2-second floor and decreases with progress through the cast.
            duration = max(0.05, victim.cast_until - victim.cast_started)
            recovery = max(0.2, 1.0 - math.sqrt(max(0.0, self.elapsed - victim.cast_started) / duration))
            victim.cast_generation += 1
            victim.cast_spell = ""
            victim.cast_until = self.elapsed
            victim.recovery_until = self.elapsed + recovery
            victim.busy_until = self.elapsed
            victim.interrupted += 1
            self._record("interrupt", target, cause=cause)
            if victim.active_potion is not None:
                self._throw(target, victim.active_potion)

    def _apply_spell(self, side: int, spell: str) -> None:
        own, other = self.fighters[side], self.fighters[1 - side]
        own.completed.add(spell)
        if spell in ("heal", "greater_heal"):
            self._heal(side, self.rng.randint(11, 15) if spell == "heal" else self.rng.randint(41, 50))
        elif spell == "cure":
            own.poison_until = 0.0
            own.poison_generation += 1
        elif self._distance() > 12 or other.hp <= 0:
            self._record("out_of_range", side, spell=spell)
        elif spell in _DAMAGE:
            low, high = _DAMAGE[spell]
            amount = float(self.rng.randint(low, high))
            resist = max(0.1, (100 - (16 + SPELL_CIRCLE[spell] * 5)) / 200) + self.resist_offset
            if self.rng.random() < resist:
                amount *= 0.6 if spell == "flamestrike" else 0.75
            if spell == "harm":
                amount *= 1.0 if self._distance() <= 1 else (0.5 if self._distance() <= 2 else 0.25)
            delay = 2.5 if spell == "explosion" else (0.0 if spell == "harm" else 0.5)
            self._schedule(self.elapsed + delay, "damage", side, (1 - side, amount, spell))
        elif spell == "poison":
            # Poison.Target calls OnCasterHurt and clears paralysis before its
            # resistance roll. Poison levels/chances remain the declared v2
            # approximation; this corrects only the immediate control effect.
            self._interrupt_cast(1 - side, spell)
            other.paralyzed_until = 0.0
            other.poison_until = self.elapsed + 30.0
            other.poison_generation += 1
            self._schedule(self.elapsed + 3.0, "poison_tick", side, (1 - side, other.poison_generation))
        elif spell == "paralyze":
            other.paralyzed_until = self.elapsed + (20.25 if self.rng.random() < 0.3 else 27.0)
        elif spell == "weaken":
            # Weaken/Clumsy.Target interrupt even an equal-strength existing
            # curse. The stat-cap change is not a damage event or damage reward.
            self._interrupt_cast(1 - side, spell)
            other.paralyzed_until = 0.0
            other.weakened_until = self.elapsed + 120.0
            other.hp_max = 90.0
            other.hp = min(other.hp, other.hp_max)
        elif spell == "clumsy":
            self._interrupt_cast(1 - side, spell)
            other.paralyzed_until = 0.0
            other.clumsy_until = self.elapsed + 120.0
            other.stamina_max = 15.0
            other.stamina = min(other.stamina, other.stamina_max)

    def _event(self, kind: str, side: int, payload: Any) -> None:
        own = self.fighters[side]
        if kind == "damage":
            target, amount, spell = payload
            self._hurt(side, target, amount, spell)
        elif kind == "cast_complete":
            spell, token, combo = payload
            if token != own.cast_generation or own.hp <= 0:
                return
            own.cast_spell = ""
            own.cast_until = self.elapsed
            own.recovery_until = self.elapsed + CAST_RECOVERY
            if combo:
                self._schedule(self.elapsed + 0.65, "combo_release", side, (spell, token, combo))
            else:
                own.busy_until = self.elapsed
                if self._consume_spell(side, spell):
                    self._apply_spell(side, spell)
        elif kind == "combo_prime":
            if payload == own.cast_generation and own.hp > 0:
                self._prime(side)
        elif kind == "combo_release":
            spell, token, finisher = payload
            if token != own.cast_generation or own.hp <= 0:
                if own.active_potion is not None:
                    self._throw(side, own.active_potion)
                own.busy_until = self.elapsed
                return
            successful = self._consume_spell(side, spell)
            if successful:
                self._apply_spell(side, spell)
            if own.active_potion is not None:
                self._throw(side, own.active_potion)
            if successful and own.hp >= own.hp_max * 0.45 and own.poison_until <= self.elapsed:
                self._schedule(max(self.elapsed + self.latency, own.recovery_until), "combo_finish", side, finisher)
            else:
                own.busy_until = self.elapsed
        elif kind == "combo_finish":
            if (own.hp > 0 and own.mana >= MANA_COST[payload]
                    and own.paralyzed_until <= self.elapsed and own.poison_until <= self.elapsed):
                self._start_cast(side, payload)
            else:
                own.busy_until = self.elapsed
        elif kind == "potion_throw":
            self._throw(side, payload)
            own.busy_until = self.elapsed
        elif kind == "potion_land":
            potion_id, x, y = payload
            potion = self.potions.get(potion_id)
            if potion:
                potion.x, potion.y = x, y
        elif kind == "potion_explode":
            potion = self.potions.pop(payload, None)
            if not potion:
                return
            if potion.held:
                potion.x, potion.y = own.x, own.y
                own.active_potion = None
                own.busy_until = self.elapsed
            self._record("potion_explode", side, potion=payload, x=potion.x, y=potion.y)
            for target, fighter in enumerate(self.fighters):
                if max(abs(fighter.x - potion.x), abs(fighter.y - potion.y)) <= POTION_RADIUS:
                    self._hurt(side, target, self.rng.randint(15, 30), "explosion_potion")
        elif kind == "poison_tick":
            target, generation = payload
            other = self.fighters[target]
            if other.poison_generation == generation and other.poison_until > self.elapsed:
                self._hurt(side, target, max(4.0, other.hp * 0.025), "poison")
                self._schedule(self.elapsed + 3.0, "poison_tick", side, payload)

    def _advance(self, until: float) -> None:
        interval = until - self.elapsed
        for own in self.fighters:
            if own.hp <= 0:
                continue
            meditation = own.meditation_until > self.elapsed
            own.mana = min(own.mana_max, own.mana + interval * self.mana_regen * (2 if meditation else 1))
            own.stamina = min(own.stamina_max, own.stamina + interval / 7.0)
            if not self.showdown and own.poison_until <= self.elapsed:
                own.hp = min(own.hp_max, own.hp + interval / 11.0)
            if own.weakened_until and own.weakened_until <= until:
                own.hp_max = 100.0
                own.weakened_until = 0.0
            if own.clumsy_until and own.clumsy_until <= until:
                own.stamina_max = 25.0
                own.clumsy_until = 0.0
        self.elapsed = until

    def _potential(self) -> float:
        a, b = self.fighters
        return 0.05 * (a.hp / a.hp_max - b.hp / b.hp_max)

    def step(self, actions: tuple[int, int]) -> tuple[tuple[Frame, Frame], tuple[float, float], bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode finished; call reset before step")
        if len(actions) != 2:
            raise ValueError("exactly two actions required")
        masks = (legal_mask(self.view(0)), legal_mask(self.view(1)))
        self_before = [own.self_damage for own in self.fighters]
        invalid = [False, False]
        validated = [0, 0]
        for side, action in enumerate(actions):
            if not isinstance(action, int) or not 0 <= action < len(ACTION_NAMES) or not masks[side][action]:
                invalid[side] = True
            else:
                validated[side] = action
        # Seeded ordering avoids permanently giving seat zero first resolution
        # for simultaneous event times. Masks were computed before either action.
        order = (0, 1) if self.rng.random() < 0.5 else (1, 0)
        for side in order:
            self._perform(side, ACTION_NAMES[validated[side]])
        self.steps += 1
        end = self.steps * self.dt
        while self.events and self.events[0][0] <= end + 1e-9:
            when, _, kind, side, payload = heapq.heappop(self.events)
            self._advance(max(self.elapsed, when))
            self._event(kind, side, payload)
        self._advance(end)
        dead = [own.hp <= 0 for own in self.fighters]
        time_limit = self.steps >= self.max_steps and not any(dead)
        # This is a game rule, so a time-limit draw has no bootstrap value.
        # The rollout collector separately truncates unfinished games when its
        # sample budget runs out and can then bootstrap the still-live state.
        terminated = any(dead) or time_limit
        truncated = False
        self.done = terminated
        if terminated and dead[0] != dead[1]:
            self.winner = 1 if dead[0] else 0
        next_potential = 0.0 if self.done else self._potential()
        shaped = self.shaping_gamma * next_potential - self._potential_previous
        self._potential_previous = next_potential
        rewards = [shaped, -shaped]
        for side, own in enumerate(self.fighters):
            rewards[side] -= 0.001 * (min(100.0, own.self_damage) - min(100.0, self_before[side]))
            if self.winner is not None:
                rewards[side] += 1.0 if self.winner == side else -1.0
        info = {
            "source": "approximate-pre-aos-simulator", "sim_version": SIM_VERSION,
            "winner": self.winner, "terminated": terminated, "truncated": truncated,
            "elapsed": self.elapsed, "steps": self.steps, "showdown": self.showdown,
            "time_limit": time_limit,
            "invalid_actions": invalid, "applied_actions": validated,
            "damage": [own.damage for own in self.fighters],
            "self_damage": [own.self_damage for own in self.fighters],
            "healing": [own.healing for own in self.fighters],
            "casts": [own.casts for own in self.fighters],
            "interrupts": [own.interrupted for own in self.fighters],
            "potions_thrown": [own.thrown for own in self.fighters],
        }
        return self.frames(), (rewards[0], rewards[1]), self.done, info

    def teacher(self, side: int) -> int:
        """Reproducible bootstrap policy using only this player's public view.

        ``completed`` is the player's own acknowledged cast history, which a
        real client can retain. No enemy resources, events, or plan are read.
        """
        view = self.view(side)
        mask = legal_mask(view)
        own_history = self.fighters[side].completed
        hp = view.hp / max(1.0, view.hp_max)
        opponent_hp = view.opponent_hp / max(1.0, view.opponent_hp_max) if view.opponent_health_known else 1.0
        choices: list[str] = []
        if view.poisoned:
            choices += ["cure_potion", "cure"]
        if hp < 0.7:
            choices.append("heal_potion")
        if hp < 0.25:
            choices.append("heal")
        if hp < 0.6:
            choices.append("greater_heal")
        if view.distance > 9:
            choices.append("approach")
        if view.elapsed < 12 and hp > 0.65:
            choices += [name for name in ("weaken", "clumsy") if name not in own_history]
        if (view.opponent_cast in ("greater_heal", "explosion", "energy_bolt", "flamestrike")
                and view.opponent_spell_age < cast_seconds(view.opponent_cast) - 0.65):
            choices.append("magic_arrow")
        if view.mana < 23 and opponent_hp > 0.2:
            if view.distance < 7:
                choices.append("retreat")
            choices.append("meditate")
        if opponent_hp < 0.14:
            choices += ["harm" if view.distance <= 1 else "energy_bolt", "magic_arrow"]
        choices += ["burst_bolt", "explosion_potion"]
        if not view.opponent_poisoned and view.mana > 35:
            choices.append("poison")
        if view.opponent_paralyzed:
            choices.append("explosion")
        choices += ["energy_bolt", "explosion", "harm" if view.distance <= 2 else "magic_arrow"]
        if hp < 0.8:
            choices.append("greater_heal")
        choices += ["meditate", "approach", "hold"]
        return next(ACTION_INDEX[name] for name in choices if mask[ACTION_INDEX[name]])
