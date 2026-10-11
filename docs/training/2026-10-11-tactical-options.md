# Public-observation tactical controller

Implemented in the training worktree; not deployed or promoted to the neural incumbent.
Select `--opponent-style tactical` in the existing neural live runner with a scripted
opponent. Existing five evaluation styles and frozen 27-action model remain intact.

## Decisions

- Survival first: cure, heal, stamina recovery, escape.
- Public opponent spell words trigger interrupt consideration. Magic Arrow requires
  remaining estimated cast time greater than its 0.4 s cast + 0.5 s pre-AOS damage
  delay + 0.15 s margin. Public words cannot prove an opponent is still casting.
- At contact range prefer armed-capable stun or punch. At two tiles approach during
  an estimated cast window when healthy. A swing/interrupt is not guaranteed.
- Existing cursor-driven burst executor primes during Explosion, releases the spell,
  reopens the potion cursor, throws, then casts Energy Bolt or Flamestrike when its
  recovery clock allows. Prefer Flamestrike against observed HP below 60%; resource
  masks preserve the existing heal/cure reserve. This is timing intent, not a claim
  that all three hits land on exactly the same server tick.
- Ground rethrow: only a visible single explosion potion within one tile, with its
  own overhead countdown and at least 1.5 s estimated remaining, can be reused.
  Double click the original serial, wait for the neutral ground target cursor, then
  throw immediately. Never restart the fuse, chase an unknown countdown, or retry a
  serial after an attempted rethrow. Opponent must be 3..12 tiles away at selection.

## Server evidence

ServUO training-reset worktree:
`Scripts/Items/Consumables/BasePotion.cs:OnDoubleClick` checks one-tile range and
500 ms same-type action lock. `BaseExplosionPotion.cs:Drink` only initializes a
fuse when m_Timer is null. `ThrowTarget` permits twelve tiles and uses a one-second
flight. Pre-AOS ticks start at .75 s and repeat each second; an observed item count
is used instead of private timer state. `Scripts/Spells/Base/SpellHelper.cs`
uses a .5 s pre-AOS delayed damage interval for Magic Arrow.

## Evidence and learning boundary

Rethrow logs `tactical_rethrow_started` with policy_override=true. Its internal
procedure has action -1 and is never a neural action label. A live round with a
rethrow is marked `ground_rethrow_outside_policy_schema` and excluded from BC/PPO.
This keeps ground-fuse information, which the existing network does not receive,
from silently becoming an unlearnable or mislabeled demonstration. Incorporating
rethrows into the learned policy requires versioned public ground-fuse features,
a new action, simulator parity checks, retraining, and independent live evaluation.
Other tactical choices use existing legal action masks and are collectable.

Verified by controlled-clock wire tests for both burst finishers, ground rethrow,
countdown expiry, unknown/spoken countdown rejection, range and blast-radius
rejection, plus teacher priority/resource tests. No new live match or weight update
was performed for this implementation.

## Incoming potion defense and mana recovery

The `tactical` profile now consumes the bridge's existing public `recent_effects`
ring. Enemy purple moving effects mark the visible landing tile as hazardous for
up to four seconds (pre-AOS fuse upper bound plus observation margin). The initial
historical ring is skipped and sequence numbers prevent old effects retriggering.
A visible ground countdown can establish that a one-tile rethrow is still feasible;
otherwise the controller immediately seeks a walkable route beyond the two-tile
blast radius. It cancels ordinary spell cursors/procedures for escape; an owned lit
potion must first be disposed through the existing cursor-safe executor. Terrain,
occupied tiles, corner checks, paralysis, stamina, and multiple hazard locations
constrain movement. No path or late packets can still result in damage; this is
not an invulnerability guarantee. Active defense rounds are excluded from neural
training under `potion_defense_outside_policy_schema`.

Below 23 mana the tactical teacher enters recovery mode until mana reaches 60
(or 75% of maximum if lower). It retreats when too close, meditates when legal,
and holds while meditation is unavailable. Offensive spells, opening debuffs,
small attacks and potion offense are not selected in recovery. Cure/heal and
incoming-potion defense retain priority. Existing five frozen benchmark styles
and neural weights are unchanged.

Validation: 96 runtime/simulator/live-contract tests passed, including projectile
ring decode/deduplication, historical and own-projectile filtering, blocked exits,
short-fuse spell cancellation, safe rethrow priority, and mana recovery hysteresis.
Deployment and live validation subsequently completed; see the report below.


## Server deployment and live validation (2026-10-11)

- Initial defense snapshot 5672268: two complete games, tactical 1W/1L.
- Found a real server mechanic mismatch: potion Reposition_OnTick retains a Mobile
  target until its one-second landing callback, so the landing location follows
  the target during flight. Tracking only the original destination was insufficient.
- Corrected snapshot 99f0184 deployed to
  `/opt/uoarena/anima3-defense-99f0184`; two further games, tactical 2W/0L.
  Unit `arena-training-defense-99f0184` finished inactive with Result=success.
- Added public target-position tracking during flight and visible ground-potion
  tracking after landing. 97 focused tests passed before deployment.
- Initial snapshot: 2/10 enemy explosions had no damage event to the defender in
  the next 50 ms; corrected snapshot: 6/9. This is an event-association audit, not
  definitive damage-source attribution or a statistically established gain.
- Corrected first-game failures include movement delayed during casting and
  insufficient time to exit after landing/direction change. Avoidance is not guaranteed.
- All four games: zero offensive action selections during the recorded mana
  recovery periods. Cure/heal, resources, retreat, meditation and hold were allowed.
- No ground rethrow occurred in these four games; the earlier two-game tactical
  pilot separately verified rethrow. No neural weights were updated or promoted.
- Raw replay SHA256 values were checked against server metadata. Audits are
  `2026-10-11-defense-v11-live.json` and `2026-10-11-defense-v12-live.json`;
  regenerate using `scripts/report_potion_defense.py`.
