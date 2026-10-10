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
