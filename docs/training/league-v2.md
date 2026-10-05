# Anima3 curriculum and league training v2

## Contract

- Schema v2: 75 public/self features, 27 masked options. V1 weights are rejected,
  not silently expanded. Enemy mana, cooldowns and inventories are never inputs.
- Executor implements punch, acknowledged stun toggle, agility/refresh, existing
  close-range Harm, potion combinations, cursor-clock held Explosion and fast
  follow-up. A wire Attack is not evidence of a landed punch.
- Fists rules follow the local ServUO source: 15 stamina, 10 s attempt delay,
  75% GM success, 4 s Freeze. Failed stun remains armed. Unknown skills mask stun.
- Simulator is approximate. Armor, swing hits, potion bonuses and latency are not
  a complete ServUO model. Only live evaluations establish live playing strength.

## Training

```
python -m anima3.neural.league --out .logs/league-new \
  --bc-steps-per-drill 4096 --bc-epochs 32 \
  --updates 16 --rollout-steps 4096 --eval-games-per-style 20
```

New output directory required. Local simulator only; it never changes shard state.
Four randomized drills: melee, survival, burst, full match. Demonstrations retain
public features, masks and labels and are explicitly not PPO-eligible. BC rebuilds
recurrent state from each episode start under current weights instead of reusing
random-initial-policy hidden states. PPO uses frozen behavior batches and truncated
BPTT with burn-in. Scripted styles: balanced, pressure, burst, defensive, potions.
Half of simulator fixtures sample a pool of frozen historical policies.

Final evaluation uses untouched seeds, both sides, and separate policy RNG.
Gate: paired seed-block bootstrap 95% lower delta > 0, >=20 games per style,
no style regresses by more than 0.10. Initial, BC and PPO comparisons are reported.
The registry points to the incumbent if a candidate fails. This is an experimental
registry; public agents are never replaced by these commands.

## Real-server league

```
PYTHONPATH=. python scripts/run_training_league.py \
  --checkpoint /path/to/v2/warmstart --out /path/to/new-live-run \
  --accounts-file /private/accounts.json \
  --bridge /path/to/anima-bridge --data-dir /path/to/assets \
  --host 127.0.0.1 --batches 1 --batch-matches 8 --eval-per-style 2
```

Uses two existing normal training accounts (arena_train_a/b by default), public
7x + explosion potion training challenges, alternates challenger. Total budget
is batches*batch-matches + 2*5*eval-per-style (28 above). Do not run concurrently
with the old runner on the same accounts. New league runners share an account lock.
Secrets are read from a private file and never logged by this runner.

Each batch freezes learner weights, mixes scripted and historical opponents, then
revalidates every replay receipt and recomputes behavior likelihoods before PPO.
Minimum four eligible episodes; otherwise the update is skipped and logged.
Off-policy historical-opponent records are excluded. Duplicate match/actor records
are rejected. Transport/build failures stop; no blind restart creating extra duels.

Frozen incumbent/candidate evaluations are interleaved after all updates. No
weights change during evaluation. Live pairing controls style and challenger order,
NOT server RNG. Two games/style is a pipeline pilot and cannot pass the evidence
gate; use >=20/style for an adequately populated promotion test. Even a passed
live gate only updates the experimental registry; rollback checkpoint is retained.

## Initial evidence (2026-10-06)

First pilot: BC16,384 steps, 8 epochs, PPO32,768 steps; no gate passed.
Found stale BC hidden-state issue; corrected full-prefix reconstruction, used
32 BC epochs, 65,536 PPO steps, and new evaluation seeds for second run.

| Second run, 100 held-out simulator games each | Score (win=1, draw=.5) |
|---|---:|
| Initial | .065 |
| BC warmstart | .400 |
| PPO candidate | .455 |

BC improvement .335, paired bootstrap CI [.265,.405]. PPO vs BC .055,
CI [-.015,.130], defensive-style delta -.125: retain BC. These are simulator
results, not real-server win rates. Selected warmstart SHA:
`39abd306f8b6dc69a10c9e26c58cd4617a4c2e67d4f8b1715291b378a8c40796`.

Artifacts: `.logs/curriculum-league-20261006-v2/` contains demonstrations,
immutable checkpoints, each evaluation fixture and `registry.json`.
