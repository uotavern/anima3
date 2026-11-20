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

### Final pre-live revision

Code review found that three initial opponent labels shared priorities. Burst now
prioritizes held Explosion combinations; potions prioritizes independent throws;
balanced retains mixed combinations. Short drills reset after 30 s to increase
rare combo examples. Repeated training with new seeds, 16,384 demonstration steps,
32 BC epochs, 65,536 PPO steps and 100 new evaluation fixtures per model:

- Initial score .075; BC .365; PPO .405.
- BC delta .290, CI [.215,.365]: pass.
- PPO vs BC delta .040, CI [-.035,.110]: retain BC.
- Selected SHA `e5915b8ddc9ecf1b5373fef4c9b60d11022584b3363faceac7aed918be67e903`.
- Exact summary and demonstration option counts: `2026-10-06-curriculum-results.json`.
- Live deployment artifact is `/opt/uoarena/anima3-training-league-20261006/warmstart`.

Final simulator W/L/D: initial 1/86/13, BC 8/35/57, PPO 11/30/59.
Most of the improvement is survival/draw conversion; it is not a 36.5% win rate.
The historical 100-duel experiment finished all fixtures before this live pilot.

### Resume and stat verification repair

The live pilot stopped after two verified games because a displayed DEX of 34
(buff/curse affected) was mistaken for an incorrect base build. Preparation now
reads the ordinary stat-ball dialog's RawStr/RawDex/RawInt fields, cancels the
read-only dialog, and independently checks the replay's first Countdown frame
for 100 HP / 100 mana / 25 stamina maxima. Bad starting templates are excluded
and stop the runner for repair. No server permissions or stat-check bypass is used.

Use the same command with `--resume`. Receipted fixtures are revalidated and
counted without playing them again. A known pre-challenge preparation failure is
archived before retry. Unknown in-match failures stop for audit. Settings and
checkpoint hash are pinned in `plan.json`; changing the budget on resume fails.
Completed batch checkpoints are hash checked and reused, not updated twice.

### Live pilot repairs and first batch

The direct training challenge did not belong to ArenaService.Sessions, so its
PrepareFighter skipped clearing stat modifiers. ServUO fix d843b12c8 applies the
reset to training matches and clears armed stun/disarm. A SHA-pinned Cecil patch
changed only PrepareFighter among 103,614 method bodies; original DLL and saves
are retained. Fixtures 5–8 all start at HP/mana/stamina maxima 100/100/25, including
fixtures whose lobby snapshots were still buffed or cursed.

Eight training games completed: 1 win, 7 losses, with fixture 4 excluded for a bad
starting template. PPO consumed 8 eligible actor episodes from 7 valid matches
(the historical opponent in fixture 8 used the identical frozen policy). Two
actor episodes from fixture 4 were rejected. One batch update used 955 rollout
steps, 3,820 optimized steps / 1,092 policy steps. New candidate SHA:
`ec9b2a7ce9f62256bb0701aca8bb031572255b8146104d4f8738925314a6b2e9`.

Before any evaluation challenge, execution was paused with fixture 9 still
connecting, no challenge marker, no match directory, no connected agents. The
watcher's kill returned an auxiliary-process error after killing the main process;
this was audited before archiving that preparation attempt. Resume reused all
8 receipts and the existing PPO update.

All 20 evaluation fixtures use executor v6: terrain radius 4 and 10 ms pump for
both models. Safe potion disposal retains walkability/path checks and >3 tile
clearance from visible mobiles. In initial live measurements, observe median fell
from 187.09 ms (fixture 5) to 61.04 ms (fixture 9); pump 63.33 to 22.08 ms.
Training and evaluation use different controller versions, so training win rate
must not be used as the incumbent evaluation baseline. Compare only the two
interleaved evaluation arms. Relevant regression suite: 158 tests passed.

### Final live pilot outcome

Completed all **28 unique matches**: 8 collection games and 20 frozen-policy
evaluation games. The interrupted run reused the single saved PPO update; no
match or update was repeated. Evidence and replay links are in
`2026-10-06-live-pilot-results.json`.

| Evaluation arm | Valid games | Wins / losses / draws | Excluded |
|---|---:|---|---:|
| Incumbent | 9 | 0 / 9 / 0 | 1 transport failure |
| PPO candidate | 10 | 0 / 9 / 1 | 0 |

Fixture 25 lost authoritative duel state: the learner's inbound byte count
remained 73,602, while the opponent received Countdown/Fighting/RoundOver and the
server recorded a completed loss (zero learner casts/damage). Root cause of the
receive stall is unresolved. The independently checksummed server receipt was
recovered, with **empty transitions**, eligible=false, and an explicit audit.
It counts against the match budget, but is excluded from learning and valid
performance statistics. No action or likelihood record was reconstructed. Restart
created fresh connections; fixtures 26–28 then completed normally. Promotion now
also rejects ineligible evaluation episodes, beyond checking starting templates.
The gate's raw planned-fixture delta/CI is diagnostic only because of this invalid
fixture and insufficient sample size; it is not evidence of playing strength.

The candidate was **not selected**. The warmstart remains the experimental
incumbent. No public agent was replaced. All 20 evaluation fixtures used the same
executor fingerprint; each arm kept exactly one policy SHA. The trial established
working collection, exclusion, batched PPO, frozen inference, resume, and rejection
of an unsupported model—not a stronger real-server policy.

Median observe RPC: training 175.98 ms, incumbent evaluation 64.11 ms, candidate
64.77 ms. Pump: 63.25 ms down to about 21.2 ms. Across learner logs, 19 successful
stun messages and 722 spell requests were recorded. These execution counts are
not win-rate evidence. Frequent hold/meditate choices warrant action-mask and
mana-management diagnosis against the server's skill cooldowns before more live
training. Simulator/real-server mismatch remains a limitation; next experiments
should prioritize real-server teacher demonstrations and scenario calibration.

Final verification: 159 focused tests passed; 28 unique server receipts and
checkpoint hashes audited; one PPO update; evaluation service exited successfully;
no anima-bridge processes remained; arena-shard stayed active.
