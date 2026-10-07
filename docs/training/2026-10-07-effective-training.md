# Effective-training experiment ledger — 2026-10-07

## Objective and completion criteria

Continue experiments until improvement is demonstrated on the real shard, not
merely a simulator score or successful gradient update. Multiple bounded runs,
including 100–200 matches when useful, are authorized. Goal remains active.
Previous goal turn was progress: completed the 28-game pilot and established
that the candidate did not beat the incumbent; two invalid fixtures were audited.

## Experiment A: skill-clock calibration + more demonstrations

Server source evidence: `Scripts/Skills/Meditation.cs::OnUse` returns 10 seconds
for both GM success/failure; `Server/Skills.cs` enforces NextSkillTime. Live
executor previously allowed repeated requests after 0.8 seconds; simulator
allowed them after 2 seconds and gave guaranteed, eight-second meditation.

Schema v3 adds a normalized 10-second meditation cooldown (76 public/self
features, same 27 actions). Old weights are explicitly incompatible. Mask also
requires free hands and >=10% health. Live executor records a conservative
10-second deadline when sending UseSkill. Simulator models GM success chance,
10-second reuse, and active regeneration until interruption or full mana.
These changes reduce wasted requests; their effect on live wins is unproven.

Offline experiment:
- directory `.logs/curriculum-league-20261007-skill-clock`
- seed 20261009; held-out seed 3640000000
- 32,768 demonstration steps, 64 BC epochs, 65,536 PPO steps (16 updates)
- max episode length 1,200 steps = 300 seconds; 100 games/model
- initial 1W/94L/5D; BC 56W/23L/21D; PPO 32W/27L/41D
- BC score .665; PPO .525; PPO-vs-BC delta -.14, CI [-.25,-.025]
- select BC SHA `0c6ca921e8bfd19fbaab3fa353d11819b2ba523cd1a810c2c4782447b7a5f56d`
- This is NOT a controlled attribution of all gain to skill-clock repair: data
  amount, BC epochs and episode horizon also changed. Old/new simulator scores
  are not comparable. Same-run held-out results are in the adjacent JSON file.

## Real-server teacher data

The scripted opponent now records a 0.25-second decision cadence, including
forced holds during owned options, exact public features/masks, executed labels,
and timestamps. `demonstration-<actor>.json` is saved only after the server replay
receipt is checked. It is marked PPO-ineligible and imported only by the separate
supervised pipeline. Actor/network/template failures remain invalid.

`anima3.neural.demonstrations.load_demonstration` checks receipt checksum, match
identity, participants, template, schema, masks, finite features and increasing
clock. `scripts/clone_live_demonstrations.py` trains a separate candidate; PPO
explicitly rejects these supervised trajectories. Use training fixture paths
for initial imitation. If evaluation demonstrations are later used as training
data, retire that evaluation set and use fresh held-out live fixtures.

## Active bounded live run

Deployed isolated source snapshot e597299, leaving the previous release intact:
- code: `/opt/uoarena/anima3-training-skill-clock-20261007`
- run: `.logs/live-pilot`
- unit: `arena-training-skill-clock-20261007.service`
- initial MainPID: 252504 (recheck systemd, never assume still active)
- budget: 8 training + 20 evaluation = 28 actual matches; seed 20261007
- workers: existing arena_train_a / arena_train_b; public 7x + explosion potions
- Restart=no, CPUQuota150%, MemoryMax3G, 5-hour maximum
- state when recorded: fixture001 fighting, unit active, no error
- local focused tests: 167 passed; lint passed

### Next work

1. Poll the actual systemd unit and status. Do not restart solely on SSH failure.
2. Verify first real demonstration receipts with the importer, examine actual
   meditation request intervals and rejected actions, then collect enough data.
3. Compare live W/L/D and execution metrics; current v3 pilot checks PPO against
   the v3 BC policy, not against the historical v2 incumbent.
4. Train a separate candidate from verified real-server demonstrations; evaluate
   on new fixtures. Consider explicit zero-column v2-to-v3 baseline migration for
   a controlled same-executor old/new comparison, with recorded lineage.
5. Investigate repeat PPO regressions (rare combo coverage, value/entropy tuning,
   imitation retention, simulator calibration). Keep rejected snapshots, do not
   promote merely because training ran.
6. Increase live evaluation sample counts only after finding a viable candidate.
   Completion requires real-server evidence, not this pilot alone.

## Follow-up: frozen-candidate comparison and first real labels

The next goal turn revalidated systemd MainPID 252504 as active (two bridges)
and found two completed, eligible losses. This is progress, not evidence of
playing-strength improvement. First two teacher recordings were downloaded to
`.logs/live-skill-clock-copy` and independently loaded against replay checksums:
214 timesteps, 62 free-choice labels. They include one burst_bolt, two stuns,
13 Harms, five Greater Heals and two explosion-potion choices. More verified
training-only fixtures are needed before supervised fine-tuning.

Commit d94df5f adds a frozen comparison mode without an intervening PPO update:

```
PYTHONPATH=. python scripts/run_training_league.py \
  --checkpoint /path/to/incumbent --candidate-checkpoint /path/to/bc-candidate \
  --batches 0 --eval-per-style 20 --out /path/to/fresh-evaluation \
  --accounts-file /private/accounts.json --bridge /path/to/anima-bridge \
  --data-dir /path/to/assets
```

This schedules 200 matches: 20 per opponent style per model. Start with a smaller
pilot if diagnosing execution, but it cannot pass the evidence gate. Both policy
hashes are pinned for resume. The runner lifecycle regression verifies no PPO
updates and no duplicate games when an evaluation-only run resumes. Existing
training-mode resume remains unchanged. Focused runner tests: 18 passed; lint
passed. This runner change is local only; active live experiment still uses its
original isolated source snapshot.

## Follow-up: imitation diagnostics and match-disjoint validation

The live unit was verified active again; it reached 3 completed games and
fixture004 Countdown. No restart or budget change was made. While it runs,
`imitation_metrics` now scores exact causal recurrent predictions against real
teacher labels, excluding forced holds from accuracy and cross entropy. On the
first two verified demonstrations the current BC model's argmax agrees on
41/62 free choices (66.13%, CE 1.059). This is teacher-state agreement, not sampled
live action agreement or playing strength. Clumsy 0/2 and burst_bolt 0/1 are
small-sample diagnostic leads; Harm 13/13 and stun 2/2 are also too small to
establish general competence. Exact metrics are recorded in adjacent JSON.

Supervised cloning now records before/after training and optional validation
metrics via `--validation-collections`. Whole matches must be disjoint, including
opposite actors of a match, to avoid leakage. With the first 8 collection fixtures,
use teacher fixtures 001/002/003/005 for training and 006/007 for validation if all
are verified (004/008 are historical-policy opponents, so contain no teacher
records). Do not use the evaluation fixtures for training this candidate.

Diagnostics regression: forced hold frames do not inflate accuracy; a wrong
free choice gives zero accuracy; empty validation is reported as None, not a
success. Six demonstration tests and lint passed. Active remote source remains
unchanged; these additions are offline diagnostics only.

## Follow-up: actual real-server imitation training completed

Live collection reached 5 completed fixtures. Verified training demonstrations
001/002/003/005 are the only input to the following two candidates; future
006/007 remain validation and the live evaluation fixtures remain unused.

Both start from v3 BC SHA 0c6ca921..., train for 32 epochs at learning rate 1e-4,
seed 20261007. They are stored separately:
- `.logs/live-imitation-20261007-unweighted`
- `.logs/live-imitation-20261007-balanced`

Training-label argmax accuracy: parent .47155, unweighted .93217, balanced .84573.
Training CE: 2.55141 → .37295 / .56070 respectively. These are fitting diagnostics,
NOT held-out improvement or live strength. Full provenance and hashes are in
`2026-10-07-live-imitation-fit.json`. Balanced loss uses inverse-square-root label
frequency clipped to [.25,4], counting free choices only. The motivation is the
third demo's 335 free hold labels versus only one burst_bolt. Seven demonstration
tests pass, including forced holds not altering class-frequency weights.

A separate inference-mode ablation on the same v3 warmstart, 100 games per mode,
fresh diagnostic seeds 3650000000, found categorical 66W/20L/14D versus greedy
12W/53L/35D. Do not switch live inference to greedy on this evidence. Per-fixture
results are `.logs/curriculum-league-20261007-skill-clock/sampling-ablation-*.json`.
This is simulator-only evidence, and these seeds are now development data.

Active local diagnostic handle: exec session 50127 runs
`/tmp/anima3-live-imitation-simulator-check.py`, log
`/tmp/anima3-live-imitation-simulator-check.log`, output
`.logs/live-imitation-20261007-diagnostics/simulator-*.json`.
It compares parent and both candidates on 100 fresh simulator fixtures each
(seed3651000000), looking for catastrophic forgetting, not live promotion.
Re-poll the handle before treating this job as finished. Remote live unit remains
the original unmodified experiment; new candidates have NOT been deployed.

Fixture006 subsequently completed while the live service remained active. It was
independently verified and scored without changing either candidate. On its 15
free-choice labels: parent accuracy .60 / CE1.1122; unweighted .6667 / CE.8083;
balanced .6667 / CE.9994. This tiny validation sample is insufficient for candidate
selection. Fixture007 is still reserved for validation. Exact results:
`2026-10-07-live-imitation-validation-006.json`. No live-strength claim follows.
