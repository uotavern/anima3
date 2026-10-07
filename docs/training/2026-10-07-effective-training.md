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
