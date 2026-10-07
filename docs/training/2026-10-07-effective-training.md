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

## Follow-up: retention result and queued frozen live comparison

Local diagnostic session50127 finished successfully. On 100 fixtures/model:
parent56W/27L/17D (score.645), unweighted23W/10L/67D (.565),
balanced34W/8L/58D (.630). Both candidates convert losses to draws but lose wins;
this does not establish improvement. Summaries are in
`2026-10-07-imitation-simulator-retention.json`.

Reserved fixture007 was independently verified after completion. On 120 free
choices, parent accuracy .4667/CE2.0796, unweighted .55/CE2.7764, balanced
.5583/CE2.3979. Accuracy increases but cross entropy worsens: some wrong choices
have become more confident. Neither candidate has passed a deployment gate.
No weights were changed after these validation results. Balanced is selected
only for the next screening experiment, due to better simulator retention than
unweighted, not for public promotion.

Queued isolated next experiment:
- code `/opt/uoarena/anima3-live-imitation-eval-20261007` (snapshot0431a3b)
- unit `arena-imitation-frozen-eval-20261007.service`
- launcher `/opt/uoarena/anima3-live-imitation-eval-20261007/queue-evaluation.py`
- out `.logs/frozen-comparison`
- baseline SHA0c6ca921e8bfd19fbaab3fa353d11819b2ba523cd1a810c2c4782447b7a5f56d
- candidate SHAd0096ef90098f589c36194e3638dccdf087125159106b0b89fc6e496d1c8cf0a
- frozen evaluation only, 4 games/style/model = 40 unique matches; seed20261008
- no PPO updates; public promotion remains disabled, sample size below gate
- launcher waits for current `arena-training-skill-clock-20261007.service` to
  terminate and requires complete28/28 unique matches; predecessor failure stops
  the queue rather than silently creating another run. Same account lock applies.
- current 28-game run plus this 40-game screen is a bounded68-match plan.
- queue max5h waiting; unit max10h total; Restart=no, CPU150%, memory3G.

Next: verify actual systemd handles; do not infer completion from this ledger.
If the predecessor fails, audit its last fixture before resuming. After successful
completion, inspect the 40-game comparison and validate model hashes remain fixed.
Only expand to a 200-game confirmation if live screening warrants it; otherwise
use observed errors and training/validation mismatch to revise the next candidate.

## CRITICAL: supply/weight confound found — both experiments halted

Latest diagnosis invalidates using this trial as a controlled strength comparison.
From recorded first learner features, fixtures001/002/003 had only ONE heal potion
and ZERO explosion/refresh potions; scripted opponent initially had five each.
By fixtures005–007 opponent supplies were also depleted. Prepared players later
weighed423 with weight_max390. An actual ordinary-client backpack inspection after
stopping found player57 has247 empty bottles and player58 has241; both weigh415.
Inventory audits and execution diagnostics are in `.logs/supply-audit-20261007/`.
Server Bottle.cs sets weight1; BasePotion adds a Bottle on consumption. Refill
uses PlaceInBackpack, which can fail capacity/weight checks; it nevertheless
announces refill. Learner lacking explosion potions also masks every burst option.
This is a material simulation/live and between-player condition confound.

Actions taken:
- Stopped queued `arena-imitation-frozen-eval-20261007` before it began ANY match.
- Waited for a safe boundary in active trial, stopped after11 completed matches,
  before fixture012 issued a challenge. SIGSTOP/recheck/SIGKILL used; auxiliary
  systemd kill returned an error but MainPID0 confirmed termination. Audit saved
  `experiment-halted.json`, match_started=false. No ongoing match interrupted.
- status.json now explicitly stopped with unequal-supplies reason, not complete.
- Both units MainPID0, no anima-bridge processes; arena-shard active.
- Added local exact training-supply audit: 100 of each reagent, 5 of each modeled
  potion, alive player, open own pack, weight<=weight_max. Collector writes
  prepared-supplies.json and refuses to challenge on failure. Not yet deployed.
- 26 relevant preparation/collector tests and lint pass.
- Live meditation request gaps ARE >=10 seconds, confirming skill-clock repair.

Next required work (do not restart the old queue):
1. Fix bottle accumulation while preserving items, e.g. stock service stores
   empty bottles in the player's bank before refill. Inspect current production
   Scripts.dll SHA and use isolated/pinned deployment as in training-reset fix;
   never replace unrelated newer shard changes. Server source worktree is
   `/Users/dkkang/dev/uo/arena-worktrees/servuo-training-reset`.
2. Verify via ordinary clients that both accounts receive complete equal supplies
   and remain comfortably below carry limit over repeated matches.
3. Deploy new client supply guard to a fresh experiment. Retain old records with
   confound annotation; don't rewrite them into purported fair results.
4. Retest the v3 warmstart with equal inventory FIRST. Existing live-imitation
   candidates learned partly from resource-starved teachers; do not treat their
   queued comparison as ready or their validations as proof of strategy quality.
5. Continue training and controlled evaluation toward actual live improvement.

## Supply correction deployed and fresh controlled run started

ServUO source commit32c2c2b81 banks exact empty Bottle items before public supply
refill, preserving serials/amounts. Current deployed Scripts.dll SHA
b30c89378266c7b0a7a2f55a162d979afecc5d7084ebc1782f338c91074012f9.
Pinned patch checked103614 existing methods: only Refill changed, one helper
added. Full DLL/Saves backup at `/opt/uoarena/supply-bottles-release-20261007`.
Shard was empty during restart; REPLAY_RELEASE.json provenance updated.

Normal-client acceptance twice per account: all8 reagents100, all5 modeled
potions5, zero backpack bottles, weight200/201 versus max390. Repeat refill is
idempotent. Evidence `2026-10-07-supply-acceptance.json`. Related client preparation,
collector and runner tests:51 passed. This fixes resources, not a learned-policy
strength claim.

Fresh run (do not resume either confounded previous unit):
- code `/opt/uoarena/anima3-training-supply-fixed-20261007`, snapshotbc3f10b
- unit `arena-training-supply-fixed-20261007.service`, confirmed MainPID257858
- run `.logs/live-pilot`, seed20261009,8 training+20 evaluation=28 matches
- parent is the original v3 simulator BC0c6ca921..., NOT any candidate trained
  on the confounded/starved live teacher data
- accounts unchanged, new pre-challenge exact supplies/weight guard enabled
- first prepared-supplies.json confirms both verified at weights200/201
- initial verified state: active, completed0, train-1

Next: verify repeated replenishment after actual potion consumption and inspect
first combat observations for five potions (not just preparation). Poll real
unit before reacting. Continue controlled model improvement; prior queued40-game
imitation comparison stays canceled. Reconsider real-teacher fine-tuning only
with properly supplied training data and new validation matches.

## First actual round and repeated replenishment verified

Verified wait on MainPID257858 through the first live match. It completed as a
300-second draw against balanced. First recorded neural observation contains
5 heal/5 explosion/5 refresh potions. The agent used5 heal and5 refresh potions,
independent explosion throws and a burst_bolt sequence. Fixture002's pre-challenge
supply audit again confirms all required counts and weights200/201. Thus the fix
works after actual consumption, not only repeated idle refill. Evidence:
`2026-10-07-first-replenishment-proof.json`, replay28d7cb6b255c4672a0e0caeb3d22071e.
One draw does not prove learned improvement. Continue the active28-game run;
collect only correctly supplied teacher data for future training. No additional
model/runner mutations were made during this verified wait.

## Follow-up: a distinct visibility deadlock diagnosed and fixed locally

The active supply-corrected unit was revalidated (MainPID257858); completed2
matches, currently collecting. No running code or budget was changed.
First corrected match had equal damage459/459 and5 potion throws each. Replay
showed both drifting beyond view after ~222s, then no more casts: by290s learner
had100mana but remained at13HP. Actor trajectory confirms256 unseen decisions,
ALL with movement actions masked. Cause: CombatExecutor._movements returned no
moves whenever opponent was absent. Simulator opponents are always visible.
This is an execution/environment mismatch, not simply poor mana strategy.

Local executor v8 remembers only an opponent's publicly seen position for up to
30seconds. When no longer visible, only approach toward that location is offered;
spell targeting remains disabled, terrain/collision checks still apply, hidden
mobile coordinates are never read, round reset clears the memory. Regression
covers hidden opponent at a different position, correct last-seen movement,
no spell/retreat, timeout and round reset. Runtime tests25 pass, lint passes.
Evidence `2026-10-07-visibility-diagnosis.json`.

IMPORTANT: v8 is local only. Active remote28-game trial remains v7 to preserve
its fixed comparison conditions. Validate v8 in a separate subsequent experiment
with the same code for both comparison arms; actual reacquisition and playing
strength remain unverified. Do not promote on these unit tests alone.

## Reconnection, audited resume, and demonstration supply provenance

On reconnect the original supply-fixed unit had failed after three completed
matches (one draw, two losses), before fixture004 challenge. Its progress showed
player58 Idle but no fresh DuelState from57 for30seconds. Shard active, online0,
live empty, no bridge processes; fixture004 contained only pre-match status and
progress, no challenge marker or match directory. Archived that attempt with
recovery-audit.json as fixture-004-handshake-attempt-1 and resumed original code,
checkpoint, seed and28-match budget using --resume in unit
arena-training-supply-fixed-resume-20261007. Completed receipts are revalidated;
no completed match is rerun. This is recovery of a transport/handshake failure,
not evidence that its underlying cause is resolved.

Offline demonstration import now requires the adjacent prepared-supplies.json
with exact participant IDs, verified counts and finite within-limit weight.
All three fresh corrected demos accepted (989,736,762 steps); three sampled
legacy supply-confounded demos rejected for absent audit. Nineteen targeted
tests pass including missing/wrong supplies, overweight and participant mismatch.
This local importer hardening is not deployed into the ongoing fixed experiment.
