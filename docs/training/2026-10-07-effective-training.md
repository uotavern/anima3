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

## Four completed corrected fixtures; teacher-data threshold enforced

Resumed unit MainPID259838 remains active and completed fixture004. Fixture004
is a frozen historical neural opponent by design (every fourth training game),
so it yields PPO data but no scripted-teacher demonstration. A local attempt to
fit fixtures001..004 was correctly refused: only three teacher records exist;
no candidate was created. Keep planned BC split001/002/003/005, validation006/007.
Parent diagnostic on the three accepted demonstrations is saved locally in
.logs/live-supply-fixed-copy/parent-imitation-diagnostics.json; this measures
imitation agreement, not strength. Need fixture005 before fitting live BC.

Local commit6dec2a0 adds maximum two pre-challenge idle-timeout retries, preserves
attempt directories and refuses retry with any challenge/results/match evidence.
Twenty league tests pass. Current running remote code remains unchanged.

Fixture005 subsequently completed (loss to potions); resumed unit moved to006.
Fresh BC candidate actually trained from001/002/003/005 only:2580 teacher steps,
8epochs, balanced actions,48optimizer updates, original parent0c6ca921.
Artifact .logs/supply-fixed-imitation-20261007-8epochs, fit summary
2026-10-07-supply-fixed-imitation-fit.json. Validation006/007 and independent
live evaluation pending; not deployed and no strength claim. This short8epoch
candidate limits initial fitting compared with the retired32epoch candidates;
selection must still rely on held-out and real matches.

## Corrected BC retention and terminal teacher collection defect

100 simulator games each, fresh seed3652000000/max1200: parent and8epochBC both
score0.735. Paired delta0, CI[-0.1,0.105]; balanced style delta-0.175. No simulator
improvement claim or promotion. Full outcomes in supply-fixed-retention JSON.

Fixture006 teacher record is INVALID (teacher_action_rejected): exactly final
steps137/138 have self_hp0, ready0 and hold-only mask. Executor correctly refuses
action in dead context, but collector was still labeling decisions before the
final DuelState arrived. Keep old receipt/record unchanged and excluded. Local
collector now requires fresh/ready view before teacher decision, so normal death
latency no longer poisons the whole demonstration.29 related tests and lint pass;
not deployed into current fixed run. Need valid held-out007 or new independent
collection before asserting generalization. Remote unit259838 still active at
last check, completed6 collecting007. Live improvement remains unproven.

## Next executor staged; held-out teacher validation unavailable

Remote isolated /opt/uoarena/anima3-training-v8-20261007 contains git snapshot
59fb090 and candidate-bc SHA5e56ab6879b253dc7b6f4b6482e06b43fb4dc3a8b5788e756a8e2424f1d7f8d0.
Verified import/checkpoint hash and executor pre-aos-options-v8-last-seen-approach
on server. No process launched there; current v7 run unchanged.

Current unit259838 completed7 and started008 (historical neural opponent).
Both intended BC validation teacher records006/007 are invalid with rejected
actions; loader correctly excludes both, so no held-out imitation metric exists.
Do not mistake empty metrics for a pass. Evidence supply-fixed-validation.json.
Need a new valid independent teacher record, with corrected collector if the
current run yields none. Full real playing strength evaluation still pending.

## Supply-corrected live PPO update completed; frozen evaluation started

Eight training matches complete. Ten eligible on-policy actor episodes (two
historical self matches contribute both actors),5357 rollout steps, four PPO
epochs/21428 optimized steps/8528 free policy steps. No rejected episodes.
Parent0c6ca921 -> candidate3d0f7b454607590cb0ff1eca5278ee5a7cac1521ee72e1b5e1c10e040df85b03.
KL0.00401, clip fraction0.03154, no KL early stop. Actual immutable checkpoint
copied locally to .logs/supply-fixed-ppo-20261007 and hash checked on load.
Summary2026-10-08-supply-fixed-ppo-update.json (Korea date).

Same100-game simulator retention seed3652000000: PPOscore0.730 vsparent0.735,
paired delta-0.005,CI[-0.105,0.095]. No established improvement or promotion.
Retention JSON saved. Unit259838 active, fixture009 evaluation-incumbent,
replay5a927bf676144e94b7bcde57a5b225aa live at30seconds. Twenty fixed-model live
comparison games now running, both models/executor unchanged. The BC candidate
5e56ab... is separate and is NOT what this current live comparison evaluates.

## First frozen evaluation and teacher-selection bias observation

Unit259838 active, completed9; fixture009 incumbent beat balanced in64727ms,
valid episode/template. Fixture010 PPOcandidate versus balanced live at79seconds,
replay81fb446e370f4d31a304ecd2cfb435d6. Single win is not comparative evidence.

Both excluded teacher validation fixtures006/007 have learner57 winning and
teacher58 dying. Rejection is specifically post-death hold labeling. Thus old
collector eligibility can systematically omit losing teachers, not merely add
random missing records. Preserve exclusions, report zero accepted validation,
and collect fresh held-out games after terminal-context fix. Do not evaluate
imitation generalization using only surviving/winning teachers. The PPO learner
records for these matches remain independently eligible and receipt-verified.

## First PPO evaluation invalidated by observation-consumption latency

Completed10, unit259838 active; fixture011 started. Fixture010 PPO lost to
balanced, replay81fb446e370f4d31a304ecd2cfb435d6 duration232409ms, normal receipt,
BUT episode ineligible with deadline. Must not present this as valid paired
strength evidence; current gate should fail closed even if later scores improve.

Exact rejected request9: observed866756.197545216, submitted866756.364342802,
received866756.379713137, consumed866757.20125142. Inference response ~15ms;
observation age1003.706ms exceeded1000ms deadline, response waited821.538ms.
Nearby observe RPC took662.980ms (previous452.859ms), pump30.493ms, state RPC31.763ms.
This is actor/bridge consumption latency, not demonstrated model inference
slowness. Keep deadline intact. Next investigate observation RPC and cgroup CPU
throttling (current150% quota includes bridges/workers); do not alter conditions
mid-comparison. Remote cpu.stat SSH query pending local exec session20679.
CPU query completed:4logicalCPUs, load1.15/1.36/1.32, unit cpu.stat nr_periods11912,
nr_throttled6052, throttled_usec81046315, usage_usec1518441126. Quota throttling
is present (~51% periods), but this cumulative counter does not prove causality
for request9. Next isolated trial can use200% for both arms and measure RPC p95/
p99 plus deadline counts; current fixed150% run remains unchanged.

## Nineteen completed, audited stat-dialog resume, queued BC comparison

Revalidated unit terminal(MainPID0/failed), completed19. Fixture020 had ONLY
status/progress/run.lock, error Could not verify base stats from Arena stat dialog,
no challenge; shard online0/liveempty, no bridge. Restarted same --resume unit;
new MainPID269830, existing known-preparation recovery archives020 automatically.
No completed match rerun and same v7/150%/checkpoint comparison remains fixed.

Valid intermediate outcomes: balanced parent2W/candidate1L(+1 invalid loss),
pressure parent2W/candidate2L; burst parent1W/candidate2W, pending other parent.
PPO candidate not promising; don't promote. Observe RPC median54..68ms, p99149..234ms,
only one inference deadline across first19learner logs; fixture010max663ms.

Prepared bounded next experiment, remote queued-bc-evaluation.py in
/opt/uoarena/anima3-training-v8-20261007, snapshot59fb090. Unit
arena-training-v8-bc-queued-20261008 waits on actual prior systemd state, then
requires complete28 status and no online/live players, pins parent0c6ca921 and
BCcandidate5e56ab hashes before launching runner(account flock). Fails if prior
run fails; no automatic takeover/retry of ambiguous matches. Maximum5h wait,
10h unit cap,40evaluation games(4/style/model), seed20261008,CPU200% botharms,
v8 botharms, corrected terminal teacher collector, bounded handshake retries.
No new training in this evaluation, no public promotion. This40game screen
cannot establish the strict200game promotion gate; use evidence to choose next
training/confirmation experiment. The queued process does not log into UO yet.

## Observation survey optimization implemented in separate client worktree

Unit269830 active(completed19/fighting020); queuedBCunit269927 remains waiting.
Inspected bridge4100ecb: each observation radius4 surveys81tiles; MapTerrain
walkable_step rebuilds TileScan across World.items for EACH tile. New worktree
/Users/dkkang/dev/uo/arena-worktrees/anima-client-observation-scan branch
fix/training-observation-scan builds once per observation and uses same scanned
planning predicate. No cache persists across network/world updates; door checks
unchanged. Original MapTerrain/pathfinder unaffected. Not deployed or added to
already queued fixed experiment.

26session unit tests pass. Added explicit asset-required integration regression:
3actualmap locations,200synthetic dynamic items,20revisions each with removals,
positions/doorgraphic/container changes. All60 complete terrain grids match old
adapter. Warm-map release benchmark total old54892us/new2536us (~21.6x terrain
survey speedup). This microbenchmark does NOT establish RPC or live-loop gain;
next deploy/benchmark separately, preserving current comparison conditions.

## Linux bridge artifact staged and building

Client25378fa archived to /opt/uoarena/anima-client-survey-25378fa. Separate
systemd build unit arena-build-survey-25378fa (MainPID270439) running locked
release cargo build, one job,CPU50%,Memory1G,Nice15,30min cap. This adds bounded
server load while current fixed trial is active; account for that limitation
when interpreting remaining timings. No existing binary overwritten and queued
BC comparison still references4100ecb. Need completed artifact hash, native
asset equivalence test, and actual ordinary-client RPC measurement before use.
Current trial269830 active completed20 collecting021; queuedBC269927 activewait.

## Native Linux optimization validation; apply only to not-yet-started BC screen

Build/test units completed successfully. Linux bridge25378fa SHA256
3d0af9bae62dd49431885490abeca510c3f313f3d10c7a4003d1e2ae73bb5814 at
/opt/uoarena/anima-client-survey-25378fa/target/release/anima-bridge.
Server native map regression60/60equivalent, original307642us/scanned5701us total
under50%CPUquota; these are quota-affected microbenchmark times, not whole RPC
or combat improvement evidence. No production binary replaced.

Current trial269830 active completed21 collecting022. Verified queued BC output
directory absent and prior trial incomplete, stopped only waiting BC unit,
updated its script to use optimized bridge for BOTH frozen arms, added pinned
binarySHA check, restarted waiting unit. Original script preserved beside it,
bridge-selection.json records provenance. This intentionally supersedes earlier
4100ecb queued-bridge plan BEFORE any BC evaluation match. Current v7 trial is
unchanged. Next40games will measure liveRPC/deadlines and v8 execution alongside
BC-vsparent outcomes; cannot attribute cross-run differences to a single change.
Transient waiting unit was removed by systemd on stop, so systemctl restart
returned unit-not-found. Recreated same unit with original resource/environment
properties via systemd-run and verified its new live MainPID; no match started
or duplicated during replacement.

## Paired-observation PPO drift diagnosis

On valid evaluation trajectories009,011..016, replayed identical full causal
feature/mask prefixes through both frozen models with initial zero hidden.
Recorded mean categorical probabilities only at free choices, no rejected010.
Artifact2026-10-08-ppo-policy-drift.json. This is same-input behavior diagnosis,
not counterfactual win estimation. Largest per-fixture mean changes ~0.0133;
pressure examples modestly reduce healing/potion probabilities and increase hold.
Therefore no evidence for a dramatic action-collapse; small probability shifts,
stochastic fight dynamics and small sample size remain possible explanations.
Do not hard-code actions from these evaluation records or count them as new
training. Use separate new training/confirmation data to investigate survival.
Current269830 active completed21 collecting022; BCwait272007 active.

## v7 trial terminated at final fixture; next BC screen actually launched

27completed collections;28actual matches. Last fixture028 learner never retained
trusted match state and actor raised lost authoritative live duel state. Teacher
saw Fighting then RoundOver. Server replay21c00c4e82964f4d9794f38e82d677d1 verified
complete, winner58,16853ms,noabort. Saved server replay+meta under028; do NOT
replay this started match or invent a learner trajectory. Original status stays
stopped; final-audit.json closes experiment with known invalid final collection.

Valid comparison scores(parent/PPOfrozen,9each): parent7W1L1D; PPO4W4L1D.
Each arm also has1invalid: PPO010deadline and parent028state loss. No promotion;
original incumbent retained. Insufficient/invalid evidence for strict gate and
observed results unfavorable. Two lost records cannot be treated as random.

Waiting BCunit correctly failed on prior experiment not-complete. Audited both
unitsMainPID0, no bridges, online0/liveempty, pinned bridge SHA verified, then
started distinct arena-training-v8-bc-screen-20261008.service. Same planned
40games,seed20261008,CPU200%,v8+optimizedbridge botharms, originalparent0c6ca921
versus BC5e56ab, no training during comparison. Output .logs/bc-frozen-evaluation
in /opt/uoarena/anima3-training-v8-20261007. This supersedes FAILED queued unit.

## First live v8/optimized-bridge latency sample

New BCscreen MainPID275277 active, fixture001 in progress, replay
b9e09bf2328e467da2ab19fa4f8b1c33. Learner observeRPC1954 samples: median34.287ms,
p9589.277ms,p99126.721ms,no inference_rejected events. Earlier v7/oldbridge150%
quota trials median54..68ms and p95116..197ms. This is encouraging combined
runtime evidence, not a randomized bridge-only benchmark:CPU200% also changed,
and arenas/states differ. No completed match or strength claim yet. Continue
checking terminal teacher data validity and last-seen pursuit in actual play.

## v8 first complete match and genuinely held-out imitation diagnosis

Unit275277 active completed1 collecting002. First fixture replay
b9e09bf2328e467da2ab19fa4f8b1c33: incumbent57 lost, teacher58won. Both records
eligible, teacher712steps accepted by supply+receipt importer. Learner had zero
unseen decisions, so last-seen pursuit wasn't exercised. Teacher survived, so
post-death exclusion fix remains unverified on a losing teacher in live data.

Frozen BC candidate on this unseen teacher episode419freechoices: parentCE1.19863
->BC0.89199, argmax agreement0.62530->0.61814. This is mixed imitation evidence,
not proof of improved play. No parameter updates, no model selection from this
record; full40game comparison continues with fixed hashes. Evidence
2026-10-08-v8-first-heldout.json. Do not claim fitting/held-out accuracy both rose.

## BC opening defeat and delayed-spell cue loss fixed locally

V8fixture002 replay33f309ba6422471e882f4a3cb840c8cb finished15543ms incl countdown,
candidate0damage/4casts, opponent98damage. Candidate chose four approaches,
Weaken,Clumsy,AgilityPotion,burst_bolt. No inference rejection;9freechoices.
Opponent potion30damage at14115ms,26at14483,42lethal at14514 (~399ms). Candidate
burst's bolt cast was interrupted. This is valid combat failure, not transport.

Observed actual features: incoming_explosion present steps14..23, then drops
immediately at24 when opponent's CorpPor word replaces Explosion. Delayed
Explosion damage had not arrived. Found same last-spell-only logic in simulator.
Local executorv9 keeps up to8 public Explosion-word expiry estimates separately
from latest spell; nearest unexpired estimate persists through follow-up spells,
expires naturally and clears on round reset. Simulator uses same public cue
history, never hidden damage queue; canceled/fizzled spells can still yield an
estimate as before. Schema dimensions unchanged; execution version incremented.
61runtime/simulator tests and lint pass, including followup preservation and
expiry and simulator queue-independent estimate/reset. NOT deployed in active
v8comparison. Needs retraining/calibration before future experiment.

Do not claim cue loss caused fixture002 defeat: candidate had already committed
to burst before its cue disappeared, and busy options restrict choices to hold.
That action granularity and burst-survival training remain distinct limitations.

## Threat-memory simulator retraining actually started

Local exec session44466 runs sourceb170aba, output
.logs/curriculum-threat-memory-20261008, log/tmp/anima3-threat-memory-training.log.
Same seed20261009 and budget as prior skill-clock experiment:32768teacher steps,
64BCepochs,16PPOupdates*4096steps,1200step horizon,100evalgames/model. New held-out
seed3653000000. Four drill collections completed; fitting still running at last
check. Independent of unchanged live v8BCscreen(unit275277 completed3 collecting4).
Do not claim improvement until frozen comparisons; old/new evaluations must use
same updated simulator/seeds before comparing effects of threat-memory change.

## Losing teacher retained after terminal-context fix

V8BCscreen fixture003 candidate57won. Losing teacher58 demonstration eligible,
no invalid reasons,783steps pass load_demonstration supply+replay verification.
This exercises the actual bug that excluded losing teachers in prior006/007,
so terminal collector repair has live evidence now. First three screening
outcomes: incumbentL,BC L,BC W; far too few for strength conclusion.

Local corrected-simulator training session44466 still live in BC stage. Original
parent independently evaluated under SAME corrected simulator and reserved
3653000000 seed,100games score0.655; artifact evaluation-original-parent.json
under .logs/curriculum-threat-memory-20261008. Prior0.735 used a different seed;
do not interpret that difference as regression. Compare with new frozen models
only once their fitting completes. Baseline evalsession40750 finished normally.

## Threat-memory training completed; controlled PPO+rehearsal experiment started

Session44466 completed. Scores on3653000000/100games: random0.045,newBC0.680,
newPPO0.575; originalparent sameupdatedsim0.655. BCvsoriginal delta0.025,
CI[-0.085,0.135] notpassed. PPOvsBC delta-0.105,CI[-0.20,-0.01],regression.
NewBC SHA772c0fc6f5cacae8e0bd8cde8faf0c51e1fd1094fc76580e08ff982ee29d6232 remains
experimental, not deployed. JSON2026-10-08-threat-memory-training.json.

Source366ec62 adds optional --bc-rehearsal-epochs0..8(default0) after each PPO
update. Reuses only original training demonstrations, reconstructs current
causal prefixes, uses separate Adam moments for BC vsPPO. No evaluation data
used for gradients.20league tests/lint passed; end-to-end experiment running.

Session11458: .logs/curriculum-rehearsal-20261008, log/tmp/anima3-rehearsal-training.log,
same seed/budgets as plainPPO but1BCepoch after each update. Reserved eval seed
3654000000. Session62303 independently scores originalparent and frozen plainPPO
on sameupdatedsim/sameseed, saves evaluation-original-parent/plain-ppo.json there.
Need compare rehearsal candidate to BOTH warmstart and plainPPO; improvement over
a regressed PPO alone isn't sufficient. Remote v8BCscreen275277 active,3complete4live.

## Four live v8 matches valid; rehearsal update stream verified

Current BCscreen275277 active,4complete collecting005. Incumbent1W1L,BC1W1L;
all4learner and all4teacher records eligible, no inference rejection. ObserveRPC
median31.4..35.6ms,p9588.9..102.1ms. No unseen-opponent decisions in these4matches,
so do not yet claim live last-seen pursuit success. Runtime audit JSON saved.

Local rehearsal11458 actually executing PPO+BC(reachedupdate3/16, each followup
BC70optimizer updates over32768 original labels); no errors. Independent matched
control eval62303completed: originalparent0.665,plainPPO0.575 on3654000000.
Await newwarmstart/rehearsalcandidate same fixtures for fair comparison.

## Rehearsal not sufficient; isolate elapsed-time discount experiment

Session11458 complete. Same3654000000 held-out100games each: original0.665,
plainPPO0.575,newBC0.625,rehearsalPPO0.590. RehearsalvsBCdelta-0.035
CI[-0.13,0.06];vsoriginal-0.075 CI[-0.17,0.02];vsplainPPO+0.015CI[-0.08,0.11].
No candidate promoted. Full artifact2026-10-08-rehearsal-training.json.

Found curriculum driver uses per-decision discount(.995 every0.25sec), while
live training uses .995 persecond. Both are internally valid objectives, but
inconsistent temporal horizons:300sec terminal weight .995^1200~.00244 versus
.995^300~.2223. Don't assert this alone caused regression. Main train CLI already
supports consistent time units; curriculum now does too(default1second, explicit
0legacy reproducibility). Potential shaping uses gamma^(sim.dt/unit), matched
to GAE, not blindly .995.72training/league tests pass, including executing tiny
whole curriculum in both units and checking rollout/GAE/manifest consistency.

Source95e5d0f local session51251 trains .logs/curriculum-time-discount-20261008,
log/tmp/anima3-time-discount-training.log. Same seed20261009/budgets,16updates,
NOrehearsal to isolate temporal-unit change. Eval3655000000. Controlsession76797
scores originalparent and frozen per-stepPPO on sameeval fixtures. Active live
BCscreen remainsv8, MainPID275277,completed4collecting5; no deployed mutations.

## Elapsed-time PPO improves aggregate score, pressure remains a regression

Session51251 completed normally. Same3655000000/100games: original50W24L26D
(score.630), per-stepPPO40W8L52D(.660), BC52W23L25D(.645), timePPO66W26L8D(.700).
TimePPO vsoriginal +.070 CI[-.035,.180]; vsBC +.055 CI[-.045,.160];
vsper-stepPPO +.040 CI[-.070,.145]. All gates fail; no live deployment.
Pressure-style score original.825, BC.775, per-stepPPO.900, timePPO.575.
TimePPO approach2053/retreat1639 vsBC1408/1198, stun78 vs86 across20games.
These are associations, not proof that movement caused losses. JSON time-discount
training and pressure artifacts preserve outcomes and aggregate actions.

To complete a controlled 2x2 comparison (per-step/elapsed-time x no/with BC
rehearsal), session15151 runs .logs/curriculum-time-rehearsal-20261008,
log/tmp/anima3-time-rehearsal-training.log. Source d5cbea0; same20261009 seed,
32768 BCsteps/64epochs,65536 PPOsteps/16updates, max1200; timeunit1 and1 rehearsal
epoch per update. Reserved evaluation3656000000,100games/model.
Session52401 evaluates originalparent and three frozen prior PPO checkpoints
(per-step,per-step+rehearsal,time) on exactly those fixtures. No heldout gradients.
This tests whether rehearsal retains pressure defense with longer-horizon PPO;
it is an experiment, not an assumed fix. Current live v8 frozen BCscreen unchanged:
unit275277 active,7complete/40,8collecting; allfirst7eligible.

## Live low-health choice audit

Read-only fixture007 (BC loss,145a73b76b7341b5b09af32d03bb4c47): learner117damage,
142healing vsopponent234damage/86healing;108eligibletransitions. At2HP index62
and4HP indices103/104 legal actions onlyhold/movement; index105addsrefresh.
Healing unavailable due cast recovery +heal potion cooldown. Do not misdiagnose
refresh as ignoring available heal. Retreat legal but survival counterfactual
unproven. JSONlive-low-health-audit preserves features/masks. Noheldout training.
Local15151 live,BC completed4480optimizer updates;52401 live,originalparent.760
andper-stepPPO.570 scored on new3656000000 fixtures. Remaining controls pending.

## Time+rehearsal complete; continue strongest incumbent directly

Sessions15151/52401 completed normally. Same3656000000/100games: original.760,
newBC.690, per-stepPPO.570, per-step+rehearsal.650, timePPO.690,
time+rehearsal.765(63W10L27D). Last vsoriginal+.005 CI[-.090,.105],pressure-.200,
notpassed. vsper-stepPPO+.195CI[.095,.285]pass, vsper-step+rehearsal+.115
CI[.035,.190]pass, but beating weaker baselines doesn't warrant promotion.
CompleteJSON2026-10-08-time-rehearsal-training.json; publicpolicy untouched.

Sourcee05ee1e adds --start-checkpoint to curriculum. Loads verified existing
checkpoint, preserves exact weights for initial/warmstart, skips initial BC refit;
training-only demos still support optional postPPO rehearsal. Parentpath/SHA
recorded. Actual tiny end-to-end test checks exact incoming PPO weights, absence
of prePPO BC, postPPO rehearsal ordering, persisted initial/warmstartSHA and
unchanged parentfile;23league tests+ruffpassed.

Session34549 runs .logs/curriculum-incumbent-continue-20261008,
log/tmp/anima3-incumbent-continue-training.log. Starts strongest originalparent
0c6ca921... directly, same seed20261009/budgets65536 PPOsteps/16updates, timeunit1,
1BCrehearsal epoch/update. Reserved3657000000 expanded40games/style=200/model
compares actual originalparent to continuedcandidate. This tests retaining
existing competence instead of first refitting weaker BC. Trainingongoing.
Liveunit275277 verifiedactive8completed/40,fixture009collecting; no active-run edits.

## Local interruption recovered; burst cancellations corroborated

Toolsession34549 disappeared; ps inventory showed no training Python, lastupdate5,
no completed registry. Preserved partialrun with interruption-audit.json. Fresh
same-seed rerun in .logs/curriculum-incumbent-continue-20261008-retry, detachedPID
12317(log/tmp/anima3-incumbent-continue-retry.log), no optimizer-resume claim.
VerifiedPID running and log advancing through16updates; evaluationongoing.
No duplicate livegames. RemoteBCscreenPID275277 active9complete10collecting.

Fixture007 burst actions end with cliloc500641 concentrationdisturbed. Replay
first Explosioncast7263ms followed by enemyClumsytarget7350ms; secondcast30273ms
followedby35damage30434ms. Thus actual adversarial cast interference corroborates
executor cancellation. Simulator already models debuff/damage interrupts; no
speculative fix applied. Evidence2026-10-08-live-burst-interruption.json.

## Direct incumbent continuation completed: no aggregate gain

Detached12317 completed, complete registry verified. 3657000000/200games/model:
initial and candidate both.6375. Paired0CI[-.075,.0725], no promotion. Style deltas
balanced+.0125,pressure-.0625,burst-.0125,defensive+.1625,potions-.100.
Direct continuation reduced prior large pressure regression but transferred
weakness to potion opponents; no claim of learned overall improvement. Original
SHA0c6ca921... retained. FullJSON2026-10-08-incumbent-continuation.json.
Next evidence priority is finishing frozen actual ServUO BC comparison; additional
simulator tuning alone has not demonstrated live strength.

## First ten runtime audit and real last-seen pursuit evidence

Live275277 active,fixture011 fighting with recent journal/health/action updates,
not hung. First10 learner+teacher all eligible. Observation medians31.19..35.60ms;
JSON2026-10-08-v8-ten-runtime.json. Four recorded policy cancellations are all
mask_changed (burst_bolt,approach,harm,approach), observationages61..145ms and
responseages1.68..1.96ms, not inference deadlines.

Fixture008/0101d318d6274af2be625d9f7a85a8ff:78unseen transitions,11approach choices,
10visibility-loss spans, all regain visibility. Longest51transitions/14.656sec,
2approaches; other spans.28..2.67sec. Approach action completion records success.
This is real evidence v8 can move and reacquire after losing visibility, unlike
old v7 disabled movement. It does not isolate causality or prove win-rate gain;
no replay/heldout data used for gradients and frozen comparison unchanged.

## Twelve matched live games complete; comparison still underpowered

Verifiedunit275277 active; fixture012 ended aftershowdown,013started normally.
First12samepairedfixtures: incumbent3W3L,candidate4W2L; all12eligible/templatesvalid.
Interimpaired+.1667CI[-.5,1.0], onlybalanced/pressure represented; gatefails
insufficientevidence. Balancedcandidate delta-.25,pressure+1.0 basedjust2games/arm,
not reliable style superiority. Artifact2026-10-08-live-twelve-interim.json
preservesmatchIDs/hashes/results. No change to models, seeds or40gamebudget.

## Real imitation coverage limits identified

CurrentBCcandidate used4verified episodes/2580steps but1467free-choice labels.
Hold641(43.7%),approach+retreat516(35.2%),burst_bolt2,punch1,stun9,
explosion_potion6,greater_heal6. Repeating epochs cannot supply missing combat
situations. Saved2026-10-08-real-imitation-coverage.json includes absent actions;
absence isn't itself a requirement to cast everyspell. Next imitation experiment
should collect new disjoint matches balanced across5teacherstyles/bothseats,
retain current40screen as heldout rather than recycle outcomes into fitting.
Remote275277active,13complete14collecting;013incumbentpressurewineligible.

## Balanced collection mode staged after frozen screen

Source7b97aba adds --collect-per-style even2..40, requiresbatches0/nocandidate.
Collects bothseats perstyle percycle; freezes original checkpoint, does no PPO,
no evaluation and no model registry/promotion. Pins purpose/count in resumeplan;
existing receipts/retry/accountflock reused. dataset.json declares demonstration
validation stillrequired by load_demonstration before fitting.24league tests+ruff
pass; mode test proves20games,balanced5styles*2seats*2repeats,zerooptimizerupdates,
noevaluation/registry and repeatresume produceszeroadditionalgames.

Remoteisolatedsnapshot /opt/uoarena/anima3-demonstrations-7b97aba CLIhelpchecked.
Unit arena-training-demonstrations-20261008.service MainPID281665 verifiedactive,
waiting inqueued-collection.py. Current275277screenstillactive, newcollection
outputabsent verified. Queue waits actualpriorunitterminal (max5h), thenrequires
priorstatuscomplete40. Anyfailed/incompleteprior stops withoutnewgames. Starts
30freshgames (--collect-per-style6), fixedoriginalparent0c6ca921..., v9executor,
optimized25378fabridge, seed20261010. Sourcev9 applies to bothteacher/learner and
is explicitnewdataset; not a mutation of ongoingv8screen. CPU200%,RAM3G,Nice5,
unitRuntimeMax8h. It collects only; fitting/independentv9evaluation stillpending.

## Prespecified whole-match split and fitting queued

Source2634bd0 scripts/fit_collected_demonstrations.py verifies completed dedicated
collection, sameparentSHA, exactstatus/dataset rows, uniquematchIDs, balanced even
styles and alternating seatpairs. Holds out lastpair/style: forplanned30games,
fixtures001..020train,021..030validation. Neither scores nor outcomes affectsplit.
Existingclonefitter verifies allreplay/supply receipts and disjointmatchIDs before
anygradient. EightfixedBCepochs,actionbalancing,seed20261010; no tuning onholdout.
6split tests pass (exact20/10, nooverlap, rejects incomplete/evaluation/duplicate/
wrongseat/mixedpolicy). Ruffclean afterliteralcleanup.

Isolatedremote /opt/uoarena/anima3-fit-balanced-20261008 snapshot2634bd0.
Unit arena-training-fit-balanced-20261008.service MainPID282728 verifiedactive;
queued-fit.py waits actualcollectionunitterminal(max8h),requirescomplete30,
then executesfit into freshcandidate-bc. Outputabsent verified. CPU150%,RAM3G,
Nice10,RuntimeMax10h. Noaccounts/duels run byfitunit and no automaticpublicdeploy.
Firstunit40screen ->second30collection ->third20train/10holdout fit now chained;
actualnewmodel and independentliveevaluation remainpending.

## Halfway real evaluation: tied, all records eligible

Verified275277active20complete21collecting. First20matchedgames incumbent6W4L,
BCcandidate6W4L; no aggregate advantage atthispoint. All20learner records and
all20teacher demonstrations eligible; teachers9867steps total. Retainthese as
heldout evaluation, not gradients forqueuedfit. SnapshotJSONlive-twenty-interim
preservesexactmatchIDs/policyhashes/eligibility andteacherstepcounts. Full40gate
stillpending; noselectedmodelorpublicdeployment change.

## Completed 40-game frozen comparison and started balanced collection

The v8 screen ended with status complete, 40/40 matches, and its systemd unit
inactive with MainPID 0. Incumbent: 14 wins, 5 losses, 1 draw; candidate: 13 wins,
7 losses. All learner episodes and templates pass the evaluation gate checks.
Paired score delta is -0.075, bootstrap 95% interval [-0.30, 0.20]. Style deltas:
balanced -0.25, pressure +0.25, burst -0.25, defensive -0.125, potions 0.0.
The gate rejects promotion and retains original SHA 0c6ca921e8bfd19fbaab3fa353d11819b2ba523cd1a810c2c4782447b7a5f56d.
This is no evidence of improved live strength; public deployment remains false.
Evidence: 2026-10-08-v8-bc-final.json includes all match IDs and source digests.
Raw final status, registry, evaluations and plan are copied under .logs/v8-bc-final-audit.

The queued collection unit (MainPID 281665) automatically advanced after the
screen terminated successfully. Verified 1/30 complete and fixture-002 collecting.
This new collection uses the predeclared v9 executor and frozen incumbent, and
is separate from the 40 held-out evaluation matches. The subsequent 20-game fit
and 10-game validation remain pending until all 30 collection games complete.

## Balanced collection stopped on fixture 019 transport silence

Collection unit 281665 and dependent fit unit 282728 are now terminal failed,
MainPID 0. Collection status stopped, completed 18, error `lost authoritative
live duel state`; no fit output created. All 18 teacher records eligible,
9028 steps / 5159 free choices. Do not restart the original queue wrappers.

Fixture 019 has a durable challenge-issued receipt but no assigned actor log.
Both unassigned actors requested state 24 times and received no trusted state.
After teleport to Yew Abbey, learner bytes_in froze at 82328 and teacher at
81935 while bytes_out increased; journal cursors stayed 10/11. No pending frame
or opcode. Thus transport silence, not merely rejected DuelState JSON, is the
current evidence. Root cause remains unproven.

Server replay 0d0f508eadbc4400b2dea568c4958c3e, SHA
08f4bd13acaf50d1d2910c04c88f0c15dda659500b9df7a1ed26a5e1d4e23f68,
started 2026-10-08T04:14:22Z, ended 04:15:15Z, 53020 ms, arena 6.
Shard journal confirms normal countdown, FIGHT, both disconnected at 04:14:42/43,
then disconnect forfeit with both HP 100% and attacks 0/0. This is an actual
19th match, not a safe pre-challenge retry. Exclude it from gradients and
strength evaluation; preserve its actual match count when recovering.
Current dataset is incomplete and strict split/fit must remain stopped.

## Transport reproduction collection

Root cause remains unproven after reviewing Rust pump_once and server Send/Flush.
To distinguish server silence from a receiving client failure, started a separate
10-game collection (five styles x two seats), same 7b97aba source, frozen original
checkpoint, bridge 25378fa, seed 20261011. This is fresh diagnostic training data;
not a replay of failed fixture019 and not held-out strength evaluation.
Unit arena-training-transport-diagnostic-20261008.service verified active PID304123.
Output .logs/transport-diagnostic-20261008 under /opt/uoarena/anima3-demonstrations-7b97aba.
CPU200%, RAM3G, RuntimeMax2h. Original stopped dataset remains untouched.
Metadata-only tcpdump unit arena-training-transport-metadata-20261008.service
PID304063 active, RuntimeMax2h, loopback TCP2593, -nn -tt -l, no payload dump.
Output transport-metadata-20261008.log under the same remote source directory.
Stop metadata capture after the diagnostic collection ends. No optimizer or
promotion runs in this diagnostic. Need resume/replace balanced fit afterwards.
