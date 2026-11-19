# Anima3 training lineage integration

Tracking: [UO workspace issue 11](https://github.com/uotavern/uo/issues/11).
This series integrates the training lineage into primary Anima3 in reviewable
units. It preserves the primary branch's later fixes and keeps experimental
model selection separate from a public rollout.

## Starting points

| Line | Revision | Unique commits since common ancestor |
| --- | --- | ---: |
| Primary `main` | `c258400401ab63059212503a648f4817051485a8` | 5 |
| `feat/training-league` | `819f4f426bfaf72e1a132abff9354480250f1444` | 160 |
| Common ancestor | `621627c7999ee7ce6552793316c8c37cc9b7a14e` | — |

The training branch changes 198 files with roughly 41,000 added lines relative
to the common ancestor. A wholesale file replacement would lose newer primary
behavior in `body.py` and `magic.py` even when it appears to resolve conflicts.
Historical worktrees and their uncommitted files remain untouched.

## Unit 1 Bridge observation and transport boundary

This unit starts from primary `main`, not the older training snapshot. It ports
only the following prerequisite behavior:

- `BridgeBody.diagnostics` exposes the last received optional transport counter
  dictionary. An older schema-33 bridge that omits it remains supported. Missing
  counters mean no new snapshot, not that traffic stopped.
- A failed `pump` response raises `BodyError` instead of silently returning zero
  applied packets. Non-object RPC responses fail at the bridge boundary.
- `Observation` retains optional target cursor ID/flag/type and recent visible
  effects. Missing fields retain their old defaults. The fields are appended to
  the dataclass so existing positional construction is not shifted.
- The handshake still requires schema 33. Additive optional fields do not imply
  accepting an arbitrary future schema version.

The current `anima-client` schema-33 emitter already provides these fields. This
unit does not add an executor, decision policy, neural dependency, or live game.
It changes no model weights, rankings or server configuration.

### Source provenance

| Hunk origin | Imported behavior | Deliberately separate work |
| --- | --- | --- |
| `6b8d6f0` | `body.py` diagnostics and pump failure propagation | Neural policy, trainer, live runner and promotion logic |
| `5f442da` | `contract.py` target cursor metadata | Debuff/opening procedures and potion timing |
| `5672268` | `contract.py` recent visible effects | Tactical potion defense and mana recovery |

The port additionally serializes diagnostic updates inside the RPC lock and
checks that replies are JSON objects. These are local boundary checks, not
changes to the upstream packet protocol. The source branch inserted cursor
fields before `war`; this port appends them to avoid changing positional APIs.

### Primary changes that must survive every later unit

| Commit | Required behavior | Current protection |
| --- | --- | --- |
| `c677025` | Headless `anima-bridge` by default; `anima-agent` for spectator mode | Mocked launch-selection tests and explicit-binary override test |
| `18885df` | Tracked `uv.lock` | Lock file preserved; locked installation succeeds |
| `c3643cd` | Hosted lab default `uo.hulryung.com:2594` and environment override | Default constants preserved |
| `ccbc571` | Reflection casts without a target cursor; each attempt resets reflection timing | `magic.py` and its existing regression tests unchanged |
| `c258400` | Experiment-five record, per-arm `caffeinate`, real child exit status | README and `experiments/run_arm.sh` unchanged |

All five commits remain ancestors because this unit branches from `main`.

### Offline acceptance

The new offline GitHub Actions job pins action commits, Python 3.12.14 and
uv 0.12.17, installs the tracked lockfile, runs the complete offline suite and
lints only this integration boundary. It receives no game or model credentials.
Legacy files outside this unit are not claimed to pass a newly imposed lint gate.

From a clean checkout with Python 3.12 and uv:

```sh
uv sync --locked --extra dev --extra jeff
uv run --locked --extra dev --extra jeff pytest -q
uv run --locked --extra dev --extra jeff ruff check \
  anima3/body.py anima3/contract.py tests/test_bridge_contract.py
```

The existing `test_backend_names_split_cloud_jev_from_local_jeff` imports the
optional `typesafe_sdk` package even though its client is mocked, so the `jeff`
extra is needed for the full offline suite. No API key, running model, shard or
bridge binary is used by these tests. The locked dependency file is unchanged.

The first clean installation with only `dev` passed 94 tests but failed that
existing missing-optional-package test. Installing the already-locked `jeff`
extra produced **95 passing tests** (81 existing and 14 new). Selected-file Ruff
checks passed. This dependency requirement is explicit rather than hidden by a
previously populated development environment. No live acceptance was run.

## Unit 2 Read-only replay evidence

Based on merged unit 1 (`e2ef73b`), this ports the standalone replay reader from
training commits `9da8784`, `5f442da` and `012436b`. It adds index/download,
SHA-256 receipt validation and observed damage, healing, cast, potion and burst
metrics. Self-inflicted damage is separate from offensive damage; unrelated
participants do not inflate offensive metrics.

The local boundary additionally rejects malformed JSON objects, invalid player
identities, bool-as-integer counters, inconsistent result types, embedded end or
header records and missing result fields. It verifies complete recordings,
ordered sequence/time, match identity, header/manifest agreement where present,
and final outcome agreement. These are archive-integrity checks, not proof of
training eligibility or a server signature. Fields consumed by metrics are type-checked before use. Other event semantics, start
fixtures, policy identity and learning eligibility remain later-unit checks.

Downloads are GET-only, have a 15-second socket timeout, at most three transport
attempts, and a 34 MiB bound on both wire and expanded gzip data. Declared response
length must match received bytes. Corrupt evidence and permanent HTTP failures
are not retried. Callers must supply a trusted server base URL and should apply
an outer deadline where a whole-job elapsed budget is required; the socket
timeout is not a total download deadline.

Offline acceptance: **146 passing tests**, including all 95 unit-1/main tests
and 51 replay cases. The new fixtures cover wrong hashes/IDs, truncation,
ordering, malformed receipts, retry bounds, URL construction, gzip expansion and
response length. Network calls are mocked; no live matches, training, promotion,
server writes or new dependencies are involved. CI now includes
`anima3/replay.py` and `tests/test_replay.py` in its existing scoped Ruff gate.
All five primary fixes remain ancestors and their source files are unchanged.

## Unit 3 Participant preparation and sparring boundaries

Based on unit 2 (`fc74273`), this ports public skill/stat-ball preparation,
server-only DuelState parsing and exact supply audits from `duel_wait.py`, and
separates the bounded handshake/receipt behavior from the training sparring
runner. Origins include `fcb98ac`, `5e148d4`, `0da70f3`, `5903e2a`, `6e5d778`,
`bc3f10b` and `a702d24`. No agent, strategy, neural or learning modules are needed.

- `prepare` first requests fresh idle state and refuses active duels or pending
  invitations before public leave/enter/supply commands. Optional raw-stat proof
  reads and cancels the Arena stat dialog, even if displayed stats already match.
- `poll` discards previously buffered journal and accepts only server system
  state. `wait_training_idle` refuses unrelated opponents/rules and requires
  fresh idle responses from two distinct participants.
- `request_training_challenge` issues one explicitly unrated supported challenge.
  `accept_invitation` checks the fresh token's opponent, rules and round count.
  Acceptance is not match completion; callers must observe the assigned match ID.
- `verify_receipt` requires that actual expected match ID, exact ordered seats,
  rules, complete SHA-bound archive, no abort and explicit training/unranked
  flags. A historical replay for the same pair cannot satisfy another match.
- `account_locks` holds nonblocking POSIX locks for each case-insensitive account.
  All runners must share the same private lock directory and canonical endpoint
  key; hostname/IP aliases are caller responsibility. This is local coordination,
  not a remote lease. A runner must hold locks across preparation and the match.

`BoundedBody` checks cancellation and a finite deadline before each operation.
An RPC already in flight remains subject to the bridge's own timeout; use an
outer process watchdog where hard elapsed limits are required. Preparation has
no reconnect/retry loop that can reissue a match. Default bounds are 180 seconds
for preparation, 30 for state/invitations and 360 for waiting out this pair's
existing training fixture. No infinite worker or autonomous fight CLI is added.

Offline acceptance: **182 tests** (146 prior and 36 new), with mocked bridge
operations and no credentials. Cases cover token spoofing/mismatch, cancellation,
stale state, timeout, preparation guards, stat recovery/raw proof, exact supplies,
historical/ranked/wrong-seat receipts and overlapping/case-variant account locks.
The scoped CI lint gate includes both participant modules and their tests.
All primary invariant files and the lockfile remain unchanged. Combat execution,
fixture orchestration, reconnect/resume and promotion stay in later units.

## Unit 4 Combat procedures and option executor

Based on unit 3 (`a58551e`), this ports the stdlib-only combat surface from the
training tip `819f4f4`: `combat_combo.py`, `neural/schema.py` (schema 3, 27
options) and `neural/executor.py` (`pre-aos-options-v12-potion-landing-tracking`),
plus the Weaken/Clumsy opener, explosion-potion procedure, Showdown no-heal rule
and out-of-range approach in `magic.py`. The executor is reached through 14
training commits from `6b8d6f0` to `99f0184`; their behavior is imported with
its tests rather than replayed commit by commit.

`magic.py` was merged by hand, not replaced. The training copy deleted
`NO_CURSOR` and moved the reflection clock to a successful cast, which would
revert `ccbc571` and void the sustain playbook again. This unit keeps both:
buffs never wait for a cursor and every reflection attempt restarts its clock.
Only cursor-targeted spells adopt the wall-clock bound
`max(3, (3 + circle) * 0.25 + 2)` seconds (at most 64 observations), measured
from the first observation after the cast request. The neural option set has
no buff, so the executor cannot reintroduce the reflection bug; a test pins this.

One local hardening beyond the source: the training `explosion_potion_proc`
targeted whichever cursor was pending, so an outstanding harmful-spell cursor
could be spent on the potion. It now targets only the neutral ground-capable
throw cursor (flag 0, type 1). Another described cursor, such as a stale spell
cursor in the frame after the potion was lit, is never targeted; the procedure
keeps waiting for the throw cursor within its existing 12-observation bound so
the lit potion is not abandoned. Boolean-only cursors from older bridges remain
accepted.
`executor.py`, `schema.py` and `combat_combo.py` are byte-identical to the source.

Behavior change for existing callers: when the duel opponent is more than ten
tiles away, `mage_verbs` now offers healing/cure or one walkable step toward the
opponent instead of out-of-range casts. Openers, potions, burst combos and
Showdown restrictions only apply when their memory flag is set.

Every emitted wire command (`CastSpell`, `TargetObject`, `TargetGround`,
`TargetCancel`, `Use`, `UseSkill`, `Walk`, `Attack`, `StunRequest`) is parsed
by current `anima-client` `anima-contract-json`; no bridge or schema change is
needed. `neural/__init__.py` does not import torch, and `inference.py`,
checkpoints and model code are not part of this unit. The six `InferenceClient`
cases of the training `test_neural_runtime.py` were removed and are deferred to
unit 5 with that module.

Offline acceptance: **236 passing tests** (182 prior and 54 new): 33 executor
runtime cases (burst cursor ordering, fizzle/clumsy interrupt before and after
priming, stale-state potion ownership, safe ground disposal, orphan cursor
cleanup, public Explosion word expiry, potion defense, landing tracking), 8
combo, 1 potion, and 12 unit-boundary cases. The boundary cases cover stun
request/arming/landing cooldown and skill/reach requirements, round-boundary
melee reset, stale server state masking, the reflection invariants, cursor waits
beyond the old eight-observation limit and at the wall-clock deadline, Showdown
healing (against a control case), survival abandonment of an opener with its cursor cancelled, and
the potion procedure waiting past a stale spell cursor, never targeting a
persistent foreign cursor and accepting boolean-only cursors. The
suite passed three consecutive runs. The scoped CI lint gate adds all new files
and `magic.py`. Body, contract, lockfile, README and `experiments/run_arm.sh`
are unchanged. No live fight, model, training or server change was run;
real-server timing of these procedures needs a separately authorized fixture.

## Unit 5 Frozen neural inference and checkpoint validation

Based on unit 4 (`74d9632`), this ports `neural/model.py` (byte-identical),
`neural/checkpoints.py` and `neural/inference.py` from the training tip `819f4f4`.
All three originate in `6b8d6f0` and `3ae14e7`; their tests come from the same
commits' `test_neural_runtime.py` (the six deferred `InferenceClient` cases, now
`tests/test_neural_inference.py`) and `test_neural_training.py` (model and
checkpoint round-trip/schema cases, now `tests/test_neural_checkpoints.py`).
Training, PPO, live updates, simulator, league and demonstrations stay in units 6–8.

The source already loaded weights with `torch.load(..., weights_only=True)`,
compared the schema and fingerprint, checked dimensions against the schema,
rejected non-finite tensors and verified the blob against the SHA-256 written
in its own manifest. That last check is self-referential: anyone who rewrites
both blob and manifest passes it. Local hardening:

- `load_checkpoint(..., expected_sha256=...)` and `InferenceClient(...,
  expected_sha256=...)` require the caller's 64-character lowercase SHA-256 of
  the weight file (the identity recorded in `model-evidence.md`). There is no
  default. The manifest must name that digest and the bytes must hash to it.
  The worker receives it as `--expected-sha256`; the parent also refuses a
  worker whose ready line reports another identity or non-positive/non-integer
  dimensions, and a later matching ready line cannot rescue it.
- The manifest is read with a hard size bound, rejects NaN/Infinity and
  non-objects, and its config must have exactly the four integer dimensions.
  Descriptors need an integer, bounded byte count and a bare file name.
- Tensors must be string-keyed, floating-point and finite, and their names and
  shapes must equal those of the declared policy before `load_state_dict`.
  Any deserialization failure, including a pickle payload refused by
  `weights_only`, is reported as `CheckpointError` (a `ValueError`); a test
  proves the payload's reducer is never executed.
- The worker reports a missing `torch` as a startup failure, not a traceback.
- The parent reads at most 2,000,001 characters of any worker line, so an
  oversized response is fatal before it can be buffered in full.
- `save_checkpoint` no longer writes optimizer or RNG blobs, and the
  `restore_optimizer`/`restore_training_rng` helpers are not ported; training
  resume belongs to unit 7. Existing manifests that list those blobs still load
  without reading them (covered by a synthetic-manifest test).

Dependency: a new `neural` extra, `torch>=2.6,<3` (2.6 is the first release
where `weights_only=True` is the default) and `numpy>=2,<3` (as on the source
branch; it only stops torch warning that NumPy is missing). On Linux, the lock
takes torch from the PyTorch CPU index, so CI downloads one wheel (about 0.2 GB)
instead of about 2.9 GB of CUDA packages. `laya` already locked torch 2.14.0 from
PyPI with CUDA on Linux. To leave that untouched, the CPU source applies only to
`neural`, and the two extras are declared conflicting (they cannot share an
environment). After `uv lock` (plus `--upgrade-package torch==2.14.0` so Linux
matches the macOS 2.14.0 rather than jumping to 2.15.0+cpu), the lock gains exactly one
package, `torch 2.14.0+cpu`. No locked version is removed or changed. Fourteen
existing packages gain extra wheel entries for the same versions in the
conflict-fork marker space. `uv export` of `dev+jeff`, `qwen`, `laya` and
`laya+qwen+jeff+dev` is byte-identical before and after.

Importing `anima3`, `anima3.neural`, `executor`, `schema` and `inference` does not
import torch; a subprocess test refuses any torch import through a meta-path hook
and checks `sys.modules`. The current schema fingerprint equals the audited
`cae99f11…0bcb` (schema 3, 76 features, 27 actions). Local copies of the
selected Reference BC (`af3cb4ad…8175`) and rollback (`0c6ca921…f56d`) weight
files under the historical `anima3-training-league/.logs` worktree were only
hashed, and their manifests read as JSON. Both hashes match, and both manifests
carry this schema and fingerprint with config 76/27/64/64. They were not loaded or
modified, and the server copies under `/opt/uoarena` were not accessed.

Offline acceptance: without torch (`--extra dev --extra jeff`), **263 passing,
1 skipped** (236 prior, 27 new inference/import cases; the checkpoint module
skips through `pytest.importorskip`). With `--extra neural`, **302 passing**
(39 model, checkpoint and real-worker cases). The real-worker cases start
`python -m anima3.neural.inference` on a temporary checkpoint, receive a
masked decision and refuse a mismatched identity. Both suites passed three
consecutive local runs. Subprocess tests use generous real deadlines and wait on
events; deadline expiry is exercised by consuming at an explicitly later clock. CI keeps the existing job and adds `neural-inference-tests`
with the same action SHAs, Python 3.12.14 and uv 0.12.17. It asserts a `+cpu`
build with no CUDA/NVIDIA/Triton distributions, fails if any torch case is
skipped, and runs the full suite with torch. The scoped Ruff gate adds all five
new files. Body, contract, README, `experiments/run_arm.sh`, `magic.py`, the
executor and the schema are unchanged.

Not done: no model was promoted, trained, evaluated or run against a game.
Inference latency on the production host, GPU/MPS inference and wiring the client
into a live duel runner are unverified (unit 6). The caller-supplied SHA binds
only the weight file. The manifest itself is not hashed by the caller, although
every manifest field that affects inference is validated.

## Unit 6 Frozen policy in the live duel runner

Based on unit 5 (`97a79de`), this adds `neural/live.py`, which joins the SHA-bound
`InferenceClient` and the unit-4 `CombatExecutor` on one actor thread per game
socket and records only receipt-verified unrated training fixtures. The actor
loop, `dispatch_option`, `policy_override`, `LiveReplayVerifier`,
`starting_template_verified` and `fight` are taken from the training tip
`819f4f4` (`neural/live.py`, 16 commits from `6b8d6f0` to `5672268`). The
collection layer around them is rewritten on the unit-3 primitives rather than
ported, because the source `run` depended on `arena.py`, `arena_learning.py`,
`arena_policy.py`, the training `sparring` handshake and a password CLI.

Left out of the actor: the melee/cast-dump probes (`neural/melee.py`), the
tactical teacher and demonstrations (`neural/curriculum.py`, unit 8). The
scripted arm is the existing scripted `Agent` (`mage_a`, Weaken/Clumsy opener,
explosion potions) built from `decision.build_client("scripted")`.

Boundaries this unit adds:

- **Immutable arm identity.** `Arm("neural", checkpoint, policy_sha256)` requires
  the 64-character weight SHA-256; a scripted arm may carry no checkpoint. The
  worker is started with that SHA; `await_workers` also refuses a ready worker
  reporting another identity, and each actor's reported `policy_sha` must equal
  its arm after the match. Episodes record the policy, plan, schema fingerprint
  and executor SHA. `LiveReplayVerifier(directory, expected_policy_sha)` requires
  both fingerprints to be present and current, always requires a well-formed
  64-hex `policy_sha` (equal to `expected_policy_sha` when given), and by default
  requires a `completed` ledger row for the episode's match and fixture.
- **No silent fallback.** A missing checkpoint manifest or an absent `torch`
  (`importlib.util.find_spec`, so the parent still never imports it) raises
  `NeuralArmUnavailable` before any account is connected, so a scripted policy
  can never be recorded under a neural label. A worker startup failure (for
  example a mismatched checkpoint SHA) is refused the same way within a
  90-second bound. A worker that fails mid-match stops the actor.
- **Mask and stale state reach the runner.** A decision consumed when the
  latest server DuelState is three seconds old or older is recorded with
  `cancellation_reason: context_stale` and never started; a decision whose
  action is no longer legal in the current frame is `mask_changed`. Both keep
  the sampled action (no substitution). Stale state also stops new submissions.
  A rejected decision in the current round (deadline, illegal, policy change)
  makes the episode ineligible.
- **Round reset.** Every new Fighting round gets a fresh epoch that resets the
  executor and the worker's recurrent state. Any other phase resets both to
  `inactive`, so neither an option nor a late decision crosses a round. A
  second round in the one-round training fixture marks `multiple_rounds`.
- **Bounded resume.** `CollectionPlan` fixes endpoint, two case-distinct
  accounts, both arms, 1–100 matches, the inference deadline and the match
  deadline. `collection.json` stores the plan and its SHA-256 once. A second
  run needs `resume=True` and the identical plan (including policy SHAs and the
  budget). `ledger.jsonl` is append-only and fsynced (the directory too, when
  the file is created; every `atomic_json` rename also fsyncs its directory):
  `issued` is written before the challenge, then exactly one of `completed`,
  `invalid_fixture`, `failed` or `unresolved`. An issued fixture without an
  outcome (crash) is closed as `unresolved` on resume and still consumes the
  budget, so no failure path can issue more challenges than planned. Seat
  alternation follows the fixture index across resumes. Every row is validated
  on load (fixture index, event, 32-hex match IDs, episode list, complete
  `completed` rows); out-of-order, malformed or double-closed ledgers raise
  `CollectionRefused`. A torn final line (crash mid-write) is also refused and
  needs manual repair: a torn `issued` row means its challenge was never sent,
  and removing a torn terminal row leaves the fixture to be closed
  `unresolved`. `results.json` is rebuilt from every `completed` row, so a
  resume keeps the matches of earlier runs.
- **Episodes follow the ledger.** All of a fixture's episodes are written to
  `<match>/.staged/`, verified, and moved to their published names only after
  the `completed` row. A refused second neural arm or an interrupted fixture
  therefore publishes nothing; a crash between the row and publication is
  finished by the next `open_collection`.
- **Duplicates and invalid fixtures.** A match identity may occur once per
  collection: each actor refuses a known identity (`DuplicateMatch`) before
  anything is written under its directory, and the ledger and pre-receipt
  checks repeat it. A `failed` row carries the `match_id` once it is known, or
  `duplicate_match_id` for a refused repeat. The receipt goes through
  unit-3 `verify_receipt` (exact ordered seats, rules, unranked training, SHA,
  no abort); a mismatch is recorded as `invalid_fixture` and stops the run
  without writing an episode. Any other failure after a challenge is recorded
  as `failed` and stops the run; nothing is retried automatically. Earlier
  verified matches remain in `results.json` (`automatic_promotion: false`).
- **Account locking and cleanup.** The run holds the unit-3 `account_locks` for
  both accounts across worker start, connection, preparation and all fixtures,
  plus a per-collection `run.lock`. One `ExitStack` closes bodies and workers
  independently (a failing close does not skip the others) before releasing the
  account locks, then the run lock. Only a held `run.lock` is reported as
  another runner; other I/O errors such as a full disk propagate unchanged.

There is no CLI or `__main__` entry: `run_collection(plan, log_dir, connect=...,
fetch_receipt=..., lock_dir=...)` receives the body factory and receipt source
from the caller. `server_receipt(web, timeout=...)` provides a bounded
read-only replay-index poller for a later authorized fixture. The receipt is
fetched while both sockets keep pumping.

Offline acceptance: without torch (`--extra dev --extra jeff`), **348 passing,
2 skipped** (263 prior, 85 new in `tests/test_neural_live.py`; the checkpoint
module and the new `tests/test_neural_live_worker.py` skip through
`pytest.importorskip`). With `--extra neural`, **389 passing** (302 prior, 87
new). The new cases cover the seven ported source cases (the bridge pump case is
already covered by unit 1), arm identity validation, missing checkpoint/torch
refusal before any login, worker identity refusal (including a real
`InferenceClient` whose fake worker serves another SHA), stale-state and
mask-change cancellation, a fresh legal start, round-boundary resets of both
executor and recurrent state, in-round rejection ineligibility, mid-match worker
failure, plan bounds, explicit identical-plan resume, budget exhaustion after an
unresolved fixture, tampered/duplicate/torn ledgers (including a non-string
`match_id`), alternating verified fixtures, failure then resume within budget
with `results.json` keeping every completed fixture, a repeated server match ID
refused before its directory is written, `match_id` on failed rows, a receipt
for other seats, an actor reporting another policy, a held account lock, a held
run lock versus a disk failure, a failing worker close that still closes both
bodies while the account locks are held, a refused second neural arm publishing
nothing, publication after a crash following the `completed` row, episodes with
missing fingerprints, missing or malformed policy, or no `completed` ledger
row, and a torch-free import. With torch, a real worker on a temporary
checkpoint feeds the real executor through `_actor` (the first decision starts
from the reset hidden state and the second continues from the first decision's
`hidden_out`), and a real worker with another identity is refused. Each of
these mutations fails a test: weakening the stale-state check, the inactive
reset, the duplicate-ledger check, the unresolved close, the fingerprint
requirement, the policy-format check, the ledger-built results, episode
staging, or the actor duplicate check. Both suites passed three consecutive local
runs. CI adds both new test files and `live.py` to the scoped Ruff gate, and
the torch-only worker file to the no-skip neural step. Body, contract, README,
lockfile, `experiments/run_arm.sh`, `magic.py`, executor, schema, inference and
checkpoint modules are unchanged.

Not done: no live server, client, bridge, fight, training or promotion was run.
Real-server timing of the actor loop, inference latency on the production host,
receipt availability timing, reconnect behaviour and the account/endpoint
aliasing caveat of unit 3 remain for a separately authorized live fixture with
its own budget. Evaluation statistics (win-rate confidence, per-style
regression) and PPO consumption of the episodes belong to unit 7; the teacher
and probes to unit 8.

## Unit 7 Evaluation statistics, selection gate and verified PPO

Based on unit 6 (`9765dc2`), this unit adds three modules. It changes `live.py`
in one place.

- `neural/evaluation.py`: fixed-budget evaluation statistics and a selection
  gate that fails closed. Stdlib only.
- `neural/splits.py`: the training/held-out split over unit-6 collections.
  Stdlib only.
- `neural/train.py`: explicit, bounded recurrent PPO on verified training
  episodes. Needs torch.
- `live.py`: `CollectionPlan` gains a `purpose` field.

Importing `anima3`, `anima3.neural`, `evaluation` or `splits` does not import torch.
A subprocess test refuses any torch import. `train.py` is never imported
implicitly, and it has no CLI, no `__main__` and no schedule.

### Source provenance

| Source at `819f4f4` | Imported | Rewritten or left out |
| --- | --- | --- |
| `neural/train.py` (`6b8d6f0`, `3ae14e7`) | `TrainConfig`, `Trajectory`, `generalized_advantage`, `clipped_policy_loss`, `_targets`, `_chunks`, `_batch`, `ppo_update`, `_live_trajectory` and `_episode_actor`, unchanged apart from the budget bounds and the `optimizer_steps` metric below | `update_verified_episodes` is rewritten: SHA-bound load, explicit config and seed, a new output directory, no optimizer restore. `collect_simulator`, `behavioral_clone`, `evaluate` and the CLI are left out. |
| `neural/league.py` `promotion_gate` (`94051a2`) | The paired bootstrap: blocks of both sides, 2000 resamples, `random.Random(619)`, order statistics 49/1949. Also kept: the domain checks, the duplicate/mismatched-pair checks, the loss/draw/win score check, invalid evidence failing the gate, and the per-style regression limit. | Rewritten as `selection_gate` around an explicit `EvaluationPlan`. Styles become planned opponents; excluded fixtures, a minimum number of paired games, Wilson intervals and a decision record are added. `evaluate_suite` and the league `main` need the simulator and teacher and are left out. |
| `neural/online.py` `update_collection(s)` (`6b8d6f0`, `94051a2`) | Ineligible and off-policy episodes are rejected and recorded. A collection batch needs a minimum size. Duplicate episodes are refused. | Rewritten as `splits.select_training_episodes` and `train.train_from_collections`. They read the ledger rather than `results.json` and check the collection's purpose. `cycle_main`, which collected and trained in a loop, is not ported. |
| `anima3/stats.py` (already on primary) | `wilson`, reused by `wilson_interval` | — |

### Boundaries

- **Training and evaluation splits.** `CollectionPlan.purpose` is either
  `evaluation`, the default, or `training`. It is part of the plan JSON and its
  SHA-256, and it is copied into `results.json` and every episode. The default is
  held out, so a collection becomes training input only by saying so
  explicitly. Adding the field changes the plan SHA-256 of every unit-6 plan; no
  unit-6 collection had been run.
  - `TrainingEpisodeVerifier` refuses an evaluation collection outright.
  - It also needs the episode's purpose and plan SHA-256 to match the
    collection's recorded and re-hashed plan, and needs `LiveReplayVerifier` to
    accept the episode for the behaviour SHA-256, including a `completed` ledger
    row.
  - `select_training_episodes` reads only episodes listed in `completed` rows.
    Their paths must stay inside the match directory.
  - Ineligible and other-policy episodes are kept in `rejected` with their
    reasons. An eligible episode that fails verification refuses the whole batch.
  - Optional `held_out` evaluation collections must not share a match identity
    with the training input. A collection cannot be on both sides.
  - In the other direction, `record_from_collection` refuses a training
    collection.
- **Fixed plan, no growth.** `EvaluationPlan` fixes, before any result is read:
  - the candidate and incumbent SHA-256s, which must differ;
  - 1–16 opponents (`scripted` or `neural:<sha>`);
  - an even number of games per opponent, from 2 to 100;
  - `max_regression` (per opponent, default 0.10);
  - `min_lower_bound` (default 0);
  - `max_excluded_per_opponent` (default 0);
  - `min_paired_games_per_opponent` (default 20, the source's minimum per style);
  - the domain.

  Its SHA-256 identifies the decision. Each policy/opponent pair is exactly one
  held-out collection whose own plan budget equals the evaluation plan's and
  whose ledger has spent the budget with no open fixture. Refused as an interim
  result or a grown count: a partly spent collection, a second collection for
  the same opponent, a missing opponent, a different budget, or rows outside the
  fixture range or duplicated. Unit 6 already makes a collection's budget
  unchangeable on resume.
- **Failed and excluded fixtures stay in the record.** `failed`,
  `invalid_fixture` and `unresolved` ledger outcomes become `excluded` rows with
  their reason, and the decision lists them. They are never re-scored. Beyond
  the allowance they fail the gate, as does falling below the minimum number of
  paired games.
- **Outcomes come from receipts.** A completed row's score (1 win, 0.5 draw,
  0 loss) is re-read from the stored SHA-checked replay and its seat, not from
  the episode. `valid` needs the full unit-6 episode verification. Invalid
  evidence fails the gate, as in the source.
- **Win-rate confidence.** `wilson_interval` (exact z = 1.95996…) reports each
  policy's wins over completed games, with draws counted as non-wins, for every
  opponent. Mean score, draws and exclusions are reported alongside.
- **Average is not superiority.** The gate passes only when every one of these
  holds:
  - the paired-bootstrap 95% lower bound of the mean score difference is above
    `min_lower_bound`;
  - no opponent's mean paired difference is below `-max_regression`;
  - the exclusion and minimum-pairing rules hold;
  - all evidence is valid.

  Tests cover two failing cases: a candidate that improves the average by 0.35
  but drops 0.15 against one opponent, and a +0.075 average whose interval spans
  zero.
- **Decisions are recorded, never acted on.** Every decision carries
  `automatic_promotion: false`, `public_deployment: false` and
  `real_server_strength_established: false`. Its outcome is
  `candidate_passed_gate` or `retain_incumbent`, never `promoted`.
  `record_decision` appends passed and rejected decisions alike to an fsynced
  JSONL record and refuses a second decision for the same plan SHA-256. It
  also refuses a torn final line and any decision claiming automatic promotion.
- **Explicit budgets and seeds.** `update_verified_episodes` and
  `train_from_collections` require a `TrainConfig` and an integer seed; neither
  has a default.
  - `TrainConfig` now needs integer `epochs` (1–64), `sequence_length`
    (1–4096), `burn_in` and `batch_sequences` (1–1024).
  - A batch is 1–200 episodes, each with at most 50,000 transitions, and at most
    1,000,000 transitions in total.
  - `ppo_update` reports `optimizer_steps`, which never exceeds
    epochs × ⌈chunks/batch⌉. The KL early stop can only lower it.
  - The update runs inside `torch.random.fork_rng`, seeded explicitly, and
    PPO shuffling uses the same seed. A fixed seed gives bit-identical weights
    on the same platform and torch build, and leaves the caller's global RNG
    untouched.
- **Immutable, SHA-bound checkpoints.** The behaviour checkpoint is loaded by
  the caller's expected SHA-256 (unit 5).
  - Every episode is re-derived against those exact weights before any
    parameter changes: action likelihood, value, recurrent state chain, legal
    mask, single terminal and accepted actions.
  - The output directory must not exist and must not be the input, inside it, or
    contain it. It is created only after every check passes.
  - The new manifest records the parent SHA, config, budget, seed, metrics,
    consumed `(match, actor)` identities, `optimizer_state:
    fresh-adam-per-update` and `automatic_promotion: false`.
  - The new checkpoint is re-loaded by its own weight SHA-256 before
    `training-update.json` is written. The input tree is byte-identical
    afterwards.
  - `live_consumed` carries over the lineage, so a descendant cannot train on
    the same match/seat twice.
  - The source restored the parent's Adam state. That needs the optimizer blobs
    left out in unit 5, so each update now starts a fresh Adam.

Offline acceptance:

| Environment | Result | Change |
| --- | --- | --- |
| Without torch (`--extra dev --extra jeff`) | **406 passing, 3 skipped** | 348 prior plus 58 new evaluation/split cases. The new training module skips through `pytest.importorskip`. |
| With `--extra neural` | **494 passing** | 389 prior, 58 evaluation and 47 training cases |

Both suites passed three consecutive local runs.

The evaluation cases cover:

- Wilson intervals against the published Newcombe (1998) Table I values:
  81/263, 15/148, 0/20, 1/29, plus 8/10, 5/10 and 10/10;
- deterministic bootstrap;
- plan bounds;
- a pass;
- the single-opponent regression and lower-bound failures above;
- equal evidence;
- excluded rows and the minimum-pairing rule;
- invalid evidence;
- growth, shrinkage and duplicates;
- malformed scores (bool, NaN, 0.75);
- domain and policy mismatch;
- the append-only decision record.

Real unit-6 `run_collection` collections, built with the mocked participant
layer, feed:

- receipt scoring with seat alternation;
- a failed fixture kept after resume;
- interim refusal;
- refusal of training collections for evaluation and of evaluation collections
  for training, including a relabelled episode;
- held-out overlap;
- tampered receipts, plan SHA-256s, purposes and relabelled plans;
- a dropped `completed` row;
- path escape;
- the end-to-end gate.

The training cases are tiny CPU loops on synthetic data:

- the ported GAE, clipping, forced-hold normalisation, critic-only and KL
  early-stop cases;
- the ported live fail-closed cases (other policy, rejected action,
  log-probability, hidden chain, unfinished episode, argmax), plus evaluation
  or unlabelled episodes;
- budget and seed refusal;
- output refusal (existing, equal to or inside the input);
- SHA-recorded output with the input unchanged;
- fixed-seed determinism and an untouched global RNG;
- consumed identities and both seats;
- the 200-episode cap;
- `train_from_collections` end to end on real unit-6 training collections whose
  transitions come from the real checkpoint, with a held-out evaluation
  collection and refusal of a tampered receipt.

Each of these mutations fails a test:

- accepting an evaluation collection for training;
- skipping (instead of refusing) an unverified eligible episode;
- dropping the held-out overlap check;
- disabling the per-opponent regression limit;
- removing both the fixture-range and the exact-count checks;
- accepting a partly spent collection;
- gating on the point estimate instead of the bootstrap lower bound;
- ignoring the exclusion allowance;
- accepting an evaluation episode in the trainer;
- allowing output inside the input or into an existing directory;
- not passing the seed to PPO shuffling;
- removing the epoch cap;
- removing purpose validation.

CI adds the five new files to the scoped Ruff gate and `test_neural_training.py`
to the neural step that must not skip. The lockfile and `pyproject.toml` are
unchanged. Body, contract, README, `experiments/run_arm.sh`, `magic.py`,
executor, schema, inference, checkpoint and model modules are also unchanged.

Not done:

- No model was trained on real episodes, evaluated, selected or promoted.
- No server, client or bridge was used.
- A real evaluation needs a separately authorized plan: held-out evaluation
  collections for the incumbent and the candidate against each opponent, with
  budgets of at least 20 games, fixed before play.
- So does any real training run, on explicitly `training` collections.

Deferred:

- The approximate simulator, BC on teacher demonstrations, league self-play and
  the simulator gates of `league.py`. They depend on `sim.py`, `curriculum.py`
  and `demonstrations.py`, so they belong with the teacher (unit 8) or a later
  simulator unit. Simulator gain is not live gain, and the gate refuses to
  compare different domains.
- Optimizer and RNG resume across updates.
- Bootstrap pairing exploits seed correlation only when both evaluations share
  fixtures. For live collections it pairs by opponent, fixture position and
  seat, which is valid for independent games but no tighter than an unpaired
  comparison.

## Remaining separately verified units

| Order | Unit and likely files | Dependencies and acceptance |
| --- | --- | --- |
| 2 (integrated) | Read-only replay evidence: `replay.py`, validation/metrics tests | Stdlib-only first; corrupt, truncated, reordered and wrong-match records must fail. No network writes or model promotion. |
| 3 (integrated) | Arena participant preparation and sparring: `arena.py`, `duel_wait.py`, `sparring.py`, public command/receipt tests | Units 1–2; preserve ordinary-account and challenge-token checks, start templates, exact rules, bounded fixtures, account locks and failure receipts. Reconcile current peer-only server assumptions. |
| 4 (integrated) | Combat procedures and executor: selected `magic.py`/`combat_combo.py`, neural executor/schema surface | Units 1 and 3. Carry reflection fix forward explicitly while adopting wall-clock cursor handling; test stun, potion ownership/disposal, recovery and delayed public effects. No training required. |
| 5 (integrated) | Frozen neural inference and checkpoint validation | Unit 4; add the minimum Torch optional dependency and regenerate/review the lock deliberately. Require dimensions, schema fingerprint, finite weights and SHA-bound loading. Preserve audited selected and rollback artifacts. |
| 6 (integrated) | Live receipts, evaluation and bounded resume: `neural/live.py` | Units 2–5; prove count/duplicate/invalid-fixture handling, immutable arm policy identities and account locking before any separately authorized live fixture. |
| 7 (this unit) | Evaluation statistics, selection gate and verified PPO: `neural/evaluation.py`, `neural/splits.py`, `neural/train.py` | Units 5–6; isolate training from evaluation, preserve holdouts, fixed budgets, elapsed-time discounting and per-opponent regression rejection. No training, evaluation or promotion is run. |
| 7b | Simulator, demonstrations, BC and league self-play: `sim.py`, `demonstrations.py`, `league.py` | Units 7–8 (teacher); seeded simulator evaluation through the unit-7 gate, demonstrations never PPO episodes, optimizer/RNG resume idempotence. Simulator gain is not live gain. |
| 8 | Tactical teacher and optional Jev shadow analysis | Unit 4 plus relevant collectors. Keep teacher demonstrations and off-policy actions out of PPO records; preserve evaluated frozen policy behavior. |
| 9 | Historical experiment record reconciliation | Preserve chronology and failed candidates; attach completed evidence without interpreting registry selection as public deployment. |

These are dependency boundaries, not permission to launch training or replace
workers. Large originating commits should be split by the table's behavior and
its tests, not cherry-picked wholesale simply because their title matches.

`feat/arena-100-duels` also has the extra commit `baaaaef` (remaining 17 fixtures
and scripted-probe distinction) absent from the training tip. Preserve it in the
historical-record unit instead of assuming `training-league` subsumes every
worktree. The independently audited selected Reference BC result and current
compatibility smoke are recorded in the parent workspace's
`docs/model-evidence.md` and `docs/model-readiness.md`.

Before a subsequent unit is merged, repeat the primary-invariant checks above,
run its offline suite from the locked environment, and record the exact source
and validation results in its PR. Live fixture budgets and model promotion are
separate acceptance decisions.
