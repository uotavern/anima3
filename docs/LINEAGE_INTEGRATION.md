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
| `neural/train.py` (`6b8d6f0`, `3ae14e7`) | `Trajectory`, `generalized_advantage`, `clipped_policy_loss`, `_targets`, `_chunks`, `_batch`, `ppo_update`, `_live_trajectory` and `_episode_actor`. `TrainConfig`'s coefficients and checks are kept. | `TrainConfig`'s budget fields and discount clock lose their defaults. `ppo_update` needs an explicit seed and reports `optimizer_steps`. `update_verified_episodes` is rewritten: it takes a `TrainingSelection`, loads by SHA, and needs explicit config and seed. It writes a new output directory and never restores an optimizer. `collect_simulator`, `behavioral_clone`, `evaluate` and the CLI are left out. |
| `neural/league.py` `promotion_gate` (`94051a2`) | The paired bootstrap: complete two-sided seed blocks, 2000 resamples, `random.Random(619)`, sorted indices 49 and 1949 (the 50th and 1950th order statistics). Also kept: the domain checks, the duplicate/mismatched-pair checks, the loss/draw/win score check, invalid evidence failing the gate, the per-style regression limit and the 20-games-per-style minimum. | Rewritten as `selection_gate` around an explicit, canonical `EvaluationPlan`. Styles become planned opponents. Added: seat checks, candidate-penalising exclusions, a block floor, Wilson intervals, `decide_from_collections` and an append-only decision record that re-derives each decision. `evaluate_suite` and the league `main` need the simulator and teacher and are left out. |
| `neural/online.py` `update_collection(s)` (`6b8d6f0`, `94051a2`) | Ineligible and off-policy episodes are rejected and recorded. A collection batch needs a minimum size. Duplicate episodes are refused. | Rewritten as `splits.select_training_episodes` and `train.train_from_collections`. They read the ledger rather than `results.json`, check the collection's purpose and require an explicit held-out set. `cycle_main`, which collected and trained in a loop, is not ported. |
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
  - `held_out` is required, with no default. It lists the evaluation
    collections that must not share a match identity with the training input,
    or is the explicit `NO_HELD_OUT`. A collection cannot be on both sides.
  - In the other direction, `record_from_collection` refuses a training
    collection.
- **Fixed, canonical plan with floors.** `EvaluationPlan` fixes, before any
  result is read:
  - the candidate and incumbent SHA-256s, which must differ, and neither of
    which may be an opponent;
  - 1–16 opponents (`scripted` or `neural:<sha>`), stored sorted;
  - an even number of games per opponent, from 20 to 100;
  - `max_regression` (per opponent, default 0.10) and `min_lower_bound`
    (default 0), both on a 0.01 grid and normalised, so reordering opponents or
    a 1e-12 nudge gives the same plan SHA-256;
  - `max_excluded_per_opponent` (default 0, at most a tenth of the games);
  - `min_paired_games_per_opponent` (default and floor 20);
  - the domain.

  Each policy/opponent pair is exactly one held-out collection whose own plan
  budget equals the evaluation plan's and whose ledger has spent the budget with
  no open fixture. Refused as an interim result or a grown count: a partly
  spent collection, a second collection for the same opponent, a missing
  opponent, a different budget, or rows outside the fixture range or
  duplicated. Unit 6 already makes a collection's budget unchangeable on resume.
- **Seats and blocks.** Every completed row states its seat (challenger or
  receiver).
  - Games pair by opponent and fixture, and a pair from different seats is
    refused.
  - Fixtures 2k and 2k+1 form a block that must hold both seats.
  - As in the source, only complete two-delta blocks enter the bootstrap, so
    every resampled block weighs the same.
  - With fewer than ten complete blocks in total, no interval is computed and
    the incumbent is retained. The plan floors already require ten per
    opponent; only exclusions can bring the count below that.
- **Failed and excluded fixtures stay in the record, scored against the
  candidate.** `failed`, `invalid_fixture` and `unresolved` ledger outcomes
  become `excluded` rows with their reason, and the decision lists them.
  - An excluded candidate game counts as a candidate loss.
  - An excluded incumbent game drops that fixture, so the candidate gains
    nothing from the incumbent's failure.
  - The choice is deliberately asymmetric in the incumbent's favour.
  - Fixtures excluded on either side beyond the allowance fail the gate, as do
    fewer than the minimum completed pairs.
- **Outcomes come from receipts.** A completed row's score (1 win, 0.5 draw,
  0 loss) is re-read from the stored SHA-checked replay and its seat, not from
  the episode. `valid` needs the full unit-6 episode verification. Invalid
  evidence fails the gate, as in the source.
- **Win-rate confidence.** `wilson_interval` (exact z = 1.95996…) reports each
  policy's wins over completed games, with draws counted as non-wins, for every
  opponent. Mean score, draws and exclusions are reported alongside.
- **Average is not superiority.** The gate passes only when every one of these
  holds:
  - the paired-bootstrap 95% lower bound of the block mean difference is above
    `min_lower_bound`;
  - no opponent's mean difference is below `-max_regression`;
  - the exclusion, minimum-pairing and block rules hold;
  - all evidence is valid.

  Tests cover two failing cases: a candidate that improves the average by 0.35
  but drops 0.15 against one opponent, and a +0.075 average whose interval spans
  zero.
- **Decisions are re-derived and decided once, never acted on.** Every decision
  carries `automatic_promotion: false`, `public_deployment: false` and
  `real_server_strength_established: false`. Its outcome is
  `candidate_passed_gate` or `retain_incumbent`, never `promoted`.
  - `decide_from_collections` re-derives both evaluations from their collection
    directories. The decision names each collection by path, plan SHA-256,
    ledger SHA-256 and server match IDs. It also states `external_candidate`.
  - `record_decision(lineage_dir, decision)` writes
    `<lineage_dir>/selection.jsonl`. Everything below runs under an exclusive
    lock on that lineage, including the re-derivation.
  - It re-hashes the plan, checks that `passed`, the outcome and `reasons`
    agree, and re-derives the decision from the named collections. The result
    must be identical.
  - **Lineage binding.** The candidate checkpoint is reached by SHA-256 through
    the collections' arms: the manifest must name that weight SHA, and the
    weight file must hash to it. A relative checkpoint path is refused, because
    it would resolve against the current directory. The manifest is read with
    a 1 MiB bound.
    - `train_from_collections` requires a `lineage_dir` and records it in the
      child manifest. Only that lineage may record a decision on the child.
    - A checkpoint without the field, such as an externally supplied one, is
      refused unless the decision declares `external_candidate: true`. A bound
      checkpoint cannot be declared external.
  - **Claims, created all or nothing before the row is appended.** Each is an
    `O_CREAT|O_EXCL` file, fsynced along with its directory. It names the
    decision's SHA-256, the plan SHA-256, the resolved record path, the row's
    sequence number and the candidate/incumbent SHAs.
    - Exclusive creation also settles a race between two lineages, whose locks
      do not serialise each other.
    - If another decision's claim exists, the decision is refused and the
      claims just created are removed.
    - A rollback that cannot remove a claim raises an error naming every claim
      left behind.

    There are three kinds of claim:
    - `selection-claim.json` in every evaluation collection directory used, so
      the same evidence can never be decided again in any lineage. A copied
      collection carries its claim with it.
    - `selection-claims/<incumbent>.json` in the candidate checkpoint
      directory, so the same candidate/incumbent pair can never be decided
      again in any lineage, even with fresh collections or after the lineage
      was reset.
    - `claims/<sequence>.json` in the lineage directory, one per row. Rows and
      claims must agree. Losing more than the final row, or reordering rows,
      is refused as tampering.
  - **Crash between claims and row.** A crash after the claims but before the
    append leaves every claim of that decision in place with no row. A lost
    final row looks the same. The decision's claim at `claims/<n>.json` (n is
    the row count) blocks the lineage with "interrupted decision at sequence
    n", and its pair and evidence claims block them everywhere.
    - Retrying the identical decision is idempotent: under the lock, if every
      one of its claims exists and matches it exactly, the claims are adopted
      and the row is appended at sequence n. Matching means the same decision
      SHA-256, plan, record path, sequence and pair.
    - Any other decision is refused with the interrupted-decision message.
    - A crash part-way through claiming leaves only some claims. A retry is
      then refused as "interrupted decision: only some of its claims exist",
      listing the missing ones.
    - **Repair:** remove exactly the claim files that name the interrupted
      decision's `decision_sha256` (pair claim, collection claims, lineage
      claim n), then retry the decision.
  - Within a record, a second decision for the same pair, or one reusing a
    collection ledger or server match, is also refused. Collection plan
    SHA-256s are recorded, but identical plans share one, so the ledger and
    match identities make evidence single-use. A new comparison against the
    same incumbent therefore needs fresh incumbent collections.
  - It also refuses a torn final line, a non-object line, an unreachable or
    mis-hashed candidate checkpoint, and any decision claiming automatic
    promotion.
  - Passed and rejected decisions are appended alike and never rewritten.
  - **Residual risk.** The claims are files under the same trust boundary as the
    collections and checkpoints. Someone with write access who deletes claim
    files can reuse evidence: from a collection copy with its claim removed,
    after deleting the checkpoint's pair claim, or after deleting a whole
    lineage directory. A copied checkpoint whose manifest has
    `training.lineage_dir` edited or stripped keeps the same weight SHA,
    because that SHA covers only the weights, not the manifest. Such a copy
    escapes the lineage binding: stripped, it can be declared external; edited,
    it can be recorded in another lineage. Its pair claim still lands in the
    copy's own directory, not the original's. No remote or signed registry
    exists. Claims are also sidecar files in the candidate checkpoint
    directory; weights and manifest are never modified.
- **Explicit budgets, discount clock and seeds.** `TrainConfig` has no default
  for `epochs` (1–64), `sequence_length` (1–4096), `burn_in`, `batch_sequences`
  (1–1024) or `discount_time_unit`. `TrainConfig()` is a `TypeError`.
  - The source's live update used `discount_time_unit=1.0` (`gamma **
    elapsed_seconds`), and the caller must now state it. The manifest records
    `gamma`, `gae_lambda`, `discount_time_unit` and the rule applied.
  - `ppo_update` has no default seed. `update_verified_episodes` and
    `train_from_collections` need an explicit integer seed.
  - A batch is 1–200 episodes, each with at most 50,000 transitions, and at most
    1,000,000 transitions in total.
  - `optimizer_steps` never exceeds epochs × ⌈chunks/batch⌉. The KL early stop
    can only lower it.
  - Training runs on CPU only, inside `torch.random.fork_rng(devices=[])`,
    seeded explicitly. `fork_rng` isolates only the CPU generator, so a model on
    another device is refused. A fixed seed gives bit-identical weights on the
    same platform and torch build, and leaves the caller's global RNG untouched.
- **The trainer verifies for itself.** `update_verified_episodes` accepts only a
  `TrainingSelection` from `select_training_episodes` for the loaded policy.
  - Its verifiers must be that policy's `TrainingEpisodeVerifier` instances.
  - Every episode is re-verified through them; a caller-supplied callable is
    never trusted.
  - The callable-based path is `_unsafe_update_from_episodes`, private and
    documented as unsafe. Only tests use it.
- **Immutable, SHA-bound checkpoints.** The behaviour checkpoint is loaded by
  the caller's expected SHA-256 (unit 5).
  - Every episode is re-derived against those exact weights before any
    parameter changes: action likelihood, value, recurrent state chain, legal
    mask, single terminal and accepted actions.
  - The output path must not exist and must not be a symbolic link, even a
    dangling one.
  - Once resolved, it must not be the input, inside it, or contain it.
  - It is created only after every check passes.
  - The new manifest records the parent SHA, config, budget, discount, seed,
    metrics, consumed `(match, actor)` identities, `optimizer_state:
    fresh-adam-per-update` and `automatic_promotion: false`.
  - The new checkpoint is re-loaded by its own weight SHA-256 before
    `training-update.json` is written. The input tree is byte-identical
    afterwards.
  - `live_consumed` carries over the lineage, so a descendant cannot train on
    the same match/seat twice.
  - The source restored the parent's Adam state. That needs the optimizer blobs
    left out in unit 5, so each update now starts a fresh Adam.

### Review round 1

The independent review of `9bea5d1` requested changes. Each item below has a
test and a mutation check.

1. A two-game plan, or four games, passed on a degenerate bootstrap. Fixed with
   plan floors and the ten-block minimum.
2. A decision could be made again on the same evidence. Fixed with the
   canonical plan, one decision per pair and single-use evidence within a
   record; the record file name is derived from the lineage directory. Round 2
   showed this still depended on callers reusing one directory, and closed it.
3. A forged or inconsistent decision could be recorded. Fixed with the
   re-hash, the outcome/reasons check, re-derivation from the collections, the
   lock, refusal of non-object lines and collection plan SHA-256s in the
   evidence.
4. Seats were never paired. Fixed with the seat requirement and two-seat
   blocks.
5. Training budgets and the discount clock had defaults. They are now
   required, and the discount is recorded.
6. The held-out set was optional. It is now required.
7. Exclusions went unpenalised. They are now capped and scored against the
   candidate.
8. The trainer trusted the caller's verifier. It now takes a selection, and
   the callable path is private.
9. Nits: the order-statistic comment, an incumbent listed as an opponent, the
   default `ppo_update` seed, and CUDA (now CPU only).

The reviewer's second probe also showed that a dangling symbolic link as the
output path redirected the write. Such a link is now refused.

### Review round 2

The re-review of `341c571` verified all nine fixes. Two issues remained.

- **Medium: the per-lineage location was only a convention.** The reviewer's
  probe re-recorded the same decision into another lineage directory. It also
  recorded fresh collections for a decided pair into a third directory, and
  reset a lineage by deleting `selection.jsonl`. Fixed by:
  - collection evidence claims;
  - a pair claim in the candidate checkpoint directory;
  - per-row lineage claims;
  - binding the candidate to its lineage via the training manifest, with an
    explicit `external_candidate` declaration for unbound checkpoints.

  Every case of the probe is now refused, with a bound candidate and with a
  declared external one.
- **Low: the re-derivation ran outside the lock.** It now runs under the lock.

### Review round 3

The third review of `0afcb22` confirmed that the claims, the lineage binding
and the rollback work, and that nothing regressed. It asked for three changes,
each now tested with a mutation check:

- **Medium:** a crash between claims and the row orphaned the claims behind a
  misleading tampering error. Fixed with the idempotent retry and the distinct
  interrupted-decision refusals described above. Leftover claims after a failed
  rollback are now reported.
- **Low:** a relative checkpoint path resolved against the current directory.
  It is now refused. The stripped-manifest copy is documented as residual risk.
- **Nit:** the manifest size bound is now enforced on the read itself.

Every case of the reviewer's `test_probe_r3.py` is refused except the retry of
the identical decision, which succeeds at sequence 0. Round 2's notes were
first committed under Unit 1 by mistake and now sit here.

### Offline acceptance

| Environment | Result | Change |
| --- | --- | --- |
| Without torch (`--extra dev --extra jeff`) | **427 passing, 3 skipped** | 348 prior plus 79 evaluation/split cases. The training module skips through `pytest.importorskip`. |
| With `--extra neural` | **520 passing** | 389 prior, 79 evaluation and 52 training cases |

Both suites passed three consecutive local runs.

The evaluation cases cover:

- Wilson intervals against the published Newcombe (1998) Table I values:
  81/263, 15/148, 0/20, 1/29, plus 8/10, 5/10 and 10/10;
- deterministic bootstrap;
- plan bounds and floors;
- canonical plan identity;
- a pass;
- the single-opponent regression and lower-bound failures above;
- equal evidence;
- too few blocks;
- excluded rows, candidate-penalising exclusions and the allowance cap;
- seat mismatch and single-seat blocks;
- invalid evidence;
- growth, shrinkage and duplicates;
- malformed scores (bool, NaN, 0.75);
- domain and policy mismatch.

Real unit-6 `run_collection` collections, built with the mocked participant
layer and 20 games per opponent, feed:

- receipt scoring with seat alternation;
- a failed fixture kept after resume;
- interim refusal;
- the end-to-end gate;
- the record, covering forged plan SHA, inconsistent outcome, altered
  statistics, promotion claims, missing evidence, an invalid plan, pairs
  re-decided under reordered, nudged or loosened plans, reused incumbent
  evidence, the lock (including re-derivation under it), a non-object line and
  a torn line;
- the round-2 probe: same evidence in another lineage, fresh evidence for a
  decided pair in the same or another lineage, a deleted, emptied or
  claim-stripped record, a missing external declaration, a bound candidate
  declared external, mis-hashed candidate weights, claim rollback when a later
  claim fails, and a copied collection keeping its claim;
- the round-3 cases: a subprocess crash after all claims, retried at sequence
  0, with different decisions refused; a crash part-way through claiming,
  reported and repaired; a rollback whose unlink fails, naming the leftover
  claim; a concurrent claim from another lineage winning by exclusive
  creation; a restored final row versus a deleted record; and a relative or
  oversized candidate checkpoint;
- refusal of training collections for evaluation and of evaluation collections
  for training, including a relabelled episode;
- the required held-out argument and overlap;
- tampered receipts, plan SHA-256s, purposes and relabelled plans;
- a dropped `completed` row;
- path escape;
- a selection that trusts only its own verifiers.

The training cases are tiny CPU loops on synthetic data:

- the ported GAE, clipping, forced-hold normalisation, critic-only and KL
  early-stop cases;
- the ported live fail-closed cases, plus evaluation or unlabelled episodes;
- required budget and discount fields with the recorded discount, and seed
  refusal;
- the required lineage directory recorded in the child manifest;
- output refusal (existing, inside or equal to the input, symbolic links,
  dangling links);
- the CPU-only refusal;
- SHA-recorded output with the input unchanged;
- fixed-seed determinism and an untouched global RNG;
- consumed identities and both seats;
- the 200-episode cap;
- the public update refusing anything but a real selection for this policy,
  including an episode altered after selection;
- `train_from_collections` end to end on real unit-6 training collections, and
  refusal of a tampered receipt.

Mutations of the original guards, each of which fails a test:

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

Mutations of the review fixes, each of which also fails a test:

- the games, paired-games and block floors;
- opponent sorting and threshold normalisation;
- one decision per pair and single-use evidence;
- the plan re-hash, the outcome/reasons check and the re-derivation;
- the lock and non-object line refusal;
- the seat, block-seat and seat-required checks;
- the candidate-loss exclusion scoring and the allowance cap;
- refusing the incumbent as an opponent;
- required `epochs` and `discount_time_unit`, and the recorded discount;
- the required `ppo_update` seed and `held_out` argument;
- the selection and verifier type checks, both in the update and in
  `TrainingSelection.verify`;
- the symbolic-link refusal;
- the CPU-only check.

Mutations of the round-2 fixes, each of which also fails a test:

- creating no collection claims or no pair claim;
- non-exclusive claim creation;
- no rollback;
- no lineage claim per row;
- no record-versus-claims check;
- no lineage binding;
- no external declaration;
- no candidate weight hash;
- no re-derivation under the lock;
- the child manifest not recording its lineage.

Both the in-record pair check and the pair claim are needed: removing either
fails a test.

Mutations of the round-3 fixes, each of which also fails a test:

- not adopting matching claims on retry;
- accepting partial claims;
- no distinct interrupted-row refusal;
- claims not naming the exact decision;
- the rollback not reporting leftovers;
- accepting relative checkpoint paths;
- no manifest size bound;
- allowing more than one orphaned lineage claim;
- non-exclusive claim creation.

CI adds the five new files to the scoped Ruff gate and `test_neural_training.py`
to the neural step that must not skip. The lockfile and `pyproject.toml` are
unchanged. Body, contract, README, `experiments/run_arm.sh`, `magic.py`,
executor, schema, inference, checkpoint and model modules are also unchanged.

Not done:

- No model was trained on real episodes, evaluated, selected or promoted.
- No server, client or bridge was used.
- A real evaluation needs a separately authorized plan: fresh held-out
  evaluation collections for the incumbent and the candidate against each
  opponent, with budgets of at least 20 games, fixed before play, and one
  fixed lineage directory, which a trained candidate names in its manifest.
- So does any real training run, on explicitly `training` collections with a
  stated held-out set.

Deferred:

- The approximate simulator, BC on teacher demonstrations, league self-play and
  the simulator gates of `league.py`. They depend on `sim.py`, `curriculum.py`
  and `demonstrations.py`, so they belong with the teacher (unit 8) or row 7b.
  Simulator gain is not live gain, and the gate refuses to compare different
  domains.
- Optimizer and RNG resume across updates.
- GPU training.
- Bootstrap pairing exploits seed correlation only when both evaluations share
  fixtures. For live collections it pairs by opponent, fixture position and
  seat, which is valid for independent games but no tighter than an unpaired
  comparison.

## Unit 8 Tactical teacher, probes, demonstrations and imitation

Based on unit 7 (`daca1dc`), this unit adds four modules and changes `live.py`,
`evaluation.py` and (after review round 1) the unit-7 `train.py`:

- `neural/teacher.py`: the scripted `TacticalTeacher` and its identity
  `TEACHER_SHA256`. Stdlib only.
- `neural/melee.py`: the scripted melee and cast-dump probes. Stdlib only.
- `neural/demonstrations.py`: receipt-bound verification and training-only
  selection of teacher demonstrations. Stdlib only.
- `neural/imitation.py`: explicit, bounded behaviour cloning (BC) on a verified
  demonstration selection. Needs torch.
- `live.py`: `teacher` and `probe` arm kinds, their actor branches and
  demonstration recording.
- `evaluation.py`: refuses a teacher or probe arm as an evaluation opponent.
- `train.py`: PPO children carry the ancestry's `demonstrations_consumed`, a
  trained parent without its consumed record is refused, and
  `train_from_collections` loads its checkpoint inside `fork_rng`.

Importing `anima3`, `anima3.neural`, `live`, `teacher`, `melee` or
`demonstrations` does not import torch. A subprocess test refuses any torch
import. `imitation.py` is never imported implicitly, and it has no CLI, no
`__main__` and no schedule.

### Scope against rows 7b and 8

Row 8 named the tactical teacher. Unit 6 deferred the probes and the teacher's
demonstrations to it, and unit 7 deferred BC on teacher demonstrations to "unit 8
or row 7b". This unit takes the part that needs only the teacher and live
receipts: demonstrations recorded by the live runner and BC on them. Row 7b
keeps everything that needs the simulator: `sim.py`, the drill helper
`setup_drill`, simulator rollouts (`collect_simulator`), simulator BC, league
self-play, the simulator gates of `league.py`, and optimizer/RNG resume. The
optional Jev shadow analysis is not ported (see Deferred).

### Source provenance

| Source at `819f4f4` | Imported | Rewritten or left out |
| --- | --- | --- |
| `neural/curriculum.py` `TacticalTeacher`, `STYLES` (`6b8d6f0` to `5672268`) | The class body is byte-identical. | Moved to `teacher.py`, whose `TEACHER_SHA256` hashes the teacher and every module its choice depends on, without the drill helper. `setup_drill` and `DRILLS` need `sim.py` and stay with row 7b. |
| `neural/melee.py` `MeleeProbe`, `CastDumpProbe` | Byte-identical. | — |
| `neural/live.py` probe and `opponent_style` actor branches | The branch bodies, the `fast_timing`/`hold_explosion` choice, `teacher_action_rejected` and `potion_defense_outside_policy_schema`. | Selected by an explicit arm instead of CLI flags. A substituted procedure (`policy_override` event) in a teacher round also marks it ineligible. The source wrote `demonstration-*.json` straight to its published name with no plan, purpose, teacher identity or ledger. Here it is staged, verified and published only after the `completed` row. |
| `neural/demonstrations.py` `load_demonstration` | The step checks (shape, finite range, legal action, accepted, increasing clock), the receipt checks and the supply-audit checks. | Rewritten as `DemonstrationReceiptVerifier` and `select_demonstrations`. They check the plan's teacher arm for that seat and read the supply audit from the `issued` ledger row, not from a `prepared-supplies.json` that later fixtures overwrite. The trajectory conversion is `imitation.demonstration_trajectory`. |
| `neural/demonstrations.py` `imitation_metrics`, `neural/train.py` `behavioral_clone` | Ported, with the forced-hold exclusion, action balancing and the frozen-reference KL. | `behavioral_clone` takes an `ImitationConfig` with no default budget and an explicit seed, refuses PPO trajectories and non-finite losses. |
| `scripts/clone_live_demonstrations.py`, `scripts/fit_collected_demonstrations.py` | The fresh-output rule, the match-disjoint validation and the duplicate refusal. | Rewritten as `clone_from_demonstrations`. Validation is an explicit list of training collections (not "last seat pair per style"), the start checkpoint is loaded by SHA, and the output records full provenance. The scripts, `run_tactical_pilot.py`, `run_melee_probe.py` and their CLIs are not ported. |
| `scripts/jev_shadow_probe.py`, `report_jev_shadow.py` | — | Not ported (see Deferred). |

### Boundaries

- **A teacher or probe is never labelled neural.** `Arm("teacher",
  variant=<style>, source_sha256=...)` and `Arm("probe", variant=<tactic>,
  source_sha256=...)` carry no checkpoint and no policy SHA-256.
  - The variant is one of the six teacher styles or ten probe variants.
  - `source_sha256` is required and must equal `TEACHER_SHA256` or
    `PROBE_SHA256`. Each is the SHA-256 of the canonical JSON map from module
    to module SHA-256 for the controller and every module its choice depends
    on: `teacher.py` or `melee.py`, plus `neural/schema.py` (`legal_mask`,
    `ACTION_INDEX`, `CombatView`), `neural/executor.py` (`cast_seconds`),
    `magic.py` (the spell circles behind `cast_seconds`) and `contract.py`
    (imported by both). The schema fingerprint covers only feature and action
    names, not this logic. `preflight_arm` checks the identity before any
    login, and before a fresh run writes `collection.json`. A change to any of
    these modules therefore refuses a plan or a resume made with the old
    identity (`ArmUnavailable`, the new base of `NeuralArmUnavailable`).
  - The self-reported Arena label is `teacher <style>-<sha12> scripted` or
    `probe <tactic>-<sha12> scripted`. Neural and scripted labels are unchanged.
  - Each actor reports its `policy_kind`, and the runner refuses a fixture
    whose actor kind differs from its planned arm, or whose non-teacher seat
    reports teacher steps.
  - `evaluation.record_from_collection` refuses a teacher or probe opponent.
    It is not scored under a neural label or any other; the plan has no label
    for it yet.
  - Unset `variant` and `source_sha256` are left out of the plan JSON, so every
    neural/scripted plan keeps its unit-7 SHA-256 (a test pins this).
- **Plans.** A collection now needs at least one neural, teacher or probe arm
  (a scripted pair is still refused). A plan with a teacher arm needs
  `require_base_template=True`. With that flag the `issued` row now carries
  both participants' pre-challenge supply audits, for neural plans too.
- **Actor behaviour.** The neural and scripted seats run exactly as in unit 6.
  The teacher seat labels a choice every 0.25 s only from fresh (under 3 s),
  ready, living state. Every labelled choice is kept, including a rejected
  one, which makes the round ineligible (`teacher_action_rejected`). Incoming
  potion defense (`tactical` only) acts outside the option schema and makes
  the round ineligible. So does any `policy_override` event, which is local
  hardening. The probe seat records neither steps nor transitions. Its raw
  `Attack`, `StunRequest` and `Use` packets are the source's and are logged.
- **Demonstrations are never PPO records.** A demonstration has its own schema
  (`anima3-live-demonstration-v2`), `policy_kind: teacher`, label
  `teacher:<style>`, `ppo_eligible: false` and no `policy_sha`.
  - It is listed in the `completed` row's `demonstrations`, never in
    `episodes`, with the file's SHA-256 in `demonstration_sha256`. A row with
    demonstrations but no matching digests is malformed.
  - `LiveReplayVerifier` refuses it, and the unit-7 episode reader refuses a
    `demonstration-` file even when it is listed under `episodes`.
  - `ppo_update` refuses demonstration trajectories, and `behavioral_clone`
    refuses anything else.
  - Each demonstration is staged with the episodes and verified before the
    row. A refused one closes the fixture `invalid_fixture` and publishes
    nothing. A crash after the row is finished by the next `open_collection`,
    as for episodes.
- **Provenance is checked, not trusted.** `DemonstrationReceiptVerifier`
  re-reads the collection on every call and requires:
  - exactly the demonstration fields (an added `policy_sha` is refused), and
    exactly the fields of the `teacher`, `verification` and `final_result`
    blocks;
  - the current schema and executor fingerprints, `collector_version`, and a
    `collection_end_phase` of `Idle` or `RoundOver`;
  - a well-formed teacher block: name, style, `TEACHER_VERSION`, and a module
    hash map (`sources`) over exactly the teacher modules that hashes to its
    SHA-256;
  - that style and SHA-256 equal the planned arm of that seat (the seat
    follows from the `issued` row and fixture parity), and `opponent_arm`
    equals the planned arm of the other seat;
  - the plan SHA-256 and purpose of the re-hashed collection plan, and an
    integer fixture (a bool is refused);
  - both seats of the `issued` row and the stored replay;
  - the replay itself (bounded to 34 MiB, SHA-checked, unranked training, not
    aborted, training rules, the server's starting template) and its outcome;
  - verified, equal supply audits for exactly those two participants in the
    `issued` row;
  - legal, accepted, finite steps with a strictly increasing clock;
  - by default, a `completed` row of that fixture listing this file.
- **Training-only selection with kept rejections.**
  `select_demonstrations(collections, teacher_sha256=..., held_out=...,
  styles=None)` requires:
  - absolute paths;
  - an explicit teacher SHA-256;
  - `held_out`: evaluation collections or `NO_HELD_OUT`, as in unit 7. A
    shared match identity refuses the selection, and a collection cannot be on
    both sides.
  - Evaluation collections are refused as input.
  - Every file read is bounded (64 MiB each, 512 MiB in total for one
    selection, rejected files included) and must hash to its ledger digest.
    A selection accepts at most 200 demonstrations.
  - Other-teacher, other-style and ineligible demonstrations are kept in
    `rejected` with their reasons. An eligible demonstration that fails
    verification refuses the whole selection, as does a duplicate
    `(match, actor)`.
  - Each accepted one is recorded with its file, file SHA-256, collection,
    plan SHA-256, fixture, match, seat, replay SHA-256, teacher block, opponent
    arm and step count.
- **Explicit, bounded imitation.** `ImitationConfig` has no default for
  `epochs` (1–64), `sequence_length` (1–4096), `batch_sequences` (1–1024) or
  `balance_actions`.
  - `clone_from_demonstrations` requires:
    - the start checkpoint's SHA-256;
    - `held_out`;
    - `validation`: training collections or `NO_VALIDATION`. They are scored
      before and after, never fitted, re-verified like the input, refused if
      the ancestry already imitated them, and may share neither a collection nor a
      match with the input.
    - `teacher_sha256`, `seed`, `lineage_dir` and `min_demonstrations`
      (1–200), all with no default;
    - absolute checkpoint, output, lineage and reference paths.
  - A reference checkpoint, its SHA-256 and a positive KL coefficient go
    together.
  - At most 200 demonstrations and 1,000,000 steps.
  - CPU only. Training, and each checkpoint load, run inside
    `torch.random.fork_rng`. A fixed seed gives bit-identical weights, and the
    caller's global RNG is untouched.
  - Each demonstration is re-verified through the selection's own collection
    verifiers. It must still equal the bytes read at selection time (same
    SHA-256), so an in-memory edit or a rewritten file is refused.
- **Immutable, SHA-bound output.** The output path follows the unit-7 rules: it
  must not exist, must not be a symbolic link and must not overlap the input.
  - The child manifest records:
    - `source: servuo-teacher-imitation`, the parent SHA, `lineage_dir`;
    - the teacher block (name, SHA-256, styles, `policy_kind: teacher`);
    - config, budget, seed and metrics;
    - before/after imitation metrics on the input and the validation set;
    - the reference SHA, collections and held-out set;
    - every demonstration's provenance, every rejection, and the validation
      provenance;
    - `ppo_input: false`, `real_server_strength_established: false`,
      `optimizer_state: fresh-adam-per-update`.
  - The output may not equal, sit inside or contain the start or reference
    checkpoint or any training, validation or held-out collection, with
    symbolic links resolved.
  - `demonstrations_consumed` and `live_consumed` are carried by every child,
    imitation and PPO alike (`train.ancestry_consumed`). A parent's lists
    therefore already cover its whole ancestry, so BC on D, then PPO, then BC
    on D is refused. A trained checkpoint (one naming `parent_policy_sha`)
    missing either list is refused rather than read as an empty record.
  - The child is re-loaded by its own SHA-256 before `imitation-update.json`
    is written. The input trees are byte-identical afterwards.
  - Because it records `lineage_dir`, only that lineage's `record_decision`
    can later decide on it, as for a PPO child. Nothing here evaluates,
    selects or promotes it.
- **Crash windows.** Collection staging, publication and resume are unit 6's.
  A BC crash before `imitation-update.json` leaves a partial output directory
  that a retry refuses as existing. Delete that directory, then retry. Nothing
  else was written.

What is enforced and what is convention:

- **Enforced in code:** everything listed above.
- **Convention: single use only within an ancestry.** A demonstration may be
  imitated again in an unrelated lineage, for example a hyperparameter sweep
  from the same parent. Only the ancestry's `demonstrations_consumed` stops
  reuse. There are no cross-lineage claims as in unit 7, because imitation is
  not a selection decision.
- **Trust boundary.** Hashes detect corruption; they are not signatures. Anyone
  with write access to a collection can rewrite a demonstration, its ledger
  digest and the issued supply audit consistently. A copied checkpoint whose
  manifest is edited to drop a consumed entry keeps its weight SHA (the SHA
  covers weights only), as in unit 7. The teacher identity covers the listed
  modules only, not the Python interpreter or standard library. Arena labels
  are self-reported.
- **Unbounded reads.** `Ledger.load` reads the whole ledger without a size
  bound. This is unchanged from unit 6.

### Review round 1

The independent review of `98d5023` requested changes. The reviewer's repro
(BC on D, then PPO, then BC on D accepted) is refused now. Each item has a test
and a mutation check.

1. **Medium: a PPO step reset the demonstration record.** The unit-7 trainer
   wrote no `demonstrations_consumed`. Every PPO and imitation child now
   carries both consumed lists, and `ancestry_consumed` refuses a trained
   parent missing either list. `train_from_collections` also loads its
   checkpoint inside `fork_rng` now (the unit-7 follow-up).
2. **Medium: the teacher identity bound only `teacher.py`.** `choose()`
   depends on `magic.SPELLS` circles (through `executor.cast_seconds`) and on
   `schema.legal_mask`. `TEACHER_SHA256` and `PROBE_SHA256` now hash a
   per-module map over the controller, `schema.py`, `executor.py`, `magic.py`
   and `contract.py`. Each demonstration records the map, and the verifier
   checks that the map gives the identity. A test edits copies of `magic.py`,
   `schema.py` and `executor.py` and sees the identity change.
3. **Low: validation demonstrations were neither checked against the
   ancestry nor re-verified.** Both are done now.
4. **Low: loose demonstration fields.** `opponent_arm` must equal the plan.
   `teacher.version`, `collector_version` and `collection_end_phase` are
   checked. Unknown keys in `teacher`, `verification` and `final_result` are
   refused.
5. **Low: output overlap.** The BC output may not equal, sit inside or contain
   the reference checkpoint or any training, validation or held-out
   collection, symbolic links included.
6. **Low: unbounded total reads.** A selection accepts at most 200
   demonstrations and reads at most 512 MiB, rejected files included.
7. **Low: preflight after the manifest.** `preflight_arm` now runs before a
   fresh run writes `collection.json`. A refused run leaves no manifest,
   ledger or `results.json`. The unit-6 absent-torch test now checks that
   instead of a `stopped` results file.

Test gaps filled:

- a busy executor accepts a held teacher choice without starting anything,
  and records any other choice as rejected;
- a teacher-versus-teacher plan is allowed and tested: both seats write
  demonstrations, each naming the other seat's planned arm.

Mutations of the round-1 fixes, each of which fails a test (17):

- the PPO child dropping the demonstration record;
- a stripped trained parent accepted;
- the `train_from_collections` load advancing the global RNG;
- the teacher identity over `teacher.py` only, or without `magic.py`;
- the source map not tied to the identity;
- unknown teacher keys allowed;
- no teacher version check;
- no end-phase check;
- unknown verification keys allowed;
- no opponent-arm check;
- no total read budget;
- no selection count cap;
- validation not checked against the ancestry;
- validation not re-verified;
- no output-overlap check;
- preflight after `collection.json`.

All 37 earlier mutations were re-run and are still killed. One of them, "no
teacher-versus-plan-arm check", first survived: the new module-map check also
caught the old tampered SHA. A self-consistent identity for other teacher code
was added as a test case, and that mutation is killed again.

### Offline acceptance

| Environment | Result | Change |
| --- | --- | --- |
| Without torch (`--extra dev --extra jeff`) | **523 passing, 4 skipped** | 427 prior plus 96 teacher/demonstration cases. The imitation module skips through `pytest.importorskip`. |
| With `--extra neural` | **636 passing** | 520 prior, 96 teacher/demonstration, 18 imitation and 2 new unit-7 training cases |

These are the counts after review round 1; the first push had 509/4 and 616.

Both suites passed three consecutive local runs.

The torch-free cases (`tests/test_neural_teacher.py`) cover:

- the ported teacher priorities: interrupts before openers or bursts, both dump
  finishers with reserves, low-mana hysteresis that never chips, distinct
  burst and potion styles;
- legal choices for every style on six views;
- the five ported probe cases plus the cast-dump sequence;
- the teacher identity over its module map, with no drill helper in it
  (round 1: edits to `magic.py`, `schema.py` or `executor.py` change it);
- eleven malformed teacher/probe/neural arms;
- non-neural labels;
- the unchanged unit-7 plan SHA-256;
- teacher plan rules;
- a changed teacher or probe source refused before any login.

They also cover the actor and the runner:

- teacher steps never becoming transitions;
- rejected actions kept and making the round ineligible;
- potion defense and `policy_override` marking it;
- no labels from unready state;
- a probe seat recording nothing;
- one policy per seat;
- real `run_collection` teacher collections (mocked participants, real ledger,
  receipts and verifier): demonstrations beside episodes, with digests and
  supply audits;
- `LiveReplayVerifier` and PPO selection refusing demonstrations;
- thirty tampered provenance fields;
- six supply-audit faults;
- a wrong server starting template;
- a missing `completed` listing;
- a kept rejected round;
- other teacher and other style;
- evaluation collections refused for imitation and as a teacher opponent;
- the held-out argument, overlap, relative paths, duplicate collections and
  missing teacher SHA;
- a tampered file caught by the ledger digest, and one re-digested but caught
  by verification;
- path escape;
- crash publication on resume;
- a refused staged demonstration;
- an actor kind mismatch;
- four malformed digest rows;
- a torch-free import.

The imitation cases (`tests/test_neural_imitation.py`) are tiny CPU loops:

- the four ported BC cases: BC-only demonstrations refused by PPO, forced
  holds excluded from metrics, action balancing, reference regularisation;
- the required budget and seed, refusing PPO trajectories;
- end to end on real teacher collections: SHA-bound output with full
  provenance, unchanged input trees, kept rejections and validation;
- fixed-seed determinism with an untouched global RNG;
- refusals: evaluation input or validation, validation overlap, relative
  paths, missing `held_out`/`validation`, existing or nested output, an unmet
  minimum, an untied reference, a wrong start SHA;
- the recorded reference identity;
- lineage re-imitation;
- in-memory and on-disk edits after selection;
- foreign verifiers;
- the CPU-only check.

Mutations, each of which fails a test (37 at first push, run one at a time,
and re-run after review round 1):

- a teacher label reported as neural;
- no teacher source SHA preflight;
- a teacher arm without the base template;
- rejected teacher actions, potion defense or teacher `policy_override` not
  marked;
- no actor policy-kind check;
- staged demonstrations not verified;
- demonstrations not published on resume;
- no ledger digest requirement;
- no supply audit in the issued row;
- a demonstration allowed to carry a policy SHA;
- no teacher-versus-plan-arm check;
- no label check;
- no `ppo_eligible: false` check;
- no supply verification;
- no starting template;
- no `completed` listing;
- a bool fixture accepted;
- a non-strict step clock;
- evaluation collections imitated;
- no held-out overlap check;
- relative paths accepted;
- rejections not kept;
- a tampered eligible demonstration skipped instead of refusing the batch;
- no ledger digest comparison;
- re-verification ignoring the file SHA;
- validation overlap allowed;
- lineage re-imitation allowed;
- the start checkpoint load advancing the global RNG;
- the BC shuffle ignoring the seed;
- BC accepting PPO trajectories;
- no verifier type check;
- no CPU check;
- no reference tie;
- a default for `balance_actions`;
- a teacher scored as an evaluation opponent.

CI adds the six new files to the scoped Ruff gate and `test_neural_imitation.py`
to the neural step that must not skip. The lockfile and `pyproject.toml` are
unchanged. Body, contract, README, `experiments/run_arm.sh`, `magic.py`,
executor, schema, inference, checkpoint, model and splits modules are also
unchanged; `train.py` changes only as described in review round 1.

Not done:

- No teacher, probe or neural policy was run against a server, client or
  bridge. Real-server timing of the teacher and probe branches, potion-defense
  behaviour and the raw probe packets need a separately authorized fixture.
- No demonstration was collected, no model was cloned on real data, and
  nothing was evaluated, selected or promoted. A real imitation run needs a
  separately authorized plan: `training` teacher collections with
  `require_base_template`, a stated held-out set and validation set, a fixed
  budget and seed, and the lineage directory the child will be judged in.

Deferred:

- Row 7b: `sim.py`, `setup_drill`/`DRILLS`, simulator rollouts and BC, league
  self-play, the simulator gates, and optimizer/RNG resume.
- Evaluation against a teacher opponent needs a plan label for it (for example
  `teacher:<style>:<sha>`); until then it is refused.
- The optional Jev shadow analysis (`scripts/jev_shadow_probe.py`). It calls a
  remote language-model service with credentials, which the offline-only rule
  excludes, and controls nothing. Its recorded result
  (`2026-10-11-jev-shadow-final.json`) was no better than the prevalence
  baseline: the 95% interval of the Brier difference spans zero. A port needs
  its own authorization and a mocked client.

## Remaining separately verified units

| Order | Unit and likely files | Dependencies and acceptance |
| --- | --- | --- |
| 2 (integrated) | Read-only replay evidence: `replay.py`, validation/metrics tests | Stdlib-only first; corrupt, truncated, reordered and wrong-match records must fail. No network writes or model promotion. |
| 3 (integrated) | Arena participant preparation and sparring: `arena.py`, `duel_wait.py`, `sparring.py`, public command/receipt tests | Units 1–2; preserve ordinary-account and challenge-token checks, start templates, exact rules, bounded fixtures, account locks and failure receipts. Reconcile current peer-only server assumptions. |
| 4 (integrated) | Combat procedures and executor: selected `magic.py`/`combat_combo.py`, neural executor/schema surface | Units 1 and 3. Carry reflection fix forward explicitly while adopting wall-clock cursor handling; test stun, potion ownership/disposal, recovery and delayed public effects. No training required. |
| 5 (integrated) | Frozen neural inference and checkpoint validation | Unit 4; add the minimum Torch optional dependency and regenerate/review the lock deliberately. Require dimensions, schema fingerprint, finite weights and SHA-bound loading. Preserve audited selected and rollback artifacts. |
| 6 (integrated) | Live receipts, evaluation and bounded resume: `neural/live.py` | Units 2–5; prove count/duplicate/invalid-fixture handling, immutable arm policy identities and account locking before any separately authorized live fixture. |
| 7 (integrated) | Evaluation statistics, selection gate and verified PPO: `neural/evaluation.py`, `neural/splits.py`, `neural/train.py` | Units 5–6; isolate training from evaluation, require an explicit held-out set, fixed budget floors, an explicit discount clock (`discount_time_unit` has no default; the source live update used 1.0, elapsed seconds), seat-paired blocks, single-use evidence and per-opponent regression rejection. No training, evaluation or promotion is run. |
| 7b | Simulator, simulator BC and league self-play: `sim.py`, `setup_drill`, `league.py` | Units 7–8; seeded simulator evaluation through the unit-7 gate, simulator rollouts never live PPO episodes, optimizer/RNG resume idempotence. Live demonstrations and BC on them moved to unit 8. Simulator gain is not live gain. |
| 8 (this unit) | Tactical teacher, probes, live demonstrations and BC: `neural/teacher.py`, `neural/melee.py`, `neural/demonstrations.py`, `neural/imitation.py` | Units 4, 6 and 7. Teacher and probe arms are never neural; demonstrations stay out of PPO records with verified provenance from training collections only; frozen-policy behaviour unchanged. The optional Jev shadow analysis is deferred (remote model service). |
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
