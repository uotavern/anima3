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

Offline acceptance: **233 passing tests** (182 prior and 51 new): 33 executor
runtime cases (burst cursor ordering, fizzle/clumsy interrupt before and after
priming, stale-state potion ownership, safe ground disposal, orphan cursor
cleanup, public Explosion word expiry, potion defense, landing tracking), 8
combo, 1 potion, and 9 unit-boundary cases. The boundary cases cover stun
request/arming/landing cooldown and skill/reach requirements, round-boundary
melee reset, stale server state masking, the reflection invariants, cursor waits
beyond the old eight-observation limit and at the wall-clock deadline, Showdown
healing and survival abandonment of an opener with its cursor cancelled. The
suite passed three consecutive runs. The scoped CI lint gate adds all new files
and `magic.py`. Body, contract, lockfile, README and `experiments/run_arm.sh`
are unchanged. No live fight, model, training or server change was run;
real-server timing of these procedures needs a separately authorized fixture.

## Remaining separately verified units

| Order | Unit and likely files | Dependencies and acceptance |
| --- | --- | --- |
| 2 (integrated) | Read-only replay evidence: `replay.py`, validation/metrics tests | Stdlib-only first; corrupt, truncated, reordered and wrong-match records must fail. No network writes or model promotion. |
| 3 (integrated) | Arena participant preparation and sparring: `arena.py`, `duel_wait.py`, `sparring.py`, public command/receipt tests | Units 1–2; preserve ordinary-account and challenge-token checks, start templates, exact rules, bounded fixtures, account locks and failure receipts. Reconcile current peer-only server assumptions. |
| 4 (this unit) | Combat procedures and executor: selected `magic.py`/`combat_combo.py`, neural executor/schema surface | Units 1 and 3. Carry reflection fix forward explicitly while adopting wall-clock cursor handling; test stun, potion ownership/disposal, recovery and delayed public effects. No training required. |
| 5 | Frozen neural inference and checkpoint validation | Unit 4; add the minimum Torch optional dependency and regenerate/review the lock deliberately. Require dimensions, schema fingerprint, finite weights and SHA-bound loading. Preserve audited selected and rollback artifacts. |
| 6 | Live receipts, evaluation and bounded resume | Units 2–5; prove count/duplicate/invalid-fixture handling, immutable arm policy identities and account locking before any separately authorized live fixture. |
| 7 | Simulator, demonstrations, BC/PPO and promotion gates | Units 5–6; isolate training from evaluation, preserve holdouts, elapsed-time assumptions, resume idempotence and per-style regression rejection. Simulator gain is not live gain. |
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
