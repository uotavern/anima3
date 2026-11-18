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

## Remaining separately verified units

| Order | Unit and likely files | Dependencies and acceptance |
| --- | --- | --- |
| 2 | Read-only replay evidence: `replay.py`, validation/metrics tests | Stdlib-only first; corrupt, truncated, reordered and wrong-match records must fail. No network writes or model promotion. |
| 3 | Arena participant preparation and sparring: `arena.py`, `duel_wait.py`, `sparring.py`, public command/receipt tests | Units 1–2; preserve ordinary-account and challenge-token checks, start templates, exact rules, bounded fixtures, account locks and failure receipts. Reconcile current peer-only server assumptions. |
| 4 | Combat procedures and executor: selected `magic.py`/`combat_combo.py`, neural executor/schema surface | Units 1 and 3. Carry reflection fix forward explicitly while adopting wall-clock cursor handling; test stun, potion ownership/disposal, recovery and delayed public effects. No training required. |
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
