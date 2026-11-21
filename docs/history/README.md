# Training lineage experiment records

This directory preserves the experiment records of the Anima3 training lineage
(integration unit 9, [UO workspace issue 11](https://github.com/uotavern/uo/issues/11)).
Records are historical evidence, not instructions. Commands in them refer to modules,
scripts and server paths of that time, and are not part of the current package.

- `lineage/` holds byte-exact copies. Each copy keeps its source path below this root,
  so the records still link to each other.
  - 76 files are read from `feat/training-league` at `819f4f4`.
  - `docs/experiments/2026-10-02-hundred-live-duels.md` is read from
    `feat/arena-100-duels` at `baaaaef`, published as the archive branch
    `archive/arena-100-duels-baaaaef`. That commit only appends the section on the
    remaining 17 matches. The training-tip version is a byte prefix of this file, and its
    length and SHA-256 are recorded.
- `index.json` lists, for every copy, its source ref, commit and path, the commit that
  introduced it, byte count, SHA-256 and Git blob ID. It also lists 27 experiments in
  chronological order, every gate decision found in the JSON records, the parent-workspace
  evidence, and every file the lineage changed that was not imported, except the
  `anima3/` and `tests/` code that units 1–8 and row 7b cover.
- `tests/test_history_records.py` checks the index against the files offline.

Nothing here was edited. A finding that a record does not contain is stated in
`index.json` or below, never written into the record.

## Status vocabulary

The index keeps three questions apart:

1. What did the experiment show (`result`)?
2. Was a candidate chosen, and by what (`selection`)?
3. Did a policy serve public players (`deployment`)?

| Field | Values |
| --- | --- |
| `result` | `passed_gate`, `failed_gate`, `simulator_only`, `negative`, `diagnostic`, `invalidated` |
| `selection` | `registry_selected`, `registry_retained`, `rollback`, `offline_selected`, `rejected`, `not_gated`, `invalidated` |
| `deployment` | `not_deployed`, `public_deployment` |

`registry_selected` means only that an experimental registry named the checkpoint as
`selected_sha`. Simulator runs keep a registry of their own, local to that run. Live
evaluations update the experimental live registry. Neither kind of registry switches a
public worker. `public_deployment` needs a `deployment_evidence` entry that names an
indexed record of kind `deployment_evidence`. No such record exists, so every
experiment and candidate is `not_deployed`.

Live matches in these records are unranked training matches between training
accounts on the arena host. The lineage text calls the `/opt/uoarena` snapshots that ran
them "deployed". They served no public player.

## Experiments

Times are the UTC author time of the commit that first recorded the experiment.

| ID | Recorded (UTC) | Experiment | Result | Candidates (SHA-256 prefix, selection) |
| --- | --- | --- | --- | --- |
| E01 | 2026-09-29 21:18 | Ten-match Jev strategy adaptation | negative | — |
| E02 | 2026-09-30 05:50 | Recurrent duel policy starter and first live update | diagnostic | `7c0cbe1a` not_gated; 150k-step run rejected; `2953462a` not_gated |
| E03 | 2026-09-30 06:53 | Simulator v5 continuation and four live updates | simulator_only | `22bc333e` offline_selected; `46fb2317`, `558a7dde` rejected; `fe5796b4` not_gated |
| E04 | 2026-10-01 19:00 | One hundred live matches with per-match PPO | negative | `4fd33a57`, `ee89aab5` not_gated |
| E05 | 2026-10-05 08:53 | Melee, stun and close Harm mechanics probe | diagnostic | — |
| E06 | 2026-10-05 10:03 | Cursor-clock cast dump timing probe | diagnostic | — |
| E07 | 2026-10-05 16:10 | Simulator curriculum league v2 | simulator_only | `39abd306`, `e5915b8d` registry_selected (run-local); PPO rejected |
| E08 | 2026-10-06 07:34 | 28-match live PPO pilot | failed_gate | `ec9b2a7c` rejected; `e5915b8d` registry_retained |
| E09 | 2026-10-07 13:51 | Schema v3 skill-clock simulator retraining | simulator_only | `0c6ca921` registry_selected (run-local); PPO rejected |
| E10 | 2026-10-07 13:56 | Skill-clock live pilot and first live imitation | invalidated | `03b06cc7`, `d0096ef9` invalidated |
| E11 | 2026-10-07 14:30 | Supply correction acceptance | diagnostic | — |
| E12 | 2026-10-07 14:41 | Unseen-opponent movement deadlock diagnosis | diagnostic | — |
| E13 | 2026-10-07 14:54 | Supply-corrected BC and PPO trial (v7) | failed_gate | `5e56ab68`, `3d0f7b45` rejected |
| E14 | 2026-10-08 01:17 | 40-game frozen v8 BC screen | failed_gate | `5e56ab68` rejected; `0c6ca921` registry_retained |
| E15 | 2026-10-08 01:25 | Threat-memory simulator retraining | failed_gate | `772c0fc6` registry_selected (run-local); PPO rejected |
| E16 | 2026-10-08 01:30 | PPO with BC rehearsal | failed_gate | `772c0fc6` registry_selected (run-local); PPO rejected |
| E17 | 2026-10-08 01:38 | Elapsed-time discount PPO | failed_gate | `772c0fc6` registry_selected (run-local); PPO rejected |
| E18 | 2026-10-08 01:42 | Elapsed-time PPO with rehearsal | failed_gate | `772c0fc6` registry_selected (run-local); PPO rejected |
| E19 | 2026-10-08 01:47 | Direct continuation of the incumbent | failed_gate | continuation rejected; `0c6ca921` registry_retained |
| E20 | 2026-10-08 04:52 | Recovered BC and 200-game live evaluation | failed_gate | `5e03b624` rejected; `0c6ca921` registry_retained |
| E21 | 2026-10-08 14:41 | Preregistered 400-game replication | failed_gate | `5e03b624` rejected; `0c6ca921` registry_retained |
| E22 | 2026-10-09 11:00 | Fresh teacher collection and reference-regularised BC | diagnostic | `0b3fb0a1`, `af3cb4ad` not_gated |
| E23 | 2026-10-09 14:02 | Reference BC simulator comparisons | simulator_only | `0b3fb0a1` rejected; `af3cb4ad` offline_selected |
| E24 | 2026-10-09 14:07 | Reference BC fresh 200-game live evaluation | passed_gate | `af3cb4ad` registry_selected; `0c6ca921` rollback |
| E25 | 2026-10-10 16:48 | Jev shadow prediction probe | negative | — |
| E26 | 2026-10-10 17:02 | Public-observation tactical controller pilot | diagnostic | — |
| E27 | 2026-10-11 11:46 | Incoming potion defence and mana recovery | diagnostic | — |

Each experiment in `index.json` gives its hypothesis, configuration, outcome, primary
files, shared ledger files and any ServUO change made on the arena shard during it.

The JSON records hold 48 gate decisions: 36 failed and 12 passed. Nine of the passes
are simulator gates: BC against a random initial policy, comparisons against weaker
baselines, or the 300-second simulator gate of E23. The other three are the single E24
live gate, stated in the registry, as an independent recomputation and in the
training-completion audit. The rejected candidates stay in the record beside the one
that passed.

Experiments overlapped in time. The order is the order in which they were first
recorded. Some candidates had no SHA-256 written into their record. They are listed
with `sha256: null` rather than given an invented identity.

## The one passing live result

E24 is the only passing live result. The Reference BC weights
`af3cb4ad…8175` beat the incumbent `0c6ca921…f56d` by +0.165 (95% interval
[+0.045, +0.285]) in 200 fresh games against five scripted styles. The potions style
fell by 0.05, inside the −0.10 floor. The experimental registry selected the candidate,
kept the incumbent as rollback and recorded `public_deployment: false`.

The parent workspace adds completed evidence. It is private, so the index records its
commit, path, SHA-256 and the asserted values only:

- **P1** `evidence/reference-bc-20261010-summary.json`: a read-only re-audit on 2026-11-17
  of all 200 receipts. It found the same selected SHA and the same registry SHA-256
  (`e41538c1…`), and recorded public deployment false.
- **P3** `evidence/model-readiness-20261117.json`: a two-game compatibility smoke on
  2026-11-17. It is compatibility evidence, not a strength estimate. It made no training
  update, left the historical registry unchanged and recorded public deployment false.
- **P2**, **P4**: `docs/model-evidence.md` and `docs/model-readiness.md`, the matching
  narratives.

## Deployment finding

Scope of the finding:

- **No neural policy was publicly deployed.**
- **Non-neural workers were publicly hosted.** Before the peer arena (2026-09-27 and
  2026-09-28), public mage and warrior workers ran as services on the public shard. The
  mage chose among five scripted playbooks, and the warrior used a fixed rule
  (`docs/ARENA_VALIDATION.md`, `docs/ARENA_LINODE.md` at `819f4f4`).
- **The playbook learner never made a real promotion.** The `arena-learner` unit ran
  with `--promote`, but it reported `collecting_training` with `baseline-v1` still
  active. Its promotion and rollback logic was tested only with synthetic evidence.
- The peer-arena handoff (`docs/ARENA_PEER_HANDOFF.md`) keeps the learner and every
  hosted worker unit stopped and disabled.

The evidence that no neural policy was deployed:

- Every `public_deployment` field in the imported JSON is false.
- `league-v2.md` says the registry is experimental and that public agents are never
  replaced by its commands.
- Each checkpoint README disclaims public promotion or replacement of the public agent.
- The parent deployment map of 2026-11-17 (P5) shows no `arena-worker` or
  `arena-learner` service running. Only that finding is recorded here, not the
  map's content.

Limits of this finding:

- Stored flags and files on disk do not prove what a running process loaded.
- ServUO patches were deployed to the arena shard during E08, E11 and E20. Those are game
  server changes, not policies.

## Not imported

`index.json` (`not_imported`) records each of these by location and SHA-256:

- the 12 checkpoint blobs of the four schema-v1 `models/` directories (6.2 MB in total;
  they stay on `origin/feat/training-league`). Their manifests and READMEs are imported.
- the 13 untracked `registry.json` files of the historical training worktree. Four of them
  hash to the registry digests that the imported records quote.
- remote run data under `/opt/uoarena` (replays, episodes, receipts, server checkpoints),
  which this offline unit did not access.
- the third-party Pluto binary analysis and its tool.
- the arena operating documents (`docs/ARENA*.md`). Two of them carry the hosted
  non-neural worker record cited above.
- `deploy/` units and configuration, including the legacy hosted worker and the
  `--promote` learner.
- `scripts/` runner and summary code, `examples/`, the branch README and
  `pyproject.toml`. None of them is an Anima3 experiment result.
- the parent workspace evidence (private repository).

An internal website brief, deleted in `88c7798` before the branch was published, is
not present at `819f4f4` and is not recorded.

The primary branch's own README experiments (Jev and playbook comparisons through
experiment five) are a different line of work. They are not part of this lineage and
must not be substituted for these results.
