# Arena acceptance

## Current topology correction — 2026-09-27

The public arena now pairs participant-run agents. Operator AI workers and the
central learner are stopped and disabled. Old hosted-AI tests below are historical.
Peer-mode live acceptance passed with two locally run agents, no server workers
or allowlist, matching server result receipts and ratings for both sides (1016/984).
A separate practice match completed without ratings. A ranked match occupied
arena 2 concurrently with practice in arena 1. That concurrent ranked match was
interrupted by a separate deployment; recovery and persistence details are in
[the current runbook](ARENA_LINODE.md). A post-recovery peer match also passed
all eight acceptance checks. Both ratings were independently checked against the
opponents' pre-match ratings (984/1016 became 1001/999); practice added no rated
records. The peer build compiles with zero warnings/errors, participant Python
files pass Ruff, and the 10 arena policy/protocol tests passed.


## Production follow-up — 2026-09-27

The Chicago shard and DNS are live at `arena.uotavern.com:2593`. External
mage and warrior ranked matches both completed with rating updates and alive
lobby recovery. Training matches complete as valid/unrated and policies collect
data. The previously described local preview has been stopped. See
[deployment verification](ARENA_LINODE.md) and the [player guide](ARENA_PLAY.ko.md).

## Local acceptance — 2026-09-26

All live tests used an isolated loopback ServUO instance, independent saves and
generated test accounts. No production shard or DNS record was changed.

## Results

| Check | Observed result |
| --- | --- |
| ServUO Release build | 0 warnings, 0 errors |
| Anima3 Python suite | 89 tests passed; new Python files pass Ruff |
| Anima client quality gate | `scripts/check.sh` passed, including Rust, WASM and desktop checks |
| Final browser JS suite | 379 tests, 1888 assertions, 30 files passed |
| Ranked mage match | Real player and worker completed 0–2; player rating became 984 |
| Ranked warrior match | Real player and worker completed 0–2; separate warrior rating became 984 |
| Practice match | Heal potion consumed during combat; completed without changing rating |
| Training self-play | Poison challenger vs baseline completed 0–2 with valid server result and action logs |
| AI failure | Worker killed during combat; match invalid for learning and unrated; player records unchanged |
| Supplies / authorization | Repeated refill bounded; ordinary player cannot register as AI |
| Persistence | Mage and warrior W/L/D/rating survived world save and restart |
| Restart recovery | Saved frozen character in ring returned to lobby and could move after restart |
| Anima UI | Arena menu, supply/cosmetic controls and rankings rendered; hotkey preset, conflict rejection, editing and reload persistence checked in Chrome |
| ClassicUO compatibility | Built unmodified client connected, authenticated and entered world; server recorded Classic Client |

Live evidence is retained locally under `anima3/.logs/arena-*.json` and the test
runtime's `Logs/Arena/events.jsonl`. Evidence and credentials are not committed.
The self-play result ID is `4c52d0dd3cda4edd95ab7bfc120383ee`;
the practice potion result is `011eb34753494df38703b240007a2896`;
the AI-failure result is `9fafde1115cb48b9a6535214c635340e`.

## Historical local preview (now stopped)

- Anima browser: <http://127.0.0.1:8098/>
- UO shard: `127.0.0.1:2598`
- The former preview used one public mage and one public warrior worker.
- Runtime: `../runtime-acceptance`; generated credentials are in its
  `test-accounts.json` (0600). Keep that file private.
- Preview worker process IDs: `.logs/preview-pids.json`. The preview is a local
  development process, not an installed service; use the deployment units for
  unattended operation. Do not log another client into the preview account
  (`arena_classic`) while its browser session is active.

## Worker follow-up — 2026-09-27

- A critically hurt warrior now prioritizes bandaging after leaving melee range.
- Bandage procedures allow the server's self-heal duration and completion latency,
  scaled to the configured worker pump interval. A slip reduces healing and does
  not mean the server timer ended. The agent no longer cancels a pending heal on
  that message or on incidental health regeneration.
- Regression tests cover 8-second delayed heals with an intermediate slip at
  50ms and 250ms pump intervals. Exactly one bandage is applied through completion.
- Final deployed warrior acceptance completed 0–2, updated the test character
  to 0W 3L / 954, and returned it alive to the lobby. Both rounds completed
  within their time limit (50s / 34s). Evidence:
  `.logs/linode-domain-warrior-healing.json`.
- The Python suite was rerun: 92 tests passed; changed Python files pass Ruff.
- The Anima browser suite was rerun: 379 tests / 1888 assertions passed.
- Native ClassicUO visual control was retried; the app-control tool timed out.
  Its previous protocol login/world entry evidence remains separate from visual QA.

## Remaining checks and limitations

- Public hosting, DNS and external mage/warrior flow are now verified; see the
  production follow-up above.
- ClassicUO login/world entry were verified. Its native window could not be
  inspected through the available UI control, so a complete visual ClassicUO
  duel walkthrough remains unverified.
- Learning selects among five mage playbooks. It does not train neural weights;
  warrior AI remains a fixed rule. The promotion/rollback logic was tested with
  synthetic evidence, but live data is still collecting: no real promotion or
  improvement in playing strength has been established.
- Match templates permanently replace skills/stats, as disclosed in the menu.
  Enable this on a dedicated arena shard. The feature defaults to disabled.
- Long-duration public load, anti-abuse operation, seasonal rating calibration
  and external-client latency still require an operational deployment test.

## Training NPC and duel preset acceptance — 2026-09-27

- ServUO release build: zero warnings/errors; commit `af99ac932`.
- `scripts/arena_training_smoke.py`: real UO protocol confirmed NPC/sign gumps,
  both ball grants, bounded repeated grants, rejection of five selections for
  a six-skill ball, six GM skills, stat sum rejection, ball consumption, stale
  queue-dialog rejection, preservation during practice and active-match block.
- `scripts/arena_rules_smoke.py`: local protocol acceptance for all five named
  modes plus custom 6x; 5GM/7GM ball application; 7GM rejected by Mage 5x;
  Paralyze and potion use rejected during a real Mage 5x duel. Saves succeeded.
- Restart: saved 7GM ball restored with its correct selection count, exactly
  one Rowan remained, and its periodic speech was observed by a nearby client.
- Evidence: ignored `.logs/training-local.json` and `.logs/rules-local.json`.
  These tests inspect client packets/gumps; they are not native visual QA.
