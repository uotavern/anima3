# Arena acceptance — 2026-09-26

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

## Local preview left running

- Anima browser: <http://127.0.0.1:8098/>
- UO shard: `127.0.0.1:2598`
- One public mage and one public warrior worker wait in the lobby.
- Runtime: `../runtime-acceptance`; generated credentials are in its
  `test-accounts.json` (0600). Keep that file private.
- Preview worker process IDs: `.logs/preview-pids.json`. The preview is a local
  development process, not an installed service; use the deployment units for
  unattended operation. Do not log another client into the preview account
  (`arena_classic`) while its browser session is active.

## Remaining checks and limitations

- `arena.uotavern.com` is configured as the intended address, but public hosting,
  DNS and external network access await the operator's server/DNS details.
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
