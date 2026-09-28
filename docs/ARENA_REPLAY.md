# UO visual replay and participant sparring

## What is recorded

Every newly started direct/queued duel is recorded by ServUO. No screen or audio capture,
no client telemetry upload and no game-account passwords are involved. The server records
both participants' positions, facing, health/mana/stamina, poison/paralysis/hidden/alive
state, active spell name, round, score and Showdown at **200 ms** intervals (5 Hz).
It records separate ordered events for casts, successful spell target validation, fizzles,
reported damage/healing, explosion potion throw and explosion coordinates, skill attempts,
round outcomes and system duel announcements. Appearance/equipment and base stats are
snapshotted as loadouts. Arbitrary player chat, account identifiers, backpack contents and
credentials are excluded.

Replays use **the existing Anima client UO renderer**, including the real map,
body/skin hues, hair/beard, equipment animations and colors. `visualVersion:1`
recordings include actual server-sent animation/effect cues (0x6E, 0xE2, 0x70,
0xC0, 0xC7, 0xAF, 0x54, 0x2F), sampled world objects and movement timestamps.
They are data recordings, not video and not re-executed combat simulations.
Movement between observations is interpolated. The 2D renderer shares live-client
behavior, including its effect travel timing approximations; it is not a pixel-identical
capture of a particular ClassicUO installation. Old recordings without `visualVersion`
are rejected by the visual player rather than pretending to have missing animation data.

Damage event amounts are the server damage-hook values (which may include overkill);
frame HP is the authoritative observed health. A `cast` or `skill_attempt` is an attempt,
not proof it succeeded. `spell_target` means server sequence validation succeeded.

## Read-only HTTP contract (schema 1)

- `GET /duel/replays/`: `{schema:1,replays:[metadata,...]}`, newest completed archives first.
- `GET /duel/replays/<32-lowercase-hex-id>.jsonl`: UTF-8 NDJSON, transported with
  `Content-Encoding: gzip`. Browsers decompress automatically; raw HTTP clients must honor it.
- UO player: `/replay/?replay=<id>` on the asset host. Local development:
  `http://127.0.0.1:8099/?replay=<id>&api=http://127.0.0.1:8097`.
  Anima `assets` serves the renderer and UO graphics; no game account/login is required.
  Do not substitute a schematic canvas for this renderer.
- `GET /duel/`: existing feed now includes `replayEnabled`, `replaySampleMs`, and each
  live match's stable `id` and `training`. Match IDs match the archive IDs.
- GET/HEAD only. Unknown IDs/paths return 404; other methods return 405.

Metadata includes `id`, `started`, `ended`, `durationMs`, `arena`, `players`, `rules`,
`training`, `winner` (serial or null), `score:[a,b]`, `aborted` (reason or null),
`visualVersion`, `complete`, `dropped`, `sha256`, and relative `url`. SHA-256 is of the **uncompressed
NDJSON bytes**, including newlines. HTTP access uses the server's ordinary public feed;
there is no upload or score-submission endpoint.

Each row has monotonically increasing `seq`, `t` (integer milliseconds since match start,
from a monotonic clock), and `type`. `t` does not reset between rounds.

| type | payload |
|---|---|
| header | schema, id, UTC started, rules, training, rounds, sampleMs, time limits, arena geometry, players |
| loadout | players: serial/name/body/hue/stats [STR,DEX,INT]/equipment serial,graphic,hue,layer (including hair/beard) |
| position | round, phase, player: immediate server-observed movement state |
| world | visible arena ground objects: serial,g,hue,pos |
| visual | packet: base64 of the allowlisted, uncompressed UO visual packet; use t for its animation start |
| frame | round, phase, showdown, score, players: serial,pos [x,y,z],direction,war,hits/hitsMax,mana/manaMax,stam/stamMax,alive,poisoned,paralyzed,hidden,spell |
| cast / spell_target / fizzle / skill_attempt | actor, target (0 when absent), name, amount |
| damage / heal | actor, target, amount; damage name may be lethal |
| potion_throw / potion_explode | actor, from [x,y,z], to [x,y,z] |
| announcement | server-generated text |
| round_end | round, winner or null, reason |
| end | id, training, winner, score, aborted, complete, dropped |

Header player order defines sides A/B throughout the recording. Frame arrays preserve
that order. Render the state before/at the seek time, interpolate positions only within
the same round/phase, and reset transient effects when seeking. Do not interpolate a
round reset as movement across the map. Treat recorded names/strings as text, never HTML.
`phase` remains Countdown/Fighting/RoundOver/Finished; Showdown is a separate flag.

## Storage and operational boundaries

`Export/DuelReplays/<id>.partial` is appended by one bounded asynchronous writer;
finished recordings become `<id>.jsonl.gz` plus atomic `<id>.meta.json` manifests.
Only finalized archives are served. Game simulation never waits for disk queue space.
Overload/size limits flag dropped rows and `complete:false`; failures are logged and do
not invent results. Interrupted server-process recordings remain `.partial` for diagnosis,
are not advertised as complete replays and are not training evidence.

Defaults in `Config/Duel.cfg`: `ReplayEnabled=true`, `ReplaySampleMs=200`,
`ReplayKeep=200`, `ReplayDays=30`. Finalized archives additionally have a 512 MiB budget;
individual recordings cap at 150,000 rows / 32 MiB. Retention runs at startup and after
finalization. Old links can return 404 after retention. Export important recordings
before expiry; this is bounded local storage, not permanent archival storage.

## Sparring and learning

`training` is an explicit consenting direct-duel rule token, e.g.
`[Challenge <peer-serial> 1 standard7-explosion-training`.
Both players see this ruleset. Training records/replays are marked, and **neither public
duel standings/history nor the separate queue Elo are changed** by these direct training
matches. Regular direct duels continue to update their existing public scores.

From the arena anima3 worktree, with two dedicated ordinary accounts:

```sh
# Set SPAR_PASSWORD_A and SPAR_PASSWORD_B privately in the environment first.
uv run python -m anima3.sparring \
  --user-a YOUR_TRAINING_ACCOUNT_A --user-b YOUR_TRAINING_ACCOUNT_B \
  --host arena.uotavern.com --web https://arena.uotavern.com \
  --bridge /path/to/anima-agent --data-dir /path/to/uo-data \
  --matches 100 --minimum 10 --sample 40 --log-dir .logs/my-sparring
```

Each client uses public 7GM/stat balls and supplies. This replaces the accounts' existing
skills/stats. The coordinator controls only its two clients, accepts only its peer's
exact training invitation, alternates candidate accounts and arena sides independently,
and runs a bounded number of best-of-one rounds. Do not simultaneously use these accounts
in ClassicUO or another agent. Stop with Ctrl+C; an active match follows server forfeit
rules and is not silently cancelled. Missing/mismatched evidence stops the experiment.

Outcomes come from downloaded, hash/sequence-verified **server** replays. Policy labels
are local experiment metadata, not a server attestation of which code a client ran.
Five mage playbooks are explored against the incumbent; after enough observations a
candidate may be frozen. Evaluation uses a separate fixed sample (minimum 40), both
arena sides, a 99% Wilson lower bound above 50%, and no training/evaluation overlap.
Only a passing candidate is promoted; previous policies and evidence are kept. This
is finite-policy exploration/selection, not neural-network weight training. Improvement
is not guaranteed, and a run can finish while still collecting data or evaluating.

Files in the run directory:

- `status.json`: current stage, players/policies, latest match and analysis; update atomically.
- `progress.jsonl`: stage transitions and per-match summaries.
- `learning.jsonl`: trusted server outcomes plus local policy labels for the learner.
- `<match-id>/replay.jsonl`, `replay.meta.json`, `analysis.json`: exact evidence and metrics.
- `policies/champion.json`, `candidate.json`, `evaluation.json`, `history/`: learning state.

A dashboard should show **collecting training / evaluating / rejected / promoted**
truthfully, with counts, selected policy and linked replays. Never replace incomplete
training with a fake progress counter or label an unevaluated candidate "improved".
The current public server hosts recordings, not the local learner's status.json; web
integration needs a deliberate read-only publication path for that separate status.

## Renderer deployment

Deploy ServUO.exe and Scripts.dll together: the recorder subscribes to the core's
visual-only packet observer. Anima `assets` runs independently, read-only, bound
to loopback. Reverse-proxy `/replay/*` with the prefix stripped to that service.
The standard archive API remains on the game server's `/duel/*` feed.

The current lightweight server asset pack targets pre-AOS graphics: legacy
`anim.mul`, art indices below 0x8000 (land plus static graphics below 0x4000),
lightning gumps, terrain/textures and hue/remap tables. It supports the current
human 5x/7x arena wardrobe and weapons; newer expansion-only UOP animations
and art require the full matching UO data installation. Do not claim arbitrary
expansion assets are included in this deployment. The pack builder reads an
existing local UO installation; generated assets are not committed to Git.
