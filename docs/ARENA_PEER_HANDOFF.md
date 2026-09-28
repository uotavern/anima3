# Handoff to the duel-results website session

## Agreed architecture

Participants run their own AI agents. The arena host provides 14 rings,
matchmaking, loadouts, refereeing and records. It runs no opponent AI and no
central learner. The website session owns the results website; this work owns
the shard and participant protocol.

## Deployment compatibility — required

- ServUO worktree: `../servuo`, branch `feat/arena-service`.
- Participant client commit: `a1f208f` in the anima3 worktree. Its default mode
  is peer participation; `--matches 1` stops after receiving its own result.
- Peer-mode server commit: **`4d2895d79`**. This is based on the web-feed commit
  `bb6c89212`; it includes the read-only `/duel/` feed.
- `Config/Arena.cfg`: `PeerAgents=true`, empty `BotAccounts`, `SelfPlay=false`.
- Keep `DisableRisingTide=true`. Keep the web session's `Config/Duel.cfg` and
  Caddy settings; they were preserved during recovery.
- Keep `arena-learner` and the public-mage/public-warrior/reference/challenger
  worker units **stopped and disabled**. Only the shard is needed for games.
- `Saves/ArenaService.bin` now uses version 1 to keep peer and legacy ratings
  separate. **Do not deploy older ArenaService binaries against this save.**
  This caused the 15:02 UTC restart loop; the current peer build recovered it.
- Deploy a completed build atomically, preserve Saves, and save before planned
  maintenance. Do not copy `._*` AppleDouble files into Config; ServUO attempts
  to parse them as configuration. Use `COPYFILE_DISABLE=1` when making macOS tarballs.
- Coordinate shard restarts with the server session; the interrupted concurrent
  test had to be excluded from completed-match verification.

## Website data and wording

The existing `/duel/` feed contains:

```json
{
  "arenas": 14,
  "arena": {
    "domain": "arena.uotavern.com",
    "mode": "peer_agents",
    "participants": 2,
    "bots": 0,
    "queue": 0,
    "leaderboard": [
      {"name": "Example agent owner", "build": "mage", "online": true,
       "wins": 1, "losses": 0, "draws": 0, "rating": 1016}
    ]
  }
}
```

The values above are illustrative. `participants` counts registered connected
agent characters, not all site visitors or guaranteed ready agents. `bots=0`
means there are no hosted opponent bots. It does not mean no participants exist.

- Suggested wording: **“Ranked matches between participant-run agents, per build.
  Everyone starts at 1000.”**
- Both participants receive W/L/D and Elo against the opponent's pre-match rating.
  Do not filter participant agents out of the leaderboard as “bots”.
- Use `arena.leaderboard` for the current peer league. The general `standings`
  and `recent` arrays retain historical ordinary duels and older hosted matches.
- Practice has no rating effect. No match outcome or rating comes from the client.
- Policy/playbook labels are participant declarations, not verified model identities.
- Do not expose private account usernames, passwords, Saves or action journals.

Participant setup: [ARENA.md](ARENA.md), [한국어 안내](ARENA_PLAY.ko.md).
Live test and recovery evidence: [ARENA_LINODE.md](ARENA_LINODE.md).

## Training and duel presets (2026-09-27)

ServUO `af99ac932` adds Rowan (Felucca 5183,332,15), the guide sign,
stat balls and 5/6/7GM skill balls, plus a targeted duel menu with invitation
acceptance. Named presets: `mage5`, `mage7`, `standard7`, `dexxer7`, `open7`.
Custom presets use canonical rule tokens. Existing mage/warrior Elo queues
remain standardized 5x; practice queues preserve a <=6x player build. Direct
5x/7x/custom challenges appear in generic duel history/standings rather than
being folded into the existing mage/warrior arena Elo leaderboard.

**Do not deploy older Scripts.dll over these saves:** the world now stores
ArenaSkillBall, ArenaStatBall, ArenaSteward and ArenaTrainingSign. A rollback
requires the matching pre-deployment world backup as well as binaries.

## 7GM / potions correction

ServUO `a1547fb86` (including `8fb5c6f25`) is the new release. Default skill
balls and saved 6GM balls now use 7GM; 5GM remains available. Practice cap is 7x.
All presets and the existing mage/warrior queues permit regular potions by
default. Explosion potions require the explicit `explosion` rules token; UI
checkbox is off by default and applies to preset and custom buttons. The web
feed carries `explosion` / `noexplosion` in the canonical rules string.

## Pre-AOS combat/support selection

Release `97bee0afd` restricts skill-ball choices to 21 combat/support skills.
Poisoning, ArmsLore, Alchemy, Inscribe and Lumberjacking are included; Tracking,
pure trade, bard/pet and post-AOS skills are excluded. UI and response validation
share the same allowlist. Existing potion options and 7GM behavior are retained.

### Incoming direct duels (participant client)

`python -m anima3.duel_wait --user <ordinary-account>` reads `ARENA_BOT_PASSWORD`, logs in,
uses public skill/stat balls to prepare Magery/EvalInt/Meditation/Resist/Wrestling/Anatomy/Alchemy
at GM and 100/25/100 stats, stocks supplies, and waits for `standard7-explosion` challenges.
It accepts only server-issued challenge tokens whose canonical rules exactly match
`7x-magic-explosion-classic`. `[DuelState]` reports only the requesting player's match/invite;
`[DuelAccept <id>]` cannot accept an invitation replaced since inspection.
The local client (not the shard) drives spellcasting and explosion potions. It reconnects
and prepares again after matches. Logs: `.logs/duel-wait/client.jsonl` and per-round JSONL.
Other rules are ignored with a chat explanation. Stop with SIGTERM to the client PID.

Deployment 2026-09-27 17:05 UTC: ServUO `55ad7d126` (includes `31fbb2deb`
Hybrid combat fixes and `97bee0afd` 21-skill filter), DLL SHA-256
`ed5f62727c0eba3364a872126b6d12b3dff95ad9a605ad22587787362dad09ab`.
Backup: `/root/uoarena-before-skills-20260927T170554Z.tar.gz`.
Verified local ordinary-client preparation, mismatched rules ignored, correct challenge
accepted, explosion potion targeted, spell combat, server-decided win, and re-preparation.
The public local participant is started via ignored `.logs/start-duel-wait.py`; credentials
are read from the protected local account file without writing passwords into CLI arguments.
PID file: `.logs/duel-wait/client.pid`. This is a Mac process, so the Mac must stay online.
Public readiness verified 2026-09-27 17:06 UTC: `Tavern Mage 3`, serial `0x3`,
7GM and 100/25/100 confirmed from server observations. Human challenge command:
`[Challenge 0x3 3 standard7-explosion`. Client PID at this verification: 35539.
Public skill-dialog smoke passed all three inspection checks; game service and existing
DuelWeb feed remained healthy. Public human match is still to be tested by the user.

## Showdown update (2026-09-28)

ServUO `2515fd211`: every direct/queued duel round now starts Showdown at 180s,
with a warning 30s before it, and draws at 300s if unfinished. Heal/Greater Heal,
bandages, heal potions and natural HP regen are blocked; checks include completion
of pre-Showdown healing. Cure and stamina/mana recovery retain ordinary rules.
Fresh rounds restore healing. Config: `Duel.ShowdownAfterSeconds` and
`Duel.RoundTimeLimitSeconds` (restart required). Existing phase/rules strings stay stable.
`DuelState` / `ArenaState` expose boolean `showdown` and seconds `showdownRemaining`.
Web root exposes `showdownAfterSeconds` / `roundLimitSeconds`; live rows expose
`showdown`, `showdownRemaining`, `roundLimitSeconds`. The reference agents avoid
healing casts in Showdown. Website instructions are appended to `ARENA_WEB_AGENT_BRIEF.md`.

Showdown deployed 2026-09-28 03:39 UTC; DLL SHA-256
`0637cefceda60235819d1a223dd82777007d21c577bc4fb7d4a122d9d77d200c`.
Backup `/root/uoarena-before-showdown-20260928T033930Z.tar.gz` includes matching world/config/binaries.
Local accelerated 30s/65s packet integration passed all 12 checks, including held Heal
resolution, unconsumed heal potion, blocked bandage/spells/regen, permitted stamina potion,
round timeout and restored healing in round 2. Production uses defaults 180s/300s.
Anima3 unit checks: 13 passed. Existing web feed, port and service verified after restart.
Reference participant was restarted to load the Showdown-aware policy.

## 2026-09-28: real UO visual replay deployed

User correction: replay must show actual UO graphics, skin/hair hues, clothing,
weapons and movement/combat/casting animation. The schematic inspector was removed.

- ServUO worktree commit `385c6a501`; both core and Scripts deployed together.
- Anima client worktree commit `69dcbcd`; read-only real UO renderer at
  `https://arena.uotavern.com/replay/?replay=<id>`.
- Verified production example:
  `https://arena.uotavern.com/replay/?replay=eca76ed797324671ae6a67e63ba129ba`.
  359 rows, visualVersion 1, complete, zero drops; server checksum and result verified.
  Browser displayed real terrain/fences, body/robes/hair and finished 0–1 after 38.2s.
- Archive list `/duel/replays/`; only show `complete:true, visualVersion:1` as playable.
  `/duel/replays/<id>.jsonl` uses gzip content encoding. Match list should label training.
- Website owns the surrounding list/results/navigation. Embed this renderer in an iframe
  or link directly. Do not build another schematic replay canvas.
- Source contract: `docs/ARENA_REPLAY.md`; deployment snippets in `deploy/`.
- `arena-replay-assets` on loopback 8096. Caddy serves the web shell/scripts directly,
  then proxies generated asset URLs. Serving app files directly resolved the observed initial script-loading stalls. No changes to `/var/www/arena` were made.
- Paired rollback backup: `/root/uoarena-before-replay-20260928T054330Z.tar.gz`.
  Scripts SHA256 `804c0abfa80b3cb38ba0dd9a9d16a074fad1146106f124e193ba0e2716fb3eee`;
  core SHA256 `6a02e1d3d7efda7498352f89142d053c625691e94cd891b01a384040fff452d8`.
- Ordinary public challenger Tavern Mage 3 restarted and reported ready. Separate
  ordinary-account training clients use 7x + explosion and do not change public
  standings/recent match history (verified before/after production training).
- Sparring/learning is bounded to 100 completed matches per run, local logs under
  `.logs/sparring/`. Currently collecting evidence; no improvement/promotion claimed.
  A transient TLS fetch failure stopped one run safely; GET retries were added and
  the coordinator resumed. Training files are not themselves a public HTTP service.
- Current deployed graphics pack targets pre-AOS; newer expansion UOP graphics require
  the full matching data files. Retention remains 30 days / 200 replays / 512 MiB.

### Replay follow-up

Public speech/spell mantras are now server-recorded; player shows overhead words,
paralysis/poison status, and sound toggle. Local file upload removed per user request.
Website worktree `/Users/dkkang/dev/uo/arena-worktrees/uotavern-replay`, branch
`feat/arena-replay-web`, adds /#replays with match choices, training/public filter,
and direct Watch replay links. Preserve Caddy's /replay/* handler when deploying.

The same website branch also installs anonymous persistent likes: Python stdlib
`services/replay_likes.py`, systemd `arena-replay-social`, loopback 8098, SQLite state
`/var/lib/uoarena-replay-social/`. Routes `/replay-social/likes` and
`/replay-social/likes/<id>` are proxied by Caddy. Match cards show counts and a
reversible Like button; the identity is a signed browser cookie, not a user account.
Server/shard restart is not needed for this service.

Important deployment correction: the live Caddyfile was missing `/replay/*` when
this follow-up started (replay links returned HTTP 404). Restored its static-first
renderer handler and verified HTTP 200. The arena website branch carries BOTH
replay handlers in `arena/Caddyfile`; merge that branch before using an older
`arena/deploy.sh` checkout, which replaces the entire live Caddyfile.
Verified new archive `1745e622495949cdb451403af58f1a55`: complete, 10 speech rows,
43 paralyzed state frames. Public speech fixture `25ed74fe40714b20830864af25617f1d`
contains both participants' ordinary speech; it was excluded from learning evidence.
