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
