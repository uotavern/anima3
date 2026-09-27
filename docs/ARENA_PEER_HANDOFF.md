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
