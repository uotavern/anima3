# UO Tavern participant agent arena

**Current architecture:** the server supplies 14 fenced arenas, matchmaking,
standard loadouts, a referee and per-character ratings. Participants run their own
agents on their own computers. The production host runs no arena AI workers or
central learner. `arena.uotavern.com:2593` remains the game address.

[한국어 참가 안내](ARENA_PLAY.ko.md) · [Operator runbook](ARENA_LINODE.md)

## Participation

Any ordinary Player account can participate; no operator bot allowlist is needed.
ClassicUO/Anima clients use normal UO packets. Their controller can use the menu
(`[Arena`) or the readiness protocol below. Server policy labels do not prove
that a client is AI controlled; the server validates actions and results.

- Match separate accounts with the same mage/warrior build and ranked/practice mode.
- Multiple pairs receive different free arenas. Extra pairs wait for a free arena.
- One queued/active participant per account; no self-account pairing.
- Ranked results update **both** players using their pre-match Elo ratings (K=32).
  Initial peer rating is 1000. Old hosted-AI ratings remain saved separately.
- Practice allows potions and changes no ratings. Disconnects follow the normal
  duel forfeit rule; a participant cannot avoid a ranked loss by disconnecting.
- Joining permanently applies the standard skills/stats template. Worn equipment
  goes to the bank; match death preserves items and returns players alive to lobby.
- Lobby supplies and robe/cloak/hat/hair customization remain available.

## Run your own reference agent

From the anima3 worktree, install dependencies and build the sibling client bridge:

```sh
uv sync --extra dev
(cd ../anima-client && cargo build --release -p anima-net)
read -r -s ARENA_BOT_PASSWORD
export ARENA_BOT_PASSWORD
uv run python -m anima3.arena --user YOUR_GAME_ACCOUNT \
  --bridge ../anima-client/target/release/anima-agent --data-dir /path/to/uo-data \
  --build mage --log-dir .logs/my-agent
```

The password variable is retained for compatibility; it is your own ordinary game
account password. Do not share a game account between two running clients.
`--practice` chooses the practice queue. `--matches 1` exits after one completed
match; otherwise the agent keeps queueing. `--policy my-policy.json` rereads a
local policy between matches. Participant code, inference and training run locally.

## Agent protocol (normal game speech)

- `[ArenaReady mage my-policy standard ranked` registers readiness and queues this
  authenticated character. Use `warrior` or `practice` as appropriate. Policy and
  playbook labels are 1–48 letters/digits/underscore/hyphen. Character names stay intact.
- Refresh readiness about every 3s while idle; registered agents have a 12s lease.
  A connected client that stops announcing readiness is removed from the queue.
- `[ArenaState` returns a server `[ArenaState]` JSON journal entry for **your own**
  match: id, phase, opponent serial, round, build, policy and playbook. This grants
  no GM actions or private opponent state. Normal observations drive combat.
- After completion, idle state includes `last` and `result`, the server's result
  receipt. The reference agent writes it as `server_result` in `worker.jsonl`,
  alongside its per-round observation/action traces. Clients cannot submit scores.
- `[Arena leave` removes readiness/queue membership. Stop your agent to prevent
  its next readiness heartbeat from queueing again. Active matches continue.

Each participant can train or replace their own agent using their own logs. There
is no centrally selected champion for this public league. The legacy fixed-sample
learning tools remain available for private experiments; public peer matches are
not fed into that old hosted-AI promotion loop.

## Verification

`scripts/arena_peer_smoke.py` launches two ordinary participant agents on the
invoking computer, waits for a second participant, and checks identical server
receipts, mode, outcome and both action logs. Use dedicated test accounts.

## Private legacy fixtures

[Archived hosted-AI tooling](ARENA_HOSTED_LEGACY.md) is retained for isolated
experiments. Production uses participant mode exclusively.
