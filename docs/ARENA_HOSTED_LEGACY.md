# Archived hosted-AI architecture

This is a private test fixture, not the current public arena.

## Historical hosted-AI test tools

The material below describes the previous optional private test architecture.
It is **not the production topology**. For an isolated legacy shard explicitly
set `Arena.PeerAgents=false` and pass `--hosted-worker` to its workers.

## Isolated local run

From the anima3 worktree, with Python 3.12 and the sibling ServUO worktree:

```sh
uv venv --python 3.12
uv pip install -e ".[dev]"
```

```sh
uv run python scripts/run_arena_local.py --servuo ../servuo --data /path/to/uo-data
```

The launcher builds ServUO and creates `../runtime-local`, test account
credentials (0600), an owner account for local administration, and one-minute
autosaves. It refuses an existing directory without its local-runtime marker.
No existing shard saves or production configuration are copied. Connect to
`127.0.0.1:2597`. Read test credentials locally from the generated JSON file.
Use `[Save` as the test owner before a deliberate shutdown/restart.

Build the Anima bridge in the client worktree (`cargo build --release -p anima-net`),
then copy `target/release/anima-agent` to `target/release/anima-bridge` (the alias
avoids the similarly named in-process runner).
The existing bridge can also be supplied explicitly with `--bridge`.

```sh
# Password is supplied via the environment, not a command-line option.
export ARENA_BOT_PASSWORD='the generated test password for arena_bot_mage'
uv run python -m anima3.arena --hosted-worker --host 127.0.0.1 --port 2597 --user arena_bot_mage \
  --bridge ../anima-client/target/release/anima-bridge --build mage
```

The worker account must be listed in `Config/Arena.cfg:BotAccounts`, and must
remain at Player access. Ordinary players cannot register as bots, request
machine state or submit match results. Bot names are assigned by the service.
A bot announces readiness every three seconds; a 12-second lease prevents an
idle/disconnected worker from being matched. Policies are bound for the match.
Workers stop issuing combat actions if machine state becomes stale, reconnect
on bridge errors, and recover the server's bound playbook after reconnecting.

A protocol acceptance script creates an ordinary player, requests supplies,
uses a standard gump response to join, allows the AI to defeat it, and verifies
match completion, rating and lobby recovery:

```sh
PYTHONPATH=. uv run python scripts/arena_live_smoke.py \
  --credentials ../runtime-local/test-accounts.json --port 2597 --build mage
# Repeat with --build warrior. Use an isolated test shard only.
```

## Learning loop

The first policy uses the measured scripted combat rule. A finite-action learner
selects among the existing mage playbooks (`standard`, `control`, `poison`,
`interrupt`, `sustain`). It does not update neural-network weights. Warrior
workers currently use the fixed melee rule.

1. Start the learner; it creates `champion.json` from the baseline. One training
   worker loads that file, and a second uses `--curriculum <policy-directory>`.
2. The challenger samples playbooks between matches. Only complete, valid,
   server-recorded self-play matches against the exact current champion enter
   training. Human matches are excluded. Disconnects/forfeits and prolonged
   zero-action rounds are excluded from learning.
3. After enough exploration data, the learner chooses a candidate by the lower
   Wilson win bound and writes a frozen candidate version. The curriculum worker
   switches to that candidate automatically.
4. Evaluate on the **first 100 new valid matches**, alternating spawn sides,
   against the exact parent champion. Training IDs cannot enter evaluation;
   both sides need at least a third of the sample. Draws count as non-wins.
   Promotion requires the 99% Wilson lower bound on wins to exceed 50%.
5. `--promote` atomically updates the champion only after this gate passes.
   Public workers read the new policy between matches. Prior policies and
   evaluation IDs are archived for rollback. Failed candidates remain frozen;
   a new training batch is needed before another candidate can be proposed.

```sh
uv run python -m anima3.arena_learning --events /path/to/shard/Logs/Arena/events.jsonl \
  --policies /path/to/policies --watch --promote
# Reference worker
uv run python -m anima3.arena --hosted-worker --user arena_train_a --training --policy /path/to/policies/champion.json
# Challenger worker (use a different password environment)
uv run python -m anima3.arena --hosted-worker --user arena_train_b --training --curriculum /path/to/policies
```

Keep one writer per policy directory. `evaluation.json` reports collection,
evaluation, rejection or promotion. No promotion or performance gain is implied
by having logs or by passing unit tests. The confidence bound is per candidate;
it is not a guarantee over indefinitely repeated proposals or against humans.
Rollback: atomically replace `champion.json` with an archived policy. Never edit
an in-flight candidate's version/playbook or its evaluation sample.

## Authoritative records and operations

- `Saves/ArenaService.bin`: atomic W/L/D/rating snapshot, also saved on world save.
- `Logs/Arena/events.jsonl`: server match IDs, identities, build, bound policy,
  round counts/actions, results and validity. Only the trusted shard log is a
  learner input. Incomplete final lines are ignored, conflicting IDs rejected.
- `Export/Arena/leaderboard.json`: public names/serials/builds/W-L-D/ratings;
  contains no account usernames or passwords. Can be published by an existing
  static host. Do not publish Saves, agent journals or private shard logs.
- Agent log directory: match/round observation, decision and procedure traces,
  plus worker lifecycle. Use distinct directories for each worker.
- Storage errors stop new matchmaking. Server resets and aborted matches do not
  rate; AI disconnects invalidate learning and do not change human ratings.
- Standard shard world saves remain essential: the rating snapshot references
  saved characters. Take a full world save before maintenance; rotate logs and
  back up the complete Saves directory together. In-flight matches are not
  resumed after a server restart; they produce no completed result to learn from.

## Production deployment

`deploy/arena/` contains systemd units and configuration examples. Prepare a
separate `/opt/uoarena/shard` instance and a dedicated `uoarena` OS user, valid
UO assets at `/opt/uoarena/data`, Python 3.12, the built Anima bridge and Mono.
Build ServUO with .NET, then use `Compiler.Dynamic=false` on the deployed shard.
Initialize the operator account interactively once, save, then use `-service`.
Create the worker accounts with unique passwords and keep them at Player access.

Provision an **A record** `arena.uotavern.com` to the shard's public IPv4 and
allow TCP 2593. The game needs direct TCP reachability; a standard HTTP reverse proxy does not
carry the UO game protocol. Configure `Server.Address` to that
hostname; no HTTP reverse proxy is required for UO clients. Worker accounts and
logs stay on the private host; do not expose Anima's local play/monitor ports.

Install env files per worker with mode 0600. Provision the policy directory and
initial `champion.json` before starting public mage/reference workers. Start
shard, learner, public mage/warrior, then reference/challenger workers. Public
queue entries have priority over self-play, which uses training workers only.
Use the normal shard account/IP limits appropriate to public access, separate
from the larger loopback-only limits in the test launcher.

## Verified behavior

See [ARENA_VALIDATION.md](ARENA_VALIDATION.md) for the local acceptance results,
remaining production checks and preview location.
