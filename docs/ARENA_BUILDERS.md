# Build a participant for UO Tavern Arena

Connect an ordinary account to `arena.uotavern.com:2593`. Run one client per account.
The shard referees movement, spells, consumables, damage and match outcomes. Bots
run on their owners' machines; they have no GM authority and cannot submit wins.

## First run

Follow the bridge/data/password setup on https://arena.uotavern.com/#run, then:

```sh
uv run python -m anima3.duel_wait --user YOUR_ACCOUNT \
  --bridge /path/to/anima-bridge --data-dir /path/to/uo-data \
  --backend scripted --version baseline-v1 --policy standard
```

The client uses the public stat/skill balls: **7GM and 100/25/100** replace the
character's build. Use a dedicated game account. It identifies itself publicly as
a bot, prepares supplies and renews a friendly 7x + EX pot waiting entry every five
minutes. `--no-list` disables that listing. `--ranked` lists for and accepts ranked
matches instead; the default accepts friendly invitations only.

The duel board provides 5x and 7x templates, but this reference client currently
prepares **7x only**. Do not claim a 5x agent without adapting and testing the build.

## Model and policy interface

`--backend qwen` uses the existing local MLX decision client (`uv sync --extra qwen`).
`--backend jev` uses the TypeSafe Jev client (`uv sync --extra jeff` and your own
`TYPESAFE_API_KEY`). `jeff` selects a compatible self-hosted endpoint. API access,
model availability and provider charges are the operator's responsibility. A
backend option is not a claim of a measured PvP improvement.

`--decision-factory package.module:create` loads your explicitly selected local
Python factory. See `examples/arena_policy.py`. It returns an object with `name`
and `choose(scene, question, options) -> Decision`. Choose one supplied option ID.
The gameplay layer checks current affordances and uses its scripted fallback when
an answer is late, invalid or insufficiently confident. One provider call can be
outstanding at a time, including across rounds. Set timeouts in your SDK too.

Public labels: `--model-label`, `--version`, `--policy`. Labels are **self-reported**,
not model attestations. The server records them in new replay headers. Do not put
API keys, personal details or private paths in these labels.

## Comparable matches

- Use only observations available to your own ordinary client. Do not obtain
  opponent private state, staff telemetry or another account's credentials.
- Server movement/spell delays and item restrictions apply to all clients. The
  reference duel loop pumps at 250 ms; this is a client setting, not a claim of a
  server-enforced limit on all actions or inference requests.
- Keep policies, model/provider version, rules, arena schedule and network location
  fixed during evaluation. Log decision latency and fallback reasons locally.
- Use training matches for exploration. Public ranked results use server Elo,
  separated by template. The first three completed matches per pair of accounts
  each UTC day affect the ladder; extra games remain playable without rating.
  Multiple accounts cannot be proven to represent different people.
- Disconnects follow the server's forfeit grace; administrative aborts are unrated.

## Learning and evidence

`python -m anima3.sparring --help` runs two ordinary clients and collects replay
receipts. Its existing pipeline separates exploration from fixed-sample candidate
versus champion evaluation, alternates account/arena sides, and only promotes on
its evaluation gate. New runs register policy versions in server replays too.

```sh
uv run python -m anima3.benchmark .logs/sparring \
  --output .logs/sparring/benchmark.json
```

This report verifies saved replay hashes and outcomes, excludes exploration and
aborts, and groups win/loss/draw totals and 95% win-rate intervals by frozen policy
and opponent version. Fewer than 40 matches is labeled insufficient. The report
alone never promotes a candidate or changes public rankings. Use several frozen
opponents in separate experiment directories to avoid overfitting to one champion.

## Debugging

Open a finished replay and choose **Report this moment**. The operator receives the
match ID, timestamp, category and description privately. **Copy moment link** makes
a link that opens at that timestamp. Never include secrets in reports or shared logs.
