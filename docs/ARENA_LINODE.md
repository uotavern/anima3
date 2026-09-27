# Current topology: participant-run agents (2026-09-27)

`arena.uotavern.com:2593` supplies 14 arenas and server-authoritative matchmaking,
refereeing and per-participant ratings. Participants run AI on their own machines.

- `Arena.PeerAgents=true`, `BotAccounts=` (empty), `SelfPlay=false`.
- `arena-shard` stays active/enabled.
- `arena-learner` and all four operator worker instances are stopped/disabled.
- The host's old agent code, private logs and policies are retained, not running.
- No new VM or paid resource was created for this correction.
- Old hosted-AI ratings are loaded as a separate league; peer records start at 1000.
- The pre-migration world and binaries are backed up privately at
  `/root/uoarena-before-peer-mode.tar.gz`. Rating persistence is now version 1;
  restore the complete backup if rolling back to binaries that only read version 0.
- Agents need ordinary Player accounts. No bot-allowlist registration is required.
- First external peer match `cb2c86beb4a34ed8afe0508f82f85aed` completed with
  identical receipts on both participant machines/processes. Both ordinary
  accounts received records: winner 1016 and loser 984 from 1000. Test processes
  ran on the operator Mac, with no AI process running on the server.
- Practice peer match `79d02062c8f6498daab841eaa1dd4f5e` completed with both
  receipts agreeing and `rated=false`. Its participants gained no rating records.
- Match `04776ab2dab64155879e54e1944d4cb5` ran in arena 2 while that practice
  match occupied arena 1, verifying concurrent allocation. It was interrupted
  by a separate deployment and is not a completed-match acceptance result.
- At 15:02 UTC a different DLL/config deployment replaced the running peer build.
  Its older rating reader could not parse the new version-1 save. The conflicting
  DLL was preserved in `/root/uoarena-conflicting-Scripts-1502.dll`; peer build
  `4d2895d79` was restored without changing the new web-feed config. AppleDouble
  `._*.cfg` files were moved out of Config to `/root/uoarena-config-metadata`.
  World load subsequently succeeded and the 984/1016 peer ratings survived.
- Post-recovery peer match `50767965c9a142998521eff403ee5167` completed. Both
  clients received identical server receipts. Ratings changed from 984/1016 to
  1001/999, verified against both pre-match opponent ratings; each player now has
  two ranked results. Practice participants still have no ranked records.
  Evidence: `.logs/peer-after-recovery-20260927/result.json`.
- Evidence: `.logs/peer-ranked-20260927/result.json` plus each participant's
  `worker.jsonl` and per-round action logs. First participant waited until the
  second joined. No hosted worker or allowlist was used.

See [participant setup and wire protocol](ARENA.md) and [한국어 안내](ARENA_PLAY.ko.md).
The original deployment record below is historical; its hosted AI service status
has been superseded by the configuration above.

---

# Linode arena deployment

Created 2026-09-27 for the UO Tavern public arena.

| Setting | Value |
| --- | --- |
| Label | `uotavern-arena-us` |
| Instance ID | `106770396` |
| Region | Chicago, `us-ord` |
| Plan | `g6-standard-4`: 4 shared vCPU, 8 GB RAM, 160 GB SSD |
| Base price at creation | USD 48/month cap, USD 0.072/hour; taxes/overages extra |
| Public IPv4 | `172.234.206.216` |
| Image | Ubuntu 24.04 LTS x86-64 |
| Firewall ID | `177886619` |
| Intended game address | `arena.uotavern.com:2593` |

No promotional credits were active at creation. Paid backups, managed service,
load balancers and extra storage were not ordered. Powered-off instances still
incur charges; deleting a backed-up instance is a separate operator decision.

## Access and layout

The operator's local SSH key is `~/.ssh/uoarena_linode`:

```sh
ssh -i ~/.ssh/uoarena_linode root@172.234.206.216
```

The shard runs as the unprivileged `uoarena` OS user. Password SSH login is
disabled. API tokens and private keys are not stored in this repository.

- `/opt/uoarena/shard`: ServUO runtime, Config, Saves, Logs and Export.
- `/opt/uoarena/data`: private game assets needed by the server/AI.
- `/opt/uoarena/anima3`: Python source and virtual environment.
- `/opt/uoarena/anima-client`: source and Linux bridge binary.
- `/etc/uoarena/*.env`: root-owned worker credentials, mode 0600.
- `/etc/uoarena/accounts.json`: initial operator/worker/test credentials, 0600.
- `/var/lib/uoarena/policies`: champion, training state and evaluation history.
- `/var/lib/uoarena/agent-logs`: worker observations and decisions.

The initial account file is private. Normal users choose their own account name
and password at first login. The shard automatically creates accounts, with a
three-account-per-IP creation limit. Bot accounts are pre-created and allowlisted.
The admin account is separate from all Player-access workers.

## Operations

```sh
systemctl status arena-shard arena-learner
systemctl status arena-worker@public-mage arena-worker@public-warrior
systemctl status arena-worker@reference arena-worker@challenger
journalctl -u arena-shard -n 60
cat /var/lib/uoarena/policies/evaluation.json
```

Save from the operator's game account with `[Save` before planned maintenance.
Stop training workers before maintenance, then public workers and the shard.
Back up the entire `Saves` directory and policy directory together after a save.
Do not restore only the rating snapshot without the corresponding world save.
Do not publish account credentials, Saves or raw journals.

To reduce CPU load, stop the two training worker units; public duels can continue.
There is no neural-model inference service or GPU on this machine.

## DNS and network

Cloudflare must use an A record `arena` → `172.234.206.216`, **DNS only**.
The UO protocol requires direct TCP 2593. Agent/play/monitor ports remain private.
SSH uses public-key authentication. The cloud firewall defaults to dropping
inbound traffic except explicitly allowed ports.

## Release inputs

- ServUO `a2c68a251` (includes arena-only Rising Tide disable option)
- anima3 `a6977e9` plus `439e5f5` (warrior healing follow-up)
- anima-client `581c511`

The new shard uses fresh accounts/world data. Local acceptance saves were not
copied.

## Live verification (2026-09-27)

- Public IPv4 TCP 2593 login, lobby, arena gump, mage ranked duel, rating update
  (0W 1L, 984) and alive return to lobby passed from an external Mac client bridge.
- Server authoritative leaderboard export agrees with the client result.
- Public mage/warrior workers and reference/challenger training workers run as
  systemd services. All six services, including shard and learner, are enabled.
- Training match completed as valid/unrated in the trusted server event stream
  after the corrected shard restart; the next policy match started automatically. Learner reports `collecting_training`, with
  `baseline-v1` still active. No policy improvement has yet been demonstrated.
- Explicit operator world save succeeded; its first snapshot and policy files
  were copied to the operator Mac under ignored `.logs/linode-first-save.tar.gz`.
- Cloud firewall permits TCP 22 and 2593; public monitor endpoints are closed.
- Cloudflare A record `arena.uotavern.com` points to `172.234.206.216`, DNS only.
  Public resolver 1.1.1.1 returns the expected IP.

Acceptance evidence is private under `.logs/linode-live-mage.json`. Run another
real public match (this changes only the designated test account's record):

```sh
PYTHONPATH=. python scripts/arena_deployment_check.py \
  --host 172.234.206.216 --credentials /private/path/accounts.json \
  --build mage --out /private/path/result.json
```

Cloudflare API authentication was provisioned using the operator's existing
`~/dev/key/cloudflare` token manager. The DNS token is scoped to `uotavern.com`
(Zone DNS Edit + Zone Read), with its private file in ignored
`.logs/cloudflare-dns.json`. The temporary discovery token was revoked.
Do not publish token values.

### Deployment incident

The default Rising Tide ocean spawner entered an unbounded deep-water search
while game data was still being transferred. A Mono thread dump located the
blocked main thread in `PlunderBeaconSpawner.Spawn` / `ValidateDeepWater`.
The shard now sets `Arena.DisableRisingTide=true`, which deactivates any persisted
spawner during initialization and prevents the unrelated event from starting.
The general world spawner algorithm is unchanged. The interrupted warrior test
is not counted as a successful acceptance test. After the corrected restart,
domain login, mage ranked match completion, rating update (0W 2L, 969), live
recovery, and a complete self-play match passed. The previous 984 rating survived
restart. Full remote warrior matches subsequently passed: 0W 1L / 984, then
0W 2L / 969 with alive lobby recovery. Evidence is in
`.logs/linode-domain-warrior.json` and `.logs/linode-domain-warrior-fixed.json`.
The later worker follow-up corrects critical-health retreat and waits for bandage
completion instead of restarting healing on a slip or a short tick timeout.
Final deployed acceptance: 0W 3L / 954, both rounds completed (50s / 34s),
all six external checks passed in `.logs/linode-domain-warrior-healing.json`.


Domain acceptance evidence: `.logs/linode-domain-mage.json`. Source compilation
passed with zero warnings/errors; deployment checker passed Ruff. The deployed
Scripts.dll SHA-256 is
`053f7e4a242d1c5006868bd21f8c1aee5cb43cd97f0421cd7d9436d3d029e789`.



