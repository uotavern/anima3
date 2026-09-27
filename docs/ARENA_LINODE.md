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
- anima3 `a6977e9` (application code)
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
restart. Full remote warrior match acceptance remains unverified.

Domain acceptance evidence: `.logs/linode-domain-mage.json`. Source compilation
passed with zero warnings/errors; deployment checker passed Ruff. The deployed
Scripts.dll SHA-256 is
`053f7e4a242d1c5006868bd21f8c1aee5cb43cd97f0421cd7d9436d3d029e789`.



