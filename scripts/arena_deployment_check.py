"""Complete one real duel against an already running public arena worker.

Uses a designated test account; this changes its skills/stats and ranked record.
Credentials are read from a local private JSON file and never printed.
"""
import argparse
import json
import time
from pathlib import Path

from anima3.body import BridgeBody
from anima3.contract import gump_response, say

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--host", required=True)
ap.add_argument("--port", type=int, default=2593)
ap.add_argument("--credentials", type=Path, required=True)
ap.add_argument("--user", default="arena_playtest")
ap.add_argument("--build", choices=["mage", "warrior"], default="mage")
ap.add_argument("--bridge", type=Path)
ap.add_argument("--data-dir", type=Path)
ap.add_argument("--seconds", type=int, default=600)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
credentials = json.loads(args.credentials.read_text())
report = {"host": args.host, "build": args.build, "checks": {}, "journal": []}
body = BridgeBody.spawn(args.host, args.port, args.user, credentials[args.user],
                        binary=args.bridge, data_dir=args.data_dir)
try:
    for _ in range(20):
        body.pump(250)
    obs = body.observe()
    report["checks"]["lobby"] = (obs.player.pos.x, obs.player.pos.y) == (5180, 332)
    body.act(say("[Arena"))
    for _ in range(8):
        body.pump(250)
    obs = body.observe()
    report["checks"]["menu"] = bool(obs.gumps)
    if not obs.gumps:
        raise RuntimeError("Arena menu missing")
    menu = obs.gumps[-1]
    body.act(gump_response(menu.serial, menu.gump_id, 2 if args.build == "mage" else 3))
    until = time.monotonic() + args.seconds
    while time.monotonic() < until:
        body.pump(250)
        obs = body.observe()
        for entry in obs.new_journal:
            if entry.text.startswith(("[Duel]", "[Arena]")):
                report["journal"].append(entry.text)
                print(entry.text, flush=True)
        if any(" rating (" + args.build + ")" in line for line in report["journal"]):
            break
    for _ in range(8):
        body.pump(250)
    obs = body.observe()
    report["checks"]["started"] = any("[Duel] Start:" in s for s in report["journal"])
    report["checks"]["finished"] = any("[Duel] Match:" in s for s in report["journal"])
    report["checks"]["rated"] = any(" rating (" + args.build + ")" in s
                                     for s in report["journal"])
    report["checks"]["recovered"] = not obs.player.dead and obs.player.pos.x == 5180
finally:
    body.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
print(json.dumps(report["checks"], indent=2))
if not all(report["checks"].values()):
    raise SystemExit(1)
