"""Local-only acceptance: regular player + a real AI worker over the UO protocol.

Run against a fresh dedicated shard on 127.0.0.1:2597. Accounts are test-only;
passwords are read from an external JSON file, never committed or printed.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from anima3.body import BridgeBody
from anima3.contract import gump_response, say, use

ap = argparse.ArgumentParser()
ap.add_argument("--credentials", type=Path, required=True)
ap.add_argument("--port", type=int, default=2597)
ap.add_argument("--build", choices=["mage", "warrior"], default="mage")
ap.add_argument("--practice", action="store_true")
ap.add_argument("--seconds", type=int, default=420)
ap.add_argument("--out", type=Path, default=Path(".logs/arena-smoke.json"))
args = ap.parse_args()
credentials = json.loads(args.credentials.read_text())
bot_user = "arena_bot_" + args.build
env = dict(os.environ, ARENA_BOT_PASSWORD=credentials[bot_user])
log_dir = Path(".logs") / ("arena-smoke-" + args.build + "-" + str(time.time_ns()))
worker = subprocess.Popen([sys.executable, "-m", "anima3.arena", "--hosted-worker", "--host", "127.0.0.1", "--port", str(args.port), "--user", bot_user, "--build", args.build, "--log-dir", str(log_dir), "--once"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
body = None
report = {"host": "127.0.0.1:" + str(args.port), "build": args.build, "practice": args.practice, "journal": [], "checks": {}}
try:
    body = BridgeBody.spawn("127.0.0.1", args.port, "arena_playtest", credentials["arena_playtest"])
    until = time.monotonic() + 8
    while time.monotonic() < until:
        body.pump(250)
        obs = body.observe()
        report["journal"].extend(j.text for j in obs.new_journal)
    report["checks"]["welcome_lobby"] = obs.player.pos.x == 5180 and obs.player.pos.y == 332
    report["checks"]["welcome_gump"] = bool(obs.gumps)
    body.act(say("[Arena supplies"))
    body.pump(500)
    obs = body.observe()
    if obs.backpack_serial():
        body.act(use(obs.backpack_serial()))
    body.pump(500)
    obs = body.observe()
    report["checks"]["supplies"] = any(i.graphic == 0x0F7A for i in obs.own_pack())
    body.act(say("[Arena style robe 6"))
    body.pump(500)
    body.act(say("[Arena"))
    body.pump(500)
    obs = body.observe()
    menu = obs.gumps[-1]
    button = (7 if args.build == "mage" else 8) if args.practice else (2 if args.build == "mage" else 3)
    body.act(gump_response(menu.serial, menu.gump_id, button))
    potion_before = None
    until = time.monotonic() + args.seconds
    while time.monotonic() < until:
        body.pump(250)
        obs = body.observe()
        if args.practice and potion_before is None and 0 < obs.player.hp_pct < 0.8:
            healing = [i for i in obs.own_pack() if i.graphic == 0x0F0C]
            if healing:
                potion_before = sum(i.amount for i in healing)
                body.act(use(healing[0].serial))
        for j in obs.new_journal:
            report["journal"].append(j.text)
            if j.text.startswith(("[Duel]", "[Arena]")):
                print(j.text, flush=True)
        if any(("[Duel] Match:" if args.practice else " rating (" + args.build + ")") in line for line in report["journal"]):
            break
        if worker.poll() is not None:
            raise RuntimeError("worker exited: " + worker.stderr.read()[-1200:])
    body.pump(500)
    obs = body.observe()
    report["checks"]["match_started"] = any("[Duel] Start:" in line for line in report["journal"])
    report["checks"]["match_finished"] = any("[Duel] Match:" in line for line in report["journal"])
    received_rating = any(" rating (" + args.build + ")" in line for line in report["journal"])
    report["checks"]["rating_mode_correct"] = not received_rating if args.practice else received_rating
    if args.practice:
        after = sum(i.amount for i in obs.own_pack() if i.graphic == 0x0F0C)
        report["checks"]["practice_potion_consumed"] = potion_before is not None and after < potion_before
    report["checks"]["restored_alive"] = not obs.player.dead and obs.player.pos.x == 5180
    report["checks"]["ai_action_logs"] = any(
        row.get("chosen", "").startswith(("cast:", "attack:")) or row.get("proc", "").startswith(("cast:", "attack:"))
        for path in log_dir.glob("*/round-*.jsonl")
        for row in (json.loads(line) for line in path.read_text().splitlines())
    )
    print(json.dumps(report["checks"], indent=2), flush=True)
finally:
    if body:
        body.close()
    worker.terminate()
    try:
        worker.wait(timeout=8)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
if not all(report["checks"].values()):
    raise SystemExit(1)
