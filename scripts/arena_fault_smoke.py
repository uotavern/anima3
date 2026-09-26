"""Local-only: an AI disconnect must not change a human's rating."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from anima3.arena_learning import matches
from anima3.body import BridgeBody
from anima3.contract import say, use

ap = argparse.ArgumentParser()
ap.add_argument("--runtime", type=Path, required=True)
ap.add_argument("--port", type=int, default=2597)
args = ap.parse_args()
creds = json.loads((args.runtime / "test-accounts.json").read_text())
board = args.runtime / "Export/Arena/leaderboard.json"
events = args.runtime / "Logs/Arena/events.jsonl"
before = json.loads(board.read_text(encoding="utf-8-sig"))["players"]
prior = {m["id"] for m in matches(events)}
worker = None
body = None
proof = {}
try:
    body = BridgeBody.spawn("127.0.0.1", args.port, "arena_playtest", creds["arena_playtest"])
    for _ in range(12):
        body.pump(250)
    body.observe()
    body.act(say("[ArenaReady mage spoof standard public"))
    body.act(say("[ArenaState"))
    body.pump(500)
    proof["player_cannot_register_bot"] = not any(j.text.startswith("[ArenaState]") for j in body.observe().new_journal)
    body.act(say("[Arena supplies")); body.pump(500)
    obs = body.observe()
    body.act(use(obs.backpack_serial())); body.pump(500)
    def reagents(obs):
        return {g: sum(i.amount for i in obs.own_pack() if i.graphic == g) for g in (0x0F7A, 0x0F7B, 0x0F84, 0x0F85, 0x0F86, 0x0F88, 0x0F8C, 0x0F8D)}
    first = reagents(body.observe())
    body.act(say("[Arena supplies")); body.pump(500)
    proof["refill_is_bounded"] = reagents(body.observe()) == first and min(first.values()) == 100
    worker = subprocess.Popen([sys.executable, "-m", "anima3.arena", "--host", "127.0.0.1", "--port", str(args.port), "--user", "arena_bot_mage", "--log-dir", ".logs/arena-fault-bot", "--once"],
                              env=dict(os.environ, ARENA_BOT_PASSWORD=creds["arena_bot_mage"]), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    body.act(say("[Arena join mage"))
    until = time.monotonic() + 120
    cut = False
    while time.monotonic() < until:
        body.pump(250)
        obs = body.observe()
        if not cut and any(j.text.startswith("[Duel] FIGHT!") for j in obs.new_journal):
            worker.terminate(); worker.wait(timeout=8); cut = True
        rows = [row for row in matches(events) if row["id"] not in prior and not row["training"]]
        if rows:
            proof["bot_disconnected"] = cut
            proof["result_not_rated"] = rows[0].get("rated") is False
            proof["result_not_training_eligible"] = rows[0]["valid"] is False
            after = json.loads(board.read_text(encoding="utf-8-sig"))["players"]
            proof["rankings_unchanged"] = before == after
            break
    else:
        raise RuntimeError("fault match did not complete")
finally:
    if body:
        body.close()
    if worker and worker.poll() is None:
        worker.terminate(); worker.wait(timeout=8)
    Path(".logs").mkdir(exist_ok=True)
    Path(".logs/arena-fault-smoke.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof, indent=2), flush=True)
if not proof or not all(proof.values()):
    raise SystemExit(1)
