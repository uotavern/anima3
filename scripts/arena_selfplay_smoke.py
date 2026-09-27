"""Exercise two training workers on an isolated local shard; retain server proof."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from anima3.arena_learning import matches

ap = argparse.ArgumentParser()
ap.add_argument("--runtime", type=Path, required=True)
ap.add_argument("--port", type=int, default=2597)
ap.add_argument("--seconds", type=int, default=600)
args = ap.parse_args()
credentials = json.loads((args.runtime / "test-accounts.json").read_text())
events = args.runtime / "Logs" / "Arena" / "events.jsonl"
before = {row["id"] for row in matches(events)}
workers = []
try:
    for name, extra in (("arena_train_a", []), ("arena_train_b", ["--explore", "--seed", "7"])):
        env = dict(os.environ, ARENA_BOT_PASSWORD=credentials[name])
        log_path = Path(".logs") / (name + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            workers.append(subprocess.Popen([sys.executable, "-m", "anima3.arena", "--hosted-worker", "--host", "127.0.0.1", "--port", str(args.port), "--user", name,
                                              "--training", "--log-dir", ".logs/" + name, "--once", *extra], env=env, stdout=log, stderr=log))
    until = time.monotonic() + args.seconds
    while time.monotonic() < until:
        if any(w.poll() is not None for w in workers):
            raise RuntimeError("a training worker exited; inspect .logs/arena_train_*.log")
        found = [row for row in matches(events) if row["id"] not in before and row.get("training")]
        if found:
            Path(".logs/arena-selfplay-smoke.json").write_text(json.dumps(found, indent=2))
            print(json.dumps(found, indent=2), flush=True)
            if not all(row["valid"] and not row["aborted"] for row in found):
                raise RuntimeError("self-play did not produce a valid result")
            break
        time.sleep(1)
    else:
        raise RuntimeError("self-play match did not finish in time")
finally:
    for worker in workers:
        worker.terminate()
    for worker in workers:
        try:
            worker.wait(timeout=8)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
