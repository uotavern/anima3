"""Run two participant-owned agents against a peer arena, then verify both receipts.

Use designated test accounts: this changes their skills/stats and ranked records.
No server worker or allowlist is required. Credentials are never printed.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--host", default="arena.uotavern.com")
ap.add_argument("--port", type=int, default=2593)
ap.add_argument("--credentials", type=Path, required=True)
ap.add_argument("--users", nargs=2, required=True)
ap.add_argument("--build", choices=["mage", "warrior"], default="mage")
ap.add_argument("--practice", action="store_true")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
if args.users[0] == args.users[1]:
    ap.error("two different accounts are required")
credentials = json.loads(args.credentials.read_text())
args.out.mkdir(parents=True, exist_ok=False)
workers = []
streams = []
checks = {}
try:
    for i, user in enumerate(args.users):
        directory = args.out / str(i)
        directory.mkdir()
        stream = (directory / "process.log").open("w")
        streams.append(stream)
        command = [sys.executable, "-m", "anima3.arena", "--host", args.host,
                   "--port", str(args.port), "--user", user, "--build", args.build,
                   "--matches", "1", "--once", "--log-dir", str(directory)]
        if args.practice:
            command.append("--practice")
        workers.append(subprocess.Popen(command, env=dict(os.environ, ARENA_BOT_PASSWORD=credentials[user]),
                                        stdout=stream, stderr=subprocess.STDOUT))
        if i == 0:
            time.sleep(10)
            checks["waits_for_other_participant"] = not list(directory.glob("*/round-*.jsonl"))
            print("First participant waiting; starting second participant", flush=True)
    deadline = time.monotonic() + 630
    for worker in workers:
        worker.wait(timeout=max(1, deadline - time.monotonic()))
    checks["both_agents_completed"] = all(w.returncode == 0 for w in workers)
    receipts = []
    for i in range(2):
        path = args.out / str(i) / "worker.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        results = [r["result"] for r in rows if r["event"] == "server_result"]
        receipts.append(results[-1] if results else {})
    a, b = receipts
    checks["matching_server_receipts"] = bool(a) and a == b
    checks["peer_mode"] = a.get("mode") == "peer_agents"
    checks["distinct_players"] = a.get("a") != a.get("b")
    checks["completed"] = a.get("aborted") is False
    checks["rating_mode"] = a.get("rated") is (not args.practice)
    checks["both_logged_actions"] = all(list((args.out / str(i)).glob("*/round-*.jsonl")) for i in range(2))
    (args.out / "result.json").write_text(json.dumps({"checks": checks, "receipts": receipts}, indent=2))
    print(json.dumps(checks, indent=2), flush=True)
finally:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=8)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
    for stream in streams:
        stream.close()
if not checks or not all(checks.values()):
    raise SystemExit(1)
