"""Participant-run AI agent for the shared Arena service.

No GM credentials, invented outcomes or client-side ratings. The shard leases a
worker only while it is online and announcing readiness. Policies change between
matches, and the normal action/observation bridge drives both supported clients.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import time
from pathlib import Path

from .agent import Agent
from .body import BridgeBody
from .contract import Observation, say, use
from .decision import build_client
from .magic import PLAYBOOKS
from .persona import Persona

TOKEN = re.compile(r"^[a-zA-Z0-9_-]{1,48}$")
MATCH_ID = re.compile(r"^[0-9a-f]{32}$")


def read_policy(path: Path | None, build: str = "mage") -> dict:
    policy = {"schema": 1, "version": "baseline-v1", "build": build, "playbook": "standard"}
    if path is not None:
        policy = json.loads(path.read_text())
    if (policy.get("schema") != 1 or policy.get("build") != build
            or not TOKEN.fullmatch(str(policy.get("version", "")))
            or policy.get("playbook") not in PLAYBOOKS):
        raise ValueError("invalid arena policy (schema, version, build or playbook)")
    if build == "warrior" and policy["playbook"] != "standard":
        raise ValueError("warrior workers currently use the standard melee rule")
    return policy


def curriculum_policy(directory: Path, rng: random.Random) -> dict:
    """Explore until a frozen candidate is ready; evaluate it until a verdict exists."""
    champion = read_policy(directory / "champion.json")
    candidate_path = directory / "candidate.json"
    if candidate_path.exists():
        candidate = read_policy(candidate_path)
        verdict_path = directory / "evaluation.json"
        verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}
        if (candidate.get("parent") == champion["version"]
                and not (verdict.get("candidate") == candidate["version"]
                         and verdict.get("status") in ("rejected", "promoted"))):
            return candidate
    chosen = rng.choice(list(PLAYBOOKS))
    return {"schema": 1, "build": "mage", "version": "explore-" + chosen, "playbook": chosen}


def server_state(obs: Observation) -> dict | None:
    """Only server system messages may assign an opponent, never player speech."""
    result = None
    for j in obs.new_journal:
        if j.serial not in (0, 0xFFFFFFFF, -1) or not j.text.startswith("[ArenaState] "):
            continue
        try:
            state = json.loads(j.text[len("[ArenaState] "):])
            if state.get("phase") == "Idle" or (state.get("phase") in ("Countdown", "Fighting", "RoundOver")
                  and MATCH_ID.fullmatch(str(state.get("id", "")))
                  and type(state.get("opponent")) is int and state["opponent"] > 0
                  and type(state.get("round")) is int and 1 <= state["round"] <= 15
                  and state.get("build") in ("mage", "warrior")):
                result = state
        except (ValueError, AttributeError):
            continue
    return result


class ObservedBody:
    """Give Agent the already inspected observation, including all its journal lines."""
    def __init__(self, body: BridgeBody) -> None:
        self.body = body
        self.pending: Observation | None = None

    def observe(self) -> Observation:
        if self.pending is not None:
            obs, self.pending = self.pending, None
            return obs
        return self.body.observe()

    def act(self, action: dict) -> None:
        self.body.act(action)

    def pump(self, ms: int) -> int:
        return self.body.pump(ms)

    def close(self) -> None:
        self.body.close()


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps({"utc": time.time(), **event}, ensure_ascii=False) + "\n")


def serve(args, stopped) -> None:
    password = os.environ.get(args.password_env)
    if not password:
        raise ValueError(f"set {args.password_env} to the worker account password")
    persona = Persona.load("mage_a" if args.build == "mage" else "duelist_a")
    log = Path(args.log_dir)
    life_log = log / "worker.jsonl"
    rng = random.Random(args.seed)
    policy_path = Path(args.policy) if args.policy else None
    # Fail on a bad initial configuration instead of reconnecting forever.
    policy = read_policy(policy_path, args.build)
    body = BridgeBody.spawn(args.host, args.port, args.user, password,
                            binary=args.bridge, data_dir=args.data_dir)
    view = ObservedBody(body)
    state: dict = {}
    connected_at = time.monotonic()
    state_at = 0.0
    ping_at = ready_at = 0.0
    round_key = None
    last_match = None
    agent = None
    opened_pack = False
    completed = 0
    try:
        while not stopped():
            obs = body.observe()
            update = server_state(obs)
            now = time.monotonic()
            if update is not None:
                state, state_at = update, now
            if not opened_pack and obs.backpack_serial():
                body.act(use(obs.backpack_serial()))
                opened_pack = True
            if not state and now - connected_at > 20:
                raise RuntimeError("no Arena handshake: check server mode, account access and readiness arguments")
            phase = state.get("phase") if now - state_at < 5.0 else None
            if phase == "Idle" or not state:
                if last_match is not None:
                    append_event(life_log, {"event": "match_released", "id": last_match})
                    result = state.get("result")
                    if isinstance(result, dict) and result.get("id") == last_match:
                        append_event(life_log, {"event": "server_result", "result": result})
                    completed += 1
                    if args.matches and completed >= args.matches:
                        return
                    last_match = None
                    round_key = None
                    agent = None
                    ready_at = 0
                if now - ready_at >= 3:
                    try:
                        policy = read_policy(policy_path, args.build)
                    except (OSError, ValueError) as e:
                        # Do not announce readiness with a missing/malformed policy.
                        append_event(life_log, {"event": "policy_error", "error": str(e)})
                        body.pump(1000)
                        continue
                    if args.explore or args.curriculum:
                        # One random arm for the next match; keep it stable through ready refreshes.
                        if round_key is None:
                            if args.curriculum:
                                policy = curriculum_policy(Path(args.curriculum), rng)
                            else:
                                chosen = rng.choice(list(PLAYBOOKS))
                                policy = {"schema": 1, "build": "mage", "version": "explore-" + chosen, "playbook": chosen}
                            round_key = ("ready", policy)
                        else:
                            policy = round_key[1]
                    mode = ("training" if args.training else "public") if args.hosted_worker else ("practice" if args.practice else "ranked")
                    body.act(say(f"[ArenaReady {args.build} {policy['version']} {policy['playbook']} {mode}"))
                    ready_at = now
            elif state.get("id"):
                last_match = state["id"]
            if now - ping_at >= 1:
                body.act(say("[ArenaState"))
                ping_at = now
            if phase == "Fighting" and state.get("build") == args.build:
                key = (state["id"], state["round"])
                if key != round_key:
                    round_key = key
                    # The shard echoes the bound policy, including after a worker restart.
                    book = state.get("playbook", policy["playbook"])
                    if book not in PLAYBOOKS:
                        raise ValueError("server assigned an unknown playbook")
                    agent = Agent(view, persona, build_client("scripted"), pump_ms=args.pump_ms,
                                  reflect_every=0, log_path=log / state["id"] / f"round-{state['round']}.jsonl")
                    agent.memory.update(duel=True, mage=args.build == "mage", duel_opponent=state["opponent"],
                                        duel_round=state["round"], playbook=book)
                    append_event(life_log, {"event": "round_start", **state})
                view.pending = obs
                agent.tick()
                # The journal already has a bounded window; bound long-running diagnostics too.
                del agent.reports[:-2000]
                del agent.proc_log[:-2000]
            else:
                if agent is not None:
                    agent = None
                    round_key = None
                    body.act({"type": "WarMode", "on": False})
                body.pump(args.pump_ms)
    finally:
        body.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="arena.uotavern.com")
    ap.add_argument("--port", type=int, default=2593)
    ap.add_argument("--user", required=True)
    ap.add_argument("--password-env", default="ARENA_BOT_PASSWORD")
    ap.add_argument("--build", choices=["mage", "warrior"], default="mage")
    ap.add_argument("--policy", help="champion/candidate JSON, reread only between matches")
    ap.add_argument("--hosted-worker", action="store_true", help="legacy operator-owned AI shard only")
    ap.add_argument("--practice", action="store_true", help="participant practice queue; no rating")
    ap.add_argument("--matches", type=int, default=0, help="stop after N matches; 0 keeps queueing")
    ap.add_argument("--training", action="store_true", help="legacy private training shard only")
    ap.add_argument("--explore", action="store_true", help="training mage: random playbook per match")
    ap.add_argument("--curriculum", help="policy directory: automatically switch exploration and candidate evaluation")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--bridge")
    ap.add_argument("--data-dir")
    ap.add_argument("--log-dir", default=".logs/arena-worker")
    ap.add_argument("--pump-ms", type=int, default=250)
    ap.add_argument("--once", action="store_true", help="exit on disconnect instead of reconnecting")
    args = ap.parse_args(argv)
    if args.matches < 0:
        ap.error("--matches must be nonnegative")
    if args.training and not args.hosted_worker:
        ap.error("--training requires --hosted-worker; use --practice for participant matches")
    if args.practice and args.hosted_worker:
        ap.error("--practice is for participant mode")
    if (args.explore or args.curriculum) and (not args.training or args.build != "mage"):
        ap.error("--explore/--curriculum requires --training --build mage")
    if not 50 <= args.pump_ms <= 1000:
        ap.error("--pump-ms must be between 50 and 1000")
    read_policy(Path(args.policy) if args.policy else None, args.build)
    if not os.environ.get(args.password_env):
        ap.error(f"set {args.password_env}")
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    while not stopping:
        try:
            serve(args, lambda: stopping)
            if args.matches:
                break
        except Exception as e:
            append_event(Path(args.log_dir) / "worker.jsonl", {"event": "worker_error", "error": str(e)})
            if args.once:
                raise
            for _ in range(20):
                if stopping:
                    break
                time.sleep(0.25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
