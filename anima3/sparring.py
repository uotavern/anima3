"""Two ordinary clients spar in unrated training duels; learn from verified server replays.

Policy labels are local experiment metadata. Match identity/outcomes come from the
server replay. Training and fixed-sample evaluation remain separate.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import replay
from .agent import Agent
from .arena import ObservedBody, append_event, read_policy
from .arena_learning import BASELINE, atomic_json, cycle
from .body import BridgeBody
from .contract import say
from .decision import build_client
from .duel_wait import RULES, prepare, pump, server_state
from .magic import PLAYBOOKS
from .persona import Persona

TRAINING_RULES = RULES + "-training"


def poll(body):
    body.act(say("[DuelState"))
    until = time.monotonic() + 10
    while time.monotonic() < until:
        body.pump(100)
        state = server_state(body.observe())
        if state is not None:
            return state
    raise RuntimeError("no trusted DuelState")


def policies(directory, game, minimum, sample):
    champion = read_policy(directory / "champion.json")
    candidate_path = directory / "candidate.json"
    if candidate_path.exists():
        candidate = read_policy(candidate_path)
        verdict_path = directory / "evaluation.json"
        verdict = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}
        if candidate.get("parent") == champion["version"] and verdict.get("status") == "evaluating":
            return candidate, champion, "evaluation"
    book = list(PLAYBOOKS)[(game // 2) % len(PLAYBOOKS)]
    return (
        {"schema": 1, "build": "mage", "version": "explore-" + book, "playbook": book},
        champion,
        "exploration",
    )


def fight(bodies, policies_by_side, log, stop):
    states = [{}, {}]
    agents = [None, None]
    views = [ObservedBody(b) for b in bodies]
    keys = [None, None]
    seen = [False, False]
    assigned_id = None
    ping = [0, 0]
    fresh = [time.monotonic(), time.monotonic()]
    deadline = time.monotonic() + 340
    while not stop() and time.monotonic() < deadline:
        for i, body in enumerate(bodies):
            obs = body.observe()
            update = server_state(obs)
            now = time.monotonic()
            if update is not None:
                states[i], fresh[i] = update, now
            state = states[i]
            if now - fresh[i] > 15:
                raise RuntimeError("lost server match state")
            if now - ping[i] >= 0.8:
                body.act(say("[DuelState"))
                ping[i] = now
            if state.get("id"):
                if assigned_id is not None and state["id"] != assigned_id:
                    raise RuntimeError("unexpected match assigned")
                assigned_id = state["id"]
                seen[i] = True
                if state.get("rules") != TRAINING_RULES:
                    raise RuntimeError("unexpected sparring rules")
            if state.get("phase") == "Fighting" and now - fresh[i] < 3:
                key = (assigned_id, state["round"])
                if key != keys[i]:
                    keys[i] = key
                    agents[i] = Agent(
                        views[i],
                        Persona.load("mage_a"),
                        build_client("scripted"),
                        pump_ms=100,
                        reflect_every=0,
                        log_path=log / assigned_id / f"player-{i}-round-{state['round']}.jsonl",
                    )
                    agents[i].memory.update(
                        duel=True,
                        mage=True,
                        explosion_potions=True,
                        duel_opponent=state["opponent"],
                        duel_round=state["round"],
                        playbook=policies_by_side[i]["playbook"],
                    )
                agents[i].memory["showdown"] = state.get("showdown") is True
                views[i].pending = obs
                agents[i].tick()
                del agents[i].reports[:-1000]
                del agents[i].proc_log[:-1000]
            else:
                agents[i], keys[i] = None, None
                body.pump(100)
        if all(seen) and all(s.get("phase") == "Idle" for s in states):
            return assigned_id
    raise RuntimeError("sparring stopped or timed out; an active duel follows server forfeit rules")


def receipt(base, match_id, ids, log):
    for _ in range(20):
        meta = next((r for r in replay.index(base) if r["id"] == match_id), None)
        if meta:
            data, rows = replay.download(base, meta)
            if (
                not meta["training"]
                or meta["aborted"] is not None
                or meta["rules"] != TRAINING_RULES
                or [p["serial"] for p in rows[0]["players"]] != ids
            ):
                raise ValueError("server recording does not match this training fixture")
            (log / match_id / "replay.jsonl").write_bytes(data)
            atomic_json(log / match_id / "replay.meta.json", meta)
            atomic_json(log / match_id / "analysis.json", replay.metrics(rows))
            return meta, rows
        time.sleep(0.5)
    raise RuntimeError("server replay missing; no learning result will be invented")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="arena.uotavern.com")
    ap.add_argument("--port", type=int, default=2593)
    ap.add_argument("--web", default="https://arena.uotavern.com")
    ap.add_argument("--user-a", required=True)
    ap.add_argument("--user-b", required=True)
    ap.add_argument("--password-a-env", default="SPAR_PASSWORD_A")
    ap.add_argument("--password-b-env", default="SPAR_PASSWORD_B")
    ap.add_argument("--bridge")
    ap.add_argument("--data-dir")
    ap.add_argument("--matches", type=int, default=100, help="bounded run, including resumed games")
    ap.add_argument("--minimum", type=int, default=10, help="exploration games per playbook")
    ap.add_argument("--sample", type=int, default=40, help="fixed evaluation games, minimum 40")
    ap.add_argument("--log-dir", type=Path, default=Path(".logs/sparring"))
    args = ap.parse_args(argv)
    if args.user_a == args.user_b:
        ap.error("use two different dedicated accounts")
    if args.matches < 1 or args.minimum < 10 or args.sample < 40:
        ap.error("matches>=1, minimum>=10, sample>=40")
    if not all(os.environ.get(k) for k in (args.password_a_env, args.password_b_env)):
        ap.error("set both password environment variables")
    log = args.log_dir
    log.mkdir(parents=True, exist_ok=True)
    # Prevent two coordinators from controlling the same experiment directory.
    import fcntl

    lock = (log / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy_dir = log / "policies"
    policy_dir.mkdir(exist_ok=True)
    if not (policy_dir / "champion.json").exists():
        atomic_json(policy_dir / "champion.json", BASELINE)
    events = log / "learning.jsonl"
    events.touch(exist_ok=True)
    old = [json.loads(line) for line in events.read_text().splitlines()]
    completed = len(old)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    bodies = []
    try:
        for user, key in [(args.user_a, args.password_a_env), (args.user_b, args.password_b_env)]:
            body = BridgeBody.spawn(
                args.host,
                args.port,
                user,
                os.environ[key],
                binary=args.bridge,
                data_dir=args.data_dir,
            )
            bodies.append(body)
            pump(body, 5)
        if any(poll(b)["phase"] != "Idle" for b in bodies):
            raise RuntimeError("account already in a duel; wait for it to finish")
        while completed < args.matches and not stopping:
            trial, champion, stage = policies(policy_dir, completed, args.minimum, args.sample)
            chosen = [trial, champion] if completed % 2 == 0 else [champion, trial]
            with ThreadPoolExecutor(max_workers=2) as pool:
                players = list(pool.map(prepare, bodies))
            ids = [p["serial"] for p in players]
            # Swap challenger (arena side A) each match as well as the candidate account.
            # Keeping these independent preserves arena-side balance in evaluation.
            order = [0, 1] if (completed // 2) % 2 == 0 else [1, 0]
            challenger, receiver = [bodies[i] for i in order]
            challenger.act(say(f"[Challenge 0x{ids[order[1]]:X} 1 standard7-explosion-training"))
            invite = None
            for _ in range(20):
                invite = poll(receiver).get("challenge")
                if invite:
                    break
                receiver.pump(100)
            if (
                not invite
                or invite["rules"] != TRAINING_RULES
                or invite["opponent"] != ids[order[0]]
            ):
                raise RuntimeError("training invitation was not the requested peer/rules")
            receiver.act(say("[DuelAccept " + invite["id"]))
            status = {
                "stage": stage,
                "game": completed + 1,
                "limit": args.matches,
                "players": [
                    {"name": p["name"], "serial": p["serial"], "policy": policy}
                    for p, policy in zip(players, chosen)
                ],
            }
            atomic_json(log / "status.json", status)
            append_event(log / "progress.jsonl", status)
            match_id = fight(bodies, chosen, log, lambda: stopping)
            meta, rows = receipt(args.web, match_id, [ids[i] for i in order], log)
            row = {
                "schema": 1,
                "event": "match_end",
                "id": match_id,
                "valid": True,
                "training": True,
                "aborted": False,
                "build": "mage",
                "a": ids[order[0]],
                "b": ids[order[1]],
                "winner": meta["winner"],
                "policy_a": chosen[order[0]]["version"],
                "policy_b": chosen[order[1]]["version"],
                "playbook_a": chosen[order[0]]["playbook"],
                "playbook_b": chosen[order[1]]["playbook"],
                "source": "verified-server-replay",
                "sha256": meta["sha256"],
            }
            append_event(events, row)
            completed += 1
            verdict = cycle(events, policy_dir, args.minimum, args.sample, promote=True)
            status.update(
                stage=verdict["status"],
                completed=completed,
                replay=meta,
                analysis=replay.metrics(rows),
                learning=verdict,
            )
            atomic_json(log / "status.json", status)
            append_event(log / "progress.jsonl", status)
            print(
                json.dumps({"completed": completed, "replay": match_id, "learning": verdict}),
                flush=True,
            )
    except Exception as e:
        atomic_json(
            log / "status.json", {"stage": "stopped", "completed": completed, "error": str(e)}
        )
        raise
    finally:
        for body in bodies:
            body.close()
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
