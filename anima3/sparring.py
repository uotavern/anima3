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
from .arena_learning import BASELINE, atomic_json, matches
from .arena_policy import load_policy
from .body import BodyError, BridgeBody
from .contract import say
from .duel_wait import RULES, prepare, pump, server_state
from .learning_run import experiment, update, verify_receipts
from .magic import PLAYBOOKS
from .persona import Persona
from .strategy import add_arguments, build_session

TRAINING_RULES = RULES + "-training"


def connect_training(args, user, password, log, stop):
    """Retry only initial login; never reissue an in-match action after reconnect."""
    for attempt in range(3):
        if stop():
            raise RuntimeError("sparring stopped during login")
        try:
            return BridgeBody.spawn(
                args.host, args.port, user, password, binary=args.bridge, data_dir=args.data_dir
            )
        except BodyError:
            append_event(
                log / "progress.jsonl",
                {
                    "stage": "login_retry",
                    "attempt": attempt + 1,
                    "account": user,
                },
            )
            if attempt == 2:
                raise RuntimeError("training login failed after 3 attempts") from None
            time.sleep(3 * (attempt + 1))


def poll(body):
    until = time.monotonic() + 30
    next_request = 0
    while time.monotonic() < until:
        if time.monotonic() >= next_request:
            body.act(say("[DuelState"))
            next_request = time.monotonic() + 2
        body.pump(100)
        state = server_state(body.observe())
        if state is not None:
            return state
    raise RuntimeError("no trusted DuelState")


def wait_training_idle(bodies, log, stop):
    """Wait out only this pair's previous training match, never take over another duel."""
    ids = [body.observe().player.serial for body in bodies]
    states, fresh, ping = [None, None], [0, 0], [0, 0]
    deadline = time.monotonic() + 360
    while not stop() and time.monotonic() < deadline:
        for i, body in enumerate(bodies):
            now = time.monotonic()
            if now - ping[i] >= 2:
                body.act(say("[DuelState"))
                ping[i] = now
            body.pump(100)
            state = server_state(body.observe())
            if state:
                states[i], fresh[i] = state, time.monotonic()
                if state["phase"] != "Idle" and (
                    state.get("rules") != TRAINING_RULES or state.get("opponent") != ids[1 - i]
                ):
                    raise RuntimeError("account is in an unrelated duel; refusing takeover")
        if all(s and s["phase"] == "Idle" for s in states) and all(
            time.monotonic() - t < 5 for t in fresh
        ):
            return
    raise RuntimeError("training accounts did not become idle within 360 seconds")


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


def fight(bodies, policies_by_side, log, stop, clients=None, strategies=None, opening="none", burst_combo=False):
    strategies = strategies or [None, None]
    clients = clients or [load_policy(), load_policy()]
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
            if now - fresh[i] > 45:
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
                        strategies[i].reflex if strategies[i] else clients[i],
                        sync=bool(strategies[i]) or clients[i].name == "scripted",
                        tactician=strategies[i].director(assigned_id, state["round"])
                        if strategies[i]
                        else None,
                        pump_ms=100,
                        reflect_every=0,
                        log_path=log / assigned_id / f"player-{i}-round-{state['round']}.jsonl",
                    )
                    agents[i].memory.update(
                        duel=True,
                        mage=True,
                        explosion_potions=True,
                        opening=opening,
                        burst_combo=burst_combo,
                        duel_opponent=state["opponent"],
                        duel_round=state["round"],
                        duel_rules=state["rules"],
                        playbook=policies_by_side[i]["playbook"],
                    )
                agents[i].memory["showdown"] = state.get("showdown") is True
                views[i].pending = obs
                agents[i].tick()
                del agents[i].reports[:-1000]
                del agents[i].proc_log[:-1000]
            else:
                # A brief stale state pauses actions, not the round's memory,
                # spell procedure or model context. Reset only on a confirmed
                # non-fighting phase; otherwise each delay restarts the brain.
                if state.get("phase") != "Fighting":
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


def receipt_while_pumping(base, match_id, ids, log, bodies):
    # Replay HTTP can be slow. Keep servicing both game sockets while the
    # verified recording downloads, before preparing the next match.
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(receipt, base, match_id, ids, log)
        while not result.done():
            for body in bodies:
                body.pump(100)
        return result.result()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    ap.add_argument("--burst-combo", action="store_true")
    ap.add_argument("--opening", choices=["none", "weaken-clumsy"], default="none")
    ap.add_argument("--opponent", choices=["same", "fixed-scripted"], default="same")
    ap.add_argument("--host", default="arena.uotavern.com")
    ap.add_argument("--port", type=int, default=2593)
    ap.add_argument("--web", default="https://arena.uotavern.com")
    ap.add_argument("--user-a", required=True)
    ap.add_argument("--user-b", required=True)
    ap.add_argument("--password-a-env", default="SPAR_PASSWORD_A")
    ap.add_argument("--password-b-env", default="SPAR_PASSWORD_B")
    ap.add_argument("--backend", choices=["scripted", "jev", "jeff", "qwen"], default="scripted")
    ap.add_argument(
        "--model", default="jev-1.13.0", help="Pinned Jev model for reproducible experiments"
    )
    ap.add_argument(
        "--max-model-calls", type=int, default=500, help="Provider calls per client per run"
    )
    ap.add_argument("--bridge")
    ap.add_argument("--data-dir")
    ap.add_argument("--matches", type=int, default=100, help="bounded run, including resumed games")
    ap.add_argument("--minimum", type=int, default=10, help="exploration games per playbook")
    ap.add_argument("--sample", type=int, default=40, help="fixed evaluation games, minimum 40")
    ap.add_argument("--log-dir", type=Path, default=Path(".logs/sparring"))
    args = ap.parse_args(argv)
    if args.opponent == "fixed-scripted" and args.brain != "hybrid":
        ap.error("fixed-scripted opponent requires --brain hybrid")
    if args.user_a == args.user_b:
        ap.error("use two different dedicated accounts")
    if args.matches < 1 or args.minimum < 10 or args.sample < 40 or args.max_model_calls < 1:
        ap.error("matches>=1, minimum>=10, sample>=40")
    if not all(os.environ.get(k) for k in (args.password_a_env, args.password_b_env)):
        ap.error("set both password environment variables")
    log = args.log_dir
    log.mkdir(parents=True, exist_ok=True)
    # Prevent two coordinators from controlling the same experiment directory.
    import fcntl

    lock = (log / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    experiment(log, args)
    clients = [
        load_policy(
            "scripted" if i == 1 and args.opponent == "fixed-scripted" else args.backend,
            model=args.model,
            max_calls=args.max_model_calls,
        )
        for i in range(2)
    ]
    strategies = [
        None
        if i == 1 and args.opponent == "fixed-scripted"
        else build_session(args, c, log / f"brain-{i}")
        for i, c in enumerate(clients)
    ]
    policy_dir = log / "policies"
    policy_dir.mkdir(exist_ok=True)
    if not (policy_dir / "champion.json").exists():
        atomic_json(
            policy_dir / "champion.json",
            {
                **BASELINE,
                "backend": args.backend,
                "model": args.model if args.backend == "jev" else args.backend,
            },
        )
    events = log / "learning.jsonl"
    events.touch(exist_ok=True)
    old = matches(events)
    verify_receipts(log, old)
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
            atomic_json(
                log / "status.json",
                {
                    "stage": "connecting",
                    "completed": completed,
                    "account": user,
                },
            )
            body = connect_training(args, user, os.environ[key], log, lambda: stopping)
            bodies.append(body)
            pump(body, 5)
        atomic_json(log / "status.json", {"stage": "waiting_for_idle", "completed": completed})
        wait_training_idle(bodies, log, lambda: stopping)
        while completed < args.matches and not stopping:
            if any(c.max_calls is not None and c.calls >= c.max_calls for c in clients):
                atomic_json(
                    log / "status.json", {"stage": "model_budget_exhausted", "completed": completed}
                )
                break
            trial, champion, stage = policies(policy_dir, completed, args.minimum, args.sample)
            if args.brain == "hybrid":
                trial = champion
                stage = "adaptive"
            chosen = [trial, champion] if completed % 2 == 0 else [champion, trial]
            if args.opponent == "fixed-scripted":
                chosen = [champion, {**BASELINE, "version": "fixed-scripted-v1"}]
            with ThreadPoolExecutor(max_workers=2) as pool:
                players = list(pool.map(lambda body, configured=completed > 0: prepare(body, previously_configured=configured), bodies))
            for i, (body, policy) in enumerate(zip(bodies, chosen)):
                body.act(
                    say(
                        f"[ArenaAgent {'jev+llm' if strategies[i] else clients[i].name} {policy['version']} {policy['playbook']}"
                    )
                )
                pump(body, 0.6)
            ids = [p["serial"] for p in players]
            # Swap challenger (arena side A) each match as well as the candidate account.
            # Keeping these independent preserves arena-side balance in evaluation.
            side = completed if args.opponent == "fixed-scripted" else completed // 2
            order = [0, 1] if side % 2 == 0 else [1, 0]
            challenger, receiver = [bodies[i] for i in order]
            challenger.act(say(f"[Challenge 0x{ids[order[1]]:X} 1 standard7-explosion-training"))
            pump(challenger, 0.6)
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
            pump(receiver, 0.6)
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
            match_id = fight(
                bodies, chosen, log, lambda: stopping, clients, strategies, args.opening, args.burst_combo
            )
            meta, rows = receipt_while_pumping(
                args.web, match_id, [ids[i] for i in order], log, bodies
            )
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
                "modelBudgetExhausted": any(c.exhausted for c in clients),
                "providerCalls": [c.calls for c in clients],
                "providerErrors": [c.errors for c in clients],
                "stage": stage,
                "brain": args.brain,
                "backend": args.backend,
                "model": args.model if args.backend == "jev" else args.backend,
                "sha256": meta["sha256"],
                "learner": ids[0] if args.opponent == "fixed-scripted" else None,
                "opponent": args.opponent,
                "opening": args.opening,
                "burstCombo": args.burst_combo,
            }
            append_event(events, row)
            completed += 1
            if args.brain == "hybrid":
                for session, serial in zip(strategies, ids):
                    if session is not None:
                        session.remember(meta, rows, serial)
                verdict = {"status": "adaptive_experience", "automaticPromotion": False}
            else:
                verdict = update(log, args.minimum, args.sample, clients)
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
