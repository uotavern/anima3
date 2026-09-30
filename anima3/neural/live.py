"""Collect version-pinned policy trajectories from ordinary ServUO clients.

Each socket and executor has its own actor thread. Neural workers run in separate
processes. Only complete, checksum-verified unrated server matches become update
inputs. Rejected policy decisions make episodes ineligible; deterministic option
failure and recovery remain experience, avoiding success-conditioned sampling.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .. import replay
from ..agent import Agent
from ..arena import ObservedBody, append_event
from ..arena_learning import atomic_json
from ..arena_policy import load_policy
from ..contract import say
from ..duel_wait import prepare, pump, server_state
from ..persona import Persona
from ..sparring import (
    TRAINING_RULES,
    connect_training,
    poll,
    receipt_while_pumping,
    wait_training_idle,
)
from .executor import CombatExecutor
from .inference import InferenceClient
from .schema import ACTION_NAMES, schema_fingerprint

EPISODE_SCHEMA = "anima3-live-policy-v1"
EXECUTION_FINGERPRINT = hashlib.sha256(
    Path(__file__).with_name("executor.py").read_bytes()
).hexdigest()


def policy_override(event):
    """Internal option failure/recovery is experience, not a substituted action."""
    return event.get("policy_override") is True or event.get("scope") == "policy_override"


def dispatch_option(executor, action, frame, obs, now, fresh):
    """A sampled option may abort without substituting a different policy action.

    Mask changes during transport are an environmental execution outcome. Retain
    the original policy sample, including failures, instead of selecting only
    episodes whose actions happened to remain possible.
    """
    if action == 0:
        return {"execution_started": False, "cancellation_reason": None}
    if not fresh:
        return {"execution_started": False, "cancellation_reason": "context_stale"}
    if not frame.mask[action]:
        return {"execution_started": False, "cancellation_reason": "mask_changed"}
    started = executor.start(action, obs, now)
    return {
        "execution_started": started,
        "cancellation_reason": None if started else "executor_refused",
    }


class LiveReplayVerifier:
    """Revalidate retained server receipts, never trust an episode's boolean.

    Receipts are downloaded from the configured server by the collector. Hashes
    detect corruption; they are not a cryptographic server signature. Keep this
    directory under the same trust boundary as the configured shard.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def __call__(self, episode):
        try:
            match_id = episode["verification"]["match_id"]
            import re

            if not re.fullmatch(r"[a-f0-9]{32}", match_id):
                return False
            root = self.directory / match_id
            meta = json.loads((root / "replay.meta.json").read_text())
            rows = replay.validate((root / "replay.jsonl").read_bytes(), meta)
            participants = [p["serial"] for p in rows[0]["players"]]
            result = episode["final_result"]
            actor = episode["actor"]
            outcome = 0.0 if meta["winner"] is None else (1.0 if meta["winner"] == actor else -1.0)
            transitions = episode["transitions"]
            return bool(
                episode.get("schema") == EPISODE_SCHEMA
                and episode.get("domain") == "servuo"
                and episode.get("sampling") == "categorical"
                and episode.get("eligible") is True
                and not episode.get("invalid_reasons")
                and meta["id"] == match_id
                and meta["training"] is True
                and meta["aborted"] is None
                and meta["rules"] == TRAINING_RULES
                and rows[0].get("rules") == TRAINING_RULES
                and episode["verification"]["replay_sha256"] == meta["sha256"]
                and len(participants) == 2
                and actor in participants
                and episode["opponent"] in participants
                and actor != episode["opponent"]
                and result["winner"] == meta["winner"]
                and result["outcome"] == outcome
                and transitions
                and transitions[-1]["done"] is True
                and transitions[-1]["reward"] == outcome
                and all(t["reward"] == 0 and t["done"] is False for t in transitions[:-1])
                and all(t["accepted"] is True and t["dt"] > 0 for t in transitions)
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return False


def _actor(body, opponent, worker, log, stop, *, deadline=340, expected_rules=TRAINING_RULES):
    """All access to one BridgeBody belongs to this thread during a match."""
    state, last_state, last_ping = {}, time.monotonic(), 0.0
    match_id = epoch = None
    seen = False
    executor = CombatExecutor(opponent)
    transitions, invalid, latencies = [], set(), []
    events = []
    end = time.monotonic() + deadline
    baseline = None
    view = ObservedBody(body)
    start_times = []
    next_decision = 0.0
    seen_rounds = set()
    health_requests, last_health_request = 0, float("-inf")

    def emit(row):
        # Bounded per-game memory; raw events are flushed once a match ID exists.
        row = {"clock": time.monotonic(), **row}
        if match_id:
            append_event(log / match_id / f"actor-{body_id}.jsonl", row)
        elif len(events) < 1000:
            events.append(row)

    def rpc(operation, call, *args):
        started_at = time.monotonic()
        try:
            return call(*args)
        finally:
            finished_at = time.monotonic()
            emit(
                {
                    "type": "bridge_rpc",
                    "operation": operation,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_ms": (finished_at - started_at) * 1000,
                }
            )

    def drain_worker_events():
        if worker:
            for event in worker.drain_events():
                emit(event)
                if (
                    event.get("type") == "inference_error"
                    or event.get("type") == "inference_rejected"
                    and event.get("epoch", epoch) == epoch
                ):
                    invalid.add(event.get("reason", event["type"]))

    body_id = body.observe().player.serial
    while not stop.is_set() and time.monotonic() < end:
        obs = rpc("observe", body.observe)
        observation_at = now = time.monotonic()
        enemy = next((m for m in obs.mobiles if m.serial == opponent), None)
        update = server_state(obs)
        if update is not None:
            state, last_state = update, now
            if state.get("id"):
                if (
                    match_id is not None
                    and state["id"] != match_id
                    or state.get("opponent") != opponent
                    or state.get("rules") != expected_rules
                ):
                    raise RuntimeError("unexpected live match assignment")
                if match_id is None:
                    match_id = state["id"]
                    (log / match_id).mkdir(parents=True, exist_ok=True)
                    for row in events:
                        emit(row)
                    events.clear()
                seen = True
            emit({"type": "server_state", "state": state})
        if now - last_ping >= 0.8:
            rpc("act:duel_state", body.act, say("[DuelState"))
            last_ping = now
            emit(
                {
                    "type": "state_requested",
                    "last_state_age": now - last_state,
                    "journal_count": len(obs.new_journal),
                    "hp": obs.player.hits,
                    "position": [obs.player.pos.x, obs.player.pos.y],
                    "opponent_health_known": bool(enemy and enemy.hits_max > 0),
                    "bridge": getattr(body, "diagnostics", {}),
                }
            )
        if now - last_state > 20:
            raise RuntimeError("lost authoritative live duel state")
        if (
            seen
            and state.get("phase") in ("Countdown", "Fighting")
            and now - last_state < 3
            and health_requests < 3
            and now - last_health_request >= 1
            and enemy is not None
            and not enemy.hidden
            and enemy.hits_max <= 0
        ):
            # The ordinary client requests the same public health bar.
            # No private opponent mana, skills or inventory are used.
            rpc("act:status_request", body.act, {"type": "StatusRequest", "serial": opponent})
            health_requests += 1
            last_health_request = now
            emit({"type": "public_health_requested", "opponent": opponent})
        final_training_round = (
            expected_rules == TRAINING_RULES
            and state.get("phase") == "RoundOver"
            and state.get("round") == 1
            and seen_rounds == {1}
        )
        if seen and (state.get("phase") == "Idle" or final_training_round):
            # The training invitation requests one round. Stop recording when
            # that round ends; run() still requires its completed server replay
            # before assigning any terminal reward or accepting the episode.
            # A deadline discovered on the final iteration must not disappear
            # merely because the authoritative match result has now arrived.
            drain_worker_events()
            if start_times:
                transitions[-1]["dt"] = max(0.001, now - start_times[-1])
            return {
                "id": match_id,
                "actor": body_id,
                "opponent": opponent,
                "transitions": transitions,
                "invalid_reasons": sorted(invalid),
                "inference_ms": latencies,
                "policy_sha": worker.policy_sha if worker else None,
                "collection_end_phase": state.get("phase"),
            }

        if state.get("phase") == "Fighting":
            current_epoch = f"{match_id}:{state['round']}:{body_id}"
            if current_epoch != epoch:
                seen_rounds.add(state["round"])
                if len(seen_rounds) > 1:
                    invalid.add(
                        "multiple_rounds"
                    )  # This collector intentionally requests one round.
                epoch = current_epoch
                executor.reset(opponent, epoch)
                if worker:
                    worker.reset(epoch)
                else:
                    baseline = Agent(
                        view,
                        Persona.load("mage_a"),
                        load_policy("scripted"),
                        sync=True,
                        pump_ms=50,
                        reflect_every=0,
                        log_path=log / match_id / f"baseline-{body_id}.jsonl",
                    )
                    baseline.memory.update(
                        duel=True,
                        mage=True,
                        explosion_potions=True,
                        opening="weaken-clumsy",
                        burst_combo=False,
                        duel_opponent=opponent,
                        duel_round=state["round"],
                        duel_rules=TRAINING_RULES,
                        playbook="standard",
                    )
            if worker:
                frame = executor.observe(obs, server_state=update, now=now)
                # Advance an already committed option even across brief state latency.
                for packet in executor.tick(obs, time.monotonic()):
                    rpc(f"act:{packet.get('type', 'unknown')}", body.act, packet)
                for event in executor.drain_events():
                    emit(event)
                    if policy_override(event):
                        invalid.add("policy_override")
                # RPCs above can block. Deadline checks and transition durations
                # use the actual consumption clock, never the old loop clock.
                now = time.monotonic()
                decision = worker.take(epoch, now)
                drain_worker_events()
                if worker.startup_error:
                    raise RuntimeError("neural worker failed during match")
                if decision is not None:
                    execution = dispatch_option(
                        executor, decision.action, frame, obs, now, now - last_state < 3
                    )
                    if transitions:
                        transitions[-1]["dt"] = max(0.001, now - start_times[-1])
                    transitions.append(
                        {
                            "features": list(decision.features),
                            "mask": list(decision.mask),
                            "action": decision.action,
                            "log_prob": decision.log_prob,
                            "value": decision.value,
                            "hidden_in": list(decision.hidden_in),
                            "reward": 0.0,
                            "done": False,
                            "dt": 0.001,
                            "accepted": True,
                            "policy_accepted": True,
                            **execution,
                        }
                    )
                    start_times.append(now)
                    latencies.append(decision.infer_ms)
                    emit(
                        {
                            "type": "policy_action",
                            "action": ACTION_NAMES[decision.action],
                            "request_id": decision.request_id,
                            "infer_ms": decision.infer_ms,
                            "response_age_ms": decision.age_ms,
                            "observation_at": decision.observed_at,
                            "submitted_at": decision.submitted_at,
                            "received_at": decision.received_at,
                            "taken_at": now,
                            "observation_age_ms": (now - decision.observed_at) * 1000,
                            "response_wait_ms": (now - decision.received_at) * 1000,
                            **execution,
                        }
                    )
                submit_now = time.monotonic()
                if submit_now >= next_decision and not worker.busy and submit_now - last_state < 3:
                    worker.submit(
                        executor.observe(obs, now=submit_now), epoch, observed_at=observation_at
                    )
                    next_decision = time.monotonic() + 0.25
                drain_worker_events()
                rpc("pump", body.pump, 50)
            elif now - last_state < 3:
                baseline.memory["showdown"] = state.get("showdown") is True
                view.pending = obs
                baseline.tick()
                del baseline.reports[:-1000]
                del baseline.proc_log[:-1000]
            else:
                rpc("pump", body.pump, 50)
        else:
            # Never execute offensive actions during countdown/round-over/lobby.
            if epoch is not None:
                executor.reset(opponent, "inactive")
                if worker:
                    worker.reset("inactive")
                epoch = None
            rpc("pump", body.pump, 50)
    raise RuntimeError("live match stopped or timed out; no unverified training result emitted")


def fight(bodies, workers, log, stop):
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = [b.observe().player.serial for b in bodies]
        futures = [
            pool.submit(_actor, b, ids[1 - i], workers[i], log, stop) for i, b in enumerate(bodies)
        ]
        try:
            result = [None, None]
            for future in as_completed(futures):
                result[futures.index(future)] = future.result()
        except BaseException:
            stop.set()
            raise
    if result[0]["id"] != result[1]["id"]:
        raise RuntimeError("actors disagree on server match identity")
    return result


def add_arguments(ap):
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--opponent-checkpoint", type=Path)
    ap.add_argument("--opponent", choices=["scripted", "self"], default="scripted")
    ap.add_argument("--host", default="arena.uotavern.com")
    ap.add_argument("--port", type=int, default=2593)
    ap.add_argument("--web", default="https://arena.uotavern.com")
    ap.add_argument("--user-a", required=True)
    ap.add_argument("--user-b", required=True)
    ap.add_argument("--password-a-env", default="SPAR_PASSWORD_A")
    ap.add_argument("--password-b-env", default="SPAR_PASSWORD_B")
    ap.add_argument("--bridge")
    ap.add_argument("--data-dir")
    ap.add_argument("--matches", type=int, default=2)
    ap.add_argument("--deadline-ms", type=float, default=500)
    ap.add_argument("--log-dir", type=Path, default=Path(".logs/neural-live"))


def run(args, stop=None):
    stop = stop or threading.Event()
    if args.user_a == args.user_b or not 1 <= args.matches <= 100:
        raise ValueError("use two different dedicated accounts and matches in 1..100")
    if not all(os.environ.get(key) for key in [args.password_a_env, args.password_b_env]):
        raise ValueError("set both password environment variables")
    log = args.log_dir.resolve()
    log.mkdir(parents=True, exist_ok=True)
    lock = (log / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (log / "results.json").exists():
        raise ValueError("collection already contains verified results; choose a new --log-dir")
    bodies, workers, results = [], [], []

    def save_results(status, error=None):
        # A later transport failure must not hide earlier verified matches.
        atomic_json(
            log / "results.json",
            {
                "domain": "servuo",
                "matches": results,
                "automatic_promotion": False,
                "collection_status": status,
                "error": error,
            },
        )

    try:
        workers = [InferenceClient(args.checkpoint, deadline_ms=args.deadline_ms).start()]
        workers.append(
            InferenceClient(
                args.opponent_checkpoint or args.checkpoint, deadline_ms=args.deadline_ms
            ).start()
            if args.opponent == "self" or args.opponent_checkpoint
            else None
        )
        for user, key in [(args.user_a, args.password_a_env), (args.user_b, args.password_b_env)]:
            atomic_json(log / "status.json", {"stage": "connecting", "account": user})
            bodies.append(connect_training(args, user, os.environ[key], log, stop.is_set))
            pump(bodies[-1], 2)
        ready_deadline = time.monotonic() + 90
        while not all(w is None or w.ready for w in workers):
            if (
                stop.is_set()
                or time.monotonic() > ready_deadline
                or any(w and w.startup_error for w in workers)
            ):
                raise RuntimeError("neural inference worker startup failed")
            for body in bodies:
                body.pump(50)
        wait_training_idle(bodies, log, stop.is_set)
        for game in range(args.matches):
            if stop.is_set():
                break
            atomic_json(log / "status.json", {"stage": "preparing", "game": game + 1})
            configured_before = game > 0 or getattr(args, "known_build", False)
            with ThreadPoolExecutor(max_workers=2) as pool:
                players = list(
                    pool.map(
                        lambda b, configured=configured_before: prepare(
                            b, previously_configured=configured
                        ),
                        bodies,
                    )
                )
            ids = [p["serial"] for p in players]
            for i, body in enumerate(bodies):
                label = workers[i].policy_sha[:12] if workers[i] else "fixed-scripted"
                body.act(say(f"[ArenaAgent neural {label} recurrent"))
                pump(body, 0.5)
            fixture_index = game + getattr(args, "fixture_offset", 0)
            order = [0, 1] if fixture_index % 2 == 0 else [1, 0]
            challenger, receiver = [bodies[i] for i in order]
            challenger.act(say(f"[Challenge 0x{ids[order[1]]:X} 1 standard7-explosion-training"))
            pump(challenger, 0.6)
            invite = None
            for _ in range(20):
                invite = poll(receiver).get("challenge")
                if invite:
                    break
            if (
                not invite
                or invite["rules"] != TRAINING_RULES
                or invite["opponent"] != ids[order[0]]
            ):
                raise RuntimeError("invitation differs from expected training fixture")
            receiver.act(say("[DuelAccept " + invite["id"]))
            atomic_json(log / "status.json", {"stage": "fighting", "game": game + 1})
            actors = fight(bodies, workers, log, stop)
            match_id = actors[0]["id"]
            meta, rows = receipt_while_pumping(
                args.web, match_id, [ids[i] for i in order], log, bodies
            )
            files = []
            for i, actor in enumerate(actors):
                if workers[i] is None:
                    continue
                transitions = actor.pop("transitions")
                outcome = (
                    0.0 if meta["winner"] is None else (1.0 if meta["winner"] == ids[i] else -1.0)
                )
                if transitions:
                    transitions[-1].update(reward=outcome, done=True)
                episode = {
                    **actor,
                    "schema": EPISODE_SCHEMA,
                    "domain": "servuo",
                    "sampling": "categorical",
                    "executor_sha256": EXECUTION_FINGERPRINT,
                    "collector_version": "fixed-cadence-options-v4",
                    "schema_fingerprint": schema_fingerprint(),
                    "eligible": bool(transitions) and not actor["invalid_reasons"],
                    "transitions": transitions,
                    "verification": {
                        "verified": True,
                        "match_id": match_id,
                        "replay_sha256": meta["sha256"],
                        "complete": True,
                        "training": True,
                        "aborted": False,
                    },
                    "final_result": {"winner": meta["winner"], "outcome": outcome},
                }
                if episode["eligible"] and not LiveReplayVerifier(log)(episode):
                    raise RuntimeError("collected episode failed receipt verification")
                file = log / match_id / f"episode-{ids[i]}.json"
                atomic_json(file, episode)
                files.append(str(file))
            result = {
                "game": game + 1,
                "challenger": ids[order[0]],
                "match_id": match_id,
                "winner": meta["winner"],
                "metrics": replay.metrics(rows),
                "episodes": files,
                "replay": f"{args.web.rstrip('/')}/replay/?replay={match_id}",
            }
            results.append(result)
            save_results("collecting")
            atomic_json(log / "status.json", {"stage": "verified", **result})
            print(json.dumps(result), flush=True)
        save_results("stopped" if stop.is_set() else "complete")
        return results
    except BaseException as exc:
        if results:
            save_results("stopped", str(exc))
        atomic_json(log / "status.json", {"stage": "stopped", "error": str(exc)})
        raise
    finally:
        for worker in workers:
            if worker:
                worker.close()
        for body in bodies:
            body.close()
        lock.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    add_arguments(ap)
    args = ap.parse_args(argv)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    run(args, stop)
    return 0
