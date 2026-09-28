"""Ordinary participant client: prepare 7GM and accept Standard 7x + explosion challenges."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from .agent import Agent
from .arena import MATCH_ID, ObservedBody, append_event, read_policy
from .arena_policy import LABEL, load_policy
from .body import BridgeBody
from .contract import say, use
from .magic import PLAYBOOKS
from .persona import Persona

RULES = "7x-magic-explosion-classic"
SKILLS = [25, 16, 46, 26, 43, 1, 0]


def server_state(obs):
    result = None
    for j in obs.new_journal:
        if j.serial not in (0, -1, 0xFFFFFFFF) or not j.text.startswith("[DuelState] "):
            continue
        try:
            s = json.loads(j.text[12:])
            if not isinstance(s, dict):
                continue

            def valid(v):
                return (
                    isinstance(v, dict)
                    and MATCH_ID.fullmatch(str(v.get("id", "")))
                    and type(v.get("opponent")) is int
                    and v["opponent"] > 0
                    and isinstance(v.get("rules"), str)
                )

            if s.get("phase") == "Idle":
                c = s.get("challenge")
                if c is None or (
                    valid(c) and type(c.get("rounds")) is int and 1 <= c["rounds"] <= 15
                ):
                    result = s
            elif (
                s.get("phase") in ("Countdown", "Fighting", "RoundOver")
                and valid(s)
                and type(s.get("round")) is int
                and 1 <= s["round"] <= 15
            ):
                result = s
        except (ValueError, TypeError):
            pass
    return result


def pump(body, seconds=2):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        body.pump(250)
    return body.observe_raw()


def prepare(body):
    """Use the same public commands and consumable dialogs as a human player."""
    for command in (
        "[Arena leave",
        "[Arena enter",
        "[Arena skills",
        "[Arena stats",
        "[Arena supplies",
    ):
        body.act(say(command))
        pump(body)
    o = body.observe_raw()
    pack = next(
        i for i in o["items"] if i.get("layer") == 21 and i["container"] == o["player"]["serial"]
    )
    body.act(use(pack["serial"]))
    o = pump(body)
    for hue, title, values in (
        (1153, "7GM SKILL BALL", {"switches": SKILLS}),
        (53, "ARENA STATS", {"entries": [[0, "100"], [1, "25"], [2, "100"]]}),
    ):
        ball = next(
            i
            for i in o["items"]
            if i["graphic"] == 0xE2D and i["hue"] == hue and i["container"] == pack["serial"]
        )
        body.act(use(ball["serial"]))
        g = None
        for _ in range(8):
            o = pump(body, 0.5)
            g = next((g for g in o["gumps"] if title in str(g)), None)
            if g:
                break
        if g is None:
            raise RuntimeError("Preparation dialog missing: " + title)
        body.act(
            dict(type="GumpResponse", serial=g["serial"], gump_id=g["gump_id"], button=1, **values)
        )
        o = pump(body, 3)
    body.act({"type": "SkillsRequest"})
    o = pump(body, 3)
    bases = {s.get("id", n): s["base"] for n, s in enumerate(o["skills"])}
    if any(bases.get(n) != 100 for n in SKILLS) or sum(bases.values()) != 700:
        raise RuntimeError("Server has not confirmed requested seven GM skills")
    if [o["player"][k] for k in ("strength", "dexterity", "intelligence")] != [100, 25, 100]:
        raise RuntimeError("Server has not confirmed 100/25/100 stats")
    return o["player"]


def serve(args, stopped, decision_client=None):
    log = Path(args.log_dir)
    life = log / "client.jsonl"
    decision_client = decision_client or load_policy(
        getattr(args, "backend", "scripted"),
        getattr(args, "decision_factory", None),
        model=getattr(args, "model", None),
        max_calls=getattr(args, "max_model_calls", None),
    )
    version = getattr(args, "version", "baseline-v1")
    policy = getattr(args, "policy", "standard")
    model = getattr(args, "model_label", None) or (
        "custom"
        if getattr(args, "decision_factory", None)
        else getattr(args, "backend", "scripted")
    )
    ranked = getattr(args, "ranked", False)
    listed_at = 0
    body = BridgeBody.spawn(
        args.host,
        args.port,
        args.user,
        os.environ[args.password_env],
        binary=args.bridge,
        data_dir=args.data_dir,
    )
    view = ObservedBody(body)
    state, state_at, ping_at, key, agent = {}, 0, 0, None, None
    prepared, last_match, last_invite = False, None, None
    started = time.monotonic()
    try:
        pump(body, 7)
        while not stopped():
            obs = body.observe()
            control_sent = False
            for line in obs.new_journal:
                if line.serial in (0, -1, 0xFFFFFFFF) and line.text.startswith(
                    ("[Arena]", "[Duel]")
                ):
                    append_event(life, {"event": "server_notice", "message": line.text})
            now = time.monotonic()
            update = server_state(obs)
            if update is not None:
                state, state_at = update, now
            if now - max(state_at, started) > 20:
                raise RuntimeError("No trusted DuelState handshake from server")
            phase = state.get("phase") if now - state_at < 5 else None
            if phase == "Idle":
                if last_match:
                    append_event(life, {"event": "match_released", "id": last_match})
                    prepared, last_match, key, agent = False, None, None, None
                if not prepared:
                    if getattr(args, "policy_file", None):
                        learned = read_policy(Path(args.policy_file))
                        if learned.get("backend", args.backend) != args.backend or (
                            args.backend == "jev" and learned.get("model", args.model) != args.model
                        ):
                            raise ValueError(
                                "Learned policy backend/model does not match this agent"
                            )
                        version, policy = learned["version"], learned["playbook"]
                        append_event(
                            life, {"event": "policy_loaded", "version": version, "playbook": policy}
                        )
                    player = prepare(body)
                    body.act(say(f"[ArenaAgent {model} {version} {policy}"))
                    body.pump(500)
                    ping_at = time.monotonic()
                    prepared = True
                    listed_at = 0
                    event = {
                        "event": "ready",
                        "name": player["name"],
                        "serial": player["serial"],
                        "rules": RULES,
                        "skills": SKILLS,
                        "model": model,
                        "version": version,
                        "policy": policy,
                        "ranked": ranked,
                    }
                    append_event(life, event)
                    print(json.dumps(event), flush=True)
                    state_at = time.monotonic()
                    continue
                if (
                    not getattr(args, "no_list", False)
                    and now - listed_at > 300
                    and not state.get("challenge")
                ):
                    body.act(say("[Arena list 7" + (" ranked" if ranked else "")))
                    append_event(life, {"event": "waiting_requested", "ranked": ranked})
                    listed_at = now
                    control_sent = True
                invite = state.get("challenge")
                if invite and invite["id"] != last_invite:
                    last_invite = invite["id"]
                    control_sent = True
                    accepted = invite["rules"] == RULES and (invite.get("ranked") is True) == ranked
                    append_event(life, {"event": "challenge", "accepted": accepted, **invite})
                    if accepted:
                        body.act(say("[DuelAccept " + invite["id"]))
                    else:
                        body.act(
                            say(
                                "I accept Standard 7x with Explosion potions. Use standard7-explosion."
                            )
                        )
            elif phase in ("Countdown", "Fighting", "RoundOver"):
                last_match = state["id"]
            if phase == "Fighting" and state.get("rules") == RULES:
                newkey = (state["id"], state["round"])
                if key != newkey or agent is None:
                    key = newkey
                    agent = Agent(
                        view,
                        Persona.load("mage_a"),
                        decision_client,
                        sync=getattr(args, "backend", "scripted") == "scripted"
                        and not getattr(args, "decision_factory", None),
                        pump_ms=250,
                        reflect_every=0,
                        log_path=log / state["id"] / f"round-{state['round']}.jsonl",
                    )
                    agent.memory.update(
                        duel=True,
                        mage=True,
                        explosion_potions=True,
                        duel_opponent=state["opponent"],
                        duel_round=state["round"],
                        playbook=policy,
                    )
                    append_event(life, {"event": "round_start", **state})
                agent.memory["showdown"] = state.get("showdown") is True
                view.pending = obs
                agent.tick()
                del agent.reports[:-2000]
                del agent.proc_log[:-2000]
            else:
                if agent:
                    agent, key = None, None
                    body.act({"type": "WarMode", "on": False})
                body.pump(250)
            # Do not batch a lobby command with the state heartbeat: some bridge/
            # shard combinations only deliver the first speech packet in a burst.
            if control_sent:
                ping_at = time.monotonic()
            elif time.monotonic() - ping_at > 1:
                body.act(say("[DuelState"))
                ping_at = time.monotonic()
    finally:
        body.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="arena.uotavern.com")
    ap.add_argument("--port", type=int, default=2593)
    ap.add_argument("--user", required=True)
    ap.add_argument("--password-env", default="ARENA_BOT_PASSWORD")
    ap.add_argument("--bridge")
    ap.add_argument("--data-dir")
    ap.add_argument("--log-dir", default=".logs/duel-wait")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--backend", choices=["scripted", "qwen", "jev", "jeff"], default="scripted")
    ap.add_argument(
        "--policy-file", type=Path, help="Reload promoted champion only between matches"
    )
    ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--max-model-calls", type=int, default=1000)
    ap.add_argument(
        "--decision-factory", help="your local module:function returning a DecisionClient"
    )
    ap.add_argument("--model-label", help="public self-reported model label")
    ap.add_argument("--version", default="baseline-v1", help="public agent version")
    ap.add_argument("--policy", choices=list(PLAYBOOKS), default="standard")
    ap.add_argument(
        "--ranked",
        action="store_true",
        help="list for and accept ranked matches instead of friendly",
    )
    ap.add_argument(
        "--no-list",
        action="store_true",
        help="accept direct invitations without a public waiting entry",
    )
    args = ap.parse_args()
    if args.max_model_calls < 1:
        ap.error("max-model-calls must be positive")
    for label in (args.version, args.model_label or args.backend, args.policy):
        if not LABEL.fullmatch(label):
            ap.error("Public labels allow 1–48 letters, digits, . _ + -")
    if not os.environ.get(args.password_env):
        ap.error("Set " + args.password_env)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    decision_client = load_policy(
        args.backend, args.decision_factory, model=args.model, max_calls=args.max_model_calls
    )
    while not stopping:
        try:
            serve(args, lambda: stopping, decision_client)
        except Exception as e:
            append_event(Path(args.log_dir) / "client.jsonl", {"event": "error", "error": str(e)})
            if args.once:
                raise
            for _ in range(20):
                if stopping:
                    break
                time.sleep(0.25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
