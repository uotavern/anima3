"""Bounded ten-game Jev/LLM experiment against one fixed scripted opponent."""

from __future__ import annotations

import argparse
import collections
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import replay
from .arena_learning import atomic_json, matches
from .learning_run import verify_receipts
from .strategy import DEFAULT_MODEL


def events(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def report(root):
    records = matches(root / "learning.jsonl") if (root / "learning.jsonl").exists() else []
    verify_receipts(root, records)
    logs = events(root / "brain-0/strategy.jsonl")
    games, prior, previous_plan = [], set(), None
    for n, record in enumerate(records, 1):
        mid = record["id"]
        meta = json.loads((root / mid / "replay.meta.json").read_text())
        rows = replay.validate((root / mid / "replay.jsonl").read_bytes(), meta)
        learner = record.get("learner")
        if record.get("opponent") != "fixed-scripted" or learner not in [record["a"], record["b"]]:
            raise ValueError("Not a fixed-opponent experiment")
        scoped = [r for r in logs if (r.get("context") or [None])[0] == mid]
        requests = [r for r in scoped if r["event"] == "plan_requested"]
        history = {h["id"] for r in requests for h in r.get("verifiedHistory", [])}
        if not history <= prior:
            raise ValueError("Strategy history contains a future or unrelated result")
        plans = [r["plan"] for r in scoped if r["event"] == "plan_applied"]
        signature = (
            json.dumps({k: plans[0][k] for k in ("primary", "responses")}, sort_keys=True)
            if plans
            else None
        )
        changed = bool(signature and previous_plan and signature != previous_plan)
        if signature:
            previous_plan = signature
        metrics = replay.metrics(rows)
        opponent = next(i for i in (record["a"], record["b"]) if i != learner)
        games.append(
            {
                "number": n,
                "id": mid,
                "result": "draw"
                if meta["winner"] is None
                else "win"
                if meta["winner"] == learner
                else "loss",
                "arena": meta.get("arena"),
                "learnerSide": "a" if record["a"] == learner else "b",
                "durationMs": meta["durationMs"],
                "verifiedHistory": sorted(history),
                "plans": plans,
                "opening": record.get("opening", "none"),
                "learner": learner,
                "burstCombo": record.get("burstCombo", False),
                "burstMetrics": replay.burst_metrics(rows),
                "openingCasts": {
                    str(serial): [
                        r.get("name")
                        for r in rows
                        if r["type"] == "cast" and r.get("actor") == serial
                    ][:4]
                    for serial in (learner, opponent)
                },
                "strategyChanged": changed,
                "tactics": dict(
                    collections.Counter(
                        r["playbook"] for r in scoped if r["event"] == "tactic_applied"
                    )
                ),
                "self": metrics[str(learner)],
                "opponent": metrics[str(opponent)],
            }
        )
        prior.add(mid)

    def summary(group):
        return {
            "games": len(group),
            "wins": sum(g["result"] == "win" for g in group),
            "draws": sum(g["result"] == "draw" for g in group),
            "meanDamageDifference": round(
                sum(g["self"]["damage"] - g["opponent"]["damage"] for g in group) / len(group), 2
            )
            if group
            else None,
        }

    result = {
        "target": 10,
        "completed": len(games),
        "games": games,
        "experienceUsedMatches": sum(bool(g["verifiedHistory"]) for g in games),
        "strategyChangedMatches": sum(g["strategyChanged"] for g in games),
        "firstFive": summary(games[:5]),
        "lastFive": summary(games[5:10]),
        "performanceImprovement": "not-established",
        "automaticPromotion": False,
        "limitation": "Ten adaptive games are a feasibility pilot, not a held-out causal win-rate evaluation. Arenas can differ.",
    }
    atomic_json(root / "ten-duels-report.json", result)
    lines = [
        "# 10경기 전략 적응 실험",
        "",
        f"완료: {len(games)} / 10",
        f"이전 경험 사용: {result['experienceUsedMatches']}경기 · 첫 계획 변경: {result['strategyChangedMatches']}경기",
        "",
        "| 경기 | 결과 | 이전 경험 | 첫 전략 | 피해 / 자해 / 상대 피해 | 리플레이 |",
        "|---|---|---|---|---|---|",
    ]
    for g in games:
        primary = g["plans"][0]["primary"] if g["plans"] else "기본 규칙"
        lines.append(
            f"| {g['number']} | {g['result']} | {len(g['verifiedHistory'])} | {primary} | {g['self']['damage']} / {g['self']['self_damage']} / {g['opponent']['damage']} | [보기](https://arena.uotavern.com/replay/?replay={g['id']}) |"
        )
    lines += ["", "## 오프닝과 순간 피해", ""]
    for g in games:
        burst = g["burstMetrics"][str(g["learner"])]
        lines.append(f"- 경기 {g['number']}: {' → '.join(burst['firstSpells'])} · 1초 최대 피해 {burst['peakOneSecondDamage']}")
    lines += ["", "## 계획의 이유", ""]
    for g in games:
        lines.append(
            f"- 경기 {g['number']}: "
            + (" / ".join(p["reason"] for p in g["plans"]) or "적용된 LLM 계획 없음")
        )
    lines += [
        "",
        "10경기는 작동 가능성을 확인하는 예비 실험입니다. 전략 변화는 성능 향상의 증명이 아니며, 모델 가중치 학습이나 자동 승격은 수행하지 않습니다.",
        "앞 5경기와 뒤 5경기의 결과는 참고용이며, 경기장 차이·무작위 전투 결과·진행 중 전략 변경의 영향을 포함합니다.",
        "",
    ]
    (root / "ten-duels-report.md").write_text("\n".join(lines))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--burst-combo", action="store_true")
    ap.add_argument("--user-a", required=True)
    ap.add_argument("--user-b", required=True)
    ap.add_argument("--opening", choices=["none", "weaken-clumsy"], default="weaken-clumsy")
    ap.add_argument("--host", default="arena.uotavern.com")
    ap.add_argument("--web", default="https://arena.uotavern.com")
    ap.add_argument("--llm-model", default=DEFAULT_MODEL)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--max-starts", type=int, default=6)
    ap.add_argument("--max-minutes", type=int, default=60)
    ap.add_argument("--jev-budget", type=int, default=600)
    ap.add_argument("--llm-budget", type=int, default=100)
    args = ap.parse_args(argv)
    if (
        not 1 <= args.max_starts <= 10
        or not 1 <= args.max_minutes <= 120
        or min(args.jev_budget, args.llm_budget) < 1
    ):
        ap.error("invalid experiment limits")
    if args.user_a == args.user_b or not all(
        os.environ.get(k) for k in ("SPAR_PASSWORD_A", "SPAR_PASSWORD_B")
    ):
        ap.error("two dedicated accounts and both password environment variables are required")
    root = args.log_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (root / "experiment.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "ten-duels-state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"starts": 0, "started": time.time()}
    )
    deadline = state["started"] + args.max_minutes * 60
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = None
    try:
        while not stopping and time.time() < deadline and state["starts"] < args.max_starts:
            result = report(root)
            if result["completed"] >= 10:
                break
            logs = events(root / "brain-0/strategy.jsonl")
            jev = args.jev_budget - sum(r["event"] == "tactic_requested" for r in logs)
            llm = args.llm_budget - sum(r["event"] == "plan_requested" for r in logs)
            if min(jev, llm) <= 0:
                state["reason"] = "model_budget_exhausted"
                break
            state.update(stage="running", starts=state["starts"] + 1, completed=result["completed"])
            atomic_json(state_path, state)
            command = [
                sys.executable,
                "-m",
                "anima3.sparring",
                "--host",
                args.host,
                "--web",
                args.web,
                "--user-a",
                args.user_a,
                "--user-b",
                args.user_b,
                "--backend",
                "jev",
                "--brain",
                "hybrid",
                "--opponent",
                "fixed-scripted",
                "--opening",
                args.opening,
                "--matches",
                "10",
                "--max-model-calls",
                str(jev),
                "--max-strategy-calls",
                str(llm),
                "--llm-model",
                args.llm_model,
                "--log-dir",
                str(root),
            ]
            if args.burst_combo:
                command.append("--burst-combo")
            with (root / "console.log").open("a") as output:
                child = subprocess.Popen(command, stdout=output, stderr=output)
            while child.poll() is None and not stopping and time.time() < deadline:
                time.sleep(1)
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            state["lastExit"] = child.returncode
            result = report(root)
            if result["completed"] >= 10:
                break
            # Preserve experience, and let the server finish any abandoned match.
            for _ in range(15):
                if stopping or time.time() >= deadline:
                    break
                time.sleep(1)
        result = report(root)
        state.update(
            stage="complete" if result["completed"] == 10 else "stopped",
            completed=result["completed"],
            finished=time.time(),
        )
        atomic_json(state_path, state)
        print(
            json.dumps(
                {
                    "stage": state["stage"],
                    "completed": result["completed"],
                    "report": str(root / "ten-duels-report.md"),
                }
            )
        )
        return 0 if state["stage"] == "complete" else 1
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
