"""Bounded live league: frozen batches, trusted updates, independent live gate.

Uses normal accounts, preserves receipt/replay verification, stops on transport
or build failures. Credentials are read privately, never supplied in this CLI.
Evaluation pairing controls opponent/style/challenger; ServUO RNG is NOT seeded.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import threading
from pathlib import Path

from anima3 import replay
from anima3.arena_learning import atomic_json
from anima3.neural.checkpoints import load_checkpoint
from anima3.neural.curriculum import STYLES
from anima3.neural.league import promotion_gate
from anima3.neural.live import run
from anima3.neural.online import NoEligibleEpisodes, update_collections


def recover_collection(log: Path, expected_sha: str):
    """Recover one receipted duel, or archive a proven pre-challenge failure.

    Unknown/partially fought fixtures stop for investigation; they are never
    silently repeated. This preserves the total actual match budget on resume.
    """
    if not log.exists():
        return None
    report_file = log / "results.json"
    if report_file.exists():
        report = json.loads(report_file.read_text())
        if report.get("domain") != "servuo" or len(report.get("matches", [])) != 1:
            raise RuntimeError("Cannot resume ambiguous collection")
        match = report["matches"][0]
        episode = json.loads(Path(match["episodes"][0]).read_text())
        meta = json.loads((log / match["match_id"] / "replay.meta.json").read_text())
        rows = replay.validate((log / match["match_id"] / "replay.jsonl").read_bytes(), meta)
        participants = [p["serial"] for p in rows[0]["players"]]
        outcome = 0 if meta["winner"] is None else (1 if meta["winner"] == episode["actor"] else -1)
        if not (episode["policy_sha"] == expected_sha and meta["training"] is True
                and meta["aborted"] is None and episode["actor"] in participants
                and episode["opponent"] in participants
                and episode["verification"]["replay_sha256"] == meta["sha256"]
                and episode["final_result"]["outcome"] == outcome
                and meta["id"] == match["match_id"]):
            raise RuntimeError("Recovered fixture provenance differs")
        if report.get("collection_status") != "complete":
            raise RuntimeError("Recovered match requires explicit completion audit")
        return report
    status_file = log / "status.json"
    state = json.loads(status_file.read_text()) if status_file.exists() else {}
    # Compatibility with the original pilot, which failed in prepare() before
    # Challenge was sent and had no durable challenge marker yet.
    known_preparation = str(state.get("error", "")).startswith((
        "Server has not confirmed 100/25/100 stats after debuff recovery",
        "Could not verify base stats from Arena stat dialog",
        "Previously verified base stats changed"))
    if ((log / "challenge-issued.json").exists() or not known_preparation
            or any(p.is_dir() and len(p.name) == 32 for p in log.iterdir())):
        raise RuntimeError("Unfinished fixture has uncertain match status; refusing replay")
    attempt = 1
    while log.with_name(log.name + f"-preparation-attempt-{attempt}").exists():
        attempt += 1
    log.rename(log.with_name(log.name + f"-preparation-attempt-{attempt}"))
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--accounts-file", type=Path, required=True)
    p.add_argument("--bridge", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--web", default="https://arena.uotavern.com")
    p.add_argument("--user-a", default="arena_train_a")
    p.add_argument("--user-b", default="arena_train_b")
    p.add_argument("--batches", type=int, default=1)
    p.add_argument("--batch-matches", type=int, default=8)
    p.add_argument("--eval-per-style", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=20261006)
    args = p.parse_args(argv)
    if not (1 <= args.batches <= 10 and 4 <= args.batch_matches <= 32
            and 2 <= args.eval_per_style <= 40 and args.eval_per_style % 2 == 0):
        p.error("invalid bounded training/evaluation budgets")
    total_budget = args.batches*args.batch_matches + len(STYLES)*args.eval_per_style*2
    lock_id = hashlib.sha256((args.host + '/'+ '/'.join(sorted((args.user_a, args.user_b)))).encode()).hexdigest()[:16]
    lock = open('/tmp/anima3-league-'+lock_id+'.lock', 'a')  # noqa: SIM115 -- process lifetime lock
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=args.resume, mode=0o700)
    accounts = json.loads(args.accounts_file.read_text())
    os.environ['SPAR_PASSWORD_A'] = accounts[args.user_a]
    os.environ['SPAR_PASSWORD_B'] = accounts[args.user_b]
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    original = args.checkpoint.resolve()
    initial = load_checkpoint(original)
    plan = {"checkpoint_sha": initial.policy_sha, "batches": args.batches,
            "batch_matches": args.batch_matches, "eval_per_style": args.eval_per_style,
            "host": args.host, "web": args.web, "user_a": args.user_a, "user_b": args.user_b,
            "seed": args.seed, "budget": total_budget}
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Resume configuration differs from pinned experiment")
    if args.resume and not plan_path.exists():
        prior = json.loads((root / "status.json").read_text())
        if prior["budget"] != total_budget or any(
                row["policy_sha"] != initial.policy_sha for row in prior.get("matches", [])):
            raise RuntimeError("Legacy pilot configuration differs")
        atomic_json(root / "status-before-resume.json", prior)
    atomic_json(plan_path, plan)
    candidate, snapshots = original, [original]
    history, updates = [], []
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    def status(state, **extra):
        atomic_json(root/'status.json', {"status": state, "budget": total_budget,
                    "completed": len(history), "matches": history, "updates": updates,
                    "public_deployment": False, **extra})

    def collect(checkpoint, style, stage, fixture, opponent=None):
        if stop.is_set():
            raise RuntimeError('stopped before next fixture')
        if len(history) >= total_budget:
            raise RuntimeError('match budget exhausted')
        log = root/f"fixture-{len(history)+1:03d}"
        opts = argparse.Namespace(
            checkpoint=checkpoint, opponent_checkpoint=opponent,
            opponent='self' if opponent else 'scripted', opponent_style=style,
            host=args.host, port=2593, web=args.web,
            user_a=args.user_a, user_b=args.user_b,
            password_a_env='SPAR_PASSWORD_A', password_b_env='SPAR_PASSWORD_B',
            bridge=args.bridge, data_dir=args.data_dir, matches=1, deadline_ms=1000,
            log_dir=log, fixture_offset=fixture, known_build=bool(history),
            experimental_tactic=None, probe_resources=False, verify_base_stats=True)
        status('collecting', stage=stage, fixture=str(log), candidate=str(candidate))
        report = recover_collection(log, load_checkpoint(checkpoint).policy_sha) if args.resume else None
        if report is None:
            run(opts, stop)
            report = json.loads((log/'results.json').read_text())
        if len(report['matches']) != 1 or report['collection_status'] != 'complete':
            raise RuntimeError('fixture did not complete')
        match = report['matches'][0]
        # Actor 0 is always learner even when challenger order is reversed.
        episode = json.loads(Path(match['episodes'][0]).read_text())
        row = {"stage": stage, "style": style if not opponent else "historical",
               "match_id": match['match_id'], "replay": match['replay'],
               "checkpoint": str(checkpoint), "policy_sha": episode['policy_sha'],
               "eligible": episode['eligible'], "invalid_reasons": episode['invalid_reasons'],
               "score": (episode['final_result']['outcome']+1)/2,
               "side": fixture % 2, "seed": fixture//2}
        history.append(row)
        status('fixture_complete')
        if not row['template_verified']:
            raise RuntimeError('Server did not normalize starting resources; fixture counted, stop for repair')
        return log, row

    try:
        for batch in range(args.batches):
            collections=[]
            for game in range(args.batch_matches):
                # Keep diverse scripted anchors plus frozen historical opponents.
                opponent = snapshots[(batch+game) % len(snapshots)] if game % 4 == 3 else None
                log, _ = collect(candidate, STYLES[(batch+game)%len(STYLES)],
                                 f'train-{batch+1}', batch*args.batch_matches+game, opponent)
                collections.append(log)
            out = root/f'candidate-{batch+1:03d}'
            try:
                saved = out / 'batch-update.json'
                if args.resume and saved.exists():
                    update = json.loads(saved.read_text())
                    if update['update']['parent_policy_sha'] != load_checkpoint(candidate).policy_sha:
                        raise RuntimeError('Saved update parent differs')
                    if update['update']['policy_sha'] != load_checkpoint(out).policy_sha:
                        raise RuntimeError('Saved update checksum differs')
                else:
                    update = update_collections(candidate, collections, out,
                                            seed=args.seed+batch, min_episodes=4)
                candidate = out
                snapshots.append(candidate)
            except NoEligibleEpisodes as exc:
                update = {"skipped": True, "rejected": exc.rejected}
            updates.append(update)
            status('batch_complete', candidate=str(candidate))
        # Freeze both policies. Interleave order to reduce server-time confounding.
        reports = {k: {"domain": "servuo", "rows": []} for k in ('incumbent','candidate')}
        for si, style in enumerate(STYLES):
            for game in range(args.eval_per_style):
                order = ('incumbent','candidate') if game % 2 == 0 else ('candidate','incumbent')
                fixture=si*args.eval_per_style+game
                for label in order:
                    _, row = collect(original if label == 'incumbent' else candidate,
                                     style, 'evaluation-'+label, fixture)
                    reports[label]['rows'].append(row)
                    atomic_json(root/f'evaluation-{label}.json', reports[label])
        gate = promotion_gate(reports['incumbent'], reports['candidate'])
        selected = candidate if gate['passed'] else original
        atomic_json(root/'registry.json', {"selected_checkpoint": str(selected),
                    "selected_sha": load_checkpoint(selected).policy_sha,
                    "incumbent": str(original), "candidate": str(candidate), "gate": gate,
                    "real_server_strength_established": gate['passed'],
                    "pairing": "style and challenger order; server RNG uncontrolled",
                    "rollback_checkpoint": str(original), "public_deployment": False})
        status('complete', gate=gate, selected_checkpoint=str(selected))
    except BaseException as exc:
        status('stopped' if stop.is_set() else 'failed', error=str(exc), candidate=str(candidate))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
