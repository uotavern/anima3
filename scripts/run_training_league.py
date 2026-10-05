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

from anima3.arena_learning import atomic_json
from anima3.neural.checkpoints import load_checkpoint
from anima3.neural.curriculum import STYLES
from anima3.neural.league import promotion_gate
from anima3.neural.live import run
from anima3.neural.online import NoEligibleEpisodes, update_collections


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
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    accounts = json.loads(args.accounts_file.read_text())
    os.environ['SPAR_PASSWORD_A'] = accounts[args.user_a]
    os.environ['SPAR_PASSWORD_B'] = accounts[args.user_b]
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    original = args.checkpoint.resolve()
    load_checkpoint(original)
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
            experimental_tactic=None, probe_resources=False)
        status('collecting', stage=stage, fixture=str(log), candidate=str(candidate))
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
