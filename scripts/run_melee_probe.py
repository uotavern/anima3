"""Six isolated, unrated mechanics fixtures. No neural updates or automatic retries."""
import argparse
import json
import os
import signal
import threading
from pathlib import Path

from anima3.arena_learning import atomic_json
from anima3.neural.live import add_arguments, run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--accounts-file', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--probe-resources', action='store_true')
    ap.add_argument('--modes', nargs='+', choices=['magic', 'punch', 'stun', 'dump-baseline', 'dump-fast', 'dump-held'],
                    default=['magic', 'punch', 'stun', 'magic', 'punch', 'stun'])
    ap.add_argument('--bridge', required=True)
    ap.add_argument('--data-dir', required=True)
    args = ap.parse_args()
    if not 1 <= len(args.modes) <= 6:
        ap.error('choose 1..6 fixtures')
    if args.root.exists():
        ap.error('choose a new root; no automatic replay of prior fixtures')
    accounts = json.loads(args.accounts_file.read_text())
    for key, user in [('SPAR_PASSWORD_A', 'arena_train_a'), ('SPAR_PASSWORD_B', 'arena_train_b')]:
        os.environ[key] = accounts[user]
    args.root.mkdir(parents=True, mode=0o700)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    history = []
    report = {'experiment': 'melee-harm-v1', 'training_updates': 0,
              'target_matches': len(args.modes), 'history': history, 'status': 'running'}
    try:
        for n, mode in enumerate(args.modes):
            if stop.is_set():
                raise RuntimeError('experiment stopped')
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            cfg = parser.parse_args([
                '--checkpoint', 'unused-scripted-probe', '--experimental-tactic', mode,
                '--host', '127.0.0.1', '--user-a', 'arena_train_a', '--user-b', 'arena_train_b',
                '--matches', '1', '--log-dir', str(args.root / f'{n+1:02d}-{mode}'),
                '--bridge', args.bridge, '--data-dir', args.data_dir,
            ])
            cfg.probe_resources = args.probe_resources
            cfg.fixture_offset = n
            report.update(current_fixture=n+1, current_tactic=mode)
            atomic_json(args.root / 'report.json', report)
            results = run(cfg, stop)
            history.extend(results)
            atomic_json(args.root / 'report.json', report)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='stopped', error=str(exc))
        raise
    finally:
        atomic_json(args.root / 'report.json', report)


if __name__ == '__main__':
    main()
