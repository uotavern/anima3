"""Two unrated tactical fixtures against the frozen incumbent; no weight updates."""
import argparse
import json
import os
import signal
import threading
from pathlib import Path

from anima3.neural.live import add_arguments, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounts-file', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bridge', required=True)
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()
    accounts = json.loads(args.accounts_file.read_text())
    for env, user in [('SPAR_PASSWORD_A', 'arena_train_a'), ('SPAR_PASSWORD_B', 'arena_train_b')]:
        os.environ[env] = accounts[user]
    live = argparse.ArgumentParser()
    add_arguments(live)
    cfg = live.parse_args([
        '--checkpoint', str(args.checkpoint), '--opponent-style', 'tactical',
        '--host', '127.0.0.1', '--user-a', 'arena_train_a', '--user-b', 'arena_train_b',
        '--matches', '2', '--log-dir', str(args.root),
        '--bridge', args.bridge, '--data-dir', args.data_dir,
    ])
    cfg.verify_base_stats = True
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    run(cfg, stop)


if __name__ == '__main__':
    main()
