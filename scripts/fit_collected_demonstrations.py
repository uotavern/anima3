"""Fit a fresh demonstration collection with the last seat pair held out per style."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from anima3.neural.checkpoints import load_checkpoint
from anima3.neural.curriculum import STYLES


def split_collection(root: Path, policy_sha: str):
    dataset = json.loads((root / 'dataset.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    rows = dataset.get('matches', [])
    if (dataset.get('purpose') != 'training-demonstrations'
            or dataset.get('policy_sha') != policy_sha
            or status.get('status') != 'complete'
            or status.get('matches') != rows or status.get('completed') != len(rows)
            or status.get('budget') != len(rows) or not rows):
        raise ValueError('complete dedicated training collection required')
    ids = [r['match_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate match in collection')
    grouped = {style: [] for style in STYLES}
    for index, row in enumerate(rows):
        if (row.get('stage') != 'training-demonstrations'
                or row.get('policy_sha') != policy_sha or row.get('style') not in grouped
                or row.get('side') not in (0, 1)):
            raise ValueError('mixed evaluation, policy or style data')
        folder = root / f'fixture-{index+1:03d}'
        if not (folder / row['match_id']).is_dir():
            raise ValueError('missing receipted match directory')
        grouped[row['style']].append((row, folder))
    sizes = {len(group) for group in grouped.values()}
    if len(sizes) != 1 or min(sizes) < 4 or min(sizes) % 2:
        raise ValueError('balanced even collection of at least four games/style required')
    training, validation = [], []
    for group in grouped.values():
        if [row['side'] for row, _ in group] != [0, 1] * (len(group) // 2):
            raise ValueError('both seats required in each pair')
        training.extend(folder for _, folder in group[:-2])
        validation.extend(folder for _, folder in group[-2:])
    return training, validation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261010)
    args = p.parse_args(argv)
    if args.out.exists() or not 1 <= args.epochs <= 256:
        p.error('fresh output and bounded epochs required')
    training, validation = split_collection(
        args.dataset.resolve(), load_checkpoint(args.checkpoint).policy_sha)
    # The existing fitter verifies every replay/supply receipt before any gradient
    # and rejects overlapping matches. Validation affects reporting only.
    subprocess.run([
        sys.executable, '-m', 'scripts.clone_live_demonstrations',
        '--checkpoint', str(args.checkpoint.resolve()), '--out', str(args.out.resolve()),
        '--epochs', str(args.epochs), '--seed', str(args.seed), '--balance-actions',
        '--collections', *map(str, training),
        '--validation-collections', *map(str, validation)], check=True)
    return 0


if __name__ == '__main__':
    main()
