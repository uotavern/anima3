"""Train a separate supervised candidate from verified server teacher records."""
import argparse
from pathlib import Path

import torch

from anima3.arena_learning import atomic_json
from anima3.neural.checkpoints import load_checkpoint, save_checkpoint
from anima3.neural.demonstrations import imitation_metrics, load_demonstration
from anima3.neural.train import TrainConfig, behavioral_clone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--collections', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--validation-collections', type=Path, nargs='*', default=[])
    p.add_argument('--epochs', type=int, default=32)
    p.add_argument('--balance-actions', action='store_true')
    p.add_argument('--seed', type=int, default=20261007)
    a = p.parse_args()
    if not 1 <= a.epochs <= 256 or a.out.exists():
        p.error('use bounded epochs and a fresh output directory')
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    loaded = load_checkpoint(a.checkpoint)
    files = sorted({f.resolve() for r in a.collections for f in r.rglob('demonstration-*.json')})
    episodes, seen = [], set()
    for f in files:
        e = load_demonstration(f, loaded.model.config.hidden_size)
        key = (e.info['match_id'], e.info['actor'])
        if key in seen:
            raise ValueError('duplicate match/actor demonstration')
        seen.add(key)
        episodes.append(e)
    if len(episodes) < 4:
        p.error('at least four verified demonstrations required')
    validation_files = sorted({f.resolve() for r in a.validation_collections
                               for f in r.rglob('demonstration-*.json')})
    validation = [load_demonstration(f, loaded.model.config.hidden_size) for f in validation_files]
    # Splitting by timestep or actor can leak the same fight into both sets.
    train_matches = {e.info['match_id'] for e in episodes}
    if train_matches & {e.info['match_id'] for e in validation}:
        raise ValueError('training/validation matches overlap')
    before = {'train': imitation_metrics(loaded.model, episodes),
              'validation': imitation_metrics(loaded.model, validation)}
    optimizer = torch.optim.Adam(loaded.model.parameters(), lr=1e-4)
    metrics = behavioral_clone(loaded.model, optimizer, episodes, TrainConfig(),
                               epochs=a.epochs, seed=a.seed, balance_actions=a.balance_actions)
    after = {'train': imitation_metrics(loaded.model, episodes),
             'validation': imitation_metrics(loaded.model, validation)}
    provenance = {'before': before, 'after': after,
                  'validation_files': [str(f) for f in validation_files],
                  'source': 'servuo-supervised-demonstrations', 'parent_sha': loaded.policy_sha,
                  'files': [str(f) for f in files], 'episodes': len(episodes),
                  'seed': a.seed, 'epochs': a.epochs, 'metrics': metrics,
                  'ppo_eligible': False, 'real_server_strength_established': False}
    save_checkpoint(a.out, loaded.model, training=provenance)
    atomic_json(a.out / 'demonstration-training.json', provenance)
    print(metrics)


if __name__ == '__main__':
    main()
