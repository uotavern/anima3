"""Summarize verified matches, agent resets and model responses in one sparring run."""
import argparse
import collections
import json
from itertools import pairwise
from pathlib import Path

from anima3 import replay


def review(root):
    games = []
    for path in sorted(root.glob('*/replay.meta.json')):
        meta = json.loads(path.read_text())
        rows = replay.validate(path.with_name('replay.jsonl').read_bytes(), meta)
        resets = {}
        for trace in path.parent.glob('player-*-round-*.jsonl'):
            ticks = [json.loads(line)['tick'] for line in trace.read_text().splitlines()]
            resets[trace.name] = sum(b < a for a, b in pairwise(ticks))
        games.append({'id': meta['id'], 'winner': meta['winner'],
                      'durationMs': meta['durationMs'], 'aborted': meta.get('aborted'),
                      'metrics': replay.metrics(rows), 'agentResets': resets})
    providers = {}
    for path in sorted(root.glob('brain-*/strategy.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        providers[path.parent.name] = {
            'events': dict(collections.Counter(row['event'] for row in rows)),
            'errors': sum(bool(row.get('error')) for row in rows),
            'lateResponses': sum(bool(row.get('late')) for row in rows),
        }
    return {'verifiedMatches': games, 'providers': providers,
            'automaticPromotion': False,
            'note': 'Self-play records and verified opponent experience; not a win-rate superiority test.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log_dir', type=Path)
    args = parser.parse_args()
    result = review(args.log_dir)
    target = args.log_dir / 'review.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
