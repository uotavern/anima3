"""Receipt-bound real-server demonstrations, exclusively for supervised learning."""
from __future__ import annotations

import json
import math
from pathlib import Path

from anima3 import replay
from anima3.sparring import TRAINING_RULES

from .checkpoints import schema_fingerprint
from .live import starting_template_verified
from .schema import ACTION_NAMES, FEATURE_NAMES
from .train import Trajectory


def load_demonstration(path: Path, hidden_size: int) -> Trajectory:
    data = json.loads(path.read_text())
    if (data.get('schema') != 'anima3-live-demonstration-v1'
            or data.get('domain') != 'servuo' or data.get('ppo_eligible') is not False
            or data.get('eligible') is not True or data.get('invalid_reasons')
            or data.get('schema_fingerprint') != schema_fingerprint()):
        raise ValueError('invalid or incompatible demonstration')
    meta = json.loads((path.parent / 'replay.meta.json').read_text())
    rows = replay.validate((path.parent / 'replay.jsonl').read_bytes(), meta)
    ids = [p['serial'] for p in rows[0]['players']]
    if (data['match_id'] != meta['id'] or data['replay_sha256'] != meta['sha256']
            or not meta['training'] or meta['aborted'] is not None
            or meta['rules'] != TRAINING_RULES or not starting_template_verified(rows)
            or len(ids) != 2 or set(ids) != {data['actor'], data['opponent']}):
        raise ValueError('demonstration receipt differs')
    trajectory = Trajectory(info={'domain': 'servuo-demonstration', 'ppo_eligible': False,
                                 'match_id': meta['id'], 'actor': data['actor']})
    previous = None
    for step in data['steps']:
        features, mask, action, at = (step[k] for k in ('features','mask','action','at'))
        if (len(features) != len(FEATURE_NAMES) or len(mask) != len(ACTION_NAMES)
                or not all(type(x) in (float, int) and math.isfinite(x) and -1 <= x <= 1 for x in features)
                or not all(type(x) is bool for x in mask)
                or type(action) is not int or not 0 <= action < len(mask) or not mask[action]
                or step.get('accepted') is not True or not math.isfinite(at)
                or previous is not None and at <= previous):
            raise ValueError('invalid demonstration step')
        previous = at
        trajectory.features.append(features)
        trajectory.masks.append(mask)
        trajectory.actions.append(action)
        # BC reconstructs the prefix from initial zero state under current weights.
        # These placeholders must never be treated as behavior likelihoods by PPO.
        trajectory.hidden.append([0.0] * hidden_size)
        trajectory.log_probs.append(0.0)
        trajectory.values.append(0.0)
        trajectory.rewards.append(0.0)
    if not trajectory.actions:
        raise ValueError('empty demonstration')
    return trajectory
