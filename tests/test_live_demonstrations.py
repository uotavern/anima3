import hashlib
import json
import math

import pytest

from anima3.neural.checkpoints import schema_fingerprint
from anima3.neural.demonstrations import load_demonstration
from anima3.neural.schema import ACTION_NAMES, FEATURE_NAMES
from anima3.neural.train import TrainConfig, ppo_update
from anima3.sparring import TRAINING_RULES


def make_demo(tmp_path):
    from anima3.duel_wait import TRAINING_SUPPLIES

    tmp_path = tmp_path / 'fixture' / ('a' * 32)
    tmp_path.mkdir(parents=True)
    supplies = {'players': [{'serial': serial, 'verified': True, 'weight': 200,
                            'weight_max': 390,
                            'counts': {hex(k):v for k,v in TRAINING_SUPPLIES.items()}}
                           for serial in (57,58)]}
    (tmp_path.parent / 'prepared-supplies.json').write_text(json.dumps(supplies))
    mid = 'a' * 32
    rows = [
        {'seq': 0, 't': 0, 'type': 'header', 'schema': 1, 'id': mid,
         'rules': TRAINING_RULES,
         'players': [{'serial': s, 'stats': [100,25,100]} for s in (57,58)]},
        {'seq': 1, 't': 0, 'type': 'frame', 'phase': 'Countdown',
         'players': [{'hitsMax': 100, 'manaMax': 100, 'stamMax': 25,
                      'alive': True, 'poisoned': False} for _ in range(2)]},
        {'seq': 2, 't': 100, 'type': 'end', 'id': mid, 'complete': True,
         'training': True, 'aborted': None, 'winner': 58, 'score': [0,1]},
    ]
    raw = '\n'.join(json.dumps(r) for r in rows).encode()
    meta = {**rows[-1], 'sha256': hashlib.sha256(raw).hexdigest(), 'rules': TRAINING_RULES}
    (tmp_path / 'replay.jsonl').write_bytes(raw)
    (tmp_path / 'replay.meta.json').write_text(json.dumps(meta))
    d = {'schema': 'anima3-live-demonstration-v1', 'domain': 'servuo',
         'ppo_eligible': False, 'eligible': True, 'invalid_reasons': [],
         'schema_fingerprint': schema_fingerprint(), 'actor': 58, 'opponent': 57,
         'match_id': mid, 'replay_sha256': meta['sha256'],
         'steps': [{'features': [0.] * len(FEATURE_NAMES),
                    'mask': [True] + [False] * (len(ACTION_NAMES)-1),
                    'action': 0, 'at': 1., 'accepted': True}]}
    p = tmp_path / 'demonstration-58.json'
    p.write_text(json.dumps(d))
    return p, d


def test_verified_demo_is_bc_only(tmp_path):
    p, _ = make_demo(tmp_path)
    e = load_demonstration(p, 64)
    assert len(e) == 1 and len(e.hidden[0]) == 64
    with pytest.raises(ValueError, match='not PPO'):
        ppo_update(None, None, [e], TrainConfig())


@pytest.mark.parametrize('damage', ['illegal_action', 'receipt', 'schema', 'rejected'])
def test_bad_demo_fails_closed(tmp_path, damage):
    p, d = make_demo(tmp_path)
    if damage == 'illegal_action':
        d['steps'][0]['action'] = 1
    elif damage == 'receipt':
        d['replay_sha256'] = 'bad'
    elif damage == 'schema':
        d['schema_fingerprint'] = 'bad'
    else:
        d['steps'][0]['accepted'] = False
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_demonstration(p, 64)


def test_imitation_metrics_excludes_forced_holds(tmp_path):
    import torch

    from anima3.neural.demonstrations import imitation_metrics
    from anima3.neural.model import PolicyConfig, RecurrentPolicy

    p, d = make_demo(tmp_path)
    free = {**d['steps'][0], 'mask': [True, True] + [False]*(len(ACTION_NAMES)-2),
            'action': 1, 'at': 2.0}
    d['steps'].append(free)
    p.write_text(json.dumps(d))
    model = RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), 8, 8))
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.bias.zero_()
        model.policy_head.bias[0] = 5
    metrics = imitation_metrics(model, [load_demonstration(p, 8)])
    assert metrics['free_choices'] == 1
    assert metrics['accuracy'] == 0
    assert metrics['cross_entropy'] > 5
    assert 'hold' not in metrics['per_action']
    distribution = metrics['action_distribution']
    assert distribution['hold']['label_share'] == 0
    assert distribution['hold']['argmax_share'] == 1
    assert distribution['hold']['mean_probability'] == pytest.approx(1/(1+math.exp(-5)))
    assert distribution[ACTION_NAMES[1]]['label_share'] == 1
    assert distribution[ACTION_NAMES[2]]['mean_probability'] == 0
    assert distribution[ACTION_NAMES[2]]['legal_share'] == 0
    assert sum(x['mean_probability'] for x in distribution.values()) == pytest.approx(1)
    assert metrics['unobserved_label_probability'] == pytest.approx(distribution['hold']['mean_probability'])
    assert imitation_metrics(model, [])['accuracy'] is None
    assert imitation_metrics(model, [])['unobserved_label_probability'] is None


def test_balanced_imitation_does_not_count_forced_holds(tmp_path):
    import torch

    from anima3.neural.model import PolicyConfig, RecurrentPolicy
    from anima3.neural.train import behavioral_clone

    p, d = make_demo(tmp_path)
    template = d['steps'][0]
    d['steps'] = [{**template, 'at': float(i)} for i in range(20)]
    for i, action in enumerate((0, 0, 1), 20):
        d['steps'].append({**template, 'at': float(i), 'action': action,
                          'mask': [True,True]+[False]*(len(ACTION_NAMES)-2)})
    p.write_text(json.dumps(d))
    episode = load_demonstration(p, 8)
    model = RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), 8, 8))
    result = behavioral_clone(model, torch.optim.Adam(model.parameters()), [episode],
                              TrainConfig(), epochs=1, balance_actions=True)
    weights = result['action_weights']
    assert weights[1] / weights[0] == pytest.approx(2**.5)
    assert result['bc_updates'] > 0


@pytest.mark.parametrize('fault', ['missing', 'potions', 'weight', 'participant'])
def test_previously_confounded_demonstrations_are_rejected(tmp_path, fault):
    p, _ = make_demo(tmp_path)
    file = p.parent.parent / 'prepared-supplies.json'
    if fault == 'missing':
        file.unlink()
    else:
        audit = json.loads(file.read_text())
        if fault == 'potions':
            audit['players'][0]['counts']['0xf0d'] = 0
        elif fault == 'weight':
            audit['players'][0]['weight'] = 415
        else:
            audit['players'][0]['serial'] = 99
        file.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='supply verification'):
        load_demonstration(p, 64)
