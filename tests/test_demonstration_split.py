import json
from types import SimpleNamespace

import pytest

from anima3.neural.curriculum import STYLES
from scripts.fit_collected_demonstrations import split_collection


def dataset(root):
    rows = []
    for pair in range(3):
        for style in STYLES:
            for side in range(2):
                i = len(rows) + 1
                row = {'match_id': f'{i:032x}', 'policy_sha': 'parent', 'style': style,
                       'side': side, 'stage': 'training-demonstrations'}
                (root / f'fixture-{i:03d}' / row['match_id']).mkdir(parents=True)
                rows.append(row)
    return rows


def write(root, rows, **overrides):
    (root/'dataset.json').write_text(json.dumps({
        'purpose': 'training-demonstrations', 'policy_sha': 'parent', 'matches': rows}))
    (root/'status.json').write_text(json.dumps({
        'status': 'complete', 'matches': rows, 'completed': len(rows), 'budget': len(rows),
        **overrides}))


def test_last_pair_per_style_is_disjoint_validation(tmp_path):
    rows = dataset(tmp_path)
    write(tmp_path, rows)
    train, validation = split_collection(tmp_path, 'parent')
    assert len(train) == 20 and len(validation) == 10
    assert set(train).isdisjoint(validation)
    assert set(train) == {tmp_path/f'fixture-{i:03d}' for i in range(1,21)}
    assert set(validation) == {tmp_path/f'fixture-{i:03d}' for i in range(21,31)}


@pytest.mark.parametrize('bad', ['incomplete', 'evaluation', 'duplicate', 'seat', 'policy'])
def test_split_rejects_contamination_and_incomplete_data(tmp_path, bad):
    rows = dataset(tmp_path)
    if bad == 'evaluation':
        rows[0]['stage'] = 'evaluation-candidate'
    elif bad == 'duplicate':
        rows[1]['match_id'] = rows[0]['match_id']
    elif bad == 'seat':
        rows[0]['side'] = 1
    elif bad == 'policy':
        rows[0]['policy_sha'] = 'other'
    write(tmp_path, rows, **({'status': 'collecting'} if bad == 'incomplete' else {}))
    with pytest.raises(ValueError):
        split_collection(tmp_path, 'parent')


def test_fit_checks_collection_policy_separately_from_start_and_reference(tmp_path, monkeypatch):
    from scripts import fit_collected_demonstrations as fit

    rows = dataset(tmp_path)
    write(tmp_path, rows)
    loaded, commands = [], []

    def load(path):
        loaded.append(path)
        return SimpleNamespace(policy_sha='parent' if path.name == 'collector' else 'different')

    monkeypatch.setattr(fit, 'load_checkpoint', load)
    monkeypatch.setattr(fit.subprocess, 'run', lambda command, **kwargs: commands.append(command))
    assert fit.main(['--dataset', str(tmp_path), '--checkpoint', str(tmp_path/'start'),
                     '--collection-checkpoint', str(tmp_path/'collector'),
                     '--reference-checkpoint', str(tmp_path/'reference'),
                     '--reference-kl-coef', '0.5', '--out', str(tmp_path/'output')]) == 0
    assert loaded == [tmp_path/'collector']
    command = commands[0]
    assert command[command.index('--checkpoint')+1] == str(tmp_path/'start')
    assert command[command.index('--reference-checkpoint')+1] == str(tmp_path/'reference')
    assert command[command.index('--reference-kl-coef')+1] == '0.5'
