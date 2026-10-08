import json

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
