"""Summarize confirmed probe evidence without attributing anonymous damage hits."""
import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(root):
    report = json.loads((root / 'report.json').read_text())
    rows = []
    for result in report['history']:
        match = next(root.glob('*/' + result['match_id']))
        meta = json.loads((match / 'replay.meta.json').read_text())
        learner = next(p['serial'] for p in meta['players']
                       if p.get('profile', {}).get('version', '').startswith('probe-'))
        counts = Counter()
        harm_distances = []
        for line in (match / f'actor-{learner}.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e.get('type') == 'probe_packet':
                counts[e['packet']['type']] += 1
            if e.get('type') == 'probe_journal':
                key = {1004013: 'stun_success', 1004010: 'stun_failure',
                       1019011: 'stun_armed', 1004009: 'stun_insufficient_stamina'}.get(e['cliloc'])
                if key:
                    counts[key] += 1
            if e.get('type') == 'action_finished' and e.get('name') == 'harm':
                counts['harm_options_completed'] += bool(e.get('success'))
            if (e.get('type') == 'wire_action' and e.get('action') == 8
                    and e.get('packet', {}).get('type') == 'TargetObject'):
                harm_distances.append(e.get('probe_distance'))
        positions = {}
        server_harm_distances = []
        for line in (match / 'replay.jsonl').read_text().splitlines():
            e = json.loads(line)
            updates = e.get('players', []) if e['type'] == 'frame' else (
                [e['player']] if e['type'] == 'position' else [])
            for player in updates:
                positions[player['serial']] = player['pos']
            if (e['type'] == 'spell_target' and e.get('actor') == learner
                    and e.get('name') == 'HarmSpell'):
                a, b = positions.get(learner), positions.get(e['target'])
                if a is not None and b is not None:
                    server_harm_distances.append(max(abs(a[0]-b[0]), abs(a[1]-b[1])))
        rows.append({'tactic': result['experimental_tactic'], 'match_id': result['match_id'],
                     'won': result['winner'] == learner, 'counts': dict(counts),
                     'harm_release_distances': harm_distances,
                     'server_harm_target_distances': server_harm_distances,
                     'probe_resources': result.get('probe_resources', False),
                     'metrics': result['metrics'],
                     'replay': result['replay']})
    return {'status': report['status'], 'completed': len(rows), 'matches': rows,
            'limitation': 'Attack packets are not hits. Anonymous damage cannot establish punch hit rate or causal cast interruption. No learned policy improvement claimed.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    print(json.dumps(summarize(ap.parse_args().root), indent=2))
