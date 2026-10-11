"""Audit public actions and server potion outcomes for a bounded live pilot."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from anima3.neural.schema import ACTION_NAMES


def audit(root):
    result = json.loads((root / 'results.json').read_text())
    report = {'status': result['collection_status'], 'training_updates': 0, 'matches': []}
    for match in result['matches']:
        path = root / match['match_id']
        raw = (path / 'replay.jsonl').read_bytes()
        meta = json.loads((path / 'replay.meta.json').read_text())
        if hashlib.sha256(raw).hexdigest() != meta['sha256']:
            raise ValueError('replay hash mismatch')
        rows = [json.loads(s) for s in raw.splitlines()]
        actor = [json.loads(s) for s in (path / 'actor-58.jsonl').read_text().splitlines()]
        demo = json.loads((path / 'demonstration-58.json').read_text())
        recovering = False
        recovery_actions, violations = Counter(), []
        allowed = {'hold', 'retreat', 'meditate', 'heal', 'greater_heal', 'cure',
                   'heal_potion', 'cure_potion', 'agility_potion', 'refresh_potion'}
        for step in demo['steps']:
            mana = step['features'][1] * 100  # verified 100-max-mana training template
            if mana < 23:
                recovering = True
            elif mana >= 60:
                recovering = False
            if recovering:
                name = ACTION_NAMES[step['action']]
                recovery_actions[name] += 1
                if name not in allowed:
                    violations.append({'at': step['at'], 'mana': mana, 'action': name})
        explosions = []
        frame = None
        for i, row in enumerate(rows):
            if row['type'] == 'frame':
                frame = row
            if row['type'] != 'potion_explode' or row['actor'] != 57 or frame is None:
                continue
            player = next(p for p in frame['players'] if p['serial'] == 58)
            distance = max(abs(player['pos'][j] - row['to'][j]) for j in (0, 1))
            damage = [r for r in rows[i+1:] if r['t'] <= row['t'] + 50
                      and r['type'] == 'damage' and r['actor'] == 57 and r['target'] == 58]
            explosions.append({'at_ms': row['t'], 'landing': row['to'],
                               'last_frame_age_ms': row['t'] - frame['t'],
                               'last_frame_distance': distance,
                               'damage_in_next_50ms': sum(r['amount'] for r in damage)})
        report['matches'].append({'id': match['match_id'], 'winner': match['winner'],
            'replay_sha256': meta['sha256'], 'explosions': explosions,
            'defense_events': dict(Counter(r['type'] for r in actor if r['type'] in
                ('potion_evasion', 'tactical_rethrow_started', 'potion_escape_blocked'))),
            'recovery_actions': dict(recovery_actions), 'recovery_violations': violations,
            'demonstration_eligible': demo['eligible'], 'invalid_reasons': demo['invalid_reasons']})
    report['limits'] = ('Two-game mechanics pilot, no strength estimate. Frame positions are sampled; '
                        'damage within 50ms is a temporal association, not explicit spell attribution. '
                        'Recovery audit uses the verified 100 maximum mana template.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root), indent=2) + '\n')
