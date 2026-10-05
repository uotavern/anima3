"""Server-accepted cast intervals, separate from client request timings."""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def stats(values):
    return {'n': len(values), 'min_ms': round(min(values), 2) if values else None,
            'median_ms': round(statistics.median(values), 2) if values else None,
            'max_ms': round(max(values), 2) if values else None}


def summarize(root):
    report = json.loads((root / 'report.json').read_text())
    output = []
    for result in report['history']:
        match = next(root.glob('*/' + result['match_id']))
        meta = json.loads((match / 'replay.meta.json').read_text())
        learner = next(p['serial'] for p in meta['players']
                       if p.get('profile', {}).get('version', '').startswith('probe-'))
        last_target = None
        gaps = defaultdict(list)
        for line in (match / 'replay.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e.get('actor') != learner:
                continue
            if e['type'] == 'spell_target':
                last_target = e
            elif e['type'] == 'cast' and last_target is not None:
                key = last_target['name'] + ' -> ' + e['name']
                gaps[key].append(e['t'] - last_target['t'])
                last_target = None
        rpc, verdicts, last_wire_target, wire_gaps = [], Counter(), None, []
        wire_pairs = defaultdict(list)
        action_name = None
        for line in (match / f'actor-{learner}.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e['type'] == 'bridge_rpc' and e['operation'] == 'observe':
                rpc.append(e['duration_ms'])
            if e['type'] == 'spell_verdict':
                verdicts[str(e['cliloc'])] += 1
            if e['type'] == 'action_started':
                action_name = e['name']
            if e['type'] == 'wire_action':
                if e['packet']['type'] == 'TargetObject':
                    last_wire_target = (e['at'], action_name)
                elif e['packet']['type'] == 'CastSpell' and last_wire_target is not None:
                    gap = (e['at'] - last_wire_target[0]) * 1000
                    wire_gaps.append(gap)
                    wire_pairs[str(last_wire_target[1]) + ' -> ' + str(action_name)].append(gap)
                    last_wire_target = None
        hits = []
        for line in (match / 'replay.jsonl').read_text().splitlines():
            e = json.loads(line)
            if (e['type'] == 'damage' and e.get('actor') == learner
                    and e.get('target') != learner):
                hits.append(e)
        burst = {'damage': 0, 'span_ms': 0, 'hits': []}
        for i, hit in enumerate(hits):
            group = [h for h in hits[i:] if h['t'] - hit['t'] <= 100]
            damage = sum(h['amount'] for h in group)
            if damage > burst['damage']:
                burst = {'damage': damage, 'span_ms': group[-1]['t'] - hit['t'],
                         'hits': [{'t': h['t'], 'amount': h['amount']} for h in group]}
        output.append({'mode': result['experimental_tactic'], 'match_id': result['match_id'],
                       'won': result['winner'] == learner,
                       'server_target_to_cast': {k: stats(v) for k, v in gaps.items()},
                       'client_target_to_cast': stats(wire_gaps),
                       'client_target_to_cast_by_pair': {k: stats(v) for k, v in wire_pairs.items()},
                       'max_damage_in_100ms': burst,
                       'observe_rpc': stats(rpc), 'verdicts': dict(verdicts),
                       'metrics': result['metrics'], 'replay': result['replay']})
    return {'status': report['status'], 'matches': output,
            'note': 'Explosion target events are absent in current server replays; client request gaps must not be called server target gaps. One match per mode: intervals include tactics, mana and opponent interference. Server timings are accepted events, not client requests; no win-rate conclusion.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    print(json.dumps(summarize(ap.parse_args().root), indent=2))
