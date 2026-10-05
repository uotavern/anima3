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
        for line in (match / f'actor-{learner}.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e['type'] == 'bridge_rpc' and e['operation'] == 'observe':
                rpc.append(e['duration_ms'])
            if e['type'] == 'spell_verdict':
                verdicts[str(e['cliloc'])] += 1
            if e['type'] == 'wire_action':
                if e['packet']['type'] == 'TargetObject':
                    last_wire_target = e['at']
                elif e['packet']['type'] == 'CastSpell' and last_wire_target is not None:
                    wire_gaps.append((e['at'] - last_wire_target) * 1000)
                    last_wire_target = None
        output.append({'mode': result['experimental_tactic'], 'match_id': result['match_id'],
                       'won': result['winner'] == learner,
                       'server_target_to_cast': {k: stats(v) for k, v in gaps.items()},
                       'client_target_to_cast': stats(wire_gaps),
                       'observe_rpc': stats(rpc), 'verdicts': dict(verdicts),
                       'metrics': result['metrics'], 'replay': result['replay']})
    return {'status': report['status'], 'matches': output,
            'note': 'One match per mode: intervals include tactics, mana and opponent interference. Server timings are accepted events, not client requests; no win-rate conclusion.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    print(json.dumps(summarize(ap.parse_args().root), indent=2))
