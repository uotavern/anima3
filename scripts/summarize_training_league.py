"""Summarize league receipts and learner execution logs without reading secrets."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from anima3.arena_learning import atomic_json


def summarize(root: Path):
    status=json.loads((root/'status.json').read_text())
    groups=defaultdict(list)
    for row in status.get('matches',[]):
        groups[row['stage']].append(row)
    actions, wires, verdicts, finishes = Counter(), Counter(), Counter(), Counter()
    latencies=[]
    rpc_latencies=defaultdict(lambda: defaultdict(list))
    stages_by_match={row["match_id"]: row["stage"] for row in status.get("matches", [])}
    eligible=0
    for file in sorted(root.glob('fixture-*/results.json')):
        report=json.loads(file.read_text())
        for match in report['matches']:
            episode=json.loads(Path(match['episodes'][0]).read_text())
            eligible+=bool(episode['eligible'])
            actor_file=file.parent/match['match_id']/f"actor-{episode['actor']}.jsonl"
            if not actor_file.exists():
                continue  # Audited transport failure can have no assigned actor log.
            for line in actor_file.read_text().splitlines():
                row=json.loads(line)
                if row.get('type')=='policy_action':
                    actions[row['action']]+=1
                    latencies.append(row['observation_age_ms'])
                elif row.get('type')=='bridge_rpc':
                    rpc_latencies[stages_by_match.get(match['match_id'], 'unknown')][row['operation']].append(row['duration_ms'])
                elif row.get('type')=='wire_action':
                    wires[row['packet']['type']]+=1
                elif row.get('type')=='melee_feedback':
                    verdicts[str(row['cliloc'])]+=1
                elif row.get('type')=='action_finished':
                    finishes[row['reason']]+=1
    latencies.sort()
    result={
        'status':status['status'], 'completed_matches':status['completed'],
        'budget':status['budget'], 'eligible_learner_episodes':eligible,
        'stages':{k:{'games':len(v), 'score':sum(r['score'] for r in v)/len(v),
                      'wins':sum(r['score']==1 for r in v),
                      'draws':sum(r['score']==.5 for r in v),
                      'losses':sum(r['score']==0 for r in v),
                      'eligible':sum(r.get('eligible',False) for r in v),
                      'eligible_score':statistics.mean([r['score'] for r in v if r.get('eligible')])
                          if any(r.get('eligible') for r in v) else None,
                      'verified_templates':sum(r.get('template_verified',False) for r in v)} for k,v in groups.items()},
        'policy_actions':dict(actions), 'wire_requests':dict(wires),
        'confirmed_stuns':verdicts['1004013'], 'failed_stun_attempts':verdicts['1004010'],
        'insufficient_stamina_messages':verdicts['1004009'],
        'option_outcomes':dict(finishes),
        'observation_age_p95_ms':latencies[int((len(latencies)-1)*.95)] if latencies else None,
        'rpc_median_ms_by_stage':{stage:{op:statistics.median(values) for op,values in ops.items()}
                                  for stage,ops in rpc_latencies.items()},
        'gate':status.get('gate'),
        'updates':len([u for u in status.get('updates',[]) if not u.get('skipped')]),
        'public_deployment':False,
        'real_server_strength_established':status.get('gate',{}).get('passed',False),
    }
    atomic_json(root/'summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.root),indent=2))
