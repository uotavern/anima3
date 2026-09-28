"""Summarize saved, hash-verified training receipts by policy and fixed opponent version.

This is an offline report, not permission to publish client-supplied rankings.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from .arena_learning import matches, atomic_json
from .stats import wilson


def summarize(directory: Path, minimum=40):
    if minimum < 40:
        raise ValueError('Use at least 40 held-out matches per comparison')
    groups = defaultdict(lambda: {'wins':0, 'losses':0, 'draws':0, 'matchIds':[]})
    skipped = 0
    for row in matches(directory / 'learning.jsonl'):
        if row.get('source') != 'verified-server-replay' or not row.get('valid') or row.get('aborted') or not row.get('training'):
            skipped += 1
            continue
        if row.get('stage') == 'exploration' or any(str(row.get('policy_'+s, '')).startswith('explore-') for s in ('a','b')):
            skipped += 1
            continue
        mid = row['id']
        if len(mid)!=32 or any(c not in '0123456789abcdef' for c in mid):
            raise ValueError('Invalid receipt ID')
        raw = (directory / mid / 'replay.jsonl').read_bytes()
        if hashlib.sha256(raw).hexdigest() != row.get('sha256'):
            raise ValueError('Receipt hash mismatch: ' + mid)
        records = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines()]
        header,end=records[0],records[-1]
        if (header.get('id')!=mid or header.get('training') is not True or end.get('type')!='end'
            or end.get('complete') is not True or end.get('aborted') is not None or end.get('winner')!=row.get('winner')
            or [p['serial'] for p in header['players']]!=[row['a'],row['b']]):
            raise ValueError('Receipt outcome/participants mismatch: ' + mid)
        for side,other in [('a','b'),('b','a')]:
            version,opponent=row['policy_'+side],row['policy_'+other]
            if version==opponent:
                continue
            group=groups[(version,opponent,header['rules'])]
            group['matchIds'].append(mid)
            group['draws' if row['winner'] is None else 'wins' if row['winner']==row[side] else 'losses']+=1
    result=[]
    for (version,opponent,rules),counts in sorted(groups.items()):
        n=len(counts['matchIds'])
        result.append(dict(version=version,opponent=opponent,rules=rules,**counts,matches=n,
            winRate=counts['wins']/n,winRate95=list(wilson(counts['wins'],n)),
            evidence='minimum reached' if n>=minimum else 'insufficient sample'))
    return {'schema':1,'minimum':minimum,'skipped':skipped,'comparisons':result,
        'notes':['Draws count as non-wins in the win-rate interval.','Policy names are local experiment labels.',
                 'Only compare frozen versions on the same opponents, rules, arena schedule and latency conditions.',
                 'This report does not promote a policy; use the predeclared fixed-sample evaluation for promotion.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum',type=int,default=40)
    args=p.parse_args()
    atomic_json(args.output,summarize(args.directory,args.minimum))


if __name__=='__main__':
    main()
