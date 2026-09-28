import hashlib
import json
import threading
from pathlib import Path
import pytest
from anima3.arena_policy import BoundedClient, load_policy
from anima3.benchmark import summarize
from anima3.decision import Decision, gate


def test_stuck_provider_has_only_one_outstanding_call():
    entered,release=threading.Event(),threading.Event()
    class Slow:
        name='slow'
        def choose(self,*_):
            entered.set();release.wait(2)
            return Decision('a',{'a':1},1,0,self.name)
    client=BoundedClient(Slow())
    thread=threading.Thread(target=lambda:client.choose('','',{'a':'a'}));thread.start();assert entered.wait(1)
    assert client.choose('','',{'a':'a'}).error=='previous provider call still running'
    release.set();thread.join(1)
    assert client.choose('','',{'a':'a'}).choice=='a'


def test_custom_policy_and_nonfinite_confidence():
    client=load_policy(factory='examples.arena_policy:create')
    assert client.choose('scene','question',{'heal':'Heal'}).choice=='heal'
    assert not gate(Decision('b',{},float('nan'),1,'custom'),{'a':'safe','b':'other'},.35).used_model
    with pytest.raises(ValueError):load_policy(factory='../arbitrary.py')


def make_receipt(root,mid='a'*32):
    folder=root/mid;folder.mkdir()
    header=dict(id=mid,training=True,players=[{'serial':1},{'serial':2}],rules='7x-training')
    end=dict(type='end',complete=True,aborted=None,winner=1)
    raw=(json.dumps(header)+'\n'+json.dumps(end)+'\n').encode();(folder/'replay.jsonl').write_bytes(raw)
    row=dict(schema=1,event='match_end',id=mid,source='verified-server-replay',valid=True,training=True,aborted=False,a=1,b=2,winner=1,policy_a='candidate-v2',policy_b='baseline-v1',sha256=hashlib.sha256(raw).hexdigest(),stage='evaluation')
    (root/'learning.jsonl').write_text(json.dumps(row)+'\n')
    return row


def test_benchmark_requires_receipt_and_counts_match_once(tmp_path):
    row=make_receipt(tmp_path)
    with (tmp_path/'learning.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    report=summarize(tmp_path);assert len(report['comparisons'])==2
    assert all(x['matches']==1 and x['evidence']=='insufficient sample' for x in report['comparisons'])
    assert sorted(x['wins'] for x in report['comparisons'])==[0,1]
    (tmp_path/row['id']/'replay.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='hash'):summarize(tmp_path)


def test_benchmark_excludes_exploration(tmp_path):
    row=make_receipt(tmp_path);row['stage']='exploration'
    (tmp_path/'learning.jsonl').write_text(json.dumps(row)+'\n')
    assert summarize(tmp_path)['comparisons']==[]
