from dataclasses import replace

import pytest
from test_neural_runtime import executor, observation

from anima3.contract import Journal, Skill
from anima3.neural.curriculum import DRILLS, STYLES, TacticalTeacher, setup_drill
from anima3.neural.league import promotion_gate
from anima3.neural.schema import ACTION_INDEX, CombatView, encode
from anima3.neural.sim import DuelSim


def test_default_evaluation_covers_full_server_showdown_window(monkeypatch):
    import torch

    from anima3.neural.league import evaluate_suite

    class HoldPolicy:
        device = 'cpu'

        def eval(self):
            return self

        def initial_state(self):
            return None

        def act(self, *args, **kwargs):
            return torch.tensor(0), None, None, None

    monkeypatch.setattr(TacticalTeacher, 'choose', lambda *args: 0)
    report = evaluate_suite(HoldPolicy(), games_per_style=2)
    assert report['round_limit_seconds'] == 300
    assert report['showdown_after_seconds'] == 180
    assert report['draws'] == 10
    assert all(r['elapsed'] == 300 and r['terminated'] for r in report['rows'])


def test_stun_needs_resources_skills_and_ack_before_next_toggle():
    obs = observation(enemy_x=101)
    obs.skills = [Skill(1,100,100,100,2), Skill(43,100,100,100,2)]
    ex = executor(obs)
    assert ex.start('stun', obs, 0)
    assert ex.tick(obs, 0) == [{'type':'StunRequest'}, {'type':'Attack','serial':2}]
    ex.tick(obs, 0.3)
    obs2 = replace(obs, new_journal=[Journal(1,'','',0,0,1019011)])
    ex.observe(obs2, {'phase':'Fighting','opponent':2}, 2.1)
    assert ex.view.stun_armed and not ex.view.stun_pending
    assert not encode(ex.view).mask[ACTION_INDEX['stun']]
    obs3 = replace(obs, new_journal=[Journal(1,'','',0,0,1004010)])
    ex.observe(obs3, {'phase':'Fighting','opponent':2}, 3)
    assert ex.view.stun_armed and ex.view.stun_cooldown == 10
    obs4 = replace(obs, new_journal=[Journal(1,'','',0,0,1004013)])
    ex.observe(obs4, {'phase':'Fighting','opponent':2}, 13)
    assert not ex.view.stun_armed
    assert not encode(ex.view).mask[ACTION_INDEX['stun']]
    frame = ex.observe(obs, {'phase':'Fighting','opponent':2}, 23.1)
    assert frame.mask[ACTION_INDEX['stun']]


def test_unknown_stun_skills_and_stale_state_fail_closed():
    v = CombatView(hands_free=True, opponent_visible=True, line_of_sight=True,
                   distance=1, stamina=14, stun_skilled=True)
    assert not encode(v).mask[ACTION_INDEX['stun']]
    assert not encode(replace(v, stamina=25, stun_skilled=False)).mask[ACTION_INDEX['stun']]
    assert encode(replace(v, stamina=25)).mask[ACTION_INDEX['stun']]
    assert sum(encode(replace(v, fresh=False)).mask) == 1


def test_sim_stun_attempt_costs_and_freeze_survives_damage():
    sim = DuelSim(seed=4, domain_randomization=False)
    sim.fighters[0].x, sim.fighters[1].x = 0,1
    sim._perform(0,'stun')
    sim.step((0,0))
    event = next(r for r in sim.trace if r['type']=='stun_attempt')
    assert event['success']
    assert 10 <= sim.fighters[0].stamina < 10.1
    assert sim.fighters[0].stun_until == pytest.approx(10.05)
    assert sim.view(1).paralyzed
    sim._hurt(0,1,1,'harm')
    assert sim.view(1).paralyzed  # Freeze differs from Paralyze spell


def test_first_circle_damage_does_not_disturb_but_second_does():
    sim = DuelSim(domain_randomization=False)
    sim._start_cast(0,'magic_arrow')
    sim._hurt(1,0,5,'punch')
    assert sim.fighters[0].interrupted == 0
    sim._start_cast(0,'harm')
    sim._hurt(1,0,5,'punch')
    assert sim.fighters[0].interrupted == 1


@pytest.mark.parametrize('drill', DRILLS)
def test_drills_reseed_and_teachers_only_choose_legal_actions(drill):
    sim = DuelSim(seed=17, max_steps=100)
    frames = setup_drill(sim, drill, 0)
    assert sim._potential_previous == sim._potential()
    teachers = [TacticalTeacher('pressure'), TacticalTeacher('defensive')]
    for _ in range(100):
        actions = [teachers[i].choose(sim.view(i)) for i in (0,1)]
        assert all(frames[i].mask[actions[i]] for i in (0,1))
        frames, _, done, info = sim.step(actions)
        assert not any(info['invalid_actions'])
        if done:
            break


def report(score, count=20):
    return {'domain':'servuo','rows':[{'style':style,'seed':i//2,'side':i%2,'score':score}
            for style in STYLES for i in range(count)]}


def test_gate_rejects_small_sample_and_regression_and_mismatched_pairs():
    assert not promotion_gate(report(0,2),report(1,2))['passed']
    assert not promotion_gate(report(.5),report(.5))['passed']
    assert promotion_gate(report(0),report(1))['passed']
    better = report(1)
    for row in better['rows'][:20]:
        row['score'] = 0
    assert not promotion_gate(report(.5),better)['passed']
    bad = report(1)
    bad['rows'].pop()
    with pytest.raises(ValueError,match='differ'):
        promotion_gate(report(0),bad)


def test_multi_collection_update_uses_one_behavior_and_each_receipt(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from anima3.neural import checkpoints, live, online, train
    monkeypatch.setattr(checkpoints, 'load_checkpoint', lambda _: SimpleNamespace(policy_sha='same'))
    verified = []
    monkeypatch.setattr(live, 'LiveReplayVerifier', lambda root: lambda e: verified.append(root.name) or True)
    def update(loaded, episodes, **kw):
        assert len(episodes) == 4
        assert all(kw['verify_episode'](e) for e in episodes)
        kw['out'].mkdir()
        return {'policy_sha':'new'}
    monkeypatch.setattr(train, 'update_verified_episodes', update)
    roots=[]
    for i in range(4):
        root=tmp_path/str(i)
        root.mkdir()
        episode=root/'episode.json'
        episode.write_text(json.dumps({'policy_sha':'same','eligible':True,'actor':57,
                                       'verification':{'match_id':str(i)}}))
        (root/'results.json').write_text(json.dumps({'domain':'servuo',
            'collection_status':'complete','matches':[{'episodes':[str(episode)]}]}))
        roots.append(root)
    result=online.update_collections('seed', roots, tmp_path/'out')
    assert result['accepted_episodes']==4 and verified==['0','1','2','3']
    with pytest.raises(ValueError,match='duplicate'):
        online.update_collections('seed', roots+roots[:1],tmp_path/'dup')
    with pytest.raises(online.NoEligibleEpisodes):
        online.update_collections('seed',roots[:3],tmp_path/'small')


def test_burst_and_potion_opponents_have_distinct_priorities():
    v=CombatView(opponent_visible=True, line_of_sight=True, distance=5,
                 reagent_counts=(100,)*8, elapsed=20, explosion_potions=5)
    burst=TacticalTeacher('burst').choose(v)
    potions=TacticalTeacher('potions').choose(v)
    assert burst == ACTION_INDEX['burst_bolt']
    assert potions == ACTION_INDEX['explosion_potion']


def test_gate_rejects_nonfinite_scores():
    before, after=report(0),report(1)
    after['rows'][0]['score']=float('inf')
    with pytest.raises(ValueError,match='scores'):
        promotion_gate(before,after)


def test_resume_archives_only_proven_preparation_failure(tmp_path):
    import json

    from scripts.run_training_league import recover_collection
    p=tmp_path/'fixture-003';p.mkdir()
    (p/'status.json').write_text(json.dumps({'stage':'stopped', 'error':
        'Server has not confirmed 100/25/100 stats after debuff recovery: [100, 34, 100]'}))
    assert recover_collection(p,'sha') is None
    assert not p.exists() and p.with_name('fixture-003-preparation-attempt-1').exists()
    p.mkdir()
    (p/'status.json').write_text(json.dumps({'stage':'stopped','error':'connection lost'}))
    with pytest.raises(RuntimeError,match='uncertain'):
        recover_collection(p,'sha')
    (p/'challenge-issued.json').write_text('{}')
    with pytest.raises(RuntimeError,match='uncertain'):
        recover_collection(p,'sha')


def test_start_template_checks_modified_resources_not_just_raw_header():
    from anima3.neural.live import starting_template_verified
    rows=[{'players':[{'stats':[100,25,100]}]*2},
          {'type':'frame','phase':'Countdown','players':[
              {'hitsMax':100,'manaMax':100,'stamMax':25,'alive':True,'poisoned':False},
              {'hitsMax':100,'manaMax':100,'stamMax':25,'alive':True,'poisoned':False}]}]
    assert starting_template_verified(rows)
    rows[1]['players'][0]['stamMax']=34
    assert not starting_template_verified(rows)


@pytest.mark.parametrize("mode", ['train', 'evaluation', 'collection'])
def test_runner_completes_and_resume_does_not_replay_or_retrain(tmp_path,monkeypatch,mode):
    import json
    from types import SimpleNamespace

    from scripts import run_training_league as runner
    counter={'games':0,'updates':0}
    monkeypatch.setattr(runner,'load_checkpoint',lambda *a:SimpleNamespace(policy_sha='same'))
    def collect(args,stop):
        counter['games']+=1
        log=args.log_dir;log.mkdir()
        match_id=f'{counter["games"]:032x}'
        folder=log/match_id;folder.mkdir()
        episode=folder/'episode.json'
        episode.write_text(json.dumps({'policy_sha':'same','eligible':True,
                    'invalid_reasons':[],'final_result':{'outcome':1}}))
        frames=[{'players':[{'stats':[100,25,100]}]*2},
                {'type':'frame','phase':'Countdown','players':[
                    {'hitsMax':100,'manaMax':100,'stamMax':25,'alive':True}]*2}]
        (folder/'replay.jsonl').write_text('\n'.join(json.dumps(f) for f in frames))
        report={'collection_status':'complete','matches':[{'match_id':match_id,
                    'episodes':[str(episode)],'replay':'local-test'}]}
        (log/'results.json').write_text(json.dumps(report))
    def update(checkpoint,collections,out,**kwargs):
        counter['updates']+=1;out.mkdir()
        summary={'update':{'parent_policy_sha':'same','policy_sha':'same'}}
        (out/'batch-update.json').write_text(json.dumps(summary))
        return summary
    monkeypatch.setattr(runner,'run',collect)
    monkeypatch.setattr(runner,'update_collections',update)
    monkeypatch.setattr(runner,'recover_collection',lambda p,sha:
                        json.loads((p/'results.json').read_text()) if p.exists() else None)
    accounts=tmp_path/'accounts.json';accounts.write_text(json.dumps({'test_a':'a','test_b':'b'}))
    argv=['--out',str(tmp_path/'run'),'--checkpoint',str(tmp_path/'seed'),
          '--accounts-file',str(accounts),'--bridge','unused','--data-dir','unused',
          '--user-a','test_a','--user-b','test_b']
    if mode == 'evaluation':
        argv += ['--batches','0','--candidate-checkpoint',str(tmp_path/'frozen')]
    elif mode == 'collection':
        argv += ['--batches','0','--collect-per-style','4']
    expected = {'games':28,'updates':1} if mode == 'train' else {'games':20,'updates':0}
    assert runner.main(argv)==0
    assert counter==expected
    assert runner.main(argv+['--resume'])==0
    assert counter==expected
    status=json.loads((tmp_path/'run/status.json').read_text())
    assert status['completed']==expected['games'] and status['status']=='complete'
    if mode == 'collection':
        from collections import Counter
        assert not status['fit_performed']
        assert not (tmp_path/'run/registry.json').exists()
        assert not list((tmp_path/'run').glob('evaluation-*.json'))
        assert Counter((r['style'],r['side']) for r in status['matches']) == {
            (style,side):2 for style in STYLES for side in range(2)}
        assert all(r['stage']=='training-demonstrations' for r in status['matches'])
    else:
        assert not status['gate']['passed']  # Pilot remains below evidence threshold.


def test_controller_upgrade_resume_requires_no_challenge(tmp_path):
    import json

    from scripts.run_training_league import recover_collection
    p = tmp_path / "fixture-009"
    p.mkdir()
    marker = {"reason": "controller_upgrade_before_challenge", "match_started": False}
    (p / "preparation-interrupted.json").write_text(json.dumps(marker))
    (p / "challenge-issued.json").write_text("{}")
    with pytest.raises(RuntimeError, match="uncertain"):
        recover_collection(p, "sha")
    (p / "challenge-issued.json").unlink()
    assert recover_collection(p, "sha") is None
    assert p.with_name("fixture-009-preparation-attempt-1").exists()


def test_gate_rejects_transport_invalid_episode():
    before, after = report(0), report(1)
    before['rows'][0]['eligible'] = False
    gate = promotion_gate(before, after)
    assert not gate['passed'] and not gate['valid_episodes']


def test_handshake_retry_preserves_attempt_and_same_fixture(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from anima3.sparring import TrainingIdleTimeout
    from scripts import run_training_league as runner
    opts = SimpleNamespace(log_dir=tmp_path / 'fixture-004', fixture_offset=3)
    calls = []
    def fake_run(actual, stop):
        calls.append(actual.fixture_offset)
        actual.log_dir.mkdir()
        (actual.log_dir / 'progress.jsonl').write_text('idle timeout evidence')
        if len(calls) == 1:
            raise TrainingIdleTimeout('no state')
        return 'done'
    stop = SimpleNamespace(is_set=lambda: False, wait=lambda seconds: False)
    monkeypatch.setattr(runner, 'run', fake_run)
    assert runner.collect_with_handshake_retry(opts, stop) == 'done'
    assert calls == [3, 3]
    assert (tmp_path / 'fixture-004-idle-retry-1/progress.jsonl').exists()


def test_handshake_retry_never_repeats_issued_challenge(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from anima3.sparring import TrainingIdleTimeout
    from scripts import run_training_league as runner
    opts = SimpleNamespace(log_dir=tmp_path / 'fixture')
    opts.log_dir.mkdir()
    (opts.log_dir / 'challenge-issued.json').write_text('{}')
    def fake_run(*args):
        raise TrainingIdleTimeout('late exception')
    monkeypatch.setattr(runner, 'run', fake_run)
    stop = SimpleNamespace(is_set=lambda: False, wait=lambda seconds: False)
    with pytest.raises(TrainingIdleTimeout):
        runner.collect_with_handshake_retry(opts, stop)
    assert (opts.log_dir / 'challenge-issued.json').exists()
    assert not list(tmp_path.glob('*retry*'))


@pytest.mark.parametrize('unit', [0.0, 1.0])
def test_curriculum_uses_matching_elapsed_discount(tmp_path, monkeypatch, unit):
    import json

    from anima3.neural import league
    original_collect, original_update = league.collect_simulator, league.ppo_update
    discounts, configs = [], []

    def collect(*args, **kwargs):
        discounts.append(kwargs['shaping_gamma'])
        return original_collect(*args, **kwargs)

    def update(model, optimizer, episodes, config, **kwargs):
        configs.append(config)
        return original_update(model, optimizer, episodes, config, **kwargs)

    monkeypatch.setattr(league, 'collect_simulator', collect)
    monkeypatch.setattr(league, 'ppo_update', update)
    league.main(['--out', str(tmp_path/'run'), '--bc-steps-per-drill', '4',
                 '--bc-epochs', '1', '--updates', '1', '--rollout-steps', '4',
                 '--max-steps', '4', '--eval-games-per-style', '2',
                 '--discount-time-unit', str(unit), '--threads', '1'])
    expected = .995 ** .25 if unit else .995
    assert len(discounts) == 5 and all(x == pytest.approx(expected) for x in discounts)
    assert configs[0].discount_time_unit == unit
    meta = json.loads((tmp_path/'run/warmstart/manifest.json').read_text())['training']
    assert meta['discount_time_unit'] == unit
    assert meta['shaping_gamma'] == pytest.approx(expected)


def test_curriculum_continues_exact_parent_and_rehearses_only_after_ppo(tmp_path, monkeypatch):
    import torch

    from anima3.neural import league
    from anima3.neural.checkpoints import load_checkpoint, save_checkpoint
    from anima3.neural.model import PolicyConfig, RecurrentPolicy
    from anima3.neural.schema import ACTION_NAMES, FEATURE_NAMES

    torch.manual_seed(83)
    parent = tmp_path / 'parent'
    model = RecurrentPolicy(PolicyConfig(len(FEATURE_NAMES), len(ACTION_NAMES), 64, 64))
    save_checkpoint(parent, model, training={'fixture': True})
    original = load_checkpoint(parent)
    events = []
    update, clone = league.ppo_update, league.behavioral_clone

    def checked_update(actual, *args, **kwargs):
        assert all(torch.equal(v, original.model.state_dict()[k])
                   for k, v in actual.state_dict().items())
        events.append('ppo')
        return update(actual, *args, **kwargs)

    def checked_clone(*args, **kwargs):
        assert events == ['ppo']
        events.append('rehearsal')
        return clone(*args, **kwargs)

    monkeypatch.setattr(league, 'ppo_update', checked_update)
    monkeypatch.setattr(league, 'behavioral_clone', checked_clone)
    out = tmp_path / 'run'
    league.main(['--out', str(out), '--start-checkpoint', str(parent),
                 '--bc-steps-per-drill', '4', '--bc-epochs', '1', '--updates', '1',
                 '--rollout-steps', '4', '--max-steps', '4', '--eval-games-per-style', '2',
                 '--bc-rehearsal-epochs', '1', '--threads', '1'])
    assert events == ['ppo', 'rehearsal']
    assert load_checkpoint(parent).policy_sha == original.policy_sha
    for name in ('initial', 'warmstart'):
        saved = load_checkpoint(out / name)
        assert saved.policy_sha == original.policy_sha
        assert saved.manifest['training']['parent_policy_sha'] == original.policy_sha
