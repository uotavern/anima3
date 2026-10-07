import json

from anima3.contract import Journal, Observation
from anima3.duel_wait import RULES, server_state


def message(state, serial=0xFFFFFFFF):
    return Journal(serial, "System", "[DuelState] " + json.dumps(state), 6, 53, 0)


def test_direct_challenge_trust_and_validation():
    invite = {"id": "a" * 32, "opponent": 123, "rounds": 3, "rules": RULES}
    state = {"phase": "Idle", "challenge": invite}
    assert server_state(Observation(new_journal=[message(state)])) == state
    assert server_state(Observation(new_journal=[message(state, 123)])) is None
    assert (
        server_state(
            Observation(
                new_journal=[message({"phase": "Idle", "challenge": {**invite, "id": "bad"}})]
            )
        )
        is None
    )
    assert server_state(Observation(new_journal=[message([])])) is None


def test_match_state_requires_server_opponent_and_round():
    state = {"phase": "Fighting", "id": "b" * 32, "opponent": 456, "round": 1, "rules": RULES}
    assert server_state(Observation(new_journal=[message(state)])) == state
    for patch in ({"round": 0}, {"opponent": True}, {"opponent": -1}, {"phase": "Finished"}):
        assert server_state(Observation(new_journal=[message({**state, **patch})])) is None


def test_configured_requires_exact_server_skill_and_stat_confirmation():
    from anima3.duel_wait import SKILLS, configured

    observation = {
        "player": {"strength": 100, "dexterity": 25, "intelligence": 100},
        "skills": [{"id": i, "base": 100} for i in SKILLS],
    }
    assert configured(observation)
    assert not configured({**observation, "skills": []})
    assert not configured(
        {**observation, "skills": observation["skills"] + [{"id": 10, "base": 10}]}
    )
    assert not configured(
        {**observation, "player": {"strength": 100, "dexterity": 100, "intelligence": 25}}
    )


def test_wait_stats_accepts_recovered_server_values_and_bounds_bad_configuration(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from anima3 import duel_wait

    body = SimpleNamespace(act=lambda action: None)
    clock = [0.0]
    monkeypatch.setattr(duel_wait.time, "monotonic", lambda: clock[0])
    debuffed = {"player": {"serial": 1, "strength": 89, "dexterity": 14, "intelligence": 100}}
    restored = {"player": {"serial": 1, "strength": 100, "dexterity": 25, "intelligence": 100}}
    def pump(body, seconds):
        clock[0] += seconds
        return restored if clock[0] >= 2 else debuffed
    monkeypatch.setattr(duel_wait, "pump", pump)
    assert duel_wait.wait_prepared_stats(body, debuffed) == restored
    clock[0] = 0
    with pytest.raises(RuntimeError, match="debuff recovery"):
        duel_wait.wait_prepared_stats(body, debuffed, timeout=1)


def test_previously_verified_build_waits_without_requesting_new_balls(monkeypatch):
    from anima3 import duel_wait
    observation = {
        "player": {"serial": 1, "strength": 89, "dexterity": 14, "intelligence": 100},
        "skills": [{"id": i, "base": 100} for i in duel_wait.SKILLS],
        "items": [{"serial": 2, "container": 1, "layer": 21}],
    }
    actions = []
    class Body:
        def act(self, action):
            actions.append(action)
        def observe_raw(self):
            return observation
    def recover(body, obs):
        obs["player"].update(strength=100, dexterity=25)
        return obs
    monkeypatch.setattr(duel_wait, "pump", lambda *args: observation)
    monkeypatch.setattr(duel_wait, "wait_prepared_stats", recover)
    assert duel_wait.prepare(Body(), previously_configured=True)["strength"] == 100
    assert not any(a.get("text") in ("[Arena skills", "[Arena stats") for a in actions)


def test_raw_stat_dialog_verifies_base_without_applying_or_accepting_buff(monkeypatch):
    from types import SimpleNamespace

    from anima3 import duel_wait
    actions=[]
    observation={'player':{'serial':1,'strength':100,'dexterity':34,'intelligence':100},
                 'items':[{'serial':2,'container':1,'layer':21},
                          {'serial':3,'container':2,'graphic':0xE2D,'hue':53}], 'gumps':[]}
    dialog={'serial':1,'gump_id':123,'layout':'ARENA STATS',
            'elements':[{'type':'entry','id':i,'s':str(v)} for i,v in enumerate((100,25,100))]}
    monkeypatch.setattr(duel_wait,'pump',lambda *a:{**observation,'gumps':[dialog]})
    body=SimpleNamespace(act=actions.append)
    assert duel_wait.read_base_stats(body,observation)==[100,25,100]
    assert actions[-1]['type']=='GumpResponse' and actions[-1]['button']==0


def test_verified_base_build_does_not_wait_for_modified_stats(monkeypatch):
    from types import SimpleNamespace

    from anima3 import duel_wait
    o={'player':{'serial':1,'strength':100,'dexterity':34,'intelligence':100},
       'skills':[{'id':i,'base':100} for i in duel_wait.SKILLS],
       'items':[{'serial':2,'container':1,'layer':21}]}
    monkeypatch.setattr(duel_wait,'pump',lambda *a:o)
    monkeypatch.setattr(duel_wait,'read_base_stats',lambda *a:[100,25,100])
    body=SimpleNamespace(act=lambda a:None,observe_raw=lambda:o)
    result=duel_wait.prepare(body,previously_configured=True,verify_base_stats=True)
    assert result['base_stat_verification']['values']==[100,25,100]
    monkeypatch.setattr(duel_wait,'read_base_stats',lambda *a:[100,50,75])
    import pytest
    with pytest.raises(RuntimeError,match='base stats changed'):
        duel_wait.prepare(body,previously_configured=True,verify_base_stats=True)


def test_training_supplies_reject_missing_potions_and_overweight():
    from anima3.duel_wait import training_supply_audit

    required = {g:100 for g in (0xF7A,0xF7B,0xF84,0xF85,0xF86,0xF88,0xF8D,0xF8C)}
    required.update({g:5 for g in (0xF0C,0xF07,0xF0D,0xF0B,0xF08)})
    o = {'player': {'serial':57, 'dead':False, 'weight':180, 'weight_max':390},
         'items':[{'serial':1000,'container':57,'layer':21,'graphic':0xE75}] +
                  [{'serial':2000+g,'container':1000,'graphic':g,'amount':n}
                   for g,n in required.items()]}
    assert training_supply_audit(o)['verified']
    o['player']['weight'] = 415
    assert not training_supply_audit(o)['verified']
    o['player']['weight'] = 180
    o['items'][-1]['amount'] = 0
    assert not training_supply_audit(o)['verified']
    o['items'][-1]['amount'] = 5
    o['items'][-1]['container'] = 999
    assert not training_supply_audit(o)['verified']
