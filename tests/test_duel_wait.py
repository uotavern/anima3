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
