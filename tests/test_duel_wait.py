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
    from anima3.duel_wait import configured, SKILLS

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
