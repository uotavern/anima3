from dataclasses import replace
from types import SimpleNamespace

from anima3.neural.melee import MeleeProbe
from anima3.neural.schema import CombatView, legal_mask


def view(**kw):
    return replace(CombatView(ready=True, fresh=True, opponent_visible=True,
                             line_of_sight=True, reagent_counts=(100,) * 8,
                             move_legal=(True,) * 4), **kw)


def test_harm_requires_close_range_and_approaches_first():
    p = MeleeProbe('punch')
    p.opening = {'weaken', 'clumsy'}
    v = view(distance=2)
    assert p.choose(v, legal_mask(v)) == 'approach'
    v = view(distance=1)
    assert p.choose(v, legal_mask(v)) == 'harm'


def test_stun_toggle_waits_for_confirmation_and_success_cooldown():
    p = MeleeProbe('stun')
    def packets(now):
        return p.packets(view(distance=1), now, 58, busy=False, hands_free=True)
    assert [x['type'] for x in packets(0)] == ['StunRequest', 'Attack']
    assert not any(x['type'] == 'StunRequest' for x in packets(3))
    p.feedback([SimpleNamespace(cliloc=1019011)], 3)
    p.feedback([SimpleNamespace(cliloc=1004010)], 4)
    assert p.stun_ready
    assert not any(x['type'] == 'StunRequest' for x in packets(15))
    p.feedback([SimpleNamespace(cliloc=1004013)], 16)
    assert not any(x['type'] == 'StunRequest' for x in packets(25))
    assert any(x['type'] == 'StunRequest' for x in packets(26))


def test_no_punch_or_stun_with_unsafe_state_or_owned_cursor():
    for v in (view(fresh=False), view(ready=False), view(paralyzed=True),
              view(distance=2), view(opponent_visible=False)):
        assert MeleeProbe('stun').packets(v, 0, 58, busy=False, hands_free=True) == []
    for busy, hands in ((True, True), (False, False)):
        assert MeleeProbe('stun').packets(view(), 0, 58, busy=busy, hands_free=hands) == []
    assert MeleeProbe('magic').packets(view(), 0, 58, busy=False, hands_free=True) == []


def test_emergency_heal_precedes_closing_distance():
    v = view(distance=6, hp=40)
    assert MeleeProbe('punch').choose(v, legal_mask(v)) == 'greater_heal'
