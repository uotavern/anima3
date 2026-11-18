from anima3.contract import Observation
from anima3.magic import explosion_potion_proc


def test_explosion_owns_target_and_cooldown():
    memory = {"tick": 7}
    gen = explosion_potion_proc(123, 456)(Observation(), memory)
    assert next(gen) == {"type": "Use", "serial": 123}
    assert memory["potion_after"] == 47
    assert gen.send(Observation()) is None
    assert gen.send(Observation(pending_target=True)) == {"type": "TargetObject", "serial": 456}
