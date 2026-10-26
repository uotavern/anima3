

def test_alternate_puts_the_challenge_in_bs_mouth():
    from types import SimpleNamespace

    from anima3.duel import ServerDuel

    def fighter(name, serial):
        said = []
        return SimpleNamespace(serial=serial, persona=SimpleNamespace(name=name),
                               body=SimpleNamespace(act=said.append), said=said)
    a, b = fighter("Ilse", 1), fighter("Torvald", 2)
    ServerDuel(a, b, 5, "5x-fists-magic", 3, a_challenges=False).challenge()
    assert not a.said and b.said[0]["text"] == "[Challenge Ilse 5 5x-fists-magic arena:3"


def _res(details, casts=({"energy_bolt": 9}, {"energy_bolt": 7}), state="done", fails=({}, {})):
    return {"state": state, "rounds": ["Round 1: Ilse defeats Torvald (hp 42%, 72 seconds)."],
            "duel_lines": details, "per": {"Ilse": {"casts_ok": casts[0], "cast_fail": fails[0]},
                                           "Torvald": {"casts_ok": casts[1], "cast_fail": fails[1]}}}


def test_a_clean_match_validates():
    from anima3.duel import validate_match
    res = _res(["Round 1 detail: Ilse hp 42%, Torvald hp 0%, attacks 17/18."])
    assert validate_match(res, {"Ilse": 0, "Torvald": 0}, mage=True) == []


def test_the_failures_that_once_passed_silently_are_void():
    from anima3.duel import validate_match
    frozen = _res(["Round 2 detail: Ilse4 hp 100%, Torvald4 hp 100%, attacks 0/0."], casts=({}, {}))
    got = validate_match(frozen, {"Ilse": 0, "Torvald": 0}, mage=True)
    assert any(g.startswith("round 2: attacks 0/0") for g in got) and "Ilse never tried to cast" in got
    clean = _res(["Round 1 detail: Ilse hp 42%, Torvald hp 0%, attacks 17/18."])
    assert validate_match(clean, {"Ilse": 2, "Torvald": 0}, mage=True) == ["Ilse's bridge reconnected 2x"]
    assert validate_match(clean, {}, missing=["Ilse:BlackPearl"]) == ["staging short of Ilse:BlackPearl"]
    assert validate_match(_res([], state="no-start"), {}) == ["match ended in state no-start"]


def test_a_fast_wipe_is_a_real_loss_not_a_frozen_fighter():
    from anima3.duel import validate_match
    wipe = _res(["Round 1: Torvald defeats Ilse (hp 100%, 8 seconds).",
                 "Round 1 detail: Torvald hp 100%, Ilse hp 0%, attacks 4/0."])
    assert validate_match(wipe, {}, mage=True) == []
    stall = _res(["Round 1: Torvald defeats Ilse (hp 100%, 150 seconds).",
                  "Round 1 detail: Torvald hp 100%, Ilse hp 0%, attacks 4/0."])
    assert validate_match(stall, {}, mage=True) == ["round 1: attacks 4/0 in 150 s"]


def test_a_mage_interrupted_every_time_lost_for_real():
    from anima3.duel import validate_match
    res = _res(["Round 1: Torvald defeats Ilse (hp 100%, 8 seconds).",
                "Round 1 detail: Ilse hp 0%, Torvald hp 100%, attacks 0/4."],
               casts=({}, {"energy_bolt": 9}), fails=({"no cursor": 6, "interrupted": 3}, {}))
    assert validate_match(res, {}, mage=True) == []


def test_a_model_that_slowed_to_seconds_voids_the_match():
    from anima3.duel import validate_match
    res = _res(["Round 1 detail: Ilse hp 0%, Torvald hp 100%, attacks 2/5."])
    res["per"]["Ilse"]["model_ms"] = 2100.0
    assert validate_match(res, {}, mage=True) == ["Ilse's model took 2100 ms per decision (limit 1000)"]
    res["per"]["Ilse"]["model_ms"] = 280.0
    assert validate_match(res, {}, mage=True) == []


def test_magic_reflection_goes_off_without_a_target_cursor():
    from types import SimpleNamespace

    from anima3.magic import SPELLS, cast_proc

    def obs(mana):
        return SimpleNamespace(player=SimpleNamespace(mana=mana), new_journal=[], pending_target=None)
    memory = {"tick": 40}
    proc = cast_proc(SPELLS["magic_reflection"], 7)(obs(100), memory)
    assert next(proc)["type"] == "CastSpell"
    proc.send(obs(100))                      # no cursor ever comes: keep watching the mana
    try:
        proc.send(obs(86))
    except StopIteration as done:
        assert done.value == "ok"
    assert memory["reflect_at"] == 40


def test_a_disturbed_magic_reflection_is_not_retried_at_once():
    from types import SimpleNamespace

    from anima3.magic import SPELLS, cast_proc, spell_order

    def obs(mana):
        return SimpleNamespace(player=SimpleNamespace(mana=mana), new_journal=[], pending_target=None)
    memory = {"tick": 100}
    proc = cast_proc(SPELLS["magic_reflection"], 7)(obs(100), memory)
    next(proc)
    result = None
    try:
        while True:
            proc.send(obs(100))              # concentration disturbed: the mana never goes
    except StopIteration as done:
        result = done.value
    assert result == "timeout" and memory["reflect_at"] == 100
    reflect_due = memory["tick"] + 10 - memory["reflect_at"] > 60
    assert spell_order("sustain", 1.0, False, False, False, reflect_due)[0] == "energy_bolt"
