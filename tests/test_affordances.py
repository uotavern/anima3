import pytest

from anima3.affordances import enumerate_affordances
from anima3.body import FakeBody
from anima3.contract import BANDAGE_GRAPHIC, GOLD_GRAPHIC
from anima3.persona import Persona
from anima3.scene import facts, render

P = Persona(name="T", combat_disposition="neutral", talkativeness=1.0, speech_examples=["hi", "yo"])


def ids(world, persona=P, memory=None):
    obs = world.observe()
    return [a.id for a in enumerate_affordances(obs, facts(obs), persona, memory if memory is not None else {})]


def test_dead_offers_nothing():
    w = FakeBody(); w.player.dead = True
    assert ids(w) == []


def test_critical_hp_with_hostile_only_flee_or_bandage():
    w = FakeBody(); w.player.hits = 10; w.add_hostile(1, 0); w.add_pack_item(BANDAGE_GRAPHIC, 2)
    got = ids(w)
    assert got[0] == "flee" and "bandage" in got and not any(g.startswith("attack") for g in got)


def test_critical_duel_warrior_heals_after_escaping_melee():
    w = FakeBody(); w.player.hits = 10
    opponent = w.add_hostile(5, 0)
    w.add_pack_item(BANDAGE_GRAPHIC, 10)
    memory = {"duel": True, "duel_opponent": opponent.serial}
    assert ids(w, memory=memory)[0] == "bandage"
    opponent.pos = type(opponent.pos)(w.player.pos.x + 1, w.player.pos.y, 0)
    assert ids(w, memory=memory)[0] == "flee"


def test_pacifist_is_never_offered_attack():
    w = FakeBody(); w.add_hostile(1, 0)
    got = ids(w, Persona(name="G", combat_disposition="pacifist"))
    assert not any(g.startswith("attack") for g in got) and got[0] == "flee"


def test_defensive_attacks_only_when_close():
    w = FakeBody(); m = w.add_hostile(5, 0)
    far = ids(w, Persona(name="A", combat_disposition="defensive"))
    m.pos = type(m.pos)(w.player.pos.x + 1, w.player.pos.y, 0)
    near = ids(w, Persona(name="A", combat_disposition="defensive"))
    assert not any(g.startswith("attack") for g in far) and near[0].startswith("attack")


def test_peaceful_loot_first_then_greet_then_wander():
    w = FakeBody(); w.add_ground_item(GOLD_GRAPHIC, 1, 0, 5); w.add_person(2, 0)
    got = ids(w)
    assert got[0].startswith("pickup:") and any(g.startswith("say:") for g in got) and any(g.startswith("walk:") for g in got)
    assert "hold" not in got                      # something worth doing is on the menu
    assert ids(FakeBody())[-1] == "hold"          # nothing to do: wander or hold


def test_scene_mentions_threat_and_health_words():
    w = FakeBody(); w.player.hits = 20; w.add_hostile(1, 0)
    obs = w.observe(); f = facts(obs)
    s = render(obs, f, "Grimm, a miner")
    assert "Hostiles:" in s and "adjacent" in s and "low" in s


def test_equip_offered_when_gear_in_pack_and_no_close_threat():
    w = FakeBody(); w.add_pack_item(0x13FF)  # a katana in the pack
    got = ids(w)
    assert got[0].startswith("equip:")
    w.add_hostile(1, 0)
    assert not any(g.startswith("equip:") for g in ids(w))  # not with a hostile adjacent
    w2 = FakeBody(); w2.add_pack_item(0x13FF)
    from anima3.contract import Item, Pos
    w2.worn.append(Item(0x9, 0x13FF, 1, Pos(), w2.player.serial, 1, 0))   # a katana already worn on layer 1
    obs = w2.observe()
    from anima3.affordances import _unequipped_gear
    assert _unequipped_gear(obs, {}) == []


def test_loot_offered_only_for_own_corpse():
    from anima3.contract import Item, Pos
    w = FakeBody()
    obs = w.observe()
    corpse = Item(0x777, 0x2006, 1, Pos(101, 100, 0), None, 0, 1)
    obs.items.append(corpse)
    from anima3.affordances import enumerate_affordances
    from anima3.scene import facts
    assert not any(a.id.startswith("loot:") for a in enumerate_affordances(obs, facts(obs), P, {}))
    obs.corpse_of = {0x777: 0x999}
    assert any(a.id == "loot:1911" for a in enumerate_affordances(obs, facts(obs), P, {"attacked": {0x999}}))


def test_equip_procedure_lifts_then_equips_and_confirms():
    from anima3.agent import Agent
    from anima3.decision import Scripted
    w = FakeBody(); w.add_pack_item(0x13FF)
    ag = Agent(w, Persona(name="W", talkativeness=0), Scripted(), pump_ms=0)
    ag.run(4)
    sent = [a for a in w.log if a["type"] not in ("AllNames", "Use")]
    assert [a["type"] for a in sent[:2]] == ["PickUp", "Equip"] and sent[1]["layer"] == 1
    assert any(v == "ok" for _, pid, v in ag.proc_log if pid.startswith("equip:"))
    assert not any(i.graphic == 0x13FF for i in w.pack) and w.worn


def test_far_threat_does_not_stop_a_healthy_worker():
    w = FakeBody(); w.add_hostile(5, 0)
    mem = {}
    obs = w.observe()
    got = [a.id for a in enumerate_affordances(obs, facts(obs), Persona(name="G", combat_disposition="pacifist"), mem)]
    assert mem.get("threat_far") is True and "flee" in got
    w2 = FakeBody(); w2.add_hostile(2, 0); mem2 = {}
    obs2 = w2.observe(); enumerate_affordances(obs2, facts(obs2), Persona(name="G", combat_disposition="pacifist"), mem2)
    assert not mem2.get("threat_far")


def test_visit_offered_for_a_person_in_the_middle_distance():
    w = FakeBody(); w.add_person(7, 0, "Grimm")
    got = ids(w)
    assert any(g.startswith("visit:") for g in got)


def test_chase_uses_walkto_then_attacks_when_adjacent():
    from anima3.agent import Agent
    from anima3.decision import Scripted
    w = FakeBody(); w.add_hostile(4, 0, aggressive=False)
    ag = Agent(w, Persona(name="R", combat_disposition="aggressive"), Scripted(), pump_ms=0)
    ag.run(8)
    types = [x["type"] for x in w.log if x["type"] != "AllNames"]
    assert "WarMode" in types and "WalkTo" in types and "Attack" in types
    assert any(v == "engaged" for _, pid, v in ag.proc_log if pid.startswith("attack:"))


def test_pacifist_being_hit_gets_flee_only():
    w = FakeBody(); w.add_hostile(1, 0)
    mem = {"hp_trend": -0.1}
    obs = w.observe()
    got = [a.id for a in enumerate_affordances(obs, facts(obs), Persona(name="G", combat_disposition="pacifist"), mem)]
    assert got == ["flee"]


def test_hostile_on_another_level_is_not_targeted():
    w = FakeBody(); m = w.add_hostile(1, 0)
    from anima3.contract import Pos
    m.pos = Pos(m.pos.x, m.pos.y, 20)      # twenty tiles up the cliff
    obs = w.observe(); f = facts(obs)
    assert f.nearest_hostile is None
    got = [a.id for a in enumerate_affordances(obs, f, Persona(name="R", combat_disposition="aggressive"), {})]
    assert not any(g.startswith(("attack:", "flee")) for g in got)


def test_blacklisted_target_is_skipped():
    w = FakeBody(); a = w.add_hostile(1, 0); b = w.add_hostile(2, 0)
    obs = w.observe(); f = facts(obs)
    mem = {"tick": 10, "target_blacklist": {a.serial: 100}}
    got = [x.id for x in enumerate_affordances(obs, f, Persona(name="R", combat_disposition="aggressive"), mem)]
    assert got[0] == f"attack:{b.serial}"


def test_only_own_kills_are_looted():
    from anima3.contract import Item, Pos
    w = FakeBody(); obs = w.observe()
    obs.items.append(Item(0x777, 0x2006, 1, Pos(101, 100, 0), None, 0, 1)); obs.corpse_of = {0x777: 0x999}
    assert not any(a.id.startswith("loot:") for a in enumerate_affordances(obs, facts(obs), P, {"attacked": set()}))
    assert any(a.id == "loot:1911" for a in enumerate_affordances(obs, facts(obs), P, {"attacked": {0x999}}))


def test_friends_are_never_threats():
    w = FakeBody(); m = w.add_hostile(1, 0)
    obs = w.observe(); f = facts(obs)
    got = [a.id for a in enumerate_affordances(obs, f, Persona(name="R", combat_disposition="aggressive"), {"friends": {m.serial}})]
    assert not any(g.startswith(("attack:", "flee")) for g in got)


def test_overloaded_offers_only_dropping_surplus():
    from anima3.contract import Item, Pos
    w = FakeBody(); w.player.weight = 250; w.player.weight_max = 250
    w.add_pack_item(0x13FF)                                                   # a spare katana in the pack
    w.worn.append(Item(0x9, 0x13FF, 1, Pos(), w.player.serial, 1, 0))       # one already worn
    obs = w.observe()
    got = [a.id for a in enumerate_affordances(obs, facts(obs), P, {})]
    assert len(got) == 1 and got[0].startswith("drop:")
    from anima3.agent import Agent
    from anima3.decision import Scripted
    ag = Agent(w, P, Scripted(), pump_ms=0)
    ag.run(3)
    assert any(x["type"] == "Drop" and x["container"] == 0xFFFFFFFF for x in w.log)


def test_heavy_pack_offers_dropping_alongside_other_verbs():
    from anima3.contract import Item, Pos
    w = FakeBody(); w.player.weight = 200; w.player.weight_max = 250
    w.add_pack_item(0x13FF); w.worn.append(Item(0x9, 0x13FF, 1, Pos(), w.player.serial, 1, 0))
    obs = w.observe()
    got = [a.id for a in enumerate_affordances(obs, facts(obs), P, {})]
    assert got[0].startswith("drop:") and len(got) > 1


def test_bandage_is_a_procedure_that_waits_for_the_wrap_to_finish():
    from anima3.agent import Agent
    from anima3.contract import BANDAGE_GRAPHIC
    from anima3.decision import Scripted
    w = FakeBody(); w.player.hits = 20; w.add_pack_item(BANDAGE_GRAPHIC, 3)
    ag = Agent(w, Persona(name="G"), Scripted(), pump_ms=0)
    ag.run(3)
    sent = [x for x in w.log if x["type"] == "BandageTarget"]
    assert len(sent) == 1 and any(v == "ok" for _, pid, v in ag.proc_log if pid == "bandage")


@pytest.mark.parametrize("pump_ms", [50, 250])
def test_bandage_waits_for_delayed_completion_even_after_slip(pump_ms):
    from anima3.agent import Agent
    from anima3.contract import Journal
    from anima3.decision import Scripted

    class SlowHealingBody(FakeBody):
        healing_ms = None

        def act(self, action):
            if action["type"] == "BandageTarget":
                self.log.append(action)
                self.healing_ms = 0
            else:
                super().act(action)

        def pump(self, ms):
            super().pump(ms)
            if self.healing_ms is not None:
                self.healing_ms += ms
                if self.healing_ms == 1000:
                    self.journal.append(Journal(0, "", "Your fingers slip!", 0, 0, 500961))
                if self.healing_ms >= 8000:
                    self.player.hits = self.player.hits_max
                    self.journal.append(Journal(0, "", "You finish applying the bandages.", 0, 0, 500969))
                    self.healing_ms = None

    w = SlowHealingBody(); w.player.hits = 10; w.add_pack_item(BANDAGE_GRAPHIC, 3)
    ag = Agent(w, Persona(name="G"), Scripted(), pump_ms=pump_ms)
    ag.run(9000 // pump_ms)
    assert w.player.hits == w.player.hits_max
    assert len([a for a in w.log if a["type"] == "BandageTarget"]) == 1
    assert any(v == "ok" for _, pid, v in ag.proc_log if pid == "bandage")


def test_critically_hurt_duel_mage_heals_rather_than_only_fleeing():
    from anima3.magic import REAGENT_GRAPHICS
    w = FakeBody(); w.player.hits = 10; w.player.mana = w.player.mana_max = 100
    opp = w.add_hostile(4, 0)
    for g in REAGENT_GRAPHICS:
        w.add_pack_item(g, 20)
    got = ids(w, memory={"duel": True, "mage": True, "duel_opponent": opp.serial})
    assert got[0] == "cast:greater_heal" and "flee" not in got and "hold" not in got
    w.player.mana = 0                      # no mana for a heal: now running is the answer
    got = ids(w, memory={"duel": True, "mage": True, "duel_opponent": opp.serial})
    assert "flee" in got and "cast:greater_heal" not in got
