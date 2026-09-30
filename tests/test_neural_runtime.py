"""Real wire sequencing under controlled time; no server or private state needed."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace

import pytest

from anima3.contract import Item, Journal, Mobile, Observation, Player, Pos, Terrain
from anima3.neural.executor import CombatExecutor, cast_seconds
from anima3.neural.inference import InferenceClient
from anima3.neural.schema import ACTION_INDEX, REAGENT_GRAPHICS, Frame


def observation(*, cursor=None, enemy_x=106, hp=100, mana=100, journal=()):
    player = Player(
        serial=1,
        pos=Pos(100, 100),
        hits=hp,
        hits_max=100,
        mana=mana,
        mana_max=100,
        stam=100,
        stam_max=100,
    )
    enemy = Mobile(2, "Opponent", Pos(enemy_x, 100), 400, 5, 100, 100, abs(enemy_x - 100))
    pack = Item(1001, 0x0E75, 1, Pos(), 1, 0x15, 0)
    items = [pack] + [
        Item(2000 + i, graphic, 100, Pos(), 1001, 0, 0)
        for i, graphic in enumerate(REAGENT_GRAPHICS)
    ]
    items += [Item(3001, 0x0F0D, 20, Pos(), 1001, 0, 0), Item(3002, 0x0F0C, 20, Pos(), 1001, 0, 0)]
    return Observation(
        player=player,
        mobiles=[enemy],
        items=items,
        new_journal=list(journal),
        pending_target=cursor is not None,
        target_cursor_flag=cursor[0] if cursor else None,
        target_cursor_type=cursor[1] if cursor else None,
        target_cursor_id=cursor[2] if cursor else None,
        terrain=Terrain((88, 88), 25, "." * 625),
    )


def executor(obs=None, now=0.0):
    ex = CombatExecutor(2)
    ex.reset(2, "match-1/round-1", now)
    ex.observe(obs or observation(), {"phase": "Fighting", "opponent": 2}, now)
    return ex


def test_burst_releases_spell_before_potion_and_finisher():
    ex, obs = executor(), observation()
    assert cast_seconds("explosion") == 1.65
    assert ex.start("burst_bolt", obs, 0)
    assert ex.tick(obs, 0) == [{"type": "CastSpell", "spell": 43}]
    assert ex.tick(observation(), 0.85) == [{"type": "Use", "serial": 3001}]
    # The potion target must never be mistaken for Explosion's target.
    assert ex.tick(observation(cursor=(0, 1, 11)), 1.0) == []
    assert ex.tick(observation(cursor=(1, 0, 12)), 1.65) == []
    assert ex.tick(observation(cursor=(1, 0, 12)), 2.31) == [{"type": "TargetObject", "serial": 2}]
    assert ex.tick(observation(mana=80), 2.36) == [{"type": "Use", "serial": 3001}]
    assert ex.tick(observation(cursor=(0, 1, 13), mana=80), 2.41) == [
        {"type": "TargetObject", "serial": 2}
    ]
    assert ex.tick(observation(mana=80), 2.46) == []
    assert ex.tick(observation(mana=80), 2.51) == [{"type": "CastSpell", "spell": 42}]
    assert ex.tick(observation(cursor=(1, 0, 14), mana=80), 4.16) == [
        {"type": "TargetObject", "serial": 2}
    ]
    assert ex.tick(observation(mana=60), 4.21) == []
    assert ex.tick(observation(mana=60), 4.92) == []
    assert not ex.busy
    result = [event for event in ex.drain_events() if event["type"] == "action_finished"][-1]
    assert result["success"] and not result["intervention"]
    assert result["potion_throw_at"] == 2.41


def test_fizzle_after_priming_emergency_throws_without_model():
    ex, obs = executor(), observation()
    assert ex.start("burst_bolt", obs, 0)
    ex.tick(obs, 0)
    ex.tick(observation(), 0.85)
    fizzled = observation(cursor=(0, 1, 12), journal=[Journal(1, "", "", 0, 0, 502632)])
    assert ex.tick(fizzled, 1.2) == [{"type": "TargetObject", "serial": 2}]
    ex.tick(observation(), 1.3)
    assert not ex.busy
    events = ex.drain_events()
    assert any(event.get("intervention") for event in events)
    assert not any(event.get("packet", {}).get("spell") == 42 for event in events)


def test_clumsy_interrupt_before_prime_never_lights_potion():
    # ServUO's Weaken/Clumsy call OnCasterHurt without dealing HP damage.
    # This exact pattern interrupted Explosion in the first real pilot.
    ex, obs = executor(), observation()
    assert ex.start("burst_bolt", obs, 0)
    assert ex.tick(obs, 0) == [{"type": "CastSpell", "spell": 43}]
    interrupted = observation(journal=[Journal(1, "", "", 0, 0, 500641)])
    assert ex.tick(interrupted, 0.21) == []
    assert not ex.busy
    assert ex.tick(observation(), 0.85) == []
    events = ex.drain_events()
    assert not any(event.get("packet", {}).get("type") == "Use" for event in events)
    result = [event for event in events if event["type"] == "action_finished"][-1]
    assert result["reason"] == "concentration disturbed"
    assert not result["policy_override"]


def test_stale_duel_state_does_not_abandon_lit_potion():
    obs = observation()
    ex = executor(obs, 0)
    # Start when state is fresh, but its 3-second TTL expires while cooking.
    assert ex.start("explosion_potion", obs, 2.5)
    assert ex.tick(obs, 2.5) == [{"type": "Use", "serial": 3001}]
    assert ex.tick(observation(cursor=(0, 1, 11)), 2.6) == []
    frame = ex.observe(observation(cursor=(0, 1, 11)), now=4.6)
    assert frame.mask[0] and sum(frame.mask) == 1
    assert ex.tick(observation(cursor=(0, 1, 11)), 4.61) == [{"type": "TargetObject", "serial": 2}]
    ex.tick(observation(), 4.7)
    assert not ex.busy
    assert not ex.start("energy_bolt", observation(), 4.8)


def test_potion_target_close_uses_distant_empty_ground():
    ex, obs = executor(), observation()
    ex.start("explosion_potion", obs, 0)
    ex.tick(obs, 0)
    ex.tick(observation(cursor=(0, 1, 11)), 0.1)
    packet = ex.tick(observation(cursor=(0, 1, 11), enemy_x=101), 0.3)[0]
    assert packet["type"] == "TargetGround"
    assert max(abs(packet["x"] - 100), abs(packet["y"] - 100)) >= 6
    assert any(event.get("intervention") for event in ex.drain_events())


def test_wrong_cursor_never_targeted_as_spell_and_deadline_cancels():
    ex, obs = executor(), observation()
    ex.start("energy_bolt", obs, 0)
    ex.tick(obs, 0)
    wrong = observation(cursor=(0, 1, 11))
    assert ex.tick(wrong, 1.7) == []
    assert ex.tick(wrong, 3.8) == [{"type": "TargetCancel"}]
    assert not ex.busy


def test_late_spell_cursor_after_timeout_is_cancelled_with_bounded_retry():
    ex, obs = executor(), observation()
    assert ex.start("poison", obs, 0)
    assert ex.tick(obs, 0) == [{"type": "CastSpell", "spell": 20}]
    assert ex.tick(observation(), 3.1) == []
    assert not ex.busy
    late = observation(cursor=(1, 0, 77))
    ex.observe(late, {"phase": "Fighting", "opponent": 2}, 11.4)
    assert ex.tick(late, 11.4) == [{"type": "TargetCancel"}]
    assert ex.tick(late, 11.5) == []
    assert ex.tick(late, 12.0) == [{"type": "TargetCancel"}]
    assert ex.tick(late, 12.6) == []
    ex.tick(observation(), 12.7)
    assert ex.start("energy_bolt", observation(), 12.8)
    cleanups = [row for row in ex.drain_events() if row["type"] == "orphan_cursor_cleanup"]
    assert len(cleanups) == 2
    assert all(
        row["scope"] == "option_execution" and not row["policy_override"] for row in cleanups
    )


def test_orphan_cleanup_requires_trusted_fighting_and_preserves_potion_cursor():
    ex = executor()
    late = observation(cursor=(2, 0, 78))
    ex.observe(late, {"phase": "RoundOver", "opponent": 2}, 1)
    assert ex.tick(late, 1) == []
    ex.observe(late, {"phase": "Fighting", "opponent": 3}, 1.1)
    assert ex.tick(late, 1.1) == []
    ex.observe(late, {"phase": "Fighting", "opponent": 2}, 1.2)
    assert ex.tick(late, 4.3) == []  # Stale state cannot authorize cleanup.
    ex.reset(2, "next-round", 4.4)
    assert ex.tick(late, 4.4) == []  # Reset revoked the old match authority.
    potion = observation(cursor=(0, 1, 79))
    ex.observe(potion, {"phase": "Fighting", "opponent": 2}, 4.5)
    assert ex.tick(potion, 4.5) == []
    ex.observe(late, now=4.6)
    assert ex.tick(late, 4.6) == [{"type": "TargetCancel"}]


def test_tracker_pins_opponent_and_never_accepts_other_mana():
    obs = observation()
    obs.new_journal = [
        Journal(3, "Other", "Corp Por", 0, 0, 0),
        Journal(2, "Opponent", "Des Mani", 0, 0, 0),
    ]
    ex = executor(obs)
    assert ex.view.opponent_cast == "weaken"
    assert not hasattr(ex.view, "opponent_mana")
    wrong = ex.observe(obs, {"phase": "Fighting", "opponent": 3}, 0.1)
    assert sum(wrong.mask) == 1
    unknown = replace(obs, mobiles=[])
    ex.observe(unknown, {"phase": "Fighting", "opponent": 2}, 0.2)
    assert not ex.view.opponent_health_known
    assert not ex.view.opponent_visible


def test_public_explosion_word_estimate_expires_without_hidden_queue():
    obs = observation()
    ex = executor(obs)
    assert ex.view.incoming_explosion_seconds == -1
    obs.new_journal = [Journal(3, "Other", "Vas Ort Flam", 0, 0, 0)]
    ex.observe(replace(obs), now=0.1)
    assert ex.view.incoming_explosion_seconds == -1
    heard = observation(journal=[Journal(2, "Opponent", "Vas Ort Flam", 0, 0, 0)])
    ex.observe(heard, now=1.0)
    assert abs(ex.view.incoming_explosion_seconds - 4.15) < 1e-9
    ex.observe(observation(), now=2.0)
    assert abs(ex.view.incoming_explosion_seconds - 3.15) < 1e-9
    ex.observe(observation(), now=5.16)
    assert ex.view.incoming_explosion_seconds == -1


def test_missing_potion_inventory_masks_actions():
    obs = observation()
    obs.items = [item for item in obs.items if item.graphic != 0x0F0D]
    ex = executor(obs)
    frame = ex.observe(obs, now=0.1)
    assert not frame.mask[ACTION_INDEX["explosion_potion"]]
    assert not frame.mask[ACTION_INDEX["burst_bolt"]]
    assert not ex.start("explosion_potion", obs, 0.1)


FAKE_WORKER = r"""
import json,sys,time
print(json.dumps(dict(type="ready",policy_sha="abc",hidden_size=2,feature_size=2,action_size=2)),flush=True)
for line in sys.stdin:
    row=json.loads(line)
    time.sleep(0.08 if row["features"][0] else 0.002)
    print(json.dumps(dict(type="decision",id=row["id"],epoch=row["epoch"],action=1,log_prob=-0.5,value=0.2,hidden_out=[x+1 for x in row["hidden"]],policy_sha="abc",infer_ms=2)),flush=True)
"""


def wait_for(callable_, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = callable_()
        if result:
            return result
        time.sleep(0.002)
    raise AssertionError("timed out waiting for inference test")


def test_inference_one_inflight_deadline_and_epoch_reset():
    client = InferenceClient(
        "unused", deadline_ms=40, worker_command=[sys.executable, "-u", "-c", FAKE_WORKER]
    ).start()
    try:
        wait_for(lambda: client.ready)
        client.reset("one")
        assert client.submit(Frame([0, 0], [True, True]), "one")
        assert client.submit(Frame([0, 0], [True, True]), "one") is None
        decision = wait_for(lambda: client.take("one"))
        assert decision.hidden_in == (0, 0) and client.hidden == (1, 1)
        assert client.submit(Frame([1, 0], [True, True]), "one")
        time.sleep(0.1)
        assert client.take("one") is None
        assert client.hidden == (1, 1)
        assert any(row.get("reason") == "deadline" for row in client.drain_events())
        assert client.submit(Frame([1, 0], [True, True]), "one")
        client.reset("two")
        time.sleep(0.1)
        assert client.take("two") is None
        assert client.hidden == (0, 0)
        assert client.submit(Frame([0, 0], [True, True]), "two")
        decision = wait_for(lambda: client.take("two"))
        assert decision.hidden_in == (0, 0)
        assert decision.features == (0, 0)
    finally:
        client.close()


def test_inference_rechecks_sampled_mask():
    client = InferenceClient(
        "unused", worker_command=[sys.executable, "-u", "-c", FAKE_WORKER]
    ).start()
    try:
        wait_for(lambda: client.ready)
        client.submit(Frame([0, 0], [True, False]), "one")
        time.sleep(0.03)
        assert client.take("one") is None
        assert any(row.get("reason") == "illegal_action" for row in client.drain_events())
        assert client.hidden == (0, 0)
    finally:
        client.close()


def queued_inference():
    """Controlled IPC timing without sleeping or invoking model computation."""
    client = InferenceClient("unused", deadline_ms=500)
    client.process = SimpleNamespace(stdin=StringIO())
    client._inbox.put(
        (
            {
                "type": "ready",
                "policy_sha": "abc",
                "hidden_size": 2,
                "feature_size": 2,
                "action_size": 2,
            },
            99.0,
        )
    )
    client._poll(100.0)
    client.reset("one")
    return client


def queue_policy_response(client, received_at):
    pending = client._pending
    client._inbox.put(
        (
            {
                "type": "decision",
                "id": pending["id"],
                "epoch": "one",
                "action": 1,
                "log_prob": -0.5,
                "value": 0.2,
                "hidden_out": [1, 1],
                "policy_sha": "abc",
                "infer_ms": 2,
            },
            received_at,
        )
    )


def test_inference_separates_observation_submission_arrival_and_consumption():
    client = queued_inference()
    assert client.submit(Frame([0, 0], [True, True]), "one", 100.2, observed_at=100.0)
    queue_policy_response(client, 100.203)
    decision = client.take("one", 100.49)
    assert decision is not None
    assert decision.observed_at == 100.0 and decision.submitted_at == 100.2
    assert decision.received_at == 100.203
    assert decision.age_ms == pytest.approx(3.0)
    assert client.hidden == (1, 1)


@pytest.mark.parametrize("poll_before_take", [False, True])
def test_fast_response_consumed_late_keeps_observation_deadline_and_hidden_uncommitted(
    poll_before_take,
):
    client = queued_inference()
    client.submit(Frame([0, 0], [True, True]), "one", 100.2, observed_at=100.0)
    queue_policy_response(client, 100.203)
    if poll_before_take:
        client._poll(100.3)
    # Request age is only310ms, but source observation is already510ms old.
    assert client.take("one", 100.51) is None
    assert client.hidden == (0, 0)
    rejected = [row for row in client._events if row.get("reason") == "deadline"]
    assert len(rejected) == 1
    assert rejected[0]["observation_age_ms"] == pytest.approx(510)
    assert rejected[0]["request_age_ms"] == pytest.approx(310)
    assert rejected[0]["response_wait_ms"] == pytest.approx(307)
    assert rejected[0]["received_at"] == 100.203


def test_stale_input_is_skipped_without_new_sample_or_fake_freshness():
    client = queued_inference()
    assert client.submit(Frame([0, 0], [True, True]), "one", 100.6, observed_at=100.0) is None
    assert client.process.stdin.getvalue() == ""
    assert client.hidden == (0, 0) and client._pending is None
    assert client._events[-1]["type"] == "inference_skipped"
    with pytest.raises(ValueError, match="observation time"):
        client.submit(Frame([0, 0], [True, True]), "one", 100.6, observed_at=100.7)


def test_stdout_reader_records_arrival_before_actor_consumption():
    client = InferenceClient(
        "unused", deadline_ms=500, worker_command=[sys.executable, "-u", "-c", FAKE_WORKER]
    ).start()
    try:
        wait_for(lambda: client.ready)
        client.submit(Frame([0, 0], [True, True]), "one")
        wait_for(lambda: not client._inbox.empty())
        inbox_ready_at = time.monotonic()
        time.sleep(0.02)
        decision = client.take("one")
        assert decision is not None
        assert decision.received_at <= inbox_ready_at
        assert time.monotonic() - decision.received_at >= 0.02
    finally:
        client.close()
