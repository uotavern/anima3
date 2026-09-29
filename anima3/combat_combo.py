"""Pre-AOS Explosion/potion burst. Distinguish harmful spell and ground potion cursors."""

import time

from .contract import target_object, use
from .magic import SPELLS, VERDICTS, cast_proc


def cursor(obs, flag, kind):
    return obs.pending_target and obs.target_cursor_flag == flag and obs.target_cursor_type == kind


def burst_proc(potion, opponent, finisher="energy_bolt"):
    if finisher not in ("energy_bolt", "flamestrike"):
        raise ValueError("Unsupported burst finisher")

    def proc(obs0, memory):
        started = time.monotonic()
        memory["combo_after_s"] = started + 20
        memory["potion_after"] = memory.get("tick", 0) + 60
        memory["combo_events"] = []

        def note(stage):
            memory["combo_events"].append(
                {"stage": stage, "elapsed": round(time.monotonic() - started, 3)}
            )

        def failed(obs):
            return any(j.cliloc in VERDICTS for j in obs.new_journal)

        def danger(obs):
            return obs.player.hp_pct < 0.45 or obs.player.poisoned or obs.player.dead

        def throw(obs):
            # The spell must have been released/cancelled before opening this cursor.
            obs = yield use(potion)
            deadline = time.monotonic() + 1.2
            for _ in range(16):
                if cursor(obs, 0, 1):
                    note("potion_throw_target")
                    obs = yield target_object(opponent)
                    return obs
                if time.monotonic() >= deadline:
                    break
                obs = yield None
            note("potion_cursor_missing")
            return obs

        note("explosion_cast")
        obs = yield {"type": "CastSpell", "spell": SPELLS["explosion"].id}
        # Prime near the latter half of the cast, not after its harmful cursor:
        # using a potion after that cursor would cancel the pending spell.
        while time.monotonic() - started < 1.25:
            if failed(obs) or danger(obs):
                note("abort_before_prime")
                return "burst not primed"
            if cursor(obs, 1, 0):
                obs = yield target_object(opponent)
                note("early_spell_cursor")
                return "explosion only"
            obs = yield None
        if failed(obs) or danger(obs) or obs.pending_target:
            note("abort_before_prime")
            return "burst not primed"
        primed = time.monotonic()
        note("potion_prime")
        obs = yield use(potion)
        released = False
        # Leave a margin before the server's pre-AOS 3.75-second fuse.
        while time.monotonic() - primed < 2.2:
            if failed(obs) or danger(obs):
                break
            if cursor(obs, 1, 0):
                # Cast recovery starts when the spell cursor appears. A short hold
                # aligns its delayed impact with the follow-up cast and potion.
                ready = time.monotonic()
                while time.monotonic() - ready < 0.55 and time.monotonic() - primed < 2.2:
                    obs = yield None
                    if danger(obs) or not cursor(obs, 1, 0):
                        break
                if cursor(obs, 1, 0) and not danger(obs):
                    note("explosion_target")
                    obs = yield target_object(opponent)
                    released = True
                break
            obs = yield None
        if not released and cursor(obs, 1, 0):
            obs = yield {"type": "TargetCancel"}
        obs = yield from throw(obs)
        if not released or danger(obs):
            note("abort_after_throw")
            return "burst abandoned after potion"
        # Do not start a follow-up while a potion cursor is still outstanding.
        if obs.pending_target:
            note("throw_unconfirmed")
            return "potion throw not confirmed"
        if obs.player.mana < SPELLS[finisher].mana:
            return "insufficient finisher mana"
        note("finisher_" + finisher)
        result = yield from cast_proc(SPELLS[finisher], opponent)(obs, memory)
        note("finished_" + str(result))
        return result

    return proc
