"""The two-rate loop.

Fast (every tick): observe → facts → affordances → execute → pump. The rule's
first affordance is always executable on its own, so the character stays alive
with no model at all. Slow (every `decide_every` ticks, or immediately when the
situation changes): ask the decision client to pick from the menu, off-thread,
with a deadline. An admitted pick becomes a short *plan* that is repeated while
it stays valid, so steering is not diluted by the rule between decisions.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .affordances import Affordance, enumerate_affordances
from .body import Body
from .contract import all_names, click
from .contract import use as click_use
from .decision import Admitted, Decision, DecisionClient, gate
from .persona import Persona
from .progression import curriculum_key, gm_count, progress_scene, train_verbs, training_delta
from .scene import facts, render
from .triage import addressed_to_me

try:
    from .economy import econ_facts, econ_scene, economy_affordances
except ImportError:  # pragma: no cover
    econ_facts = econ_scene = economy_affordances = None

QUESTION = "What should you do right now?"


@dataclass
class TickReport:
    tick: int
    hp_pct: float
    dead: bool
    hostiles: int
    gold: int
    chosen: str | None = None
    used_model: bool = False
    reason: str = ""
    confidence: float | None = None
    backend: str = ""
    ms: float | None = None
    options: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self, body: Body, persona: Persona, client: DecisionClient, *,
        decide_every: int = 4, plan_ticks: int = 4, threshold: float = 0.35, deadline_s: float = 1.5,
        pump_ms: int = 250, log_path: str | Path | None = None, sync: bool | None = None,
        economy: bool = False, proc_max_ticks: int = 60, triage=None, speech=None,
        reflect_every: int = 150, chronicle_path: str | Path | None = None, tactician=None,
    ) -> None:
        self.tactician = tactician   # the duel's slow layer: sets memory["playbook"] at phase boundaries
        self.reflect_every = reflect_every
        self.chronicle_path = Path(chronicle_path) if chronicle_path else None
        self.aim: str | None = None
        self._reflecting = False
        self.journal_log: list[tuple[int, int, str]] = []   # (tick, speaker serial, text), trimmed
        self._reconnects_seen = 0
        self.journal_seq = 0                                 # total lines ever appended (the trim-safe cursor)
        self.triage, self.speech = triage, speech
        self.speech_log: list[tuple[int, str, str | None, str]] = []   # (tick, heard, said, reason)
        self._speech_thread: threading.Thread | None = None
        self.economy, self.proc_max_ticks = economy, proc_max_ticks
        self.profession = persona.profession or "adventurer"
        self.skill_log: list[tuple[int, dict[str, float]]] = []   # (tick, {skill: +delta})
        self._prev_obs = None
        self._proc: tuple[str, Any, int] | None = None   # (affordance id, generator, started tick)
        self.proc_log: list[tuple[int, str, str]] = []    # (tick, id, verdict)
        self.body, self.persona, self.client = body, persona, client
        self.decide_every, self.plan_ticks, self.threshold = decide_every, plan_ticks, threshold
        self.deadline_s, self.pump_ms = deadline_s, pump_ms
        self.sync = (client.name == "scripted") if sync is None else sync
        self.log_path = Path(log_path) if log_path else None
        self.memory: dict = {"economy": economy, "pump_ms": pump_ms}
        self.tick_no = 0
        self.reports: list[TickReport] = []
        self._gen = 0
        self._lock = threading.Lock()
        self._pending: tuple[int, float] | None = None          # (gen, started_at)
        self._result: tuple[int, Decision, list[str]] | None = None
        self._plan: tuple[str, int] | None = None                # (affordance id, valid until tick)
        self._last_sig: tuple | None = None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # --- decision plumbing --------------------------------------------------
    def _start(self, scene: str, options: dict[str, str]) -> None:
        self._gen += 1
        gen = self._gen
        self._pending = (gen, time.monotonic())

        def work() -> None:
            try:
                d = self.client.choose(scene, QUESTION, options)
            except Exception as e:  # noqa: BLE001 — a broken model must never stop the body
                d = Decision("", {}, 0.0, 0.0, getattr(self.client, "name", "?"), error=f"{type(e).__name__}: {e}")
            with self._lock:
                if self._pending and self._pending[0] == gen:
                    self._result = (gen, d, list(options))
                    self._pending = None

        threading.Thread(target=work, daemon=True).start()

    def _take_result(self, options: dict[str, str]) -> Decision | None:
        with self._lock:
            r, self._result = self._result, None
        if r is None:
            return None
        gen, d, keys = r
        return d if gen == self._gen and keys == list(options) else None

    def _signature(self, hostiles: int, hp_pct: float, options: dict[str, str]) -> tuple:
        return (hostiles, round(hp_pct, 1), tuple(options))

    # --- one tick --------------------------------------------------------------
    def tick(self) -> TickReport:
        self.tick_no += 1
        if self.tick_no % 20 == 1:
            self.body.act(all_names())  # names arrive asynchronously; refresh them now and then
        obs = self.body.observe()
        reconnects = getattr(self.body, "reconnects", 0)
        if self.tick_no == 1 or reconnects != self._reconnects_seen:
            # A character that has never opened its own backpack is not told what is in it:
            # `own_pack()` reads empty and every pack-dependent verb disappears (live-caught).
            # A reconnected bridge is a fresh client that has never opened it either: a mage
            # that lost its reagents this way stood frozen through whole matches (live-caught).
            self._reconnects_seen = reconnects
            if self.tick_no > 1:
                self.body.act(all_names())
            bp = obs.backpack_serial()
            if bp is not None:
                self.body.act(click_use(bp))
        for j in obs.new_journal:
            if j.text:
                self.journal_log.append((self.tick_no, j.serial, j.text))
                self.journal_seq += 1
        del self.journal_log[:-2000]
        self._hear(obs)
        self._track_target(obs)
        if self._prev_obs is not None and obs.skills:
            d = {k: v for k, v in training_delta(self._prev_obs, obs).items() if 0 < v < 5.0}  # a drop or a jump is GM staging
            if d:
                self.skill_log.append((self.tick_no, d))
        self._prev_obs = obs if obs.skills else self._prev_obs
        self._last_obs = obs
        self.memory["tick"] = self.tick_no
        hist: list = self.memory.setdefault("hp_hist", [])
        hist.append(obs.player.hp_pct)
        del hist[:-6]
        self.memory["hp_trend"] = hist[-1] - hist[0] if len(hist) > 1 else 0.0
        if obs.corpse_of:  # death links are transient; remember our kills' corpses
            self.memory.setdefault("my_corpses", set()).update(obs.corpse_of)
        self._learn_names(obs)
        f = facts(obs)
        if self.tactician is not None and not f.dead:
            self.tactician.tick(self, obs, f)   # before the menu, and even while a cast owns the tick
        # An active procedure owns the tick unless danger interrupts it.
        if self._proc is not None:
            pid, gen, started = self._proc
            limit = self.proc_max_ticks * (8 if pid.startswith("goto:") else 1)  # a long walk is legitimate
            if pid == "bandage":
                limit = max(limit, 20000 // max(50, self.pump_ms) + 1)
            black = self.memory.get("target_blacklist", {})
            live = [m for m in f.hostiles if black.get(m.serial, -1) <= self.tick_no]   # the same threats the menu sees
            danger = bool(live and live[0].distance <= 3) and not pid.startswith(("attack:", "drop:", "bandage", "cast:", "meditate"))  # chasing, unburdening and binding wounds are what danger calls for
            critical = f.hp_pct < 0.35 and not pid.startswith(("bandage", "cast:"))   # binding the wound IS the remedy for critical health
            interrupted = f.dead or danger or critical or self.tick_no - started > limit
            if not interrupted:
                rep = TickReport(self.tick_no, f.hp_pct, f.dead, len(f.hostiles), obs.player.gold, chosen=pid, reason="procedure")
                if self.speech is not None and self.reflect_every and self.tick_no % self.reflect_every == 0 and not self._reflecting:
                    self._reflect(getattr(self, "_last_scene", render(obs, f, self.persona)))
                try:
                    step = gen.send(obs)
                    if step is not None:
                        self.body.act(step)
                    self._log_proc(pid, obs, step)
                except StopIteration as done:
                    self.proc_log.append((self.tick_no, pid, str(done.value)))
                    rep.reason = f"procedure done: {done.value}"
                    self._proc = None
                    if done.value != "ok" and pid.split(":")[0] in ("sell", "craft", "smelt", "goto"):
                        # a refused or failed verb is not retried immediately; let the others run
                        self.memory.setdefault("backoff", {})[pid] = self.tick_no + 40
                self.body.pump(self.pump_ms)
                self.reports.append(rep)
                return rep
            self.proc_log.append((self.tick_no, pid, "interrupted"))
            self._proc = None
        affs = enumerate_affordances(obs, f, self.persona, self.memory)
        econ_lines: list[str] = []
        far = self.memory.pop("threat_far", False)
        if self.economy and econ_facts is not None and (not f.hostiles or far) and not f.dead:
            ef = econ_facts(obs, self.memory)
            econ = economy_affordances(obs, ef, self.memory)
            econ_lines = econ_scene(ef)
            # curriculum: among admissible work, the verb training the largest skill gap leads
            if obs.skills:
                econ.sort(key=lambda a: curriculum_key(obs, self.profession)(a.id))
                line = progress_scene(obs, self.profession)
                if line:
                    econ_lines.append(line)
            # economy verbs go ahead of wandering/hold, after survival/loot
            keep = [a for a in affs if not a.id.startswith("walk:") and a.id != "hold"]
            econ = [a for a in econ if a.id != "wait:work"] or []
            practice = train_verbs(obs, self.profession, self.memory) if obs.skills else []
            affs = keep + econ + practice or [Affordance("wait:work", "Wait at the workplace; nothing can be done right now.")]
        if not self.economy and obs.skills and not f.hostiles and not f.dead and not self.memory.get("duel"):
            practice = train_verbs(obs, self.profession, self.memory)
            if practice:   # between fights a hunter practises rather than idles
                affs = [a for a in affs if a.id != "hold" and not a.id.startswith("walk:")] + practice
        rep = TickReport(self.tick_no, f.hp_pct, f.dead, len(f.hostiles), obs.player.gold, options=[a.id for a in affs])
        if not affs:
            rep.reason = "no affordances (dead or nothing valid)"
            self.body.pump(self.pump_ms)
            self.reports.append(rep)
            return rep
        options = {a.id: a.description for a in affs}
        by_id = {a.id: a for a in affs}
        scene = render(obs, f, self.persona)
        if econ_lines:
            scene += "\n" + "\n".join(econ_lines)
        if self.aim:
            scene += f"\nYour current aim: {self.aim}"
        self._last_scene = scene
        if self.speech is not None and self.reflect_every and self.tick_no % self.reflect_every == 0 and not self._reflecting:
            self._reflect(scene)
        sig = self._signature(len(f.hostiles), f.hp_pct, options)
        changed = sig != self._last_sig
        self._last_sig = sig
        need = changed or (self.tick_no % self.decide_every == 0)

        decision: Decision | None = None
        admitted: Admitted | None = None
        if len(options) == 1:
            # Nothing to decide. Asking anyway keeps the model busy, and in async mode the next
            # real choice then falls to the rule while that useless call is still pending.
            admitted = Admitted(next(iter(options)), False, "only option")
        elif self.sync:
            if need or self._plan is None:
                decision = self.client.choose(scene, QUESTION, options)
                admitted = gate(decision, options, self.threshold)
        else:
            decision = self._take_result(options)
            if decision is not None:
                admitted = gate(decision, options, self.threshold)
            elif self._pending is not None and time.monotonic() - self._pending[1] > self.deadline_s:
                self._pending = None
                admitted = Admitted(next(iter(options)), False, "deadline")
            elif self._pending is None and need:
                self._start(scene, options)
        if changed:
            self._plan = None  # the world moved; a stale plan must not carry over

        if admitted is not None:
            self._plan = (admitted.choice, self.tick_no + self.plan_ticks)
            rep.chosen, rep.used_model, rep.reason = admitted.choice, admitted.used_model, admitted.reason
        elif self._plan is not None and self._plan[0] in by_id and self.tick_no <= self._plan[1]:
            rep.chosen, rep.reason = self._plan[0], "plan"
        else:
            rep.chosen, rep.reason = next(iter(options)), "rule" if not (self._pending and not self.sync) else "rule (deciding)"
        if decision is not None:
            rep.confidence, rep.backend, rep.ms = decision.confidence, decision.backend, decision.ms

        self._execute(by_id[rep.chosen], f)
        self.body.pump(self.pump_ms)
        self._log(rep, scene, options, decision, admitted)
        self.reports.append(rep)
        return rep

    def _track_target(self, obs) -> None:
        """A target we have been attacking, adjacent, for 40 ticks without it dying is
        not reachable (line of sight, height); blacklist it for a while."""
        eng = self.memory.get("engaged")
        if eng is None:
            return
        t = next((m for m in obs.mobiles if m.serial == eng), None)
        if t is None or t.distance > 1:
            self.memory["engaged_since"] = None
            return
        since = self.memory.get("engaged_since") or self.tick_no
        self.memory["engaged_since"] = since
        if self.tick_no - since >= 40:
            self.memory.setdefault("target_blacklist", {})[eng] = self.tick_no + 120
            self.memory["engaged"] = None
            self.memory["engaged_since"] = None
            self.proc_log.append((self.tick_no, f"attack:{eng}", "abandoned (no kill in 40 adjacent ticks)"))

    def _hear(self, obs) -> None:
        """Triage speech from nearby people into `memory['heard_pending']` (once each)."""
        if self.triage is None:
            return
        me = obs.player.serial
        near = {m.serial: m for m in obs.mobiles if m.serial != me}
        seen: set = self.memory.setdefault("heard_seen", set())
        for j in obs.new_journal:
            if j.serial not in near or not j.text or (j.serial, j.text) in seen:
                continue
            seen.add((j.serial, j.text))
            m = near[j.serial]
            friend = j.serial in self.memory.get("friends", set())
            if not addressed_to_me(j.text, obs.player.name, m.distance) and m.distance > (6 if friend else 2):
                continue   # a villager talking nearby is always worth hearing
            tr = self.triage.classify(j.text)
            self.memory.setdefault("heard_pending", []).append(
                {"serial": j.serial, "name": m.name or self.memory.get("names", {}).get(j.serial, ""), "text": j.text,
                 "kind": tr.kind, "conf": tr.confidence, "tick": self.tick_no})
        # a pending line older than 20 ticks is stale
        self.memory["heard_pending"] = [h for h in self.memory.get("heard_pending", []) if self.tick_no - h["tick"] <= 20]

    def _recent(self) -> str:
        """A compact, factual account of the last stretch for the slow layer."""
        n = self.reflect_every
        procs = [f"{pid.split(':')[0]}={v}" for t, pid, v in self.proc_log if t > self.tick_no - n]
        gains = {}
        for t, d in self.skill_log:
            if t > self.tick_no - n:
                for k, v in d.items():
                    gains[k] = round(gains.get(k, 0) + v, 1)
        said = [s_ for t, _, s_, _ in self.speech_log if t > self.tick_no - n and s_]
        parts = []
        if procs:
            import collections
            parts.append("did " + ", ".join(f"{k}×{v}" for k, v in collections.Counter(procs).most_common(6)))
        if gains:
            parts.append("skills up " + ", ".join(f"{k} +{v}" for k, v in gains.items()))
        if said:
            parts.append("said " + " / ".join(said[-2:]))
        r = self.reports[-1] if self.reports else None
        if r:
            parts.append(f"health {r.hp_pct:.0%}, gold {r.gold}")
        return "; ".join(parts) or "nothing much"

    def _reflect(self, scene: str) -> None:
        """Off-thread: a new aim for the scene, and a chronicle entry."""
        self._reflecting = True
        recent = self._recent()
        tick = self.tick_no

        def work() -> None:
            try:
                line = self.speech.aim(self.persona, scene, recent)
                if line.text:
                    self.aim = line.text
                entry = self.speech.chronicle(self.persona, scene, recent)
                if entry.text and self.chronicle_path:
                    self.chronicle_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.chronicle_path.open("a") as fh:
                        fh.write(f"## tick {tick}\n\n{entry.text}\n\n*aim:* {self.aim or '-'}  \n*facts:* {recent}\n\n")
            finally:
                self._reflecting = False

        threading.Thread(target=work, daemon=True).start()

    def _reply(self, h: dict, intent: str, scene: str) -> None:
        """Generate one line off-thread; the body says it when it arrives."""
        if self.speech is None:
            return

        def work() -> None:
            line = self.speech.say(self.persona, scene, h["text"], intent)
            self.speech_log.append((self.tick_no, h["text"], line.text, line.reason))
            if line.text:
                self.memory.setdefault("say_pending", []).append(line.text)

        self._speech_thread = threading.Thread(target=work, daemon=True)
        self._speech_thread.start()

    def _learn_names(self, obs) -> None:
        """Names arrive as journal lines answering a Click; cache them by serial and
        click one unnamed nearby mobile per tick so the scene stops saying 'a creature'."""
        names: dict[int, str] = self.memory.setdefault("names", {})
        serials = {m.serial for m in obs.mobiles}
        for j in obs.new_journal:
            if j.serial in serials and j.text and j.serial not in names:
                names[j.serial] = j.name or j.text
        for m in obs.mobiles:
            if not m.name and m.serial in names:
                m.name = names[m.serial]
        clicked: set[int] = self.memory.setdefault("clicked", set())
        for m in obs.mobiles:
            if not m.name and m.serial not in clicked and m.distance <= 12:
                clicked.add(m.serial)
                self.body.act(click(m.serial))
                break

    def _execute(self, aff: Affordance, f) -> None:
        from .contract import say as _say
        if aff.id.startswith("attack:"):
            tgt = int(aff.id.split(":")[1])
            self.memory["engaged"] = tgt
            self.memory.setdefault("attacked", set()).add(tgt)
        for text in self.memory.pop("say_pending", []):   # lines generated earlier land now
            self.body.act(_say(text))
        if aff.id.startswith(("reply:", "ignore:")):
            serial = int(aff.id.split(":")[1])
            pending = self.memory.get("heard_pending", [])
            h = next((x for x in pending if x["serial"] == serial), None)
            self.memory["heard_pending"] = [x for x in pending if x["serial"] != serial]
            last: dict = self.memory.setdefault("replied_at", {})
            if aff.id.startswith("reply:") and h is not None and self.tick_no - last.get(serial, -999) >= 30:
                last[serial] = self.tick_no
                intent = {"wary": "tell them to keep away, curtly", "answer": "answer what they asked, briefly",
                          "greet": "greet them back in your own way", "remark": "react to what they said"}[aff.id.split(":")[2]]
                self._reply(h, intent, self._last_scene)
            return
        if aff.procedure is not None:
            gen = aff.procedure(self._last_obs, self.memory)
            try:
                step = next(gen)  # the first action; the generator then waits for next tick's obs
                if step is not None:
                    self.body.act(step)
                self._proc = (aff.id, gen, self.tick_no)
                self._plan = None  # a procedure, not a plan, now owns the ticks
            except StopIteration as done:
                self.proc_log.append((self.tick_no, aff.id, str(done.value)))
            return
        for action in aff.actions:
            self.body.act(action)
        if aff.id.startswith("say:"):
            self.memory.setdefault("greeted", set()).add(int(aff.id.split(":")[1]))
        elif aff.id.startswith("attack:"):
            self.memory["engaged"] = int(aff.id.split(":")[1])
            self.memory.setdefault("attacked", set()).add(int(aff.id.split(":")[1]))
        elif aff.id == "cast:magic_reflection":
            self.memory["reflect_at"] = self.tick_no

    def _log_proc(self, pid: str, obs, step) -> None:
        if not self.log_path:
            return
        eng = self.memory.get("engaged")
        tgt = next((m for m in obs.mobiles if m.serial == eng), None) if eng else None
        row = {"tick": self.tick_no, "proc": pid, "step": step,
               "me": (obs.player.pos.x, obs.player.pos.y, obs.player.pos.z),
               "target": None if tgt is None else (tgt.serial, tgt.pos.x, tgt.pos.y, tgt.pos.z, tgt.distance),
               "journal": [(j.cliloc, j.text[:80]) for j in obs.new_journal if j.text or j.cliloc],
               "gumps": [(g.serial, g.gump_id, sorted(set(re.findall(r"[0-9]{7}", g.layout)))[:12]) for g in obs.gumps],
               "cursor": obs.pending_target, "popup": None if obs.popup is None else len(obs.popup.entries),
               "shop_sell": None if obs.shop_sell is None else len(obs.shop_sell.items)}
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _log(self, rep: TickReport, scene: str, options: dict[str, str], decision: Decision | None, admitted: Admitted | None) -> None:
        if not self.log_path:
            return
        row = {"tick": rep.tick, "scene": scene, "options": options, "chosen": rep.chosen, "reason": rep.reason,
               "used_model": rep.used_model, "hp_pct": round(rep.hp_pct, 3), "hostiles": rep.hostiles, "gold": rep.gold,
               "decision": None if decision is None else {"choice": decision.choice, "probs": decision.probs,
                                                          "confidence": decision.confidence, "ms": decision.ms,
                                                          "backend": decision.backend, "error": decision.error}}
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run(self, ticks: int, on_tick=None) -> list[TickReport]:
        for _ in range(ticks):
            rep = self.tick()
            if on_tick:
                on_tick(rep)
        return self.reports

    def skill_gains(self) -> dict[str, float]:
        total: dict[str, float] = {}
        for _, d in self.skill_log:
            for k, v in d.items():
                total[k] = round(total.get(k, 0.0) + v, 1)
        return total

    def summary(self) -> dict:
        r = self.reports
        decided = [x for x in r if x.confidence is not None]
        reasons: dict[str, int] = {}
        for x in r:
            reasons[x.reason] = reasons.get(x.reason, 0) + 1
        return {"ticks": len(r), "model_calls": len(decided), "model_admitted": sum(x.used_model for x in r),
                "dead": bool(r and r[-1].dead), "gold": r[-1].gold if r else 0,
                "min_hp_pct": round(min((x.hp_pct for x in r), default=1.0), 2),
                "avg_decision_ms": round(sum(x.ms for x in decided) / len(decided), 0) if decided else None,
                "procedures": [f"{t}:{pid}={v}" for t, pid, v in self.proc_log][-40:],
                "skill_gains": self.skill_gains(),
                "speech": [(t, h[:40], (s_ or "")[:60], r) for t, h, s_, r in self.speech_log][-10:],
                "aim": self.aim,
                "gm": gm_count(self._last_obs, self.profession) if self._last_obs is not None and self._last_obs.skills else None,
                "reasons": reasons}
