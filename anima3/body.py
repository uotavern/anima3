"""Bodies: the thing that turns actions into packets and packets into Observations.

`BridgeBody` drives anima-client's `anima-agent` NDJSON bridge (built here as
`anima-bridge` to dodge the bin-name collision with the in-process runner).
`FakeBody` is a tiny deterministic world for offline runs and tests — enough
physics to exercise perception, movement, combat and pickup; not a UO simulator.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .contract import (
    BANDAGE_GRAPHIC,
    DIRECTION_DELTAS,
    GOLD_GRAPHIC,
    SCHEMA_VERSION,
    Item,
    Journal,
    Mobile,
    Observation,
    Player,
    Pos,
    Terrain,
    chebyshev,
    direction_toward,
)


class BodyError(RuntimeError):
    pass


class Body(Protocol):
    def observe(self) -> Observation: ...
    def act(self, action: dict) -> None: ...
    def pump(self, ms: int) -> int: ...
    def close(self) -> None: ...


#: The shard. The hosted one by default; `ANIMA3_HOST=127.0.0.1` for a local ServUO.
DEFAULT_HOST = os.environ.get("ANIMA3_HOST", "uo.hulryung.com")
DEFAULT_PORT = int(os.environ.get("ANIMA3_PORT", "2593"))
DEFAULT_BRIDGE = Path.home() / "dev" / "uo" / "anima-client" / "target" / "release" / "anima-bridge"
DEFAULT_DATA_DIR = Path.home() / "dev" / "uo" / "uo-resource"
_MONITOR_RE = re.compile(r"monitor on (http://[^\s]+)")


class BridgeBody:
    """One bridge subprocess == one logged-in character."""

    def __init__(self, proc: subprocess.Popen, ready: dict[str, Any], terrain_radius: int) -> None:
        self._proc = proc
        self.ready = ready
        self.terrain_radius = terrain_radius
        self.monitor_url: str | None = None
        self.diagnostics: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    @classmethod
    def spawn(
        cls, host: str, port: int, user: str, password: str, *,
        binary: str | os.PathLike | None = None, data_dir: str | os.PathLike | None = None,
        monitor_port: int | None = None, terrain_radius: int = 12,
    ) -> BridgeBody:
        exe = Path(binary or DEFAULT_BRIDGE)
        if not exe.exists():
            raise BodyError(f"bridge not built: {exe} — run `cargo build --release -p anima-net` in anima-client")
        env = dict(os.environ)
        if monitor_port is not None:
            env["ANIMA_MONITOR_PORT"] = str(monitor_port)
        proc = subprocess.Popen(
            [str(exe), host, str(port), user, password, str(data_dir or DEFAULT_DATA_DIR)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, bufsize=1,
        )
        first = proc.stdout.readline()  # type: ignore[union-attr]
        if not first:
            err = proc.stderr.read() if proc.stderr else ""
            raise BodyError(f"bridge exited before ready: {err.strip()[-400:]}")
        ready = json.loads(first)
        if ready.get("event") != "ready":
            raise BodyError(f"unexpected first line: {first.strip()[:200]}")
        if ready.get("schema_version") != SCHEMA_VERSION:
            proc.kill()
            raise BodyError(f"unsupported bridge schema {ready.get('schema_version')}; expected {SCHEMA_VERSION}")
        return cls(proc, ready, terrain_radius)

    def _drain_stderr(self) -> None:
        for line in self._proc.stderr:  # type: ignore[union-attr]
            m = _MONITOR_RE.search(line)
            if m:
                self.monitor_url = m.group(1)
            sys.stderr.write(line)

    def _rpc(self, obj: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            assert self._proc.stdin and self._proc.stdout
            self._proc.stdin.write(json.dumps(obj) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        if not line:
            raise BodyError("bridge closed the pipe")
        response = json.loads(line)
        # Optional transport counters from newer bridges, without packet payloads.
        # Preserve compatibility with bridges predating diagnostics.
        if isinstance(response.get("diagnostics"), dict):
            self.diagnostics = response["diagnostics"]
        return response

    def observe_raw(self) -> dict[str, Any]:
        r = self._rpc({"cmd": "observe", "terrain_radius": self.terrain_radius})
        if not r.get("ok"):
            raise BodyError(r.get("error", "observe failed"))
        return r["obs"]

    def observe(self) -> Observation:
        return Observation.from_json(self.observe_raw())

    def act(self, action: dict) -> None:
        r = self._rpc({"cmd": "act", "action": action})
        if not r.get("ok"):
            raise BodyError(f"act {action.get('type')}: {r.get('error')}")

    def pump(self, ms: int) -> int:
        r = self._rpc({"cmd": "pump", "ms": int(ms)})
        if not r.get("ok"):
            raise BodyError(r.get("error", "pump failed"))
        return int(r.get("applied", 0))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._rpc({"cmd": "quit"})
        try:
            self._proc.wait(timeout=3)
        except Exception:  # noqa: BLE001
            self._proc.kill()


class ResilientBody:
    """A `BridgeBody` that reconnects when its subprocess dies.

    A long unattended run loses bridges: the machine sleeps, the shard restarts, a
    pipe breaks. Every such failure used to end the run with `io error: Broken pipe`
    mid-command. This retries the spawn a few times and re-issues the call; the
    character reappears where the server left it, which is all the brain needs.
    """

    def __init__(self, spawn_args: dict, max_retries: int = 5, backoff_s: float = 3.0) -> None:
        self._args = spawn_args
        self._retries = max_retries
        self._backoff = backoff_s
        self.reconnects = 0
        for attempt in range(max_retries):
            try:
                self._body = BridgeBody.spawn(**spawn_args)
                break
            except BodyError as e:
                # A character whose previous session just ended is still held by the shard for
                # a few seconds and the login is refused; the first spawn waits it out too.
                if attempt == max_retries - 1:
                    raise
                sys.stderr.write(f"[body] login {attempt + 1}/{max_retries} as {spawn_args.get('user')} failed: {e}\n")
                import time as _t
                _t.sleep(backoff_s * (attempt + 1))

    @property
    def ready(self) -> dict[str, Any]:
        return self._body.ready

    @property
    def monitor_url(self) -> str | None:
        return self._body.monitor_url

    def _reconnect(self) -> None:
        import time as _t
        for attempt in range(self._retries):
            _t.sleep(self._backoff * (attempt + 1))
            try:
                with contextlib.suppress(Exception):
                    self._body.close()
                self._body = BridgeBody.spawn(**self._args)
                self.reconnects += 1
                sys.stderr.write(f"[body] reconnected as {self._args.get('user')} (#{self.reconnects})\n")
                return
            except Exception as e:  # noqa: BLE001 — keep trying while the shard settles
                sys.stderr.write(f"[body] reconnect {attempt + 1}/{self._retries} failed: {e}\n")
        raise BodyError(f"could not reconnect as {self._args.get('user')} after {self._retries} tries")

    def _call(self, name: str, *a, **kw):
        try:
            return getattr(self._body, name)(*a, **kw)
        except BodyError:
            self._reconnect()
            return getattr(self._body, name)(*a, **kw)

    def observe(self) -> Observation:
        return self._call("observe")

    def observe_raw(self) -> dict[str, Any]:
        return self._call("observe_raw")

    def act(self, action: dict) -> None:
        self._call("act", action)

    def pump(self, ms: int) -> int:
        return self._call("pump", ms)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._body.close()


# --- Offline world ------------------------------------------------------------
@dataclass
class FakeMobile:
    serial: int
    name: str
    pos: Pos
    body: int = 0x0027  # mongbat
    notoriety: int = 3
    hits: int = 12
    hits_max: int = 12
    damage: int = 3          # per pump when adjacent and hostile
    aggressive: bool = True


@dataclass
class FakeBody:
    """Deterministic pocket world. One tick == `pump(ms)`."""

    player: Player = field(default_factory=lambda: Player(
        serial=0x1, name="Grimm", pos=Pos(100, 100, 0), hits=50, hits_max=50,
        mana=10, mana_max=10, stam=30, stam_max=30, gold=0, weight=20, weight_max=200))
    mobiles: list[FakeMobile] = field(default_factory=list)
    ground: list[Item] = field(default_factory=list)
    pack: list[Item] = field(default_factory=list)
    journal: list[Journal] = field(default_factory=list)
    war: bool = False
    blocked: set[tuple[int, int]] = field(default_factory=set)
    log: list[dict] = field(default_factory=list)
    worn: list[Item] = field(default_factory=list)
    held: Item | None = None
    route: tuple[int, int] | None = None
    _serial_next: int = 0x4000_0000
    combat_target: int | None = None

    # --- world building helpers ---
    def add_hostile(self, dx: int, dy: int, **kw) -> FakeMobile:
        self._serial_next += 1
        m = FakeMobile(self._serial_next, kw.pop("name", "a mongbat"),
                       Pos(self.player.pos.x + dx, self.player.pos.y + dy), **kw)
        self.mobiles.append(m)
        return m

    def add_person(self, dx: int, dy: int, name: str = "Sara") -> FakeMobile:
        self._serial_next += 1
        m = FakeMobile(self._serial_next, name, Pos(self.player.pos.x + dx, self.player.pos.y + dy),
                       body=0x0191, notoriety=1, hits=40, hits_max=40, damage=0, aggressive=False)
        self.mobiles.append(m)
        return m

    def hear(self, speaker: FakeMobile, text: str) -> None:
        self.journal.append(Journal(speaker.serial, speaker.name, text, 0, 0, 0))

    def add_ground_item(self, graphic: int, dx: int, dy: int, amount: int = 1) -> Item:
        self._serial_next += 1
        it = Item(self._serial_next, graphic, amount, Pos(self.player.pos.x + dx, self.player.pos.y + dy), None, 0, 0)
        self.ground.append(it)
        return it

    def add_pack_item(self, graphic: int, amount: int = 1) -> Item:
        self._serial_next += 1
        it = Item(self._serial_next, graphic, amount, Pos(), 0x4000_0001, 0x15, 0)
        self.pack.append(it)
        return it

    # --- Body protocol ---
    def observe(self) -> Observation:
        p = self.player
        mobs = [Mobile(m.serial, m.name, m.pos, m.body, m.notoriety, m.hits, m.hits_max, chebyshev(p.pos, m.pos))
                for m in self.mobiles if m.hits > 0]
        items = [Item(i.serial, i.graphic, i.amount, i.pos, None, 0, chebyshev(p.pos, i.pos)) for i in self.ground]
        items += [Item(0x4000_0001, 0x0E75, 1, Pos(), p.serial, 0x15, 0)]  # the backpack itself
        items += [Item(i.serial, i.graphic, i.amount, Pos(), p.serial, i.layer, 0) for i in self.worn]
        items += [Item(i.serial, i.graphic, i.amount, i.pos, i.container, i.layer, 0) for i in self.pack]
        mobs.sort(key=lambda m: m.distance)
        items.sort(key=lambda i: i.distance)
        j, self.journal = self.journal, []
        # 9x9 walk window
        side, r = 9, 4
        ox, oy = p.pos.x - r, p.pos.y - r
        walk = "".join("#" if (ox + cx, oy + cy) in self.blocked else "." for cy in range(side) for cx in range(side))
        return Observation(player=Player(**p.__dict__), mobiles=mobs, items=items, new_journal=j,
                           war=self.war, terrain=Terrain((ox, oy), side, walk))

    def act(self, action: dict) -> None:
        self.log.append(action)
        t = action["type"]
        p = self.player
        if p.dead and t != "Say":
            return
        if t == "Walk":
            dx, dy = DIRECTION_DELTAS[action["dir"] % 8]
            nx, ny = p.pos.x + dx, p.pos.y + dy
            if (nx, ny) not in self.blocked and not any(m.pos == Pos(nx, ny, 0) and m.hits > 0 for m in self.mobiles):
                p.pos = Pos(nx, ny, 0)
        elif t == "WalkTo":
            self.route = (int(action["x"]), int(action["y"]))
        elif t == "WarMode":
            self.war = bool(action["on"])
        elif t == "Attack":
            self.combat_target = int(action["serial"])
        elif t == "PickUp":  # lift onto the cursor; a Drop finishes the move
            for it in list(self.ground) + list(self.pack):
                if it.serial == action["serial"] and (it in self.pack or chebyshev(p.pos, it.pos) <= 2):
                    (self.ground if it in self.ground else self.pack).remove(it)
                    self.held = it
        elif t == "Drop" and self.held is not None and action.get("container") == 0xFFFFFFFF:
            it = self.held; self.held = None
            self.ground.append(Item(it.serial, it.graphic, it.amount, Pos(action["x"], action["y"], 0), None, 0, 0))
            p.weight = max(0, p.weight - 6 * it.amount)
        elif t == "Drop":
            it = self.held
            if it is not None and it.serial == action["serial"] and action.get("container") == 0x4000_0001:
                self.held = None
                if it.graphic == GOLD_GRAPHIC:
                    p.gold += it.amount
                else:
                    self.pack.append(Item(it.serial, it.graphic, it.amount, Pos(), 0x4000_0001, 0x15, 0))
        elif t == "BandageTarget":
            for it in list(self.pack):
                if it.serial == action["bandage"] and it.graphic == BANDAGE_GRAPHIC:
                    it.amount -= 1
                    if it.amount <= 0:
                        self.pack.remove(it)
                    p.hits = min(p.hits_max, p.hits + 15)
                    self.journal.append(Journal(0, "", "You finish applying the bandages.", 0, 0, 500969))
                    break
        elif t == "Say":
            self.journal.append(Journal(p.serial, p.name, action["text"], 0, 0, 0))
        elif t == "Equip":
            it = self.held
            if it is not None and it.serial == action["serial"]:
                self.held = None
                self.worn.append(Item(it.serial, it.graphic, it.amount, Pos(), p.serial, int(action.get("layer", 1)), 0))
        elif t == "UseSkill":
            self.journal.append(Journal(0, "", "You begin practising.", 0, 0, 0))
        elif t == "AllNames":
            pass
        elif t == "Click":
            for m in self.mobiles:
                if m.serial == action["serial"]:
                    self.journal.append(Journal(m.serial, m.name, m.name, 0, 0, 0))

    def pump(self, ms: int) -> int:
        if ms > 0:
            time.sleep(ms / 1000)  # offline time passes at the same cadence as live
        p = self.player
        if p.dead:
            return 0
        if self.route is not None:  # one A* step per pump, greedy is fine offline
            tx, ty = self.route
            if (tx, ty) == (p.pos.x, p.pos.y):
                self.route = None
            else:
                d = direction_toward(p.pos, Pos(tx, ty, 0))
                dx, dy = DIRECTION_DELTAS[d]
                if (p.pos.x + dx, p.pos.y + dy) not in self.blocked:
                    p.pos = Pos(p.pos.x + dx, p.pos.y + dy, 0)
        # our swing
        if self.war and self.combat_target is not None:
            for m in self.mobiles:
                if m.serial == self.combat_target and m.hits > 0 and chebyshev(p.pos, m.pos) <= 1:
                    m.hits -= 4
                    if m.hits <= 0:
                        self.journal.append(Journal(0, "", f"You have slain {m.name}.", 0, 0, 0))
                        self.add_ground_item(GOLD_GRAPHIC, m.pos.x - p.pos.x, m.pos.y - p.pos.y, 25)
        # their swing / approach
        for m in self.mobiles:
            if m.hits <= 0 or not m.aggressive:
                continue
            d = chebyshev(p.pos, m.pos)
            if d <= 1:
                p.hits -= m.damage
                if p.hits <= 0:
                    p.hits, p.dead = 0, True
                    self.journal.append(Journal(0, "", "You are dead.", 0, 0, 0))
            elif d <= 8:
                dx, dy = DIRECTION_DELTAS[direction_toward(m.pos, p.pos)]
                m.pos = Pos(m.pos.x + dx, m.pos.y + dy, 0)
        return 1

    def close(self) -> None:
        pass
