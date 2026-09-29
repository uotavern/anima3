"""Typed views over the anima-client Observation JSON, and Action JSON builders.

Field names mirror `anima-contract-json` (schema 33) verbatim. Only what this
brain reads is modelled; unknown keys are ignored so additive schema bumps are
harmless. Actions are plain dicts of exactly the shape `action_from_json` parses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 33

#: UO direction numbering, as `Walk{dir}` expects. Index -> (dx, dy).
DIRECTION_DELTAS: list[tuple[int, int]] = [
    (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
]
DIRECTION_NAMES = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]

#: Notoriety values a character may lawfully attack (gray/criminal/enemy/murderer).
HOSTILE_NOTORIETY = frozenset({3, 4, 5, 6})
INNOCENT_NOTORIETY = frozenset({1, 2})
#: Human/elf body graphics — "a person", as opposed to a creature.
HUMAN_BODIES = frozenset({0x0190, 0x0191, 0x025D, 0x025E})
#: Item graphics this brain recognises on the ground / in the pack.
GOLD_GRAPHIC = 0x0EED
BANDAGE_GRAPHIC = 0x0E21
ORE_GRAPHICS = frozenset({0x19B7, 0x19B8, 0x19B9, 0x19BA})
BACKPACK_LAYER = 0x15


@dataclass(frozen=True)
class Pos:
    x: int = 0
    y: int = 0
    z: int = 0

    @classmethod
    def from_json(cls, d: Any) -> Pos:
        d = d or {}
        return cls(int(d.get("x", 0)), int(d.get("y", 0)), int(d.get("z", 0)))


def chebyshev(a: Pos, b: Pos) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


def direction_toward(frm: Pos, to: Pos) -> int:
    dx = (to.x > frm.x) - (to.x < frm.x)
    dy = (to.y > frm.y) - (to.y < frm.y)
    if (dx, dy) == (0, 0):
        return 0
    return DIRECTION_DELTAS.index((dx, dy))


@dataclass
class Player:
    serial: int = 0
    name: str = ""
    pos: Pos = field(default_factory=Pos)
    direction: int = 0
    hits: int = 0
    hits_max: int = 0
    mana: int = 0
    mana_max: int = 0
    stam: int = 0
    stam_max: int = 0
    gold: int = 0
    weight: int = 0
    weight_max: int = 0
    poisoned: bool = False
    dead: bool = False
    # Our own condition (schema 33): a looter that went grey reads notoriety 3.
    hidden: bool = False
    paralyzed: bool = False
    notoriety: int = 1
    mounted: bool = False

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Player:
        return cls(
            serial=int(d.get("serial", 0)), name=str(d.get("name", "")),
            pos=Pos.from_json(d.get("pos")), direction=int(d.get("direction", 0)),
            hits=int(d.get("hits", 0)), hits_max=int(d.get("hits_max", 0)),
            mana=int(d.get("mana", 0)), mana_max=int(d.get("mana_max", 0)),
            stam=int(d.get("stam", 0)), stam_max=int(d.get("stam_max", 0)),
            gold=int(d.get("gold", 0)), weight=int(d.get("weight", 0)),
            weight_max=int(d.get("weight_max", 0)),
            poisoned=bool(d.get("poisoned", False)), dead=bool(d.get("dead", False)),
            hidden=bool(d.get("hidden", False)), paralyzed=bool(d.get("paralyzed", False)),
            notoriety=int(d.get("notoriety", 1)), mounted=bool(d.get("mounted", False)),
        )

    @property
    def hp_pct(self) -> float:
        return self.hits / self.hits_max if self.hits_max else 1.0


@dataclass
class Mobile:
    serial: int
    name: str
    pos: Pos
    body: int
    notoriety: int
    hits: int
    hits_max: int
    distance: int
    # What any client shows of another mobile's condition (schema 32).
    poisoned: bool = False
    paralyzed: bool = False
    war_mode: bool = False
    hidden: bool = False

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Mobile:
        return cls(
            serial=int(d.get("serial", 0)), name=str(d.get("name", "")),
            pos=Pos.from_json(d.get("pos")), body=int(d.get("body", 0)),
            notoriety=int(d.get("notoriety", 0)), hits=int(d.get("hits", 0)),
            hits_max=int(d.get("hits_max", 0)), distance=int(d.get("distance", 0)),
            poisoned=bool(d.get("poisoned", False)), paralyzed=bool(d.get("paralyzed", False)),
            war_mode=bool(d.get("war_mode", False)), hidden=bool(d.get("hidden", False)),
        )

    @property
    def person(self) -> bool:
        return self.body in HUMAN_BODIES

    #: Set by `facts()` from the player's z: the core's `distance` is x/y only, and a
    #: mongbat twenty tiles down the cliff face reads as "adjacent" (live-caught).
    dz: int = 0

    @property
    def reachable(self) -> bool:
        return abs(self.dz) <= 5

    @property
    def hostile(self) -> bool:
        """A creature that may lawfully be attacked; a *person* only when criminal/enemy/murderer.
        A gray (3) human is another player or staff — someone to be wary of, not a monster."""
        if self.person:
            return self.notoriety in (4, 5, 6)
        return self.notoriety in HOSTILE_NOTORIETY


@dataclass
class Item:
    serial: int
    graphic: int
    amount: int
    pos: Pos
    container: int | None
    layer: int
    distance: int
    hue: int = 0        # schema 33: ore, ingot, leather and board kinds differ by hue alone
    name: str = ""

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Item:
        c = d.get("container")
        return cls(
            serial=int(d.get("serial", 0)), graphic=int(d.get("graphic", 0)),
            amount=int(d.get("amount", 1)), pos=Pos.from_json(d.get("pos")),
            container=None if c is None else int(c), layer=int(d.get("layer", 0)),
            distance=int(d.get("distance", 0)), hue=int(d.get("hue", 0)), name=str(d.get("name") or ""),
        )


@dataclass
class Journal:
    serial: int
    name: str
    text: str
    msg_type: int
    hue: int
    cliloc: int

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Journal:
        return cls(
            serial=int(d.get("serial", 0)), name=str(d.get("name", "")),
            text=str(d.get("text", "")), msg_type=int(d.get("msg_type", 0)),
            hue=int(d.get("hue", 0)), cliloc=int(d.get("cliloc", 0) or 0),
        )


@dataclass
class Skill:
    id: int
    value: float     # displayed (with bonuses), tenths already divided
    base: float      # trained base
    cap: float
    lock: int        # 0 up, 1 down, 2 locked

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Skill:
        def tenths(k: str) -> float:
            v = d.get(k, 0) or 0
            return float(v) / 10.0 if isinstance(v, int) and v > 200 else float(v)
        return cls(int(d.get("id", 0)), tenths("value"), tenths("base"), tenths("cap"), int(d.get("lock", 0)))


@dataclass
class Terrain:
    """Walkability window: `walk` is `side*side` chars, row-major from `origin`
    (top-left), '.' walkable / anything else blocked."""

    origin: tuple[int, int]
    side: int
    walk: str

    @classmethod
    def from_json(cls, d: dict[str, Any] | None) -> Terrain | None:
        if not d:
            return None
        o = d.get("origin") or [0, 0]
        return cls(origin=(int(o[0]), int(o[1])), side=int(d.get("side", 0)), walk=str(d.get("walk", "")))

    def walkable(self, x: int, y: int) -> bool | None:
        cx, cy = x - self.origin[0], y - self.origin[1]
        if not (0 <= cx < self.side and 0 <= cy < self.side):
            return None
        i = cy * self.side + cx
        return self.walk[i] == "." if i < len(self.walk) else None


@dataclass
class Gump:
    serial: int
    gump_id: int
    layout: str

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Gump:
        return cls(int(d.get("serial", 0)), int(d.get("gump_id", 0)), str(d.get("layout", "")))

    def has_cliloc(self, cliloc: int) -> bool:
        return str(cliloc) in self.layout


@dataclass
class Popup:
    serial: int
    entries: list[dict[str, Any]]

    def index_of(self, cliloc: int, text: str | None = None) -> int | None:
        for i, e in enumerate(self.entries):
            if int(e.get("cliloc", -1)) == cliloc or (text and str(e.get("text", "")).lower() == text.lower()):
                return int(e.get("index", i))
        return None


@dataclass
class ShopSellItem:
    serial: int
    graphic: int
    amount: int
    price: int
    name: str


@dataclass
class ShopSell:
    vendor: int
    items: list[ShopSellItem]

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ShopSell:
        return cls(int(d.get("vendor", 0)), [ShopSellItem(int(i.get("serial", 0)), int(i.get("graphic", 0)),
                                                          int(i.get("amount", 1)), int(i.get("price", 0)),
                                                          str(i.get("name", ""))) for i in d.get("items") or []])


@dataclass
class Observation:
    player: Player = field(default_factory=Player)
    mobiles: list[Mobile] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    new_journal: list[Journal] = field(default_factory=list)
    pending_target: bool = False
    target_cursor_id: int | None = None
    target_cursor_flag: int | None = None
    target_cursor_type: int | None = None
    war: bool = False
    terrain: Terrain | None = None
    gumps: list[Gump] = field(default_factory=list)
    popup: Popup | None = None
    shop_sell: ShopSell | None = None
    corpse_of: dict[int, int] = field(default_factory=dict)   # corpse serial -> killed mobile serial
    skills: list[Skill] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Observation:
        pu = d.get("popup")
        ss = d.get("shop_sell")
        return cls(
            player=Player.from_json(d.get("player") or {}),
            mobiles=[Mobile.from_json(m) for m in d.get("mobiles") or []],
            items=[Item.from_json(i) for i in d.get("items") or []],
            new_journal=[Journal.from_json(j) for j in d.get("new_journal") or []],
            pending_target=bool(d.get("pending_target")),
            target_cursor_id=(d["pending_target"].get("cursor_id") if isinstance(d.get("pending_target"), dict) else None),
            target_cursor_flag=(d["pending_target"].get("cursor_flag") if isinstance(d.get("pending_target"), dict) else None),
            target_cursor_type=(d["pending_target"].get("target_type") if isinstance(d.get("pending_target"), dict) else None),
            war=bool(d.get("war", False)),
            terrain=Terrain.from_json(d.get("terrain")),
            gumps=[Gump.from_json(g) for g in d.get("gumps") or []],
            popup=Popup(int(pu.get("serial", 0)), list(pu.get("entries") or [])) if pu else None,
            shop_sell=ShopSell.from_json(ss) if ss else None,
            corpse_of={int(x.get("corpse", 0)): int(x.get("killed", 0)) for x in d.get("corpse_of") or []},
            skills=[Skill.from_json(x) for x in d.get("skills") or []],
        )

    # --- convenience views -------------------------------------------------
    def backpack_serial(self) -> int | None:
        """The player's own backpack: the item worn on layer 0x15 by the player."""
        bp = next((i for i in self.items if i.layer == BACKPACK_LAYER and i.container == self.player.serial), None)
        return bp.serial if bp else None

    def own_pack(self) -> list[Item]:
        """Items inside the player's backpack only — never a vendor's stock or worn gear."""
        bp = self.backpack_serial()
        return [i for i in self.items if bp is not None and i.container == bp]

    def in_pack(self, graphic: int) -> list[Item]:
        return [i for i in self.own_pack() if i.graphic == graphic]

    def on_ground(self) -> list[Item]:
        return [i for i in self.items if i.container is None]


# --- Action builders (exact `action_from_json` shapes) ----------------------
def walk(direction: int, run: bool = False) -> dict:
    return {"type": "Walk", "dir": int(direction), "run": bool(run)}


def walk_to(x: int, y: int) -> dict:
    return {"type": "WalkTo", "x": int(x), "y": int(y)}


def say(text: str) -> dict:
    return {"type": "Say", "text": text}


def attack(serial: int) -> dict:
    return {"type": "Attack", "serial": int(serial)}


def use(serial: int) -> dict:
    return {"type": "Use", "serial": int(serial)}


def pick_up(serial: int, amount: int = 1) -> dict:
    return {"type": "PickUp", "serial": int(serial), "amount": int(amount)}


def war_mode(on: bool) -> dict:
    return {"type": "WarMode", "on": bool(on)}


def bandage_target(bandage: int, target: int) -> dict:
    return {"type": "BandageTarget", "bandage": int(bandage), "target": int(target)}


def open_door() -> dict:
    return {"type": "OpenDoor"}


def target_object(serial: int) -> dict:
    return {"type": "TargetObject", "serial": int(serial)}


def target_ground(x: int, y: int, z: int, graphic: int = 0) -> dict:
    return {"type": "TargetGround", "x": int(x), "y": int(y), "z": int(z), "graphic": int(graphic)}


def all_names() -> dict:
    """Ask the server for the names of everything on screen (ClassicUO 'AllNames')."""
    return {"type": "AllNames"}


def click(serial: int) -> dict:
    """Single-click: the server answers with the object's name as a journal line."""
    return {"type": "Click", "serial": int(serial)}


def gump_response(serial: int, gump_id: int, button: int) -> dict:
    return {"type": "GumpResponse", "serial": int(serial), "gump_id": int(gump_id), "button": int(button)}


def popup_request(serial: int) -> dict:
    return {"type": "PopupRequest", "serial": int(serial)}


def popup_select(serial: int, index: int) -> dict:
    return {"type": "PopupSelect", "serial": int(serial), "index": int(index)}


def sell_items(vendor: int, items: list[tuple[int, int]]) -> dict:
    return {"type": "SellItems", "vendor": int(vendor), "items": [{"serial": int(s), "amount": int(a)} for s, a in items]}


def equip(serial: int, layer: int = 1) -> dict:
    """ServUO's EquipReq uses the item's own layer; `layer` is only what the packet carries."""
    return {"type": "Equip", "serial": int(serial), "layer": int(layer)}


def drop(serial: int, container: int, x: int = 0, y: int = 0, z: int = 0) -> dict:
    """Finish a lift: put the held item into `container` (a backpack, a bank box)."""
    return {"type": "Drop", "serial": int(serial), "x": int(x), "y": int(y), "z": int(z), "container": int(container)}


def use_skill(skill: int) -> dict:
    """Invoke a skill by id (the skill-button macro); targeted skills then open a cursor."""
    return {"type": "UseSkill", "skill": int(skill)}
