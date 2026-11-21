"""Simulator drills and the simulator's code identity; stdlib only.

``DRILLS`` and ``setup_drill`` are ported unchanged (AST-identical) from
``neural/curriculum.py`` at the training tip ``819f4f4`` (``94051a2``). Unit 8 moved
the teacher from that file to ``teacher.py``; the drill helper needs the simulator
and lives here. A drill only sets isolated initial simulator state.

``SIMULATOR_SHA256`` identifies the simulator's behaviour, not just one file: it is
the SHA-256 of the canonical JSON map ``SIMULATOR_SOURCES`` from each module the
simulated game depends on to that module's SHA-256 (``sim.py``, this file and
``schema.py``, which supplies the masks, costs and encoding). Every simulator
rollout and simulator evaluation records it, and a simulator evaluation plan made
with another identity is refused, so an edit to the simulator cannot silently
reuse an older plan or result.
"""
from __future__ import annotations

from .teacher import source_hashes, source_identity

# Every simulator result carries this domain. It is never "servuo".
SIM_DOMAIN = "approximate-pre-aos-simulator"
DRILLS = ("melee", "survival", "burst", "full")
SIMULATOR_MODULES = ("neural/sim.py", "neural/drills.py", "neural/schema.py")
SIMULATOR_SOURCES = source_hashes(SIMULATOR_MODULES)
SIMULATOR_SHA256 = source_identity(SIMULATOR_SOURCES)


def setup_drill(sim, name, side):
    """Set only isolated simulator initial state and reset potential afterwards."""
    if name not in DRILLS:
        raise ValueError(name)
    own, other = sim.fighters[side], sim.fighters[1-side]
    if name != "full":
        distance = sim.rng.choice((1, 2, 3)) if name == "melee" else sim.rng.randint(3, 7)
        own.x, own.y, other.x, other.y = 0, 0, distance, 0
        own.mana = sim.rng.uniform(45, 100)
        if name == "survival":
            own.hp = sim.rng.uniform(25, 60)
            if sim.rng.random() < 0.5:
                own.poison_until = 20
                own.poison_generation += 1
                sim._schedule(3, "poison_tick", 1-side, (side, own.poison_generation))
        elif name == "burst":
            own.mana = 100
            other.hp = sim.rng.uniform(45, 90)
        elif name == "melee":
            own.stamina = sim.rng.uniform(5, 25)
    sim._potential_previous = sim._potential()
    return sim.frames()
