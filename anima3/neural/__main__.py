"""One entry point for simulation training, live collection and verified updates."""

from __future__ import annotations

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: python -m anima3.neural {train|evaluate|live|wait|update|cycle} [options]")
        print("train: recurrent PPO simulator training; evaluate: held-out simulation")
        print("live: ordinary ServUO clients; update: verified real-episode PPO")
        print("wait: prepare and accept human-friendly 7x + explosion potion challenges")
        print("cycle: bounded live collection -> verified update loop (no promotion)")
        return 0
    command, rest = argv[0], argv[1:]
    if command in {"train", "evaluate"}:
        from . import train

        return train.main((["--evaluate-only"] if command == "evaluate" else []) + rest)
    if command == "live":
        from .live import main as run
    elif command == "wait":
        from .wait import main as run
    elif command == "update":
        from .online import update_main as run
    elif command == "cycle":
        from .online import cycle_main as run
    else:
        raise SystemExit(f"unknown neural command: {command}")
    return run(rest)


if __name__ == "__main__":
    raise SystemExit(main())
