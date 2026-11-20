# Live-trained continuation candidate

Four verified, unranked 7x + explosion-potion matches, with challenger order
alternating and clients reconnecting between matches. The parent is the frozen
[`duel-continuation`](../duel-continuation/README.md) checkpoint.

| Match | Result | Recorded transitions | Next policy SHA prefix |
|---|---|---:|---|
| `af9e520e0cbd4bffa6f7b00d8d7d0914` | Loss | 133 | `5a9afe696cea` |
| `d1cdce16976848d0b7b3ca3dac9b3b59` | Win | 50 | `f5e9c7e1ff53` |
| `0bd5a94b6934449986938bfd2cbc9bf0` | Win | 204 | `0698f95bbd31` |
| `27e442010d144d16959852721e6d71d6` | Win | 233 | `fe5796b401fe` |

Each update used four PPO epochs; the next match used the newly saved weights.
The 620 transitions include 86 free choices and forced casting/waiting frames.
Actor optimization used the free choices; the critic retained all frames.

Final SHA: `fe5796b401fe276cb6c916d5c613e8dc286dcaa0432b136d5aee3465f2803a38`.
Weights, optimizer and RNG state were trained on Linux and restored on macOS.
All four replays and policy lineage were independently checked after download.

A regression check on the same 80 simulator diagnostic games moved from the
parent's 29 wins, 50 losses, 1 draw to 43 wins, 37 losses. This reused diagnostic
set is not a new held-out evaluation. Four real matches do not establish a
long-run real-server win rate. No public agent was replaced.

- [Last winning replay](https://arena.uotavern.com/replay/?replay=27e442010d144d16959852721e6d71d6)
- [Experiment record](../../docs/experiments/2026-09-30-neural-continuation.json)
- [Commands and limitations](../../docs/NEURAL_DUEL.ko.md)
