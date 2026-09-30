# Continued duel training candidate

Frozen simulator-trained checkpoint, selected before the final evaluation.
Policy SHA: `22bc333e08741e402c0245fb97ca8cc0b588199cfc6531364e7e908ee8f70526`.

This continues `models/duel-starter` for one million additional PPO transitions
(two million in this model's lineage). Two other one-million-step experiments
were retained as comparisons and were not selected.

Changes in the selected training run:

- Simulator v5 includes deployed pre-AOS damage-related stamina loss.
- Discounting uses elapsed seconds, with matching potential shaping.
- Actor updates use decisions with multiple legal options; critic updates keep
  casting and waiting frames too.

## Frozen evaluation

Both policies used the same 100 new environment seeds, each from both spawn
sides, against the fixed scripted opponent in simulator v5.

| Selection mode | Previous starter | This candidate |
|---|---|---|
| Categorical, matching live inference | 49 wins, 151 losses | 99 wins, 100 losses, 1 draw |
| Greedy, separate seed set | 91 wins, 106 losses, 3 draws | 110 wins, 89 losses, 1 draw |

For categorical evaluation, the paired score gain (draw = half win) was 25.25
percentage points; the 95% bootstrap interval, resampling whole paired seeds,
was 15.75 to 34.5 points. These are approximate-simulator results, not a human
or real-shard win rate. The greedy interval included zero.

Weights, optimizer and RNG state are included. Live learning saves descendants
separately so this evaluated checkpoint remains frozen. No public promotion is
automatic.

[Training and live commands](../../docs/NEURAL_DUEL.ko.md)
