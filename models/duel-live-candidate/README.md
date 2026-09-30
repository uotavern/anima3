# Verified live-learning candidate

This checkpoint is the current simulator-trained V6 policy after one verified
real-server match and PPO update on the arena server. The neural player's
observations, sampled actions, probabilities and GRU states were verified against
the exact policy and complete server replay before PPO.

| Stage | Policy SHA prefix | Accepted transitions |
|---|---|---:|
| V6 simulator-trained parent | `7c0cbe1a0472` | — |
| Server-local live update | `2953462ad056` | 83 |

The parent completed 1,000,000 simulated PPO transitions using the current
5-potion supply limits. In this real match, neural player 57 lost to the fixed
scripted opponent, player 58. Its 83 eligible decisions were used for four PPO
epochs (332 optimized steps). This update does **not** establish an improvement
in playing strength. The artifact remains experimental.

Weights, optimizer and RNG state are included for inference and resuming training.
All three were loaded successfully on macOS after training on Linux; a separate
inference worker also returned three valid decisions. It has not replaced the
public arena agent. The earlier two-cycle self-play validation, including use of
updated weights in the next match, is preserved in the experiment report.

- [Training match replay](https://arena.uotavern.com/replay/?replay=b470dc944875413194d96fcc97b8e079)
- [Experiment report](../../docs/experiments/2026-09-30-neural-duel.json)
- [Commands and evidence](../../docs/NEURAL_DUEL.ko.md)
