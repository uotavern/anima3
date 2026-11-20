# Experimental duel starter

This is the actual 127,384-parameter GRU policy trained in the September 30,
2026 experiment: 16,384 teacher transitions and 1,000,000 recurrent PPO steps.
It was trained from scratch; it does not contain Pluto or third-party weights.

The SHA256 and observation/action schema are checked by the loader. Weights,
optimizer and RNG state are included for inference and resuming training.

Corrected simulator v4 starts with **5 potions of each type**, matching arena
supplies. On 100 new seeds played from both sides: **87 wins, 111 losses,
2 draws**. This does not establish live-server strength. The policy selected
Explosion + potion + Flamestrike 246 times in greedy evaluation, but selection
does not prove all combined impacts landed. It still does not select the
suggested Weaken/Clumsy opener. It is not a promoted public-arena champion.

The failed 150,000-step run and the older 20-potion experiment are retained in
the evidence report. No tuning followed the final 200-game evaluation.

```sh
.venv/bin/python -m anima3.neural evaluate \
  --resume models/duel-starter --out .logs/starter-evaluation --eval-games 20

# Set ARENA_BOT_PASSWORD from your local secret store first.
.venv/bin/python -m anima3.neural wait \
  --checkpoint models/duel-starter --user YOUR_DEDICATED_ACCOUNT
```

See [training and runtime guide](../../docs/NEURAL_DUEL.ko.md) and
[experiment evidence](../../docs/experiments/2026-09-30-neural-duel.json).
