# Melee / stun / close Harm mechanics probe

Separate worktree `feat/melee-harm-experiment`. Six unranked, one-round Standard
7x + explosion training matches, two each for magic, punch and stun. Same scripted
opponent, same seven GM build, alternating challenger. Existing 100-match budget
remains at 83 completed; these fixtures do not count towards its remaining 17.

This is a scripted mechanics ablation, not a new trained network. No checkpoint
is updated, no scripted actions are mislabelled as neural policy samples. The
magic control shares healing and opening logic with the melee variants. Melee
variants approach within one tile, attack with free hands, and prioritize Harm;
stun additionally arms the pre-AOS special and observes server acknowledgements.

Local ServUO Harm.cs applies full damage at <=1 tile, half at 2, quarter beyond 2.
Actual target release distance is logged (the opponent can move during a cast).
Fists.cs requires 80 Wrestling and Anatomy and 15 stamina; success freezes for
4 seconds with a 10-second attempt cooldown. Failed stun stays armed. Pending
acknowledgements are never blindly toggled again. Live outcome must be verified.

Runtime: `/opt/uoarena/anima3-melee-probe-20261005`
Unit: `arena-melee-probe-20261005.service` (bounded 2 hours, no restart).
Logs: `.logs/comparison`; private account credentials remain in the original lab.

Use `scripts/summarize_melee_probe.py LOG_ROOT` for wins, server-confirmed stun
outcomes, attack requests and Harm release distances, plus replay links. Replay
anonymous damage cannot prove punch hit rate or causal interruption; do not
report all fizzles as punch interrupts. Two games per arm is only a smoke test,
not a statistically reliable win-rate comparison. Existing validation gates and
state-loss stop remain enabled. Server/shard binaries are unchanged.
