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

## First observed mechanics and resource follow-up

The first stun-labelled fixture reached one tile and released 23 Harm targets,
with 22 executor completions and 28 Attack requests. No StunRequest was sent:
Clumsy reduced the 25 DEX build to 14 maximum stamina, below the required 15.
Do not label this fixture as successful stun use. The first punch fixture stayed
at distance 2 and never punched; labelled modes alone do not prove exposure.

A separate two-fixture follow-up uses supplied Greater Agility (0xF08) before
Refresh (0xF0B) when needed. Acknowledged observation, rather than the Use request,
controls whether stun is legal. Both resources wait for an idle executor/free
hands and are rate-limited. The follow-up waits for the six-fixture service to
exit successfully and checks no bridge is active; failure does not auto-retry.

- Runtime: `/opt/uoarena/anima3-melee-resources-20261005`
- Unit: `arena-melee-resources-20261005.service`
- Results: `.logs/stun-resources`
- 41 focused tests passed, including stale-state/cursor guards, toggle semantics,
  and agility-before-refresh resource recovery.
- Total authorized mechanics probe is bounded to 8 matches, 0 neural updates.

## Completed base comparison

All six base fixtures completed and were receipt-verified. Each arm won 2/2
against the fixed scripted opponent; this does not establish a difference
between the arms. Only two fixtures actually reached melee: together they
recorded 37 Attack requests and 31 server Harm targets at one tile. Neither
stun-labelled base fixture produced a stun success. Raw summaries are retained
in `2026-10-05-melee-harm-results.json`.

The first resource-assisted fixture `ed73efec0c79457490f46f23acdf8b1d` won and
recorded two server-confirmed stun successes, ten one-tile Harm targets, and
four insufficient-stamina messages. This validates actual stun use and shows
resource pressure, not guaranteed success from merely issuing a request.

## Final result

All 8/8 fixtures completed; both transient services exited successfully, no
anima-bridge remained, and arena-shard stayed active. The resource-assisted arm
won both matches. Across those two fixtures: 6 server-confirmed stun successes,
2 failed attempts, 4 insufficient-stamina notices, 72 Attack requests and 36
server-confirmed Harm targets at one tile. See
`2026-10-05-melee-resources-results.json`. The second replay is
https://arena.uotavern.com/replay/?replay=90526361b25f49cf86b9de7343cd5429 .

Across all 8 fixtures there were 67 server-confirmed one-tile Harm targets and
109 Attack requests. All arms won against this fixed opponent, so wins cannot
establish superiority of the added mechanic. Punch hit rate and causal cast
interrupt counts remain unmeasured by this replay format. This run proves live
execution of close Harm and stuns, not reinforcement-learning improvement.
No policy weights or original 100-match budget were modified.
