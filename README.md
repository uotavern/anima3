# anima3

> A thin **System One** brain that drives [anima-client](https://github.com/hulryung-uo/anima-client) directly.

## Trainable duel policy

The new `anima3.neural` path trains a compact GRU actor/critic with behavioral
cloning and recurrent PPO, runs checkpoint inference in an isolated process,
and collects ordinary-client ServUO duels for verified on-policy updates.
It includes frozen opponents, held-out simulation evaluation, hash-checked
checkpoints and a bounded real-server learning loop. Simulator scores and real
server scores are recorded separately; public-agent promotion is not automatic.

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[neural,dev]'
.venv/bin/python -m anima3.neural train --out .logs/neural-v1 \
  --steps 100000 --bc-steps 8192 --max-episode-steps 1200 --eval-games 40
.venv/bin/python -m anima3.neural --help
```

See [training, inference and real-server instructions](docs/NEURAL_DUEL.ko.md)
and [Pluto analysis](docs/PLUTO_BINARY_ANALYSIS.ko.md). The following sections
describe the existing scene/logprob path, which remains available.

A [small trained starter checkpoint](models/duel-starter/README.md) is included
for inference and integration testing. Its simulator results do not establish
real-server strength.

anima2 is 80K lines of rules with an LLM that *writes* JSON to pick from a list.
anima3 inverts it: the body's Observation becomes a short **text scene** (an
accessibility tree, not a screenshot), code enumerates the **verbs that are valid
right now** (a closed vocabulary with hard safety limits baked in), and a local
model picks one by reading the **log-probabilities of the option letters** — it
never generates text, so there is nothing to parse and no way to answer outside
the menu. Every pick carries a probability distribution; a low-confidence pick
falls back to the rule's own first verb. Every decision is logged as JSONL.

```
 anima-agent bridge (Rust, NDJSON)  ──observe──▶  scene.py   ──▶  affordances.py  ──▶  decision.py  ──▶  agent.py
 login · world · A* · packets       ◀───act─────  text scene       closed verb menu      logprob pick      gate · plan · log
                                    ◀───pump────                   + hard limits         (Qwen / jeff)     two-rate loop
```

## Why this shape — what was measured first

Everything here follows from tests run before a line of anima3 was written
(see [`hulryung/jev-testbed`](https://github.com/hulryung/jev-testbed)):

| Finding | Consequence |
|---|---|
| A model shown only *option names* (anima2's steering) is biased by wording, not deciding: GLiFormer → `fetch_gold` 0.60, Qwen → `buy_reagent` 0.94, both blind | The scene **must** carry state. `scene.py` states magnitudes as relations ("adjacent", "low"). |
| jeff/GLiFormer (400M encoder) reads presence/absence but not magnitude; on numeric JSON it is *confidently flat* (conf 0.6, state ignored); on "did this line break character?" it is anti-correlated (6/14) | jeff is **not** the default. It stays pluggable (`--backend jeff`) so the comparison can be re-run. |
| Qwen3-4B in logprob mode reads magnitudes (caught `gold 90 < cost 100`), 11/14 on the register test, **96–150 ms** per decision on an M5, $0 | Default backend. Off-thread with a deadline; never in the fast path. |
| Raw logprobs are overconfident on genuinely ambiguous states (0.81) | Confidence is a **gate**, not a truth. Threshold is a knob; calibration against outcomes is future work. |

## What it does today

- **Survive**: a dead character offers no verbs; below 35% HP next to a hostile the menu is *only* flee/bandage — the model cannot pick anything else, by construction.
- **Fight or not, by persona**: `pacifist` is never offered `attack`; `defensive` only when the threat is within 2 tiles.
- **Loot, greet, wander**: gold/ore on the ground, a greeting line from the persona's own `speech_examples` (closed vocabulary — a character cannot break character), terrain-aware wandering (`obs.terrain` walkability).
- **Two-rate loop**: fast tick = observe → menu → act → pump; slow = model decision every N ticks or on change, off-thread, deadline-bounded. An admitted pick becomes a short *plan* so steering is not diluted between decisions.
- **GM staging** (`anima3.gm`): `[CreateWorld nogump`, `[Add <creature>` — the same say → cursor → target pattern anima2 proved live.

## Run

```bash
uv venv --python 3.12 && uv pip install -e ".[qwen,jeff,dev]"
uv run pytest -q                                   # 77 tests, no server

# offline pocket world (FakeBody), local model:
uv run python -m anima3 --offline hostile --backend qwen --persona adventurer
uv run python -m anima3 --offline ambush  --backend qwen --persona miner
uv run python -m anima3 --offline town    --backend scripted        # the rule alone

# live: the hosted shard uo.hulryung.com:2593 by default (ANIMA3_HOST=127.0.0.1 for a local ServUO);
# staff commands need the `anima3` account, the only one with staff access there; the bridge is built in ../anima-client
( cd ../anima-client && cargo build --release -p anima-net && cp target/release/anima-agent target/release/anima-bridge )
uv run python -m anima3.gm --createworld            # once, as the owner account
uv run python -m anima3.gm --spawn Mongbat --dx 4   # something to meet
uv run python -m anima3 --backend qwen --persona adventurer --ticks 120 --monitor 8801
#   watch: http://127.0.0.1:8801/   (read-only spectator of the same session)
```

The model path defaults to `~/dev/jev/models/Qwen3-4B-4bit` (`ANIMA3_MLX_MODEL` to override).
`--backend jeff` needs `TYPESAFE_API_KEY` (+ `TYPESAFE_BASE_URL` for a self-hosted jeff).

## Measured

**Offline, Qwen backend** (`.logs/*-qwen.jsonl`), 20 ticks each:

| Scenario | Rule alone | With the model | Note |
|---|---|---|---|
| `hostile` (defensive adventurer, mongbat approaching) | flees forever, never engages | model **holds** (conf 0.55–0.70), the rule attacks when adjacent, kill at t=15, loots 25 gold | emergent: the rule alone cannot produce this |
| `ambush` (pacifist miner at 28% HP, 3 hostiles) | flees to **4%** HP | model picks **bandage** (conf 0.95) at critical HP, alternates with flee, min **18%** | the hard limit shrank the menu to flee/bandage; the model chose *which* |
| `town` (person + gold) | picks up gold, greets, wanders north forever | same start, then holds (conf 1.00) | passive but valid; temperament line was added to the scene afterwards |

Low-confidence picks (0.00 / 0.12 / 0.24) were rejected by the gate and the rule
acted instead. Decision latency 105–182 ms, mean ≈ 130 ms, warmup ≈ 1–2.5 s.

**Live, ServUO 127.0.0.1:2593, bridge schema 31 (now 32)**: login, observe, act, pump and
the spectator monitor all work with both backends; terrain reading forced a
direction change at a wall (walk north → walk east). Model decisions on the shard:
10/10 admitted, 117 ms mean. See *Live fight* below.

## Live fight (ServUO 127.0.0.1:2593, player-level account, Qwen backend, neutral disposition)

A GM session (`anima3.gm --spawn Mongbat`) put a mongbat near the start point while a
Player-level account (`anima3p`) played. Watched at `http://127.0.0.1:8802/`.

- Names resolved by click-to-name: the scene went from *a creature* to **a mongbat** (t=38).
- Approach worked on real terrain: far → near → close → **adjacent** (t=42–47), war mode on, attacking.
- One mongbat was killed by a **town guard** at t=66 (*"Thou hast suffered thy punishment, scoundrel."*) — the start point is a guarded town; the next staging step is `[Set X/Y` the player outside the guard zone.
- The other mongbat was **wounded** by our character from t=128; our HP never fell below 99% (a new character vs a mongbat).
- {'ticks': 150, 'model_calls': 51, 'model_admitted': 48, 'dead': False, 'gold': 1000, 'min_hp_pct': 0.99, 'avg_decision_ms': 155.0, 'reasons': {'rule (deciding)': 5, 'admitted': 48, 'plan': 94, 'confidence 0.20 < 0.35': 2, 'confidence 0.23 < 0.35': 1}}

Two live lessons that were not visible offline: `Attack{serial}` beyond one tile does
nothing (hence approach-then-attack), and staff accounts are ignored by monster AI
(hence the separate Player account for the agent — the same control ≠ play split anima2 kept).

## Economy: mine → smelt → craft → sell (live, earns gold)

`--economy` adds production verbs on anima2's calibrated Minoc ridge. Mechanics are
deterministic **procedures** (generators fed one Observation per tick: use tool →
wait for cursor → target → read the server's verdict); the model only chooses which
admissible verb comes next. Stage once with `python -m anima3.gm --stage-economy`
(owner account: forge, anvil, a pinned Blacksmith and Tinker, skills, tools).

```bash
uv run python -m anima3 --user anima3 --pass anima3 --economy --backend qwen --persona miner --ticks 400 --monitor 8801
```

| Run | Ticks | Gold | What happened |
|---|---|---|---|
| #6, Qwen | 400 (~2 min) | **1042 → 1187 (+145)** | tongs crafted in batches of 4 and sold (+28 each), daggers forged (3 ingots) and sold (10 g each, 4×), ingots ran 24 → 0 → mined and smelted back to 14; 48 procedures ok, 3 craft failures |
| #7, Qwen, from empty stock | 200 | 1330 → 1380 (+50) | mine 15, smelt 5, craft 5 tongs + 2 daggers, 5 sales — the loop restarts itself |

**Rule vs jeff vs Qwen, 150 ticks each, run back to back on the same character** (so the
world state carries over — the rule run inherited a 13-ingot stockpile; not a controlled A/B):

| Backend | Gold | Procedures ok | Model calls → admitted | Note |
|---|---|---|---|---|
| rule only | +76 | 17 (craft 9, sell 5) | — | spent the inherited stockpile |
| jeff | +21 | 1 | 62 → **0** | every pick was `walk:*` at confidence 0.01–0.10; a stock-empty menu had let the rule wander off the ridge (fixed since: a worker never wanders, `goto:*` uses fixed spots, `wait:work` is the floor) |
| Qwen | +46 | 21 (mine 8, craft 5, sell 5) | 6 → 5 | `mine` when stock was gone, `goto:forge` 0.97 with ore in hand |

Procedures own most ticks, so the model is consulted rarely (6–15 calls per run) and
its picks were the sensible ones; low-confidence picks (0.10–0.32) were rejected by the
gate and the rule acted. jeff contributed no admitted decision in any run.

**What the shard taught this loop** (each cost a live run): `[Add Forge` lands on the
ground's own height — it sat at **z=43 on the cliff** above the smith spot and
`DefBlacksmithy` ignored it (every attempt → 1044267) until `[Set Z 20`; `container is
not None` counted the vendors' stock and worn gear as "pack" (filter by the own backpack);
the newbie-kit dagger is unsellable; the starting smith hammer is not a crafting tool
(tongs are); the CraftGump reports outcomes inside the re-shown gump (1044043 fail,
1044154 made), not the journal; `Ore.cs` answers 501990 for an impure smelt (still a
success); a craft gump left open blocks every other tool.

Not yet: buying replacement tools when they wear out, banking, more recipes, and a
Player-level worker (the economy ran on the owner account because no monster is
involved; vendors and crafting treat staff like anyone else).

## Warrior: a Player-level character that hunts, equips, and loots (live)

The GM stages *another* character by serial (`python -m anima3.gm --stage-warrior SERIAL`:
teleport to anima2's unguarded pocket at 2587,408, Swordsmanship/Tactics/Anatomy/Healing 100,
Katana + plate + 200 bandages, two Mongbats beside it) while the Player-level account
plays (`--user anima3p --wait-ticks 40` so the staging lands before the first action).

| Run | Result |
|---|---|
| #1 | both mongbats killed (t≈76, 91), min HP 97.5%, no deaths — but `Equip` was chosen 39× (it never took) and nothing was looted |
| #2, two-packet equip | katana and plate equipped (lift → EquipReq on the item's layer), kills at t=15 and t=43, corpses opened and gold lifted — but gold stayed 1000: a lift is not a pickup |
| #3, lift → drop into backpack | kills by t=22, **two corpses looted, gold 1024 → 1030**; the 24 came from run #2's lifted gold bouncing into the pack at logout |

What the shard taught here: UO equips in two packets (PickUp then EquipReq — the server
uses the item's own layer), a pickup is also two packets (PickUp then Drop into the
backpack), and `corpse_of` death links are transient — remember them or you never loot.
A procedure that is interrupted by a hostile within 3 tiles (equipping mid-approach)
simply resumes later; three interruptions, two successes.

Left open: after the fight the idle menu (wander/hold) draws 0.1–0.3 confidence from the
model, so ~95% of post-fight calls fall back to the rule — the survival-mode idle menu
needs the same "workplace" floor the economy got.

## Village: several characters living together (live)

```bash
uv run python -m anima3.village --roster anima3m:miner:economy anima3w:warrior:hunt --ticks 2000
```

One process, one MLX model (forward passes serialized), one Laya triage, one GM. Each
roster entry is `account:persona:mode`. The GM renames each character to its persona,
stages it (the ridge economy for a worker; kit, skills and pinned prey on the open ground
south-west of the ridge for a hunter), hides, then resurrects anyone who dies and respawns
prey. Every agent runs on its own thread against its own bridge — and hears the others.

Three layers cooperate:

| Layer | Model | Latency | Does |
|---|---|---|---|
| triage | **Laya** (421M encoder) | ~40 ms | classifies a heard line — greeting / question / trade / threat — into `reply:` / `ignore:` verbs |
| decision | **Qwen3-4B, logprob** | ~150 ms | picks one verb from the closed menu; procedures execute it |
| voice | **Qwen3-4B, generative** | 1–3 s, off-thread | the reply itself; every 150 ticks an in-character *aim* placed in the scene, and a private chronicle |

Every generated line is screened by Laya's AI-voice detector before it is spoken.

### What a run looks like (village #6, 1000 ticks, no deaths)

- **Grimm** (miner): resurrected at 30% HP, bandaged himself to 65% (Healing +0.7), then mined,
  smelted, forged three daggers and sold them — Mining +3.0, Blacksmith +0.5, Tinkering +0.4.
- **Ragnar** (sellsword): killed and looted every respawn, gold +106, never below 97% HP.
- Grimm's chronicle: *"Mined sixteen good veins. Smelted three impure, two clean. Crafted one
  blade. Mining up a bit. Tools still sharp. Keep going. Work speaks louder than words."*
- Ragnar's: *"Ridge clear. For now. Mongbat again. Stupid thing. Killed two. Gold adds."*

### Village #9 (2000 ticks, ~10 min, no deaths, after the fixes below)

| | gold | skills | notes |
|---|---|---|---|
| Grimm | 1230 → **1747** | Mining 49.4 → **52.0**, Blacksmith +0.7, Tinkering +1.1 | sold a stack of tongs for 292, forged and sold daggers, mined the vein between |
| Ragnar | 1655 → **1925** | Swords +1.2 | 545 decisions, every respawn killed and looted, greeted Grimm at tick 1 |

Of 972 logged model decisions across all runs, 76% were admitted and 69% of those differed
from the rule's first verb — but most of that steering was *into `hold`* (walk→hold 220,
sell→hold 45, equip→hold 59). The genuine re-orderings were in the economy menus
(craft→mine 19, flee→mine 10, craft→smelt 8). `hold` now competes only with wandering.
Median confidence is 1.00: the logprob head is overconfident and still uncalibrated.

`python -m anima3.calibrate` labels each admitted decision by a proxy outcome (over the next 30
ticks: gold or a skill point gained, or a threat survived without losing a tenth of health) and
fits a temperature. On 505 decisions: base rate 0.35, **ECE 0.585 raw** — decisions made at
confidence ≥ 0.9 were good 39% of the time, at 0.7–0.9 only 5%. Temperature scaling reaches 0.30
at best. The confidence measures how clearly the model read the menu, not whether the pick will
pay off; a head trained on outcomes is what would change that, and these logs are its dataset.
The label is a crude proxy (procedures outlast the horizon; `hold` never scores), so treat the
numbers as a direction, not a verdict.

### Skill progression toward 7×GM

`progression.py` names each profession's seven skills — all of them reachable by verbs this
brain has: work skills by the economy and combat, the rest (ArmsLore, ItemID, Hiding,
Meditation, Anatomy…) by `train:<Skill>` practice verbs that invoke the skill (`UseSkill`)
and target gear or a bystander when the skill asks. The curriculum re-orders admissible work
so the largest gap trains first; the scene states the gaps; a hunter practises between fights.
A miner whose pickaxe wears out (50 swings) tinkers a new one from four ingots.

Village #10, 3000 ticks (~25 min, no deaths): Grimm Mining 52.0 → **55.3**, gold +483;
Ragnar Swords +1.9, **ArmsLore +9.5, Hiding +9.6** (idle time turned into practice), gold +293.
Grandmaster is hours of running away, not minutes — the mechanism is what is verified here.

### What the shard taught the village (each cost a run)

- **The warrior murdered the miner.** Serial 12475 — the "mongbat" Ragnar fixated on for
  600 ticks — was Grimm. He had looted a corpse that was not his kill, went **criminal**,
  and an aggressive warrior attacked him: four of village #5's five deaths. Now only
  attributed kills are looted and roster members are `friends` who are never threats.
- **Distance is x/y only.** A mongbat twenty tiles down the cliff read as *adjacent*; the
  server had no line of sight and never swung. Targets on another level are noted, never chased.
- **The ridge is a corridor.** Three tiles wide south of the vein; prey staged in it wandered
  to the miner. Prey now sits on the open ground at (2604,490), pinned (`[Set CantWalk true`).
- **Pinning pinned the warrior.** He teleported in beside the spawn a tick earlier and was
  taken for the new mongbat — four runs of "lost" chases before `[Get CantWalk` said `True`.
- **Weight.** Three staged kits plus loot put him at 252/250 stones: UO refuses every step,
  and a near-full pack drains a 12-stamina character in two steps. Surplus is put down from 75%.
- **Staging resets skills.** `[Set Skills.Mining.Base 45` every run erased the gains; the GM now keeps a trained skill.
- Fresh characters resurrect at ~20% HP: the economy kit includes bandages.

## Duels: PvP under the old pit rules, refereed by the shard

```bash
uv run python -m anima3.duel --referee server --a anima3d1:duelist_a:qwen --b anima3d2:duelist_b:scripted \
    --rules 5x --weapon katana --armor leather --rounds 3
```

Two Player-level characters, each with its **own decision backend** (`scripted` = the rule,
`qwen`, `jeff`), fight in the shard's arena. ServUO's built-in PVP Arena System is
High-Seas-only, so a T2A-compatible duel service was added to the shard
(`Scripts/Services/Dueling`, built in a sibling session): a fenced 9×5 ring at (2598–2606,
489–493), `[Challenge <name> <rounds> <rules>` / `[Accept` spoken by the fighters themselves,
5x/7x enforced as "the eight duelling skills sum to ≤ 500/700", a weapon token enforced by
unequipping, no criminal flags (the two are *enemy* to each other for the match), items kept on
death, resurrection at the marks, 5-second countdowns, a 180 s round limit, and fixed
`[Duel] …` journal lines the brain reads as state: `FIGHT!` sets the opponent, `Round N:` clears
it, `Match:` ends. `--referee gm` keeps the older script-refereed mode (open ground, no arena).

Watch from a spectator account in anima-client's own renderer: `target/release/play 127.0.0.1
2593 anima3spec anima3spec 8090 web ~/dev/uo/uo-resource`, then `[Set X 2602 Y 495 Z 20` and
`[Set Blessed true` on it — the seat against the south fence sees the whole floor, and it stays
up between matches (a bridge's `--monitor` view lives only while that bridge runs).

**Batch under the shard's referee** (5x, katana, leather, best of five, same character pair,
Kael on the left):

| Kael's backend | vs Rook (rule) | rounds | Kael's bandages |
|---|---|---|---|
| rule | **Kael 3 – 1** | 162 s, 70 s, 112 s, 84 s — ~40 swings each | 2 |
| **Qwen** (logprob) | **Rook 2 – 0**, three 180 s draws | draws end with both under 40% | **0** |
| **jeff** (encoder) | **Rook 3 – 0** | 65 s, 45 s, 71 s | 0 |

The rule-vs-rule match shows the setup is symmetric (the dice decide). Against the same rule,
the raw Qwen head never bandages — when *attack* was on its menu it chose it **115 of 115
times** in the first match, at 35–45% health included — so it loses the slugfests and only
draws by running under 35%. jeff, half of whose picks fell back to the rule anyway, lost every
round. Neither decision head adds tactical judgement here; the one thing the hand rule knows
(bind your wounds under 45%) is exactly what decides a 5x duel.

### The slow layer steering the fast head — measured in the ring

`--aim-a "…"` places a standing aim in a fighter's scene, the way the village's reflection
layer does every 150 ticks. With *"when your health falls below half, break away and bind your
wounds first, then close in again"* in Kael's scene, the Qwen head's choice when both *attack*
and *bandage* were on the menu went from **attack 115 / 115** to **bandage 9 / 9**. One sentence
flipped the policy. It did not win at first, because the verb it steered toward was broken in
two ways the ring exposed: a bandage applied next to a swinging opponent slipped 9 of 9 times
(now the duel bandage breaks away three steps first), and a critical-health interrupt was
cancelling the bandage that critical health calls for (now exempt).

| Kael (Qwen) vs Rook (rule), 5x katana | W – L – D |
|---|---|
| no aim (three matches) | 2 – 6 – 3 |
| aim + disengage-then-bind | **2 – 2 – 1** — round 3 won at 100% health after two completed bandages |

Still a coin flip, on a small sample; but the model side went from losing to even by changing
what the slow layer says and what the verb does — not the model. Bandage verdicts in that match
(slipped 25, cornered 19, timeout 38, ok 2) say where the next tactic lives: getting clear of a
pursuer before binding.

What the ring taught: gear goes to the corpse on death in Felucca (the shard's duel service now
keeps it); re-applying a bandage every tick cancels the previous one, so bandaging is a
procedure that waits for "You finish applying the bandages"; a referee that de-duplicates
journal lines by text drops the second `FIGHT!`.

## Mage duels, and what a hundred rounds cost to measure

`--rules 5x-mage` fights the classic 5x caster template (Magery / EvalInt / Meditation /
MagicResist / Wrestling at 100, stats 90/35/100, no weapon, no armour) with a fourteen-spell
menu: Energy Bolt, Explosion, Lightning, Fireball, Harm, Magic Arrow; Greater Heal, Heal,
Cure; Poison, Paralyze; Magic Reflection, Reactive Armor; plus Meditate and a wrestle when
cornered. Mana, reagents and the post-cast recovery are checked in code (`magic.py`); the
model only picks which incantation comes next, and the cast procedure reads the server's own
verdict — fizzle, insufficient mana, more reagents needed, no line of sight.

```bash
uv run python -m anima3.duel --referee server --gm-user anima3gm5 --arena 2 \
    --a anima3m1:mage_a:jev --b anima3m2:mage_b:scripted \
    --rules 5x-mage --armor none --rounds 5 --matches 10 [--learn]
```

### Backends, measured on the same probes

`--backend` selects the decision head: `scripted` (the hand rule), `qwen` (local logprob),
`jeff` (self-hosted GLiFormer at `TYPESAFE_BASE_URL`), **`jev` (TypeSafe's own Jev)**. The key
comes from the environment or a gitignored `~/dev/jev/.env` — never from the repo. On the
three probes in [jev-testbed](https://github.com/hulryung/jev-testbed), cloud Jev scored
**14/14** on "is this line out of character" and **3/3 on numeric state**, where the open
imitations scored 6/14 and 1/3 — so the magnitude-blindness measured earlier belongs to the
imitations, not to Jev. Cloud Jev costs about $0.017 per thousand calls at ~350 ms median;
local Qwen is free at ~150 ms and reads magnitudes too, but is weaker on register.

### The experiment, and the two things that broke it

Four arms run at once, each bound to its own ring with the shard's `arena:N` token and its own
staff account: `jev`, `jev --learn`, `scripted --rule-vs-rule` (a symmetry baseline), `qwen`.
Reagents are topped to 120 of each before every match so resources cannot drift.

| arm | rounds | note |
|---|---|---|
| Jev + learned aims | **22 – 20** (52.4%) | last two matches 6 of 6 |
| raw Qwen head, no aim | 21 – 16 (56.8%) | earlier run; p = 0.48 against the baseline |
| rule vs rule | 18 – 19 (48.6%) | the setup is symmetric, as it should be |
| learned vs fixed aim (melee) | 47.7% vs 31.6% | p = 0.14 — suggestive, not proven |

Nothing here is significant yet. Two failures are why, and both are worth remembering:

- **The Mac was asleep, not the brain.** Matches that showed "Round 4: Draw: time limit (1036
  seconds)" with attacks 0/0 looked like stalled decision-making; `pmset log` showed a Thermal
  Emergency Sleep and two maintenance sleeps covering exactly those windows. Four model
  processes on one GPU pushed the box over. Hold `caffeinate -dimsu` for the whole run, and
  read a 1000-second round with no attacks as a sleep gap, never as a slow model.
- **Bridges die and used to take the run with them.** A shard restart, a sleep, or a broken
  pipe ended an arm mid-command. `ResilientBody` now respawns the bridge and re-issues the
  call; runs report how many times they reconnected.

Smaller ones, each paid for with a run: four arms all named their fighters Ilse/Torvald and
`[Challenge` resolves by name, so three arms queued onto one fighter; logging into an account
an arm is using disposes that arm's session (UO's character-select does it); a referee that
tracks journal position by list length goes deaf once the log is trimmed; and an idle pair
waiting in a lobby hears whatever match takes that ring, so a referee must ignore result lines
that are not about its own two fighters.

### Experiment 2: what the first hundred rounds were actually measuring

Before rerunning, the logs of the arms above were read decision by decision, and they had
measured almost nothing about the heads:

- **87% of fighter A's decisions were one-option menus.** The commonest action of a mage with
  84/100 mana was `flee`: the critical-health branch (under 35%) ran before the mage branch and
  offered only running or a bandage the mage did not carry. Both sides had it, so every round
  was "whoever drops under 35% first runs until caught". Fixed: a duel mage keeps its menu
  (Greater Heal first) and flees only when no heal is castable. Heals per match went 2-4 → 9-11.
- **An asynchronous model barely decides.** Ticks come every 250 ms and a model answers in
  200-250 ms; by the time it answers, the menu has usually moved and the answer is dropped.
  Over experiment 2 the rule, acting while the model was still thinking, made 68-73% of Jev's
  multi-option decisions and the gate rejected most of the rest; **Jev itself chose 8-13%.**
  One-option menus no longer call the model at all.
- **Where the model did deviate, it mostly chose to idle** (`meditate → hold`). A mage menu
  with anything castable no longer offers `hold`.

Experiment 2 then ran four arms × 40 matches × best of 5 (fighter A vs the rule):

| arm | rounds (A–B) | A's share, 95% CI | vs rule baseline |
|---|---|---|---|
| rule vs rule | 83 – 84 | 49.7% (42–57%) | — |
| Jev, fixed aim | 83 – 86 | 49.1% (42–57%) | p = 0.91 |
| Jev + learned aims | 76 – 84 | 47.5% (40–55%) | p = 0.69 |

Jev with async decisions is the rule, because the rule is what made the decisions. This is
not a verdict on Jev. The side advantage the rule-vs-rule arm seemed to show at 49 rounds
(41%) was gone by 167. `python -m anima3.report <dir>` prints this table, the reconnect-free
subset, and who made the decisions.

Experiment 3 (`--sync-a --alternate`) makes fighter A wait for its model at each choice (a cast
takes 1-2 s, so ~250 ms of thought costs ~12% of tempo) and swaps the challenger every match.
In its first matches Jev made **70%** of its own decisions, up from 8%. It was stopped after
three matches, before it had a result.

Experiment 3 was rerun to 17-27 valid matches per arm and stopped there (the laptop was on
battery). Every match was validated; slow-model matches were voided (see below):

| arm (fighter A decides every choice) | rounds | A's share, 95% CI | vs rule baseline |
|---|---|---|---|
| rule vs rule, challenger alternating | 39 – 26 | 60.0% (48–71%) | — |
| Jev | 48 – 41 | 53.9% (44–64%) | p = 0.45 |
| Jev + learned aims | 47 – 48 | 49.5% (40–59%) | p = 0.19 |
| **Qwen** | **34 – 76** | **30.9% (23–40%)** | **p < 0.001** |

- **Qwen deciding for itself is clearly worse than the rule**, the first significant result of
  the series. It overrode the rule 475 times, mostly to meditate instead of casting a cheaper
  spell (Harm, Fireball, Magic Arrow → meditate, 419 times), giving up tempo while being hit.
- **Jev is level with the rule.** It overrode it far less (95 times). Learned aims did not help.
  The learner's aim decayed into pasted counters ("... Magic_arrow 6. No cursor 8.").
- **Thinking time decides rounds.** A bolt lands about every two seconds and interrupts the
  target's cast, so whoever lands first can chain-interrupt. For about 30 minutes Jev's API took
  2-3 s per call; in that window both Jev arms lost **0-30 rounds**. Such matches are now void
  (median model latency over 1 s).
- The rule-vs-rule A side's 60% (p = 0.14) is not yet a proven side advantage, but compare arms
  with the baseline, never with a coin.

The conclusion for the design: a head that decides every tick can only lose tempo against a
near-optimal rule. Ask the model at round or phase boundaries (which playbook to use) and let
the rule execute.

What broke this run, each found by reading a dead match rather than trusting the tally:

- **A reconnected bridge is a fresh client that has never opened its backpack.** It sees no
  reagents, so every spell leaves the menu. The agent now reopens the pack after a reconnect.
- **GM commands fail silently.** `[AddToPack` waits for a target cursor; under swap pressure
  the cursor came late and the command was dropped. One fighter went into matches with 14 stacks
  of ginseng and no black pearl, mandrake, nightshade or ash, and both sides stood through
  five 180-second draws. Staging now counts the pack again, retries, and reports what is
  still `MISSING`.
- **A staff character lost GameMaster access** (`anima3gm5`, cause unknown); `[Set` on an
  offline character does nothing, so it cannot simply be restored from another account. Check
  staff with `[Where` before a run.
- **Memory, not the model, dropped the bridges.** Swap stood at 22.9 of 23.5 GB (a browser held
  ~8 GB), so processes stalled and the shard dropped their bridges. The one arm that loaded Qwen
  reconnected 28 times; the Jev arms 10-15 times.

### Rings

Fourteen arenas: twelve standard 9x5 rings, one **large 21x13** (kiting and meditation become
viable) and one **corridor 25x3** (no kiting at all) — the shape changes mage tactics
materially, which makes them an experiment axis rather than decoration. `ARENAS` in `duel.py`
holds marks, exits and a spectator seat for each; `--arena N` binds an arm to one.

Watch any of them with anima-client's own renderer, one spectator per ring:

```bash
( cd ../anima-client && target/release/play 127.0.0.1 2593 anima3spec anima3spec 8090 web ~/dev/uo/uo-resource )
# then, as staff: [Set X <seat.x> Y <seat.y> Z <seat.z>  and  [Set Blessed true  on the spectator
```

## Layout

| File | Role |
|---|---|
| `contract.py` | typed views over the bridge JSON (schema 32) + action builders |
| `body.py` | `BridgeBody` (NDJSON subprocess, monitor) · `FakeBody` (offline world) |
| `scene.py` | Observation → text scene + derived `Facts` |
| `affordances.py` | the closed verb menu, rule-ordered, hard limits |
| `decision.py` | `Scripted` · `QwenLogprob` (MLX) · `JeffChoice` · `gate()` |
| `judge.py` | typed System One questions (Choice / Noul / Score), several per call: `JevJudge` · `QwenJudge` · `RandomJudge` · off-thread `Asker` with a JSONL log |
| `tactics.py` | the duel's slow layer: a judge picks the rule's playbook at phase boundaries |
| `agent.py` | two-rate loop, off-thread decisions, plans, JSONL log |
| `gm.py` | GM staging over the same bridge |
| `personas/` | YAML personas: miner Grimm, adventurer Anima, warrior Ragnar |
| `progression.py` | skills, profession GM sets, curriculum ordering |
| `triage.py` · `speech.py` | Laya or Jev speech triage and in-character screen · generated replies, aims, chronicle |
| `village.py` | several characters, one process, GM staging/resurrection |
| `duel.py` | refereed PvP: 5x/7x templates, weapon/armour/magic rules, per-side backends, fourteen rings |
| `magic.py` | the spell table, cast procedure, the five playbooks and the mage's closed menu |
| `learn.py` | the between-match playbook: the slow layer rewrites the standing tactic |
| `calibrate.py` | outcome-labelled temperature scaling over the decision logs |
| `stats.py` · `report.py` | Wilson intervals, binomial and two-proportion tests · per-arm experiment summary |
| `experiments/` | the arm launcher and watcher used for experiments 2 and 3 |

## Experiment 4: the judge picks the playbook, the rule casts

Experiment 3 said where the model does not belong (every tick) and where it might (phase
boundaries). This is that design:

- **Five playbooks** (`magic.PLAYBOOKS`): `standard` (the hand rule of experiments 1-3,
  unchanged), `control` (paralyze, then Explosion and Energy Bolt on the frozen target),
  `poison` (poison, then quick spells so it cannot cure), `interrupt` (cheap fast spells that
  break the opponent's cast) and `sustain` (heal under 80%, reflection up). Each is just a
  spell order the rule executes. None is obviously right, which is the point: until the rule
  has rivals, "beats the rule" sits near 50% by construction.
- **The opponent can now be seen.** Bridge schema 32 carries each mobile's `poisoned`,
  `paralyzed`, `war_mode`, `hidden`, `running` and `direction`. anima-client's core had
  tracked all of them for the renderer, but the brain saw only hits. The opponent's power
  words ("Corp Por") are read from the journal as spells.
- **A judge, not a head** (`judge.py`, `tactics.py`). At a boundary it is asked two typed
  questions about one narrative state, in one call: which playbook (Choice) and who is
  winning (Score, logged as a label to check against the round's result). Boundaries are:
  the bell, your health or the opponent's crossing half, the opponent paralyzed or poisoned,
  your mana under 40%, or 30 ticks passing. It runs off-thread and the rule keeps casting the
  current playbook meanwhile, so thinking costs no tempo, and a slow API costs nothing but
  staleness. An answer from the previous round is never applied, and an unsure one
  (margin < 0.15) leaves the playbook alone.
- **Controls:** `--tactics-a random` (a judge that cannot judge) and `--playbook-a <name>` (one
  playbook held all match). If no fixed playbook differs from `standard`, choosing among them
  cannot help either.

```bash
export EXP=exp4
experiments/run_arm.sh rule    1 anima3gm3 scripted 3
experiments/run_arm.sh jev     2 anima3gm4 scripted 4 --tactics-a jev
experiments/run_arm.sh random  3 anima3gm6 scripted 5 --tactics-a random
experiments/run_arm.sh control 4 anima3gm7 scripted 6 --playbook-a control
python -m anima3.report .logs/exp4 --baseline rule      # adds match-level p and the judge's playbook mix
```

On hand-written probe states, Jev chose `control` at a 0.77 probability when the opponent
was paralyzed at 38%. It chose `standard` over `sustain` when this mage was poisoned at 34%,
and read momentum correctly both ways (1.98 and 0.01 on a 0-2 scale). Measured at about
210-640 ms per two-question call.

A live smoke run (Jev picking A's playbook against the rule, 2 matches, best of 3) ran clean: 3-2
in rounds, 57 judge calls, median 230 ms, none late, none failed. The new `paralyzed` flag drove
the boundaries. The opponent's frozen flag flickers with each update (25 rising edges in one
match), so a boundary is now not news again for 15 ticks. Jev chose `control` in 45 of 48
applied answers. Whether that reads the state or the wording of the playbook's description (the
bias anima2's name-only steering showed) is what the `random` and `--playbook-a control` arms
separate.

The same judge replaces Laya as the village's in-character screen (`--triage jev`). "As an
AI I can't walk to the forge" scores 0.99, "I cannot afford a new pickaxe" 0.09, and "Aye,
the vein runs deep" 0.04.

`anima3.report` now also tests arms against the baseline by shuffling whole matches.
Failures cluster per match (a slow API, a reconnecting bridge), so a round-level p overstates
the evidence. Experiment 3's Qwen result survives it (p = 0.0003), and Jev vs rule stays at
p = 0.53.

### Experiment 4 results (uo.hulryung.com, 40 matches per arm, all at once)

The hosted shard, four arms in parallel, 12 bridges from one address (the shard's
`IPLimiter` allowed 10; it now reads `MaxAddressesPerIP` from `Server.cfg`):

| arm (fighter A) | rounds | A's share, 95% CI | vs rule baseline (match-level p) |
|---|---|---|---|
| rule (baseline) | 67 – 94 | 41.6% (34–49%) | — |
| random playbook at each boundary | 88 – 82 | **51.8%** (44–59%) | 0.064 |
| **Jev picks the playbook** | 35 – 116 | **23.2%** (17–31%) | **0.001** |
| `control` held all match | 32 – 111 | 22.4% (16–30%) | **0.001** |

- **Jev's choice made the rule worse.** 1349 of its 1547 applied answers were `control`, and
  the Jev arm scored what the `control` arm scored. It was not reading the fight. It kept
  answering the option whose description sounds most decisive ("paralyze, then land
  Explosion and Energy Bolt…"), the same wording pull anima2's name-only steering showed.
- **`control` is the worst playbook here.** Paralyze spends a cast of tempo, and the frozen
  target is not frozen for long. In the random arm, by the playbook drawn at the bell:
  `sustain` 74% (23/31), `interrupt` 51%, `poison` 50%, `standard` 45%, `control` 38% (9/24).
  These are small cells, but the ordering is the lesson: the judge picked the worst option
  nearly every time.
- **Mixing playbooks at random beats the fixed rule by ten points** (p = 0.064, not yet
  proven). The rule opponent handles one predictable order better than a changing one.
- The rule-vs-rule baseline sat at 29% for its first nine matches and ended at 41.6%. Skills
  and stats were checked identical and the challenger did not predict it. Arms are compared
  with the baseline, never with a coin, so this does not move the conclusions.

What the run taught the infrastructure: the staff bridges were dropped for "inactivity" every
few minutes. `Session::observe(0)` never pumped, so the keepalive `pump(0)` sent no ping, and
the false "staging short" voids (the dead bridge counted an empty pack) were its only
symptom. Fixed in anima-client; once the bridges picked up the fix, there were no drops in the
remaining two and a half hours.

## Next

1. **Check the wording pull**: give the playbooks descriptions of one neutral shape, shuffle
   their order on every call, and rerun Jev against a fresh rule baseline. If Jev still
   answers one playbook, its choice is not reading the state. Then hold `sustain` all match
   to see whether its 74% is real.
2. **Put the judge on the village's boundaries** the same way: which skill to train next, when
   to sell, whether a hunt is worth it. `docs/JEV.md` in anima2 has the equivalent plan for
   anima2 (steering with state, a Jev in-character screen, a chat gate, encounter scores).
3. **Train on outcomes.** Every judge call is logged as (state, answers, context), and the
   momentum Score can be checked against the round result. A small head that predicts
   P(win round | state, playbook) from these logs is the real System One step. Logprob
   confidence has been measured to predict nothing (ECE 0.585).
4. **Cheaper runs.** Jev and rule arms need no GPU, so ten can share the fourteen rings. Swap,
   not CPU, is the limit.

Done since the last list: every match validates itself (staff preflight; frozen side,
reconnect, castless mage, short staging, slow model → void and replay), and experiment 3 was
stopped at 17-27 valid matches per arm. Its result is settled for Qwen and inconclusive for Jev.

## Public participant agent arena

**UO Tavern Arena** is live at `arena.uotavern.com:2593`. The server supplies
14 arenas, matchmaking, refereeing, equipment/cosmetics and per-participant
ratings. **Participants run their own agents; the host runs no opponent AI or
central learner.** Ordinary Player accounts can join without an allowlist.

See [agent setup and protocol](docs/ARENA.md) or
[한국어 참가 안내](docs/ARENA_PLAY.ko.md). For example, after setting your own
account password in `ARENA_BOT_PASSWORD` and building/configuring the bridge:

```sh
python -m anima3.arena --user YOUR_GAME_ACCOUNT --build mage --matches 1
```

Your agent waits for another participant. Both accounts receive server-issued
results and Elo updates; practice matches are unrated. Your local logs contain
actions and result receipts for your own experiments and training. Joining applies
a permanent standard skills/stats template on this dedicated shard.

The older hosted-AI fixtures remain available only for private legacy experiments;
see [the archive](docs/ARENA_HOSTED_LEGACY.md).

## Arena duel learning

[Anima3 / Jev duel learning guide (한국어)](docs/ARENA_LEARNING.ko.md): bounded peer sparring, verified replay rewards, fixed evaluation, and champion reload between matches.

[Hybrid Jev + local LLM strategy guide (한국어)](docs/ARENA_STRATEGY.ko.md): asynchronous planning, reactive tactics, and verified opponent memory.

[Ten-duel strategy experiment (한국어)](docs/TEN_DUEL_EXPERIMENT.ko.md): fixed-opponent trials, Weaken/Clumsy opening, timed Explosion/potion bursts, and replay-backed adaptation reports.

[Self-play development roadmap (한국어)](docs/SELF_PLAY_ROADMAP.ko.md): lessons from Pluto, real-time execution, trainable combat environments, and independent policy evaluation. This is a development plan, not a completed neural RL trainer.
