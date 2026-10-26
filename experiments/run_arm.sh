#!/bin/sh
# One duel arm. $1 name, $2 pair (accounts anima3m{2p-1,2p}), $3 staff account, $4 backend, $5 arena, rest = extra flags
# e.g.  experiments/run_arm.sh jev-sync 2 anima3gm3 jev 3 --sync-a      (logs land in .logs/$EXP/<name>, EXP=exp3 by default)
cd ~/dev/uo/anima3
# On battery a closed lid sleeps the Mac whatever caffeinate says, and every match after that
# is void (experiment 3's first attempt: a 180 s round timed out at 951 s). Refuse to start.
if ! pmset -g batt | grep -q "AC Power" && [ -z "$ALLOW_BATTERY" ]; then
  echo "preflight: on battery power — plug in (or set ALLOW_BATTERY=1)" >&2; exit 3
fi
EXP=${EXP:-exp3}; mkdir -p ".logs/$EXP"
NAME=$1; PAIR=$2; STAFF=$3; BACKEND=$4; ARENA=$5; shift 5
A=anima3m$(( (PAIR-1)*2 + 1 )); B=anima3m$(( (PAIR-1)*2 + 2 ))
AIM="Open with Energy Bolt; Greater Heal below half; meditate only when the opponent is far."
# Each arm holds its own assertion for exactly as long as it runs. A shared caffeinate found by
# pgrep never appeared in experiment 5's rerun: the Mac idle-slept at 02:14 on AC and two arms
# lost eight hours to sleep and reconnect voids.
caffeinate -dims .venv/bin/python -m anima3.duel --referee server --gm-user "$STAFF" --arena "$ARENA" \
  --a "$A:mage_a:$BACKEND" --b "$B:mage_b:scripted" \
  --rules 5x-mage --armor none --rounds 5 --matches 40 --max-ticks 900 --monitor-base 0 \
  --suffix "$PAIR" --log-dir ".logs/$EXP/$NAME" --aim-a "$AIM" --alternate "$@" > ".logs/$EXP/$NAME.out" 2>&1
rc=$?   # before $(date), which would set $? to its own 0
echo "arm $NAME finished $(date '+%H:%M:%S') rc=$rc" >> .logs/$EXP/arms.done
