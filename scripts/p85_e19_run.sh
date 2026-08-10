#!/usr/bin/env bash
# E19 -- pre-registered in docs/experiments/E19-clock-mechanism.md
#
#   cell A  one overrule per game (od1)      -- separates H-compound from H-fusion
#   cell B  fire only when losing (ow0.5)    -- the validated half of the user's trigger
#
# Shard counts are chosen so both cells finish at roughly the same time: A costs
# ~10 s/game (the cap stops all searching once used), B costs ~97 s/game.
# 6 processes total = the same machine load E18 ran under, so the 12 s
# per-decision cap aborts at the same ~2% rate rather than a worse one.
#
# ⛔ --deck-a/--deck-b grimmsnarl is NOT optional (sample collapses the trigger).
# ⚠ Launch with NO tool-level timeout; verify completion by counting shards.
set -u
cd "$(dirname "$0")/.."
mkdir -p out/arena/e19 out/logs

# cell A: 2 shards x 400 matches = 1,600 games
for i in 0 1; do
  python -X utf8 scripts/arena.py play \
    "bc:cap,orc,od1,net=out/policy_v5_s2.npz" \
    "bc:base,net=out/policy_v5_s2.npz" \
    --deck-a grimmsnarl --deck-b grimmsnarl \
    --matches 400 --archive "out/arena/e19/capA_$i.jsonl" \
    > "out/logs/p85_e19_capA_$i.txt" 2>&1 &
done

# cell B: 4 shards x 75 matches = 600 games
for i in 0 1 2 3; do
  python -X utf8 scripts/arena.py play \
    "bc:wp,orc,om99,ow0.5,os12,net=out/policy_v5_s2.npz" \
    "bc:base,net=out/policy_v5_s2.npz" \
    --deck-a grimmsnarl --deck-b grimmsnarl \
    --matches 75 --archive "out/arena/e19/wpB_$i.jsonl" \
    > "out/logs/p85_e19_wpB_$i.txt" 2>&1 &
done
wait
echo "=== E19 shards done ==="
