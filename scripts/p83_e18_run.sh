#!/usr/bin/env bash
# E18 -- the clock's arena A/B, pre-registered in docs/experiments/E18-clock-arena.md
#
# ⛔ --deck-a/--deck-b grimmsnarl is NOT optional. On arena's default `sample`
# deck the oracle's free trigger fires on 0.7% of decisions instead of 24%
# (79% of sample decisions carry >=12 options vs 19.7% on our real deck), so a
# sample run measures a component that barely fires and reports a null.
#
# Sharded by PROCESS with SEPARATE archives -- never let shards append to
# out/arena/games.jsonl.
set -u
cd "$(dirname "$0")/.."
mkdir -p out/arena/e18 out/logs
N=${1:-34}          # matches per shard; 6 shards x 34 = 204 matches = 408 games
for i in 0 1 2 3 4 5; do
  python -X utf8 scripts/arena.py play \
    "bc:orc,orc,net=out/policy_v5_s2.npz" \
    "bc:base,net=out/policy_v5_s2.npz" \
    --deck-a grimmsnarl --deck-b grimmsnarl \
    --matches "$N" --archive "out/arena/e18/shard_$i.jsonl" \
    > "out/logs/p83_e18_$i.txt" 2>&1 &
done
wait
echo "=== shards done ==="
grep -h "score=" out/logs/p83_e18_*.txt
grep -h "health" out/logs/p83_e18_*.txt
