#!/usr/bin/env bash
# E17 collection. `fastsearch` keeps process-global state, so shard by PROCESS.
#
# 🔴 Concurrency is capped at 6 and the two phases run SEQUENTIALLY. The first
# attempt launched 10 shards at once; each was parsing all 385 MB of
# `submission_v5_s2` into one list, and on a 7.9 GB box they starved each other
# into skipping most of the population *silently*. Collection now streams one
# replay at a time (`index_positions`) and refuses to emit a shard that skipped
# >2% of candidates.
#
# Control shards are drawn from the SAME --positions 300 sample as the treatment
# (shards 0-3 of 12 = 100 of the same positions), so the winner's-curse control
# is measured on the same board states it licenses.
set -u
cd "$(dirname "$0")/.."
mkdir -p out/logs
rm -f out/logs/p82_e17_trt_*.json out/logs/p82_e17_ctl_*.json

echo "=== phase 1: treatment, 6 shards ==="
for i in 0 1 2 3 4 5; do
  python -X utf8 scripts/p82_e17_self_oracle.py --collect \
    --positions 300 --pairs 50 --shard "$i" --shards 6 \
    > "out/logs/p82_collect_trt_$i.txt" 2>&1 &
done
wait
tail -n 1 out/logs/p82_collect_trt_*.txt

echo "=== phase 2: identical-arms control, 4 shards ==="
for i in 0 1 2 3; do
  python -X utf8 scripts/p82_e17_self_oracle.py --collect --control \
    --positions 300 --pairs 50 --shard "$i" --shards 12 \
    > "out/logs/p82_collect_ctl_$i.txt" 2>&1 &
done
wait
tail -n 1 out/logs/p82_collect_ctl_*.txt
echo "=== all shards done ==="
