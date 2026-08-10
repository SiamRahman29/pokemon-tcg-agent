#!/usr/bin/env bash
# Resolve the identical-arms control: shards 4-11 of the SAME 300-position
# sample add 200 more control positions, taking the control to k=300 and the
# SE on the order effect from 0.0047 to ~0.0027 -- enough to separate a real
# +0.005 baseline bias from noise. The bias is a third of the treatment effect,
# so this decides whether E17's headline is +0.016 or +0.011.
set -u
cd "$(dirname "$0")/.."
for i in 4 5 6 7 8 9 10 11; do
  python -X utf8 scripts/p82_e17_self_oracle.py --collect --control \
    --positions 300 --pairs 50 --shard "$i" --shards 12 \
    > "out/logs/p82_collect_ctl_$i.txt" 2>&1 &
done
wait
