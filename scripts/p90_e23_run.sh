#!/usr/bin/env bash
# E23 -- pre-registered in docs/experiments/E23-fscrap-anchor.md.
#
#   usage: p90_e23_run.sh a|b
#
# Repairs E21's VOID arm: `fscrap` never fired in the mirror because our 60 runs
# zero Pokemon Tools (§8cc). Each cell is a TWO-CELL DELTA against the same
# anchor -- treatment arm and byte-identical control arm -- so the driver runs
# both arms of a cell and the delta's interval is sqrt(2)x a single cell's
# (§8aw), which the scorer prints rather than the single-cell width.
#
# 4 shards x M matches per arm; each shard gets its OWN --archive (HANDOFF's
# standing rule: concurrent processes must never append to games.jsonl).
# Deliberately 4 shards on a 6-core box: a second agent is working this repo.
#
# ⚠ Pool from arena's own printed W/D/L lines, NEVER by re-deriving from the
# archives -- rule 18.
set -u
cd "$(dirname "$0")/.."
mkdir -p out/arena/e23 out/logs
CELL=${1:-a}
NET=out/policy_v5_s2.npz

case "$CELL" in
  a) ANCHOR="rule:v10,noS"; DECKB=lucario_v10;   M=500 ;;   # 4,000 games/arm
  b) ANCHOR="rule:archaludon"; DECKB=archaludon_ex; M=250 ;; # 2,000 games/arm
  # b2: an exact REPLICATION of cell b on fresh games -- same anchor, same deck,
  # same n, nothing tuned. Added after b read -0.0317 (z=-2.24) on the harmful
  # branch, under the decision rule frozen in the experiment doc BEFORE it ran.
  b2) ANCHOR="rule:archaludon"; DECKB=archaludon_ex; M=250 ;;
  *) echo "cell must be a, b or b2"; exit 2 ;;
esac

for ARM in fscrap base; do
  if [ "$ARM" = fscrap ]; then A="bc:e23,fscrap,net=$NET"; else A="bc:base,net=$NET"; fi
  for i in 0 1 2 3; do
    python -X utf8 scripts/arena.py play "$A" "$ANCHOR" \
      --deck-a grimmsnarl --deck-b "$DECKB" \
      --matches "$M" --archive "out/arena/e23/${CELL}_${ARM}_$i.jsonl" \
      > "out/logs/p90_e23_${CELL}_${ARM}_$i.txt" 2>&1 &
  done
  wait
  echo "=== E23 cell $CELL arm $ARM done ==="
done

echo "=== E23 cell $CELL: raw lines ==="
for ARM in fscrap base; do
  echo "--- arm $ARM"
  grep -h "^A=" out/logs/p90_e23_${CELL}_${ARM}_*.txt
  grep -h "\[health\]" out/logs/p90_e23_${CELL}_${ARM}_*.txt
done
python -X utf8 scripts/p90_e23_score.py "$CELL"
