#!/usr/bin/env bash
# E21 -- pre-registered in docs/experiments/E21-petrel-fetch.md, frozen at 5be502d.
#
#   usage: p87_e21_run.sh a|b [matches-per-shard]
#
# 4 shards x 250 matches = 2,000 games per cell. Each shard gets its OWN
# --archive: HANDOFF's standing rule is that concurrent processes must never
# append to out/arena/games.jsonl.
#
# ⚠ Pool from arena's own printed W/D/L lines, NEVER by re-deriving from the
# archives -- that is the seat-indexing bug rule 18 was written about and the
# other agent's e16a07d fixed for the Kaggle harness.
#
# Deliberately 4 shards on a 6-core box: a second agent is working this repo.
set -u
cd "$(dirname "$0")/.."
mkdir -p out/arena/e21 out/logs
CELL=${1:-a}
M=${2:-250}
NET=out/policy_v5_s2.npz

case "$CELL" in
  a) FLAG=fstad ;;
  b) FLAG=fscrap ;;
  *) echo "cell must be a or b"; exit 2 ;;
esac

for i in 0 1 2 3; do
  python -X utf8 scripts/arena.py play \
    "bc:e21,$FLAG,net=$NET" \
    "bc:base,net=$NET" \
    --deck-a grimmsnarl --deck-b grimmsnarl \
    --matches "$M" --archive "out/arena/e21/${CELL}_$i.jsonl" \
    > "out/logs/p87_e21_${CELL}_$i.txt" 2>&1 &
done
wait
echo "=== E21 cell $CELL done ==="
grep -h "^A=" out/logs/p87_e21_${CELL}_*.txt
grep -h "\[health\]" out/logs/p87_e21_${CELL}_*.txt
