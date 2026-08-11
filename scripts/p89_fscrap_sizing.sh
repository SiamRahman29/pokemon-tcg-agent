#!/usr/bin/env bash
# p89 -- ON-POLICY sizing for E21's VOID arm (`fscrap`), in the matchups a cell
# could actually run in.
#
#   usage: p89_fscrap_sizing.sh [matches-per-anchor]
#
# §8cc killed E21b for a reason that was already in the file: our 60 runs ZERO
# Pokemon Tools, so in the MIRROR "a Tool is on their board" is unsatisfiable by
# construction and the rule fired 0/3082. Its verdict was
# **"size the condition IN THE MATCHUP THE CELL WILL RUN IN"** -- this is that,
# run before any cell is proposed.
#
# ON-POLICY on purpose: the flag is ON in every run here, so `fetch_fired` is
# what the rule would really do, not an off-policy replay estimate. §8cc measured
# that replay sizing UNDER-predicts (0.72 realized vs 0.461 sized, 1.6x).
#
# ⚠ The scores printed here are n=200 and are NOT a result -- 95% CI is ~±0.07.
# This script answers "does the condition ever arise", nothing else.
set -u
cd "$(dirname "$0")/.."
mkdir -p out/arena/p89 out/logs
M=${1:-100}
NET=out/policy_v5_s2.npz

# anchor spec | deck module | tool count in that 60
ANCHORS=(
  "rule:v10,noS|lucario_v10"
  "rule:lucario|mega_lucario_ex"
  "rule:crustle|crustle_v1"
  "rule:archaludon|archaludon_ex"
  "rule:dragapult|dragapult_ex"
)

for row in "${ANCHORS[@]}"; do
  spec=${row%%|*}
  deck=${row##*|}
  python -X utf8 scripts/arena.py play \
    "bc:p89,fscrap,net=$NET" "$spec" \
    --deck-a grimmsnarl --deck-b "$deck" \
    --matches "$M" --archive "out/arena/p89/${deck}.jsonl" \
    > "out/logs/p89_${deck}.txt" 2>&1 &
done
wait
echo "=== p89 fscrap sizing done (${M} matches = $((M*2)) games per anchor) ==="
for row in "${ANCHORS[@]}"; do
  deck=${row##*|}
  echo "--- $deck"
  grep -h "^A=" "out/logs/p89_${deck}.txt"
  grep -h "\[health\]" "out/logs/p89_${deck}.txt"
done
