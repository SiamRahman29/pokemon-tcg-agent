#!/usr/bin/env bash
# E19 -- pre-registered in docs/experiments/E19-clock-mechanism.md
#
#   usage: p85_e19_run.sh A|B
#
# 🔴 THE CELLS RUN ONE AT A TIME, and the reason is a sizing error worth
# recording. The first launch ran both concurrently, sized from each cell's
# SOLO speed (A 10 s/game, B 142 s/game). That is the wrong unit: the machine
# delivers a roughly fixed ~21 rollouts/s in total (E18: 187,403 rollouts in
# ~2.5 h), so the two cells together demand ~1.06M rollouts = ~14 h, and cell B
# at ~1,500 rollouts/game starves cell A. Size in ROLLOUTS, not games.
#
#   cell A  ~100 rollouts/game x 1,600 =   160k  ⇒ ~2 h on the whole box
#   cell B  ~1,500 rollouts/game x n            ⇒ 71 s/game machine-wide
#
# Cell A decides whether cell B's premise survives at all, so A goes first.
#
# ⛔ --deck-a/--deck-b grimmsnarl is NOT optional (sample collapses the trigger).
# ⚠ Launch with NO tool-level timeout; verify completion by counting shards.
set -u
cd "$(dirname "$0")/.."
mkdir -p out/arena/e19 out/logs
CELL=${1:-A}
N=${2:-0}

if [ "$CELL" = "A" ]; then
  M=${N:-0}; [ "$M" -eq 0 ] && M=134       # 6 x 134 matches = 1,608 games
  for i in 0 1 2 3 4 5; do
    python -X utf8 scripts/arena.py play \
      "bc:cap,orc,od1,net=out/policy_v5_s2.npz" \
      "bc:base,net=out/policy_v5_s2.npz" \
      --deck-a grimmsnarl --deck-b grimmsnarl \
      --matches "$M" --archive "out/arena/e19/capA_$i.jsonl" \
      > "out/logs/p85_e19_capA_$i.txt" 2>&1 &
  done
else
  M=${N:-0}; [ "$M" -eq 0 ] && M=34        # 6 x 34 matches = 408 games
  for i in 0 1 2 3 4 5; do
    python -X utf8 scripts/arena.py play \
      "bc:wp,orc,om99,ow0.5,os12,net=out/policy_v5_s2.npz" \
      "bc:base,net=out/policy_v5_s2.npz" \
      --deck-a grimmsnarl --deck-b grimmsnarl \
      --matches "$M" --archive "out/arena/e19/wpB_$i.jsonl" \
      > "out/logs/p85_e19_wpB_$i.txt" 2>&1 &
  done
fi
wait
echo "=== E19 cell $CELL done ==="
