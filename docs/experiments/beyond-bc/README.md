# Beyond-BC experiment index

Branch: `experiments/beyond-bc`

Goal: strengthen the shipped behavioral clone without destabilizing its fast,
legal action prior. First-round work must remain compatible with the CPU
submission; expensive planning is a post-freeze Round-2 track.

Start each new experiment session with `docs/experiments/beyond-bc/HANDOFF.md`;
do not use the repository-wide handoff as this program's task list.

## Baseline

- Policy: `out/policy_v5.npz`
- Training corpus: `artifacts/pds_v4`
- Architecture: state `[512, 256]`, option head `[256, 128]`, listwise loss,
  v3 option features, v4 state features, v5 option-set pool
- Logged corpus: 248,985 decisions from 1,603 games
- Strength instrument: weighted seven-anchor arena, not leaderboard score
- Promotion gate: screen cheaply, confirm at `n >= 2,000`; approximately
  `+50 Elo` weighted is the First Round shipping bar

## Experiment order

1. `E1` — shared state encoder with outcome and selection-count auxiliary heads
2. `E3` — uncertainty-gated DAgger collection and human review
3. `E2` — observable matchup routing and residual adapters
4. `E4` — support-constrained outcome learning, conditional on E1 and E3
5. `E5` — BC-guided planning and distillation after the First Round freeze

The numbering follows the architecture canvas; execution order follows
dependencies.

Current state:

- E1 is settled null across outcome, count, and combined auxiliary arms.
- E3 has a 160-item uncertainty queue from the fresh rating-977 v5
  trajectories, but is paused because neither a qualified human teacher nor a
  validated stronger automated teacher is available. The 15-label pilot is
  audit-only and must not train.
- E2 is settled null: hard-routed mirror/Alakazam residual adapters left
  general-route agreement untouched, but failed both strength screens
  (`0.521 [0.490, 0.552]` mirror; Alakazam treatment 0.782 vs control 0.792).
- E4 remains blocked. `out/policy_v5.npz` is still the frozen baseline.
- E5's first scale curve continues: `0.380 → 0.420 → 0.515` at `n=200`, with
  rising realized work per plan. High still overlaps 0.5, so one confirmation
  cell (`M=32`, `8.0s`) is preregistered and not yet run. Distillation remains
  gated on a planner strength win.

## Required user handoffs

Completed:

1. Two fresh decision-identical v5 replay populations at ratings 897 and 977.
2. Private Kaggle GPU run for E1 and returned result archive.

Still planned:

1. A qualified TCG reviewer, if E3 is ever resumed.
2. Approval before any Kaggle submission, remote push, or final merge.

## Record format

Each experiment gets a Markdown record in this directory containing:

- hypothesis and intervention;
- exact control and treatment commands;
- code/data/checkpoint hashes;
- train diagnostics;
- arena sample size, score, and confidence interval;
- pre-registered pass/fail rule;
- settled verdict.

Only settled findings are copied into `report/EVIDENCE.md`,
`report/STRATEGY.md`, and the architecture canvas.

Large datasets, checkpoints, replays, and raw arena logs stay under the ignored
`artifacts/`, `replays/`, and `out/` trees.
