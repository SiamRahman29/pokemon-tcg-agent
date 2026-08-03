# E5 — Round-2 planning scale probe

Status: first scale curve cleared the continue gate; one confirmation cell is
preregistered and not yet run.

## Hypothesis

The repaired B4 turn sequencer may improve as hidden-state averaging receives
more compute. Round 2 changes the available budget enough to test a scaling
claim, but it does not erase B4's measured loss at the small setting.

This is a re-probe under a new resource regime, not a revival of the old
verdict. The algorithm stays fixed: propose BC-guided continuations, simulate
the opponent's complete reply turn, and score when control returns. Only the
number of determinizations and its proportional wall-time cap change.

Distillation is conditional. No planner output trains a policy unless the
planner first beats frozen v5 in the arena.

## Frozen baseline

- Policy: `out/policy_v5.npz`
- SHA-256:
  `26c681c4845a7eb017def4ee5d353bbedd128767bfc034cb7091e95e0949849e`
- Deck: `grimmsnarl`
- Control: the same v5 checkpoint and shipped rule defaults, without sequencing
- Treatment: the same agent plus `seq,reply`
- Primary instrument: paired Grimmsnarl mirror

The sequencer must use the explicit `net=` checkpoint supplied to its owning
agent. The bundled `agents/sa/policy_net.npz` is not v5, so silently loading the
module singleton inside planning would confound the scale experiment.

## Fixed scale points

Candidate proposal count remains fixed at `K=8`. The first action is sampled
from the BC top three, so increasing `K` beyond coverage of those actions is not
a clean compute axis. Only hidden-state averaging scales:

```text
arm       determinizations    cap per eligible MAIN select
low                4                      1.0 s
medium             8                      2.0 s
high              16                      4.0 s
confirm           32                      8.0 s   # preregistered after curve
```

Each arm plays the no-sequencer control for 100 paired matches / 200 games.
Do not tune `K`, the evaluator, reply policy, top-k continuation sampling, or
time ratios after seeing outcomes.

The initial `n=200` cells estimate the direction of the scale curve; they do not
promote an agent. The confirmation cell is one higher fixed point on the same
axis, not a search over settings.

## Commands

```powershell
python -X utf8 scripts/p50_e5_smoke.py
python -X utf8 scripts/p51_e5_scale.py --arms low,medium,high --force
```

Confirmation, only after the first curve's continue gate:

```powershell
python -X utf8 scripts/p51_e5_scale.py --arms confirm
```

The smoke's game outcomes are deliberately not reported or used as a strength
screen. It verifies checkpoint injection, legal execution, planning activity,
fallback counters, and realized compute.

## Integrity smoke

The three two-game execution cells passed:

```text
E5_SMOKE_OK
low     planned=38  overruled=19  aborts=0  0.299 s/completed
medium  planned=44  overruled=23  aborts=0  0.408 s/completed
high    planned=76  overruled=55  aborts=0  0.805 s/completed
fallbacks=0
```

## First scale curve

All three preregistered cells completed at `n=200`, with archives and arena
score lines matching `out/e5/manifest.json`:

```text
arm     M    cap   score               completed  s/plan  aborts
low     4   1.0s   0.380 [0.316,0.449]     2046    0.319      0
medium  8   2.0s   0.420 [0.354,0.489]     1442    0.427      0
high   16   4.0s   0.515 [0.446,0.583]      837    0.724      0
```

Point estimates are non-decreasing. Realized work per completed plan rises at
every step with zero budget aborts and zero sequencer errors. Absolute completed
plans fall as each plan gets more expensive; that is recorded as a diagnostic,
not as license to retune the axis.

The low cell reproduces B4's repaired loss almost exactly (`0.375` then;
`0.380` now against frozen v5). The high cell's point estimate crosses 0.5, but
its interval still includes 0.5, so it is not a promotion candidate.

## Gates

1. **Integrity:** the planner holds the exact explicit v5 net object; both
   agents finish the smoke with zero policy fallbacks and zero missing-net
   events; the treatment produces at least one completed plan. ✅
2. **Mechanism:** realized sequencer time per completed plan must increase from
   low to medium to high. A point that only spends a larger cap without doing
   more work is not a scale point. ✅ (`0.319 → 0.427 → 0.724`)
3. **Scale signal:** run all three preregistered `n=200` cells. E5 closes locally
   if the point estimates are not non-decreasing, or if the high arm's 95%
   upper bound is at or below 0.5. If they are non-decreasing and the high arm
   is above 0.5, preregister one higher-compute confirmation before running it.
   Do not select the best of the three as a candidate. ✅ continue
   (`0.380 → 0.420 → 0.515`; high upper bound `0.583`)
4. **Confirmation:** the preregistered `confirm` arm (`M=32`, `sb=8.0`, `n=200`)
   must keep the monotone direction. If its point estimate falls below `high`,
   or its 95% upper bound is at or below 0.5, close E5 as a local near-miss.
   If it stays at or above `high` and its interval excludes 0.5, escalate that
   fixed configuration to the planner-promotion gate. Do not invent a fifth
   compute point after seeing confirmation.
5. **Planner promotion:** a fixed planner configuration must beat v5 with a 95%
   interval excluding 0.5 at `n >= 1,000`, then survive the weighted anchor set
   at `n >= 2,000`. Approximately `+50 Elo` weighted remains the shipping bar.
6. **Distillation:** only a promoted planner may label trajectories. A distilled
   treatment must be compared with a byte-identical-data self-distillation
   control before any shipping claim.

## Prior evidence and limits

B4's repaired one-reply form scored `0.375 [0.311, 0.444]` against v3 at
`n=200` with a 1.0 s cap, while the matched no-reply arm scored
`0.165 [0.120, 0.223]`. The design repair was real and the resulting agent still
lost. The E5 low cell matches that loss against v5; the question now is whether
the confirmation point continues the recovery past noise.

Local CPU measurements cannot establish H100 or 16-vCPU scaling. They can kill
or continue the hypothesis before any Round-2 engineering spend.
