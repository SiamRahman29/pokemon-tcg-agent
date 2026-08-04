# E5 — Round-2 planning scale probe

Status: settled null / local near-miss; planner not promoted; distillation not
opened.

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

## Fixed scale points

```text
arm       determinizations    cap per eligible MAIN select
low                4                      1.0 s
medium             8                      2.0 s
high              16                      4.0 s
confirm           32                      8.0 s
```

`K=8` stayed fixed. No result-dependent retuning was allowed.

## Scale results

```text
arm     M    cap   score               completed  s/plan  aborts
low     4   1.0s   0.380 [0.316,0.449]     2046    0.319      0
medium  8   2.0s   0.420 [0.354,0.489]     1442    0.427      0
high   16   4.0s   0.515 [0.446,0.583]      837    0.724      0
confirm 32  8.0s   0.230 [0.177,0.293]     6228    1.331      0
```

Archives: 200 games each under `out/arena/e5_*_vs_control.jsonl`. Manifest:
`out/e5/manifest.json`.

## Settled verdict

Gate readings:

1. Integrity and mechanism passed on the first curve
   (`0.319 → 0.427 → 0.724` s/plan).
2. The first curve continued (`0.380 → 0.420 → 0.515`).
3. Confirmation failed both fail rules: point estimate fell below high
   (`0.230 < 0.515`), and the 95% upper bound stayed at or below 0.5
   (`0.293`).

E5 is closed as a local near-miss. The apparent high-arm recovery did not
survive the preregistered higher-compute check. Do not invent a fifth compute
point. Do not promote any sequencer configuration. Do not distill planner
labels. `out/policy_v5.npz` remains the frozen shipping baseline.

Confirm also completed far more plans and overrules than high
(`6228` / `3647` vs `837` / `503`) while spending `1.331` s per completed plan.
More override of the clone at higher compute correlated with a large loss, which
is consistent with the search-selects-noise failure mode rather than a monotone
compute curve.

## Memory note

Early confirm attempts OOM'd because every determinization stayed live until the
end of `plan()`. The shipped fix releases each world immediately and calls
`SearchEnd` per determinization. The settled confirm cell used that fix.
