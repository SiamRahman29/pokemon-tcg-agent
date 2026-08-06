# E5 — Round-2 planning scale probe

Status: closed; planner not promoted; distillation not opened. ⚠ The "scaling
curve" framing is **corrected** — realized compute was flat across low/medium/
high and the real dose-response is on the planner's FIRING RATE. See the
day-22 correction under "Settled verdict". EVIDENCE §8bb (renumbered from
§8aw to avoid a collision with `main`'s E8 section).

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

E5 is closed. The apparent high-arm recovery did not survive the preregistered
higher-compute check. Do not invent a fifth compute point. Do not promote any
sequencer configuration. Do not distill planner labels. `out/policy_v5.npz`
remains the frozen shipping baseline.

## 🔴 Corrected day 22: the scale axis never scaled

Total planning time per arm, from this manifest's own `sim_s`:

```text
arm      nominal cap   total planning s   plans/game   firing rate   score
low          1.0 s            652            10.2         10.8%      0.380
medium       2.0 s            616             7.2          7.4%      0.420
high         4.0 s            606             4.2          4.2%      0.515
confirm      8.0 s          8,288            31.1         35.0%      0.230
```

The cap went 1 → 2 → 4 s and realized compute went **652 → 616 → 606 s**. The
three cells that opened the confirmation gate are three draws at essentially
constant compute (n=200 each, two-cell resolution ±0.098); pooled they are
**0.4383 [0.399, 0.478] over 600 games**, already a clean loss. "Higher compute
collapses the curve" attributes the result to a variable that did not vary —
the same shape as the Crustle deck confound (`main` §8ax).

**What actually varied is the firing rate, and it is monotone across all four
arms:** 4.2% → 0.515, 7.4% → 0.420, 10.8% → 0.380, 35.0% → 0.230. The planner
plays worse the more it engages, over an 8× range, while overruling the clone at
a near-constant 58–61%. ⚠ Four cells at n=200 with no repeat: the ordering is
4/4 and the extremes separate, but adjacent steps are inside ±0.098. Direction,
not slope.

✅ `errors: 0` and `budget_aborts: 0` in **all four** arms, so confirm's 0.230 is
a healthy cell, not a degraded agent. Recorded here because a day-22 pass looked
for these counters in `out/logs/`, did not find them, and nearly filed confirm as
unauditable — they live in `out/e5/manifest.json`.

⇒ The verdict is unchanged and now rests on 600 pooled games plus a 4-for-4
ordering rather than on a three-point pattern. A fifth compute point would be
the same mistake with a larger budget.

## Memory note

Early confirm attempts OOM'd because every determinization stayed live until the
end of `plan()`. The shipped fix releases each world immediately and calls
`SearchEnd` per determinization. The settled confirm cell used that fix.
