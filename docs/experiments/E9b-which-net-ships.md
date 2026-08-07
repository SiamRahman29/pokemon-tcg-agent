# E9b — which net goes in the free slot

**Frozen 2026-08-07 (day 25), BEFORE the ens5 cell reported.** At the time of
writing, ens5 vs `policy_v5_s1` stood at ~456/1,400 games with no score line
printed and none read. The point of this file is that the rule below cannot be
chosen after seeing the number it consumes.

## Standing

The user has authorised a submission into the free active slot ("wait for ens5
then submit the better one"). The slot currently holds `55169114` (917.2), our
weakest active agent; a new submission evicts it and **retains** `55321893`
(954.3), so the board keeps showing the max of the two. **Downside of submitting
is zero; the only question is which bundle.**

## What is already measured (shipped config, `--no-rules` both arms, mirror,
direct, seat-balanced, n=1,400/cell)

| cell | score | 95% CI |
|---|---|---|
| ens2 vs `policy_v5` | 0.559 | [0.533, 0.585] |
| `policy_v5_s1` vs `policy_v5` | 0.531 | [0.505, 0.557] |
| ens2 vs `policy_v5_s1` | 0.505 | [0.479, 0.531] (null) |
| **`policy_v5_s2` vs `policy_v5_s1`** | **0.537** | [0.511, 0.563] |

Ranking so far: **`s2` > `s1` > `v5`**. All ten seed pairs agree on 80.4–81.4%
of 12,939 held-out decisions, so no member is correlated enough to poison a vote
(§8bf).

## The decision rule

Let **X** = ens5 vs `policy_v5_s1`, the cell in flight.

1. **X resolves above 0.500** (lower CI bound > 0.500) ⇒ **ship ens5.**
2. **X does not resolve above 0.500** (CI spans it, or resolves below) ⇒
   **ship `policy_v5_s2` alone.**

No other outcome branches. In particular there is **no** branch that re-runs
ens5 against `s2` before submitting — that cell would cost ~25 min and the
leaderboard cannot resolve the difference it would settle (§8ak: two
decision-identical agents read 63–87 points apart, against a largest-ever
candidate gap here of ~25 Elo). If it is worth measuring, it is worth measuring
**after** the slot is filled, not instead of filling it.

## Why the tiebreak favours ens5 in branch 1 even though `s2` may score higher

`s2` won a screen of three seeds against a member that was itself selected as
the better of an earlier pair. **Selection inflates the winner**, so `s2`'s 0.537
is biased upward by an unknown amount and honestly owes a confirmation run on
fresh games before anyone quotes it. Its margin does survive a Bonferroni
correction for three comparisons (p≈0.016), so this is a caveat, not a
retraction.

**Averaging all five members is a pre-specified rule containing no selection at
all.** Whatever ens5 measures is unbiased and owes nothing. Between two options
whose true difference the ladder cannot resolve anyway, the one without a
confirmation debt is worth more than a slightly higher point estimate with one.

In branch 2 that argument is unavailable — a vote that cannot beat a member is
not worth shipping over the best member — so `s2` ships despite its debt, and
the debt gets recorded rather than hidden.

## Predictions, recorded so they can be wrong

- **X lands in [0.52, 0.56].** Reasoning: averaging 5 independent functions
  should cancel more fitted variance than averaging 2, and ens2 vs `s1` read
  0.505 — but two of ens5's five members (`v5`, `s1`) are the weakest two we
  own, which caps how far the average can move.
- **ens5 will NOT beat `s2` by a resolved margin** if that cell is ever run.
  A vote's members bound it; three good members cannot carry two weak ones far
  past the best of them.

## Not covered by this file

Latency is settled, not assumed: ens5 measured **6.7 ms/move mean, 34.4 ms max,
598.8 s of the 600 s pool unspent**. Member count is not cost-constrained.

The bundle must still pass the extracted-tarball smoke (`MEMBERS=5 want=5`,
distinct member md5s, dim guard, `RESULT=0`) before submission. A smoke failure
overrides both branches: **nothing ships that has not run from an extracted
bundle the way Kaggle loads it.**
