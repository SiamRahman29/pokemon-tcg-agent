# E15 — R2: average out the bench-slot relabelling

**Frozen 2026-08-09 (day 27), BEFORE any arena game was played.**

## Hypothesis

The shipped net reads bench **slot number**, a nuisance variable with no game
meaning. Measured on the shipped `55326513` npz (`p78_symmetry_probe.py`,
EVIDENCE §8bt, after the `inPlayIndex` repair):

* **7.7%** of relabellings change the chosen option
* **16.9%** of decisions are unstable under ≥1 of 6 relabellings (MAIN 24.8%)
* keyset control 0 violations, option-order control 0/23,952

Averaging the option probabilities over K relabellings marginalises the nuisance
out. Firing rate measured before the run: **`sym8` changes 8.36% of real selects
(160 / 1,915)** against plain `bc` — this is not too rare to measure.

## Arms

| arm | spec |
|---|---|
| control | `bc:base,net=out/tmp/sa/policy_net.npz` |
| no-op control | `bc:sym1,sym1,net=…` — **proven bitwise identical to control, 0/1915 selects differ** |
| treatment | `bc:sym8,sym8,net=…` |

Direct mirror head-to-head, `--deck-a grimmsnarl --deck-b grimmsnarl`, seat-swapped.

## Pre-registered kill criterion

**`sym8` must beat the control with a 95% Wilson interval excluding 0.500 at
n ≥ 2,000 games.** If the interval covers 0.500, E15 is a **null** and does not
ship, regardless of how clean §8bt's defect is.

⚠ No training seed is involved — this is a test-time transform of a FIXED net —
so the 0.482 seed-null (§8z) and the ≥3-seeds rule do **not** apply. The relevant
null is 0.500, and the instrument is a single cell (±0.021 at n=2,000), not a
difference of two cells.

## The prior, recorded before the result

**Against.** §8bd measured the near-tie band and found it *indifferent*:
flipping the clone's k-th choice for its (k+1)-th reads **0.494 [0.467, 0.520]**.
§8bt found the unstable decisions sit in exactly that band (median top1−top2
margin **0.310** vs **1.298** overall). So the honest expectation is a null.

**For.** Variance reduction over a nuisance variable is not the same operation as
deliberately taking a worse-ranked option — §8bd moved *away* from the net's
preference, this moves *toward* the net's preference marginalised over an
arbitrary labelling. Untested either way.

⇒ **A null here is the expected outcome and must be reported as such, not
re-cut at a different K until something clears.** If it nulls, the defect stays
recorded in §8bt as a correctness finding (the code stays; see
`correctness-fixes-are-wanted-regardless-of-elo`) and the axis closes.
