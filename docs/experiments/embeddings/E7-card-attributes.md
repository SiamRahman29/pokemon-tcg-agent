# E7 — card attributes as an identity channel that transfers

Status: **PRE-REGISTERED.** Written before any arena number exists. Nothing
below may be edited after results land except the Results and Verdict sections.

## Hypothesis

E6 measured that identifying the opponent's Pokemon is worth ~0.25 win rate
where the corpus supports it (`rule:crustle`, 4/4 of its Pokemon in vocabulary)
and ~0 where it does not (`rule:v10`, 0/6 in vocabulary). A per-card embedding
row can only ever describe cards the corpus contained.

Card *attributes* come from the engine's card DB, which covers all 1,267 cards.
An unseen Pokemon therefore arrives as "Fighting, weak to Psychic, Stage 1,
340 HP, has an ability" instead of an untrained N(0, 1) vector.

**Prediction: v6 gains more against `rule:v10` than against `rule:crustle`.**

That asymmetry is the whole claim. A uniform gain across all anchors would be
a better feature block, which is worth having, but it would **not** be evidence
for the out-of-vocabulary mechanism — the attributes would simply be carrying
information the dense features lacked for every opponent alike.

## Why the mechanism is not guaranteed

Checked before building, and it is thinner than the story wants:

- All six Lucario Pokemon are `energyType=6` (Fighting) with `weakness=5` or
  `1`, and all of those values do occur in the corpus.
- But `weakness=5` appears on exactly **one** trained card, and `energyType=6`
  on **five**. The columns that must carry Mega Lucario have single-digit
  support behind them.

So the channel exists. Whether there is enough gradient behind it to matter is
the open question, and a null here is a real possible outcome.

There is also a confound this experiment cannot remove: the corpus contains no
Lucario games at all. "Never trained on the matchup" remains a sufficient
explanation for the weakness on its own. A v6 gain against `rule:v10` would
show the attribute channel transfers; it would not show the vocabulary gap was
the *only* cause.

## Intervention

`--attr` appends 276 state columns (12 slots x 23) and widens the option vector
from 37 to 46 with a `cardType` one-hot plus two target flags. Sized first
(`scripts/p55_attr_sizing.py`); the gate killed `aceSpec` (one value corpus-wide)
and `pokemonType`/`evolutionType` (fully redundant with the six stage/ex flags
already encoded).

Corpus `artifacts/pds_v6`, 248,985 rows — the same count as `pds_v4`, with
`dense`, `xdense` and `opt_dense[:, :37]` verified byte-identical to it. The
control therefore trains on **identical rows**.

⚠ The treatment differs from the control in **four** ways, not one: the 276
state attribute columns, the 9 new option columns, — because `pool_width` is a
function of `opt_cols` — a v5 pool that is 190 wide instead of 172, and
consequently **more parameters: 855,745 against 702,913**. So a positive result
licenses "the v6 block pays", not "the attribute columns pay". Sub-attribution
needs `--drop-a` and `--opt-cols 37`, below.

The capacity difference is worth naming rather than waving away, though EVIDENCE
8w measured 8.2x the parameters buying **-43 decisions** on this corpus, so
extra width is not a free win here and a +22% param count is unlikely to be the
mechanism on its own.

This does **not** weaken the mechanism test in rule 4. Arms B and C use the
same treatment net, so the confound is constant across them and cancels out of
`gain(C) - gain(B)`.

## Arms

Treatment and control are trained by the same recipe on the same corpus:

```
--epochs 12 --bs 1024 --loss listwise --state-h 512,256 --head-h 256,128 --pool
treatment:  --attr                 (opt_cols 46, state 1002)
control:    --opt-cols 37          (opt_cols 37, state  726 == v5)
```

Seeds 0 and 1 for both, because the measured seed-only floor is +/-13 Elo
(~+/-0.019 win rate) and a single-seed result below that is claiming nothing.

⚠ **Checkpoint selection is by best `val_top1`, not `--export-last`** — so the
two arms may export different epochs, and part of any difference could be epoch
count rather than the intervention. This matches how v4-vs-v4ctrl and v5 were
compared, so it is the repo's standing precedent for two *pure clones* sharing
one objective; EVIDENCE 8ao's demand for `--export-last` is specifically for
arms whose objective departs from corpus fit (advantage weighting), which this
is not. Recorded here rather than discovered later. v6 seed 0 exported **epoch
10 of 12** (`val_top1` 0.7190).

⚠ And rule 3 still binds: `val_top1` does **not** predict strength in either
direction (8z moved it by 8 decisions for +37 Elo; 8aa by 214 for +14). No
verdict below may cite it as evidence of anything.

## Instrument

Every arm runs its own control **back to back in the same session**. No arm is
compared against a stored number: `rule:crustle`'s code changed on 08-02 and
moved a fixed net's score by +0.100 with no change to our side, and the archive
records both eras under one unversioned name.

| arm | opponent | opp Pokemon in vocabulary | role |
|---|---|---|---|
| A | mirror, direct v6 vs control | 19/19 (ours) | 33.3% field weight; a difference measured *within* one run |
| B | `rule:crustle` | 4/4 | does it hurt where identity already works? |
| C | `rule:v10,noS` | **0/6** | **the hypothesis** |
| D | `rule:alakazam5` | 21/23 | 22% field share; generalisation |

## Decision rule, fixed in advance

1. Screen every cell at n=300. Confirm at n=2,000 any arm screening >= +0.03.
2. The reported effect per arm is the **mean over the two seeds**, with the
   seed spread reported beside it. A gain smaller than the spread is a null.
3. **Promote** only if arm A's confirmed CI excludes 0.500 in v6's favour, and
   no anchor arm regresses with disjoint CIs.
4. **Mechanism confirmed** only if `gain(C) > gain(B)` with non-overlapping
   intervals. Otherwise report as a feature-block result and explicitly retract
   the out-of-vocabulary story.
5. A null is a publishable result and gets written up as one. This repo has
   shipped clean nulls before (E1, E2, B8) and the E6 permutation numbers stand
   on their own regardless of what E7 does.

## Sub-attribution, only if the block passes

`--drop-a` zeroes any member of `features.A_GROUPS`
(`attrEnergyType`, `attrWeakness`, `attrAbility`, `attrResist`, `attrWeakHit`)
and records the surviving mask in the npz, so a drop-one arm needs no corpus
rebuild. Not run before the block clears, because five extra retrains against a
block that measures null is wasted compute.

EVIDENCE 8ab is the reason this handle exists at all: the v4 block shipped
whole, and when it was finally decomposed its five leftover members measured
**-22 Elo against having no block at all**.

## Results

Training: v6 855,745 params, control 702,913. Best `val_top1` — v6 0.7190 /
0.7152, control 0.7163 / 0.7139 (seeds 0/1). **v6 fits the corpus marginally
better on both seeds.** Rule 3 says that predicts nothing, and it did not.

### Arm A — mirror, direct (n=300 per seed)

| seed | v6 vs its own control |
|---|---|
| 0 | 0.533 [0.477, 0.589] |
| 1 | 0.487 [0.431, 0.543] |
| **pooled n=600** | **0.510 [0.470, 0.550]** |

CI includes 0.500. **Rule 3 not met.**

### Arms B / C / D — screen at n=300 per cell

| arm | opponent | seed 0 | seed 1 | mean | seed spread |
|---|---|---|---|---|---|
| B | `rule:crustle` | −0.007 | −0.028 | −0.018 | 0.021 |
| C | `rule:v10,noS` | −0.030 | **+0.073** | +0.021 | **0.103** |
| D | `rule:alakazam5` | +0.023 | −0.027 | −0.002 | 0.050 |

🔴 **A design error, and it is the most useful thing in this record.** B/C/D are
a *difference of two independent cells*, so at n=300 the 95% half-width on each
delta is **+/-0.080**. Every number in that table is inside it. These rows are
not evidence of a null — **they are uninformative**, and arm C's 0.103 sign-
flipping seed swing is exactly what that resolution predicts. Arm A's direct
design is **2x tighter for the same number of games**, which is `p33`'s `direct`
flag earning its keep.

### Arm C — confirmed at n=2,000 per cell

| seed | v6 | control | delta |
|---|---|---|---|
| 0 | 0.602 [0.580, 0.623] | 0.616 [0.595, 0.637] | −0.014 |
| 1 | 0.595 [0.573, 0.616] | 0.571 [0.549, 0.592] | +0.024 |
| **pooled** | **0.5985** | **0.5935** | **+0.005 [−0.017, +0.027]** |

Includes zero.

## Verdict

🔴 **E7 is a CLEAN NULL. v6 does not promote.** Arm A's confirmed CI includes
0.500 (rule 3), and the hypothesis arm resolves to +0.005 [−0.017, +0.027] over
4,000 games a side.

🔴 **The mechanism is NOT confirmed, and per rule 4 the out-of-vocabulary story
is retracted for this intervention.** It is *not* falsified either — the two
seeds disagreed in sign at both sample sizes, so what E7 establishes is that
**card attributes as implemented do not recover the identity channel**, not that
identity is unimportant. E6 stands untouched: permuting opponent card ids still
costs 0.838 → 0.587 against Crustle.

### ⚠ A correction made inside this record

After seed 0 alone this was written up mid-session as *"the pre-registered
prediction is falsified... fails on direction"*. **Seed 1 reversed the sign
(−0.030 → +0.073) and that characterisation was wrong.** It is the same rule-1
error §8an documents from an n=6 smoke, committed by the same discipline that
wrote the rule down. One seed of a two-cell delta at n=300 licenses nothing.

### Why it probably failed, stated as hypothesis not fact

The 18 per-slot dense features already carry HP, max HP, damage taken, stage,
ex/megaEx/tera, energy count, retreat cost, prize value and estimated damage —
all correct for *any* card in the DB. `energyType` and `weakness` add little on
top of that, while 276 columns that are ~83% zero at any given decision add
variance. That is the EVIDENCE 8ab failure mode, where five leftover v4 columns
measured **−22 Elo against having no block at all**.

And the support was named as thin before the run: `weakness=5` — Mega Lucario's
— has **one** trained card behind it, `energyType=6` has **five**.

### ⚡ A methods finding worth more than the null

Against `rule:v10` at n=2,000, the two **control** seeds read 0.616 and 0.571 —
a **0.045** spread — while the two treatment seeds read 0.602 and 0.595, a 0.007
spread. §8z's seed-only floor of ±0.019 was measured **mirror-direct**, and this
is a caution that it should not be assumed to carry to third-party anchors. Two
seeds cannot characterise a variance, so this is a flag for the next design, not
a number to quote.

### What this leaves

The corpus contains **zero** Lucario games. E7 tried to repair an unseen
archetype by re-encoding cards; it did not work. The remaining untested
explanation is the simpler one — that the fix for an archetype you have never
observed is **training data containing it**, not a better encoding of its cards.
That is a data question, not an embedding question, and it is where the E6
diagnosis actually points.

`--drop-a` sub-attribution is **not run**, per the pre-registration: five extra
retrains against a block that measures null is wasted compute.
