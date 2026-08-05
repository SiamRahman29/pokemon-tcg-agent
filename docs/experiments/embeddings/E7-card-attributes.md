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

⚠ The treatment differs from the control in **three** ways, not one: the 276
state attribute columns, the 9 new option columns, and — because
`pool_width` is a function of `opt_cols` — a v5 pool that is 190 wide instead
of 172. So a positive result licenses "the v6 block pays", not "the attribute
columns pay". Sub-attribution needs `--drop-a` and `--opt-cols 37`, below.

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

_(empty — to be filled once the runs land)_

## Verdict

_(empty)_
