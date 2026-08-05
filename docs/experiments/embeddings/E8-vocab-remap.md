# E8 — the v7 vocabulary remap: PAD, UNK, and 92% fewer embedding parameters

Pre-registered 2026-08-06 (day 21), before any arena game was run.
Follows [E6](E6-identity-channel.md) (identity carries weight) and
[E7](E7-card-attributes.md) (card attributes do not recover it).

---

## The audit this came from

Four candidate defects were measured on the shipped `out/policy_v5.npz`.
**One of the four was retracted before anything was built** — it is recorded
here because the retraction is the reason the intervention has the shape it
does.

### ❌ RETRACTED — "`mode="mean"` on `bag_emb` erases copy counts"

The claim was that `mean(4x Munkidori) == mean(1x Munkidori)`, and that this
destroys hand-composition information on the 57.4% of decisions whose hand
holds a duplicate (measured: 34,043 of 59,356 rows in shard 0).

**It is wrong.** The bag flats keep duplicates — a real hand reads
`[1227, 1113, 7, 303, 1152, 1121, 7]` — and `EmbeddingBag(mode="mean")` divides
by the bag LENGTH, not by the distinct count. So the pooled vector is

    mean = SUM_over_cards (count_c / n) * e_c

which is a **count-weighted average**: multiplicity survives as a proportion.
Verified directly — `[A,A,B,C]` and `[A,B,B,C]` pool to different vectors, and
`[A,A,B,C]` equals `(2 e_A + e_B + e_C) / 4` exactly. The degenerate case that
motivated the claim (`4xA` vs `1xA`) collides only because those bags are all
one card, and even then `handCount` is already a dense feature, so the two
states are separable anyway.

What is genuinely lost is only the overall scale `n`, which dense already
carries. ⇒ **not a defect, and the `--bagsum` arm that was going to be built
for it was dropped.** A sum-pool would have added nothing: `sum = n * mean`,
and `n` is known, so the appended block would have been an exact linear
function of columns the net already had.

⚠ This is [rule 15](../../../HANDOFF.md) again — a candidate defect that
dissolved on contact with the actual code. Third instance.

### ✅ DEFECT — 90% of every table ships untrained, and unseen cards read it

Per-table census over `artifacts/pds_v4` (`p53_emb_vocab.py`):

| table | rows | ever got a gradient | untrained |
|---|---|---|---|
| `slot_emb` | 1300 | 104 | **1196 (92.0%)** |
| `bag_emb` | 1300 | 134 | 1166 (89.7%) |
| `card_emb` | 1300 | 135 | 1165 (89.6%) |
| `atk_emb` | 1600 | 57 | **1543 (96.4%)** |

**88,000 embedding parameters ship; ~6,880 (7.8%) ever received a gradient.**

Waste is not the problem — the rows cost 350 KB and no decisions. The problem
is what happens when one is *read*: an untrained row is a random draw whose
norm (3.908–3.953) is **indistinguishable from a trained row's** (3.970–4.068).
So a card the corpus never contained does not arrive labelled "unknown"; it
arrives as a confident, arbitrary identity. The net has no way to tell the two
apart.

### ✅ DEFECT — row 0 is overloaded and carries a quarter of all slot lookups

`slot_emb[0]` means *empty slot*, *card id out of range*, *no stadium* and *no
effect* at once, and it is **25.5% of all slot lookups**. There is no
`padding_idx`, so it trains like a card.

Evidence the net dislikes this: it drove `|slot_emb[0]|` down to **2.337**
against a 3.958 table mean — the **11th smallest of 1,300 rows**. It taught
itself to make that row mean "nothing", which `padding_idx=0` gives for free
and exactly.

### ✅ COSMETIC — the id space is oversized

`N_CARD_IDS = 1300` against a real maximum card id of **1267**;
`N_ATTACK_IDS = 1600` against a real maximum attack id of **1556**. Subsumed by
the remap below; on its own it changes nothing.

---

## The intervention (`train_policy.py --vocab`, exported as the v7 block)

Each table is collapsed to the rows the corpus actually trained:

    row 0    PAD    empty slot / no stadium / no effect, `padding_idx=0`
    row 1    UNK    every card the corpus never contained
    row 2..  the seen ids, in ascending order

Per-table, **not shared** — a card seen in hand but never on an opponent's
board is trained in `bag_emb` and untrained in `slot_emb`, and one shared vocab
would reintroduce the defect for exactly those rows.

| table | rows before | rows after |
|---|---|---|
| `slot_emb` | 1300 | **105** |
| `bag_emb` | 1300 | 136 |
| `card_emb` | 1300 | 136 |
| `atk_emb` | 1600 | **58** |

**Embedding parameters 88,000 → 6,960 (−92.1%); total net 703k → 622k.**
The map travels inside the npz (`vocab_<table>`) and inference rebuilds the
lookup from it, so the tables and the map cannot drift apart; `policynet.load`
refuses any net where `rows != 2 + len(vocab)`.

**Every layer width is unchanged** — `state_in` is 708 on both arms, identical
to v5. Only the embedding table row counts differ. This is the cleanest A/B the
repo has run: same corpus, same rows, same seed, same architecture, same
parameter shapes everywhere downstream.

---

## 🔴 Sizing gate (rule 14), run BEFORE the arena and it is not encouraging

Where can UNK actually fire? Measured per anchor deck against the census —
Pokémon in `slot_emb`, all cards in `bag_emb`:

| anchor | field weight | Pokémon in vocab | cards in vocab |
|---|---|---|---|
| mirror (grimmsnarl) | 33.3% | 6/6 | 19/19 |
| rule:alakazam5 | 22.0% | 7/9 | 21/23 |
| rule:archaludon | 8.0% | **2/4** | 14/15 |
| rule:crustle | 6.7% | 4/4 | 24/24 |
| bc:garchomp | 6.7% | 6/6 | 20/20 |
| rule:dragapult | 5.3% | 7/7 | 21/22 |
| **rule:v10 (M Lucario)** | **4.0%** | **0/6** | **7/17** |

⚠ **The UNK half can only bite on ~12% of the weighted field** (v10 4.0% +
archaludon 8.0%). Even a large per-arm gain converts to almost nothing
weighted: +0.05 on both is +0.006 overall. **This is stated before the run so
that a positive C arm cannot later be quoted as a field-wide result.**

The **pad half is different** — id 0 is 25.5% of slot lookups in *every* game,
mirror included, so arm A measures it at full field weight on the tightest
instrument we own.

⇒ **the two halves are run as separate treatment nets**, `--vocab` (both) and
`--pad` (pad only), against one shared control.

---

## Pre-registered arms and predictions

Control `out/policy_v5c_s{0,1}.npz`; treatments `policy_v7_s{0,1}` (remap+UNK+
pad) and `policy_v7pad_s{0,1}` (pad only). Recipe identical on every arm:
`--ds artifacts/pds_v4 --epochs 12 --bs 1024 --loss listwise --state-h 512,256
--head-h 256,128 --pool --opt-cols 37`.

| arm | opponent | what it measures | prediction |
|---|---|---|---|
| A | mirror, **direct** | the pad half, field-wide | ≥ 0.500; a loss kills v7 |
| B | rule:crustle (4/4) | **negative control** — UNK cannot fire | null |
| C | rule:v10,noS (0/6) | UNK at its maximum | > B if the mechanism is real |
| D | rule:archaludon (2/4) | UNK, partial | between B and C |

**Falsifier:** if `delta(C) <= delta(B)` with both resolved, the UNK mechanism
is not what any gain is, whatever the arm A number does.

### ⚠ Two priors that argue against this, recorded now

1. **E7 pre-registered this same asymmetry shape and failed it.** Its arms
   disagreed in sign across seeds at both n=300 and n=2000. A second failure of
   the same shape would say the out-of-vocab axis is not reachable from the
   embedding tables at all.
2. **E6 measured that permuting the identity channel against v10 costs
   −0.018 [overlapping zero]** — i.e. scrambling opponent identity in the one
   matchup where UNK fires hardest changes nothing measurable. The defence is
   that E6 could only compare *random against random*: permuting rows that were
   already untrained swaps one arbitrary vector for another, so it could not
   test random-against-UNK. That defence is real but it is a defence, not
   evidence.

Against those: this arm **removes** noise rather than adding a channel that has
to be learned, and the pad half is field-wide and independent of both priors.

### Resolution, so nothing gets overread (EVIDENCE §8aq's error)

Arms B/C/D are a difference of two independent cells: **±0.080 at n=300 games,
±0.057 at n=600, ±0.022 at n=2000**. Arm A is a direct head-to-head and is 2×
tighter for the same games. Screen at n=600/cell; **a delta inside the interval
is uninformative, not null**, and only a confirmed n=2000 arm may be called
either way.

---

## Results — v7 (remap + UNK + pad), n=1500 games per cell per seed

`out/emb/e8_v7.log`, archive `out/arena/p57_e8.jsonl`.

| arm | opponent | seed 0 | seed 1 | pooled Δ |
|---|---|---|---|---|
| A | mirror, **direct** | 0.503 [0.478, 0.528] | 0.471 [0.446, 0.497] | **0.487** [0.469, 0.505] |
| B | rule:crustle (in-vocab) | +0.004 | −0.010 | **−0.003** |
| C | rule:v10 (0/6 out-of-vocab) | +0.018 | +0.023 | **+0.021** |

### ⚠ A resolution error in this script's own output, corrected

The first run printed `two-cell resolution +/-0.025` at n=1500. That is **one
cell's** width. A treatment-minus-control delta is a difference of two
INDEPENDENT cells, so its standard error is √2× a single cell's:

| | per seed (n=1500) | pooled 2 seeds (n=3000) |
|---|---|---|
| arms B/C/D/E (two-cell) | **±0.036** | **±0.025** |
| arm A (direct head-to-head) | ±0.025 | ±0.018 |

Every delta below is read against the corrected figures. This is the same class
of mistake EVIDENCE §8aq made and it was caught by recomputing rather than by
trusting the printed number.

### What the three arms say

**Arm B is the null it had to be.** −0.003 pooled, against ±0.025. Crustle is
4/4 and 24/24 in vocabulary, so UNK cannot fire, and it did not move. Without
this the C arm would be uninterpretable.

**Arm C is directionally consistent and NOT resolved.** +0.021 pooled with a
±0.025 interval — [−0.004, +0.045], which includes zero. What it does have,
and what E7 never had, is **agreement between seeds**: +0.018 and +0.023, same
sign, near-identical magnitude, against a negative control at −0.003. The
pre-registered asymmetry (C − B = **+0.024**) points the way the mechanism
requires.

⇒ **suggestive, unresolved, and not worth resolving.** Driving ±0.025 down to
±0.010 needs ~18,000 games per cell. The sizing gate already priced what that
would buy: v10 is **4.0%** of the field, so +0.021 there is **+0.0008
weighted** — a hundred minutes of arena to confirm a thousandth of a win rate.

**Arm A is a null that leans negative.** 0.487 pooled [0.469, 0.505] includes
0.500, so v7 is not measurably worse in the mirror — but the seeds disagree
(0.503 / 0.471) by about what the measured seed-only floor (0.482) produces on
its own, and the point estimate is on the wrong side.

### Weighted verdict

Over the 44.0% of field weight actually measured:

    mirror   0.333 x (-0.013)  = -0.0043
    crustle  0.067 x (-0.003)  = -0.0002
    v10      0.040 x (+0.021)  = +0.0008
                                 -------
                                 -0.0037

**v7 does not promote.** Even taking arm A at face value as a null (Δ = 0), the
total is **+0.0006** — indistinguishable from zero. The mirror carries 8.3× the
weight of the one arm that gained, so the arithmetic was decided by the sizing
gate before a game was played.

### The one result here that is not a null

**92% of the embedding parameters can be deleted for nothing.** 88,000 → 6,960
(11.5% of the whole net) costs 0.0018 of corpus fit on seed 0 and 0.0003 on
seed 1, and moves no anchor outside noise. Read against §8w — which *added*
8.2× the parameters and lost 43 decisions — capacity is now bounded from both
directions on the same net. Nothing in this project has ever been
capacity-limited.

## Results — decomposition (v7pad) and reach (arms D, E)

_Pending._
