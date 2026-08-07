# E11 — the bench-development gap (pre-registered 2026-08-07, day 25, 3rd session, BEFORE any cell ran)

**Found by `p70_perturn_sweep.py`, which exists because §8bj's rule 21 said the
per-DECISION ranking picks the wrong clusters — and then F1 was closed after
examining only the two clusters that ranking had already chosen.** This is the
first candidate this project has produced where **we are measurably worse than
the 1150+ pilots at something ordering-free.**

## The measurement (already made, not part of the test)

`Buddy-Buddy Poffin` (PLAY): *"search your deck for up to 2 Basic Pokémon with
70 HP or less and put them onto your Bench."* Share of AVAILABLE TURNS in which
the card is actually played, conditioned on our own board occupancy
(active + bench, of 6 slots), mirror games only:

| board size | expert turns | expert plays | our turns | our plays | gap |
|---|---|---|---|---|---|
| 1 | 94 | 98.9% | 18 | 100.0% | +1.1% |
| 2 | 63 | 96.8% | 20 | 85.0% | −11.8% |
| 3 | 76 | 84.2% | 26 | 69.2% | −15.0% |
| **4** | **114** | **70.2%** | **51** | **29.4%** | **−40.8%** |
| **5** | **147** | **46.9%** | **69** | **7.2%** | **−39.7%** |

Pooled: **0.80 fewer plays per game**, above the 0.5 firings/game sizing gate
that killed Morgrem (0.2), Pokégear (0.27) and the Archaludon rule (0.187).
Sources: `artifacts/pds_mirror_exp` (257 expert mirror games) and
`artifacts/pds_ours_mirror1` (80 recorded games of the shipped agent, one seat).

⚠ **Confound checked before pre-registering:** both sides decline at the same
mean board occupancy (4.46 expert, 4.45 ours), so this is not a different mix of
situations — conditioned on the identical board, the behaviours differ.

## Why this is tested rather than closed by rule 11

Rule 11 forbids building rules for **tradeoffs**, and benching a 70-HP basic *is*
a tradeoff: board development against handing the mirror's own bench-snipe an
extra target. **The precedent that governs is rule 10 / `boss_veto`**, whose
docstring says it "sits between the P4b class and the P4a class … per rule 10 it
lives or dies by its A/B." Same shape here, and the same resolution: **one cheap,
seed-cancelling A/B decides it.** ⛔ If it is null, the finding is a chapter and
no rule ships — the deviation is running the test, not lowering the bar.

## The intervention

`targeting.poffin_force`: in a MAIN select, if the net's top pick is **not**
`Buddy-Buddy Poffin` and a `Buddy-Buddy Poffin` PLAY option exists and our board
has **≥ 2 free slots** (board size ≤ 4), play it instead.

**Why ≥2 free and not ≥1:** the experts are themselves near a coin flip at board
5 (46.9%), so forcing there overshoots *their* behaviour. The rule targets the
bucket where the gap is largest and the experts are clearly committed (board 4:
70.2% vs our 29.4%). Deliberately conservative; a wider variant is a separate
experiment, not a knob to tune after seeing the result.

## The design (frozen)

- `bc` both arms with the **byte-identical** net `out/policy_v5_s2.npz`, rule
  toggled — so the ±13 Elo seed nuisance (§8bk) cancels **exactly**.
- Shipped configuration otherwise: `noChip,noSpread,noSrc`.
- Mirror, DIRECT, seat-balanced, **n = 2,800** (±~0.019).

## The bar (frozen, same as F2's)

**Ship iff point ≥ 0.53 AND the 95% CI excludes 0.50.** Below → keep the shipped
agent, write the null. ⚠ If it ships, it must **also** clear the 7-anchor
weighted check before any submission — a mirror-only verdict is rule 16's trap,
and the mirror is 71.4% of our field above rating 1000 but not 100% of it.

## Predictions (registered now, scored later)

1. **Point estimate lands in [0.50, 0.53] and does NOT resolve.** The mechanism
   is real but a 0.8-plays/game intervention on a tradeoff is small, and this
   project's rules are 0 for 4 on tradeoffs.
2. **If it resolves at all, it resolves positive**, because the direction is
   "do more of what the 1150s do" rather than "stop doing something".
3. ⚠ **A specific way this could be WRONG that I cannot rule out:** the clone
   may be declining because its *own* subsequent play differs — it may lack the
   Basics in deck to fetch, in which case the option is available but the search
   whiffs and forcing it wastes the card. Not measured before freezing.

---

## ▶ VERDICT (2026-08-07, same session) — 🔴 NULL, and slightly negative

**`poffin` ON vs OFF, byte-identical net, n=2,800: 0.487 [0.469, 0.506].**
Fails the 0.53 bar and does not resolve. **The rule does not ship.**

Positive control passed first (39.7% → 61.6% play rate over 40 recorded games),
so this is a real intervention measured cleanly, not a silent no-op.

Prediction 1 was half right (did not resolve ✅, but landed below the [0.50,
0.53] band ❌). Prediction 3 — that the forced search may whiff for want of
Basics in deck — was never measured and stays on the record as the most likely
mechanism.

⛔ **Do not now run a milder threshold.** This file pre-registered that a
different threshold is a separate experiment, not a knob to tune after seeing
the result. Full write-up: `EVIDENCE` §8bl.
