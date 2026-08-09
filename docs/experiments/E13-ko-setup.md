# E13 — the KO-setup chip target (pre-registered 2026-08-09, day 26, BEFORE any code and BEFORE the sizing run)

**Found by `p73_target_policy.py`, which exists because E12's dominated test only
covered the 1.2% of damage-placement decisions where a KO was already on the
board — and I let "the net is 99.1% correct" read as a claim about targeting in
general when it was a claim about that subset.** This experiment is about the
other 98.8%.

⚠ **This file is frozen before the sizing script exists.** The condition below is
declared now precisely so the 0.5 firings/game gate can kill it — see "the
sizing gate" — rather than being widened after seeing a disappointing count.

## The measurement (already made, not part of the test)

Share of opponent-targeting damage-placement decisions in which the counter is
put on the opponent's **Active**, mirror games, KO-available decisions excluded
(that is the tradeoff regime, not the arithmetic one):

| corpus | chose Active | n |
|---|---|---|
| us (`replays/submission_v5_s2`, shipped agent) | **4.1%** | 194 |
| `李秉叡（ntumlnoob）` (#2, 1162.8) | **14.1%** | 794 |

Conditioned on the Active being **already damaged**:

| corpus | chose Active | n |
|---|---|---|
| us | **7.0%** | 86 |
| them | **22.4%** | 441 |

⚠ **Confound checked before pre-registering** (§8bo): direct standardisation of
our per-bucket rates onto their board mix moves 4.1% → **5.5%** against their
14.1%, so **~86% of the gap is behavioural**, not a different mix of situations.

Coheres with a second, independent number: KO-available decisions are **1.2%** of
our stream and **6.3%** of theirs. They concentrate damage to *manufacture* KOs;
we spread it and the KOs never arrive.

## Why this is tested rather than closed by rule 11

**It is a tradeoff, and tradeoffs are 0 for 5.** Spending the counter on the
Active forgoes developing damage elsewhere, and whether the KO actually lands
depends on the opponent retreating, healing or switching. `HANDOFF.md:3185`
warns that "dominated" is easy to talk yourself into — `counter_source` was
filed as dominated on a judgment that was asserted rather than measured. One
judgment puts this in the tradeoff column no matter how good the arithmetic
half looks. **I initially pitched this as near-dominated. It is not.**

What earns it a single A/B anyway is rule 11's *own* note on why the 3/3 won:
what the net **cannot see is HP, damage and attached energy**. §8bo shows the
net already branches targeting by matchup on its own — 4.1% (mirror) to 90.9%
(Crustle), a 22× swing with every rule off — because archetype is card identity,
which the encoding does carry. Damage state it does not carry, and the encoding
route to it is closed (+115 → +37 → +14 → 0 → 0). A rule is the only instrument
left for this input.

Governing precedent is rule 10 / `boss_veto` and E11: **one cheap,
seed-cancelling A/B decides it.** ⛔ If it is null, the finding is a chapter and
no rule ships. The deviation is running the test, not lowering the bar.

## The intervention

`targeting.ko_setup_target(obs, chosen)` — a **post-net promoter** in the
`counter_source` / `poffin_force` position (it reorders the net's pick; it does
not replace the ranker). Returns `None` — leaving the net alone — unless **all**
of the following hold:

1. `select.context` is in `CHIP_CONTEXTS` (13 `DAMAGE_COUNTER`,
   14 `DAMAGE_COUNTER_ANY`, 15 `DAMAGE`).
2. Every option is an **opponent-side** Pokémon and readable — the same guard
   `chip_target` uses, so mixed and own-side selects fall through untouched.
3. One option names the opponent's **Active**, and the net did not already pick it.
4. The Active is **already damaged** (`hp` < the card's printed HP).
5. **The KO-setup arithmetic.** With `A = best_damage(our active, mypl, oppl,
   their active)` and `c = CHIP_DAMAGE` (30):

       A < hp            (the KO is NOT already there)
       A >= hp - c       (this placement is what puts it in reach)

   Fire only in that band. Above it the chip buys nothing that our attack was
   not already going to take; below it the Active survives anyway and the
   counter is better spent developing the bench.

When it fires, the Active option is promoted to the front of the net's ordering;
everything else keeps the net's relative order.

**Why this form and not "prefer a damaged Active":** the plain version fires on
100% of condition-4 decisions, and matching a **22.4%** behaviour with a 100%
rule is exactly E11's error — it forced a 70% expert preference to 100% and read
0.487. Clause 5 is the mechanism rather than the rate: it selects the subset of
damaged-Active decisions where concentrating measurably converts a survivable
turn into a lethal one. **A rule without clause 5, or a different value of `c`,
is a separate experiment — never a knob to tune after seeing this result.**

⚠ **Predicted interaction with walls, stated now:** `best_damage` does not model
prevention and reports 180 into a Crustle (`targeting.py:76`). That over-read
makes clause 5's first test (`A < hp`) fail, so the rule **cannot fire against a
wall**. That is the correct behaviour, but it is an accident of an approximation
rather than a designed guard, and it is on the record as such.

## The sizing gate (rule 14, run AFTER this file is committed)

Count decisions satisfying 1–5 over `replays/submission_v5_s2` (76 games).
**⛔ Below 0.5 firings/game the rule is dead and is not written** — the same gate
that closed Morgrem (0.2), Pokégear (0.27) and the Archaludon rule (0.187), and
that closed both halves of E12 (missed-KO 0.09, source under-move 0.20).

The denominator is known healthy: 10.2 damage-placement decisions per game.

## The design (frozen)

- `bc` both arms with the **byte-identical** net `out/policy_v5_s2.npz`, rule
  toggled — so the ±13 Elo seed nuisance (§8bk) cancels **exactly**.
- Shipped configuration otherwise: `noChip,noSpread,noSrc`.
- Mirror, DIRECT, seat-balanced, **n = 2,800** (±~0.019).
- **Positive control first:** record games with the rule ON and confirm the
  chose-Active rate actually moves. E12 spent a day on a measurement whose
  option→Pokémon alignment was wrong and only a control caught it (11/48 → 55/55);
  a silent no-op must not be allowed to masquerade as a null.

## The bar (frozen, same as E11's and F2's)

**Ship iff point ≥ 0.53 AND the 95% CI excludes 0.50.** Below → keep the shipped
agent, write the null. ⚠ If it ships, it must **also** clear the 7-anchor
weighted check before any submission: the mirror is **31.6%** of the field at our
current rating (§8bn), not the 71.4% §8ac projected, so a mirror-only verdict is
rule 16's trap with a worse denominator than E11 faced.

## Predictions (registered now, scored later)

1. **The sizing gate is the most likely killer.** Condition 4 held on 86 of our
   194 tradeoff-regime decisions, and clause 5 is a narrow band on top of that.
   If it fires under 0.5/game the experiment ends before the A/B, and that is a
   result, not a failure.
2. **If it reaches the arena, the point estimate lands in [0.50, 0.53] and does
   not resolve** — tradeoff rules are 0 for 5 and this is a small intervention.
3. **If it resolves at all, it resolves positive**, because the direction is "do
   more of what the 1150s do" rather than "stop doing something".
4. ⚠ **A specific way this could be WRONG that I cannot rule out:** the expert
   22.4% may be driven by *their* follow-up — a Boss's Orders or a switch that
   converts the setup — in which case reproducing the target choice alone
   reproduces the cost without the payoff. Not measured before freezing. This is
   E11's prediction 3 in a new suit, and E11's prediction 3 was the one that
   survived contact.
