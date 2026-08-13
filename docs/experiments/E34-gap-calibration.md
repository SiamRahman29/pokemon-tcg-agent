# E34 — is the rollout estimator's GAP BETWEEN ARMS calibrated? The randomized-overrule design. Pre-registered.

**Status: PRE-REGISTERED 2026-08-13 (day 32), BEFORE the first measured game.**
Frozen at the commit adding this file. User-directed.

⛔ **Pilot #1 and the first full run are VOID — wrong net (`EVIDENCE` §8cn).**
The design, statistics, controls and branches below are **unchanged and still
binding**; only the pilot numbers and the sizing derived from them are
withdrawn. ⚠ The instrument defect was in `p96`'s net handling, not in this
pre-registration, so nothing here is re-opened for editing after seeing data —
`p96` now takes the seat's own net object and asserts identity, and re-pilots
from scratch.

---

## The hole E33 left, stated exactly

E33 concluded on the CALIBRATED branch: **bias = −0.0083 [−0.0278, +0.0111]**
over 1,484 games / 15,271 decisions, so §8ca's *"the rollout estimate is biased,
most likely by determinization"* was retracted.

⚠ **But E33 rolled out ONE action — the clone's own pick — and therefore tested
a LEVEL.** Nothing this project spends is a level:

| where the currency is spent | the quantity |
|---|---|
| E17 | **+0.0139**/decision = `p̂(best arm) − p̂(own pick)` |
| `oracle.py` stage 2 | `argmax_j (p̂_j − p̂_0)` — a ranking of gaps |
| E19 cell A | one such gap, cashed once per game |
| §8bw's **+0.120** scale bar, §2.7's sizing | denominated in the same gaps |

**A level-calibrated estimator can still get gaps wrong.** If both arms carry a
shared position-specific error, it cancels in the level and survives in the
difference. E33's own post-hoc dispersion finding (**ratio 1.1445 [1.1139,
1.1759]**) is a hint in exactly that shape — an estimator whose spread exceeds
what the real process supports.

⇒ **E34 measures the gap directly, by randomizing which arm is actually
played.**

## Why this is not E19 again

E19 cell A read **0.4963 [0.4719, 0.5207]** at n=1,608 against a *predicted*
0.535 and concluded the estimate was biased. Two weaknesses, both fixed here:

1. ⛔ **E19's baseline was an ASSUMED 0.500.** A mirror A/B's expected score is
   0.500 by symmetry, but the overruling arm is only *one* seat and only *one*
   decision; the contrast was against a constant someone wrote down. **E34's
   baseline is measured on the same games by the same instrument** — the coin
   decides which arm gets played, so the counterfactual is observed, not
   assumed.
2. ⛔ **E19's prediction (+0.035) came from E17's offline table over DIFFERENT
   positions.** E34's prediction is `p̂_B − p̂_A` computed **at the very position
   that was then played out**. Predicted and realized are the same estimand on
   the same rows.

🔴 **And no common random numbers are available.** `harness.play_game(agent0,
agent1, deck0, deck1, recorder=None)` takes **no seed** — verified by
inspection, extending §8bw's C2 finding from the `fs` rollouts to the real
harness. ⇒ the two arms cannot be played on a shared world, the design is an
**unpaired** difference in means, and the sizing below is priced accordingly.

## The design

One randomized decision per game, on the shipped clone, in the mirror.

1. **Eligibility.** MAIN context, `search_begin_input` present, single pick
   (`minCount ≤ 1 ≤ maxCount`), ≥ 2 legal options, game not over, `turn ≥ 2`.
2. **Fire.** At each eligible decision, Bernoulli(`q`). The **first** fire in a
   game is the measured one; the rest of the game is untouched.
   ⚡ **`q = 0.025`, and the value is reasoned rather than convenient.** E33
   measured ~41 eligible MAIN decisions per game (15,271 at `every=4` over
   1,484 games). A constant `q` makes the first-fire index geometric with mean
   `1/q`, so `q = 1/41` spreads the sampled position ≈ **uniformly across the
   whole game**; the obvious `q = 0.1` would cluster it in the opening ten
   decisions. It costs almost nothing: firing in ~65% of games instead of ~100%
   adds 0.2 s of unfired game per fired game against ~4 s of rollouts.
   ⚠ **Stated as a property of the population, not a defect:** short games are
   less likely to fire, so the fired set over-weights long games. Internal
   validity is untouched — `z` is randomized *after* the fire — but the estimand
   is "the gap at a position sampled this way", and that is what gets reported.
3. **Arms, chosen BEFORE any rollout is read.**
   `A` = the agent's own pick. `B` = the highest-`net.scores` option ≠ A.
4. **Predict.** `R` rollouts of each arm, paired on a shared determinized world
   per replicate index (`oracle.py`'s stage-2 pairing, worth ρ≈0.53), to
   terminal, clone piloting both seats. → `p̂_A`, `p̂_B`, **`d̂ = p̂_B − p̂_A`**.
5. **Randomize.** Fair coin `z`. `z=1` ⇒ **play B**. `z=0` ⇒ play A.
6. **Observe.** The game finishes normally; `y ∈ {0, 0.5, 1}` from the deciding
   seat's view.

⇒ Because `z` is assigned **after** `d̂` is computed and independent of
everything, the `z=1` and `z=0` games are exchangeable. The difference in their
mean outcomes is an unbiased estimate of the true one-step gap, and `mean(d̂)`
is the estimator's claim about that same quantity **on the same games**.

⛔ **Two selection traps this design is built to avoid, and they are the whole
reason for step 3's ordering.**

- **Winner's curse.** If `B` were "whichever arm the rollout preferred", the
  selection would be on rollout noise and the realized gap would fall short of
  the predicted one **even under perfect calibration**. `B` is fixed by
  `net.scores`, which is computed without any rollout. This is the same family
  of error as E33's finite-`R` calibration curve, which was pre-registered as
  non-reportable for the same reason.
- **Regression dilution.** Regressing realized on predicted `d̂` attenuates
  toward zero because `d̂` is noisy. ⇒ **the primary is a difference of MEANS,
  which is immune to `R`.** The per-position slope is secondary with the
  attenuation stated, never the headline.

⚡ **`B` = the net's rank-2 option is the operationally relevant arm**, not an
arbitrary one: `oracle.py`'s `promoted_rank2` is the dominant bucket in its own
health line, so this is the comparison the shipped clock actually makes.

## Statistics

**Primary:** `miscal = realized_gap − predicted_gap`, where

    realized_gap  = mean(y | z=1) − mean(y | z=0)
    predicted_gap = mean(d̂)          over ALL fired games, both z groups

CI by **bootstrap over GAMES**, which is also the unit of randomization.

⚡ **No clustering problem here, and that is a deliberate property of the
design.** E33 needed a cluster-robust SE because it measured many decisions
inside one game; E34 measures **exactly one** per game, so games are the
independent unit by construction. (§8bw's lesson is paid by design rather than
by correction.)

**Secondary, reported alongside:**
- **`shrinkage = realized_gap / predicted_gap`** with bootstrap CI — the number
  that would reprice E17 and §2.7. Reported only if `predicted_gap` is bounded
  away from 0 by its own CI.
- **Covariate-adjusted primary.** `m = (p̂_A + p̂_B)/2` is a **pre-randomization**
  quantity, so adjusting `y` for it is legitimate and costs no validity. Expected
  gain is modest (~10% on SE, from E33's `Var(p)` against `Var(y)`); it is
  reported, not relied on.
- The primary restricted to `3 ≤ nopt ≤ 5` — `oracle.py`'s live firing window.
- The turn distribution of fired positions, descriptive.

## Controls — all four must pass or the cell VOIDs

1. **Randomization is real.** Observed `P(z=1)` within 3σ of 0.5, **and**
   pre-treatment covariates balanced across arms: `mean(p̂_A)`, `mean(turn)`,
   `mean(nopt)`, `mean(net margin)`. ⚠ An imbalance means the coin is reading
   something it must not.
2. **The overrule actually happened.** On `z=1` the returned pick is `B`,
   counted and asserted; on `z=0` it is the inner agent's own. **Exactly one
   fired decision per fired game.** (§8g had to infer a silent no-op once;
   §8cl's `[plan]` health line exists for the same reason.)
3. **Rollout health.** `none` rate reported; **VOID above 10%**. Both arms must
   return ≥ `R/3` completed rollouts or the position is skipped **without
   firing** — a skip must not consume the game's one fire.
4. **The fork is live.** AUC(`p̂_A`, `y`) within the `z=0` arm must clearly
   exceed 0.5, replicating E33's control 1 on this cell's own rows.

## Pre-registered readings

The bar is written against **the question E19 actually asked**, not an arbitrary
half-width: the CI must be able to separate *"the gap is real"* from *"the gap
is illusory"*.

| branch | condition | what it means |
|---|---|---|
| 🔴 **INFLATED** | CI on `miscal` excludes 0, with \|realized\| < \|predicted\| | **The currency is inflated by the measured `shrinkage`.** E17's +0.0139, §8bx, §8bw's +0.120 bar and §2.7's sizing all reprice by that factor. This is a mechanism for E19's null that E33 could not see. |
| ✅ **GAPS CALIBRATED** | CI on `miscal` contains 0 **and** excludes `−predicted_gap` (the fully-illusory alternative) | The estimator ranks arms honestly at the scale it is spent. ⇒ **E19's null has NO evaluator explanation left** — neither level (E33) nor gap — and the search family's failure is selection at the margin or per-decision gains genuinely not compounding. |
| ⚠ **UNRESOLVED** | CI contains both 0 and `−predicted_gap` | underpowered. Report the width; do not narrate a null. |
| ⛔ **VOID** | `predicted_gap` CI contains 0 | the chosen arm pair carries no predicted difference, so there is nothing to check calibration *of*. See sizing — this is the pilot's job to prevent. |

🔬 **The prediction, written in advance, and it is a null.** E33's dispersion
ratio of **1.1445 in variance ≈ 1.07 in sd** predicts a shrinkage of ≈ **0.93** —
a 7% effect. ⛔ **This design cannot detect 7% at any n we would pay for**, and
that is stated here so a CALIBRATED reading is not later narrated as exonerating
dispersion. ⇒ **I expect ✅ GAPS CALIBRATED**, and its value is *closure*: it
would leave the evaluator with no remaining way to explain E19.

⚠ **What would surprise me, and it is not nothing.** E18 measured search picking
the genuinely best arm only **67%** of the time, and **37% of its overrules take
an option the net scores > 3 worse**. If gaps are inflated by ≫ 7%, the cause is
something beyond dispersion and this cell will say so.

## Sizing — pilot first (rule 14)

⚠ **Power here is driven almost entirely by `|predicted_gap|`, so the pilot's
first job is to measure it.** The primary's SE is ≈ `2σ/√G` with `σ ≈ 0.5`, i.e.
≈ `1.0/√G`, falling to ≈ `0.9/√G` with the covariate adjustment. Against the
GAPS CALIBRATED bar (the CI must exclude `−predicted_gap`):

| `\|predicted_gap\|` | G for 2σ separation |
|---|---|
| 0.10 | ~360 |
| 0.05 | ~1,300 |
| 0.03 | ~3,600 |
| 0.02 | ~8,100 |

⇒ **if the pilot returns `|predicted_gap| < 0.02`, the primary population is
narrowed** to positions with a larger *pre-rollout* expected gap — net rank-1
minus rank-2 score margin ≥ `m` — because that covariate is computed without
reading a rollout and therefore **cannot induce winner's curse**. ⚡ E19 cell B
already established the direction: the value sits where the net is **confident
and wrong** (margin > 3), not where it is torn. The narrowed population, if used,
is fixed here as the rule and its `m` recorded before the full run.

`R` is fixed at **20** — `oracle.py`'s shipped `r_sel`. ⚡ **The estimator is
measured at the setting it is spent at**, and `R` does not enter the primary's
expectation at any value, so it is not a knob to turn after seeing a number.
`d̂`'s own noise contributes ≈ `0.09/√G` to the primary, an order of magnitude
below the outcome noise, which is why `R` can be pinned rather than swept.

Cost model going in: 2·`R` = 40 rollouts × ~101 ms (§8bw) ≈ **4.0 s/game** on
top of a ~0.2 s game. `scripts/shard.py` (§8cl) makes it ~3.1× cheaper in wall
clock. ⇒ 5,000 games ≈ 5.8 h serial ≈ **1.9 h wall**.

    python -X utf8 scripts/p96_gap_calibration.py --games 60 --rollouts 20 --q 0.025

### ⛔ PILOT #1 VOIDED — WRONG NET. Recorded in full; do not size off it.

🔴 `p96` took `net.scores` and the rollout continuation from `pnet.get()`, the
**v2** clone `#a25b904d`, while both seats played `out/policy_v5_s2.npz`
`#4790c469`. ⇒ **arm B was the OLD net's rank-2 option** and the rollouts
continued under the OLD policy, so `predicted_gap = −0.0245` describes an
estimator nobody ships. The full run launched off it was killed at ~1,400 fired
games and archived to `out/logs/e34_wrongnet/`. See `EVIDENCE` §8cn.

⚡ **The pilot's own rows are what exposed it**, an hour later: `margin =
score[A] − score[B]` was **negative on 51 of 220 rows (23%)**, meaning the
agent's own pick ranked below the net's argmax — impossible with rules off,
where `oracle.py` verified `argmax(scores) == net.choose` at **106/106**. With
the net pinned it is **0%**. I had printed that 23% as a routine integrity note
and moved on.

⚠ **Sizing must be re-derived, not adjusted.** `|predicted_gap|` is the one
quantity the full run's cost depends on, and arm B is now a genuinely different
option: with the correct net, A is the true argmax at every decision, so B
should be *worse* than it was, and `|predicted_gap|` **larger**. That would make
E34 cheaper — which is exactly why it must be measured rather than assumed.
⛔ **The narrowing rule is re-armed and re-tested against the new pilot**: it
fires below `|predicted_gap| < 0.02`, and pilot #1's 0.0245 no longer counts as
having cleared it.

⚠ Everything below is retained for the record only.

### ▶ PILOT #1 (VOID) RAN 2026-08-13 — 6 shards × 60 games, R=20, q=0.025

**All four controls pass.** `none = 0.0%` of 8,800 rollouts (control 3);
`played_b + kept_a == fired` on every shard and `double_fire = 0` (control 2);
**AUC(p̂_A, y) = 0.7327** on the `z=0` arm (control 4); balance `|t| ≤ 1.67` on
all of `p̂_A`, `turn`, `nopt`, `margin`, and `P(z=1) = 0.473` (control 1).

| statistic | pilot (220 fired games of 360) |
|---|---|
| fire rate | **62%** of games ⇒ ~38 eligible decisions/game |
| **PREDICTED gap** | **−0.0245 [−0.0464, −0.0035]** |
| REALIZED gap | +0.0020 [−0.1283, +0.1313] |
| MISCAL | +0.0265 [−0.1076, +0.1596] |
| realized, ANCOVA-adjusted | −0.0264 [−0.1494, +0.0971] |
| cost | 6.5 s/game at 6 shards ⇒ **0.55 fired-games/s** |

⚡ **`q = 0.025` is confirmed by the data it was derived from.** A 62% fire rate
implies `(1−q)^N = 0.38` ⇒ **N ≈ 38 eligible decisions per game**, against the
~41 estimated from E33. The sampled position spreads across the whole game as
intended (turn bands 0–4 / 4–8 / 8–12 / 12+ hold 63 / 78 / 56 / 23 fires).

⚡ **The SE model is validated rather than assumed:** observed
`SE × √G = 1.014` against the **1.0** the sizing table was built on.

**Sizing, fixed here BEFORE the full run.** `|predicted_gap| = 0.0245`, so the
GAPS CALIBRATED bar (CI must exclude the fully-illusory `miscal = +0.0245`)
needs `1.96 × 1.014/√G ≤ 0.0245` ⇒ **G ≥ 6,580 fired games**. Run **16,200
games as 6 shards of 2,700** ⇒ ~10,000 fired ⇒ half-width ≈ **0.0199**, and
~5 h wall at the measured throughput.

⛔ **The narrowing rule does NOT fire, and that is binding.** It was
pre-registered to trigger at `|predicted_gap| < 0.02`; the pilot returns
**0.0245**, above the line. Narrowing to a large-`margin` population now would
be choosing the population *after* seeing which one gives a bigger effect —
the same class of error the arm-ordering rule exists to prevent. **The full
population stands.**

⚠ **The pilot's point estimates are recorded and are NOT a lean.** MISCAL's
half-width is **0.134**, five times the effect the full run is sized to
resolve, and the unadjusted (+0.0265) and ANCOVA-adjusted (−0.0264) estimates
sit on opposite sides of zero — which is itself a statement that n=220 resolves
nothing here. 🔬 They are noted only because the point estimate leans **against**
this file's pre-registered prediction of ✅ CALIBRATED: a realized gap near 0
against a predicted −0.0245 would be the 🔴 INFLATED branch. **That is a reason
to run the cell, not a result from it**, and it is written here so the full
run cannot later be narrated as having confirmed a hunch formed at n=220.
