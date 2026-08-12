# E27 — on-policy policy iteration with a trust region on the PREVIOUS policy. Pre-registered.

**Status: PRE-REGISTERED 2026-08-12 (day 31), BEFORE any generation, training or
arena game.** Frozen at the commit adding this file.

---

## The licence, and it is narrower than the slogan

§8ch measured that a **trained policy's** deviations cost about a quarter of what
arbitrary deviations of the same rate, depth and location cost
(**f = 0.758 [0.703, 0.814]** against the evaluator family's 0.12). In the units
that matter:

```
incoherent deviation   -0.11148 log-odds per changed pick    (E25, replicated out-of-sample by E26 cell B)
a trained policy       -0.01899 log-odds per changed pick    (E26 cell A: logit(0.4053) / 20.19)
                        5.9x cheaper
```

⇒ **the sharp local optimum of §8cg forbids JUMPS, not PATHS.**

⚠ **What that does NOT license.** §8ch cannot separate sequential coherence from
per-decision plausibility, so "coherence is worth 0.76" is not a measured claim.
The composite — *a trained policy moves cheaply* — is what E27 rests on, and the
composite is enough, because E27's steps are taken by a trained policy either way.

## 🔴 The honest case AGAINST, first

Three standing measurements point at a null, and they are named here so the
result cannot be narrated past them:

1. **§8bv — conditioned on the board, winners and losers play the same**
   (−0.0024 bits). That is close to a direct measurement that the outcome label
   carries little about the action. E27 uses a *finer* signal (per-decision TD
   residual, not per-game outcome), but V is trained on those same labels, so
   V's information about action quality is **bounded by what outcomes carry**.
2. **B8 (§8ao) already failed with outcome-weighted fine-tuning** — 20,000
   games, two runs, and 4× the data moved the estimate DOWN. It closed *on the
   method*.
3. 🔴 **E27 risks LAUNDERING V's contaminated ranking through a training step.**
   E25's mechanism is that V is confident exactly where it extrapolates. A
   gradient that up-weights high-advantage actions up-weights the ones V liked
   most — which are the ones furthest off-policy. **The defence is that V is
   evaluated only at states the game actually reached, plus a trust region and a
   per-round A/B — but that is an argument, not a measurement, and it is the
   most likely way this fails.**

## Why this is nevertheless not B8 re-run

| | B8 (§8ao) | E27 |
|---|---|---|
| rounds | **one** | **iterated**: each round re-collects on-policy from the NEW policy |
| signal | terminal `won`, AWR β=1.0 | **per-decision TD residual** `V(s_{t+1}) − V(s_t)` |
| anchor | the **human corpus** at weight 1.0 | the **previous policy** π_r — a trust region, not a leash to the mode |
| trainable | head only, **17.2%** | more than the head (frozen set named per round, before the round) |
| where V comes from | — | **retrained each round on that round's own games**, i.e. actual policy evaluation |

⚡ **The anchor change is the conceptual one.** B8 anchored to the corpus, which
by construction prevents leaving the clone's basin — it was built not to escape.
Anchoring to π_r instead keeps steps *small* without keeping them *in place*,
which is what "paths not jumps" means operationally.

## ▶ CELL 0 — the calibration that makes every later gate quantitative

**E26 gives ONE point on the coherent cost curve** (c = 20.19 changed
picks/game). E27's steps will be much smaller, and the curve's shape at small c
is exactly what "paths are cheap" asserts. ⛔ **Assuming it is a slogan; measuring
it is one 2,000-game cell.**

`xsub<p>` accepts the expert's deviating pick with probability p, so cell A's
machinery reruns at a reduced rate over the *same* decisions.

| cell | spec | expected c | n |
|---|---|---|---|
| **0a** | `bc:e27c25,net=…v3,xnet=…ntum,xsub0.25` | ≈ 5.0 | 2,000 |

Fit through the origin (a policy identical to π₀ scores 0.500 by construction):

- **If log-odds-linear** (cell 0a lands near `sigmoid(−0.0190 × 5.0)` = **0.476**),
  the null for every E27 round is `null(c) = sigmoid(−0.0190 c)`.
- ⚡ **If SUBLINEAR** (cell 0a reads meaningfully above 0.476), small coherent
  steps are cheaper than proportional — **"paths not jumps" becomes measured**,
  and the null uses the fitted curve instead.
- 🔴 **If SUPERLINEAR** (below 0.476), small steps are *not* cheap, the licence
  E27 rests on is weaker than §8ch suggested, and **that is a finding worth more
  than the round it would have gated.**

### ▶ CELL 0a HAS RUN — the coherent cost curve is LINEAR in log-odds, and the null is now a measured object

```
cell 0a  bc:e27cal,...,xsub0.25   0.479 [0.458, 0.501]   n=2,000
         c = 11,419 / 2,000 = 5.71 changed picks/game     mean depth 1.86 (cell A: 1.92)
         health OK  fallbacks=0  xerr=0  clipped=0
```

| | c | score | logit | per-pick slope |
|---|---|---|---|---|
| origin (identical policy) | 0 | 0.500 | 0 | — |
| **cell 0a** | 5.71 | **0.479** | −0.0841 | −0.0147 |
| **E26 cell A** | 20.19 | **0.4053** | −0.3835 | −0.0190 |

**Linear prediction for 0a was 0.4729; measured 0.479.** ⇒ **the LINEAR branch
fires.** Pooled OLS through the origin over both points:

```
null(c) = sigmoid(-0.018677 * c)          <- E27's null, frozen
```

⚠ **Two honest qualifications, because this interval is wide.**
1. At c = 5.71 the cost is **0.021, only 1.88σ from zero** (p ≈ 0.06). The slope's
   own CI from cell 0a alone is **[−0.0295, +0.0007]** — it contains cell A's
   −0.0190 comfortably, and it also contains **zero**. So "linear" here means
   *not distinguishable from linear*, not *confirmed linear*.
2. The point estimate leans mildly **sublinear** (−0.0147 vs −0.0190), which
   would favour E27's premise. **It is not resolved and must not be quoted as
   support.**

⚡ **What it does establish, and it is what the gate needed:** a trained policy
deviating at ~6 changed picks/game costs **about 2 points of win rate**. So a
round-1 step of that size starts from ≈0.479, and the measurement SE at
n = 2,000 is ≈0.011 — a workable gate, where "the direction paid" needs roughly
**≥ 0.51** to exclude the null.

## The design

Round *r* (π₀ = `out/policy_v5_s2.npz`, the shipped net):

1. **Generate** N self-play games from π_r at τ (exploration), sharded on Kaggle.
2. **Evaluate**: train V_r on *that round's* games (`train_value.py`), so V is
   on-distribution for π_r. Held out **by game**.
3. **Advantage**: `A_t = V_r(s_{t+1}) − V_r(s_t)`, seat-corrected
   (`V_me(s) = 1 − V_them(s)` in a zero-sum two-player game), terminal
   `V(s_T) = won`. Written as a column by `scripts/p92_td_advantage.py`.
4. **Improve**: fine-tune π_r → π_{r+1} weighting by `A_t`, anchored to π_r.
5. **Measure**, and this is the part that is not optional:
   - the **realised deviation rate** of π_{r+1} from π_r, via E26's own wrapper
     (`xnet=`), in changed picks/game — the same unit the null is written in;
   - the **A/B** π_{r+1} vs π_r, n ≥ 2,000, mirror;
   - the **A/B** π_{r+1} vs **π₀** — because rounds compound and the incumbent
     is what a submission must beat.

## The configuration, frozen before round 1 — every value set on a property of the DATA, never on a score

| knob | value | why this value and not another |
|---|---|---|
| AWR β | **0.5**, in units of sd(A) | a TD residual has sd ≈ 0.07 against B8's `won − base` sd ≈ 0.5, so the same β would push **7× more gently** than the run that already measured null. β = 0.5 after standardising reproduces **B8's weight ratio of e ≈ 2.72** between a +1sd and a −1sd decision ⇒ **E27 differs from B8 in the SIGNAL, not in how hard it pushes** |
| advantage clip | **±2 sd** | the advantage is a spike at zero plus a heavy tail (§8by's shape). Unclipped at ±4 the **effective sample size measured 50.8%** against B8's 91.3% — half the corpus lost to variance before any signal is asked for. At ±2 the full weight range is `[e⁻¹, e⁺¹]`, **exactly B8's win-row/loss-row range end to end**, and ESS reads **85.5%** |
| architecture | `--state-h 512,256 --head-h 256,128 --pool --opt-cols 37` | v5's recipe, verified against `policy_v5_s2.npz` (702,913 params, `n_pool=172`). ⚠ **`--opt-cols 37` is load-bearing**: the corpus now writes 46-wide option features and the v6 attribute block is *appended*, so 37 slices exactly the v5 layout `--init` requires |
| V | retrained per round on that round's own games | policy **evaluation**, on-distribution for π_r |

⛔ **None of these may be swept.** B8 was denied a β sweep and E17's post-hoc arm
selection is what E19 priced. A different β requires a new pre-registration.

## ⚠ The gate arm must run through the SAME INSTRUMENT the null was calibrated on

Recorded before round 1 produced any number, because it is the kind of mismatch
that reads as a result.

`null(c)` was fitted on **E26 cell A and E27 cell 0a, both of which ran through
the `xnet=` wrapper** — which substitutes only on **single-pick** decisions and
lets multi-selects fall through (8,816 of them in cell A). A round-1 A/B run as
a plain `net=` swap would substitute on **every** decision, so its deviation
rate is not the `c` the null is a function of, and comparing the two would be
comparing an intervention against a curve calibrated on a weaker one.

⇒ **Two arms per round, and they answer different questions:**

| arm | spec | reads against |
|---|---|---|
| **gate** | `bc:…,net=π₀,xnet=π_{r+1}` | `null(c)` — *did the direction pay more than the cost of moving?* |
| **ship** | `bc:…,net=π_{r+1}` vs `bc:…,net=π₀` | the 0.541 bar — *is this a better agent?* |

⛔ **The gate arm never ships and the ship arm never gates.** Mixing them is
E17's post-hoc arm selection with extra steps.

## Reading rule — frozen before any round

Let `c` be π_{r+1}'s measured changed picks/game against π_r, and `null(c)` the
cell-0 curve.

| branch | condition | reading |
|---|---|---|
| ✅ **THE DIRECTION PAYS** | score vs π_r **above** `null(c)`, CI excluding it | the step's *direction* bought more than the cost of moving. **Continue to the next round.** |
| 🔴 **MOVEMENT ONLY** | CI contains `null(c)` | the policy moved and the direction was worth nothing — **exactly the B8 result at finer grain.** Two consecutive rounds here ⇒ **STOP**, and the axis closes with a number |
| 🔴 **WORSE THAN MOVING** | below `null(c)`, CI excluding it | the advantage signal is **anti-informative** ⇒ hypothesis 3 above (V laundering) is confirmed ⇒ **STOP IMMEDIATELY**, do not tune, do not sweep β |
| ⚠ **VOID** | any health flag, or `c` = 0 | a fine-tune that changes no decisions is not a treatment |

**Round budget: 3.** ⛔ **No β sweep, no τ sweep, no "one more round because it
was trending" — B8 was declined a β sweep and E17's post-hoc arm selection is
what E19 priced.** A fourth round requires a new pre-registration.

## Ship rule, written before there is anything to ship

⛔ **Both active slots currently hold byte-identical `v5_s2`, so a submission
evicts our own best agent.** The bar is therefore the incumbent's:
**π_final vs π₀ at n ≥ 2,000 in the mirror, point ≥ 0.541** (B8's precedent: the
seed-only null 0.482 plus its width), **AND** the weighted anchor score must not
fall (§8ac weights — self-play can buy wins against our own net while losing to
the field's rule agents, and the mirror alone cannot see that).
If it ships, **submit twice** (§8ak, max-of-two-draws). **Last safe day 08-15.**

## What this cannot say

- A win does not show *which* of the four changes from B8 mattered. Four factors
  move together and the calendar does not permit an ablation; **say so, and file
  the ablation as Round-2 work** rather than implying a mechanism.
- Self-play measures strength **against ourselves**. §8ac's 71.4%-mirror field
  above rating 1000 is the reason that is not fatal here, and the anchor sweep is
  the guard — but it is a guard, not a proof.
- The value net's own quality bounds everything, and it is trained on the same
  outcome labels §8bv found nearly uninformative about actions.
