# E25 — is V's 12% recovery a fixed FRACTION of the deviation cost, or does it grow as the rate falls? Pre-registered.

**Status: PRE-REGISTERED 2026-08-11 (day 30), after the threshold probe and
BEFORE either cell.** Frozen at the commit adding this file. E23 is reserved by
E22's doc for value iteration and E24 is the `fscrap` anchor cell, so this is
E25.

---

## What §8cf established, and the one question it leaves

Three points, all n = 2,000, mirror, byte-identical policy net both sides:

| arm | score | 95% CI |
|---|---|---|
| `bc:base` — identical arms (C0, Kaggle) | 0.5082 | [0.4897, 0.5267] |
| coin flip over top-3 @ 55.5% of firings | 0.1115 | [0.0896, 0.1334] |
| V-guided (LCB K=1.0) over top-3 @ 55.1% | 0.1580 | [0.1361, 0.1799] |

```
deviating from the clone at 23.2% of visited decisions   0.500 -> 0.1115  =  -0.389
what a 5-net pessimistic ensemble recovers of that              +0.0465  =    12.0%
```

Define **f = (V-guided − matched random) / (0.500 − matched random)** — the
fraction of the deviation cost the evaluator buys back. §8cf measured
**f = 0.12 at one rate.**

🔴 **If f is a constant of the setup, the net effect is −(1 − f) × cost, which
is negative at EVERY rate, and the whole family — "override the clone using a
better evaluator" — dies at once instead of one configuration at a time.** That
would close E20, E22, the oracle, the sequencer and the clock under a single
statement rather than five separate nulls.

⚡ **The escape is f > 1 at a low deviation rate**: if V's *confidence* ranks its
own overrides, the few it is surest about could be net-positive even though the
average one is not.

## Why the rate must be lowered by CONFIDENCE and not by subsampling

⛔ **Random subsampling of firings would be a null experiment.** It holds the
average quality of a deviation fixed, so cost and benefit both scale with the
rate and **f is invariant by construction**. It would confirm linearity and
learn nothing.

The informative knob is therefore a **minimum V-gap `vtau`** — which is the
`tau` knob E22's addendum forbade, and the distinction has to be stated or this
document is a rationalisation:

> ⛔ **E22 forbade tau as a SEARCH FOR A WINNING CONFIGURATION** — E17's
> post-hoc arm selection, which E19 priced. ✅ **Here tau is set ONCE, in
> advance, from a probe, to hit a pre-registered RATE, and the primary reading
> is a point prediction about f.** ⛔ **No tau is swept. If this cell fails, no
> second tau is licensed.**

## The threshold, fixed by the probe and not by any score

`p`-probe: 80 games of the exact E22 configuration, logging the distribution of
V's gap at each override (1,318 overrides over 2,445 firings):

| gap >= | overrides | share of firings |
|---|---|---|
| 0.01 | 950 | 38.9% |
| 0.02 | 769 | 31.5% |
| 0.04 | 510 | 20.9% |
| **0.08** | **254** | **10.4%** |
| 0.16 | 92 | 3.8% |
| 0.32 | 19 | 0.8% |

⚡ **`vtau = 0.08` is chosen because it lands nearest a ~1/5 rate**, which is the
lowest rate at which the predicted cost (~0.073) is still several SE at
n = 2,000. It was **not** chosen by looking at any win rate — no arm has been
run at any tau.

## The two cells

| cell | spec | n |
|---|---|---|
| **A** | `bc:e25,vlp,vlcb1.0,varm3,vtau0.08` over the 5 nets | 2,000 |
| **B** | `bc:e25ctl,vlp,varm3,vrnd<A's realised rate>` | 2,000 |

Cell B is §8cf's rate-matched coin flip re-pointed at cell A's realised
deviation rate. ⛔ **B is a CONTROL and cannot ship whatever it reads.**

## 🔴 The point prediction, written before either cell runs

If f is scale-free at 0.12 and cost is proportional to rate:

```
rate ratio           0.104 / 0.551          =  0.189
predicted cost       0.389 x 0.189          =  0.0734
predicted cell B     0.500 - 0.0734         =  0.427
predicted cell A     0.427 + 0.12 x 0.0734  =  0.436
```

**Cell A is predicted to read 0.436 and cell B 0.427.** E19 is the model here:
a point prediction that can be missed is worth more than a bar that cannot.

## Reading rule — frozen before either cell

| branch | condition | reading |
|---|---|---|
| 🔴 **SCALE-FREE CONFIRMED** | A in [0.41, 0.47] **and** f's CI contains 0.12 | f is a constant of the setup ⇒ **net effect is negative at every rate** ⇒ **the entire "override the clone with a better evaluator" family closes**, E20/E22/oracle/sequencer/clock under one statement. Report chapter, not another cell |
| ⚡ **CONFIDENCE SELECTS, INSUFFICIENTLY** | f significantly > 0.12 but A's CI excludes 0.500 from below | V's confidence does rank its own overrides, and the ceiling is still below break-even. Quantify the ceiling; **do not** chase it with a third tau |
| ✅ **CROSSOVER** | A >= 0.500, CI excludes 0.500 | there IS a rate at which V-guided override wins. ⛔ **A screen never ships (§8bh)** ⇒ fresh-games confirmation at n = 2,000, then weighted anchors |
| ⚠ **ANTI-INFORMATIVE CONFIDENCE** | A < 0.41 | high-confidence overrides are **more** damaging than average ones ⇒ V's gap is anti-informative about its own errors, which strengthens §8cd's extrapolation diagnosis rather than E22's remedy |

⛔ **Void conditions**, unchanged: `--deck-a/--deck-b grimmsnarl`; the component
must fire with a printed count; health clean (`fallbacks=0 net_missing=0
skip_noclock=0 skip_thin=0 errors=0`); every value net's bytes in the archived
identity (rule 20); and E19's standing constraint — **no internal gate licenses
anything, only the end-to-end A/B does.**

⚠ **A fault carried forward from §8cf, fixed here:** that audit's criterion was
"CIs disjoint" and they were disjoint by 0.0027. **Every comparison in this
document is a two-sample z on the pooled W/D/L, stated as such**, not a picture
of two intervals.
