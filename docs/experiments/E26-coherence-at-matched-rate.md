# E26 — is a COHERENT policy's deviation cheaper than an incoherent one at the same rate? Pre-registered.

**Status: PRE-REGISTERED 2026-08-12 (day 31), after the p91 sizing probe and
BEFORE any arena game.** Frozen at the commit adding this file.

---

## Why this is not another cell in the family E25 closed

§8cg closed *"override the clone with a better **evaluator**"*. Every member of
that family — §2's search, B4's sequencer, the clock, E20, E22, E25 — keeps the
clone as the policy and substitutes an argmax chosen by scoring options
**off-policy**. E25 found why they all fail: the evaluator's confidence and its
distance from the clone are the same variable, so the ranking signal is the
damage signal.

**A different policy is not an evaluator.** `policy_b7_ntum` is on-distribution
for itself: its deviations are chosen by a network that was *trained* to make
them, in sequence, and it never scores an option it would not play. Whatever
kills the evaluator family is not obviously present here — and §N.3 has carried
this as *"the actual open question"* since day 28:

| | claim | predicts |
|---|---|---|
| **H1 "no plan"** | the expert's moves are good only *as a sequence*; partial copying breaks coherence | a coherent deviation is much cheaper than an incoherent one at the same rate |
| **H2 "the mode is the local optimum"** | any deviation costs, regardless of direction or source | a coherent deviation costs the same as a coin flip at the same rate |

## What the sizing probe found, and how it changed this document

`scripts/p91_phase_disagree.py`, no arena games spent. Rows are real selects
(≥2 options); rate is *how often the expert clone's argmax differs from ours*.

| state distribution | early (1–4) | mid (5–9) | late (10+) | all |
|---|---|---|---|---|
| **`pds_ours` — our own games** | **0.265** | **0.257** | **0.268** | 0.263 |
| `pds_ntum_r` — the expert's games | 0.283 | 0.312 | 0.333 | 0.311 |

🔴 **On the distribution the agent actually meets, disagreement is FLAT across
the game — 1.04× between the largest and smallest phase.** The originally
planned experiment (substitute the expert in one phase) therefore varies almost
nothing but the *rate*, and E25 has already priced rate knobs. ⇒ **the phase
design is dropped before it was built.** ⚠ This kills a *rationale*, not the
question: flat rate does not imply flat quality. What it means is that phase is
the wrong axis to spend games on, and **coherence at a fixed rate is the right
one**, because it is the axis H1 and H2 actually disagree about.

⚠ The mild monotone rise on the expert's own states (0.283 → 0.333) is on
**their** trajectories, not ours, and §8s already showed the two policies differ
near-symmetrically on both distributions. It is not evidence for a phase effect
in our own play.

## The measuring stick already exists

E25 measured the cost of **incoherent** deviation at two rates, same harness,
same deck, byte-identical nets:

| arm | changed picks/game | score | logit |
|---|---|---|---|
| coin flip @ 55% of firings | 18.21 | 0.1115 | −2.0755 |
| coin flip @ 13% of firings | 4.56 | 0.3650 | −0.5538 |

Cost is close to linear in **log-odds per changed pick** (E25's own note: −0.114
and −0.121 per changed pick at the two rates), which fixes the line

```
logit(score) = -0.11148 * changed_per_game - 0.04545        [E25's two points]
              intercept -> 0.4886 at zero changed picks, i.e. the C0 null
```

**This line is the null hypothesis of E26.** It says a deviation costs what a
deviation costs, whoever chose it. That is H2, written as an equation.

## The two cells

Generation is **v3 on both sides**, deliberately:

- `policy_b7_ntum` is a **v3-era net** (`sfc0_w` 496 wide, no `xdense` block)
  while we ship v5 (708). Substituting it into a v5 agent would confound the
  expert's *policy* with a feature set worth ≈ +51 Elo (§8z + §8aa). Discovered
  by the sizing probe, before any games.
- Running at v3 makes **cell A a replication of §8u (0.370 [0.349, 0.391])**,
  which gives this experiment a positive control it would otherwise lack.

| cell | spec | opponent | n |
|---|---|---|---|
| **A** | `bc:e26x,net=out/policy_b1_v3.npz,xnet=out/policy_b7_ntum.npz` | `bc:base,net=out/policy_b1_v3.npz` | 2,000 |
| **B** | `bc:e26r,net=out/policy_b1_v3.npz,xrnd<A's realised rate>,xrank=<A's histogram>` | same | 2,000 |

Both arms run the **same wrapper with the same two forward passes**, so the
plumbing is paid for on both sides (the `sym1` / `flip0` discipline). Cell A
plays the expert's pick; cell B plays a random option drawn to match cell A's
realised **rate** and **rank depth**.

⚠ **Rank matching is the part E25 wished it had.** §8cf recorded that ~20% of
V's apparent edge might be net-margin rather than selection, because the control
deviated *deeper* than the treatment. Here cell B samples its deviation rank
from cell A's measured rank histogram, so depth is matched by construction and
the residual is direction alone.

⚠ Cell B is a **CONTROL and cannot ship whatever it reads.**

## 🔴 The point prediction, written before either cell runs

Cell A is predicted at **0.370** (§8u's measured value, same nets, same matchup,
same n — a replication, not a guess). Any reading outside **[0.33, 0.41]** means
the harness is not reproducing a settled result and **the cell is VOID**, not
interesting.

Cell B is predicted by the E25 line at cell A's realised `changed_per_game`. At
the offline rate of 0.263 and this agent's real-select count the expectation is
firmly **below 0.15**; the exact number is computed from A's realised counter and
recorded **before B runs**.

Define, exactly parallel to E25's f:

```
f_coh = (A - B) / (0.500 - B)          the fraction of the deviation cost that COHERENCE buys back
                                        E25 measured f_eval = 0.12 for the best evaluator we can build
```

## Reading rule — frozen before either cell

| branch | condition | reading |
|---|---|---|
| ✅ **COHERENCE IS THE VARIABLE** | f_coh > 0.5, CI excludes 0.12 | a coherent policy's deviations cost **far less** than matched incoherent ones ⇒ **H1 supported over H2**, and the escape direction is a policy that is coherent *and* better than the mode — which is E27, not more imitation. ⛔ Still ships nothing: A is 0.370, i.e. losing |
| ⚡ **PARTIAL** | f_coh in [0.12, 0.5], CI excludes 0 | coherence is worth more than evaluator quality and less than the whole cost. Quantify; the escape argument weakens but survives |
| 🔴 **H2 CONFIRMED** | f_coh's CI contains 0.12, or B ≈ A | **the cost is the deviation, not its incoherence.** Then the local optimum is sharp in every direction and E27's premise is damaged in advance — record that before spending Kaggle time on it |
| ⚠ **VOID** | A outside [0.33, 0.41], or any health flag | the instrument is not reproducing §8u; fix before reading anything |

⛔ **Void conditions** (unchanged, and each one paid for): `--deck-a/--deck-b
grimmsnarl`; the wrapper must fire with a printed count (`xfired`, `xdiff` —
E24's lesson that a firing the net agrees with is **not** a treatment); health
clean (`fallbacks=0 net_missing=0 errors=0`); both net paths in the archived
identity (rule 20); seat-corrected score read from the arena's own line, never
from raw `winner=` counts (rule 18).

⛔ **No third cell on this axis without a new pre-registration.** If f_coh is
large, the follow-up is **not** "clone a better expert" — B7 measured that at
−55 and −92 Elo, and §8u's monotone ordering is the reason. It is E27.

## What this cannot say

- It is measured at **v3**. Applicability to the shipped v5 agent is an
  assumption; the confounded alternative (substituting a v3 net into a v5 agent)
  is worse, and a v5 expert clone would need the ntum corpus rebuilt with the
  `xdense` block. **State the generation in every sentence that quotes the
  number.**
- A large f_coh does **not** say the expert is good — cell A is 0.370, a rout.
  It says the *cost structure* differs, which is a statement about what kind of
  intervention could ever pay, not about this one.
- Neither cell measures anything about **our own** incoherence. §N.4.1 (do we
  switch commitments more often than the experts?) remains unbuilt and unrun.
