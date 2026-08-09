# E16 — is the expert's move actually better than ours? Counterfactual move value by paired rollouts

**Status: PRE-REGISTRATION DRAFT. Frozen before any measurement cell is run.**
Feasibility for the instrument was established first and separately in
`EVIDENCE` §8bw / `scripts/p80_rollout_feasibility.py`; no result below has been
computed.

## The question, and why nothing in this repo can currently answer it

§8u is the sharpest number we have: we cloned the #2 player **successfully**
(held-out agreement 59.9% → 67.2%) and it cost **−92 Elo**, with covariate shift
ruled out (§8s). §8r then showed why no existing eval can explain that — every
rate-vs-experts metric is a **conformity** metric, and conformity to the mode is
what a ~1000 rating *is*. The one question that would separate the live
hypotheses has never been askable:

> **In THIS exact position, is their move better than ours?**

E16 asks it directly. Fork a real expert position out of a replay, force the
expert's move in one arm and our net's move in the other, let the clone play
both seats to a terminal state, and difference the win rates. No corpus, no
mode, no agreement.

## What the feasibility probe fixed about the design (§8bw)

These are measurements, not assumptions, and three of them **changed** the
design sketched in HANDOFF §N.4.0:

| finding | consequence for E16 |
|---|---|
| an sbi captured in another process reconstructs the position **exactly** (60/60, option list bitwise identical) | the probe is possible at all |
| the rollout is **not** reproducible given a fixed world — the engine draws its own shuffles/coins | ⛔ **true common random numbers are NOT available.** §N.4.0's "CRN" is unachievable; a shared determinized world is the only pairing there is |
| a shared world still removes **~50%** of the variance (ρ≈0.50) | pairing is worth keeping, it just is not CRN |
| **101 ms** per rollout to terminal | 🔴 **§N.4.0's "per-decision resolution is unaffordable" is FALSE** — ±0.020 on a pooled Δ costs ≈96 min on one core. Pooling across a class is a *choice*, not a necessity |
| the fork **silently accepts a decklist the seat is not playing** and returns a plausible number | ✅ **defused, not merely flagged:** each seat's registered 60 is read out of the replay (`seat_decklist()`, 50/50 recovered, validated 20/20 on our own seat). ⚠ The first fix — "restrict to the mirror" — was **wrong**: only **18 of 50** `mirror_experts` seats run our exact list |
| the clone's own top vs last option reads **+0.120 [+0.052, +0.189]** (clustered) | the instrument has real resolution, and this is the **scale bar**: a deliberately bad move is worth ~0.12 |
| only **11 of 40** of our own positions sit in win-probability [0.15, 0.85] | most positions are near-ceiling and bound what any instrument can see there |
| 🔴 **pairs are clustered inside positions** — three runs of the same cell read +0.130 / +0.107 / +0.120 against a nominal ±0.017 | **every interval here is clustered on the POSITION.** The naive pair-level interval is 4.1× too narrow and is forbidden |

## Population

`replays/mirror_experts` — **257 games**, both seats on the Grimmsnarl
archetype. Expert seats are the 1150+ pilots F1 already mined from this dump
(22,665 expert decisions).

🔴 **Correction to this section's first draft, made before any cell ran.** It
said the wrong-deck hazard was "controlled by construction" because both seats
are Grimmsnarl. **That is false** — only **18 of 50** sampled seats run our
exact 60; the rest are 1–3 card variants. Every seat is therefore determinized
with **its own registered list**, recovered from the replay, and a seat whose
list cannot be recovered is **skipped, not defaulted**.

**Unit:** a MAIN decision of an expert seat with `minCount ≤ 1 ≤ maxCount`,
≥3 options, turn ≥ 2, live game, **where the expert's actual pick differs from
our net's top-ranked option.** Agreements carry no contrast and are used as a
control instead (below).

## Arms

At each qualifying position, paired on the determinized world (same seed, same
world, both arms):

| arm | forced first action | continuation |
|---|---|---|
| **A** | the **expert's actual** pick | shipped clone, both seats |
| **B** | **our net's top** pick | shipped clone, both seats |

Value is scored from the **expert seat's** view: 1.0 win / 0.5 tie / 0.0 loss.
**Δ = E[A − B].**

⚠ **What Δ means, stated before it is read.** This is win probability under
**clone-vs-clone continuation** — the value of a *one-step deviation from our
own policy*. That is exactly the right question for "should our net have played
their move" and the wrong question for "is this move good in the abstract".
Every use of the number must carry that clause.

## Controls — each can kill the run

1. **C1/C2/C4 from `p80` re-run on THIS dump**, not inherited from the
   `submission_v5_s2` run. Reconstruction fidelity must be ≥99%.
2. 🔴 **The agreement control (free, and the strongest one).** On decisions
   where the expert's pick and our net's top pick are **identical**, the two
   arms are the same action, so the procedure must read **Δ = 0** within noise.
   A non-zero reading there means the harness is broken and every other number
   is void.
3. **The scale bar.** §8bw's top-vs-last **+0.130** on our own games is the
   reference magnitude. A Δ of +0.005 is not "small but real" — it is 4% of a
   deliberately bad move, and must be reported against that ruler.
4. **`--exclude` ourselves** from the expert population (rule 18's third form):
   `Scio` must not appear as an "expert" seat.

## Sizing, pre-committed

🔴 **Size on positions, not pairs.** §8bw's between-position sd is what sets the
interval; the pair-level sd understates the budget 4×. **600 positions × 30
pairs ⇒ ≈ ±0.018**, costing 600 × 30 × 2 × 101 ms ≈ **60 min** on one core.
Budget 3× for contention (rule 7).

⚠ **30 pairs per position is deliberate.** Pairs beyond ~30 buy almost nothing
once the position is the unit of variation — the budget belongs in *more
positions*, not deeper ones. This is the same arithmetic as rule 21 (the unit of
the size must be the unit the effect lives in), one instrument over.

## Pre-registered kill criterion and readings

**To claim the experts' moves are better, Δ's 95% CI — clustered on the
position — must exclude 0 at ≥ 600 positions.** Otherwise E16 reports a null.
⛔ **The naive pair-level interval may not be quoted anywhere**, per §8bw.

| reading | interpretation, written before the run |
|---|---|
| **Δ > 0, CI excludes 0** | the experts' moves are genuinely better *in our own continuation*. First non-conformity evidence that they outplay us move-by-move; the exploratory per-class breakdown then says where |
| **Δ ≈ 0** (CI inside ±0.02) | their per-move advantage is invisible to a clone continuation. ⚠ **H1 and H2 both predict this**, which is why arm C exists |
| **Δ < 0, CI excludes 0** | their moves are *worse* under our continuation — the strongest possible form of H1 ("partial copying breaks the coherence"), and a direct explanation of §8u's −92 Elo |

## Arm C — the H1/H2 discriminator, run ONLY if Δ ≤ 0

§N.3 states the two hypotheses E16 exists to separate. If Δ ≤ 0 they are still
tied, and the tie-breaker is the **continuation policy**: repeat both arms with
the clone replaced by `out/policy_b7_ntum.npz` — the B7 single-expert fine-tune
(agreement with ntumlnoob 67.2%, measured **−92 Elo as a standalone agent**,
§8u). It is not a good player; it is a player that follows up like the expert.

**Read it as a difference-in-differences**, never as a level:

```
DiD = [A − B | expert continuation] − [A − B | clone continuation]
```

⚠ **The DiD is required, not optional.** `b7_ntum` is a weaker net overall, and
a weaker continuation changes how long any early advantage persists — a *level*
effect that hits both arms and cancels only in the difference. Reporting the
expert-continuation arm on its own would be §8ax's error one instrument over.

| DiD reading | conclusion |
|---|---|
| **DiD > 0** | the expert's move is good *when followed up their way* and neutral when followed up ours ⇒ **H1: coherence is the mechanism**, and commitment (N.4.2) becomes the lever |
| **DiD ≈ 0** | the move's value does not depend on who continues ⇒ **H2**, and the thread turns to credit assignment (N.4.3) |

## Exploratory, and labelled as such in advance

A per-decision-class breakdown of Δ (option type, turn bucket, board
occupancy). **Exploratory means it cannot produce a shipped rule on its own** —
any candidate it suggests re-enters through rule 14 (size first) and rule 11
(dominated vs tradeoff, currently 0-for-5 on tradeoffs).

## What this cannot become

⛔ **Not a training target.** Demonstrator selection is closed twice over
(§8t −55 Elo, §8u −92 Elo) and "agreement with the expert anti-predicts
strength" is measured. E16's output is a **measurement**, and at most a rule in
the §8f/§8y audit lineage.
⛔ **No re-cut on a knob after seeing the result** — E11 and E15 both
pre-registered that prohibition and it holds here.
