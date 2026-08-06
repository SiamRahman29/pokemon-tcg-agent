# E3b — the teacher-free near-tie gate

**Pre-registered 2026-08-06 (day 23), before any arm was run.** Written after the
sizing pass and before the A/B, per rule 14.

## Why this exists

E3 is parked at its teacher gate: the review pilot produced no usable labels
because the reviewer had no Pokémon TCG expertise, and the planner is
disqualified as an automatic teacher (it lost to BC at `0.375 [0.311, 0.444]`).
So E3 is not a policy null — it never measured a policy.

But E3's *premise* is testable without any teacher. The queue is built from
decisions where the clone's selected/unselected boundary is nearly tied, on the
assumption that those are decisions worth relabelling. **Whether that band is
worth anything at all can be measured by taking the other side of it and playing
the games.**

## What the sizing found first (rule 14)

`p43_dagger_queue.py --dump-margins` over 115 live ladder games, 19,573 decisions:

| | |
|---|---|
| rankable candidates | **8,963** (77.9 per game) |
| bitwise-equivalent free ties, excluded | 799 (4.1%) |
| median boundary margin | **1.479** logits |
| the review queue's 160 items | margins `[0.0001, 0.1316]` = **the bottom 10%** |

⇒ **The human queue is the extreme tail of the distribution, not the band.** And
the free-by-construction ties (§8x: two copies of one card in one role) are only
8.2% of decisions with a boundary, so they cannot be what makes near-ties cheap.

Flip rates measured in the mirror (8 matches per τ, read off `[health]`):

| τ (logits) | decisions flipped | nearest §8am arm |
|---|---|---|
| 0.10 | **3.0%** | — |
| 0.50 | **17.7%** | tau 0.50 → 20.4% off-argmax, **free** |
| 1.00 | **32.9%** | tau 1.00 → 30.5% off-argmax, **−135 Elo** |
| 2.00 | **48.1%** | tau 2.00 → 44.0% off-argmax, **−494 Elo** |

## Hypothesis

§8am measured the same axis by **temperature** and found a cliff: the first ~20%
of deviations from argmax are free, the next 10% cost ~150 Elo. It could not say
*which* decisions are in the free band, because a softmax deviates stochastically
everywhere. **This experiment indexes the same effect by boundary margin**, which
is a quantity any future intervention can read off a single forward pass.

## Design

`bc:<label>,net=out/policy_v5.npz,flipN` versus plain `bc:v5,net=<the same
file>`, mirror, direct head-to-head, n = 2,000 games per arm.

⚡ **This is the only experiment in this project with no training-seed term.**
§5.6's finding — that a two-seed A/B measures two networks rather than one
intervention — cannot apply here: both arms load the **same weight file**, so the
only difference between them is the flip. The published interval is the whole
interval.

Arms: τ ∈ {0, 0.10, 0.50, 1.00, 2.00}. τ=0 never fires (a margin is ≥ 0 by
construction) and is a harness control that must read 0.500.

## Pre-registered predictions

1. **τ=0 → 0.500**, W≈L. Anything else means the harness is broken, not that the
   probe found something.
2. **τ=0.10 (3.0% of decisions) → null.**
3. **τ=0.50 (17.7%) → null.** This is the E3-relevant arm: it covers the whole
   near-tie band the DAgger queue is drawn from, flipped adversarially.
4. **τ=1.00 (32.9%) → a clear loss, ≲0.40.** 5. **τ=2.00 (48.1%) → ≲0.20.**

⚡ **Arms 4 and 5 are a positive control for the whole probe.** §8am says
deviation at those rates is expensive. If they do not lose, the instrument is
wrong and arms 2–3's nulls mean nothing. **A probe whose null arm has no
companion that fires is not evidence** — that is §8ai's lesson, and it is cheap
to satisfy here.

## What each outcome licenses, written before the data

**If arm 3 is null** — the near-tie band carries no exploitable *average* signal.
⇒ No relabelling that is not better-informed than v5, case by case, can pay: the
band is not systematically mis-ranked, so E3's value rests entirely on the
reviewer being right on individual states.
🔴 **This does NOT kill E3, and the day-23 plan's claim that it would is wrong.**
§8am's own reading is the counter-argument: *"a band of indifferent-on-average
choices is precisely where some are better and some worse, netting to zero. That
is the population an outcome signal exists to sort."* An oracle's value is bounded
by E[|effect|]; this measures |E[effect]|. **The two are different numbers and
only the second is being measured here.**

**If arm 3 loses** — the "near-ties" are not ties. v5's ranking inside the band is
right more often than wrong, and the boundary margin is measuring logit geometry
rather than decision difficulty. ⇒ E3's queue is selecting the wrong states, and
that *is* a finding about E3's construction, obtainable with no teacher.

**If arm 3 wins** — v5 is systematically wrong in the band and the fix needs no
human at all. Considered unlikely and stated anyway, because an unstated third
branch is how a two-branch pre-registration becomes unfalsifiable.

## Command

```powershell
python -X utf8 scripts/p59_e3_flip.py --matches 1000
```

Archive: `out/arena/p59_e3_flip.jsonl`. Log: `out/logs/p59_e3_flip.txt`.
