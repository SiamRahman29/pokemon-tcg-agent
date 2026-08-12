# E28 — do WE repeat our own last action more than the experts do? Pre-registered.

**Status: PRE-REGISTERED 2026-08-12 (day 31), before any pass over any replay.**
Frozen at the commit adding this file. Round context:
`ROUND-2026-08-12.md`. **Zero arena games** — this reads replays we already hold.

**Confirmed never run.** Repo-wide grep for copycat / perseveration /
previous-action / action-history returns only `scripts/p77_wp_regret.py` and its
log, which use "last action" for boundary attribution (our last action of a turn
is folded together with the opponent's whole reply, reported and never
attributed). §N.4.1 — *do we switch commitments more often than the experts?* —
is recorded in `E26-coherence-at-matched-rate.md:189` as **"unbuilt and unrun"**.

---

## 1. The hypothesis, and why it is not a re-run of the plan thread

[Wen et al. 2020, arXiv:2010.14876] — the **copycat problem**. When a policy is
cloned from data where the expert's actions are correlated in time, and
`a_{t-1}` is recoverable from the observation, the net learns to predict the
*previous* action rather than the *next* one. It is a shortcut: held-out
agreement goes **up**, causal grounding goes **down**. At deployment the agent
conditions on **its own** previous action, so an error becomes the evidence for
the next one, and errors lock in rather than averaging out.

⚡ **Why it is worth a cell here specifically.** The closed encoding work (E7/E8)
ended on the rule *"a defect gradient descent can compensate for is a bug in the
code, not a limit on the agent"* — the optimiser had already routed around all
three embedding defects. **Copycat is the opposite class: the optimiser does not
route around it, it creates it**, because the shortcut lowers training loss.
That is the one class of defect where "is the net's behaviour currently
CONSTRAINED by this?" plausibly answers yes.

⚡ **And the channel provably exists in our input.** `agents/sa/features.py`
encodes `turnActionCount` (line 91) and the `my_discard` / `opp_discard` id bags
(`BAG_NAMES`, lines 203–204, 314–315). Anything played this turn sits in
`my_discard`, so `a_{t-1}` is partially recoverable **without any explicit
history feature**. ⚠ The honest split, which this cell does not prejudge:

- the once-per-turn quartet (`supporterPlayed`, `energyAttached`, `retreated`,
  `stadiumPlayed`) is **legality** state — "you already did this, you cannot
  again" — which pushes *away* from repetition. It is **anti**-copycat.
- `my_discard` is **dual-use**: it is also how the net counts resources (how many
  Ultra Balls remain). The shortcut hypothesis is narrow — that the net reads it
  as *"we are mid-item-chain, chain another"* rather than as a count.
- ⚠ Counter-evidence on the record: `turnActionCount` shipped inside the v4
  block that measured **+37 Elo**. The temporal channel is net-useful. That is
  not evidence that no shortcut rides along with it; both can be true.

**This is not the plan thread re-opened.** B10/§8bv, E16/§8bx and arm C all asked
*does coherence help?* This asks *are we mechanically repeating ourselves?* —
the opposite sign, a different instrument, and no latent variable, no clustering
and no MI-over-clusters, so it dodges every trap §8bv hit.

---

## 2. ⛔ Step 0 — the sizing gate, before any statistic is computed

Report, per side, the number of **consecutive within-turn decision pairs** after
stratification by decision kind. Available on paper: ~4,700 dated-dump games
(`replays/2026-07-26/` … `2026-08-07/`), `ntumlnoob_31-07-2026` (331),
`sixth_sense_31-07-2026`, `mirror_experts`, and our own
`ours_mirror_rec` / `submission_optv3`.

⛔ **If any stratum holds < 500 pairs on either side, that stratum is dropped and
said so.** ⛔ If the pooled count falls below 2,000 pairs per side, the cell is
**VOID**, not null.

⚠ **Boundary pairs are excluded, not attributed** — `p77`'s rule binds here for
the same reason: a pair spanning the handover carries the whole opponent turn.

---

## 3. The instrument — one pass, three readings

**Sides.** `us` = our shipped net's own games. `them` = the same-deck expert
dumps. Both are on our exact 60, so the action alphabet matches.

⚠ **The expert dumps are dated 07-31, before the board's top reshuffled.** They
are the best demonstrators we hold, and §8bq's "three current Grimmsnarl
experts" may already be stale. If E29's mine lands fresh same-deck dumps before
this pass runs, use them **in addition**, never instead, and report both.

### Reading 1 — self-predictability (the copycat statistic)

Predict `a_t` from `a_{t-1}` **alone**, over a **coarse pre-registered action
alphabet** (decision kind × card class), fitted on half the pairs and scored on
the held-out half so that neither side is inflated by fitting.

Report **held-out accuracy** and **plug-in MI**, pooled *and* stratified by
decision kind. ⚠ **Stratification is mandatory, not optional**: a difference in
*which decisions we face* would otherwise masquerade as self-predictability.
⚠ **Coarse on purpose** — §8bv's plug-in MI positive control read **negative**
under a too-fine conditioning bucket.

**Two controls, both required before the comparison may be read** (§8bv's rule:
an estimator gets a control or its number is not admissible):

| control | construction | pre-registered prediction |
|---|---|---|
| ⬆ **positive** | a synthetic trace that repeats its previous action wherever it is still legal | held-out accuracy **≥ 0.90**; if it does not clear that, the estimator cannot see repetition and the cell is **VOID** |
| ⬇ **negative** | our own trace, actions shuffled **within each turn** | accuracy at the marginal base rate, MI **≈ 0**; if it reads materially above, the alphabet is leaking turn position and the cell is **VOID** |

### Reading 2 — commitment switches (§N.4.1, HANDOFF probe 1)

Per game **and** per decision, same denominators both sides:

- **attacker switches** — the active we are investing into changes while the old
  active is still alive;
- **target switches** — the opponent Pokémon we are damaging changes before it
  is KO'd.

### Reading 3 — prize maps (report track, no verdict branch)

Classify each game's realised KO sequence (2-2-2 / 1-1-2-2 / 2-1-2-1 / other)
and report whether ours is consistent with a single map or wanders. Descriptive
only; it is vocabulary for `STRATEGY.md` and the commitment object reading 2
needs. ⛔ **Not** a feature and **not** a rule — the encoding axis is closed, and
the dominated/tradeoff discriminator predicts a rule here loses.

---

## 4. Reading rules — keyed on the COMPARISON, written before the pass

⚠ E25's own branch condition keyed on one arm and fired correctly for a reason
it did not establish. **Every branch below is a joint condition on readings 1
and 2**, whose predictions point in *opposite* directions.

| reading 1 (self-predictability) | reading 2 (switches) | verdict |
|---|---|---|
| **ours > theirs**, CI excludes 0 | **ours < theirs** | ✅ **PERSEVERATION.** The two agree. Licenses **E31** (§5) and 🔴 **predicts HANDOFF probe 2 — adding hysteresis — is NEGATIVE**; do not run it as written |
| ours ≈ theirs | ours > theirs | **H1 as originally stated.** Copycat is not our failure mode; E31 unlicensed |
| ours ≈ theirs | ours ≈ theirs | 🔴 **Both die.** §N.4.1 closes the way §8bv did, and the thread turns to N.4.3 (credit assignment) by its own logic |
| **ours > theirs** | **ours > theirs** | ⛔ **VOID, not a finding** — the readings contradict; suspect the instrument, not the agent |

⚡ **On point predictions.** E25's lesson is that a *missed point prediction*
does the work a bar cannot. A numeric prediction for the treatment is not
available before the pass, so the **controls carry the point predictions**
(§3, ≥0.90 and ≈0) and the treatment is judged on the **comparison**, which is
the instrument here. Recorded so this cannot be narrated as a bar afterwards.

---

## 5. What a positive result licenses — and what it does not

⛔ **A fix is NOT licensed by this cell.** If reading 1 fires, the follow-on is
**E31**, pre-registered separately before any retrain:

1. **Split the channel, don't ablate it.** Separate `my_discard` at the *start of
   the turn* (resource counting — causally sound, keep) from cards discarded
   *during this turn* (the trace — drop). A blanket zero of the bag also
   destroys resource counting, so its null would be ambiguous by construction.
2. The adversarial-head fix (gradient-reversed `a_{t-1}` predictor) is **third
   in line and unlicensed** unless the split shows the trace is load-bearing and
   too costly to remove outright.

Either way E31 is a retrain plus an n≥2000, ≥3-seed A/B against a byte-identical
control, and it competes with E30 for arena time before 08-17.

---

## 6. ⚠ Limits, stated before the result

- **A realised trajectory cannot see an error of omission** (§8bm). This counts
  the switches we *made*; it cannot price the ones we failed to make.
- It measures the shipped net's **behaviour**, not the channel's **causal role**.
  Only E31's split A/B can do the latter — which is exactly why the fix is
  gated rather than bundled.
- `us` and `them` are different populations at different ratings, so any
  difference is confounded with strength. ⚠ **This is the §8r conformity trap
  wearing new clothes**, and the defence is the *direction*: H1 and copycat
  predict opposite signs on reading 2, so the sign is informative even though
  the level is not. **The level alone will not be quoted.**

---

# ▶ RESULT — 2026-08-12, same day. Step 0 PASSES; reading 1 is VOID on its own positive control.

Instruments: `scripts/p87_e28_pairs.py` (gate), `scripts/p88_e28_reading1.py`
(reading 1 + both controls). ⛔ **The comparison was never computed** — see §R3.

## R1. ✅ Step 0 — the sizing gate PASSES, after a 32% double-count was removed

| side | games | mirror seats | **within-turn pairs** |
|---|---|---|---|
| `us` (`submission_v5_s2` + `submission_optv3` + `ours_mirror_rec`) | 210 | 289 | **26,318** |
| `them` (ntumlnoob / Raja Biswas / Sixth Sense / Dominic Peel) | 554 | 566 | **46,029** |

Both are **>10× the 2,000-pair VOID floor**. **11 strata** clear ≥500 pairs on
both sides (`ctx` 0, 3, 7, 13, 15, 16, 21, 22, 30, 40, 43), holding **97.3%** of
`us` pairs and **96.2%** of `them`. Seven strata dropped and named. Adjacent-but-
excluded (boundary/interleaved) pairs: 3,261 `us`, 6,009 `them`.

🔴 **An instrument defect caught at the gate, before it could matter.** The three
expert dumps **overlap**: `mirror_experts` is a *re-cut* of the other two, not an
independent pull — `ntumlnoob ∩ mirror_experts = 148`, `sixth_sense ∩
mirror_experts = 112`, union **555 against a naive sum of 816**. Pooling them
unguarded double-counts **261 games (32%)** and would have passed the gate on
duplicated data. Deduped by episode id; `mirror_experts` contributes **0** new
games. ⚠ Also: `sixth_sense_31-07-2026` is mostly **Raja Biswas** (158 seats) and
only 69 `Sixth Sense`, so matching on the dump's own name would have silently
dropped 70% of it.

## R2. ⬇ The negative control PASSES — the alphabet does not leak turn position

Within-turn shuffle: held-out accuracy **0.6969** against a marginal base rate of
**0.6939** — lift **+0.0030**, MI **0.0126**. ✅ Ordering carries essentially
nothing once the marginal is matched.

⚡ This also clears a worry raised at the gate: `ctx` 13 and 16 have *exactly*
equal pair counts on both sides (1,921 / 3,237), which looked like a
deterministic alternation that could leak position. The negative control says it
does not.

⚠ **The shuffle was made STRICTER than the text.** Permuting within `turn`
alone would move a card class into a slot where it cannot legally occur, which
*depresses* accuracy below base rate and lets the control pass trivially.
Permuting within `(turn × ctx)` preserves each stratum's marginal exactly and
destroys only the ordering. Recorded because it makes the control harder, not
easier.

## R3. 🔴 The positive control FAILS — 0.8774 against a pre-registered ≥0.90 ⇒ reading 1 is VOID, not null

The frozen rule (§3): *"if it does not clear that, the estimator cannot see
repetition and the cell is VOID."* **It reads 0.8774. The cell is VOID as
written, and the bar has not been moved.**

⛔ **The comparison was NOT computed.** `p88` exits before evaluating `them`;
the gate is enforced in code, not left to the reader. Nothing about `them` is
known, which is what makes §R4 legitimate rather than a rescue.

**The cause is measured, and it is the control's construction, not the
estimator's sensitivity:**

| | |
|---|---|
| within-turn pairs in kept strata | 24,096 |
| **repetition is LEGAL at `b`** | **12,365 — 51.3%** |
| slots where it is illegal (control kept the *real* action) | 11,731 — **48.7%** |
| the synthetic trace does repeat, where it can | 11,852 / 12,365 |
| what `us` actually does | repeats the previous class **37.3%** of the time |

⇒ **The "positive control" was only ~51% synthetic.** §3 specified *"repeats its
previous action wherever it is still legal"* but left **unspecified what the
trace does where repetition is illegal**; that gap was filled with *"keep the
real action"*, which injects genuine unpredictable behaviour into half the
trace. A trace that is half-real cannot reach 0.90 no matter how sensitive the
estimator is — and reading **0.8774 while only half-deterministic is evidence
the estimator sees repetition well**, not that it is blind.

⚡ **The transferable lesson, and it is E25's in a new place:** a point
prediction is only as good as the construction it is keyed to. **≥0.90 was set
before anyone knew repetition is legal only 51.3% of the time.** A bar on a
quantity whose ceiling is an unmeasured property of the data is a bar on the
data, not on the instrument. ⚠ **This is the third session in a row in which a
control was found mis-specified** (E27 found three biased *toward* its
hypothesis); this one is biased *against*, which is why it surfaced as a VOID
rather than as a false positive.

## R4. ⛔ Pre-registered correction, frozen BEFORE it runs

Re-registered because the defect is in the **control's construction** and is
diagnosed from a structural fact (the 51.3% legality rate) that is **independent
of the treatment** — and the treatment was never read. Two changes, both fixing
the same unspecified degree of freedom:

1. **The positive-control trace becomes fully synthetic.** Where repetition is
   illegal, the trace takes a **deterministic fallback (lowest-index option)**
   rather than the real action, so the trace is a deterministic function of
   (previous action, option list). ⇒ **the ≥0.90 bar becomes a statement about
   the estimator**, which is what it was always meant to be.
2. **Reported alongside it: accuracy restricted to the legal-repetition
   subset**, which is the bar's intent expressed on the slots where repetition
   is even possible.

⛔ **The ≥0.90 threshold is UNCHANGED**, and it now applies to construction 1.
⛔ If the corrected control still misses 0.90, reading 1 is **VOID and stays
VOID** — there is no third construction. ⚠ Reading 2 (commitment switches) does
not depend on this estimator and is unaffected either way; §4's branches need
both, so no verdict may be issued until reading 1 resolves.
