# E31 — trace reliance: does the clone over-use "what I just did"?

**Pre-registered 2026-08-12 (day 31), frozen at the commit adding this file.**
Charter: `ROUND-2026-08-12.md`. Predecessor: `E28-replay-trace-audit.md`, whose
reading 1 is **VOID** — this is the re-attempt §R8 required, and §R8's two
conditions are met explicitly in §2 below.

⚠ **The user has directed that a fix land by 08-17.** I said the calendar was
tight; that concern was heard and overruled, which is their call. §6 states the
schedule honestly and §5 names the one branch that cannot make it.

---

## 0. What E28 left, and what changed on inspection

E28 could not answer the question. Its reading 1 died on a positive control
(0.8774, then 0.8759 corrected, against a frozen ≥0.90) because **the cue
`(ctx_a, class_a)` could not represent the target** on the 48.7% of slots where
repetition is illegal. §R8: a re-attempt *"needs a NEW pre-registration with a
cue that includes the option list at `b`* — *and a reachability check on its bar
before the bar is frozen."*

⚡ **Before writing this file, the two candidate channels were measured in the
corpus, and the result reverses E28 §5's proposed fix.**

| channel | what it carries | measured |
|---|---|---|
| **`turnActionCount`** (`xdense[:,0]`) | an **exact within-turn action index**, one dense scalar, present at every decision, undiluted | steps +1 at **49,521 of 59,598** same-game adjacencies in shard 0 |
| **`my_discard`** (id bag) | the whole pile, **mean-pooled** into one embedding (`policynet.py:177`) | pile is **10.1 cards** mean / 9 median ⇒ one card moves the mean ~10%; and **82.8% of within-turn steps do not change it at all** |

🔴 **E28 §5 aimed its fix at `my_discard`. That was wrong and is withdrawn
here.** A channel that is unchanged on 83% of consecutive decisions, and diluted
tenfold when it does change, cannot be a meaningful within-turn trace.
**`turnActionCount` is the trace channel** — it is the only input that tells the
net, exactly and always, how many things it has already done this turn.

⚠ **This is not yet evidence that the net over-uses it.** §8ab measured dropping
`turnActionCount` alone as **within noise** (±13 Elo, 2 seeds) — weak, and it
measured *removal*, not *reliance*. §8y sized the channel and it targets real
miss mass (2,629 of 3,902 misses are MAIN). **A trace channel that is genuinely
informative is not a defect.** The defect is only ever *over*-use, which is why
every reading below is a **difference against the demonstrator**, never a level.

---

## 1. Hypothesis

**Copycat / causal confusion** (Wen et al., arXiv:2010.14876): a cloned policy
latches onto a nuisance correlate of the expert's *previous* action, because
that correlate predicts the *next* action better than the causal state does —
inside the demonstration distribution. It then compounds errors when it must act
on its own trace instead of the expert's.

⚡ **The sharp form of the claim, and the only one this cell tests: the net uses
the trace MORE than its own demonstrator does.** If net and expert lean on it
equally, the clone has faithfully copied a real regularity of the game and there
is nothing to fix. Copycat is *amplification*, not *presence*.

---

## 2. ⛔ Why this pre-registration is reachable where E28's was not

E28's VOID was a statement about its pre-registration, not about the agent. Both
causes are removed by construction, and this section is the reachability check
§R5 demanded:

1. ✅ **The cue includes the option list at `b`.** The base predictor is
   *"which symbol is chosen from the symbols available"*; the trace predictor
   adds the previous symbol. The 48.7%-illegal slots that sank E28 are now
   handled by the base model, which sees exactly the constraint that determines
   them. ⚡ The available-symbol list `avail` was **already extracted** by
   `p87_e28_pairs.py:118` — reading 1 collected it and then excluded it from the
   cue. The repair is to use a field we already had.
2. ✅ **No absolute accuracy bar appears anywhere in the primary reading.** The
   statistic is a **difference of increments computed on identical rows**
   (§3.2). An unreachable threshold cannot be written, because no threshold is
   written. This is charter §3 rule 2 — key on the comparison, not one arm —
   applied to the failure that rule was written for.

---

## 3. Part A — diagnosis. Offline, on the corpus, **zero arena games**.

### 3.0 Frame, and the sizing gate ⛔ before any statistic

- **Population:** `artifacts/pds_v6` — 248,985 decisions, the only corpus
  carrying both `xdense` and the discard bags, and the population the shipped
  net was trained and evaluated on.
- **Pairs:** strict within-turn adjacency — same `gid`, `turnActionCount` step
  **exactly +1**. Turn boundaries and gaps are excluded, not repaired.
- ⚠ **The split is BY GAME, never by row.** Decisions inside a game are
  dependent; a row-wise split would inflate every held-out number here. Fit on
  half the `gid`s, score on the other half.
- ⚠ **All CIs are paired bootstrap over GAMES** for the same reason.

### ⛔ AMENDMENT, 2026-08-12 — made BEFORE the probe existed and before any statistic was computed

**The gate as first frozen was ≥50,000 held-out pairs. It was unreachable, and
this is the second time in two cells that a bar was written without checking it
against the instrument** (§2 was written about exactly this failure, and then
committed it). Measured immediately after freezing:

| | |
|---|---|
| corpus rows / games | 248,985 / **1,603** |
| the trainer's val split (`gid % 20 == 0`) | 13,671 rows / **85 games** |
| all within-turn pairs | **203,676** |
| within-turn pairs *inside the val split* | **11,155** |

🔴 A ≥50,000 floor read against the net's own val split would have VOIDed a
perfectly good cell on arithmetic. **The bar is not lowered to rescue the cell —
the frame is corrected, because the frame was wrong.**

⚡ **Why the val split was the wrong frame.** `Δ_net` is **not an accuracy
against a memorizable label**. It asks how predictable **the net's own argmax**
is from the previous symbol — a property of the policy function, evaluated at
states. What must be disjoint is the **symbol predictors' fit/score split**, and
that has no reason to coincide with the *net's* train/val split.

⇒ **Amended frame, two arms:**

| arm | rows | fit/score split | gate |
|---|---|---|---|
| **primary** | all **203,676** pairs | the 1,603 games split in half by `gid` | **≥50,000** ✅ reachable |
| **robustness** | the **11,155** pairs in the net's val split | same halving, restricted | **≥5,000** ✅ reachable |

⚠ **The primary arm carries a stated bias, and it points AGAINST the
hypothesis.** On the ~95% of games the net trained on, its argmax is pulled
toward the expert's actual action, which drags `Δ_net` toward `Δ_expert` — i.e.
**toward the null**. A confirming primary is therefore conservative. The
robustness arm carries no such bias and is where the sign is checked; ⛔ if the
two arms **disagree in sign**, the cell is **VOID** and neither is reported as a
finding.

⛔ **Stratum floor unchanged:** ≥500 pairs for a stratum to be reported.

### 3.1 The alphabet — coarse on purpose

Symbol = **(select type, card class of the option)**, matching E28 §3 so the two
cells stay comparable. ⚠ Coarse deliberately: §8bv's plug-in MI positive control
read *negative* under a too-fine conditioning bucket.

### 3.2 ▶ Reading A1 — the copycat statistic (PRIMARY)

Two symbol-level predictors, both fitted on the training games, both scored on
the held-out games, on **identical rows**:

| predictor | cue |
|---|---|
| **base** | the multiset of symbols **available** at `t` |
| **+trace** | the above **plus the previous chosen symbol at `t−1`** |

Let **Δ = accuracy(+trace) − accuracy(base)**, computed twice against two label
sources on the very same rows:

- **Δ_expert** — labels are the demonstrator's actual pick (`opt_chosen`);
- **Δ_net** — labels are the **shipped net's argmax** over the same option list.

> ### 🔴 **The statistic is `Δ_net − Δ_expert`.**

| outcome | verdict |
|---|---|
| **Δ_net − Δ_expert > 0**, bootstrap CI excludes 0 | ✅ **COPYCAT CONFIRMED.** The net leans on the trace *beyond* what its demonstrator does. Opens Part B |
| CI contains 0 | 🔴 **NO COPYCAT PROBLEM.** The net inherits exactly the trace dependence the data contains. ⛔ Part B does not run; no retrain, no arena game |
| **Δ_net − Δ_expert < 0**, CI excludes 0 | 🔴 **NO COPYCAT PROBLEM**, and the net is *less* trace-driven than the humans. Same closure |

Reported pooled **and** stratified by select type. ⚠ Stratification is
mandatory: a difference in *which decisions each side faces* would otherwise
masquerade as trace reliance.

**Both controls must pass before the comparison may be read** (§8bv's rule):

| control | construction | prediction, and why it is reachable |
|---|---|---|
| ⬆ **positive** | a **synthetic trace-follower**: picks the option matching its own previous symbol whenever one is available, else falls back to the net's pick | **Δ_synth − Δ_expert > 0, CI excluding 0.** ⚡ Reachable **by construction** — the agent is literally trace-driven, and the quantity is a difference, not a level. This is the property E28's ≥0.90 lacked |
| ⬇ **negative** | the previous symbol **permuted across pairs within the same select type** | **Δ ≈ 0 for both label sources.** If it reads materially above 0, the trace column is leaking stratum identity and the cell is **VOID** |

### 3.3 Reading A2 — localization (interventional)

Runs regardless of A1, because **Part B needs to know which channel to fix** and
this is the only reading that says. On the frozen shipped net, perturb one
channel at a time and measure **Δ argmax rate** over the legal option list:

| channel | perturbation |
|---|---|
| **`turnActionCount`** | resample from the observed marginal *within the same select type* |
| **`my_discard`** | drop the cards added this turn (i.e. the start-of-turn pile) |
| ⬇ *calibration:* `retreated`, `stadiumPlayed` | the once-per-turn legality quartet — **known-used**, and **anti**-trace by nature (they say what may no longer be done) |
| ⬇ *calibration:* a random dense column at matched variance | null reference |

⚠ **Read only relative to the calibration channels.** A perturbed input is
off-manifold and the net was never trained there; the resulting response is
partly OOD for *every* channel, so only the **comparison** between channels is
admissible. ⛔ This bounds **sensitivity, not harm** — it cannot promote or kill
on its own, and it appears in no verdict row of §3.2.

### 3.4 Reading A3 — where it costs (conditional on A1 firing)

Split held-out rows by whether the +trace predictor and the expert **disagree**.
Does the net's own top-1 error rate rise on the disagree rows against matched
agree rows? This is what turns *"uses the trace"* into *"is misled by it"*, and
it is the reading Part B's expected effect size comes from.

---

## 4. Part B — the fix. ⛔ **GATED: runs only if A1 confirms.**

If A1 does not fire, **E31 closes at Part A with zero arena games spent**, like
§8bm/§8bp/§8br/§8bs and E30. That is a real outcome, not a failure.

### 4.1 Which fix, keyed to A2 — frozen now, before A2 reports

| A2 localizes reliance to | the fix |
|---|---|
| **`turnActionCount`** | retrain with it **coarsened to a 3-bucket turn phase** (early / mid / late) instead of a 24-valued index |
| **`my_discard`** | **split the bag** — start-of-turn pile (resource counting, kept) vs discarded-this-turn (trace, dropped) |
| **neither separates from the calibration channels** | ⛔ **VOID for localization.** No fix ships. The diagnosis is reported and E31 closes |

⚠ **"Drop `turnActionCount`" is deliberately NOT on the menu.** §8ab: drop-one
is within noise but **drop-all-three is −36 Elo, disjoint** — the three are
mutually redundant, so removing one risks the whole block for reasons unrelated
to copycat. **Coarsening keeps the phase information §8y sized and destroys the
exact index**, which is precisely the copycat cue. That is the intervention the
hypothesis names, and nothing wider.

### 4.2 Arena protocol — pre-registered before the first game

- Retrained treatment vs a **byte-identical control**: same corpus, same arch,
  same seed, same rows, channel intact. Only the content of a few columns
  differs — the §8ab discipline, which is tighter than removing dimensions.
- **≥3 seeds per arm, n≥2000 per cell.** A treatment-minus-control interval is
  **√2×** a single cell's.
- ⚠ **Report the REALISED changed-picks/game from the run's own logs** and place
  the result on E25/E26's cost law (charter §3 rule 1). A knob is not a
  variable. A fix that moves many picks must beat the **f = 0.758** line, not
  merely 0.500.

### 4.3 ⚠ Pre-registered so it cannot be misread afterwards

**`val_top1` is expected to FALL if the fix works.** The shortcut is present in
the corpus too, so removing the net's access to it *must* cost imitation
accuracy while — on the hypothesis — buying strength. 🔴 Per charter §3 rule 3,
`val_top1` may neither promote nor kill; here it is additionally expected to
point the **wrong way**, and a drop in it is **not** evidence the fix failed.

---

## 5. Calendar, stated honestly

| date | |
|---|---|
| **08-12** (today) | this pre-registration frozen; Part A built |
| **08-13** | Part A reports. ⛔ If A1 does not fire, E31 closes here |
| **08-14** | retrain (treatment + byte-identical control × 3 seeds) |
| **08-15** | arena, and the result is an input to the submission decision |
| **08-17** | simulation deadline |

⚠ **The one branch that cannot make 08-17** is *A1 fires and A2 localizes to
`my_discard`*: that fix needs new bag-splitting machinery in both
`features.py` and `train_policy.py` plus a **corpus rebuild**, because the bags
are not currently maskable (`X_GROUPS` covers only the eight v4 scalars).
🔴 On §0's measurements that branch is also the *unlikely* one. The
`turnActionCount` branch needs **no corpus rebuild** — it is a transform of one
existing column — which is what makes the directed schedule feasible.

---

## 6. ⚠ Limits, stated before the result

- **A2 is off-manifold** and can only be read as a between-channel comparison
  (§3.3). It localizes; it never decides.
- **A1's net labels are argmax**, so the reading is about the net's *decisions*,
  not its full score distribution. Chosen deliberately: the argmax is what plays.
- ⚠ **Reading A1 cannot see errors of omission** — it measures reliance on the
  trace among options actually offered, and says nothing about lines the clone
  never considers. Same limit E28 §6 carried.
- **Part A is entirely within-distribution.** Copycat's worst damage is
  compounding error *off* the demonstrator distribution, which only the arena in
  Part B can see. ⇒ A confirming A1 with a null Part B is a coherent outcome and
  is **not** an instrument failure: it would mean the shortcut exists and does
  not cost enough to matter at our rating.

---

# ▶ RESULT — 2026-08-12, same day. 🔴 **NO COPYCAT PROBLEM. Part B does not run. Zero arena games.**

`scripts/p93_e31_trace.py`, net `out/policy_v5_s2.npz`, corpus
`artifacts/pds_v6`. **181,517 within-turn pairs**, 1,603 games split 801/802.

## R1. ⚠ A third correction, found while wiring the probe: we were about to measure the wrong net

`agents/sa/policy_net.npz` is a **pre-v5 net** (no pooling block, `state_in`
496) and is **not** what plays. Hashing the npz inside `dist/submission.tar.gz`
identifies the live agent as **`policy_v5_s2`**. `build_submission.py` overrides
the bundled net via `--policy-net`, so the file sitting in the agent directory
has been stale for days.

⚡ **This is rule 20 again** (*"a `net=` path is not an identity"*) arriving from
a new direction: **an agent-directory path is not an identity either.** Every
offline probe defaulting to `agents/sa/policy_net.npz` — `context_accuracy.py`
among them — reads a net the arena has not seen. 📌 Left for whoever owns
`HANDOFF.md`: that default is a trap and should be named there.

## R2. ✅ Both controls PASS — the first admissible numbers this instrument has produced

| control | reading | pre-registered requirement | |
|---|---|---|---|
| ⬆ positive (synthetic trace-follower) | **Δ_synth − Δ_expert = +0.0452 [+0.0409, +0.0496]** | > 0, CI excluding 0 | ✅ **PASS** |
| ⬇ negative (previous symbol permuted within select type) | Δ_expert **+0.0019**, Δ_net **−0.0013** | ≈ 0 both | ✅ **PASS** |

⚡ E28 died here twice (0.8774, 0.8759 against an unreachable 0.90). §2's
reachability argument held: **a difference of increments on identical rows is
controllable where an absolute bar was not.**

## R3. 🔴 The statistic is NEGATIVE and its CI excludes zero

| | base | +trace | **Δ** |
|---|---|---|---|
| **expert** | 0.6656 | 0.7287 | **+0.0632** |
| **net** | 0.6912 | 0.7375 | **+0.0464** |
| synthetic follower | 0.6930 | 0.8014 | +0.1084 |

> ## 🔴 **Δ_net − Δ_expert = −0.0168 [−0.0201, −0.0133]**, n = 90,591 scored pairs

| arm | statistic | n |
|---|---|---|
| **primary** (all pairs) | **−0.0168 [−0.0201, −0.0133]** | 90,591 |
| **robustness** (net's val split) | **−0.0381 [−0.0570, −0.0169]** | 9,977 |

⇒ By §3.2's third row: **NO COPYCAT PROBLEM, and the net is LESS trace-driven
than the humans it copied.** ⛔ Part B does not run. No retrain, no arena game.

**Two internal checks, both of which the pre-registration named in advance:**

1. ✅ **The arms agree in sign**, so §3.0's VOID condition did not fire.
2. ⚡ **They disagree in MAGNITUDE in exactly the direction the amendment
   predicted.** The primary arm was declared biased *toward the null* by
   memorization on the ~95% of games the net trained on; the unbiased val arm
   should therefore sit **further** from zero. It does — **−0.0381 against
   −0.0168.** The bias analysis was written before the run and the data
   reproduced it.

⚡ **The shortcut was real and available and the clone declined it.** The
experts' own Δ is **+0.0632** and a purely trace-driven agent reaches
**+0.1084**, so trace dependence is a genuine regularity of this game sitting in
plain view in `turnActionCount`. The net uses it **less** than its demonstrators.

## R4. ⚡ The finding that outlives the null: the clone is MORE predictable, but not from the trace

**Net base accuracy 0.6912 vs expert 0.6656.** From the option list alone —
before any trace is supplied — **the clone's next action is more predictable
than a human's.**

⇒ Its excess predictability is **modality, not perseveration.** It is not
repeating what it just did; it is playing the modal option more often than the
people it learned from. 🔴 That is the *opposite* failure from copycat, and it
is the same object as the standing `val_top1`/conformity result — **a perfect
conformity score is compatible with being modal, and modal is ~1000** — reached
here from an instrument that had nothing to do with agreement.

## R5. A2 — localization. Both trace channels are mildly load-bearing; nothing dominates.

Resampling one channel and counting argmax changes, n = 248,985 decisions:

| channel | argmax changed | vs null reference |
|---|---|---|
| `my_discard` (reverted to the start-of-turn pile) | **1.93%** | 2.5× |
| `turnActionCount` (resampled within select type) | **1.73%** | 2.3× |
| `retreated` (legality calibration) | 0.83% | 1.1× |
| `stadiumPlayed` (legality calibration) | 0.79% | 1.0× |
| **`dense_ctl`** — matched-variance dense column, **null reference** | **0.76%** | — |

- ⛔ **This changes no verdict.** §3.3 excluded A2 from every branch of §3.2: it
  bounds **sensitivity, not harm**, and it is off-manifold.
- Both trace channels separate from the null reference and from the legality
  pair, so they are not inert — but **destroying either changes under 2% of
  decisions**, which is a small lever to have been worth a retrain even if A1
  had fired.
- 🔴 **§0's prediction that `my_discard` would be the weaker channel was NOT
  borne out** — it reads marginally higher. ⚠ But the two perturbations are
  **not magnitude-matched** (a within-stratum resample of a scalar vs reverting
  a whole pooled bag to its turn-start state), so this neither confirms nor
  refutes §0. It is recorded as an open mismatch, not as a result.

## R6. ⛔ What E31 does and does not license

- ✅ **The copycat question is ANSWERED**, and answered in the negative. It moves
  from *unmeasured* (E28 §R8) to *measured and refuted* on 181,517 pairs with
  both controls passing.
- 🔴 **No fix ships**, and this is the pre-registered outcome, not a retreat.
  Coarsening `turnActionCount` would now be a retrain aimed at a defect measured
  **absent**, removing a channel the net **under**-uses, against §8ab's −36 Elo
  showing the v4 block is not free to disturb.
- ⛔ **HANDOFF probe 2 (hysteresis) is now predicted NEGATIVE and should not be
  run as written.** E28 §R8 reverted it to *unprioritised* because the
  perseveration row never fired. It has now fired the **other way**: a net that
  is *less* trace-driven than its demonstrators will not be improved by adding
  stickiness. ⚡ This is the one place E31 closes standing work beyond itself.
- 📌 **Add to the closed list**: copycat / causal confusion, refuted 2026-08-12.
  Sixteen closed axes. ⛔ Do not re-open on a literature argument — §1 of the
  charter, applied to the proposal that generated this cell.
- ⚡ **R4 is the live thread**, and it is not this cell's: the clone is *more*
  modal than its demonstrators. That is a statement about **which** option it
  picks, not about **what it just did**, and it belongs to the conformity thread.
