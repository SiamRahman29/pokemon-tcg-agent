# E17 — what does a rollout oracle over OUR OWN options buy, at a budget the 600 s clock can actually pay?

**Status: PRE-REGISTERED 2026-08-10 (day 29). Frozen before the first treatment
cell. Nothing below was written after seeing a number.**

Driver: `scripts/p82_e17_self_oracle.py` (to be written).
Instrument: `scripts/p80_rollout_feasibility.py` (`EVIDENCE` §8bw) — fork, rollout,
`seat_decklist`, `positions`, C0, C1. Net pinned to the shipped `55326513`
weights.

---

## 1. Why this exists, given that §2.7's gate already PASSED

ROADMAP §2.7's sizing gate ("90th-percentile |gap| ≥ 0.10 ⇒ build") was answered
by E16's dispersion at **0.1263** (§8bx). That gate is passed and this document
does not re-litigate it. **It was passed by a proxy, and §8bx says so in its own
caveat (3):**

> *"The dispersion is measured between our pick and the expert's pick; a rollout
> oracle ranks the net's **own** options, and their gap distribution is similar
> but not identical."*

The clock has no expert. At a real decision it will rank **the net's own top-k
options** and play the argmax. Two quantities decide whether that is worth
building, and **neither has been measured**:

1. **The gap distribution among our own top options.** E16's population was
   *positions where a 1050+ expert played what we would not* — a set selected
   for disagreement with a strong player. The clock fires at every decision we
   choose to spend on, selected by nothing.
2. **How much of the ceiling a *noisy* oracle actually keeps.** §8bx's caveat
   (1) puts it at *"roughly half survives at gap/SE ≈ 2"*. **That is an
   arithmetic guess, not a measurement**, and it is the multiplier the whole
   build rests on. Selection on noisy estimates is exactly the failure mode that
   killed F2 (§8bh: the screen's error equalled the effect's sd, so the max
   selected for measurement error).

This is the sizing step §2.7 skipped, in the lineage of rule 14 (*size before
you build*) and §8ae (*price the blocking objection before paying for the
build*). It costs ~1.5 h of one core and it gates an engineering programme
(batched rollouts, an online budget manager) measured in days.

⚠ **And it is deliberately run before the throughput engineering**, because
§2.7's own analysis says the engineering is the expensive half (batching,
process parallelism, rewriting featurize — *"all of it is engineering, not a
flag"*). A null here saves that entire investment.

## 2. Population — OUR games, not the experts'

`replays/submission_v5_s2` (76 games of the shipped agent), our own seat only,
via `our_seat()`. Live MAIN decisions, turn ≥ 2, ≥ 3 single-index options
(`positions()`), each determinized with the seat's own registered 60
(`seat_decklist()`, validated 20/20 on our seat, §8bw).

**Why on-policy and not `mirror_experts`:** the clock runs in *our* games at
*our* positions. E16's population cannot answer this question because it is
conditioned on an expert disagreeing with us, which is not available at play
time and is not the distribution the clock will meet.

## 3. Design

At each sampled position, with `net.scores(o)` ordering the options:

* arms = the net's **top-1, top-2, top-3** (the realistic contest — a real
  oracle ranks a handful, not all ~6);
* **R = 40 replicates**. Replicate *k* determinizes **one** world
  (`random.Random(seed)`, same seed for all three arms) and rolls each arm out
  to terminal with the clone piloting both seats. That shared world is the only
  pairing available (⚠ **not CRN** — §8bw C2: the engine draws its own
  shuffles/coins);
* 120 rollouts/position ≈ 12 s at §8bw's 101 ms; **250 positions ≈ 50 min**.

Every per-replicate value is **saved to disk** (§8bx had to re-run because the
first E16 run discarded them).

## 4. The quantities, and what each is for

| # | quantity | what it decides |
|---|---|---|
| **Q1** | mean Δ(top-2 − top-1), clustered on position | is the net's own ranking right *on average*? Prediction: **≤ 0**. A significantly **positive** Q1 would be a much bigger and cheaper finding than the clock — a free re-ranking — and would be reported as such |
| **Q2** | true between-position sd of that gap, by deconvolution (observed sd² − measurement noise²), ⇒ E\|X\| | the ceiling's raw material, and the direct comparable to §8bx's 0.0768 / 0.0866 |
| **Q3** ⭐ | **perfect-oracle ceiling** = E[max over the 3 arms − top-1], estimated with a leave-out correction | the upper bound. **The kill gate reads this** |
| **Q4** ⭐⭐ | **realized gain of a BUDGETED oracle**: split the R replicates into disjoint selection (R_sel) and evaluation sets, pick argmax on the selection means, score the pick on the evaluation set, minus top-1's evaluation mean. Averaged over many random splits, clustered on position. Reported as a **curve** over R_sel ∈ {5, 10, 20, 30} | **the headline. This is the deliverable the clock would buy**, and the split-sample construction is what makes it unbiased under selection noise |
| **Q5** | semi-analytic extension of Q4 to R_sel ∈ {60, 130, 200} from the measured per-arm noise and the deconvolved gap distribution | the empirical R_sel caps at 30 for R=40; a real 600 s allocation is **60–200 pairs/arm** (§7). ⚠ **Labelled MODEL-BASED**, never quoted as a measurement |
| **Q6** | Q4 and \|gap\| conditioned on the **logit margin** (top-1 − top-2 score), turn, option count, and the position's win probability | an online budget manager must decide **where** to spend. If the gap is flat in every free covariate, the clock must spend blind and Q4 is the whole story. WP is estimated from the **selection** split only, so the stratification is independent of the evaluated gain |

## 5. Controls — any failure voids the treatment

* **C0** (`p80.c0_alignment`) — the recorded action reproduces our net's pick on
  our own games, ≥ 95%. Pins the observation/action pairing.
* **C1** (`p80.c1_fidelity`) — the forked position is bitwise the replay's,
  ≥ 99%.
* 🔴 **C-identical — the winner's-curse control, and the one this design exists
  to survive.** Run the *entire* Q4 procedure at ~60 positions with all three
  arms set to the **same** option (top-1 three times). The realized gain **must
  read 0**. If it reads positive, the estimator is selecting on noise and
  scoring on that same noise — the exact defect that produced F2's phantom
  screen winner (§8bh) — and every number in §4 is void. This control is free,
  it is the reason the split is disjoint, and **it is checked before the
  treatment is reported**.
* **scale bar:** §8bw's clone top-vs-last = **+0.120**. Every Δ is quoted
  against it.
* **clustering:** every interval is clustered on the **position**, never the
  pair (§8bw: the pair-level interval is 4.1× too narrow, caught by replication).

## 6. 🔴 The decision rule — written before the run

Primary quantity is **Q4 at R_sel = 30** (≈ 90 rollouts ≈ 9 s at one decision —
*inside* a realistic allocation, so it reads as a **lower bound** on a real
budget). Secondary is **Q3** (the ceiling).

| branch | condition | consequence |
|---|---|---|
| 🔴 **KILL** | Q3 ≤ **+0.015** WP/decision, **or** Q4@30 ≤ **+0.010** with a CI covering 0 | **The clock dies for Round 1.** Not "parked" — closed, written up, and §2.7 is marked with the verdict. The value E16 found at *expert-disagreement* positions does not exist among our own options at a payable budget |
| ✅ **BUILD** | Q4@30 ≥ **+0.020** with the CI excluding 0 | proceed, in this order: (1) batched rollouts + process parallelism for throughput, (2) the online budget manager, (3) the agent, (4) the large-or-nothing A/B |
| 🟡 **NARROW** | between the two | build **only** if Q6 finds a free, online-computable trigger selecting ≥ 25% of decisions at Q4@30 ≥ **+0.030**. Otherwise it is a chapter |

## 7. What the budget arithmetic behind R_sel actually is

600 s/game, currently spending 1.12 s. At §8bw's 101 ms/rollout (⚠ ±30%, rule 7
CPU contention) a 500 s allocation is ~5,000 rollouts/game:

| decisions bought | rollouts each | pairs/arm at k=3 |
|---|---|---|
| 10 | 500 | **165** |
| 15 | 333 | **110** |
| 30 | 166 | **55** |

⇒ R_sel = 30 is **conservative by 2–6×** and Q3 bounds the other end.

## 8. ⚠ What E17 CANNOT establish, stated in advance

1. **Per-decision WP gains do not add across a game.** A positive Q4 does not
   give a win-rate number. Only the A/B does, and the A/B is the
   **large-or-nothing** blocker (§2.7: n=2000 at 60 s/game is ~33 h/arm).
2. **The rollout value is win probability under clone-vs-clone continuation**,
   not game-theoretic value (§8bw). An option that is good only because the
   *clone* mishandles the alternative will score well here and gain nothing
   against a different opponent.
3. **Selection noise is modelled, opponent adaptation is not.** A real opponent
   is not the clone.
4. Q5 is model-based. Q3 is an upper bound. Neither ships a number on its own.

## 9. Prediction, recorded before the run

Q1 ≤ 0 (the net's ranking is right on average — it is a clone of the field mode
and §8bx found the *experts'* moves worth only +0.007 over ours). Q2 in the
0.05–0.09 band, i.e. §8bx's dispersion roughly replicates on our own options.
**Q3 in +0.02…+0.05, Q4@30 in +0.010…+0.025** — that is, straddling the gate,
which is why the gate has three branches and not two.
