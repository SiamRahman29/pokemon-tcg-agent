# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** public LB **and** the Strategy Category. Sim deadline **2026-08-17**,
then ~2 weeks continued play; strategy report due **2026-09-14**. Kaggle CLI is
authenticated.

**Standing (read 2026-08-07, full LB — now 6,483 rows):** we are **`Scio`, rank
129 of 6,483, 990.7** — best rank and best score this project has had, on an
agent submitted 08-01 and settled since. Top is `LiamK` 1166.0, then `flg`
1162.7, `Raihan Ramadistra` 1151.1, `Sixth Sense` 1144.9. ⚠ The board grew
3,000 → 5,000 → 6,024 → 6,088 → **6,483** and **the top has reshuffled completely
since day 13** (the old top four — `Majkel1337` 1251.3, `keidroid` 1174.3 — are
no longer there, and the ceiling fell ~85 points). **Treat any ranking as a
snapshot, and never compare a score across board sizes** (§8p: the same tarball
read 958 and 834 on a 4,000- and a 6,024-row board).

✅ **Getting the LB no longer needs 300 paginated calls:**
`competition_leaderboard_download('pokemon-tcg-ai-battle', path='out/lb')`
returns **all 6,024 rows as a zipped CSV in ONE call** (columns: `Rank`,
`TeamId`, `TeamName`, `Score`, `SubmissionCount`, …). The `page_token` walk in §5
still works but is obsolete. **This is what made the day-10 analysis possible** —
it lets any team name in a replay be joined to its rating.

> # ▶ START HERE — DAY 31, LATE (2026-08-12): **FIVE CELLS CLOSED IN ONE DAY, ZERO ARENA GAMES SPENT, AND THE ROUND IS OUT OF LEVERS.**
>
> **Routing: `docs/experiments/ROUND-2026-08-12.md` is the charter for everything below and it EXPIRES on 08-17** — after that it is history, not routing. Read it before any experiment doc.
>
> - 🔴 **E31 (`E31-trace-reliance.md`): THE COPYCAT QUESTION IS ANSWERED, IN THE NEGATIVE.** **Δ_net − Δ_expert = −0.0168 [−0.0201, −0.0133]** on 181,517 within-turn pairs, **both estimator controls passing** — the clone leans on its own previous action **LESS** than the humans it copied. The shortcut was real and available (experts **+0.0632**, a pure trace-follower **+0.1084**, sitting in plain view in `turnActionCount`) and it declined it. ⛔ **Sixteenth closed axis. No fix shipped, no retrain, no arena game.**
> - ⚡ **The finding that outlived the null (E31 §R4): the clone is MORE predictable than its demonstrators — 0.6912 vs 0.6656 from the option list ALONE, before any history.** Its excess predictability is **modality, not perseveration** — the *opposite* failure from copycat, and the conformity thread reached from an instrument that has nothing to do with agreement. **This is the only live diagnosis left and it has no intervention attached.**
> - ⛔ **HANDOFF probe 2 (hysteresis) is now PREDICTED NEGATIVE — do not run it as written.** E28 reverted it to *unprioritised*; E31 fired it the other way. Adding stickiness to a net already less sticky than its demonstrators is aimed backwards.
> - 🔴 **E28 closed without a copycat verdict** (reading 1 VOID on an unreachable control), but bought **§N.4.1 outright** on a tight null: attacker switches +0.0006 [−0.0010, +0.0022], target switches −0.0001 [−0.0029, +0.0027] ⇒ **we do NOT switch commitments more than the experts**, which excludes H1 **and** perseveration at once.
> - 🔴 **E29 REFUTES the 71.4% mirror share** — 20.9% at n=67 in the 1000+ band, z = −3.81; **five documents quoting it as hard were overclaiming**, and the monotone curve is dead too (it FALLS at the top). The top is a **four-way field and we are fourth at 10.5%** (23.7% one week earlier, same frame).
> - 🔴 **E30 VOID ON BUILD SIZE** — the reweighted corpus caps at 26,250 decisions against a pre-registered floor of 149,391. ⛔ The parked coverage lever is dead a second time, for a new reason.
> - ⚠ **THE ROUND IS OUT OF LEVERS.** All four charter cells plus E31 are closed, four of them negative, **none spent an arena game**. Nothing is pre-registered-and-unrun.
> - ⚠ **08-15 (two agents, two decks) LOST ITS SECOND DECK.** E29 was the input and it came back no: **no archetype in the target set clears the sizing gate to clone.** Directive 2 reverts to a pure variance hedge priced exactly as §8al prices it. ⚠ Both active slots already hold `v5_s2`.
> - 🔴 **RULE 20, NEW DIRECTION — an agent-directory path is not an identity either.** `agents/sa/policy_net.npz` is a **stale pre-v5 net** (no pooling block, `state_in` 496). Hashing the npz inside `dist/submission.tar.gz` shows the live agent is **`policy_v5_s2`**; `build_submission.py` overrides the bundle via `--policy-net`. ⛔ **Every offline probe defaulting to `agents/sa/policy_net.npz` — `context_accuracy.py` included — reads a net the arena has never seen.**
> - ✅ **Report track advanced** (the qualification gate): `STRATEGY.md` **§7d** (prize maps — ~84% of mirror takes are single-prize, so the canonical 2-2-2 / 1-1-2-2 / 2-1-2-1 maps do not describe this matchup) and **Appendix A.1** (the literature round: three of seven proposals closed on arrival by opening our own files).
>
> ⚠ **Superseded but NOT retracted:** the day-31 header below (E26/E27) stands unchanged; everything above sits on top of it.
>
> # ▶ DAY 31, EARLIER (2026-08-12): **THE TRAP IS NOW CHARACTERISED, NOT JUST NAMED — we can MOVE cheaply and we cannot AIM.**
>
> **Read in this order: §N.14 (E27 closed) → §N.13 (E26, the positive result it spends) → ROADMAP §2.8 → `EVIDENCE` §8ch/§8ci/§8cj.**
>
> - ✅ **E26 (§8ch): a trained policy's deviations cost a QUARTER of matched-random ones — f = 0.758 [0.703, 0.814] against the evaluator family's 0.12.** The local optimum forbids **jumps, not paths**. First positive result about a *mechanism* since §8z.
> - 🔴 **E27 (§8cj): CLOSED by its own pre-registered rule at two consecutive MOVEMENT ONLY.** On-policy TD-advantage iteration, better critic each round (AUC 0.7963 → 0.8311), corpus anchor removed, all parameters free: ship **0.5010 → 0.4805**. Both `f_round` intervals contain 0.758 ⇒ **a critic-aimed policy deviates no better than one imitating a stranger.**
> - ⛔ **NOTHING is pre-registered-and-unrun.** Seven aiming signals are measured dead. ⛔ No round 3, no β sweep, no eighth variant.
> - ⛔ **NO SUBMISSION IS INDICATED** — no candidate this session beat `v5_s2`, and both active slots already hold it (max-of-two-draws, §8ak).
> - 📌 **The premise for suspending `STRATEGY.md` has expired**: that decision was "Track A only", and Track A now has no open levers. **The report is the qualification gate** (top 8 advance on the Strategy Category), and day 31 produced two chapters that exist only in `EVIDENCE`.
>
> ⚠ **Superseded but NOT retracted:** day 30's header (below) closed the evaluator family via E25 (§8cg). That verdict stands unchanged; E26/E27 sit on top of it. Two agents are working this repo; see the routing note below before editing a shared file.
>
> ## 🔴 N.14 DAY 31 (2026-08-12): **E27 IS CLOSED — two rounds of on-policy policy iteration move the policy 6.57 picks/game and buy nothing** (§8cj)
>
> Pre-registered `docs/experiments/E27-policy-iteration.md` at `4070eb1`, before
> any generation or training. Every subsequent rule frozen before the cell it
> governs.
>
> | | round 1 (π₀→π₁) | round 2 (π₁→π₂) |
> |---|---|---|
> | V, held out by game | AUC 0.7963 | **AUC 0.8311** |
> | step `c` (vs π_r) | 5.29 | 4.31 |
> | **gate** | 0.4920 [0.470, 0.514] | **0.4760** [0.454, 0.498] |
> | `null(c)` | 0.4753 | 0.4799 |
> | matched-random control | 0.4240 | 0.4150 |
> | **f_round** | 0.895 [0.605, 1.185] | **0.718 [0.450, 0.985]** |
> | **ship vs `v5_s2`** | **0.5010** | **0.4805** |
>
> - 🔴 **Both gates contain their null ⇒ MOVEMENT ONLY twice ⇒ the frozen rule
>   CLOSES the axis.** Pooled, the direction bought **+0.006 ± 0.008**.
> - ⚡ **The sharpest statement: both `f_round` intervals contain E26's 0.758**,
>   which is the value for a policy that is merely *different* (an expert clone
>   67 Elo worse). **The outcome signal did not aim anything.**
> - ⛔ **The obvious objections are all ruled out**: not the critic (0.8311, on
>   par with the E20-era net E22 ensembled — and the *better* critic gave the
>   *worse* round); not the corpus anchor (removed); not the frozen head (all
>   702,913 params); not aggressiveness (weight ratio 2.72, identical to B8's);
>   not entropy collapse (`val_top1` flat at 0.872, step fell only 5.29 → 4.31);
>   not sampling luck (two independent 8,000-game draws from π₁ agree to 3 dp).
> - ⚠ **NOT closed: the same family under R2 hardware.** This ran 16,000 games on
>   2 vCPU-equivalents. Say *"at this scale, on this board, measured nothing"*.
> - 🔬 **Three controls were found biased TOWARD the hypothesis and fixed before
>   their cells ran** — carry this forward, it is the session's most reusable
>   lesson: (1) a pooled rank histogram made the control 17% clipped at depth
>   1.59 vs the treatment's 1.92; (2) deviation *rate* scales 9.4% → 50% with
>   option count, so a flat-rate control deviates in the wrong places; (3)
>   `null(c)` was calibrated at depth ~1.9 and round 1 deviated at 1.33.
>   **"Rate-matched" is not matched.**
> - 🔬 **Two instrument defects fixed at the cause, not papered over:**
>   `p26_selfplay_gen` never passed `attr=` (it predates the day-20 v6 block), so
>   the writer stacked `None` into an object array — breaking its own re-open
>   **after** the shard was already on disk, and it would have made `--attr`
>   train on nothing. And a mis-aimed `--collect` **discarded a completed Kaggle
>   round at `rc=0`**; the runner now names every `.npz` the run wrote that
>   nobody collects, checked before the tree is deleted, and exits non-zero.
>
> ⚠ **ROUTING, day 30, third correction.** This header read *"the only pre-registered-but-unrun thread is E22"* and the line below it read *"▶ The live thread is E20"* — **both stale**, E20 having been refuted (§8cd) and E22 having reported and been audited (§8cf). This is failure mode 2 from ROADMAP's own doc-discipline audit for the third time in one day, which is itself the finding: **an additive entry point decays faster than the experiments it indexes.** Corrected as **routing only** — no verdict here is restated or revised.
>
> ⚠ **ROUTING, day 30, second correction.** This header previously read *"E20 … is PRE-REGISTERED and running"*; E20 reported and was refuted, so the line was stale in the same way §8ca left the one below it stale. Corrected as **routing only** — no verdict here is restated or revised.
>
> ⚠ **ROUTING CORRECTED DAY 30.** The header below this line read *"E17 RAN … a build is licensed"* and the read-order pointed at *"the clock is BUILT and its A/B is in flight"* — **both were superseded by E19 on day 29** (§8ca: 0.4963 [0.4719, 0.5207], n=1,608, the clock closed and E17's +0.0139/decision retracted with it). Failure mode 2 from ROADMAP's own doc-discipline audit: additive updates leave a stale claim load-bearing in the entry point. Corrected as *routing only* — no verdict here is restated or revised.
>
> **▶ The E20/E22 thread is CLOSED — read §N.11 first, then §N.9.** Its artefacts stand and are reusable regardless: V trainer `scripts/train_value.py`, inference `agents/sa/valuenet.py` (verified against the trainer at 2.7e-7), the one-ply search `agents/sa/vlook.py` (including `vrnd`, the rate-matched control), the Kaggle harness `scripts/kaggle/`, and `score.py --dir` for local sharded runs. Bars and void conditions are frozen by `docs/experiments/E20-value-lookahead.md` and `E22-pessimistic-lookahead.md`.
>
> ## ✅ N.13 DAY 31 (2026-08-12): **E26 — a TRAINED POLICY's deviations cost a quarter of what ARBITRARY ones cost, at matched rate, depth and location: f = 0.758 [0.703, 0.814] against the evaluator family's 0.12** (§8ch)
>
> | arm (n=2,000, mirror, v3 both sides, health clean) | score | 95% CI | changed/game | mean depth |
> |---|---|---|---|---|
> | **A — expert policy substituted** | **0.4053** | [0.384, 0.427] | 20.19 | 1.92 |
> | **B — rate/depth/location-matched random** | **0.1080** | [0.095, 0.122] | 19.06 | 1.85 |
>
> **z(A − B) = 22.9. f = (A−B)/(0.500−B) = 0.758, ratio to E25's f_eval 6.3×.**
>
> - ✅ **THE FIRST POSITIVE RESULT ABOUT A MECHANISM SINCE §8z.** The
>   deviation cost that closed six axes (§8cg) is **overwhelmingly a property of
>   deviating INCOHERENTLY**. ⇒ **the sharp local optimum forbids JUMPS, not
>   PATHS.** That is the licence E27 runs on.
>   ⚠ **Corrected same session:** this first read *"the first positive-signed,
>   well-controlled result since §8z"* and that is wrong — day 24's **seed swap**
>   (`v5`→`v5_s1`, **0.549 [0.527, 0.571]**, n=2,000) is positive and **still
>   stands**; it was *ensembling* that day 25 retracted. The seed swap was a
>   better **draw** from a nuisance distribution (§8bg/§8bh), not new capability,
>   and the harvest built on it measured null. §8ch carries the full wording.
> - ⚡ **E25's cost law replicated OUT OF SAMPLE**: predicted 0.1025 at cell B's
>   realised rate, measured 0.1080 — fitted on v5 in the `vlook` harness, tested
>   on v3 with a rank-histogram sampler. **It is a law of this game, not one
>   configuration.**
> - 🔴 **"Rate-matched" was two things short of matched, both biased toward the
>   hypothesis.** Pooled-histogram sampling ran the control at depth 1.59 vs 1.92
>   with 17% clipped; and deviation *rate* is a function of option count (9.4% at
>   2 options, ~50% at 10), so a flat-rate control deviates in the wrong places.
>   Fixed by sampling both from the treatment's own per-option-count histogram.
>   **Carry this into every future matched control.**
> - ⚠ **The limit, stated first not last: this does NOT separate sequential
>   coherence from per-decision plausibility.** The licensed claim is the
>   composite. And it is measured at **v3**.
> - ⛔ **Nothing ships** — cell A is 0.4053, still a rout, and §8u is untouched.
> - 🔬 Two sizing finds that cost no games: phase is **flat** on our own states
>   (0.265/0.257/0.268), which killed the original design; and `policy_b7_ntum`
>   is a **v3-era net**, which would have confounded generation with policy.
>
> ## 📌 STANDING USER DIRECTIVES — day 31 (2026-08-12). **Live, not yet executed.**
>
> **Priority order given by the user: (1) escape the local optimum and improve
> the net we have; (2) the two notes below, which are for LATER and are recorded
> here so they are not re-derived.**
>
> 1. ⛏ **MINE NEWER EPISODES — and the FIRST question it must answer is whether
>    the mirror is still 71.4% of our field above rating 1000 (user, day 31).**
>    🔴 **That figure is 10 of 14 games.** §8ac's band table reads `1000+ (n=14)`,
>    so the Wilson 95% CI is roughly **[40%, 83%]**, and it was measured
>    **2026-08-01** — before the top of the board reshuffled completely. It has
>    since been quoted as a hard number in ROADMAP §2.6, §8bi, §8ce and **E27's
>    pre-registration**, where it is the stated reason self-play is not fatally
>    misaligned with our real opponent pool. ⚠ **A premise repeated in five
>    places is not thereby verified — it is load-bearing (rule 15).**
>    ⚡ **And there is a live tension to resolve, not just a stale number:**
>    §8bq's mined top-of-ladder feed (08-03…08-07, `avg_score` ≥1100) puts our
>    archetype at **23.7% of that population**, while §8ac's curve would predict
>    *more* than 71.4% up there. The two measure different frames — opponents
>    **we** faced vs the composition of a **censored published feed** — so they
>    need not agree, but the gap is large enough that one of them is misleading
>    about where we now play. **Re-census from our own recent replays
>    (`p9_field_census.py --us`), which is the frame that governs anchor
>    weights.**
>    The last corpus build (`pds_v4`) drew on four
>    dumps, 1,603 games, `avg_score` 1057–1223, and nothing has been mined since
>    day 25. ⚠ **ROADMAP §0's rules bind and have each been paid for:** pull
>    **manifests broadly, episodes selectively**; keep manifests even when
>    episodes are pruned; **NEVER let mining pick an anchor** (that is what
>    retired `rule:v10`, which turned out to be 12.8% of our field); and Kaggle's
>    datasets stop at `avg_score` 1055 while we play at ~976, so mined data
>    describes a band above ours. ⚡ **What is genuinely new since the last mine:
>    the top of the board reshuffled completely** (HANDOFF standing note), so
>    fresh dumps are a different population, not more of the same.
> 2. 🃏 **08-15 FINAL SUBMISSIONS: two agents on two DIFFERENT decks**, if a
>    second deck can be justified — a variance hedge across matchup
>    distributions rather than a claim that the second deck is stronger.
>    ⚠ **Read against §8al/§8as before choosing it:** strength falls
>    **monotonically** with distance from the consensus 60 (1 swap 0.4911, 2
>    swaps 0.4757, 4 swaps 0.4637 against a same-deck control of 0.4980), and
>    §8ar/§8as's 11-variant search was ≤ 0 throughout. ⇒ **the honest framing is
>    "we are buying variance and paying Elo for it"**, and the price is roughly
>    known. The decision is the user's and is scheduled for 08-15.
>
> ## 🔴 N.12 DAY 30 (2026-08-11): **E25 closes the ENTIRE "override the clone with a better evaluator" family — V's edge over a coin flip FLIPS SIGN as you tighten the threshold** (§8cg)
>
> Pre-registered `docs/experiments/E25-deviation-cost-curve.md` at `8176ac1`
> before either cell, with a **point prediction** (A = 0.436, B = 0.427).
> **Missed by ~9σ.**
>
> | arm (all n=2,000, mirror, health clean) | changed/game | rate/firing | net-margin | score |
> |---|---|---|---|---|
> | `bc:base` (C0) | 0 | — | — | 0.5082 |
> | coin flip @ 55% | 18.21 | 0.554 | +2.55 | 0.1115 |
> | V-guided @ 55% | 16.24 | 0.551 | +2.34 | **0.1580** |
> | coin flip @ 13% | 4.56 | 0.129 | +2.40 | **0.3650** |
> | V-guided @ 13% (`vtau0.08`) | 4.02 | 0.130 | +2.77 | **0.3300** |
>
> **f = (V-guided − rate-matched coin flip) / (0.500 − coin flip):**
>
> | rate | V − coin flip | z | f | 95% CI |
> |---|---|---|---|---|
> | 55% | +0.0465 | +2.94 | **+0.120** | [+0.040, +0.199] |
> | 13% | −0.0350 | −2.21 | **−0.259** | [−0.489, −0.030] |
> | change | −0.0815 | **−3.64** | | |
>
> - 🔴 **Tightening on V's own confidence makes it WORSE than a coin flip over
>   the same arms at the same rate.** The pre-registered escape (f > 1 at a low
>   rate) is refuted **with the sign wrong**. ⚡ The control even deviates *more*
>   per game (4.56 vs 4.02) and still wins.
> - 🔴 **THE MECHANISM: V's confidence and its distance from the clone are the
>   same variable.** Filtering to high-gap overrides raises net-margin
>   +2.34 → +2.77. **V is confident precisely where it disagrees most with the
>   clone, and it disagrees most where it is extrapolating — so the ranking
>   signal IS the damage signal, and no threshold separates them.**
> - ⚠ **Stated as a limit:** because they are the same quantity, this cell
>   **cannot** decompose "confidence is anti-informative" from "distance costs".
>   The licensed claim is the composite — *no confidence threshold isolates good
>   deviations* — which is the operationally decisive one.
> - ⛔ **CLOSED AS A FAMILY, not as configurations:** §2's rollout search (0.323),
>   B4/E5's `evalfn` sequencer (−89 Elo), the clock (0.4963, §8ca), E20 (0.0065),
>   E22 (0.1580), E25 (0.3300 and worse-than-random). **One statement: the clone
>   sits at a sharp local optimum, and any evaluator's score for an off-policy
>   option is contaminated by extrapolation in proportion to how far off-policy
>   it is. Better evaluators do not fix this, because the contamination grows
>   with exactly the signal you would rank by.**
> - ⛔ **E23 (value iteration) is now MORE clearly unlicensed**, not less: its
>   premise was that better data at search-selected successors fixes selection,
>   and those successors are chosen by the same contaminated ranking.
> - ⛔ **No third tau, no fourth arm.**
> - ⚡ **The point prediction is what did the work.** A bar ("≥ 0.530") would have
>   filed this as one more null; a *missed prediction* is what forced the control
>   that found the sign flip.
> - ⚠ **Defect in my own frozen rule:** branch ⚠ keyed on `A < 0.41` — cell A
>   alone, against a linear cost-vs-rate model. Only cell B can separate
>   nonlinearity-in-rate from bad selection. It fired correctly for a reason its
>   own condition did not establish. **Key a reading rule on the comparison, not
>   on one arm.**
>
> ## ⚡ N.11 DAY 30 (2026-08-11): **E22 fails at 0.1580 — and its audit arm produces the number the whole search thread was missing: deviating from the clone costs −0.389, and the best evaluator we can build recovers 12% of it** (§8cf)
>
> **E22 was the last pre-registered-but-unrun thread.** It is now run, audited,
> and closed. `docs/experiments/E22-pessimistic-lookahead.md` (frozen `527e26a`;
> its audit rule frozen separately at `b4dbe14`, after the cell reported and
> before the audit arm ran).
>
> | arm | score | 95% CI |
> |---|---|---|
> | `bc:base` — identical arms (C0, Kaggle) | 0.5082 | [0.4897, 0.5267] |
> | E20 — argmax over one V, every option | 0.0065 | [−0.0154, +0.0284] |
> | **coin flip over top-3 @ 55.5% (audit control)** | **0.1115** | [0.0896, 0.1334] |
> | **E22 — LCB K=1.0 + top-3 coverage, 5 nets** | **0.1580** | [0.1361, 0.1799] |
>
> All n = 2,000, mirror, byte-identical policy net both sides, health clean.
>
> - ⚠ **E22 lands on the HARMFUL branch, NOT the KILL branch** (KILL needs the CI
>   to *contain* 0.500; this misses by 16σ). The doc routes ⚠ to *audit before
>   interpreting* — so the KILL text is not invoked, and the audit is what
>   produced the finding.
> - ⚡ **Both pre-registered changes worked on the internal gate.** Agreement with
>   the clone went from **0.92× its chance rate to 1.35×** — crossing from below
>   chance to above it, the pre-registered signature that §8cd's mechanism was
>   correctly identified. ⛔ And it bought a rout anyway. **E19's standing
>   constraint holds: no internal gate licenses anything.**
> - 🔴 **The audit arm is the entry: a coin flip over the SAME covered arms at a
>   MATCHED deviation rate** (55.4% vs 55.1% of firings; 23.6% vs 23.2% of
>   visited decisions). Rate-matching is the design — an unmatched random arm
>   deviates on 2/3 of firings and re-measures the confound.
>   ⇒ **V beats the coin flip by +0.0465 [+0.0155, +0.0775], z = 2.94.**
> - 🔴 **THE TRANSFERABLE NUMBER: deviating at 23% of decisions costs
>   0.500 → 0.1115 = −0.389. V recovers +0.0465 of it = 12.0%.**
>   **The binding constraint is deviation from the clone, not evaluator
>   quality.** A V verified against its trainer at 2.7e-7, AUC 0.827 held out by
>   game, 5-net ensembled, pessimism-penalised and coverage-restricted, recovers
>   one eighth of the damage that deviating does at all.
> - ⚠ **Two faults in my own rule, recorded.** (1) The pre-registered criterion
>   was "CIs disjoint" and they are disjoint **by 0.0027** — a criterion that
>   could flip on 3 games of noise should have named the two-sample test (z=2.94
>   agrees, which is why the reading stands). (2) The control's override carries
>   net-margin +2.55 vs E22's +2.34, so **~20% of V's edge may be margin rather
>   than selection** on a crude two-point extrapolation.
> - ⛔ **CLOSED: H-eval in every cheap form.** ⛔ **E23 (value iteration) is NOT
>   licensed** — its dependency clause was written for the 🟡 *alive but short*
>   branch, and harmful is not partial. ⛔ **No `tau` knob** — that is E17's
>   post-hoc arm selection, which E19 priced.
> - ⚡ **THE ONE QUESTION LEFT, and it is sharper than what closed.** If V's
>   benefit is a fixed fraction f ≈ 0.12 of the deviation cost **at every rate**,
>   the net effect is −(1−f)×cost, **negative at every rate** — and the entire
>   "override the clone with a better evaluator" family dies at once instead of
>   one configuration at a time. The escape is **f > 1 near a low deviation
>   rate**. That is **one number**: one more rate point plus its matched control,
>   ~45 min. **Pre-register it before running it.**
> - 🔬 **It also reframes E16–E19 / §8bw–§8ca.** Fifteen days searched for a
>   better *decision rule*; this cell says the clone sits at a **sharp** local
>   optimum where off-policy movement is catastrophic **independent of how it is
>   chosen**.
>
> ⚠ **Seat trap, one layer out:** an interim read off raw `winner=` counts showed
> 428/384 — a near coin flip — because those are **seat**-indexed and the arena
> alternates seats. Arm A's wins are `winner == seat` (pooled 0.167 at that
> point). Verified against the smoke's official W3/L7 before use. Rule 18.
>
> ✅ `scripts/kaggle/score.py --dir` pools LOCAL sharded runs through the same
> scorer as the Kaggle cells. ⛔ `out/kaggle_pack/` is now git-ignored — a 45 MB
> zip of the licensed engine entered a commit via `git add -A` and was amended
> out.
>
> ## 🔴 N.10 DAY 30 (2026-08-11): **E24 closed E21's VOID arm — the board fact reaches 1,045 decisions and buys +0.0041** (§8ce)
>
> Pre-registered at `2d36ce8` before any cell. §8cc's `fscrap` read 0.5175 with
> `fetch=0/3082` — a wiring statement, because our 60 runs zero Pokémon Tools
> and the condition is unsatisfiable in the mirror. Re-run where a Tool exists,
> after **on-policy sizing** across five anchors (0.300/game vs `lucario_v10`
> down to 0.115 vs `crustle_v1`, **0.000** in the mirror):
>
> | cell | anchor | delta | 95% CI | n/arm |
> |---|---|---|---|---|
> | **a (primary)** | `rule:v10,noS`@`lucario_v10` | **+0.0041** | [−0.0168, +0.0250] | 4,000 |
> | b (exploratory) | `rule:archaludon` | −0.0317 | [−0.0596, −0.0039] | 2,000 |
> | b2 (replication) | same, fresh games | −0.0055 | [−0.0334, +0.0224] | 2,000 |
> | b+b2 pooled | — | −0.0186 | [−0.0383, **+0.0011**] | 4,000 |
>
> - ✅ **Controls passed everywhere** — 0.261 changed picks/game in cell a
>   (1,045 of 1,123 firings), and the control arms printed **no `fetch=` field at
>   all**. This is the reading E21b could not produce.
> - ⚡ **New counter, and it is the transferable bit: `fetch_diff`.** A firing the
>   net agrees with is **not a treatment**. `fired` overstates it by ~7% here and
>   could overstate it by any amount elsewhere. `bcagent.STATS` now prints both.
> - 🔬 **Cell b tripped the harmful branch by 0.0017 and dissolved on
>   replication.** The reading rule was frozen and committed (`7d576da`) BEFORE
>   b2 ran, on two grounds available in advance: b implied **−0.19 per changed
>   fetch** against cell a's **+0.016**, and it was the second of two cells.
>   ⚠ **Pooled b+b2 clears zero by 0.0011** — say *"not resolved as harm at
>   n=4,000"*, never *"archaludon is clean"*. A third cell is **not indicated**.
> - ⚡ **Sizing does not predict on-policy firing in a fixed direction.** Realized
>   came in **under** here (0.261 vs 0.300) and **1.6× over** for `fstad` (§8cc).
> - 🔴 **The finding that outlives the cells: E21 filed `fscrap` as
>   "dominated-class" and it is not** — it promotes Scrapper over Unfair Stamp
>   and Night Stretcher, all live cards. The genuinely dominated version (delete
>   Scrapper when no Tool is anywhere) sizes at **0.066–0.08/game** and is
>   unmeasurable. ⇒ **On this seam the winning class is too rare to test and the
>   losing class is the only one big enough.** Tradeoff rules: **0-for-7**.
> - ⛔ Nothing ships; both flags stay OFF. The rule **cannot fire in the mirror**
>   (71.4% of our field above 1000), so even a win was matchup tech.
> - ⚠ **Numbering: renumbered E23 → E24 before write-up**, because E22's frozen
>   doc reserves **E23 for value iteration**. Scripts/logs/archives keep their
>   `p90_e23_*` names — they are the receipts the runs wrote.
>
> ## 🔴 N.9 DAY 30 — **E20 IS REFUTED: 0.0065 at n=2,000. A real instrument defect was found on the way and was NOT the cause** (§8cd)
>
> **The value net's live path computed a different function than was trained.**
> `train_value.py` pads an empty card bag with row 0; `sa/valuenet.py`
> substituted **zeros**. `p88_value_equivalence.py`: max \|diff\| **0.126** on the
> **7.0%** of rows with an empty bag, against a within-position sibling range of
> **0.186** — comparable to the whole signal an argmax uses, and **structured**,
> because hands empty exactly when played out. Fixed; the shipped path now
> matches the trainer at **2.7e-7**.
>
> 🔴 **THE VERDICT: broken path 0.0040 → corrected path `0.0065 [−0.0154,
> +0.0284]`, n=2,000. Repairing a 0.126 mismatch on 7% of evaluations moved
> NOTHING.** 13 wins in 2,000 games ⇒ **H-eval in E20's pre-registered form —
> argmax over a learned V across every option — is REFUTED**, on the
> pre-registered HARMFUL branch.
> ⚡ **The reusable lesson: a real defect is not automatically the explanation.**
> Only re-running separated them; otherwise "we found the bug" would have stood
> in for a verdict.
> - ⚡ **The anti-selection diagnosis is REINSTATED, weaker:** argmax agrees with
>   the clone **11.1% against a 20.5% chance rate** on the corrected path (was
>   6.1% on the broken one). Real, and previously overstated by ~half.
> - ⛔ **Still retracted: the "live AUC" figure.** It is clustered by game
>   (~15 effective units) and swung 0.7042 → 0.5222. Quote only the
>   per-decision agreement statistic from `p87`.
> - ✅ **What stands:** the Kaggle harness, **commissioned on two controls**
>   (C0 identical arms **0.5082** contains 0.500; C1 `s2` v `s1` **0.4996**
>   contains §8bh's 0.510, n=2,800 each); **V itself — AUC 0.827 held out BY
>   GAME**, now verified equivalent through the path that plays; and `p86`'s
>   structural fact that after one `fs.step` players are indexed **absolutely**
>   (98%) and the mover changes seat only **1%** of the time.
> - ⚡ **A by-product:** C1 is the third reading of `s2` v `s1` (0.537 → 0.510 →
>   0.4996). Pooled with §8bh: **0.5031 [0.488, 0.518] ⇒ ≈ +2 Elo, CI containing
>   zero.** The shipped net's seed edge is gone; any seed-harvest plan is weaker.
> - 🔬 **Two method failures, both already in §2's catalogue:** a 2,000-game A/B
>   ran on a component never reconciled with its trainer (**rule 18**, and the
>   check cost ten minutes); and a **clustered** AUC (~1,000 decisions from 15
>   games) was read as evidence — it re-ran at 0.5222 against 0.7042. **§8bw's
>   own lesson, repeated.**
> - ⏭ **E22** (`docs/experiments/E22-pessimistic-lookahead.md`, **renumbered from
>   E21** because §8cc owns that number) is pre-registered and **NOT run**. Its
>   5-seed ensemble is **trained** (`out/value_e0..4.npz`, AUC 0.827–0.829,
>   shared `--split-seed 0` so spread is epistemic). Baseline is **0.0065**.
>   ⚠ **Read the distance honestly before spending on it: E22 must clear 0.530
>   from 0.0065.** ⛔ **If it fails, the axis closes** — the evaluator is not the
>   missing piece the three dead searches lacked, and that becomes the chapter.
> - ⚡ **Harness usage:** `scripts/kaggle/pack.py --push` (one zip, repo-relative
>   paths — `--dir-mode zip` 400s), `launch.py push/status/pull`, `score.py --job
>   <j> --expect <v>`. ⛔ **Never hardcode `/kaggle/input/<slug>`** — Kaggle
>   mounted this dataset under `/kaggle/input/datasets/`; the runner searches.
>
> ⛔ **CONCURRENCY, day 30.** `HANDOFF.md`, `report/EVIDENCE.md` and `ROADMAP.md` are edited by both agents. **Commit before a long run, not after** — 140 lines of §8cb/§N.7 sat uncommitted through four commits from the other agent and only luck kept them. **And check the next free `EVIDENCE` section letter before writing one** (`grep -oE "^## 8[a-z]+\." report/EVIDENCE.md | sort -u | tail -3`); §8cb is taken.
>
> **📌 USER DECISION: report/`STRATEGY.md` stays SUSPENDED until the sim closes 08-17** (report due 09-14 has runway). Track A only.
>
> **✅ THE VALIDATION COST IS MEASURED, AND THE A/B IS AFFORDABLE.** `arena.py`
> has **no built-in parallelism** (no `Pool`, no `--jobs`) — but it does not need
> a code change: **process-sharding works**, verified 6 shards × 40 rows with
> separate `--archive` files and a random `run_id` per process (`battle_start`
> takes no seed, so shards do not replay each other). ⚠ **The speedup is ~3.6×,
> not 6×** — E17's own contention measured it (76 ms/rollout solo → 127 ms at
> 6-way), and startup is ~6 s per process, which only amortises on long runs.
> ⇒ **n=2,000 oracle mirror A/B ≈ 85 core-hours ≈ 24 h wall.** One long run,
> inside the 08-15 last-safe-day, with no room for a second attempt.
> ⛔ **Shard to separate archives and merge** — never let 6 processes append to
> `out/arena/games.jsonl`, and never let a throughput test touch it at all.
> ⚡ **The obvious optimisation, unbuilt: early stopping.** τ does NOT save
> compute (the rollouts are what produce the margin), but a racing / successive-
> elimination schedule would — 57% of decisions have essentially zero gap and
> could be abandoned after a handful of rollouts.
>
> **▶ Read in this order:** **§N.6c + §8ca (E19 closed the clock — start here, it retracts §8by's headline)** → `docs/experiments/E20-value-lookahead.md` (the live thread) → §8by (E17, for what E19 retracted and what survives) → ROADMAP §2.7 → §N.0d (how the lead arose).
>
> ## 🔴 N.8 DAY 30 (2026-08-11): E21 RAN — the clone's board-blind fetch BEATS a board-aware rule, 0.4405, z=−5.36 (§8cc)
>
> Pre-registered at `5be502d` before either cell. `bc:e21,fstad` (fetch Spikemuth
> Gym when no Stadium is ours) in the mirror, byte-identical nets, n=2,000:
> **0.4405 [0.4187, 0.4623]** — the pre-registered **HARMFUL** branch, not a null.
> ✅ Control 1 passed hard: **1,439/3,023 fetches redirected, 0.72/game**, so the
> intervention happened and lost.
> - ⚡ **Realized firing is 1.6× the offline sizing** (0.72 vs 0.461/game): rule-14
>   estimates from recorded games under-predict, because the rule changes the
>   trajectory it is measured on.
> - 🔴 **Arm 2 (`fscrap`) is VOID: `fetch=0/3082`.** Our 60 runs **no Pokémon
>   Tool**, so in the mirror the condition is unsatisfiable *by construction* —
>   §8aj said this about the same card and I did not apply it. ⇒ **SIZE THE
>   CONDITION IN THE MATCHUP THE CELL WILL RUN IN.** Its accidental payoff: with
>   zero firings the arms are identical, giving a clean **C0 at 0.5175 [0.496,
>   0.539]**, CI containing 0.500.
> - 🔬 **Audit (diagnostic only): the harm does NOT generalise.** Against
>   `rule:v10` on `lucario_v10`, `fstad` − `base` = **+0.014 [−0.028, +0.056]**,
>   and the mirror's −0.060 sits outside it. Spikemuth Gym is **symmetric in the
>   mirror** (both sides run Marnie's), so we pay a scarce tutor to hand both
>   players the same engine — and it is a **4-of** displacing a 1-of ACE SPEC.
> - ✅ **Tradeoff rules 0-for-6**, on the class's largest-ever firing rate.
> - ⛔ Nothing ships; both flags default OFF. E20 owns the submission slots.
>
> ## ⚡ N.7 DAY 30 (2026-08-11): the two named Petrel scenarios — §8cb
>
> Both close by sizing, but **§8br's verdict is sharper and worse-sounding now.**
> - ⛔ **Tool Scrapper into their Active's tool is a NON-EVENT.** Ours 1/9
>   (11.1%), **experts 5/149 (3.4%) against a 2.2% no-tool baseline** — nobody
>   conditions on it, and the situation arises **0.12/game**. Both sides fetch
>   Unfair Stamp / Night Stretcher / Boss's Orders instead.
> - 🔴 **Unfair Stamp is not a decision our agent makes: 56 of 56 legal turns,
>   including 18/18 opening with a hand of ≥7.** Experts 508/530 (95.8%), and
>   their 22 declines are legible — bigger hand (6.82 vs 5.72), more legal plays
>   (4.00 vs 2.67), **smaller opponent hand** (5.82 vs 8.45).
> - ⛔ **Sizing: 0.031 declines/game** (4.2% of 0.74 legal turns/game), 16× under
>   the gate. **This is what carries the verdict, and it is prior to whether the
>   policy is right.**
> - ⛔ **RETRACTED same session — "the unconditional policy is right 55/56 on
>   card differential" is NOT supported.** `D = 4 − H + O` and **85.7% of its
>   variance is the OPPONENT's hand size**; at the median O=7 no observed play
>   reads negative, so a 100%-unconditional policy and a perfect one score
>   alike. It is also blind to card *quality*, which is what "a strong hand"
>   means. ✅ **All that survives: 0/56 strictly DOMINATED plays** (experts
>   3/508) — one failure mode excluded, not a correctness claim. **Whether the
>   Stamp policy is right is OPEN**; deciding it needs outcome linkage that 18
>   big-hand plays cannot resolve, and §8ca closed the oracle route.
> - ⚡ **The reusable finding: a marginal take-rate table cannot tell a bad
>   policy from NO policy.** §8br's "+17.8% Unfair Stamp" was the shadow of a
>   decision the net never makes.
> - 🔴 **RULE 21, THIRD VICTIM.** Per-select this reads 56/172 = 32.6% taken with
>   mean hand 4.73 played vs 5.48 declined — selective-looking, in the expected
>   direction, and pure within-turn ordering (3.1 offers per turn). **Pick the
>   unit before reading the number.**
>
> ## 🔴 N.6c E19 CELL A CLOSES THE CLOCK — and retracts E17's headline (§8ca)
>
> `bc:cap,orc,od1` — **at most ONE overrule per game**, so the one-step
> assumption every rollout rests on is **exactly** satisfied.
>
> | | predicted | observed |
> |---|---|---|
> | **H-compound** (value real, multi-deviation destroys it) | 0.535 | ❌ **3.1σ away** |
> | **H-fusion** (the rollout value never transferred) | ≈0.500 | ✅ **0.4963 [0.4719, 0.5207]**, n=1,608 |
>
> 🔴 **A +0.035 gain in win probability at one decision IS +0.035 on that
> game's win rate if the estimate is unbiased. It is not there ⇒ the estimate
> is biased.** Upper bound +0.021 against a predicted +0.035, and consistent
> with zero. Controls all held: cap 0.93 overrules/game, 0.0% rollout errors
> over 137,158, worst pool 502 s.
>
> 🔴 **This retracts more than the agent.** §8by's +0.0139/decision, §8bz's
> +0.0353/overrule, the 67% best-arm rate and §8bw's +0.120 scale bar are ALL
> rollout values under clone-vs-clone continuation. E19 is the first test of
> whether that currency buys games and **it does not**. ⇒ **ROADMAP §2.7's
> sizing framework rested on an unvalidated assumption.**
> ⚡ **The lesson to carry:** every internal control passed — C0 99.8%, C1 100%,
> identical arms at zero, selection verified at z=5.5 against stored truth —
> and the instrument still measured the wrong quantity. **Internal validity is
> not external validity, and only the end-to-end test can tell you.**
>
> **Leading mechanism (named, NOT isolated):** determinization. We sample the
> opponent's whole deck and each simulated world is played as if the hidden
> cards were known — textbook strategy fusion. Sign: 37% of overrules take an
> option the net scores >3 worse. ⛔ The rival ("the opponent model is just
> wrong") is untested; separating them needs an information-set-aware rollout
> that does not exist here.
>
> ⛔ **CELL B IS CANCELLED, by E19's own pre-registered dependency** — no firing
> policy can rescue value that was never there, and it would have cost ~8 h.
> ⚡ Its **offline** result stands and is a report finding: `wp<0.50` ("we are
> losing") holds +0.0150/decision at 22% of firings, while "the net is confused"
> is **refuted** (adding `margin<1.5` drops it to +0.0108) — the wins come from
> options the net scored **>3 worse**, where it was confident and *wrong*.
>
> ⚠ **Do NOT report capping as an improvement.** E18 0.4828 → cap 0.4962 is
> **+0.0134 [−0.0414, +0.0682], z=0.48**, an interval four times the effect.
> The informative contrast is cap vs its own **point prediction**.
>
> ⇒ **The clock is CLOSED for Round 1.** What survives is narrower: the failure
> is in the **evaluator**, not the idea. An information-set-aware search, or a
> *learned* value function trained on real outcomes rather than determinized
> simulations, is untouched — the policy-iteration family §2.7 already names.
>
> ## 🟡 N.6b E18 RAN: 0.4764 [0.4281, 0.5252], n=403 — INCONCLUSIVE, and NOTHING SHIPS
>
> `EVIDENCE` §8bz, log `out/logs/p83_e18.txt`. **The clock plays the game and
> wins nothing**, and the diagnostics say precisely why it is not a wiring bug:
>
> | | |
> |---|---|
> | score (mirror, byte-identical net both sides) | **0.4764 [0.4281, 0.5252]** |
> | live oracle picks the genuinely best arm | **40/60 = 67%** vs a 1/3 null ⇒ **z=5.5** |
> | overrules | **3.32/game** of 7.98 fires (32% of decisions) |
> | true gain **when it overrules** | **+0.0353**, and **68% were real improvements** |
> | the pre-search net's opinion of what search took | **3.01 below its own top-1**; 37% were >3 worse |
> | value left on the table when it KEEPS the net's pick | **+0.0090 [+0.0036, +0.0145]** ⇒ it under-fires |
> | rollout errors | **0.0%** over 187,403 rollouts ✅ the 6.9% was the memory leak |
> | worst 600 s pool | 251.6 s ✅ never near the reserve |
>
> ⛔ **Not a kill** (the KILL branch needed the CI to exclude 0.500 downward)
> and ⛔ **not a ship** (point estimate below 0.500). Per the pre-registration,
> a null at n=400 is **uninformative**: SE≈0.025, power 0.34 against a true +0.03.
>
> 🔴 **The mechanism, named:** every measurement — E17's and the autopsy's —
> is **Q^π(s,a), the value of a ONE-STEP deviation** with the clone continuing.
> The deployed agent deviates **3.32 times a game**, so it is a different policy
> and the estimate is strictly valid only for the first switch. ⚠ Second worry:
> 37% of overrules take an option the net scores >3 worse, and those look good
> *under a clone continuation* — which is exactly what simulates them.
>
> ⚠ **The falsifiable next cell, unrun and NOT yet pre-registered:** if
> over-deviation is the mechanism, an oracle that overrules **less** should score
> **better**. E17's τ sweep holds the per-decision value at **7% of decisions
> instead of 38%** (τ=0.15). ⛔ τ is post-hoc — pre-register before running.
>
> ## 🔨 N.6 THE CLOCK IS BUILT — `agents/sa/oracle.py`
>
> **📌 USER DECISIONS (day 29):** *(a)* build the agent and run **n=400**, and
> **submit if it reads well** — recorded as a deviation in `E18-clock-arena.md`
> §4, because a screen that ships is what §8bh forbids (`s7` screened 0.528 and
> read **0.487** on 2,800 fresh games). *(b)* **NO draw-rolling** — the active
> pair stays frozen at `55326513` (973.6) and `55382430` (878.9).
>
> - **Agent:** `bc:<label>,orc`, OFF by default. Two-stage: free trigger
>   (option count ≤ 5) → 10-rollout probe that declines already-won positions →
>   20 pairs × top-3 arms → argmax. Flags `op/os/oa/ow/om/ot/oc` in `arena.py`.
> - ⚡ **Two things §2.7 called engineering already existed:** `planner.py` and
>   `sequencer.py` fork mid-game, and **`timemgr.py` IS the budget manager**.
> - **Cost:** ~65–100 s/game of the 600 s pool; worst observed game left 500 s.
> - **E18** pre-registered at `cc070b0` **before the first game**, with its own
>   power written down: n=400 has SE≈0.025 and detects a true +0.03 with
>   probability **0.34**, so ⛔ **a null at n=400 is NOT a kill.**
> - **Score it with** `python -X utf8 scripts/p83_e18_score.py` (pools
>   `out/arena/e18/shard_*.jsonl`, checks the 45 s reserve, refuses a verdict if
>   the oracle never fired).
>
> ### 🔴 Three defects the instrumentation caught, all of which would have produced a confident wrong number
> 1. **`arena.py` defaults to the `sample` deck**, where **79% of decisions
>    carry ≥12 options against 19.7% on grimmsnarl** — so the free trigger fired
>    on **0.7%** of decisions instead of 24%. A `sample` A/B measures a component
>    that barely fires and returns a null. **Rule 20 / §8ax one seat over.**
>    ⛔ **Always pass `--deck-a grimmsnarl --deck-b grimmsnarl`.**
> 2. 🔴 **`fs.release(root)` LEAKS.** A rollout creates a fresh search id at each
>    of ~100 steps; `release` reclaims one node, `fs.end()` reclaims the arena.
>    Measured **1.68 GB in 8 min** (climbing) vs **0.063 GB flat** after the fix
>    — 6 shards went from ~10 GB (unrunnable on 7.9 GB) to **0.39 GB**. ⚡ It
>    also probably explains an intermittent **6.9% rollout error rate**:
>    `fs.begin` throwing under memory exhaustion. **160 games collected before
>    the fix were DISCARDED**, not pooled — a starved oracle fires less.
> 3. **A `python -c` process-killer matches its own command line AND its parent
>    shell's**, so a kill loop SIGTERM'd the bash that was about to write the
>    E18 runner. Use **`scripts/killarena.py`** (in a file, excludes its own
>    parents). ⚠ Also: **launch long runs with NO tool-level timeout** — a 600 s
>    cap killed the first attempt at 160 of 408 games, and killing the wrapper
>    does **not** kill the python grandchildren, so a half-dead run keeps
>    writing rows.
>
> ## What E17 established (§8by, 300 treatment + 300 control positions, ~90,000 rollouts)
> - ⭐ **A budgeted rollout oracle over the net's OWN options is worth +0.0139 [+0.0027, +0.0250] per decision**, control-corrected. Against the §8bw scale bar of 0.120 that is 12%.
> - ✅ **The net's own ranking is right on average** (top-2 − top-1 = −0.0078). **There is no free re-ranking** — the value is entirely in per-decision *dispersion*, which at **0.1045** is LARGER than §8bx's our-vs-expert 0.0768/0.0866.
> - 🔴 **The 600 s is not the resource.** The budget curve saturates by 20 pairs/arm: an 8× increase from R_sel=5 to 40 buys **+0.004**.
> - 🔴 **§2.7's play-time arithmetic used the wrong denominator** (~318 selects; the real figure is **47.1 qualifying decisions/game**). The triggered design costs **153 s of 600 s** ⇒ **no batching is needed to PLAY**; the engineering is only for validation.
> - 🔴 **57% of our decisions carry nothing** (win prob > 0.85 ⇒ +0.0015). **The value is where we are LOSING** (+0.074 below 0.15). Spend the clock there or waste it.
> - ✅ **Trigger that licenses the build: option count ≤ 5** — +0.0373 [+0.0109, +0.0638] on 36% of decisions. ⚠ Gate met on the **point** estimate only; and it is **post-hoc among four**.
> - ⚠ **τ (a minimum margin before overruling) is exploratory** — τ=0.15 keeps the value at **1/5 the overrule rate**, but six values were swept and none pre-registered. **Pre-register it before any build leans on it.**
>
> ## ✅ Both owed items are CLOSED
> - **Rule-2 read on `55382430`: settled at 878.9** (two readings, 04:02 and 04:32 UTC 08-10, identical). Its **byte-identical twin `55326513` reads 973.6** ⇒ **a 94.7-point gap between two agents with the same bytes**, against §8ak's 63.2. **The ladder's noise floor is wider than the project has been quoting.**
> - **Draw-rolling is therefore PRICED and is with the user:** the whole 94.7 is nuisance, the LB ranks on the displayed draw, ~12 draws remain before 08-15, and it costs no local CPU. ⚠ Every submission evicts by recency — name the victim first.
> ⛔ **Do NOT re-open:** latent plans (§8bv), expert move-quality (§8bx), coherence/commitment via imitation (arm C), demonstrator selection (§8t/§8u), and the rest of §N.2's kill list.
>
> ---
>
> # ▶ §N — THE IMITATION-WITHOUT-A-PLAN THREAD
>
> **Read §8u, §8r, §8bv and §8bs before proposing anything.** The thread is
> live, but one operationalisation of it is already dead and re-running that is
> the main way to waste this session.
>
> > ## ✅ N.0 UPDATE (day 27, 3rd session): PROBE 0's FEASIBILITY GATE IS PASSED, AND IT CORRECTED TWO OF ITS OWN PREMISES
> >
> > `scripts/p80_rollout_feasibility.py`, `EVIDENCE` §8bw, log
> > `out/logs/p80_rollout_feasibility.txt`. **No experiment ran.**
> >
> > - ✅ **The blocking risk is dead.** An sbi captured in **another process**
> >   reconstructs the position **exactly** — 60/60, option list bitwise
> >   identical, both boards, turn and acting seat. Expert seats reconstruct
> >   32/32.
> > - ✅ **The instrument resolves.** The clone's own **top vs last** option
> >   reads **+0.120 [+0.052, +0.189]**. That is also the **scale bar** — read
> >   every future Δ against it.
> > - 🔴 **CRN does not exist here.** The engine draws its own shuffles/coins
> >   beyond the determinized world, so §N.4.0's "common random numbers" is
> >   unachievable. A **shared world** is the only pairing — worth ρ≈0.53, so
> >   keep it, just do not call it CRN.
> > - 🔴 **"Per-decision resolution is unaffordable" was WRONG.** 101 ms per
> >   rollout to terminal; ±0.020 on a pooled Δ ≈ 96 min on one core.
> > - ⛔→✅ **The fork accepts a decklist the seat is NOT playing** and returns
> >   a plausible number (Crustle's 60 on a Grimmsnarl seat read exactly what
> >   the right deck read). The first fix was "mirror only" — 🔴 **and that fix
> >   was wrong: only 18 of 50 `mirror_experts` seats run our exact 60**, so it
> >   would have mis-determinized 64% of expert seats silently. ✅ **Real fix:
> >   read each seat's registered 60 out of the replay** (`seat_decklist()`,
> >   50/50 recovered, 20/20 against `decks/grimmsnarl.py` on our own seat).
> >   **Scope constraint lifted — any seat in a replay we hold is usable.**
> > - 🔴 **And a defect in my own estimator, caught by replication:** three runs
> >   of the same cell read +0.130 / +0.107 / +0.120 against a nominal ±0.017.
> >   Pairs are **clustered inside positions**; clustering widens the interval
> >   **4.1×** and all three runs then agree. Fixed in the tool. **Size on
> >   positions, not pairs.**
> > - ⚠ **Most positions cannot show anything:** 11 of 40 sit in win-probability
> >   [0.15, 0.85]. Stratify on competitiveness, and estimate that WP on an
> >   **independent** rollout batch or the selection biases the effect.
> >
> > ▶ **Pre-registered as E16** (`docs/experiments/E16-counterfactual-move-value.md`),
> > **not yet run and not yet user-approved.** It scores the expert's actual
> > move against our net's move on mirror positions, with a free agreement
> > control (identical arms must read 0) and a **difference-in-differences** arm
> > using `out/policy_b7_ntum.npz` as an expert-like continuation — which is the
> > only design on the table that separates **H1 from H2** in §N.3.
>
> ## 🔴 N.0b DAY 28 — E16 RAN AND IS A NULL: the experts' MOVES are not better than ours
>
> `EVIDENCE` §8bx, log `out/logs/p81_e16.txt`, pre-registered at `ed22624`.
> 600 positions where a 1050+ pilot played what our net would not:
>
> | cell | reading |
> |---|---|
> | **Δ(expert − ours)** | **+0.0066 [−0.0018, +0.0150]** ⇒ NULL |
> | agreement control (identical arms) | −0.0009 [−0.0144, +0.0126] ✅ |
> | **1100+ band alone** | **−0.0000 ±0.0217**, k=93 |
> | scale bar (clone's own top vs last) | +0.120 |
>
> ⚡ **Third independent route to the same place, first one by OUTCOME rather
> than behaviour** (§8bj dissolved the clusters, §8bl's A/B read 0.487).
> **The strongest band shows the least**, which is the wrong ordering for "their
> moves are better". ⇒ **the gap is not in per-move choice quality.**
> ⚠ **But E16 measures the MEAN.** A null mean is compatible with a large
> per-decision gap whose SIGN varies, with the experts landing on the right side
> no more often than we do. **The dispersion is the live quantity** — and it is
> simultaneously the sizing gate for ROADMAP §2.7 (the clock).
> ## 🔴 N.0c ARM C RAN — H1 UNSUPPORTED, AND THE IMITATION THREAD IS CLOSED
>
> Same 300 positions, continuation swapped to `policy_b7_ntum`:
> clone +0.0107, expert-like +0.0056, **DiD −0.0051 [−0.0228, +0.0126]**.
> The expert's move is worth no more when followed up their way.
> ⚠ **Treatment is weak** — b7 differs from the clone by only ~7 agreement
> points — so H1 is **unsupported, not refuted** (§8ao's label for B8's β).
> 🔴 **The thread closes on ACTIONABILITY:** exploiting H1 requires a coherent
> expert imitator, which *is* B7, which measured **−55 / −92 Elo**. All three
> operationalisations are now closed — latent plan (§8bv), per-move quality
> (§8bx), coherence (arm C).
>
> ## ✅ N.0d WHERE IT HANDS OFF: the clock, with its gate already green
>
> E16's by-product is the live lead. **Mean gap between the experts' moves and
> ours ≈ 0; TRUE per-position spread 0.077–0.087** over three independent
> samples ⇒ typical |gap| **0.061–0.069**, 90th percentile **0.126** against
> ROADMAP §2.7's pre-declared **0.10** build line. **The value is real and
> nobody captures it — not us, not the 1100+ band.** A problem search can solve
> and imitation cannot, because there is no demonstrator to clone.
> ⛔ Unchanged: the **large-or-nothing** validation blocker (§2.7) — an agent
> spending real time per move cannot be A/B'd at n≥2000 on this box.
>
> 📌 **USER DECISION (day 28): run this thread until it is dead, THEN take up
> the clock.** ROADMAP §2.7 holds the parked clock design, its price, its kill
> gate, and the §8v "+154 Elo" correction.
>
> ## N.1 What is ESTABLISHED (do not re-measure)
> - 🔴 **§8u — the founding datum.** We cloned the #2 player **successfully**
>   (held-out agreement 59.9% → 67.2%) and measured **−92 Elo**. Field
>   disagreement 30.2% → 32.0% → 36.2% maps monotonically to Elo 0 → −55 → −92.
>   ⚡ **Covariate shift is RULED OUT** for that arm (§8s: 26.7% vs 31.9%,
>   near-symmetric, with a 1.7% positive control) — so "we dragged the expert
>   off-distribution" is not the explanation.
> - 🔴 **§8r** — agreement peaks at rating 1050–1100 and falls in **both**
>   directions over 87 same-deck, zero-exposure demonstrators. Agreement measures
>   distance from the fitted mode, **not skill**. ⇒ every rate-vs-experts eval we
>   run is a *conformity* metric, which is why they all say we match the experts
>   while we sit at 976.
> - 🔴 **§8bs** — no blunder signature in the 27 losses (worst decision −0.069 in
>   losses vs −0.070 in wins vs −0.078 for the players who beat us).
> - 🔴 **§8bv** — conditioned on the board, **winners and losers play the same**
>   (−0.0024 bits) at the coarse bucket used. ⇒ a **strong prior against**
>   outcome-conditioned / "upside-down RL" cloning — not a proof; a net reading
>   the full board could resolve finer structure than this estimator.
>
> ## N.2 What is KILLED (⛔ do not rebuild)
> - ⛔ **Plan-as-latent-cluster, predicting the next MAIN action** (§8bv):
>   +0.001 / −0.009 / +0.037 / +0.090 bits at k=2/3/4/6 against a **+0.372**
>   estimator control with only two groups. ⚠ And note the two traps, because
>   the next instrument will meet both: a too-fine state bucket drove the plug-in
>   MI bias high enough to make the positive control read **negative**, and the
>   first signature was **circular** (cluster on card play rates → predict which
>   card was played, +0.27→+0.46 of tautology).
> - ⛔ **Bench-symmetry averaging** (E15/§8bu): 0.513 [0.492, 0.535] vs a
>   pre-registered 0.500, both controls holding. Do not re-cut at another K.
> - ⛔ **Demonstrator selection in any form** (B7/§8u), **more data**, **capacity**,
>   **deck perturbation**, **search**, **within-turn sequencing**, **B8 at β=1.0**.
>
> ## N.3 🔴 THE TWO HYPOTHESES §8u CANNOT YET SEPARATE — this is the actual open question
>
> | | claim | predicts |
> |---|---|---|
> | **H1 "no plan"** | the expert's moves are individually good but only coherent as a *sequence*; partial copying breaks the coherence | a **coherently committed** expert imitation beats a partial one, and incoherence is measurable in our own play |
> | **H2 "the mode is the local optimum for THIS field"** | any deviation from the field mode costs, regardless of direction or of the source's strength | nothing recovers by committing; the monotone 30.2→32.0→36.2 ⇒ 0→−55→−92 is the whole story |
>
> ⚡ **The fact that breaks the tie is already on the board: the top players DO
> beat the mode.** So the mode is not optimal, and H2 in its strong form is
> false. What is true is narrower and more uncomfortable: **our deviations from
> the mode have all been in bad directions and theirs are in good ones.** That is
> not an imitation problem — it is a *credit-assignment* problem, and the only
> instruments that can tell a good deviation from a bad one are outcome feedback
> (B8, closed on **method** at β=1.0 with the sweep never run — "unfalsified
> rather than refuted") and a stronger evaluator than `evalfn`.
>
> ## N.4 Ranked probes for the next session (none built, none pre-registered yet)
>
> 0. ⭐⭐ ✅ **FEASIBILITY RESOLVED — see §N.0 above; this entry is kept for the
>    reasoning, but its "CRN" and its affordability claim are both corrected
>    there, and the live artifact is E16.**
>    **THE INSTRUMENT THIS PROJECT HAS NEVER HAD: counterfactual action value
>    by paired rollouts with common random numbers.** Every eval we own is either
>    a conformity metric (rate vs experts — §8r says that cannot measure skill) or
>    a weak evaluator (`evalfn`, AUC 0.667 early). Neither can answer the only
>    question that matters: **"in THIS exact position, is their move better than
>    ours?"** A rollout instrument can: fork a real position, play option A and
>    option B from it with the clone piloting both seats under the **same seeds**,
>    and difference the win rates. No corpus, no mode, no conformity.
>    - ✅ ~~**Feasibility check FIRST, ~20 min, and it is a real risk**~~ — **RAN
>      AND PASSED, 60/60 (§8bw).** The engine accepts an sbi captured in another
>      process and reconstructs the position exactly.
>    - 🔴 ~~**Per-DECISION resolution is unaffordable and that is fine.**~~
>      **MEASURED FALSE** — 101 ms per rollout to terminal. Pooling across a
>      decision class is a choice, not a necessity. ⚠ But size on **positions**,
>      not pairs: the pair-level interval is 4.1× too narrow (§8bw).
>    - ⚡ **What makes it worth the build: it can be pointed at the EXPERTS' games.**
>      Take positions from `flg` / `Raihan Ramadistra` / `Sixth Sense`, score
>      *their* move against *our net's* move by rollout, and you have measured
>      whether they are better and where — with no assumption that agreement means
>      skill. That is the experiment §8r's conformity trap has blocked all project.
>
> 1. ⭐ **Measure OUR incoherence directly, no plan recovery needed.** Count
>    commitment switches per game — how often we change attacker / abandon a
>    setup line mid-game — for us vs the three current Grimmsnarl experts
>    (`Raihan Ramadistra`, `flg`, `Sixth Sense`; §8bq). H1 predicts we switch
>    MORE. This needs no latent variable, no clustering, and no MI estimator, so
>    it dodges every trap §8bv hit. **If we do not switch more, H1 is in serious
>    trouble and the thread should turn to N.4.3.**
> 2. **Commitment as a mechanism, tested without a corpus.** Add hysteresis to
>    the shipped net (stick with the previously-preferred attacker/target unless
>    the logit margin exceeds a threshold) and A/B it. This tests whether
>    *commitment per se* is worth Elo, independent of whose plan it is. Cheap:
>    it is a wrapper like `symavg.py`, and E15's harness is reusable.
> 3. **The credit-assignment reframe (N.3).** If 1 and 2 null, the thread's own
>    logic points here, not at more imitation work.
> ⚠ **Pre-register every one of these** (`docs/experiments/`) before the first
> arena game — E15 is the template, and writing the prior down first is what
> turned a +0.013 into an honest null.
>
> ## N.5 Live state at handoff
> - **LB: rank ~150 of 6,653, displayed 976.2** (= best of the ACTIVE pair; an
>   older submission at 978.4 does **not** display, which is how "best active"
>   was confirmed empirically).
> - **Active pair is now two byte-identical `v5_s2` agents** — `55326513` (971.5)
>   and `55382430` (submitted 15:04 UTC 08-09, reading 857.7 and still
>   converging; μ starts at 600 and needs 4 h+). `55321893` (ens2, 855.9) is
>   evicted and frozen. ⚠ **Do not read `55382430` as a result before ~20:00 UTC.**
> - Rank-20 cutoff **1089.0**; every top-8 team runs exactly 2 submissions.
>

> ## 0. 🔴 E14 — WP-REGRET AUTOPSY: BUILT, RUN, CLOSED THE SAME SESSION (`p77_wp_regret.py`, §8bs)
>
> The user's named seam — *"a few games where 1 bad decision cost us games"* —
> is the one shape no rate miner in this repo can see. It is now measured and
> **it is not there.** `scripts/p77_wp_regret.py`, log `out/logs/p77_wp_regret.txt`.
>
> | control | reading |
> |---|---|
> | our worst decision, **27 losses** | **−0.069 WP** |
> | our worst decision, **49 wins** | **−0.070 WP** ← identical |
> | **their** worst decision, the 27 games **they won** | **−0.078 WP** ← worse than ours |
> | events at \|ΔWP\| ≥ 0.20 | **3 in 76 games = 0.039/game**, 1 of them FORCED, **1 across all 27 losses** |
>
> ⛔ **0.039/game against the 0.5 gate — 13× under, the smallest candidate rate
> this project has measured.** No concentration either (worst decision carries
> 45.8% of our negative flow in losses, **40.0% in wins**), and in absolute terms
> our own within-turn decisions bleed **0.132 WP/game in losses vs 0.166 in
> wins** — the stream a blunder would live in is *bigger in the games we won*.
> ⚡ **`evalfn`'s §8l AUC replicates on real ladder games: 0.667 early / 0.905
> late** (§8l's 0.685/0.901 was 200 self-play games), fitted per turn bucket on
> an independent 250-game corpus. The instrument is sound; the seam is empty.
>
> ## 0a. 🔴 R1 AND R2 BOTH RAN AND BOTH ARE NULLS — day 27, 2nd session
>
> **R2 / E15 — average out the bench-slot nuisance.** Pre-registered at `c2ce197`
> BEFORE any game. `bc:sym8` vs `bc:base`, same net, mirror, n=2,000:
> **0.513 [0.492, 0.535]** against a pre-registered 0.500 ⇒ **NULL, does not
> ship.** Controls held: `sym1` bitwise identical to `bc` (0/1,915 selects) and
> `sym8` fired on **8.36%** of selects, so it is not a null for want of firing.
> The prior (§8bd's indifferent near-tie band) was written down first and was
> right. ⛔ **Do not re-cut at another K.** §8bu. Code stays, off by default.
>
> **R1 — the latent "plan".** 🔴 **Killed at its sizing gate before any net was
> trained** (§8bv, `p79_plan_audit.py`, 56,611 MAIN decisions). Plan clusters
> carry **+0.001 / −0.009 / +0.037 / +0.090 bits** (k=2/3/4/6) about the next
> action beyond the board, against an estimator control that reads **+0.372 with
> only two groups**. ⚠ Two traps caught on the way: a too-fine bucket made the
> control read NEGATIVE, and the first signature was **circular** (clustering on
> card play rates to predict which card was played, +0.27→+0.46 of pure
> tautology).
> ⚡ **And the by-product is the day's most reusable fact: conditioned on the
> board, WINNERS AND LOSERS PLAY THE SAME (−0.0024 bits)** — a third independent
> corroboration of §8bs. ⚠ **Stated correctly:** the bucket is coarse
> `(turn//3, prize diff)`, so this is a **strong prior against** outcome-conditioned
> / "upside-down RL" cloning, **not a proof against it** — a net reading the full
> board could see finer structure. The first write-up said "retires" and that was
> an overclaim, corrected the same session.
>
> ## 0b. 🔴 THE PART TO CARRY FORWARD: a realized trajectory CANNOT see an error of omission
>
> The discriminator was run because a null needs one. §8bm's seven known
> dominated plays ("a KO was available and we spread instead") score
> **+0.002…+0.005 ΔWP and rank mid-pack** (31 of 58, 44 of 76, 86 of 124); only
> one is rank 1 and it is −0.016, inside the noise floor. **Declining a prize
> still deals damage, so the state improves — the whole cost sits in the branch
> that never happened.**
> ⇒ **§8bs is a null about SELF-INFLICTED damage only. Never quote it as "no
> decision cost us a game."** ⚠ Same cause, second blind spot: **the attack is
> the last select of a turn, so a bad attack's cost lands in `boundary` and is
> never attributed.**
> ▶ **The honest version, if this axis is reopened: the OPTION-LEVEL
> counterfactual** — score every legal option at the same state instead of the
> one realized trajectory. For damage placement it is pure arithmetic and needs
> no engine (`p72`'s option→Pokémon mapping is verified 839/840).
>
> ## 0c. ⚠ THREE INSTRUMENT DEFECTS, none of which reached a number — and one was MINE
>
> 1. The IRLS **diverged** (slope 74,173, every state pinned to 0/1); the
>    reliability table caught it. Fixed + a `saturated` column now prints.
> 2. 🔴 **`evalfn` is UNDEFINED during setup — it returns −8.2 on an empty
>    board**, because it reads prizes as `6 - len(prize)` *taken* and one pile is
>    dealt before the other. ✅ Never touched a shipped number (live callers are
>    `planner`/`sequencer` at MAIN, turn ≥ 1, both closed axes) — but **it is
>    inside §8l's early-game 0.685**, whose clean value is 0.667.
> 3. ⚠ **My first fix was a selection bias worse than the bug:** guarding on
>    "both actives non-empty" deleted **158 of 177 damage deltas** — the state
>    after a DAMAGE select has the defender's active empty exactly when we
>    **KO'd** it. **A guard that moves a denominator 88% is a finding, not a fix.**
>
> ## 1. The correctness inventory (compiled, not new measurement)
> **Every decision class with a definable right answer (= dominated/arithmetic) is ≥96% correct unaided:** lethals 316/316 (§8), KO-in-targeting 99.1% (§8bm), Adrena-Brain source 96.1% (§8bm — dominance = source's counter load caps the move; 15/387 under-moved, 2 dmg/game), promotion 7/803 all-forced (B2), Pokégear 39/39 (§8ag). **The tradeoff mass (~98.8% of targeting, most of MAIN/TO_HAND) has NO ground truth** — only corpus agreement (71% `--equiv`, §8x; measures distance-from-mode, not skill, §8r) and expert-relative gaps (§8bo/§8bl/§8br), which are 0-for-5 when forced by rule. ⇒ the 135-Elo gap lives entirely where correctness is undefinable.
>
> ## 2. DS audit verdicts (assessment, no new cells)
> - **Data quality is NOT the binding constraint** — volume/labels/demonstrators/coverage all measured dead (§1, §8t/§8u, §8bi). One untested variant: matchup-share resampling (corpus 56.9% mirror vs field 31.6%, §8bn) — prior against it (all reweighting lost monotonically with distance from the field mode), needs ≥3 seeds + fresh-game confirm, does not fit pre-freeze. Not scheduled.
> - **Not overfitting — underfitting the corpus** (71% vs 95.6% ceiling; 8.2× capacity → −43; B8 fit better, played worse). Real overfitting is meta-level and already measured: field-mode-by-design (mirror ~0.500 cap), winner's curse 0.027–0.031, instrument weights wrong twice (§8ac vs §8bn).
>
> ## 3. Proposed evals beyond the arena (ranked)
> 1. ~~**WP-regret autopsy**~~ 🔴 **BUILT AND CLOSED — see §0/§0b above and §8bs.** The promise ("finds frequency-1 blunders no rate-miner can") was **half true**: it finds frequency-1 *self-inflicted* drops — there are none above the noise floor — and is **provably blind** to frequency-1 *missed opportunities*, which is what the discriminator established.
> 2. **Frozen tactics/regression suite** — puzzle set of provable positions + past-bug positions; scores any net in seconds; non-regression instrument + report material.
> 3. **Entropy/calibration profiling** — map where the net is uncertain per context/phase; the targeting map for Round-2's 1,800 s clock.
> 4. **CRN seed pairing in the local engine** (`battle_start` takes no seed, §8ad) — paired A/B arms would collapse arena variance; best instrument investment for R2.
> (1–2 fit pre-freeze; 3–4 are freeze-window/R2 work.)
>
> ## 4. Standing recommendation (unexecuted)
> **Second-slot duplicate of `v5_s2` by ~08-14**: next submission evicts `55321893` (ens2, weaker); decision-identical agents read 63–87 apart (§8ak) and displayed = best active ⇒ max-of-two-draws ≈ +25–35 displayed, zero idea risk (E10 F2 step 4 logic). Rule-2 settling reads on `55326513` first.
>
> ---
>
> # ▶ DAY 26 (2026-08-08): THE PASSIVE-DAMAGE SEAM IS SIZED ON OUR OWN LADDER GAMES AND BOTH HALVES DIE AT THE GATE — and the field at 1027 is not the field we planned for
>
> **📌 USER DIRECTIVE (day 26): report work stays SUSPENDED, Track A only —
> push the agent. Named seams: passive-damage targeting, Petrel, and "a few
> games where 1 bad decision cost us games."** The user supplied **76 real
> ladder games of the shipped agent** in `replays/submission_v5_s2`, and asked
> to start there. `STRATEGY.md` untouched; `EVIDENCE.md` gets entries.
>
> **Live board (user-read): `55326513` 1027.2 · `55321893` 906.2.** ⚠ Both moved
> from 1044.3/937.1 — **inside the 63-point noise floor (§8ak), so nothing has
> gone wrong and nothing is confirmed.** Rule-2 reads still owed.
>
> ## 🔴 1. E12 — BOTH HALVES OF THE PASSIVE-DAMAGE FAMILY CLOSE ON SIZING
>
> The decisions are frequent and they are **the net's alone** — the shipped
> bundle reads `{'chip_targeting': False, 'energy_spread': False,
> 'counter_source': False}`. Denominators are **10.2 damage-placement and 5.09
> Adrena-Brain source choices per game**, both ~10–20× the 0.5/game gate, so
> nothing here dies for want of opportunities:
>
> | dominated test | rate | verdict |
> |---|---|---|
> | a KO was available and we spread instead | **0.09/game**, 0.9% of real choices, **2 of 27 losses** | 🔴 under the gate that killed Morgrem (0.2), Pokégear (0.27), Archaludon (0.187) |
> | Adrena-Brain source under-moved (the cap) | **0.20/game**, **1.5% of the ability = 2 damage/game** | 🔴 under the gate |
>
> ⚡ **The complement is the real finding: unaided, the net takes the available
> KO 99.1% of the time and picks the source correctly 96.1% of the time.** That
> is `p2_lethal`'s 316/316 one level down, and it explains why `counter_source`
> is switched off and *harmless* rather than merely unproven.
> ✅ **A fifth confidently-wrong script was avoided by a control run FIRST:**
> the obvious `steps[i][seat]["action"]` route maps the chosen option to the
> Pokémon that actually lost HP **11 of 48 times**; the miner's
> `steps[0][0].visualize` / `v["selected"]` route matches **839 of 840**.
> `p72_loss_autopsy.py --verify` re-runs it. **Check it first if a number here
> looks wrong.**
> ⛔ **The "would it have survived" counterfactual is NOT computable from these
> replays** — log type 16's `value` reconciles with the board only 62% of the
> time. The dominated test was chosen *because* it needs no such claim. §8bm.
>
> ## ⚡ 2. THE FIELD AT ~1027, AND A CONTROL THAT REFRAMES THE GAP
>
> 76 games, **64.5% overall**, 68 distinct opponent teams. Loss mass: **mirror
> 31.6% share / 58.3% WR / 10 of 27 losses**, Alakazam 23.7% / 77.8% / 4, Mega
> Lucario 9.2% / 57.1% / 3, **Teal Mask Ogerpon 6.6% / 40.0% / 3**.
> 🔴 **The mirror is 31.6%, not §8ac's projected 71.4% above rating 1000.** The
> ordering survives (it is still the biggest bucket and the most loss mass); the
> **weight does not** — any weighted verdict quoting §8ac's shares is quoting a
> share this dump contradicts. **Alakazam is back at 23.7%** after being dropped
> from planning since day 9.
> 🔴 **Censusing ntumlnoob's 330 games from THEIR seat (#2, 1162.8): Teal Mask
> Ogerpon beats them 37.5% vs our 40.0%** ⇒ a **matchup property of the
> archetype, not a piloting failure** — off the list before anything was spent.
> Their overall is **64.5%, identical to ours** — ⚠ **not equal strength**,
> their field is ~130 points stronger, so it means they are better. What it does
> say: **the gap is not concentrated in a matchup we could close by targeting
> better**, the same place §8bj and §8bl landed from the behavioural side. §8bn.
>
> ## 🔴 3. E13 — THE KO-SETUP RULE, PRE-REGISTERED AND KILLED AT THE GATE THE SAME DAY
>
> The user approved building a targeting rule off §8bo's 22.4%-vs-7.0% gap.
> Frozen first in `docs/experiments/E13-ko-setup.md` at **50a6344, before
> `p74_ko_setup_sizing.py` existed** — then sized: **0.04 firings/game against
> the 0.5 gate, 12× under.** No rule written, no arena time spent.
>
> ⚡ **The kill is bigger than the sizing.** The same funnel from ntumlnoob's
> seat (149 games) puts **7 of 406** damaged-Active decisions in the KO-setup
> band — **1.7% vs our 1.0%.** The band is nearly empty for the experts too, so
> **the mechanism E13 was built on ("they concentrate to manufacture KOs") is
> measured FALSE**, not merely too rare. ⚠ That retires my reading of §8bo's
> "KO-available 1.2% vs 6.3%" as evidence of deliberate KO manufacture.
> ✅ **One thing was gained:** the funnel controlled a confound §8bo never did —
> whether our attack *already* kills their Active, making the counter genuinely
> wasted. Ours **45.4%**, theirs **45.8%**. §8bo's ~86%-behavioural conclusion
> survives; only this explanation of it is dead. §8bp.
> ⛔ **Do not revive at a different chip value, and do not reach for the
> clause-4-only "prefer a damaged Active" variant** (it sizes at 2.6–4.0/game
> and will look tempting). It was frozen as a separate experiment *precisely* so
> it could not become the fallback the moment clause 5 failed, and it is a
> 100%-forcing rule against a 22.4% behaviour — E11's error verbatim (0.487).
>
> ## ▶ 4. WHAT IS RUNNING / WHAT IS NEXT
>
> - ✅ **08-03…08-07 mined AND censused: 1,972 games, 3,944 seats** (`p75_day_census.py`).
>   ⚠ **The feed's floor has RISEN — `avg_score` min 1100, median 1166** (§8i
>   said ~1055). This is ~140 points above our 1027.
>   🔴 **Our archetype is the MOST-PLAYED deck up there (23.7%) and wins 47.9%
>   — 46.2% excluding its top pilot, so it is a DECK property, not a pilot one.**
>   ⚡ **"Mega Lucario ex is the best deck (62.6%)" is FALSE — Majkel1337 is
>   84.2% of its games; every other Lucario pilot wins 43.5%.** The
>   pilot-concentration control in `p75` is what caught it; run it before
>   quoting any archetype win rate from a top-of-ladder dump. §8bq.
>   ⚡ **`李秉叡（ntumlnoob）` has switched off Grimmsnarl to Dudunsparce.** §8bo's
>   comparison stands (it was like-for-like), but they are no longer the
>   demonstrator to mine. **`flg` (90 games, 55.6%) is the best current
>   Grimmsnarl source; `Raihan Ramadistra` (472) is the volume.**
> - 🔴 **PETREL IS NOW INSTRUMENTED AND CLOSED BY SIZING** (`p76_petrel_fetch.py`,
>   §8br). Petrel resolves **1.59×/game**; against 501 games from three current
>   Grimmsnarl pilots we **over-fetch Unfair Stamp (+17.8%) and Night Stretcher
>   (+12.5%)** and **under-fetch Spikemuth Gym (−8.1%) and Rare Candy (−5.4%)**.
>   ⚠ **Do NOT size this by adding the take-rate gaps** — they are conditional
>   rates on overlapping denominators. The correct sizing is total variation
>   between the fetch distributions: **18.3% × 1.59 = 0.29 fetches/game, under
>   the 0.5 gate**, and that is the CEILING (whole distribution at once). The
>   biggest single-card rule is 0.13/game.
>   🔴 **Two mapping bugs here, worth carrying forward: a PLAY option (type 7)
>   has NO `area` field** — filtering on `area == HAND` makes every card play
>   invisible (it returned "Poffin offered 9 times in 76 games", which E11
>   contradicts). Take option card ids from **`optfeat.option_features`**, the
>   extractor that built the training data. ⚠ **And the first positive control
>   passed at n=10** because it required the *next* record to be our seat;
>   scanning to our next record gives n=1375 at 96.8%. **A control with n=10
>   licenses nothing.**
> - ⛔ **BOTH seams the user named on day 26 are now closed by the SAME gate:**
>   passive-damage targeting (0.09/0.20 §8bm, 0.04 §8bp) and Petrel (0.29 §8br).
> - ⛔ **Do not build a passive-damage rule on §8bm** — it is a sizing kill, and
>   both dominated tests are an order of magnitude under the gate. **E13 (§8bp)
>   closes the tradeoff half the same way. The passive-damage seam is now
>   measured from both ends and has produced nothing shippable.**
>
> ---
>
> # ▶ DAY 25, 3rd SESSION (2026-08-07): THE FINAL PUSH RAN IN FULL AND SHIPPED NOTHING — F1 closed, F2 null, F3 killed, and the board reads **1010.1**
>
> **All four E10 items concluded in one session. Nothing cleared a bar, and every
> bar was written before its cell ran.** ⇒ **`55326513` (`policy_v5_s2`) stands
> as the final agent unless something new appears; no submission is due, and the
> 08-15 slot is unspent.** Full record: `EVIDENCE` **§8bh** (the selection
> debt), **§8bi** (both sizing gates), **§8bj** (F1's verdict). Pre-registration:
> `docs/experiments/E10-final-push.md`, frozen in `ad7d29f` before any cell.
>
> ## 📈 1. THE LIVE STANDING — our best number ever, and it is NOT settled
>
> | active? | submission | score | what it is |
> |---|---|---|---|
> | ✅ **active** | **`55326513`** (08-07 14:05) | **1004.5 → 1010.1 → 1044.3** | `policy_v5_s2`, single net, rules off |
> | ✅ active | `55321893` (08-07 09:59) | 954.3 → 934.7 → 917.3 → **937.1** | ens2 vote — **not monotone either; it is noise, not a trend** |
>
> ⚡ **RANK 55 / 6,525 AT 1044.3** (17:35 UTC, +3h30m) — **by far the best rank
> and score this project has had** (previous best 129 at 990.7); top is `LiamK`
> **1202.3**, so the gap is **158**. 🔴 **NOT SETTLED: three reads, all inside
> or near the 4-hour window, and the score rose at every one** (1004.5 → 1010.1
> → 1044.3). Rule 2 is **not** satisfied and a rising score is unconverged, not
> momentum — `55054446` once read 916.8 → 936.0 → 979 and settled at **905.2**.
> ▶ **Next: two reads ≥1 h apart, both after 18:05 UTC, and they must AGREE
> before this number is quoted anywhere.**
>
> ## 🔴 2. THE SEED PREMIUM WAS MOSTLY SELECTION — `s2` is +7 Elo, not +26
>
> The day-25 debt is paid. `s2` vs `s1` on **fresh games**, same config, same
> weight files: **0.510 [0.484, 0.536]** against the screen's 0.537. It **does
> not resolve.** `s4` screens at 0.480. ⇒ §8bg's "50 Elo seed spread" was a
> **max-minus-min over three draws whose max was selected**; the between-seed sd
> over four *unselected* offsets is **≈11 Elo**, so two random seeds differ by
> ~15, not 50. **The shipped net is a median seed, not a lucky one.**
> ⚠ **F2 is re-priced before it spends its budget: best-of-12 buys ≈+18 Elo, not
> +35–40, and one screen→confirm pair already gave back 0.027.** The protocol is
> unchanged — **screens select, only a fresh-game confirmation ships.**
>
> ## 🔴 3. F1 IS CLOSED, AND IT CLOSED ON A COUNTING UNIT
>
> The mirror gate passed hugely (**257 games, 22,665 expert decisions**). The
> extraction found **4,785 confident disagreements (18.6/game)** and its top
> cluster — the clone wanting **Munkidori**, 8.4/game — **dissolved under an
> on-policy control**: we fire Adrena-Brain **6.42×/game**, the 1150+ pilots
> **6.23×/game**. Identical. We take it at the first opportunity, they take it
> later in the turn ⇒ **sequencing, a closed axis.** The one ordering-free
> difference (Spikemuth Gym's search: they stop at turn ~9.7, we never stop) is a
> **tradeoff** ⇒ no rule (rule 11, 0/4). **F1's pre-registered kill criterion is
> met and it closes as a chapter.**
> ⚡ **New standing rule (21): SIZE AND RANK PER TURN, not per decision.** §8ai's
> lesson was remembered as an anecdote and E10 still let the ranking run per
> decision; the per-decision view overstated this cluster by ~25×.
>
> ## 🔴 3b. E11 — F1 WAS CLOSED ONE STEP EARLY, AND THE STEP FOUND SOMETHING (which then measured zero)
>
> Rule 21 says rank per turn — but F1 applied the per-turn *correction* only to
> the two clusters the per-**decision** ranking had already chosen.
> `scripts/p70_perturn_sweep.py` ranks **every** option class per turn with no
> pre-selection, and found the first candidate where **WE are the weaker
> player**: **`Buddy-Buddy Poffin`**, conditioned on identical board occupancy —
> experts **70.2%** of available turns at board 4, us **29.4%**; at board 5,
> 46.9% vs 7.2%. **0.80 plays/game**, ordering-free, confound-checked (both sides
> decline at the same mean board size).
> ✅ Rule built (`targeting.poffin_force`, arena flag `poffin`, default OFF),
> pre-registered in `docs/experiments/E11-poffin.md` at **`a50a240` before the
> code**, positive control passed (play rate 39.7% → 61.6%).
> 🔴 **A/B, byte-identical net, n=2,800: 0.487 [0.469, 0.506].** Fails the bar,
> does not resolve, slightly negative. **Does not ship. Tradeoff rules are 0/5.**
> ⚡ **The lesson is the thesis:** a behavioural difference from the 1150s that is
> real, sized, ordering-free and confound-checked still converts to **zero Elo**.
> ⛔ **Do NOT now try a milder threshold** — E11 pre-registered that as a separate
> experiment precisely so it cannot become a knob tuned after seeing the result.
> §8bl.
>
> ## 🔴 4. F3 IS KILLED ON AVAILABILITY
>
> Over the four dumps that built `pds_v4` (1,603 games, avg_score 1057–1223):
> **Archaludon and Mega Lucario appear in ZERO games**, and the miner discards
> nothing — it clones both seats of every game. **The data does not exist**
> (§8i: the episode feed stops at ~1055; §8ac: those decks are 0/47 above rating
> 900). And the mismatch is self-closing — the corpus is 56.9% mirror against a
> field that is **71.4% mirror above 1000**, which is where we now play.
> `PARKED-corpus-coverage.md` is marked CLOSED.
>
> ## ▶ 5. WHAT IS STILL RUNNING / WHAT IS NEXT
>
> - 🔴 **F2 IS DONE AND IT SHIPS NOTHING.** Ten seeds of one recipe; the single
>   screen winner `s7` (**0.528** vs `s1`) then read **0.487 [0.468, 0.505]**
>   against the incumbent `s2` over 2,800 fresh games — **below the 0.53 bar and
>   not even above 0.500.** ⇒ **keep `policy_v5_s2`; `55326513` stands; no
>   submission is due.** The winner's curse measured **0.031** here against
>   §8bh's 0.027. ⚡ **Why it could not work, quantitatively:** between-seed sd
>   **0.0190** vs a 1,400-game screen's error **0.0134** — *the same size*, so
>   the max of ten screens selects mostly for measurement error. A real harvest
>   needs **~5,100 games/screen (~3.5 h)** for a prize of ≈+20 Elo on a ladder
>   whose noise floor is 63. §8bk.
> - ⏱ **Rule-2 reads on `55326513`** at ≥17:10 and ≥18:10 UTC.
>   🔴 **BLOCKED 17:04 UTC — the Kaggle OAuth token EXPIRED.**
>   `~/.kaggle/credentials.json` is an OAuth credential (`access_token` +
>   `refresh_token`), **not** the old `kaggle.json` API key, and its
>   `access_token_expiration` was **16:56:20 UTC** — every call after that
>   returns **401 Unauthorized**, including `competition_leaderboard_download`.
>   The SDK did **not** silently refresh despite holding a `refresh_token`.
>   🔴 **The OAuth token's TTL is 12 h** (issued 04:56 UTC, expired 16:56), so
>   `kaggle auth login` buys half a day and then breaks mid-session — it did
>   exactly that here, and a re-login attempt left the file **unmodified**
>   (same mtime, same expiry), so the flow does not reliably take.
>   ✅ **FIXED 17:35 UTC, and the recipe is VERIFIED — ⛔ it is NOT `kaggle.json`,
>   which is the OLD SDK's scheme and does not work here.** This SDK wants a raw
>   token string:
>   1. kaggle.com/settings/api → under **API**, *Generate New Token* → copy it
>   2. save the bare string (no JSON, no quotes, no trailing newline) to
>      **`C:\Users\USER\.kaggle\access_token`** — or export `KAGGLE_API_TOKEN`
>
>   It does not expire, so this is the durable fix for the 08-15 submission and
>   the daily reads through the 08-31 continued-play window.
>   ⚠ **A token is a credential: never paste it into a chat, an issue or a
>   commit.** If one is ever exposed, *Expire API Token* on the same settings
>   page invalidates it and a new one can be issued.
> - ✅ **A defect-shaped reading that died to one look at the raw data, recorded
>   so nobody re-finds it:** 22.3% of `TO_HAND` decisions offer options with no
>   card identity (`opt_card == 0`) and they survive `--equiv`. They are **PRIZE
>   selections — face down.** Both sides are guessing; the only differing feature
>   is the index disambiguator, which is the correct encoding. **Not a blind
>   spot, nothing to price.** §8bj.
>
> ---
>
> # ▶ DAY 25, 2nd SESSION (2026-08-07): THE FINAL PUSH — MIRROR MINING, SEED HARVEST, COVERAGE GATE
>
> **📌 USER DIRECTIVE (day 25, 2nd session): a FINAL PUSH at the ~150-Elo gap to
> the leaders (we show 990.7-class play; the top holds 1145–1166). Track A only
> still stands; `STRATEGY.md` got ONE user-authorised edit (§7b.4, the thesis
> statement) and is otherwise untouched; `EVIDENCE.md` keeps getting entries.**
>
> 🔒 **The full plan is pre-registered in `docs/experiments/E10-final-push.md`
> — frozen before any cell, with bars and predictions. Read it before running
> anything. ROADMAP §2.6 carries the ranked table; this box is the summary.**
>
> ## The frame — three measured facts pick the experiments
>
> 1. **The climb runs through the MIRROR** (§8ac): 33.3% of our games at 955,
>    51.1% above opponent rating 900, **71.4% above 1000**.
> 2. **A field clone converges to 0.500 against field-modal play in the mirror
>    by construction** — and the leaders are rule agents. Whatever they do
>    differently in our own 60 is minable: we hold **557 games from two 1150+
>    teams on our exact list** (`replays/ntumlnoob_31-07-2026`,
>    `replays/sixth_sense_31-07-2026`).
> 3. **True strength is the target, not the displayed number** — the LB reads
>    decision-identical agents 63–87 apart (§8ak), but continued play after
>    08-17 converges toward true level, and Round 2's BO3 plays the real agent.
>
> ## The four items, ranked (details + kill criteria in E10)
>
> - **F1 — mirror-conditioned disagreement mining** (highest ceiling): filter
>   the expert dumps to MIRROR games (sizing gate: ≥100 games), extract
>   large-margin disagreements between `policy_v5_s2` and the 1150+ pilots
>   (`context_accuracy`/`p16` machinery, `--equiv`), cluster, size (≥0.5
>   firings/game), classify with the discriminator — **dominated → rule
>   candidate; tradeoff → chapter, no rule (0/4)**. Watch the top clusters
>   (§8ah's method). Rule A/Bs run byte-identical-net rule-toggled, so the seed
>   nuisance cancels. ⛔ This is an AUDIT of experts, not B7 cloning and not an
>   E3 teacher.
> - **F2 — seed harvest, done honestly** (guaranteed EV): seed = ±25 Elo pure
>   nuisance (§8bg) and only 4 seeds ever sampled. Screen `s4` + confirm `s2`
>   on fresh games (the item-5 debt), train/screen ~6–8 more seeds vs `v5_s1`
>   at n=1,400 shipped-config, then **one confirmation on fresh games vs the
>   incumbent `s2`: ship bar = point ≥0.53 AND CI excluding 0.50.** Screens
>   never ship. If shipping: **submit twice, by 08-15.**
> - **F3 — the corpus-coverage sizing gate** (30 min): run the probe in
>   `docs/experiments/embeddings/PARKED-corpus-coverage.md`; expected kill via
>   §8ac (the blind archetypes vanish above rating 900); verdict written either
>   way, no training this side of the freeze.
> - **F4 — B8 β: DECLINED, recorded** — the one honestly open door on a closed
>   axis, declined for the freeze window (two nulls, closed on the method, hard
>   stop spent). Do not relitigate.
>
> Plus carryover: **rule-2 LB reads on `55326513`** (≥1 h apart, both slots in
> one call).
>
> ## ⛔ Standing constraints that bind this push
>
> - Fifteen closed axes + ensembling stay closed; F1's feature-shaped findings
>   need a §8au-priced defect before any retrain (the B1-instance-5 bar).
> - Every A/B in shipped config (`--no-rules` both arms — the §8be trap).
> - Interventions need ≥3 seeds or a seed-cancelling design; a two-cell delta's
>   interval is √2× a single cell's.
> - Name what a submission evicts before quoting any bar (next eviction:
>   `55321893`, ens2, 934.7).
> - EVIDENCE entries (§8bh+) the session each item concludes; verdicts blank
>   while cells are in flight.
>
> ---
>
> # ▶ DAY 25, 1st SESSION (2026-08-07): THE SEED IS WORTH 50 Elo, ENSEMBLING IS NOT, AND `policy_v5_s2` IS SUBMITTED
>
> **📌 USER DIRECTIVE, STANDING: report work is SUSPENDED. Track A only.**
> `STRATEGY.md` gets no edits. `EVIDENCE.md` keeps getting entries.
>
> Full record: `EVIDENCE` §8bf (shipped-config re-measurement) and **§8bg** (the
> seed result). Pre-registration: `docs/experiments/E9b-which-net-ships.md`,
> frozen in `e214c66` before the deciding cell reported.
>
> ## 📈 1. THE LIVE STANDING — and a submission was made
>
> | active? | submission | score | what it is |
> |---|---|---|---|
> | ✅ **active** | **`55326513`** (08-07 14:05) | *pending* | **`policy_v5_s2`, single net, rules off — the day-25 ship** |
> | ✅ active | `55321893` (08-07 09:59) | **934.7** | ens2 (v5 + v5_s1 vote); read 954.3 earlier, **drifting down** |
> | ⛔ inactive | `55169114` | 918.5 | evicted by the new submission |
> | ⛔ inactive | `55160229` | 978.4 | the old 990.7 — **frozen, gone, and NOT recoverable** |
>
> 🔴 **A RESUBMISSION DOES NOT INHERIT A SCORE.** Every submission starts at
> μ=600 and draws fresh. The 990.7/978.4 cannot be "restored" by resubmitting
> that tarball — and we know its true level, because `55169114` is
> **decision-identical** to it and sits at **918.5**. Same function, 978.4 vs
> 918.5. ⇒ **the 990.7 was the top of a range, not an agent we lost.** Same for
> the 1017 the user observed on `55321893`, now 934.7.
>
> ## 🔴 2. THE TRAINING SEED IS WORTH ~50 Elo AND IT IS PURE NUISANCE
>
> Shipped config (`--no-rules` both arms), mirror, direct, n=1,400/cell, all vs
> `policy_v5_s1`. Same corpus, architecture, hyperparameters, epochs — **only
> `--seed` differs**:
>
> | net | score | 95% CI | Elo |
> |---|---|---|---|
> | `policy_v5_s2` | **0.537** | [0.511, 0.563] | **+25.8** |
> | `policy_v5_s3` | **0.465** | [0.439, 0.491] | **−24.4** |
>
> **A 50 Elo spread from the seed alone**, against the v4 state block's **+37**
> (systematic feature enumeration, the project's second-largest confirmed win)
> and v5's **+14**. ⚡ Independently reproduces §8be's incidental 0.073 swing.
> ⇒ **every 2-seed A/B this project ran measured its intervention on top of a
> ±25 Elo nuisance term.** §8aw's warning was understated, not overstated.
>
> ## 🔴 3. ENSEMBLING FAILED TO BEAT ITS OWN BEST MEMBER TWICE
>
> Both in the shipped configuration, both null:
> **ens2 vs `v5_s1` 0.505** [0.479, 0.531] · **ens5 vs `v5_s1` 0.522** [0.496,
> 0.548]. ⇒ **a vote is bounded by its members; mediocre nets do not average
> into a good one.** The live `55321893` bundle's whole gain over the old shipped
> net was the **seed swap** (0.531), not the vote.
> ⚠ **NOT a refutation of §8be** — its cells were rules-ON and the intervals
> overlap. The claim is *unproven where it counts*, not disproven (rule 4).
> 🔴 **And §8be's weighted-anchor table, which CHOSE ens2 over the seedswap, is
> all rules-on. No weighted verdict in it describes the shipped agent.**
>
> ## ✅ 4. TWO INSTRUMENT FACTS, both cheap and both settled
>
> - **`scripts/p63_net_agreement.py`** is the decorrelation gate. Reproduces
>   §8be's numbers independently over 12,939 held-out decisions (`v5c_s1` vs
>   `v5_s1` **100.0%** exact; `v5` vs `v5c_s0` **88.4%** vs the published 87.5%).
>   🔴 **All ten fresh-seed pairs sit in 80.4–81.4%** ⇒ **seed variation cannot
>   produce correlated members; mixing RECIPES does.** §8be's diagnosis was right
>   and its implied cause was wrong. ⚠ Gate default 85% is a **midpoint guess**;
>   HANDOFF's old "~90%" would have waved through the pair that made ens3 lose.
> - **Latency is a non-issue, measured not assumed:** ens5 runs **6.7 ms/move
>   mean, 34.4 ms max, 598.8 s of 600 s pool unspent.** Member count is not
>   cost-constrained at any size worth considering.
>
> ## ⚠ 5. THE OUTSTANDING DEBT — read before quoting 0.537
>
> **`s2` won a screen of THREE seeds, so 0.537 is inflated by selection.** It
> survives Bonferroni (p≈0.016) so it is a caveat, not a retraction — but the
> honest estimate is **below** 0.537 and **a confirmation run on fresh games does
> not exist yet.** That is day 26's cheapest real task (~12 min).
> ⛔ **DO NOT vote only the screen winners** — that compounds the same bias.
> 🟡 **`policy_v5_s4` was trained and never screened.**
>
> ## ▶ THE DAY-26 PLAN — ⚠ SUPERSEDED by the FINAL PUSH box above (items 1–3 are folded into F2 and the carryover line; item 4's "the ladder cannot show it" is answered in the frame: continued play and the Round-2 BO3 can)
>
> 1. ⏱ **Read the LB ≥1 h after 14:05 UTC** for `55326513`, and take a second
>    read ≥1 h later (rule 2). ⚠ A rising score is unconverged, not momentum.
> 2. 🔬 **Confirm `s2` on fresh games** (~12 min) — pays the debt in item 5.
> 3. 🔬 **Screen `s4`** (~12 min). Cheap, and it widens the seed distribution
>    from 4 points to 5.
> 4. ⚠ **The honest strategic read: none of this is an LB lever.** §8ak's
>    decision-identical pair reads **63–87 points apart**; our whole candidate
>    spread is ~50 Elo. **The board cannot resolve any of it.** Both slots are
>    now filled with drawn-fresh agents, which is the one thing that WAS worth
>    doing. ⇒ **further seed-hunting buys a number the ladder cannot show.**
> 5. ⛔ **Do NOT reopen a closed axis** (fifteen — ensembling joins the list).
>    ⛔ **Do not push without asking** — 30+ commits are local by design.
>
> ---
>
> # ▶ DAY 24 (2026-08-07): ENSEMBLING WORKS, AND WE HAD BEEN SHIPPING THE WEAKER OF TWO NETS WE ALREADY OWNED
>
> ⚠ **SUPERSEDED IN PART BY DAY 25 (above): re-measured in the shipped
> configuration, the vote does NOT beat its better member and the gain is the
> seed swap. Item 5's "⛔ NOT SUBMITTED" is also stale — see the standing box.**
>
> **📌 USER DIRECTIVE, STANDING UNTIL THEY LIFT IT: report work is SUSPENDED.
> Track A only — the simulation leaderboard and a stronger agent.** `STRATEGY.md`
> gets no edits. `EVIDENCE.md` keeps getting entries because it is what stops the
> next session re-running an experiment; that is engineering, not report writing.
>
> Full record: `EVIDENCE` §8be. Pre-registration: `docs/experiments/E9-ensemble.md`
> (frozen before any cell, including two predictions that were wrong).
>
> ## 🔴 1. THE SHIPPED NET IS THE WEAKER SEED
>
> `out/policy_v5_s1.npz` has been sitting in `out/` since 08-01 and **beats the
> shipped `out/policy_v5.npz` 0.549 [0.527, 0.571]**, n=2,000 mirror direct
> (arm C read 0.451 from the incumbent's side). ≈ +34 Elo, free, one file.
> ⇒ **Every A/B this project ever ran "against v5" used the weaker of two
> available nets.** I predicted this arm would be a null. It is not.
>
> ## ⚡ 2. A 2-NET VOTE BEATS BOTH MEMBERS — the largest confirmed gain since the v4 state block
>
> `Ensemble` (softmax each member over the option set, then average — **not** a
> raw-logit mean, because a listwise loss pins ranking and not scale). Spec:
> `bc:<label>,net=a.npz+b.npz`. Mirror, direct, fixed weight files, n=2,000:
>
> | cell | score | 95% CI |
> |---|---|---|
> | ens2 vs `policy_v5` (shipped) | **0.541** | [0.519, 0.563] |
> | ens2 vs `policy_v5_s1` (better member) | **0.531** | [0.510, 0.553] |
>
> **Weighted anchor confirmation, 90.6% of the field, n=1,500/cell, one session:**
> **ens2 ΔW = +0.0289 ± 0.0115, 2.5× outside, positive on 6 of 7 anchors.**
> The seed swap alone is **+0.0215**, 1.9× outside, positive on only 4 of 7.
> ⇒ **ens2 is the candidate.**
>
> ## 🔴 3. MORE MEMBERS IS NOT BETTER — and the reason generalises
>
> Four v5-recipe nets are on disk but there are only **three policies**:
> `policy_v5c_s1` is **100.0% decision-identical** to `policy_v5_s1` (different
> md5, same function). And the honest 3-net vote **lost anyway** (0.491 vs the
> best member): `policy_v5c_s0` agrees with `policy_v5` on **87.5%**, so ens3 is
> effectively two votes for the v5-ish policy against one for the stronger `s1`.
> ⇒ **Members must be DECORRELATED. 87.5% agreement is already enough to hurt.**
> `build_submission.py` refuses byte-identical members; it cannot detect the
> 87.5% case, so **check agreement before adding a member** (the one-off script
> pattern is in this box's item 6).
>
> ## 🔴 4. THE CONFIGURATION TRAP — READ THIS BEFORE TRUSTING ANY OLD ΔW
>
> Arena `bc` defaults to `chip_targeting`/`energy_spread`/`counter_source` **ON**.
> The submission builds with `--no-rules`, all three **OFF** (§8f: those rules
> measure 0.427 against a v3-optfeat net). **Every E9 cell ran rules-ON, both arms
> alike** — internally valid, but a delta between two rules-*on* agents, while the
> bundle is rules-*off*. ⚠ **This likely affects earlier verdicts too (§8aa, §8z):
> the shipped configuration may never have been the measured one.** Re-run in the
> shipped config: `out/arena/p62_ship_config.jsonl`.
>
> ## 📦 5. THE BUNDLE IS BUILT AND VERIFIED — ⛔ NOT SUBMITTED
>
> ```powershell
> python -X utf8 scripts/build_submission.py --agent bc --deck grimmsnarl \
>     --nets policy --no-rules --policy-net out/policy_v5.npz \
>     --ensemble-net out/policy_v5_s1.npz
> ```
> `dist/submission.tar.gz`, 6.9 MiB. Smoke on the **extracted** bundle, loaded the
> way Kaggle loads it: `MEMBERS=2 want=2`, 2 distinct member md5s, both pass the
> dim guard, `RESULT=0 turns=17 lat_max=0.01s pool_left=599.9s`.
> ✅ **Fail-soft is preserved**: `main.py` tries the vote and on ANY exception
> falls back to the single bundled net (which is member 0), logging loudly. An
> explicit `net=` is strict *by design* since day 22 — strict means it raises, and
> raising on the shipped path would forfeit a live episode.
>
> ## 🎲 6. IF YOU SUBMIT: SUBMIT TWICE, NOT ONCE
>
> The board shows your **best ACTIVE** submission and only the **latest 2** are
> active. Today's two are *the same v5 policy* and they landed **990.7 and 904.1**
> — so the 990.7 is a lucky draw, not v5's level (mean ≈ 947, and §8ak now has the
> decision-identical gap at **86.6**). Submitting once evicts the 990.7 and leaves
> one fresh draw beside the stale 904.1. **Submitting twice replaces both slots
> with two draws of the better agent and the board takes the max.** Expected
> displayed score ≈ 1015 vs the 990.7 held today — but the spread is wide enough
> to land lower, so it IS a gamble against a settled number.
> ⛔ **Do not submit once.** It is dominated by both alternatives (submit twice, or
> don't submit).
>
> ## ▶ THE DAY-25 PLAN — the retrain, which is the obvious next move
>
> 1. 🔬 **Train 3 more v5-recipe seeds (~2 h unattended), then vote across the
>    decorrelated set.** This is the direct consequence of item 3: the vote is
>    limited by having two independent members, and correlated ones hurt.
>    ```powershell
>    python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 \
>        --bs 1024 --loss listwise --state-h 512,256 --head-h 256,128 --pool \
>        --opt-cols 37 --seed 2 --out out/policy_v5_s2.npz
>    ```
>    ⚠ **`--opt-cols 37` is mandatory** — `optfeat.OPT_DENSE` grew 37 → 46 on the
>    merge, and anything sharing an ensemble with `policy_v5` must match its
>    layout. Repeat for `--seed 3`, `--seed 4`.
>    ⚠ **Check pairwise agreement before adding any member** and drop anything
>    above ~90% agreement with a member already in the vote.
> 2. 🔬 **Then re-confirm in the SHIPPED configuration** (`noChip,noSpread,noSrc`),
>    not the arena default — item 4. Do the mirror first; it is 32% of the field
>    and the tightest instrument we own.
> 3. ⚠ **Inference cost is not free forever.** 2 members = 0.01 s/move against a
>    600 s pool, so 5 is still nothing — but the smoke prints `lat_max`, so read
>    it rather than assuming.
> 4. ⛔ **Do NOT reopen a closed axis** (fourteen). ⛔ **Do not push without
>    asking** — 30+ commits are local by design.
> 5. 🟡 **Still open, unresolved:** `bc:garchomp` read **0.641** where §8ap
>    recorded **0.857**, and its fingerprint says it is piloted by the stale
>    width-496 `lw2` singleton. The E9 *deltas* are unaffected (all three arms met
>    that identical build back-to-back) but the level is unexplained — possible
>    sixth instance of anchor drift.
>
> ---
>
> # ▶ DAY 23 (2026-08-07)
>
> ## 📈 STANDING FIRST: RANK 129 / 6,483 AT 990.7 — BEST EVER, AND THE ANSWER TO "SHOULD WE SUBMIT" IS NO
>
> Read 2026-08-07 via `competition_leaderboard_download` (one call, all 6,483
> rows). Previous bests were 185/6,103 at 955.1 and 198/6,136 at 942.7.
>
> | active? | submission | score | what it is |
> |---|---|---|---|
> | ✅ **active** | `55160229` (08-01 10:39) | **990.7** | **v5 — the pooled option-set net. This is the score the board shows.** |
> | ✅ **active** | `55169114` (08-01 18:42) | 904.1 | **decision-identical to v5** (health logging only) |
> | ⛔ inactive | `55156480` | 910.5 | v4 state block |
> | ⛔ inactive | `55072063` | 952.0 | the frozen 07-29 reading, ~4,000-entrant board (§8p) |
>
> 🔴 **ANY NEW SUBMISSION EVICTS `55160229`, THE 990.7.** Eviction is by recency
> and only the latest 2 are active, so a third submission drops the *older* active
> one — which is our best-scoring agent and the one displaying our rank. **Day
> 13's standing rule ("name the agent a submission would evict before quoting the
> bar") is now answered with a name and a number.**
> ⇒ ⛔ **Do not submit.** Nothing in the repo measures better than v5 in the arena;
> v7 is an unresolved null with the point estimate on the wrong side (−0.0078);
> and the ladder cannot adjudicate any of it (below). **There is no candidate
> whose expected gain exceeds the certain cost of evicting a 990.7.**
> ⚠ The calendar's "last safe day to submit ~08-15" is therefore moot for us —
> **the freeze has effectively already started, by arithmetic rather than by date.**
>
> ⚡ **AND THE PAIR ITSELF PRODUCED A RESULT: the decision-identical gap WIDENED to
> 86.6** (990.7 vs 904.1). §8ak hedged that its 63.2 was *"a lower bound observed
> at one moment"* because four reads had been shrinking (81.7 → 76.2 → 69.0 →
> 63.2) and might converge. **Six days on, they diverged instead.** Two agents whose
> true difference is *exactly zero* have now been observed 63–87 points apart.
> ⚠ **RULE 2: one reading.** Logged in `EVIDENCE` §8ak, deliberately **not** yet
> promoted into `STRATEGY` §5.1b (which still cites the settled 63.2). **Taking the
> confirming read ≥1 h apart is day 24's first five minutes.**
>
> ---
>
> # ▶ DAY 23: THE E3 GATE RAN, AND IT RETRACTED A DIFFERENT EXPERIMENT
>
> Full record: `EVIDENCE` §8bd; pre-registration
> `docs/experiments/beyond-bc/E3b-near-tie-gate.md` (frozen in `675d09c`, before
> any arm ran, **including the two predictions that turned out wrong**).
>
> 🔴 **1. THE NEAR-TIE BAND E3 TARGETS IS INDIFFERENT.** `bc,flip<τ>` swaps the
> clone's *k*-th choice for its (*k*+1)-th whenever their logit gap is under τ.
> The 160-item review queue sits entirely inside the τ=0.10 band; flipping **every
> decision in that band** — 7.0% of all decisions, far more aggressive than
> relabelling 160 — reads **0.494 [0.467, 0.520]**, n=1,400. At τ=0.50 (21.8% of
> decisions) still null at **0.487**. ⇒ **No systematic re-ranking of the band
> pays.** ⚡ **This is the only experiment here with no training-seed term**: both
> arms load the same weight file, `#dc1c9acc` on both sides of every row.
> ⚠ **E3 IS NOT KILLED, and the day-23 plan's claim that a null would kill it is
> WITHDRAWN — by the pre-registration, before the data.** The flip measures
> |E[effect]|; a teacher's value is E[|effect|]. §8am's own reading says an
> indifferent-on-average band is exactly where a sorter has room.
>
> 🔴 **2. THE BIGGER RESULT IS A RETRACTION OF §8am.** Two arms MISSED their
> pre-registered predictions (τ=1.00 predicted ≲0.40, read **0.455**; τ=2.00
> predicted ≲0.20, read **0.356**) — both predictions were §8am's, matched on
> deviation *rate*. At matched rate: ~31–35% deviated, **0.455 [0.429, 0.481]**
> here vs §8am's **0.315 [0.255, 0.382]**; ~44–51% deviated, **0.356 [0.332,
> 0.382]** vs **0.055 [0.031, 0.096]**. **Disjoint.** ⇒ **A softmax temperature
> raises deviation RATE and DEPTH together; §8am credited the rate.** This probe
> pins depth at exactly one rank and finds **no cliff at all**: 0.495 → 0.494 →
> 0.487 → 0.455 → 0.356, monotone 5/5. ✅ τ=0.5 and B8's exploration budget are
> untouched; what changes is what the free band *means*.
> ⚡ **Third instance in three sessions of "the effect was credited to the
> variable we named"** (§8ax's deck, §8bb's compute, now this) — and the **first
> caught by a designed experiment rather than an audit afterwards**. Written up as
> `STRATEGY` §5.8, with the defence: read the *realized* value of the thing you
> claim varied, from the run's own logs.
>
> ⚠ **NOT separated:** the flip differs from softmax in depth **and** targeting.
> One more sweep (sample from the top-2 only, at matched rate) would apportion
> them. **Not run.**
>
> ### 📝 The report moved a lot this session
> `STRATEGY` §5.7 (the day-22 audit as its own chapter — seven defects, why five
> could not have moved a number), §5.8 (above), a **B8 chapter that did not
> exist**, an E3 entry, and §2's summary table corrected — it read *"self-play RL:
> 0 — never run"*, true of league self-play and hiding a 20,000-game measured
> null. `EVIDENCE` gained §8bc (the five audit defects that had only ever been in
> this file) and §8bd.
> 🔴 **And a propagation failure was found and fixed:** day 22 corrected E8's
> weighted figure **−0.0099 → −0.0078** and the correction reached **two** places
> while **six** files kept the old number — including `STRATEGY` §4g contradicting
> its own §8. Same for §8aq's retracted headline (still live in `ROADMAP`'s
> calendar) and p37's ΔW. **ROADMAP's own rule — grep the claim across all four
> files in the same commit — is the one that keeps getting skipped.**
>
> # ▶ THE DAY-24 PLAN — AND THE HONEST ANSWER TO "WHAT IS LEFT TO EXPLORE"
>
> **Nothing on this list is an Elo lever, and that is not a gap in the list.**
> Model Score is 70% across five bullets with the leaderboard as *one* of them
> (~14% of the grade); Deck Score is 20%. **~76% of the grade is analysis and
> writing, and the Round-2 gate (§3.5) is the Strategy Category, not the ladder.**
> Ranked by value per hour:
>
> 1. ⏱ **FIVE MINUTES, DO IT FIRST: the rule-2 confirming read** of the
>    decision-identical pair (above). If 86.6 holds, `STRATEGY` §5.1b gets a
>    stronger headline *and* loses a hedge that turned out to be wrong. That is a
>    rubric-scoring correction for the price of one API call.
> 2. 📝 **THE REPORT IS THE WORK.** Two chapters have never been re-read against
>    what days 19–23 measured, and both are load-bearing:
>    **§7 (opponent modelling)** predates §8ay's weight correction, so its shares
>    may be the broken-classifier ones; **§1 (approach and rationale)** was written
>    when the feature axis was still paying and now has to account for five
>    generations ending +115 → +37 → +14 → 0 → 0. ⚠ Check both against `EVIDENCE`
>    rather than trusting the prose — that exact check found the stale −0.0099 in
>    six files today.
> 3. 🔬 **Two ~25-minute runs that FIRM UP PUBLISHED RETRACTIONS, not new axes.**
>    Both are optional and neither can change a verdict:
>    (a) **separate depth from targeting** in §8bd — sample from the top-2 only at
>    matched rate; right now the retraction of §8am's cliff bounds the two
>    mechanisms jointly and apportions neither;
>    (b) **confirm §8ax's additivity assumption** — run the v1/v2/v3 Crustle
>    pilots on the *other* deck (restore from `b7869d2` / `83daa48`). The
>    retraction does not depend on it; the *sizes* do.
> 4. 🟢 **THE ONLY GENUINELY OPEN THREAD, and its prior is poor — say so before
>    starting.** Every axis is closed except one: the corpus contains **zero**
>    Lucario games and 40.7% of the field is under-represented >3×
>    (`PARKED-corpus-coverage.md`, §8au). The **cheap probe is a sizing gate, not
>    a build**: do the replays we already hold contain the missing archetypes at
>    all? ⚠ **State the prior first:** E7 and E8 both tried to fix an unseen
>    archetype by re-encoding cards and both failed, and the "more data" axis is
>    0 for 5. This is worth doing because it *closes* the last green thread with a
>    measurement and becomes a chapter — **not** because it is likely to pay.
> 5. ⛔ **Do NOT reopen a closed axis. Thirteen:** search, data, demonstrators,
>    capacity, sequencing, RL, deck, encoding, embeddings, multitask, routing,
>    planning, and now the near-tie band. ⚠ **E3 is "unresolved for want of a
>    teacher", which is NOT the same as open** — §8bd measured that no *systematic*
>    re-ranking of its band pays, and a teacher's value (E[|effect|]) is simply not
>    measurable with what we have. **Do not build a teacher.**
> 6. ⛔ **Do NOT submit** (see the standing box — it evicts a 990.7), and **do not
>    push without asking**: 29 commits are local by design.
>
> #### 🟡 Open judgement calls, none urgent, all the user's
>
> - **Ship v7 or keep v5?** Unresolved by design since day 21. Default: v5 ships —
>   and the eviction arithmetic above now makes that near-decisive.
> - **Which Crustle cell belongs in the anchor table?** Neither `@crustle_v1`
>   (0.755, strong pilot / wrong deck) nor `@crustle` (0.893, weak pilot / right
>   deck) is "the field's Crustle played well". A real fix is a pilot for the
>   consensus list. No verdict depends on it.
> - **Keep `experiments/beyond-bc` tracking `main`, or reset it to `eff71fd`** as a
>   historical marker?
>
> ---
>
> # ▶ DAY 22: THE VALIDATION FLOW WAS AUDITED AND SIX THINGS WERE WRONG (2026-08-06)
>
> Full record: `EVIDENCE` §8ax; the tool changes are in `scripts/arena.py`,
> `agents/sa/bcagent.py`, `scripts/p56_e7_arena.py`, `scripts/p57_e8_arena.py`.
> **No net-vs-net verdict in this repo changes.** Every published difference ran
> both arms back-to-back against one instrument, and that is what saved them.
>
> 🔴 **1. THE CRUSTLE ANCHOR CHANGED DECK AS WELL AS PILOT, AND THE DECK IS THE
> BIGGER TERM.** `rule:crustle` ran on `crustle_v1` in p10/p19/p20/p34/p35/p37
> and on `crustle` in p27/p28/p54/p56/p57 — **20 of 60 slots different**,
> archived under one identity. Measured with the pilot held fixed at the repo's
> v4, n=2,000/cell: **0.753 [0.734, 0.772] vs 0.893 [0.879, 0.906] = +0.140**
> against a ±0.031 resolution. ⇒ **§8an's "+0.09 from the empty-bench guard" and
> §8aq's "−0.111 from the Dwebble tie-break" both straddle a deck swap and
> neither isolates the pilot.** Same-deck, every pilot term is **≤0.027**.
> ⚡ **§8ah's originally expected sign is restored** — the repair made the pilot
> *stronger* (≈ −0.04 to us), which is what §8ah predicted and §8an reported as a
> surprise. 🔴 **§8aq's "WHICH Pokémon it benches matters more than WHETHER" is
> retracted.** ⇒ **RULE 20.**
>
> 🔴 **2. `arena.py elo` WAS NUMERICALLY DIVERGENT FOR FIFTEEN DAYS.** Fixed
> `lr=8.0` on an unnormalised batch gradient: past ~175 games per player the
> iteration oscillates. `rule:crustle` swung **8,586 Elo** between consecutive
> passes and read −3632 / +258 / +3397 / −3275 at iterations 499/500/501/502.
> **Every rating it ever printed was an arbitrary sample of an oscillation.**
> Nothing published rests on it — every Elo figure in `EVIDENCE` is a win-rate
> conversion — **which is exactly why it survived: an unused instrument is never
> checked.** Now a damped diagonal-Newton fit that converges to 1e-4 and
> reproduces the bc-vs-crustle head-to-head **0.652 → 0.652**; it refuses to
> print an unconverged fit, and flags the **12 agents that never played a game
> connected to the anchor** (their level is prior, not evidence).
>
> 🔴 **3. A `net=` THAT FAILED THE LOAD GUARD SILENTLY PLAYED `sa/policy_net.npz`.**
> `policynet.load` returns None rather than raising, and `__call__` falls back to
> the singleton — the old width-496 `policy_lw2`. Demonstrated with a v7 net
> whose vocab map was one entry short (§8aw's exact "stale map" hazard): accepted
> by `build_agent`, archived under the requested net's name, would have played
> 496-wide lw2 against a 708-wide control and printed an ordinary score.
> ✅ **All 32 nets on disk load, so no past result is affected.** Now refuses.
>
> 🟠 **4. THE DEGRADATION COUNTERS WERE WIRED ONLY INTO THE SUBMISSION.** Day 15
> built `bcagent.STATS` + `health_line()` to catch the index-order fallback and
> called it *"the highest value-per-byte thing to log"* — then wired it into
> Kaggle's `main.py` and **not into the arena**, the instrument day 17 calls *"the
> ONLY instrument"*. Worse, `p57` ran arena with `capture_output=True` and printed
> stderr **only on non-zero exit**, so the tracebacks were discarded on every
> successful run. `arena.py play` now prints `[health]`; p56/p57/p58 surface
> stderr on success and **hard-stop on DEGRADED**.
>
> 🟠 **5. `bc` WITH NO `net=` IS AN UNVERSIONED IDENTITY** — 1,226 games in
> `games.jsonl` under the bare name `bc`, spanning 07-28 → 07-31, across which
> `sa/policy_net.npz` was a moving target. Rule 19 one seat over. Agent names now
> carry `#<md5-8>` **of the weight bytes**, so a retrain that reuses a path
> archives as a new agent instead of pooling. (Shipped v5 fingerprints
> `#dc1c9acc`, matching the bundle md5 already recorded below.)
>
> 🟡 **6. ARCHIVES APPEND AND A RE-RUN WAS INVISIBLE.** `p57_e8.jsonl` holds
> **3,000 games per v5c control cell against 1,500 per treatment cell** — the
> control was re-run for the v7pad pass into the same file. Published numbers are
> safe (drivers parse the printed score line), but re-deriving from that archive
> gives a control that was never the published one. Rows now carry `run`
> (schema 2) and `play` announces when the target already holds that exact cell.
>
> 🔴 **7. AND THEN THE WEIGHTING LAYER — the `w` in every `W = Σ wᵢΔᵢ` headline**
> (§8ay). `p9_field_census.py` keyed evolution lines by card **id**, but
> `evolvesFrom` is a **name** and a name has many printings: **228 broken links
> over 106 basic printings**, so one archetype split by which reprint the
> opponent happened to draw. **Riolu #677 and #974 both lost Mega Lucario ex**,
> and `rule:v10`'s share is a published weight. Fixing it exposed a second bug it
> had masked (`ex` outranked copy count, so a 2-of tech beat a 4/3/3 engine).
> Hand-checked against all 75 games: **69/75 → 74/75 correct**. Shares: mirror
> 33.3 → **32.0**, alakazam5 22.0 → **25.3**, crustle 6.7 → **8.0**, v10 4.0 →
> **5.3**.
>
> ⚡ **AND THE BUG IS NOT THE PROBLEM — n=75 IS.** The mirror weight, which
> carries a third of every weighted verdict, has a 95% interval of **[22.5%,
> 43.2%]**. *Every correction above sits inside the interval of the estimate it
> corrects.* Bootstrapped through: weight uncertainty adds **±0.0031** to p37's
> ΔW and **±0.0023** to E8's. ⇒ **Weight error bites in proportion to how much
> the per-anchor deltas DIFFER.** p37's quoted ±0.0050 treats the weights as
> exact; honestly combined it is **±0.0059**. For E8 the ±0.025 game noise
> swamps it — **the weighting layer is not E8's problem, the 2-seed budget is.**
> ✅ **No verdict changes** (p37 −0.0140 → −0.0155, still 2.6× outside its kill
> line, negative in 100% of bootstraps).
> 🔴 **Separately, E8's −0.0099 was an arithmetic error: it is −0.0078.** Its own
> table's mirror row is a **score** in a column of **Δ**s; 0.487 − 0.5 = −0.0128,
> confirmed from the archive at 0.4872 over 3,000 games. Still a null.
>
> ✅ **8. THE BRANCH IS RECONCILED.** `experiments/beyond-bc` merged into `main`
> (`ba32c45`); both are the same commit. **No beyond-BC experiment needed
> re-running** — every run there is mirror-direct or vs `rule:alakazam5` on its
> own deck, so it has zero exposure to §8ax or §8ay. Two corrections landed:
> **E5's "compute curve" never scaled** (realized compute 652→616→606 s across
> the three arms that opened the gate; the real dose-response is on the planner's
> FIRING RATE, monotone 4/4) and **E2's Alakazam arm is uninformative, not null**
> (−0.010 against ±0.036). Sections renumbered **8au/8av/8aw → 8az/8ba/8bb** to
> clear a collision with E6/E7/E8.
> 🔴 **Two integration breaks the smokes caught, neither of them a git conflict:**
> `p45_dagger_export.py` was writing unloadable shards (`main`'s `Writer` gained
> an `attr` column it never passed), and **`optfeat.OPT_DENSE` grew 37 → 46**, so
> anything warm-starting from `policy_v5.npz` must now pin `--opt-cols 37`.
> ⚠ **`git` merged `EVIDENCE.md` and `STRATEGY.md` CLEANLY and silently produced
> duplicates** — six sections under three numbers, and a duplicated E1 entry in
> STRATEGY §8. The conflicts were the easy part. Check prose after a doc merge.
>
> ✅ **9. THE REPORT: §6 AND §7c ARE WRITTEN.** `STRATEGY.md` now has **no
> section marked in progress**.
> **§6 Robustness and consistency** answers two Model Score bullets that had no
> answer: consistency is measured (dispersion **0.984 ± 0.023** over 163 cells /
> 3,811 blocks — the arena's intervals are honest), matchup-dependence is stated
> with its own unflattering finding (in variance terms the mirror *is* the anchor
> set; 45.3% of every weighted verdict sits near the ceiling), and §6.3 lists
> **four separate ways the anchor table has been wrong**.
> **§7c The deck** gained the thing Deck Score literally asks for and did not
> have: the concept, the key cards, and the game plan — Punk Up welding energy
> acceleration to the evolution, Adrena-Brain repairing the wall while adding
> reach, Froslass chipping through it. ⚡ **And the alignment has a twist worth
> keeping:** the three hand-written rules that encode this deck's arithmetic
> **ship turned OFF**, because once the encoding carried the same information as
> features they measured 0.427. *The representation won and made the rules
> redundant* — the project's thesis told through the deck. §7c.4 adds the
> pre-registered 11-variant search that the old text said we never built.
>
> ⛔ **NOT FIXABLE, so nobody should propose it:** the arena has no common random
> numbers. `cg.game.battle_start` takes only the two decklists, `StartData`
> carries no seed, and the RNG lives inside `cg.dll` — verified, two fresh
> processes running an identical 5-game script diverge. **`evaluate_paired` is
> seat-balanced, not variance-reduced.** The ±0.036/cell floor is structural;
> more games or more seeds are the only levers.
>
> # ⛔ SUPERSEDED — THE DAY-23 PLAN (executed; see the day-24 plan above). Item 2's "a null kills E3" claim is WITHDRAWN by §8bd.
>
> 🔴 **Read the rubric before planning anything.** Model Score is **70% across
> five bullets**, and *"performance within the competition track"* is **one of
> them** — so the leaderboard is **~14% of the grade**. Deck Score is 20%.
> **Roughly three quarters of the grade is analysis and writing.** §8ak already
> measured that the LB cannot resolve any change we would make (63.2-point floor
> against a largest-ever measured effect of +40). ⇒ **Stop buying Elo we cannot
> measure with 14% of the marks, and finish the 76%.** The only date that still
> matters is the report deadline, **2026-09-14**; we are not shipping a new agent,
> so **08-17 is not a deadline for us**.
>
> 1. 📝 **STRATEGY.md is the work.** §6 and §7c are done and nothing is marked
>    *in progress*. What is thin: **§8's newer entries** (E6/E7/E8 and the E2/E5
>    corrections just landed — check each against `EVIDENCE` rather than
>    trusting the prose), and there is still no chapter on the **day-22
>    validation audit as a story in its own right** — six defects found in our
>    own instrument, quantified, fixed, **and no verdict changed**. That is the
>    single strongest "technically sound" exhibit this project owns.
> 2. 🔬 **The one experiment still worth running: the teacher-free E3 gate**
>    (~1 h). E3 is parked on "no qualified reviewer", but *"is there anything to
>    learn in the near-ties?"* needs no teacher. On decisions with boundary
>    margin < τ, take the other boundary option; A/B vs v5 in the mirror at
>    n=2,000. **A null kills E3 without a reviewer and is a good report chapter**
>    ("we built a DAgger pipeline, then proved the states it targets are
>    indifferent"). ⚠ A loss does **not** prove a teacher would help — say so.
>    Rule 14 first: dump the margin histogram over all 8,963 candidates
>    (`p43` already computes them) so you know what fraction τ covers.
> 3. ⛔ **Do NOT reopen a closed axis.** Search, data, demonstrators, capacity,
>    sequencing, RL, deck, encoding, embeddings, multitask, routing, planning.
>    Twelve. The absence of leads is not a problem to solve — **it is the best
>    Model Score narrative we have**, and the rubric explicitly rewards it.
> 4. ⛔ **Do NOT submit**, and do not push without asking (20+ commits are local
>    by design).
>
> #### 🟡 Open judgement calls, none of them urgent, all of them the user's
>
> - **Ship v7 or keep v5?** Unresolved by design (day 21). Default: v5 ships.
> - **Which Crustle cell belongs in the anchor table?** The field plays ~the
>   consensus list (47% card overlap vs 27% for `crustle_v1`), but the anchor
>   set pairs the pilot with `crustle_v1`. `@crustle_v1` is a strong pilot on the
>   wrong deck (0.755); `@crustle` is a weak pilot on the right one (0.893).
>   **Neither is "the field's Crustle played well"** — a real fix is a pilot for
>   the consensus list, not a choice between these two.
> - **Confirm §8ax's additivity assumption?** ~20 min: run the v1/v2/v3 pilots
>   on the *other* deck. The retraction does not depend on it; the sizes do.
> - **Keep `experiments/beyond-bc` pointing at `main`, or reset it to `eff71fd`**
>   as a historical marker? It currently tracks `main`.
>
> ---
>
> # ▶ DAY 21: THE EMBEDDING COMPONENT IS SPENT (2026-08-06)
>
> Full record: `docs/experiments/embeddings/E8-vocab-remap.md`, `EVIDENCE` §8aw.
>
> ✅ **THE FIX IS KEPT AND SUPPORTED — this is a user decision, not a leftover.**
> *"I would have liked embeddings to be fixed regardless of their impact on
> Elo."* Shipping 88,000 parameters of which 92% are untrained noise is
> indefensible on its own terms whatever the scoreboard says. So:
>
> ```
> python -X utf8 scripts/p53_emb_vocab.py --pds artifacts/pds_v4
> python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 \
>     --bs 1024 --loss listwise --state-h 512,256 --head-h 256,128 --pool \
>     --opt-cols 37 --seed 0 --vocab out/emb/vocab.json --out out/policy_v7_s0.npz
> ```
>
> `--vocab` implies `--pad`; `--pad` alone is the isolated padding fix. The map
> ships inside the npz as `vocab_<table>` and `policynet.load` refuses any net
> whose row count ≠ `2 + len(vocab)`. ⛔ **Two scripts feed RAW ids straight to
> the tables and both now refuse a v7 net by name** — `context_accuracy.py` and
> `p54_emb_ablate.py`. If you add a third consumer, guard it or map through
> `vocab_<table>` first. ⚠ **The vocabulary is corpus-derived**: rebuild the
> corpus and a net's map is stale — that means retraining, not remapping.
>
> 🟡 **Shipping v7 is an open judgement call, deliberately not taken.** Default
> is **v5 keeps shipping** — v7's −0.0078 is not resolved as a loss (every arm
> spans zero) but the point estimate is on the wrong side, and the LB's
> 63.2-point floor cannot adjudicate it. Correctness gain real, strength gain
> measured zero. To change it, name the agent the submission would evict first.
>
> ⛔ **DO NOT REOPEN THE EMBEDDING TABLES** *(as a source of Elo — the code
> stands).* Three experiments, three nulls.
> E6 measured that identity carries a quarter of the win rate; E7 tried to
> recover it from card attributes (null); E8 fixed the two *real* defects — 90%
> of rows shipping untrained, and row 0 overloaded across 25.5% of lookups — and
> both fixes measure **weighted −0.0078 (v7) and −0.0047 (v7pad)** over 74% and
> 44% of the field. **A real defect is not a lever.** Nets kept at
> `out/policy_v7_s{0,1}.npz` / `out/policy_v7pad_s{0,1}.npz`; **v5 still ships**.
>
> ⚡ **CAPACITY IS BOUNDED FROM BOTH DIRECTIONS NOW.** 88,000 → 6,960 embedding
> parameters (−92.1%, 11.5% of the whole net) costs **0.0018 / 0.0003** of
> corpus fit and moves no anchor. With §8w (8.2× params, −43 decisions):
> **nothing in this project has ever been capacity-limited.** Stop proposing
> bigger or smaller nets.
>
> 🔴 **TWO SEEDS × 1,500 GAMES UNDER-RESOLVES EVERY ANCHOR — INCLUDING THE
> MIRROR.** Day 20 warned §8z's ±0.019 floor was mirror-direct only. E8 found
> the *direct mirror* arm swinging **0.073** between seeds against ±0.036
> sampling, and archaludon swinging **0.091** against ±0.051. ⇒ **budget ≥3
> seeds, or stop quoting anchor deltas under ~0.05.** This does not retract §8z
> or §8aa (n=2,000, replicated) but their intervals were optimistic.
>
> ⛔ **A two-cell delta's resolution is √2× a single cell's.** The E8 driver
> shipped printing one cell's width and understated it by 41%. At n=1500/cell:
> **±0.036 per seed, ±0.025 pooled**. Arm A (direct) is √2× tighter.
>
> ⛔ **RULE 1 APPLIES TO PATTERNS ACROSS ARMS, NOT ONLY TO SINGLE ARMS.** E8
> published a monotone dose-response across four arms built from single-seed
> cells each inside its own interval; the next data point destroyed it. Knowing
> the trap and restating it in the same message did not prevent it.
>
> 🟢 **The untested lever is still DATA, not encoding** (§8au): the corpus has
> **zero** Lucario games and 40.7% of the field is under-represented >3×
> (`PARKED-corpus-coverage.md`). E7 and E8 both tried to fix an unseen archetype
> by re-encoding cards. Neither could. **The cheap probe is a sizing gate, not a
> build:** do the replays we hold contain the missing archetypes at all?
>
> ---
>
> # ▶ DAY 20: THE EMBEDDING VOCABULARY IS THE BLIND SPOT (2026-08-04/05)
>
> Full records: `docs/experiments/embeddings/E6-identity-channel.md` (settled,
> also `EVIDENCE` §8au) and `E7-card-attributes.md` (pre-registered, running).
>
> 🔴 **We ship 4 embedding tables of which 3.6–10.4% of rows ever saw a
> gradient.** Per-table seen counts: `slot_emb` 104/1300, `bag_emb` 134/1300,
> `card_emb` 135/1300, `atk_emb` **57/1600**. The rest ship at random N(0,1).
>
> 🔴 **Permuting only the OPPONENT's card ids on the frozen v5 net costs
> 0.838 → 0.587 vs `rule:crustle` (4/4 of its Pokémon in vocabulary) and
> 0.625 → 0.607 vs `rule:v10` (0/6).** Identifying the opponent's Pokémon is
> worth ~a quarter of the win rate where we can do it — and **against Mega
> Lucario we already cannot**, so there was nothing left to destroy. Mirror
> scoping control 0.550 [0.493, 0.605], CI spanning 0.500. No retraining.
>
> ⚠ **Permutation, not zeroing.** Zeroing moves the input distribution the
> downstream layers were trained against and cannot separate "identity
> destroyed" from "activations off scale". `scripts/p54_emb_ablate.py`.
>
> ⛔ **A STORED ANCHOR SCORE IS NOT A CONTROL.** `arena.build_agent` archives
> anchors as `rule:<name>` with **no version**, and `83daa48` changed Crustle on
> 08-02: the *same net, same flags* reads **0.767 before / 0.867 after** —
> +0.100 of apparent gain from changing nothing. 49,320 Crustle games pool both
> eras under one identity and `arena.py elo` fits over all of them. **Every
> strength claim must run its control back-to-back in the same session.**
>
> ✅ **The shipped bundle is v5.** `dist/submission.tar.gz` carries
> `out/policy_v5.npz` byte-identical (md5 `dc1c9acc…`, width 708). The tracked
> `agents/sa/policy_net.npz` is the old `policy_lw2` (width 496) and is **stale
> in the tree but not what ships** — do not "fix" it by retraining.
>
> 🔴 **v6 IS A CLEAN NULL — DO NOT REBUILD IT** (`EVIDENCE` §8av). Arm A
> (mirror, direct, pooled 2 seeds, n=600) **0.510 [0.470, 0.550]**; the
> hypothesis arm vs `rule:v10` confirmed at n=2,000/cell resolves to
> **+0.005 [−0.017, +0.027]**. Card attributes **do not** recover the identity
> channel, and per rule 4 the out-of-vocabulary story is retracted *for this
> intervention*. §8au's diagnosis is untouched. The nets are kept at
> `out/policy_v6_s{0,1}.npz`; **the shipped net is unchanged**.
>
> ⛔ **SCREEN ON THE DIRECT ARM.** A two-cell anchor delta (treatment vs control
> against a third party) carries **±0.080** at n=300 — every screened delta in
> E7 was inside it, and arm C's seed swing flipped sign. The mirror's direct
> head-to-head is **2× tighter for the same games**. Take a two-cell delta to
> n≥2,000 or do not quote it.
>
> ⚠ **§8z's ±0.019 seed floor is MIRROR-DIRECT and may not carry.** Against
> `rule:v10` at n=2,000 the two control seeds read 0.616 / 0.571 (spread 0.045).
>
> 🟢 **Where §8au actually points:** the corpus has **zero** Lucario games. E7
> tried to fix an unseen archetype by re-encoding cards and failed. The untested,
> simpler lever is **training data containing the archetype** — a data question,
> not an embedding one.
>
> 🟡 **The v6 machinery is built and works if ever wanted:** `--attr`,
> `--drop-a` + `features.A_GROUPS`, `artifacts/pds_v6`, and the dim guard
> handles all four state widths (496 / 536 / 726 / 1002). `--drop-a`
> sub-attribution was deliberately NOT run — five retrains against a null block.
>
> <details><summary>What v6 was (kept for the report chapter)</summary>
>
> `--attr` adds 276 state
> columns (energyType / weakness / ability / resistance / weak-to-facing-type)
> + a `cardType` one-hot on the option, all from the card DB, which covers all
> 1,267 cards and therefore **transfers to cards the corpus never contained**.
> Corpus `artifacts/pds_v6` is verified byte-identical to `pds_v4` on every v3/v4
> column, so the control trains on identical rows. Sized first
> (`p55_attr_sizing.py`): the gate **killed `aceSpec`** (one value corpus-wide)
> and **`pokemonType`/`evolutionType`** (fully redundant with the six stage/ex
> flags). Pre-registered prediction: **gain vs `rule:v10` > gain vs
> `rule:crustle`**; a uniform gain falsifies the mechanism.
>
> ⚠ **Thin support, stated before the result:** `weakness=5` (Mega Lucario's)
> appears on **one** trained card and `energyType=6` on **five**. A null was
> called live in advance, and a null is what it measured.
>
> </details>

> # 🔴 READ THIS FIRST — B7 RAN ON DAY 11 AND IS CLOSED (2026-07-31 night)
>
> **Day 10 measured; day 11 trained, and both arms lost.** The pre-registered
> bar was **+50 Elo weighted over the five anchors**. Neither arm was close, and
> neither was a null — both are losses against the net they were built to beat:
>
> | net | what it clones | miss vs **the field** | miss vs **ntumlnoob** | mirror vs v3, n=2,000 |
> |---|---|---|---|---|
> | **v3** (live) | the ~50-pilot mixture | **30.2%** | 40.1% | — control |
> | `rw25` | mixture, weighted to LB rating | 32.0% | **40.2%** | **0.421** [0.400, 0.443] ≈ **−55 Elo** |
> | `b7_ntum` | one 1163-rated expert | **36.2%** | 19.4% *(32.8% held out)* | **0.370** [0.349, 0.391] ≈ **−92 Elo** |
>
> 🔴 **Read the first and last columns together — the ordering is exact.** Field
> disagreement 30.2 → 32.0 → 36.2; Elo 0 → −55 → −92. **Every step away from the
> field's modal policy costs strength, and the net that best imitates the #2
> player is the weakest agent this project has built.** `EVIDENCE` §8t, §8u.
>
> ⚡ **And §8q's headline was NARROWED by a much harder test — this is the part
> to carry forward.** Over **87 same-deck, same-week demonstrators the net has
> never trained on a single row of**, agreement **peaks at 1050–1100 (76.1%) and
> falls in BOTH directions** — 66.7% below 900, 70.9% at 1100–1150, 65.6% at
> 1152, 59.9% at 1163. **Agreement measures distance from the fitted mode, not
> skill.** A 780-rated player is 33% unpredictable to us too, and nobody wants to
> clone them. §8r.
>
> ✅ **Covariate shift is RULED OUT** (it was §8q's one unanswered objection):
> policy-vs-policy disagreement is **26.7% on our own states vs 31.9% on theirs**
> — near-symmetric, so the expert clone is a genuinely different policy, not one
> policy measured off-support. The test carries a **1.7% positive control** (the
> v3 net reproducing its own submission's replays). §8s.
>
> ⛔ **Do not build a third demonstrator-selection variant.** Five axes of
> more/better training have now measured null or negative; **exactly one
> intervention ever worked and it was representational** (§8f). Spend the
> remaining days accordingly.
>
> # ⚡ AND TWO MORE CLOSED THE SAME NIGHT — read these before planning anything
>
> **1. ✅ THE CLONE IS NOT CAPACITY-BOUND. The ~30% residual is the ENCODING.**
> Identical corpus, identical recipe, only the width changed:
>
> | net | params | vs v3 | misses of 12,939 | agreement |
> |---|---|---|---|---|
> | **v3** (live) | 594,369 | 1.0× | **3,902** | 69.8% |
> | `cap_big` | 1,559,489 | **2.6×** | **3,900** | 69.9% |
> | `cap_xl` | 4,865,985 | **8.2×** | **3,945** | 69.5% |
>
> **2.6× buys two decisions out of 12,939; 8.2× loses 43.** Both big nets drive
> train loss far below v3's while validation peaks early and declines — they
> already have more capacity than the features can use. §8w.
>
> 🔴 **This is the gate result for RL and it cuts against it.** A policy gradient
> reads the **same** feature vectors. Where two options are bitwise-identical
> inputs with different right answers (§8f's exact finding), their **gradients
> are identical too** — exploration cannot break a tie the representation cannot
> express. **RL inherits this ceiling rather than escaping it.** ⇒ **The feature
> audit is a PREREQUISITE for the RL program, not a competing priority.**
>
> **2. 🔴 B4 IS DEAD, and the manner of death is the lesson.** The §8n design
> diagnosis was **right** — end-of-our-turn myopia was most of the rout — and
> simulating the opponent's reply moved it **0.075 → 0.375 [0.311, 0.444] n=200**,
> the largest movement any B4 change produced. **Still ≈ −89 Elo** against a
> clone costing 1 ms, and below the 0.40 line set before the run. **A correct
> explanation of a failure does not entitle you to a fix.** §8v.
> ✅ **The time-budget confound is RESOLVED, not caveated.** At matched budget,
> `seq,sb1.0` **without** reply scores **0.165 [0.120, 0.223]** against the reply
> arm's **0.375 [0.311, 0.444]**, n=200 each — **disjoint CIs**. So the design fix
> is worth **≈ +0.21 on its own** (≈ +193 Elo) and the extra time ≈ +154 Elo.
> ⚠ **CORRECTED day 28 (ROADMAP §2.7): the +154 is the recovery from budget
> ABORTS — at `sb0.35` the sequencer overran 62 times against 56 plans — so it
> is a floor repair, NOT a time→strength scaling curve. Do not cite it as
> evidence that spending the clock pays.**
> **The diagnosis was confirmed by a controlled experiment and B4 died anyway.**
> 🔴 **Consequence for NNUE: B4 was its only consumer that is not the dead
> game-tree search (§2), so an incremental evaluator buys nothing here until some
> consumer exists.** Do not build one on spec.

<details><summary>Day 10's headline box — the restore and the expert dumps (superseded in part by §8r–§8u, kept for the reasoning)</summary>

> # 🔴 TWO THINGS CLOSED ON DAY 10 (2026-07-31 pm)
>
> **1. ✅ THE P4b RESTORE IS ANSWERED BY THE LADDER: the 952 was a BOARD-SIZE
> ARTIFACT, not a better agent.** The identical tarball was resubmitted as
> `55129730` and read **833.9 at 4.0 h of play**, where the original read
> **958.2 at ~4 h** — same code, same age, **−124 points**, board ~4,000 →
> **6,024**. Our three agents now read **833.9 / 818.1 / 841.5** live.
> **§8k ("everything we own is within 36 Elo") is confirmed on the LB itself.**
> ⛔ **Do not reopen the restore. Do not chase 952.** `EVIDENCE` §8p.
> ⚠ One reading only — re-read ≥1 h later before quoting 833.9 (rule 2). The
> matched-age comparison does not depend on convergence.
>
> **2. ⚡ WE NOW HAVE EXPERT DEMONSTRATIONS OF OUR EXACT DECK, AND THEY SAY
> SOMETHING UNCOMFORTABLE.** 227 games from **Sixth Sense (#3, 1152.4)** and 330
> from **李秉叡（ntumlnoob）(#2, 1162.8)** — both playing `decks/grimmsnarl.py`
> **card-for-card**. Scoring the live v3 net against their actual choices:
>
> | demonstrator | rating | rows | miss rate |
> |---|---|---|---|
> | 48 other grimmsnarl pilots (mostly 1090–1140) | ~1110 | 10,088 | **27.2%** |
> | Sixth Sense (#3) | 1152 | 18,296 | **34.4%** |
> | ntumlnoob (#2) | 1163 | 25,775 | **40.1%** |
>
> **Agreement falls monotonically as the demonstrator gets better.** And the two
> top players diverge from us in *different* contexts (Sixth Sense almost entirely
> in `TO_HAND`, −31.5 pp; ntumlnoob broadly in MAIN/damage/switch) — **so there is
> no single "expert move" to copy.**
>
> 🔴 **The structural fact behind it: our corpus is ALREADY elite and already
> concentrated.** flg (1125) **527 seats**, Dries @ Tufa Labs (1102) **490**,
> James Cox (1166) **414**, Dominic Peel (1136) 238, LiamK (1128) 216. **We clone
> 1100–1166 play and score 833.9.** More/better demonstrators is therefore NOT
> the obvious lever; **what has never been tried is cloning ONE policy instead of
> a ~50-pilot mixture.** `EVIDENCE` §8q.
>
> ⚠ **The alternative explanation is NOT ruled out and must be stated first in
> any write-up: COVARIATE SHIFT.** Agreement is measured on the demonstrator's own
> trajectory distribution, so part of that 40% is BC's compounding-error problem,
> not a policy we can copy. **Low agreement does not prove they play better. Only
> an arena A/B can.**
>
> ✅ **ANSWERED day 11 — covariate shift is ruled out (§8s), the monotone trend
> is narrowed to a PEAK at 1050–1100 (§8r), and both interventions this box
> motivated LOST (§8t, §8u).** The box is kept because its reasoning was sound
> and its bar was set honestly; the arena is what changed the answer.

</details>

<details><summary>Day 9's headline box — the "−130 regression" artifact (superseded by §8p, kept for the reasoning)</summary>

>

> **⚠ An earlier version of this box, written mid-session, said the gap was
> SOLVED and that v3's loss was reproduced locally. That was written after 2 of 5
> anchors and it was wrong. Corrected here after the full sweep.**
> `report/EVIDENCE.md` §8i.
>
> **1. There is a real v3 weakness, and the retired anchor is what found it.**
> Both agents vs a fixed `rule:v10,noS` on `lucario_v10`, n=2000, CIs disjoint:
> **P4b 0.576 [0.554, 0.598] vs v3 0.505 [0.483, 0.527]** — v3 is ≈ **−50 Elo**
> against **12.8%** of the field. B1 could not have seen this; the anchor had been
> retired two days before.
>
> **2. But it does NOT explain the regression — it points the other way.**
> Weighted by field share over the 61.4% now measured, **the arena says v3 is
> ≈ +35 Elo BETTER** than P4b (mirror head-to-head **0.657**, Crustle **+91 Elo**,
> Alakazam a dead heat, Lucario **−50**).
>
> **3. 🔴 And the −130 was a comparison rule 2 forbids.** `55072063`'s **952.0 is
> FROZEN** — earned 07-29 against a ~4,000-entrant board; the board is now
> **6,000**. The only same-time, both-active comparison is:
>
> | submission | agent | read 2026-07-31 |
> |---|---|---|
> | `55116557` | **v3, rules off** | **819.8** |
> | `55077709` | P6a (lw2 + chip + spread + `counter_source`) | **845.0** ⚠ *still climbing*: 824.9 → 837.5 → 845.0 |
>
> **−25 points, against an agent that has not converged — well inside the LB's
> own ±50–100 swing.** So the honest status is **"v3 and P6a are indistinguishable
> on the ladder"**, not "v3 lost 130 points".
>
> **⇒ §8g's "the arena is systematically wrong, n=2" is WEAKENED.** Both its
> instances compared against non-comparable numbers — `counter_source` against a
> converging score, B1 against a frozen one. **There may be no systematic arena
> bias to explain.** Do not spend more days explaining one.
>
> **🔴 The genuinely load-bearing finding of day 9 is the sampling frame, and it
> is independent of all of the above.**
> `fetch_top_episodes.py` mines the **top** episodes by `avg_score`, and Kaggle's
> daily datasets **bottom out at 1055** — buckets 800–900 and 900–1000 contain
> **zero** episodes. We play at **825–952**. **No amount of episode mining can
> ever describe the field we face.** §8b's "Lucario is 0% of the meta" was true
> at 1150+; in our own 109 real games it is **12.8%** — tied for the largest deck
> we play against.
>
> **Our own submission replays are the only evidence about our own opponents.**
>
> ✅ **Fixed the same day:** `scripts/p9_field_census.py` names the real field,
> and `scripts/import_field_agents.py` imported the two missing anchors
> (`rule:alakazam5`, `rule:archaludon`). Anchor coverage **39.4% → 71.6%**.
>
> ⚠ **v3 is NOT refuted as a net.** It wins big on 26.6% of the field and loses
> on 12.8%; the other 60.6% is unmeasured. **Do not discard it and do not reship
> it** until the 5-anchor sweep finishes (▶ item 2).

</details>

**Read §2 before trusting any number. §3 is the live plan. This file must always
end with a live plan, never a summary.**

⚠ **Day 9 note on reading this file:** several load-bearing claims dated
2026-07-30 were **narrowed, not deleted** — the meta-shift table (§1), rule 12's
"`lucario_v10` is 0% of the meta", and rule 16's "the arena does not measure
ladder strength". Each is now prefixed with what it is actually true *of*. If a
statement in here about "the meta" does not say **which score band** it describes,
distrust it: mined episodes are the top-1150 band, and
`scripts/p9_field_census.py` on our own replays is ours (`EVIDENCE` §8i).

### ⛔ SUPERSEDED — DAY 19's PLAN (B3, rule-based archetype counters). NOT the live plan; the live one is the DAY-23 PLAN at the top of this file.

> ⚠ **Kept for its reasoning, not as a task list.** B3 never ran: day 19's
> gating input (a fresh own-field census) was overtaken by the embedding
> programme, and by day 22 the strategic picture had changed — every Elo axis is
> closed and the report carries ~76% of the grade against the leaderboard's ~14%.
> **Do not start B3 from this section.** The archetype-share table it depends on
> is also pre-§8ay and therefore misweighted.

> # 📥 WHAT THE USER IS BRINGING (asked for at the end of day 18)
>
> **1. 🔴 A FRESH REPLAY DUMP OF OUR OWN AGENT — this is the gating input, not a
> nice-to-have.** §8ac measured that our opponent pool is **a function of our own
> rating**, and our last own-field census is from **day 15**. Day 18's meta mine
> describes the **≥1055 band**; we play at **~942**. **Coding counters for
> archetypes we do not actually face is rule 14's exact failure mode**, and it
> killed three rules this week (Morgrem 0.2/game, Pokégear 0.27, empty-bench
> 0.187). Dump `55160229` (v5), then:
>
> ```powershell
> python -X utf8 scripts/p9_field_census.py --us Scio --dir replays/<new_dump>
> ```
>
> **That table decides WHICH counters are worth writing. Do not write one before
> it exists.**
>
> **2. A targeted per-team dump of a top Dunsparce pilot** — `Majkel1337` (284
> games, **64.1% WR**) or `Brahim` (Buneary/Dunsparce, 102 games) or `Luca` (#1).
> ROADMAP calls targeted per-team dumps the best data source we have, and unlike
> mined episodes they are **not censored by band**.
> ⚠ **FOR READING THE ARCHETYPE'S LINE OF PLAY, NOT FOR CLONING.** §8u measured
> that cloning a 1163-rated expert cost **−92 Elo** and that agreement with an
> expert **anti-predicts** strength. A 1279 demonstrator is further away, not
> closer. Read it; do not fit to it.
>
> # ✅ BOTH RUNS LANDED — THE DECK SEARCH IS CLOSED
>
> **`p37` — stage 2, the pre-registered confirmation of candidate G. IT DIED.**
> 57,600 fresh games, all seven anchors:
>
> | anchor | weight | control | G | Δ |
> |---|---|---|---|---|
> | mirror | 33.3% | 0.504 | 0.500 [0.492, 0.507] | −0.004 |
> | alakazam5 | 22.0% | 0.793 | 0.768 | 🔴 **−0.025** |
> | archaludon | 8.0% | 0.688 | 0.656 | 🔴 **−0.032** |
> | crustle (v4) | 6.7% | 0.764 | 0.760 | −0.004 |
> | garchomp | 6.7% | 0.837 | 0.818 | −0.019 |
> | dragapult | 5.3% | 0.807 | 0.787 | −0.020 |
> | v10 | 4.0% | 0.615 | 0.564 | 🔴 **−0.051** |
> | **WEIGHTED** | **86.0%** | | | 🔴 **−0.0140** |
>
> **ΔW = −0.0140 against ±0.0050 resolution — 2.8× outside, NEGATIVE on 7 of 7.**
> ⛔ **The kill line is not met. G dies, and per the pre-registration THE SEARCH
> IS OVER — no second candidate is promoted.** That clause is what made this one
> test instead of eleven. `EVIDENCE` §8as.
>
> ⚡ **The cheap screen predicted the expensive confirmation**: stage 1 called the
> mirror at **0.501** on 4,000 games, stage 2 says **0.500** on 15,800. The
> two-stage design's central bet is confirmed on first use.
>
> **`p38_xero2` — the Xerosic isolation. −0.040, and it does NOT convict the
> card.** `EVIDENCE` §8at.
>
> | deck | cards changed | score | Δ | Elo |
> |---|---|---|---|---|
> | control | 0 | 0.504 [0.488, 0.519] | — | — |
> | **Xerosic ×2 isolated** | **2** | **0.464** [0.449, 0.479] | **−0.040** | ≈ −28 |
> | community + Xerosic bundle | 5 | 0.431 [0.415, 0.446] | −0.073 | ≈ −51 |
>
> ⚡ **§8ab's no-bundling rule paid for itself** — the bundle's −0.073 splits into
> Xerosic ×2 at −0.040 and the other three changes at −0.033.
> 🔴 **But "the card is wrong here" and "our clone misplays it" both predict
> −0.040**, and the replays show the net firing it at opponent hand size **4**
> while **nine offers at 7** went by (it discards down to 3, so those plays
> removed **one card each** against a best-available **five**). **Only a timing
> rule separates them — a day-19 candidate, rule 11 tradeoff column, 0 for 4.**
>
> # 🃏 WHAT DAY 18 SETTLED
>
> **1. ⛔ THE DECK SEARCH FOUND NOTHING — 11 pre-registered variants, ALL ≤ 0**
> (§8ar, `out/logs/deck_search_prereg.txt`, frozen in commit `d93cf04` **before**
> any variant deck file existed). Six of eight mirror candidates lost
> significantly. **Ultra Ball was held fixed across six different cut slots and
> lost in all six (0.439–0.488)** — which separates *"we cut six good slots"*
> from *"the add card is wrong"*, and only the second explains six losses.
> ⚡ **A single A/B could not have distinguished those, and that is the part of
> the design that earned its cost.**
> ⚡ **The cheap screen predicted the expensive confirmation almost exactly:**
> stage 1 called the mirror at **0.501** on 4,000 games; stage 2 says **0.500**
> on 15,800.
>
> **2. 🔴 §8af's EXPOSURE FILTER IS NECESSARY BUT NOT SUFFICIENT, and this is the
> most reusable finding of the day.** Ultra Ball sits at **5.59×** the exposure of
> our weakest card and lost every slot. Energy Switch sits at **3.61×** and the
> net **played it 1 time in 28 offers**. **Card-level exposure is not the binding
> constraint; card × DECK-CONTEXT is**, and nothing in this repo measures that.
>
> **3. ⛔ THE COMMUNITY LIST IS NOT PLAYED ON THIS BOARD.** The 08-01 consensus
> Grimmsnarl list is **identical to `decks/grimmsnarl.py`, card for card, seen
> 158×**. Budew, Yveltal and Energy Switch appear in **zero** 08-01 lists;
> **Special Red Card is not implemented in this engine at all.**
> The user's revision (Budew + Yveltal out, Xerosic ×2 in) measured
> **0.431 [0.415, 0.446]** vs the 0.504 control ≈ **−51 Elo**, the largest deck
> loss measured here — ⚠ **but it is a FIVE-card bundle and §8ab forbids
> attributing it to any one card.** That is what `p38_xero2` isolates.
>
> **4. ⚡ THE XEROSIC MECHANISM IS MEASURED AND IT IS A PILOT PROBLEM, NOT
> NECESSARILY A CARD PROBLEM.** Over 6 recorded games the card was offered 28
> times with the opponent holding a mean of **5.2** cards — **nine of those at 7
> cards** — and the net played it **twice, both at hand size 4**, discarding
> **1 card each time** (it discards down to 3). Best available moment would have
> discarded **5**. ⚠ n=6, a smoke test; and the 7% take rate is NOT itself low
> (Boss's Orders takes 6% — supporters are offered many times a turn and only one
> is playable). 🔴 **This is a live B3-shaped rule candidate for day 19** — *play
> Xerosic at the highest opponent `handCount`* — ⚠ but rule 11 puts it in the
> **tradeoff** column (it competes for the one Supporter play), and tradeoff
> rules are **0 for 4** here.
> 🔧 **A correction made before publishing:** the first pass read the opponent's
> `hand` array and got "0 cards both times", a far more dramatic claim. **That
> array is hidden and always empty**; the true count is in `handCount`. Same
> shape as rule 18.
>
> **5. 🔴 AN ANCHOR HAD DRIFTED (§8aq) ⇒ RULE 19.** `rule:crustle` is a **fourth**
> pilot committed **26 minutes after** its last measurement and verified on
> **n=6**. It reads **0.755 [0.735, 0.773]**, not the **0.866** three docs quoted
> — confirmed three times independently (p35 0.755, p34 0.748, p37 ctrl 0.764).
> **The one-line Dwebble tie-break inside the empty-bench guard is worth 0.111**,
> larger than the whole repair it was a footnote to. ⇒ **WHICH Pokémon a pilot
> benches matters more than WHETHER it benches**, and §8an's Result 2 is reversed
> for the pilot we ship. ⛔ **No verdict retracted** — both nets faced the same
> pilot and §8an showed the shift is a level shift that cancels in differences.
>
> **6. ⚠ THE "META SHIFT" AT THE TOP IS MOSTLY A SAMPLING ARTIFACT.** By **games**
> our archetype fell 52.1% → 20.1% and Dunsparce variants are **58.8%**; by
> **teams** ours is still the most-played at **37.9%** against all Dunsparce
> variants' **17.2%**. **One team (Majkel1337) supplied 284 of 399 games** and
> mining caps at 400 by `avg_score`, so a hyperactive team crowds the sample —
> rule 16's sampling frame, third instance. ⚠ Our team count *did* fall 22 → 11,
> which activity does not explain on its face, but with 284 of 399 games taken
> there are only ~115 left to spread over everyone else, so crowding confounds
> that too. **Neither reading is clean and both are recorded.**
>
> # ▶ THE DAY-19 PLAN — B3, ARCHETYPE COUNTERS (user-directed)
>
> **The precedent is good and it is the ONLY axis with a shipped win.** B3
> instance 1 (the Crustle branch) recovered **+0.104** and is live in v5. Every
> other axis is closed: search, data, demonstrators, capacity, sequencing, RL,
> and now deck.
>
> 1. 📥 **Census our own field from the fresh dump FIRST** (above). Nothing else
>    runs until the archetype shares for *our band* exist.
> 2. 🔴 **THE BLOCKER NOBODY HAS NAMED YET: we cannot SIMULATE 58.6% of the top
>    band.** There is **no deck file and no pilot** for any Dunsparce variant, nor
>    for Teal Mask Ogerpon (9.4%). **A counter we cannot A/B is a hunch.** The
>    consensus lists are already mined and sitting in `out/meta/day_0801.txt`;
>    building the deck is the `cynthia_garchomp` recipe. ⚠ **The pilot is the hard
>    part, and §8ap warns a net-piloted anchor is both uninformative and
>    flattering** (`bc:garchomp` 0.857 is an upper bound, biased optimistic).
> 3. ⚡ **The Dunsparce shell, for reference** (consistent across all three
>    variants — a speed/disruption deck, not a damage race):
>    `4x Mist Energy · 4x Dunsparce · 4x Dudunsparce · 4x Buneary · 3x Mega
>    Lopunny ex · 4x Ultra Ball · 4x Poké Pad · 4x Buddy-Buddy Poffin · 4x Air
>    Balloon · 4x Hilda · 4x Lillie's Determination · 4x Wally's Compassion ·
>    1x Fan Rotom`. ⚡ **Xerosic's Machinations ×1 is in Majkel1337's 64.1% list** —
>    the user's card is real, just as a 1-of in a free-retreat/disruption deck.
> 4. ⚠ **RULE 11 GOVERNS EVERY RULE WRITTEN TOMORROW: dominated → build,
>    tradeoff → distrust. 3 for 3 and 0 for 4.** Before writing one, say which
>    column it is in.
> 5. ⚠ **RULE 14 BEFORE RULE 11: size it first.** How often does the condition
>    fire per game, and how big is it per instance? Three rules died on that
>    question this week without an A/B being spent.
> 6. ⛔ **Do NOT submit.** Rank 198 with v5 live; calendar is consolidate
>    08-08→08-14 then freeze.
> 7. 📝 **`STRATEGY.md` is the standing one-edit-per-session obligation** and is
>    now owed **seven** chapters (§8am, §8an, §8ao, §8ap, §8aq, §8ar, and the deck
>    search). It is 30%+ of the rubric against LB's one bullet of five.

<details><summary>Day 18's own plan and results as written mid-session</summary>

### ▶ DAY 18 — THE STRATIFIED DESIGN IS PRICED, AND ITS OWN INSTRUMENT NARROWED IT TO TWO SLOTS.

> # 🃏 DAY 18 SO FAR — measurement only. ⛔ NO DECK VARIANT HAS BEEN BUILT OR A/B'd (user is reviewing the design first).
>
> **1. ✅ §8ap's "near ceiling ⇒ cannot resolve" does NOT block Track C** (§8ar,
> `p33_anchor_resolution.py`). It is true in **Elo** units and false in the units
> a deck decision uses: `W = Σ wᵢpᵢ` is linear in **win rate**, and noise *falls*
> near the ceiling. Worst anchor costs **2.04×** the games for equal Elo
> resolution, not infinity; in win-rate units `bc:garchomp` is our **most**
> sensitive cell. Full design: **±0.0050 on W for 57,600 games ≈ 1.1 h** at 2–3
> jobs (Neyman allocation, 55% of naive equal-n).
>
> 🔴 **But the case for stratifying is BIAS, not precision** — the same games
> spent mirror-only measure Δ to **±0.0041**, *tighter*. Stratify only where the
> mirror is biased. That is a liveness question, and it needed a new instrument.
>
> **2. 🔴 THE MIRROR-ONLY CRITIQUE IS REAL BUT NARROW — 17 of 19 slots are
> mirror-safe** (§8ar, `p34_matchup_liveness.py`, 400 games × 7 anchors).
>
> | card | mirror | alakazam5 | crustle | weighted | spec |
> |---|---|---|---|---|---|
> | **Tool Scrapper** | 0.02 | 0.56 | 0.47 | **0.28** ⚠ under every sizing floor | **0.93** |
> | **Froslass** ⭐ | 1.44 | **5.57** | **6.83** | **3.14** | **0.54** |
> | *(17 others)* | — | — | — | — | **≤ 0.16** |
>
> ✅ **Positive control:** it recovered §8al's Tool Scrapper fact unprompted.
> 🔴 **§8al's example was the deck's single most extreme card**, so the critique
> its plan was built on is far narrower than assumed. ⚡ **The one slot that
> matters is the one ROADMAP already named — the Froslass line** (Track C step 4,
> "the only growable passive-damage line"): a mirror A/B sees **under a quarter**
> of its real use. ⚡ **And the bias runs both ways** — Munkidori 18.6 mirror vs
> 11.3–14.2 elsewhere, so a mirror A/B **overstates the core engine** too.
>
> **3. 🔴 AN ANCHOR HAD DRIFTED AND EVERY DOC QUOTED THE OLD NUMBER** (§8aq).
> `rule:crustle` is a **fourth** pilot (`83daa48`, committed **26 min after** the
> 0.866 run's last game, verified on **n=6**). At n=2,000 it reads **0.755
> [0.735, 0.773]**, confirmed independently by `p34` at 0.748. **The one-line
> Dwebble tie-break inside the guard is worth 0.111 — larger than the entire
> +0.098 empty-bench repair, in the opposite direction.** ⇒ **WHICH Pokémon a
> pilot benches matters more than WHETHER it benches.**
> ✅ **§8an's Result 2 is reversed for the pilot we actually ship**: v4 resolves
> *better* than the broken v1 (0.768) **and** keeps the guard. §8ap's headline
> (40.7% above 0.75) survives; its Crustle row does not.
> ⛔ **No verdict is retracted** — both nets in every comparison faced the same
> pilot, and §8an showed the shift is a level shift that cancels in differences.
> ⇒ **HANDOFF rule 19**, and it is a NEW shape: not a buggy script, but **two
> correct scripts with the world changed between them.** Only a timestamp catches
> it. ✅ Swept all seven anchors — Crustle was the only drift.
>
> #### ▶ WHAT IS NEXT (blocked on the user's review of the design)
>
> 1. 🃏 **The deck search, two-stage, pre-registered.** Stage 1 screens
>    candidates **in the mirror** (33.3% of the field, the only matchup we win
>    exactly 50% of, and 4× cheaper per unit precision) against §8aj's same-deck
>    control **0.4980 [0.483, 0.513]** — **ranking, not testing.** Stage 2
>    confirms **the top-1 only** at p33's stratified allocation. 🔴 **The
>    multiplicity rule is the point:** k variants at α=0.05 manufactures a winner
>    at k≈20, which is the shopping the B8 β-sweep was declined for.
>    ⚠ **Froslass and Tool Scrapper skip stage 1** — the mirror cannot judge them.
> 2. ⚠ **The expected outcome is a NULL and it is on the record before the run.**
>    §8al measured strength falling monotonically with distance from the consensus
>    60. "A proper search over the slot space, and the list survived it" is a good
>    Deck Score result and is **not** an Elo lever.
> 3. 📈 Re-sweep the anchor table — ⚠ **now known to have been about to quote a
>    stale Crustle row**; use 0.755.
> 4. 📝 `STRATEGY.md` — §8am/§8an/§8ao/§8ap are owed chapters, plus §8aq/§8ar.
> 5. ⛔ **Do NOT submit.** Rank 198 with v5 live; consolidate 08-08→08-14.

</details>

<details><summary>The day-18 plan as set at the end of day 17 (items 1–2 are now in progress above)</summary>

### ▶ DAY 18 AS PLANNED — B8 IS DEAD, THE LADDER IS PROVABLY UNUSABLE, AND TRACK C IS THE ONLY RUBRIC-WEIGHTED TRACK LEFT UNSTARTED.

> # 🔴 THE DAY-17 HEADLINE: TWO DECISION-IDENTICAL AGENTS READ 63.2 POINTS APART
>
> `55169114` differs from v5 (`55160229`) only by health counters and one
> `print` — weights, deck, engine byte-identical. A fourth read closed §8ak's
> withheld verdict: **942.7 vs 879.5**, both settled by the same test.
>
> **A true difference of EXACTLY ZERO, displayed as 63 points.** Against every
> effect this project has measured: day-15's +40.5 headline, §8z's **+37**,
> §8ab's **−36**, §8aa's **+14**.
>
> ⇒ 🔴 **RULE 2'S SECOND CLAUSE IS NOW A MEASUREMENT, NOT AN INFERENCE. The
> ladder cannot adjudicate any net change we make or are likely to make. The
> arena — n≥2000, byte-identical control, 0.482 seed floor — is not the weaker
> instrument. It is the ONLY instrument.** ⚠ Quote it as *"≥63 and still
> closing"* (81.7 → 76.2 → 69.0 → 63.2); read 4 closed because the *converged*
> agent rose **+4.2**, so **both** agents fail day-scale convergence.
>
> 📈 **Standing: rank 198 / 6,136 at 942.7 — best rank ever, and our SCORE FELL
> while our RANK ROSE.** New #1 is `Luca` at 1322.6.
>
> # 🔴 B8 IS CLOSED — TWICE, ON 20,000 SELF-PLAY GAMES
>
> | corpus | treatment vs its byte-identical control | control vs v5 |
> |---|---|---|
> | 4,000 games | **0.512** [0.491, 0.534] | 0.480 [0.458, 0.502] |
> | **16,000 games** | **0.506** [0.484, 0.528] | 0.491 [0.469, 0.513] |
>
> **Bar was 0.541 (computed before either A/B). Both failed. Each control sits
> on v5, so the fine-tuning procedure itself is harmless — the null is clean,
> not two damaged nets compared to each other.**
>
> ⚡ **4× the data moved the estimate DOWN (0.512 → 0.506).** A decision rule was
> committed **while the second arena was still running** (`out/logs/b8_prereg.txt`):
> *estimate above ⇒ scale to 40,000; at or below ⇒ the axis closes on the METHOD,
> not the budget.* **Branch (b). ⛔ Do not run the 40,000-game version.**
>
> ⚡ **A parameter diagnostic, also taken before the result, says what the null
> MEANS:** total abs Δ treatment-vs-control **455.6** against control-vs-v5's
> **1349.1** — the advantage weighting moved the head **34%** as far as the
> fine-tune moved it from v5. **The parameters moved substantially and the win
> rate did not.** That is a stronger negative than a bare 0.506.
>
> ⚠ **β is UNTESTED, and it is named rather than buried.** Both runs used β=1.0
> (2.7× win/loss ratio). A sweep was declined by rule — reporting the best of
> several configs is shopping. *"A stronger reweighting might work"* is
> **unfalsified, not refuted**, and belongs in the report as an open question.
>
> ✅ **Also built and reusable regardless:** `p26_selfplay_gen.py` writes
> corpus-format shards straight from self-play (no replay dump, no adapter), and
> `train_policy.py` gained `--advantage / --anchor-ds / --margin-max /
> --freeze-except / --export-last`.
>
> # ⚡ THE CRUSTLE ALARM IS RETIRED, AND THE ANCHOR SET IS COMPLETE
>
> **§8an — the re-run the user authorised.** All three nets score **+0.087…
> +0.102** higher against the repaired pilot, every CI disjoint, **and the sign
> is the OPPOSITE of what §8ah predicted.** But it is a **LEVEL** shift, so it
> cancels in the net-vs-net *differences* every weighted verdict is built from
> (v4−v3 moves +0.013, v5−v4 +0.003; at 6.7% weight, +0.0009 and +0.0002) **and
> the net ordering is unchanged under every pilot version.** ✅ **Nothing needed
> rewriting. §8ah's "every verdict carrying a Crustle term is suspect" is
> answered, for two independent reasons.**
>
> ⚡ **Decomposed, because the user asked the right question** (*"I thought we
> only made it bench when the bench was empty"*): the fix had changed **three**
> things where **one** was authorised. Narrowed to the guard alone and
> re-measured — the unauthorised bench-anything default contributes **+0.011 /
> −0.013 / −0.004**, sign-flipping noise. **The empty-bench guard is the whole
> +0.09.** ⛔ **Its MAGNITUDE (90000, vs Dwebble's 25000) is NOT understood and
> was deliberately left alone** — one mechanism hypothesis was already refuted
> today. **The honest next step is watching a game, which is how the original
> bug was found.**
>
> **§8ap — both missing anchors closed, and the finding is not the coverage.**
> `rule:dragapult` **already existed and had never once been used** in nine days
> (0.809); Garchomp was **built** from our own meta snapshot after checking
> §8af's exposure filter first — all 20 card ids in the corpus, 0 of 60
> untrained (0.857).
>
> 🔴 **Sorting the anchors by resolution also sorts them by
> UNrepresentativeness.** The only two near 0.5 are `rule:v10` (**4.0%**) and
> `rule:archaludon` (**8.0%**) — both **0 of 47 games above rating 900**.
> Everything representative we beat **77–87%** of the time. **§8ac's re-weighting
> was correct and moved weight ONTO the anchors that cannot resolve a
> difference: 40.7% of every weighted verdict now sits above 0.75.**
> ⚡ **The mirror is the only anchor that is both, and it carries the set** —
> which is also why §8ao's B8 A/B was right to run there.
> ⚠ **`bc:garchomp` is weak twice:** our net holding someone else's 60 measures
> *deck × how well OUR net pilots it*, so 0.857 is an **upper bound, biased
> optimistic**. Never quote it as "we beat Garchomp 86%".
>
> #### 🔧 Three of this session's own errors, corrected in the sections that made them
>
> 1. **The seat bug, AGAIN.** Crustle was first reported as **0.489 / 0.510 /
>    0.502** by reading `winner==0` as "agent A won". Seats alternate. True:
>    **0.857 / 0.888 / 0.870**. §8ae documented this five days earlier; the fix
>    went into `p21` and a throwaway snippet felt exempt. ⇒ **rule 18**.
> 2. **Dragapult characterised from n=6.** Written up mid-session as "far more
>    competitive than Crustle" off a 2/6 smoke; **n=2000 says 0.809**, the
>    opposite direction. Rule 1, violated by its own maintainer.
> 3. **Memory over-estimated by 40%**, which sized the B8 rerun at 4× instead of
>    the 10× requested. Measured **1.48 KB/row**, so ~55,000 games would fit.
>    ⛔ Moot — the decision rule closed the axis anyway.
>
> #### ▶ THE DAY-18 ORDER OF WORK
>
> 🔴 **The strategic situation, stated plainly: every LB axis is now closed, and
> §8ak says the LB could not have shown us the difference anyway.** Search (§2),
> data (§1), demonstrators (§8t/§8u), capacity (§8w), sequencing (§8v), deck
> guess-a-swap (§8al), and now RL (§8ao). The feature axis is spent (+115 → +37
> → +14). **Chasing marginal Elo is now measurably unmeasurable.** ROADMAP §4
> already commits to the consequence: *the dossier does not get sacrificed for
> marginal Elo.*
>
> 1. 🃏 **TRACK C, DESIGNED PROPERLY — the only rubric-weighted track with a
>    written next step nobody has taken.** Deck Score is **20%**. §8al retired
>    guess-a-swap (three hunches → one null, two significant losses, monotone
>    worse with distance from the consensus 60) and named the successor:
>    **a MATCHUP-STRATIFIED SEARCH DESIGN over the whole slot ranking.**
>    ⚠ **All four deck A/Bs so far were MIRROR-ONLY**, which flatters any variant
>    cutting mirror-dead tech (Tool Scrapper: 0.00 plays/game in the mirror) and
>    cannot judge a card aimed anywhere else. ⚡ **We now have the anchors to fix
>    that** — dragapult and garchomp exist as of today, and `p25_deck_slot_audit.py`
>    already sized all 60 slots.
> 2. 📈 **Re-sweep the five-anchor table against the CURRENT anchor set** — the
>    repaired Crustle, plus dragapult and garchomp — so the weighted verdicts
>    quote instruments that exist today. ⚠ Carry §8ap's warning: report the
>    mirror term separately, because it is doing nearly all the work.
> 3. 👀 **User task: watch one Crustle game** (`out/replays/audit_crustle_v4`,
>    `notebooks/visualizer.html`). The 90000 guard fires at ~9 decision points a
>    game and nobody knows what it costs. **The last time a human watched an
>    anchor, it found a game-throwing bug in one replay.**
> 4. 📝 **`STRATEGY.md`** — day 17 satisfied the one-edit rule (§5.1b, the LB
>    resolution measured rather than argued). Day 17's other four sections
>    (§8am, §8an, §8ao, §8ap) are **owed chapters** and the user has deferred
>    them; the deadline is **09-14** and the material is now unusually strong.
> 5. ⛔ **Do NOT submit anything** without a reason that survives §8ak. We are at
>    rank 198 with v5 live; the calendar is consolidate 08-08→08-14 then freeze.

</details>

<details><summary>Day 17's original plan as set at the end of day 16 (B8 ran and is closed above)</summary>

### ▶ DAY 17 IS B8: THE RL FINE-TUNE. IT IS SCHEDULED, IT HAS A KILL LINE, AND IT HAS A HARD STOP.

> # 🤖 NEXT SESSION STARTS HERE. EVERYTHING ELSE IS PARKED.
>
> **The user's decision at the end of day 16: park deck work, do RL.** Both halves
> are evidenced — §8al retired the guess-a-swap method, and B8 is the only axis
> left that is neither dead nor spent.
>
> 🔴 **WHY IT IS THE LAST ONE.** Search (§2), more data (§1), demonstrator
> selection (§8t/§8u), capacity (§8w — 8.2× params bought **−43** decisions),
> within-turn sequencing (§8v), and now **deck (§8al)** are all closed. The
> feature axis is the only one that ever paid and it is falling ~3× a generation:
> **+115 → +37 → +14 Elo.**
>
> ⚡ **AND THE THING A CLONE STRUCTURALLY CANNOT KNOW IS EXACTLY WHAT AN OUTCOME
> SIGNAL SUPPLIES.** The corpus records *what humans did*, never *whether it
> worked*. That is not a re-parameterisation of existing information — it is new
> information, which is why B8 is different from capacity (§8w) and from
> per-card expert heads (same data, same features, strictly less context).
>
> ✅ **PREREQUISITES ARE ALL BUILT — nothing blocks the first line of code.**
> `harness.Recorder` (§8ad) emits trajectories; the feature audit has been done
> twice (§8y/§8z, §8ab); the encoding ceiling is computed (§8x, 95.6% vs a clone
> at 71%, so expressiveness binds ≤4.4 pp); the seed-only null is measured
> (**0.482**, §8z); `arena.py play --archive` is the A/B harness.
>
> ✅ **AND THE VARIANCE OBJECTION IS PRICED, NOT ARGUED** (§8ae): **5.96 games/s
> per process ⇒ ~5.5M games to the deadline.** +37 Elo separates in **800 games**;
> a **1%** effect at a context recurring 20×/game needs **960**. Every row is
> affordable by three orders of magnitude.
>
> #### The build, smallest version only
>
> 1. **Fine-tune a SMALL parameter set** — final layer or a low-rank delta, not
>    the whole net — on our own recorded game outcomes.
> 2. **Keep the corpus loss as an anchor term** so the policy cannot drift off the
>    clone it started from. The clone is worth ~955 on the ladder; the downside
>    risk is undoing that.
> 3. **A/B at n≥2000 against a BYTE-IDENTICAL control**, seed floor carried in.
> 4. ⛔ **Never screen on `val_top1`** — rule 3, paid for twice: §8z moved
>    agreement by **8 decisions** and bought **+37 Elo**; §8aa moved it by **214**
>    and bought **+14**. A 70× exchange-rate difference means the fit metric
>    cannot screen in either direction.
>
> 🔴 **PRE-REGISTERED KILL LINE, WRITTEN BEFORE ANY CODE:** *if the fine-tuned net
> does not beat its byte-identical control by a margin whose CI excludes the
> seed-only null at n≥2000, B8 DIES and becomes a report chapter.*
> ⚠ **And §8ae's own warning binds:** *a sizing probe that fails to kill is not
> evidence that a thing works.* **B4 passed all three of its kill criteria and
> then died at n=200.**
> ⏰ **HARD STOP 08-08.** After that the calendar is consolidate (08-08→08-14)
> then freeze; a winner found later could not be integrated and submitted with
> time to converge (~4 h+, last safe submission ~08-15).
>
> #### Carried into day 17
>
> - ✅ **DONE FIRST, AND IT IS THE BIGGEST NUMBER IN THE REPO** (§8ak). The
>   fourth read landed at **10:33 UTC**: v5 **942.7**, health bundle **879.5**,
>   **gap 63.2**. v5 moved −1.6 in 3 h 08 m against −12.2 in the prior 2 h 17 m,
>   and **the reference agent moved MORE (+4.2)** — so the withheld condition is
>   met and the verdict is written. 🔴 **Two decision-identical agents read 63.2
>   points apart, which is LARGER than every effect this project has measured**
>   (+40.5 day-15 headline, +37 §8z, −36 §8ab, +14 §8aa). **Rule 2's second
>   clause is now a measurement, not an argument: the ladder cannot adjudicate
>   any net change we make. The arena is not the weaker instrument — it is the
>   only one.** ⚠ Quote it as *"≥63 and still closing"* (81.7 → 76.2 → 69.0 →
>   63.2); read 4 closed because the **health bundle rose**, after rule 2 had
>   certified it converged, so **both** agents fail day-scale convergence.
>   📈 **Standing: rank 198 / 6,136 at 942.7** — best rank ever, and our score
>   *fell* while our rank rose. New #1 is `Luca` at 1322.6.
> - ✅ **THE CRUSTLE RE-RUN IS AUTHORISED (user, day 17).** Every verdict
>   carrying a Crustle term was measured against the pilot that threw games
>   (§8ah). ⚡ **It is now also forced rather than optional**: B8's five-anchor
>   sweep will carry a *repaired* Crustle term while every published number
>   carries the broken one, and comparing them would be apples to oranges at
>   6.7% of the field.
> - 📝 `STRATEGY.md` §7c is written; §6 was corrected; §4f now exists. The
>   standing one-edit-per-session rule is satisfied for day 16.

</details>

<details><summary>Day 16 (2026-08-02) — the anchor bug, the deck programme, and the health-check submission (complete; B8 above supersedes its plan)</summary>

### ▶ DAY 16 (2026-08-02; user-directed. The goal is to WIN)

> # 🔴 THE DAY-16 HEADLINE: THE USER WATCHED ONE REPLAY AND FOUND AN ANCHOR THAT WAS THROWING GAMES
>
> **Day-15 item 6 said the five anchors carry 71.5% of every weighted verdict
> here, were imported rather than written by us, and had never been watched. The
> user watched `out/replays/anchor_vs_anchor/game000` and reported the Crustle
> pilot never benched a second Pokémon and lost when its active was KO'd. It was
> correct.** `EVIDENCE` §8ah.
>
> `sources/crustle.py:338` scored **every Pokémon except Dwebble at −5000** for a
> bench play, with **no empty-bench guard**, so once the Dwebbles were gone it
> played on an empty bench until the first KO ended the match. ✅ **Fixed** —
> bench-full −5000, **empty bench 90000**, otherwise Dwebble 25000 / any 12000.
>
> | agent | games | EXPOSED turn-ends | empty-bench losses |
> |---|---|---|---|
> | **`rule:crustle` before** | 3 | **0.667/game** | **2 of 2 losses** |
> | **`rule:crustle` after** | 12 | **0.000** | 2 of 10 |
> | `rule:archaludon` | 12 | 0.000 | 1 of 3 |
> | `rule:alakazam5` | 18 | 0.000 | 0 of 11 |
> | `rule:lucario` | 12 | 0.000 | 0 of 5 |
> | **`bc:v5` (ours)** | 51 | 0.000 | **0 of 23** |
>
> 🔴 **CONSEQUENCE — EVERY VERDICT CARRYING A CRUSTLE TERM IS SUSPECT and must be
> re-run against the repaired pilot.** Our arena reads **0.663** vs `rule:crustle`
> against a **57.1%** real win rate; §8i filed that as "the arena reads
> optimistic" and this is a mechanism for part of it. ⚠ **NOT YET DONE — user
> has not authorised the re-run** (hours of compute, rewrites published numbers).
> At §8ac weights Crustle is 6.7% of the field, so the dilution is real.
>
> ⚡ **The methods lesson is about OUR detector, not the pilot.** The obvious
> screen — "bench empty, bench play offered, chose something else" — **overcounts
> and is not an error rate**: a pilot that plays three items and *then* benches
> scores three declines and did nothing wrong. On it, `rule:archaludon` looked
> **worse than Crustle** (1.333/game) and **is clean**. The sharp detector adds
> *"...and it ATTACKED or ENDED THE TURN anyway"*. **Fourth confident-but-wrong
> script in three days** (§8ad, §8ae, §8af, this) — ⚡ but the first one caught
> *before* reporting, by asking **"what is the benign reading of this count?"**
>
> ⛔ **AND THE RULE IT SUGGESTED FOR OUR OWN AGENT DIED BY SIZING** (§8ai). Empty
> bench is the most *dominated* option there is (rules go 3/3 there), so it was
> promoted as the best-shaped candidate in days — then sized: over 75 real ladder
> games and 7,094 decisions it fires **0.187/game** and matches **1 of 22
> losses**. The Morgrem out died at ~0.2 (§8e), Pokégear at 0.27 (§8ag). ~1.3% of
> games against an n=2000 A/B that resolves 2.1%. **Not built. Third sizing
> closure in three days.**
>
> #### 📋 The health-check submission — `55169114`, and it is decision-identical to v5
>
> ⚠ **It was submitted 2026-08-01 18:42 UTC and recorded in NO doc until now.**
> `dist/submission_bc-grimmsnarl-netspolicy_20260802-004209.tar.gz`. `diff -rq`
> against the v5 bundle: **only `main.py` and `sa/bcagent.py` differ**, and only
> by the health counters + one `print`. **Weights, deck, engine, every other
> module byte-identical. No decision path changed.**
>
> | submission | what it is | age | read 05:08 UTC | read 05:25 UTC |
> |---|---|---|---|---|
> | `55160229` | **v5** | 18.8 h | 956.5 | **951.0** |
> | `55169114` | **v5 + stdout counters** | 10.7 h | **874.8** | **874.8** |
> | `55156480` | v4 (frozen, evicted) | 22.2 h | 910.5 | 910.5 |
>
> 🔴 **≈ −80 points between two agents that make the same move in the same
> state**, both past the 4 h convergence window, read in the same call. If it
> holds it is **the project's first measured LB null**, and it is **double** the
> +40.5 that day 15 called "rule 2 satisfied on a net pair". ⚠ **Two readings
> only 17 min apart — rule 2 wants ≥1 h. A settling read is armed for ~07:25 UTC
> and **§8ak** is NOT to be written until it lands.** (§8aj was taken by the deck
> sizing, which concluded first.)
>
> ✅ **THE SUBMISSION-LOG TRACK IS CLOSED, and only one of its four goals paid.**
> Three episode logs collected:
> - ✅ **pool usage settled** — 318 selects, **1.12 s of a 1,800 s budget**, worst
>   call 0.153 s (startup). The "0.1 s of 600 s" claim is real.
> - ✅ **net is live, two ways in one file** — 86–98 net calls at **1.2–1.6 ms**
>   against 7–10 fallback-shaped calls at **23–37 µs**, a 50× separation.
> - 🔴 **Kaggle starts a FRESH PROCESS PER EPISODE** — all three logs read
>   `calls=1`, so the cumulative counter line can never print. The heartbeat fix
>   would cost a submission slot.
> - ⛔ **Nothing further is worth a slot.** The crash alarm already fires on the
>   next select (it does *not* wait for the heartbeat) and has reported zero over
>   3 games; `net_missing` belongs in a **build-time assertion**, not a log; and
>   the margin-distribution idea is obtainable **offline from replays we already
>   download**. Leave the counters in as passengers; never submit for logging.
>
> #### The day-16 order of work
>
> 1. ✅ `crustle.py:338` fixed · ✅ anchor pathology audit (`p24`) · ✅ empty-bench
>    sizing closed · ✅ EVIDENCE §8ah/§8ai · ✅ STRATEGY §4f written and §6's stale
>    day-9 shares corrected.
> 2. ⏳ **Ladder settling read (~07:25 UTC) → §8aj**, the identical-agent null.
> 3. ✅ **DECK WORK RAN — Track C has its first measurement-driven result** (§8aj).
>    `p25_deck_slot_audit.py` sized all 60 slots over 75 real games. ⚡ **Deck
>    swaps PASS the sizing gate that killed three rules this week**, because the
>    relevant frequency is the **draw** rate not the play rate — Tool Scrapper is
>    played 0.13×/game but **drawn in 81% of games**. 🔴 **And the obvious cut was
>    disqualified by the MATCHUP: Tool Scrapper is played 0.00 times per mirror
>    game** (our list runs no tools), so a mirror A/B would return "cutting it is
>    free" by construction — **rule 16 in deck clothing.** Tested instead
>    `Dawn ×1 → 4th Marnie's Grimmsnarl ex`, n=4,000 per arm:
>
>    | arm | score | 95% CI |
>    |---|---|---|
>    | CONTROL `grimmsnarl` v `grimmsnarl` | **0.4980** | [0.483, 0.513] |
>    | TEST `grimmsnarl_g4` v `grimmsnarl` | **0.4911** | [0.476, 0.507] |
>
>    **Δ = −0.0069 ± 0.0112, z = −0.61 — a clean NULL, exactly as pre-registered.
>    The list stands.** ✅ **The control is the keeper: 0.4980 is the project's
>    first same-deck variance floor for deck A/Bs** (the deck-side analogue of the
>    0.482 seed null), and it sits on 0.500, so the harness is unbiased.
>    ⚠ **First player is worth ~1 pp** (P0 0.510/0.513 vs P1 0.486/0.470) —
>    `arena.py` alternates seats, anything that doesn't reads a ~2 pp bias.
>    ⇒ **Next deck candidates, in order: Pokégear 3.0 ×1** (0.33/0.31, the other
>    slot weak in both populations, and §8ag left it open) — then **Tool Scrapper,
>    but ONLY against a tool-running anchor**, which needs the §8ah repair
>    validated first.
> 4. ✅ **DECK WORK IS DONE FOR THIS PHASE AND ITS METHOD IS RETIRED** (§8al).
>    Two more user-directed variants tested at n=8,000 each, against §8aj's
>    same-deck control of **0.4980 [0.483, 0.513]**:
>
>    | variant | swaps | score | vs control | p |
>    |---|---|---|---|---|
>    | `Dawn → 4th Grimmsnarl ex` | 1 | 0.4911 | −0.007 | 0.54 |
>    | `Poffin −1, Scrapper −1 → Budew ×2` | 2 | **0.4757** | **−0.022** | **0.021** |
>    | four-card reconfiguration | 4 | **0.4637** | **−0.034** | **0.0004** |
>
>    🔴 **Strength falls MONOTONICALLY with distance from the consensus 60.**
>    ✅ **And the charitable defence is ruled out** — over 6 recorded games
>    (`out/replays/budew_v2_watch/`, watchable in `notebooks/visualizer.html`)
>    Budew reached the **Active spot 3/6** and Itchy Pollen fired **4×**. The
>    mechanism works at the intended rate and the deck is worse anyway: **the
>    plan loses, not the execution.**
>    ⇒ **The consensus 60 is a local optimum and our net is tuned to it.**
>    ⛔ **THE GUESS-A-SWAP METHOD IS RETIRED** (user's own call, and the data
>    agrees): three single-card hunches → one null, two significant losses. **The
>    next deck programme needs a MATCHUP-STRATIFIED SEARCH DESIGN over the whole
>    slot ranking** — ⚠ all four A/Bs so far were **mirror-only**, which flatters
>    variants cutting mirror-dead tech (Tool Scrapper 0.00 plays/game there) and
>    cannot judge a card aimed anywhere else. **Do not run another one-off swap.**
> 5. ⏸ Re-run the Crustle-term verdicts against the fixed pilot — **user's call**.

</details>

<details><summary>Day 15's headline box (2026-08-01) — the rating-dependent field; still live, superseded only in its item-6 status</summary>

> # 🔴 THE DAY-15 HEADLINE: THE "META SHIFT" IS OUR OWN CLIMB, AND IT RE-PRICES EVERY WEIGHTED VERDICT IN THIS REPO
>
> ⚡ **RANK 185 / 6,103 AT 955.1 — our best ever, and ✅ RULE 2 IS SATISFIED FOR
> THE FIRST TIME ON A NET PAIR.** Two readings 61 minutes apart, and they agree:
>
> | submission | 16:06 UTC | 17:07 UTC | age at 2nd |
> |---|---|---|---|
> | **`55160229` v5** | **955.1** | **955.1** | 6.5 h |
> | `55156480` v4 | 914.9 | **914.6** | 9.9 h |
>
> **Both converged, both active, same time — the only comparison rule 2 permits,
> and this project has never had one for two nets before.** v5 is **+40.5**.
>
> 🔴 **It still does not adjudicate §8aa, and saying so is the point.** The arena
> put v5 at **+14 Elo** (**+13.8 re-weighted**, §8ac), and rule 2's second clause
> is that **the LB cannot resolve an effect that size at all** — it confirmed
> `chip_target` at ~150 points and could never have adjudicated `counter_source`
> at ~12. The ladder agreeing in *sign* at ~3× the magnitude is consistent with
> §8i's calibration (the arena ranks matchups right and reads optimistic) and
> with §8ab's compression caveat; it is **not** independent confirmation.
> ⚠ Minor: v5 read **exactly** 955.1 twice, which more likely means few rated
> games in that hour than a perfectly stable rating. v4 moved 0.3.
>
> **The supplied replays answered day-15 item 2, and the answer was not the one
> the item expected.** Pooled over v4+v5 (75 games) the field looks transformed
> since day 9 — the **mirror 13.8% → 33.3%** (Fisher **p=0.002**), **Mega Lucario
> 12.8% → 4.0%**, win rate 63.0% → 70.7%. **But hold the opponent-rating band
> fixed and every era difference vanishes** (all Fisher p ≥ 0.065, n=181 games
> over four dumps). What actually moved is **us**: mean opponent rating **799 →
> 867**, tracking our own 820 → 955.
>
> | archetype | opp <800 | 800–900 | 900–1000 | 1000+ |
> |---|---|---|---|---|
> | **mirror** | 5.3% | 18.6% | **42.4%** | **71.4%** |
> | Alakazam | 13.3% | 28.8% | 33.3% | 14.3% |
> | Crustle | 16.0% | 5.1% | 9.1% | 7.1% |
> | **Mega Lucario** | 17.3% | 6.8% | **0.0%** | **0.0%** |
> | **Archaludon** | 10.7% | 15.3% | **0.0%** | **0.0%** |
>
> 🔴 **The opponent pool is not a population we sample — it is a function of our
> own rating, and it moves when we do.** Every anchor weight in this repo carries
> an invisible parameter: the score we held when the census was taken. **Rule 16's
> sampling-frame trap, committed a second time, on our own data.** `EVIDENCE` §8ac.
>
> **Re-weighted, measurements untouched — only the shares change:**
>
> | verdict | day-9 weights | **day-15 weights** |
> |---|---|---|
> | §8i `v3 − P4b` | +35.6 | **+62.1** |
> | §8j **rules ON − OFF** | **+0.8** | **−18.1** 🔴 **sign flip** |
> | §8z `v4 − v3` | +23.4 | +24.8 |
> | §8aa `v5 − v4` | +10.2 | +13.8 |
>
> ✅ **Nothing shipped has to change** — the v5 bundle already pins
> `chip_targeting/energy_spread/counter_source = False` (verified by reading
> `main.py` out of the tarball). **§8j's "the rules are worth nothing" was the
> right call for ~18× weaker reasons than the true ones.**
>
> ⚡ **AND IT RESOLVES A STANDING CONTRADICTION INSTEAD OF CREATING ONE.** §8b
> (mined ≥1144 band) said **52.1% of seats play our archetype**; §8i (our games at
> ~820) said the mirror is **13.8%**; day 9 filed these as irreconcilable. **They
> are two points on one monotone curve and we have been walking up it.**
>
> #### What this changes for the remaining 16 days
>
> 1. ⛔ **Track C's Archaludon lead is CLOSED BY SIZING, before it was built**
>    (rule 14). Promoted on "10.1% of the field and our worst matchup"; it is
>    **8.0% overall and 0 of 47 games above rating 900**. Same for B3's Mega
>    Lucario instance (**4.0%**) and, more mildly, Crustle (6.7%).
> 2. ⚡ **THE MIRROR IS THE MATCHUP THAT MATTERS AND GETS MORE SO AS WE CLIMB** —
>    33.3% now, **51.1% above 900**, 71.4% above 1000. It is also what our
>    head-to-head net A/Bs already measure, so **our most sensitive instrument is
>    now also our most representative one.** Weight it accordingly.
> 3. ⚠ **Two archetypes have no anchor and now outrank two that do:** Cynthia's
>    Garchomp ex **6.7%** + Dragapult ex **5.3%** = 12.0%, against Crustle +
>    Lucario's 10.7%. `decks/dragapult_ex.py` already exists.
>
> ✅ **ITEM 4 IS BUILT AND VERIFIED: `harness.Recorder`.** Optional recorder on
> `play_game`; `visualize_data()` output is byte-compatible with Kaggle replays,
> so recorded local games are read **unmodified** by `p9_field_census.py`,
> `build_policy_dataset.py` et al., and watchable in `notebooks/visualizer.html`.
> `scripts/p20_record_games.py` (CLI), `scripts/p20_recorder_equivalence.py`
> (12/12 exact checks). 🔴 **Its first version was a test that could not have
> failed** — it demanded identical games run-to-run, but `battle_start` takes no
> seed. **Before trusting an equivalence test, ask what would have counted as
> success.** `EVIDENCE` §8ad.
> 📼 **Ready to watch:** `out/replays/v5_vs_alakazam`, `out/replays/anchor_vs_anchor`
> (⚠ one anchor-vs-anchor game ran **39 turns** against 11 and 11 — first thing to
> explain in the item-6 audit).
>
> ⚡ **ITEM 5 RAN, AND RL SURVIVED ITS OWN KILL CRITERION.** The probe cost zero
> new games — four archives already carry pairs of known separation.
> **Throughput 5.96 games/s per process ⇒ ~5.5M games to the deadline.**
> Detecting §8z's +37 Elo from outcomes takes **800 games (0.015% of budget)**;
> resolving a **1-percentage-point** effect at a single select's context takes
> **960 games** if that context recurs ~20×/game (201 selects/game).
> 🔴 **So the credit-assignment objection — the last one standing after §8x
> narrowed the encoding argument — does not bind. It dies with a NUMBER, which
> is what §2 never had.** `EVIDENCE` §8ae.
> ⛔ **This is NOT a licence to build.** B4 passed all three of its kill criteria
> and then died at n=200. The model prices one context in isolation and ignores
> non-stationarity and shared parameters; training cost is not priced at all;
> and the nearest real measurement (`--winners-only` 0.375) still points the
> wrong way. **The next step is the smallest real thing: fine-tune a SMALL
> parameter set on our own recorded outcomes, A/B at n≥2000 vs a byte-identical
> control with the seed floor carried in.**
> ⚠ **Its first run was garbage in an instructive way:** arena archives are
> **seat-indexed and the seats swap every game**, so reading seat 0 as agent A
> averaged both agents together. It reported +37 as undetectable and +14 as
> detectable. **A bug that biases everything toward the null looks like a
> finding, not a crash.**

</details>

> ## 📍 THE SITUATION AT THE TOP OF DAY 15
>
> - ⏸ **DAY 14 WAS DELIBERATELY IDLE — nothing ran, by user instruction.** Both
>   submissions were left to play for 8–9 h so their replays could be downloaded.
>   **Day 14's item B (the centred option encoding) and item C (Track C deck
>   work) were NOT executed and are NOT cancelled** — they are parked below.
> - 📥 **THE USER IS SUPPLYING THE INPUTS.** Expect at the start of the session:
>   **(a) the replays of `55160229` (v5) and `55156480` (v4)** from their 8–9 h of
>   play. ⚠ That is our own-opponent census data (`p9_field_census.py --us`) for
>   two agents at once, and the **first replay set for the v4/v5 feature blocks.**
> - 🔴 **A NEGATIVE RESULT WAS RETRACTED — READ IT BEFORE PLANNING RL.**
>   "Self-play RL" has been struck from the settled-negative list in all four
>   docs **and from the assistant's memory file**: it was **never run.** No code,
>   no `n`, no CI — a **compute prior inherited from the search result**, filed
>   beside the measured negatives for twelve days. **Rule 15, third instance, and
>   this time the unmeasured claim was living inside `EVIDENCE.md` itself.**
>   ✅ Verified against the old repo too (`E:\Kaggle\pokemon-tcg-simulation`):
>   **no RL code, no training script, no reward function** outside `.venv`.
>   ⇒ **The status is "never attempted", not "dead".** `EVIDENCE` §2's box.
> - ⚡ **AND §8w'S GATE AGAINST RL IS SUBSTANTIALLY SATISFIED.** Its argument —
>   a policy gradient reads the same vectors, so bitwise-identical options get
>   identical gradients — was **narrowed by §8x the next day**: the tie ceiling is
>   **95.6%** against a clone at **71%**, so the encoding binds **at most 4.4 pp**,
>   and the ties that exist are two copies of one card in one role (free choices).
>   §8w named the feature audit as RL's **prerequisite**; it has now been done
>   **twice** (§8y/§8z, §8ab).
> - 🔴 **SO THE LIVE OBJECTION IS NEITHER COMPUTE NOR EXPRESSIVENESS — IT IS
>   CREDIT-ASSIGNMENT VARIANCE**, the same term that killed search (terminal 0/1
>   ⇒ SE ≈ 0.14; the max over ~9 rivals sits 0.21–0.28 above truth by chance).
>   One binary reward over a ~40-turn game with hundreds of selects.
>   **Rule 14 binds: SIZE IT BEFORE BUILDING IT.** The nearest measurement is
>   unfriendly (`--winners-only` **0.375**, §1) but is **not the same mechanism** —
>   that filtered *other people's* games by outcome and discarded half the corpus;
>   a gradient signed on *our own* trajectories does neither.
> - ⚡ **THE PLUMBING FOR ITEMS 2–4 MAY ALREADY EXIST, IN THE OLD REPO.**
>   `notebooks/how-to-output-local-battle-as-json-and-view.ipynb` +
>   `notebooks/visualizer.html`: the engine's own **`cg.game.visualize_data()`**
>   emits a replay the **official viewer** (`ptcgvis.heroz.jp`) renders, and the
>   notebook captures an **obs log + action log** in the *same*
>   `battle_start`/`battle_select` loop **our `harness.py:48-75` already runs**.
>   ⇒ **One optional recorder on `play_game` yields BOTH the human-watchable
>   replay AND the RL/exploration trajectories.** This is a contained change to
>   one function, not a build.
>
> **Deadlines: sim closes 2026-08-17 (16 days). Report due 2026-09-14 (44 days).**
> **Rubric: Model 70% (LB is ONE bullet of five) + Deck 20% + writing 10%.**

#### The day-15 order of work (user-set; items 1–4 are theirs, 5–6 are the parked engineering)

**1. 📈 READ THE LADDER FIRST, TWICE, ≥1 h APART — and this time it is a real
   question, not a ritual.** After 8–9 h **both** submissions are converged, so
   `55160229` (v5) vs `55156480` (v4) is finally a **same-time, both-active,
   both-settled** comparison — the only kind rule 2 permits. ⚠ **It still does
   not adjudicate §8aa** (+7.3 weighted, far under the LB's ±50–100), but it is
   the first honest live read on the pooled block.

**2. 📥 INGEST THE SUPPLIED REPLAYS.** `p9_field_census.py --us` on each
   submission separately. Two questions worth more than the census: **has our
   field composition moved** (it drives every anchor weight in this repo), and
   **does v5 face a different field than v4** (it should not — same deck; if it
   does, the census is measuring rating band, not deck choice).

**3. 🔬 SUBMISSION LOGS — instrument the agent.** ✅ **SUB-ITEM 1 IS BUILT AND
   READS CLEAN.** `sa/bcagent.STATS` + `health_line()`: counters for calls,
   catch-all `fallbacks`, `net_missing`, and the first traceback verbatim.
   Free on the happy path (one dict increment against ~1 ms of decision time),
   one line per game, never per-decision spam. **Measured locally over 733
   selects: `OK calls=396 fallbacks=0 net_missing=0` (v5) and `calls=337`
   (v4)** — 🔴 **the first DIRECT confirmation the net is live**, where §8g
   could only argue it from a 40.7% index-0 rate against the 100% a real
   fallback would show.
   ⚠ **Still to do: nothing prints it yet on Kaggle.** `build_submission.py`
   must emit `health_line()` once per game for the log to exist, and that costs
   a submission slot to verify — **the user's call, not mine.**
   🔴 **SUB-ITEM 2 IS MOOT AND SHOULD BE STRUCK:** "rule firing rates in the
   wild" cannot be measured, because the shipped bundle pins
   `chip_targeting/energy_spread/counter_source = False`. **There are no rules
   firing.** §8ac's re-weighting makes that pinning look better, not worse
   (rules are −18 Elo at the real weights), so the sub-item dies rather than
   becoming urgent.
   3. **pool usage** — we claim 0.1 s of 600 s; confirm it on the real harness;
   4. ⚠ **"cite a reason per action" is the expensive one** — ~1 ms/move ×
      thousands of selects is a lot of stdout, and **the log size cap and
      retention are UNKNOWN and must be checked before designing a format.**
      Cheap 90%: log the net's **top-1 logit margin** + a one-byte code for who
      decided (net / which rule / fallback), aggregated per game. That also buys
      the **margin distribution on real ladder states vs arena states** — a
      covariate-shift instrument for free.
   ⚠ **Design rule: compact per-game summary + rare event lines, never
   per-decision spam.**

**4. 🎬 THE TRAJECTORY RECORDER — build this before 5 and 6; it unblocks both,
   plus the user's own inspection.** 🔴 **`arena.py` archives ONE SUMMARY ROW PER
   GAME** — winner/turns/selects/latency/pool (`scripts/arena.py:281-294`).
   **No observations, no actions, no trajectories.** So today there is nothing to
   watch and nothing to learn from. Add an optional recorder to
   `harness.play_game` that (a) accumulates `obs` + chosen action per select and
   (b) calls `game.visualize_data()` before `battle_finish()`. **Port the old
   repo's `visualizer.html` so the user can watch games in the official viewer.**
   ⚠ Keep it **opt-in** — the A/B path must stay byte-identical and fast, and per
   §8aa's methods rule, **if this is meant to be a no-op for existing runs, prove
   it with an equivalence test, not with the arena.**

**5. 🤖 RL — SIZE THE VARIANCE FIRST, DO NOT BUILD.** The user's framing is
   **fine-tuning an already-decent clone on its own outcomes**, which is a
   different cost regime from the league self-play §0 declined — and §0 only ever
   considered the from-scratch version. ⚠ **Also correct the record with the
   user's own recollection**: RL did not fail against rule agents; **what matches
   that description is `search`.** The pre-registered probe, before any training
   code: **with the item-4 recorder, measure how many games the terminal-outcome
   signal needs to separate two policies of KNOWN Elo separation** (we have
   several — v4 vs `v4ctrl` at +37, the `no3` ablation at −36, and a **measured
   seed-only null at ±13**). **Kill criterion: if separation needs more games
   than ~1.4 cores can produce in the remaining days, RL dies for a few CPU-hours
   instead of a week — and it dies with a NUMBER**, which is the thing §2 never
   had. ⚠ **And whatever the probe says, it is a report chapter**: a retracted
   negative, re-derived honestly, is exactly §5's material.

**6. 🕵️ AUDIT THE ARENA OPPONENTS (user wants to watch first, then I analyze).**
   The five anchors carry **71.5% of every weighted verdict in this repo** and
   they were **imported, not written by us — nobody on this project has ever
   watched one play.** Gated on item 4. ⚠ Relevant to item 5 too: our anchors are
   rule pilots and our own nets, so **which opponent we generate exploration
   against decides what the data can teach.**

**PARKED FROM DAY 14 — not cancelled, and item B has a closure condition:**
   - **B. The centred option encoding** — append `opt_enc − mean(opt_enc)` to
     each **option** rather than pooling into the state (§8aa pooled on the
     *state* side, where the summary must survive the state MLP before it can
     affect a ranking; centring puts the comparison directly in the vector the
     head scores). ~20 min of compute. **If it lands ≤ +15 Elo, DECLARE THE
     FEATURE AXIS CLOSED** and write it up as a three-generation
     diminishing-returns curve (+115 → +37 → +14) — a better chapter than a
     fourth null.
   - **C. Track C deck work — 20% of the rubric and NOW FOUR SESSIONS UNTOUCHED.**
     Only one decklist variant has ever been A/B'd (0.490, null).
     🔴 **ITS CONCRETE LEAD DIED ON DAY 15, BY SIZING, BEFORE ANYTHING WAS
     BUILT — and that is rule 14 working, not a setback.** The lead was:
     Archaludon runs **Full Metal Lab ×4 (card 1244)**, we run **Spikemuth Gym
     ×4 (1259)**, and `WALL_POKEMON = {345}` models neither. It was promoted on
     *"Archaludon is 10.1% of the field and our worst real matchup"*. **At our
     current rating Archaludon is 8.0% of the field and 0 of 47 games above
     rating 900** (§8ac) — the tech would serve a band we are leaving. ⛔ Do not
     build it. **Same sizing kills B3's Mega Lucario instance (4.0%, also 0/47
     above 900).**
     ⚡ **What Track C should aim at instead, in order:**
     1. **The MIRROR — 33.3% of our field, 51.1% above rating 900, and rising
        with every point we gain.** A deck edge in the mirror is worth more than
        one anywhere else on the board, and it is the matchup our A/Bs measure
        best. Nobody has ever asked what beats our own 60.
     2. **Cynthia's Garchomp ex (6.7%) and Dragapult ex (5.3%)** — 12.0%
        together, more than Crustle + Lucario, and **neither has an anchor**.
        `decks/dragapult_ex.py` already exists; the pilot notebook is in
        `notebooks/`. Build the anchor before the deck opinion (rule 12).
     3. **The stewardship write-up is owed either way** — "we measured a change
        and kept the list" is deck analysis, and this closure is exactly that.
   - **D. `report/STRATEGY.md` — one edit per session, minimum.** §6 (opponent
     modelling) is still *in progress*; §8 needs the v5 entry. ⚡ **Day 14 already
     handed it a chapter for free: the self-play retraction belongs in §5's
     process-failure section**, next to the three failures already written there.

<details><summary>Day 14's plan as it was set at the end of day 13 (superseded — the day was idle by instruction; B and C are parked above)</summary>

> ## 📍 THE SITUATION
>
> - ⚡ **RANK 268 / 6,088 AT 923.0 — our best live number ever, and it is still
>   climbing.** `55156480` (the v4 state block) read **489.3 → 853.4 → 822.3 →
>   894.7 → 923.0** over its first 3 hours. **We were 465/6,075 at 864.1
>   yesterday.** Top is Majkel1337 1251.3, then Sixth Sense 1181.7.
>   ⚠ **923.0 IS NOT A SETTLED NUMBER** — rule 2 wants two *agreeing* readings
>   ≥1 h apart and these disagree because the agent is still converging. The
>   P4b restore took 4 h. **Re-read before quoting it.**
> - Active pair: **{`55156480` v4 923.0, `55129730` P4b 836.4}**. v3 is evicted
>   and frozen at 864.1. **v4 has now beaten every number this project has
>   produced except the original P4b's board-inflated 952** (§8p).
> - 🔴 **AND THE LADDER DOES NOT ADJUDICATE §8z** — v4 was +16.5 Elo weighted,
>   far below the ±50–100 the LB resolves. **A 60-point climb is not evidence
>   the block works**; the arena at n=4,000 with a seed control already answered
>   that, and this is a board that moved 3,000 → 6,088 entrants in a week.
> - ⚡ **v5 WAS SUBMITTED AS `55160229`** (`dist/submission_bc-grimmsnarl-netspolicy_20260801-163829.tar.gz`,
>   `NET_OK opt_in=37 state_in=708` — 536 + the 172-wide pool, so the block is
>   live in the bundle and not silently sliced off; sha verified against
>   `out/policy_v5.npz`). Active pair becomes **{`55160229` v5 climbing from
>   μ=600, `55156480` v4}**; **P4b is evicted.**
> - 🔴 **AND THE REASONING WAS CORRECTED MID-SESSION — read this, it is a
>   decision-framing error, not a new measurement.** The first verdict was "do
>   not submit: +7.3 weighted, negative on 2 anchors of 5, wrong shape". That
>   answers **"is v5 better than v4?"** (no) when the question a submission
>   actually asks is **"is v5 better than what it EVICTS?"** Eviction is by
>   recency, so v5 displaced **P4b — 836.4 against v4's 908–923, dominated on
>   the displayed score (best ACTIVE) and last of everything we own in the
>   arena (§8k).** v4 keeps its rating and stays active throughout, so the
>   displayed score cannot fall. ⇒ **The +50 bar was written when slots were
>   scarce and every submission evicted something valuable; the user relaxed
>   exactly that premise, and the bar was still being applied to the old one.**
>   ⚠ **Standing correction: before quoting the bar, name the agent the
>   submission would EVICT.** A candidate that loses to our best can still
>   dominate our worst.
>
> **Deadlines: sim closes 2026-08-17 (16 days). Report due 2026-09-14 (44 days).**
> **Rubric: Model 70% (LB is ONE bullet of five) + Deck 20% + writing 10%.**

#### ✅ Done on day 13 — two results, and together they are the best report material the project has

- **§8aa — the v5 pooled option-set block: the deep-sets fix, and it BARELY
  PAYS.** Every option is scored independently against one shared state vector,
  so the net has never seen the option *set*. Mean/max pool of the option
  encodings + count scalars, appended after the v4 block (`--pool`).
  **Agreement 71.0% → 72.7% — 214 more correct decisions of 12,939, the largest
  agreement gain this project has ever produced — for +14 Elo pooled over two
  seeds**, one noise-width, mixed-sign across the anchors.
- 🔴 **THE PAIR IS THE FINDING, and it is now measured in both directions:**

  | intervention | Δ agreement (of 12,939) | Δ Elo | Elo per decision |
  |---|---|---|---|
  | **v4 state block** (§8z) | **+8** | **+37** | 4.6 |
  | **v5 pooled option set** (§8aa) | **+214** | **+14** | **0.07** |

  **The exchange rate between fit and strength differs 70× between two
  interventions run a day apart on the same corpus.** ⇒ **`val_top1` is not a
  screening metric in either direction. Nothing may be promoted or killed on
  it.** This is rule 3 with both signs paid for.
- ⚡ **§8ab — the v4 ablation, and it validates the METHOD rather than the
  block.** `--drop-x` zeroes a member's columns (identical arch, params, init,
  rows, seed; `x_mask` stored in the npz so inference matches training):
  - **Drop any ONE of `turnActionCount` / stadium / effect card → within noise**
    (0.527, 0.526, 0.483 vs full v4).
  - 🔴 **Drop all THREE → 0.449 [0.427, 0.470], −36 Elo, disjoint.** They are
    **mutually redundant and jointly necessary** — and they are essentially the
    whole +37.
  - ⚡ **The five leftover members alone are WORSE THAN NO BLOCK AT ALL**
    (0.469 vs `v4ctrl`, −22 Elo, disjoint). **The three that went through §8y's
    sizing step carry everything; the five that skipped it are negative.**
    ⇒ **Derive and size. Do not bundle.**
- ⚠ **A caveat that touches every weighted table in this repo:** head-to-head
  Elo among these nets **orders consistently but compresses ~23 points over two
  hops** (v4−ctrl +37, ctrl−no3 +22, v4−no3 measured **+36** against an additive
  +59). **Weighted five-anchor totals are ordinal, not arithmetic.**
- 🔧 **A methods rule bought the hard way (§8aa's last section).** The refactor
  that enabled the pool moves the option encoding ahead of the state MLP *for
  every net*. A regression A/B read **0.503** where §8z had 0.567 — which would
  have meant the live net was broken. **The arena cannot settle that: it is not
  deterministic run to run.** A direct equivalence test (load the pre-edit
  module from git, same observations, compare scores) said **max |old − new| =
  0.000e+00 over 588 selects**. ⚡ **When a refactor is supposed to be a no-op,
  prove it with an equivalence test, not the noisy end-to-end instrument.**
- **Report:** `STRATEGY.md` §4c (audit-by-enumeration + the ablation), §4d
  (§8z's decoupling), §4e (§8aa's converse). Three new chapters.

#### The day-14 order of work

**A. 📈 READ THE LADDER FIRST, TWICE, ≥1 h APART.** v4 is mid-climb at 923.0.
   The question is where it settles — **against 864.1, which is what v3 reached
   on the same board size.** ⚠ Do not quote 923.0 as converged, and do not read
   a rank change as evidence about the v4 block (rule 2).

**B. 🔬 THE FEATURE AXIS IS NARROWING — one concrete lead left, then stop.**
   Three generations: option binding **+115 Elo** (§8f), state block **+37**
   (§8z), option-set pool **+14 and mixed-sign** (§8aa). **The returns are
   falling by roughly 3× a generation and the next one lands under the noise
   floor.** The single untested variant worth one day:
   1. ⚡ **The CENTRED option encoding** — append `opt_enc − mean(opt_enc)` to
      each **option** rather than pooling into the state. §8aa pooled on the
      *state* side, where the summary must survive the state MLP before it can
      affect a ranking; centring puts the comparison directly in the vector the
      head scores. **Different mechanism, same cheap append-and-slice, ~20 min
      of compute.** If it also lands ≤ +15 Elo, **declare the feature axis
      closed and write it up as a three-generation diminishing-returns curve** —
      which is a better chapter than a fourth null.
   ⛔ **Do NOT re-open capacity (§8w), demonstrator selection (§8u), data volume
   (§1) or search (§2).** Six axes are dead; this is the seventh probe of the
   one that lives, and it is nearly spent.

**C. 🃏 TRACK C DECK WORK — now the largest untouched item on the board.**
   20% of the rubric, and **only ONE decklist variant has ever been A/B'd**
   (0.490, null). It has not been reached on days 12 or 13. **Promote it to
   first item if B's lead does not land.** Concrete starting point found on
   day 13: **Archaludon runs Full Metal Lab ×4 (card 1244), a stadium**, and we
   run **Spikemuth Gym ×4 (1259)**. Playing ours removes theirs, and
   `WALL_POKEMON = {345}` does not model Full Metal Lab's damage reduction at
   all. **Audit before rule (rule 14):** how often do we hold Spikemuth Gym
   while Full Metal Lab is in play? Archaludon is 10.1% of the field, our worst
   real matchup (45.5% over 11 games), and the anchor v4 *and* v5 both barely
   move (+7, −6).

**D. 📝 `report/STRATEGY.md` — one edit per session, minimum.** §6 (opponent
   modelling) is still marked *in progress* and §8's negative-results list now
   needs the v5 entry. Day 13's three chapters are written.

</details>

<details><summary>Day 13's plan and situation (superseded — all four items ran; A/B/D done, C not reached and re-stated above)</summary>

> ## 📍 THE SITUATION (as of day 13's start)
>
> - **Rank 465 / 6,075, score 864.1** (two readings 12:02 and 13:03 BST: 869.7
>   then 864.1 — agreeing, rule 2 satisfied). Top is **Majkel1337 at 1300.6**,
>   which is 135 points clear of the old top and is new.
> - 🔴 **THE ACTIVE-PAIR FACTS IN THE OLD BOX WERE STALE AND BACKWARDS.** Live
>   per-submission scores read 08-01: **`55116557` v3 = 864.1** (our best, and
>   still climbing) and **`55129730` P4b-restore = 824.3**. ✅ **That is §8i's
>   arena prediction confirmed on the ladder** — the sweep said v3 +36 Elo over
>   P4b, the ladder says +40, both active, both converged. **The day-9 "B1 lost
>   130 points" story is now fully inverted.**
> - ⚡ **A NET WAS SUBMITTED ON DAY 12 — the first since 07-31: `55156480`**,
>   the v4 state block (`dist/submission_bc-grimmsnarl-netspolicy_20260801-131057.tar.gz`,
>   `NET_OK opt_in=37 state_in=536`, sha verified against `out/policy_v4.npz`).
>   It **evicts `55116557` (v3, 864.1)**; the active pair becomes {`55156480`
>   climbing from μ=600, `55129730` P4b 824.3}, so **the displayed score will
>   DROP to ~824 for ~4 h** — expected, not a regression. It read **489.3** at
>   ~5 minutes old, which is **7 games of TrueSkill and means nothing** (the P4b
>   restore went 600 → 715.9 → 833.9 over 4 h). §8z.
> - ✅ **DAY 12 BROKE THE PLATEAU, and it did it on the one axis that has ever
>   worked.** The v4 state block beats its own byte-identical control
>   **0.567 [0.545, 0.588] n=2000**, replicates at a second seed
>   (**0.539 [0.518, 0.561]**), against a **measured seed-only null
>   (0.482 [0.460, 0.504])**. Pooled **≈ +37 Elo**. Better on **5 anchors of 5**.
> - 🔴 **AND IT MOVED HELD-OUT AGREEMENT BY EIGHT DECISIONS OUT OF 12,939.**
>   Rule 3's converse, measured for the first time: **the agreement metric the
>   whole B7 programme rested on is blind to a 37-Elo intervention** (§8z).
> - ✅ **The deck is NOT the bottleneck** (§8o); "clone better demonstrators" is
>   not the lever (§8u); capacity is not the lever (§8w). **State features are.**
>
> **Deadlines: sim closes 2026-08-17 (17 days). Report due 2026-09-14 (45 days).**
> **Rubric: Model 70% (LB is ONE bullet of five) + Deck 20% + writing 10%.**
> **Winning is not the same as ranking.** Read §0 of `ROADMAP.md` before deciding
> that a rank point is worth more than a report chapter — the competition
> description says outright that a mid-tier LB with deep analysis can win.

#### ✅ Done on day 12 (read before planning day 13)

- **§8x — the encoding ceiling, computed rather than argued.** Bitwise-identical
  options get identical logits from any net, so `Σ(1/g)/N` bounds top-1 for this
  layout: **95.6%, against the clone's 69.8%.** So §8w's "the residual is the
  encoding" **cannot** mean the answer is inexpressible — un-expressibility is at
  most 4.4 of the 30.2 points. ✅ And every tie is **two copies of one card in one
  role**, so `context_accuracy.py --equiv` now counts those as hits: honest
  agreement is **71.0%**, TO_HAND **67.1%** not 61.2%.
- **§8y — the feature audit BY ENUMERATION** (`p18_missing_state_audit.py`), and
  it **retracted the candidate list three files had carried since day 10**: turn
  number, prizes and both hand counts are all encoded already (`features.py`
  88–99). **Rule 15, second instance — caught before anything was built.** Two
  more died on sizing (`remainDamageCounter` constant at 100% of decisions,
  `remainEnergyCost` at 99.1%).
- **§8z — the v4 state block: BUILT, MEASURED, REPLICATED, SUBMITTED.**
  `turnActionCount` + the select's **effect card** + the **stadium** + `retreated`
  / `stadiumPlayed` + tool counts + bench cap + pool size. Corpus `pds_v4` is
  **byte-identical to `pds_v3r`** on every pre-existing array, and `--no-extra` is
  the control on those identical rows.
- **A noise floor exists now.** `train_policy.py --seed`; two identical-recipe
  controls at different seeds measure **0.482 [0.460, 0.504]** — a null. **Every
  net-vs-net number in this repo previously had an unmeasured confound.**
- **Report:** `STRATEGY.md` §4b (the ceiling, new), §8's capacity bullet narrowed.

#### The day-13 order of work

**A. 📈 READ THE LADDER FIRST, TWICE, ≥1 h APART.** `v4` needs ~4 h to converge.
   ⚠ **The expected path is DOWN then UP**: displayed drops to ~824 (P4b) while
   v4 climbs from 600. **Do not react to the dip.** The question that matters is
   where v4 settles against **864.1**, the number v3 reached.
   🔴 **And whatever it reads, it does not adjudicate §8z** — +16.5 Elo weighted
   is far below the ladder's ±50–100 (rule 2). It was submitted because it is
   better on 5/5 anchors and slots are not scarce, not because the LB can see it.

**B. 🔬 THE FEATURE AXIS IS LIVE AGAIN — WORK IT, it is the only one that has
   ever paid, and it has now paid TWICE (§8f, §8z).** Two concrete leads, both
   derived rather than guessed:
   1. ⚡ **THE NET NEVER SEES THE OPTION SET.** Every option is scored
      independently against a shared state vector, so it cannot know whether it
      is choosing among 3 Trainers or 40 deck cards. That is *why* the effect
      card paid (§8y). **The direct fix is a pooled summary of the option
      encodings (mean/max) concatenated into the state** — deep-sets, one extra
      block, the same append-and-slice trick. **This is the declined appendix's
      Set-Transformer plank in its cheapest possible form**, and §8z is the first
      evidence it would pay.
   2. **The v4 block was shipped whole; nobody knows which member did the work.**
      An ablation (drop `turnActionCount` alone, drop the effect card alone) is
      two trainings and two A/Bs, and it is a report table either way.
   ⚠ **Carry the noise floor in**: ±13 Elo between seeds. **Any ablation arm must
   clear that**, so run each at n=2000 and prefer two seeds.

**C. 🃏 Track C deck work FOR DECK SCORE (20% of the rubric, still untouched).**
   Only **one** decklist variant has ever been A/B'd (0.490, null). Re-aim at
   **Archaludon** — it is the one anchor where v4 barely moved (+7 Elo) and our
   worst real matchup (45.5% over 11 games). Full Metal Lab is a second
   damage-reduction effect `WALL_POKEMON = {345}` does not model.

**D. 📝 `report/STRATEGY.md` — one edit per session, minimum.** Day 12 handed it
   two strong chapters that are **not** written yet: **§8z's decoupling** (an
   intervention worth 37 Elo that moves agreement by 0.06 pp — this is the
   sharpest statement of rule 3 the project has) and **§8y's method** (a feature
   audit done by diffing the observation against the code, which retracted a list
   three files were asserting).

> ✅ **A, B and D RAN ON DAY 13; C did not and is re-stated as day 14's item C.**
> A: the ladder was read four times and v4 climbed to **923.0, rank 268/6,088**.
> B: **both** leads ran — the pooled option set (§8aa, **+14 Elo for the largest
> agreement gain in the project**) and the ablation (§8ab, **the three derived
> members are jointly necessary and the five unsized extras are negative**).
> D: three chapters written (§4c, §4d, §4e).
> ⚠ **Item B's framing was right and its prediction was wrong in a useful way**:
> it called the pooled summary "the first evidence it would pay", citing §8z.
> It moved the *fit* more than anything ever has and bought a noise-width of
> strength — **which is exactly the decoupling §8z had just demonstrated, applied
> in the opposite direction and not anticipated.**

</details>

<details><summary>Day 11's order of work (superseded — all five items ran)</summary>

#### ✅ Done on day 11

- **Item A shipped: every corpus row carries its demonstrator's LB rating.**
  `build_policy_dataset.py --ratings` (+ `--exclude`, `--aliases`), 94–98% seat
  coverage. ⚠ **The first build silently lost 24.6% of d26 seats and 182 of the
  198 misses were ONE team — the LB's #1, appearing as `James Cox`, as
  `zoroark190` (a member username) and as the merged `James Cox & Henry Chao`.**
  Fixed by exact member-username matching plus a hand-verified
  `replays/team_aliases.tsv`. **A census keyed on a display name splits your
  most valuable demonstrator into three.**
- **`p15_rating_curve.py`** (agreement vs rating, with a `--seen-from` exposure
  control) and **`p16_policy_disagree.py`** (the covariate-shift discriminator).
- **`p9_field_census.py --us / --emit-players`** — censuses any named seat's
  opponents and writes the same-deck ones to a `--players-file`. **This makes
  the day-10 control population reproducible**; the original 48-name list was
  never on disk.
- **`train_policy.py --rating-temp / --rating-min / --init`** — per-row weighted
  listwise loss (ESS reported before training) and warm-start fine-tuning.
- **Three concluded experiments: §8r, §8s, §8t/§8u.** Both B7 arms killed
  against a bar set before the first run.

#### The day-12 order of work

**A. 📝 `report/STRATEGY.md` IS NOW THE HIGHEST-VALUE WORK IN THE PROJECT, and
   that is a measured claim rather than a consolation.** The LB is inside its
   own noise band (§8k, confirmed on the ladder by §8p), five training axes are
   dead, and the report is 30%+ of the rubric before counting the soundness /
   consistency / robustness bullets inside Model's 70%. **Day 11 alone handed it
   a genuinely publishable result**: *agreement with a demonstrator measures
   distance from the fitted mode, not skill* — with a peak, a zero-exposure
   control group, a symmetry test against covariate shift, a positive control at
   1.7%, and two pre-registered interventions that failed in the predicted
   direction. **Write §7b.1/§7b.2 up properly and start §6 (opponent modelling)
   from §8r.** One edit per session remains the floor, not the target.

**B. 🔬 THE FEATURE AUDIT IS NOW THE WHOLE ENGINEERING TRACK, and §8w promoted
   it from "the only axis that ever paid" to "the only axis with a live
   mechanism".** Capacity is ruled out (8.2× params, no gain), demonstrator
   selection is ruled out (§8u), data volume was already dead (§1) — **by
   elimination the residual is what the option encoding cannot bind.**
   The day-10 list is still unworked and is still the best candidate set:
   read `agents/sa/optfeat.py` and `features.py` against
   `context_accuracy.py`'s **MAIN misses (3,930 of 6,424)** and ask §8f's
   question — is the input **absent** (informational) or **present but
   unbindable** (representational)? Only the second has ever paid.
   ⛔ **The cheap-candidate list this item carried is RETRACTED (§8y): opponent
   hand size, prizes remaining and turn number are ALL already encoded**
   (`features.py` lines 88–99); only the stadium was really absent.
   `scripts/p18_missing_state_audit.py` now derives the list by diffing the
   observation against what `featurize()` reads, and sizes each candidate.
   ⚡ **§8s gives this a new instrument**: `p16_policy_disagree.py` names the
   contexts where a stronger policy actually diverges from ours — **MAIN 45.8%,
   DAMAGE_COUNTER 30.5%, SWITCH 27.6%** — so the feature audit now has a
   ranked target list derived from a 1163-rated policy rather than from guessing.

**C. 🃏 Track C deck work FOR DECK SCORE (20% of the rubric, and untouched).**
   §8o closed it as a rank lever, which is exactly why it should now be done
   honestly as deck analysis: only **one** decklist variant has ever been A/B'd
   (0.490, null). Re-aim it at our real worst matchups — **Archaludon (45.5% of
   11 real games)** and **Mega Lucario (50%)** — both of which now have anchors.

**D. ⛔ DO NOT SUBMIT.** Nothing clears **+50 Elo weighted over the five
   anchors**; the two newest nets are **−55** and **−92**. Every submission
   evicts a live agent (§8h). ⚠ **And do not submit `b7_ntum` "to see what the
   ladder says"** — that trade spends a slot and evicts a live agent to test a
   net the arena puts 92 Elo down, against an instrument that resolves ±50–100.

> ✅ **ALL FOUR RAN ON DAY 12.** A: two chapters written (§4b + the §8 narrowing).
> B: the audit was done and it **retracted its own candidate list** (§8y), then
> the derived replacement measured **+37 Elo pooled** (§8z). C: not reached —
> re-stated as day 13's item C. D: **deliberately broken, with the reasoning
> written down in §8z** — v4 is below the +50 bar at +16.5 weighted, and was
> submitted anyway because it is better on 5/5 anchors and the user relaxed
> submission scarcity. **The bar was re-priced, not the evidence.**

</details>

<details><summary>Day 10's order of work (superseded — item B ran and is closed by §8u)</summary>

#### ✅ Done on day 10

- **`55129730` P4b restore: 833.9 at 4.0 h vs the original's 958.2 at ~4 h.**
  Question closed by experiment, for the price of one submission. §8p.
- **Two expert dumps ingested and verified** (`replays/sixth_sense_31-07-2026`
  227 games, `replays/ntumlnoob_31-07-2026` 330 games). Both teams play our exact
  60. `selected` is present for third-party seats, so **the BC pipeline works on
  them unchanged**.
- **`build_policy_dataset.py --player / --players-file`** — builds a corpus from
  named seats only. Corpora on disk: `artifacts/pds_expert` (Sixth Sense, 19,107
  rows), `artifacts/pds_ntum` (ntumlnoob, 27,318), `artifacts/pds_grimm_ctrl`
  (48 other grimmsnarl pilots, 10,498 — **the same-deck control**).
- **`context_accuracy.py --all-rows`** — scores every row, not the trainer's
  `gid % 20` split. **Required for a corpus the net never trained on**; without it
  you silently score 5% of the data.
- **The agreement-vs-rating result** (§8q), plus two explanations killed:
  familiarity (`haggle`: 0 corpus seats, 75% agreement) and one-team idiosyncrasy
  (the ntumlnoob dump was fetched to break exactly that tie).

#### The day-11 order of work

**A. ⚡ TAG EVERY CORPUS ROW WITH ITS DEMONSTRATOR'S LB RATING — do this first,
   it gates B and C.** Team names are in every replay's `info.TeamNames`; the
   full leaderboard is one `competition_leaderboard_download` call. Store a
   per-row `rating` (and `submissionId` where the dump has `episodes_meta.json`).
   Immediate payoff: turn day 10's three points into a **proper agreement-vs-
   rating curve over all 1,603 corpus games** — a report figure either way.

**B. 🔬 THE TWO TRAINING EXPERIMENTS, in this order.** Bar for both, pre-
   registered: **+50 Elo weighted across the five anchors or it is a chapter, not
   a submission** (§8k).
   1. **Rating-weighted clone** on the full corpus — keeps all 248,985 rows while
      fixing mode-averaging. **Most likely of the two to work**, and original
      enough to be its own report chapter.
   2. **Single-expert clone** — fine-tune v3 on `artifacts/pds_ntum`, then on
      ntum + Sixth Sense. ⚠ 27k rows against a 249k corpus: **underfitting is the
      expected failure mode** — fine-tune, do not train from scratch, and early-
      stop on a held-out expert split.
   ⚠ **Carry the standing prior in**: three axes of more/better training measured
   null or negative (§1), and the only thing that ever moved the clone was
   representational (§8f). This is a different axis — demonstrator *selection* —
   but the prior is not friendly.

**C. 🔍 THE ALTERNATIVE EXPLANATION, which a good report must address first:
   covariate shift.** Some of the 40% miss is BC's compounding-error problem, not
   a copyable policy. **Cheapest discriminator: score the expert net on OUR
   agent's trajectories and ours on theirs** — if disagreement is symmetric it is
   policy difference; if it collapses when the states are ours, it was shift.

**D. 📝 `report/STRATEGY.md` — one edit per session, minimum.** §7b already
   argues "the ceiling is the clone, not the deck"; day 10 gave it a much better
   instrument (identical 60 at +310 rating) and a new figure (the curve). Also
   worth a methods line: **`--all-rows` and the `--player` filter are the kind of
   detail the soundness bullet rewards.**

**E. ⛔ DO NOT SUBMIT** unless something clears **+50 Elo weighted over the five
   anchors**. Every submission evicts. The restore is closed; do not spend
   another slot re-testing a settled question.

> ✅ **ALL FIVE ITEMS RAN ON DAY 11.** A shipped; **B killed both arms** (−55 and
> −92 Elo, §8t/§8u); **C answered — covariate shift ruled out** (§8s); D done
> (§7b.1 rewritten, §7b.2 added); E honoured — **nothing was submitted.**
> ⚠ Note item B's prediction was **wrong in an instructive way**: it called the
> rating-weighted arm "most likely to work" and expected underfitting to sink the
> expert arm. The rating-weighted arm moved expert agreement by **0.1 pp** — it
> did nothing at all — and the expert arm imitated *successfully* (+7.3 pp held
> out) and lost anyway. **The standing prior in the same item was the part that
> held.**

</details>

<details><summary>Day 9's close and the day-10 order of work (kept for the record; A and C below are still live and were re-stated in the day-11 list)</summary>

#### ✅ Done at the end of day 9 (read before planning day 10)

- ~~**`55129730` — THE P4b RESTORE IS LIVE AND CLIMBING** (600 → 715.9 → …).~~
  ✅ **SETTLED day 10: it converged to 833.9 at 4.0 h against the original's
  958.2 at ~4 h — the same code, 124 points lower, on a board 2,000 entrants
  bigger** (§8p). The reasoning below was sound and the experiment was worth its
  submission: it converted a three-day argument into a measurement.
  Active pair is `{55129730, 55116557 v3 818.1}`; **P6a is evicted, frozen 841.5**.
  **The reasoning reversed an earlier recommendation of mine and the user was
  right to push:** the LB said P4b 952 vs P6a 846 for three days — a 105-point
  gap at/above the resolution limit — and "the board grew so it is not
  comparable" was a weaker argument than I presented. Risked 33 points (P6a →
  v3's floor) to chase ~100.
- ❌ **Boss's Orders rule #5 (`bossPrize`) — NULL, and the card is now properly
  closed.** All three anchors overlap; weighted **+6 Elo**. The user's
  observation was correct and the rule fixes exactly the defect they described
  (fires on 28.6% of plays vs a 29% measured misplay rate, corroborated three
  independent ways) — **it just decides ~0.09 prizes per game, 1.5% of a 6-prize
  game, against an A/B that resolves 0.021.** ⚠ **Rule 14 was violated: built
  first, sized after. The sizing takes two minutes and predicts the null
  exactly.** `EVIDENCE` §6.
- 🔧 **`scripts/p14_prize_audit.py`** — automates the misplay hunting that used
  to require the user watching games. It re-found the Boss's Orders defect
  independently. ⚠ **Its "did not attack" bucket is NOT trustworthy** (14.9–27.7%
  against P5c's established 3,683/3,683); `_available()` prices 180 damage
  without checking our Active can legally attack. **Fix that before using it.**

#### The day-10 order of work

**A. ⚡ HUNT THE NEXT REPRESENTATIONAL DEFECT — the only proven rank lever.**
   B1 is the single largest effect this project has produced, and it was found by
   **reading the feature code against a premise nobody had checked** (§8f), not
   by guessing. Do that again, deliberately:
   - `python -X utf8 scripts/context_accuracy.py` — the per-context miss table.
     **MAIN holds 3,930 of 6,424 misses.** That is the target.
   - Read `agents/sa/optfeat.py` and `features.py` **against** the top miss
     contexts and ask the §8f question for each: is the input **absent**
     (informational) or **present but unbindable** (representational)? Only the
     second kind has ever paid.
   - ⛔ **This list was WRONG and was carried for two days — see §8y.** Hand
     size, turn number, prizes and the opponent's discard are all encoded
     already (`features.py` 88-99, and `opp_discard` is an id bag). Use
     `p18_missing_state_audit.py`, which derives the list instead of recalling
     it.
   - **Bar: any candidate must clear +50 Elo weighted across the five anchors
     before it is worth a submission** (§8k). Below that it is a report chapter.

**B. 📝 `report/STRATEGY.md` — the highest EV per hour in the project, and it is
   NOT a consolation prize.** Deck 20% + writing 10% + the soundness /
   consistency / robustness bullets inside Model's 70% dwarf the LB's one bullet,
   **and the LB is stuck inside its own noise band while the report is not.**
   Day 9 alone produced three strong chapters — the censored sampling frame
   (§8i), matchup-conditional rules at **+47 / −51 Elo** (§8j), and the
   everything-is-within-36-Elo result (§8k). ⚠ **One edit per session, minimum.**

**C. 🔬 B4 — DECIDE, do not drift.** The prototype exists and **loses 0.075
   [0.026, 0.199] n=40** (§8n). Two bugs eliminated; the live hypothesis is a
   *design* flaw — maximising end-of-OUR-turn value cannot see the opponent's
   reply. **Either** test the one-ply-reply fix (small change to `_rollout`'s
   terminal evaluation) **or kill it and write it up.** It is already a good
   negative-result chapter either way. Do not let it consume day 10.

**D. 🃏 Track C deck work — reframe it, then do it FOR DECK SCORE.** §8o proved
   it is not a rank lever, so stop selling it as the counter-meta fix. It is 20%
   of the rubric on its own and **only ONE decklist variant has ever been A/B'd**
   (0.490, null). Also **re-aim it**: Track C is written against Crustle, but our
   worst matchups are **Archaludon (45.5% over 11 real games)** and **Mega
   Lucario (50%)**, and anchors for both now exist.

**E. ⛔ DO NOT SUBMIT** unless something clears **+50 Elo weighted over the five
   anchors**. Every submission evicts, and the pair {P6a, v3} is already the
   arena's top two (§8k). The restore of P4b is **closed — do not reopen it.**

</details>

<details><summary>Day 9's completed items (kept for the record)</summary>

**Day 9 answered the question day 8 ended on.** The blocking problem — "no arena
number in this repo predicts ladder strength" — is **closed** (§8i): the arena
predicts fine, the anchor set was wrong, and it is now fixed.

0. ✅ **THE 5-ANCHOR SWEEP IS COMPLETE** (n=2000 per cell, 71.5% of the field).
   **Weighted by field share, v3 is +36 Elo over P4b** — it wins four anchors and
   loses only Mega Lucario. Δ Elo is `elo(v3 vs anchor) − elo(P4b vs anchor)`;
   the mirror row is a head-to-head, so its Δ is `elo(0.657)` directly — **do not
   compute that one as `elo(v) − elo(1−v)`, which doubles it.**

   | anchor | share | P4b | v3 | Δ Elo | weighted |
   |---|---|---|---|---|---|
   | `rule:alakazam5` | **22.0%** | 0.727 [0.707, 0.746] | 0.731 [0.711, 0.750] | **+4** dead heat | +0.8 |
   | mirror, head-to-head | 13.8% | (0.343) | **0.657 [0.636, 0.677]** | **+113** | +15.6 |
   | `rule:crustle` | 12.8% | 0.663 | 0.770 | **+92** | +11.8 |
   | `rule:v10` | 12.8% | 0.576 [0.554, 0.598] | 0.505 [0.483, 0.527] | **−50** 🔴 | −6.4 |
   | `rule:archaludon` | 10.1% | 0.621 [0.599, 0.642] | 0.669 [0.648, 0.690] | **+36** | +3.7 |
   | | **71.5%** | | | | **+36 Elo** |

   **And the ladder agrees, once compared honestly:** v3 819.8 vs P6a 845.0
   (both active, same time) = **−25**, against an agent still climbing, inside
   the LB's ±50–100. **Arena +36, ladder −25, instrument ±75: these are not in
   conflict.** The apparent 130-point contradiction came from comparing against
   a frozen 07-29 score (§8i).

   ⚠ **The one place v3 is genuinely worse is `rule:v10` (−50 Elo on 12.8%).**
   That is the live engineering lead — see item 3.

<details><summary>The sweep as it was being run (superseded, kept for the reasoning)</summary>

   **⚡ FINISH THE 5-ANCHOR SWEEP. Nothing should be submitted before it.** Two of
   the four runs are done; `p9_field_census.py`'s top 5 covers 71.6% of the field:

   | anchor | share | `bc:p4b,noSrc` | `bc:v3off,…` |
   |---|---|---|---|
   | `rule:v10` / `lucario_v10` | 12.8% | **0.576 [0.554, 0.598]** | **0.505 [0.483, 0.527]** |
   | **`rule:alakazam5`** / `alakazam5` | **22.0%** | **0.727 [0.707, 0.746]** | **0.731 [0.711, 0.750]** |
   | `rule:archaludon` / `archaludon_ex` | 10.1% | ⏳ running | ⏳ TODO |
   | `rule:crustle` / `crustle_v1` | 12.8% | 0.663 (§8c) | 0.770 (§8f) |
   | mirror (`grimmsnarl` v `grimmsnarl`) | 13.8% | — head-to-head — | ⏳ **running** |

   **v3 − P4b so far: Alakazam +0.004 (dead heat, CIs overlap), Crustle +0.107,
   Lucario −0.071.**

   ```powershell
   python -X utf8 scripts/arena.py play "bc:v3off,net=out/policy_b1_v3.npz,noChip,noSpread,noSrc" `
       rule:archaludon --deck-a grimmsnarl --deck-b archaludon_ex --matches 1000 `
       --archive out/arena/p9_v3off_vs_archaludon.jsonl
   ```

   ⚠ **Weight each anchor by its share before concluding anything.** A rule that
   wins 22% of the field and loses 12.8% is not "2 anchors to 1" — it is +9.2 pp
   of the field. That arithmetic is the whole point of the census, and it is the
   thing rule 12 was missing.

   🔴 **AND CHECK BOTH ARMS ARE THE SAME COMPARISON — this nearly went wrong.**
   §8f's mirror number (**0.661**) is v3 vs `out/policy_b1_ctrl.npz`, a
   **v2-feature net trained on the same `pds_v3` corpus**. That is *not* P4b
   (`lw2` net, `pds_v2` corpus, chip + spread rules **ON**). Dropping 0.661 into
   the column above as "v3 is +0.161 in the mirror" mixes a **feature ablation**
   with an **agent comparison**, and it flips the weighted verdict: done naively
   it totals ≈ +0.045 for v3, while the ladder says v3 is **132 points worse**.
   **The honest cell is `bc:v3off` vs `bc:p4b,noSrc` head to head — which had
   never been run in this project.**

   ✅ **It landed at 0.657 against the 0.661 that was being reused, so the reuse
   was harmless — but it was harmless by luck.** Run the cell you are actually
   weighting; it cost 12 minutes.

</details>

1. ⛔ **DO NOT RESTORE P4b — ANSWERED 2026-07-31 IN THE ARENA, FOR ZERO
   SUBMISSIONS** (§8k). All three agents swept across all five anchors, n=2000
   per cell, Elo relative to P4b:

   | agent | weighted | note |
   |---|---|---|
   | **v3** (rules off, `55116557`) | **+36** | active |
   | **P6a** (`55077709`) | **+7** | active, and our best LIVE score (845.0) |
   | **P4b** (`55072063`) | **0** | frozen 952.0 — **the arena ranks it LAST** |

   **The entire spread is 36 Elo and the LB resolves ±50–100** (§1). So restoring
   P4b would cost a submission, **evict `55077709` (845.0, our best active and
   still climbing)**, and restart at μ=600 for ~4 h — to install the agent the
   arena ranks last, on the strength of a **frozen** score earned on a board
   2,000 entrants smaller. **The active pair {v3, P6a} is already the arena's
   top two. No action needed.**

   ✅ Also settled: **`counter_source` is vindicated a second time** — P6a beats
   P4b by +24 Elo in the mirror and +7 weighted, independent of §8c's +0.052.

   🔴 **The strategic consequence, and it should steer the remaining 17 days:**
   if our best and worst agents differ by 36 Elo and the LB cannot see 36 Elo,
   **no further rule-sized improvement can move the rank.** The only levers big
   enough to clear the band are a materially better net or **ROADMAP B4**
   (turn-level sequencing — we use 0.1 s of the 600 s pool). **Another targeting
   rule is a report chapter, not a rank.**

<details><summary>The open-question framing this replaced (kept — the reasoning is report material)</summary>

   **⚠ THE P4b RESTORE IS NOW A GENUINELY OPEN QUESTION — DO NOT DO IT ON
   AUTOPILOT.** Every earlier version of this item assumed "952 > 837.5, so
   restoring is free value". **That premise is a frozen-vs-live comparison,
   which is exactly what §8i retracted.** The evidence now points both ways:

   | for restoring P4b | against restoring P4b |
   |---|---|
   | It *did* read **952.0**, the highest number this project has produced | That 952.0 was earned 07-29 on a **~4,000**-entrant board; the board is now **6,000**, and a frozen rating is not comparable to a live one (rule 2) |
   | The LB is the real referee and it liked P4b best | The **arena now covers 71.5% of the field** and says **v3 is +36 Elo over P4b** — and the arena's credibility was the only reason to doubt it |
   | | A restore **costs a submission and evicts** (`55077709`, 845.0). It restarts at μ=600 and needs ~4 h |
   | | We would be evicting the agent the arena ranks **highest** to install the one it ranks lower |

   **My read: this is now finely balanced and it is the user's call, not a
   default.** ⚠ **Do not treat "restore P4b" as settled just because three
   earlier versions of this file said so** — all three were written before the
   anchor set covered the field.

</details>

   **The decision that actually binds is item 2 (what is ACTIVE on 08-17), and
   there is time.** Nothing is at risk of being lost:
   - **Kaggle's copy of `55072063` is permanent** and keeps showing 952.0.
   - **P4b is rebuildable from git even without `dist/`** — `dist/**` is
     gitignored, but `agents/sa/policy_net.npz` **is tracked** and is the same
     lw2 net (`sha256 bba02a42…`), and the code is in history.
   - **You CANNOT re-activate an old submission.** The API has `competition_submit`
     and nothing like select/activate — "latest 2" is recency, not a choice. So a
     restore is always a *new* submission climbing from μ=600, whenever it happens.

   **What actually binds: the best agent must be ACTIVE at the 08-17 close and
   through the 08-31 continued-play window** (§8h). Two mild arguments against
   leaving it to the deadline: the climb takes ~4 h, and the field is growing fast
   (3,000 → 5,000 → **6,000** entrants in 3 days), so a late restore may not land
   on 952. ⚠ **And note that same growth is the reason 952 is not comparable to
   819.8** — it cuts both ways.

   ⚠ **Real gap worth closing cheaply: `out/policy_b1_v3.npz` and
   `artifacts/pds_v3/` are gitignored and exist ONLY on this disk.** Losing them
   means re-running the 4-day shard rebuild plus a 12-epoch train to get v3 back.
   **This is the one part of item 1 that is unambiguously worth doing now.**

   **The cheapest way to settle the restore question without spending a
   submission:** the arena already ranks P4b vs v3 (+36 Elo to v3, 71.5%
   coverage). The missing arm is **P6a** — the agent a restore would evict, and
   the only one whose live score (845.0) is comparable to v3's. Run
   `bc` (= P6a's exact config) against all five anchors and weight it. **If P6a
   ≈ v3, the restore evicts nothing and the only question is whether P4b beats
   both — which the arena says it does not.**

   ⛔ **SETTLED 2026-07-31 (day 10): the restore was DONE, and it read 833.9 at
   4.0 h against the original's 958.2 at ~4 h** (`EVIDENCE` §8p). Everything below
   is kept only as the record of how the decision was made. **Do not run it
   again.**

   If you do restore anyway:

   ```powershell
   python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_submit('dist/submission_bc-grimmsnarl-netspolicy_20260729-103819.tar.gz','P4b restore: lw2 + chip_target + energy_spread (the 952 agent)','pokemon-tcg-ai-battle')"
   ```

   - That tarball is **verified** to be `55072063`'s exact code (flags: chip +
     spread, `counter_source` absent from the signature) and **smoke-tested**
     (`scripts/restore_smoke.py`). §3.0's table.
   - **Eviction arithmetic (updated 2026-07-31): it evicts `55077709` at
     845.0 — which is now our BEST ACTIVE**, not our worst, and it is still
     climbing. Active becomes {P4b-restored, v3 819.8}. ⚠ **This is the reverse
     of what earlier drafts said** and it is the main new argument against.
   - ⚠ It restarts at **μ=600** and needs ~4 h to reach ~950; displayed dips to
     819.8 meanwhile. That is the cost, and it is unavoidable — a rating cannot be
     restored, only re-earned (§8h).
   - ⚠ **Read the LB before and after** (§5) and confirm with **two readings ≥1 h
     apart** (rule 2).

2. ~~**Build anchors for the REAL field.**~~ ✅ **DONE 2026-07-31 — this was the
   blocker on everything and it is gone** (§8i). `scripts/p9_field_census.py`
   names the field from our own 109 ladder games; `scripts/import_field_agents.py`
   imported the two anchors we lacked. Coverage **39.4% → 71.6%**. The remaining
   work is item 0's sweep, not more infrastructure.

   ⚠ **The one thing to internalise from it:** the 63% "other" was **not** an
   exotic tail. It was four ordinary archetypes plus a classifier with four
   hardcoded card ids. **Before believing a bucket called "other", check whether
   the classifier or the field is the thing that is small.**

3. ✅ **THE MEGA LUCARIO LEAD IS SIZED AND THE ANSWER IS "NOT WORTH SHIPPING
   ALONE"** (§8j, 2026-07-31). **The cause is not the v3 net — it is the rules
   being off.** v3+rules reads **0.572** vs `rule:v10` against rules-off's
   **0.505** (disjoint), i.e. level with P4b's 0.576.

   **But turning them on globally is worth +1 Elo** — the mirror loses exactly
   what Lucario gains (−7.1 vs +6.0 weighted), and Alakazam/Crustle/Archaludon
   are dead heats. **A Lucario-only branch sizes at ~+8 Elo, below the LB's
   ±50–100 resolution** (rule 2), so by rule 14 it is a **bundle candidate, not a
   solo submission, and not urgent.** The branch machinery already exists
   (`wall_defer` is the template) if it is ever bundled.

   ⚠ **Two traps this closed, both worth carrying:**
   - **"Rule X is harmful vs anchor Y" expires when X is modified.** We predicted
     rules-on would lose badly to Crustle (`chip_target` measured −0.126 there);
     it was a dead heat, because `wall_defer` has been ON by default since 07-30.
   - **Do not sum dead heats.** Only 2 of 5 cells have disjoint CIs. Branching
     wherever Δ>0 scores +12 Elo, of which **+4 is noise** from three overlapping
     cells.

3b. **B4 (turn-level sequencing) — probed GREEN, prototyped RED.** ⚠ **Superseded
   by §8n: the prototype LOSES 0.075 [0.026, 0.199] n=40.** Kept because the
   probe numbers are sound and are report material; see item C above for the
   decision. The probe said (`EVIDENCE` §8l, §8m):
   - **62% of our turns** have ≥2 real selects (rule 13 passed).
   - Space is median **98M** — exhaustive dead — but `fs.step` runs at
     **7,698/s**, so **~78,000 candidate sequences per turn** are affordable.
   - `evalfn` ranks **within** a turn: split-half top-1 agreement **62.0% vs
     6.2% chance** over 93 turns. Not determinization luck.
   - ⚠ **A pre-registered kill criterion was corrected mid-probe** (SNR at M=1
     answers a question B4 never asks). §8m documents it in full — **read that
     before trusting the verdict.**

   **Next: prototype + arena A/B at n≥1000 vs all five anchors, with pool-usage
   logging.** ⚠ Estimated gain is small and upward-biased (0.099 eval units per
   turn, a max over 16), and the 600 s pool is a real risk (§7). By rule 3 this
   is licensed as an experiment, not as an expected win.

3c. **The original Mega Lucario framing, kept because the two-instrument
   agreement still holds and still points at this matchup:**
   - **Arena:** v3 is **−50 Elo** vs `rule:v10` — the only anchor of five it
     loses, and the only negative term in the weighted table.
   - **Ladder:** we won **36.4% of 11 real games** against Mega Lucario
     opponents averaging **735**, i.e. **85 points below us** (§8i's calibration
     table). Losing to weaker players is not a matchup tax, it is a defect.

   It is **12.8% of the field**, we have a **real LB-950 pilot** for it already
   (`rule:v10,noS`), and `replays/submission_optv3` holds 11 real games to read.
   **Start with the audit, not a rule** (rule 14): what does Mega Lucario ex
   actually do to us — 340 HP, and `decks/lucario_v10.py` runs Gravity Mountain
   (bench HP?) plus Premium Power Pro ×4. Size the effect before writing
   anything, and check whether `chip_target`/`energy_spread` are net-negative
   here the way `chip_target` was against Crustle (§8c is the template).

   ✅ **The 5-anchor rule sweep is DONE — that is what item 3 above reports.**

4. **Re-mine the meta?** ⛔ **NO — and this is now a permanent rule, not a
   scheduling note.** Kaggle's daily episode datasets bottom out at `avg_score`
   **1055**; we play at 825–952; the 800–1000 buckets are **empty**. Mining
   produces an accurate picture of a band we never meet, and acting on it is what
   retired `rule:v10` — the anchor that turned out to be our worst matchup
   (§8i). Mining is still useful for
   **decklist consensus** (§1's "our 60 is the field's 60" is real Deck Score
   evidence) and for **Track B report figures about the top of the ladder** — but
   **never again as the input to an anchor decision.** Use
   `p9_field_census.py` on our own replays for that, and re-run it after every
   submission dump.

5. **Fix the two measured defects — but as questions, not licences** (§6 closed
   Boss's Orders after four null rules; all four were on the **lw2** net with the
   other rules on, so they do not settle this net):
   - **Boss's Orders: 9 of 31 real drags were misplays (29%)**, 5 of them throwing
     away a **double KO** (Shadow Bullet is 180 to the Active **plus 30 to a
     bench** — a ≤30 HP bench sitter means attacking takes two prizes).
   - **Froslass: 7 of 63 (11.1%)** evolves happened with more ability Pokemon on
     our side **and no armed Munkidori** — pure self-damage. ⚠ The other 19
     "we have more" rows are the intended engine (Shroud loads, Adrena-Brain
     ships); do not "fix" those.
6. **The Alakazam matchup is a strategy question nobody has asked.** It is 22% of
   the field — the biggest single thing we play against — and its attack is
   **Powerful Hand: 20 damage per card in the attacker's hand.** Nothing in
   `targeting.py` or the feature set reasons about the opponent's hand size, and
   the whole deck is a draw engine (Kadabra/Alakazam Psychic Draw, Dudunsparce
   Run Away Draw, Fezandipiti ex Flip the Script). **Size it before building it**
   (rule 14): how often is their hand large enough to matter, and is there any
   action of ours that shrinks it? ⚠ We already win this matchup 66.7%, so the
   headroom is small — check that first.

</details>

### The B1 arena result — kept because the CONTRAST with the ladder is the finding

> ⚠ **Read this as the specimen, not as a plan — and note that §8i has since
> explained it.** `optfeat` v3 beat the shipped agent **0.661 [0.640, 0.681]
> n=2000** in the mirror (≈ +115 Elo) and **0.770 vs `rule:crustle`**
> (shipped: 0.663) — two anchors, one adversarial, both agreeing, the first effect
> in the project larger than the LB's own resolution. With v3 features the hand
> rules measured **harmful** (`v3+rules` vs `v3 alone` = **0.427**), which is why
> it shipped with rules off.
>
> **It then read 825 against P4b's 952** (§8g). Nothing above was miscomputed and
> nothing above is retracted — every number reproduces from `out/arena/b1_*.jsonl`.
> **What was wrong was the coverage, not the measurement**: those two anchors are
> 26.6% of the field, and against the third-largest deck (`rule:v10`, 12.8%) the
> same v3 agent scores **0.505 vs P4b's 0.576** (§8i). The mirror's +0.161 and
> Lucario's −0.071 are both real; only one of them was in the anchor set.
> Nets: `out/policy_b1_v3.npz` (treatment), `out/policy_b1_ctrl.npz` (control).
> Corpus: `artifacts/pds_v3`.

<details><summary>The v3 bundle as built and shipped (kept for reproducibility)</summary>

**Built, smoke-tested, and SUBMITTED as `55116557` on 2026-07-30 18:14 UTC.**

   ```
   dist/submission_bc-grimmsnarl-netspolicy_20260731-000752.tar.gz  (4.0 MiB)
   dist/submission.tar.gz  <- same file (the `latest` copy)
   ```

   Built with, and this exact command is the reproducer:

   ```powershell
   python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc `
       --nets policy --policy-net out/policy_b1_v3.npz --no-rules
   ```

   **Verified, not assumed:**
   - `NET_OK opt_in=37` — the **v3** net is live in the extracted bundle. ⚠ This
     check is new and it matters: a net that fails the dim guard makes the agent
     play `list(range(minCount))` — **random-legal, and it still "runs"**. The
     builder now fails the build instead (`--policy-net` runs `policynet.load`),
     and the smoke asserts `NET_OK`.
   - `FLAGS chip=False spread=False src=False` — the rules are off, pinned in
     `main.py` as `AGENT_KWARGS`. ⚠ **Global defaults deliberately NOT flipped** —
     they remain correct for `lw2`, which is what is live right now. The
     `(net, flags)` **pair** is pinned at build time; `wall=True` is inert
     because `chip_target` never runs.
   - **sha256 of `sa/policy_net.npz` == `out/policy_b1_v3.npz`** — the packaged
     net is byte-identical to the one that measured 0.661.
   - `agent_pool_left=599.9s lat_max=0.04s` — 0.1 s of the 600 s pool.
   - Layout `main.py` + `deck.csv` + `cg/` + `sa/` at top level; 4.0 MiB of the
     197.7 MiB cap; smoke `exec`s the source with **no `__file__`** (the §7 gotcha
     that killed `55028078`).

   ⚠ **The packaging was all correct — and it did not save the result.** Every
   check above passed and the agent still lost ~130 points. **Build hygiene
   protects against shipping a broken bundle; it cannot protect against shipping
   a worse agent.** The thing that failed was the *decision*, and the decision
   came from the arena.

</details>

### Closed earlier on day 8 (kept for the record)

1. ~~**Size, then build, the Morgrem out**~~ ✅ **SIZED AND CLOSED 2026-07-30 —
   do not build it** (`EVIDENCE` §8e, `out/logs/p7_morgrem_200.txt`). The veto
   would fire **~0.2× per game**; the *free* version of the same out (post-KO
   promotion into a wall) is **already taken 95.4%** of the time; and the trade is
   *60 onto a wall they heal 22.5% off* vs *30 onto a 70-HP Dwebble that dies to
   it + 220 more HP of body* — a **tradeoff**, rule 11's 0-for-4 column. The
   effect is ~2.6% of our damage output in this matchup, which **an n=2000 A/B
   cannot resolve** (±0.021), so no A/B was spent. **Also corrected a load-bearing
   claim:** "our attacker deals 0 into theirs" is true of their **Active only** —
   Shadow Bullet's bench snipe lands **unprevented on Dwebble (82 events, mean
   73.9, 0 zeroed)** and kills the Crustle line's basics.
2. ⛔ **A pilot for `crispin_toolbox` DOES NOT EXIST PUBLICLY — searched
   2026-07-30, and the public-notebook well is dry for competitive pilots of any
   deck.** All **272** public notebooks for this competition were enumerated
   (4 sort orders × 3 pages). No Crispin/toolbox pilot at all. Three candidates
   whose titles claimed high ratings were pulled and **all three refuted against
   the 4,000-row LB** (rule 10, the same trap as the "1084.5 baseline"):

   | notebook (claim) | author's actual standing |
   |---|---|
   | `soutasakurai/max-elo-1208-libraryout-w-crustle-great-tusk` ("Max Elo 1208") | **`SOUTA Sakurai`, rank 3439/4000, 605.0** — *below the μ=600 start* |
   | `prvsiyan/ptcg-ai-battle-static-deck-tusk-1208-v24` ("Tusk 1208") | `prvsiyan`, rank 1083, 789.1 |
   | `pcxxxxxx/explainable-ptcg-agent-with-legal-ogerpon-deck` | `pcxxxxxx`, rank 2454, 686.6 |

   Every other verifiable notebook author also sits **below us**: `kokinnwakashuu`
   832.9, `jazivxt` 816.3, `pllinas` 739.1, `penguin069` 689.8, `naoto714` 633.0.
   **The top 10 (1187–1147) have published nothing.** So there is no public agent
   stronger than ours to import, and this avenue is closed — not deferred.

   **Consequence, and it is good news:** rule 12's bar (**≥2 anchors, one
   adversarial**) is *already met* by the mirror + `rule:crustle`, and
   `rule:crustle` is competitive on our own measurement (we score 0.663, not a
   0.911 blowout — a real number beats any notebook title). **Writing a Crispin
   pilot ourselves is NOT recommended:** a 5-attacker multi-type toolbox with
   Crispin tutoring is far harder to pilot than Crustle's single lockdown line,
   and a weak self-written pilot reproduces the 0.911 no-resolving-power failure.
   By rule 14, size that before building it.
3. **Re-mine the meta — BLOCKED UNTIL 07-31.** 07-30's episodes publish the
   following day (the current day always 403s) and 07-29 is already mined, so
   there is nothing new to fetch today. On 07-31: confirm the Crustle/Crispin
   shares and build the **deck matchup win-rate matrix** among high-rated players
   (ROADMAP Track B/C figure). ⚠ This also gates the Crispin-anchor question —
   check Crispin's share is still ~17% before spending any work on it.
4. ~~**ROADMAP B1** (feature-augmented retrain)~~ ✅ **DONE AND WON 2026-07-30/31
   — see the green box at the top.** `EVIDENCE` §8f. Follow-ups it opened, in
   value order:
   - **Retrain v3 on a bigger corpus.** v3 won on **1,603 games vs the shipped
     net's 2,810** — 43% less data. The pruned days are re-fetchable from
     `replays/manifests/` (12 days of episode ids). ⚠ But §1 says more data is
     *not* a lever, so treat this as a cheap check, not an expected gain.
   - **Re-A/B each rule against the v3 net individually.** We know the three
     together are harmful (0.427); we do **not** know whether one of them is
     still positive. `noChip` / `noSpread` / `noSrc` one at a time.
   - **The v3 features make `wall_defer`'s hardcoded `WALL_POKEMON = {345}`
     obsolete in principle** — "our damage into this target" is now feature 34,
     so the wall condition is readable off the board for *any* prevention
     ability. Only matters if a second wall deck appears.
5. ~~**Do not submit yet.**~~ ⚠ **SUPERSEDED BY B1 (item 0).** That advice was
   written when the best candidate was a ~12-Elo rule, which the LB cannot
   resolve. **B1 measures ≈ +115 Elo on two anchors — above the instrument's
   precision** — so the reasoning that said "wait and bundle" now says "submit
   this one". The bundle it was waiting for exists.
6. ~~**`report/STRATEGY.md` does not exist yet**~~ ✅ **CREATED 2026-07-31** after
   slipping ~4 sessions. §1–5 and §8 are written from concluded experiments;
   §6–7 are outlined against work in flight. **Standing rule: one edit per
   session, however small** — it is 30%+ of the rubric against the LB's one
   bullet of five, and it was the only deliverable with no same-day feedback
   loop to force it. See ROADMAP's doc-discipline audit.

### The five files, and what each owns

| file | owns |
|---|---|
| **`HANDOFF.md`** (this) | live state, the live engineering plan, the anti-self-deception rules, commands, gotchas |
| **`ROADMAP.md`** | the strategy-competition plan — what the engineering is *for*, the breakthrough hunt, the calendar |
| **`report/EVIDENCE.md`** | the hypothesis log: every concluded experiment with n, CI, verdict. **All closed-experiment detail lives there, not here.** |
| **`report/STRATEGY.md`** | **the report itself** — the deliverable due 09-14. ⚠ **One edit per session, however small.** It slipped ~4 sessions because it is the only file with no same-day feedback loop |
| **`competition_details_and_rubric.md`** | the rubric, verbatim |

**End of every session: update HANDOFF (plan), ROADMAP (calendar), EVIDENCE
(any experiment that CONCLUDED) and STRATEGY (one edit, however small) together.**

⚠ **And when you retract a claim, `grep` it across all five files in the same
commit.** Updates here have been additive — HANDOFF went 135 → 1,579 lines in
5 days and carries 27 retraction markers — so a wrong claim survives in whatever
copied it. "`lucario_v10` is 0% of the meta" propagated to four places and cost
us the anchor that would have caught B1. Rule 15 warns about this; we did it
anyway. ROADMAP's doc-discipline audit has the numbers.

> **Submission state (2026-07-31). ⚠ The previous version of this box was WRONG
> on the one point that mattered — see the ✅ below.**
>
> - **Daily quota: 5/day.** Never the binding constraint.
> - **Only the latest 2 submissions play episodes.** Active pair right now
>   (2026-07-31 10:46 UTC): **`55129730` (P4b restored, 833.9) + `55116557`
>   (v3, 818.1)**. `55077709` (P6a) is **evicted, frozen at 841.5**; `55072063`
>   (**952.0**) has been frozen since 07-30.
> - 🔴 **The 952.0 is not a target and never was one.** Re-running that exact
>   agent on today's board produced **833.9** (§8p). **Every one of our agents
>   reads 818–842 when played concurrently.**
> - ✅ **ANSWERED (was "the open question that decides the endgame"): the
>   displayed score is the best ACTIVE submission, NOT the best ever.** Proof:
>   best-ever is `55072063` at 952.0, best-active is 837.5, and **the board shows
>   us at 837.5 / rank 605.** We fell 224 → 605 on the eviction alone.
> - 🔴 **So "freezing is cheaper than it sounds" was FALSE and is retracted.** A
>   frozen score counts for nothing. **The best agent MUST be in the active pair
>   at the 08-17 close and through the 08-31 continued-play window.**
> - 🔴 **Every submission is therefore a real risk, not a free option.** It
>   evicts, and the evicted score stops counting the moment it does.
>
> **The bar on submitting is "do we expect this to beat the best agent we would
> be evicting" — and 🔴 as of 2026-07-31 NOTHING WE HAVE CLEARS IT.** All three
> agents are within **36 Elo** and the LB resolves **±50–100** (§8k), so no
> current candidate is distinguishable from what it would evict. **The active
> pair {v3 819.8, P6a 845.0} is the arena's top two; leave it alone.**
>
> ⚠ The rollback argument ("952 > 837.5") is **retracted**: 952.0 is frozen, was
> earned on a ~4,000-entrant board, and the arena ranks P4b **last** of the three.

---

## 1. Where we are (day 8 end, 2026-07-30)

| submission | what | LB |
|---|---|---|
| `55077709` | + `counter_source` (P6a) | 600 → 762.2 → 746.4 → **824.9** ⚠ still climbing |
| `55072063` | clone v2 + `chip_target` + `energy_spread` | 958.2 → 970.1 → **948.1** ✅ **our best** |
| `55054446` | clone v2 + `chip_target` | 916.8 → 936 → 979 → 901.6 → **905.2** (inactive) |
| `55048039` | clone v2, no targeting | 752 → 758.6 (settled) |
| `55049206` | `rule:iono` sample agent | ~700–716 (settled) |

**What ships:** a behavior clone of the field (2,810 human games) plus
arithmetic rule overrides for the decisions its features cannot express. ~1 ms
per move; uses 0.1 s of the 600 s pool. See §4.

**The method, confirmed end to end:** *find decisions the features cannot
express, and write a rule for them.* Three axes of more training bought nothing;
one missing feature (`chip_target`) bought ~150 LB points.

**The sharpening, which matters more than the method:** 7 rules have been A/B'd.
The three that won delete a **provably worthless** option; the four that did
nothing pick a side in a **tradeoff** — and every one of those four moved its
audit rate exactly as designed first. **Rule 11 in §2 is the test.** Full
numbers: `report/EVIDENCE.md` §3.

**The open problem, reframed 2026-07-30:** `counter_source` won both local bars
and then read low on the LB. **The gap has since halved on its own** (224 → 123)
with the two agents converging *toward each other from opposite directions*, so
the day-7 "confident false positive" framing was itself premature. §3.0.

### ⚠ The resolution limit — the day-8 lesson, and it constrains everything

**A 0.534 mirror A/B is ≈ +12 Elo. Our LB readings swing ±50–100 while
converging.** So the LB could never have resolved `counter_source` in either
direction, and the local arena and the LB were never actually in conflict — the
error was asking a ±75-point instrument to measure a 12-point effect.

Consequences, all of which bind on the rest of the project:

- **Never nominate the LB as the referee for a rule again.** It can confirm a
  ~150-point intervention (`chip_target`) and nothing smaller. Small rules are
  decided in the arena, and the arena's trustworthiness is therefore the whole
  game — which is why §3.1 outranks everything.
- **Rules worth ~10 Elo cannot be validated one at a time on the LB, ever.** If
  the remaining lever is a stack of small rules, the only honest validation is
  multi-anchor arena A/Bs plus one LB submission of the *bundle*.
- **This is a rule-2 amendment, not a new rule:** two readings ≥1 h apart that
  agree are necessary but not sufficient — the effect also has to be **larger
  than the instrument's precision** before an LB reading can speak to it.

### ⚠ The meta shift (2026-07-30) — TRUE, but about a band we never play in

> 🔴 **READ THIS BEFORE THE TABLE (added 2026-07-31, `EVIDENCE` §8i).** Every row
> below was mined from the **top 400 episodes by `avg_score`**, and Kaggle's daily
> datasets **contain nothing below `avg_score` 1055**. We play at **825–952**;
> the 800–900 and 900–1000 buckets are **empty**. So this table is an accurate
> description of the **top of the ladder** and says nothing about our opponents.
>
> **Acting on it cost ~130 LB points.** Row 1 below retired `rule:v10` as "0% of
> the meta"; in our own 109 real games Mega Lucario is **12.8% of the field**, and
> it is the anchor that would have caught B1. **For what we actually face, use
> `scripts/p9_field_census.py` on our own replay dumps — never this table.**
>
> What it IS still good for: the decklist-consensus finding (item 3 below, real
> Deck Score evidence) and Track B report figures about the top of the board.

Mined with `mine_meta.py`: **pre-shift** = 07-22 + 07-24, 800 games / 1,600 seats
(`out/meta/pre_shift_0722_0724.txt`); **post-shift** = 07-29, 400 games / 800
seats (`out/meta/post_shift_0729.txt`).

| archetype | pre-shift share | post share | pre WR | post WR |
|---|---|---|---|---|
| **`{D}`/Munkidori — OUR deck** | 829 (51.8%) | 417 (**52.1%**) | 52.2% | **47.5%** ⚠ |
| **Crustle** (`Mist`/`Spiky`) | **1 (0.06%)** | **145 (18.1%)** | — | **56.6%** |
| **Crispin toolbox** (`{G}`/Crispin) | 2 (0.1%) | **135 (16.9%)** | — | **58.5%** |
| Abra/Alakazam + Abra/Telepath | 214 (13.4%) | 37 (4.6%) | 45.8% | 38–45% |
| **`{F}`/Rock Fighting = `lucario_v10`** | 159 (9.9%) | **0 (0.0%)** | 54.1% | — |

**Three findings, each of which changes the plan:**

1. ❌ ~~**`lucario_v10` — the single opponent every routine number in this repo is
   measured against — is 0 of 400 games.** Our arena bar has been measuring a
   deck that has left the meta entirely. This is rule 12's worst case,
   realised.~~ 🔴 **RETRACTED 2026-07-31 — this is the most expensive wrong
   sentence in the project.** It is 0 of 400 games **at avg_score ≥ 1144**. In
   our own 109 real games Mega Lucario is **12.8% of the field**, and it is the
   matchup we lose worst (36.4% over 11 games, against opponents rated **85
   points below us**). Retiring that anchor is what let B1 ship unseen. **The
   error was not the measurement — it was reading a top-band sample as "the
   meta".** §8i.
2. **🔴 Crustle went from 1 seat in 1,600 to 18.1% of the field at a 56.6% win
   rate, and the LB's top two players are both on it** (`flg` 1205.7,
   `Majkel1337` 1186.4 in this sample). **Our own deck's win rate fell 52.2% →
   47.5% across the same window** while staying half the field. §3.2 is not a
   side quest — **it is the meta**, and the pilot we don't have is the instrument
   we most need.
3. **🟢 Our decklist is still exactly the field's consensus.** The most common
   exact 60 on 07-29 was seen **353×** and `decks/grimmsnarl.py` is **identical
   to it** (verified card-for-card). We are not playing a stale or fringe list —
   direct Deck Score evidence, and it also means no decklist change is needed for
   *consensus* reasons, only for matchup reasons (Track C).

Also mined: the **Crispin toolbox** at 16.9% / 58.5% — the highest win rate in the
sample, though **all 135 games are one team**, so read it as one strong pilot, not
a field average. It contests the stadium slot (Area Zero Underdepths ×4 vs our
Spikemuth Gym), which no current rule of ours reasons about.

**New anchor decks committed:** `decks/crustle.py` (rebuilt — the previous
reconstruction was **12 card slots stale**, including 4× Crushing Hammer the
current list does not run) and `decks/crispin_toolbox.py`. Both resolve to 60.

### What the top of the board does

Nothing strong here is learned. `notebooks/` holds three checked-in reference
agents: `strong-start-baseline-agent-v10-lb-950` (LB 950+, hand-written
deck-specific scoring, ~350 readable lines), `rule-based-not-psychic-alakazam-best-5th`
(**5th place, pure rules, no ML, no search**), and
`a-sample-archaludon-75-wr-vs-my-1300-starmie` (author reports 1300+; matchup
rules with grid-searched thresholds). The competition rewards **deck expertise +
matchup rules + damage arithmetic**. And **V10's MCTS has never once executed**
(`EVIDENCE` §2) — LB 950+ is 100% handcrafted policy.

---

## 2. How not to fool yourself

Every rule here was paid for. Rules 1, 2 and 8 have each invalidated real work.

1. **n=24 is noise.** A BC game costs ~0.17 s — n=1000 is 17 s of CPU. **Never
   accept an n<100 strength claim for anything cheap to measure.** ~2 pp effects
   need n≈2000.
2. **One LB score is not a result.** `55049206` read 743.0 → 697.4 → 704.1.
   **Require two readings ≥1 h apart that agree.** `55054446` is the standing
   warning: day 6 logged "916.8 → 936 → **979**, trending up" and planned against
   it; it settled at **905.2**, below its own first reading. **A rising score is
   unconverged, not momentum** — and a *falling* young one is equally
   uninformative (everything starts at μ=600 and climbs for hours).

   ⚠ **And agreement is not sufficient — the effect must exceed the
   instrument.** LB readings swing ±50–100 while converging, so **the LB cannot
   resolve a rule worth ~10 Elo (a ~0.53 A/B) at all.** It confirmed
   `chip_target` (~150 points) and it could never have adjudicated
   `counter_source` (~12). Day 8 spent a whole session's prior on that mistake
   (§1, §3.0). **Ask what size effect the instrument can see before you let it
   overrule the arena.**
3. **Validation metrics do not predict playing strength — five times.** Value-net
   loss, policy top-1 ×3, `--winners-only`. The net with the *best* val accuracy
   lost its A/B. **Judge every net in the arena, head-to-head.**
4. **Compare nets head-to-head, not through a third opponent.**
   `bc:<tag>,net=<path>` runs two nets in one process.
5. **A cross-deck arena score is mostly a DECK MATCHUP, not agent skill.**
   `rule:lucario` scores 0.781 vs `rule:iono`; the ~104-Elo-stronger
   `rule:v10,noS` scores 0.788 — indistinguishable; the pilot is invisible
   through that anchor (head-to-head they are 0.646 [0.616, 0.675]).
   **Measure skill in near-mirror matchups only.**
6. **CPU contention distorts wall-clock-budgeted agents** (`search:*`, `rule:v10`
   without `noS`). BC and `rule:*,noS` are untimed, so cross-run comparison is
   valid for them.
7. **This machine gives ~1.4 cores of real throughput** (Ryzen 5500U, 15 W). Run
   2–3 jobs, not 4+.
8. **Frequency is not correctness, and per-turn binary audits hide
   multiplicity.** `munkidori_adrena_brain` read 99.4% per *turn* but 96.9% per
   *opportunity* — with two Munkidori a turn offers two activations and `any()`
   scores one as 100%. **Count opportunities, not turns** (`MULTIPLICITY` in
   `opportunity_audit.py`).
9. **A metric that never prints is not a metric that passed.** `drag_target` read
   zero rows for days: it was keyed on `TO_ACTIVE`, but Boss's Orders drags
   through **`SWITCH`**, and the opponent-only filter then dropped every row
   silently. **Check each row has a non-zero denominator before believing the
   table.**
10. **Moving an audit rate is not winning games.** The P4a rules took the drag
    from 85/99 to 99/99 and the conversion turns from 36.9% to 100%, then
    measured 0.489 and 0.493. Rule 3 one level up, in the *rule* pipeline instead
    of the training one. **Arena-A/B every rule, no exceptions.**
11. **Prefer rules that delete a DOMINATED option; distrust rules that pick a
    side in a TRADEOFF.** The project's most reliable predictor, **3 for 3 and
    0 for 4** (table in `EVIDENCE` §3). The net has watched 2,810 games of humans
    making those trades and is as good at them as our arithmetic; what it
    *cannot* see is HP, damage and attached energy. **Before writing a rule, ask
    which column it is in.**

    ⚠ **Be strict — "dominated" is easy to talk yourself into.** `counter_source`
    was filed as dominated because the heavily-damaged source is better *both* on
    damage transferred and on healing. The first is arithmetic; the second is a
    **judgment** (a heal is only worth it if the Pokemon is savable) that was
    asserted, not measured. It then won the arena and read 762 on the LB. **A
    rule is only dominated if EVERY dimension it moves is arithmetic — one
    judgment puts it in the tradeoff column no matter how good the other looks.**
12. **A single-anchor arena will eventually lie to you, and on 2026-07-30 it
    did.** Everything routine was measured against `rule:v10,noS` on
    `lucario_v10` — ⚠ **which we then wrongly wrote off as "0% of the meta"; it
    is 12.8% of the field we actually play (§8i), and this rule's own example
    below is the good part of the story, not that clause.** Re-anchored on
    `rule:crustle`, **`chip_target` — the rule that bought ~150 LB points and
    defined the project's whole method — measures −0.126, i.e. actively
    harmful**, while it is worth +0.077 in the mirror (`EVIDENCE` §8c).
    **An arithmetic rule encodes an objective, and an objective is only correct
    while the strategic context holds.** So:
    - **Every rule A/B needs ≥2 anchors, one of them adversarial**, and every
      archived number carries an anchor label.
    - **A rule that wins on one anchor is a matchup branch candidate, not a
      shipped rule**, until a second anchor agrees.
    - Also: a pattern the user watched in a real game can be genuinely absent
      locally. **When the local audit says it never happens, measure it on
      `replays/submission_replay_2026-07-29/`** — `scripts/p5a_replays.py` reads
      our real selects against 54 distinct LB opponents.
    - ⚠ **An anchor must be COMPETITIVE to resolve anything.** `bc` piloting an
      off-distribution deck gave a 0.911 blowout — a ceiling that squeezes any
      rule delta to nothing. Import a real pilot (`import_crustle_agent.py`).
13. **Check the denominator is a real CHOICE, not just a real count.** P5a read
    "the rule takes the best target 26/26" — but 90 of its 95 pooled-KO rows
    offered only one prize value, so nothing could go wrong. The honest
    denominator was **5**. A rate over forced moves measures nothing.
14. **SIZE BEFORE YOU BUILD: a dramatic per-instance number says nothing about
    frequency, and frequency is where rules die.** "Morgrem deals 60 through the
    wall while Grimmsnarl ex deals 0" was filed as *the biggest known lever in the
    matchup*. Sized: the rule would fire **~0.2 times per game**, the free version
    of the same out was **already taken 95.4%** of the time, and the effect
    (~2.6% of our damage output) was **smaller than an n=2000 A/B can resolve**.
    Closed for the price of one probe and no A/B (`EVIDENCE` §8e). This is rule 10
    one stage earlier: **moving an audit rate is not winning games, and counting
    an opportunity is not finding one.** Ask "how often, and how big per
    instance?" *before* writing code — and check whether the cheap version of the
    behaviour already happens.

    ⚠ **A corollary that caught this one:** state the rule's *alternative*
    explicitly and measure it too. The whole argument rested on the alternative
    being worth zero; it was not (the bench snipe kills their basics), and nobody
    would have noticed without asking what the other branch actually does.
15. **RE-READ THE CODE THAT THE WHOLE METHOD RESTS ON. The project's founding
    premise was false for eight days and nobody checked.** "The net cannot see
    HP" was written in `targeting.py`, repeated in `HANDOFF`, and used to justify
    every rule — while `features.py` had been feeding the net per-slot HP,
    damage, energy and prize value since v1. The true gap was one line's worth:
    `opt["index"]` was never encoded, so two options naming two copies of the
    same card were **bitwise identical inputs with different right answers**.
    Fixing that measured **0.878** and made the rules harmful (`EVIDENCE` §8f).
    **A premise repeated in three files is not thereby verified — it is just
    load-bearing.** When a claim about the code justifies weeks of work, open the
    file and confirm it, especially if it has never been questioned.

    ⚠ **The general form, and the thing to carry forward:** ask whether a blind
    spot is **informational** (the input is absent) or **representational** (the
    input is present but cannot be bound to the decision). They look identical
    from the outside — the agent gets it wrong at chance — and they have opposite
    cures. Four hand rules cured the symptom; 12 features cured the cause and
    dominated them.
16. **AN ARENA RESULT IS A WEIGHTED AVERAGE OVER YOUR ANCHOR SET, AND NOTHING
    ELSE. State the weights before you read the score.** ✅ **Resolved
    2026-07-31** (`EVIDENCE` §8i) — the earlier version of this rule said the
    arena does not measure ladder strength and treated that as the project's
    central problem. **That was wrong, and believing it would have cost far more
    than the original mistake.** The arena predicts fine. Both LB "contradictions"
    were the same error: the anchor set did not span the field.

    - v3 measured **0.661 in the mirror** and **0.505 vs `rule:v10`** (P4b:
      0.576). Both are true. Only the first was in the anchor set, and the
      ladder averaged over both.
    - **Weight every anchor by its measured share before concluding anything.**
      "Wins 2 anchors, loses 1" is not a verdict; "+0.16 on 13.8% and −0.07 on
      12.8%" is the start of one. `p9_field_census.py` supplies the shares.

    🔴 **AND THE SHARES ARE AN ESTIMATE WITH AN INTERVAL — quote it (day 22,
    §8ay).** They come from **75 replays**, so the mirror's 33.3% is really
    **[22.5%, 43.2%]**. Every published `±` on a weighted ΔW — p33's ±0.0050,
    E8's ±0.025 — is **game-sampling noise only and treats the weights as
    exact**. Bootstrapped, weight uncertainty adds ±0.0023–0.0031, so p37's
    honest interval is **±0.0059, not ±0.0050**.
    ⚡ **It bites in proportion to how much the per-anchor deltas DIFFER.** All
    same sign ⇒ reweighting barely moves the sum. Mixed signs, or one anchor
    carrying the result ⇒ the weight is doing real work and its interval belongs
    in the answer.
    ⚠ The census itself was also **wrong** until day 22 (evolution lines keyed by
    card id, 228 broken links, 6 of 75 games mislabelled) — but every correction
    landed *inside* the n=75 interval, which is the point.

    ⚠ **And the deeper trap, which is the one to actually carry forward:**
    **CHECK WHERE YOUR POPULATION DATA COMES FROM BEFORE YOU LET IT RETIRE AN
    ANCHOR.** `fetch_top_episodes.py` mines the **top** episodes by `avg_score`,
    and Kaggle's daily datasets are **censored below `avg_score` 1055** — the
    800–1000 buckets are literally empty. We play at 825–952. So the mined meta
    was a perfectly accurate description of a population we never meet, and it
    said `lucario_v10` was **0% of the field** when in our own games it is
    **12.8%**. Retiring that anchor on that evidence is what let B1 ship.

    **The general form: a sampling frame you did not choose is a hypothesis, not
    a fact.** Ask what the data-generating process excludes — not whether the
    numbers are right.

    ✅ **The positive control still holds, and it is why the arena is trusted
    again:** the arena predicted 0.770 vs `rule:crustle` and we won **76.9% of 13
    real Crustle games**. The arena is accurate exactly where the anchor
    resembles the opponent — which is now most of the field (71.6%), and was
    26.6% when B1 was decided.

    **Standing requirement: measure against the top-5 anchors, weighted, and
    never again let a mirror A/B alone decide whether to turn a rule off** — the
    mirror is 13.8% of reality.
17. **HELD OUT BY GAME IS NOT HELD OUT BY PLAYER — and the identity you are
    holding out is not stable.** Day 11 tried to explain §8q's rating trend and
    found that the obvious confound could not even be *tested* with the corpus as
    built: the trainer splits on `gid % 20`, so **every demonstrator in the
    held-out split also appears in the training split**, and the "0 exposure"
    bucket was empty. A same-week, same-deck dump of **87 players the net had
    never seen a row of** was needed to answer it (the answer: exposure buys
    nothing, 73.6% unseen vs 69.3% for the most-trained-on). **Before believing
    "the net generalises to player X", check whether X is in the training set —
    the split does not do it for you.**

    ⚠ **And the key is a display name, which means it is not a key.** Naive
    matching left 24.6% of one day's seats unrated; **182 of those 198 misses
    were the LEADERBOARD'S #1 TEAM**, appearing as `James Cox`, as `zoroark190`
    (a member username) and as the post-merge `James Cox & Henry Chao`. §8q hit
    the same bug on `Sixth Sense` / `Raja Biswas`. **Teams rename and merge
    mid-competition, so any per-player statistic silently splits your most
    valuable demonstrator into three and reports it as sparse data.** Resolve on
    `teamId` from the episode sidecar where you have one; match member usernames
    exactly; keep verified renames in `replays/team_aliases.tsv`; and **print the
    match rate and the biggest unmatched names every time** (rule 9).

18. **🔴 DO NOT RE-DERIVE A STATISTIC THE TOOL ALREADY PRINTS — and when two
    numbers for the same quantity disagree, that disagreement is the most
    valuable signal you will get all day.** Day 17 read `winner == 0` out of an
    arena archive as "agent A won" and reported the Crustle re-run as **v3 0.489
    / v4 0.510 / v5 0.502** — three numbers hugging 0.5, from which a draft
    concluded the pilot repair "moved things in both directions, none of it
    significant". **`agent0`/`agent1` are seat-indexed and `arena.py` alternates
    seats every game**, so that computation averaged every net together with its
    own opponent. The true values were **0.857 / 0.888 / 0.870**.

    ⚡ **§8ae documented this exact bug five days earlier** and it was committed
    again anyway — because the earlier fix was applied to `p21`, and this was a
    throwaway analysis snippet, which felt exempt. **Nothing is exempt.**

    ✅ **What caught it was `arena.py`'s own summary line printing `score=0.888`
    next to the snippet's 0.510.** Every arena run already emits the
    seat-corrected score, the CI, and the per-seat W/D/L. Reading the archive by
    hand re-implements all of that and can only *introduce* error.

    ⚠ **The general shape, and it is the FIFTH instance this week** (§8ad's
    equivalence test that could not fail, §8ae's seat bug, §8af's third-in-two-days,
    §8ah's overcounting detector, this): **a bug in an analysis script produces a
    PLAUSIBLE NUMBER, not a crash.** Three of the five biased toward the null,
    which reads as a finding. The defence is not care; it is **redundancy** —
    compute the headline number a second way, or take it from a tool that
    already computes it, and reconcile the two before writing a word.

    ⛔ **Corollary, paid for the same day:** `train_policy.py` exported the
    checkpoint with the best `val_top1`, so an arm whose objective deliberately
    departs from corpus fit would have exported an *earlier epoch* than its
    control and the A/B would have measured training length. **Any A/B between
    two training arms must pin the checkpoint rule (`--export-last`), not just
    the data and the seed.** Rule 3 is not only about what you *report* — it is
    about what the training loop *selects*.

    ⚠ **A third form, same session: a control population built from "the
    opponents of X" contained US.** `Scio` is on that list because we played
    them, and our own agent's selects are exactly what the net was fitted to —
    left in, it scores ~98% against itself and inflates the control.
    **`--exclude` yourself from any population you intend to treat as
    independent.**

19. **🔴 AN ANCHOR IS A FILE, AND A NUMBER QUOTED FROM IT IS ONLY VALID FOR THE
    VERSION THAT PRODUCED IT. Check the source is OLDER than the archive before
    quoting a score.** §8ap's anchor table published `rule:crustle` at **0.866**
    and called it "the shipped guard-only pilot". `83daa48` replaced that pilot
    at **17:48:55**; the last game behind the 0.866 finished at **17:22**. The
    repo held a **fourth** version, verified on **six games**, and three
    documents quoted the old number for a day. Measured at n=2,000 it is
    **0.755** — and the one-line tie-break separating them is worth **0.111**,
    larger than the entire empty-bench repair it was a footnote to (§8aq).

    ⚠ **This is NOT the "buggy script, plausible number" shape** that §8ad,
    §8ae, §8af, §8ah and §8an's seat bug all had. **Both scripts were correct
    and the world changed between them.** Redundancy (rule 18) does not catch
    it; only a timestamp does:

    ```powershell
    git log -1 --format='%cd' -- agents/agentkit/rulebased/sources/<pilot>.py
    python -X utf8 -c "import json,time;r=[json.loads(l) for l in open('out/arena/<run>.jsonl')];print(time.ctime(r[-1]['ts']))"
    ```

    ✅ **Swept over all seven anchors on day 18: Crustle was the only drift.**
    ⚡ **The cheap standing defence, and it is what caught this one:** any script
    that plays an anchor should **print that anchor's arena score beside its own
    output**, so a changed instrument announces itself on the next run instead
    of on the next audit.

    ⚠ **This rule's own example turned out to be mostly rule 20.** The 0.866 and
    the 0.755 were also two different DECKS, and the deck is the bigger term
    (§8ax). The rule is right; the story attached to it was half wrong, which is
    itself the lesson — a timestamp check answers the question it was designed
    for and reports nothing about the one nobody asked.

20. **🔴 AN ANCHOR IS A FILE *AND AN ARGUMENT*. Check the DECK a pilot was run
    on before comparing two of its scores.** A `rule:<name>` pilot is tuned for
    exactly one 60 and plays any other through a generic fallback —
    `decks/crustle_v1.py` says so in its own docstring, and HANDOFF §3.2 carried
    an n=20 probe measuring it at **+0.08**. `rule:crustle` was nonetheless run
    on `crustle_v1` in `p10/p19/p20/p34/p35/p37` and on `crustle` in
    `p27/p28/p54/p56/p57`, **archived under one identity both times**, and the
    published score tracks the deck (0.748–0.768 vs 0.866–0.870) and not the
    pilot version. Measured directly with the pilot held fixed: **+0.140**
    [±0.031, n=2,000/cell] — larger than either pilot effect §8an and §8aq
    published. §8ax.

    ⛔ **Rule 19's timestamp check cannot see this.** Both scripts were correct,
    both source files were older than their archives, and the *call site*
    differed. The only thing that catches it is the identity carrying the deck.

    ✅ **Fixed in the tool, not just in the rule:** `arena.build_agent` now
    archives `rule:<name>@<deck>` and prints a loud warning when the deck is not
    the one `agentkit.rulebased.DECK_MODULE` names for that pilot.

    ⚡ **The general form, and it is the fourth instance:** *the identity a
    result is filed under must contain everything that can change the result.*
    §8aq was the pilot source; §8ax is the deck; day 22 also found `bc` archived
    with **no net** (1,226 games over four days of a moving `sa/policy_net.npz`)
    and a `net=` **path** that a retrain can silently repoint. All four now
    carry their content in the archived name.

21. **🔴 SIZE *AND RANK* PER TURN, NOT PER DECISION — a within-turn ordering
    difference inflates a per-decision count without changing what happens.**
    F1's largest mirror cluster was "the clone wants Munkidori's ability and the
    1150+ pilot does not": **75.1% vs 38.5%** of the decisions where it is
    offered, 8.4 confident disagreements per game, top of 519 clusters. Per
    **turn** it is 96.9% vs 93.8%, and an on-policy control — the shipped agent
    made to play 80 mirror games and mined with the same miner — reads **6.42
    uses/game against the experts' 6.23.** Identical behaviour; the ability is
    *"Once during your turn"*, so wanting it at action 1 rather than action 4
    generates a disagreement on every decision in between. **The per-decision
    view overstated the cluster by ~25×.** §8bj.

    ⛔ **This is §8ai's empty-bench detector, third instance** (the first
    overcounted "declines to bench" when the pilot benched later in the same
    turn; the second made `rule:archaludon` look worse than a broken pilot).
    Rule 14 already says *size before you build* — this says **the unit of the
    size must be the unit the effect lives in**, and for anything that happens
    once per turn that unit is the turn.

    ✅ **`scripts/p67_option_rate.py` is the tool**: availability, take-rate and
    per-turn use for one option class, on any corpus, with the demonstrator and
    the net scored side by side (on our own games those two columns must agree
    exactly — a free positive control that exercises the whole chain).

---

## 3. THE PLAN (day 9 → day 10)

**Day 9 closed the one question the whole project was blocked on** (§3.4): the
arena/ladder gap is an anchor-coverage problem, not an instrument problem, and
the anchor set is now rebuilt to 71.6% of the field. **The arena is trustworthy
again, with the weighting discipline in rule 16.**

**Day 8 closed all three of day 7's open questions** (§3.0 `counter_source` is
good and stays; §3.1 we were *not* measuring against the right opponents, and the
correction found a harmful rule; §3.2 the Crustle premise is verified) **and
shipped a fix (§3.3).** It also killed ROADMAP B2. The live work is now the
▶ START HERE list at the top of this file; §2.5 of `ROADMAP.md` holds the ranked
breakthrough candidates (B1, B3–B5) that run alongside.

### 3.0 ✅ RESOLVED (2026-07-30): `counter_source` stays

Re-measured against the new meta anchor: **`counter_source` is worth +0.052 vs
`rule:crustle`** (0.559 with, 0.507 without, n=2000 each) — *more* than it was
worth in the mirror (+0.034) or vs `lucario_v10` (+0.033). The LB scare was an
artifact of reading a ±75-point instrument at 12-Elo precision (§1). **Keep the
rule; no rollback.** `EVIDENCE` §8c. History below, kept because the reasoning is
report material.

### 3.0b The original write-up — "unresolvable on the LB"

**Readings (all read against the contemporaneous score, never a remembered one):**

| when (UTC) | P6a `55077709` | P4b `55072063` | gap |
|---|---|---|---|
| 07-29 09:21 | submitted (μ=600) | — | — |
| 07-29 10:22 / 10:27 | 762.2 → 746.4 | 970.1 | ~224 |
| 07-30 08:19 (**+23 h of play**) | **824.9** | **948.1** | **123** |

**The two agents are converging toward each other from opposite directions** —
P6a +78 while climbing off μ=600, P4b −22 while settling off an overshoot. That
is the signature of two close true ratings, not of a 220-point regression. **Do
not read the remaining 123 as the rule's effect**: see the resolution limit in §1
— the rule is worth ≈ +12 Elo and this instrument has ±50–100 of swing, so **the
LB cannot answer this question and no further reading will change that.**

**Therefore §3.0 is closed as an LB question and re-opened as an arena question.**
The decisive experiment is local and is §3.1 step 5: **re-measure `counter_source`
against the post-shift anchors.** If it wins against all of them, keep it and the
story is "the LB was never able to see a 12-Elo rule". If it loses against
post-shift anchors, we have both the diagnosis (Cause A) and the fix (Cause B).

- **Cause A — the local anchor is stale.** Both "independent" confirmations share
  the same opponent deck and the same era of the meta. Fix: §3.1, then
  re-measure. **This is now the leading hypothesis for anything that looks like a
  rule/LB disagreement.**
- **Cause B — the dominance argument was half wrong.** *Transfer* is dominated
  (3+ counters move 30, 1 moves 10). *Healing* is a tradeoff that was asserted:
  moving 30 off our most-damaged Pokemon is only the best heal if that Pokemon is
  **savable**. The clone may have been judging that correctly with information
  the rule discards. Fix: a narrower rule — **redirect only when the net's pick
  is strictly worse on transfer AND not obviously the better heal** (e.g. leave
  the net alone when its pick is the Active, or when the max-counter source is
  already beyond saving: HP ≤ incoming damage). That keeps the arithmetic and
  returns the judgment to the net. **Worth building regardless of the verdict** —
  rule 11's ⚠ clause says the current rule is mis-classified, and the narrow
  variant is the version that belongs in the dominated column.

**⛔ There is NO free submission slot, and the rollback is worse than pointless.**
Corrected slot arithmetic (the earlier note in this file was wrong — it said "be
willing to lose `55054446`'s slot", but **`55054446` is already inactive**):

- Active pair today = `55077709` (824.9) + `55072063` (948.1, our best).
- **Any** new submission → active = {new, `55077709`}, i.e. it **evicts
  `55072063`** and freezes our best at ~948.
- A `counter_source=False` rollback *is* `55072063`'s agent. So rolling back
  would evict a 23-h-converged 948.1 agent in order to restart **the identical
  code** from μ=600 and spend 4+ h climbing back. **Never do this.**
- **The next submission must therefore be something we expect to beat 948, not a
  rollback and not a small rule.** That is a high bar, and it is the real reason
  §3.1 and the ROADMAP §2.5 breakthrough candidates (B1/B2/B4 — none of which
  cost a submission slot) are the priority.

**Preserved builds — labels VERIFIED and both smoke-tested 2026-07-31**, so
nothing needs rebuilding under time pressure:

| tarball in `dist/` | bundled rule flags | = submission | restores? |
|---|---|---|---|
| `...20260729-103819.tar.gz` | chip, spread (**no `counter_source` in the signature at all**) | **`55072063` — the 950.2 agent** | ✅ `NET_OK`, full game, 0.1 s pool |
| `...20260729-152103.tar.gz` | + `counter_source` | `55077709` (824.9) | ✅ `NET_OK`, full game |
| `...20260730-151057.tar.gz` | + `chip_wall_defer` | **never submitted** (day-8 wall branch) | — |
| `...20260731-000752.tar.gz` | **rules OFF + v3 net** | **not submitted yet** (item 0) | ✅ `NET_OK opt_in=37` |

All three lw2 bundles carry the same net (`sha256 bba02a42…` = `out/policy_lw2.npz`
= the live `agents/sa/policy_net.npz`) and **their own copies of `sa/` and `cg/`**,
so later repo changes cannot break them — the 07-29 bundles still report
`opt_in=n/a` because they predate that property, and they run fine.

⚠ **"Restorable" is NOT "recoverable to 950."** Re-submitting the P4b bundle
restarts it at **μ=600** and it must climb 4+ h; the 950.2 rating itself cannot be
restored, only re-earned. So the insurance is against *losing the code*, which we
have not, and never against a bad submission decision.

### 3.1 ⚠ Re-anchored (2026-07-30) — and re-anchored AGAIN on 07-31, see §3.4

> 🔴 **The premise below is retracted.** Day 8 rebuilt the anchor set because
> `lucario_v10` was "0% of the meta" — true only of the top-1150 band. **Adding
> `rule:crustle` was right; dropping `rule:v10` was the mistake**, and §3.4 put
> it back. The current anchor set is §4's five-deck table. Kept because the
> Crustle work in it is sound and the reasoning is report material.

Every number in §3, §6 and `EVIDENCE.md` was earned against `lucario_v10`, which
was believed to be **0% of the meta** (§1). So the bar itself has to be rebuilt.

- ✅ **1. Fetch + mine.** 07-28 and 07-29 fetched (400 each); **07-30 is not
  publishable yet — its dataset 403s, episodes appear the following day**, so the
  newest available day is always yesterday. Both meta snapshots are archived in
  `out/meta/`.
- ✅ **2–3. Rank and diff.** Table in §1. The delta is the report's meta-shift
  figure.
- ✅ **4. Reconstruct the new opponents.** `decks/crustle.py` (rebuilt from the
  current 77×-seen list) and `decks/crispin_toolbox.py` (135×-seen). Both resolve
  to 60 cards.
- ⏭ **5. THE WORK: re-run every shipping A/B against the new anchors.** At minimum
  `bc` vs `bc:x,noSrc` / `bc:x,noChip` / `bc:x,noSpread`, n≥2000, against **each**
  anchor. A rule that wins against all of them is real; one that wins only
  against `rule:v10` was never measured properly. **This is also what settles
  §3.0.**

✅ **The pilot blocker is SOLVED for Crustle: `rule:crustle` now exists.**
`scripts/import_crustle_agent.py` lifts the public `pixiux/ptcg-crustle-v1-submit`
agent (409 lines of readable option scoring) into
`agents/agentkit/rulebased/sources/crustle.py` + `decks/crustle_v1.py` (its own
tuned 60), registered in `DECK_MODULE`. Idempotent. It plays the real lockdown:
bench Dwebble → evolve → arm 3 energies incl. Grass → Hero's Cape → Battle Cage
stadium → heal with Jumbo Ice Cream / Cook at damage ≥50 → retreat to a ready
Crustle → Superb Scissors (479, **120 damage**, ×2 into Grass weakness).

```powershell
python -X utf8 scripts/arena.py play bc rule:crustle `
    --deck-a grimmsnarl --deck-b crustle_v1 --matches 1000
```

⚠ **Two Crustle decks, and the difference matters.** `crustle_v1` is the pilot's
own list — use it when you want the strongest Crustle we can run locally.
`crustle` is the **field consensus** list (77×-seen). The pilot scores ~20 of the
consensus list's cards through a generic fallback, so it plays them legally but
badly; early n=20 probes read 0.620 on its own list vs 0.700 on the consensus
one, in the direction that confirms this.

⛔ **`crispin_toolbox` has no pilot and CANNOT GET ONE from public code — the
search is complete, not pending (2026-07-30).** All 272 public notebooks were
enumerated; there is no Crispin/toolbox pilot, and **no public author outranks
us** (details and the refuted-title table are in the ▶ START HERE item 2 above).
The first attempt already showed why a substitute won't do: `bc` piloting it
scored 0.089 — **we beat it 0.911 [0.898, 0.923] at n=2000**, and an anchor we
beat 91% of the time has almost no resolving power for a rule worth ~1 pp because
the ceiling squeezes the delta. **A `bc`-piloted anchor is not good enough; do not
spend A/B time on one.** Rule 12's ≥2-anchor bar is met by the mirror +
`rule:crustle` in the meantime.

**Public notebooks worth mining (pulled to `notebooks/pulled/`, 2026-07-30):**

| ref | why |
|---|---|
| `pixiux/ptcg-crustle-v1-submit` | ✅ imported — `rule:crustle`. **Its competitiveness rests on our own number (we score 0.663), not on the title** — `pixiux` does not appear on the LB at all |
| ~~`makthanithin/pokemon-tcg-ai-battle-1084-5-baseline`~~ | ⚠ **DO NOT TRUST THE TITLE.** "1084.5" is the author's self-report. Checked against the full LB: they are **`Nithin maktha`, rank 750, 819.1** — **hundreds of places below us**, and no `makthanithin` appears at all. **A notebook title is not a measurement** (rule 10). Kept only as a lesson |
| ~~`soutasakurai/max-elo-1208-libraryout-w-crustle-great-tusk`~~ | ⚠ **THE SAME TRAP, SECOND TIME.** "Max Elo 1208" — the author is **rank 3439/4000 at 605.0, below the μ=600 start.** Pulled and rejected 2026-07-30 |
| ~~`prvsiyan/ptcg-ai-battle-static-deck-tusk-1208-v24`~~, ~~`pcxxxxxx/explainable-ptcg-agent-with-legal-ogerpon-deck`~~ | ⚠ also pulled, also refuted: 789.1 (rank 1083) and 686.6 (rank 2454) |
| `jazivxt/crustle-counter-al220-v29-agents-only` | someone else's *anti-Crustle* agent — directly Track C |
| `kokinnwakashuu/ptcg-lucario-public-lab-anti-crustle-log` | anti-Crustle analysis + logs |
| `prvsiyan/ptcg-ai-battle-control-v11-meta-portfolio` | "meta router"/portfolio = ROADMAP B3 (archetype detection → matchup branches) |
| `busyaprime/what-actually-wins-on-the-ladder`, `myso1987/...deck-meta-by-score-band` | independent meta analyses to cross-check our mining against |

⚠ **Do not treat a cross-deck score as skill** (rule 5) — use each new anchor the
way `rule:v10` was used: a fixed opponent for A/B *deltas*, both sides facing the
identical opponent. And **archive the per-anchor tables**; they are the rubric's
consistency/robustness exhibit and go into the report verbatim.

⚠ **Do not treat a cross-deck score as skill** (rule 5) — use each new anchor the
way `rule:v10` is used: a fixed opponent for A/B *deltas*, both sides facing the
identical opponent.

Also: **archive the per-anchor A/B tables.** They are the rubric's
consistency/robustness exhibit and go into the report verbatim.

### 3.2 ✅ Crustle — **this is the meta now**, piloted, and the premise is verified

**Measured (§1): 1 seat in 1,600 pre-shift → 18.1% of the field at 56.6% WR on
07-29, with the LB's top two players on it, while our win rate fell 52.2% →
47.5%.** Crustle is no longer a curiosity to probe eventually; it is the most
likely single explanation for our ceiling.

`decks/crustle.py` has been **rebuilt to the current 77×-seen list** — the old
reconstruction was 12 slots stale (it ran 4× Crushing Hammer, which the current
list drops for Colress's Tenacity / Tool Scrapper / Battle Cage / {G} Energy).
Notable contents: Dwebble ×4 / Crustle ×3, Cornerstone Mask Ogerpon ex, Mega
Kangaskhan ex ×2, Jumbo Ice Cream ×4, Boss's Orders ×4.

**⚠ VERIFY THE PREMISE FIRST — one probe, before anything else.** The whole line
rests on *"Adrena-Brain and Freezing Shroud move/place damage counters, which is
not damage from an attack, so **Mysterious Rock Inn** should not prevent them."*
Mysterious Rock Inn is an **ABILITY on Crustle itself** (card 345; 344 is
Dwebble) that prevents damage from opponent {ex} attacks — and Grimmsnarl ex is
`ex=True`, so Shadow Bullet should deal **zero**. Our card db exposes no ability
text for 345 (`abilities: None`), so this cannot be settled by reading — only by
playing. **If counters do not bypass the prevention, the entire passive-damage
line is dead and no decklist work should happen.** (The `probe_adrena.py` pattern
that settled P4b's four mechanics is described in `EVIDENCE` §5; the script
itself is no longer on disk — write a fresh throwaway probe.)

Also unverified: that Grimmsnarl ex really deals **zero** to Crustle.
`attack_into_ex_immune_active` has been in `opportunity_audit.py` for days and
has **never fired**, purely because there was no Crustle deck to fire against
(rule 9). **It can fire now.**

**Two things missing, both blocking:**

1. **No pilot.** A decklist alone cannot reproduce the lockdown — the wall only
   works if the pilot sets it up and sits behind it. `bc` plays it
   off-distribution; `rule:v10` is Lucario-specific scoring. Options: find the
   public Crustle bot (`dashimaki360/beating-the-day-1-1-crustle-bot` implies one
   exists) or write a minimal rule pilot. **A weak pilot under-reads the matchup
   and makes the hole look smaller than it is.** No deck experiment is
   interpretable until this exists (ROADMAP Track C step 1).
2. **The `crustle-replays/` directory the decklist docstring cites is not in the
   repo** — only the decklist survived. Ask the user if the source games are
   needed.

**User's idea, recorded but not committed to:** lean into passive damage
(Adrena-Brain, Freezing Shroud) either by (a) more copies or (b) prioritising
those Pokemon when fetching. Established facts: **Munkidori is already at 4, the
copy cap** — only the Froslass line (2 Snorunt / 2 Froslass) can grow, which
badly weakens (a); the one decklist variant ever measured scored 0.490 n=2000;
and any change is off-distribution for the net. **(b) has the better prior** — no
cards change, and *conditional on the matchup* "fetch the Pokemon whose damage
actually goes through" is near-dominated rather than a tradeoff. It lands on
`TO_HAND` (15.3% of selects, where only duplicate-avoidance has been closed).

### 3.3 ✅ FIXED AND SHIPPED (2026-07-30): the matchup branch

`chip_target` now defers to the net when the opponent's Active is a wall
(`targeting.WALL_POKEMON = {345}`), **ON by default**, `bc:<label>,noWall` to
disable.

| variant | vs `rule:crustle`, n=2000 |
|---|---|
| `bc` before (unconditional) | 0.559 [0.537, 0.581] |
| **`bc` now (branch on)** | **0.663 [0.642, 0.684]** |
| `bc:x,noChip` (ceiling) | 0.685 [0.665, 0.705] |

**Recovers 82% of the −0.126**, and the mirror control reads 0.521 [0.490, 0.552]
n=1000 (contains 0.5 — no bleed, and none is possible by construction).

⚠ **Do NOT submit this alone.** It is worth ~+10–15 Elo overall (a +0.10 swing in
18% of the field), which is **below the LB's resolution** (§1). Bundle it.
Remaining headroom: a wall-aware *ranker* instead of deferral, worth at most the
0.663 → 0.685 gap. **Next, bigger, and in the same matchup: the Morgrem out
below.**

### 3.3b The original diagnosis (kept — it is the report's argument)

**The measured defect** (`EVIDENCE` §8c): vs `rule:crustle`, `bc` scores **0.559**
and `bc:x,noChip` scores **0.685** — **our founding rule costs us 12.6 points of
score in 18% of the field.** In the mirror (52% of the field) it is worth +0.077
head-to-head, so **do not delete it — branch it.**

**Why it fails, measured:** `chip_target` ranks "dies to 30 first, most prizes
among those, then lowest HP", which against Crustle farms **Dwebble** (a 1-prize
basic) while the immune wall sits untouched. Counter-placement events onto Dwebble
drop **235 → 24** when the rule is off, and events onto Crustle rise **1,386 →
1,583** at a higher mean (12.9 → 15.0).

**The rule to write** — and note it is a *dominated-option* rule by rule 11, which
is the 3-for-3 column: **when the opponent's Active cannot be damaged by our
attacks, damage counters are the only way to remove it, so concentrate them
there** rather than spending them on a killable basic. The condition is factual,
not a judgment: we can test "would our attack deal 0 to this target" directly
(that is what `best_damage` / the census measures), so this is arithmetic, not a
guess about what matters.

**Design sketch (implement in `targeting.py`, default OFF until A/B'd):**

1. Detect the immune-wall condition per target, not per archetype: for the
   opponent's Active, `best_damage(our_active, ...) == 0` while a counter effect
   is available. That generalises past Crustle to any prevention ability, and
   needs no archetype classifier — **so it is cheaper than B3 and should be tried
   first.**
2. When it holds, rank counter targets by "damage that actually lands, most on
   the blocker" instead of by killability.
3. A/B against **all three** anchors: `rule:crustle`, the grimmsnarl mirror, and
   `rule:v10` (for continuity with the archived numbers). It must not bleed the
   mirror.

⚠ **Also test the cheap alternative first:** simply switching `chip_target` off
when the opponent's Active is undamageable is a one-line version of the same idea
and already has a measured +0.126 upper bound in this matchup. **Measure the
one-liner before building the ranker.**

~~🆕 **And a second, independent out from `EVIDENCE` §8d: Marnie's Morgrem
(non-ex) deals 60 through the wall while Grimmsnarl ex deals 0.**~~
❌ **CLOSED BY SIZING 2026-07-30 — do not build it** (`EVIDENCE` §8e,
`scripts/p7_morgrem.py`, `out/logs/p7_morgrem_200.txt`, 3× 200 games).

| measurement | result |
|---|---|
| turns the evolve-veto would actually fire | **38 / 49 / 53 per 200 games** = ~0.2/game |
| Morgrem Active vs a wall but **cannot pay {D}{D}** | 66% of such turns |
| **post-KO promotion into a wall** — the *free* route, no retreat cost | **288/302 = 95.4% already promote the Morgrem** |
| damage healed back off their Crustle | **22.5%** — the 60 is worth ~47 net |
| attack damage onto their **Dwebble** | **82 events, mean 73.9, 0 prevented** |

**Three reasons, any one sufficient.** (1) ~0.2 firings/game × ~47 net damage
against the ~352/game we already land = **~2.6%**, and an n=2000 A/B resolves
±0.021 — **the instrument cannot see it** (§1, now applied to the arena, not the
LB). (2) The cheap version of the out is already taken 95.4% of the time — the
"316/316 lethals, all forced" shape. (3) It is a **tradeoff**, not a dominated
option: 60 onto a healing 150-HP wall vs 30 onto a 70-HP Dwebble that *dies* to it
plus 220 more HP of body. Prizes are a genuine tie (1 per hit either way: ex = 2
prizes and survives exactly two 240s; Morgrem = 1 prize and dies to one), which is
what made it look dominated on paper — but "which target matters" is a judgment,
and rule 11's ⚠ clause is explicit that one judgment is enough.

⚠ **And it corrected a load-bearing sentence.** "Our main attacker deals 0 into
theirs" is true of their **Active only**. Shadow Bullet's 30 bench snipe is
**unprevented**, and onto a 70-HP Dwebble it kills the Crustle line's basics. Any
future anti-wall play is measured against *that*, not against zero.

⚠ **Not closed:** the retreat/promotion route — 451 turns per 200 games (2.3/game)
where Grimmsnarl ex attacks a wall for zero *with a Morgrem benched*. 10× the
denominator, but Grimmsnarl ex's retreat cost is **2** (the whole attack
investment), so it is a worse trade than it looks. Filed, not recommended.

### 3.4 ✅ RESOLVED (2026-07-31): the arena/ladder gap was anchor coverage

**The finding, in one line: the arena is accurate, and we retired the anchor that
would have caught B1 two days before B1 was decided.** Full write-up
`EVIDENCE` §8i; the numbers are in the top box of this file.

Three things came out of it, in decreasing order of how much they change:

1. **🔴 The public episode data cannot describe our opponents, ever.** Kaggle's
   daily datasets stop at `avg_score` **1055**; we play at **825–952**. This is
   censorship in the data-generating process, not a sampling choice we can tune.
   **`replays/submission_*` is the only evidence about our own field**, which
   makes those dumps the repo's most valuable asset and makes pulling replays
   after every submission a standing task.
2. **The anchor set is rebuilt to 71.6%** (§4's table) — `rule:alakazam5` and
   `rule:archaludon` imported, `rule:v10` reinstated.
3. **Rule 16 is rewritten** from "the arena does not measure ladder strength" to
   "an arena result is a weighted average over your anchor set" — with the
   sampling-frame warning as the general lesson.

**What is NOT resolved and is now item 0:** whether v3 is better than P4b once
all five anchors are weighted. Two of four runs are in; v3 loses Lucario and
wins the mirror and Crustle.

### The board

| | item | state |
|---|---|---|
| **§3.0** | is `55077709` (P6a) actually good? | ✅ **RESOLVED — yes, keep it.** +0.052 vs the new anchor |
| **§3.1** | re-anchor the arena on the current meta | ✅ **SUPERSEDED BY §3.4.** Day 8 re-anchored on the *mined* meta and that is what broke it: the mined meta is the top-1150 band, not ours. Day 9 re-anchored on **our own replays** — 5 anchors, 71.6% coverage. ⛔ `crispin_toolbox` stays pilot-less and is now **low priority: 0 appearances in 109 real games** |
| **§3.4** | why did the arena disagree with the LB? | ✅ **RESOLVED — anchor coverage, not the instrument.** v3 reads 0.505 vs `rule:v10` against P4b's 0.576, CIs disjoint (`EVIDENCE` §8i) |
| **§3.2** | Crustle premise probe | ✅ **VERIFIED — counters bypass the wall, AND a non-ex attacker gets through.** Track C steps 3–4 unblocked |
| **§3.3** | `chip_target` is HARMFUL vs Crustle (−0.126) | ✅ **FIXED AND SHIPPED** — the `wall_defer` branch recovers +0.104 |
| **§3.3b** | the Morgrem out (the non-ex attacker) | ❌ **CLOSED BY SIZING 2026-07-30 — do not build.** ~0.2 firings/game, the free route is already 95.4% right, and it is a tradeoff (`EVIDENCE` §8e) |
| **P2** | MAIN-decision rules, via the **lethal audit** | **lethal is CLOSED (2026-07-30): this deck has one attack, so the choice doesn't exist.** MAIN's arithmetic half is empty; what remains is tradeoffs |
| P1 | re-rank decks | **superseded by §3.1** |
| P6b/P6c, P5a/b/c, P4a/b/c | — | **all closed** — see `report/EVIDENCE.md` |

### P2 — the remaining mass (MAIN), and how to enter it

`context_accuracy.py` says **MAIN holds 3,930 of the net's 6,424 misses** (18,924
rows, 33.9% miss); `p6_recon` says MAIN is **47.7% of all selects with ≥2
options**. Every other bucket is owned by a rule, at measured parity, or measured
too small (`EVIDENCE` §7).

⚠ **Carry rule 11 in.** MAIN is mostly tradeoffs (which Supporter, attach now or
later, evolve or develop) — precisely where four straight rules did nothing.

**The arithmetic half of MAIN was the plan, and it is now measured EMPTY.**
`scripts/p2_lethal.py --matches 200` (2026-07-30) closed both cuts:

- **same-attacker cut: 316/316 lethals taken, and all 316 were forced** — the
  lethal was the *only* attack offered. Honest denominator **0** (rule 13).
- **needs-promotion cut: 7 of 803 no-KO turns** had a bench Pokemon that could
  KO, and **retreat was illegal in all 7**. Zero actionable cases.

**Why: Grimmsnarl ex has exactly one payable attack (Shadow Bullet, 180 flat), so
"which attack" is never a decision in this deck.** Missed lethal — the classic
handcrafted-agent edge — cannot exist for us. Full entry and its three
consequences: `EVIDENCE` §8 (including that a decklist change adding a second
attacker would *create* this decision class, unpatched).

**So do not write a lethal detector, and do not write a general MAIN scorer
either.** What is left in MAIN is tradeoffs, where hand rules are 0-for-4. The
live routes into MAIN are therefore ROADMAP **B1** (give the net the features
instead of writing the rule) and **B4** (sequence the whole turn rather than
score one option). `p6_recon.py` is the template for any further counter;
`p5b_check.py` is the template for confirming a rule fires (rule 9) before
spending an A/B on it.

### P3 — the abomasnow hole (open, low priority)

0.360 vs 0.475–0.519 elsewhere (pre-P2c, re-measure), and our selects/turn
collapse from 12.5–16.6 to **8.6** with shorter games — a lockdown, not subtle
misplay. Replay a loss with `SA_DEBUG=1` and read the actual select options.

---

## 4. What ships

`agents/sa/bcagent.py` `PolicyAgent` + `agents/sa/policy_net.npz` (= `policy_lw2`,
listwise, 2,810-game corpus, val top-1 0.6755) + `agents/sa/targeting.py`.
~1 ms/move, 0.1 s of the 600 s pool.

### Code map (`agents/sa/`)

- `bcagent.py` — **what we ship.** `net_path` pins an npz; each rule has a flag.
- `targeting.py` — **all the rule overrides. Every new rule belongs here.** Each
  has its own `PolicyAgent` flag and its own `bc:` arena switch, so any one can
  be A/B'd alone.

  ⚠ **EVERY NUMBER IN THIS TABLE IS CONDITIONAL ON THE `lw2` NET (2026-07-31).**
  Against the **v3** net the same three rules measure **0.427 together — actively
  harmful** (`EVIDENCE` §8f). They are *proxies for the option→target binding*, so
  once the features supply it the rules override a better-informed net with cruder
  arithmetic. **Read this table as "what the rules are worth to a net that cannot
  see its options' targets", not as a property of the rules.**

  **Two anchors per row now (rule 12).** Mirror = head-to-head vs the variant;
  Crustle = this variant's score against a fixed `rule:crustle`, so its rule
  value is the *difference from `bc`'s 0.559* (`EVIDENCE` §8c).

  | function | select | switch | mirror | vs Crustle |
  |---|---|---|---|---|
  | `chip_target` | DAMAGE / DAMAGE_COUNTER(_ANY) | `noChip` | 0.577 → +~150 LB | −0.126 unconditional 🔴 |
  | ↳ `wall_defer` branch | ditto, when their Active is a wall | `noWall` | no effect by construction (0.521 control) | **+0.104 recovered** ✅ |
  | `energy_spread` | MAIN, {D} ATTACH onto a Munkidori | `noSpread` | **0.702** n=4000 | **+0.193** ✅ |
  | `counter_source` | REMOVE_DAMAGE_COUNTER (ours) | `noSrc` | 0.534 n=2000 | **+0.052** ✅ |
  | `drag_target` | SWITCH (Boss's Orders' drag) | `drag`, **off** | 0.489 — null |
  | `drag_target(prefer_high_hp)` | ditto, KO-able tiebreak | `dragHi`, **off** | 0.490 — null |
  | `boss_converts` | MAIN, plays Boss's Orders | `boss`, **off** | 0.493 — null |
  | `boss_veto` | MAIN, suppresses Boss's Orders | `veto`, **off** | 0.493 — null |

  Three shapes, and **the shape predicts the result** (rule 11):
  - **Replace the whole ranking** — `chip_target`, `drag_target`. Fire only when
    *every* option is an opponent's Pokemon.
  - **Redirect the net's own pick** — `energy_spread`, `counter_source`. Never
    create or suppress an action, only change its target. Both need
    `full_rank(net, obs)` because MAIN and REMOVE_DAMAGE_COUNTER selects have
    `maxCount == 1`, so `choose()` returns one index with no runner-up.
  - **Force or suppress an action outright** — `boss_converts`, `boss_veto`. Both
    null. Both tradeoffs.
- `policynet.py` — numpy inference. `SA_PNET_PATH` env override; **dim guard**
  (stale net → `None` → fallback; never remove it).
- `features.py` (v2, DENSE_DIM=242, PER_SLOT=18) / `optfeat.py` (**v3 as of
  2026-07-30, OPT_DENSE 25 → 37**) — shared by trainer and inference.

  ⚠ **The project's stated blind spot was MISDIAGNOSED until 2026-07-30**
  (`EVIDENCE` §8f). "The net cannot see HP" is **false** — `features.py` has always
  given it per-slot HP, damage, energy and prize value for all 12 slots. The real
  gap: the v2 per-option vector encoded position only as *area* flags and **never
  encoded `opt["index"]`**, so two options naming two different bench slots were
  identical vectors — and two options naming **two copies of the same card were
  bitwise identical with different right answers.** That is exactly
  `energy_spread` (bare vs loaded Munkidori, and note it is the largest effect
  ever measured here, 0.702) and `chip_target`. **The rules restore a missing
  BINDING, not missing arithmetic.**

  **v3 appends 12 target-state features** (target HP, maxHP, damage fraction,
  dies-to-30, prize, energy count, own-type energy, ours/theirs, our damage into
  it, can-KO, and the **slot index**). ⚠ **Appended, never inserted** — dims 0..24
  are byte-identical to v2, and `policynet.Net.opt_in` derives each net's width
  from `head_in` and slices. **That is what lets a v2 and a v3 net run in ONE
  process for a head-to-head A/B (rule 4) across a feature change** — and it is
  also what stops a dim bump from silently falling the shipped net back to
  `list(range(minCount))`. **Do not replace `opt_in` with the global constant.**
- `evalfn.py` + `textdmg.py` — handcrafted eval / expected damage.
  `targeting.best_damage` wraps `textdmg.estimate` with weakness and energy
  payability and is what every damage-vs-HP rule should call. Approximate in
  general, **exact for this deck** — every attack grimmsnarl can pay for is flat
  damage. Same object as V10's `evaluate_state`; read both together.
- `agent.py` (`SearchAgent`), `planner.py`, `timemgr.py`, `worlds.py`,
  `tracker.py`, `fastsearch.py`, `valuenet.py` — the search path. **Dead
  (`EVIDENCE` §2); kept as the record.** `planner.py` imports `valuenet`, so
  don't delete pieces of it piecemeal.
- Both agents never raise: fallback = `list(range(minCount))`.

### The anchor set — five decks, 71.6% of the field (rebuilt 2026-07-31)

Shares and our win rates are from **our own 109 ladder games**
(`scripts/p9_field_census.py`, `out/logs/p9_field_census_pooled.txt`), which is
the only source that describes the band we play in (`EVIDENCE` §8i).

| anchor | deck | share | our WR | pilot |
|---|---|---|---|---|
| `rule:alakazam5` | `alakazam5` | **22.0%** | 66.7% | author reports **5th place**, pure rules |
| mirror: `bc` v `bc` | `grimmsnarl` | 13.8% | 60.0% | ourselves |
| `rule:crustle` | `crustle_v1` | 12.8% | 57.1% | `pixiux/ptcg-crustle-v1-submit` |
| `rule:v10,noS` | `lucario_v10` | 12.8% | **50.0%** | the LB-950 notebook |
| `rule:archaludon` | `archaludon_ex` | 10.1% | **45.5%** ⚠ | `a-sample-archaludon-75-wr…` |

⚠ **Weight by share, always.** Every A/B in this repo before day 9 is a number
against *one* of these — usually `rule:v10` (pre-07-30) or the mirror + Crustle
(07-30/31). **A pre-day-9 number is not wrong, it is partial**; check which
anchor produced it before reusing it.

⚠ **Two anchors are new and their per-rule deltas are unmeasured.** In
particular `chip_target`'s wall branch hardcodes `WALL_POKEMON = {345}`
(Crustle), and **Archaludon's Full Metal Lab is a second damage-reduction effect
it has never seen** (−30 into any Metal Pokemon, and Hero's Cape puts Archaludon
ex at 400 HP). That is the most likely reason we lose that matchup.

⚠ **`crispin_toolbox` remains pilot-less and is now also low priority** — it did
not appear once in 109 real games, which is consistent with §1's box: it was
16.9% *of the top-1150 band*.

#### `rule:v10` — retired on 07-30, reinstated on 07-31

`scripts/import_v10_agent.py` lifts the LB-950 notebook into
`agents/agentkit/rulebased/sources/v10.py` plus `decks/lucario_v10.py` (its own
retuned 60 — *not* `decks/mega_lucario_ex.py`). Idempotent. Flags: `noS` disables
its MCTS, `tb<sec>` sets its budget — **both are no-ops in practice because the
MCTS never runs**; pass `noS` anyway so the archived name records intent.
`rule:v10x` makes the search reachable (still falls back).

---

## 5. Commands

```powershell
# LB / submission status  (§3.0 step 1 -- read BOTH scores in this one call)
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(s.ref, s.date, s.status, s.public_score, '|', str(s.description)[:60]) for s in a.competition_submissions('pokemon-tcg-ai-battle')[:5]]"

# The leaderboard, top 20
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(i, r.team_name, r.score) for i, r in enumerate(a.competition_leaderboard_view('pokemon-tcg-ai-battle')[:20], 1)]"

# ⚡ THE FULL LEADERBOARD, ONE CALL (found 2026-07-31) -- USE THIS, not the
# pagination walk below. Writes out/lb/pokemon-tcg-ai-battle.zip containing one
# CSV of ALL 6,024 rows: Rank, TeamId, TeamName, LastSubmissionDate, Score,
# SubmissionCount, TeamMemberUserNames. Joining TeamName -> Score is what let
# day 10 rate every demonstrator in the training corpus (EVIDENCE §8q).
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_leaderboard_download('pokemon-tcg-ai-battle', path='out/lb')"

# ⛔ SUPERSEDED by the one-liner above; kept because it still works and the
# reasoning is report material. The client PRINTS "Next Page Token = ..." rather
# than returning it, so capture stdout and feed it back via page_token. This is
# how "1084.5 baseline" was refuted (its author is rank 750 at 819.1).
python -X utf8 -c "
from kaggle.api.kaggle_api_extended import KaggleApi
import io, contextlib
a=KaggleApi(); a.authenticate(); rows=[]; tok=None
for _ in range(40):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        batch=a.competition_leaderboard_view('pokemon-tcg-ai-battle', page_size=100, page_token=tok)
    if not batch: break
    rows+=batch; tok=None
    for line in buf.getvalue().splitlines():
        if 'Next Page Token' in line: tok=line.split('=',1)[1].strip()
    if not tok: break
print('rows', len(rows))
for i,r in enumerate(rows,1):
    if 'Scio' in (r.team_name or ''): print('RANK', i, r.score, r.team_name)
"

# Skill measurement: near-mirror head-to-head (rule 5). The only kind that counts.
python -X utf8 scripts/arena.py play "rule:v10,noS" rule:lucario `
    --deck-a lucario_v10 --deck-b mega_lucario_ex --matches 500

# Against the real bar
python -X utf8 scripts/arena.py play bc "rule:v10,noS" `
    --deck-a grimmsnarl --deck-b lucario_v10 --matches 500

# A/B a rule override against the pure clone (how every targeting.py rule is judged).
# Off-switches: noChip, noSpread, noSrc, noWall. Opt-in (default off): drag, dragHi, boss, veto.
# Isolate ONE rule per run: the P4a pair measured 0.452 while each alone was null.
# NOTE the first token after `bc:` is a LABEL, not a flag (§7). Write `bc:<label>,<flag>`.
python -X utf8 scripts/arena.py play "bc:s,noSrc" bc `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 --archive out/arena/ab_x.jsonl

# Net A/B, two nets in one process (~5 min, n=2000)
python -X utf8 scripts/arena.py play "bc:new,net=out/policy_X.npz" bc `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 --archive out/arena/ab_X.jsonl

powershell -File scripts/deck_sweep.ps1        # all decks vs the anchor; no args (§7)
python -X utf8 scripts/tally.py "<agent>" "out/arena/foo_*.jsonl"

# Audits -- run these BEFORE writing any rule
python -X utf8 scripts/opportunity_audit.py --matches 100        # our games
python -X utf8 scripts/opportunity_audit.py --corpus artifacts/pds_v2   # demonstrators
python -X utf8 scripts/context_accuracy.py                       # per-context top-1
# ⚡ --equiv: count a hit when the argmax option is BITWISE IDENTICAL to the
# chosen one. Those are two copies of ONE card in one role (two Trainers in the
# deck, two energies in hand onto the same target) -- picking either produces the
# same game, so plain top-1 charges the net for a coin flip. 30.2% -> 29.0%
# corpus-wide, TO_HAND 61.2% -> 67.1%. Use it for any agreement claim. EVIDENCE 8x.
python -X utf8 scripts/context_accuracy.py --net out/policy_b1_v3.npz `
    --ds artifacts/pds_v3r --equiv

# ── DAY 12: is the residual the ENCODING? Two probes, neither needs a net ──
# The CEILING. Bitwise-identical options get identical logits from ANY net, so
# sum(1/g)/N bounds top-1 for this layout. It is 95.6% and the clone gets 69.8%,
# i.e. un-expressibility explains at most 4.4 of the 30.2 points. --opt-cols 25
# reruns it against the v2 layout (the 8f control). EVIDENCE 8x.
python -X utf8 scripts/p17_encoding_ceiling.py --ds artifacts/pds_v3r

# THE FEATURE AUDIT, BY ENUMERATION. Diffs the observation against what
# featurize() actually reads, then SIZES each dropped field (rule 14: an absent
# input that is constant where the decisions happen explains nothing).
# ⛔ Use this instead of the remembered candidate list -- 3 of the 4 items that
# list carried for two days were already encoded. EVIDENCE 8y.
python -X utf8 scripts/p18_missing_state_audit.py --games 300

# The v4 state block: rebuild the corpus (byte-identical to pds_v3r plus
# xdense/xslots), then treatment and control on the IDENTICAL rows.
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds_v4/d26 `
    --ratings out/lb/pokemon-tcg-ai-battle.zip replays/2026-07-26   # ...d27-d29
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 --bs 1024 `
    --loss listwise --state-h 512,256 --head-h 256,128 --out out/policy_v4.npz
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 --bs 1024 `
    --loss listwise --state-h 512,256 --head-h 256,128 `
    --no-extra --out out/policy_v4ctrl.npz        # control: the v3 state vector

# ⚡ THE NOISE FLOOR, and every net-vs-net number in this repo needed it.
# Two IDENTICAL-recipe nets differing only in --seed measure 0.482 [0.460, 0.504]
# against each other -- a null, i.e. run-to-run variance is ~±13 Elo. Any A/B
# claiming less than that is claiming nothing. EVIDENCE 8z.
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 --bs 1024 `
    --loss listwise --state-h 512,256 --head-h 256,128 --seed 1 `
    --no-extra --out out/policy_v4ctrl_s1.npz
python -X utf8 scripts/p6_recon.py --matches 120   # EVERY select, bucketed -- the menu
python -X utf8 scripts/p5_audit.py --matches 200   # sizes the three P5 findings
python -X utf8 scripts/p5a_replays.py              # the same counters on 55 REAL games

# What the SHIPPED v3 agent did against the real field (the arena's reality check).
# Reports the archetype mix, the Boss's Orders drag audit and the Froslass timing
# audit, all with honest denominators. EVIDENCE 8g.
# NOTE its archetype table uses a 4-card hardcoded classifier and buckets 63% as
# "other" -- use p9 below for the field, and p8 only for the two audits.
python -X utf8 scripts/p8_optv3_replays.py --dir replays/submission_optv3

# ⚡ WHAT THE FIELD ACTUALLY IS. The ONLY honest source -- our own games. Mining
# public episodes CANNOT answer this (they stop at avg_score 1055; we play at
# 825-952). Names every archetype by evolution LINE, ignores 1-of techs, and
# reconstructs each deck's card list. Pass both dumps to pool them. EVIDENCE 8i.
# ⚠ RE-RUN THIS AFTER EVERY SUBMISSION REPLAY DUMP -- the mix moves.
python -X utf8 scripts/p9_field_census.py `
    --dir replays/submission_optv3 replays/submission_replay_2026-07-29

# The two anchors the census said we were missing (idempotent; from notebooks/).
# rule:alakazam5 = the field's #1 deck (22.0%), a 5th-place pure-rules pilot.
# rule:archaludon = our worst matchup (45.5% WR over 11 real games).
python -X utf8 scripts/import_field_agents.py
python -X utf8 scripts/arena.py play bc rule:alakazam5 `
    --deck-a grimmsnarl --deck-b alakazam5 --matches 1000
python -X utf8 scripts/arena.py play bc rule:archaludon `
    --deck-a grimmsnarl --deck-b archaludon_ex --matches 1000

# Can a preserved bundle still be restored? (run from inside an extracted tarball)
python -X utf8 scripts/restore_smoke.py
python -X utf8 scripts/p5b_check.py --matches 150  # does a rule actually fire? (rule 9)

# Mine the TOP of the ladder. On disk: 07-26..07-30.
# ⚠ The CURRENT day 403s -- episodes publish the following day, so mine yesterday.
# 🔴 THIS IS NOT OUR FIELD. These datasets contain nothing below avg_score 1055
# and we play at 825-952. Use it for decklist consensus and report figures about
# the top of the board -- NEVER to decide which anchors to keep (EVIDENCE 8i).
python -X utf8 scripts/fetch_top_episodes.py --date 2026-07-30 --max 400
python -X utf8 scripts/mine_meta.py replays/2026-07-29    # takes dirs as arguments
powershell -File scripts/fetch_days.ps1        # several days; edit $Dates default (§7)

# Crustle: the counter-meta anchor (import once; idempotent)
python -X utf8 scripts/import_crustle_agent.py
python -X utf8 scripts/arena.py play bc rule:crustle `
    --deck-a grimmsnarl --deck-b crustle_v1 --matches 1000

# Is damage even landing? (the wall/counter census -- and the log-reading template)
python -X utf8 scripts/p2_lethal.py --matches 200          # lethal audit (closed)
python -X utf8 scripts/p3_crustle_probe.py --matches 60    # attack vs counter damage

# SIZE a rule before building it (rule 14). p7 is the per-TURN template -- resolve
# a decision once per turn, not once per select, or multiplicity inflates it.
python -X utf8 scripts/p7_morgrem.py --matches 200         # the Morgrem out (closed)

# Train (12 epochs; artifacts/pds_v2 is the shipped corpus)
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v2 --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 --out out/policy_X.npz

# ROADMAP B1: the feature A/B. artifacts/pds_v3 = 1,603 games at 37 opt-cols,
# rebuilt from the 4 raw replay days on disk. The CONTROL is the SAME rows
# truncated to the v2 layout (--opt-cols 25) -- so features are the only
# difference. `--opt-cols` exists for exactly this and nothing else.
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v3 --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 `
    --opt-cols 25 --out out/policy_b1_ctrl.npz        # control (v2 features)
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v3 --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 `
    --out out/policy_b1_v3.npz                        # treatment (v3 features)
python -X utf8 scripts/arena.py play "bc:v3,net=out/policy_b1_v3.npz" `
    "bc:ctrl,net=out/policy_b1_ctrl.npz" `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 `
    --archive out/arena/b1_v3_vs_ctrl.jsonl

# Rebuild shards from raw replays (more data is NOT a lever -- EVIDENCE §1)
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds/d30 replays/2026-07-30

# ── ROADMAP B7 / day 11: WHO is demonstrating, and does it matter? (§8r-§8u) ──
# The full 6,024-row leaderboard in ONE call -- this is what makes any of it work.
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_leaderboard_download('pokemon-tcg-ai-battle', path='out/lb')"

# Tag every row with the demonstrator's LB score. ⚠ ALWAYS read the coverage line
# it prints: the first run silently lost 24.6% of d26, and 92% of that was ONE
# team under three names (display name, member username, post-merge name). The
# fix is exact member matching (automatic) + replays/team_aliases.tsv (by hand).
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds_v3r/d26 `
    --ratings out/lb/pokemon-tcg-ai-battle.zip replays/2026-07-26   # ...d27-d29

# Agreement vs demonstrator rating. ⚠ DEFAULTS to the trainer's held-out split
# on purpose -- scoring all rows of a corpus the net trained on manufactures the
# very correlation being tested. --seen-from gives a real zero-exposure bucket.
python -X utf8 scripts/p15_rating_curve.py --net out/policy_b1_v3.npz `
    --ds artifacts/pds_v3r

# A REPRODUCIBLE control population: census any third-party dump from its
# owner's seat and emit the opponents on our archetype. ⚠ --exclude Scio, or the
# control contains our own agent and scores ~98% against itself.
python -X utf8 scripts/p9_field_census.py --dir replays/sixth_sense_31-07-2026 `
    --us "Sixth Sense" --us "Raja Biswas" --emit-players out/ctrl_players.txt

# Covariate shift: compare the two POLICIES to each other, not to human labels.
# Symmetric disagreement = a real policy difference; collapse on our own states
# = it was shift. artifacts/pds_ours doubles as the 1.7% positive control.
python -X utf8 scripts/p16_policy_disagree.py --a out/policy_b1_v3.npz `
    --b out/policy_b7_ntum.npz --ds artifacts/pds_ours artifacts/pds_ntum_r

# The two B7 nets, both KILLED (-55 and -92 Elo). Kept as reproducers only.
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v3r --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 `
    --rating-temp 25 --out out/policy_b7_rw25.npz     # ESS 41% of rows
python -X utf8 scripts/train_policy.py --ds artifacts/pds_ntum_r --epochs 30 `
    --lr 2e-4 --loss listwise --state-h 512,256 --head-h 256,128 `
    --init out/policy_b1_v3.npz --out out/policy_b7_ntum.npz

# Build + submit (smoke-tests the bundle the way Kaggle loads it)
python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc --nets policy

# ... with a CANDIDATE net + its rule flags pinned as a PAIR (the v3 config).
# --policy-net runs the dim guard at build time: a net this code cannot feed
# would otherwise ship happily and play random-legal on Kaggle. --no-rules is
# REQUIRED with a v3 net (the three rules measure 0.427 against it, EVIDENCE 8f).
python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc `
    --nets policy --policy-net out/policy_b1_v3.npz --no-rules
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_submit('dist/submission.tar.gz','msg','pokemon-tcg-ai-battle')"

# Import public notebook agents
python -X utf8 scripts/import_v10_agent.py     # rule:v10 + decks/lucario_v10
python -X utf8 scripts/import_rule_agents.py   # the four sample agents

# Find new public notebooks (this is how V10 was found -- redo periodically)
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(k.ref,'|',k.title) for k in a.kernels_list(competition='pokemon-tcg-ai-battle',sort_by='voteCount',page_size=30)]"
```

### Data on disk

- **`replays/`**: `2026-07-26` and `2026-07-27` (400 each) — the last pre-shift
  days. **Nothing newer than 07-27; §3.1 needs 07-28/29/30.** Older days were
  pruned 2026-07-30 (they were 15 GB, the corpus is already compiled, and more
  training data is measured dead). Their **`manifest.csv` files are kept in
  `replays/manifests/<date>/`** (episode ids + avg_score, re-fetchable from
  those), and the meta they encoded is archived in
  `out/meta/pre_shift_0722_0724.txt`.
- **`replays/submission_replay_2026-07-29/`** — 55 games, 54 distinct LB
  opponents, team `Scio` is us. Use for diagnosis, not training.
  ⚠ **Despite the date these are `55054446`'s games — the chip-only agent, before
  `energy_spread`.** Anything depending on *two armed* Munkidori is understated
  there. **Always check which agent produced a replay dump.**

  🔴 **These two dumps are the ONLY data in existence about the field we play.**
  Kaggle's public episode datasets are censored below `avg_score` 1055 and we sit
  at 825–952, so `replays/2026-07-*` cannot substitute (`EVIDENCE` §8i).
  **`replays/submission_*` must never be pruned**, and every future submission
  should have its replays pulled and fed to `p9_field_census.py`. ⚠ Each dump is
  ~50 games from **one agent at one rating**, so the mix moves between them
  (Lucario 20% vs 5%, Alakazam 13% vs 31%) — pool them, and treat any single
  archetype share as ±8 pp.
- **`replays/submission_optv3/`** — 56 files, **54 usable** (2 are bare
  step-arrays, not replays — skip anything where the JSON root is a list).
  **These are `55116557`'s games: the optfeat-v3 agent with every rule OFF.**
  The single most valuable diagnostic asset in the repo right now, because it is
  the only record of what our agent does against the **real field** rather than
  against our two anchors. Analysed by `scripts/p8_optv3_replays.py`
  (`out/logs/p8_optv3_replays.txt`); findings in `EVIDENCE` §8g.
  **Archetype mix — the number that invalidates the arena: 63% "other", Crustle
  24%, mirror 9%.**
- **`artifacts/**` is gitignored.** `artifacts/pds/` = 4,010 games (the *rejected*
  lw3 corpus); **`artifacts/pds_v2/` = 2,810 (the shipped corpus)** and exists
  only on this disk — `pds` minus the three days that made lw3 worse:

  ```powershell
  foreach ($d in @('old','d21','d22','d23','d24','d25','d26','d27')) {
    New-Item -ItemType Directory -Force "artifacts/pds_v2/$d" | Out-Null
    Copy-Item "artifacts/pds/$d/shard_*.npz" "artifacts/pds_v2/$d/" -Force
  }
  ```

- **`out/arena/*.jsonl`** — 51 archived A/B runs; **the primary receipts for every
  number in `EVIDENCE.md`. Do not delete.** `out/logs/RECEIPTS.txt` is the index:
  every `score=… [CI] W/D/L over N games` line from every run, in one file.
- **`out/policy_lw2.npz`** = the shipped net; `lw3` / `policy_win` are kept as the
  negative-result receipts. Other checkpoints were pruned.
- `decks/crustle.py` was reconstructed from a `crustle-replays/` dir that is
  **not in the repo**.
- Old repo `E:\Kaggle\pokemon-tcg-simulation` = failed pure-RL attempts; it also
  holds 366 replays at `replay_miner\replays\2026-07-06..12`. **Take its replays,
  not its approach.**

---

## 6. Settled — do not redo

Full numbers, mechanisms and interpretations: **`report/EVIDENCE.md`**. The
one-line verdicts:

- **The clone is plateaued — three training axes negative.** More data (0.491),
  winners-only (**0.375**), and higher val accuracy all fail. `EVIDENCE` §1.
- **Search is out, ours and the field's.** Ours scored 0.323 and was selecting
  rollout noise (SE≈0.14); **V10's MCTS has never executed** (two bugs, confirmed
  by timing). ~~Self-play RL dropped on the same evidence.~~ 🔴 **RETRACTED day
  14 — self-play RL was NEVER RUN.** It was a compute prior inherited from the
  search result and filed beside the measured negatives in four files for twelve
  days; there is no RL code in this repo or the old one. **Status is "never
  attempted", not "dead"** — the live objection is credit-assignment variance,
  and it must be SIZED before anything is built. `EVIDENCE` §2.
- **Boss's Orders — all four interventions null, the card is closed. Do not write
  a fifth.** `EVIDENCE` §6.
- **The Morgrem out is closed by SIZING, not by an A/B** — ~0.2 firings/game, the
  free route already 95.4% right, and a tradeoff besides. It also corrected "our
  attacker deals 0 into theirs": true of their **Active only**. `EVIDENCE` §8e.
- **Closed cheaply and correctly:** P5c never-end-without-attacking (3,683/3,683),
  `REMOVE_DAMAGE_COUNTER_COUNT` (100% already), post-KO promotion (9 misses/120
  games), `TO_HAND` duplicate-avoidance (parity), the decklist variant (0.490),
  P5a pooled Adrena-Brain (~0.5 real decisions per 200 games). `EVIDENCE` §8.
- **Do not resurrect:** the `rule:iono` arena→LB ladder; the old deck sweep's
  ranking; "the clone is comfortably above the rule baseline"; every n=24 number
  and every strength claim dated before 2026-07-27 pm; "3× compute made it
  worse". `EVIDENCE` §10.

⚠ **Everything above was measured against ONE opponent** — `rule:v10` on
`lucario_v10`. **That is far better news than day 8 thought.** Day 8 read it as
"measured against a dead deck" and discounted it; day 9 measured the actual field
and `lucario_v10` is **12.8% of it**, tied for the largest deck we face
(`EVIDENCE` §8i). So these results are *narrow*, not *stale* — they are one
genuine slice of the field, and the missing slices are the other four anchors,
not a replacement for this one.

The negatives are probably safe (a rule that does nothing against a real opponent
rarely becomes a winner against another). The **positives** still need the other
four anchors before they are treated as general.

⚠ **Open loose end:** the P2b "already at demonstrator parity" verdicts were only
re-derived for `munkidori_adrena_brain` after the P4c multiplicity fix; the
demonstrator side of the `opps` column has never been run
(`--corpus artifacts/pds_v2`). `EVIDENCE` §8.

---

## 7. Gotchas (all paid for)

- 🔴 **`cmd | tee log | grep ...` REPORTS THE EXIT CODE OF `grep`, NOT OF `cmd`.**
  The day-11 capacity run **crashed with a CPU OOM at epoch 8 of 12** and the
  harness reported **"completed (exit code 0)"**, because the last stage of the
  pipeline succeeded. The filtered view showed eight tidy epochs and no error;
  only reading the *unfiltered* log revealed the traceback. **A truncated run
  looks exactly like a finished one when you only read the grep.** Redirect
  (`> log 2>&1`) and echo `$?` when the exit status matters, and never conclude
  from a filtered log — this is rule 9 ("a metric that never prints is not a
  metric that passed") applied to the runner instead of the metric.
- ⚠ **MEMORY, not CPU, is this machine's binding constraint on model size**, and
  it bit twice in one hour. A 1.5M-param net (`--state-h 1024,512`) at
  `--bs 1024` OOMs mid-training on the 249k-row corpus; at `--bs 512` it OOMs
  during *data loading* — 231 MiB for one `(1633243, 37)` array — **whenever an
  arena process is running alongside it.** The `Data` class holds every shard in
  RAM plus per-row bag lists, so the load is a hard spike before a single epoch
  starts. **Do not run a large train and an arena concurrently** (rule 7 said
  2–3 jobs for CPU reasons; for the big nets it is 1), and **size this before
  planning any RL run** — a policy+value pair plus a replay buffer has to fit in
  what is left.
- **`__file__` DOES NOT EXIST on Kaggle.** `kaggle_environments/agent.py` does
  `exec(code_object, env)` → `NameError` → ERROR before the agent runs. This
  killed `55028078`. The smoke test `exec`s the source with no `__file__` in
  globals, exactly as Kaggle does — **keep it that way**.
- **Kaggle sets no env vars.** `SA_NO_PNET`/`SA_NO_VNET`/`SA_PNET_PATH` are inert
  there, so any bundled `.npz` is LIVE. Pin with `--nets none|policy|value|both`.
- **Do not set `SA_COUNT_MODE=expect` with a listwise net** — it assumes
  calibrated probabilities; listwise gives a valid *ranking* only.
- **Kaggle enforces the 600 s pool** (exhausted = loss) though the harness does
  not. `arena.py` records `pool0`/`pool1` and warns below 300 s. BC uses 0.1 s.
  ⚠ **If the machine sleeps mid-run, one game eats the whole nap** and `arena.py`
  prints `WOULD TIME OUT ON KAGGLE` off that single game. Check the distribution
  first: in `ab_spread.jsonl` the worst pool was −3606.9 s and the *next* worst
  was 599.2 s, median 599.9 s, p99 latency 1.6 ms.
- **⛔ "Latest 2 active" is a TRAP, not a footnote.** Submitting a third agent
  silently **evicts your best one from active play** — it stops playing episodes
  and its score freezes. **Before every submission, list the active pair and name
  which one you are willing to lose.** A rollback pays this cost too.
- **A young submission reads low and it means nothing** (μ=600 start; `55072063`
  took ~4+ h to reach 958). Never compare a fresh submission against a mature
  one, and never against a remembered number — read both in the same call.
- **Submission:** `.tar.gz`, `main.py` + `deck.csv` at TOP level (+ `cg/`, `sa/`).
  Cap 197.7 MiB. 5/day, **latest 2 active**. New submissions start μ=600. The
  validation episode is self-play first — a crash there means Error.
  `kaggle competitions submit` may 400 despite a 100% upload; the Python client
  works, and that call **submits** — it is not a dry run.
- **Submission discipline:** submit only what has won head-to-head at n≥500
  against the current anchors. Always `--nets`-pin. Rebuild rather than trusting
  an old tarball in `dist/`.
- ⚠ **A REJECTED NET DOES NOT CRASH — IT PLAYS RANDOM-LEGAL.** `policynet.load`
  returns `None` on a feature-dim mismatch and `PolicyAgent` falls back to
  `list(range(minCount))`, so a mis-built bundle smoke-tests "fine", uploads
  fine, and quietly scores ~600. Since 2026-07-31 `--policy-net` runs the dim
  guard at build time and the smoke asserts `NET_OK`. **Never ship a bundle whose
  build log lacks that line.**
- ⚠ **`dist/submission.tar.gz` is whatever was built LAST.** As of 2026-07-31 it
  is the **v3 + rules-off** candidate, not the live `lw2` agent. Check the
  timestamped filename before uploading.
- Kaggle Python API returns **snake_case** (`public_score`, `team_name`);
  `competition_leaderboard_view` paginates at 20 rows.
- **`obs["logs"]` is a per-observation DELTA, not a cumulative game log.**
  Observed lengths across our own selects: `[0, 0, 48, 14, 3, 1, ...]` —
  non-monotonic. **Never index into it as if it held the whole game**; concatenate
  deltas, or (better) tally events without needing offsets. This produced a probe
  that read 0.0 damage in every bucket including ones that cannot be zero
  (`EVIDENCE` §8d). Useful entry types: **`type 15`** = an attack
  (`cardId`, `attackId`, `playerIndex`); **`type 16`** = an HP change
  (`playerIndex` = the owner of the changed Pokemon, `cardId`, `value` negative
  for damage / positive for healing, and **`putDamageCounter`** True for
  placed/moved counters vs False for attack damage). ⚠ **A PREVENTED attack logs
  as `value: 0`**, so a filter of `value < 0` silently drops exactly the events
  that prove a prevention ability exists.
- **Third-party replay dumps (`replays/<team>_<date>/`) are a different animal
  from mined episodes, and three things about them bite:**
  - ⚠ **`info.TeamNames` is the display name AT EPISODE TIME and teams rename.**
    The Sixth Sense dump reports "Raja Biswas" on 113 games and "Sixth Sense" on
    30 — **one team, teamId 16452116**. A census keyed on the name splits one
    demonstrator in two. **Join on `teamId` from `episodes_meta.json`.**
  - ⚠ **A dump spans several of that team's SUBMISSIONS**, i.e. several different
    agents. `episodes_meta.json` carries `submissionId` per seat — use it, and
    tag rows so the weaker agent can be ablated out.
  - **The sidecar is not an episode.** `build_policy_dataset.py` now skips any
    file whose stem is not all digits; before that, `episodes_meta.json` was
    parsed as a replay and counted as `errors=1`.
- ⚠ **A player filter that matches nothing used to build a corpus of EVERYTHING.**
  An empty `--players-file`, or a CJK name off by one homoglyph
  (李秉**叡** vs 李秉**睿**), silently produced an unfiltered corpus under an
  expert corpus's name — the `bc:` label trap in a new place. **Both cases now
  `SystemExit`.** Take exact team names from `episodes_meta.json`, never retype
  them.
- ⚠ **`context_accuracy.py` scores the `gid % 20` val split by default.** On a
  corpus the net never trained on that silently measures **5% of your data**.
  Pass **`--all-rows`** for any external/expert corpus.
- **The first token after `bc:` is a LABEL, not a flag.** `bc:veto` silently
  builds a plain `bc` named "veto" — the flag parser starts at token 1, so the
  A/B compares the clone against itself. Write `bc:<label>,<flag>`
  (`bc:p5b,veto`). `arena.py` now raises on an unrecognised flag, which is what
  caught it, but the label slot still swallows anything.
- **PowerShell `-File script.ps1 -Days a,b,c` does not bind an array.** Edit the
  script default and launch with no args. **Never name a param `$Matches`** —
  collides with the automatic regex variable.
- Windows: `python -X utf8` everywhere. Run from repo root; `sys.path` needs
  `src/`, `agents/`, root. Launch long jobs with `Start-Process` (detached) and
  pass `-u`, or python block-buffers redirected stdout.
- Some replays download truncated (exactly 3 MiB) and fail JSON parse; builders
  skip them (`errors=N`). Delete + re-fetch to recover.
- Commit style: fine-grained, one-line semantic messages + Claude co-author
  trailer.
