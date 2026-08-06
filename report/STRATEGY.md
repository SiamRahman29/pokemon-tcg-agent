# Clone the field, then audit the clone's blindness

**A behavior-cloned Pokémon TCG agent, and the measurement discipline that kept
it honest.**

Team `Scio` — Kaggle `pokemon-tcg-ai-battle`

> **STATUS: DRAFT, started 2026-07-31 (day 9 of 18).** Sections 1–5 and 8 are
> written from concluded experiments in `EVIDENCE.md`. Sections 6–7 are outlined
> against work still in flight. **Every number here must trace to an archived run
> in `out/arena/*.jsonl` with n and CI — if a claim below has no citation, it is
> a placeholder, not a finding.** Final due 2026-09-14.
>
> ⚠ **Do not write a verdict into this file until the experiment producing it has
> concluded.** On 2026-07-31 a conclusion was written into `EVIDENCE.md` §8i after
> 2 of 5 anchors had reported and had to be retracted within the same session
> (§5.4). That failure is *in* the report — but only once, and deliberately.

---

## 1. Approach and rationale

**The constraint that determined everything: 2 vCPU and a 600 s pool for a whole
game.** Under that budget we asked what the strongest available prior is, and
answered: *the field itself.*

We behavior-cloned **2,810 human games** mined from Kaggle's published top
episodes into a listwise policy net (~1 ms/move, using **0.1 s of the 600 s
pool**), then systematically enumerated the decisions the clone gets wrong and
repaired them with arithmetic rules.

**Why not search.** We built it and measured it: our search agent scored **0.323**
against the clone, and was selecting rollout noise (SE ≈ 0.14 on a terminal 0/1
signal). Then we found the decisive external evidence — **the public LB-950
baseline's MCTS has never once executed** (two bugs, confirmed by timing
instrumentation), and it holds its rating anyway. A search stack is not what wins
this board. `EVIDENCE` §2.

**Why not reinforcement learning — and this sentence has been corrected.** It
previously read "dropped on the same evidence, plus ~1.4 cores of real
throughput". That describes a *decision*, not an experiment: **self-play RL was
never run** — no code, no `n`, no interval, in this repository or the one it
inherited from. The claim had been filed beside the measured negatives in four
documents for twelve days. It is retracted in §8 and its post-mortem belongs
there and in §5; the honest status is **"declined on a compute prior, never
attempted"**. `EVIDENCE` §2's retraction box.

What *is* measured and must not be laundered into an RL verdict: **search**
(0.323, above) and **`--winners-only`** (0.375, §1's table) — the latter filters
*other people's* games by outcome and discards half the corpus, which is not the
same mechanism as a gradient signed on our own trajectories. When the objection
that remained — credit-assignment variance — was finally sized rather than
asserted, **it did not bind**: separating two policies of known ±37 Elo takes 800
games against a ~5.5M-game budget (`EVIDENCE` §8ae). A small-parameter fine-tune
of the clone on its own recorded outcomes was scheduled on that basis.

**Why not more data.** Three independent scaling axes all measured negative: a
4,010-game corpus lost to the 2,810-game one (**0.491**), winners-only training
scored **0.375**, and the net with the *best* validation accuracy lost its
head-to-head. `EVIDENCE` §1. **This is the result we would most like other
competitors to check**, because it inverts the usual instinct.

---

## 2. The hypothesis log

The full log is `report/EVIDENCE.md` — every concluded experiment with its
hypothesis, command, n, confidence interval and verdict. It is maintained as the
primary record; this report cites it rather than restating it.

**Summary of what has been decided, and how:**

| class | experiments | outcome |
|---|---|---|
| training scale/objective | 5 | all negative (§1) |
| demonstrator selection | 2 (B7) | both negative, −55 and −92 Elo (§7b.3) |
| search | 2 | both negative (§2) |
| self-play RL | **0 — never run** | not a result; a decision misfiled as one (§1, §8) |
| targeting rules | 7 A/B'd at n≥2000 | 3 win, 4 null (§3) |
| feature representation | 1 (B1) | large win (§8f) |
| matchup branches | 1 shipped | +0.104 recovered (§8c) |
| sized and closed without an A/B | 6 | (§8, §8e) |

---

## 3. The discriminator: dominated options vs tradeoffs

**The central claim of this report, and it is falsifiable.**

> **A hand-written rule beats a behavior clone when it deletes a *provably
> dominated* option, and fails when it picks a side in a *tradeoff*.**

Record: **3 for 3** on dominated-option rules, **0 for 4** on tradeoff rules, each
judged by arena A/B at n≥2000. `EVIDENCE` §3.

**Why it predicts.** The clone watched 2,810 games of humans making tradeoffs and
is roughly as good at them as our arithmetic. What it cannot do is *bind* an
option to the board state that option refers to — see §4.

⚠ **The discriminator has a sharp edge that cost us a session.** "Dominated" is
easy to talk yourself into. `counter_source` was filed as dominated because the
heavily-damaged source is better *both* on damage transferred and on healing. The
first is arithmetic; the second is a **judgment** (a heal only pays if the Pokémon
is savable) that was asserted, not measured. **A rule is only dominated if every
dimension it moves is arithmetic — one judgment puts it in the tradeoff column.**

---

## 4. The misdiagnosis, and what it taught us about blind spots

**For eight days this project's founding premise was false, and it was repeated in
three files without once being checked.**

The premise: *"the net cannot see HP, so we must supply the arithmetic."* It was
written in `targeting.py`, restated in the handoff, and used to justify every
rule. **It was wrong** — `features.py` had been feeding the net per-slot HP,
damage, energy and prize value since v1.

The real gap was one line's worth: the per-option vector encoded position only as
*area* flags and **never encoded the option's index**, so two options naming two
copies of the same card were **bitwise identical inputs with different right
answers**. Encoding it (v2 → v3, 25 → 37 per-option columns) measured **0.878**
against a same-corpus control and **inverted the method**: with the binding
supplied, the hand rules became *harmful*. `EVIDENCE` §8f.

**The generalisable lesson, and we think it is the most transferable thing here:**

> Ask whether a blind spot is **informational** (the input is absent) or
> **representational** (the input is present but cannot be bound to the decision).
> They look identical from outside — the agent is wrong at chance either way — and
> they have opposite cures. Four hand rules cured the symptom; twelve features
> cured the cause and dominated them.

### 4b. How much blindness is left? A computable ceiling

Having been wrong once about what the features can express, we stopped asserting
it. **The bound is computable exactly, with no model involved.** Two options
whose per-option encoding is *bitwise identical* receive identical scores from
any net reading that encoding — the state vector is shared across a row's
options, so it cannot break the tie. If the demonstrator's choice sits in a tie
group of size `g`, no such net beats `1/g` on that row, and `Σ(1/g)/N` is a hard
upper bound on achievable agreement.

Over 235,654 single-choice decisions the bound is **95.6%**. Our clone reaches
**69.8%**.

| context | rows | chosen option is tied | ceiling | clone |
|---|---|---|---|---|
| MAIN | 127,683 | 5.0% | 97.4% | 62.7% |
| TO_HAND | 31,901 | **32.4%** | **81.0%** | 61.2% |
| ATTACH_TO | 2,395 | 45.3% | 74.1% | 71.1% |
| 12 other contexts | 60,000+ | **0.0%** | 100.0% | 77–100% |
| **all** | **235,654** | **7.8%** | **95.6%** | **69.8%** |

**Two things follow, and they point in opposite directions.**

**First, the residual is not un-expressibility.** At most 4.4 of the 30.2
missing points can be blamed on options the encoding cannot tell apart. The §8f
defect is real and we fixed it; what is left is not more of the same.

**Second, the ties that remain are not errors at all — and finding that changed
how we measure.** A tie requires the same card id, so every tie group is *two
copies of one card in one role*: two identical Trainers sitting at different
positions in the deck, two identical energies in hand attaching to the same
Pokemon. **Picking either produces an identical game.** Plain top-1 agreement
charges the model for a coin flip between interchangeable cards. Scoring a hit
whenever the model's pick is bitwise identical to the human's puts corpus
agreement at **71.0% rather than 69.8%**, and TO_HAND at **67.1% rather than
61.2%**.

That is a small correction and we report it because it is the *kind* of thing
that silently inflates a disagreement metric — and because every conclusion in
§7b is built on one.

### 4c. The feature audit, done by enumeration rather than by memory

§4b says the residual is not un-expressibility, so it must be *absence*: state
the encoder never reads. For two days our plan named four such candidates —
opponent hand size, prizes remaining, turn number, the stadium — and all four
had been copied forward from session to session without anyone opening
`features.py`.

**Three of the four were already encoded** (`features.py`, lines 88–99). We had
made §4's mistake a second time, in the same project, with the same mechanism:
a premise repeated in three files is not thereby verified, it is just
load-bearing.

So we stopped recalling the list and derived it. `p18_missing_state_audit.py`
walks 300 real games, and at every decision point diffs **every key of the
observation** against the set of keys `featurize()` actually reads. Each
survivor is then *sized* before anything is built — how many distinct values
does it take at a real decision, and what is its modal share? An input that is
constant where the decisions happen cannot explain a single miss:

| candidate | distinct values at a decision | modal share | verdict |
|---|---|---|---|
| `turnActionCount` | 20 | 17% in MAIN | **build** |
| the select's `effect` card | 31 | 26% in TO_HAND | **build** |
| the stadium in play | 7 | 61% | **build** |
| `retreated` / `stadiumPlayed` | 2 | 57% in SWITCH | **build** |
| `remainDamageCounter` | 1 | **100%** | dead on sizing |
| `remainEnergyCost` | 2 | **99.1%** | dead on sizing |
| hand counts, prizes, turn | — | — | **already encoded** |

Two candidates died for the cost of one probe rather than one training run, and
the retraction happened before a line of model code was written. **We regard the
method as more transferable than the block it produced:** diff the observation
against what the encoder reads, then size each survivor, and never trust a
candidate list you can recite from memory.

**And we can now show the method, not just the block, is what worked.** The
three derived fields shipped alongside five cheap extras that were *not* put
through the sizing step — `retreated`, `stadiumPlayed`, tool counts, bench cap,
pool size. Ablating members (zeroing their columns, so architecture, parameter
count, seed and rows are all identical) separates them:

| arm, n=2,000 | vs the full block | reading |
|---|---|---|
| drop `turnActionCount` alone | 0.527 | within noise |
| drop the stadium alone | 0.526 | within noise |
| drop the effect card alone | 0.483 | within noise |
| **drop all three derived fields** | **0.449** [0.427, 0.470] | **−36 Elo, disjoint** |
| *the same arm vs no block at all* | *0.469* [0.447, 0.490] | *−22 Elo, disjoint* |

**The three derived fields are mutually redundant and jointly necessary** — any
one can stand in for the others, and losing all three loses the entire +37. The
five unsized extras are **worse than nothing**: a net given only them plays 22
Elo below a net given no block at all. Unhelpful state columns are not free;
they are somewhere to overfit.

> The sizing step killed two candidates for being constant where decisions
> happen, and we nearly dismissed it as pedantry. The five features that skipped
> it are precisely the ones that measure negative. **Derive, size, and do not
> bundle.**

### 4d. The result: +37 Elo of play, and eight decisions of agreement

The survivors became one appended block — `turnActionCount`, the effect card,
the stadium, `retreated`/`stadiumPlayed`, tool counts, bench cap, pool size.
The corpus is byte-identical to its predecessor on every pre-existing array, so
the control (`--no-extra`) trains the old state vector on **the same 248,985
rows with the same recipe and the same seed**. Features are the only difference.

| grimmsnarl mirror, n=2,000 | score | reading |
|---|---|---|
| **v4 vs its own control**, seed 0 | **0.567** [0.545, 0.588] | +47 Elo |
| **v4 vs its own control**, seed 1 | **0.539** [0.518, 0.561] | +27 Elo — replicates |
| control seed 0 vs control seed 1 — **seed only** | 0.482 [0.460, 0.504] | ✅ null |

The third row is the one we would ask a reviewer to look at first. **Every
net-vs-net comparison in this project, for twelve days, compared two
independently trained nets and silently assumed run-to-run variance was zero.**
It is not zero, it is about ±13 Elo, and we only know that because we measured
it. Pooled over n=4,000 the block is worth **≈ +37 Elo**, and it is better on
**five anchors of five** covering 71.5% of the field we actually face —
including Mega Lucario, previously our only losing matchup, which moves 0.505 →
**0.549** with disjoint intervals. A generic feature repair fixed the matchup
that two sessions of targeted rule-writing could not.

**And here is the part we did not expect.** Held-out top-1 agreement with the
demonstrators — the metric the whole of §7b is built on:

| net | misses of 12,939 held-out decisions |
|---|---|
| control | 3,756 |
| **v4** | **3,748** |

**Eight decisions out of 12,939, for 37 Elo of playing strength.** Rule 2 of our
codex — *validation metrics do not predict strength* — had been paid for five
times in the direction "better agreement, worse play". This is the first
instance of the converse, and it is the more informative one: an intervention
worth 37 Elo is **invisible** to the instrument. The per-context breakdown shows
the block did not raise agreement, it **moved** it — 35 misses *worse* in MAIN,
19 better in ATTACH_FROM, 14 better in TO_HAND. It agrees less with the human
mixture and plays better.

### 4e. The other direction, measured the next day

The obvious objection to §4d is that one dead-heat metric proves little — maybe
agreement is simply hard to move. So we tried to move it, on purpose.

The last defect of §4's class was this: **every option is scored independently
against one shared state vector, so the network has never seen the option
_set_.** It cannot tell whether it is choosing among three Trainers in hand or
forty cards in a deck search, and it cannot see how the option in front of it
compares to its alternatives. We added the cheapest possible deep-sets encoder —
an elementwise mean and max over the option encodings, plus two count scalars,
appended to the state vector, with the same byte-identical control.

It worked, as a *fit*. Held-out agreement went **71.0% → 72.7%**: 214 more
correct decisions out of 12,939, the largest agreement gain any intervention in
this project has produced, concentrated exactly where the mechanism predicted
(MAIN misses 2,630 → 2,454).

**It bought almost no playing strength.**

| grimmsnarl mirror | n | score | reading |
|---|---|---|---|
| pooled net vs the v4 net, seed 0 | 2,000 | 0.514 [0.492, 0.536] | null on its own |
| pooled net vs the v4 net, seed 1 | 2,000 | 0.527 [0.505, 0.549] | +19 Elo |
| **pooled — the honest number** | **4,000** | **0.521** [0.505, 0.536] | **+14 Elo** |
| pooled net vs the v4 *control* — *positive control* | 2,000 | 0.539 [0.517, 0.561] | +27 Elo |

The last row is why the third is readable: the instrument still resolves the v4
effect at full size on the same afternoon, so +14 is a real measurement of a
small thing rather than a broken harness. It is also **one noise-width** — the
seed-only null is ±13 Elo.

**The two experiments together are the result:**

| intervention | Δ agreement (of 12,939) | Δ Elo | Elo per decision |
|---|---|---|---|
| v4 state block | **+8 decisions** | **+37** | 4.6 |
| v5 pooled option set | **+214 decisions** | **+14** | **0.07** |

> **The exchange rate between fit and strength differs by a factor of 70 between
> two interventions run a day apart, on the same corpus, with the same recipe.**
> Taken with §7b, this closes a loop: agreement with a demonstrator measures
> **distance from the fitted mode**, and playing strength is a different
> quantity that the mode tracks at no reliable rate — an intervention worth 37
> Elo the metric cannot see, and one the metric loves that is worth a
> rounding error.
>
> We think this is the most portable finding in the report. Behaviour cloning is
> usually tuned on validation accuracy because it is the only cheap signal
> available. On this task that signal is not weakly predictive; it is
> **uninformative at any scale we can measure**, and we would not have believed
> that without watching it fail in both directions in one week.

---

### 4f. The opponent pool is a function of your own rating

Every result above is a weighted average over an anchor set, and the weights come
from a census of who we actually play. We measured that census once, at ~820
rating, and then spent five days quoting it while climbing to 955.

Pooled over 75 games from two live agents, the field looked **transformed** since
the first census: the mirror **13.8% → 33.3%** (Fisher **p=0.002**), Mega Lucario
**12.8% → 4.0%**, our win rate 63.0% → 70.7%. The obvious reading is a meta
shift, and it is wrong.

`scripts/p19_field_drift.py` runs the discriminator: bucket all 181 rated games
from all four dumps by **opponent rating** rather than by date.

| archetype | opp <800 | 800–900 | 900–1000 | 1000+ |
|---|---|---|---|---|
| **mirror** | 5.3% | 18.6% | **42.4%** | **71.4%** |
| Alakazam | 13.3% | 28.8% | 33.3% | 14.3% |
| Crustle | 16.0% | 5.1% | 9.1% | 7.1% |
| Mega Lucario | **17.3%** | 6.8% | **0.0%** | **0.0%** |
| Archaludon | 10.7% | 15.3% | **0.0%** | **0.0%** |

🔴 **Hold the band fixed and every era difference vanishes** — all Fisher
p ≥ 0.065 across four dumps, and the smallest points the *opposite* way from the
pooled table. What moved was us: mean opponent rating **799 → 867**, tracking our
own 820 → 955.

**The field did not change its decks. We changed our seat in it.**

The consequence is not cosmetic. Re-weighting with band-correct shares — the
measurements themselves untouched, only the weights — moves every verdict, and
**flips one**: our "the arithmetic rules are worth nothing" result went **+0.8 →
−18.1**, because it rested on a −51 mirror term at 13.8% cancelling a +47 Lucario
term at 12.8%, and the true weights are 33.3% and 4.0%. We had shipped the right
configuration for reasons that were wrong by a factor of 18.

⚡ **It also dissolved a contradiction we had filed as irreconcilable.** A census
of top-rated episodes said 52.1% of seats played our archetype; our own games
said the mirror was 13.8%. Both are true — they are two points on one monotone
curve, and we had been walking up it.

> **The general form, and it is why this sits in a methods chapter:** *any*
> statistic gathered from your own matches carries an invisible parameter — the
> skill you had when you gathered it. This is the sampling-frame trap, committed
> a second time, on data we generated ourselves after writing the rule that warns
> about it. The fix is not a better census; it is **reporting the band alongside
> every share**, the way one reports n.

---

### 4g. Where the feature axis ended: three real defects that were not levers

The audit method of §4c kept working right up to the point where it stopped
paying, and the way it stopped is the most transferable thing in this section.

Over three days we examined the four embedding tables that carry card identity
and found **three genuine defects**, each measured rather than asserted:

| defect | measurement |
|---|---|
| Most of every table ships untrained | 104/1300, 134/1300, 135/1300, **57/1600** rows ever received a gradient. 88,000 parameters ship; ~6,880 were trained. |
| Untrained rows are not inert | Their norms (3.908–3.953) are indistinguishable from trained rows' (3.970–4.068), so an unseen card arrives as a *confident arbitrary identity*, not as "unknown". |
| One row is overloaded | Row 0 means empty slot / out-of-range / no stadium / no effect at once, across **25.5% of all slot lookups**, with no `padding_idx`. |

All three are real. All three were repaired, and the repairs were verified to
fire exactly where predicted. **All three measured nothing.** Weighted over the
anchor set: **−0.0099** for the full repair and **−0.0047** for the isolated
padding fix — both inside the run-to-run noise of simply retraining with a
different seed.

**Why they could not have paid, in hindsight and now in evidence:**

1. **The optimiser had already routed around each one.** The net drove row 0's
   magnitude to the 11th-smallest of 1,300 rows entirely on its own — it had
   *learned* that row means "nothing", and pinning it to zero only formalised a
   conclusion it had already reached.
2. **The channel was not load-bearing where the defect fired.** An ablation
   measured that scrambling opponent identity costs **0.838 → 0.587** against an
   opponent whose cards are in vocabulary, and **−0.018 (nothing)** against the
   one whose cards are not. Against exactly the opponents the repair targets,
   the net had already stopped deciding on identity and was deciding on hit
   points, energy and damage. There was no signal to clean up.
3. **The blast radius was priced before the build and it was small.** A sizing
   gate showed the unknown-card repair could only fire against ~12% of the
   weighted field; every other opponent is ≥78% in vocabulary, making the change
   a no-op there by construction.

⇒ **The question that predicts a win is not "is this wrong?" but "is the
network's behaviour currently constrained by this being wrong?"** Gradient
descent compensates for a great deal. A defect the optimiser can route around is
a defect in the code, not a limit on the agent — and only the second kind is a
lever. We did not have this distinction when the section began; §4c–4f's wins
all happened to be of the second kind, which is why the method appeared general.

The returns curve makes the same point without any theory. Across five
generations of feature work: **+115 → +37 → +14 → 0 → 0 Elo.**

**We repaired them anyway, and would again.** Shipping 88,000 parameters of
which 92% are untrained noise is indefensible on its own terms whatever the
scoreboard says, and the repair paid a dividend the Elo column does not show:
it removed **11.5% of the entire network** for 0.0018 of held-out fit. Set
beside the opposite experiment — 8.2× the parameters for **−43 decisions** —
capacity is now bounded from both directions on the same network. Nothing in
this project was ever capacity-limited, and we can now say so from measurement
rather than from suspicion.

---

## 5. Measurement discipline, and six failures of our own process

The full codex is 20 rules, each paid for by invalidated work (`HANDOFF.md` §2).
The load-bearing ones:

1. **n=24 is noise.** ~2 pp effects need n≈2000.
2. **Validation metrics do not predict playing strength** — five times.
3. **Count opportunities, not turns** — a per-turn binary audit hid multiplicity
   and read 99.4% where the honest figure was 96.9%.
4. **Check the denominator is a real *choice*.** One audit read "the rule takes
   the best target 26/26" while 90 of its 95 rows offered only one option. The
   honest denominator was 5.
5. **Size before you build.** One candidate died for the price of a probe: it
   would have fired ~0.2×/game, the free version of the same play was already
   taken 95.4% of the time, and the effect was smaller than n=2000 can resolve.

### 5.1 Failure one: asking a ±75-point instrument to measure a 12-Elo effect

A rule measuring 0.534 in the arena is worth ≈ +12 Elo. Our leaderboard readings
swing **±50–100 while converging**. We nonetheless treated a low LB reading as
evidence against that rule and spent a session's prior on it. **The arena and the
LB were never in conflict; the question was below the instrument's resolution.**
`EVIDENCE` §7.

#### 5.1b And we later measured that instrument's resolution directly

§5.1 argued the point from the leaderboard's observed swing. On 2026-08-02 we
measured it instead, by accident and then on purpose. A submission made purely
to add logging turned out to be **decision-identical** to the agent it was
built from — `diff -rq` over the extracted bundles showed only the counters and
one `print` differing, with weights, deck and engine byte-identical. Two agents
that make the same move in the same state therefore played the ladder side by
side for a day, and **they read 63.2 points apart** (942.7 against 879.5, after
23.9 h and 15.8 h of play, both settled by the same test).

**That is a true difference of exactly zero, displayed as 63 points.** Set it
against every effect this project has produced:

| effect | magnitude |
|---|---|
| **two decision-identical agents** | **63.2 LB points** |
| our best LB-measured net gap | +40.5 |
| the v4 state block (§4d) | +37 Elo |
| dropping its three derived members | −36 Elo |
| the v5 option-set pool (§4e) | +14 Elo |

⇒ **The leaderboard cannot adjudicate any change we have made or are likely to
make.** This is not a complaint about Kaggle's rating system, which is doing the
ordinary thing a rating system does over a few hundred games against a moving
population of 6,136. It is an argument about which instrument a claim may rest
on: **a 2,000-game arena A/B against a byte-identical control, with a measured
seed-only noise floor, is not the weaker evidence here — it is the only
evidence.** Every number in this report that could have been sourced from either
is sourced from the arena, and the ladder appears only where the effect is large
enough to clear 63 points.

⚠ Two honest limits. The gap was still **closing** across four reads (81.7 →
76.2 → 69.0 → 63.2 over 5.4 h), so 63 is a lower bound observed at one moment
rather than a stable floor. And the last of those reads closed because the
*converged* agent moved **+4.2** — after our own two-readings-an-hour-apart rule
had certified it as settled. **A convergence test at hour scale does not license
a claim at day scale**, which cost us a headline number the day before.

### 5.2 Failure two: an anchor set that was not the field

Every routine number was measured against one opponent deck. When the metagame
appeared to shift we re-anchored — and **retired the original anchor on the
strength of a sample that could not contain our opponents.**

**The structural finding, which we believe is the most useful thing in this report
for anyone else competing on this board:**

> **Kaggle's daily episode datasets are censored.** The published episodes stop at
> `avg_score` **1055**; our agent plays at **825–952**. The 800–900 and 900–1000
> buckets contain **zero** episodes. **No amount of episode mining can describe
> the field you actually face** unless you are already at the top of the ladder.

Mining said our original anchor was **0% of the metagame**. Measured against our
own 109 real ladder games, that deck is **12.8% of the field and our worst
matchup** — we win 36.4% of those games against opponents rated **85 points below
us**. `EVIDENCE` §8i.

**The fix:** the field census (`scripts/p9_field_census.py`) reconstructs
archetypes from our *own* submission replays. Anchor coverage went from 39.4% to
**71.5%** of the field.

### 5.3 What a census gets wrong if you let it

Two methodology traps, both hit and both fixed, because they change the answer:

- **Naming a deck by its highest-prize Pokémon is wrong.** A single 2-prize card
  run as a draw tech split the field's largest archetype across four names.
  **A card the deck runs one copy of is a tech, not an identity.**
- **Naming a deck by whichever card you happened to see fragments one deck into
  three** — a short game may only reveal the middle evolution stage. Resolving
  every Pokémon to its evolution line first is required.

Together these moved the top archetype's share from 16.5% to **22.0%** and cut
28 apparent "archetypes" to 19.

### 5.4 Failure three: we published a conclusion before the experiment finished

On 2026-07-31, after 2 of 5 anchors reported, we wrote "the arena/ladder gap is
solved" into the evidence log. The full sweep contradicted it: weighted by field
share, the arena favours the agent we thought had been proven worse.

**And the headline effect we had been trying to explain was itself an artifact.**
The "130-point regression" compared a **live** score against a **frozen** one
earned two days earlier on a board 2,000 entrants smaller — a comparison our own
rule 2 forbids. The only same-time, both-active comparison is **−25 points**,
against an agent that had not converged, inside the instrument's own ±50–100.

**We report this because the correction is the finding.** Two days of work were
directed at explaining a contradiction that largely did not exist.

### 5.5 Failure four: the census we built to fix §5.2 had the same flaw

§5.2 ends on a fix — stop mining a censored public dataset, census your *own*
replays instead — and coverage duly went from 39.4% to 71.5% of the field. **We
then used those shares to weight every arena verdict for six days without once
asking what determined them.**

They were determined by our own rating.

The competition matches you against opponents near your own score. So the
opponent pool is **not a fixed population being sampled** — it is a function of
where you sit, and it moves when you move. Our census was taken at ~820. By day
15 we were at 955, and the same measurement on 75 fresh games looked like a
metagame upheaval:

| archetype | census at ~820 (n=54) | census at ~955 (n=75) | Fisher *p* |
|---|---|---|---|
| the **mirror** (our own archetype) | 13.8% | **33.3%** | **0.002** |
| Alakazam | 22.0% | 21.3% | 1.000 |
| Crustle | 12.8% | 6.7% | 0.222 |
| Mega Lucario | 12.8% | 4.0% | 0.067 |
| Archaludon | 10.1% | 8.0% | 0.797 |

**It is not an upheaval.** Bucketing all 181 rated games we have by *opponent
rating* rather than by date shows a clean monotone structure, and **holding the
band fixed, not one archetype differs significantly between the two eras** (all
*p* ≥ 0.065):

| archetype | opp <800 | 800–900 | 900–1000 | 1000+ |
|---|---|---|---|---|
| **mirror** | 5.3% | 18.6% | **42.4%** | **71.4%** |
| Mega Lucario | 17.3% | 6.8% | **0.0%** | **0.0%** |
| Archaludon | 10.7% | 15.3% | **0.0%** | **0.0%** |

The decks we had built our counter-meta program around are **0 for 47 above
opponent rating 900.** They were never the metagame; they were the metagame
*beneath us*, and we were climbing away from them while tuning against them.

**What it cost, quantified.** Re-weighting our four headline verdicts with
band-correct shares — the measurements untouched, only the weights changed —
moves one of them across zero:

| verdict | as published | re-weighted |
|---|---|---|
| feature-block generations | +35.6 / +23.4 / +10.2 | +62.1 / +24.8 / +13.8 |
| **"our handcrafted rules are worth nothing globally"** | **+0.8 Elo** | **−18.1 Elo** |

That verdict rested on an almost exact cancellation: a **−51 Elo** loss in the
mirror at an assumed 13.8% weight against a **+47 Elo** gain versus Mega Lucario
at 12.8%. The true weights are **33.3%** and **4.0%**. The cancellation was an
artifact of the wrong denominator, and the rules are not neutral — they are
**actively harmful at the level we now play.** (We had already disabled them, on
narrower mirror-only evidence. The right call for the wrong reason is still a
process failure.)

**And the correction dissolved a contradiction we had recorded as unresolvable.**
§5.2 reports mined data saying 52.1% of high-rated seats play our archetype,
while our own census said the mirror was 13.8% — filed as proof that mined data
"can never describe our field". Both numbers are right. They are two points on
one curve, and the whole disagreement was the 300 rating points between where
they were measured.

> **The transferable lesson, and it is the sharper form of §5.2's.** There, the
> sampling frame was chosen by someone else and we failed to ask what it
> excluded. Here we chose the frame ourselves, fixed the first bug, and still
> failed — because **the parameter that governed it was our own success.** Any
> measurement taken on a ladder is conditioned on your position in it. If your
> agent improves, your evaluation set changes underneath you, and every weight
> derived from it silently expires.

### 5.6 Failure five: our A/Bs measured two networks, not one intervention

Every net-vs-net verdict in this project was built the same way: train the
treatment and its control at two seeds, play n≈1500–2000 games per cell, pool.
The interval we printed was the **sampling** interval — how much the games
wobble. It silently assumed the two nets were the intervention.

They are not. Retraining with a different seed produces a *different network*,
and on the final experiment we caught that term exceeding the one we were
quoting:

| arm | seed 0 → seed 1 | swing | 95% sampling interval |
|---|---|---|---|
| vs `rule:archaludon` | +0.018 → −0.073 | **0.091** | ±0.051 |
| **mirror, direct head-to-head** | 0.524 → 0.451 | **0.073** | ±0.036 |

The second row is the damaging one. The direct mirror match-up is our *tightest*
instrument — treatment against control in a single head-to-head, no third party,
√2× the resolution of any anchor — and it still moved twice as far between seeds
as the games alone can explain. Four of five arms flipped sign between seeds.

⇒ **two seeds under-resolves every anchor we own, the mirror included.** A
verdict at two seeds is a statement about two particular trained networks; to
make it a statement about an *intervention* takes 3–5 seeds and a comparison of
distributions. This does not retract the results that ran at n=2000 with
replicated seeds — their point estimates stand and their signs replicated — but
their published intervals were optimistic, and every single-seed anchor reading
elsewhere in this work is worth less than its stated interval.

Related, and caught in the same session: a treatment-minus-control delta is a
difference of **two independent cells**, so its standard error is √2× a single
cell's. Our own driver printed the single-cell width, understating resolution by
**41%**, until it was recomputed by hand rather than read off the log.

### 5.7 Failure six: we had audited every result and never once audited the apparatus

The five failures above were each found by a result that looked wrong. On the
twenty-second day we ran the opposite exercise: **nothing looked wrong, and we
audited the validation flow anyway** — the arena, the agent constructor, the
archive format, the rating fitter and the census that supplies the weights —
asking of each part only *what does this do that nobody has ever checked?*

It found **seven defects**. This section reports them together because the
pattern across them is worth more than any one of them.

| # | defect | size | did a published number move? |
|---|---|---|---|
| 1 | The Crustle anchor changed **deck** as well as pilot, under one archived name | **+0.140** deck term vs ≤0.027 for every pilot term | 🔴 **two attributions retracted** (§6.3) |
| 2 | The census keyed evolution lines by card **id**; `evolvesFrom` is a **name** | 228 broken links; 69/75 → **74/75** correct | 🟡 **weights restated**, no verdict changed (§6.3) |
| 3 | The Elo fitter was **numerically divergent** for fifteen days | `rule:crustle` swung **8,586 Elo** between consecutive iterations | ✅ **no** — every published Elo is a win-rate conversion |
| 4 | A pinned net that failed its load guard **silently played a different net** | would have run a 496-wide net against a 708-wide control | ✅ **no** — all 32 nets on disk load |
| 5 | The degradation counters were wired into the submission, **not the arena** | an arm falling back on every decision returned a score and no complaint | ✅ **no** — measured separately, and never fired |
| 6 | `bc` with no explicit net is an **unversioned identity** | **1,218 games** pooled under one name over four days of a moving checkpoint | ✅ **no** — no cross-era comparison was published |
| 7 | Archives **append**, so a silent re-run left a control that was never published | 3,000 control games against 1,500 treatment games in one file | ✅ **no** — scores are parsed from the run, not the file |

**Two of the seven changed something we had written; five could not have.** The
arithmetic of *why* is the finding, and it is not luck in the way it first looks.

🔴 **Defects 3–7 all live in parts of the flow that produce no number a human
reads.** The rating fitter's output is never quoted — every Elo figure in this
report is converted from a win rate — so fifteen days of divergence cost nothing
and, for exactly the same reason, went unnoticed. The health counters print
nowhere. The archive's redundancy is visible only to a reader re-deriving a
result, and until §6.1 nobody had. ⇒ **An instrument nobody quotes is an
instrument nobody checks, and the two facts have the same cause.** Our own codex
already contained the rule — *a metric that never prints is not a metric that
passed* — and we had applied it to the agent's diagnostics and never to our own
tooling.

✅ **And the reason the published verdicts survived is a discipline adopted for an
unrelated reason.** After an anchor was found to have drifted between two
sessions, we made it a rule that **every strength claim runs its control
back-to-back in the same session against the same instrument.** That rule was
written to defeat drift. What it actually buys is much broader: **any defect that
shifts an instrument's level cancels in a difference measured through it.** A
divergent fitter, a mis-specified anchor, a stale weight — none of them can move
a treatment-minus-control delta whose two arms met the identical apparatus
minutes apart.

🔴 **Which is precisely why defect 1 was the one that bit.** It is the single
defect where the two arms did *not* share an instrument: the runs before and
after the pilot repair were separated by days and were handed different decks,
and the archive recorded both under one name, so nothing announced that the
comparison had stopped being back-to-back. **The exception proves the mechanism
rather than the discipline's luck.**

⚠ **What we are not claiming.** "No verdict changed" is a statement about the
seven defects we found, not about the ones we did not. The audit was a day of
reading our own code with a specific question, not a proof of correctness, and
the fact that its two real findings were both in the *oldest and most reused*
component of the flow — the anchor set — is the part we would extrapolate from.

---

## 6. Robustness and consistency

Two questions, and they are not the same one:

* **Does the agent perform consistently under repeated matches?** That is a
  question about the *instrument's* noise, and we measured it rather than
  assuming it (§6.1).
* **Does any result depend on a particular matchup or starting position?** That
  is a question about the *design*, and it is the harder one (§6.2).

The exhibit for both is the **weighted multi-anchor A/B table** — seven anchor
decks covering 90.6% of the measured field, n≥2000 per cell, every anchor a real
pilot rather than a self-play stand-in. **That table has now been caught wrong
four separate times** (§6.3), which is the most useful thing we can say about it.

### 6.1 Consistency under repeated matches, measured

Every interval in this report assumes games are independent Bernoulli trials.
That assumption is testable and had never been tested: the engine's RNG is a
single continuous stream inside a compiled library, so correlation between
consecutive games is a live possibility rather than a pedantic one.

We split every archived cell with n≥1000 into blocks of 100 consecutive games
and compared the observed between-block variance to the binomial expectation:

| | |
|---|---|
| cells tested | **163** |
| blocks | **3,811** |
| mean dispersion ratio | **0.984** |
| median | **0.987** |
| pooled standard error | **±0.023** |

**1.00 is exactly binomial.** The arena sits 0.7 standard errors from it. There
is no over-dispersion, no hidden correlation from the shared RNG stream, and a
Wilson interval at *n* games means what it says.

⚡ **Read this together with §5.6 and the pair is the honest picture.** The
*game-sampling* term is exactly as wide as we print it; the *training-seed* term
is wider than we printed it. Our published intervals were correct about the part
they modelled and silent about the part they did not.

✅ A second consistency check fell out of the same audit: re-deriving ~15
published scores directly from the raw game archives, with an independently
written seat-correction, reproduced them to 3–4 decimal places. That is the check
which would have caught the seat-indexing bug that once turned an 0.888 into an
0.510.

### 6.2 Dependence on specific matchups

The anchor set is chosen to make this answerable rather than arguable:

* **Seven anchors spanning 90.6% of the measured field**, each weighted by its
  observed share, so no single matchup can carry a verdict by itself.
* **Seats alternate every game** and each match plays both, so first-player
  advantage cancels by construction rather than by correction.
* **Every anchor is a real opponent pilot**, not our own net holding someone
  else's deck — the one exception (`bc:garchomp`) is labelled an upper bound
  wherever it appears, because our net piloting a foreign 60 measures *deck ×
  how well we pilot it*.
* **A per-card liveness instrument** (`p34`) asks, for each of the 19 slots a
  deck change might touch, how often it is live in each matchup. **17 of 19 are
  mirror-safe.** The two that are not are exactly the two a mirror-only test
  would have judged wrongly: Tool Scrapper is played **0.00 times per mirror
  game** while being drawn in 81% of real games, and the Froslass line sees under
  a quarter of its real use in the mirror.

🔴 **The uncomfortable finding is that breadth is not the same as sensitivity.**
Decomposing the weighted variance shows the mirror does not merely dominate the
anchor set — *in variance terms it essentially **is** the set*, and the anchors
we added for representativeness are cheap to carry but contribute almost nothing
to resolution. Sorting anchors by how well they separate two of our nets also
sorts them, inversely, by how representative they are: the only two near 0.5 are
`rule:v10` (5.3% of the field) and `rule:archaludon` (8.0%), both measured at
**0 of 47 games above opponent rating 900**, while every anchor that represents
the field we actually play is one we already beat 75–89% of the time. **45.3% of
every weighted verdict now sits on anchors near the ceiling.** Adding them bought
honesty, not statistical power, and we report both facts because only reporting
the first would be flattering.

### 6.3 Four ways this exhibit has been wrong

Here is the table as we first wrote it, over 109 games while we sat at ~820
rating:

| anchor | share of field | our WR (real games) |
|---|---|---|
| Alakazam / Telepath | 22.0% | 66.7% |
| Grimmsnarl mirror | 13.8% | 60.0% |
| Crustle | 12.8% | 57.1% |
| Mega Lucario ex | 12.8% | **50.0%** |
| Archaludon ex | 10.1% | **45.5%** |

**Failure one — the shares are a function of our own rating, not of the meta**
(§4f). Re-measured over 75 games at 915–955, the same five anchors read:

| anchor | share at ~820 | **share at ~955** | share above opp. rating 900 |
|---|---|---|---|
| **Grimmsnarl mirror** | 13.8% | **33.3%** | **51.1%** (71.4% above 1000) |
| Alakazam / Telepath | 22.0% | 21.3% | 33.3% |
| Archaludon ex | 10.1% | 8.0% | **0 of 47 games** |
| Crustle | 12.8% | 6.7% | 9.1% |
| Mega Lucario ex | 12.8% | **4.0%** | **0 of 47 games** |

**Failure two — one of the anchors was throwing games** (§8ah). The Crustle
pilot scored every Pokémon except Dwebble at −5000 for a bench play, so once its
Dwebbles were gone it played on an empty bench until the first KO ended the
match; it **ended its turn exposed 0.667 times per game and lost 2 of 2 that
way.** **An anchor that loses games does not add noise — it biases every A/B
that uses it in our favour, in the direction that looks like progress.** It was
found by a human watching a replay, in a project that had run arena A/Bs at
n=2000 for fifteen days, because *an anchor that throws games still returns a
number.*

**Failure three — the anchor was a file *and an argument*, and we only checked
the file** (§8ax). Repairing that pilot moved our score against it by +0.09, and
we wrote that down as the value of the repair. It was not. A rule-based pilot is
tuned for exactly one 60-card list and plays any other through a generic
fallback, and the runs before and after the repair had *also been given different
decks* — 20 of 60 cards apart. Measured directly with the pilot held fixed, **the
deck alone is worth +0.140** [n=2,000/cell], larger than the repair it was
credited to. The archive recorded the pilot under one name for both decks, so
nothing announced the change.

⇒ Two consequences we would rather state than bury. First, the size of the pilot
repair is **≈ −0.04**, not +0.09 — which restores the sign we originally
*predicted* and then talked ourselves out of when the data appeared to disagree.
Second, a later "one-line tie-break is worth more than the whole repair" finding
was the same confound read backwards, and is **retracted**.

**Failure four — the weights themselves were an unaudited estimate** (§8ay).
Every headline here is `W = Σ wᵢΔᵢ`. We had audited the Δ exhaustively and never
once audited the **w**. They come from classifying our own ladder replays by
archetype, and that classifier keyed evolution lines by *card id* when the game's
data keys them by *name* — so of 106 basic printings that share a name, only one
kept its evolutions, breaking **228 links**. One archetype split according to
which reprint the opponent happened to draw. Hand-checking all 75 games:
**69/75 correct before, 74/75 after.**

🔴 **And the bug is not the headline — the sample size is.** These weights come
from **75 games**, so the mirror's 33.3% share is really **[22.5%, 43.2%]**.
*Every correction above lands inside the interval of the estimate it corrects.*

**The key methodological point: an arena result is a weighted average over your
anchor set and nothing else** — and the weights, the members, the members'
*arguments*, and the census that produced the weights have each now been caught
wrong. "Wins two anchors, loses one" is not a verdict.

### 6.4 What we could not make robust, and the honest interval

**Runs are not reproducible, and this is a property of the engine.** The
simulator owns its RNG inside a compiled library; `battle_start` accepts only the
two decklists and returns no seed. We verified that two fresh processes running
an identical five-game script diverge. **Common random numbers are therefore
unavailable**, seat-swapped pairing balances seats but buys no variance
reduction, and no measurement here can be repeated — only spent again. The
±0.036-per-cell floor is structural.

**Propagating the weight uncertainty.** Bootstrapping the 75 census labels
through a completed verdict adds **±0.0031** to a weighted ΔW. Our design
resolution was quoted as ±0.0050, which treated the weights as exact; combined,
the honest figure is **±0.0059** — an 18% widening. It does not overturn that
verdict (still 2.6× outside its kill line, negative in 100% of bootstraps), and
we report it because a conclusion that survives a wider interval is worth more
than one quoted against a narrower one.

⚡ **Weight error bites in proportion to how much the per-anchor deltas differ.**
Where every anchor moves the same way, reweighting barely moves the sum. Where
they disagree in sign, the weight is doing real work and its interval belongs in
the answer.

**Calibration — how much to trust the whole apparatus.** Comparing each anchor's
arena score to the *same agent's* real win rate on that archetype: **the ordering
is correct 4 for 4, and the level reads ~13–27 pp optimistic.** So the arena is
sound for A/B deltas and matchup ranking, and must never be read as a predicted
win rate.

## 7. Opponent modelling: arithmetic rules are matchup-conditional

**The finding, and it is the cleanest single result in the project:**

> **The same three hand-written rules are worth +47 Elo in one matchup and −51 Elo
> in another.** Measured against five anchor decks at n=2000 each.

| anchor | share of field | Δ Elo, rules on vs off | CIs |
|---|---|---|---|
| Mega Lucario ex | 12.8% | **+47** | **disjoint** |
| Grimmsnarl mirror | 13.8% | **−51** | **disjoint** |
| Alakazam | 22.0% | +7 | overlap |
| Crustle | 12.8% | −9 | overlap |
| Archaludon ex | 10.1% | +12 | overlap |
| **global** | 71.5% | **+1 Elo** | — |

**Globally the setting is worth nothing** — the two large effects cancel. A
single global on/off switch is therefore the *wrong control surface*, and any
experiment that measured it in one matchup would have concluded confidently and
wrongly in whichever direction that matchup happened to point. **Ours did: the
agent shipped with rules off on the strength of a mirror-only number.**

**The generalisation we propose:** an arithmetic rule encodes an *objective*
("remove the killable target", "spread energy"), and an objective is only correct
while the strategic context holds. Against a damage-prevention wall, "remove the
killable target" farms a 1-prize basic while the immune attacker sits untouched —
we measured our founding rule at **−0.126** there. **The repair is not to delete
the rule but to branch it**, and the branch condition is readable straight off the
board (*"would our attack deal 0 to this target?"*) with no archetype classifier:
that recovered **+0.104** of the −0.126.

⚠ **We are not shipping the second branch, and the reason is discipline, not
laziness.** The Mega Lucario branch sizes at **~+8 Elo** once the three
statistically-indistinguishable anchors are excluded from the sum. Our
leaderboard readings swing **±50–100 while converging**, so **an 8-Elo change
cannot be validated on the instrument available.** It is logged as a bundle
candidate. **Reporting an effect we cannot measure as though we had measured it
is the failure mode this whole report is organised against.**

## 7b. The ceiling is the clone, not the deck — and we can prove it

**The strongest structural claim in this report, because it is measured on the
band we are trying to reach rather than the one we play in.**

Of 800 seats in the top 400 episodes of 2026-07-29 (`avg_score` ≥ 1144 — roughly
320 points above us), **417, or 52.1%, are playing our exact archetype**, and our
60 is card-for-card the consensus list (that exact list seen 353×).

> **So the gap between our 846.6 and the leader's 1169.2 is a piloting gap, not a
> deck gap.** No decklist change can be worth 320 points when the players at 1169
> are on our list.

Two things follow, and they are the reason this project spent its last days where
it did:

1. **Deck experimentation is not a rank lever here.** We still report it (§9)
   because "we measured a change and kept the list" is deck analysis — but we
   stopped describing it as the fix for our ceiling once it was measured.
2. **The bottleneck is imitation quality, and it is quantified.** Our corpus is
   2,810 games scraped from *these same* 1144+ players. We clone them and play
   ~320 points below them. `context_accuracy.py` puts the disagreement at
   **33.9% of decisions — 3,930 of 6,424 misses in the MAIN context alone.**

**And exactly one intervention has ever closed any of that gap: fixing a
*representational* defect** (§4 — options that referred to different board slots
were bitwise-identical inputs). It was found by reading the feature code against
a premise nobody had checked, not by adding rules. **That is the paper's practical
recommendation: when a behavior clone underperforms its demonstrators, audit what
the option encoding can and cannot bind before you write a single rule.**

### 7b.1 The sharper version of the same claim: identical decklists, +310 rating

The archetype-share argument above is population-level. On 2026-07-31 we obtained
**every recent game of the players ranked #2 and #3** — 330 games from
`李秉叡（ntumlnoob）` (1162.8) and 227 from `Sixth Sense` (1152.4) — and
reconstructed their lists from play and discard. **Both are card-for-card
identical to ours**, down to the 1-ofs. *Same 60 cards, +310 rating.* The deck
question is closed as tightly as this competition permits.

Scoring our live policy net against their actual choices then produced the result
this section did not expect (`EVIDENCE` §8q):

| demonstrator | rating | rows | top-1 disagreement |
|---|---|---|---|
| 48 other pilots of the same deck | ~1110 | 10,088 | **27.2%** |
| Sixth Sense (#3) | 1152 | 18,296 | **34.4%** |
| ntumlnoob (#2) | 1163 | 25,775 | **40.1%** |

Read on its own, that table says **agreement falls as the demonstrator gets
better**. But its three rows come from three different sources, so rating is
confounded with dump, date, opponent pool and collection process — and the
obvious rival explanation is that our net has simply *seen* the mid-rated
players and not these two. So we tagged every one of the 248,985 corpus rows
with the leaderboard score of the demonstrator who made that choice, and
re-asked the question inside a single collection process (`EVIDENCE` §8r).

**The effect survived, and it changed shape.** Against 87 teams playing our
identical deck, in the same two dumps, in the same week, **none of whom the net
has ever trained on a single row of** (n = 22,768 scoreable decisions):

| demonstrator rating | rows | top-1 agreement |
|---|---|---|
| below 900 | 1,288 | 66.7% |
| 900–1000 | 1,879 | 75.0% |
| 1000–1050 | 2,740 | 75.1% |
| **1050–1100** | **8,915** | **76.1%** ← peak |
| 1100–1150 | 7,946 | 70.9% |
| Sixth Sense (1152) | 18,296 | 65.6% |
| ntumlnoob (1163) | 25,775 | 59.9% |

> **Our clone does not agree less with better players. It agrees less with
> everyone who is not in the 1050–1100 band — and it falls off just as steeply
> downward (66.7% below 900) as upward.**

That is a much more useful statement than the monotone one, and it is the
paper's central claim about behavior cloning here: **the objective reproduces
the modal policy of a mixture, so agreement measures distance from that mode,
not skill.** Two rival explanations were eliminated on the way, both of which we
had previously accepted on thin evidence:

- **Familiarity is not it**, and this time with a real control group: 87
  demonstrators the net trained on *zero* rows of are predicted at 73.6%, while
  the six it trained on more than 8,000 rows of each sit at 69.3%. Exposure buys
  nothing. (Our earlier refutation of this rested on a single player at n=61.)
- **Identity is not rating.** Inside the training corpus the curve is flat —
  **−0.03 pp per +100 rating** across 12 demonstrators — and two players 3 points
  apart in rating (1166.1 and 1162.8) sit **9 pp** apart in agreement.

One incidental finding is worth recording because it dates every claim of this
kind: **a demonstrator is a submission, not a person.** Sixth Sense's two agents
in the same dump are predicted at 67.0% and 62.2% with disjoint confidence
intervals, and their games from two days earlier score 74.8%. An imitation
target has a shelf life of days.

### 7b.2 The objection that had to be answered first: covariate shift

Agreement is always measured on the demonstrator's *own* trajectory
distribution. A stronger pilot reaches board states our clone rarely occupies,
so some of that 40% could be behavior cloning's classic compounding-error
problem rather than a policy anyone could copy. Every number above would then be
measuring where we go, not how we play.

We tested it by taking the two policies off the human labels entirely and
comparing **them to each other, on both state distributions** (`EVIDENCE` §8s).
Policy A is our live net; policy B is A fine-tuned on the #2 player's 330 games.

| states | rows | **A disagrees with B** | A vs the human who played | B vs the human who played |
|---|---|---|---|---|
| **our own** 54 ladder games | 4,476 | **26.7%** | **1.7%** | 27.2% |
| **ntumlnoob's** games | 25,775 | **31.9%** | 40.1% | 19.4%* |

**Disagreement does not collapse on our own board states — 26.7% against 31.9%.
It is near-symmetric, so the two nets are genuinely different policies and not
one policy measured off its support.** Covariate shift is not the explanation,
and "is the expert's policy better?" becomes a question the arena is entitled to
answer.

The 1.7% cell is the positive control this line of work previously lacked: those
replays *are* the submission that *is* policy A, so a sound pipeline has to
reproduce them almost exactly, and it does. (*The 19.4% is in-sample for B; its
honest held-out figure is 32.8%.)

**Where the two policies differ is where the misses always were:** MAIN (38.7%
of our states, 45.8% of theirs), damage-counter placement, and fetch. Where the
engine forces the move — `REMOVE_DAMAGE_COUNTER_COUNT`, `ACTIVATE` — they agree
100.0%, which is the instrument checking itself.

### 7b.3 So we tried to fix it — and the way it failed is the result

Two interventions, both aimed at the mode-averaging diagnosis, both with the
same bar fixed in writing before either was trained: **+50 Elo weighted across
our five field anchors, or it is a chapter and not a submission.**

1. **Rating-weighted cloning.** Weight every one of the 248,985 rows by
   `exp((rating − max)/T)`. T was chosen by a stated rule — the most aggressive
   reweighting keeping effective sample size above 100,000 rows — giving
   **41.0%** ESS. Everything else is the control's recipe, on rows that are
   byte-for-byte the control's rows.
2. **Single-expert cloning.** Fine-tune the same net on the #2 player's 330
   games. It worked *as imitation*: held-out agreement with them rose
   **59.9% → 67.2%**.

| net | what it clones | disagreement with **the field** | disagreement with **the expert** | head-to-head vs the live net, n=2,000 |
|---|---|---|---|---|
| live | the ~50-pilot mixture | **30.2%** | 40.1% | — (control) |
| rating-weighted | mixture, tilted to the top | 32.0% | **40.2%** | **0.421** [0.400, 0.443] ≈ **−55 Elo** |
| single-expert | one 1163-rated player | **36.2%** | 19.4%* | **0.370** [0.349, 0.391] ≈ **−92 Elo** |

> **Read the third and fifth columns together. Disagreement with the field goes
> 30.2 → 32.0 → 36.2; strength goes 0 → −55 → −92. Every step away from the
> field's modal policy cost strength, in order — and the net that best imitates
> the second-best player on the leaderboard is the weakest agent we have built.**

Two details make this more than a null:

- **The rating-weighted net moved agreement with the expert by 0.1 pp — from
  40.1% to 40.2%, which is to say not at all — while paying the full price of
  discarding 59% of its effective sample.** §7b.1 predicted exactly that, in
  advance: our corpus's median demonstrator is already rated 1125, past the
  agreement peak, so there was very little mode left to un-average.
- **The single-expert net genuinely learned the expert** — §7b.2 shows the
  difference is a real, symmetric policy difference and not covariate shift —
  **and got worse anyway.** Imitating one strong player at the cost of six points
  of field agreement was a losing trade at every point we measured.

⚠ **The honest limit.** Our arena's anchors are field-like by construction, so
"closer to the field wins here" is partly what the instrument rewards. The
defence is calibration — the arena ranked four of four real matchups correctly
and predicted 0.770 against a deck we then beat in 76.9% of 13 real ladder games
— but it is a defence, not a proof. Settling it would need a ladder submission
of a net measured 92 Elo down, on a board that resolves ±50–100, at the cost of
evicting a live agent. **We judged that a bad trade and we are reporting the
negative result on the arena instead.** (*In-sample; 32.8% held out.)

**What we take from B7 as a general claim:** *behavior cloning gives you the
mode of your demonstrator mixture, and moving the target off that mode costs
more than the better target gains — even when the better target is measurably,
symmetrically better and you successfully imitate it.*

## 7c. The deck

### 7c.1 The concept: a 320 HP attacker that refuels itself, behind chip damage the opponent cannot block

`Marnie's Grimmsnarl ex / Munkidori`, {D}-type. The plan is not a damage race —
it is to field a body nothing removes cleanly, then win on arithmetic the
opponent cannot interact with.

**The win condition.** *Marnie's Grimmsnarl ex* (×3) has **320 HP** and attacks
for **180, plus 30 to a benched Pokémon** (Shadow Bullet). The bench 30 is the
part that matters: it is not a bonus, it is the deck's clock.

**Why the deck can afford it.** Grimmsnarl ex's ability **Punk Up** fires *when
you evolve into it* and searches **up to 5 Basic {D} Energy out of the deck**,
attaching them to any Marnie's Pokémon. Energy acceleration is welded to the
evolution itself, which is why a deck that attacks with a 320 HP ex runs **10
energy and zero manual acceleration**. *Rare Candy* (×3) skips Morgrem so the
trigger lands a turn early.

**Two damage sources the opponent cannot block.**

* ***Munkidori* ×4 — Adrena-Brain**: once per turn, move up to **3 damage
  counters** from one of your Pokémon to one of theirs. It does two jobs in one
  activation — it *repairs the 320 HP wall* and *adds reach* — which is why the
  deck runs the full four rather than treating it as tech.
* ***Froslass* ×2 (behind *Snorunt* ×2) — Freezing Shroud**: during every
  Pokémon Checkup, put 1 damage counter on **every Pokémon with an Ability, on
  both sides**, except Froslass. Unblockable, symmetric, and asymmetric in
  practice — our attacker is being healed by Munkidori while theirs is not.

**The consistency package exists to make one line happen on schedule.**
*Spikemuth Gym* (×4, Stadium) lets each player tutor a **Marnie's** Pokémon once
per turn — our entire evolution line is Marnie's, so it is a repeatable engine
tutor for us and much weaker for most opponents. *Buddy-Buddy Poffin* (×4) puts
the basics down, *Team Rocket's Petrel* (×4) tutors **any Trainer**, and
*Lillie's Determination* (×4) shuffle-draws 6 (8 while we still hold 6 prizes).
*Boss's Orders* (×2) drags the prize we need; *Unfair Stamp* (×1) is the comeback
card, playable only after we lose a Pokémon.

⇒ **How the win actually arrives.** Rarely by one big attack. Shadow Bullet's
bench 30, plus Adrena-Brain's up-to-3 counters, plus Froslass's per-Checkup
counter, mean the opponent's whole board accumulates damage it cannot heal, until
everything on it dies to a number *smaller than a full attack*. **The deck wins
by making the opponent's board fragile, not by making our attacker bigger.**

### 7c.2 How the deck and the agent line up — and the twist

This engine demands arithmetic: *which* target dies to exactly 30, *which*
Munkidori still needs a {D}, *which* Pokémon has 3 counters available to move.
We first supplied that as three hand-written rules — `chip_target`,
`energy_spread`, `counter_source` — precisely because the network's option
encoding showed neither HP nor attached-energy counts, so the clone was aiming
chip damage at chance. Each cleared its own A/B.

🔴 **And the shipped agent runs all three turned OFF.** Once the option encoding
was extended to carry that same information as *features*, the three rules
measured **0.427** against the resulting net — actively harmful, because they
were now overriding a network that could finally see what they were computing.
`build_submission.py` pins them off at build time.

⚡ **That is this project's whole thesis, told through the deck.** The deck's
game plan defined what information the agent needed; we supplied it twice, once
as rules and once as representation, and **the representation won and made the
rules redundant.** The alignment between deck and agent is real — it just lives
in the feature vector rather than in an `if` statement.

### 7c.3 Can we do better than the list we copied?

We did not design our 60. We mined it — it is card-for-card the most common exact
list among top episodes, seen **353 times in a single day's data**. That is an
honest starting point and a slightly embarrassing one, so the question we owed
the reader is: **can we do better than the list we copied?**

**Answering it required building an instrument first.** For most of the project
we avoided decklist changes with a hand-waved caveat — *"every change is
off-distribution for the net"* — which is an excuse, not a measurement. It has a
mechanism, so we measured it. A card is encoded twice: **derived properties** (HP,
retreat, damage, cost satisfaction), computed from the card database and
therefore correct for any card on first sight; and a **card-id embedding**, which
is meaningful only if that id appeared in training. Our corpus contains exactly
**134 distinct card ids**, so 1,166 of the 1,300 embedding rows are still random
initialisation. ⇒ **A swap inside the 134 is low risk; a swap outside it is the
real hazard.** The excuse became a filter.

**Then we sized every slot**, over 75 real ladder games and 7,094 of our own
decisions. Two things fell out that we would not have guessed.

**First, deck swaps are measurable where play-rules are not — and the reason is
which frequency you count.** Three arithmetic-rule candidates died this week
because the *situation* was rare (0.187–0.27 firings per game against an n=2000
A/B that resolves 0.021 of win rate). A replacement card is different: it sits in
the deck whether or not the old card is ever played, so the relevant rate is the
**draw** rate. Our thinnest card is played 0.13 times per game but is **drawn in
81%** of them. That single distinction is why deck work deserved days and the
last three rule candidates did not.

**Second, the obvious cut was disqualified by the matchup rather than by its
value.** Tool Scrapper is our least-used card — and it is played **0.00 times per
game across 24 mirror games**, necessarily, because our own list runs no Tools
and there is nothing to scrap. Testing it in the mirror would have returned
*"cutting it is free"* by construction, with the matchup producing the answer
instead of the card. **A slot audit must be stratified by matchup before it can
rank anything.** This is the same sampling-frame error as §4f and §5.2, in a
third costume.

**Then we tested three variants against a same-deck control.** The control
matters: identical decks, identical net on both seats, `n=4,000` — it measures
what the harness does when nothing has changed, and it came out at **0.4980
[0.483, 0.513]**, essentially on 0.500. Every deck claim below clears that floor
or it is noise.

| variant | swaps from consensus | score | vs control | p |
|---|---|---|---|---|
| **control** — identical decks | 0 | **0.4980** | — | — |
| `Dawn → 4th Grimmsnarl ex` | 1 | 0.4911 | −0.007 | 0.54 |
| `Poffin −1, Scrapper −1 → Budew ×2` | 2 | **0.4757** | **−0.022** | **0.021** |
| four-card reconfiguration | 4 | **0.4637** | **−0.034** | **0.0004** |

🔴 **Strength falls monotonically with distance from the list we copied.** One
swap is indistinguishable from noise; two lose ≈17 Elo; four lose ≈25.

**And the comfortable explanation is ruled out.** The natural defence of a losing
variant is that the clone never learned to use the new card. We recorded six
games and counted: Budew reached the Active spot in **3 of 6**, and its attack
fired **4 times**. The mechanism works at roughly the intended rate and the deck
is worse anyway. **This is not an execution failure — the plan loses.**

> **What we take from it.** The consensus 60 behaves like a **local optimum**, and
> our policy is a clone trained on games played with that exact list, so it is
> tuned to it twice over. "We measured a change and kept the list" is a deck
> result, and it is a stronger one than an unmeasured opinion in either
> direction. ⚠ **Its scope is narrow and we state it rather than bury it: all
> four A/Bs were run in the mirror.** That flatters variants which cut
> mirror-dead tech — and they lost regardless — but a card aimed at a different
> matchup cannot be judged by these runs at all.
>
> 🔴 **The methodological failure is ours and worth naming: three of these
> variants were single-card guesses.** Guessing produced one null and two
> significant losses, which is what an unstructured search over a
> near-optimal 60 should produce. A serious deck programme needs a
> matchup-stratified design over the whole slot ranking, not a sequence of
> hunches.

### 7c.4 So we built the search, pre-registered it, and it found nothing

Guessing was retired as a method and replaced with a design. **The candidate list
— 11 variants — was frozen in a file committed *before* any variant deck
existed**, because a search over *k* variants at α=0.05 manufactures a winner at
k≈20, and the whole point was to be unable to shop for one.

**Two stages, with the multiplicity rule doing the work.** Stage 1 *ranks* all 11
cheaply in the mirror against the same-deck control; it computes no p-values by
design. **Only the top-1 is promoted**, and the pre-registration said in advance
that if it fails, the search is over and no second candidate is tried. Stage 2
confirms that one variant across all seven anchors at a Neyman-allocated
**57,600 games** — an allocation chosen because it reaches the same precision as
equal-n for 55% of the cost.

| stage | result |
|---|---|
| stage 1, 11 variants | 🔴 **all 11 at or below the control.** Six of eight mirror candidates lost significantly |
| stage 2, candidate G, 7 anchors | **ΔW = −0.0155** against a design resolution of ±0.0059 — **negative on 7 of 7** |

⛔ **The kill line was not met, so the search is over and the consensus 60
stands.** That clause is what made this one test instead of eleven.

⚡ **Three things this bought that a single A/B could not.** First, **Ultra Ball
was held fixed as the added card across six different cut slots and lost in all
six** (0.439–0.488) — which separates *"we cut six good slots"* from *"the added
card is wrong"*, and only the second explains six losses. Second, the cheap
screen predicted the expensive confirmation almost exactly: stage 1 called the
mirror at **0.501** on 4,000 games, stage 2 at **0.500** on 15,800 — the
two-stage design's central bet, confirmed on first use. Third, a separate
isolation run showed a five-card bundle's −0.073 splitting into **−0.040 for one
2-card change and −0.033 for the other three**, which is why bundled changes are
not attributable and are no longer run.

🔴 **We also learned the exposure filter is necessary but not sufficient**, and
it is the most reusable finding here. Ultra Ball sits at **5.59×** the training
exposure of our weakest card and lost every slot; Energy Switch sits at 3.61× and
the net played it **once in 28 offers**. **Card-level exposure is not the binding
constraint — card × deck-context is**, and nothing we built measures that.

> **The honest summary for Deck Score.** We netdecked a list, built an instrument
> to judge changes to it, pre-registered a search over the whole slot ranking,
> ran 57,600 confirmation games, and **kept the list we started with**. The
> expected outcome was written down as a null *before the run* — §7c.3 had
> already measured strength falling monotonically with distance from the
> consensus 60 — and a null is what it returned. **"A proper search over the slot
> space, and the list survived it" is a deck result.** It is not an Elo lever,
> and we do not present it as one.

## 8. Negative results

Honest nulls at n≥2000 are the section we are most confident in.

- **Search** (ours 0.323; the public baseline's MCTS never executes).
- **Self-play RL** — ⚠ **not a negative result, and we are correcting our own
  filing of it.** This entry previously read "dropped on the above plus the
  compute budget", which describes a *decision*, not an experiment: no run, no
  `n`, no interval. We had propagated it into four documents as though it were
  measured. It belongs in §5's failures-of-our-own-process, not here, and it is
  the third instance of the same error the rest of §5 describes.
- **E1 auxiliary representation learning** — three seed-matched arms, all null
  at n=2,000: outcome **0.505 [0.484, 0.527]**, selection count **0.507
  [0.486, 0.529]**, and both heads **0.500 [0.478, 0.522]**. The auxiliary
  diagnostics learned, but their gradients did not improve playing strength.
- **Data scaling** — three axes, all negative.
- **Embedding repair** — the terminal chapter of the feature axis (§4g). Three
  *genuine* defects in the card-identity tables, all repaired and all null:
  weighted **−0.0078** for the full repair, **−0.0047** for the isolated padding
  fix, both inside the noise of a seed change. (The full-repair figure was
  published as −0.0099; recomputing it from its own table during a later audit
  gave −0.0078. Same verdict, and we corrected it rather than leave it.) The reason is the finding: the
  optimiser had already routed around every one of them. ⚡ Not wasted — the
  repair also removed **11.5% of the network** for 0.0018 of held-out fit,
  which with the opposite experiment (8.2× parameters, −43 decisions) bounds
  capacity from both directions.
- **Demonstrator weighting** — a fourth axis, and the one we most expected to
  work. Weighting every training row by its demonstrator's leaderboard rating
  (effective sample size held at 41% of the corpus, everything else identical)
  scored **0.421 [0.400, 0.443]** against the net it was built to improve. It
  did not merely miss the pre-registered +50 Elo bar; it lost by more than the
  bar's width. The measurement in §7b.1 predicted this before the run — there
  was little mode left to un-average, because our corpus's median demonstrator
  is already rated 1125 and the net fits its strongest demonstrators as well as
  its weakest.
- **Boss's Orders — four separate interventions, all null** (0.489, 0.490, 0.493,
  0.493). Every one moved its audit rate exactly as designed first. **Moving an
  audit rate is not winning games.**
- **The pooled option-set encoder** (§4e) — the most instructive near-miss here.
  It is the one intervention that did exactly what it was designed to do at the
  level of the fit (+214 correct held-out decisions, the project's largest) and
  returned **+14 Elo, one noise-width, negative on two anchors of five.** We
  report it as a negative because the pre-registered bar is playing strength and
  it did not clear it — **not** because the mechanism is refuted. The honest
  statement is narrower and more useful: *summarising the option set into the
  shared state vector improves imitation of the mixture and barely improves
  play*, which is the same lesson as §7b arriving by a different road.
- **Observable matchup adapters (E2)** — hard-routed residual adapters on
  visible opponent Grimmsnarl / Alakazam lines left general-route agreement
  exactly at **0.7137**, raised mirror-route fit **0.7300 → 0.7340**, then
  scored **0.521 [0.490, 0.552]** against the seed-matched control in the
  mirror. A correct observable router that protects the base path is still not a
  strength win. ⚠ We previously also cited "0.782 versus 0.792 against
  `rule:alakazam5`" as evidence; **that arm is two independent cells and its
  −0.010 sits 3.6× inside its own ±0.036 resolution.** It is uninformative, not
  a null, and the mirror arm carries this verdict alone.
- **Planning scale (E5)** — the most instructive negative on this list, because
  the first way we read it was wrong. Repaired `seq,reply` vs frozen v5 scored
  **0.380 → 0.420 → 0.515** as determinizations went 4 → 8 → 16, and the
  preregistered confirmation cell at 32 scored **0.230 [0.177, 0.293]**. We filed
  that as "more compute collapses the curve." 🔴 **But realized compute across
  those first three arms was 652 → 616 → 606 seconds — flat.** The nominal budget
  quadrupled and the work did not, so the three cells that opened the
  confirmation gate are three draws at constant compute (pooled **0.4383
  [0.399, 0.478]**, n=600 — already a loss). The variable that actually moved is
  **how often the planner fires**, and score falls monotonically with it across
  all four arms over an 8× range: 4.2% → 0.515, 7.4% → 0.420, 10.8% → 0.380,
  35.0% → 0.230, while it overrules the clone at a near-constant 58–61%
  throughout. **The planner is harmful in proportion to how often it engages** —
  which is the §4 encoding argument arriving from a third direction, since the
  sequencer reads the same feature vectors the clone does and can only overrule a
  better-calibrated prior with a worse one. No planner is promoted and
  distillation is not opened.
- **The five unsized state features** (§4c) — worse than nothing, at −22 Elo
  against a net with no block at all. A negative result about *our own process*:
  they were bundled in because they were cheap, and cheap was the only argument.
- **Model capacity** — the cleanest null in the report, and the one that
  redirected the remaining work. Identical corpus, identical recipe, only the
  width changed: **594k → 1.56M parameters moved held-out agreement by two
  decisions out of 12,939** (3,902 misses → 3,900), and **4.87M made it worse**
  (3,945). Both larger nets drove training loss far below the small one's while
  validation peaked early and declined — they already had more capacity than the
  features could use. Combined with the demonstrator result above, this left the
  *encoding* as the only surviving explanation for the residual — and the
  encoding is precisely where our one large win came from (§4).
  ⚠ **We then checked that elimination instead of trusting it**, because an
  elimination argument is only as good as its enumeration. §4b's ceiling says
  the option layout permits 95.6% agreement, so "the encoding" cannot mean *the
  right answer is inexpressible*. The surviving form of the claim is narrower
  and testable: the **state** does not carry what would let the net choose
  between options it can already tell apart.
- **Turn-level sequencing** — three builds, and the one that is worth reporting
  is the last. The prototype scored **0.075** (n=40) by maximising the value of
  the board at the end of *our* turn, which structurally cannot see the
  opponent's reply. We diagnosed that, simulated the opponent's whole turn
  before scoring, and the result moved to **0.375 [0.311, 0.444]** at n=200 —
  the largest movement any change to it produced, and still ≈89 Elo behind a
  clone that costs 1 ms per move. We then checked that the recovery was the
  *diagnosis* and not the extra compute the fix required: at matched budget, the
  same searcher **without** the reply scores **0.165 [0.120, 0.223]** against the
  reply arm's **0.375 [0.311, 0.444]**, disjoint intervals at n=200 each. **The
  design fix is worth ≈ +0.21 on its own.** So the explanation was confirmed by
  controlled experiment — and the agent died anyway. **A correct explanation of a
  failure does not entitle you to a fix.**
- **Decklist variants** — a pre-registered two-stage search over the whole slot
  ranking: **11 candidates frozen before any variant deck existed**, all 11 at
  or below the same-deck control in stage 1, and the single promoted candidate
  confirmed negative on **7 anchors of 7** over 57,600 games (ΔW **−0.0155**
  against ±0.0059). The consensus 60 survived a proper search. §7c.4.
- **Six candidates closed by sizing before an A/B was spent.**

---

## Appendix A — the research program we declined

We began from a generic "synthesise the best of game AI" brief (NNUE-style
incremental evaluation → policy/value nets → neural-guided MCTS → belief states
and opponent modelling → GNN/Set-Transformer encoders → large-scale self-play).
We declined most of it **on our own measurements rather than on taste**, and the
disposition of each plank is tabulated in `ROADMAP.md`. The one plank our thesis
predicted should work — richer state representation — is the one that produced the
project's largest measured effect (§4).

## Appendix B — reproducibility

Every number traces to an archived run. `out/arena/*.jsonl` holds the per-game
records; `out/logs/RECEIPTS.txt` indexes every `score= [CI] W/D/L over N` line
produced by the project. Commands for each experiment are in `HANDOFF.md` §5.
