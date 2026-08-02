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

**Why not reinforcement learning.** Dropped on the same evidence, plus ~1.4 cores
of real throughput on the development machine.

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
| search / RL | 3 | all negative (§2) |
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

## 5. Measurement discipline, and two failures of our own process

The full codex is 17 rules, each paid for by invalidated work (`HANDOFF.md` §2).
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

---

## 6. Robustness and consistency *(in progress)*

The exhibit is the **weighted multi-anchor A/B table** — five anchor decks
covering 71.5% of the measured field, n=2000 per cell, every anchor a real pilot
rather than a self-play stand-in.

🔴 **Both columns of the obvious version of this table are wrong, and finding out
why is the chapter.** Here it is as we first wrote it, measured over 109 games
while we sat at ~820 rating:

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

**Failure two — one of the five anchors was throwing games** (§8ah). The Crustle
pilot scored every Pokémon except Dwebble at −5000 for a bench play, so once its
Dwebbles were gone it played on an empty bench until the first KO ended the
match; it **ended its turn exposed 0.667 times per game and lost 2 of 2 that
way.** Our arena number against it was **0.663** against a **57.1%** real win
rate. **An anchor that loses games does not add noise — it biases every A/B that
uses it in our favour, in the direction that looks like progress.** It was found
by a human watching a replay, in a project that had run arena A/Bs at n=2000 for
fifteen days, because *an anchor that throws games still returns a number.*

**The key methodological point: an arena result is a weighted average over your
anchor set and nothing else** — and both the weights and the members have now
been caught wrong. "Wins two anchors, loses one" is not a verdict.

**Calibration — how much to trust it.** Comparing each anchor's arena score to
the *same agent's* real win rate on that archetype: **the ordering is correct 4
for 4, and the level reads ~13–27 pp optimistic.** So the arena is sound for A/B
deltas and matchup ranking, and must not be read as a predicted win rate.

*To add: the completed rules-on sweep; the P6a sweep; seat-balance tables.*

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

## 8. Negative results

Honest nulls at n≥2000 are the section we are most confident in.

- **Search** (ours 0.323; the public baseline's MCTS never executes).
- **Self-play RL** — ⚠ **not a negative result, and we are correcting our own
  filing of it.** This entry previously read "dropped on the above plus the
  compute budget", which describes a *decision*, not an experiment: no run, no
  `n`, no interval. We had propagated it into four documents as though it were
  measured. It belongs in §5's failures-of-our-own-process, not here, and it is
  the third instance of the same error the rest of §5 describes.
- **Data scaling** — three axes, all negative.
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
- **Decklist variants** — 0.490.
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
