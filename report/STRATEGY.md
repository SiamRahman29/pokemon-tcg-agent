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

---

## 5. Measurement discipline, and two failures of our own process

The full codex is 16 rules, each paid for by invalidated work (`HANDOFF.md` §2).
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

---

## 6. Robustness and consistency *(in progress)*

The exhibit is the **weighted multi-anchor A/B table** — five anchor decks
covering 71.5% of the measured field, n=2000 per cell, every anchor a real pilot
rather than a self-play stand-in.

| anchor | share of field | our WR (real games) |
|---|---|---|
| Alakazam / Telepath | 22.0% | 66.7% |
| Grimmsnarl mirror | 13.8% | 60.0% |
| Crustle | 12.8% | 57.1% |
| Mega Lucario ex | 12.8% | **50.0%** |
| Archaludon ex | 10.1% | **45.5%** |

**The key methodological point: an arena result is a weighted average over your
anchor set and nothing else.** "Wins two anchors, loses one" is not a verdict.

**Calibration — how much to trust it.** Comparing each anchor's arena score to
the *same agent's* real win rate on that archetype: **the ordering is correct 4
for 4, and the level reads ~13–27 pp optimistic.** So the arena is sound for A/B
deltas and matchup ranking, and must not be read as a predicted win rate.

*To add: the completed rules-on sweep; the P6a sweep; seat-balance tables.*

## 7. Opponent modelling and metagame adaptation *(in progress)*

*To write: the matchup-branch result (a damage-prevention wall makes our founding
rule actively harmful, −0.126, recovered +0.104 by a classifier-free branch read
straight off the board); the Mega Lucario investigation; whether a second
damage-reduction archetype needs the same treatment.*

## 8. Negative results

Honest nulls at n≥2000 are the section we are most confident in.

- **Search** (ours 0.323; the public baseline's MCTS never executes).
- **Self-play RL** — dropped on the above plus the compute budget.
- **Data scaling** — three axes, all negative.
- **Boss's Orders — four separate interventions, all null** (0.489, 0.490, 0.493,
  0.493). Every one moved its audit rate exactly as designed first. **Moving an
  audit rate is not winning games.**
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
