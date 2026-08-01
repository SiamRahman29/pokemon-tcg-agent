# ROADMAP — dual goal: public LB + WIN the Strategy Category

> **This file decides what the engineering is FOR.** `HANDOFF.md` §3 is the live
> engineering plan; `report/EVIDENCE.md` is the hypothesis log. Update all three
> at the end of every session.
>
> **Deadlines (user-confirmed 2026-07-30): simulation 2026-08-17** (~18 days),
> then ~2 weeks continued play; **strategy report 2026-09-14**. Agent work is the
> scarce-time track; the report has a month of runway afterward — **but its
> evidence is generated now, so log it now.**
>
> **Rubric (user-confirmed): Model Score 70% + Deck Score 20% + report writing
> 10%.** Verbatim text in `competition_details_and_rubric.md`.

---

## 0. The verdict (2026-07-30): partial pivot, not a track switch

**Keep the engineering line. Do NOT adopt the research program in the appendix.
ADD a strategy-dossier track starting now.**

Why not build MCTS + GNN + belief nets + self-play:

1. **Our own measurements already falsified its core planks *for this
   competition*:** search 0.323 vs the clone; V10's MCTS has never executed and
   holds LB 950+ anyway; ~~self-play RL,~~ more data and more val accuracy all
   negative; nothing at the top of the board is learned. 18 days on ~1.4 cores
   cannot change those results.
   🔴 **AMENDED day 14: strike self-play RL from that list — it was never run.**
   A compute prior was filed as a measurement in four files (`EVIDENCE` §2's
   retraction box). The *decision* to decline it in the day-8 pivot still stands
   on its own terms; what is withdrawn is the claim that we had **evidence** for
   it. The rest of the sentence is measured and unaffected.
2. **The rubric rewards reasoning, not architecture.** 70% Model Score = clarity
   of rationale + originality + *soundness* + consistency + robustness + LB
   performance (**one bullet of five**). A half-built GNN is neither performant
   nor sound. The description says outright that mid-tier LB + deep analysis can
   win.
3. **The declined program becomes report material.** "We derived testable
   hypotheses from the AlphaGo/poker/NNUE program and here is the experimental
   evidence for why each plank does or doesn't transfer under these constraints"
   is a *stronger* originality story than half-building the stack — because we
   have the negative-result receipts.

Why not pure status quo either — three rubric-shaped holes:

- **Deck Score is 20% and we have almost no documented deck rationale** (we
  netdecked grimmsnarl from top episodes). *Partly fixed 07-30, see Track C.*
- **"Avoids over-reliance on specific matchups"** — every routine number is vs
  ONE anchor (`rule:v10,noS`/`lucario_v10`), and the meta shifted. HANDOFF §3.1
  (re-anchor) is therefore *also* the robustness evidence. Same work, double
  credit.
- **The report does not exist**, and its best material (the day-by-day
  hypothesis→measurement→verdict log) is being produced now and is cheapest to
  capture now. *Started 07-30: `report/EVIDENCE.md`.*

**Episode data:** ~20 GB of episodes is published per day. More *training* data is
dead (EVIDENCE §1) — **never bulk-download for imitation.** Targeted mining is
still cheap and is the right tool for a **deck matchup win-rate matrix among
high-rated players** (a headline figure for the robustness and Deck Score
sections). **Pull manifests broadly (they're tiny), episodes selectively**, and
keep the manifest even when episodes are pruned (`replays/manifests/`).

> 🔴 **REVISED 2026-07-31 — mining must NEVER pick our anchors again.** Kaggle's
> daily episode datasets contain **nothing below `avg_score` 1055**; we play at
> **825–952**, and the 800–1000 buckets are empty. So mined episodes describe a
> band 200–320 Elo above ours, and clause (a) above — "mining picks our anchors" —
> is what retired `rule:v10`, the deck that turned out to be **12.8% of our field
> and our worst matchup** (EVIDENCE §8i). **Anchors come from
> `scripts/p9_field_census.py` on our own submission replays. Mining is for the
> top-of-ladder figures and decklist consensus only.**
>
> ⚡ **AMENDED 2026-07-31 (day 10) — there is now a THIRD data source, and it is
> the best one we have: targeted per-team replay dumps.** Every recent game of one
> named team, with an `episodes_meta.json` sidecar carrying `submissionId`. Two
> exist (`replays/sixth_sense_31-07-2026`, `replays/ntumlnoob_31-07-2026` — #3 and
> #2 on the LB, both on our exact 60). **This is not subject to the censoring
> problem** because we are not sampling a band, we are naming a demonstrator.
> **But the band rule still binds for ANCHORS**: a 1160 pilot is 330 points above
> our field, and an anchor stronger than the population compresses A/B deltas the
> way the 0.911 Crispin anchor did in reverse. **Dumps are for IMITATION targets
> (B7); anchors still come from `p9_field_census.py` on our own replays.**

---

## 1. The narrative we are building toward

> **"Clone the field, then audit the clone's blindness."** ⚡ **Amended day 12: the audit has now paid twice, and the second time it was systematic rather than lucky** — §8x computes what the encoding CAN express (95.6%) instead of arguing it, and §8y derives the missing inputs by diffing the observation against the feature code rather than recalling them. The narrative's second half is now a *method*, not an anecdote. We behavior-cloned
> 2,810 human games, systematically enumerated the decisions its features cannot
> express (no HP, no damage, no attached energy), and repaired them with
> arithmetic rules. We present a falsifiable discriminator — **rules that delete
> a *dominated* option win (3/3); rules that pick a side in a *tradeoff* lose
> (0/4)** — validated by arena A/B at n≥2000 every time. Plus a
> measurement-discipline codex (HANDOFF §2, 17 rules, each paid for) and
> evidence-backed negative results on search, RL and data scaling.

Everything from here should either (a) raise the LB, (b) add a chapter, or (c)
both. **Prefer (c).**

---

## 2. Tracks

### Track A — Leaderboard (live plan: HANDOFF §3)

Priority order unchanged: resolve `55077709` (§3.0) → re-anchor the arena (§3.1)
→ Crustle probe + pilot (§3.2) → P2 lethal audit. **Bar for shipping: head-to-head
win at n≥2000 vs the current anchors.**

New lens: §3.1's multi-anchor arena is the rubric's consistency/robustness
exhibit — **archive the per-anchor A/B tables, they go in the report verbatim.**

### Track B — Strategy dossier

Deliverables: **`report/EVIDENCE.md`** (the log — exists, backfilled 07-30) and
**`report/STRATEGY.md`** (the report — not started). Outline:

1. **Approach & rationale** — why imitation of the field + rule repair under
   2 vCPU / 600 s; why not search or RL (our measurements *and* the V10
   never-runs finding).
2. **Hypothesis log** — from `EVIDENCE.md`; remaining backfill sources are the 76
   semantic git commits and `out/arena/` (indexed in `out/logs/RECEIPTS.txt`).
3. **The discriminator** — dominated vs tradeoff, the 7-rule table, and *why* it
   predicts (the net has seen 2,810 games of human tradeoffs; it has seen zero HP
   values).
4. **Measurement discipline** — HANDOFF §2's 17 rules as a methods section.
5. **Robustness & consistency** — multi-anchor A/B tables, seat balance, n and CI
   everywhere, and the `counter_source` false-positive post-mortem (§3.0). **A
   documented failure of our own validation process, diagnosed and fixed, is
   rubric gold — not embarrassment.**
6. **Opponent modeling / meta adaptation** — the measured meta shift, archetype
   detection, the Crustle case study.
7. **Deck concept** (feeds Deck Score — Track C).
8. **Negative results** — search, ~~self-play,~~ data scaling, Boss's Orders ×4,
   decklist variants. Honest nulls at n≥2000 are rare on Kaggle and scream
   soundness. ⚠ **Self-play removed day 14 — it was never measured**, and a
   chapter that boasts of honest nulls must not pad itself with an unmeasured
   one. The retraction itself belongs in §5 (process failures) instead.

**Process rule:** every experiment gets an `EVIDENCE.md` entry **the session it
concludes** — hypothesis, command, n, CI, verdict, one sentence of
interpretation. End-of-session checklist: HANDOFF plan ✓, EVIDENCE entries ✓,
this file's calendar ✓.

> ⚠ **AMENDED 2026-07-31 — "the session it concludes" means CONCLUDES, not
> "looks decided".** §8i was written into `EVIDENCE.md` after **2 of 5 anchors**
> reported and had to be retracted the same session; the full sweep reversed its
> sign. **If runs are still in flight, log the numbers and leave the verdict
> blank.** A verdict written early gets copied into HANDOFF, then ROADMAP, and by
> then three files assert it (rule 15).

### 🔴 The doc-discipline audit (2026-07-31) — what is actually going wrong

**Maintenance is not the problem; two other things are.**

| file | commits | days touched | verdict |
|---|---|---|---|
| `HANDOFF.md` | 62 | 5 of 5 | ✅ updated every session |
| `ROADMAP.md` | 10 | 2 of 2 since it existed | ✅ |
| `report/EVIDENCE.md` | 14 | 2 of 2 since it existed | ✅ |
| `report/STRATEGY.md` | — | — | ❌ **promised 07-30, created 07-31 after slipping ~4 sessions** |

**Failure 1 — the deliverable with a real deadline was the one that slipped.**
Everything with a same-day feedback loop got updated religiously; the thing due
**09-14** got deferred every session because nothing forced it. It is now started.
**Standing rule: `STRATEGY.md` gets one edit per session, however small.** It is
30%+ of the rubric (Deck 20% + writing 10%, plus the soundness/consistency/
robustness bullets inside Model Score's 70%), against LB's **one bullet of five**.

**Failure 2 — updates are ADDITIVE, and that has cost real points.** `HANDOFF.md`
went **135 → 1,579 lines in 5 days** and now carries **27 lines** of
retraction/superseded markers. Claims get appended and annotated rather than
revised, so a wrong claim survives in the files that copied it. This is not
cosmetic:

> "`lucario_v10` is 0% of the meta" was written once on 07-30 and propagated to
> HANDOFF §1, rule 12, §3.1 **and** ROADMAP's calendar. Acting on it retired the
> anchor that would have caught B1 — the deck that turned out to be **12.8% of
> our field and our worst matchup.**

**HANDOFF rule 15 already warns about exactly this** ("a premise repeated in three
files is not thereby verified — it is just load-bearing") and the project did it
anyway, twice. **Standing rule: when a claim is retracted, grep for it across all
four files in the same commit.** `grep -rn "<the claim>" *.md report/` — cheap,
and it is the step that was skipped.

### Track C — Deck Score (20%)

**Documentation half — the core argument is now measured, not asserted.**
Both snapshots are mined (`out/meta/`, full table in HANDOFF §1 / EVIDENCE §8b):

- **Our archetype was and remains the field's most-played deck** — 51.8% of seats
  pre-shift (52.2% WR, 22 teams, top three rated teams all on it), 52.1%
  post-shift. And **`decks/grimmsnarl.py` is card-for-card identical to the
  current consensus 60** (seen 353× on 07-29, re-verified after the shift). That
  answers "why grimmsnarl" with data instead of "we netdecked it".
- **But our win rate fell 52.2% → 47.5% while our share held** — the field did not
  abandon our deck, it *learned to beat it*. **Crustle: 1 seat in 1,600 → 18.1% at
  56.6% WR, with the LB top two on it.** That is the counter-meta, measured, and
  it makes the experimentation half urgent rather than speculative.
- Still to add: the re-anchored deck sweep (HANDOFF P1 caveat: also run
  `bc:plain,noChip,noSpread,noSrc` — with our grimmsnarl-only rules on, the sweep
  answers "what to ship", not "which deck has the higher ceiling").
- **Key-cards section:** Munkidori/Adrena-Brain economy (two armed Munkidori =
  60-point swing/turn), Shadow Bullet snipe + `chip_target`, and the Boss's Orders
  saturation finding (we play it on 38% of legal turns; more copies measured null
  at 0.490).

**Experimentation half — the anti-counter program** (ordered; each step gates the
next; the user asked for this explicitly on 2026-07-30):

1. **Build the measuring instrument first: a Crustle anchor.** Deck is in repo,
   no pilot. **No deck experiment is interpretable until this exists** — a
   decklist change measured only against `rule:v10` answers the wrong matchup.
2. **Probe the mechanic** (HANDOFF §3.2): do Adrena-Brain / Freezing Shroud
   counters bypass Mysterious Rock Inn's damage prevention? One probe run. **If
   no → the passive-damage line is dead**; enumerate other outs (Tool Scrapper?
   stadium replacement? a non-ex attacker line?) from the card pool *before*
   touching the list.
3. **Play-priority adaptation before decklist surgery** (better prior, zero
   off-distribution cost): in Crustle mode, prioritize fetching/arming the
   passive-damage Pokemon (`TO_HAND` and attach rules, conditional on archetype
   detection — B3).
4. **Then decklist changes, measured properly:** the Froslass line is the only
   growable passive-damage line (Munkidori capped at 4). Candidate swaps get A/B'd
   **against the Crustle anchor AND the mirror AND `rule:v10`** — a tech card must
   pay for its slot in the bad matchup without bleeding the good ones. ⚠ Every
   change is off-distribution for the net; mitigation = rules own the changed
   cards' decisions, or a B1 retrain absorbs them.

**Stewardship narrative:** "we measured a change and kept the list" and "we
measured a change and made it" are *both* deck analysis. Only **unmeasured** list
edits are forbidden.

---

## 2.5 The breakthrough hunt — ranked candidates for 970 → 1200

**The innovation gap and the LB gap are the same problem.** Incremental targeting
rules bought ~150 points and are hitting diminishing returns (the last three
A/B'd rules were null; the remaining selects are closed or tiny). The top is ~210
points away, and the originality score won't be won by a fourth chip-targeting
rule. So: qualitatively different levers, each with a **cheap probe → kill
criterion → full A/B** ladder. **Run probes in rank order; kill fast; write up
every kill** — every candidate is a report chapter regardless of outcome.

| # | candidate | why it could be the jump | probe (cheap) | kill criterion |
|---|---|---|---|---|
| ✅ **B1** ⭐ **three generations now measured, and the returns are falling ~3× each: +115 → +37 → +14 Elo (§8f, §8z, §8aa). The axis is nearly spent — one variant left (day 14 item B), then close it.** | ~~**Feature augmentation + retrain**~~ **WON 2026-07-30/31** | The premise was even better than stated: the blindness was **representational, not informational** — the net always had per-slot HP, but `opt["index"]` was never encoded, so two options naming two copies of the same card were bitwise identical | **DONE.** `optfeat` v3 (25→37 cols), corpus `artifacts/pds_v3`, 7 A/Bs at n=2000 | **Pre-registered kill line was ≤0.52. Measured 0.661 vs the SHIPPED agent** (≈ +115 Elo) and **0.878** vs a same-corpus control. **The rules are now harmful (0.427)** — the method inverts. `EVIDENCE` §8f. ⚡ **Instance 2, the STATE block (day 12): 0.567 [0.545, 0.588] vs a byte-identical control, replicated 0.539 at a second seed against a measured seed-null of 0.482, pooled ≈ +37 Elo, better on 5/5 anchors** — and found by ENUMERATION (`p18`), which is the part to reuse: diff the observation against what `featurize()` reads, then size each survivor. §8y/§8z |
| ~~**B2**~~ | ~~**Arithmetic MAIN layer** — the lethal audit~~ | ~~missed lethal is the classic handcrafted-agent gap~~ | **KILLED 2026-07-30** by `scripts/p2_lethal.py`, 200 games | **both cuts empty: 316/316 lethals taken and all 316 FORCED (honest denominator 0), 7/803 promotion cases with retreat illegal in all 7. Grimmsnarl ex has ONE payable attack, so the decision does not exist in this deck** — EVIDENCE §8 |
| **B3** ⬆⬆ | **Matchup branches** (archetype detection only if needed) | **PROMOTED TO #1 2026-07-30 — the repair for a MEASURED defect: `chip_target` scores −0.126 against `rule:crustle` while paying +0.077 in the mirror** (EVIDENCE §8c). Also the report's opponent-modeling chapter. ⚠ **Try the classifier-free version first** — the Crustle condition reads straight off the board ("our attack would deal 0 to their Active"), no archetype model needed (HANDOFF §3.3). ✅ **Instance 1 (Crustle) SHIPPED and recovered +0.104.** ⚡ **Instance 2 is now open and better-evidenced: MEGA LUCARIO.** v3 is **−50 Elo** there (the only anchor of five it loses) and we won **36.4% of 11 real games against opponents rated 85 points BELOW us** — two independent instruments, same matchup (EVIDENCE §8i). ⚠ **Also a THIRD damage-reduction deck now has an anchor:** Archaludon's Full Metal Lab is −30 into any Metal Pokemon, which `WALL_POKEMON = {345}` does not model | **Lucario first: audit before rule (rule 14).** Read the 11 real games + an `opportunity_audit`-style probe vs `rule:v10`; ask whether `chip_target`/`energy_spread` are net-negative there the way `chip_target` was vs Crustle. Then (a) one-line branch, A/B n=2000 vs **all five** anchors | the audit sizes the effect below what n=2000 resolves (±0.021) → close it by sizing like the Morgrem out, no A/B spent |
| ~~**B4**~~ 🔴 **DEAD 2026-07-31 (day 11)** | **Turn-level planning with the unused 600 s pool** — enumerate within-turn action *sequences*, score end-of-turn states with `evalfn`/`textdmg` | We use 0.1 s of 600 s; the #1 player reportedly uses the full budget. **Distinct from the dead game-tree search**: no rollouts, no determinized opponent turns — just sequencing of our own turn, where the variance problem that killed our search (terminal 0/1, SE≈0.14) does not exist. Novel for this board | ✅ **PROBE RUN 2026-07-31 — B4 SURVIVES ALL THREE KILL CRITERIA** (`EVIDENCE` §8l). Denominator **62% of turns** have ≥2 real selects (rule 13 passed, unlike the Morgrem out); space is median **98M** so exhaustive is dead, but `fs.step` runs at **7,698/s** ⇒ **~78,000 candidate sequences/turn** — an ordinary beam; and `evalfn` has real signal (**AUC 0.685 early / 0.901 late** vs the eventual result). **Throughput is NOT the binding constraint**, which is the opposite of what was predicted here | ⚠ **The decisive question is still UNMEASURED and is strictly harder: within-turn, same-position ranking.** Across-game AUC is compatible with zero discrimination between twenty end-of-turn states reachable from the *same* position. **Next probe (~1 h): enumerate k≈20 sequences per real turn, report the eval spread against the eval's own noise and how often argmax ≠ the clone's pick. Near-identical scores kill B4 there** — no beam width saves it. Rule 3: nothing so far licenses a build, only the next probe. 🔴 **CLOSED day 11 (§8v): the DESIGN diagnosis was right and it was not enough.** Adding the one-ply opponent reply — simulate their whole turn, score when control returns, score terminal games as RESULTS — moved the prototype **0.075 → 0.375 [0.311, 0.444] n=200**, the largest movement any B4 change produced, and it is **still ≈ −89 Elo** against a clone costing 1 ms. Pre-registered line was 0.40. ✅ **Confound RESOLVED same session:** at matched budget, `seq,sb1.0` without reply scores **0.165 [0.120, 0.223]** vs the reply arm's **0.375**, n=200 each, **disjoint** — the design fix is worth **≈ +0.21 on its own**, not the extra time. **The diagnosis was confirmed by controlled experiment and B4 died anyway.** 🔴 **Consequence for NNUE: B4 was its only non-search consumer, so an incremental evaluator buys nothing here until one exists.** |
| **B5** ⬆ | **Deck adaptation vs the counter-meta** (= Track C experimentation half) | **PROMOTED 2026-07-30 — the premise is now measured, not hypothesised: Crustle went 0.06% → 18.1% of the field at 56.6% WR with the LB top two on it, while our WR fell 52.2% → 47.5%.** Matchup EV has moved more than any play-skill rule can recover | Track C steps 1–2 (Crustle pilot, then the mechanic probe) | probe says counters don't bypass the prevention → passive-damage line dead; enumerate other outs before any list change |

| ~~**B7**~~ ⚡ **NEW day 10 — arm 1 KILLED day 11** | **Demonstrator selection: rating-weighted and single-expert cloning** | **The premise is measured, not assumed** (EVIDENCE §8q): top-1 agreement with a demonstrator **falls monotonically as the demonstrator improves** — 27.2% miss against ~1110 pilots, 34.4% against #3 (1152), **40.1% against #2 (1163)**, n≥10k per group. Every net we have trained targets the **modal action of a ~50-pilot mixture**, and the best players are furthest from that mode. **We have never cloned ONE policy.** Newly possible: 330 games from #2 and 227 from #3, both on our exact 60. ⚠ Distinct from the dead "more data" axis (§1) — this is demonstrator *selection*, not volume | ✅ **the measurement half is DONE.** Next, in order: (1) tag every corpus row with its demonstrator's LB rating (one `competition_leaderboard_download` + `info.TeamNames`); (2) **rating-weighted clone** on all 248,985 rows; (3) **single-expert fine-tune** of v3 on `artifacts/pds_ntum` (27,318 rows) | 🔴 **DAY 11 VERDICT — the bar was set before the run and arm 1 failed it outright.** (1) ✅ rating-tagging shipped (`--ratings`, 94–98% seat coverage). (2) 🔴 **rating-weighted clone: 0.421 [0.400, 0.443] n=2000 vs v3 in the mirror ≈ −55 Elo** — not a null, a loss (§8t). (3) ✅ **covariate shift RULED OUT** — policy-vs-policy disagreement is 26.7% on our states vs 31.9% on theirs, near-symmetric, with a 1.7% positive control (§8s). (4) ⚡ **§8q's headline NARROWED by a much harder test**: agreement **peaks at 1050–1100 and falls in BOTH directions** (66.7% below 900, 59.9% at 1163) over 87 same-deck, same-week, **zero-exposure** demonstrators — so agreement measures **distance from the fitted mode, not skill**, and familiarity is refuted with a real control (§8r). (5) 🔴 **arm 2, the single-expert fine-tune, loses HARDER: 0.370 [0.349, 0.391] ≈ −92 Elo** — and it imitated *successfully* (ntumlnoob agreement 59.9% → **67.2%** held out). **⇒ B7 CLOSED.** The two failures are ordered by how far each net moved from the field (miss 30.2% → 32.0% → 36.2%, Elo 0 → −55 → −92): **agreement with the FIELD predicts strength; agreement with the EXPERT anti-predicts it** (§8u) |

**Sequencing (updated 2026-07-30 — B2 is dead, so the ladder shortened):** B1 is
independent of the re-anchor and is now the **top-ranked unstarted candidate** —
and B2's kill strengthens it, because the reason B2 died (this deck's arithmetic
decisions are all in *targeting*, none in *attack selection*) means the remaining
MAIN mass is exactly the tradeoff class where only better features, not more
rules, can help. B3 gates B5-step-3 and half of Track C. B4 is the most
speculative and the most original — **run its cheap probe early** (one afternoon)
so the investment decision is data-driven.

**B6 — `deckfacts.py`, rule generalization (user-agreed, parked with a
condition).** Compute payable attacks / provably-dead attaches / damage
thresholds from the decklist + card db, so `targeting.py` rules consume *facts*
instead of hardcoded card ids. **Do it WITH the first decklist change** (a hard
prerequisite for Track C step 4), **not before** — it buys zero Elo on an
unchanged deck and needs its own regression A/B. It now has three concrete
instances waiting: `MUNKIDORI`/`DARK_ENERGY`/`BOSS_ORDERS` in `targeting.py`, and
as of 2026-07-30 **`WALL_POKEMON = {345}`** — the Crustle branch's condition,
which the general version would learn from the event log (an attack that logged
`value: 0` against that card id) instead of naming a card. Also the natural home
for keeping only `counter_source`'s arithmetic half.

**Rule-11 discipline applies inside every candidate:** B2's edge is the arithmetic
cut, not a general MAIN scorer; B4's scorer is handcrafted eval, not a learned
value net (measured dead); B5's changes are measured against three anchors, not
vibes.

---

## 3. Calendar (sim closes 2026-08-17; report due 2026-09-14)

| window | Track A (LB) | Track B/C (dossier) | gate |
|---|---|---|---|
| **07-30** (day 8, DONE) | ✅ §3.0 resolved — `counter_source` **stays** (+0.052 vs the new anchor); ⚠ meta measured **at avg_score ≥1144 only** (`lucario_v10` → 0% ⚠ **retracted 07-31: it is 12.8% of OUR field**, Crustle 0.06% → **18.1%**); ✅ `rule:crustle` pilot imported + 2 anchor decks; ✅ **all 3 shipped rules re-A/B'd at n=2000 — found `chip_target` at −0.126** and ✅ **shipped the matchup branch** (0.559 → 0.663); ✅ §3.2 premise verified in-engine (+ the **Morgrem** out); ✅ **B2 killed** | ✅ `EVIDENCE.md` created and backfilled (§8b/8c/8d added same session); ✅ both meta snapshots archived; ⏭ `STRATEGY.md` still not started | ✅ met: `counter_source` verdict, the real meta, B2 verdict. **Not** done: B1, B4 probe |
| **07-30 pm** (day 8, 2nd session) | ❌ **Morgrem out CLOSED BY SIZING — not built** (~0.2 firings/game, the free route already 95.4% right, and a tradeoff; `EVIDENCE` §8e). Bought **rule 14 — size before you build** and a correction: "our attacker deals 0 into theirs" is true of their **Active only** (the bench snipe kills Dwebble unprevented). ⛔ **Crispin pilot ruled UNOBTAINABLE** — 272 public notebooks enumerated, no toolbox pilot, and 3 high-Elo titles pulled + refuted against the LB (one author is at 605.0). Standing re-read: **rank 224/4,000 at 950.2** | — | the sizing gate worked as designed — one candidate died for the price of one probe, no A/B spent; a second died for the price of an LB query |
| **07-30 night → 07-31** (day 9) | 🔴 **B1 SHIPPED AND LOST.** `55116557` reads **825 at 10 h** where P4b was **958 at 4 h** — the arena said +115 Elo, the ladder said ≈ **−130**. Bundle verified fine (the net WAS live: 40.7% index-0 over 4,682 selects, vs 100% for a fallback). **Second systematic arena↔LB inversion** — `counter_source` is still **837.5 vs P4b's 952.0** after 2 days of converged play. **Cause measured: 63% of real opponents are archetypes we have never A/B'd against; the mirror, where B1 was decided, is 9%.** ✅ **The endgame question is ANSWERED: displayed = best ACTIVE, so the eviction alone cost rank 224 → 605.** Good news that transferred: **Crustle 76.9% over 13 real games** (arena predicted 0.770) | `EVIDENCE` §8g + §8h; `scripts/p8_optv3_replays.py`; `replays/submission_optv3` is now the anchor source | ⏭ ~~restore P4b first~~ 🔴 **both halves of this gate were wrong** — the anchors came first and they showed the restore is not clearly right (it evicts our BEST active) and the −130 was a frozen-score artifact. EVIDENCE §8i |
| **07-31 pm** (day 9, 2nd session) | ✅ **THE ANCHOR SET IS REBUILT AND THE PANIC WAS ARTIFACTUAL.** `p9_field_census.py` names the field from **our own 109 games** (Alakazam 22.0%, mirror 13.8%, Crustle 12.8%, **Mega Lucario 12.8%**, Archaludon 10.1% — top 5 = **71.5%**); `import_field_agents.py` added `rule:alakazam5` (a **5th-place** pilot) and `rule:archaludon`. Full 5-anchor sweep at n=2000: **v3 is +36 Elo over P4b**, losing only Mega Lucario (**−50**). 🔴 **Retracted the same session: the "−130 regression" was a FROZEN-vs-live comparison** — live-vs-live is v3 819.8 vs P6a 845.0 = **−25**, inside the LB's ±50–100. 🔴 **And the structural finding: Kaggle's episode datasets stop at avg_score 1055 while we play at 825–952, so mined meta can NEVER describe our field** | `EVIDENCE` §8i (incl. the arena-calibration table: ranks matchups 4/4 correctly, reads ~15 pp optimistic) | ✅ gate met — and the gate itself ("an arena whose verdicts survive contact with the ladder") is now satisfiable |
| **08-01** | ⚡ **1. Re-test v3 WITH the rules on, across all 5 anchors** — the 0.427 that justified rules-off was a **mirror-only** result and the mirror is 13.8% of the field. Cheapest possible gain, ~35 min of CPU. **2. The Mega Lucario matchup** — the only negative term in the weighted table (−50 Elo) *and* 36.4% real WR against opponents rated 85 points BELOW us. This is **B3's second instance**, and B3's first instance (Crustle) recovered +0.104. **3. Sweep P6a** to settle the P4b-restore question without spending a submission | matchup win-rate matrix; **start `STRATEGY.md`** | a weighted, 5-anchor verdict for what to ship |
| **08-01** (day 10) ⚡ | **THE ONLY RANK LEVER IS IMITATION QUALITY** — §8o proved the deck is not the bottleneck (**52.1% of the ≥1144 band plays our exact archetype**) and §8k proved every agent we own is within **36 Elo**, below the LB's resolution. So: **hunt the next REPRESENTATIONAL defect the way B1 was found** — read `optfeat.py`/`features.py` against `context_accuracy.py`'s **3,930 MAIN misses**, starting with inputs that are absent entirely (⛔ **the list named here — hand size, stadium, prizes, turn — was 3/4 WRONG and is retracted by §8y**; only the stadium was absent. The derived replacements are `turnActionCount`, `select.effect` and the stadium). **Bar: +50 Elo weighted over the 5 anchors or it is a chapter, not a submission.** Decide B4 (prototype loses 0.075, §8n) | **`STRATEGY.md` — one edit minimum, every session.** Day 9 produced 3 chapters: the censored sampling frame, ±47/−51 matchup-conditional rules, and everything-within-36-Elo | a candidate that clears +50 Elo weighted, **or** an honest write-up of why the ceiling is the clone |
| **07-31 pm** (day 10) ✅ | **THE P4b RESTORE IS ANSWERED BY EXPERIMENT AND THE ANSWER IS "THE 952 WAS THE BOARD"** — the identical tarball read **833.9 at 4.0 h** against the original's **958.2 at ~4 h**, board ~4,000 → **6,024** (EVIDENCE §8p). §8k is now confirmed on the ladder, not just the arena: our three agents read **833.9 / 818.1 / 841.5** live. ⚡ **And the user supplied targeted expert dumps** — 227 games from **Sixth Sense (#3, 1152.4)** and 330 from **ntumlnoob (#2, 1162.8)**, both on our **exact 60, card for card**. Measured: agreement with a demonstrator **falls as the demonstrator improves** (27.2% → 34.4% → **40.1%** miss), the two top players diverge from us in **different** contexts, and **our corpus is already elite and concentrated** (flg 527 seats at 1125, Dries 490 at 1102, James Cox 414 at 1166) — we clone 1100–1166 play and score 833.9. **⇒ B7** | `EVIDENCE` §8p + §8q; corpora `artifacts/pds_expert`, `pds_ntum`, `pds_grimm_ctrl`; `--player`/`--players-file` and `--all-rows` tooling | ⚠ **gate NOT met and deliberately so: nothing was trained.** Day 10 was measurement; the intervention is day 11's item B, with the +50 Elo bar set **before** the first run |
| **08-01** (day 11) ⚡ | **B7, in order: rating-tag the corpus → rating-weighted clone → single-expert fine-tune.** Bar: **+50 Elo weighted over the five anchors**, else chapter. ⚠ **Address covariate shift before claiming the experts are "better"** — cross-score each policy on the other's states. ⛔ No submission unless the bar is cleared; every submission evicts and the restore already spent one slot | **`STRATEGY.md` §7b** — day 10 handed it a much better instrument (**#2 and #3 play our identical 60**) and a new figure (agreement-vs-rating). One edit minimum | a trained candidate with a weighted 5-anchor number, **or** a written negative result on demonstrator selection |
| **07-31 night** (day 11) ✅ | **B7 RAN, AND THE PRE-REGISTERED BAR DID ITS JOB.** Item A shipped: every corpus row now carries its demonstrator's LB score (`--ratings`, 94–98% of seats; the last 25% of d26 was **one team appearing under three names** — the LB's #1 — fixed with member-username matching + `replays/team_aliases.tsv`). Then four results: 🔴 **BOTH B7 arms LOSE — rating-weighted 0.421 [0.400, 0.443] ≈ −55 Elo, single-expert 0.370 [0.349, 0.391] ≈ −92 Elo, n=2000 each** (§8t, §8u), taking the count of dead "more/better training" axes to **five** against one representational win; 🔴 **and they lose in ORDER of distance from the field** (field miss 30.2 → 32.0 → 36.2%), so **agreement with the field predicts strength and agreement with the expert anti-predicts it**; ✅ **covariate shift is ruled out** by near-symmetric policy-vs-policy disagreement, with a **1.7% positive control** on our own submission's replays (§8s); ⚡ **§8q is narrowed, not retracted** — over 87 same-deck, same-week, **never-trained-on** demonstrators, agreement **peaks at 1050–1100 (76.1%) and falls both ways** (66.7% below 900, 59.9% at 1163). **Agreement measures distance from the fitted mode, not skill** (§8r). Also: a demonstrator is a **submission**, not a person — one team's two agents differ 67.0% vs 62.2%, disjoint | `EVIDENCE` §8r + §8s + §8t + §8u; `p15_rating_curve.py`, `p16_policy_disagree.py`, `p9_field_census.py --us/--emit-players`, `train_policy.py --rating-temp/--init`; **`STRATEGY.md` §7b.1 rewritten around the peak, new §7b.2 (shift) and §7b.3 (the ordering)** | ✅ gate met in the honest sense: a trained candidate with a real number, and the number says no. ⛔ **Nothing submitted** — correct, the bar was not cleared |
| **08-01** (day 12) ✅ | 🔴 **THE PLATEAU BROKE, on the only axis that has ever paid.** First: the "residual is the encoding" claim was checked instead of inherited — `p17_encoding_ceiling.py` computes a hard bound from the shards (bitwise-identical options ⇒ identical logits ⇒ `Σ(1/g)/N` caps top-1) and it is **95.6% against the clone's 69.8%**, so un-expressibility explains **at most 4.4 of the 30.2 points**; the ties that exist are **two copies of one card in one role**, i.e. free choices, and `--equiv` puts honest agreement at **71.0%** (§8x). Then the feature audit was done **by enumeration** (`p18_missing_state_audit.py` diffs the observation against what `featurize()` reads and sizes each field) — which **retracted the candidate list HANDOFF/ROADMAP/EVIDENCE had carried since day 10**: turn, prizes and both hand counts were already encoded (**rule 15, second instance**), and two more died on sizing (§8y). The derived replacements — `turnActionCount`, the select's **effect card**, the **stadium** — became the **v4 state block**: **0.567 [0.545, 0.588] n=2000 vs a byte-identical control**, replicated at a second seed (**0.539**), against a **measured seed-only null (0.482)**, pooled **≈ +37 Elo**, and **better on 5 anchors of 5** including v3's only losing matchup (Mega Lucario 0.505 → **0.549**, disjoint). ⚡ **And it moved held-out agreement by 8 decisions in 12,939 — rule 3's converse, measured** (§8z). ⚡ Also: `55116557` (v3) has climbed to **864.1** against the P4b restore's **824.3**, **confirming §8i's +36 Elo arena prediction on the ladder** | `EVIDENCE` §8x + §8y + §8z (+ the §8p addendum); `p17_encoding_ceiling.py`, `p18_missing_state_audit.py`, `features.extra_feats`, `train_policy.py --seed/--no-extra`, `context_accuracy.py --equiv`; **`STRATEGY.md` §4b new, §8's capacity bullet narrowed** | ⚠ **gate NOT met on its own terms and that is stated, not smoothed: +16.5 Elo weighted is below the pre-registered +50.** Submitted anyway (5/5 anchors, slots not scarce) with the deviation and its reasoning recorded in §8z |
| **08-01** (day 13) ✅ | ⚡ **RANK 268 / 6,088 AT 923.0** — `55156480` (v4) climbed 489 → 853 → 822 → 895 → **923.0** in 3 h, our best live number ever, ⚠ **still climbing and therefore not a settled reading** (rule 2), and **not evidence about the v4 block either way** (+16.5 weighted is far under the LB's ±50–100). Two experiments, and **the pair is the result**: 🔴 **§8aa — the v5 pooled option-set block** (deep-sets in its cheapest form: mean/max over the option encodings appended to the state, so the net can finally see the option SET) **moved held-out agreement 71.0% → 72.7%, +214 correct decisions of 12,939 — the largest agreement gain the project has produced — for +14 Elo pooled over two seeds**, one noise-width, and **negative on 2 anchors of 5** (weighted +7.3). Set against §8z's **+8 decisions for +37 Elo**, the exchange rate between fit and strength differs **70×** between two interventions a day apart ⇒ **`val_top1` is uninformative in BOTH directions and may not screen anything.** ⚡ **§8ab — the v4 ablation** (`--drop-x` zeroes a member's columns, so arch/params/init/rows/seed are identical and `x_mask` in the npz keeps inference honest): dropping any **one** of `turnActionCount`/stadium/effect is **within noise**, dropping **all three is −36 Elo, disjoint** — mutually redundant, jointly necessary, and essentially the whole block — while **the five unsized extras alone are −22 Elo against having no block at all.** ⇒ **§8y's derive-and-size method is what worked, not the bundle.** ⚠ Also: pairwise Elo here **orders consistently but compresses ~23 points over two hops**, so weighted anchor totals are **ordinal, not arithmetic** | `EVIDENCE` §8aa + §8ab; `optfeat.pool_width/pool_scalars`, `train_policy.py --pool/--drop-x`, `features.X_GROUPS`, `x_mask` in `policynet`/`context_accuracy`; **`STRATEGY.md` §4c (audit + ablation), §4d, §4e — three new chapters** | ⚡ **v5 SUBMITTED as `55160229`** — but only after the first verdict (⛔ no: +7.3 weighted, negative on 2 of 5, wrong shape) was **corrected as a framing error**. That verdict answers *"is v5 better than v4?"*; a submission asks *"is v5 better than what it EVICTS?"* — and eviction is by recency, so it displaced **P4b (836.4, last of everything we own)** while v4 keeps its rating and stays active. 🔴 **The +50 bar was written when slots were scarce and every submission evicted something valuable; that premise was relaxed two days earlier and the bar was still carrying its old cost model.** ⇒ **standing rule: name the agent a submission would evict before quoting the bar.** ❌ Track C still not reached — third session running |
| **08-01 – 08-08** | re-run shipping A/Bs vs the new anchors; **B3 archetype detector; Track C steps 1–3 (Crustle pilot + mechanic probe + priority adaptation)**; follow up whichever of B1/B2/B4 survived; matchup win-rate matrix | log every concluded experiment same-session; deck sweep vs the new anchors | ≥1 breakthrough candidate validated at n≥2000 **OR** all five killed with written verdicts |
| **08-08 – 08-14** | consolidate: winners integrated, large-n consistency runs vs every anchor; **Track C step 4 only if steps 1–3 justified it**; submit early enough to converge (~4 h+ climb, slots scarce) | collect figures/tables from arena archives as produced | best agent submitted with ≥3 days of episode history before the deadline |
| **08-14 – 08-17** | **freeze** — no risky submissions (latest-2 eviction trap) | — | sim track locked with the settled pair active |
| **08-17 – 08-31** (continued play) | watch the final pair's rating — **this IS the rubric's "consistency under repeated matches" evidence**; record the trajectory | draft chapters 1–4 | continued-play data captured |
| **09-01 – 09-14** | — | chapters 5–8, figures, full review pass, format for Kaggle, submit | report submitted well before 09-14 |

**Standing constraints:** 5 submissions/day, **latest 2 active** — every
submission near the deadline risks evicting the best agent (HANDOFF §7 ⛔). Score
convergence takes 4+ h, so the *last safe day* for a new agent is ~08-15, and
earlier is better because continued play rewards a settled high-μ pair.

---

## 4. Decision principles

- **Rule 11 governs new rules:** dominated → build; tradeoff → distrust.
- **Rule 10 governs everything:** the arena A/B decides; audit rates and
  narratives don't. This applies to *report claims* too — every number in the
  dossier must trace to an archived run with n and CI.
- **Appendix ideas enter through the same gate as any rule:** cheap probe first,
  then A/B. No architecture work on faith.
- **When LB work and dossier work conflict for time**, note that LB is ~1 bullet
  of 5 in a 70% category and the description explicitly says mid-tier can win —
  **the dossier does not get sacrificed for marginal Elo.** (Being a finalist
  likely needs a respectable LB; 970+ and climbing is already strong on this
  board.)

---

## Appendix — the research program we declined (was `possible.md`)

Kept as a compressed record so the report can say *what* we rejected and why,
without carrying 300 lines of open questions. The original was a generic
"synthesize the best of game AI" brief: modular engine → handcrafted eval +
search → NNUE-style incremental features → policy/value nets → neural-guided
MCTS (PUCT / IS-MCTS) → belief states and opponent modeling → object-centric
architecture (GNN over attachments/evolutions, Set Transformers over unordered
zones like hand/deck/discard) → large-scale self-play with league training →
ablation studies. Inspirations and the principle taken from each: **Stockfish/
NNUE** (incremental sparse evaluation, feature caching), **AlphaGo Zero/KataGo**
(policy+value+search, multi-task value heads), **DeepStack/Libratus/Pluribus**
(information sets, belief updating, opponent modeling), **Hearthstone/MTG AI**
(action generation, sequencing, combo detection), **GNNs** (Pokémon is
relational), **Set Transformers** (zones are unordered).

**Status of each plank against our own measurements:**

| plank | our evidence | disposition |
|---|---|---|
| neural-guided search / MCTS | ours 0.323 (n=31, rollout SE≈0.14); **V10's MCTS never executes and holds LB 950+** | **dead** — EVIDENCE §2 |
| large-scale self-play / league | ~1.4 cores; nothing at the top of this board is learned | **declined on a compute prior — NOT measured** (day-14 correction; EVIDENCE §2's retraction box). ⚡ **And the plank being re-opened is not this one**: fine-tuning an already-decent clone on its own outcomes is a different cost regime from league self-play from scratch, which is all §0 ever considered |
| more data / better representations via training | lw3 (4,010 games, best val acc) lost 0.491; winners-only 0.375 | **dead as stated** — EVIDENCE §1 |
| richer state features | this is the *one* plank our own thesis predicts should work | **live as B1** (§2.5) |
| opponent modeling / beliefs | the top notebooks do the cheap version (matchup rules on revealed cards) | **live as B3**, minus the neural machinery |
| within-turn sequencing (Hearthstone shape) | we use 0.1 s of a 600 s budget; no rollout variance in this restricted form | **live as B4** |
| GNN / Set Transformer encoders | no measurement, and 18 days on 1.4 cores | **not attempted; documented as out of budget** |

**The principle worth keeping from it**, which the project already follows: for
any borrowed idea, ask *what problem does this solve; does PTCG have that
problem; what is the PTCG-specific version; how would we falsify it cheaply?*
