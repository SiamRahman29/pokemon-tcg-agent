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
>
> 🔴 **SECOND ROUND ANNOUNCED 2026-08-03 — see §3.5. It changes what this file is
> FOR, in one sentence: the top 8 advance on the STRATEGY CATEGORY result, not the
> simulation leaderboard.** The dossier stopped being 30% of a score and became
> the qualification gate. §3.5 also records the BO3/log-access rule and the
> Round-2 hardware.

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
> measurement-discipline codex (HANDOFF §2, 18 rules, each paid for) and
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
4. **Measurement discipline** — HANDOFF §2's 18 rules as a methods section.
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

> 🔴 **AMENDED 2026-08-01 (day 15) — THE ANTI-COUNTER PROGRAM BELOW IS AIMED AT
> A BAND WE ARE LEAVING, and the correction is a Deck Score chapter in itself.**
> Every share in this section was measured when we sat at ~820. `EVIDENCE` §8ac
> shows the opponent pool is **a function of our own rating**: at 955 the mirror
> is **33.3%** of our games (**51.1% above opponent rating 900, 71.4% above
> 1000**), while **Crustle is 6.7%** and **Mega Lucario and Archaludon are 0 of
> 47 games above rating 900**. The "Crustle counter-meta" is real and it lives
> **below us**.
> ⇒ **Re-aim, in this order: (1) the MIRROR — 33.3% and rising, and nobody has
> ever asked what beats our own 60; (2) anchors for Cynthia's Garchomp ex
> (6.7%) and Dragapult ex (5.3%), which together outrank Crustle + Lucario and
> have none; (3) the stewardship write-up, which this closure supplies.**
> ⚠ The steps below are kept in full because the *method* is right and because
> a re-climb into the 800s would make them live again — **and because "we sized
> a planned change and did not make it" is deck analysis, not a gap.**
>
> ⚡ **AMENDED 2026-08-02 (day 16) — DECK WORK IS NOW SAFE, INSTRUMENTED, AND THE
> USER'S PRIORITY. Three things landed that between them remove every standing
> excuse:**
> 1. ✅ **§8af — the off-distribution worry is SIZED, not hand-waved.** The corpus
>    holds exactly **134 distinct card ids**, so 1,166 of the 1,300 embedding rows
>    are still random init. **Swapping to a card inside the 134 is low risk**
>    (Tool Scrapper, which we play today, sits at 2,820 "as our option"); swapping
>    outside it is the real hazard. That converts "every change is
>    off-distribution" from a blocker into a **filter**.
> 2. ⚡ **Our net is a FIELD clone, not a Grimmsnarl clone** — every card in the
>    Crustle, Crispin/Raging Bolt and Team Rocket lists is trained. So it can
>    pilot those decks, which gives Track C the instrument it never had: **hold
>    the pilot constant and vary the 60.** ⚠ It measures *deck × how well our net
>    pilots it*, not deck strength in the abstract — state that or the number
>    lies.
> 3. ⛔ **But anchor-based deck A/Bs are BLOCKED until the §8ah repair is
>    re-validated** (rule 12, one level up: a decklist measured against a pilot
>    that throws games answers the wrong question). ⚡ **The MIRROR is not
>    blocked** — both seats are our own net on our own 60 — and it is **33.3% of
>    the field, 51.1% above rating 900, 71.4% above 1000**. **Start there.**
>
> 📊 **Top-band sizing for the two archetypes the user spotted among the leaders**
> (`out/meta/post_shift_0729.txt`, 400 games = 800 seats, `avg_score` ≥1144):
> ours **52.1% at 47.5% WR** · Crustle 18.1% at **56.6%** · **Crispin toolbox w/
> Raging Bolt ex 16.9% at 58.5%** (one team, James Cox & Henry Chao) · **Team
> Rocket Spidops 6.6% at 41.5%**. 🔴 **At the top of the board our archetype is
> the most-played and the WORST-performing of the big three** — that is the Deck
> Score question worth answering, and Team Rocket is the weakest deck up there,
> so it is not the one to build for.
> ✅ **§8ag closed Pokégear as a play question** (0.27 real choices/game; we take
> a free Supporter 39/39) but **explicitly left "does the 1-of earn its slot"
> open as a deck question** — and §8af now makes that A/B safe to run.

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
| ~~**B8**~~ 🔴 **CLOSED 2026-08-02 (day 17) — two nulls, 20,000 self-play games** | **RL fine-tune of the clone on its OWN recorded outcomes** (NOT league self-play from scratch — §0 only ever declined that) | 🔴 **Every other axis is dead or spent.** Search (§2), more data (§1), demonstrator selection (§8t/§8u), capacity (§8w, 8.2× params → −43 decisions), within-turn sequencing (§8v), and now **deck (§8al: four A/Bs, monotone worse with distance from consensus)**. The feature axis is the one that ever paid and it is falling ~3× a generation (**+115 → +37 → +14**). ⚡ **RL's blocking objection has been PRICED and it does not bind** (§8ae): a 1% effect at a context recurring 20×/game needs **960 games** against a **5.5M-game** budget; +37 Elo separates in **800**. ⚡ **And the thing a clone structurally cannot know is exactly what an outcome signal supplies** — the corpus says *what humans did*, never *whether it worked*. Prerequisites are BUILT: `harness.Recorder` (§8ad) for trajectories, the feature audit done twice (§8y/§8z, §8ab), the encoding ceiling computed (§8x) | **SMALLEST REAL THING ONLY.** Fine-tune a **small** parameter set (final layer / a low-rank delta) on our own recorded game outcomes; keep the corpus loss as an anchor so it cannot drift off the clone. **A/B at n≥2000 vs a BYTE-IDENTICAL control**, seed floor carried in (§8z's 0.482 null), never on `val_top1` (rule 3: §8z moved agreement 8 decisions for +37 Elo, §8aa moved 214 for +14) | 🔴 **Pre-registered, written before any code: if the fine-tuned net does not beat its byte-identical control by a margin whose CI excludes the seed-only null at n≥2000, B8 DIES and becomes a report chapter.** ⚠ **And §8ae's own warning binds: a sizing probe that fails to kill is not evidence a thing works — B4 passed all three of its kill criteria and died at n=200 (§8v).** **Hard stop 08-08**; after that the calendar is consolidate-then-freeze and there is no room to integrate a winner. 🔴 **VERDICT (day 17): the bar was 0.541 at n=2000 and BOTH runs failed it — 4,000 games 0.512 [0.491, 0.534], 16,000 games 0.506 [0.484, 0.528], with each control sitting on v5 (0.480, 0.491) so the procedure itself is harmless.** ⚡ **4× the data moved the estimate DOWN, which is branch (b) of a decision rule committed before the second A/B reported (`out/logs/b8_prereg.txt`) — so the axis closes on the METHOD, not the budget, and the now-feasible 40,000-game run is NOT indicated.** ⚡ **And a parameter diagnostic taken before the result says what the null means: the advantage weighting moved the head by 34% of the distance the fine-tune itself moved it from v5, so the parameters moved substantially and the win rate did not.** ⚠ **β is named as untested rather than buried: both runs used β=1.0, a sweep was declined by rule, so 'a stronger reweighting might work' is unfalsified rather than refuted.** `EVIDENCE` §8ao |
| **B5** ⬆ | **Deck adaptation vs the counter-meta** (= Track C experimentation half) | **PROMOTED 2026-07-30 — the premise is now measured, not hypothesised: Crustle went 0.06% → 18.1% of the field at 56.6% WR with the LB top two on it, while our WR fell 52.2% → 47.5%.** Matchup EV has moved more than any play-skill rule can recover | Track C steps 1–2 (Crustle pilot, then the mechanic probe) | probe says counters don't bypass the prevention → passive-damage line dead; enumerate other outs before any list change |

| ~~**B7**~~ ⚡ **NEW day 10 — arm 1 KILLED day 11** | **Demonstrator selection: rating-weighted and single-expert cloning** | **The premise is measured, not assumed** (EVIDENCE §8q): top-1 agreement with a demonstrator **falls monotonically as the demonstrator improves** — 27.2% miss against ~1110 pilots, 34.4% against #3 (1152), **40.1% against #2 (1163)**, n≥10k per group. Every net we have trained targets the **modal action of a ~50-pilot mixture**, and the best players are furthest from that mode. **We have never cloned ONE policy.** Newly possible: 330 games from #2 and 227 from #3, both on our exact 60. ⚠ Distinct from the dead "more data" axis (§1) — this is demonstrator *selection*, not volume | ✅ **the measurement half is DONE.** Next, in order: (1) tag every corpus row with its demonstrator's LB rating (one `competition_leaderboard_download` + `info.TeamNames`); (2) **rating-weighted clone** on all 248,985 rows; (3) **single-expert fine-tune** of v3 on `artifacts/pds_ntum` (27,318 rows) | 🔴 **DAY 11 VERDICT — the bar was set before the run and arm 1 failed it outright.** (1) ✅ rating-tagging shipped (`--ratings`, 94–98% seat coverage). (2) 🔴 **rating-weighted clone: 0.421 [0.400, 0.443] n=2000 vs v3 in the mirror ≈ −55 Elo** — not a null, a loss (§8t). (3) ✅ **covariate shift RULED OUT** — policy-vs-policy disagreement is 26.7% on our states vs 31.9% on theirs, near-symmetric, with a 1.7% positive control (§8s). (4) ⚡ **§8q's headline NARROWED by a much harder test**: agreement **peaks at 1050–1100 and falls in BOTH directions** (66.7% below 900, 59.9% at 1163) over 87 same-deck, same-week, **zero-exposure** demonstrators — so agreement measures **distance from the fitted mode, not skill**, and familiarity is refuted with a real control (§8r). (5) 🔴 **arm 2, the single-expert fine-tune, loses HARDER: 0.370 [0.349, 0.391] ≈ −92 Elo** — and it imitated *successfully* (ntumlnoob agreement 59.9% → **67.2%** held out). **⇒ B7 CLOSED.** The two failures are ordered by how far each net moved from the field (miss 30.2% → 32.0% → 36.2%, Elo 0 → −55 → −92): **agreement with the FIELD predicts strength; agreement with the EXPERT anti-predicts it** (§8u) |

**🔴 B1 INSTANCE 4 — RAN AND CLOSED 2026-08-05 (day 20). The defect is REAL and
this repair for it is a CLEAN NULL.** Arm A (mirror, direct, pooled 2 seeds)
**0.510 [0.470, 0.550]**; hypothesis arm vs `rule:v10` confirmed at n=2,000/cell
**+0.005 [−0.017, +0.027]**. Per rule 4 the out-of-vocabulary story is retracted
*for this intervention* — not falsified, since the two seeds disagreed in sign
at both sample sizes. **§8au's diagnosis is untouched; card attributes just do
not recover it.** `EVIDENCE` §8av. ⚡ **B1 has now measured +115 → +37 → +14 → 0
across four instances. The feature axis — the only one that ever paid — is
SPENT.** ⛔ Do not open instance 5 without a defect priced the way §8au priced
this one, and do not rebuild v6. 🟢 **Where §8au actually points: the corpus
contains ZERO Lucario games.** The untested lever is training data containing
the archetype — a data question, not an encoding one. ⛔ **And a method rule
that cost this experiment its screen: a two-cell anchor delta carries ±0.080 at
n=300. Screen on the mirror's DIRECT arm (2× tighter for the same games) or take
it to n≥2,000.**

**⛔ B1 INSTANCE 5 WAS OPENED ANYWAY ON 2026-08-06 (day 21) — E8 — AND IT IS THE
THIRD NULL IN A ROW. THE EMBEDDING COMPONENT IS CLOSED.** It cleared the bar the
line above sets: two defects priced before the build (**90% of every table ships
untrained and unseen cards read those rows as confident identities**; **row 0
overloaded across 25.5% of slot lookups with no `padding_idx`**), a fix that
verifiably fires (UNK hits 6/6 v10 Pokémon, 0/4 crustle, 0/6 mirror), a sizing
gate run first, and a same-session control. **Weighted −0.0078 (v7, 74% of the
field — published as −0.0099 and corrected day 22, §8ay) and −0.0047 (v7pad,
44%).** The one arm whose seeds agreed (`rule:v10`
+0.021) **cannot be attributed to the mechanism** — the pad-only net, which has
no UNK row at all, scored +0.034 on the same arm at seed 0. `EVIDENCE` §8aw.
⇒ **a real, measured, correctly-repaired defect is still not a lever.** ⚡ The
one non-null: **92% of the embedding parameters delete for free** (88,000 →
6,960, 11.5% of the net, −0.0018 corpus fit) — with §8w that bounds capacity
from **both** directions. 🔴 And the instrument got worse, not better: the
*direct mirror* arm swung **0.073** between seeds against ±0.036 sampling, so
**two seeds under-resolves every anchor we own, the mirror included.**

✅ **THE REPAIR IS RETAINED — user decision 2026-08-06, on correctness grounds
rather than Elo.** `train_policy.py --vocab` / `--pad` are permanent supported
machinery on `main`, guarded at both the loader (`policynet.load` refuses a
row-count/map mismatch) and at the two consumers that feed raw ids
(`context_accuracy.py`, `p54_emb_ablate.py`). ⛔ **This closes the axis as a
source of Elo; it does not deprecate the code.** 🟡 **Whether v7 SHIPS is an
open call, deliberately left to the user** — default is v5, because −0.0078 is
unresolved-but-wrong-signed and the ladder cannot adjudicate it. `EVIDENCE`
§8aw. ⚠ **Method changes this forces on everything downstream: budget ≥3 seeds
per arm** (2 measures two networks, not an intervention) **and remember a
two-cell delta's interval is √2× a single cell's** — our own driver printed the
single-cell width and understated resolution by 41%.

<details><summary>The original day-20 entry, kept for the report chapter</summary>

**⚡ B1 INSTANCE 4 — REOPENED 2026-08-04 (day 20), and this time the defect was
MEASURED before the build.** B1 was written down as "nearly spent, one variant
left". E6 (`EVIDENCE` §8au) found a different defect in the same family and
priced it without retraining anything: **permuting only the opponent's card-id
embedding rows on the frozen v5 net costs 0.838 → 0.587 against `rule:crustle`
and 0.625 → 0.607 against `rule:v10`.** Identifying the opponent's Pokémon is
worth ~a quarter of the win rate where the corpus supports it — and **all six
Mega Lucario Pokémon are out of vocabulary**, so that matchup is played blind by
construction, not by tuning. Only **8.0–10.4%** of the rows in three of the four
embedding tables ever received a gradient (`atk_emb`: **3.6%**).

This is the same lesson as §8f and §8y one level down: the information is
present, the *binding* is not — except here the binding is impossible in
principle, because a per-card embedding row cannot describe a card the corpus
never contained. **v6 (`--attr`) replaces identity-by-row with
identity-by-attribute** — energyType, weakness, ability, resistance,
weak-to-facing-type, plus a `cardType` one-hot on the option — all read from the
card DB, which covers all 1,267 cards. Sized first (`p55_attr_sizing.py`), and
the gate **killed `aceSpec` and `pokemonType`/`evolutionType` before they cost
anything**, which is the §8ab lesson applied in advance rather than after.
Pre-registered in `docs/experiments/embeddings/E7-card-attributes.md`;
prediction is an **asymmetry** (gain vs `v10` > gain vs `crustle`), and a
uniform gain falsifies the mechanism. ⚠ Support is thin — `weakness=5` appears
on one trained card — and the corpus has zero Lucario games, so a null is live
and "never trained on the matchup" remains a competing explanation.

</details>

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

### 🔴 B9/B10 — added and closed the same day (day 27, 2026-08-09)

Both came out of the user's question *"every eval says we match the top players,
so why are we not one?"*, whose answer is §8r: **every rate-vs-experts eval is a
conformity metric, and conformity to the mode is exactly a ~1000 rating.**

| # | candidate | verdict |
|---|---|---|
| **B9** | **Marginalise the bench-slot nuisance** the net provably reads (§8bt: 16.9% of decisions flip under a semantically-null relabelling, MAIN 24.8%) | 🔴 **NULL, pre-registered.** `bc:sym8` 0.513 [0.492, 0.535] vs 0.500, n=2,000, both controls holding (`sym1` bitwise identical, `sym8` firing on 8.36% of selects). §8bu. Code stays, off by default |
| **B10** | **"Imitation without a plan"** — condition the policy on a latent LINE and commit to it, attacking §8u's −92 Elo directly | 🔴 **Premise killed at the sizing gate, no net trained.** §8bv: plan clusters carry ≤ **+0.090 bits** about the next action beyond the board, against a **+0.372** estimator control with only two groups |

⚡ **The thread is NOT closed — one operationalisation of it is.** §8u's founding
datum stands (cloning the #2 player *successfully* cost 92 Elo, with covariate
shift ruled out by §8s), and **HANDOFF §N** carries the two hypotheses it cannot
yet separate plus three ranked probes. ⚠ The most valuable of those needs no
latent variable at all: **measure whether WE switch commitments mid-game more
often than the experts do.**

---

## 2.6 THE FINAL PUSH (day 25, 2026-08-07) — user-directed, pre-registered in `docs/experiments/E10-final-push.md`

**The user asked for a final push at the ~150-Elo gap before the freeze.** §2.5
is a graveyard — every B-candidate is closed — so this section starts from the
three measured facts that survive all of it:

1. **The opponent pool is a function of our own rating and the climb runs
   through the MIRROR** (§8ac: 33.3% at 955 → 51.1% above 900 → **71.4% above
   1000**). Rank 129 → top-20 means beating agents on our exact 60.
2. **A field clone converges to 0.500 in the mirror against field-modal play by
   construction**, and nothing at the top of this board is learned (§0) — the
   1145–1166 rule agents are doing something in our own matchup that the field
   mode does not. We hold 557 of their games ON our exact 60. That difference
   can be audited even though it cannot be cloned (§8u).
3. **The seed is a ±25 Elo pure-nuisance term sampled only four times** (§8bg),
   and true strength — not the displayed draw (§8ak: 63–87 noise floor) — is
   what continued play and the Round-2 BO3 reward.

| # | candidate | why it could pay | probe (cheap) | kill criterion |
|---|---|---|---|---|
| **F1** ⭐ | **Mirror-conditioned disagreement mining** on the ntumlnoob / Sixth Sense dumps | the only unexamined seam left: WHAT the 1150s do in the mirror. An **audit** in the §8f/§8y/§8ah lineage — the only methods that ever paid — not B7 cloning (trained on experts, −55/−92) and not an E3 teacher (⛔ closed). Output = a rule (rule 11: dominated only) or a §8au-priced defect, never a training target | sizing gate: ≥100 mirror games in the dumps; then large-margin disagreement extraction → cluster → size (≥0.5 firings/game) → discriminator classification → watch the top clusters | no cluster passes sizing AND classifies dominated → **closes as a chapter** ("the 1150s' mirror edge is tradeoffs — the class rules cannot repair"), which slots straight into `STRATEGY` §7b |
| **F2** | **Seed harvest, screen→confirm** | best-of-~12 draws from a σ≈25 nuisance ≈ +35–40 Elo over the median for zero new ideas; we shipped best-of-3 | screen `s4` + new seeds vs `v5_s1` (n=1,400 shipped-config, ~12 min each); **confirm the winner on FRESH games vs incumbent `s2`** | ship bar pre-registered: point ≥0.53 AND CI excluding 0.50 on the confirmation; below → keep `s2`, write the null. Screens NEVER ship |
| **F3** | **Corpus-coverage sizing gate** (the parked "last untested lever") | 40.7% of the field >3× under-represented in training; the one axis never probed | run `PARKED-corpus-coverage.md`'s probe: do our held replays contain the missing archetypes at all? (~30 min) | §8ac predicts a kill (blind archetypes are 0/47 above rating 900, and the mirror is OVER-represented 1.92×); verdict written either way, ⛔ no training this side of the freeze |
| ~~**F4**~~ | ~~one pre-registered β for B8~~ | the only honestly open door on a closed axis ("unfalsified rather than refuted", §8ao) | — | **DECLINED at planning time**: two nulls over 20,000 games, closed on the METHOD, hard stop spent, no integration runway by 08-15. A report line, not a run |

### ▶ OUTCOMES (day 25, 3rd session — three of the four items concluded in one sitting)

| # | verdict | the number that decided it |
|---|---|---|
| **F1** | 🔴 **CLOSED as a chapter — its pre-registered kill criterion is met** | gate passed hugely (257 mirror games, 22,665 expert decisions) and the extraction found 4,785 confident disagreements, 18.6/game. **The top cluster dissolved under an on-policy control**: the clone wants Munkidori's ability at 75.1% of offered decisions vs the experts' 38.5%, but **6.42 uses/game vs 6.23** when the shipped agent is made to play — identical behaviour, different *timing*, and sequencing is closed. The one ordering-free difference (Spikemuth Gym's search: the 1150s stop at turn ~9.7, we never stop) is a **tradeoff** ⇒ no rule (0/4). `EVIDENCE` §8bj, **HANDOFF rule 21** |
| **F2** | 🔴 **NULL — the harvest ships nothing, and it says why** | the selection debt is paid first: `s2` on fresh games is **0.510**, not the screen's 0.537. Then ten seeds of one recipe ⇒ between-seed sd **0.0190 ≈ 13.2 Elo** while the max−min reads **48.7** (§8bg's "50" was a *range*). The single screen winner `s7` (0.528 vs `s1`) then read **0.487 [0.468, 0.505]** vs the incumbent over 2,800 fresh games — **fails the pre-registered 0.53 bar.** ⚡ Why it *couldn't* work: the screen's error (0.0134) equals the effect's sd (0.0190), so the max of ten selects for measurement error; a real harvest needs ~5,100 games/screen for ≈+20 Elo on a ladder with a 63-point floor. `EVIDENCE` §8bh + §8bk |
| **F3** | 🔴 **KILLED on AVAILABILITY** | over the four dumps that built `pds_v4` (1,603 games, avg_score 1057–1223) **Archaludon and Mega Lucario appear in ZERO games**, and the miner discards nothing. The data does not exist at any band we can mine, and the mismatch is self-closing (corpus 56.9% mirror vs a field that is 71.4% mirror above 1000 — where we now play). `EVIDENCE` §8bi; `PARKED-corpus-coverage.md` marked CLOSED |
| ~~F4~~ | declined at planning time, unchanged | — |

**Discipline that binds all of it:** shipped config (`--no-rules`) in every cell;
byte-identical-net rule-toggled A/Bs where possible (the seed nuisance cancels);
otherwise ≥3 seeds; name what a submission evicts; if shipping, **submit twice,
last safe day ~08-15**; EVIDENCE §8bh+ entries the session each item concludes.

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
| **08-01** (day 15) ⚡ | 🔴 **THE ANCHOR WEIGHTS WERE MEASURING OUR OWN RATING.** The user supplied 8–9 h of replays for both live agents. Pooled (75 games) the field looks transformed since day 9 — **mirror 13.8% → 33.3% (Fisher p=0.002)**, Mega Lucario 12.8% → 4.0%, win rate 63.0% → 70.7% — but `p19_field_drift.py` buckets all 181 rated games from all four dumps by **opponent rating** and **every era difference vanishes once the band is held fixed** (all p ≥ 0.065). What moved is us: mean opponent rating **799 → 867**, tracking our climb 820 → 955. The mirror rises **5.3% → 18.6% → 42.4% → 71.4%** across bands while **Mega Lucario and Archaludon are 0 of 47 above rating 900**. ⇒ **the opponent pool is a function of our own rating, so every weighted verdict carried an invisible parameter — the score we held at census time.** Re-weighted (measurements untouched): §8i +35.6 → **+62.1**, §8z +23.4 → +24.8, §8aa +10.2 → +13.8, and 🔴 **§8j flips sign, +0.8 → −18.1** (its "worth nothing" rested on a −51 mirror term at 13.8% cancelling a +47 Lucario term at 12.8%; the real weights are 33.3% and 4.0%). ✅ **Nothing shipped changes — v5 already pins rules OFF**, verified by reading `main.py` out of the tarball. ⚡ **And §8b (52.1% of the ≥1144 band on our archetype) vs §8i (mirror 13.8%) was never a contradiction — one monotone curve, and we are walking up it.** ✅ **Item 4 built: `harness.Recorder`** — `visualize_data()` output is byte-compatible with Kaggle replays, so recorded local games are read unmodified by every replay tool here and watchable in `notebooks/visualizer.html`. 🔴 **Its first equivalence test could not have failed** (demanded identical games run-to-run; `battle_start` takes no seed) — rewritten around four exact per-game checks, 12/12 | `EVIDENCE` §8ac + §8ad; `p19_field_drift.py`, `p20_record_games.py`, `p20_recorder_equivalence.py`, `harness.Recorder`, `out/lb_snapshot_0801pm.json`; **`STRATEGY.md` §4f** | ⚡ **rank 185 / 6,103 at 955.1, our best ever, and ✅ rule 2 satisfied on a net pair for the first time** — two readings 61 min apart agree (v5 955.1 → 955.1, v4 914.9 → 914.6), both converged and both active, **v5 +40.5**. 🔴 **Still does not adjudicate §8aa**: the arena put v5 at +14 Elo (+13.8 re-weighted) and rule 2's second clause says the LB cannot resolve that size at all. ⛔ **Track C's Archaludon lead and B3's Lucario instance CLOSED BY SIZING before either was built** — 8.0% and 4.0% of the field, both 0/47 above rating 900 |
| **08-02** (day 16) 🔴 | **AN ANCHOR WAS THROWING GAMES, AND A HUMAN FOUND IT BY WATCHING.** Day-15 item 6 (the anchors carry 71.5% of every weighted verdict and had never been watched) paid on its first use: the user watched `anchor_vs_anchor/game000` and saw the Crustle pilot decline to bench **Mega Kangaskhan ex** with an **empty bench**, then lose to the next KO. `sources/crustle.py:338` scored every Pokémon but Dwebble at **−5000** with **no empty-bench guard**. ✅ Fixed (empty bench → 90000); **exposed turn-ends 0.667/game → 0.000**, declines 6 → 0. `scripts/p24_anchor_pathology.py` clears the other four anchors **and our own net (0 of 51 games)**. 🔴 **Every verdict with a Crustle term is suspect** — arena 0.663 vs a 57.1% real WR — **re-run NOT yet authorised**. ⚡ **The methods half is about our own script: the obvious detector OVERCOUNTS** (benching later in the same turn still counts), and on it `rule:archaludon` looked worse than Crustle and **is clean** — fourth confident-but-wrong script in three days, **first one caught before reporting**. ⛔ **The rule it suggested for our own agent died by sizing** (0.187 firings/game, 1 of 22 losses, vs Morgrem's 0.2 and Pokégear's 0.27). 📋 Also recorded at last: **`55169114` is decision-identical to v5** (only `main.py` + `bcagent.py` differ, by counters and one `print`) and reads **≈80 points below it** — a candidate first measured LB null, ⚠ pending a rule-2 settling read. ✅ **Submission-log track CLOSED**: pool usage settled (**1.12 s of 1,800 s**), net proven live by timing (1.2–1.6 ms vs 23–37 µs), **fresh process per episode** so cumulative counters can never print, and nothing left worth a slot | `EVIDENCE` §8ah + §8ai; `scripts/p24_anchor_pathology.py`, `out/replays/audit_*`; **`STRATEGY.md` §4f WRITTEN** (it had been claimed in this table and did not exist) **and §6's stale day-9 anchor shares corrected** to the §8ac band-aware table plus the anchor defect | ✅ gate met: one real defect found, fixed and verified before/after; one candidate rule killed by sizing before any code; two report chapters |
| **08-02** (day 16, 2nd half) 🃏 | ✅ **TRACK C GOT ITS REAL ANSWER, and it took four A/Bs instead of one.** `p25_deck_slot_audit.py` sized **all 60 slots** over 75 real games (§8aj) and produced two findings before any deck was built: ⚡ **deck swaps PASS the sizing gate that killed three rules this week** — the relevant frequency is the **draw** rate, not the play rate (Tool Scrapper: 0.13 plays/game but **drawn in 81%**) — and 🔴 **the obvious cut is disqualified by the MATCHUP**, since Tool Scrapper is played **0.00 times per mirror game** (our list runs no tools), so a mirror A/B would return "cutting it is free" by construction — **rule 16 in deck clothing.** Then three variants against a measured same-deck control of **0.4980 [0.483, 0.513]**: 1 swap **0.4911** (null) · 2 swaps **0.4757** (**−17 Elo**, p=0.021) · 4 swaps **0.4637** (**−25 Elo**, p=0.0004). 🔴 **Strength falls monotonically with distance from the consensus 60.** ✅ **And the charitable defence is ruled out** — 6 recorded games show Budew reaching the Active spot **3/6** and Itchy Pollen firing **4×**, so the mechanism works and the plan still loses. ⇒ **The consensus 60 is a local optimum and our net is tuned to it; measured, and we kept the list** | `EVIDENCE` §8aj + §8al; `p25_deck_slot_audit.py`, `decks/grimmsnarl_g4.py`, `grimmsnarl_budew.py`, `grimmsnarl_budew_v2.py`, `out/arena/deck_*.jsonl`, `out/replays/budew_v2_watch`; **`STRATEGY.md` §7c** | ✅ gate met: Track C went from ONE decklist A/B ever to a sized 60, a per-matchup split, four controlled tests and a same-deck variance floor. ⛔ **The guess-a-swap method is retired** — the next deck programme needs a matchup-stratified search design |
| **08-02** (day 17) 🔴 | ⚡ **THE LADDER'S NOISE FLOOR IS 63.2 POINTS AND IT RE-PRICES WHICH INSTRUMENT ANY CLAIM MAY REST ON.** `55169114` is decision-identical to v5 and a fourth read closed §8ak's withheld verdict: **942.7 vs 879.5, gap 63.2**, both settled by the same test. A true difference of **exactly zero** displayed as 63 points — larger than the day-15 +40.5 headline, §8z's +37, §8ab's −36, §8aa's +14. ⇒ **rule 2's second clause stops being an inference: the LB cannot adjudicate any net change we make, and the arena is not the weaker instrument but the only one.** 📈 **Rank 198 / 6,136 at 942.7 — best rank ever, and the score FELL while the rank rose.** 🔴 **B8 RAN AND IS CLOSED, twice.** τ was sized first (**§8am**: 20.4% of selects randomisable at zero cost, then a CLIFF — 30.5% costs −135 Elo; ⛔ **the CLIFF is retracted by §8bd day 23 — temperature moves deviation rate AND depth together, and at fixed one-rank depth there is no cliff.** The 20% free band survives as a measurement). Then **4,000 games → 0.512 [0.491, 0.534]** and **16,000 games → 0.506 [0.484, 0.528]** against a pre-registered **0.541**; each control sat on v5, so the procedure is harmless; and a **decision rule committed before the second A/B** sent it to *close on the method, not the budget* when 4× the data moved the estimate down. ✅ **CRUSTLE RE-RUN AUTHORISED AND DONE (§8an): the alarm is retired.** All three nets score **+0.087…+0.102** higher against the repaired pilot, every CI disjoint and the sign the OPPOSITE of §8ah's expectation — but it is a **level** shift, so it cancels in the net-vs-net differences every weighted verdict is built from (+0.013 and +0.003 at 6.7% weight) and the net ordering is unchanged. **Nothing needed rewriting.** ⚡ Decomposed to the **empty-bench guard alone**; the unauthorised bench-anything default contributed **+0.011 / −0.013 / −0.004** — sign-flipping noise. ✅ **BOTH MISSING ANCHORS CLOSED (§8ap)** — `rule:dragapult` already existed and had **never been used** (0.809), Garchomp was **built** from our own meta snapshot (0.857). 🔴 **And measuring them found the set's resolution runs INVERSE to its representativeness**: the only anchors near 0.5 are the two §8ac put at 0/47 above rating 900, 40.7% of the weighted verdict now sits above 0.75, and **the mirror is the only anchor that is both** | `EVIDENCE` §8ak (verdict) + §8am + §8an + §8ao + §8ap; `p26_selfplay_gen.py` (+`--keep-margin`), `train_policy.py --advantage/--anchor-ds/--margin-max/--freeze-except/--export-last`, `decks/cynthia_garchomp.py`, `out/arena/p27..p32_*`; **`STRATEGY.md` §5.1b**; **HANDOFF rule 18** | ✅ gate met in the honest sense: the last un-killed LB axis died with two numbers and a pre-registered rule, the anchor set is complete, and three of the session's own errors are corrected in the sections that made them |
| **08-02** (day 18) 🃏 | ⚡ **TRACK C'S DESIGN IS PRICED, AND ITS OWN INSTRUMENT NARROWED IT FROM 60 SLOTS TO 2.** §8ap looked like a blocker (*"near ceiling ⇒ cannot resolve"*); `p33_anchor_resolution.py` shows that is true in **Elo** units and false in the units a deck decision uses — `W = Σ wᵢpᵢ` is linear in **win rate**, noise *falls* near the ceiling, and the worst anchor costs **2.04×** the games for equal Elo resolution rather than infinity. Full stratified design: **±0.0050 on W for 57,600 games ≈ 1.1 h** (Neyman allocation with costs, 55% of naive equal-n). 🔴 **But the case for stratifying is BIAS, not precision** — the same games spent mirror-only measure Δ to **±0.0041**, *tighter* — so it is worth its cost only where the mirror is biased, which nothing here could measure. ✅ **Built it** (`p34_matchup_liveness.py`, 400 games × 7 anchors, agent-wrapped so rule 18's seat bug cannot arise) and it **recovered §8al's Tool Scrapper fact unprompted** as a positive control. 🔴 **The finding: 17 of 19 slots are mirror-SAFE (spec ≤ 0.16), so §8al's example was the deck's single most extreme card and the critique its plan rested on is far narrower than assumed.** ⚡ **The one slot that matters is the one this file already named — Froslass** (Track C step 4, "the only growable passive-damage line"): **1.44** plays/game in the mirror against **5.57 / 6.83** vs alakazam5 and Crustle, so a mirror A/B sees under a quarter of its use. ⚡ **And the bias runs both ways** — Munkidori 18.6 mirror vs 11.3–14.2 elsewhere, so mirror-only **overstates the core engine** too. 🔴 **Separately, an ANCHOR HAD DRIFTED**: `rule:crustle` is a **fourth** pilot committed **26 min after** its last measurement and verified on **n=6**; at n=2,000 it is **0.755 [0.735, 0.773]**, not §8ap's 0.866, and the one-line Dwebble tie-break inside the guard is worth **0.111** — larger than the entire empty-bench repair, in the opposite direction. **⇒ ~~WHICH Pokémon a pilot benches matters more than WHETHER it benches~~ — 🔴 RETRACTED day 22 (§8ax): the 0.866 and the 0.755 are two different DECKS, 20 of 60 slots apart and worth +0.140. Same-deck the tie-break is +0.027 and the guard −0.038, both small. The rule-19 method finding stands; the pilot attribution does not.** ⛔ **No verdict retracted** (both nets faced the same pilot; the shift is a level shift) | `EVIDENCE` §8aq + §8ar; `p33_anchor_resolution.py`, `p34_matchup_liveness.py`, `out/arena/p35_v5_vs_crustle_v4.jsonl`, `out/logs/p34_liveness.{txt,json}`; **HANDOFF rule 19** | ⚠ **gate deliberately NOT met: no deck variant built or A/B'd.** The two-stage design (mirror screen → single pre-registered stratified confirmation) is **with the user for review**, because k variants at α=0.05 manufactures a winner at k≈20 and that is the shopping the B8 β-sweep was declined for |
| **08-02 – 08-08** 🤖 | **B8 — THE RL FINE-TUNE, AND IT IS THE PRIORITY.** Scheduled at last: it has been "alive but unscheduled" since §8ae passed its sizing probe on day 15 — decided, never dated, which is the exact drift this file's own doc-discipline audit describes. **Smallest version only**: fine-tune a small parameter set on our own recorded outcomes with the corpus loss as an anchor, A/B at n≥2000 vs a byte-identical control. 🔴 **Kill line written before any code (see §2.5 B8); hard stop 08-08** — after that the calendar is consolidate-then-freeze and a winner could not be integrated. ⛔ **Deck work is PARKED, not cancelled**: §8al closed the *guess-a-swap* method (three user-proposed edits → one null, two significant losses, monotone worse with distance from the consensus 60), and the next deck programme needs a **matchup-stratified search design** — all four A/Bs so far were mirror-only | **the deck chapter is WRITTEN, not owed** — `STRATEGY.md` §7c; B8 is a chapter either way, and a retracted negative re-derived honestly is §5's material | B8 clears its pre-registered bar → integrate and submit by ~08-15, **or** it dies with a number and a written verdict |
| **08-01 – 08-08** | ⚡ **re-aimed by §8ac: the MIRROR is 33.3% of the field and 51.1% above rating 900** — deck and play work there outranks every counter-meta branch. Build anchors for **Cynthia's Garchomp (6.7%) and Dragapult (5.3%)**, which now outrank Crustle + Lucario combined and have none; ~~B3 archetype detector~~ ~~Track C Crustle/Archaludon tech~~ **demoted by sizing**; follow up whichever of B1/B4 survived; matchup win-rate matrix | log every concluded experiment same-session; deck sweep vs the **re-weighted** anchors | ≥1 breakthrough candidate validated at n≥2000 **OR** all five killed with written verdicts |
| **08-06 – 08-07** (days 22–23) ✅ | 🔬 **THE INSTRUMENT WAS AUDITED AND THEN AN EXPERIMENT RETRACTED ANOTHER ONE.** Day 22's audit of the validation flow found **seven** defects with no result prompting it (`EVIDENCE` §8ax/§8ay/§8bc); **two changed something published, five could not have** — and the reason is that a same-session back-to-back control makes any *level* shift cancel in a difference. Day 23 then ran the **teacher-free E3 gate** (§8bd, pre-registered in `675d09c` before any arm): flipping the clone's *k*-th choice for its (*k*+1)-th across the whole near-tie band reads **0.494 [0.467, 0.520]**, n=1,400, same weight file both sides — **the band E3 wants relabelled is indifferent**, and E3 is *unresolved for want of a teacher*, not refuted (the flip measures \|E[effect]\|, a teacher's value is E[\|effect\|]). 🔴 **The bigger result: two arms MISSED their pre-registered predictions and the misses retract §8am's "cliff".** A softmax temperature raises deviation **rate and depth together**; at fixed one-rank depth there is no cliff at all (0.495 → 0.494 → 0.487 → 0.455 → 0.356, monotone 5/5), and at ~half of decisions deviated the two instruments read **0.356 vs 0.055**, disjoint. **Third instance in three sessions of crediting an effect to the variable we named** — and the first caught by a designed experiment rather than an audit | `EVIDENCE` §8bc + §8bd (+ the §8ak addendum); `p59_e3_flip.py`, `p43 --dump-margins`, `bcagent.flip_margin`; **`STRATEGY` §5.7 (the audit as a chapter), §5.8 (the variable-we-named failure), a B8 chapter that did not exist, an E3 entry, §2's table corrected**; day-22's −0.0099 → −0.0078 correction propagated to the **six** files that had kept the old number | ✅ gate met: one pre-registered experiment ran and answered its question, one published finding was retracted on evidence, and the report gained five sections. ⛔ Nothing submitted |
| **08-08 – 08-14** | 🔴 **RE-REVISED day 25 (2nd session) — THE WINDOW IS THE FINAL PUSH (§2.6), superseding day 23's "report only".** Two premises of that verdict expired: the user re-opened Track A with a final-push directive, and §8bg then measured the seed as a **±25 Elo nuisance sampled only four times** — so "nothing left to integrate" was true of *ideas*, not of *draws from a distribution we own*. ⚠ Its eviction warning is also stale: `55160229` (990.7) was already evicted on 08-07; the next submission evicts `55321893` (ens2, 934.7). The window, per `E10-final-push.md`: **08-08** F3 gate + F2 screens (s4, s2 fresh-game confirmation) + F1 sizing gate + rule-2 reads on `55326513` · **08-09–08-11** F1 mining, F2 seed training/screens in background · **08-12–08-14** any F1 rule A/B'd, F2 confirmation, ship decision · **08-15** last safe submission, twice if shipping | E10 (frozen pre-registration); EVIDENCE §8bh+ same-session; `STRATEGY` §7b.4 (the one user-authorised edit) states the mirror thesis for the report | a shipped improvement **confirmed on fresh games**, OR three written kills (F1/F2/F3) that close the project's last open questions before the freeze — either outcome is a report chapter |
| **08-08 – 08-09** (days 26–27) 🔴 | **FOUR USER-NAMED SEAMS MEASURED, FOUR KILLS, NO ARENA TIME SPENT ON ANY OF THEM — and then two designed experiments, both null.** Days 26: passive-damage targeting (§8bm, 0.09/0.20 per game), KO-setup (§8bp, 0.04 — and the *mechanism* measured FALSE for the experts too), Petrel's fetch (§8br, 0.29 ceiling). Day 27: **the WP-regret autopsy** (§8bs) — no blunder signature in the 27 losses (our worst decision −0.069 vs −0.070 in our wins vs −0.078 for the players who beat us; 0.039 events/game at \|ΔWP\|≥0.20, **13× under the gate**) and, more usefully, **a realized trajectory is PROVABLY BLIND to errors of omission** (§8bm's seven known dominated plays score **+0.002…+0.005** and rank mid-pack). Then **R2/E15** — averaging out the bench-slot nuisance the net demonstrably reads (§8bt: 16.9% of decisions flip under a null relabelling) — **0.513 [0.492, 0.535] vs a pre-registered 0.500, NULL**; and **R1 killed at its sizing gate before a net was trained** (§8bv: plan clusters carry ≤+0.090 bits against a **+0.372** estimator control). ⚡ **By-product, and the most reusable fact of the two days: conditioned on the board, WINNERS AND LOSERS PLAY THE SAME** (−0.0024 bits), which retires outcome-conditioned cloning before it is built. ✅ **Submitted `55382430`**, a byte-identical `v5_s2` duplicate, so both active slots are now our best agent (max-of-two-draws, §8ak) | `EVIDENCE` §8bs/§8bt/§8bu/§8bv; `p77_wp_regret.py`, `p78_symmetry_probe.py`, `p79_plan_audit.py`, `agents/sa/symavg.py`, `docs/experiments/E15-symmetry-averaging.md`; **HANDOFF §N is the next session's entry point** | ✅ gate met in the honest sense: **six written kills**, two of them pre-registered, and three instrument defects found and repaired (`evalfn` undefined during setup; a diverging IRLS; a stale `inPlayIndex` in my own permutation). ⛔ Nothing shipped — no candidate cleared its bar |
| **08-09** (day 27, 3rd session) ✅ | ⚡ **THE PROJECT GOT A NEW KIND OF INSTRUMENT, and the feasibility gate that guarded it passed 60/60.** Every eval we own is a **conformity** metric (§8r) or a weak evaluator (`evalfn`); neither can ask *"in THIS position, is their move better than ours?"*. `p80_rollout_feasibility.py` shows a real position **forks out of a replay** — an sbi captured in another process reconstructs it exactly, option list bitwise identical, expert seats 32/32 — and its options can be scored by rolling the clone out to terminal at **101 ms** a rollout. Positive control: the clone's own **top vs last** option reads **+0.120 [+0.052, +0.189]**, so it resolves, and that is the scale bar. 🔴 **Three of the design's own premises died in the process:** CRN is unavailable (the engine draws its own shuffles — a shared world is the only pairing, worth ρ≈0.53); "per-decision resolution is unaffordable" is **false**; and the fork **silently accepts a decklist the seat is not playing** (Crustle's 60 on a Grimmsnarl seat read 0.975, identical to the correct deck) ⇒ **mirror only**. 🔴 **Plus a defect in my own estimator caught by replication** — three runs of one cell read +0.130/+0.107/+0.120 against a nominal ±0.017, because pairs are **clustered inside positions**; clustering widens it 4.1× and all three then agree | `EVIDENCE` §8bw; `scripts/p80_rollout_feasibility.py`, `out/logs/p80_rollout_feasibility.txt`; **`docs/experiments/E16-counterfactual-move-value.md` pre-registered** (agreement control + a difference-in-differences arm on `policy_b7_ntum` that separates §N.3's H1 from H2) | ⚠ **gate deliberately NOT met: nothing was measured and E16 has NOT run.** This was the feasibility step §N.4.0 demanded first, and E16 awaits a user go/no-go on spending the pre-freeze days on it |
| **08-14 – 08-17** | **freeze** — no risky submissions (latest-2 eviction trap) | — | sim track locked with the settled pair active |
| **08-17 – 08-31** (continued play) | watch the final pair's rating — **this IS the rubric's "consistency under repeated matches" evidence**; record the trajectory | draft chapters 1–4 | continued-play data captured |
| **09-01 – 09-14** | — | chapters 5–8, figures, full review pass, format for Kaggle, submit | report submitted well before 09-14 |

**Standing constraints:** 5 submissions/day, **latest 2 active** — every
submission near the deadline risks evicting the best agent (HANDOFF §7 ⛔). Score
convergence takes 4+ h, so the *last safe day* for a new agent is ~08-15, and
earlier is better because continued play rewards a settled high-μ pair.

---

## 3.5 SECOND ROUND — announced 2026-08-03 (day 20)

**Source: competition organisers, relayed by the user. Recorded verbatim first,
consequences second, and nothing here is measured yet.**

### The announcement

- **Qualification is by the STRATEGY CATEGORY result.** The top eight teams after
  First Round's Strategy Category advance; withdrawals are backfilled by rank.
  A team must enter Strategy with the **same team** as Simulation.
- **Simulation rank is "taken into consideration"** but is explicitly one input
  among several — evaluation is "holistic", naming **deck construction,
  originality of the proposed approach, and quality of the explanations in the
  report.** Criteria page:
  `kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/overview/evaluation`
- **Second Round is an in-person event in Tokyo**, a tournament among the eight.
- **Format changes: best-of-three.** Same deck and same code for the whole match;
  first to two games wins. **Games are played SEQUENTIALLY, and a team may read
  the Game-1 log during Game 2 and the Game-1+2 logs during Game 3.**
- **Resources: AWS p5.4xlarge or equivalent — 1× NVIDIA H100 80 GB, 256 GiB RAM,
  16 vCPUs.**
- ⏱ **Clock (announced same day): 30 minutes of TOTAL THINKING TIME PER GAME,
  per team. Separate per game, NOT carried over between games of a BO3. Running
  out loses that game.**
- 🃏 **Card pool (announced same day): every First Round card remains legal, and
  the pool is EXPANDED. The additional cards are announced only AFTER the
  Simulation Category final rankings are confirmed** — i.e. after 08-17.

### What it changes, largest first

**1. 🔴 The dossier is no longer 30% of a score — it is the gate.** §4's
tie-break rule ("the dossier does not get sacrificed for marginal Elo") was
argued from rubric weights; it is now a structural fact about advancing. The
standing **one-edit-per-session rule on `STRATEGY.md` is a qualification
requirement**, and the seven owed chapters (§8am–§8as, plus the deck search) are
the backlog that matters most in the repo. ⚠ **This does not license abandoning
the ladder** — sim rank is an input, and a top-200 finish is part of the
holistic case. It re-orders, it does not cancel.

**2. ⚡ The BO3 log channel is NEW INFORMATION AT DECISION TIME — which is the
only axis that has ever paid here.** Every win this project has had was
representational or informational: §8f (`opt["index"]` was never encoded, +115
Elo), §8z (the state block, +37). Every *more-of-the-same* axis measured null or
negative — capacity §8w, data §1, demonstrators §8t/§8u, RL §8ao. **Game-2 and
Game-3 log access is a strictly new input the agent has never had**, so it is the
same shape as the two things that worked. ⚠ **It is not a compute lever and must
not be filed as one.**

**3. ⚡ The opponent field collapses from 6,000 anonymous entrants to 7 named
teams.** ROADMAP already calls **targeted per-team replay dumps the best data
source we have** (§0's day-10 amendment) — not band-censored, because we name the
demonstrator rather than sampling a rating band. In Round 2 the entire field is
nameable. ⚠ **Read them, do not clone them** — §8u measured that cloning a
1163-rated expert cost **−92 Elo** and that agreement with an expert
*anti*-predicts strength.

**4. ⚠ THE HARDWARE DOES NOT RESURRECT WHAT DIED ON MEASUREMENT.** 16 vCPUs and
an H100 make several closed axes *possible* again; that is not the same as
*indicated*. Sorted by whether the cause of death was the clock:

| closed axis | cause of death | does R2 hardware change it? |
|---|---|---|
| capacity (§8w) | 8.2× params bought **−43** decisions of 12,939 | ⛔ **No.** Measured on the features, not the clock |
| more data (§1), demonstrator selection (§8t/§8u) | monotone worse with distance from the field | ⛔ **No** |
| deck guess-a-swap (§8al), the 11-variant search (§8ar/§8as) | all ≤ 0 against a same-deck control | ⛔ **No** |
| RL fine-tune (§8ao) | pre-registered rule; **4× the data moved the estimate DOWN** | ⛔ **Closed on the METHOD, not the budget** — that is the exact wording of the decision rule. ⚠ The one honest opening is the **untested β**, named as unfalsified rather than refuted |
| game-tree search (§2) | ours 0.323, n=31, rollout SE≈0.14 | ⚠ **Partly.** Variance was the killer and n is affordable now; but V10's MCTS never executing while holding 950+ is not a compute finding |
| **within-turn sequencing, B4 (§8v)** | −89 Elo vs a 1 ms clone | 🔴 **YES, and it is the only clean case.** The controlled arm measured **extra time ≈ +154 Elo** on its own, and the prototype spent **~12 s/game — 2% of even the 600 s First Round pool.** R2 gives **3× the clock × 8× the cores + a GPU**, i.e. **150× what B4 actually used.** **A re-probe candidate, not a revival** — it was −89 with the design fix already applied |

**5. ⏱ THE CLOCK TRIPLES, AND IT COMPOUNDS WITH THE CORES: 600 s → 1,800 s PER
GAME, PER PLAYER.**

> ✅ **User-confirmed 2026-08-03: in the FIRST ROUND format, 600 s is the total
> game clock for each player.** So this repo's **600 s** — carried in ten places
> including `STRATEGY.md`'s opening constraint sentence — **is CORRECT and stands.
> Nothing needs sweeping.**
>
> ⚠ **An earlier version of this box claimed the opposite** (that 600 s was 3×
> wrong and the real budget was 1,800 s) **and it was itself the error it accused
> the repo of.** Retracted in place, same session, per §2's rule.
> 🔴 **What remains genuinely unexplained is HANDOFF:758's "1.12 s of a 1,800 s
> budget", read off a real Kaggle episode log.** The format rule is authoritative
> and the log line is not overridden by being unexplained — **re-check what field
> that 1,800 s came from** (per-episode harness value? summed across seats?) and
> record the answer. ⛔ **Until then, do not quote 1,800 s as a First Round number.**

⇒ **Round 2 adds ~3× the wall clock AND ~8× the vCPUs AND a GPU.** These
multiply on any throughput-bound method: **~24× the CPU work per decision** before
counting batched net evaluation on the H100.

**Budget arithmetic, at ~318 selects/game (day-16 measurement):**

| | per-game clock | per decision | vs our clone's 1.2–1.6 ms call |
|---|---|---|---|
| **First Round** | 600 s | **≈ 1.9 s** | ~1,300× headroom, on 2 vCPU |
| **Second Round** | **1,800 s** | **≈ 5.7 s** | ~3,800× headroom, on **16 vCPU + H100** |

**We used 1.12 s of the 600 s. The headroom was already enormous and we never
spent it** — which is why the R2 clock is an opportunity and not an excuse.

🔴 **NEW FAILURE MODE WE HAVE NEVER BEEN NEAR: timeout = loss, per game, with no
carry-over.** HANDOFF §7 notes Kaggle enforces the pool, but at 1.12 s of 600 s
the risk was structurally zero. **The moment any agent here actually spends the
budget, a hard-guarantee time manager becomes a correctness requirement** — and
in a single in-person tournament, one game lost on clock is unrecoverable.

⚡ **AND THIS RE-PRICES B4'S DEATH — as a named confound, NOT a revival, and the
gap is far larger than first stated.** §8v's prototype used **~12 s/game of
planning** (`EVIDENCE` §8v, "~40× the clone, well inside the 600 s pool") — that
is **2% of even the First Round budget**. It lost by ≈ 89 Elo there, while its
own controlled arm measured the extra time at **≈ +154 Elo**.
**The Second Round budget is 1,800 s/game — 150× what the prototype actually
spent — on 8× the cores with GPU-batchable net evals.**
⚠ **This does not reopen B4**: the design fix was already applied when it lost,
and a confound is not a result. It means the honest statement is *"B4 was never
run within two orders of magnitude of the budget it would have, and we cannot
close that gap locally"* — n=200 × ~300 selects × 5.7 s ≈ **95 h per arm** on one
core.
⇒ **The scaling-curve probe is the only affordable answer**: fit time→strength at
3–4 budget points and extrapolate, because extrapolation is cheap and measurement
is not.

**6. 🃏 THE EXPANDED CARD POOL IS A DECK-SCORE OPPORTUNITY AND A NET HAZARD, and
§8af already sized the hazard exactly.** The corpus holds **exactly 134 distinct
card ids**; **1,166 of the 1,300 embedding rows are still random init.** **Every
new card is, by construction, outside the 134** — maximally off-distribution.
§8ar/§8as then measured that **card-level exposure is necessary but not
sufficient**: Ultra Ball at **5.59×** our weakest card's exposure lost **all six**
slots it was tried in, and Energy Switch at 3.61× was played **1 time in 28
offers**. ⇒ **Our clone cannot pilot new cards, and no amount of H100 fixes that,
because there is no corpus for them to be cloned from.**

🔴 **AND WE CANNOT OPT OUT, WHICH IS THE PART THAT MATTERS: the opponent's cards
enter OUR observation.** Declining to play new cards protects our own decklist
and protects nothing else — a finalist who plays three new cards puts three
unseen ids on the board that `featurize()` must encode, and (⚠ **VERIFY THIS
FIRST — it is inferred from §8af, not measured**) they land on random-init
embedding rows. **So the hazard is not a deck choice we control; it is a property
of the match.** ⇒ **A cheap probe exists and should run before any posture is
chosen: inject unseen card ids into a live observation and measure what the net
does** — degrade gracefully, or produce garbage logits.

⚠ **The asymmetry is not in our favour and should be stated plainly: OUR
ARCHITECTURE IS THE ONE A POOL EXPANSION DAMAGES MOST.** §0 records that *nothing
at the top of this board is learned* — a rule- or search-based finalist absorbs
new cards at the cost of reading them, while a clone has no demonstrations of
them at any price.

⇒ **Two survivable postures, and both are report material:**
- **(a) Decline the new cards, with the exposure measurement as the reason.**
  "We measured why our architecture cannot absorb an unseen card and chose the
  known-good 60" is an evidenced Deck Score position, not a punt.
- **(b) Play them and have the RULE LAYER own every decision they touch.** ⚡
  **This is the strongest argument the clone+rule-repair architecture has ever
  had** — rules generalise to unseen cards and clones structurally cannot.

⇒ **B6 (`deckfacts.py`) IS PROMOTED, and its parking condition is met.** It was
parked as *"do it WITH the first decklist change"*; a pool expansion where rules
must own new cards **is** that condition. Hardcoded ids — `MUNKIDORI`,
`DARK_ENERGY`, `BOSS_ORDERS`, **`WALL_POKEMON = {345}`** — do not survive a pool
change, and computing facts from decklist + card db is exactly the generalisation
a new pool demands.

⚠ **It also DEGRADES the seven-opponent anchor idea:** the finalists' First Round
lists will not be their Second Round lists. **Read their replays for style and
play patterns, not for decklists.**
⚠ **And the engine must implement the new cards, so every anchor and pilot needs
re-validation after the pool update** — rule 19 (an anchor drifted and every doc
quoted the old number) at seven-anchor scale.

⚡ **The one genuinely re-opened question: §8al/§8as's "the consensus 60 is a
local optimum, and strength falls monotonically with distance from it" is a fact
about the FIRST ROUND pool and meta.** A new pool can move that optimum, and the
two-stage pre-registered search (§8ar/§8as) is already built and would be
answering a genuinely new question rather than re-asking a settled one.
⏰ **But the pool drops after 08-17 and the report is due 09-14, so this work
lands inside the report window.** Schedule it as a conflict, not a freebie.

**7. ⚠ Cross-game memory must travel through the LOG, because the process does
not persist.** Day 16 measured that **Kaggle starts a fresh process per episode**
(all three submission logs read `calls=1`, which is why the cumulative counter
line could never print). The announcement's phrasing — teams *access the log* —
is consistent with that. **Plan for parse-the-log, not for in-memory state
across games.**

### Unknowns that gate any Round-2 plan — confirm before building on them

1. ✅ **ANSWERED 2026-08-03: 1,800 s per game per player, no carryover,
   timeout = loss — against First Round's 600 s. The clock triples.** What
   remains open is narrower and is worth **a 16× swing** in any search sizing:
   **is the pool wall-clock or CPU-time?** On 16 vCPUs a wall-clock pool lets a
   parallel search spend 16 core-seconds per wall-second; a CPU-time pool does
   not. ⚠ Also unconfirmed: whether the H100 counts against it at all.
   🔴 And re-check what HANDOFF:758's stray **1,800 s** log field actually was.
2. **Whether the agent process can reach the GPU at all** — image, drivers,
   frameworks, and whether model weights ship inside the submission bundle.
3. **The exact log format and access mechanism** in Game 2/3, and whether it is
   the same `visualize_data()` shape `harness.Recorder` already emits
   byte-compatibly (§8ad). If it is, our whole replay toolchain reads it unmodified.
4. **Whether the other seven finalists' First-Round replays are obtainable** —
   this is the difference between 7 real anchors and 7 guesses.
5. **The Round-2 date**, which decides whether R2 engineering competes with the
   **09-14** report deadline or follows it. ⚠ **The card pool already does** — it
   drops after 08-17 and the report is due 09-14.
6. 🃏 **The additional cards themselves** — count, power level, and whether they
   change the archetype hierarchy or only add tech slots. **Announced after the
   Simulation rankings confirm**, so it cannot be pulled forward.

### Standing position until those are answered

**Round-2 engineering is CONTINGENT — the report is not.** The report is what
qualifies us, and a written Round-2 design section is *itself* report material
under "originality of the proposed approach". ⇒ **Design R2 on paper now, write
it into `STRATEGY.md`, and build only what a First-Round deadline already
justifies.** ⏳ **Ranked R2 candidates and their probes are being brainstormed
with the user (day 20) and are NOT yet committed** — nothing below the line above
has been sized, and rule 14 (size before you build) applies to all of it.

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
