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
   holds LB 950+ anyway; self-play RL, more data and more val accuracy all
   negative; nothing at the top of the board is learned. 18 days on ~1.4 cores
   cannot change those results.
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

---

## 1. The narrative we are building toward

> **"Clone the field, then audit the clone's blindness."** We behavior-cloned
> 2,810 human games, systematically enumerated the decisions its features cannot
> express (no HP, no damage, no attached energy), and repaired them with
> arithmetic rules. We present a falsifiable discriminator — **rules that delete
> a *dominated* option win (3/3); rules that pick a side in a *tradeoff* lose
> (0/4)** — validated by arena A/B at n≥2000 every time. Plus a
> measurement-discipline codex (HANDOFF §2, 13 rules, each paid for) and
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
4. **Measurement discipline** — HANDOFF §2's 13 rules as a methods section.
5. **Robustness & consistency** — multi-anchor A/B tables, seat balance, n and CI
   everywhere, and the `counter_source` false-positive post-mortem (§3.0). **A
   documented failure of our own validation process, diagnosed and fixed, is
   rubric gold — not embarrassment.**
6. **Opponent modeling / meta adaptation** — the measured meta shift, archetype
   detection, the Crustle case study.
7. **Deck concept** (feeds Deck Score — Track C).
8. **Negative results** — search, self-play, data scaling, Boss's Orders ×4,
   decklist variants. Honest nulls at n≥2000 are rare on Kaggle and scream
   soundness.

**Process rule:** every experiment gets an `EVIDENCE.md` entry **the session it
concludes** — hypothesis, command, n, CI, verdict, one sentence of
interpretation. End-of-session checklist: HANDOFF plan ✓, EVIDENCE entries ✓,
this file's calendar ✓.

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
| ✅ **B1** | ~~**Feature augmentation + retrain**~~ **WON 2026-07-30/31** | The premise was even better than stated: the blindness was **representational, not informational** — the net always had per-slot HP, but `opt["index"]` was never encoded, so two options naming two copies of the same card were bitwise identical | **DONE.** `optfeat` v3 (25→37 cols), corpus `artifacts/pds_v3`, 7 A/Bs at n=2000 | **Pre-registered kill line was ≤0.52. Measured 0.661 vs the SHIPPED agent** (≈ +115 Elo) and **0.878** vs a same-corpus control. **The rules are now harmful (0.427)** — the method inverts. `EVIDENCE` §8f |
| ~~**B2**~~ | ~~**Arithmetic MAIN layer** — the lethal audit~~ | ~~missed lethal is the classic handcrafted-agent gap~~ | **KILLED 2026-07-30** by `scripts/p2_lethal.py`, 200 games | **both cuts empty: 316/316 lethals taken and all 316 FORCED (honest denominator 0), 7/803 promotion cases with retreat illegal in all 7. Grimmsnarl ex has ONE payable attack, so the decision does not exist in this deck** — EVIDENCE §8 |
| **B3** ⬆⬆ | **Matchup branches** (archetype detection only if needed) | **PROMOTED TO #1 2026-07-30 — the repair for a MEASURED defect: `chip_target` scores −0.126 against `rule:crustle` while paying +0.077 in the mirror** (EVIDENCE §8c). Also the report's opponent-modeling chapter. ⚠ **Try the classifier-free version first** — the Crustle condition reads straight off the board ("our attack would deal 0 to their Active"), no archetype model needed (HANDOFF §3.3). ✅ **Instance 1 (Crustle) SHIPPED and recovered +0.104.** ⚡ **Instance 2 is now open and better-evidenced: MEGA LUCARIO.** v3 is **−50 Elo** there (the only anchor of five it loses) and we won **36.4% of 11 real games against opponents rated 85 points BELOW us** — two independent instruments, same matchup (EVIDENCE §8i). ⚠ **Also a THIRD damage-reduction deck now has an anchor:** Archaludon's Full Metal Lab is −30 into any Metal Pokemon, which `WALL_POKEMON = {345}` does not model | **Lucario first: audit before rule (rule 14).** Read the 11 real games + an `opportunity_audit`-style probe vs `rule:v10`; ask whether `chip_target`/`energy_spread` are net-negative there the way `chip_target` was vs Crustle. Then (a) one-line branch, A/B n=2000 vs **all five** anchors | the audit sizes the effect below what n=2000 resolves (±0.021) → close it by sizing like the Morgrem out, no A/B spent |
| **B4** | **Turn-level planning with the unused 600 s pool** — enumerate within-turn action *sequences*, score end-of-turn states with `evalfn`/`textdmg` | We use 0.1 s of 600 s; the #1 player reportedly uses the full budget. **Distinct from the dead game-tree search**: no rollouts, no determinized opponent turns — just sequencing of our own turn, where the variance problem that killed our search (terminal 0/1, SE≈0.14) does not exist. Novel for this board | measure within-turn branching first (`p6_recon`-style, 100 games); if tractable, prototype on MAIN-heavy turns only, A/B n≥1000 with pool-usage logging | branching intractable even with beam limits, or A/B ≤0.52 — note that *eval quality*, not time, becomes the binding constraint |
| **B5** ⬆ | **Deck adaptation vs the counter-meta** (= Track C experimentation half) | **PROMOTED 2026-07-30 — the premise is now measured, not hypothesised: Crustle went 0.06% → 18.1% of the field at 56.6% WR with the LB top two on it, while our WR fell 52.2% → 47.5%.** Matchup EV has moved more than any play-skill rule can recover | Track C steps 1–2 (Crustle pilot, then the mechanic probe) | probe says counters don't bypass the prevention → passive-damage line dead; enumerate other outs before any list change |

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
| large-scale self-play / league | ~1.4 cores; nothing at the top of this board is learned | **dropped** — EVIDENCE §2 |
| more data / better representations via training | lw3 (4,010 games, best val acc) lost 0.491; winners-only 0.375 | **dead as stated** — EVIDENCE §1 |
| richer state features | this is the *one* plank our own thesis predicts should work | **live as B1** (§2.5) |
| opponent modeling / beliefs | the top notebooks do the cheap version (matchup rules on revealed cards) | **live as B3**, minus the neural machinery |
| within-turn sequencing (Hearthstone shape) | we use 0.1 s of a 600 s budget; no rollout variance in this restricted form | **live as B4** |
| GNN / Set Transformer encoders | no measurement, and 18 days on 1.4 cores | **not attempted; documented as out of budget** |

**The principle worth keeping from it**, which the project already follows: for
any borrowed idea, ask *what problem does this solve; does PTCG have that
problem; what is the PTCG-specific version; how would we falsify it cheaply?*
