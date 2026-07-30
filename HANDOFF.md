# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** public LB **and** the Strategy Category. LB top is **1179.8**
(`Majkel1337`); our best agent reads **970.1**. Sim deadline **2026-08-17**, then
~2 weeks continued play; strategy report due **2026-09-14**. Kaggle CLI is
authenticated.

**Read §2 before trusting any number. §3 is the live plan. This file must always
end with a live plan, never a summary.**

### The four files, and what each owns

| file | owns |
|---|---|
| **`HANDOFF.md`** (this) | live state, the live engineering plan, the anti-self-deception rules, commands, gotchas |
| **`ROADMAP.md`** | the strategy-competition plan — what the engineering is *for*, the breakthrough hunt, the calendar |
| **`report/EVIDENCE.md`** | the hypothesis log: every concluded experiment with n, CI, verdict. **All closed-experiment detail lives there, not here.** |
| **`competition_details_and_rubric.md`** | the rubric, verbatim |

**End of every session: update HANDOFF (plan), ROADMAP (calendar), and
EVIDENCE (any experiment that concluded) together.**

> ⛔ **THERE IS NO FREE SUBMISSION SLOT.** Only the **latest 2** submissions play
> episodes, and the active pair is `55077709` (824.9) + `55072063` (**948.1, our
> best**) — `55054446` is already inactive. **Any** submission evicts
> `55072063` and freezes it at ~948; the only un-evict is resubmitting and waiting
> out another 4 h+ climb from μ=600. **So the next submission must be something we
> expect to beat 948 — not a rollback, and not a single small rule** (§3.0).

---

## 1. Where we are (day 7 end, 2026-07-29)

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

### ⚠ The meta shift is now MEASURED (2026-07-30), and it is worse than reported

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

1. **🔴 `lucario_v10` — the single opponent every routine number in this repo is
   measured against — is 0 of 400 games.** Our arena bar has been measuring a
   deck that has left the meta entirely. This is rule 12's worst case, realised.
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
12. **The local arena is ONE opponent deck.** Everything routine is measured
    against `rule:v10,noS` on `lucario_v10`. A pattern the user watched in a real
    game can be genuinely absent locally without being absent on the LB. **When
    the user reports something and the local audit says it never happens, measure
    it on `replays/submission_replay_2026-07-29/`** — `scripts/p5a_replays.py` is
    the worked example; it reads our real selects against 54 distinct LB
    opponents.
13. **Check the denominator is a real CHOICE, not just a real count.** P5a read
    "the rule takes the best target 26/26" — but 90 of its 95 pooled-KO rows
    offered only one prize value, so nothing could go wrong. The honest
    denominator was **5**. A rate over forced moves measures nothing.

---

## 3. THE PLAN (day 7 → day 8)

P5 is closed, the Boss's Orders lever is closed four interventions deep, and P6a
won locally and shipped. **The open questions: (0) is P6a actually good, (1) are
we measuring against the right opponents at all, (2) Crustle.** `ROADMAP.md`
§2.5 holds the ranked breakthrough candidates (B1–B5) that run alongside.

### 3.0 `55077709` (P6a / `counter_source`) — DOWNGRADED to "unresolvable on the LB"

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

Preserved builds, so nothing ever needs rebuilding under time pressure:
`dist/submission_bc-grimmsnarl-netspolicy_20260729-103819.tar.gz` (P4b) and
`...-152103.tar.gz` (P6a).

### 3.1 Re-anchor the arena on the CURRENT meta — steps 1–4 DONE, 5 is the work

Every number in §3, §6 and `EVIDENCE.md` was earned against `lucario_v10`, which
**is now 0% of the meta** (§1). So the bar itself has to be rebuilt.

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

🔴 **Blocker on step 5, and it is now the project's critical path: the new anchors
have no pilots.** `rule:v10` is Lucario-specific scoring and `bc` would play them
off-distribution. Options, cheapest first:

1. **`bc` piloting the new deck as a *fixed* opponent.** Crude, but for A/B
   *deltas* the opponent only has to be identical on both sides (rule 5), not
   strong. **Do this first — it unblocks step 5 today** — while being explicit in
   the write-up that a weak pilot under-reads the matchup (§3.2).
2. **A minimal rule pilot for Crustle** (the real instrument; Track C step 1).
   Look for the public bot first: `dashimaki360/beating-the-day-1-1-crustle-bot`.
3. Check `import_rule_agents.py` / new public notebooks for anything piloting
   these archetypes (the LB top two play Crustle, so a notebook may exist).

⚠ **Do not treat a cross-deck score as skill** (rule 5) — use each new anchor the
way `rule:v10` was used: a fixed opponent for A/B *deltas*, both sides facing the
identical opponent. And **archive the per-anchor tables**; they are the rubric's
consistency/robustness exhibit and go into the report verbatim.

⚠ **Do not treat a cross-deck score as skill** (rule 5) — use each new anchor the
way `rule:v10` is used: a fixed opponent for A/B *deltas*, both sides facing the
identical opponent.

Also: **archive the per-anchor A/B tables.** They are the rubric's
consistency/robustness exhibit and go into the report verbatim.

### 3.2 Crustle — **this is the meta now**, and it is still unpiloted

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

### The board

| | item | state |
|---|---|---|
| **§3.0** | is `55077709` (P6a) actually good? | **closed as an LB question** (unresolvable — 12 Elo vs a ±75 instrument); re-opened as §3.1 step 5 |
| **§3.1** | re-anchor the arena on the current meta | **steps 1–4 DONE 07-30** (meta measured, `lucario_v10` is 0% of it, two new anchor decks committed); **step 5 = re-run every shipping A/B, blocked only on a pilot** |
| **§3.2** | Crustle | **now known to BE the meta (18.1%, 56.6% WR, top-2 LB)**; deck rebuilt to the current list; **no pilot**, premise unverified — the critical path |
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

  | function | select | switch | arena |
  |---|---|---|---|
  | `chip_target` | DAMAGE / DAMAGE_COUNTER(_ANY) | `noChip` | 0.577 n=2000 → **+~150 LB** |
  | `energy_spread` | MAIN, {D} ATTACH onto a Munkidori | `noSpread` | **0.702 n=4000** |
  | `counter_source` | REMOVE_DAMAGE_COUNTER (ours) | `noSrc` | 0.534 n=2000 — **LB on trial** |
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
- `features.py` (v2, DENSE_DIM=242, PER_SLOT=18) / `optfeat.py` — shared by
  trainer and inference. **Any npz trained pre-v2 fails the dim guard.** Adding
  an HP/damage feature here bumps `VERSION` and retrains every net — that is
  ROADMAP candidate **B1**.
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

### The arena's real opponent: `rule:v10`

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

# The leaderboard (paginates at 20)
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(i, r.team_name, r.score) for i, r in enumerate(a.competition_leaderboard_view('pokemon-tcg-ai-battle')[:20], 1)]"

# Skill measurement: near-mirror head-to-head (rule 5). The only kind that counts.
python -X utf8 scripts/arena.py play "rule:v10,noS" rule:lucario `
    --deck-a lucario_v10 --deck-b mega_lucario_ex --matches 500

# Against the real bar
python -X utf8 scripts/arena.py play bc "rule:v10,noS" `
    --deck-a grimmsnarl --deck-b lucario_v10 --matches 500

# A/B a rule override against the pure clone (how every targeting.py rule is judged).
# Off-switches: noChip, noSpread, noSrc. Opt-in (default off): drag, dragHi, boss, veto.
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
python -X utf8 scripts/p6_recon.py --matches 120   # EVERY select, bucketed -- the menu
python -X utf8 scripts/p5_audit.py --matches 200   # sizes the three P5 findings
python -X utf8 scripts/p5a_replays.py              # the same counters on 55 REAL games
python -X utf8 scripts/p5b_check.py --matches 150  # does a rule actually fire? (rule 9)

# §3.1 re-anchor: what is the field ACTUALLY playing now? (last fetched: 07-27)
python -X utf8 scripts/fetch_top_episodes.py --date 2026-07-30 --max 400
python -X utf8 scripts/mine_meta.py replays/2026-07-30    # takes dirs as arguments
powershell -File scripts/fetch_days.ps1        # several days; edit $Dates default (§7)

# Train (12 epochs; artifacts/pds_v2 is the shipped corpus)
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v2 --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 --out out/policy_X.npz

# Rebuild shards from raw replays (more data is NOT a lever -- EVIDENCE §1)
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds/d30 replays/2026-07-30

# Build + submit (smoke-tests the bundle the way Kaggle loads it)
python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc --nets policy
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
  by timing). Self-play RL dropped on the same evidence. `EVIDENCE` §2.
- **Boss's Orders — all four interventions null, the card is closed. Do not write
  a fifth.** `EVIDENCE` §6.
- **Closed cheaply and correctly:** P5c never-end-without-attacking (3,683/3,683),
  `REMOVE_DAMAGE_COUNTER_COUNT` (100% already), post-KO promotion (9 misses/120
  games), `TO_HAND` duplicate-avoidance (parity), the decklist variant (0.490),
  P5a pooled Adrena-Brain (~0.5 real decisions per 200 games). `EVIDENCE` §8.
- **Do not resurrect:** the `rule:iono` arena→LB ladder; the old deck sweep's
  ranking; "the clone is comfortably above the rule baseline"; every n=24 number
  and every strength claim dated before 2026-07-27 pm; "3× compute made it
  worse". `EVIDENCE` §10.

⚠ **Everything above was measured against ONE opponent in the pre-shift meta.**
The negatives are probably safe; the **positives** are what §3.1 must re-check,
and `counter_source` is already under suspicion for exactly this reason (§3.0).

⚠ **Open loose end:** the P2b "already at demonstrator parity" verdicts were only
re-derived for `munkidori_adrena_brain` after the P4c multiplicity fix; the
demonstrator side of the `opps` column has never been run
(`--corpus artifacts/pds_v2`). `EVIDENCE` §8.

---

## 7. Gotchas (all paid for)

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
- Kaggle Python API returns **snake_case** (`public_score`, `team_name`);
  `competition_leaderboard_view` paginates at 20 rows.
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
