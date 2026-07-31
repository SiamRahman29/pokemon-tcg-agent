# EVIDENCE — the hypothesis log

**What this file is:** every concluded experiment as *hypothesis → measurement →
verdict → interpretation*. It is the raw material for `report/STRATEGY.md`
chapters 2, 4 and 8 (see `ROADMAP.md` Track B), and the archive of detail that
used to live in `HANDOFF.md` §3–§6.

**Rules for entries.** Every number traces to an archived run with `n` and a CI.
Every entry gets written **the session the experiment concludes**, not later.
Unless stated otherwise: arena A/B, grimmsnarl mirror, `bc` = our shipped clone,
intervals are 95%.

⚠ **Every measurement dated before 2026-07-30 was taken against ONE opponent**
(`rule:v10,noS` piloting `lucario_v10`) in the **pre-shift meta**. Negatives are
probably safe (a rule that could not beat that anchor is unlikely to be a hidden
gem); **positives are the ones to re-check** after the re-anchor. See
`HANDOFF.md` rule 12 and §3.1.

---

## 1. Training the clone — the plateau (3 axes, all negative)

**Hypothesis:** the behavior clone improves with more data / better validation
accuracy / cleaner labels.

| net | corpus | val top-1 | head-to-head vs prev shipped, n=2000 |
|---|---|---|---|
| v1 (pointwise BCE) | 2,410 games | 0.6596 | — |
| **v2 `policy_lw2` (listwise)** | **2,810** | **0.6755** | **0.524 — SHIPPED** |
| `policy_lw3` (more data) | 4,010 | 0.6933 | 0.491 |
| `policy_win` (`--winners-only`) | 2,810 | 0.6410 | **0.375 — decisive** |

**Verdict: all three negative.** More data, higher val accuracy, and
winners-only all fail. `--winners-only` is 12 pp *worse* — **cloning the losing
side is helping.** `lw3` has the best val accuracy of any net and lost the A/B.

**Interpretation:** validation metrics do not predict playing strength in this
game (five separate confirmations, incl. the value net and three policy top-1
runs) — this is `HANDOFF` rule 3, and it is the reason every claim in this file
is an arena A/B. `--loss listwise` reaches in 1 epoch what BCE took 4 to reach,
so the loss function mattered and the data volume did not.

---

## 2. Search — dead, ours *and* the field's

**Hypothesis (ours):** determinized rollout search over sampled worlds beats the
clone's single forward pass.

**Measured:** `search:M,noV,roll,mo,mc20,pb0.15` vs `bc` = **0.323, n=31**. It
overruled the clone on 52% of decisions. More determinizations and adding the
value net were also negative.

**Diagnosis (the useful part):** a terminal rollout returns 0/1, so a mean over
12 determinizations has **SE ≈ 0.14**, and the max over ~9 rival actions sits
~0.21–0.28 above truth *by chance alone*. The search was not mis-tuned; it was
selecting noise. **Any future search must attack this variance term first** —
which is exactly why candidate B4 (within-turn sequencing, no rollouts, no
opponent determinization) is a different bet and not a retry.

**Hypothesis (the field's):** the LB-950 reference agent's strength comes from
its shallow MCTS.

**Measured: V10's MCTS has never once executed.** Two independent bugs:
1. its candidate set comes from `choose()` truncated to `select.maxCount`, which
   is **1 for every MAIN select measured (70/70)** — so there is nothing to
   search over;
2. `search_begin(obs, your_deck=yd)` passes 1 of 7 required arguments and raises
   `TypeError` into a bare `except`.

Confirmed by timing: **200 games in 11.8 s** for a nominally
wall-clock-budgeted agent.

**Verdict/interpretation:** **LB 950+ is 100% handcrafted policy.** Nothing in
this competition has ever demonstrated that search is worth anything. This is
the single most load-bearing negative result we have, because it redirected the
whole project from architecture to decision-repair. Loose end if ever revisited:
`agents/sa/worlds.py`'s `World` is exactly the `search_begin` argument bundle.

**Also dropped on this evidence: self-play RL.** Days of work on ~1.4 cores to
maybe reach where hand-written rules already sit, in a competition where nothing
at the top is learned.

---

## 3. The discriminator — dominated vs tradeoff (3 for 3, 0 for 4)

**Hypothesis:** `optfeat` gives the net no HP, no damage and no attached-energy
count, so any select whose right answer is *that arithmetic* is decided at
chance and a rule can win it.

**Sharpened after 7 rules:** it is not enough for the net to be blind — the
decision must have a right answer arithmetic can *prove*. Rules that delete a
**dominated** option win; rules that pick a side in a **tradeoff** lose.

| rule | select | class | arena | LB |
|---|---|---|---|---|
| `chip_target` | DAMAGE / DAMAGE_COUNTER | dominated | **0.577** [0.555, 0.599] n=2000 | **+~150** |
| `energy_spread` | MAIN, {D} onto Munkidori | dominated | **0.702** [0.687, 0.715] n=4000 | in the 970 agent |
| `counter_source` | REMOVE_DAMAGE_COUNTER | **half/half — see §7** | 0.534 [0.513, 0.556] n=2000 | **unresolved, 762→746** |
| `drag_target` (aim the drag) | SWITCH | tradeoff | 0.489 [0.467, 0.511] n=2000 | null |
| `boss_converts` (force the play) | MAIN | tradeoff | 0.493 [0.471, 0.515] n=2000 | null |
| `boss_veto` (suppress the play) | MAIN | tradeoff | 0.493 [0.471, 0.515] n=2000 | null |
| `drag_target(prefer_high_hp)` | SWITCH | tradeoff | 0.490 [0.469, 0.512] n=2000 | null |

**Interpretation:** the net has watched 2,810 games of humans making those
trades and is already as good at them as our arithmetic. What it *cannot* do is
see HP, damage or attached energy — so it loses to pure arithmetic every time
the answer *is* arithmetic. **This is the report's central falsifiable claim**,
and §7 below is the one datapoint that tests its boundary.

---

## 4. `chip_target` — the founding result (+~150 LB)

**Hypothesis:** the net aims chip damage (Shadow Bullet's snipe, Adrena-Brain's
counters) at chance, because it cannot see HP.

**Measured:** it picked the lowest-HP target **25.7%** of the time — chance.

**Rule:** override every select where *all* options are the opponent's Pokemon
(`agents/sa/targeting.py`).

**Measured after:** `bc` vs `bc:noChip` = **0.577** [0.555, 0.599] n=2000 (+54
Elo). Against the independent bar: `bc` vs `rule:v10,noS` went **0.418 → 0.537**
[0.506, 0.568] n=1000 — i.e. it flipped us past the public LB-950 agent.
**The LB agreed: ~+150 points.**

**Interpretation:** the project's central method, confirmed end to end — *find
decisions the features cannot express, and write a rule for them.* Three axes of
more training bought nothing; one missing feature bought ~150 LB points.

---

## 5. `energy_spread` (P4b) — the largest effect measured here (0.702)

**Hypothesis (user's reading):** we waste manual attaches by stacking a second
{D} on a Munkidori that already has one, instead of arming the second Munkidori.

**Four mechanics verified in-engine** (the `probe_adrena.py` pattern: 40 games
with a wrapper that greedily takes every Munkidori ability):

1. Adrena-Brain is **once per Pokemon**, not once per turn — we activated it
   twice in a turn 35 times; a slot that had used it was never re-offered.
2. The {D} condition is a **threshold, not a cost** — energy after use unchanged
   **138/138**.
3. **Munkidori is not a "Marnie's Pokemon"** (card 112 is plain `Munkidori`; the
   others are `Marnie's Impidimp/Morgrem/Grimmsnarl ex`), so Punk Up cannot
   attach to it — in 40 games every attach option targeting a Munkidori came
   from the hand, i.e. the 1-per-turn manual attach. **That is what makes a
   wasted attach expensive.**
4. **A second {D} on a Munkidori is dead, full stop.** Munkidori's only attack
   is Mind Bend ({P}{C}) and this deck runs zero Psychic energy — so it cannot
   even be attack setup.

**Arithmetic:** two Munkidori at 1 {D} move 6 damage counters a turn (a **60-point
swing** — Adrena-Brain both heals us and damages them); one Munkidori at 2 {D}
moves 3.

**Measured:** the clone chose the wasted attach **143 times to 94** — worse than
a coin flip, because `optfeat` carries no attached-energy count. The rule takes
that to 0 and lifts activations from **1.26 → 1.60 per turn**.

**Verdict: 0.702 [0.687, 0.715], n=4000.** For scale, the +150-LB
`chip_target` scored 0.577.

**Interpretation:** the cleanest dominated-option rule we have — the overridden
option is *provably worthless*, no judgment involved. This is the datapoint that
anchors the top of the discriminator.

---

## 6. Boss's Orders — CLOSED, four interventions deep

**Hypothesis (user observation, accurate):** *"we had the chance to play Boss's
Orders to bring out a weaker benched Pokémon and knock it out with Shadow Bullet
but we didn't."* Plus, later: a dragged Pokemon got evolved into their main
attacker, so sometimes the play is actively bad.

| intervention | fires / audit rate achieved | arena |
|---|---|---|
| `drag_target` — aim the drag | 85/99 → **99/99** best KO taken | 0.489 n=2000 |
| `boss_converts` — force the play when it converts | 36.9% → **100%** of converting turns | 0.493 n=2000 |
| `boss_veto` — suppress the play when it converts nothing | fires on **57.6%** of wanted plays | 0.493 n=2000 |
| `drag_target(prefer_high_hp)` — KO-able tiebreak | decides **56.5%** of drag selects | 0.490 n=2000 |
| both P4a rules together | — | **0.452** [0.435, 0.470] n=3000 |

Supporting facts: 157 converting turns in 300 games; the clone played Boss on
36.9% of them vs 25.7% of all other legal turns (so it *does* discriminate,
barely); we play the card on **38.2% of legal turns** vs demonstrators' 31.4%;
the veto's fallback is verified safe — of 50 vetoed plays that fell through to
END, **50/50 had no attack available**, so it never threw a turn away.

**Verdict: four interventions, four nulls, on a card we play 38% of legal
turns. The card is closed. Do not write a fifth.** (The pair at 0.452 is only
marginally outside the intervals — read it as "no evidence of benefit, some
evidence of harm", not a precise interaction estimate.)

**Interpretation — this is the origin of `HANDOFF` rule 10:** every one of them
moved its audit rate *exactly as designed* and not one won a game. Moving an
audit rate is not winning games — rule 3 (val accuracy ≠ strength) reappearing
one level up, in the **rule** pipeline instead of the training one. And per the
discriminator: every one picks a side in a trade (Supporter for a prize, this KO
vs that KO) the net has seen thousands of humans make.

All the code stays in `targeting.py` behind default-off flags as the record.
Repro today: `bc:drag` vs `bc:base`, `bc:boss` vs `bc:base` (the original
isolation runs predate the opt-out default flip).

---

## 7. `counter_source` (P6a) — a win in the arena, unresolved on the LB

**How it was found (P6 recon, the systematic version of P5's live-game
watching):** `scripts/p6_recon.py --matches 120` buckets every select by
(context, whose options) and reports how many are a real decision. The menu:

| select | share of selects with ≥2 options | verdict |
|---|---|---|
| `MAIN` | 47.7% | the remaining mass — P2 |
| `TO_HAND` ours | 15.3% | at demonstrator parity (§8) |
| `DAMAGE_COUNTER` theirs | 5.6% | `chip_target` owns it |
| `ATTACH_FROM` ours | 5.5% | resource tradeoff, unexamined |
| `REMOVE_DAMAGE_COUNTER_COUNT` | 5.2% | closed — already 100% max |
| `TO_ACTIVE` ours | 3.9% | closed — 91.2% right already |
| `DAMAGE` theirs | 3.7% | `chip_target` owns it |
| `REMOVE_DAMAGE_COUNTER` ours | 2.9% | **the one live finding** |

**Hypothesis:** Adrena-Brain moves "up to 3 damage counters" from one of our
Pokemon to one of theirs, and **the source is its own select** — all options
*ours*, which is exactly the case `chip_target` declines by design. How many
counters then move is capped by what the source carries.

**Measured:** the clone takes the maximum on the *follow-up* count select
**100% of the time (n=481)** — so all of the loss is one select earlier, in the
source pick, where the features go blind. In **59 of 291** source selects with
≥2 options (**20.3%**) it picked a source that moves fewer counters than an
available alternative — 10 or 20 damage where 30 was on the table.

**Rule:** `targeting.counter_source` takes that to **0**; full 3-counter moves
go 67.1% → 76.5% of activations.

**Measured:** **0.534 [0.513, 0.556] n=2000** mirror, balanced across seats
(534/466 as P0, 535/465 as P1 — not a seat artifact). Against the independent
bar: `bc:s,src` vs `rule:v10,noS` = **0.626 [0.604, 0.647] n=2000** against a
bare `bc`'s **0.593 [0.562, 0.623] n=1000** — +0.033 there vs +0.034 in the
mirror, two measurements, same size, same sign.

**Then it read 762.2 → 746.4 on the LB while the otherwise-identical
`55072063` sat at 970.1.**

### The day-8 resolution (2026-07-30): the LB was the wrong instrument

| when (UTC) | P6a `55077709` | P4b `55072063` | gap |
|---|---|---|---|
| 07-29 09:21 | submitted (μ=600) | — | — |
| 07-29 10:22 / 10:27 | 762.2 → 746.4 | 970.1 | ~224 |
| 07-30 08:19 (+23 h of play) | **824.9** | **948.1** | **123** |

**The agents converged toward each other from opposite directions** — P6a +78
climbing off μ=600, P4b −22 settling off an overshoot. That is the signature of
two close true ratings, not of a 220-point regression.

**And the arithmetic nobody did on day 7: a 0.534 mirror A/B is ≈ +12 Elo, while
LB readings swing ±50–100 during convergence.** So the leaderboard could never
have adjudicated this rule in either direction, and **the arena and the LB were
never in conflict.** The day-7 "the local arena produced a confident false
positive" framing was itself a false positive — produced by asking a ±75-point
instrument to measure a 12-point effect.

**Verdict: unresolvable on the LB; re-opened as an arena question** (re-measure
against the post-shift anchors, `HANDOFF` §3.1 step 5). The rule stays shipped
meanwhile, because rolling it back would cost our best agent's active slot for a
re-run of identical code (`HANDOFF` §3.0).

**Interpretation — this is the most transferable methodological result of the
week, and it is a report chapter on its own:** *state the resolution of your
instrument before you let it overrule another instrument.* We have two measuring
devices with a ~6× precision difference and we spent a session letting the coarse
one veto the fine one. It also has a hard forward consequence: **if the remaining
levers are stacks of ~10-Elo rules, no single-rule submission can ever be
validated on the LB** — only multi-anchor arena A/Bs, then one submission of the
bundle.

⚠ **The narrower `counter_source` variant is still worth building** regardless of
the verdict: rule 11's ⚠ clause says the shipped rule is *mis-classified* (a
dominated half welded to an asserted-judgment half), and the narrow version —
redirect only when strictly better on transfer AND not obviously the worse heal —
is the one that actually belongs in the dominated column.

**Interpretation (written before the verdict, deliberately):** two candidate
causes with opposite fixes.

- **The anchor was stale.** Both "independent" confirmations share the same
  opponent deck and the same era of the meta, so they are far less independent
  than they look. Fix: re-anchor, re-measure.
- **The dominance argument was half wrong.** *Damage transferred* is genuinely
  dominated (3+ counters move 30, 1 moves 10 — pure arithmetic). *Healing* is a
  **tradeoff that was asserted, not measured**: moving 30 off our most-damaged
  Pokemon is only the best heal if that Pokemon is *savable*. The clone may have
  been making that judgment correctly with information the rule discards.

**The methodological lesson, which outlives the rule either way:** a rule is
only in the dominated column if **every** dimension it moves is arithmetic. If
one of them is a judgment, it is a tradeoff no matter how good the other looks.
This is now `HANDOFF` rule 11's ⚠ clause.

---

## 8. Sized and closed cheaply (the cuts that saved days)

Each of these was a plausible lever, measured, and closed for the cost of one
audit — the *reason* they are here is that measuring first is cheaper than
building first.

| item | measurement | verdict |
|---|---|---|
| **"never end a turn without attacking" (P5c)** | the clone attacked on **3,683/3,683** turns where END and a payable ATTACK were both offered | nothing to fix |
| **`REMOVE_DAMAGE_COUNTER_COUNT`** | moves the maximum offered **100%** of the time, n=481 / 120 games | nothing to fix |
| **post-KO promotion (`TO_ACTIVE`)** | promotes a Pokemon that cannot attack while an attacker was benched **9 times in 120 games** (91.2% right on the 102 selects that mattered) | too small, and a tradeoff |
| **`TO_HAND` duplicate-avoidance** | demonstrators fetch a duplicate 5.8% (n=57,053); we fetch one 5.8% (n=482) | already correct |
| **decklist variant** (+2 Boss's Orders / −1 Tool Scrapper / −1 Spikemuth Gym) | **0.490** [0.468, 0.512] n=2000 | null — see §9 |
| **`REMOVE_DAMAGE_COUNTER` (general)** | lowest lift on the board, but demonstrators are themselves inconsistent (Active 33.6%, max-prize 60.6%, ~2.8 options, n=9,911) | **a low lift can mean a noisy label, not a blind feature** |

### P5a — pooled Adrena-Brain budget: closed, *and the instrument was broken*

This entry is kept in full because the bug is the lesson.

Day 6 sized it at "26 pooled-KO selects in 200 games, 26/26 correct". **Both
halves were wrong, and it still closes.**

1. **The budget was off by one activation.** `p5_audit` computed
   `left = max(1, len(armed - used))`, but the Munkidori being activated *right
   now* is already in `used` (the MAIN select that fires the ability precedes
   the DAMAGE_COUNTER select). So `armed - used` is what remains *afterwards*,
   and the pool is this activation **plus** those. It read `left == 1` on every
   row ever measured — including all 54 real-replay rows with two armed
   Munkidori — so **a 60-point pool was never once representable**: the audit
   was structurally incapable of detecting the exact scenario the user
   described. Fixed to `left = 1 + len(armed - used)`; the pooled-KO denominator
   roughly tripled (26 → 89–95 per 200 games).
2. **The 26/26 was mostly forced moves.** Of 95 pooled KOs, **90 offered only
   one prize value** — nothing to get wrong. The real denominator is ~5–7 per
   200 games; across three runs the rule missed 2 of ~19.
3. **On the real replays it is rarer still.** `scripts/p5a_replays.py` over the
   55 live games: 266 chip selects, 6 pooled KOs, and **all 6 had a single
   prize value — not one real choice in 55 games against the actual field.**

**Verdict:** real mechanism, right arithmetic, worth ~0.5 decisions per 200
games. Not a lever, **do not rebuild it** — but the fixed budget term matters to
anything else reasoning about Adrena-Brain's per-turn output.

**Interpretation:** this is the origin of `HANDOFF` rule 13 (*check the
denominator is a real CHOICE, not just a real count*) — a rate over forced moves
measures nothing.

### B2 / P2 — the lethal audit: closed, because this deck has one attack

**Hypothesis (ROADMAP candidate B2, ranked #2):** the entire top-10 is
handcrafted deck expertise, missed lethal is the classic gap in card-game AI, it
is pure arithmetic (the dominated column), and `textdmg.estimate` is exact for
this deck — so "a payable attack KOs their Active and we didn't take it" should
be a real and winnable defect.

**Instrument:** `scripts/p2_lethal.py`, 200 games vs `rule:v10,noS`, one row per
turn scored at the turn's **final** MAIN select (the opponent's Active HP moves
during our own turn, so an early lethal can be stale or unnecessary by the end).
ATTACK options arrive on the MAIN select carrying `attackId`, and the engine only
offers attacks the Active can pay for, so payability needs no modelling.

| cut | n | result |
|---|---|---|
| **1. same attacker** (dominated, high prior) | 316 lethal-offered turns | **took the lethal 316/316** |
| **1, on the honest denominator** — lethal **and** a non-lethal attack both offered | **0** | **the choice does not exist** |
| **2. needs promotion** (tradeoff, lower prior) | 803 no-KO turns | 7 (0.9%) had a bench Pokemon that could KO — and **in all 7 retreat was not legal**, so the action didn't exist either |

**Verdict: B2 is killed on both cuts. Do not build a lethal detector.** The kill
criterion ("same-attacker cut reads <2% missed") is met at 0% — but the *reason*
matters more than the number.

**Interpretation — the structural finding:** **Grimmsnarl ex has exactly one
payable attack (Shadow Bullet, 180 flat).** So "which attack" is never a decision
in this deck, and the missed-lethal class of expertise — a real edge for decks
with multiple attackers or damage tiers — **cannot exist for us.** Three
consequences:

1. It explains why a behavior clone with no deck expertise can be competitive
   here: the arithmetic-heavy decisions that handcrafted agents win are
   concentrated in **targeting** (which we patched, +150 LB) rather than in
   **attack selection** (which is empty).
2. It removes B2 from the breakthrough ladder and raises B1 and B4 by
   elimination.
3. **It is a hidden cost of any decklist change** (ROADMAP Track C step 4): add a
   second viable attacker and this whole decision class appears, unpatched and
   off-distribution for the net.

⚠ **Methodological note, and the reason this entry is trustworthy:** the first
run of this audit read **"94.5% took the lethal, 5.5% missed"** — 19 apparent
misses. All 19 were turns the **game ended on**: Adrena-Brain moves counters onto
any of their Pokemon, so a winning turn often takes the last prize via the
*ability* and never reaches an attack, which is indistinguishable from "ended the
turn with a lethal on the table" unless you separate it. Scoring those as misses
would have justified building a rule to fix wins. **Rule 9/13 again: check what
the denominator is made of before believing the rate.**

**Hypothesis (user):** *"I think we are not using Adrena-Brain at every chance."*
The instrument was indeed wrong; the corrected number is small.

`opportunity_audit.py` now declares a `MULTIPLICITY` per line and prints an
`opps` column beside `turns`. `munkidori_adrena_brain` reads **99.4% per turn
but 96.9% per opportunity** (452 opportunities over 359 turns, 150 games).

**Verdict:** real bug, ~3% miss — the activation was never the lever. The lever
was upstream: getting a second Munkidori armed at all (§5).

Only `munkidori_adrena_brain` is a `"count"` line, because it is the only one
whose copies are countable **on both sides** (one ABILITY option per Munkidori,
live and in the shards). Items are repeatable too, but a Rare Candy option
carries no target, so counting options would *invent* a denominator. **Do not
widen `"count"` without a real target count.**

⚠ **Open:** the P2b "already at demonstrator parity" verdicts
(`munkidori_adrena_brain` 99.3%, `rare_candy_play` 82.0%,
`evolve_impidimp_to_morgrem` 91.6%, `dark_energy_to_munkidori` 78.3%) were only
re-derived for `munkidori_adrena_brain` after this fix. The others are
once-per-turn lines and so unaffected in principle, but the demonstrator-corpus
side of the `opps` column has never been run (`--corpus artifacts/pds_v2`).

---

## 8b. The meta shift, measured (2026-07-30) — the report's headline figure

**Hypothesis (user-reported 2026-07-29):** the field has moved and the top of the
board has reshuffled.

**Method:** `scripts/mine_meta.py` over top-episode replays, archetype = the
deck's Pokemon/energy signature, win rate excluding draws. **Pre-shift** = 07-22 +
07-24 (800 games / 1,600 seats, `out/meta/pre_shift_0722_0724.txt`);
**post-shift** = 07-29 (400 games / 800 seats, `out/meta/post_shift_0729.txt`).
Both are top-episode samples ranked by `avg_score`, so this is the *high-rated*
field, which is the population we actually play.

| archetype | pre share | post share | pre WR | post WR |
|---|---|---|---|---|
| `{D}`/Munkidori — **ours** | 829 (51.8%) | 417 (52.1%) | 52.2% | **47.5%** |
| **Crustle** (`Mist`/`Spiky`) | **1 (0.06%)** | **145 (18.1%)** | — | **56.6%** |
| **Crispin toolbox** (`{G}`/Crispin) | 2 (0.1%) | **135 (16.9%)** | — | **58.5%** |
| Abra/Alakazam + Abra/Telepath | 214 (13.4%) | 37 (4.6%) | 45.8% | 38–45% |
| `{F}`/Rock Fighting = **`lucario_v10`** | 159 (9.9%) | **0 (0.0%)** | 54.1% | — |

**Verdict: the shift is real, larger than reported, and it invalidates our
measuring instrument rather than our agent.**

1. **`lucario_v10`, the single opponent behind every routine number in this
   project, is 0 of 400 games.** Rule 12 warned that one opponent deck is not the
   field; the deck has now left the field altogether. **Every positive result in
   this file is therefore provisional until re-measured** (§10) — and this, not
   any property of the rule, is the leading explanation for the `counter_source`
   puzzle in §7.
2. **Crustle went from 1 seat in 1,600 to 18.1% of the field at 56.6%**, and the
   LB's top two players are both on it. **Our win rate fell 52.2% → 47.5% over
   the same window while our share held at ~52%** — i.e. the field did not
   abandon our deck, it learned to beat it. That is a counter-meta, and it is the
   most likely single explanation for our ceiling.
3. **Our decklist is still exactly the field's consensus:** the most common exact
   60 on 07-29 was seen **353 times**, and `decks/grimmsnarl.py` is **identical to
   it, card for card**. So the deck is not stale; only the *matchup* is.

**Interpretation.** This is the cleanest robustness result in the project and it
cuts both ways, which is what makes it worth reporting: our *agent* work is
vindicated (we netdecked and kept the consensus best list, verified twice, months
apart in meta-time), and our *methodology* takes a direct hit (we validated seven
rules against a deck that no longer exists). The honest framing for the report:
**a single-anchor arena is a latent bug that a meta shift converts into a real
one, and the only defence is multi-anchor A/Bs — which is why every number from
here on carries an anchor label.**

⚠ **Operational note:** the current day's episode dataset **403s** — episodes are
published the following day, so the newest minable day is always yesterday. Plan
the pre-deadline re-mine accordingly.

---

## 8c. The re-anchored A/Bs (2026-07-30) — a shipped rule is HARMFUL in the new meta

**Hypothesis:** rules validated against `lucario_v10` (now 0% of the field, §8b)
still pay against the decks we actually face.

**Method:** each variant plays grimmsnarl against the **same fixed opponent** —
`rule:crustle` piloting `crustle_v1` — so per rule 5 the *differences* are
interpretable even though the absolute level is a cross-deck matchup. n=2000 per
row, archives `out/arena/anchor_rulecrustle_*.jsonl`.

⚠ **Scale note:** these are scores against a common opponent, so a rule's value
is the **difference between rows**. The mirror numbers in §3 are head-to-head
scores between variants. Different scales — compare *signs and rank*, not
magnitudes.

| A-side | score vs `rule:crustle` | rule's contribution here | its mirror result |
|---|---|---|---|
| `bc` — all rules on | **0.559** [0.537, 0.581] | — | — |
| `bc:x,noSpread` | 0.366 [0.345, 0.388] | **`energy_spread` +0.193** | 0.702 ✅ |
| `bc:x,noSrc` | 0.507 [0.485, 0.529] | **`counter_source` +0.052** | 0.534 ✅ |
| **`bc:x,noChip`** | **0.685** [0.665, 0.705] | **`chip_target` −0.126** 🔴 | 0.577 ✅ |

### Verdict 1: `counter_source` is vindicated — §3.0 is resolved

It pays **more** against the current meta (+0.052) than against the mirror
(+0.034) or `lucario_v10` (+0.033). Combined with §7's resolution-limit finding,
the whole "confident false positive" scare was an artifact of reading a ±75-point
instrument at 12-Elo precision. **The rule stays.**

### Verdict 2: `chip_target` — the founding +150-LB rule — is HARMFUL vs Crustle

0.559 with it vs **0.685 without**, non-overlapping intervals at n=2000 each.

**Mechanism, measured not assumed** (`scripts/p3_crustle_probe.py`, 60 games per
arm, counting damage-counter placement events):

| | counters onto **Crustle** | counters onto **Dwebble** |
|---|---|---|
| `chip_target` ON | 1,386 events, mean 12.9 | **235** |
| `chip_target` OFF | **1,583 events, mean 15.0** | **24** (−90%) |

`chip_target` ranks targets "anything that dies to 30 first, most prizes among
those, then lowest HP". Against Crustle that rule farms **Dwebble** — a 1-prize
basic — while the wall that must come down sits untouched. Switching it off moves
~200 counter events onto Crustle and lifts total counter damage there ~33%. The
net, left alone, concentrates correctly.

**Interpretation — this is the most important methodological result of the
project, and it is a report chapter on its own.** Our founding result, the rule
that bought ~150 LB points and defined the whole "find the blind select and write
a rule" method, **is matchup-conditional and we could not have known** while every
measurement came from one opponent deck. It does not overturn the method; it
bounds it: **an arithmetic rule encodes an objective ("kill what is killable")
that is only correct while the strategic context holds.** Against a wall deck the
objective itself changes — remove the wall, don't collect cheap prizes — and no
amount of correct arithmetic inside the wrong objective helps.

Forward consequence: **the fix is not to drop `chip_target`** (it is worth +0.077
head-to-head in the mirror, which is 52% of the field) **but to branch it on the
matchup** — exactly ROADMAP candidate B3, now promoted from speculation to the
repair for a measured defect.

### Verdict 3: the matchup branch works, and it shipped the same session

`targeting.chip_target(obs, wall_defer=True)` — hand the select back to the net
whenever the opponent's Active is a known damage-prevention Pokemon
(`WALL_POKEMON = {345}`), because there our counters are the only way to remove
it and the net was measured to aim them correctly on its own.

| variant | score vs `rule:crustle`, n=2000 |
|---|---|
| `bc` — unconditional `chip_target` | 0.559 [0.537, 0.581] |
| **`bc:w,wall` — the branch** | **0.663 [0.642, 0.684]** |
| `bc:x,noChip` — rule off entirely | 0.685 [0.665, 0.705] |

**It recovers +0.104 of the −0.126 (82%)**, with an interval disjoint from
unconditional `chip_target`'s and overlapping the "off entirely" ceiling. So the
branch captures nearly all of the available gain while keeping the rule where it
pays.

**Mirror control: 0.521 [0.490, 0.552], n=1000 — contains 0.5, i.e. no bleed.**
This is expected *by construction*: the branch cannot fire unless a card 345 is
the opponent's Active, and neither `grimmsnarl` nor `lucario_v10` nor
`crispin_toolbox` contains one, so `bc:w,wall` and `bc` are behaviourally
identical there. **The control is still worth running** — it is the only thing
that would catch an implementation error that fired the branch when it shouldn't.

**Shipped ON by default** (`chip_wall_defer=True`; `bc:<label>,noWall` restores
the old behaviour). Remaining headroom is the 0.663 → 0.685 gap, i.e. a bespoke
wall-aware *ranker* rather than deferral — small, and explicitly deferred until
something larger is exhausted (HANDOFF §3.3).

⚠ **LB expectation, stated in advance so it cannot be rationalised later:** the
gain is ~+0.10 of score in **18%** of the field, so of order **+10–15 Elo
overall** — **below this instrument's resolution** (§7). **This change must not be
submitted on its own to "see if it works"**; that is the exact error this file
documents twice. Bundle it with further improvements and submit once.

## 8d. Crustle mechanics — the §3.2 premise, verified in-engine

**Hypothesis (never checked before today):** Mysterious Rock Inn, an ABILITY on
Crustle (345), prevents damage from opponent {ex} attacks — so Marnie's Grimmsnarl
ex (`ex=True`) deals **zero** — but Adrena-Brain and Freezing Shroud *place/move
damage counters*, which is not "damage done by an attack", so they should go
through. Our card db exposes no ability text for 345, so only play settles it.

**Method:** `scripts/p3_crustle_probe.py`, 60 games, a census of the engine's own
`type 16` HP-change events, which carry **`putDamageCounter`** (True for
placed/moved counters, False for attack damage), the owner, the card and the
value. A prevented attack logs as **`value: 0`**.

| measurement | n | result |
|---|---|---|
| attack damage onto Crustle | 224 events | **209 ZEROED (93.3%)** |
| the 15 that landed | 15 | **60 damage each — all from Marnie's Morgrem (647, attack 936), a NON-ex attacker** |
| damage counters onto Crustle | 1,386 events | **1,298 landed (93.7%)**, 17,850 total damage / 60 games |
| their Superb Scissors (479) onto our Grimmsnarl ex line | 168 events | **240 every time** (120 doubled by weakness), used 253× |

**Verdict: the premise holds, and there are TWO outs rather than one.**

1. **Damage counters bypass the prevention** — confirmed at n=1,386. The
   passive-damage line is **live**, so ROADMAP Track C steps 3–4 are unblocked.
2. 🆕 **A non-ex attacker is not covered by an anti-{ex} ability.** Marnie's
   Morgrem deals **60** through the wall, and we already run 3 copies as the
   evolution stage. **Nobody had considered this out**; it needs no decklist
   change, only a play-priority rule (don't always evolve Morgrem → Grimmsnarl ex
   into a Crustle board).
   ⚠ **Finding 2 was sized on 2026-07-30 and does not survive it — see §8e.**
   The rule fires ~0.2× per game, the free version of the same out is already
   taken 95% of the time, and the "deals 0" half of the argument is only true of
   their **Active** (below).

**And the matchup's shape in one line:** they deal **240** per attack into our
main attacker; our main attacker deals **0** into theirs. We still win 55.9%,
entirely on damage counters (~298 per game). That asymmetry is why the two
counter-rules pay more here than anywhere else, and why the attack-targeting rule
pays negative.

⚠ **Correction (2026-07-30): "our main attacker deals 0 into theirs" is true of
their ACTIVE only, and reading it as "Shadow Bullet is worthless here" is wrong.**
Shadow Bullet also does 30 to a benched Pokemon, and re-reading the same census
per-target shows **attack damage onto Dwebble: 82 events, mean 73.9, 0 prevented**
against a 70-HP basic. So the attack kills the Crustle line's basics even while
the wall itself takes nothing. This matters because it is the *alternative* every
proposed anti-wall play is measured against (§8e), and the original one-liner
made that alternative look like zero.

⚠ **Instrument note (rule 9, twice in one script).** The first version of this
probe read **0.0 damage in every bucket**, including buckets that cannot be zero.
Cause: **`obs['logs']` is a per-observation DELTA, not a cumulative log** (lengths
`[0, 0, 48, 14, 3, 1, ...]`, non-monotonic), so offset-based attribution was
nonsense. Rewritten as a census needing no attribution. Then the census itself
filed **`value == 0` events with the heals**, hiding the prevented attacks — the
single event class the probe existed to count. **Both bugs were caught only by
checking a bucket whose answer was known in advance.** Always include one.

## 8e. The Morgrem out — CLOSED BY SIZING, before an A/B was spent (2026-07-30)

**Hypothesis (§8d finding 2, and the top-ranked next action at the end of day 8):**
against a Crustle board our {ex} attacker deals 0 while Marnie's Morgrem — a
non-ex Stage-1 we already run 3 of — deals 60. So a play-priority rule *"do not
evolve Morgrem into Grimmsnarl ex into a wall"* should recover real damage at no
decklist cost. It was described in `HANDOFF.md` as "the biggest known lever in the
matchup".

**Method:** `scripts/p7_morgrem.py`, 200 games vs `rule:crustle` piloting
`crustle_v1`, three independent runs. The decision is resolved **per turn, not per
select** — one turn hands us many MAIN selects and the evolve-or-attack question
stays open across all of them, so scoring each select separately (the first
version did) reports mid-turn ability activations as if they resolved the choice.
That is rule 8 read in reverse: multiplicity can *inflate* a denominator as easily
as it hides one. Known-in-advance bucket per rule 9: MAIN selects taken while
their Active is a wall must be *large* vs this opponent — it printed 7,821.

| measurement | 200 games | verdict |
|---|---|---|
| turns where an **armed** Morgrem was Active vs a wall **and** the evolution was in hand — the rule's honest denominator | **91–102** (three runs) | ~0.5/game |
| …of which the clone **took the evolve** (the only turns a veto changes) | **38 / 49 / 53** | **~0.2 firings per game** |
| turns where a Morgrem was Active vs a wall but **could not pay {D}{D}** | 252–257 (**66%** of Morgrem-Active turns) | the out is not even available |
| **post-KO promotion into a wall with a Morgrem available** — the *free* version of the same out, no retreat cost | **288/302 = 95.4%** already promote the Morgrem | **already right** |
| damage healed back off their Crustle | **21,070 of 72,630 (22.5%)** | the 60 is worth ~47 net |
| attack damage onto their **Dwebble** | 82 events, **mean 73.9, 0 prevented** | the alternative attack is not zero |

**Verdict: do not build it, and do not spend the A/B.** Three independent reasons,
any one of which would be enough:

1. **Frequency.** The veto would fire ~0.2 times per game inside the ~18% of the
   field on Crustle. At ~47 net damage a firing against the ~352 damage per game
   we already land on a Crustle, that is a **~2.6%** change in output — and an
   n=2000 arena A/B resolves ±0.021 of win rate. **The instrument cannot see it**
   (the §1 resolution limit, applied to the arena instead of the LB). Measuring it
   honestly would need n≈20,000 per arm.
2. **The free route is already taken.** Promotion after a KO costs nothing, while
   retreating Grimmsnarl ex costs its retreat of **2** — the entire attack
   investment. The clone already promotes an available Morgrem into a wall 95.4%
   of the time. This is the "316/316 lethals, all forced" shape from §8/B2: the
   behaviour the rule wanted already happens wherever it is cheap.
3. **It is a TRADEOFF, not a dominated option** — rule 11's 0-for-4 column. The
   marginal turn is *60 onto a 150-HP wall they heal 22.5% off* versus *30 onto a
   70-HP Dwebble that dies to it, plus 220 more HP of body*. The prize arithmetic
   is a genuine tie (Grimmsnarl ex = 2 prizes and survives exactly two 240s;
   Morgrem = 1 prize and dies to one — 1 prize per hit either way), which is what
   made this look dominated on paper. But "kill their basic and deny the next
   Crustle" versus "chip the current Crustle" is a **judgment about which target
   matters**, and rule 11's ⚠ clause is explicit: one judgment puts a rule in the
   tradeoff column no matter how clean the other dimension looks.

**Interpretation — this is the sizing discipline paying for itself.** The mechanic
is real and was verified in-engine (§8d); what failed is the leap from *"this
mechanic exists"* to *"this mechanic is a lever"*. A dramatic per-instance number
(60 vs 0) says nothing about frequency, and frequency is where this died. Cost:
one probe and no A/B. Compare rule 10 — moving an audit rate is not winning games
— of which this is the earlier-stage cousin: **counting an opportunity is not
finding one.**

⚠ **What is NOT closed by this.** Route 3 was never measured: 451 turns per 200
games (2.3/game) have Grimmsnarl ex attacking a wall for zero on the Active *with
a Morgrem on our bench*. That is 10× the denominator above, but converting it
needs a retreat costing 2 energy, so it is a much worse trade than it looks and it
is a tradeoff besides. Filed, not recommended.

## 8f. B1 — the blind spot was MISDIAGNOSED for the whole project (2026-07-30)

**This entry is the setup for B1's A/B, and the diagnosis alone is a report
result**, because it corrects the sentence the project's entire method rests on.

**The standing claim** (in `targeting.py`'s docstring, `HANDOFF` §4, and every rule
write-up here): *"`optfeat.option_features` gives the net a card-id embedding and
eight positional scalars per option, and **no HP and no damage** — so the net
cannot represent 'this one dies to 30' at all."*

**What is actually true.** `features.py` `_slot_feats` has **always** written, for
each of the 12 board slots: `hp/300`, `maxHp/300`, damage fraction, attached
energy count, own-type energy count, prize value, retreat cost, and
`best_estimated_damage`. **The net has had every HP and damage number on the board
since v1.** The claim as written was false.

**The real gap, and it is narrower and more interesting.** The v2 per-option
vector encoded position only as *area* flags — `area == ACTIVE`, `area == BENCH`,
`area == HAND`, plus the same two for `inPlayArea`. **`opt["index"]` and
`opt["inPlayIndex"]` were never encoded at all.** So:

- two options naming two different benched Pokemon produced **identical** dense
  vectors, separable only by the card-id embedding;
- therefore two options naming **two copies of the same card** were *exactly*
  identical — bitwise indistinguishable inputs with different correct answers.

**That is the mechanism behind both winning rules, and it predicts their sizes.**
`energy_spread` is precisely the two-copies case (bare vs already-loaded
Munkidori) and it is the **largest effect ever measured here (0.702)** — because
there the net was not merely uninformed, it was *unable to represent the
distinction*, and it picked the loaded Munkidori 143 times to 94, **worse than a
coin flip**. `chip_target` (which of their Pokemon dies to 30) is the same defect
one select over.

**So the correct statement of the project's thesis is not "the net cannot see HP".
It is: the net could see the board but could not see WHICH OPTION POINTED WHERE.**
The rules did not supply missing arithmetic so much as supply a missing *binding*.

**The intervention (`optfeat` v3, 25 → 37 columns, appended never inserted):** per
option, the resolved target's presence, HP, maxHP, damage fraction, dies-to-30,
prize value, energy count, own-type energy count, ours-or-theirs, our best damage
into it, can-we-KO-it, and **the slot index** — the last being the one that fixes
the representability failure outright.

**Instrument check before training (rule 9), 20 games / 1,851 multi-option
selects:** every new column varies *within* a select, which is the only kind of
variation that can change a ranking. `slot_index` varies in **72.5%** of selects —
the highest of any column, confirming the diagnosis directly. Known-in-advance
buckets both passed: target resolution is **424/424 = 100%** on chip selects
(these selects name Pokemon, so anything less would be a bug), and own-type energy
varies in **53%** of MAIN selects, i.e. `energy_spread`'s signal demonstrably
exists in the new features.

**Experimental design, and why it is not the obvious one.** Comparing a v3 net
against the shipped `policy_lw2` would confound features with corpus — `pds_v2` is
2,810 games and 8 of its 10 raw replay days were pruned from disk, so it cannot be
rebuilt. Instead: **one corpus at 37 columns (1,603 games, 248,985 rows, from
07-26…07-29), and the control is the same corpus truncated to the first 25
columns** (`--opt-cols 25`). Same games, same selects, same labels, same seed —
**the features are the only difference.** Because the v3 block is appended,
truncation reproduces the v2 layout exactly.

⚠ **A shipping hazard handled, worth recording.** `policynet.load` guards on
feature dims, so bumping `OPT_DENSE` would have made the *shipped* net fail its
own guard and silently fall back to `list(range(minCount))` — i.e. quietly
destroyed the live agent. Fixed by deriving the option width **per net** from
`head_in` and slicing, which also lets a v2 and a v3 net run **in one process**
for the head-to-head A/B rule 4 requires. Regression-checked: `bc` vs
`rule:crustle` reads **0.640 [0.601, 0.677] n=600** against the archived
0.663 [0.642, 0.684] — unchanged.

**Verdict framing, committed BEFORE the A/B ran** (so it cannot be rationalised
after): **v3 wins** → the thesis is confirmed at its root and the hand rules are
*manual patches for a representational hole* the features close directly;
**v3 does not win** → representation was not the binding constraint and the
rule-writing method is vindicated against its best alternative.

### The result (2026-07-30/31): v3 WINS, and it is the largest effect in the project

Seven arena A/Bs, **n=2000 each**, archived in `out/arena/b1_*.jsonl`.

**A. The feature effect — clean.** Same corpus, same rows, same labels, same seed;
the *only* difference is the option-feature width.

| | v3 vs the v2 control | |
|---|---|---|
| rules **OFF** both sides | **0.878** [0.863, 0.892] | the pure feature effect |
| rules **ON** both sides | 0.661 [0.640, 0.682] | the rules carry most of the control's gap |

**0.878 is the largest effect ever measured here** — `energy_spread`, the previous
record, was 0.702. And the collapse from 0.878 to 0.661 when both sides get the
rules *is itself the evidence for §8f's diagnosis*: what the rules were adding is
almost exactly what the v2 features could not represent.

**B. The thesis test — the rules are now ACTIVELY HARMFUL, not merely redundant.**

| | |
|---|---|
| `v3 + rules` vs `v3 alone` (mirror) | **0.427** [0.405, 0.449] |
| `v3 + rules` vs `v3 alone` (vs `rule:crustle`) | 0.773 vs 0.770 — **neutral**, CIs overlap |

In the mirror the CI clears 0.5 comfortably: **three of the four rules that define
this project's method now cost us games.** The mechanism is not mysterious —
`chip_target` *replaces the whole ranking* with fixed arithmetic ("dies to 30,
most prizes, lowest HP"), which was right when the net was **incapable of
representing** which option pointed at which Pokemon, and is wrong now that it can
see target HP, dies-to-30, prize value and our damage into each target. The rule
overrides a better-informed judgment with a cruder one. On the Crustle anchor they
are neutral rather than harmful, which fits: `wall_defer` already hands that exact
matchup back to the net.

**So the rules and the v3 features are ALTERNATIVES, not complements.** Ship v3
with rules off, or ship `lw2` with rules on — never the combination.

**C. Against the shipped agent** (`policy_lw2` + rules), i.e. the shipping question:

| | v3 rules-OFF | v3 rules-ON | shipped `bc` |
|---|---|---|---|
| mirror vs shipped `bc` | **0.661** [0.640, 0.681] | 0.597 [0.575, 0.618] | 0.5 |
| vs `rule:crustle` (adversarial) | **0.770** [0.751, 0.788] | 0.773 [0.754, 0.791] | **0.663** |

0.661 in the mirror is **≈ +115 Elo** — for the first time, an effect *larger than
the LB's ±50–100 resolution* (§1), i.e. the first candidate all project that the
leaderboard could actually adjudicate. **Two anchors, one adversarial, both agree**
— rule 12 satisfied, which is precisely the check `chip_target`'s −0.126 taught us
to run.

**Internal consistency (worth stating, since it is free):** three independent runs
order the three agents transitively — `v3 alone` > `v3 + rules` > `shipped` — via
0.427, 0.597 and 0.661 measured in separate processes. Nothing contradicts.

⚠ **The confound, stated rather than buried.** The **vs-shipped** rows conflate two
changes: the features *and* the corpus. v3 trains on 1,603 games from 07-26…07-29
(which includes the **post-shift** meta) while `policy_lw2` trains on 2,810 older
games. So **do not cite 0.661-vs-shipped as the feature effect** — the clean
feature number is panel A's 0.878, which holds corpus fixed. For a *shipping*
decision the conflation is harmless (we want the better agent either way); for the
report's causal claim only panel A counts. Note also that v3 wins on **43% less
data**, which independently re-confirms §1's "more data is not a lever".

⚠ **Pool cost checked, not assumed.** v3 calls `best_damage` per *option*, and the
A/Bs ran ~2× slower in wall-clock. Measured over 8,000 game-seats: **minimum
599.7 s left of 600, zero games below 300 s.** No timeout risk. (Run 3's archive
did print `WOULD TIME OUT ON KAGGLE`; per the §7 gotcha the distribution was
checked — **1 game in 4,000** at −556 s, p1 = 599.8, median 599.9. Machine sleep,
not the agent.)

**Interpretation — this is the project's central result, and it reverses the
method.** For eight days the working theory was *find decisions the features
cannot express and write a rule for them*, and it was right about the diagnosis
(the net was blind) and wrong about the cure. The blindness was **representational,
not informational** — the net had the HP all along and could not bind it to an
option — and the correct fix was 12 features, not four hand rules. The rules got
~150 LB points because they *simulated* the missing binding; supplied properly, the
binding is worth ~115 Elo more and makes the rules a liability. **The report's
thesis should be stated as: hand-written arithmetic is a proxy for a missing
representation, and it is dominated by fixing the representation.**

## 8g. B1 on the LEADERBOARD — the arena was wrong, and it is now SYSTEMATIC (2026-07-31)

**This is the most important entry in this file.** §8f recorded B1 winning every
local A/B. It was submitted (`55116557`, 2026-07-30 18:14 UTC) and **lost badly.**

| submission | config | LB | age when read |
|---|---|---|---|
| `55072063` **P4b** | lw2 + `chip_target` + `energy_spread` | **952.0** (958 within ~4 h) | 27 h, converged, now **frozen** |
| `55077709` P6a | P4b + `counter_source` | **837.5** | ~2 days, converged |
| `55116557` **v3** | **optfeat v3, rules OFF** | **819.8 → 824.6** | **9.6 h, rising ~5/h** |

**The comparison that matters is at equal age: P4b was at 958 by 4 h; v3 is at 825
at 10 h.** It is not an unconverged reading — it is a different trajectory. The
arena predicted **+115 Elo**; the LB delivered roughly **−130**.

### This is the SECOND time the arena said "better" and the LB said "much worse"

| intervention | arena verdict | LB verdict |
|---|---|---|
| `counter_source` (P6a) | +0.052 vs `rule:crustle`, 0.534 mirror | **837.5 vs P4b's 952.0 — a 114-point gap that has NOT closed in 2 days** |
| **optfeat v3** (B1) | **0.661 vs shipped, 0.770 vs `rule:crustle`** | **825 vs 952** |

⚠ **Day 8 called the `counter_source` gap "two agents converging toward each
other" and closed the question. That was wrong.** The gap was 224 → 123 → and it
has now sat at **114.5 for two more days with both agents converged.** The
"convergence" was P6a finishing its climb, not the two ratings meeting.

**So the arena is not merely noisy — it is BIASED, in a consistent direction, on
the interventions that matter.** Two independent cases, both large enough to
exceed the instrument. That is a property of our measurement, not bad luck.

**The leading hypothesis, and the thing to test next: our anchors are not the
field.** We A/B against the grimmsnarl mirror and `rule:crustle`. In 54 real
ladder games the opponent was:

| archetype | games | our win rate |
|---|---|---|
| **"other"** (nothing we model) | **34 (63%)** | 58.8% |
| Crustle | 13 (24%) | **76.9%** |
| Grimmsnarl mirror | 5 (9%) | 60.0% |
| Crispin toolbox | 1 | — |

**63% of our real games are against decks we have never once A/B'd against, and
the mirror — which every rule was tuned on — is 9%.** An agent optimised on two
anchors that jointly cover a third of the field can win both and lose the ladder.

⚠ **Do NOT read the 0.639 overall replay win rate as vindication.** Win rate is
against rating-matched opponents, so it is ~0.5-seeking by construction and says
nothing about level. Split by episode order it is 0.639 / 0.500 / 0.778 (n=18
each) — no trend, just noise.

### The bundle was NOT broken — checked first, because it would explain everything

A net rejected by the dim guard makes the agent play `list(range(minCount))`
silently (§7). Ruled out from the replays: over **4,682 multi-option selects the
agent picked index 0 only 40.7% of the time** (a fallback agent is 100%). The
shipped net was live. **The regression is play strength.**

### Two concrete defects, found by the user watching games and then quantified

Both audited by `scripts/p8_optv3_replays.py` over the 54 games
(`out/logs/p8_optv3_replays.txt`). ⚠ Both are **decided by the net alone** in this
agent, because v3 ships with every rule off.

**1. Boss's Orders drags away a Pokemon we could have KO'd — 9 of 31 real
choices (29%).** Denominator is honest: 22 of 31 drags happened when their Active
was *not* KO-able, where dragging is free and is excluded.

| | n |
|---|---|
| their Active not KO-able — drag is free | 22 |
| **MISS: could KO the Active *and* snipe a ≤30 HP bench (a DOUBLE KO) — dragged instead** | **5** |
| **MISS: dragged something we cannot KO, abandoning a KO-able Active** | **2** |
| drag traded up or equal — defensible | 2 |

The double-KO miss is the expensive one: **Shadow Bullet does 180 to the Active
*and* 30 to a benched Pokemon**, so with a ≤30 HP sitter on their bench, attacking
takes two prizes and dragging takes one. Real examples:

```
89013303 t11: could KO Archaludon ex hp=170 (2p) + snipe Duraludon hp=10; dragged Duraludon
89015107 t14: could KO Crustle hp=70 (1p); dragged Mega Kangaskhan ex hp=220 which survives
89021174 t9 : could KO Alakazam hp=80 (1p) + snipe Abra hp=20; dragged Abra
```

⚠ **This does NOT license writing a fifth Boss's Orders rule.** §6 closed the card
after four null interventions — but **all four were measured against the lw2 net
with the other rules on**, and this agent is a different net with no rules at all.
The card is re-opened *as a question*, not as a licence.

**2. Freezing Shroud is symmetric and we are usually its bigger victim — but the
clearly-bad case is 11%, not the majority.** Froslass: *"put 1 damage counter on
each Pokémon that has an Ability (both yours and your opponent's), except any
Froslass."* **Our ability Pokemon are Munkidori (×4) and Marnie's Grimmsnarl ex
(Punk Up, ×3)** — our own main attacker is on the clock.

At the 63 moments we chose to evolve into Froslass:

| board at that moment | n | |
|---|---|---|
| they have more ability Pokemon — clock favours us | 26 (41.3%) | good |
| equal | 11 (17.5%) | neutral |
| **we have more, but an armed Munkidori is out** | 19 (30.2%) | **this is the intended engine** — Shroud loads counters onto our Pokemon and Adrena-Brain ships them across |
| **we have more AND no armed Munkidori** | **7 (11.1%)** | **pure self-damage** |

⚠ **The user's instinct is directionally right, but the honest number is 7/63, not
"most".** The 19 "armed Munkidori" rows are the deck working as designed, and
calling them misplays would repeat rule 10 (moving an audit rate is not winning
games). Worst observed: `89012406 t19: ours=4 theirs=0`.

### The good, and it is real

- **Crustle: 10 wins of 13 real games (76.9%)** — and the arena predicted 0.770
  against `rule:crustle`. **The one matchup we actually engineered transferred to
  the field almost exactly.** That is a genuine validation of the day-8 wall work
  *and* of the anchor when the anchor matches the opponent.
- **Morgrem attacked 5 times, 4 of them into a Crustle** — the non-ex out the user
  saw. §8e closed it as too small to build a *rule* for; it happens anyway.

### Interpretation — what this costs and what it buys

**Costs:** a submission slot, the rank (below), and the day-8 conclusion about
`counter_source`. **Buys:** the single most valuable negative result of the
project — *our local arena does not measure ladder strength*, established at
n=2 independent large interventions. Everything downstream of the arena is now
suspect, and **§3.1's re-anchoring was necessary but nowhere near sufficient: two
anchors covering 33% of the field is still a single-anchor error in disguise.**

## 8h. The endgame question is ANSWERED: only ACTIVE agents count (2026-07-31)

`HANDOFF.md`'s submission box flagged this as *"the open question that decides the
endgame"*. It is now settled by observation:

- best-ever submission: **`55072063` at 952.0** — but **inactive**;
- best **active** submission: `55077709` at **837.5**;
- **the leaderboard shows us at 837.5, rank 605 of 5,000.**

**The displayed score tracks the best ACTIVE submission, not the best ever.** We
were rank **224** at 950.2 before the eviction and **605** after it, with no change
in play strength — the drop is purely the frozen agent no longer counting.

**Consequences, and they bind on the whole endgame:**
1. **A frozen high score is worth NOTHING at the close.** The best agent must be
   in the active pair on 2026-08-17 and through the 08-31 continued-play window.
2. **Every submission is now a genuine risk**, not a free option — it evicts, and
   the evicted score stops counting immediately.
3. The day-8 note *"freezing is cheaper than it sounds"* was **wrong** and is
   corrected in `HANDOFF.md`.

## 8i. The anchor set was wrong — but it does NOT explain the ladder gap, and the gap was never as big as reported (2026-07-31)

> ⚠ **This section was rewritten mid-session after the full sweep landed. An
> earlier draft was headed "the arena/ladder gap is SOLVED" and claimed the
> −130 was reproduced locally. That was wrong on the arithmetic, and it was wrong
> because it was written after 2 of 5 anchors instead of 5 of 5.** The retraction
> is in "What this does NOT show" below. The anchor-coverage finding survives; the
> conclusion drawn from it did not.

**Two separate results, and they must not be merged.**

### Result 1 (solid): v3 has a real weakness the old anchor set could not see

Both agents vs a fixed `rule:v10,noS` piloting `lucario_v10`, n=2000 each
(`out/arena/p9_{p4b,v3off}_vs_v10.jsonl`):

| agent | vs `rule:v10` |
|---|---|
| **P4b** = lw2 + chip + spread (`55072063`) | **0.576 [0.554, 0.598]** |
| **v3 rules-off** (`55116557`) | **0.505 [0.483, 0.527]** |

**Confidence intervals disjoint. v3 is −0.071 (≈ −50 Elo) against Mega Lucario**,
which is **12.8% of the field we actually play**, while being **+0.157 in the
mirror**. B1 could not have seen this: the anchor had been retired two days
earlier on the strength of a meta mined from a population 200–320 Elo above ours.

### The full 5-anchor sweep (n=2000 per cell, day 9)

| anchor | share | P4b | v3 | Δ Elo (v3 − P4b) | weighted |
|---|---|---|---|---|---|
| `rule:alakazam5` | **22.0%** | 0.727 [0.707, 0.746] | 0.731 [0.711, 0.750] | **+4** (dead heat) | +0.8 |
| mirror, **head-to-head** | 13.8% | (0.343) | **0.657 [0.636, 0.677]** | **+113** | +15.6 |
| `rule:crustle` | 12.8% | 0.663 | 0.770 | **+92** | +11.8 |
| `rule:v10` | 12.8% | 0.576 [0.554, 0.598] | 0.505 [0.483, 0.527] | **−50** 🔴 | −6.4 |
| `rule:archaludon` | 10.1% | 0.621 [0.599, 0.642] | 0.669 [0.648, 0.690] | **+36** | +3.7 |
| | **71.5%** | | | | **+36 Elo** |

⚠ **Δ Elo is `elo(v3 vs anchor) − elo(P4b vs anchor)` for the fixed-anchor rows.
The mirror row is a head-to-head, so its Δ is `elo(0.657) = +113` directly** —
computing it as `elo(v) − elo(1−v)` doubles it to +226 and inflates the weighted
total from +36 to +57 Elo. Different row types, different arithmetic.

⚠ **A methodology check that nearly went wrong, and then didn't.** §8f's mirror
number (0.661) is v3 vs `policy_b1_ctrl` — a **v2-feature net on the same
`pds_v3` corpus** — not vs P4b (`lw2`, `pds_v2`, rules on). Those are a *feature
ablation* and an *agent comparison*, and mixing them into one weighted table is
invalid. The honest head-to-head had never been run in this project; it was run,
and it reads **0.657** against the 0.661 that was being reused. **The reuse was
in fact harmless — but it was harmless by luck, and the check cost 12 minutes.**

### What this does NOT show — the retraction

**Weighted by field share over the 71.5% we can now measure, the arena says v3 is
≈ +36 Elo BETTER than P4b.** It does not reproduce the regression at all. The
Lucario weakness is real and worth −50 Elo on 12.8% of the field — about **−6 Elo
overall** — and it is swamped by the mirror (+15.6) and Crustle (+11.8) gains.

**And the "−130" itself was an artifact of a comparison this repo's own rule 2
forbids.** `55072063`'s **952.0 is frozen**: it was earned on 07-29 against a
~4,000-entrant board, and the board is now **6,000**. The only same-time,
both-active comparison available is:

| submission | agent | read 2026-07-31 |
|---|---|---|
| `55116557` | **v3, rules off** | **819.8** |
| `55077709` | P6a = lw2 + chip + spread + `counter_source` | **845.0** ⚠ still climbing (824.9 → 837.5 → 845.0) |

**−25 points, against an opponent that has not converged.** §1's own resolution
limit says the LB swings ±50–100 while converging. **So the residual
arena/ladder disagreement is smaller than the instrument can resolve** — the
dramatic contradiction was manufactured by comparing a live score against a
frozen one from a smaller field.

🔴 **Therefore §8g's headline — "the arena is systematically wrong, n=2, both
larger than the instrument" — is WEAKENED, and so is the version of rule 16 that
was written from it.** Both of its instances compared against numbers that were
not comparable: `counter_source` against a converging score (already conceded in
§7), and B1 against a frozen one. **There may be no systematic arena bias to
explain.**

### What to actually conclude

1. ✅ **Keep the rebuilt anchor set.** It found a genuine −50 Elo hole in v3 that
   26.6% coverage could not. That is worth the day regardless of the above.
2. ✅ **Keep the sampling-frame lesson** (below). It is independent of the gap.
3. 🔴 **Stop trying to explain a 130-point regression. Measure a 25-point one
   properly instead** — and note the LB cannot resolve 25 points, so the honest
   answer may be "v3 and P6a are indistinguishable on the ladder".
4. ⚠ **Never compare against a frozen score again.** Rule 2 says it; §8g did it
   anyway; this section nearly did it twice.

### Why the anchor was retired: the mined meta describes a band we never play in

`scripts/fetch_top_episodes.py` sorts each day's manifest by `avg_score` and
keeps the top `--max`. Every meta snapshot in this repo (§8b) is therefore the
**top of the ladder**. Worse, and this is structural rather than a tuning
mistake — **Kaggle's daily episode datasets do not contain our band at all**:

| day | episodes | min `avg_score` | top-400 cutoff |
|---|---|---|---|
| 2026-07-26 | 4,554 | — | 1156 |
| 2026-07-29 | 4,386 | **1055** | 1144 |

**The lowest-rated published episode on 07-29 is 1055. We play at 825–952.**
Buckets 600–800, 800–900 and 900–1000 contain **zero** episodes. So:

> **No amount of episode mining can ever describe the field we face.** The public
> data is censored below ~1055 by construction. Our own submission replays are
> the only evidence that exists about our own opponents.

That is why §8b read "`{F}`/Rock Fighting = `lucario_v10` is **0 of 400 games**"
and concluded the anchor was dead. It was 0% *at 1150+*. In our own games it is
**12.8% of the field** — tied for the largest deck we face.

### The real field, from 109 real ladder games (`scripts/p9_field_census.py`)

Pooled over `replays/submission_optv3` (54 games, the v3 agent) and
`replays/submission_replay_2026-07-29` (55 games, the chip-only agent).
Full table: `out/logs/p9_field_census_pooled.txt`.

| archetype | share | our WR | anchor status |
|---|---|---|---|
| **Alakazam** (Abra/Kadabra, Telepath Psychic) | **22.0%** | 66.7% | ❌ → ✅ imported day 9 |
| Marnie's Grimmsnarl ex (mirror) | 13.8% | 60.0% | ✅ had it |
| Crustle | 12.8% | 57.1% | ✅ `rule:crustle` |
| **Mega Lucario ex** | 12.8% | **50.0%** | ✅ had it, **wrongly retired** |
| **Archaludon ex** | 10.1% | **45.5%** ⚠ worst | ❌ → ✅ imported day 9 |
| *(14 more archetypes)* | 28.4% | — | — |
| **top 5** | **71.6%** | | |

**Before day 9 our anchor set covered 39.4% of the field and excluded both the
largest archetype and our worst matchup.** The B1 decision was taken on the
mirror (13.8%) and Crustle (12.8%) — **26.6%**.

⚠ **Two methodology traps the census hit, both worth carrying forward.**
1. **Naming a deck by its highest-prize Pokemon is wrong.** A single
   Fezandipiti ex run as a draw tech made 18 Alakazam games read as a
   "Fezandipiti ex" archetype. **A card the deck runs one copy of is a tech, not
   an identity** — score archetypes on multiples only.
2. **Naming a deck by the card you happened to see fragments one deck into
   three.** A short game may only ever reveal the Kadabra. The census resolves
   every Pokemon to its evolution line via `evolvesFrom` before counting; without
   that, Abra/Kadabra/Alakazam read as three separate 11% archetypes and the
   field looks far flatter than it is. Both fixes together moved the #1
   archetype's share from 16.5% to 22.0% and cut 28 "archetypes" to 19.

### What this does and does not overturn

- ✅ **Rule 16's diagnosis stands and is now quantified** — the arena is accurate
  where the anchor resembles the opponent, silent elsewhere. §8g already had the
  positive control (arena predicted 0.770 vs Crustle; we won 76.9% of 13 real
  Crustle games). This adds the negative one.
- 🔴 **§8b's "the meta shifted" is retracted as a claim about OUR meta.** What
  shifted was the top-1150 band. Our band's mix over two dumps is different
  again (Lucario 20%/5%, Alakazam 13%/31% between the two), so **treat even this
  census as a snapshot with n≈50 per dump.**
- 🔴 **"`chip_target` is harmful" was measured vs Crustle only** (§8c). It is
  still a matchup branch, but the branch was chosen from a 2-anchor set.
- ⚠ **v3 is not refuted as a net.** It wins big on 26.6% of the field and loses
  on 12.8%. Nothing yet measures it on the other 60.6%. **Do not discard it and
  do not reship it** until the 5-anchor sweep is complete.

### The anchors added (`scripts/import_field_agents.py`)

Both pilots were sitting checked into `notebooks/` unused for the whole project:

| anchor | source | why it is a good anchor |
|---|---|---|
| `rule:alakazam5` + `decks/alakazam5.py` | `rule-based-not-psychic-alakazam-best-5th` | author reports **5th place**, pure rules, no ML, no search — the strongest pilot in the repo, on the field's biggest deck. Its 60 matches the field's **engine** exactly (checked below) |
| `rule:archaludon` + `decks/archaludon_ex.py` | `a-sample-archaludon-75-wr-vs-my-1300-starmie` | our worst matchup. Also a **second damage-reduction deck** (Full Metal Lab: −30 into any Metal Pokemon), which `targeting.WALL_POKEMON = {345}` does not model |

⚠ Rule 12's competitiveness clause was checked, not assumed: at n=30 smoke we
score 0.633 vs `rule:alakazam5` and 0.733 vs `rule:archaludon` — real opponents,
not the 0.911 ceiling that made a `bc`-piloted Crispin anchor useless.

### Calibration: the arena ranks matchups right and reads ~15 pp optimistic

The only external check available — v3's arena score per anchor against **the
same agent's real win rate** on the same archetype (`replays/submission_optv3`):

| archetype | arena (v3) | real WR | n | mean opponent rating |
|---|---|---|---|---|
| Crustle | 0.770 | 70.0% | 10 | 713 |
| Alakazam | 0.731 | 57.1% | 7 | 779 |
| Archaludon | 0.669 | 40.0% | 5 | 787 |
| Mega Lucario | 0.505 | 36.4% | 11 | 735 |

**The rank order is exactly right, 4 for 4, and the level is ~13–27 pp
optimistic.** So: **trust the arena for A/B deltas and for ordering matchups;
do NOT read an arena score as a predicted win rate.** The optimism is expected —
these pilots are public notebooks, not the players we meet.

### ⚠ The census describes the pool we were MATCHED against, not our own band

`p9_field_census.py --lb` joins each opponent to a leaderboard snapshot. Over the
54 v3 games, opponents average **759** while that agent sat at **819.8** — we
were matched *down*, and only 17% of opponents outrated us. **Every share and win
rate in this section is a property of that pool.** Per archetype it makes two
matchups look far worse than the raw WR suggests:

- **Mega Lucario: 36.4% against opponents averaging 735 — 85 points BELOW us.**
  That is not "a hard matchup", it is losing to weaker players, and it is the
  same deck the arena independently flags at −50 Elo. **Two independent
  instruments, same conclusion: this is the matchup to fix.**
- **Archaludon: 40.0% vs 787.**
- Crustle's healthy-looking 70.0% is against the **weakest** pool we faced (713).
- The mirror is the bright spot: **60.0% against opponents averaging 871**, the
  only pool that outrated us.

**How closely does the Alakazam pilot's 60 match the field's?** Checked against
the 24-game reconstruction rather than asserted — **16 of 23 observed cards match
or exceed, and the mismatches are all tech slots, never the engine**:

- ✅ **exact on everything that defines the deck**: Abra ×4, Alakazam ×3,
  Kadabra ×3→4, Dunsparce ×3, Dudunsparce ×2, Fezandipiti ex ×1, Shaymin ×1,
  Telepath Psychic Energy ×4, Poké Pad ×4, Buddy-Buddy Poffin ×4, Dawn ×4,
  Hilda ×3→4, Basic {P} ×2, Sacred Ash ×1, Enriching Energy ×1.
- ⚠ **pilot runs fewer**: Rare Candy 4→3, Night Stretcher 3→1, Boss's Orders 3→2.
- ⚠ **absent from the pilot**: Xerosic's Machinations ×2, Nighttime Mine ×2,
  Lana's Aid ×1, Neutralization Zone ×1 — note **`Nighttime Mine` is a stadium
  the field plays and this pilot does not**, so a stadium-contest effect will be
  under-read through this anchor.
- The pilot additionally runs Enhanced Hammer ×3, Lucky Helmet ×3, Genesect,
  Psyduck, which we never saw (invisible cards are expected — reconstruction is
  a lower bound).

⚠ **And the reconstruction pools 24 games from 24 different players**, so it is
the *union* of several variants, not one canonical list. Some "absent" cards are
almost certainly one opponent's tech, not a slot this pilot is missing. **Do not
"fix" the pilot's deck toward the reconstruction** — it was tuned for its own 60,
and `rule:crustle`'s ~20 fallback-scored cards on the consensus list is the
standing warning about what that costs (§8c).

## 8j. Does v3 want the hand rules back on? — IN FLIGHT, verdict deliberately blank

> ⚠ **This entry is open on purpose.** §8i was written after 2 of 5 anchors and
> had to be retracted the same session. The amended process rule (ROADMAP Track B)
> is: **if runs are in flight, log the numbers and leave the verdict blank.**
> 2 of 4 cells are in. **Do not act on this section yet.**

**The question.** v3 shipped with all three hand rules **off**, justified by
`v3+rules` vs `v3 alone` = **0.427** (§8f). That measurement was taken **in the
mirror — 13.8% of the field.** §8i's whole lesson is that a mirror-only result
does not generalise, so the rules-off decision was never actually tested.

**Cells so far** (`bc:v3on,net=out/policy_b1_v3.npz` = v3 with chip + spread +
src + wall all on, i.e. the defaults; n=2000 each):

| anchor | share | v3 rules OFF | **v3 rules ON** | Δ | P4b (reference) |
|---|---|---|---|---|---|
| `rule:alakazam5` | 22.0% | 0.731 [0.711, 0.750] | **0.739 [0.719, 0.758]** | +0.008 (overlaps) | 0.727 |
| `rule:v10` | 12.8% | 0.505 [0.483, 0.527] | **0.572 [0.550, 0.594]** | **+0.067** 🟢 | 0.576 |
| `rule:crustle` | 12.8% | 0.770 | **0.761 [0.742, 0.779]** | −0.009 (overlaps) | 0.663 |
| `rule:archaludon` | 10.1% | 0.669 [0.648, 0.690] | ⏳ running | | 0.621 |
| mirror (h2h vs `v3 alone`) | 13.8% | — | 0.427 (§8f) | **−0.073** 🔴 | — |

**What the first cell shows, stated narrowly:** *the rules are what close the
Mega Lucario hole.* Rules-off v3 reads **0.505** there; rules-on reads **0.572**,
disjoint CIs, and that is **level with P4b's 0.576**. So §8i's "−50 Elo weakness"
is **not a property of the v3 net** — it is a property of having turned the rules
off, and it was invisible because the decision to turn them off was made in the
mirror.

⚠ **The cells point opposite ways** — rules ON wins Lucario (+0.067) and loses
the mirror (−0.073), while Alakazam and Crustle are **dead heats**. **That is the
signature of a matchup branch (B3), not of a global on/off setting**, which is
exactly the shape that paid +0.104 against Crustle (§8c).

⚠ **A prediction of ours failed here and it is worth recording.** We expected
rules-on to lose *badly* to Crustle, because `chip_target` measured **−0.126**
there (§8c). It did not — 0.761 vs 0.770, overlapping. **The reason is that the
`wall_defer` branch is already ON by default in `bc:v3on`**, so the harmful
behaviour it was predicted to reproduce has been branched away since 07-30. The
lesson is narrow but real: **"rule X was harmful vs anchor Y" expires the moment
X is modified.** Re-read the flag defaults before predicting a cell.

**Reproducer:**

```powershell
python -X utf8 scripts/arena.py play "bc:v3on,net=out/policy_b1_v3.npz" `
    "rule:v10,noS" --deck-a grimmsnarl --deck-b lucario_v10 --matches 1000 `
    --archive out/arena/p10_v3on_vs_v10.jsonl
```

## 9. Deck stewardship so far (feeds Deck Score — see ROADMAP Track C)

- **The list is an exact 60 seen 290× in one day's top episodes**, and the net is
  trained on it — so *every* variant is off-distribution for the policy as well
  as untested. That is the standing cost any decklist change must pay.
- **Spikemuth Gym is played ~100% by both sides — do not cut it.**
- **Team Rocket's Petrel (×4) tutors any Trainer**, so Trainer *access* is
  already there; adding copies of a tutorable card buys much less than it looks.
- **Munkidori is already at 4, the copy cap.** Only the Froslass line (2 Snorunt
  / 2 Froslass) can grow — which substantially weakens "run more passive damage"
  as a plan.
- The one variant measured scored **0.490** [0.468, 0.512] n=2000 (§8).

---

## 10. Standing caveats on this corpus of evidence

- **`replays/submission_replay_2026-07-29/` is `55054446`'s games** — the
  chip-only agent, **before** `energy_spread` — despite the folder's date.
  Any measurement there that depends on *two armed* Munkidori is understated,
  because `energy_spread` is precisely the rule that arms the second one.
  **Always check which agent produced a replay dump before using it.**
- **`55054446` is the standing warning about LB readings.** Day 6 recorded
  "916.8 → 936.0 → **979**, three readings, trending up" and wrote a plan against
  it. It settled at **905.2** — below its own first reading. A rising score is
  unconverged, not momentum.
- **Do not resurrect:** the arena→LB ladder anchored on `rule:iono`; the old deck
  sweep's ranking; "the clone is comfortably above the rule baseline"; every
  n=24 number and every strength claim dated before 2026-07-27 pm (measured
  through stale nets silently rejected by dim guards, a compute knob that could
  not bind, and a mirror matchup compared against cross-deck runs); "3× compute
  made it worse" (`SA_SPEND_MULT` only grants time, and time was never binding).
- **Pre-shift meta snapshot** for the "the meta moved" figure:
  `out/meta/pre_shift_0722_0724.txt` (mined 2026-07-30 from 07-22 + 07-24,
  the last days before the reported shift; the raw JSONs for those days were
  pruned afterwards, per-day `manifest.csv` kept in `replays/manifests/`).
