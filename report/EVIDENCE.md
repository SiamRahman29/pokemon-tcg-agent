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

> 🔴 **RETRACTED 2026-08-01 (day 14) — THIS IS NOT A MEASUREMENT AND IT HAS BEEN
> FILED AS ONE FOR TWELVE DAYS.** The paragraph above has no `n`, no CI, no
> command and no code behind it. **Self-play RL was never run.** It was dropped
> on a *compute prior*, inherited from the **search** result — a different
> experiment — and then propagated into `ROADMAP.md` §0/§8/appendix,
> `STRATEGY.md` §8's negative-results list and `HANDOFF.md` §6 as though it sat
> beside the measured negatives. **Rule 15, third instance**, and the first one
> where the unmeasured claim was sitting inside the file whose entry rule is
> "every number traces to an archived run with n and a CI".
>
> ✅ **Checked the old repo too** (`E:\Kaggle\pokemon-tcg-simulation`, which this
> project inherited from): **no RL code, no training script, no self-play loop,
> no reward function anywhere outside `.venv`.** Its only learned artifact is
> `artifacts/policy.pt` (359 KB, 2026-07-05) with no script that trains it.
> **So the claim has no experimental basis in either repository.**
>
> **What IS measured, and must not be laundered into an RL verdict:**
> - **search** = 0.323, n=31 — and the diagnosis is about *rollout variance*
>   (terminal 0/1 ⇒ SE ≈ 0.14; the max over ~9 rivals sits 0.21–0.28 above truth
>   by chance). §2 above.
> - **`--winners-only`** = 0.375, n=2000 — outcome-*filtering* other people's
>   games, discarding half the corpus. §1.
> - **§8w's gradient argument** — real, but **narrowed by §8x the next day**: the
>   bitwise-tie ceiling is **95.6%** against a clone at 71%, so the encoding
>   binds at most 4.4 pp, and the ties that exist are two copies of one card in
>   one role (free choices). §8w also named the feature audit as RL's
>   *prerequisite*; that audit has since been done twice (§8y/§8z, §8ab).
>   **⇒ the gate §8w set is substantially satisfied.**
>
> ⚠ **The honest status is therefore "never attempted", not "dead".** The live
> objection is neither compute nor expressiveness — it is **credit-assignment
> variance**, the same term that killed search: a ~40-turn game with hundreds of
> selects and one binary terminal reward. **That is a sizeable quantity and it
> gets sized before anything is built** (rule 14). ⛔ **Until it is sized, do not
> cite this section as evidence against RL, and do not cite it as evidence for
> RL either.**

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
turns. The card is closed. Do not write a fifth.**

> ### ⚠ A FIFTH WAS WRITTEN (2026-07-31), AND IT IS ALSO NULL — but the reason is new and it is the useful part
>
> The user watched real games and said: *"if we just knew how to play Boss's
> Orders correctly we should have converted a few wins."* **The observation was
> correct and the four rules above genuinely did not cover it.** They answered
> *which* Pokemon to drag and *whether the drag itself converts*; none compared
> the drag against **the attack we already had**.
>
> `targeting.boss_prize_veto` does: it suppresses the play when attacking now
> takes **strictly more prizes** than any drag, pricing both branches honestly
> (a drag can double-KO too, so the dragged Pokemon is excluded from its own
> snipe) and firing only on strictly greater. **It is in the DOMINATED column** —
> both branches are prize counts and damage-vs-HP, no judgment — which is rule
> 11's 3-for-3 side. It should have won.
>
> **Corroboration was strong before any A/B was spent:** 29% of 31 real ladder
> drags were misplays (`p8_optv3_replays.py`), the rule fires on **28.6%** of
> Boss's Orders plays in the arena, and `p14_prize_audit.py` re-found the same
> defect by a third, independent route.
>
> | anchor | share | plain `bc` | `bossPrize` | Δ Elo |
> |---|---|---|---|---|
> | `rule:alakazam5` | 22.0% | 0.728 | 0.736 [0.716, 0.755] | +7 |
> | mirror (head-to-head) | 13.8% | — | **0.505 [0.483, 0.527]** | +3 |
> | `rule:v10` | 12.8% | 0.583 | 0.590 [0.569, 0.612] | +5 |
> | | 48.6% | | | **+6 Elo** |
>
> **All three CIs overlap. Weighted: +6 Elo — indistinguishable from zero and
> far below the LB's ±50–100.**
>
> **🔴 THE REASON, AND IT IS THE THING TO CARRY FORWARD: the play is too RARE to
> matter, and this was computable before writing any code.** From the same 54
> ladder games:
>
> - drags where attacking was a genuine alternative: **0.57 per game**
> - misplays among them: **0.17 per game**
> - prizes actually thrown away: **~0.09 per game = 1.5% of a 6-prize game**
>
> An n=2000 A/B resolves ±0.021. **A 1.5% effect is below the instrument** — the
> identical arithmetic that closed the Morgrem out at ~2.6% (§8e).
>
> ⚠ **Process failure, recorded because it is the point of rule 14.** Rule 14
> says *size before you build*. This was built first and sized afterwards. The
> sizing takes two minutes and would have predicted the null exactly. **Being
> right about the defect is not the same as the defect being worth fixing**, and
> a vivid per-instance error is exactly the shape that fools you into skipping
> the frequency check.
>
> **So the card really is closed, and now we know WHY rather than just that it
> is: five interventions, five nulls, because Boss's Orders decides ~0.09 prizes
> a game.** The rule stays in `targeting.py` behind `bc:<label>,bossPrize`,
> default off, as the record. (The pair at 0.452 is only
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

## 8j. Does v3 want the hand rules back on? — ✅ CONCLUDED: globally NO (+1 Elo), and the value is in a BRANCH

> ✅ **5 of 5 cells reported before this verdict was written.** The entry was held
> open with the numbers logged and the verdict blank for three intermediate
> reports, per the process rule amended earlier the same day after §8i was
> published at 40% of its data. **The discipline changed the answer twice** — at
> 2 cells the story was "the rules fix everything", at 4 cells it was "a matchup
> split", and only at 5 is the global figure (**+1 Elo**) visible.

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
| `rule:archaludon` | 10.1% | 0.669 [0.648, 0.690] | **0.684 [0.663, 0.704]** | +0.015 (overlaps) | 0.621 |
| mirror (h2h vs `v3 alone`) | 13.8% | — | 0.427 (§8f) | **−0.073** 🔴 | — |

### The verdict, weighted

| anchor | share | Δ Elo (on − off) | weighted | CIs |
|---|---|---|---|---|
| `rule:alakazam5` | 22.0% | +7 | +1.6 | overlap |
| mirror (h2h) | 13.8% | **−51** | **−7.1** | **disjoint** |
| `rule:crustle` | 12.8% | −9 | −1.1 | overlap |
| `rule:v10` | 12.8% | **+47** | **+6.0** | **disjoint** |
| `rule:archaludon` | 10.1% | +12 | +1.2 | overlap |
| **global rules-on** | **71.5%** | | **+0.6 → +1 Elo** | |

**1. Turning the rules on globally is worth NOTHING (+1 Elo).** The mirror loss
(−7.1) almost exactly cancels the Mega Lucario gain (+6.0), and the other three
anchors are dead heats. **So shipping v3 with rules off was not a mistake — it
was arbitrary.** The 0.427 that justified it was a mirror-only number and
therefore bad evidence, but it happened to land on a defensible config.

**2. Only two of five cells have disjoint CIs**, and they are the two that point
opposite ways. **The three "dead heat" rows must not be summed as if they were
signal** — done naively, a branch that switches the rules on wherever Δ is
positive scores +12 Elo, but +4 of that is noise from three overlapping cells.
**The honest branch value is the Lucario cell alone: +6.0 weighted ≈ +8 Elo.**

**3. ⚠ And ~8 Elo is below what we can validate.** §1's resolution limit: the LB
swings ±50–100 while converging, so **this can never be confirmed on the
leaderboard**. By rule 14 the sizing is now done and the honest read is: *a
Lucario rules-branch is a legitimate BUNDLE candidate, not a solo submission,
and not urgent.*

**4. What it does buy is a report chapter** — B3's third instance, after the
Crustle wall branch (+0.104) and the Lucario finding itself. The pattern
"arithmetic rules are matchup-conditional, and the condition is readable off the
board" is the opponent-modelling argument, and this is its cleanest example: the
same three rules are **+47 Elo** in one matchup and **−51** in another.

**The one robust positive, stated narrowly:** *the rules are what close the
Mega Lucario hole.* Rules-off v3 reads **0.505** there; rules-on reads **0.572**,
disjoint CIs, and that is **level with P4b's 0.576**. So §8i's "−50 Elo weakness"
is **not a property of the v3 net** — it is a property of having turned the rules
off, and it was invisible because the decision to turn them off was made in the
mirror. ⚠ **This does NOT mean the rules should go back on** — see the weighted
verdict below; globally they are worth +1 Elo.

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

## 8k. ✅ The three-way sweep: every agent we have is within 36 Elo, i.e. below the instrument (2026-07-31)

**The question:** should we spend a submission restoring P4b (`55072063`, the
frozen 952.0)? Answered in the arena instead of on the ladder, so it cost no
submission and no eviction.

**All three agents, all five anchors, n=2000 per cell. Elo relative to P4b.**
The mirror column is a head-to-head against P4b, so P4b is 0 there by definition;
every other column is `elo(agent vs anchor) − elo(P4b vs anchor)`.

| agent | Alakazam 22.0% | mirror 13.8% | Crustle 12.8% | Lucario 12.8% | Archaludon 10.1% | **weighted** |
|---|---|---|---|---|---|---|
| **v3** (rules off, `55116557`) | +4 | **+113** | **+92** | **−50** | +36 | **+36** |
| **P6a** (lw2 + 3 rules, `55077709`) | +1 | **+24** | +0 | +5 | +6 | **+7** |
| **P4b** (lw2 + chip + spread, `55072063`) | 0 | 0 | 0 | 0 | 0 | **0** |

Raw scores for the P6a row (the new ones): Alakazam **0.728 [0.708, 0.747]**,
Lucario **0.583 [0.561, 0.604]**, Archaludon **0.629 [0.608, 0.650]**. Crustle
(**0.663**) and the mirror (**0.534**) were already at n=2000 and were re-verified
from `out/arena/anchor_rulecrustle_wall.jsonl` and `out/arena/ab_src.jsonl`
rather than copied from the docs.

### The verdict, and it is a strategic one

**1. The whole spread is 36 Elo, and the leaderboard resolves ±50–100.**
So **the instrument that decides the competition cannot tell these three agents
apart.** Everything the project has shipped since `chip_target` sits inside one
noise band.

**2. Therefore: do NOT restore P4b.** It is the *weakest* of the three by the
arena, and a restore would cost a submission, **evict `55077709` (845.0, our best
active and still climbing)**, and restart at μ=600 for ~4 h — to install an agent
the arena ranks last, on the strength of a **frozen** 952.0 earned on a board
2,000 entrants smaller (§8i). **The premise of every earlier version of that plan
was a frozen-vs-live comparison.**

**3. The active pair {v3, P6a} is already the arena's top two.** No action needed.

**4. `counter_source` is independently vindicated a second time.** P6a beats P4b
by **+24 Elo in the mirror** (0.534) and **+7 weighted**, which is the same
direction as §8c's +0.052 vs Crustle, measured a different way. Day 8's decision
to keep the rule was right.

**5. ⚠ The uncomfortable implication, and it should steer the remaining 17 days:**
if our best and worst current agents differ by 36 Elo and the LB cannot see 36
Elo, then **no further rule-sized improvement can move the leaderboard.** The
remaining levers are the ones big enough to clear the band — a materially better
net, or ROADMAP **B4** (turn-level sequencing with the 599.9 s we never use).
**Another targeting rule is a report chapter, not a rank.**

⚠ **Do not read this as "the arena disagrees with the LB" again.** The LB order
is P4b 952.0 > P6a 845.0 > v3 819.8 and the arena order is the reverse — but
952.0 is frozen and not comparable, and the two live agents differ by 25 points
against an unconverged opponent. **Both instruments are saying "these are close";
only the frozen number makes it look otherwise.**

## 8l. B4's cheap probe — it SURVIVES all three kill criteria, and the decisive question is still unmeasured (2026-07-31)

**B4 (ROADMAP §2.5): spend the unused pool enumerating our own turn's action
sequences and score end-of-turn states with `evalfn` — no rollouts, no opponent
determinization, so §2's terminal-0/1 variance does not arise.** It is the last
unstarted breakthrough candidate and, after §8k, **the only lever left that could
clear the 36-Elo band the leaderboard cannot resolve.** Probe run before any
build, per rule 14.

### 1. Is there a decision at all? ✅ passes (`p12_b4_probe.py`, 992 of our turns)

| | turns | share |
|---|---|---|
| every option forced | 40 | 4.0% |
| exactly ONE real select | 337 | 34.0% (greedy is optimal by definition) |
| **≥2 real selects** | **615** | **62.0%** ← the honest denominator |

Median **6** real selects per turn, **4** of them MAIN. **Unlike the Morgrem out
(§8e), this candidate has a real denominator** — rule 13 does not kill it.

### 2. Is the space tractable? ⚠ exhaustive NO, beam YES

Naive sequence count (product of option counts, an upper bound):
**median 98,122,752**, and **64.7% of turns exceed 1M**. Exhaustive enumeration
is dead.

But throughput, measured rather than assumed (`p12b_step_bench.py`, 36 begins /
432 steps on real mid-game observations):

| | median | note |
|---|---|---|
| `fastsearch.step` | **0.130 ms** (**7,698/s**) | one per action per candidate |
| `fastsearch.begin` | 0.78 ms | one per turn — negligible |

**600 s pool ÷ 9.9 of our turns per game ≈ 60 s/turn ⇒ ~78,000 candidate
sequences per turn.** That is 0.08% of the median space — and a perfectly
ordinary beam width. **Throughput is NOT the binding constraint**, which is the
opposite of what we expected going in.

### 3. Can `evalfn` tell good states from bad? ✅ real signal (`p12c_evalfn_signal.py`, 200 games)

AUC of end-of-our-turn `evaluate()` against the eventual game result:

| turn | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.47 | 0.41 | 0.58 | 0.69 | 0.74 | 0.78 | 0.80 | **0.87** | 0.83 | 0.91 | 0.90 | 0.92 |

**Early (≤8) mean AUC 0.685; late (>8) 0.901.** Turns 0–1 are noise, as they
should be — nothing has happened yet. From turn 3 the eval is genuinely
discriminative. **`evalfn` is not the broken component**, and the value net's
failure (§1) does not transfer to it.

### 🔴 The verdict, and the caveat that matters more than the three passes

**B4 survives its cheap probe on all three criteria.** That is a real result: it
is now the best-supported unstarted candidate in the project.

⚠ **But the quantity that decides B4 is NOT the one measured above, and it is
strictly harder.** AUC-vs-result asks *"can `evalfn` tell a winning board from a
losing board, across different games?"* A sequencer asks *"can `evalfn` rank
twenty end-of-turn states reachable from the SAME position this turn?"* Those
states share almost all their structure — same deck, same prizes, same opponent
board — so the eval differences are tiny compared with the between-game spread
this AUC is built on. **A high across-game AUC is compatible with zero
within-turn discrimination.**

**This is the same error class as §8i**: measuring on a convenient population
rather than the one the decision will be taken over. We are not going to make it
twice in one day.

**Next probe before any build (cheap, ~1 h):** from real turns, enumerate k≈20
legal sequences via `fastsearch.step`, score each end-of-turn state, and report
**the within-turn eval spread against the eval's own noise**, plus how often the
`argmax` differs from the clone's choice. **If candidates within a turn score
nearly identically, B4 dies there** — and no beam width or pool budget can save
it.

⚠ And rule 3 stands: five times a metric that looked good failed to predict
playing strength. **Nothing above licenses a build; it licenses the next probe.**

## 8m. B4's decisive probe: the eval DOES rank within a turn — and a pre-registered criterion had to be corrected (2026-07-31)

§8l named the missing measurement: not "can `evalfn` tell a winning board from a
losing one across games" (AUC 0.685/0.901) but **"can it rank end-of-turn states
reachable from the SAME position"**. `scripts/p12d_within_turn_signal.py`,
**93 turns, K=16 candidate sequences, M=8 determinizations, full K×M matrix**:

| quantity | value |
|---|---|
| between-candidate sd (determinization-averaged merit) | **0.111** |
| residual candidate×determinization sd (does NOT transfer) | 0.055 |
| SNR at **M=1**, `sqrt(between/resid)` | **1.05** |
| between-sd vs **SE of a candidate mean at M=8** | **5.7×** |
| **split-half top-1 agreement** (chance 6.2%) | **62.0%** (n=150) |
| best − median candidate, per turn | 0.099 eval units ⚠ upward biased |

**Verdict: the eval ranks real, transferable merit within a turn.** The argmax
chosen on one half of the determinizations wins on the *other, independent* half
**62% of the time against 6.2% chance** — ten times chance. That is not
determinization luck, which is what killed the rollout search (§2).

### 🔴 A pre-registered criterion was changed after seeing data. Read this before trusting the verdict.

**The criterion written into the script before running it was: "SNR ≤ ~1.5 or
top-1 agreement near chance ⇒ kill". SNR came out at 1.05, which would have
killed B4. It was not killed, and the criterion was changed to agreement-only.**
That is exactly the move that should attract suspicion, so the reasoning is here
in full and the reader is invited to disagree:

1. **The FIRST baseline was wrong by construction.** It compared spread across
   candidates (one determinization) against spread of one candidate across
   determinizations. A new determinization changes what everybody draws, moving
   **all candidates together** — a common-mode shift that inflates "noise" while
   leaving the *ordering* untouched. It reported SNR 1.05 **and** agreement 66.7%
   vs 6.2% chance in the same run: a contradiction that only resolves if the
   baseline measured level noise, not rank noise. **Fixed** by scoring the full
   K×M matrix and doing a two-way decomposition, which removes common mode.
2. **The corrected SNR is still ~1.05, and that is NOT a repeat of the same
   error — it is a different one.** `sqrt(between/resid)` compares merit against
   **single-observation** noise, i.e. the M=1 operating point. **A sequencer does
   not run at M=1.** `p12b` measured ~78,000 affordable rollouts per turn, so it
   averages, and averaging shrinks noise as `1/sqrt(M)` while leaving true merit
   alone. At M=8 the ratio is already **5.7×**.
3. **So the metric was mis-specified, not the result inconvenient.** SNR-at-M=1
   answers "could a single-sample chooser rank these?" — a question B4 never
   asks.

⚠ **Judge it on the agreement number, which was co-pre-registered and never
moved: 62.0% against 6.2%.** If that had come out near chance, B4 would be dead
regardless of any of the above.

### What this does and does not license

- ✅ **B4 is the first breakthrough candidate to clear every cheap gate**
  (denominator 62% of turns, ~78k candidates/turn affordable, eval ranks within
  turn) and, after §8k, the only lever left that could clear the **36-Elo band
  the leaderboard cannot resolve**.
- ⚠ **The gain estimate is upward biased and small.** 0.099 eval units per turn
  is a **max over 16 noisy estimates** — the same selection bias that flattered
  the dead search. The true per-turn gain is smaller, and it is in "rough
  prize-card units" against games decided by 6 prizes.
- 🔴 **Rule 3 stands: five metrics have looked good here and not paid.** Nothing
  above is evidence of playing strength. **The next step is a prototype plus an
  arena A/B at n≥1000 against the five anchors, with pool-usage logging** — and
  the pool is a real risk (exhausting 600 s is a loss, §7).

## 8n. B4's prototype LOSES — 0.075 at n=40, and rule 3 claims its sixth victim (2026-07-31)

`agents/sa/sequencer.py`, wired as `bc:<label>,seq` (opt-in, default off).
Head-to-head vs plain `bc` in the mirror:

| version | score vs `bc` |
|---|---|
| first build | **0.083** [0.015, 0.354] n=12 |
| after the mid-turn-eval fix | **0.075** [0.026, 0.199] n=40 |

**This is not a marginal loss — it is a rout, and it is the exact signature of
the dead rollout search: the sequencer overrules the clone on 50% of MAIN
selects** (`EVIDENCE` §2: the search overruled 52% and scored 0.323).

### What has been ruled out

1. **Option-index mismatch — NO.** The simulated root's option list was compared
   field-by-field against the real select's over four positions: identical
   (`type`, `area`, `index`, `playerIndex`, length).
2. **Mid-turn vs end-of-turn comparison — REAL BUG, FIXED, DIDN'T HELP.** The
   first build evaluated any candidate that hit the step cap *mid-turn*, still
   holding its cards, against candidates that had attacked and committed.
   `evalfn` scores the uncommitted state higher, making "stall" the winning
   move — and the agent duly played ~50% more turns per game. Candidates that do
   not actually end the turn are now discarded. **Score moved 0.083 → 0.075,
   i.e. not at all.**

### The live hypothesis, and it is a DESIGN flaw rather than a bug

**B4 maximises the value of the board at the end of OUR turn, which structurally
cannot see the opponent's reply.** A line that leaves our 2-prize attacker
exposed scores well right up to the moment it is knocked out. The clone, trained
on 2,810 human games, prices that risk implicitly; the sequencer discards it.
`evalfn` does carry a `−0.45 × prize_value` term for being KO-able, but it is one
term among many and the argmax over K candidates will happily pay it.

⚠ **And the selection bias is the §2 failure in a new costume.** §8m measured
that the argmax *reproduces* across determinizations (62% vs 6.2%) — that it is a
**stable** choice. It never showed the argmax is a **good** choice. A max over K
noisy estimates of a biased objective is stably wrong.

### Verdict

**Not "B4 is dead" and not "B4 needs one more fix".** The state is: the
prototype exists, two candidate bugs are eliminated, and the remaining
explanation is that the objective itself is wrong. Options, cheapest first:

1. **Test the design flaw directly** — score candidates with a one-ply opponent
   reply (their best attack into our board) instead of end-of-our-turn value.
   That is a small change to `_rollout`'s terminal evaluation.
2. **Constrain rather than replace the clone** — only overrule when the eval gap
   exceeds a threshold, turning a 50% overrule rate into ~5%. The dead search's
   `PRIOR_BONUS` was the same idea and it did not save that search.
3. **Kill it.** 17 days remain; §8k says no rule-sized change can move the LB
   anyway, and B4's own gain estimate (0.099 eval units/turn) was upward biased.

🔴 **Rule 3, sixth instance: a metric that looked good did not predict playing
strength.** §8m's 62%-vs-6.2% was real and reproducible and the thing built on
it lost 37 of 40 games. **The rule now reads: five training metrics and one
search metric.**

## 8o. ✅ THE DECK IS NOT THE BOTTLENECK — 52.1% of the 1144+ band plays our exact archetype (2026-07-31)

**The question ("is our deck as good as it could possibly be?") is answerable
from data we already hold, and mining is the RIGHT tool here** — §8i's warning is
that mined episodes describe the ≥1055 band, and *that is precisely the band this
question is about.*

Re-classified the 400 top episodes of 07-29 (800 seats, `avg_score` ≥ 1144) with
the line-aware census classifier:

| archetype at ≥1144 | seats | share |
|---|---|---|
| **Marnie's Grimmsnarl ex — OUR DECK** | **417** | **52.1%** |
| Crustle | 96 | 12.0% |
| Mega Kangaskhan ex | 68 | 8.5% |
| Meowth ex | 56 | 7.0% |
| Teal Mask Ogerpon ex | 38 | 4.8% |
| Team Rocket's Mewtwo ex | 37 | 4.6% |
| Alakazam | 32 | 4.0% |

**Over half of every seat in the band ~320 points above us is playing the deck we
play**, and `decks/grimmsnarl.py` is card-for-card the consensus 60 (seen 353×).

**Therefore the ~320-point gap between our 846.6 and the top's 1169 is a PILOTING
gap, not a deck gap.** No decklist change can be worth 320 points when the people
at 1169 are on our list. This closes the question the ROADMAP has carried since
07-30 as an open risk.

**Consequences, and they reorder the whole plan:**

1. 🔴 **Track C's deck experimentation is NOT a rank lever.** It remains worth
   doing for **Deck Score (20% of the rubric)** and the stewardship narrative —
   "we measured a change and kept the list" is deck analysis — but it should stop
   being described as the counter-meta fix for our ceiling.
2. ✅ **It also retires a standing worry cheaply**: we are not playing a stale or
   fringe list, and the field did not abandon it. The 52.2% → 47.5% win-rate
   fall (§8b) is the field learning to beat the *archetype*, and everyone at the
   top absorbed that and kept playing it.
3. ⚡ **The lever is imitation quality.** Our corpus is 2,810 games mined from
   *these* top episodes — we are cloning 1144+ players and playing at 846.
   `context_accuracy.py` says the clone disagrees with its demonstrators on
   **33.9%** of decisions, **3,930 of 6,424 misses in MAIN alone**. **B1 is the
   only intervention that ever moved that** (a representational fix, +36 Elo
   weighted, §8k) — and it was found by reading the feature code, not by
   guessing.

## 8p. The P4b restore: the 952 was a BOARD-SIZE artifact, and the LB says so itself (2026-07-31, day 10)

**Hypothesis under test (HANDOFF item 1, reopened by the user):** `55072063`'s
**952.0** was the best number this project ever produced, and three days of
argument turned on whether it was a better agent or a stale reading. It was
resubmitted as `55129730` — **the same verified tarball, the same code.**

| submission | agent | score | age when read |
|---|---|---|---|
| `55072063` | P4b, submitted 07-29 04:45 | **958.2** | ~4 h |
| **`55129730`** | **P4b restored, submitted 07-31 06:46** | **833.9** | **4.0 h** |

**Read at matched submission age — 4.0 h in both cases — the identical agent is
124 points lower.** The board grew from ~4,000 to **6,024** entrants between the
two runs. `55072063`'s later readings only fell (958.2 → 970.1 → 948.1), so the
restore is not obviously mid-climb toward 950.

Full active state, read 2026-07-31 10:46 UTC (`competition_submissions` +
the leaderboard snapshot, which agree):

| submission | agent | score | state |
|---|---|---|---|
| `55129730` | P4b restore | **833.9** | active — **our displayed score** |
| `55116557` | v3, rules off | 818.1 | active |
| `55077709` | P6a | 841.5 | **evicted, frozen** |
| `55072063` | P4b original | 952.0 | evicted, frozen |

**We are rank 632 of 6,024.**

**Verdict: ✅ the restore question is CLOSED, and §8k is now confirmed on the
ladder rather than only in the arena.** Every agent we own reads 818–842 when
played at the same time against the same board. The 952 was never 100 points of
play strength; it was a smaller denominator. **Do not reopen this.**

⚠ **Rule 2 is only half-satisfied:** 833.9 is one reading (the earlier one, 715.9,
was mid-climb). **Re-read it next session ≥1 h later before quoting it.** The
matched-age comparison above is the part that does not depend on convergence.

⚠ **The general lesson, and it is worth more than the submission:** a rating is a
statement about a population, not about an agent. **A score earned against a
different-sized board is not comparable to a current one — in either direction.**
This file previously used that fact to argue the restore *down* (§8i) and the
user correctly pushed back that it cut both ways. The experiment settled it.

### ✅ Read again a day later (2026-08-01) — and the ORDER has resolved, in the arena's favour

Both submissions are now fully converged and both were active the whole time,
which is the only comparison rule 2 allows:

| submission | agent | 07-31 | **08-01** |
|---|---|---|---|
| `55116557` | **v3, rules off (B1)** | 818.1 | **864.1** ⬆ |
| `55129730` | P4b restore | 833.9 | **824.3** |

🔴 **v3 is now our best agent on the ladder by 40 points, and the day-9 panic is
fully inverted.** B1 "shipped and lost 130 points" (§8g); the true story is that
it was compared against a frozen score on a smaller board and had not converged.
✅ **The arena predicted this exactly: §8i's five-anchor sweep put v3 at +36 Elo
over P4b, weighted by field share, and the ladder now says +40 points.** Two
independent instruments, same sign, near-identical magnitude.

**This is the strongest calibration evidence in the project** — better than the
Crustle positive control, because it is a *ranking of two of our own agents*
made by the arena before the ladder had separated them. It is also why the v4
sweep (§8z) is trusted enough to act on.

⚠ **And it retires a claim three files were carrying:** "our three agents read
833.9 / 818.1 / 841.5, all within the noise" was true on 07-31 and is not the
right summary now. §8k's *conclusion* (the spread is small) survives; the
specific numbers do not. **Re-read before quoting, every time.**

## 8q. Expert demonstrations: agreement with the demonstrator FALLS as the demonstrator gets better (2026-07-31, day 10)

**⚠ STATUS: the measurements below are concluded and reproducible. The
INTERVENTION they motivate (expert / rating-weighted cloning) is NOT run — no net
has been trained and no arena A/B exists. Per ROADMAP's amended process rule the
numbers are logged and the verdict is deliberately left blank.**

### The asset

The user supplied two targeted replay dumps — **not** mined top episodes, but
every recent game of one named team, with an `episodes_meta.json` sidecar
carrying `submissionId` per seat:

| dump | team | LB | games | their submissions (games, WR) |
|---|---|---|---|---|
| `replays/sixth_sense_31-07-2026` | Sixth Sense (teamId 16452116) | **#3, 1152.4** | 227 | `55115972` (162, 66.0%), `55128220` (69, 69.6%) |
| `replays/ntumlnoob_31-07-2026` | 李秉叡（ntumlnoob） | **#2, 1162.8** | 330 | `55076771` (305, 64.3%), `55133032` (29, 65.5%) |

⚠ **Both dumps mix two of that team's submissions**, i.e. two different agents.
The WR difference is inside noise in both cases (Sixth Sense Δ3.6 pp, SE≈6.6), so
both were kept — **but rows should be tagged by `submissionId` so this is
ablatable.** Not yet done.

⚠ **`info.TeamNames` in the JSON and `teamName` in the metadata can disagree**:
the Sixth Sense dump reports "Raja Biswas" on 113 games and "Sixth Sense" on 30.
**Same teamId — the team renamed mid-window.** A census keyed on the display name
splits one demonstrator in two.

**Two facts verified before any of the analysis below:**

1. **Both experts play our exact 60.** Reconstructed from play + discard, both
   resolve card-for-card to `decks/grimmsnarl.py`'s `DECKLIST`
   (10/4/4/4/4/4/4/4/3/3/3/3/2/2/2/1/1/1/1). **Identical list, +310 rating** —
   a far cleaner instrument for §8o's "the gap is piloting" than the band
   argument was.
2. **The replays carry `selected` for the third-party seat**, so the whole BC
   pipeline works on them unchanged. `build_policy_dataset.py` gained
   `--player` / `--players-file`.

### The measurement

`context_accuracy.py --all-rows` scores the **live v3 net**
(`out/policy_b1_v3.npz`) against each demonstrator's actual choices. The
`--all-rows` flag is new and is *required* here: the `gid % 20` split is the
trainer's, and these corpora were never trained on at all.

```powershell
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds_expert `
    --player "Raja Biswas" --player "Sixth Sense" replays/sixth_sense_31-07-2026
python -X utf8 scripts/context_accuracy.py --net out/policy_b1_v3.npz `
    --ds artifacts/pds_expert --all-rows
```

| context | control: 48 other grimmsnarl pilots | Sixth Sense (#3) | ntumlnoob (#2) |
|---|---|---|---|
| MAIN | 64.6% | 58.3% | **48.1%** |
| TO_HAND | 67.4% | **35.9%** | 57.2% |
| DAMAGE_COUNTER | 75.3% | 76.7% | **61.4%** |
| REMOVE_DAMAGE_COUNTER | 73.8% | 78.8% | 62.4% |
| DAMAGE | 81.2% | 75.6% | **64.5%** |
| SWITCH | 72.0% | 69.8% | 57.9% |
| TO_ACTIVE | 89.4% | 87.8% | 78.5% |
| ATTACH_FROM | 89.1% | 88.2% | 89.5% |
| **rows / overall miss** | **10,088 / 27.2%** | **18,296 / 34.4%** | **25,775 / 40.1%** |

**The headline: top-1 agreement falls monotonically with demonstrator rating —
27.2% → 34.4% → 40.1% miss.** Our clone looks most like mid-1100 pilots and least
like the #2 player. Random baselines are equal across groups (23.6–23.8% in
TO_HAND), so it is not an option-count artifact, and every cell has n ≥ 1,000.

**And the two top players diverge in DIFFERENT places** — Sixth Sense almost
entirely in `TO_HAND` (−31.5 pp vs control), ntumlnoob broadly across MAIN,
damage placement and switching (−16.5 / −13.9 / −14.1) but only −10.2 in
`TO_HAND`. **So there is no single "expert move" to copy.** Sixth Sense's fetch
behaviour is real but is *not* what makes a 1163 player.

### Two explanations killed, one left standing

**❌ Familiarity ("we predict the players we trained on") — refuted.**
Per-player agreement against seats contributed to `artifacts/pds_v3`:

| player | LB | seats in our corpus | TO_HAND agreement |
|---|---|---|---|
| Sixth Sense / Raja Biswas | 1152 | 0 / 34 | 31% (n=779) / 38% (n=1,709) |
| Dominic Peel | 1136 | **238** | 55% (n=168) |
| やる気元気ミワハルキ | 1126 | 9 | 69% (n=93) |
| カントー地方マスター | 1098 | 6 | 66% (n=61) |
| **haggle** | 1092 | **0** | **75% (n=61)** |

`haggle` contributed **zero** games and is predicted at 75%; Dominic Peel
contributed 238 seats and sits at 55%. ⚠ n is small (61–168) for every row but
the experts — treat the ordering, not the values.

**❌ "One team's idiosyncrasy" — refuted by the ntumlnoob dump**, which was
fetched specifically to break this tie. Both top players are unpredictable to the
clone; only the location differs.

**⚠️ NOT killed, and it is the main threat to the whole reading: COVARIATE
SHIFT.** Agreement is measured on the *demonstrator's own* trajectory
distribution. A strong pilot reaches board states our clone rarely occupies, so
some of that 40% is BC's classic compounding-error problem rather than a policy
we could copy. **Low agreement does not prove they play better than us. Only an
arena A/B can.**

### The structural finding this produced, which is bigger than the dumps

**Our corpus is ALREADY elite, and it is dominated by a handful of teams.** Seats
in the 1,603-game `pds_v3` corpus, against those teams' current ratings:

| demonstrator | LB score | seats |
|---|---|---|
| flg | 1125.1 | **527** |
| Dries @ Tufa Labs | 1101.9 | **490** |
| Dominic Peel | 1135.7 | 238 |
| James Cox (+ & Henry Chao) | 1166.1 | 229 + 185 |
| LiamK | 1127.6 | 216 |

**We clone 1100–1166 play and score 833.9.** So "get better demonstrators" is not
obviously the lever — the demonstrators are already ~300 points above our result.
**What has never been tried is cloning ONE policy instead of a ~50-pilot
mixture.** Every net this project has trained targets the modal action of a
mixture, and the best players are the furthest from that mode (the table above is
a direct measurement of it). That is a textbook mode-averaging failure and it is
newly testable: 330 games from #2 and 227 from #3.

### What follows — pre-registered before anything is trained

1. **Tag every corpus row with its demonstrator's LB rating.** Team names are in
   every replay and the full 6,024-row leaderboard downloads in one call; this
   turns the 3-point trend above into an agreement-vs-rating curve over all 1,603
   games and enables (3).
2. **Single-expert clone** — fine-tune v3 on `artifacts/pds_ntum` (27,318 rows)
   and on ntum + Sixth Sense. ⚠ 27k rows vs the corpus's 248,985: underfitting is
   the expected failure, so fine-tune rather than train from scratch, and early-
   stop on a held-out expert split.
3. **Rating-weighted clone** on the full corpus — keeps all 249k rows while
   fixing mode-averaging. **The most likely of the three to work, and original
   enough to be a report chapter on its own.**

**Bar for all three, set before the first run: +50 Elo weighted across the five
anchors, or it is a chapter and not a submission** (§8k). ⚠ And note the standing
prior against this family: **three axes of more/better training have measured
null or negative** (§1), and the only intervention that ever moved the clone was
representational (§8f). **This is a different axis — demonstrator *selection*
rather than data volume — but the prior is not friendly and the bar is not
negotiable.**

## 8r. §8q survives a much harder test — and changes shape: agreement PEAKS at 1050–1100 and falls in BOTH directions (2026-07-31, day 11)

**⚠ STATUS: the measurements are concluded. The intervention (B7 training) is
in flight and its verdict is deliberately blank below.**

§8q rested on three points from three different dumps, so **rating was
confounded with dump, date, deck and collection process**, and it flagged one
untested alternative (covariate shift). Day 11 tagged every corpus row with its
demonstrator's LB score, which lets rating be varied *inside* one collection
process. Three tests followed, and the first two overturn parts of §8q.

### The tooling (item A of the day-11 plan)

`build_policy_dataset.py --ratings out/lb/pokemon-tcg-ai-battle.zip` writes a
per-row `rating`, `opp_rating`, `team_id` and `sub_id`. Coverage on the four
day-dumps is **94–98% of seats**; the residue is one three-person team that has
no submission on the 07-31 board at all, so NaN is the honest value.

Getting there required two fixes that are themselves findings:

- **A replay's `TeamNames` is a display name, and it is not stable.** Naive
  name-matching left **24.6%** of d26 seats unrated, and **182 of those 198
  misses were one team** — the LB's **#1**, appearing as `James Cox` and as
  `zoroark190` (a member username) because the team merged and renamed.
  Matching member usernames exactly plus a hand-verified alias file
  (`replays/team_aliases.tsv`) took coverage to **98.0%**. **A census keyed on
  the display name silently splits the single most valuable demonstrator into
  three** — §8q hit the same bug on Sixth Sense / Raja Biswas.
- **`--exclude` exists because the control population contained `Scio`.** The
  experts played *us*; a "control demonstrator" list built from their opponents
  therefore included our own agent, whose selects are what the net was fitted
  to. Left in, it inflates the control's agreement with rows that are not
  evidence of anything.

### Test 1 — inside the training corpus, the rating curve is FLAT

`p15_rating_curve.py`, v3 net, `artifacts/pds_v3r` (= `pds_v3` re-tagged;
**248,985 rows / 1,603 games, identical**), scored on the trainer's held-out
`gid % 20` split:

| demonstrator rating | rows | games | top-1 | 95% CI | miss |
|---|---|---|---|---|---|
| 900–1000 | 392 | 6 | 74.0% | [0.694, 0.781] | 26.0% |
| 1000–1050 | 643 | 9 | 69.7% | [0.660, 0.731] | 30.3% |
| 1050–1100 | 1,851 | 23 | 72.3% | [0.703, 0.743] | 27.7% |
| 1100–1150 | 7,935 | 71 | 69.1% | [0.681, 0.701] | 30.9% |
| **1150+** | 1,868 | 28 | **70.0%** | [0.678, 0.720] | 30.0% |

Row-weighted over 12 demonstrators: **−0.03 pp of agreement per +100 rating**
(r = −0.36). **No decline.** And two players at effectively the same rating sit
9 pp apart — `James Cox & Henry Chao` (1166.1) at **68.8%** versus ntumlnoob
(1162.8) at **59.9%** — which is larger than anything the rating axis explains.

### Test 2 — familiarity is REFUTED, this time with a real zero

§8q's familiarity refutation rested on `haggle` at n=61. This one has n=22,768
and an actual unexposed group. `--seen-from` joins each demonstrator to the
number of rows of theirs **in the training split**:

| rows of this demonstrator the net trained on | rows scored | teams | mean rating | top-1 |
|---|---|---|---|---|
| 1–500 | 229 | 3 | 1026 | 70.3% |
| 500–2,000 | 904 | 8 | 1043 | 72.6% |
| 2,000–8,000 | 4,164 | 13 | 1092 | 70.3% |
| 8,000+ | 7,392 | 6 | 1128 | 69.3% |
| **0 — never seen (same-deck control)** | **22,768** | **87** | **1063** | **73.6%** |

**Exposure buys nothing.** Demonstrators the net trained on 40,000 rows of are
predicted no better than 87 players it has never seen. ⚠ The first four rows are
held-out *games* of seen players; the last row is a different dump, so this is
not a single controlled contrast — but the direction is unambiguous and it is
the opposite of the confound.

### Test 3 — the decisive one: same dump, same date, same deck, all-unseen players

The control population is now **reproducible**: `p9_field_census.py --us <team>
--emit-players` was generalised to census any named seat's opponents and write
out the ones on our archetype. Over both expert dumps the census names **89
teams**; excluding ourselves leaves **87 demonstrators / 266 games / 23,726
rows** (**22,768** single-choice and therefore scoreable), all on `Marnie's
Grimmsnarl ex` — 46.9% of what the two experts play against — all with **zero**
training exposure, spanning 700–1140.

| demonstrator rating | rows | games | top-1 | 95% CI | miss |
|---|---|---|---|---|---|
| <900 | 1,288 | 17 | 66.7% | [0.641, 0.692] | 33.3% |
| 900–1000 | 1,879 | 24 | 75.0% | [0.730, 0.769] | 25.0% |
| 1000–1050 | 2,740 | 33 | 75.1% | [0.734, 0.767] | 24.9% |
| **1050–1100** | **8,915** | **105** | **76.1%** | [0.752, 0.770] | **23.9%** ← peak |
| 1100–1150 | 7,946 | 87 | 70.9% | [0.699, 0.719] | 29.1% |
| Sixth Sense, 1152.4 | 18,296 | 227 | 65.6% | [0.649, 0.663] | 34.4% |
| ntumlnoob, 1162.8 | 25,775 | 330 | 59.9% | [0.593, 0.604] | 40.1% |

**This is the result.** Every row is the same net, the same deck, the same two
dumps, the same week, and no demonstrator the net has ever trained on:

1. **§8q's decline above 1100 is REAL and is not a dump artifact** — it is
   already visible *within the control alone*, 76.1% → 70.9%, CIs disjoint,
   before either expert is added. The two experts then extend the same slope.
2. **But it is not monotone in rating — it is a PEAK.** Agreement falls just as
   hard below 900 (66.7%) as it does at 1150 (65.6%). §8q's "falls monotonically
   as the demonstrator gets better" is **narrowed: our clone models the modal
   policy of the 1050–1100 band and is worse at everything on either side.**
   That is the signature of mode-averaging over a mixture, which is what B7
   claims — but the claim is now about *distance from the mode*, not *skill*.
3. ⚠ **Consequence for the "experts are better, copy them" reading:** low
   agreement is a distance measurement, not a quality one. A 780-rated
   demonstrator is also 33% unpredictable to us and nobody wants to clone them.
   **Only an arena A/B can say the experts' policy is better** — §8q's covariate
   shift caveat therefore stands unretracted, and this test does not address it.

### A fourth finding, from the sidecars: a "demonstrator" is a SUBMISSION, not a person

`sub_id` splits each dump by the agent that actually played it:

| team | submission | rows | games | top-1 | 95% CI |
|---|---|---|---|---|---|
| Sixth Sense | `55115972` | 12,991 | 160 | **67.0%** | [0.662, 0.678] |
| Sixth Sense | `55128220` | 5,305 | 67 | **62.2%** | [0.609, 0.635] |
| ntumlnoob | `55076771` | 23,892 | 303 | 59.9% | [0.593, 0.606] |
| ntumlnoob | `55133032` | 1,883 | 27 | 58.8% | [0.566, 0.610] |

**One team's two agents differ by 4.8 pp with disjoint CIs** — as large as a
whole rating band in the table above. ⚠ **So "agreement with player X" has a
shelf life measured in days**, and pooling a dump that spans an upload mixes two
policies. Sixth Sense also scores **74.8%** (n=313) on their 07-26–29 games
inside `pds_v3r` against **65.6%** on their 07-30–31 dump: same player, same
net, **+9 pp**, because the older games are a different agent.

### What this licenses, and what it does not

- ✅ **B7 keeps its premise**: there is real, measured distance between our
  clone and the top band, it is not familiarity, not deck, not dump, not date.
- ❌ **The "rating-weighted clone is the most likely of the two to work" ranking
  (§8q) is weakened.** Our corpus's demonstrators run 801–1166 with a **median
  of 1125** — already past the agreement peak — and Test 1 says the net fits all
  of them about equally. There is less mode to un-average than §8q assumed.
- ⚠ **Anything that reads a rating as a quality axis needs the peak, not the
  slope.** Written down before the training runs so it cannot be retrofitted.

Reproduce:

```powershell
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds_v3r/d26 `
    --ratings out/lb/pokemon-tcg-ai-battle.zip replays/2026-07-26   # +d27..d29
python -X utf8 scripts/p15_rating_curve.py --net out/policy_b1_v3.npz `
    --ds artifacts/pds_v3r
python -X utf8 scripts/p9_field_census.py --dir replays/sixth_sense_31-07-2026 `
    replays/ntumlnoob_31-07-2026 --us "Sixth Sense" --us "Raja Biswas" `
    --us "李秉叡（ntumlnoob）" --emit-players out/ctrl_players.txt
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds_grimm_ctrl_r `
    --ratings out/lb/pokemon-tcg-ai-battle.zip --players-file out/ctrl_players.txt `
    --exclude Scio replays/sixth_sense_31-07-2026 replays/ntumlnoob_31-07-2026
python -X utf8 scripts/p15_rating_curve.py --net out/policy_b1_v3.npz `
    --ds artifacts/pds_grimm_ctrl_r --all-rows --seen-from out/logs/pds_v3r_seen.json
```

Logs: `out/logs/p15_rating_curve_v3.txt`, `out/logs/p15_ctrl_samedate.txt`.

## 8s. Covariate shift is NOT the explanation — the expert clone is a genuinely different policy (2026-07-31, day 11)

**The objection §8q could not answer, answered.** Agreement with a demonstrator
is always measured on that demonstrator's own trajectories, so a strong pilot
reaching states we rarely occupy inflates disagreement without either policy
being better. The discriminator (`scripts/p16_policy_disagree.py`) stops
comparing each policy to *human labels* and compares **the two policies to each
other, on both state distributions**.

Policies: `A` = v3 (`out/policy_b1_v3.npz`, live), `B` = v3 fine-tuned on
ntumlnoob (`out/policy_b7_ntum.npz`). States: `artifacts/pds_ours` (our own
agent's 54 ladder games, `replays/submission_optv3`) and `artifacts/pds_ntum_r`.

| state distribution | rows | **A ≠ B** | A ≠ human | B ≠ human |
|---|---|---|---|---|
| our own games | 4,476 | **26.7%** | **1.7%** | 27.2% |
| ntumlnoob's games | 25,775 | **31.9%** | 40.1% | 19.4% |

**Disagreement does not collapse on our own states — 26.7% vs 31.9%. That is
near-symmetric, so the difference is a real policy difference and not an
artifact of being measured off-support.** Covariate shift is therefore not the
explanation for §8q, and the question "is their policy better?" is a legitimate
one for the arena to answer (§8t).

Three supporting details:

- **A ≠ human is 1.7% on our own games — the positive control the earlier
  measurements lacked.** `replays/submission_optv3` *is* submission `55116557`,
  which *is* the v3 net, so a correct pipeline must reproduce those choices
  almost exactly. It does. Every disagreement number in §8q/§8r is therefore
  measuring policies, not a scoring bug.
- ⚠ **`B ≠ human` on ntumlnoob's games (19.4%) is IN-SAMPLE and must not be read
  as the fine-tune's accuracy** — B trained on 95% of those rows. The honest
  held-out figure is **32.8%** (val top-1 0.672). The `A ≠ B` column is the one
  this section rests on, and `pds_ours` is fully out-of-sample for both nets.
- **The difference is concentrated exactly where §8q said the misses live:**
  MAIN (38.7% of our states / 45.8% of theirs), then DAMAGE_COUNTER (20.5% /
  30.5%) and TO_HAND (19.7% / 23.6%). `REMOVE_DAMAGE_COUNTER_COUNT` and
  `ACTIVATE` are 0.0% — the two policies are identical wherever the decision is
  forced, which is a second sanity check on the instrument.

Log: `out/logs/p16_shift_ntum.txt`.

## 8t. 🔴 B7's rating-weighted arm FAILS its pre-registered bar — and loses outright (2026-07-31, day 11)

**Hypothesis (§8q, pre-registered):** every net we have trained targets the modal
action of a ~50-pilot mixture, so weighting each row by its demonstrator's LB
rating should fix mode-averaging while keeping all 248,985 rows. §8q called it
"the most likely of the three to work".

**Build.** `train_policy.py --rating-temp T` weights row *i* by
`exp((rating_i − max) / T)`, normalised to mean 1 so the effective step size
matches the unweighted control. **T was chosen by a stated rule before training:
the most aggressive reweighting that keeps effective sample size above 100,000
rows.** T=25 gives ESS **102,180 of 248,985 (41.0%)**, weights spanning
[0.0000, 4.04] — still 4× the single-expert corpus. The 8,740 unrated rows (3.5%)
are held at the median rating rather than dropped or given weight 1.

```powershell
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v3r --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 `
    --rating-temp 25 --out out/policy_b7_rw25.npz
```

Everything else is the control's recipe verbatim, and `artifacts/pds_v3r` is
`pds_v3` re-tagged — **248,985 rows / 1,603 games, identical** — so the *only*
difference between treatment and control is the per-row weight. Best val top-1
**0.6844** at epoch 10 (and 0.6850 restricted to 1120+ demonstrators).

### The result

| A/B | share | n | `rw25` | v3 (control) | Δ | verdict |
|---|---|---|---|---|---|---|
| **mirror, head-to-head** vs v3 | 13.8% | 2,000 | **0.421** [0.400, 0.443] | (0.579) | **≈ −55 Elo** | 🔴 CI excludes 0.5 |
| `rule:alakazam5` | 22.0% | 2,000 | 0.721 [0.700, 0.740] | 0.731 [0.711, 0.750] (§8i) | **≈ −7 Elo** | dead heat, CIs overlap |

**It does not merely miss the +50 Elo bar — it loses, by more than the bar's
width, against the net it was built to improve.** Seat-balanced in both cells
(mirror 0.431 as P0 / 0.410 as P1; Alakazam 0.741 / 0.700), so neither is a seat
artifact.

⚠ **Rule 16 was honoured, and then the sweep was stopped on arithmetic rather
than on a hunch.** The mirror is 13.8% of the field and may not close a question
alone, so the largest anchor (22.0%) was run as the adversarial second: it is a
dead heat. **The remaining three anchors were not run, and here is why that is
not a shortcut:** with −55 Elo on 13.8% and −7 on 22.0%, clearing **+50
weighted** would require the other 35.7% of the field to average **≈ +160 Elo**
— larger than any effect this project has ever measured, from a net that is
strictly worse on the 35.8% already sampled. **Stating the arithmetic is the
honest way to stop; "it looked bad in the mirror" would not have been.**

### Why this is a useful negative rather than a wasted day

**§8r predicted it, and said so before the run.** The premise of the
rating-weighted arm was that strong demonstrators sit far from the fitted mode.
§8r measured the opposite *inside this corpus*: the net fits its 1150+
demonstrators (70.0%) as well as its 900–1000 ones (74.0%), the row-weighted
slope is **−0.03 pp per +100 rating**, and the corpus's median demonstrator is
already **1125**. There was little mode to un-average, and upweighting a band
the net already fits cost it 59% of its effective data for nothing.

**This is the fourth axis of "more/better training" to measure null or
negative** (§1: more data, winners-only, best-val-accuracy; now
demonstrator-weighting). The only intervention that has ever moved this clone
remains representational (§8f). ⚠ **The standing prior was stated in §8q before
the run and it was right; the bar was set before the run and it was not moved.**

## 8u. 🔴 B7 IS CLOSED, AND THE WAY IT FAILED IS THE FINDING: arena strength tracks agreement with the FIELD, and moves INVERSELY to agreement with the expert (2026-07-31, day 11)

**Both arms ran. Both failed the pre-registered +50 Elo bar, and both lost
outright.** Arm 2 is the single-expert fine-tune §8q ranked second:
`train_policy.py --init out/policy_b1_v3.npz --ds artifacts/pds_ntum_r --lr 2e-4
--epochs 30`, best val top-1 **0.672** at epoch 22 (agreement with ntumlnoob's
held-out games rose **59.9% → 67.2%**, so the fine-tune worked *as imitation*).

### The joint table — three nets, one row each, and it is monotone

| net | what it optimises | miss vs **the field** (held-out `pds_v3r`) | miss vs **ntumlnoob** | mirror vs v3, n=2,000 |
|---|---|---|---|---|
| **v3** (live) | the mixture | **30.2%** | 40.1% | — (control) |
| `rw25` | mixture, weighted to high ratings | 32.0% | **40.2%** | **0.421** [0.400, 0.443] ≈ −55 Elo |
| `b7_ntum` | one 1163-rated expert | **36.2%** | 19.4% *(in-sample; 32.8% held out)* | **0.370** [0.349, 0.391] ≈ **−92 Elo** |

**Read the first and last columns together.** Field disagreement goes
30.2% → 32.0% → 36.2%; arena Elo goes 0 → −55 → −92. **Every step away from the
field's modal policy costs strength, in order, with no exception** — and the net
that best imitates the #2 player is the weakest agent we have built.

### The mechanism, and it is not the one that was pre-registered

§8q expected arm 1 to underfit and arm 2 to be limited by 27k rows. The
measured mechanism is different and sharper:

- **The rating-weighted net moved agreement with the expert by 0.1 pp —
  40.1% → 40.2%, i.e. not at all — while costing 1.8 pp of field agreement.**
  It paid the full price of discarding 59% of its effective sample and bought
  *nothing* in the intended direction. **§8r predicted exactly this** (the
  corpus's median demonstrator is already 1125 and the net fits its strongest
  demonstrators as well as its weakest: −0.03 pp per +100 rating), so this is a
  pre-registered prediction confirmed, not a post-hoc story.
- **The single-expert net genuinely learned the expert** (held-out agreement
  +7.3 pp, and §8s shows it is a real, symmetric policy difference — not
  covariate shift) **and got worse anyway.** Imitating one strong player at the
  cost of 6 pp of field agreement is a losing trade at every point measured.

### What this closes, and the one thing it does not

⛔ **B7 is closed. Do not build a third demonstrator-selection variant.** Five
axes of "more/better training" have now measured null or negative — more data,
winners-only, best-val-accuracy, rating-weighting, single-expert selection —
against exactly one intervention that ever worked, which was **representational**
(§8f). ⚠ **That asymmetry is the project's most reproducible result and it should
govern how the remaining days are spent.**

⚠ **The honest limit on the claim.** Our arena's anchors are field-like by
construction (the mirror opponent *is* the field clone, and the rule anchors are
field archetypes), so "closer to the field wins here" is partly what the
instrument is built to reward. §8i's calibration is the defence — the arena
ranked 4/4 real matchups correctly and predicted 0.770 against Crustle where we
then won 76.9% of 13 real games — but it is a defence, not a proof. **What would
settle it is a ladder submission of `b7_ntum`, and that is explicitly NOT worth a
slot**: it would evict a live agent to test a net measured at −92 Elo, against an
LB that resolves ±50–100 (§8k). **The negative result stands on the arena, and
the report should say so in exactly these terms.**

Archives: `out/arena/b7_rw25_vs_v3_mirror.jsonl`,
`out/arena/b7_rw25_vs_alakazam5.jsonl`, `out/arena/b7_ntum_vs_v3_mirror.jsonl`.
Training logs: `out/logs/b7_rw25_train.txt`, `out/logs/b7_ntum_train.txt`.

## 8v. 🔴 B4 IS DEAD — the design diagnosis was RIGHT, the fix recovered most of the rout, and it still loses (2026-07-31, day 11)

§8n left B4 in an honest but unresolved state: the prototype scored **0.075
[0.026, 0.199] n=40**, two candidate *bugs* had been eliminated without moving
it, and the remaining explanation was that the objective itself is wrong —
maximising the value of the board **at the end of OUR turn** structurally cannot
see the opponent's reply. §8n listed testing that directly as option 1.

**Built** (`agents/sa/sequencer.py`, `reply=True`, exposed as `bc:<label>,seq,
reply`): after our turn ends the simulation continues through the **opponent's
whole turn**, piloted by the same clone on the same determinization, and the
candidate is scored when control returns to us. A game that ends inside the
simulation is scored on the **result** (±1e4), not on the wreckage — `evalfn`
scores a board, so a line that loses on the spot would otherwise be graded on
the corpse.

### Pre-registered before the run

**n=200, and below 0.40 B4 dies; above 0.40 it escalates to n=1000.**

| version | score vs `bc` (v3, rules off) | n |
|---|---|---|
| first build (§8n) | 0.083 [0.015, 0.354] | 12 |
| mid-turn-eval fix (§8n) | 0.075 [0.026, 0.199] | 40 |
| **+ one-ply opponent reply** | **0.375 [0.311, 0.444]** | **200** |

**0.375 < 0.40, so B4 is closed.** The CI excludes 0.5 outright — this is still
a clear loss, ≈ **−89 Elo**, not a near miss. ⚠ Seat split is wide (0.30 as P0,
0.45 as P1) and worth noting, but both seats lose.

### The interesting part: the diagnosis was correct and it was not enough

**0.075 → 0.375 is the single largest movement any B4 change has produced**, and
it came from the one intervention that was reasoned about rather than debugged.
End-of-turn myopia really was most of the rout. **And the objective is still
worse than the clone's single forward pass.**

### ✅ The confound was flagged, then resolved the same session

The reply arm also runs at a **3× time budget** — the reply triples simulation
cost, and at the default `sb0.35` the sequencer blew its budget **62 times
against 56 plans**, so a like-for-like comparison required `sb1.0` (aborts → 0).
That left `0.075 → 0.375` mixing **reply** with **budget**. The clean arm —
`seq,sb1.0` with reply **off**, same n, same everything else — was then run:

| arm | score vs `bc` (v3, rules off) | n | ≈ Elo |
|---|---|---|---|
| `seq`, `sb0.35` (§8n) | 0.075 [0.026, 0.199] | 40 | −436 |
| **`seq,sb1.0` — budget alone** | **0.165 [0.120, 0.223]** | 200 | **−282** |
| **`seq,sb1.0,reply` — + the design fix** | **0.375 [0.311, 0.444]** | 200 | **−89** |

**The two n=200 CIs are disjoint (0.223 < 0.311), so the reply fix is worth
≈ +0.21 at MATCHED budget — it is not the extra time.** Budget alone accounts
for ≈ +154 Elo of the recovery and the reply for ≈ +193 Elo. **§8n's design
diagnosis is therefore confirmed by a controlled experiment, not merely
consistent with one:** maximising end-of-OUR-turn value really was the dominant
defect, and it was the largest single effect in B4's history.

**And B4 still dies.** Confirming the diagnosis did not save it — 0.375 is below
the pre-registered 0.40 line and its CI excludes 0.5 outright.

### What it costs and what it closes

- **~12 s/game of planning**, ~40× the clone. Well inside the 600 s pool, but it
  makes every A/B 40× more expensive — which is why the staged n=200 rule
  existed.
- ⛔ **B4 is closed as a rank lever.** Three builds, one correct design
  diagnosis, and the best version is 89 Elo behind a net that costs 1 ms.
- 🔴 **The consequence for NNUE, and it is the one that matters for planning:**
  an incrementally-updatable evaluator's whole value proposition is scoring many
  states per decision. **B4 was the only consumer for one in this project that
  is not the dead game-tree search** (§2). With B4 closed, **building an NNUE
  buys nothing here until some consumer for it exists.**

Archives: `out/arena/b4_reply_vs_v3.jsonl`, `out/arena/b4_noreply_sb10_vs_v3.jsonl`
(200 rows each). Logs: `out/logs/b4_reply_ab.txt`, `out/logs/b4_noreply_ab.txt`.

## 8w. ✅ THE CLONE IS NOT CAPACITY-BOUND — 8.2× the parameters buys nothing, so the ~30% residual is the ENCODING (2026-07-31, day 11)

**Why this was run.** After B7 failed on both arms (§8u), the live hypothesis
became: the bottleneck is neither the demonstrator nor the data volume, but how
much of *any* policy this feature set can express. The supporting observation
was that fitting **one** player with 27,318 rows plateaus at ~67% while fitting
**~50** players with 248,985 rows plateaus at ~70% — nine times the data and a
fifty-fold harder target, for ~2.5 pp. **That is the signature of a model or
representation limit, not a target limit.** This experiment separates the two.

**Design.** Identical corpus (`artifacts/pds_v3r`), identical recipe
(`--epochs 12 --bs 1024 --loss listwise`), identical seed. **Only the width
changes.** Held-out agreement is measured the same way for all three —
`context_accuracy.py` on the trainer's `gid % 20` split, 12,939 single-choice
rows — so the three numbers are directly comparable.

| net | state / head | params | vs v3 | **misses of 12,939** | agreement | best val, epoch |
|---|---|---|---|---|---|---|
| **v3** (live) | 512,256 / 256,128 | **594,369** | 1.0× | **3,902** | **69.8%** | — |
| `cap_big` | 1024,512 / 512,256 | **1,559,489** | **2.6×** | **3,900** | **69.9%** | 0.7028 @ e8 |
| `cap_xl` | 2048,1024 / 1024,512 | **4,865,985** | **8.2×** | **3,945** | **69.5%** | 0.6994 @ e5 |

🔴 **2.6× the parameters buys TWO decisions out of 12,939. 8.2× LOSES 43.**
Agreement is flat-to-declining across an order of magnitude of capacity.

**And the training curves say why — this is overfitting, not underfitting:**

| net | train loss, first → last | val peak | val at epoch 11 |
|---|---|---|---|
| `cap_big` | 1.194 → **0.633** | 0.7028 (e8) | 0.6944 |
| `cap_xl` | 1.168 → **0.566** | 0.6994 (e5) | 0.6928 |

Both drive training loss far below v3's while validation **peaks early and then
declines**, and the larger net peaks *earlier* (e5 vs e8) and *lower*. The models
already have more capacity than the data and features can use. **Adding
parameters is not merely useless here — past ~1.5M it is actively harmful.**

### What this establishes, and it is the load-bearing conclusion of day 11

**The ~30% residual disagreement is not the model, not the data volume, and not
the demonstrator. By elimination it is the ENCODING** — which is exactly what
§8f proved once already, when the single fix that ever moved this clone
(+≈115 Elo) turned out to be representational: `opt["index"]` was never encoded,
so two options naming two copies of the same card were **bitwise-identical
inputs with different right answers**.

🔴 **The consequence for reinforcement learning, and it should be read before any
RL work is scheduled.** A policy gradient reads the **same** `optfeat`/`features`
vectors as the clone. Where two options are bitwise identical, their gradients
are identical too — **exploration cannot break a tie the representation cannot
express.** RL fixes credit assignment; it does not fix expressiveness.
**So a representational ceiling binds RL exactly as it binds imitation, and this
measurement says we are at one.** The feature audit is therefore a *prerequisite*
for the RL program, not a competing priority.

⚠ **What it does NOT establish.** This measures agreement, and rule 3 says
agreement does not predict playing strength (six instances). The claim here is
narrow and negative: **more parameters do not help.** It does not prove that
better *features* would — §8f is the only evidence for that, and it is one
instance. The next candidate list is not a guess either: §8s names the contexts
where a 1163-rated policy actually diverges from ours (**MAIN 45.8%,
DAMAGE_COUNTER 30.5%, SWITCH 27.6%**), and the known-unencoded inputs are
opponent hand size (Alakazam is 22% of the field and attacks for 20 per card in
hand), stadium, prizes remaining and turn number.

🔴 **RETRACTED THE NEXT DAY (§8y): three of those four are already encoded** —
`features.py` lines 88-99 have fed the net `turn`, both prize counts and both
players' `handCount` since v1. Only the stadium was genuinely absent. The list
was carried unchecked in three files for two days; `p18_missing_state_audit.py`
now DERIVES it instead. Rule 15, second instance.

Reproduce:

```powershell
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v3r --epochs 12 `
    --bs 1024 --loss listwise --state-h 1024,512 --head-h 512,256 `
    --out out/policy_cap_big.npz
python -X utf8 scripts/context_accuracy.py --net out/policy_cap_big.npz `
    --ds artifacts/pds_v3r --min-rows 100000
```

⚠ **A memory fix was required to run this at all, and it is worth knowing.**
`train_policy.Data` materialised one small array per row per bag — 248,985 rows
× 3 bags ≈ 750k objects — for data the shards already store flat. That
allocation, not the model, OOM'd this 7.3 GB machine on every net above ~1.5M
params. Bags are now kept flat with global offsets and gathered per batch.
**The rewrite was proven equivalent before any result was trusted**: all 46,425
rows × 3 bags of a two-shard corpus match the old per-row slicing byte-for-byte,
and so does the batch-assembly path.

Logs: `out/logs/cap_big_train.txt`, `out/logs/cap_xl_train.txt`.

## 8x. 🔴 "The residual is the encoding" is NARROWED: the option layout permits **95.6%** and the clone gets 69.8%, so un-expressibility is at most 4.4 pp (2026-08-01, day 12)

**Why this was run.** §8w concluded *by elimination* that the ~30% residual is
the encoding: capacity is ruled out (8.2× params, no gain), demonstrator
selection is ruled out (§8u), data volume was already dead (§1). **An
elimination argument is only as good as its enumeration**, and this one had
never been checked directly. `scripts/p17_encoding_ceiling.py` measures it with
no net involved at all.

### The instrument

Two options whose `(opt_dense, card_id, attack_id, target_id)` are **bitwise
identical** get identical logits from *any* net reading those inputs — the state
block is shared across a row's options, so it cannot break the tie. If the
demonstrator's choice sits in a tie group of size `g`, no such net can beat
`1/g` on that row. **Σ(1/g)/N is therefore a hard upper bound on top-1
agreement for this feature layout**, and it is exactly the §8f defect
(`opt["index"]` unencoded ⇒ two copies of a card bitwise identical) counted over
the whole corpus instead of found by reading code.

### The result — the bound is high, and the ties are HARMLESS

| context | rows | tied rows | ceiling | v3 top-1 | misses |
|---|---|---|---|---|---|
| **MAIN** | 127,683 | **5.0%** | **97.4%** | 62.7% | **2,629** |
| **TO_HAND** | 31,901 | **32.4%** | **81.0%** | 61.2% | **674** |
| ATTACH_TO | 2,395 | 45.3% | 74.1% | 71.1% | 37 |
| TO_BENCH | 574 | 38.2% | 79.4% | — | — |
| EVOLVE | 952 | 22.8% | 88.2% | — | — |
| SWITCH / DAMAGE / DAMAGE_COUNTER / ATTACH_FROM / … | 60k+ | **0.0%** | **100.0%** | — | — |
| **all** | **235,654** | **7.8%** | **95.6%** | **69.8%** | **3,902** |

🔴 **The encoding permits 95.6% and the clone delivers 69.8%. Exact
un-expressibility can account for at most 4.4 pp of the 30.2 pp residual.**

**And the ties that do exist are not defects.** A tie requires the *same*
`card_id` (it is part of the key), so every tie group is **two copies of one
card in one role** — two identical Trainers in the deck at indices 12 and 30
(TO_HAND), two identical energies in hand attaching to the same Pokemon
(ATTACH_TO). **Picking either produces the same game.** Plain top-1 charges the
net for a coin flip between interchangeable cards.

✅ **Fixed in the instrument, not just noted:** `context_accuracy.py --equiv`
scores a hit when the argmax is bitwise identical to the chosen option. **The
honest agreement is 29.0%, not 30.2%**, and TO_HAND moves **61.2% → 67.1%**
(ATTACH_TO 71.1% → 78.1%). Small overall, but it is the difference between
measuring play and measuring label arbitrariness, and it should be carried into
any future agreement number.

### What this does and does not say

- ⛔ **It does not resurrect capacity or demonstrator selection.** §8w and §8u
  are untouched.
- 🔴 **It does narrow §8w's conclusion, which three files now assert.** "The
  residual is the encoding" cannot mean *the right answer is not expressible* —
  on 95.6% of rows it is. What remains available is the weaker and more specific
  claim: **the STATE does not carry what would let the net pick between options
  it can already tell apart.** That is a different repair (§8y) and a much
  narrower target.
- ⚠ **The Bayes floor could not be measured this way and that is worth
  recording.** Of 235,654 rows only **43 full inputs repeat at all**, covering
  0.8% of rows (almost all `IS_FIRST`), where demonstrators agree 98.7% of the
  time. **Exact-duplicate states essentially do not occur in this game**, so
  "how much of the residual is human noise" cannot be answered by de-duplication.
  The one thing that does bear on it is §8u's expert arm: fitting **one** player
  plateaus at 67.2% held out, barely above the 50-player mixture's ~70% —
  **mixture entropy is not the explanation either.**

Reproduce:

```powershell
python -X utf8 scripts/p17_encoding_ceiling.py --ds artifacts/pds_v3r
python -X utf8 scripts/p17_encoding_ceiling.py --ds artifacts/pds_v3r --opt-cols 25   # the §8f control
python -X utf8 scripts/context_accuracy.py --net out/policy_b1_v3.npz --ds artifacts/pds_v3r --equiv
```

## 8y. ⚡ The feature audit, done by ENUMERATION — and it killed three candidates the plan had been carrying for two days (2026-08-01, day 12)

**Why this was run.** §8w's closing paragraph named the next candidate list:
"opponent hand size (Alakazam is 22% of the field and attacks for 20 per card in
hand), stadium, prizes remaining and turn number." That list has been in
`HANDOFF.md` and `ROADMAP.md` since day 10. **Three of its four items are
already encoded.**

🔴 **`features.py` lines 88–99 have fed the net `turn`, both players' prize
counts and both players' `handCount` since v1.** This is **rule 15 for the
second time in one project** — a claim about the code, repeated in three files,
justifying planned work, never opened and checked. The first instance cost eight
days ("the net cannot see HP"); this one was caught before anything was built,
by the same cure: read the file.

### The instrument

`scripts/p18_missing_state_audit.py` walks real observations, **diffs the
observation's keys against the set `featurize()` actually reads**, and reports
how much each dropped field varies at a decision point. Rule 14 applies to
features exactly as to rules: *an absent input that is constant where the
decisions happen cannot explain a single miss.* 44,992 decision points, 300
games.

**Everything the state carries and the net never sees:**
`current`: `looking`, `retreated`, `stadium`, `stadiumPlayed`, `turnActionCount`;
`player`: `benchMax`; `pokemon`: `energyCards`, `preEvolution`, `serial`;
`select`: `contextCard`, `deck`, `effect`, `remainDamageCounter`,
`remainEnergyCost`.

### Sized, and two more died on the spot

| candidate | distinct | modal share | verdict |
|---|---|---|---|
| `remainDamageCounter` | **1** | **100.0%** | ⛔ **dead — constant 0 at every decision** |
| `remainEnergyCost` | 4 | **99.1%** | ⛔ **dead — 100% modal inside every context** |
| `contextCard` | 22 | 89.8% | ⛔ weak, and ~constant per context |
| `my/opp_tools_n` | 3 | 90.2% / 85.3% | ⚠ marginal (tools present 10–15%) |
| **`turnActionCount`** | **20** | **9.9%** (17% in MAIN) | ✅ **the highest-variance dropped field** |
| **`select.effect` card** | **45** | 66.5% (**26% in TO_HAND**) | ✅ **the highest-variance field where the misses are** |
| **`stadium` id** | 7 | 60.7% (57% in MAIN) | ✅ absent entirely, incl. from every id bag |
| `retreated` / `stadiumPlayed` | 2 | 91.0% / 90.2% | ✅ cheap; `retreated` is **43% non-modal in SWITCH** |
| `opp_prize_left` *(control)* | 7 | 40.7% | — already encoded; included to prove the probe reads real variation |

**The two survivors land exactly where the misses are** (v3 net, 12,939
held-out rows: **MAIN 2,629 + TO_HAND 674 = 84% of all misses**):

- **`turnActionCount`** — 1…25+, smoothly distributed. The net re-scores a
  barely-changed board several times per turn **with no idea how deep into the
  turn it is.** MAIN is 54% of the corpus and 67% of the misses.
- **`select.effect`** — *which card caused this select*. In TO_HAND that is
  Spikemuth Gym 16.4%, Poké Pad 13.0%, Team Rocket's Petrel 10.1%, Night
  Stretcher 7.1%, Dawn 5.5%, Ultra Ball 2.9%, … — "tutor a Trainer", "take a
  Supporter", "recover from the discard" and "search anything" are **the same
  context with the same select type**. 🔴 **And the net cannot infer it from the
  option list, because it never sees the option list**: every option is scored
  independently against a shared state vector. The effect card is the one input
  that says *what kind of choice this is*.
  ⚡ This is also where the strongest measured expert divergence lives — §8q:
  Sixth Sense (#3) diverges from us **almost entirely in TO_HAND, −31.5 pp**.

**Three instruments now point at TO_HAND** (lowest encoding ceiling, largest
expert divergence, highest dropped-field variance) and one at MAIN (67% of the
miss mass). ⏳ **The intervention is built and training; the verdict is §8z, and
it is deliberately not written here yet** (ROADMAP's day-11 amendment: log the
numbers, leave the verdict blank while runs are in flight).

### The intervention, and its bar — written down before any result

**The v4 state block** (`features.extra_feats`, 8 dense scalars + 2 card ids
embedded through the existing slot table): `turnActionCount`, `retreated`,
`stadiumPlayed`, has-stadium, `benchMax`, own/opponent tool counts, the search
pool size, plus embeddings of **the stadium** and **the select's effect card**.
⚠ **Appended after `seld`, the last block of the state vector**, so a v3 net
slices to its own `state_in` and reads byte-identical input — the same
compatibility trick `opt_in` uses, and what lets v3 and v4 run head-to-head in
one process (rule 4).

**The control is as tight as this project can make one.** `artifacts/pds_v4` was
rebuilt from the same four replay days and verified **byte-identical to
`artifacts/pds_v3r` on every pre-existing array** — same 248,985 rows, same
labels, same option features — with only `xdense`/`xslots` added.
`--no-extra` then trains the v3 state vector on those identical rows with the
same seed and recipe. **Features are the only difference.**

**Pre-registered, 2026-08-01, before the first arena game:**

| test | bar |
|---|---|
| primary: `v4` vs `v4ctrl`, mirror, n=2,000 | **≤ 0.52 kills it** — a null at an instrument that resolves ±0.021 |
| secondary: `v4` vs the live v3 net, mirror, n=2,000 | reported either way |
| ship: weighted over the **five** field anchors | **+50 Elo or it is a chapter, not a submission** (§8k) |

⚠ **The standing prior is unfriendly and is recorded here so the verdict cannot
be reframed afterwards:** six axes of more/better training have now measured
null or negative, and **exactly one intervention ever worked — a
representational one (§8f, 0.878 against its same-corpus control).** This is the
closest thing to a second instance of that axis, which is the reason to run it;
it is not a reason to expect it to work. Rule 3 also applies: **held-out
agreement will move and that predicts nothing** — the arena decides.

Reproduce:

```powershell
python -X utf8 scripts/p18_missing_state_audit.py --games 300
```

## 8z. ⚡ THE v4 STATE BLOCK WINS THE ARENA AND MOVES HELD-OUT AGREEMENT BY ZERO — the cleanest decoupling of the two in the project (2026-08-01, day 12)

**The hypothesis, from §8y:** the residual is not un-expressibility (§8x says the
option layout permits 95.6%); it is that the **state** does not carry what would
let the net choose between options it can already tell apart. The v4 block adds
the fields the audit derived — `turnActionCount`, the select's **effect card**,
the **stadium**, `retreated`/`stadiumPlayed`, tool counts, bench cap, pool size.

**The control is byte-tight.** `artifacts/pds_v4` is identical to
`artifacts/pds_v3r` on every pre-existing array — same 248,985 rows, same
labels, same option features — plus `xdense`/`xslots`. `--no-extra` trains the
v3 state vector on those identical rows, same recipe, same seed. **Features are
the only difference.**

### Result 1 — the arena, and it clears the pre-registered kill line

| A/B, grimmsnarl mirror, n=2,000 | score | Elo |
|---|---|---|
| **`v4` vs `v4ctrl`** (its own same-corpus control) | **0.567 [0.545, 0.588]** | **+47** [+31, +62] |
| **`v4` vs the live v3 net** | **0.541 [0.519, 0.562]** | **+29** |

Both intervals exclude 0.5; the pre-registered kill line was ≤0.52 and the
primary's lower bound is 0.545. Seat-balanced (569/430 as P0, 563/436 as P1).

### Result 2 — held-out agreement did NOT move, and that is the finding

| net | misses of 12,939 | agreement (`--equiv`) |
|---|---|---|
| `v4ctrl` | **3,756** | 71.0% |
| `v4` | **3,748** | 71.0% |

🔴 **Eight decisions out of 12,939 — a dead heat — for +47 Elo of play.** Best
validation top-1 was 0.7031 (control) against 0.7037 (treatment).

**Rule 3 has been paid for six times as "better agreement, worse play". This is
the first instance of the converse, and it is the more useful one**: the metric
that the entire B7 programme was built on (§8q, §8r, §8u) is *insensitive* to an
intervention worth 47 Elo. The per-context breakdown shows why — the block did
not raise agreement, it **moved** it:

| context | `v4ctrl` misses | `v4` misses | Δ |
|---|---|---|---|
| MAIN | 2,595 | 2,630 | **+35 worse** |
| TO_HAND | 572 | 558 | −14 |
| ATTACH_FROM | 74 | 55 | **−19** |
| DAMAGE_COUNTER | 117 | 105 | −12 |
| DAMAGE / SWITCH / TO_ACTIVE / DC_ANY | 155 | 141 | −14 |
| ATTACH_TO / REMOVE_DC | 162 | 171 | +9 |

**It agrees less with the humans in MAIN and more in the execution contexts, and
plays 47 Elo better.** Whatever the block bought is not "imitate the mixture
more closely".

### Result 3 — the five-anchor sweep: better on every anchor, and below the bar

Δ is `elo(v4 vs anchor) − elo(v3 vs anchor)`, v3's numbers from §8i's sweep. The
mirror row is a head-to-head, so its Δ is `elo(0.541)` directly (§8i's warning:
do **not** compute that one as `elo(v) − elo(1−v)`).

| anchor | share | v3 | **v4** | Δ Elo | weighted |
|---|---|---|---|---|---|
| `rule:alakazam5` | **22.0%** | 0.731 | **0.759** [0.740, 0.777] | **+26** | +5.6 |
| mirror, head-to-head | 13.8% | — | **0.541** [0.519, 0.562] | **+29** | +3.9 |
| `rule:crustle` | 12.8% | 0.770 | **0.788** [0.770, 0.806] | **+18** | +2.3 |
| `rule:v10` | 12.8% | 0.505 | **0.549** [0.528, 0.571] | **+31** 🔴 **disjoint** | +3.9 |
| `rule:archaludon` | 10.1% | 0.669 | **0.678** [0.657, 0.698] | +7 | +0.7 |
| | **71.5%** | | | | **+16.5 Elo** |

⚡ **Mega Lucario is the one worth reading twice.** It was v3's *only* losing
anchor — 0.505 against P4b's 0.576, the single negative term in the day-9 table
and the lead that item 3 of the day-9 plan chased for two sessions. **v4 takes
it to 0.549 with intervals disjoint from v3's**, without a matchup branch, a
rule, or a decklist change. The generic feature fix did what the targeted rule
could not.

🔴 **And the verdict on shipping is NO by the bar that was written down first:
+16.5 Elo weighted (+23 renormalised to full coverage) against a pre-registered
+50.** It is positive on five anchors of five, which is a better shape than the
number suggests — but the bar exists because **the leaderboard resolves ±50–100**
(rule 2), and 23 Elo is invisible to it. **We are not going to discover whether
this net is better by submitting it.**

### The confound that had to be sized first, and the answer

⚠ **Every net-vs-net A/B in this repo compares two independently trained nets,
and this project has never measured how much of such a gap is the seed.** The
treatment and control differ in weight-init shape as well as in features, so
"+47 Elo" could in principle be run-to-run variance. `--seed` was added and both
arms retrained at seed 1.

| A/B, mirror, n=2,000 | score | reading |
|---|---|---|
| `v4ctrl` (seed 0) vs `v4ctrl_s1` (seed 1) — **seed only** | **0.482 [0.460, 0.504]** | ✅ **null**; run-to-run variance is ≈ ±13 Elo |
| `v4` vs `v4ctrl`, **seed 0** | 0.567 [0.545, 0.588] | +47 Elo |
| `v4_s1` vs `v4ctrl_s1`, **seed 1** | **0.539 [0.518, 0.561]** | ✅ **+27 Elo — replicates** |

✅ **The effect survives.** Both treatment intervals exclude 0.5 and both are
disjoint from the seed-only interval; the pooled estimate over n=4,000 is
**≈ +37 Elo**. **The seed explains none of it** — which also means every earlier
net-vs-net number in this repo is now backed by a measured noise floor instead
of an assumption.

⚠ **The honest size is the pooled +37, not the headline +47.** Seed 0 was the
luckier draw and it is the one that was swept against the anchors, so the
weighted +16.5 above may itself be a point or two generous.

### What was done with it

**Submitted anyway, and the reasoning is recorded because it deviates from the
bar.** The pre-registered rule was "+50 weighted or it is a chapter". That bar
existed for two reasons: the LB cannot resolve a small effect, and **every
submission evicts a live agent** (§8h). The user relaxed the second — daily
submissions are not scarce for us — so the trade was re-priced, not the
evidence:

- v4 is better on **5 anchors of 5**, no negative term. **That is the specific
  shape B1 lacked** — B1 won two anchors covering 26.6% of the field and lost
  `rule:v10` outside its anchor set (§8i). Coverage is now 71.5%.
- It repairs **v3's only losing matchup** with disjoint intervals.
- The effect is replicated at a second seed against a measured noise floor.

🔴 **And what it will NOT do: validate itself.** +16.5 Elo weighted is far below
the ladder's ±50–100 (rule 2). **Whatever `55137...` reads, that reading is not
evidence for or against the v4 block** — the arena, at n=2,000 per cell with a
seed control, is the better instrument here and it has already answered.

Reproduce:

```powershell
python -X utf8 scripts/arena.py play "bc:v4,net=out/policy_v4.npz,noChip,noSpread,noSrc" `
    "bc:v4ctrl,net=out/policy_v4ctrl.npz,noChip,noSpread,noSrc" `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 `
    --archive out/arena/p19_v4_vs_v4ctrl.jsonl
```

Archives: `out/arena/p19_v4_vs_{v4ctrl,v3live,alakazam5,crustle_v1,lucario_v10,archaludon_ex}.jsonl`.
Logs: `out/logs/v4_train.txt`, `out/logs/v4ctrl_train.txt`.

## 8aa. 🔴 THE v5 POOLED OPTION-SET BLOCK: 27× THE AGREEMENT GAIN FOR A THIRD OF THE ELO — §8z's result run backwards, 24 hours later (2026-08-01, day 13)

> ⚠ **This entry was written after ONE seed and its headline was wrong.** The
> first arm read 0.514 [0.492, 0.536] — a clean null — and this section said so.
> The seed-1 replicate then read **0.527 [0.505, 0.549]**, which excludes 0.5,
> and the pooled estimate is **+14 Elo**, not zero. **ROADMAP's own rule — "log
> the numbers and leave the verdict blank while runs are in flight" — was
> written after §8i had to be retracted for exactly this, and it was broken
> again here.** The corrected numbers are below; the correction is left visible
> because the failure mode is the report's §5.4 and this is its second instance.

**The hypothesis, and it was the best-motivated one left.** Every option is
scored independently against one shared state vector, so the net has never been
able to see the option **set**: it cannot tell whether it is choosing among 3
Trainers in hand or 40 cards in the deck, nor how the option in front of it
compares to its alternatives. That is the same *class* of defect as §8f (no
binding between an option and its target) and §8y (no `effect` card saying what
kind of choice this is) — and those are the only two interventions in twelve
days that ever paid.

**The v5 block** is deep-sets in its cheapest possible form: φ is the per-option
encoding the head already builds, the pool is an elementwise **mean and max**
over the select's options plus two count scalars, and ρ is the existing state
MLP. `optfeat.pool_width`, `train_policy.py --pool`. Appended after the v4
block, never inserted, so `--pool` off reproduces the v4 state vector
byte-for-byte — the same control discipline for the third generation running.

### Result 1 — the arena says "barely anything happened"

| A/B, grimmsnarl mirror | n | score | reading |
|---|---|---|---|
| **`v5` vs `v4`, seed 0** | 2,000 | 0.514 [0.492, 0.536] | null on its own |
| **`v5` vs `v4`, seed 1** | 2,000 | 0.527 [0.505, 0.549] | excludes 0.5, +19 Elo |
| **pooled — the honest number** | **4,000** | **0.521 [0.505, 0.536]** | **+14 Elo** [+4, +25] |
| `v5` vs `v4ctrl` — the positive control | 2,000 | **0.539 [0.517, 0.561]** | ✅ +27 Elo |
| (`v4` vs `v4ctrl`, §8z, pooled) | 4,000 | 0.553 | **+37 Elo** |

**Two things make that +14 readable rather than impressive.** The seed-only null
is 0.482 [0.460, 0.504] (§8z) — run-to-run variance is ≈ ±13 Elo, so +14 is one
noise-width, and the pooled interval's lower bound is 0.505. And the positive
control fires at the same size v4 does, so the instrument is working: the block
has not broken anything, it has added **about a third of what v4 added**, at the
edge of what n=4,000 can resolve.

### Result 2 — and held-out agreement moved a LOT

| net | misses of 12,939 | agreement (`--equiv`) | best val top-1 |
|---|---|---|---|
| `v4ctrl` | 3,756 | 71.0% | 0.7031 |
| `v4` | 3,748 | 71.0% | 0.7037 |
| **`v5`** | **3,534** | **72.7%** | **0.7201** |

🔴 **214 decisions of 12,939 — the largest agreement gain any intervention in
this project has produced.** And most of it lands exactly where the mechanism
predicted: **MAIN misses fall 2,630 → 2,454**, the context the option-set
summary was designed for. The block did what it was built to do, as a *fit*.

### 🔴 The finding, and it is the pair rather than either half

Read §8z and this entry as one table. Same corpus, same recipe, same held-out
split, same arena, same measured noise floor, 24 hours apart:

| intervention | Δ agreement (of 12,939) | Δ Elo | Elo per decision gained |
|---|---|---|---|
| **v4 state block** (§8z) | **+8 decisions** | **+37** | 4.6 |
| **v5 pooled option set** (here) | **+214 decisions** | **+14** | **0.07** |

**The exchange rate between the two quantities differs by a factor of 70
between two interventions run a day apart on the same corpus.** Rule 3 had been
paid for six times as "better agreement, worse play"; §8z gave the first
converse (a large Elo gain the metric could not see); this gives the second and
sharper one — **a large agreement gain that buys almost nothing.** Neither
number predicts the other in either direction or at any scale.

⇒ **The practical consequence for the remaining days: `val_top1` is not a
screening metric for this project, in either direction, and no candidate may be
promoted or killed on it.** Only the arena decides. That was already rule 3;
what is new is that we can now show it costs a *day* to relearn.

⚠ **What this does NOT show.** It does not show that the option set is
irrelevant to strong play — it shows that *this* summary of it, fed to *this*
state MLP, buys about a noise-width. A per-option **centred** encoding (each
option minus the set mean, i.e. the pool on the option side rather than the
state side) is a different experiment and is untested. But the prior is now
unfriendly: it would be the seventh "fit the mixture better" axis, and the six
before it produced one win.

### Result 3 — the five-anchor sweep, and the SHAPE is what decides it

Δ is `elo(v5 vs anchor) − elo(v4 vs anchor)`, v4's numbers from §8z. The mirror
row is a head-to-head, so its Δ is `elo(0.521)` directly (§8i's warning: do
**not** compute that one as `elo(v) − elo(1−v)`).

| anchor | share | v4 | **v5** | Δ Elo | weighted |
|---|---|---|---|---|---|
| `rule:alakazam5` | **22.0%** | 0.759 | **0.789** [0.771, 0.807] | **+30** | +6.6 |
| mirror, head-to-head | 13.8% | — | 0.521 [0.505, 0.536] | +15 | +2.0 |
| `rule:crustle` | 12.8% | 0.788 | **0.768** [0.749, 0.786] | **−20** 🔴 | −2.6 |
| `rule:v10` | 12.8% | 0.549 | **0.569** [0.547, 0.591] | +14 | +1.8 |
| `rule:archaludon` | 10.1% | 0.678 | **0.671** [0.650, 0.691] | −6 | −0.6 |
| | **71.5%** | | | | **+7.3 Elo** |

### The verdict on shipping — first NO, then corrected to YES, and the correction is the lesson

**The first verdict, written from the table above, was ⛔ NO.** +7.3 Elo
weighted (+10 renormalised) is below the +50 bar; §8z shipped below the same bar
only because it was positive on **five anchors of five** with no negative term,
whereas v5 is **positive on three, negative on two**, its largest single term a
**−20 Elo loss to Crustle** (12.8% of the field, the deck the counter-meta is
built on, §8b). A mixed-sign +7 is what a noise-width effect looks like measured
five times.

🔴 **That reasoning is sound and it answers the wrong question.** It asks *"is
v5 better than v4?"* — no. **A submission does not replace our best agent; it
replaces the one that falls out of the latest-2 window.** Eviction is by
recency (§8h), so the trade on the table was:

| | rating when submitted | arena standing |
|---|---|---|
| **v5**, submitted as `55160229` | climbs from μ=600 | ≈ v4 (+14 mirror, +7.3 weighted) |
| **P4b `55129730`**, evicted | **836.4** | **last of everything we own** (§8k) |
| **v4 `55156480`**, untouched | **908–923** | the reference |

**v4 stays active and keeps its rating, and the displayed score is the best
ACTIVE agent — so the displayed number cannot fall, and the slot costs a
strictly dominated agent.** ⇒ **submitted.**

⚠ **The failure mode is worth more than the submission.** The +50 bar was
written when slots were scarce and *every submission evicted something
valuable*. The user relaxed exactly that premise two days earlier, §8z recorded
the relaxation — and the bar was still being applied with its original cost
model attached. **A threshold outlives the assumptions that set it, and this one
survived its own retraction by two days.**

⇒ **Standing rule: before quoting the shipping bar, name the agent the
submission would EVICT.** A candidate that loses to our best can still dominate
our worst, and only the second comparison decides a slot.

### A methods note that nearly cost the day

The refactor that made the pool possible moves the per-option encoding **ahead**
of the state MLP for every net, including v4. A regression A/B of `v4` vs
`v4ctrl` at n=600 read **0.503**, against §8z's 0.567 — which would have meant
the refactor had silently broken the live net. **The arena cannot settle that
question: it is not deterministic run to run** (match 0 was 15 turns in one run
and 11 in the next), so the two numbers are not comparable game for game.

The question was settled instead by a direct equivalence test — load the
pre-edit `policynet.py` from git as a second module, run both against the same
observations from real games, compare scores exactly:

```
net policy_v4.npz: state_in=536 opt_in=37 n_pool=0
compared 588 selects; max |old - new| = 0.000e+00
```

**Bitwise identical.** The 0.503 was a 2.4σ draw at n=600 and nothing more.
⚡ **The general rule this buys: when a refactor is *supposed* to be a no-op,
prove it with an equivalence test, not with the noisy end-to-end instrument.**
n=600 could not have distinguished "unlucky" from "broken" at any p-value worth
acting on, and three hours of A/Bs would have been spent on the wrong question.

Reproduce:

```powershell
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 --bs 1024 `
    --loss listwise --state-h 512,256 --head-h 256,128 --pool --out out/policy_v5.npz
python -X utf8 scripts/arena.py play "bc:v5,net=out/policy_v5.npz,noChip,noSpread,noSrc" `
    "bc:v4,net=out/policy_v4.npz,noChip,noSpread,noSrc" `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 `
    --archive out/arena/p20_v5_vs_v4.jsonl
```

Archives: `out/arena/p20_v5_vs_{v4,v4ctrl}.jsonl`,
`out/arena/p20_regress_v4_vs_v4ctrl.jsonl`. Log: `out/logs/v5_train.txt`.

## 8ab. ⚡ ABLATING THE v4 BLOCK: the three DERIVED members carry all of it, the other five are worse than nothing, and no single one is necessary (2026-08-01, day 13)

**The question §8z left open:** the block shipped whole, so nothing said which of
its eight members bought the +37 Elo. §8y *derived* three of them by enumeration
(`turnActionCount`, the select's **effect** card, the **stadium**) and the other
five came along as cheap extras (`retreated`, `stadiumPlayed`, tool counts,
bench cap, pool size).

**The instrument** (`train_policy.py --drop-x`, `features.X_GROUPS`): the named
columns are **zeroed** in the corpus rather than deleted, so an arm has the
identical architecture, parameter count, weight init, rows, recipe and seed as
v4 — **only the content of a few columns differs.** The surviving mask is stored
in the npz (`x_mask`) and applied by `policynet` at inference, so an arm can
never be fed a column it was not trained on.

### Drop-one: every member is individually redundant

| arm, vs full v4, n=2,000 | score | Elo | reading |
|---|---|---|---|
| drop `turnActionCount` | 0.527 [0.505, 0.549] | +19 | removing it is *better*, barely |
| drop the **stadium** | 0.526 [0.504, 0.548] | +18 | removing it is *better*, barely |
| drop the **effect** card | 0.483 [0.461, 0.505] | −12 | removing it is worse — **null** |

All three sit within ~1.5 noise-widths of v4 (the seed-only floor is ±13 Elo,
§8z). **Read alone, this table says the block's three headline members do
nothing, and two are mildly harmful.** That reading is wrong, and the next test
is why.

### Drop-all-three: they are jointly necessary, and they are the whole block

| arm, n=2,000 | score | Elo | reading |
|---|---|---|---|
| drop all three, **vs full v4** | **0.449 [0.427, 0.470]** | **−36** | 🔴 disjoint |
| drop all three, **vs `v4ctrl`** (no block at all) | **0.469 [0.447, 0.490]** | **−22** | 🔴 disjoint |

🔴 **Removing the three derived members costs 36 Elo — essentially the entire
+37 the block was worth — while removing any one of them costs nothing.** They
are mutually redundant and jointly necessary: each can stand in for the others,
and losing all three loses the effect.

⚡ **And the second row is the sharper one. The five leftover members, on their
own, are WORSE than having no block at all** (−22 Elo against the control, CIs
disjoint). Extra state columns that do not resolve a real decision are not free
— they are somewhere to overfit. **§8y's sizing step, which killed
`remainDamageCounter` and `remainEnergyCost` for being constant, was not
pedantry; the five survivors that were never sized are exactly the ones that
turn out to be negative.**

⇒ **This validates the day-12 method rather than the day-12 block.** The audit
picked three fields out of an observation by enumeration and sizing, and those
three carry 100% of a measured 37-Elo gain while the unsized extras carry −22.
**Derive and size; do not bundle.**

### ⚠ A methods caveat this produced, and it touches every weighted table here

The three pairwise results **order consistently** — v4 > `v4ctrl` > drop-three
on every head-to-head — but their magnitudes do not add:

| comparison | measured | additive prediction |
|---|---|---|
| v4 − `v4ctrl` | +37 | — |
| `v4ctrl` − drop-three | +22 | — |
| **v4 − drop-three** | **+36** | **+59** |

**Head-to-head Elo among these nets compresses by ~23 points over two hops.**
Nothing here is intransitive, so the *rankings* this project derives from
pairwise A/Bs stand — but **summing or chaining Elo differences across nets
overstates them**, and the five-anchor weighted totals (§8i, §8z, §8aa) are
chained quantities. Treat them as ordinal, not as arithmetic.

Reproduce:

```powershell
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 --bs 1024 `
    --loss listwise --state-h 512,256 --head-h 256,128 `
    --drop-x turnAction,effect,stadium --seed 0 --out out/policy_v4_no3.npz
python -X utf8 scripts/arena.py play "bc:no3,net=out/policy_v4_no3.npz,noChip,noSpread,noSrc" `
    "bc:v4,net=out/policy_v4.npz,noChip,noSpread,noSrc" `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 `
    --archive out/arena/p21_v4no3_vs_v4.jsonl
```

Archives: `out/arena/p21_v4no{turnAction,effect,stadium,3}_vs_v4.jsonl`,
`out/arena/p21_v4no3_vs_v4ctrl.jsonl`. Logs: `out/logs/v4_no*_train.txt`.

## 8ac. 🔴 THE "META SHIFT" IS OUR OWN CLIMB — anchor weights are a function of OUR rating, and every weighted verdict in this repo was weighted for a band we have left (2026-08-01, day 15)

**Hypothesis.** The user supplied 8–9 h of replays for both live submissions
(`replays/submission_v5`, 46 games; `replays/submission_v4`, 29). Day 15's plan
asked two questions of them: *has our field composition moved* — it drives every
anchor weight in this repo — and *does v5 face a different field than v4*, which
it should not, since the two agents play the same 60 cards.

**Both answers came back "yes", and the second one is what explains the first.**

### Result 1 — the pooled census has moved a long way from day 9

`p9_field_census.py --dir replays/submission_v4 replays/submission_v5`, 75 games:

| archetype | day 9 (`submission_optv3`, n=54) | day 15 (v4+v5, n=75) | Fisher p |
|---|---|---|---|
| **Marnie's Grimmsnarl ex** (the **mirror**) | 13.8% | **33.3%** | **0.002** |
| `rule:alakazam5` | 22.0% | 21.3% | 1.000 |
| `rule:crustle` | 12.8% | 6.7% | 0.222 |
| `rule:v10` (Mega Lucario) | 12.8% | **4.0%** | 0.067 |
| `rule:archaludon` | 10.1% | 8.0% | 0.797 |

Our win rate over the same games rose **63.0% → 70.7%**. On its face this is a
meta shift: the mirror has more than doubled and Mega Lucario has collapsed.

### Result 2 — 🔴 but it is not a shift in the meta, it is a shift in US

The two dumps are the same agent architecture on the same 60 cards, 3½ hours
apart, and **their opponents are drawn from the same strength pool** — mean
rating **860 (v5)** vs **878 (v4)**, medians 877 and 882. So the difference
between the two dumps cannot be a rating-band effect *between them*. But
between **day 9 and day 15** we climbed from ~820 to 915–955, and the mean
opponent rating went **799 → 867** with us.

`scripts/p19_field_drift.py` runs the discriminator: bucket all 181 rated games
from all four of our dumps by **opponent rating** and read the shares along that
axis instead of along time.

| archetype | opp <800 (n=75) | 800–900 (n=59) | 900–1000 (n=33) | 1000+ (n=14) |
|---|---|---|---|---|
| **mirror** | 5.3% | 18.6% | **42.4%** | **71.4%** |
| Alakazam | 13.3% | 28.8% | 33.3% | 14.3% |
| Crustle | 16.0% | 5.1% | 9.1% | 7.1% |
| **Mega Lucario** | **17.3%** | 6.8% | **0.0%** | **0.0%** |
| **Archaludon** | 10.7% | 15.3% | **0.0%** | **0.0%** |

⚡ **Mega Lucario and Archaludon are 0 for 47 above rating 900** (≤7.6% by the
rule of three). The mirror rises monotonically across every bucket.

**And now hold the band fixed and compare the eras** — old = the 07-29 and 07-31
dumps, new = v4+v5:

| band | n old / new | largest era difference | Fisher p |
|---|---|---|---|
| <800 | 59 / 16 | mirror 3.4% → 12.5% | 0.198 |
| 800–900 | 28 / 31 | mirror 14.3% → 22.6% | 0.513 |
| 900–1000 | 13 / 20 | Alakazam 53.8% → 20.0% | 0.065 |
| 1000+ | 6 / 8 | mirror 66.7% → 75.0% | 1.000 |

🔴 **Not one archetype differs significantly between the eras once the band is
held fixed** — every p ≥ 0.065, and the smallest one points the *opposite* way
from the pooled table. **The uncontrolled comparison is confounded by our own
rating, and the confound is the entire effect.**

⇒ **The field did not change its decks. We changed our seat in it.**

### Result 3 — what that costs, re-weighted verdict by verdict

Rule 16 says an arena result is a weighted average over the anchor set and
nothing else. The weights were measured at ~820 and every verdict since has
used them. Re-weighting with the band-correct shares (measurements untouched —
only the shares change):

| verdict | day-9 weights | **day-15 weights** | change |
|---|---|---|---|
| §8i `v3 − P4b` | +35.6 | **+62.1** | +26.5 |
| §8j **rules ON − OFF** | **+0.8** | **−18.1** 🔴 | **−18.9** |
| §8z `v4 − v3` | +23.4 | +24.8 | +1.5 |
| §8aa `v5 − v4` | +10.2 | +13.8 | +3.6 |

🔴 **§8j changes sign.** "Turning the rules on globally is worth NOTHING
(+1 Elo)" rested on a near-exact cancellation between a **−51 Elo mirror loss at
13.8%** and a **+47 Elo Mega Lucario gain at 12.8%**. At the real weights the
mirror term is **33.3%** and the Lucario term is **4.0%**, so the cancellation
collapses and the rules are **−18 Elo — actively harmful, not neutral.**

✅ **This does not change what is shipped, and that is the good news.** The v5
bundle pins `chip_targeting=False, energy_spread=False, counter_source=False`
(verified by reading `main.py` out of
`dist/submission_bc-grimmsnarl-netspolicy_20260801-163829.tar.gz`). The decision
to ship rules-off was made on §8f's mirror evidence and is now supported ~18×
more strongly than the table that was used to justify it.

### The general form, and it is the report's methods chapter

Rule 16's deep trap was **"a sampling frame you did not choose is a hypothesis,
not a fact"** — written after Kaggle's episode datasets turned out to be
censored below 1055. This is the same error committed with our *own* data:

> 🔴 **The opponent pool is not a fixed population we are sampling. It is a
> function of our own rating, and it moves when we do.** Every anchor weight in
> this repo therefore has an invisible parameter — the score we held when the
> census was taken — and re-deriving the weights is not optional maintenance,
> it is part of reading any arena number at all.

⚡ **It also resolves a standing tension in these files rather than creating
one.** §8b mined the ≥1144 band and found **52.1% of seats on our archetype**;
§8i censused our own games at ~820 and found the mirror at **13.8%**; day 9
recorded these as a contradiction ("mined meta can NEVER describe our field").
**They were never in conflict — they are two points on one monotone curve**
(5.3% → 18.6% → 42.4% → 71.4% → ~52% of *seats* at the very top), and we have
been walking up it. Both numbers were right about the band they measured.

### Consequences taken

1. **The anchor weights are re-derived** and carry the band they were measured
   at. Any future weighted total states our score at census time.
2. **The mirror is now the dominant matchup and gets more so as we climb** —
   33.3% overall, **51.1% above rating 900**, 71.4% above 1000. It is also the
   matchup our head-to-head net A/Bs already measure best, so the most sensitive
   instrument we own is now also the most representative one.
3. 🔴 **Track C's Archaludon lead is closed by sizing, before it was built**
   (rule 14). It was promoted on "10.1% of the field and our worst real matchup".
   It is **8.0% overall and 0 of 47 games above rating 900** — the work would
   serve a population we are leaving. Same for the Mega Lucario branch (B3
   instance 2, 4.0%) and, more mildly, Crustle (6.7%).
4. ⚠ **Two archetypes have no anchor and now outrank two that do:** Cynthia's
   Garchomp ex **6.7%** and Dragapult ex **5.3%** (12.0% together) against
   Crustle + Lucario's 10.7%. `decks/dragapult_ex.py` already exists.

⚠ **Honest limits.** n=75 for the new shares (Wilson 95% CIs: mirror
[23.7%, 44.5%], Lucario [1.4%, 11.1%]); only the mirror and Lucario moves have
the old share outside the new CI. Opponent ratings come from a snapshot taken
2026-08-01 16:07 UTC, not from the time each game was played, which blurs the
bucket assignment for the older dumps — it cannot manufacture a monotone trend,
but it widens every band boundary.

```powershell
python -X utf8 scripts/p9_field_census.py --dir replays/submission_v4 replays/submission_v5 `
    --lb out/lb_snapshot_0801pm.json
python -X utf8 scripts/p19_field_drift.py --lb out/lb_snapshot_0801pm.json
```

## 8ad. 🔧 THE TRAJECTORY RECORDER — and a test that could not have failed (2026-08-01, day 15)

**The gap.** `arena.py` archives **one summary row per game** (winner, turns,
selects, latency, pool — `arena.py:281-294`). No observations, no actions, no
trajectories. After fifteen days that meant the five anchors carrying **71.5% of
every weighted verdict in this repo had never been watched playing a single
turn**, the RL variance probe had no data source, and a losing A/B could not be
inspected.

**The fix is small because the format is not ours.** `harness.Recorder` +
`play_game(..., recorder=)`. `cg.game.visualize_data()` emits exactly the
structure Kaggle puts at `steps[0][0]["visualize"]` in a downloaded replay, and
attaching `obs`/`action` per step is what the engine's own notebook does
(`notebooks/how-to-output-local-battle-as-json-and-view.ipynb`). So a recorded
local game is read **unmodified by every replay tool already in this repo** —
`p9_field_census.py`, `p5a_replays.py`, `build_policy_dataset.py`,
`p16_policy_disagree.py` — and by the official viewer through
`notebooks/visualizer.html`. ⚠ `visualize_data()` must be called **before**
`battle_finish()`, which frees the buffer.

### 🔴 The methods lesson: the first equivalence test was worthless

§8aa's rule was *"when a refactor is supposed to be a no-op, prove it with an
equivalence test, not the noisy end-to-end instrument"*. The first version of
`p20_recorder_equivalence.py` obeyed the letter of that and broke it anyway: it
played each game twice, once with `recorder=None` and once with a `Recorder`,
and demanded identical action streams. **All four games "failed", diverging at
select 2–4.**

The recorder was fine. **`cg.game.battle_start` takes no seed** — the engine
shuffles internally, so two consecutive games diverge no matter what, and §8aa
had *already recorded this* ("match 0 was 15 turns in one run and 11 in the
next"). ⇒ **A test that cannot pass for the reason you are testing is not
evidence, and neither is one that cannot fail.** It is rule 9 — *a metric that
never prints is not a metric that passed* — one level up: **before trusting an
equivalence test, ask what result would have looked like success.**

**The rewrite tests only claims that can be made exact, per game:**

| check | how | 12/12 games |
|---|---|---|
| the tap is **faithful** | wrap `game.battle_select`; the args the **engine** received must equal `action_log` element for element | ✅ |
| the tap has **no side effects** | serialise `obs` before and after `on_select`; require equality | ✅ |
| the capture is **complete** | observations == selects, visualize steps == selects + 1 | ✅ |
| the artifact **round-trips** | write it, re-read it with `p9_field_census.analyse` | ✅ |

⚠ **What is deliberately NOT claimed:** that `recorder=None` reproduces the
identical *game* to the pre-edit code. No seed exists to prove that with. That
path differs from the original by two `is not None` guards, one local
assignment and one hoisted list comprehension — behaviour-preserving by
inspection — and the script reports the turns/selects distributions with the
recorder on and off (12.8 / 205.7 vs 12.5 / 191.4, n=12) explicitly labelled
**weak**, because agreement there is not proof while a difference would have
been evidence.

```powershell
python -X utf8 scripts/p20_recorder_equivalence.py --games 12
python -X utf8 scripts/p20_record_games.py --a rule:alakazam5 --deck-a alakazam5 `
    --b rule:crustle --deck-b crustle --games 3 --out out/replays/anchor_vs_anchor
```

Recorded so far: `out/replays/v5_vs_alakazam` (v5 2–1), `out/replays/anchor_vs_anchor`
(alakazam5 2–1 crustle — ⚠ one of the three ran **39 turns** against 11 and 11,
the first thing worth explaining in the day-15 item-6 audit).

## 8ae. ⚡ RL, SIZED AT LAST: the pre-registered kill criterion is NOT met, and the twelve-day-old prior does not survive contact with arithmetic (2026-08-01, day 15)

**Why this had to be measured.** "Self-play RL is dead" was asserted in four
documents for twelve days and struck on day 14 (§2's retraction box): it was
**never run** — no code, no `n`, no reward function, in this repo or the old one.
Rule 15's third instance, and the unmeasured claim was living inside
`EVIDENCE.md` itself. §8w's separate objection (a policy gradient reads the same
feature vectors, so bitwise-identical options get identical gradients) was
narrowed by §8x: the encoding binds **at most 4.4 pp** against a clone at 71%.

⇒ **The live objection was neither compute nor expressiveness but
credit-assignment variance** — one binary reward over a ~40-turn game with
hundreds of selects, the same term that killed search. **Rule 14: size it before
building it.** The probe and its kill criterion were written before any code:

> measure how many games the terminal-outcome signal needs to separate two
> policies of KNOWN Elo separation. **Kill if separation needs more games than
> ~1.4 cores can produce in the remaining days.**

⚡ **It cost zero new games.** Four head-to-head archives already exist whose
separations were measured at n≥2000. `p21_rl_variance_probe.py` bootstraps them.

### Result 1 — the outcome signal is cheap

Throughput measured from the archives' own timestamps (so it includes real
contention): **5.96 games/s per process**, ≈ **43,000 games/h** at 2 processes,
≈ **5.5M games** to the 08-17 deadline.

| pair | measured Δ | archived score | games for 80% power |
|---|---|---|---|
| `v4` vs `v4ctrl` (§8z) | **+37 Elo** | 0.567 | **800** — 0.015% of the budget |
| `no3` vs `v4` (§8ab) | **−36 Elo** | 0.449 | **800** — 0.015% of the budget |
| `v5` vs `v4` (§8aa) | +14 Elo | 0.514 | **not detected even at 6,400** |

### Result 2 — 🔴 and the credit signal is affordable too, which is the surprise

A REINFORCE advantage at one select is the game outcome minus a baseline, so its
per-visit SE is ~0.5. There are **201 selects per game** (median 200, max 380,
over 8,000 archived games), and a context recurs `k` times per game:

| true effect on win prob | visits needed | games at k=1 | k=5 | **k=20** |
|---|---|---|---|---|
| 10% | 192 | 192 | 38 | 10 |
| 5% | 768 | 768 | 154 | 38 |
| 2% | 4,802 | 4,802 | 960 | 240 |
| **1%** | 19,208 | 19,208 | 3,842 | **960** |

**Against a 5.5M-game budget, every row is affordable by three orders of
magnitude.** ⇒ **The pre-registered criterion is not met. RL does not die
here.** The variance argument that stood in for a measurement since day 8 is,
when actually sized, **not the binding constraint** — and it dies with a number,
which is the thing §2 never had.

### 🔴 What this does NOT license, stated before anyone acts on it

1. **A sizing probe that fails to kill is not evidence that a thing works.**
   Rule 3 in its original form. B4 passed all three of its kill criteria (§8l)
   and then died at n=200 (§8v). This probe is exactly as weak as that one was.
2. **The model in Result 2 is crude and optimistic.** It prices one context in
   isolation with a two-sample test. Real policy-gradient training adds
   non-stationarity (the data distribution moves as the policy moves),
   correlated returns within a game, and shared parameters — so contexts are not
   estimated independently and the effective sample size is smaller, unmeasured.
3. **Game generation is the cheap part; training is not priced here at all.**
4. ⚠ **The nearest real measurement remains unfriendly**: `--winners-only`
   scored **0.375** (§1). It is **not the same mechanism** — that filtered
   *other people's* games by outcome and discarded half the corpus, where a
   gradient signed on *our own* trajectories does neither — but it is the
   closest thing to evidence we have and it points the wrong way.

⇒ **The honest next step is the smallest real thing**: fine-tune a *small* set
of parameters on our own recorded outcomes (the day-15 recorder, §8ad, supplies
the trajectories) and A/B it at n≥2000 against a byte-identical control with the
seed floor carried in — the same discipline as §8z. **Not a league, not
from scratch, and not on faith.**

### A methods note: the first run of this probe was garbage in an instructive way

`agent0`/`agent1` in an arena archive are **seat-indexed, and the seats swap
every game** (`arena.py:280-283`). The first version read seat 0 as though it
were always agent A, averaging A's wins together with B's. The output claimed
the **+37 Elo** pair was undetectable at n=6,400 while the **+14** pair detected
comfortably, and put **61% "false positives"** on a measured null.

**Three impossibilities in one table is what made it obvious** — and the general
form is worth keeping: **a bug that biases everything toward the null looks like
a finding, not a crash.** Had the pair ordering happened to come out plausible,
this section would have reported the opposite verdict.

⚠ **And one true caveat surfaced by the same table.** The bootstrap resamples an
archive's own outcomes, so it sizes the effect that archive *observed*. The
seed-only pair observed **0.482**, and at large `n` the instrument duly
"detects" it. That is not a false-positive rate — it is a demonstration that
**past ~2,000 games the arena begins resolving its own noise floor**, and that
the measured seed deviation (0.018) is **larger than the v5 block's real effect
(0.014)**. §8aa's "+14 Elo is one noise-width", in arithmetic.

```powershell
python -X utf8 scripts/p21_rl_variance_probe.py --boot 1500
```

## 8af. 🃏 WOULD SWAPPING A CARD BREAK THE CLONE? The exposure audit, and it makes deck work SAFE-ISH rather than forbidden (2026-08-02, day 15)

**The user's question, and it is the right one:** *"Would changing a card
dramatically decrease our strength because the agent wouldn't know how to play
that new card?"* Track C has carried this as a hand-waved caveat since day 8
("every change is off-distribution for the net"). It has a mechanism, so it is
measurable. `scripts/p22_deck_change_risk.py`.

### The mechanism: a card is encoded TWICE, and only one channel is fragile

| channel | what it carries | behaviour on an unseen card |
|---|---|---|
| **derived properties** | HP, max HP, damage fraction, stage, ex/megaEx/tera, retreat cost, prize value, attached energy, cost satisfaction, `best_estimated_damage` (`features._slot_feats`, `optfeat` 25..36) | ✅ **computed from the card DB, correct on first sight** |
| **card-id embeddings** | `slot_emb`, `bag_emb`, `card_emb` (1300×16), `atk_emb` (1600×16) | 🔴 **random init — pure noise** |

### The measurement

**The training corpus contains exactly 134 distinct card ids.** So **1,166 of
the 1,300 rows in each of the three embedding tables never received a gradient
and are still random initialisation.** A card outside those 134 injects three
random 16-dimensional vectors into the sums the net reads.

⚡ **But the 134 are not our 60 — they are the FIELD's cards.** The corpus is
2,810 games of many demonstrators on many archetypes, and every one of the 134
was at some point *the acting player's own option* (1,345,021 occurrences). So
the net has chosen Mega Kangaskhan ex, Latias ex, Ultra Ball, Crispin, Area Zero
Underdepths and 129 others **from the driving seat**, despite none being in our
list.

**And the bar for "well known" is lower than it looks.** Our own least-offered
cards, by how often *we* were given them as a choice:

| our card | as OUR option | anywhere |
|---|---|---|
| Tool Scrapper | **2,820** | 41,753 |
| Unfair Stamp | 6,197 | 126,306 |
| Pokégear 3.0 | 7,759 | 157,345 |

⇒ **A swap-in with ≥ ~3,000 "as our option" occurrences is no more
off-distribution than Tool Scrapper, which we play today.**

### The verdict, and it is conditional rather than a yes or a no

1. ✅ **Swapping to a card inside the 134 is low risk.** Both channels are
   populated and the exposure is comparable to cards we already run.
2. 🔴 **Swapping to a card outside the 134 is the real hazard** — the user's
   worry is correct *for that case*, and it is the case to avoid.
3. ⚠ **Exposure is necessary, not sufficient.** It says the net has seen the
   card, not that it plays it well. **Any real swap still needs an arena A/B at
   n≥2000 against the §8ac-re-weighted anchors, with the seed floor carried in.**

### 🔴 A methods failure, the third in two days, and the same shape every time

The first version of this script tried to detect untrained rows **from the net
alone**: an untrained row sits at its init, so untrained rows should share one
tight norm. **Measured: seen rows average 4.008 and unseen 3.952, with 1,032 of
1,166 unseen rows inside the seen rows' 5–95 percentile band.** The heuristic
separated nothing and the column it printed ("row trained: yes") was meaningless
for every card in the table — including the ones it marked safe.

The bug: **the init is i.i.d. random, so every untrained row has its OWN random
norm.** There was never a cluster to find. And the corpus id set was sitting
right there as ground truth, needing no inference at all.

⇒ **Third instance in two days of the same failure: `p20_recorder_equivalence`
(a test that could not fail), `p21_rl_variance_probe` (seat-indexed archives
read as agent-indexed), and this. All three PRINTED CONFIDENT NUMBERS.** The
standing lesson is now earned three times over: **a check that cannot come out
the other way is not evidence, and the tell is a column with no variance in it.**

## 8ag. 🃏 POKÉGEAR 3.0: the user is right about the mechanism and the sizing closes it anyway (2026-08-02, day 15)

**The user's observation:** *"Pokégear lets us see 7 cards and pick one. I see
that we are picking cards but I don't think we have any mechanism of using the
knowledge."* `scripts/p23_pokegear_audit.py`, over our **75 real ladder games**.

**The mechanism claim is correct.** `features.BAG_NAMES` is `slots`, `my_hand`,
`my_discard`, `opp_discard`. **There is no bag for "cards I have seen in my
deck"**, and `optfeat` reads the `looking` zone (area 12) only while the select
is open. Once it closes, the 5.3 non-Supporter cards we saw per look — 207 over
39 looks — are gone.

⚠ **But the card destroys most of that value itself:** *"Shuffle the other cards
back into your deck."* The ordering is randomised by the card, not lost by our
encoding. What a perfect player retains is only *"these cards are in my deck,
therefore not prized"* — real, and far smaller than "I know my next draws".

### And the decision itself is already played correctly

⚡ **The engine PRE-FILTERS the options to the Supporters found in the top 7**
(options are `{"area": 12, "index": N, "type": 3}` naming only Supporters), and
`optfeat._card_at` resolves area 12, **so the net does see which Supporters are
on offer.** `minCount=0`, so declining is legal.

| | count | share |
|---|---|---|
| Pokégear selects | **39** over 75 games | **0.52 / game** |
| FORCED (1 Supporter offered — nothing can go wrong) | 19 | 48.7% |
| **REAL CHOICE (2+ offered)** | **20** | 51.3% |
| **took a Supporter** | **39 / 39** | **100%** |
| declined | **0** | — |

✅ **We never decline a free Supporter.** That is the *dominated* half of the
card (rule 11's good column, where rules have gone 3 for 3) and it is already
100% right, so **no rule can buy anything there.**

🔴 **And the remaining half is closed by sizing (rule 14): 0.27 real choices per
game.** The Morgrem out was closed **without spending an A/B** at ~0.2
firings/game (§8e), and an n=2000 arena A/B resolves ~0.021 of win rate. Which
Supporter to take is also a **tradeoff**, not a dominated choice — rule 11's bad
column, 0 for 4.

⇒ **Nothing to build here.** When 2+ were offered we took Lillie's Determination
12×, Boss's Orders 4×, Petrel 4×, and passed Petrel 14× — whether that
preference is right is exactly the judgment the net has watched 2,810 games of
humans make.

⚠ **This does NOT answer whether the 1-of Pokégear earns its slot** — that is a
deck question, not a play question, and it needs an A/B. §8af says such an A/B
is now safe to run provided the replacement is one of the 134 known cards.

```powershell
python -X utf8 scripts/p22_deck_change_risk.py --net out/policy_v5.npz
python -X utf8 scripts/p23_pokegear_audit.py
```

## 8ah. 🔴 THE FIRST ANCHOR ANYONE WATCHED WAS THROWING GAMES — day-15 item 6, and the user found it by watching (2026-08-02, day 16)

**Item 6 existed because the five anchors carry 71.5% of every weighted verdict
in this repo, they were imported rather than written here, and nobody on this
project had ever watched one play.** The user watched
`out/replays/anchor_vs_anchor/game000` and reported that the Crustle pilot never
benched a second Pokémon and lost when its active was KO'd.

**The report was correct.** `scripts/p24_anchor_pathology.py`.

### Result 1 — the game, and the reason code was NOT what established it

Game 000 is `rule:alakazam5` (seat 0) vs `rule:crustle` (seat 1), rewards `[1,0]`.
At **turn 10** seat 1 had active Dwebble (70 HP), **bench empty**, and **Mega
Kangaskhan ex** — a 300 HP card whose DB entry is `"basic": true,
evolvesFrom: null` — in hand. The engine **offered the bench play twice**, at two
consecutive decisions. Both were declined for an energy attach. Turn 11,
Alakazam attacked, Dwebble was KO'd, game over.

⚠ **The `Result` log said `reason: 3`, and that is not evidence** — real ladder
replays carry `result: 1, reason: 3` on ordinary prize-out wins. What settles it
is the **prize counts**: seat 0 won *holding 2 prizes* and seat 1 never took one,
so the only available win condition was seat 1 having no Pokémon in play.
**Read the board state, not the reason code.**

### Result 2 — the cause was one line, and it inverted a default every other pilot gets right

`agents/agentkit/rulebased/sources/crustle.py:338`, before:

```python
if data.cardType == CardType.POKEMON:
    return 25000 if card_id == DWEBBLE and len(player.bench) < player.benchMax else -5000
```

**Only Dwebble was ever benchable**; Mega Kangaskhan ex ×2 and Cornerstone Mask
Ogerpon ex scored **−5000** and lost to every other play, with **no empty-bench
guard at all**. Once the Dwebbles were gone it played on an empty bench until the
first KO ended the match. Every other pilot defaults a Pokémon to *benchable* and
subtracts for redundancy — `alakazam5`/`lucario`/`v10` **20000**, `iono`
**100000**, `abomasnow` **10000**. This one inverted the default.

Fixed: bench-full still −5000, **empty bench returns 90000** (filling it
dominates every other play in the set), otherwise Dwebble 25000 / anything 12000.

### Result 3 — 🔴 and the first detector OVERCOUNTED, which nearly produced a second false finding

The obvious detector — *"bench empty, a bench play was offered, agent chose
something else"* — **is not an error rate.** A turn is many selects, so a pilot
that plays three items and *then* benches scores three "declines" and has done
nothing wrong. On that detector `rule:archaludon` looked worse than Crustle
(1.333/game, Duraludon ×14) and **it is fine**.

The sharp detector is *"...and it ATTACKED or ENDED THE TURN anyway"*, which has
no benign reading. 12 recorded games per anchor vs `bc:v5`, plus the existing
dumps:

| agent | games | declined | **EXPOSED turn-ends** | empty-bench losses |
|---|---|---|---|---|
| **`rule:crustle` (before)** | 3 | 6 | **2 (0.667/game)** | **2 of 2 losses** |
| **`rule:crustle` (after)** | 12 | **0** | **0** | 2 of 10 |
| `rule:archaludon` | 12 | 16 | **0** | 1 of 3 |
| `rule:alakazam5` | 18 | 1 | **0** | 0 of 11 |
| `rule:lucario` | 12 | 1 | **0** | 0 of 5 |
| **`bc:v5` (ours)** | 51 | 1 | **0** | **0 of 23** |

⇒ **One real defect existed, it is fixed (0.667 → 0.000 exposed/game), and the
other four anchors — and our own net — are clean on this pathology.**

⚠ **Unresolved and deliberately not closed:** the repaired Crustle pilot still
loses **2 of 10** with an empty bench, and Archaludon **1 of 3**, in games where
no bench play was ever *offered*. That is a card-search priority question
(`wanted_card_score` stops fetching once `dwebble_total >= 3` regardless of an
empty bench), not a play-selection one, and n is far too small to call it.

### The cost, and what it does NOT license

🔴 **An anchor that throws games biases every A/B that uses it in our favour, in
the direction that looks like progress.** Our arena number vs `rule:crustle` is
**0.663** against a **57.1%** real win rate on that archetype; §8i logged that
gap as "the arena reads optimistic" and this is a concrete mechanism for part of
it. **Every verdict carrying a Crustle term is now suspect and must be re-run
against the repaired pilot.** At §8ac's re-weighting Crustle is 6.7% of the
field, so the dilution is real but the correction is owed.

⛔ **It does NOT license re-running everything on faith.** Crustle is one of five
anchors; the pathology audit covers one failure mode; and §8ab's caveat still
binds — weighted five-anchor totals are ordinal, not arithmetic.

### The methods lesson, and it is the fourth of the same shape in three days

`p20_recorder_equivalence` (a test that could not fail), `p21_rl_variance_probe`
(seat-indexed archives read as agent-indexed), `p22`'s embedding-norm heuristic
(no cluster to find), and now a pathology detector that **overcounted by design
and made a clean pilot look like the worst one on the board.** All four printed
confident numbers. ⚡ **The new part: this one was caught by asking "what is the
benign reading of this count?" before reporting it** — which is the same question
`p20`'s rewrite should have asked and didn't.

⚡ **And the finding itself came from a human watching a replay, not from any
script.** Fifteen days of arena A/Bs at n=2000 never surfaced it, because an
anchor that loses games still returns a number.

```powershell
python -X utf8 scripts/p20_record_games.py --a "bc:v5,net=out/policy_v5.npz" `
    --b rule:crustle --deck-b crustle --games 12 --out out/replays/audit_crustle_fixed
python -X utf8 scripts/p24_anchor_pathology.py
```

## 8ai. ⚡ THE EMPTY-BENCH RULE FOR OUR OWN AGENT: right shape, wrong frequency — closed by sizing (2026-08-02, day 16)

§8ah's bug is the most **dominated** option class in the game (skip it and the
next KO wins), and this project's discriminator says rules deleting a dominated
option go **3 for 3** while rules picking a side in a tradeoff go **0 for 4**. So
it was promoted as the best-shaped rule candidate in days — **and then sized
before anything was built** (rule 14).

Over our **75 real ladder games** (`replays/submission_v4` + `submission_v5`),
7,094 of our own decisions, 22 losses:

| measure | value |
|---|---|
| decisions with an empty bench | 283 (3.77/game) |
| ...where a legal bench play was **offered and declined** | **14 → 0.187/game** |
| games with ≥1 decline | 12 of 75 |
| losses matching the empty-bench signature | **1 of 22** |

🔴 **0.187 firings/game.** The Morgrem out was closed **without spending an A/B**
at ~0.2 firings/game (§8e) and Pokégear at 0.27 (§8ag). The failure costs ~1.3%
of games; an n=2000 arena A/B resolves ~0.021 of win rate. **We could not measure
the fix even if we built it.** ⇒ **Not built.**

⚠ Scope: this counts declines where the engine *offered* a bench play, which is
an upper bound on "ended a turn with an empty bench holding a benchable basic",
so the closure is sound. It does **not** measure turns where we had no basic at
all — and in **269 of the 283** empty-bench decisions we had nothing to play.
The declined cards were our own basics (Munkidori ×6, Marnie's Impidimp ×5,
Snorunt ×4); we run **10 basic Pokémon in 60**. Whether 10 is right is a genuine
deck question, but the same 1.3% ceiling caps what changing it could pay.

⇒ **Third sizing closure in three days, and the pattern is now worth naming: the
discriminator tells you whether a rule *would* work; sizing tells you whether it
would be *visible*. Both gates, in that order, before any code.**

## 8aj. 🃏 SIZING ALL 60 SLOTS — deck swaps PASS the gate that killed three rules, and the obvious candidate is disqualified by the matchup (2026-08-02, day 16)

Track C had **one** decklist A/B to its name (0.490, null) and it was chosen by
argument. §8af removed the safety excuse (swaps inside the corpus's **134** known
ids are no more off-distribution than Tool Scrapper, which we already play), so
the question became *which* slot — and that is a measurement, not an opinion.
`scripts/p25_deck_slot_audit.py`, our **75 real ladder games**, 7,094 of our own
selects, resolving every option through **`optfeat.option_features`** (the same
resolver the net uses — hand-rolling it silently misattributes decisions, because
option `type` decides whether `index` means a hand slot or an `(area,index)`).

### Result 1 — ⚡ the sizing gate that killed three rules does NOT bind here

The Morgrem out (§8e, ~0.2 firings/game), Pokégear's real choices (§8ag, 0.27)
and the empty-bench rule (§8ai, 0.187) all died because the **situation** was
rare. **A card swap is different, and the difference is exactly the frequency you
measure.** The replacement sits in the deck whether or not the old card gets
played, so the relevant rate is the **draw** rate, not the **play** rate.

Tool Scrapper is played **0.13 times/game** — under every floor above — but is
*unseen* in only **14 of 75 games**, so it is **drawn in 81%** of them and sits
dead in hand in 20%. **A swap changes something in four games out of five**,
against the ~0.021 of win rate an n=2000 A/B resolves.

⇒ 🔴 **Deck work is measurable where the play-rules were not.** That single
distinction is why Track C is worth days and the last three rule candidates were
not, and it had never been stated in this repo.

### Result 2 — 🔴 and the obvious cut is disqualified BY THE MATCHUP, not by its value

Split the same 75 games into **24 mirror / 51 non-mirror** (opponent's board ever
showed Marnie's Grimmsnarl ex):

| card | n | plays/game **mirror** | plays/game **other** |
|---|---|---|---|
| **Tool Scrapper** | 1 | **0.00** | 0.20 |
| Boss's Orders | 2 | 0.29 | 0.37 |
| **Dawn** | 1 | **0.29** | **0.25** |
| **Pokégear 3.0** | 1 | **0.33** | **0.31** |
| Rare Candy | 3 | 0.46 | 0.63 |
| Snorunt | 2 | 0.50 | 1.08 |
| Buddy-Buddy Poffin | 4 | 0.50 | 0.75 |
| Munkidori | 4 | 9.08 | 7.45 |
| **Marnie's Grimmsnarl ex** | **3** | **9.12** | 8.96 |

**Tool Scrapper is played 0.00 times in 24 mirror games, and it must be: our list
runs no tools, so there is nothing to scrap.** A mirror A/B on it would return
*"cutting it is free"* **by construction** — the matchup producing the answer
rather than the card. **This is rule 16 in deck clothing**, and it is the same
error that retired `rule:v10` on a meta share measured in the wrong band. Tool
Scrapper is anti-tool tech; it can only be judged against a tool-running anchor,
which §8ah's repair currently blocks.

⇒ **A slot audit must be run per-matchup before it can rank anything.** The
pooled table alone would have sent us to test the one card the test cannot see.

### Result 3 — the shortlist, and the variant actually built

Cuts that survive the mirror check are the ones weak in **both** populations —
**Dawn (0.29 / 0.25)** and **Pokégear 3.0 (0.33 / 0.31)**. Boss's Orders is rare
*and* game-winning, which is precisely the §8e trap, so it is not touched. Add
candidates verified inside the 134, exposure relative to Tool Scrapper's 1.00×
baseline: Boss's Orders **9.82×**, Night Stretcher 8.17×, Snorunt 7.29×, Froslass
6.31×, Ultra Ball 5.59×.

**Built: `decks/grimmsnarl_g4.py` — `Dawn ×1 → 4th Marnie's Grimmsnarl ex`.** The
cut is our thinnest genuine slot; the add is our single most-played card, sitting
at 3 of a legal 4, and it is maximally mirror-relevant — the matchup that is
**33.3% of our field, 51.1% above rating 900** (§8ac) and the only one testable
without an anchor, since both seats are our own net.

⚠ **Pre-registered prior: expect a null.** Our 60 is card-for-card the consensus
list seen **353×** among the field's strongest players, and the informative
outcome is most likely *"we measured a change and kept the list."* Run with a
**same-deck control** (`grimmsnarl` vs `grimmsnarl`, identical net both seats) to
measure the seat/variance floor, because `battle_start` takes no seed (§8ad) and
these games are not reproducible run-to-run.

### Result 4 — ✅ THE A/B RAN AND IT IS A CLEAN NULL. The list stands.

2,000 matches per arm, both seats played, **n=4,000 games each**:

| arm | score | 95% CI | W/D/L |
|---|---|---|---|
| **CONTROL** `grimmsnarl` vs `grimmsnarl` | **0.4980** | [0.483, 0.513] | 1991/2/2007 |
| **TEST** `grimmsnarl_g4` vs `grimmsnarl` | **0.4911** | [0.476, 0.507] | 1964/1/2035 |

**TEST − CONTROL = −0.0069, SE 0.0112, z = −0.61, 95% CI [−0.029, +0.015].**
The A/B resolves effects larger than **≈2.2 percentage points** and this is not
one. ⇒ **`Dawn ×1 → 4th Marnie's Grimmsnarl ex` is null. Dawn stays in the 60.**

**The pre-registered prior was right, and it was written down first** — which is
what makes this a result rather than a shrug. The consensus list seen 353× by the
field's strongest players survives the first slot we had measured grounds to
doubt.

⚡ **The control is the more valuable half, and it is new.** `0.4980
[0.483, 0.513]` over 4,000 games is the **first measured same-deck variance floor
for deck A/Bs in this project** — the deck-side analogue of §8z's seed-only null
(0.482) for nets. Every future decklist claim now has a floor to clear, and the
floor sits essentially on 0.500, which also says the harness itself is unbiased.

⚠ **A seat effect exists and is worth recording**: P0 scores **0.510 / 0.513**
against P1's **0.486 / 0.470** across the two arms. First player is worth roughly
a point of win rate. `arena.py` alternates seats so it cancels here — **but any
measurement that does not alternate is reading a ~2 pp bias as a result.**

⇒ **Track C's stewardship entry, earned rather than asserted: we sized all 60
slots, derived a candidate from our own games instead of from argument, tested it
against a same-deck control at n=4,000, and kept the list.**

```powershell
python -X utf8 scripts/p25_deck_slot_audit.py
python -X utf8 scripts/arena.py play "bc:v5,net=out/policy_v5.npz" `
    "bc:v5,net=out/policy_v5.npz" --deck-a grimmsnarl_g4 --deck-b grimmsnarl --matches 2000
```

## 8ak. 🔴 TWO DECISION-IDENTICAL AGENTS READ ≥63 POINTS APART — the LB's noise floor is LARGER THAN EVERY EFFECT THIS PROJECT HAS EVER MEASURED (2026-08-02, days 16–17)

`55169114` is **decision-identical to v5** (`55160229`): `diff -rq` over the
extracted bundles shows only `main.py` and `sa/bcagent.py` differing, and only by
health counters plus one `print`. Weights, deck, engine, every other module are
byte-identical; no branch changed. So any ladder gap between them is the
instrument, not the agent — **if the readings are settled.**

| read (UTC) | `55169114` (health) | `55160229` (v5) | gap | `55156480` (frozen) |
|---|---|---|---|---|
| 05:08 | 874.8 | 956.5 | **81.7** | 910.5 |
| 05:25 | 874.8 | 951.0 | **76.2** | 910.5 |
| 07:25 | 875.3 | 944.3 | **69.0** | 910.5 |
| **10:33 (day 17)** | **879.5** | **942.7** | **63.2** | 910.5 |
| ⚡ **2026-08-07 (day 23)** | **904.1** | **990.7** | 🔴 **86.6** | 910.5 |

### 🔴 SIX DAYS LATER THE GAP DID NOT CLOSE — IT WIDENED TO 86.6

The day-17 entry hedged that 63.2 was *"a lower bound observed at one moment
rather than a stable floor"*, because the gap had been shrinking across four
reads (81.7 → 76.2 → 69.0 → 63.2) and an obvious extrapolation was that two
identical agents would eventually converge. **They did not.** After six further
days of ladder play — far past any convergence argument — the same pair reads
**990.7 and 904.1**, a gap of **86.6**, wider than any of the four original reads.

⇒ **The 63.2 figure was not a floor and not a settling point; it was one sample
of a quantity that wanders.** The claim this section exists to support is
strengthened rather than weakened: the ladder's noise between two agents whose
true difference is *exactly zero* has now been observed anywhere from 63 to 87
points, against a largest-ever measured effect of +40.5.
⚠ **RULE 2: this is ONE reading.** It is logged, not yet quoted in
`STRATEGY` §5.1b, which continues to cite the fully-settled 63.2. Confirm with a
second read ≥1 h apart before promoting it.

📈 **Standing at the same moment: rank 129 of 6,483 at 990.7 — our best rank and
best score ever** (previous bests: 185/6,103 at 955.1, 198/6,136 at 942.7). The
board displays the best **active** submission, which is `55160229`.

### ✅ The fourth read landed and the verdict is now writable

**v5 satisfies the withheld condition.** It moved **−1.6 over 3 h 08 m**
(944.3 → 942.7) against **−12.2 over the preceding 2 h 17 m** — an order of
magnitude of deceleration. The test is not a fixed threshold but a comparison:
**the reference agent moved MORE.** `55169114` went 875.3 → **879.5**, **+4.2**,
in the identical window. Any bar that certifies the health bundle as settled
certifies v5 as settled, so the verdict is written.

🔴 **THE RESULT: two agents that make the same move in the same state read
63.2 points apart** after **15.8 h** and **23.9 h** of play. `diff -rq` over the
extracted bundles finds only `main.py` and `sa/bcagent.py` differing, and only by
health counters plus one `print` — weights, deck, engine, every other module
byte-identical, no branch changed.

⚡ **And that number is bigger than everything this project has ever measured.**

| effect | magnitude | instrument |
|---|---|---|
| **decision-identical pair, LB** | **63.2 points** | this section |
| v5 − v4, "rule 2 satisfied" (day 15) | +40.5 | LB, live-vs-live |
| §8z v4 state block | +37 Elo | arena, n=2,000 |
| §8ab drop-all-three ablation | −36 Elo | arena, n=2,000 |
| §8aa v5 pooled block | +14 Elo | arena, n=4,000 |

⇒ **Rule 2's second clause — "the LB cannot resolve an effect that size at all" —
has stopped being an argument and become a measurement.** It was inferred on day
15 from the LB's ±50–100 swing; it is now demonstrated directly, by a pair whose
true difference is **exactly zero**. **The ladder cannot adjudicate any net
change this project has produced or is likely to produce.** The arena, with its
byte-identical controls and its measured 0.482 seed floor, is not a weaker
instrument than the LB — **it is the only instrument.**

⚠ **Two things this does NOT establish, and both matter.**

1. **63.2 is a lower bound observed at one moment, not a stable floor.** The gap
   has closed **monotonically across all four reads** — 81.7 → 76.2 → 69.0 →
   **63.2** over 5.4 h — and nothing here says it stops. Quote it as *"at least
   63 points, still closing"*, never as "the noise floor is 63".
2. 🔴 **The mechanism flipped between read 3 and read 4, and that is the reason
   for caveat 1.** Reads 1–3 closed because **v5 fell**. Read 4 closed mostly
   because **the health bundle ROSE (+4.2)** — after rule 2 had already
   certified it as converged. **Both agents fail day-scale convergence**, which
   is the amendment below, arriving a second time and from the other direction.

### 🔴 The part that was already a result: rule 2's one-hour clause is not enough

**v5 satisfied rule 2 on day 15** — 955.1 twice, 61 minutes apart, and it was
written into HANDOFF as the first settled net-pair reading this project ever had.
**It now reads 944.3.** Two agreeing readings an hour apart certified a number
that moved **11 points within a day**.

⇒ **Rule 2 amendment: an hour of agreement licenses an hour-scale claim, not a
day-scale one.** Anything quoted across sessions needs re-reading in the session
that quotes it. The day-15 headline's **v5 +40.5 over v4** was built on exactly
this and is now **+32.2** on the same arithmetic — ⚠ and even that is no longer a
legal comparison, because `55156480` is **evicted and frozen** while v5 is live.

🔴 **The amendment needed strengthening the same day it was written.** The health
bundle satisfied it too — 874.8 → 875.3 over 2 h 17 m — and then moved **+4.2**
over the next three hours. **A two-hour agreement does not license a day-scale
claim either.** The operative rule is now: *quote a ladder number only from a
read taken in the session that quotes it, and only against another agent read in
the same call.*

### The active-pair model, confirmed as a side effect

`55156480` (910.5) and `55129730` (836.4) are **identical across all four
reads** spanning 5.4 h, while both active submissions moved every time. That is
the "only the latest 2 play episodes" rule observed directly rather than
inferred — evicted submissions freeze and still display.

### Standing, read 2026-08-02 10:33 UTC

**Rank 198 of 6,136 at 942.7** — our best rank, on a board that grew 6,103 →
6,136 since day 15. Top is `Luca` **1322.6** (new, and 127 points clear of the
old #1), then やる気元気ミワハルキ 1195.1, `Majkel1337` 1187.5 (was #1 at 1251.3),
`Raihan Ramadistra` 1176.3, `ntumlnoob` 1172.4. ⚠ **Our rank improved while our
score fell** (955.1 → 942.7) — the board moves under us, so rank and score are
separate readings and neither substitutes for the other.

## 8al. 🃏 THREE VARIANTS, AND STRENGTH FALLS MONOTONICALLY WITH DISTANCE FROM THE CONSENSUS 60 (2026-08-02, day 16)

Two user-directed variants were tested after §8aj's slot audit, both built on
**Budew** — a card whose case is mechanically sound: `Itchy Pollen` costs
**`energies: []`**, deals 10, and *"during your opponent's next turn, they can't
play any Item cards from their hand."* Our list runs **17 Items in 60**, and in
the mirror the opponent runs the same 17. Exposure was not a concern (§8af):
Budew appears **4,375 times as our own option**, 1.55× Tool Scrapper's.

All arms are the mirror — our own net on both seats, only the 60 differing —
against the same-deck control floor from §8aj.

| variant | swaps | score | 95% CI | vs control | p |
|---|---|---|---|---|---|
| **CONTROL** identical decks | 0 | **0.4980** | [0.483, 0.513] | — | — |
| `Dawn → 4th Grimmsnarl ex` | 1 | 0.4911 | [0.476, 0.507] | −0.0069 | 0.54 |
| `Poffin −1, Scrapper −1 → Budew ×2` | 2 | **0.4757** | [0.465, 0.487] | **−0.0223** | **0.021** |
| `Petrel −1, Poffin −2, Scrapper −1 → Grimmsnarl ex +1, Boss +1, Budew ×2` | 4 | **0.4637** | [0.453, 0.475] | **−0.0343** | **0.0004** |

🔴 **The ordering is monotone in distance from the consensus list: 0.498 →
0.491 → 0.476 → 0.464.** One swap is null; two swaps lose ≈17 Elo; four lose
≈25 Elo. Three points do not establish a functional form, and the two Budew arms
are not distinguishable from each other (−0.0120 ± 0.0079, z = −1.52) — but every
deviation is at or below the floor and **none is above it.**

### 🔴 The charitable explanation is ruled out: the net DOES play the card

The obvious defence of a losing variant is *"the clone never learned to use the
new card, so we measured two dead slots."* Six recorded games
(`out/replays/budew_v2_watch/`) say otherwise:

| | count of 6 games |
|---|---|
| Budew reached our hand | **5** |
| Budew reached the **Active spot** | **3** (all on turn 0, as the opening active) |
| **Itchy Pollen attacks fired** | **4** |

⇒ **The mechanism fires at roughly the rate the design intended and the deck is
still worse.** This is not an execution failure; the plan loses. The leading
hypothesis — **untested**, so recorded as a hypothesis — is that opening with
Budew means *not* opening with Marnie's Impidimp, delaying the
Impidimp → Morgrem → Grimmsnarl ex line in a deck whose Stage 2 is played
**9.12 times per game**. Trading a turn of setup for one denied Item play is a
bad trade when you must evolve twice.

### ⚠ What this does and does not license

**Does:** the consensus 60 — card-for-card the list seen **353×** among the
field's strongest players — behaves like a **local optimum**, and our net is
tuned to it. Combined with §8o (the deck is not the bottleneck) and §8ac (deck
shares are a function of our own rating), **Track C's experimentation half is
answered: measured, and we kept the list.** That is the stewardship result, and
it now rests on four A/Bs rather than one.

**Does NOT:** ⚠ **every arm was run in the MIRROR ONLY.** That matters
asymmetrically — cutting Tool Scrapper is **free by construction** in the mirror
(§8aj: 0.00 plays/game there, our list runs no tools), so the mirror **flatters**
these variants on that cut and they lost anyway. But a variant aimed at a
non-mirror matchup cannot be judged here at all.
⚠ **And the four-card arm is a bundle** — §8ab's "derive and size, do not bundle"
applies: it lost, and we cannot say which of the four changes lost it. That is
the acknowledged cost of testing a configuration rather than a component.

```powershell
python -X utf8 scripts/arena.py play "bc:v5,net=out/policy_v5.npz" `
    "bc:v5,net=out/policy_v5.npz" --deck-a grimmsnarl_budew_v2 --deck-b grimmsnarl --matches 4000
python -X utf8 scripts/p20_record_games.py --a "bc:v5,net=out/policy_v5.npz" `
    --deck-a grimmsnarl_budew_v2 --b "bc:v5,net=out/policy_v5.npz" --deck-b grimmsnarl `
    --games 6 --out out/replays/budew_v2_watch
```

## 8as. 🔴 THE DECK SEARCH IS CLOSED — 11 pre-registered variants, all ≤ 0, and the confirmation is negative on 7 anchors of 7 (2026-08-02, day 18)

**The design** is §8ar's, pre-registered in `out/logs/deck_search_prereg.txt` and
committed (`d93cf04`) **before any variant deck file existed** — git enforces the
ordering, which is what makes stage 2 a test rather than shopping. Two stages:
a mirror screen that **ranks** (no p-values, by construction), then a **single**
stratified confirmation of the top-ranked candidate and nothing else.

### Step 0 — the falsification check, run before anything else

Stock vs stock: **0.504 [0.488, 0.519]** over 4,000 games against §8aj's
**0.4980 [0.483, 0.513]**. Consistent, so the harness had not moved.

⚡ **It also priced a rule-18 trap caught in this script before it ran.** The
first ranker scored archives directly and derived our seat from
`deck0 == <variant name>` — correct for a variant, whose two deck names differ,
and **silently wrong for the stock-vs-stock control**, where both sides are named
`grimmsnarl` and every row reads as seat 0. This run measured **P0 52.6% vs P1
48.1%**, so that control would have returned **0.526** instead of 0.504 and
biased every candidate's ΔW downward by **0.022** — larger than any single-card
effect §8al ever measured, and it would have read as a clean finding that every
variant loses. The ranker now parses the seat-corrected `score=` line `arena.py`
already prints.

### Stage 1 — the screen. All eleven at or below stock.

| id | swap | screen | ΔW |
|---|---|---|---|
| **G** | Pokegear 3.0 → Fezandipiti ex | 0.501 [0.486, 0.517] | **−0.0010** ⭐ leader |
| J | Pokegear 3.0 → Froslass 2→3 | −0.012 / **+0.015** | −0.0016 |
| D | Rare Candy 3→2 → Ultra Ball | 0.488 | −0.0053 |
| F | Poffin 4→3 → Ultra Ball | 0.485 | −0.0063 |
| I | Tool Scrapper → Ultra Ball | −0.027 / −0.010 | −0.0066 |
| K | Dawn → Snorunt 2→3 | −0.029 / −0.033 | −0.0086 |
| A / B | Dawn, Pokegear → Ultra Ball | 0.475 | −0.0097 |
| E | Night Stretcher 3→2 → Ultra Ball | 0.472 | −0.0107 |
| H | Dawn → Latias ex | 0.450 | −0.0180 |
| C | Unfair Stamp → Ultra Ball | 0.439 | −0.0217 |

🔴 **Every one of eleven is ≤ 0; six of eight mirror candidates lose
significantly.** §8al's monotone result reproduced by a mechanical search — and
this time **the cut slots were chosen by measured liveness (§8ar) rather than
intuition**, which removes the standing objection that §8al simply picked bad
cards.

⚡ **The design's real payoff: Ultra Ball was held fixed across SIX different cut
slots and lost in all six (0.439–0.488).** That separates *"we cut six good
slots"* from *"the add card is wrong"*, and only the second explains six
independent losses. **A single A/B cannot distinguish those two, and every deck
A/B this project ran before today was a single A/B.**

### Stage 2 — the confirmation. Negative on 7 of 7.

Candidate G only, 57,600 fresh games at §8ar's Neyman allocation. **Fresh** so the
confirmation is independent of the screen.

| anchor | weight | control | G | Δ | w·Δ |
|---|---|---|---|---|---|
| mirror | 33.3% | 0.504 | 0.500 [0.492, 0.507] | −0.004 | −0.0013 |
| `rule:alakazam5` | 22.0% | 0.793 [0.785, 0.802] | 0.768 [0.759, 0.777] | 🔴 **−0.025** | −0.0055 |
| `rule:archaludon` | 8.0% | 0.688 [0.673, 0.703] | 0.656 [0.640, 0.671] | 🔴 **−0.032** | −0.0026 |
| `rule:crustle` (v4) | 6.7% | 0.764 | 0.760 | −0.004 | −0.0003 |
| `bc:garchomp` | 6.7% | 0.837 [0.821, 0.852] | 0.818 [0.801, 0.833] | −0.019 | −0.0013 |
| `rule:dragapult` | 5.3% | 0.807 | 0.787 | −0.020 | −0.0011 |
| `rule:v10` | 4.0% | 0.615 [0.593, 0.636] | 0.564 [0.542, 0.586] | 🔴 **−0.051** | −0.0020 |
| **WEIGHTED** | **86.0%** | | | | **−0.0140** |

**ΔW = −0.0140 against a design resolution of ±0.0050 — 2.8× outside, and
negative on every anchor.**
⚠ **Restated day 22 (§8ay): at the corrected field shares this is −0.0155, and
against the honestly combined resolution (game sampling ⊕ weight uncertainty,
±0.0059) it is 2.6× outside — negative in 100% of bootstraps. The weights above
are the ones published on the day; the verdict does not depend on which set is
used.** ⛔ **The pre-registered kill line is not met. G dies,
and per the pre-registration THE SEARCH IS OVER: no second candidate is
promoted.** That clause is what makes this one test instead of eleven.

### ⚡ The methodological result: the cheap screen predicted the expensive confirmation

Stage 1 called the mirror at **0.501** on 4,000 games; stage 2 says **0.500** on
15,800. **The two-stage design's central bet — that a cheap screen can rank
honestly if it is never asked to test — is confirmed on its first use.**

### 🔴 And the finding that will outlive the null: §8af is necessary, not sufficient

Two independent lines say the same thing:

- **Ultra Ball** sits at **5.59×** the corpus exposure of our weakest card and
  lost **all six** slots it was tried in.
- **Energy Switch** sits at **3.61×** and, over recorded games, was **offered 28
  times and played once**.

⇒ **Card-level exposure is not the binding constraint; card × DECK-CONTEXT is.**
The corpus knows Ultra Ball from *decks that play it* — never alongside Marnie's
Impidimp and Buddy-Buddy Poffin. **Nothing in this repo measures that, and §8af
has been quoted as if it did.**

### ⚠ What this does NOT establish

1. **Eleven of ~8,000 possible single-card swaps.** This is not proof that no
   improvement exists — it is evidence that the easy ones do not, and that the
   prior on any given swap is negative. **14 deck variants have now been measured
   across three sessions with zero wins.**
2. **Every number here is deck × how well OUR net pilots that deck** (§8ap's
   warning, one level up). A list our clone plays badly is not thereby a bad
   list — see §8at, where exactly that ambiguity is live.
3. ⚠ **The mirror cell's Δ depends on a convention**: it is a direct
   head-to-head, so referencing the measured stock-vs-stock 0.504 gives −0.004
   and the symmetric 0.500 gives 0.000. That is 0.0013 of ΔW and changes nothing.

```powershell
python -X utf8 scripts/p36_deck_search.py          # writes the 11 variants + commands
python -X utf8 scripts/p36_deck_search.py --rank   # stage 1 ranking
```

## 8at. 🃏 THE COMMUNITY LIST AND XEROSIC'S MACHINATIONS — a −0.040 that does NOT convict the card (2026-08-02, day 18)

**The user supplied a community-updated Grimmsnarl list.** Three findings before
any game was played, and they matter more than the A/B:

1. 🔴 **The list counts to 54, not 60**, and **`Special Red Card` is not
   implemented in this engine** — every card id was searched; no name contains
   "Red Card". **The community pool and this board's pool have diverged.**
2. 🔴 **It is not played here.** The 08-01 consensus Grimmsnarl list is
   **identical to `decks/grimmsnarl.py`, card for card, seen 158×**. Budew,
   Yveltal and Energy Switch appear in **zero** 08-01 lists of any archetype.
3. ✅ Completed with the user's judgement calls (one slot, +1 Poffin, is the
   assistant's and is marked as such in `decks/grimmsnarl_community.py`), it is a
   **five-card** change, not the 18 a diff against the incomplete core suggested.

### The measurements, and why the second one was necessary

| deck | cards changed | mirror score, n=4,000 | Δ vs 0.504 control | Elo |
|---|---|---|---|---|
| control (stock v stock) | 0 | 0.504 [0.488, 0.519] | — | — |
| community + Xerosic ×2 | **5** | 0.431 [0.415, 0.446] | −0.073 | ≈ −51 |
| **Xerosic ×2, ISOLATED** | **2** | **0.464** [0.449, 0.479] | **−0.040** | **≈ −28** |

⚡ **§8ab's no-bundling rule paid for itself.** The five-card bundle convicts
nothing; the two-card isolation splits it — **Xerosic ×2 carries −0.040 and the
other three changes carry −0.033.** For scale, §8al's two-card Budew swap was
−0.022, so **Xerosic ×2 costs about twice as much per card changed.**

### 🔴 But it is a PILOT result as much as a CARD result, and the replays show the mechanism

Xerosic's Machinations discards the opponent down to **3** cards. Over 6 recorded
games (`out/replays/xerosic_vs_stock`) it was:

- **offered 28 times**, opponent holding a mean of **5.2** cards — **nine of
  those offers at 7 cards**, one at 8;
- **played twice, both at hand size 4** → **1 card discarded each time**;
- best available moment would have discarded **5**.

⚠ **The 7% take rate is NOT itself the evidence** — Boss's Orders takes 6%,
because supporters are offered many times a turn and only one is playable
(`state.supporterPlayed`; the list runs 10 supporters, 12 with Xerosic).
**The evidence is the hand-size distribution.**

⇒ 🔴 **"The card is wrong for this deck" and "the card is fine and our clone
misplays it" BOTH predict −0.040.** The isolation test cannot separate them.
**Only a timing rule can** — *play Xerosic at the highest opponent `handCount`* —
⚠ and rule 11 puts that in the **tradeoff** column (it competes for the one
Supporter play), where this project is **0 for 4**.

### 🔧 A correction made before publishing, not after

The first pass read the opponent's `hand` array and reported *"both plays at hand
size 0"* — i.e. the card was a total blank, a far more dramatic claim. **That
array is hidden in the observation and is always empty**; the true count lives in
a separate **`handCount`** field. Same shape as rule 18: a plausible number, not
a crash. The figures above are the corrected ones.

## 8ar. 🃏 THE MATCHUP-STRATIFIED DESIGN, PRICED AND THEN NARROWED BY ITS OWN INSTRUMENT — 17 of 19 slots are mirror-safe, and the mirror-only critique is real but narrow (2026-08-02, day 18)

**The standing item.** §8al retired guess-a-swap and named its successor: *"the
next deck programme needs a MATCHUP-STRATIFIED SEARCH DESIGN over the whole slot
ranking"*, because **all four deck A/Bs so far were mirror-only**, which flatters
a variant cutting mirror-dead tech (Tool Scrapper: 0.00 plays per mirror game)
and cannot judge a card aimed anywhere else. §8ap then appeared to block it:
sorting the anchors by resolution sorts them by UNrepresentativeness, and
everything representative is something we beat 77–87% of the time, filed as
⛔ *"near ceiling"*.

**Both claims were checked before anything was built (rule 14), and both move.**

### Result 1 — §8ap's "near ceiling" is true in ELO units and false in the units a deck decision uses

`p33_anchor_resolution.py`. A fixed **Elo** delta maps to a win-rate delta
proportional to `p(1-p)` while the noise falls only as `sqrt(p(1-p))`, so Elo
resolution degrades as `1/sqrt(p(1-p))` — but the noise **falls** near the
ceiling, and a deck maximises the field-weighted **win rate** `W = Σ wᵢpᵢ`,
which is linear in win rate.

| anchor | p | min **WR** change @ n=8,000 | min **Elo** | n× vs mirror |
|---|---|---|---|---|
| mirror | 0.500 | **0.0110** | 7.6 | 1.00 |
| `rule:v10` | 0.569 | 0.0153 | 10.9 | 1.02 |
| `rule:archaludon` | 0.671 | 0.0146 | 11.5 | 1.13 |
| `rule:crustle` (v4, §8aq) | 0.755 | 0.0133 | 12.5 | 1.35 |
| `rule:alakazam5` | 0.789 | 0.0126 | 13.2 | 1.50 |
| `rule:dragapult` | 0.809 | 0.0122 | 13.7 | 1.62 |
| `bc:garchomp` | 0.857 | **0.0108** | 15.4 | 2.04 |

⇒ **The worst anchor in the set costs 2.04× the games for equal ELO resolution —
not "cannot resolve".** And in win-rate units `bc:garchomp` is the **most**
sensitive cell we own. §8ap's alarm is correct about nets and does not transfer
to decks.

### Result 2 — 🔴 and the case for stratifying is entirely BIAS, not precision

Weighted over all seven anchors the design resolves **±0.0050 on W**, at **57,600
games ≈ 1.1 h** at rule 7's 2–3 jobs (Neyman allocation with costs; **55%** of
the naive equal-n cost, because the mirror is a *direct* head-to-head and every
other cell needs two arms). ⚠ **But spend those same 57,600 games mirror-only and
Δ is measured to ±0.0041 — tighter.** Every game in one cell beats games spread
over seven.

⇒ **Stratifying buys an unbiased estimate of the right quantity, and pays for it
in precision.** A more precise estimate of the wrong quantity is worse, not
better — it is rule 16 with a tighter CI. So the design question is not *"should
we stratify"* but *"which cards require it"*, and that is a **liveness** question
per card per matchup, which nothing in this repo could answer.

⚠ **A model check was attempted and FAILED to be informative, which is why the
table above is stated in win-rate units throughout.** §8an's three nets × two
Crustle pilots is the only handle on how a real difference compresses near the
ceiling; predicted 0.0121 / −0.0134 against observed 0.0080 / −0.0090 with
**SE 0.0119**. Both observations sit inside one SE of the prediction *and* of
zero. **It cannot distinguish the Elo model from any other, and is reported as a
failed check rather than a confirmation.**

### Result 3 — the instrument, and it reproduces the one fact it could be checked against

`p34_matchup_liveness.py` wraps our agent so every select is tallied as it
happens — no replay files (§8ad's recorder writes multi-MB per game, which is
~7 GB at this n), and **the wrapper travels with the agent rather than a seat
index**, so rule 18's seat bug cannot arise. Option → card resolution uses
`optfeat.option_features`, the same resolver `p25` and the net itself use.
400 games × 7 anchors.

✅ **Positive control, unprompted:** it puts **Tool Scrapper at 0.02 plays per
mirror game against 0.28 field-weighted (spec 0.93)** — independently recovering
the single fact §8al built its whole critique on, without being told.

### 🔴 Result 4 — the finding: the mirror-only critique is REAL but NARROW

| card | mirror | alakazam5 (22%) | crustle (6.7%) | weighted | spec |
|---|---|---|---|---|---|
| **Tool Scrapper** | 0.02 | 0.56 | 0.47 | **0.28** ⚠ under every sizing floor | **0.93** |
| **Froslass** ⭐ | 1.44 | **5.57** | **6.83** | **3.14** | **0.54** |
| Snorunt | 1.53 | 2.41 | 2.31 | 1.77 | 0.13 |
| Rare Candy | 0.82 | 0.99 | 0.97 | 0.91 | 0.10 |
| *(15 others)* | — | — | — | — | **≤ 0.05** |

**17 of the 19 distinct cards in our 60 have spec ≤ 0.16, and 15 have ≤ 0.05.**
⇒ **For almost the whole deck the mirror is an UNBIASED screen**, and the
expensive stratified design is needed for exactly **two** slots.

🔴 **§8al's Tool Scrapper example was not representative — it is the single most
extreme card in the list.** The mirror-only critique stands as stated and is far
narrower in scope than the plan built on it assumed.

⚡ **And the one card that matters is the one ROADMAP already named.** Track C
step 4 says *"the Froslass line is the only growable passive-damage line
(Munkidori capped at 4)"*. It is played **1.44×** per mirror game and **5.57 /
6.83** against alakazam5 and Crustle — **a mirror A/B sees under a quarter of its
real use.** Tool Scrapper, the other mirror-blind slot, is at **0.28 weighted
plays/game, under every sizing floor this project has killed a rule at** (§8e
0.2, §8ag 0.27, §8ai 0.187), so its blindness is real and its ceiling is not.

⚡ **The bias runs the other way too, and nobody had looked:** Munkidori (18.6
mirror vs 11.3–14.2 elsewhere), Marnie's Impidimp (10.6 vs 6.4–7.7) and Morgrem
(8.4 vs 4.1–6.3) are played **more** in the mirror. ⇒ **A mirror-only A/B
overstates the core engine as much as it understates the tech**, which is the
same error with the opposite sign and was not part of §8al's argument.

### ⚠ What this does NOT establish

1. **Liveness is not value.** A card played 0.1×/game can win those games
   (Boss's Orders — §8e's trap, restated in `p25`'s own footer). This says
   **where** a swap could pay and caps **how much**; it never says a card is bad.
2. **These are arena games we win 75–82% of**, not ladder games. Liveness
   measured while dominating need not equal liveness in a close game.
3. **`spec` is defined against OUR anchor set and OUR §8ac weights**, both of
   which §8ac showed are a function of our own rating.
4. ⚠ **One cross-check is marginally disjoint:** `bc:garchomp` read 0.802
   [0.761, 0.839] here against §8ap's 0.857 [0.841, 0.872]. Its deck and pilot
   files predate that measurement (checked, §8aq), and **seven cross-checks at
   95% produce one marginal failure by construction** — so this is logged, not
   chased.

```powershell
python -X utf8 scripts/p33_anchor_resolution.py
python -X utf8 scripts/p34_matchup_liveness.py --games 400
```

## 8ay. 🔴 THE WEIGHTING LAYER AUDITED: the field census split archetypes by CARD PRINTING, and the weights it feeds come from 75 games (2026-08-06, day 22)

**Why this exists.** §8ax audited the arena — the **Δ** in every headline. Every
headline is **W = Σ wᵢ Δᵢ**, and nobody had ever audited the **w**. They come
from `p9_field_census.py` over our own ladder replays (§8ac), and they set the
weight on every anchor in `p33.ANCHORS`, `p37`, E7 and E8.

### Defect 1 — evolution lines were keyed by card ID, and a name has many

`_evolution_index` resolved `evolvesFrom` (a **name**) to a single card id and
took the first match. **106 basic printings share a name with another, and 228
links were broken.** A deck seen through Abra #741 was labelled "Abra"; the
identical deck seen through Abra #109 was labelled "Alakazam". The function's own
docstring says it exists to stop exactly this ("naming that deck 'Kadabra' and
the next one 'Alakazam' splits one archetype in two") — one level down.

It reached the anchors: **Riolu #677 and #974 both lost Mega Lucario ex**, so
three `lucario_v10` games labelled as "Hariyama", and `rule:v10`'s share is a
published weight. Fixed by indexing the whole line graph **by name**; linking
every printing is not enough on its own, because `_signature` groups by ROOT and
two printings of one basic are two roots.

### Defect 2 — `ex` outranked copy count, and the 1-of guard did not reach it

Repairing defect 1 exposed a second one it had been masking: the ranking tried
every `ex` line before considering copies, so a **2-of** tech beat a 4/3/3
engine and five Abra/Kadabra/Alakazam games labelled "Dudunsparce ex". Same trap
as the Fezandipiti one the docstring already records, one copy higher. Ranking is
now **evolved-line first** (a deck evolves its engine and merely plays its
support basics — the only signal here about deck ROLE), then total copies across
the line, with `ex` as a tie-break.

### Measured against a hand-check of all 75 games

| | old | corrected |
|---|---|---|
| correctly labelled | **69 / 75** | **74 / 75** |
| mirror | 33.3% | **32.0%** |
| alakazam5 | 21.3% (`p33` used 22.0%) | **25.3%** |
| crustle | 6.7% | **8.0%** |
| `rule:v10` | 4.0% | **5.3%** |
| archaludon / garchomp / dragapult | 8.0 / 6.7 / 5.3% | unchanged |

⚠ **The one remaining error is a mirror game** (`4xMunkidori, 2xSnorunt,
2xMarnie's Impidimp, 2xFroslass, 1xMarnie's Morgrem`) where the opponent's
Grimmsnarl ex never reached play, so the Snorunt/Froslass support line outweighed
the Impidimp line and it labels "Froslass". **Partial observation is the limit
here, not the tie-break order** — no ordering of these features fixes it without
breaking the Alakazam-vs-Dunsparce cases. A real fix is matching observed cards
against the known `decks/*.py` 60s; that is a build, and it is not done.

### 🔴 The finding that dominates both defects: n = 75

| anchor | share | 95% Wilson |
|---|---|---|
| **mirror** | 32.0% | **[22.5%, 43.2%]** — 20.7pp wide |
| alakazam5 | 25.3% | [16.9%, 36.2%] |
| crustle / archaludon | 8.0% | [3.7%, 16.4%] |
| `rule:v10` | 5.3% | [2.1%, 12.9%] |

**Every correction above is inside the interval of the estimate it corrects.**

### What it does to the published verdicts — bootstrap over the 75 labels

| verdict | published | corrected | weight-only 95% | sign stable |
|---|---|---|---|---|
| `p37` deck search (§8as) | −0.0140 | **−0.0155** | ±0.0031 | 100% |
| E8 v7 (§8aw) | −0.0099 | **−0.0078** | ±0.0023 | 100% |

⚡ **Weight error bites in proportion to how much the per-anchor deltas DIFFER.**
p37's deltas are all the same sign, so ±20pp on a share moves ΔW by ±0.0031;
against p37's ±0.0050 game-sampling resolution that is not negligible — combined
**±0.0059**, an 18% widening of an interval that was quoted as if the weights
were exact. For E8 the ±0.025 game noise swamps it entirely: **the weighting
layer is not E8's problem, the 2-seed budget is**, exactly as day 21 said.

✅ **No verdict changes.** p37's ΔW is still 2.6× outside its kill line and
negative in 100% of bootstraps; E8 is still an unresolved null with the point
estimate on the wrong side.

🔴 **But E8's −0.0099 was ALSO an arithmetic error, independent of the weights.**
Its own table gives mirror Δ = 0.487 − 0.5 = −0.0128, and the archive confirms
0.4872 pooled over 3,000 games. Recomputed from the published table with the
published weights: **−0.0078**, not −0.0099. The total implies a mirror Δ of
−0.019 that appears nowhere. ⚠ The table also prints a **score** in a column
headed **Δ** for the mirror row and deltas everywhere else, which is how it
happened.

```powershell
python -X utf8 scripts/p9_field_census.py --dir replays/submission_v4 replays/submission_v5
```

## 8ax. 🔴 THE CRUSTLE ANCHOR CHANGED **DECK** AS WELL AS PILOT, AND THE DECK IS THE BIGGER TERM — §8an and §8aq both attribute it to the wrong thing (2026-08-06, day 22)

**How it surfaced.** An audit of the local validation flow, not a new
experiment. `arena.build_agent` archived a rule pilot as `rule:<name>` with **no
deck**, so `out/arena/*.jsonl` was asked which 60 the Crustle anchor was actually
piloting in each published run. The answer splits perfectly, and not by pilot
version:

| deck played | runs | our score |
|---|---|---|
| `crustle_v1` (the pilot's own list) | p10, p19, p20, p34, p35, p37 ctrl | 0.768 / 0.7885 / 0.768 / 0.748 / 0.755 / 0.764 |
| 🔴 `crustle` (field consensus) | **p27, p28** — §8an's and §8aq's own v2/v3 rows — plus p54, p56, p57 | **0.870 / 0.866** |

The two decks differ in **20 of 60 slots**. `decks/crustle_v1.py`'s own docstring
already said what that does: *"A pilot run on the other list scores ~20 of its
cards through a generic fallback, so it plays them legally but badly."* HANDOFF
§3.2 even carries an n=20 probe reading **0.620 on its own list vs 0.700 on the
consensus one** — the right sign, and most of the magnitude, sitting in the repo
the whole time.

**The measurement** (`p58_crustle_deck.py`, `out/arena/p58_crustle_deck.jsonl`).
One net (`policy_v5`), **one pilot — the v4 in the repo today** — two decks,
back-to-back in one session, n=2,000 per cell:

| deck | our score | 95% CI |
|---|---|---|
| `crustle_v1` | **0.7530** | [0.734, 0.772] |
| `crustle` | **0.8930** | [0.879, 0.906] |
| **DECK TERM** | 🔴 **+0.1400** | two-cell resolution ±0.031 |

✅ **Positive control:** the `crustle_v1` cell reproduces §8aq's p35 (0.755) at
0.753, so this is the same instrument §8aq measured, not a drifted one.

🔴 **+0.140 is larger than either effect the two sections published** — §8an's
+0.087…+0.102 "empty-bench guard" and §8aq's −0.111 "Dwebble tie-break". Both
comparisons straddle a deck swap: §8an's v1 row is `crustle_v1` and its v2/v3
rows are `crustle`; §8aq's 0.866 is `crustle` and its 0.755 is `crustle_v1`.
**Neither isolates the pilot.** That statement needs no model — it is what the
archives say.

### What the pilot terms actually are

Every *same-deck* pilot comparison available, which is the part that needs no
assumption:

| held fixed | pilot change | our score | Δ |
|---|---|---|---|
| deck `crustle_v1` | v1 (no guard) → v4 (shipped) | 0.768 → 0.753 | **−0.015** |
| deck `crustle` | v4 → v2 (guard + bench-anything) | 0.893 → 0.870 | **−0.023** |
| deck `crustle` | v4 → v3 (guard only) | 0.893 → 0.866 | **−0.027** |

⚠ **Every pilot term is ≤ 0.027 — a fifth of the deck term.**

**Under additivity** (stated as an assumption; no cell exists for v1/v2/v3 on the
other deck), §8an's observed +0.102 decomposes into deck **+0.140** and pilot
**−0.038**, and §8aq's observed −0.111 into deck **−0.140** and pilot **+0.027**.
Two consequences:

1. ⚡ **§8ah's originally EXPECTED sign is restored.** §8ah predicted the broken
   pilot was flattering us and that repairing it would *lower* our score. §8an
   found the opposite and called it out in bold — *"the expected sign was the
   other one"*. With the deck term removed the repair is worth **≈ −0.04 to
   us**, i.e. the repaired pilot is stronger, exactly as predicted. **The
   surprise was the confound.**
2. 🔴 **§8aq's headline reverses.** *"WHICH Pokémon a pilot benches matters more
   than WHETHER it benches"* compared a tie-break at −0.111 against a repair at
   +0.09. The same-deck numbers are **+0.027** and **−0.038**: comparable, both
   small, and if anything *whether* is the larger. The "one line worth more than
   the whole repair it was a footnote to" is a deck swap.

### What does NOT move

✅ **No net-vs-net verdict changes, for the reason §8an already gave** — both
arms of every published difference faced the same pilot *and* the same deck, so
a level shift on one anchor cancels in the difference. §8an's Result 1 (the §8ah
alarm is retired) stands on that argument, which was never the confounded one.
✅ **E6/E7/E8 are internally clean** for the same reason: `p54`/`p56`/`p57` ran
treatment and control against `crustle` back-to-back.
⚠ **But the E-series' Crustle cell is not the anchor table's Crustle cell.**
`p33.ANCHORS` weights Crustle at 0.755 (`crustle_v1`); E8 measured its 6.7% term
against `crustle`, where we score 0.893. The weight was calibrated on a different
instrument from the one the experiment used.

### The fix

`arena.build_agent` now archives a rule pilot as **`rule:<name>@<deck>`** and
prints a loud warning when the deck is not the one `DECK_MODULE` says the pilot
was tuned for. The two Crustles can no longer pool under one identity.

⇒ **HANDOFF rule 20.** Rule 19 said *an anchor is a file*. It is a file **and an
argument**, and rule 19's timestamp check cannot see the argument.

```powershell
python -X utf8 scripts/p58_crustle_deck.py --matches 1000
```

⚠ **The open piece, named rather than buried:** the decomposition assumes the
deck and pilot terms add. Confirming it directly needs the v1/v2/v3 pilots run
on `crustle` (or v2/v3 on `crustle_v1`), which means restoring them from
`b7869d2` / `83daa48`. **Not run.** The two model-free facts above — a +0.140
deck term, and both published comparisons straddling a deck change — are enough
to retract the attributions without it.

## 8ce. 🔴 E24 — E21's VOID ARM RE-RUN WHERE THE CONDITION CAN EXIST: the board fact reaches 1,045 decisions and buys +0.0041. Tradeoff rules go 0-for-7 (2026-08-11, day 30)

Pre-registered in `docs/experiments/E24-fscrap-anchor.md`, frozen at **`2d36ce8`
before the first arena game of any cell** (renumbered E23 → E24 before write-up:
E22's frozen doc reserves E23 for value iteration). Sizing
`scripts/p89_fscrap_sizing.sh`, driver `scripts/p90_e23_run.sh`, scorer
`scripts/p90_e23_score.py`, logs `out/logs/p89_*.txt` / `out/logs/p90_e23_*.txt`,
archives `out/arena/e23/`.

**Why it exists: a VOID cell is a debt, not a result.** §8cc ran `fscrap` in the
mirror and it read 0.5175 with **`fetch=0/3082`** — our 60 runs zero Pokémon
Tools, so "a Tool is on THEIR board" is unsatisfiable there *by construction*.
§8cc's own verdict was **"size the condition IN THE MATCHUP THE CELL WILL RUN
IN"**. This is that verdict executed, and nothing else about the rule changed.

### The sizing, run first and ON-POLICY

The flag is ON in every sizing run, so `fetch_fired` is what the rule really
does. 200 games per anchor, each rule pilot on its own tuned 60 (rule 20); Tool
counts from the card DB (`cardType == 2`):

| anchor | deck | Tools in the 60 | fired/game |
|---|---|---|---|
| `rule:v10,noS` | `lucario_v10` | 1 | **0.300** |
| `rule:lucario` | `mega_lucario_ex` | 1 | 0.270 |
| `rule:archaludon` | `archaludon_ex` | 1 | **0.225** |
| `rule:dragapult` | `dragapult_ex` | 1 | 0.200 |
| `rule:crustle` | `crustle_v1` | 1 | 0.115 |
| mirror (§8cc) | `grimmsnarl` | **0** | **0.000** |

⚡ **`fetch_fired` is NOT the treatment size, and this experiment adds the
counter that says so.** A firing the net agrees with is a no-op for the A/B.
`bcagent.STATS` now carries **`fetch_diff`** — firings whose pick differs from
`net.choose` — and it ran at **93%** of firings, consistent with §8br's
off-policy "conditioned on a target existing we take Scrapper 1 time in 14".

### The cells — two-cell deltas, treatment vs a byte-identical control

Same net (`policy_v5_s2#4790c469`), shipped rule config, differing **only** by
the flag; the anchor is identical in both arms, so the delta's interval is √2× a
single cell's (§8aw) and the scorer prints that width.

| cell | anchor | treatment | control | **delta** | 95% CI | z | n/arm |
|---|---|---|---|---|---|---|---|
| **a (primary)** | `rule:v10,noS`@`lucario_v10` | 0.6512 | 0.6471 | **+0.0041** | [−0.0168, +0.0250] | +0.39 | 4,000 |
| **b (exploratory)** | `rule:archaludon`@`archaludon_ex` | 0.7033 | 0.7350 | −0.0317 | [−0.0596, −0.0039] | −2.24 | 2,000 |
| **b2 (replication)** | same as b, fresh games | 0.7165 | 0.7220 | −0.0055 | [−0.0334, +0.0224] | −0.39 | 2,000 |
| **b + b2 pooled** | — | 0.7099 | 0.7285 | **−0.0186** | **[−0.0383, +0.0011]** | −1.85 | 4,000 |

✅ **Both controls passed in every cell, which is the whole point of re-running
it.** Treatment reached **0.261 changed picks/game** in cell a (1,045 of 1,123
firings over 5,619 fetches) and 0.168 / 0.179 in b / b2 — all far above the
pre-registered 0.10 VOID bar. And **the control arms printed no `fetch=` field
at all**, so the arms differed only in the flag.

⚡ **Realized firing came in slightly UNDER the sizing** (0.261 vs 0.300; 0.168
vs 0.225), which is the *opposite* direction from §8cc's `fstad` (0.72 realized
vs 0.461 sized, 1.6× over). ⇒ **On-policy firing is not biased in a fixed
direction against replay sizing — it is simply a different quantity.** §8cc's
"sizing under-predicts" should be read as "sizing does not predict", which is
the claim the two measurements jointly support.

### 🔬 The harmful branch fired on cell b, and the frozen rule dissolved it

Cell b tripped the pre-registered harmful branch by **0.0017**, with its CI
clearing zero by **0.0039**. Before running anything else I wrote the reading
rule into the pre-registration and committed it (`7d576da`), because the reasons
to disbelieve it were available *before* the replication existed:

* the implied effect is **−0.19 win probability per changed fetch** against cell
  a's **+0.016** — a 12× disagreement in magnitude and a disagreement in sign,
  for the same rule and the same card;
* it is the **second of two cells**, with no multiplicity correction.

**b2 replicated it unchanged on fresh games and read −0.0055, CI containing 0.**
Per the frozen rule (b2's CI contains 0 **and** the pooled CI contains 0) ⇒
**cell b was a sampling artifact and no matchup-specific harm is claimed.**

⚠ **Stated rather than smoothed: the pooled interval clears zero by 0.0011.**
Pooled b+b2 is −0.0186 [−0.0383, **+0.0011**], z = −1.85. That is a *hair*, and
the honest summary is **"not resolved as harm at n=4,000"**, not "archaludon is
clean". The rule was frozen before the data existed and it is not being moved
after the fact; a third archaludon cell would settle it and is **not indicated**,
because the primary is a clean null and nothing here can ship.

### What this closes

- 🔴 **H-scrap is refuted on the primary anchor**: the board fact reached
  **1,045 decisions** and moved the score **+0.0041 [−0.0168, +0.0250]**. E21b's
  0.5175 is now retired as the wiring statement it was, with a measured
  counterexample beside it.
- ✅ **Tradeoff rules go 0-for-7** (§3's discriminator). E21a's `fstad` lost at
  z = −5.36; `fscrap` merely does nothing. **The class has now failed at both
  ends of the firing-rate range it can reach** — 0.72/game and 0.26/game.
- 🔴 **A correction to E21's own filing, and it is the reusable half.** E21 filed
  `fscrap` as *"closer to the dominated column"*. It is not: as implemented it
  **promotes** Tool Scrapper over Unfair Stamp and Night Stretcher, all live
  cards, which is a tradeoff. The genuinely dominated version — *delete* Scrapper
  from the fetch when no Tool is anywhere — sizes at **0.066/game** (§8br) and
  **~0.08/game** in the mirror. ⇒ **On this seam, the rule class the discriminator
  says WINS is an order of magnitude too rare to measure, and the class it says
  LOSES is the only one large enough to test.** That is why the seam produces
  nulls and losses and never a win, and it is a stronger statement than either
  cell.
- ⛔ **Scope, written before the result and unchanged by it:** our 60 runs no
  Pokémon Tool, so this rule **cannot fire in the mirror** — 71.4% of our field
  above rating 1000 (§8ac). Even a win would have been matchup tech, not a ladder
  lever.
- ⛔ **Nothing ships.** Both E21 flags stay in the tree, OFF by default.

## 8cd. 🔴 E20 IS REFUTED — a learned V + one-ply argmax reads 0.0065 at n=2,000; a real train/inference defect was found on the way and turned out NOT to be the cause (2026-08-11, day 30)

Pre-registered in `docs/experiments/E20-value-lookahead.md`, frozen at
`5811b9a` **before V was trained and before any arena game**. Kaggle harness
`scripts/kaggle/`, V trainer `scripts/train_value.py`, agent `agents/sa/vlook.py`,
probes `p86`/`p87`/`p88`.

### ✅ First, the instrument: a Kaggle arena harness, commissioned on two controls

`arena.py` sharded across Kaggle kernels (4 vCPU each, two concurrent), pulled
back and pooled from **the `score=` line the arena already prints** — never
re-derived from the archives, because `agent0`/`agent1` are seat-indexed and
re-deriving is rule 18's exact bug.

| control | pooled, n=2,800 | pre-registered | |
|---|---|---|---|
| **C0** identical arms (`s2` v `s2`) | **0.5082** [0.4897, 0.5267] | 0.500 | ✅ |
| **C1** `s2` v `s1` | **0.4996** [0.4811, 0.5182] | 0.510 (§8bh) | ✅ |

Health clean on all 8 shards (`fallbacks=0 net_missing=0`), net hash archived.
⚡ **A by-product worth carrying:** C1 is the *third* independent reading of
`s2` v `s1` — screen **0.537** → §8bh fresh games **0.510** → here **0.4996**.
Pooling §8bh's 1,400 with these 2,800 gives **0.5031 [0.488, 0.518]** ⇒ the
shipped net's seed edge is **≈ +2 Elo with a CI containing zero**. §8bh took it
from +25.8 to +7.0; this takes it to nothing, and it further weakens any
seed-harvest plan.

### ✅ V(s) trains, and it reproduces §8az's warning on self-play data

`scripts/train_value.py` — **the trainer `sa/valuenet.py` has cited in its own
docstring since day 1 and which had never existed.** 1,914,025 rows / 20,000
self-play games from B8's corpus, on the `won` column, split **by `gid`**:

```
ep 1  val 0.5037  AUC 0.8258
ep 2  val 0.5026  AUC 0.8270   <- best, exported
ep 3  val 0.5082  AUC 0.8273
ORIENTATION  V|won 0.7089  vs  V|lost 0.3602
```

**AUC 0.827 held out by game** against `evalfn`'s 0.685 early (§8m). And E1's
overfit-after-one-epoch warning (§8az) reproduces on self-play data — val turns
up at epoch 3 — which is why early stopping and the export rule were pinned in
the pre-registration rather than chosen afterwards.

### 🔴 The defect, and it invalidates the cell

`train_value.py` pads an **empty card bag** with row 0 (`EmbeddingBag(mode="mean")`
returns NaN on a truly empty bag). `sa/valuenet.py` substituted **zeros**. The
weights were therefore fitted against `bag_emb[0]` and the agent scored with a
different vector. `p88_value_equivalence.py`, 3,000 corpus rows:

| | max \|diff\| | verdict |
|---|---|---|
| `valuenet.py` **as shipped** (empty → zeros) | **0.126210** | 🔴 not equivalent |
| after the fix (empty → `bag_emb[0]`) | **0.00000027** | ✅ equivalent |

**7.0% of rows carry an empty bag**, and the within-position spread across
sibling successors is only **0.186** — so the perturbation is comparable to the
entire signal an argmax depends on. ⚠ **And it is structured, not random:** a
hand empties exactly when it has been played out, so the error lands on the
options that spend resources.

### The readings

| cell | reading | status |
|---|---|---|
| E20 primary, broken path | **0.0040 [−0.0179, +0.0259]**, n=2,000 | 🔴 VOID — measured a mismatched component |
| **E20 primary, corrected path** | **0.0065 [−0.0154, +0.0284]**, n=2,000 | ⚠ **the pre-registered HARMFUL branch** |

🔴 **THE DEFECT WAS REAL AND IT WAS NOT THE CAUSE.** Repairing a mismatch worth
0.126 on 7% of evaluations moved the win rate **0.0040 → 0.0065** — nothing.
⇒ **H-eval, in E20's pre-registered form (argmax over a learned V across every
option), is REFUTED on a valid measurement.** 13 wins in 2,000 games.

⚡ **And that ordering is the reusable lesson: finding a real defect is not the
same as finding the explanation.** The fix was mandatory for correctness and
irrelevant to the result, and only re-running distinguished the two. Had the
corrected cell not been run, "we found the bug" would have stood in for a
verdict — the §8ah shape (a genuine defect, correctly repaired, that changed
no published number) one level over.

### ✅ The withdrawn diagnosis is REINSTATED, weaker, on the corrected path

`p87` re-run through the fixed evaluator, 16 games:

| | broken path | **corrected path** | chance |
|---|---|---|---|
| V's argmax == the clone's pick | 6.1% / 8.5% | **11.1%** | **20.5%** |
| within-position spread / across | 0.215 | 0.183 | — |

**Still below chance**, so the anti-selection is real and "one-sided
extrapolation error under a max" survives — but the defect inflated it, and the
published 6.1% overstated the effect by roughly half. ⚠ The AUC figure from
that probe stays retracted at any n this small; **only the per-decision
agreement statistic is quoted, because it is not clustered by game.**

### 🔴 Two method failures, both already in this file's own catalogue

1. **A 2,000-game A/B ran on a component whose inference path had never been
   reconciled with its trainer.** Two implementations of one function is
   precisely the situation **rule 18** covers — *compute the headline a second
   way and reconcile before writing a word*. The check cost ten minutes.
2. **A clustered statistic was read as evidence.** "Live AUC 0.7042 ⇒ plumbing
   is sound" pooled ~1,000 decisions from **15 games**; the effective n is
   games. Re-running the same probe read **0.5222**. §8bw caught this exact
   clustering in its own estimator (intervals widen 4.1×) and it was repeated
   two messages after being quoted.

### ⛔ What is WITHDRAWN, not merely unconfirmed

The diagnosis published from the broken path — *"one-sided extrapolation error
under a max"*, supported by argmax agreeing with the clone at **6.1%** against a
20.3% chance rate, and by a top-3 coverage constraint lifting it to **37.9%** —
was measured through the same defective evaluator. **Both are withdrawn.** They
may well survive re-measurement; they are not evidence today.

### What stands

- The Kaggle harness, commissioned (C0/C1), and the pooling discipline in
  `scripts/kaggle/score.py`.
- V itself: AUC 0.827 by game, and now **verified equivalent** through the path
  that plays.
- `p86`'s structural fact: after one `fs.step` the observation indexes players
  **absolutely** (`players[me]` is still ours 505/517 = 98%) and the mover
  changes seat only **1%** of the time, so a pre-step seat stays valid — which
  refuted the seat hypothesis written into `vlook.py`'s own docstring.
- **E22** (`docs/experiments/E22-pessimistic-lookahead.md`, renumbered from E21
  because §8cc owned that number) is pre-registered and **NOT run**. Its
  5-seed V ensemble is trained (`out/value_e0..4.npz`, AUC 0.827-0.829). Its
  baseline now exists: **0.0065**.
- ⚠ **What E20 does NOT refute.** It tested ONE consumer of V — an unconstrained
  argmax. V ranks *games* at AUC 0.827 and the failure is in how the search
  reads it, not in whether the state is evaluable. E22's two changes (LCB
  pessimism, top-k coverage) target exactly the measured mechanism, and the
  honest prior after E20 is that they must clear **0.530 from 0.0065**, which is
  an enormous distance. ⛔ **If E22 also fails, the axis closes**: the evaluator
  is not the missing piece the three dead searches lacked, and that is the
  report chapter.


## 8cc. 🔴 E21 — THE CLONE'S BOARD-BLIND FETCH BEATS A BOARD-AWARE RULE, DECISIVELY (0.4405, z=−5.36) — and the second arm never fired at all (2026-08-11, day 30)

Pre-registered in `docs/experiments/E21-petrel-fetch.md`, frozen at **`5be502d`
before the first arena game of either cell**, prediction included. Driver
`scripts/p87_e21_run.sh`, logs `out/logs/p87_e21_*.txt`, archives
`out/arena/e21/`. User-directed: *"test in the arena to see if the experiments
pan out instead of comparing if the experiments make us similar to the experts."*

**H-fetch:** §8br showed a Petrel fetch option's vector carries **no board at
all**; supplying the board fact directly, at the one select that cannot see it,
should beat the clone. Two conditions, both from card text rather than expert
behaviour (§8u: agreement with the expert anti-predicts strength):

* `fstad` — fetch Spikemuth Gym when no Stadium is in play or the one in play is
  theirs. Sized offline at **0.461 firings/game**, the largest rate this seam
  has produced.
* `fscrap` — fetch Tool Scrapper only when a Tool is on THEIR board. 0.171/game.

### 🔴 The primary cell: not a null, a LOSS

Mirror, byte-identical weights both sides (`out/policy_v5_s2.npz#4790c469`),
4 shards × 500 games, pooled from arena's own printed W/D/L (rule 18):

| cell | score | 95% CI | n | verdict |
|---|---|---|---|---|
| **`fstad` vs `base`** | 🔴 **0.4405** | [0.4187, 0.4623] | 2,000 | **HARMFUL branch — z = −5.36** |

✅ **Control 1 passed decisively**, which is what makes this readable: the rule
fired **1,439 / 3,023 fetches = 47.6%, 0.72 per game**. This is not the §8be
family — the intervention happened and it lost.

⚡ **The realized firing rate is 1.6× the offline sizing (0.72 vs 0.461/game),**
because off-policy replay sizing cannot see that the rule changes the
trajectory it is measured on. **Sizing from recorded games under-predicts
on-policy firing** — worth carrying into every future rule-14 estimate.

### 🔴 The second cell is VOID, and it is rule 16 in a new costume

`fscrap` reads **0.5175 [0.4956, 0.5394]** — and **`fetch=0/3082 (0.0%)`. It
never fired once.** Per this experiment's own control 1, *"a null with a zero
firing count is a statement about the wiring, not about H-fetch"*, so the score
is not a result about the rule.

🔴 **The cause was already in this file and I did not apply it.** `decks/grimmsnarl.py`
runs Tool Scrapper and **zero Pokémon Tools**, so in the MIRROR neither side can
ever put a Tool on the board and the rule's condition is unsatisfiable *by
construction*. §8aj recorded exactly this for the same card — *"Tool Scrapper is
played 0.00 times per mirror game, so a mirror A/B would return 'cutting it is
free' by construction — rule 16 in deck clothing"* — and §8br's own addendum
lists the tools our real opponents carry (Air Balloon 178, Hero's Cape 87 for
the experts). **The 0.171/game sizing came from ladder games against a mixed
field and does not transfer to the mirror.** ⇒ **Size the condition IN THE
MATCHUP THE CELL WILL RUN IN**, not on the corpus that suggested it.

⚡ **The one thing it did buy:** with zero firings the two arms are behaviourally
identical, so the cell is an accidental **C0 control** — identical agents should
read 0.500, and it reads 0.5175 with the CI containing 0.500. The harness is
sound, which is worth having under E21a's −5.36.

### 🔬 The audit the "harmful" branch demanded — and the harm does NOT generalise

Diagnostic, **not a ship path** (no confirmation cell, no anchor sweep, and no
threshold tuning is permitted — the pre-registration forbids it). Both arms
against `rule:v10,noS` on `lucario_v10`, where Spikemuth Gym is **asymmetric**
because the opponent runs no Marnie's line:

| arm | score | 95% CI | n |
|---|---|---|---|
| `fstad` vs `rule:v10` | 0.6670 | [0.6378, 0.6962] | 1,000 |
| `base` vs `rule:v10` | 0.6530 | [0.6235, 0.6825] | 1,000 |
| **delta** | **+0.0140** | **[−0.0275, +0.0555]** | — |

🔴 **The mirror's −0.0595 lies outside the audit's interval.** The injected fact
is not wrong in general — **forcing it is wrong where the Stadium is
SYMMETRIC.** In the mirror both players run Marnie's lines, so Spikemuth Gym
tutors for the opponent exactly as much as for us, and we spend a Petrel — a
scarce tutor — to hand both sides the same engine. ⚠ Two-cell delta, so its
width is √2× a single cell's (§8aw); it separates the two matchups, it does not
license anything.

⚡ **The deeper reading, and it is about scarcity rather than symmetry:**
Spikemuth Gym is a **4-of** that arrives on its own, while the cards the rule
displaces are a **1-of ACE SPEC** (Unfair Stamp) and a 3-of (Night Stretcher).
Spending the deck's only any-Trainer tutor on its most plentiful card is the
error, and the net had already learned not to do it (5.1% take rate, §8br).

### What this closes

- 🔴 **H-fetch is REFUTED in the mirror, which is 71.4% of our field above
  rating 1000 (§8ac).** The clone's board-blind fetch is **better** than the
  board-aware rule, by 6 points of win rate at z=−5.36.
- ⚡ **This is the first rule in this family to produce a DECISIVE signal rather
  than a null, and the sign is negative.** §8br called the fetch "not a worse
  judgement but the absence of one"; E21 shows that where we can express a
  judgement cheaply, the clone's is better. ⇒ **The Petrel seam is closed a
  second time, now on strength rather than on sizing.**
- ✅ **Tradeoff rules go 0-for-6**, and this is the class's strongest test: the
  largest firing rate it has ever had (0.72/game), so *"too rare to matter"* was
  unavailable as an excuse. The prediction written before the run (§E21, "my
  prediction, written first: `fstad` reads NULL") was **wrong in magnitude and
  right in direction** — it lost by more than predicted.
- ⛔ **Nothing ships.** Both flags stay in the tree, OFF by default. E20 owns the
  submission slots.

## 8cb. ⚡ THE TWO NAMED PETREL SCENARIOS: one is a NON-EVENT, the other is a REAL blind spot the experts do not share — and both die at the sizing gate anyway (2026-08-11, day 30)

**Hypothesis (user, day 30):** §8br closed Petrel *by sizing*, on a marginal
take-rate distribution. A marginal rate can hide a conditional policy that is
simply absent. Two spots were named where the fetch might be not-merely-different
but **wrong**: (A) their Active carries a Tool — do we fetch Tool Scrapper?
(B) do we fetch and play Unfair Stamp on a **strong hand** merely because it is
legal?

`scripts/p86_petrel_scenarios.py`, log `out/logs/p86_petrel_scenarios.txt`.
Corpus: our 76 ladder games (`replays/submission_v5_s2`) and, as the control
that decides whether any gap is a *defect*, 1,972 games from three current
Grimmsnarl pilots at `avg_score` ≥1100 (§8bq). Card ids come from
`optfeat.option_features`; the mapping control is p76's and it re-ran clean at
**1331/1375 = 96.8%**.

### 🔴 A. Tool Scrapper — a non-event, and the experts are WORSE at it than we are

§8br's bucket was "a tool anywhere on THEIR board", pooling the Active (where
scrapping changes this turn's maths) with the bench (where it usually does not).
Split, over fetches where Scrapper was still in the deck:

| board at the fetch | ours | took | rate | experts | took | rate |
|---|---|---|---|---|---|---|
| **THEIR ACTIVE has a tool** | 9 | 1 | **11.1%** | 149 | 5 | **3.4%** |
| their bench only | 5 | 0 | 0.0% | 47 | 2 | 4.3% |
| no tool anywhere | 73 | 5 | 6.8% | 409 | 9 | 2.2% |

⛔ **Neither side conditions on it.** The experts' own lift for a live target is
**3.4% vs 2.2%** on n=149 — the honest read of "does a 1150 pilot fetch Scrapper
when there is something to scrap" is **no**. Our 11.1% is 1 event on n=9 (95% CI
≈ [0.3%, 48%]) and cannot be distinguished from either.

⚠ **And the situation barely arises: 9 fetches in 76 games = 0.12/game**, against
the 0.5/game gate — before any rule's take rate is applied. What we fetched
instead: Unfair Stamp ×5, Poké Pad, Lillie's, Rare Candy. The experts, given the
same board, took Unfair Stamp 43×, Night Stretcher 26×, Boss's Orders 23×,
Spikemuth Gym 20×. **Tool Scrapper loses the slot to tempo and disruption for
everybody**, which is §8br's Scrapper addendum arriving from the other direction.

### 🔴 B. Unfair Stamp — the net makes NO decision here, and this one the experts DO make

⚠ **The per-select denominator says the opposite and it is an ORDERING artifact
(rule 21, third instance).** Counting every MAIN select where Stamp was legal:
56 plays / 172 offers = **32.6% taken**, mean hand **4.73** when played vs
**5.48** when "declined" — which reads as a policy that plays Stamp on small
hands. It is not. There are **3.1 such selects per turn**, and we decline holding
8, play four other cards, then Stamp holding 4. The ordering-free unit is the
**turn**, scored at its first offer:

| | turns Stamp legal | played that turn | rate |
|---|---|---|---|
| **ours** | 56 | 56 | **100.0%** |
| ours, turn opens H≥7 | 18 | 18 | **100.0%** |
| **experts** | 530 | 508 | **95.8%** |
| experts, turn opens H≥7 | 175 | 163 | **93.1%** |
| experts, turn opens H≤4 | 157 | 154 | 98.1% |

🔴 **56 of 56. Unfair Stamp is not a decision for our agent — legal implies
played**, which is exactly what §8br's structural addendum predicts (a fetch
option's vector carries no board; everything situational must survive the head
MLP's interaction with a shared state vector).

⚡ **And the experts' 22 declines are legible, which is what makes this a real
gap rather than noise.** At the turn's first offer, declining turns have a
**bigger** hand (6.82 vs 5.72), **more legal plays** (4.00 vs 2.67) and a
**smaller opponent hand** (5.82 vs 8.45) — precisely "I have a working hand and
there is nothing over there to strip". The user's hypothesised condition exists,
is used by stronger players, and is absent from our policy.

### ⛔ Rule 14 kills both, and the card maths says the unconditional policy is nearly right anyway

Unfair Stamp shuffles both hands away; we draw 5, they draw 2. At a play with our
hand H (Stamp included) and theirs O, our net is `5−(H−1)` and theirs `2−O`.

| | ours (56 plays) | experts (508 plays) |
|---|---|---|
| mean card differential (our net − their net) | **+8.32** | +7.96 |
| plays that lost the exchange on raw cards | **1/56** | 6/508 |
| plays strictly DOMINATED (we lose cards, they do not) | **0/56** | 3/508 |

⛔ **RETRACTED IN THE SESSION THAT WROTE IT (user challenge: "how do you know it
is *right*?").** This paragraph first read *"the unconditional policy is right 55
times in 56"*. **The differential does not support that and barely reflects our
decision at all.** Expand it:

```
D = [5-(H-1)] - [2-O] = 4 - H + O
        var(H) = 3.34      var(O) = 27.37      var(D) = 31.93
        share of var(D) from O alone = 85.7%
```

🔴 **86% of the metric's variance is the OPPONENT's hand size — a quantity our
policy does not control.** At the median opponent hand (O=7), **0 of 56 plays
read negative at any of our observed hand sizes**, so "55 of 56" mostly records
that our opponents held a lot of cards. **A 100%-unconditional policy and a
perfect one score alike on it.** ⚠ And it is blind to card *quality*, which is
exactly what "a strong hand" means — shuffling away a 4-card hand that is
precisely Rare Candy + Grimmsnarl ex + energy is a disaster at D=+7. ⚠ Our 1/56
against the experts' 6/508 is **1.8% vs 1.2%, indistinguishable**, and was
presented as if it favoured us.

✅ **What the table DOES establish, and it is narrow:** we do not walk into the
**dominated** case defined before the run — we lose cards while they gain —
at **0/56** (experts 3/508). A specific failure mode is excluded. **That is not
evidence the policy is right.**

⛔ **Knowing whether it is right needs outcome linkage** (do big-hand Stamps lose
more games?), which **18 big-hand plays in 76 games cannot resolve**, and the
oracle route is closed by §8ca (rollout value under clone-vs-clone continuation
does not transfer). ⇒ **The question is OPEN, not answered in our favour.**

⚡ **The verdict below is unaffected, and the reason matters:** it never rested
on this paragraph. The sizing (0.031 declines/game) kills the rule whether the
policy is right or wrong, which is the whole point of a sizing gate — it is
prior to the question of correctness. **The single worst play is H=9 into O=3,
1 event in 76 games = 0.013/game.**

⛔ **Sizing the expert-matching rule:** they decline on **4.2%** of legal turns,
and Stamp is legal on **0.74 turns/game** for us ⇒ **0.031 declines/game**, 16×
under the gate that killed Morgrem (0.2), Pokégear (0.27), Archaludon (0.187),
Petrel-as-a-whole (0.29) and the WP-regret seam (0.039). Scenario A is 0.12/game
before a take rate is applied, 4× under.

### What this closes, and what it does not

- ⛔ **Both named scenarios close by sizing**, joining §8bm/§8bp/§8br/§8bs. Six
  user-named seams, six kills, still no arena time spent on any of them.
- 🔴 **But §8br's verdict is now sharper and worse-sounding: on Unfair Stamp the
  policy is not a worse judgement, it is the ABSENCE of a judgement (56/56).**
  §8br said "we over-fetch Unfair Stamp +17.8%" — that gap is the marginal shadow
  of a decision the net never makes. **A marginal take-rate table cannot
  distinguish a bad policy from no policy**, and this is the first place the
  project has separated them.
- ⚡ **Method, and the reason the first table was wrong:** rule 21 caught a third
  victim. A per-select rate over a 3.1-select turn measured within-turn ordering
  and reported it as judgement, with a plausible mean-hand gap (4.73 vs 5.48) in
  the *expected direction* to make it convincing. **Sixth "confident but wrong
  first table" in this project, and the first one caught by choosing the unit
  before reading the number rather than after.**
- ⚠ **A script defect worth recording because it is a class:** `--us` used
  argparse `action="append"` with `default=["Scio"]`, and **append does not
  replace a default** — so `--us flg` meant `{Scio, flg}` and the first expert
  run carried our own seat under the experts' label. It changed nothing here
  (Scio contributes 0 rows to dumps floored at `avg_score` 1100, §8bq — the B0
  counts were 272/278 both before and after the fix), but the same pattern on a
  corpus we *do* appear in silently pools the arms of a comparison.

## 8ca. 🔴 E19 CELL A KILLS THE CLOCK, AND IT KILLS E17's HEADLINE WITH IT: at exactly one deviation per game the rollout's +0.035 delivers nothing (2026-08-10, day 29)

Pre-registered in `docs/experiments/E19-clock-mechanism.md`, frozen at `aa19968`
**before either cell's first game**. Driver `scripts/p85_e19_run.sh A`, log
`out/logs/p85_e19_capA.txt`. `bc:cap,orc,od1` vs `bc:base`, mirror, net
`#4790c469` on both sides.

| cell | reading |
|---|---|
| **score, one overrule per game** | **0.4963 [0.4719, 0.5207]**, n=1,608 |
| overrules actually taken | **0.93/game** (cap held: `skip_capped`=49,578) |
| rollout errors | **0.0%** over 137,158 rollouts |
| worst remaining 600 s pool | 502.1 s ✅ |

### The two hypotheses, and the one that survives

E18 left the puzzle: search picks the genuinely better option (67% best-arm,
z=5.5, +0.035 per overrule against stored ground truth) and the win rate reads
0.4828. Two explanations, pre-registered with **point** predictions:

| | claim | predicted | observed |
|---|---|---|---|
| **H-compound** | the value is real but only for a ONE-STEP deviation; E18 deviated 3.32×/game and played a policy it never evaluated | **0.535** | ❌ **3.1σ away** |
| **H-fusion** | the rollout value is biased and never transferred | **≈0.500** | ✅ **0.4963** |

🔴 **At cap=1 the one-step assumption is EXACTLY satisfied and the effect is
still zero.** That is the whole point of the design: a +0.035 gain in win
probability at one decision *is*, by definition, +0.035 on that game's win rate
if the estimate is unbiased. It is not there. ⇒ **the estimate is biased.**

⚠ **Stated at the right strength.** The upper bound is **+0.021** against a
predicted +0.035, so the one-step value is overstated by **at least ~40%** and
is consistent with **zero**. We can exclude +0.035; we cannot exclude +0.01.

### 🔴 What this retracts, and it is more than the agent

**E17's headline measured the same biased quantity.** §8by's +0.0139/decision,
§8bz's +0.0353/overrule, the 67% best-arm rate and the +0.120 scale bar (§8bw)
are all *rollout* values under clone-vs-clone continuation. E19 is the first
test of whether that currency converts to won games, and **it does not**. ⇒
**ROADMAP §2.7's entire sizing framework rested on an unvalidated assumption**,
and the E17 gate it passed should be read as "the rollout says there is value
here", not "there is value here".

⚡ **The instrument was verified and still measured the wrong thing.** Every
control held — C0 99.8%, C1 100%, identical arms read zero, the autopsy showed
selection working at z=5.5 against stored truth. **A component can be correct
in every internal check and still not measure the quantity you want**, and no
amount of internal validation substitutes for the end-to-end test.

### The leading mechanism, named but not isolated

**Determinization.** We sample the opponent's *whole deck* from a library by
best overlap, then inside each sampled world the rollout plays as if the hidden
cards were known — the classic strategy-fusion failure of perfect-information
Monte Carlo. A line that only works because the simulation "knows" the world
scores well and gains nothing in a real game. Supporting sign from §8bz: **37%
of overrules take an option the net scores >3 worse**, i.e. lines the field-mode
clone would never choose, which is exactly where a fused evaluation would be
most wrong. ⛔ **Not isolated by experiment** — the alternative (the opponent
model is simply wrong) is untested, and distinguishing them needs an
information-set-aware rollout that does not exist here.

### ⛔ The pre-registered consequence: cell B is CANCELLED

E19 §"Ordering and dependency" says it in advance: *"If A supports H-fusion, the
per-decision value never transferred and no trigger can help."* **Cell B — the
validated half of the user's `wp<0.50` trigger — would have measured a dead
premise at ~8 h of compute, and is not run.** ⚡ Its offline finding stands on
its own as a report result and is worth keeping: *"we are losing"* concentrates
the (rollout-measured) value at **+0.0150/decision on 22% of firings**, while
*"the net is confused"* is **refuted** — adding `margin<1.5` lowers it to
+0.0108, because the wins come from options the net scored **>3 worse**, where
it was confident and *wrong*, not where it was torn.

### ⚠ A comparison that does NOT support what it looks like

E18 (no cap) 0.4828 → cell A (cap=1) 0.4962 looks like capping helped. It is
**+0.0134 [−0.0414, +0.0682], z=0.48** — an interval four times the effect.
⛔ **Do not report capping as an improvement.** The informative contrast is not
cap-vs-no-cap (two noisy runs differenced) but cap **against its own point
prediction**, which is why E19 was designed to make a point prediction at all.

⇒ **The clock is CLOSED as a Round-1 axis.** ⚡ What survives for Round 2 is
narrower and honest: the failure is in the **evaluator**, not the idea. A
search whose rollouts are information-set-aware, or a *learned* value function
trained on real outcomes rather than determinized simulations, is untouched by
this result — and that is the value-based policy-iteration family §2.7 already
names as never-tried.

## 8bz. 🟡 E18 — THE CLOCK PLAYS THE GAME AND WINS NOTHING: search selects demonstrably better options (z=5.5) and the win rate reads 0.476 (2026-08-10, day 29)

Pre-registered in `docs/experiments/E18-clock-arena.md`, frozen at `cc070b0`
**before the first game**. Agent `agents/sa/oracle.py`, driver
`scripts/p83_e18_run.sh`, scorer `scripts/p83_e18_score.py`, autopsy
`scripts/p84_oracle_autopsy.py`. Net `#4790c469` on **both** sides — the only
difference is whether the oracle runs, so the seed nuisance cancels.

| cell | reading |
|---|---|
| **score, oracle vs the shipped agent, mirror** | **0.4764 [0.4281, 0.5252]**, n=403 |
| as P0 / as P1 | 0.5248 / 0.4279 |
| worst remaining 600 s pool | **251.6 s** ✅ never near the 45 s reserve |
| rollout error rate | **0.0%** over 187,403 rollouts |

🟡 **By the pre-registered rule this is INCONCLUSIVE, and §3 named that as the
EXPECTED outcome**: n=400 carries SE ≈ 0.025 and detects a true +0.03 with
probability **0.34**. ⛔ It is **not** a kill — the KILL branch required the
interval to exclude 0.500 downward and it does not. ⛔ And it does not ship:
the point estimate is below 0.500.

### ✅ The component is NOT broken, and this is the part that makes the null mean something

A sub-0.500 A/B has two incompatible readings — *the clock does not help* and
*the clock is inverted* — and they demand opposite responses. `p84` separates
them against ground truth the project already owns: **E17 stored 50 paired
rollouts per arm at 300 real positions**, so each option's value is known to
±0.04. Replay those exact positions through the **live** `RolloutOracle.choose()`
and score the option it returns against E17's stored means.

| | reading |
|---|---|
| **picked the genuinely best arm** | **40/60 = 67%** against a 1/3 null ⇒ **z = 5.5** |
| value of the live oracle's picks | +0.0112 [−0.0016, +0.0239] |
| uniform choice over the same arms | −0.0119 |
| perfect oracle over the same arms | +0.0243 ⇒ it captures **46%** |

⚡ **The agreement rate is the decisive statistic, not the mean gain.** The mean
is diluted by every position where the arms are genuinely close — most of them
— so it is underpowered by construction; picking the best of three at 67%
against 33% is not. ⇒ **selection works, so the null is a fact about the CLOCK,
not the wiring.**

### 🔴 What the extra time actually CHANGES — the diagnostic that makes the null readable

Without this, a null reads identically whether search agreed with the net
everywhere (nothing could move) or overruled it constantly and gained nothing.
It is the latter.

| | reading |
|---|---|
| decisions where search chose **differently** | **19/60 = 32%** (live A/B: **3.32 overrules/game** of 7.98 fires) |
| true gain **when it overruled** | **+0.0353 [−0.0035, +0.0740]** |
| overrules that were genuine improvements | **13/19 = 68%** |
| how the **pre-search net** scored the option search took | **3.01 below its own top-1** |
| …overrules taking an option the net scored **>3 worse** | **37%** |
| value **left on the table** when it kept the net's pick | **+0.0090 [+0.0036, +0.0145]** |

⇒ Search overrules a third of the time, is right about two thirds of those,
takes options the clone actively dislikes, and is if anything **under**-firing
(the "left on the table" interval excludes zero).

### 🔴 The mechanism the null most likely reveals, named rather than hand-waved

Every number above measures **Q^π(s,a) — the value of a ONE-STEP deviation**,
with the clone playing everything afterwards. That is exactly what E17 measured
and what the rollout computes. **But the deployed agent deviates 3.32 times per
game**, so it is not a one-step deviation from π — it is a different policy, and
the estimate is strictly valid only for the first switch. E17's pre-registration
flagged the symptom (*"per-decision gains do not add"*) without naming the
mechanism; this is the mechanism.

⚠ **A second, sharper worry the diagnostics raise.** 37% of overrules take an
option the net scores **>3 below** its own top-1. Those look good *under a clone
continuation* — and the clone is precisely what plays them out in the rollout.
A line that only works because the simulated continuation mishandles it will
score well here and gain nothing in a real game. **This is strategy fusion's
cousin and it is untested.**

### ⚡ Confirmed: the 6.9% rollout error rate was a MEMORY LEAK, not a logic bug

`fs.release(root)` reclaims **one** search node; a rollout creates a fresh id at
each of up to `ROLLOUT_CAP` steps. Measured **1.68 GB in 8 minutes and climbing**
per process, against **0.063 GB flat** after switching to `fs.end()` — a 23×
reduction, and six shards went from ~10 GB (unrunnable on a 7.9 GB box) to
0.39 GB. **The clean run reports 0.0% errors over 187,403 rollouts**, which
confirms the diagnosis. ⛔ **160 games collected before the fix were DISCARDED,
not pooled** — a memory-starved oracle fires less and declines silently, so
those games measure a different agent.

### ⛔ Three process defects, each of which would have produced a confident wrong number

1. **`arena.py` defaults to the `sample` deck**, where **79% of decisions carry
   ≥12 options against 19.7% on grimmsnarl** — so the free trigger fired on
   **0.7%** of decisions instead of 24%. A `sample` A/B measures a component
   that barely fires and returns a null. **Rule 20 / §8ax one seat over**,
   caught by the option-count histogram in the health line.
2. **A `python -c` process killer matches its own command line *and its parent
   shell's*** — a kill loop SIGTERM'd the bash that was about to write the E18
   runner, so the "launch" produced nothing and a later check reported four
   phantom shards by matching itself. `scripts/killarena.py` now lives in a file
   and excludes its own parents.
3. **A tool-level 600 s timeout killed the first attempt at 160 of 408 games** —
   and killing the wrapper does **not** kill the python grandchildren, so a
   half-dead run keeps writing rows. Long runs launch with no timeout, and
   completion is verified by counting shard files, never by the wrapper's own
   "done" message.

### What is owed next

⚠ **The diagnostics make a falsifiable prediction and it is the natural next
cell:** if over-deviation is the mechanism, an oracle that overrules **less**
should score **better**. E17's τ sweep keeps the full per-decision value at
**7% of decisions instead of 38%** (τ=0.15: +0.0146 corrected vs +0.0139). ⛔ τ
is **post-hoc** and must be pre-registered before it is run.

## 8by. 🟡 E17 — THE CLOCK'S OWN GATE: a budgeted rollout oracle over the net's OWN options is worth +0.014/decision, the 600 s is not the resource, and 57% of our decisions carry nothing (2026-08-10, day 29)

Pre-registered in `docs/experiments/E17-self-oracle-value.md`, frozen at
`c0a2cc9` **before the driver existed**. Driver `scripts/p82_e17_self_oracle.py`,
log `out/logs/p82_e17.txt`, net `#4790c469` (the shipped `55326513` weights).
**300 treatment positions and 300 control positions**, each × 3 arms × 50
paired replicates on shared worlds — ~90,000 rollouts.

**Why it existed at all.** §2.7's sizing gate had already passed on §8bx's
dispersion — but that was measured between **our** pick and the **expert's**,
at positions selected because a 1050+ expert disagreed with us. The clock has no
expert: it ranks **the net's own options** at positions selected by nothing, and
§8bx's caveat (3) says so itself. The second unmeasured quantity was the
multiplier the whole build rested on — caveat (1)'s *"roughly half survives"*,
an arithmetic guess. Selecting on noisy estimates is what killed F2 (§8bh).

### The pre-registered readings

| # | quantity | reading |
|---|---|---|
| **Q1** | Δ(top-2 − top-1) / Δ(top-3 − top-1) | **−0.0078** [−0.0205, +0.0049] / **−0.0186** [−0.0356, −0.0016] |
| **Q2** | TRUE between-position sd of the top-2 gap | **0.1045** ⇒ typical \|gap\| **0.083** |
| **Q4** ⭐ | realized gain of a budgeted oracle, R_sel=30 | **+0.0163 [+0.0060, +0.0266]**, k=300 |
| **Q4 corrected** | minus the identical-arms control | **+0.0139 [+0.0027, +0.0250]** |
| scale bar | §8bw clone top vs last | +0.120 |

✅ **Q1 held as predicted: the net's own ranking is right on average.** There is
**no free re-ranking** — all of the value is in per-decision *dispersion*.
⚡ And the dispersion among **our own** options (0.1045) is **larger** than
§8bx's our-pick-vs-expert-pick figure (0.0768 / 0.0866). The quantity §8bx
could only call "similar but not identical" is in fact bigger.

🟡 **VERDICT: the NARROW branch**, by the letter of the rule frozen at
`c0a2cc9` — +0.0163 is above the +0.010 kill line and below the +0.020 build
line. NARROW licenses a build **only** if a free, online-computable trigger
selects ≥25% of decisions at ≥ +0.030, control-corrected:

| trigger (free at play time) | corrected gain | share | |
|---|---|---|---|
| **option count ≤ 5** | **+0.0373 [+0.0109, +0.0638]** | **36%** | ✅ passes |
| turn ≥ 11 | +0.0488 [+0.0097, +0.0878] | 21% | ❌ share |
| score margin < 1.5 | +0.0187 [+0.0034, +0.0341] | 67% | ❌ size |
| option count ≤ 3 | +0.0319 [−0.0204, +0.0841] | 11% | ❌ both |

⚠ **The gate is met on the POINT estimate; the interval is not.** [+0.0109,
+0.0638] contains plenty of values under 0.030, so "≥ +0.030" is a point
estimate, not an established fact. ⚡ **But the two live triggers are
independent** — 19 positions carry both against 23 expected by chance — so they
are two distinct handles on the value, not one covariate in two costumes.

### 🔴 Three findings that change ROADMAP §2.7's own design

**1. The 600 s is NOT the resource — allocation is.** The empirical budget
curve saturates almost immediately:

```
R_sel      5      10      20      30      40
Q4    +0.0122 +0.0143 +0.0158 +0.0163 +0.0164
```

**An 8× budget increase buys +0.004, and it is flat by 20 pairs/arm.** §2.7's
framing ("we use 1.12 s of a 600 s budget") points at the wrong quantity.

**2. §2.7's play-time arithmetic used the wrong denominator, and the correction
is large.** It divided the budget across **~318 selects**. The oracle only fires
at live MAIN decisions with ≥3 single-pick options, and there are **47.1 per
game** (3,581 over 76 games), not 318. At 90 rollouts × 100 ms:

| firing rule | decisions/game | think time | of the 600 s budget |
|---|---|---|---|
| every qualifying decision | 47.1 | **424 s** | 71% |
| **option count ≤ 5** | **17.0** | **153 s** | **25%** |
| turn ≥ 11 | 9.9 | 89 s | 15% |

⇒ **the play-time budget is not binding, and no batching is needed to PLAY.**
§2.7's "10–15× of engineering" is required only for **validation**.

**3. 57% of our decisions carry no value at all**, and that is the single most
actionable number here. WP estimated on the **front half** of the replicates,
gain on the **disjoint back half** (§8bw M3's warning, obeyed):

| position win probability | gain/decision | share |
|---|---|---|
| < 0.15 | **+0.0743** | 12% |
| 0.15 – 0.50 | **+0.0671** | 10% |
| 0.50 – 0.85 | −0.0079 | 20% |
| **> 0.85** | **+0.0015** | **57%** |

**The value is where we are LOSING.** A clone move in a won position has nothing
to improve on; comeback lines are exactly where field-modal play is worst.
⚠ WP is not free at play time — but `evalfn` reads **AUC 0.905 late** (§8bs),
and the value is concentrated late, so the cheap proxy is accurate precisely
where it is needed.

### ✅ The controls, and the scare that was not real

**C0 caught the run being pointed at the wrong network before a single
treatment rollout.** `pnet.get()` returns whatever `SA_PNET_PATH` says, and the
repo default `agents/sa/policy_net.npz` (`#a25b904d`) is the **v2** clone —
three generations behind the agent that played these games. C0 read **67.3%**
on it and **99.8%** on `out/policy_v5_s2.npz`. p80/p81 depended on an
environment variable for this; **the net is now pinned in the script.**
⚡ Separately verified that arm 0 really is the shipped agent's move:
`argmax(scores) == net.choose` **106/106**.

**The identical-arms control (the winner's-curse control) passed at every
budget** — but its point estimate at k=100 was **+0.0055 and rising with
budget**, a third of the treatment effect and exactly the shape a residual bias
would have. The suspect was structural: **arm 0 is always rolled first *and* is
the baseline in every estimator.** ⚠ **The obvious placebo does not
discriminate** — arm2 − arm1 reads zero under "no bias" *and* under "arm 0 is
low". So 200 more control positions were collected, and at **k=300 the order
effect reads +0.0017 [−0.0036, +0.0071]** against +0.0053 at k=100. **It was
noise.** The control at k=300 is +0.0025 ±0.0042, and the corrected effect is
+0.0139 [+0.0027, +0.0250].

⚡ **The general lesson:** a control that *passes* its pre-registered test can
still carry a point estimate large enough to matter, and "the CI covers zero" is
not the same as "the bias is zero". The cost of resolving it was 12 minutes.

### 🔴 A model-based ceiling would have been wrong by 4×

Q3's normal fit says the perfect-oracle ceiling is **+0.0749** (62% of the scale
bar) and that a budgeted oracle keeps 92% of it at R_sel=30. The **empirical**
curve saturates at **+0.0164**. Normality fails because the gap distribution is
a **spike at zero plus a heavy tail** — 57% of decisions read +0.0015. ⇒ making
the split-sample estimator the pre-registered **primary** and labelling the
model as model-based is the only reason +0.075 did not become the headline.
⚠ **§8bx's own +0.0372 oracle figure rests on the same normal assumption and
should be read the same way.**

### ⚠ Exploratory, and labelled: the deviation threshold

The oracle overrules the net on **38%** of decisions at τ=0, and §8bd measured
that deviating at ~half of decisions is expensive when the deviations are not
earned. Requiring a minimum estimated gap first:

| τ | fires | treatment | control | corrected |
|---|---|---|---|---|
| 0.00 | 38% | +0.0164 | +0.0025 | +0.0139 [+0.0027, +0.0250] |
| 0.10 | 12% | +0.0153 | +0.0008 | +0.0145 [+0.0046, +0.0244] |
| 0.15 | **7%** | +0.0146 | +0.0000 | **+0.0146 [+0.0051, +0.0242]** |

**Five times fewer overrules for the same value.** The treatment is nearly flat
in τ while the control collapses — the shape a bias concentrated in marginal
calls would make. ⛔ **τ was NOT pre-registered and six values were swept**;
this is the shopping a B8 β-sweep was declined for on day 18. It needs its own
pre-registered confirmation before any build leans on it.

### ⚠ What E17 does NOT establish

1. **Per-decision WP gains do not add across a game.** 17 decisions at +0.037 is
   **not** +0.63. E17 cannot produce a win rate; only an A/B can.
2. **The value is win probability under clone-vs-clone continuation** (§8bw). In
   a mirror A/B the opponent *is* the clone, so the rollout model is exactly
   right there and **flatters the design**; against the real ladder it is a
   model mismatch.
3. **Determinization is not free of bias.** We sample the opponent's *whole
   deck* from a library by best overlap, and inside each determinized world the
   rollout plays as if the hidden cards were known. Averaging outcomes does not
   restore the information-set structure.
4. **The trigger is post-hoc** among four candidates, and the ≥0.030 gate is met
   on the point estimate only.

⇒ **The gate passed on its own terms and the decision is now about SCHEDULE,
not evidence.** With the corrected arithmetic an n=2,000 mirror A/B at 153 s a
game is ≈**85 core-hours** — one overnight run if `arena.py` can use the 6 local
cores, ~4 days if it cannot. **That is the number the build decision turns on,
and it should be measured before any agent is written.**

## 8bx. 🔴 E16 IS A NULL — the 1050+ experts' moves are worth +0.007 over ours at their own positions, and the 1100+ band reads exactly zero (2026-08-10, day 28)

Pre-registered in `docs/experiments/E16-counterfactual-move-value.md`, frozen at
`ed22624` **before the first treatment cell**. Driver `scripts/p81_e16_move_value.py`,
log `out/logs/p81_e16.txt`, net `#4790c469` (the shipped `55326513` weights).

**The question §8u left open and no conformity metric could answer:** *in THIS
exact position, is their move better than ours?* Fork a real expert position,
force their move in one arm and our net's top-1 in the other, let the clone
pilot both seats to a terminal state, difference the win rates, cluster on the
position.

**Population:** `replays/mirror_experts`, 257 games, seats rated ≥1050 on the
08-09 board. **2,457 disagreements / 3,724 agreements — our net's top-1 differs
from the expert on 39.8% of their MAIN decisions**, which independently
reproduces §8q's ~40% miss against the #2 player.

| cell | reading |
|---|---|
| **Δ(expert − ours)** | **+0.0066 [−0.0018, +0.0150]**, k=600 positions × 30 pairs |
| agreement control (identical arms) | **−0.0009 [−0.0144, +0.0126]** ✅ |
| scale bar (§8bw, clone's own top vs last) | **+0.120** |

⛔ **The interval covers 0. By the pre-registered criterion E16 is a NULL.** The
point estimate is **5.5% of the scale bar** — the experts' move advantage, if it
exists at all, is a twentieth of the gap between the clone's own best and worst
option.

🔴 **The pre-declared rating split sharpens it instead of rescuing it:**

| band | Δ | k |
|---|---|---|
| 1050–1075 | +0.0296 ±0.0546 | 18 |
| 1075–1100 | +0.0070 ±0.0092 | 489 |
| **1100+** | **−0.0000 ±0.0217** | **93** |

**The strongest band shows exactly nothing.** "Their moves are better" predicts
the opposite ordering. Turn buckets show no coherent pattern either
(+0.009/+0.016/−0.001/+0.016/−0.008 over five buckets).

⚡ **This is the third independent route to the same place, and the first by
OUTCOME rather than behaviour.** §8bj (F1) dissolved the mirror disagreement
clusters under an on-policy control; §8bl (E11) took a real, sized,
ordering-free behavioural difference to an A/B and got 0.487. E16 skips
behaviour entirely and scores the moves by simulated result. ⇒ **the expert gap
is not in per-move choice quality.**

⚠ **WHAT THIS DOES NOT SAY, and the distinction is the whole next step.** E16
measures the **mean** — |E[X]|. A mean of zero is fully compatible with a large
per-decision gap whose **sign varies**, i.e. E[|X|] ≫ 0, with the experts simply
landing on the right side of it no more often than we do. That is the same
|E[X]| vs E[|X|] distinction E15's pre-registration drew about §8bd's near-tie
band. **The dispersion, not the mean, is what an oracle could capture** — and it
is the sizing gate for the whole "spend the 600 s clock" axis (ROADMAP §2.7).
⛔ The per-position values were **not saved** by the first run, so the estimate
was recomputed properly rather than backed out of a summary interval.

### ✅ THE DISPERSION RUN — and it is the first lever this project has sized onto the RIGHT side of its own instrument

Independent sample, `--positions 220 --pairs 40`, log
`out/logs/p81_e16_dispersion.txt`. **The mean replicates as a null**
(+0.0123 [−0.0004, +0.0251] against the 600-position run's +0.0066), and the
spread is the finding:

```
observed per-position sd  0.0963
measurement noise         0.0581   (40 pairs/position)
⇒ TRUE between-position sd 0.0768
```

| derived quantity | value |
|---|---|
| typical \|gap\| at a disagreement (E\|X\|) | **0.0613** |
| 90th-percentile \|gap\| | **0.1263** |
| gain of an oracle choosing max(theirs, ours) over always-ours | **+0.0372 / decision** |
| resolution available at 200 pairs (≈15 decisions/game of the 600 s budget) | **0.0304** ⇒ **gap ≈ 2× the instrument** |

⚡ **So the null and the headroom are both real, and they are not in conflict.**
At a disagreement the two moves genuinely differ by ~0.06 win probability —
half the clone's own best-vs-worst gap — but the **sign varies**, and the 1050+
experts land on the good side no more often than we do (mean ≈ 0, and the
1100+ band reads 0.0056 ± 0.0201). ⇒ **there is exploitable value at these
decisions that NOBODY in the corpus is capturing.** That is the signature of a
problem **search can solve and imitation structurally cannot** — there is no
demonstrator to clone, because nobody is doing it right.

🔴 **This is ROADMAP §2.7's pre-declared sizing gate and it PASSES**: the gate
was *"90th percentile ≥ 0.10 ⇒ build; ≤ 0.03 ⇒ the clock cannot buy anything"*,
and it reads **0.1263**. ⚠ Declared before the number was seen, in the same
session.

⚠ **Four things that keep this honest.** (1) The oracle figure is an **upper
bound**: a real rollout oracle selects on *noisy* estimates, so at gap/SE ≈ 2 it
picks correctly ~75–80% of the time and captures roughly half of the 0.0372.
(2) **Per-decision gaps do not add across a game** — 15 disagreements/game does
not mean 15 × 0.037. (3) The dispersion is measured between *our pick and the
expert's pick*; a rollout oracle ranks the net's **own** options, and their gap
distribution is similar but not identical. (4) None of this touches the
**validation** blocker (§2.7): an agent spending real time per move cannot be
A/B'd at n≥2000 on this box, so the axis remains **large-or-nothing**.

### 🔴 ARM C — the H1/H2 discriminator ran, and H1 is not supported

`--arm-c out/policy_b7_ntum.npz #9e27e172`, log `out/logs/p81_e16_armc.txt`.
Both continuations measured on the **same 300 positions**, differenced pairwise.

| continuation | Δ(expert − ours) |
|---|---|
| shipped clone | +0.0107 [−0.0019, +0.0232] |
| **b7_ntum** (67.2% agreement with the expert, §8u) | +0.0056 [−0.0090, +0.0201] |
| **DiD** | **−0.0051 [−0.0228, +0.0126]** |

⛔ **The DiD covers zero.** The expert's move is worth no more when an
expert-imitating policy follows it up than when ours does. **H1 — "their moves
are only good as a coherent sequence, so partial copying breaks them" — gets no
support at the strongest contrast we can build.**

⚠ **And the honest limit on that, which caps the claim: the treatment is WEAK.**
`b7_ntum` agrees with ntumlnoob on **67.2%** of decisions against the clone's
**59.9%** (§8u) — the two continuations differ by only **~7 agreement points**.
A null DiD therefore bounds the effect *per agreement point*; it does **not**
rule out that a fully coherent expert continuation would show something.
⇒ **H1 is unsupported, not refuted** — the same honest label §8ao gave B8's β.

🔴 **But the thread still closes, on actionability rather than truth.** Even if
H1 holds, the only mechanism for exploiting it is building a coherent expert
imitator — **which is exactly B7, and B7 measured −55 and −92 Elo** (§8t/§8u).
There is no stronger expert-like policy in the repo and no route to one. ⇒ every
operationalisation of "imitation without a plan" is now closed: the latent plan
(§8bv, killed at its sizing gate), per-move quality (§8bx, null), and coherence
(arm C, null). **The thread's own logic points where §N.3 said it would — credit
assignment — and E16's dispersion is already that instrument.**

⚡ **Third independent estimate of the dispersion, and it holds:** true sd
**0.0866** here (0.0768 on the 220-position run), typical |gap| **0.069**.
The headroom finding does not depend on which sample measured it.

## 8bw. ✅ THE INSTRUMENT THIS PROJECT HAS NEVER HAD IS FEASIBLE — a real position forks out of a replay and its options can be scored by rollout (2026-08-09, day 27, 3rd session)

`scripts/p80_rollout_feasibility.py`, log `out/logs/p80_rollout_feasibility.txt`,
net pinned to the shipped `55326513` weights (`#4790c469`). **No experiment ran
— this is the feasibility gate HANDOFF §N.4.0 demanded before any design work.**

**Why it matters.** Every eval this project owns is either a **conformity**
metric (§8r: agreement measures distance from the fitted mode, not skill) or a
**weak evaluator** (`evalfn`, AUC 0.667 early). Neither can answer §8u's open
question — *"in THIS exact position, is their move better than ours?"* A rollout
instrument can, and §N.4.0 named one blocking risk: `fastsearch.begin` had only
ever been handed a `search_begin_input` captured **in the same process**.

### The four controls

| control | reading |
|---|---|
| **C1** forked position identical to the replay's — option list **bitwise**, both boards, turn, acting seat | **60/60 = 100%** ✅ |
| **C2** same determinized world, same pick, 8 repeats | trajectories differ (52–92 steps) 🔴 **not reproducible** |
| **C3** the clone's **top** option vs its **last**, paired on the world | **+0.120 [+0.052, +0.189]**, k=40 positions ✅ |
| **C4** hand the fork a decklist the seat is **not** playing | **accepted, 40/40, plausible number, no error** 🔴 |

⇒ **C1 kills the blocking risk: an sbi captured in another process reconstructs
the position exactly.** The instrument is buildable.

### Three findings that CHANGE §N.4.0's own design

1. 🔴 **Common random numbers are NOT available.** C2 shows the engine draws its
   own shuffles/coins beyond the determinized world, so §N.4.0's "paired
   rollouts with common random numbers" is unachievable as written. A **shared
   world** is the only pairing there is — and it still removes **ρ≈0.53** of the
   variance, so pairing survives; only the name was wrong.
2. 🔴 **"Per-decision resolution is unaffordable" is FALSE.** A rollout to
   terminal costs **101 ms**. ±0.020 on a pooled Δ costs ~96 min on one core
   (below), so pooling across a decision class is a **choice**, not a
   necessity — the opposite of what the design assumed.
3. ⛔ **C4 was the binding scope constraint, and it is now DEFUSED — but only
   after it nearly landed on the population I had just called safe.** `begin`
   takes the seat's hidden deck as an *argument* and cannot check it: given
   Crustle's 60 for a Grimmsnarl seat it returned **exactly what the correct
   deck returned** — rule 18's "plausible number, not a crash", in a new place.
   The first fix was to restrict the instrument to the mirror. 🔴 **That fix
   was wrong, because "both seats are Grimmsnarl" is not "both seats are our
   60": over 25 `mirror_experts` games, only 18 of 50 seats run our exact
   list** — the rest are 1–3 card variants. Determinizing those with our 60
   would have mis-filled the hidden zones of **64% of expert seats**, silently.
   ✅ **The real fix reads each seat's registered 60 out of the replay**
   (`seat_decklist()`, a bare 60-int action at step 1; recovered **50/50** on
   `mirror_experts` and validated **20/20** against `decks/grimmsnarl.py` on
   our own seat). ⇒ **the scope constraint is lifted entirely** — any seat in a
   replay we hold is usable, mirror or not.

### 🔴 And a defect in my own estimator, caught by replication

Three runs of the identical C3 cell read **+0.130 / +0.107 / +0.120** against a
nominal ±0.017. **They cannot all be true.** The cause is clustering: pairs are
nested inside 40 positions, and the per-position effect varies far more than
within-position sampling implies. Clustering on the position widens the interval
**4.1×** — and all three runs sit inside the clustered interval while none sits
inside the naive one. **The fix is in the tool** (`p80` now prints both and
labels the naive one *DO NOT QUOTE*), and the sizing formula was making the same
error: **472 positions × 60 pairs**, not 1,704 pairs.

⚠ **The general form, and it is the sixth instance of the family:** an analysis
choice that is *statistically* wrong produces a plausible number, not a crash.
What caught it was running the same cell twice — the cheapest form of rule 18's
redundancy, and one this project had not applied to an interval before.

### What the numbers say about the design

* **scale bar:** the clone's own top-vs-worst option is worth **+0.120** win
  probability. Any Δ from a future experiment must be read against that ruler.
* **cost:** 101 ms/rollout, 17–195 ms by turn (early turns are dearer; ⚠ two
  runs of the same cell read 78 and 101 ms — rule 7 CPU contention, so treat
  timings as ±30%).
* 🔴 **most positions cannot show anything: only 11 of 40 sit in win-probability
  [0.15, 0.85]** (mean 0.802 on our own ladder games). A near-ceiling position
  bounds the visible action-value gap. Any design must stratify on
  competitiveness — and must estimate that WP on an **independent** rollout
  batch, or selecting positions on the same rollouts that measure the effect
  buys a regression-to-the-mean bias.
* ✅ **the payoff use case is available:** expert seats reconstruct **32/32**
  over two dumps, and `replays/mirror_experts` holds **257 archetype-mirror
  games** — ⚠ *archetype* mirror, not identical-60 mirror, per the correction
  above; each seat is determinized with its own recovered list.

⇒ **Pre-registered as E16** (`docs/experiments/E16-counterfactual-move-value.md`)
— score the expert's actual move against our net's move by paired rollout, with
the agreement control, the +0.120 scale bar, and a difference-in-differences arm
that separates §N.3's H1 from H2. ⛔ **Not a training target**: demonstrator
selection is closed twice (§8t, §8u).

## 8bv. 🔴 R1 DIES AT ITS SIZING GATE — there is no latent "plan" in the corpus, and winners and losers play the SAME given the board (2026-08-09, day 27)

`scripts/p79_plan_audit.py`, log `out/logs/p79_plan_audit.txt`. **700 games,
1,376 seats, 56,611 MAIN decisions** from 08-05…08-07.

**The hypothesis.** §8u is the sharpest number in the repo on why we are not a
top player: we cloned the #2 player **successfully** — held-out agreement
59.9% → 67.2% — and measured **−92 Elo**. The reading that survives is that you
copy a strong player's moves without their *plan*. If that is right, the corpus
contains several coherent LINES and our memoryless net fits their mean.

**The gate.** If demonstrators run distinct lines, knowing which line a seat is
on must predict their next action **beyond what the board already says**:
`MI(action ; plan | state-bucket)` against the same statistic with plans
shuffled across seats (the shuffle preserves label marginals *and* the
within-seat correlation, so it absorbs the plug-in bias).

| plans (k-means on game shape) | MI | shuffled | **excess** |
|---|---|---|---|
| 2 | 0.0562 | 0.0551 | **+0.0010** |
| 3 | 0.0953 | 0.1039 | **−0.0087** |
| 4 | 0.1793 | 0.1425 | **+0.0368** |
| 6 | 0.2671 | 0.1773 | **+0.0898** |
| **ESTIMATOR CONTROL** — label = plays Petrel above median rate | 0.4232 | 0.0515 | **+0.3717** ✅ |
| **label = did this seat WIN** | 0.0531 | 0.0555 | **−0.0024** |

⛔ **R1's premise fails.** A label that genuinely defines behaviour reads
**+0.372 bits with only two groups**; the best plan clustering reads **+0.090
with six**, two of which are micro-clusters (18 and 26 seats), and the
well-populated k=2/k=3 splits read **zero**. The estimator had the power to see
an effect of the relevant size and did not.

🔴 **Two methodological traps, both sprung and both caught:**
1. **The first bucket was too fine** — `(turn, hand, prizes, opp_prizes,
   n_opts)` drove the shuffled baseline to 0.36 bits and made the positive
   control come out **negative**. Plug-in MI bias grows with
   (#buckets × #plans × #actions)/N; the bucket is the lever.
2. 🔴 **The first signature was CIRCULAR.** Clustering seats on their *card play
   rates* and then predicting *which card was played* reads **+0.27 to +0.46** —
   which is the estimator control's construction, not a finding. Re-clustering
   on game-shape features only (bench size, energy pace, evolution timing, prize
   pace, length), which share no variable with the label, collapses it to the
   table above.

⚡ **The `won` row is a finding in its own right:** at the resolution measured,
**conditioned on the board state, winners and losers choose the same actions**
(−0.0024 bits). It corroborates §8bs (no blunder signature) and §8bn from a third
direction.

⚠ **CORRECTED SAME SESSION — the first version of this entry said this "retires
outcome-conditioned / upside-down-RL cloning before it is built", and that is an
OVERCLAIM.** The bucket here is deliberately coarse — `(turn//3, prize
differential)` — and a net conditioning on the *full* board could exploit
fine-grained differences this estimator cannot see. What is established is a
**strong prior against** outcome conditioning, not a proof against it. The
distinction matters because the same coarseness that makes the plan nulls
trustworthy (it suppresses plug-in bias) is what limits this row's reach.

⚠ **What is NOT claimed.** This tests one operationalisation of "plan" (k-means
over six shape features). A richer plan representation could carry more. But the
instrument demonstrably resolves a real game-level label at 0.37 bits, so
whatever remains is smaller than that.

## 8bu. 🔴 E15 IS A NULL — averaging out the bench-slot nuisance reads 0.513 [0.492, 0.535] against a pre-registered 0.500 (2026-08-09, day 27)

Pre-registered in `docs/experiments/E15-symmetry-averaging.md` at commit
`c2ce197`, **before any arena game was played**. Log `out/logs/e15_sym8_ab.txt`.

`bc:sym8` (option probabilities averaged over 8 bench relabellings,
`agents/sa/symavg.py`) vs `bc:base`, same net (`#4790c469`), direct grimmsnarl
mirror, seat-swapped, **n = 2,000 games**:

```
score = 0.513 [0.492, 0.535]   W1027/D0/L973
[health] OK calls=424480 fallbacks=0 net_missing=0
```

⛔ **The interval covers 0.500. By the pre-registered criterion E15 is a null and
does not ship.** The point estimate is +0.013 (≈ +9 Elo) — positive, and far
under the 63–87-point LB noise floor (§8ak), so it is unmeasurable where it
would have to show up even if real.

✅ **Two controls held.** `sym1` (identity relabelling only) is **bitwise
identical** to plain `bc` — 0 of 1,915 selects differ — so the no-op arm is
proven rather than assumed. And `sym8` changes **8.36%** of real selects
(160/1,915), so this is not a null for want of firing: the intervention was
live on ~4 decisions per game.

⚡ **The prior recorded before the run was right, and that is the point.** §8bd
measured the near-tie band as indifferent (0.494 [0.467, 0.520]) and §8bt showed
the unstable decisions sit in exactly that band (median margin 0.310 vs 1.298).
Writing that down first is what makes this a null instead of a +0.013 headline.
⛔ **Do not re-cut at a different K.** The pre-registration forbids it explicitly,
and the mechanism — not the K — is what §8bd already priced.

⚠ **The defect itself stands** (§8bt): the net does read a nuisance variable. The
code stays; it is off by default (`sym_k=0`), and it is a correctness finding
that measured null, exactly the case
`correctness-fixes-are-wanted-regardless-of-elo` was written for.

## 8bt. ⚡ THE NET DECIDES PARTLY ON BENCH SLOT NUMBER — 16.9% of decisions flip under a relabelling that changes nothing about the game (2026-08-09, day 27)

`scripts/p78_symmetry_probe.py`, log `out/logs/p78_symmetry.txt`, run against the
**shipped** `55326513` npz extracted from its own tarball.

**The question no audit here has asked.** `p18`, §8y/§8z and §8ab all asked *"what
is in the observation that `featurize` does not read?"* — the dual question is
**"what does the net read that carries no game meaning?"** A bench **slot
number** is the clean case: moving a Pokémon from bench slot 1 to slot 3 changes
nothing, but it changes `opt["index"]`, which §8f encoded deliberately (+115 Elo)
because it was the only thing separating two options naming two copies of a card.

| arm | reading |
|---|---|
| **option-list ORDER** (positive control — deep-sets pool the option set, §8aa, so this MUST read ~0) | **0 / 23,952 = 0.000%** ✅ harness sound |
| **option-identity multiset preserved** (control, added after the bug below) | **0 violations** ✅ |
| **our own BENCH relabelling** | **1,670 / 21,732 = 7.69%** of relabellings change the chosen option |
| …per decision | **613 / 3,622 = 16.9%** of decisions flip under ≥1 of 6 relabellings |

Worst context is **MAIN at 24.8%** (443/1,788), then TO_HAND 14.8%.

🔴 **The first version of this probe read 18.2% / 27.5% and was wrong**, because
`_permute_bench` rewrote `opt["index"]` but not **`opt["inPlayIndex"]`**, which
ATTACH/EVOLVE (types 8, 9) use to name their in-play target and which
`optfeat._target_pokemon` and `slot_ix` both read. That does not relabel a
position, it **corrupts the option** — so part of the original instability was
measured damage, not a symmetry. Caught by an **option-identity multiset
control**, which is now arm 0 and reads 0. The effect survives the repair at
about 1.3 pp lower. ⚠ Fourth instance this session of "a guard/permutation I
wrote myself was the bug".

⚠ **The margin decomposition is what keeps this honest.** Median top1−top2 logit
margin is **1.298 over all decisions but 0.310 over the unstable ones** — the
flips live in the near-tie band. **§8bd measured that band and it is
INDIFFERENT**: flipping the clone's *k*-th choice for its (*k*+1)-th across it
reads **0.494 [0.467, 0.520]**. ⇒ **the prior is against this being worth Elo**,
and it must not be described as a found gain.

**What it is:** a measured correctness defect (the net consumes a nuisance
variable), and a **free** variance reduction if wanted — averaging logits over K
bench relabellings costs ~1 ms × K against a 600 s pool we spend 0.1 s of, and
at K=1 it is bitwise today's agent. ⛔ **Untested in the arena. Do not ship it on
this entry alone** — §8bd is a real prior against it, and this project's own rule
(§8aw) is that a defect gradient descent has already routed around is a bug in
the code, not a limit on the agent.

## 8bs. 🔴 THE WP-REGRET AUTOPSY RUNS ON THE 27 LOSSES — no blunder signature exists, and the method is provably blind to the half of the problem that matters (2026-08-09, day 27)

Day 27 §3 proposal 1, built as `scripts/p77_wp_regret.py`, archived at
`out/logs/p77_wp_regret.txt`. **Hypothesis (the user's named seam):** "a few
games where 1 bad decision cost us games" — a frequency-1 blunder that every
rate miner in this repo is structurally blind to, because a mistake made once
has no rate to difference against the experts.

**Method.** Score every decision state in the 76 shipped-agent ladder games with
`evalfn`, map score → win probability through a logistic fitted **per turn
bucket on an independent corpus** (250 fresh top-of-ladder games, 08-05…08-07),
and difference consecutive states. A pair with the same actor in the same turn
is caused by *that player's action* and is called **attributable**; a pair
spanning the handover carries the opponent's whole reply and is called
**boundary** and never attributed.

### C1–C3: the controls, and §8l reproduced outside self-play

| turn bucket | n states | AUC |
|---|---|---|
| 0–2 | 4,962 | 0.587 |
| 3–5 | 10,512 | 0.747 |
| 6–8 | 9,067 | 0.807 |
| 9–11 | 6,846 | 0.889 |
| 12+ | 4,819 | 0.920 |

⚡ **early 0.667 / late 0.905 against §8l's 0.685 / 0.901** — the first time that
number has been taken on *real ladder games by other players* rather than 200
self-play games, and it replicates. The calibration transfers to our corpus
(Brier **0.209** vs a base rate of 0.232, reliability monotone across all five
bins). The noise floor of one attributable ΔWP is **sd 0.039**, 1st percentile
**−0.141** — the scale any "blunder" has to clear.

### 🔴 The verdict: there is no blunder signature

| stream | n | mean worst ΔWP | median | min |
|---|---|---|---|---|
| **US, in the 27 losses** | 27 | **−0.069** | −0.064 | −0.205 |
| US, in the 49 wins | 49 | **−0.070** | −0.061 | −0.215 |
| THEM, in the 27 games they won | 27 | **−0.078** | −0.069 | −0.233 |
| THEM, in the 49 games they lost | 49 | −0.052 | −0.056 | −0.123 |

**Our worst decision in a game we lose is indistinguishable from our worst
decision in a game we win (−0.069 vs −0.070), and milder than what the players
who beat us did in those same games (−0.078).** A lost game must contain a big
drop — the trajectory ends at zero — so the drop alone was never evidence; these
three baselines are what make the statement falsifiable, and it does not survive
any of them.

⛔ **Sizing: 3 events at |ΔWP| ≥ 0.20 in 76 games = 0.039/game, of which one was
FORCED (a single option — not a decision at all). One event across all 27
losses.** Against the 0.5/game gate that killed Morgrem (0.2), Pokégear (0.27),
Archaludon (0.187) and Petrel (0.29), this is **13× under**, and it is the
smallest candidate rate this project has ever measured.

**No concentration either:** the single worst decision carries 45.8% of our
negative attributable ΔWP in losses — and **40.0% in wins**. And in absolute
terms our own within-turn decisions bleed **0.132 WP/game in the losses vs 0.166
in the wins**: the stream a blunder would live in is *larger in the games we
won*.

⚡ **Inspected, the ranked list contains no misplay.** The worst single decision
in all 27 losses (−0.205) is *placing damage on the opponent's Mega Lucario ex*;
the rest of the top ten is damage onto Mega Kangaskhan ex / Alakazam, and
`DISCARD_ENERGY` picks where **the alternative is another copy of the same
card**. The negative ΔWP after a DAMAGE select is mechanical — spending the
attack makes `evalfn` re-read both threat terms — not evidence about the target
chosen.

### 🔴 The finding that actually matters: this method cannot see an error of omission

The discriminator was run *because* a null needs one. §8bm found seven plays
where the same damage on a different legal target would have KO'd something and
what we hit survived — **known errors, owned by `p72`.** Where does the WP
ranking put them?

| game | result | ΔWP | rank in that game | the dominated event |
|---|---|---|---|---|
| 90744917 | win | −0.016 | **1 of 111** | KO available: Crustle@20hp |
| 90754696 | win | **+0.004** | 86 of 124 | KO available: Mega Kangaskhan ex@30hp |
| 90754696 | win | **+0.005** | 89 of 124 | KO available: Mega Kangaskhan ex@10hp |
| **90780054** | **LOSS** | **+0.002** | 31 of 58 | KO available: Teal Mask Ogerpon ex@10hp |
| **90796954** | **LOSS** | **+0.005** | 44 of 76 | KO available: Alakazam@20hp |
| 90863103 | win | +0.005 | 86 of 116 | KO available: Dwebble@10hp |
| 90863103 | win | +0.005 | 84 of 116 | KO available: Dwebble@10hp |

🔴 **Six of seven score POSITIVE and rank mid-pack; the seventh clears the noise
floor by less than half.** The reason is structural and it generalises past this
deck: **a realized trajectory only contains the branch we took.** Declining a
prize still deals damage, so the state improves and ΔWP is positive; the entire
cost sits in the branch that never happened. ⇒ **the null above is a null about
SELF-INFLICTED damage, and says nothing about foregone value.** Any future
reading of this entry that drops that clause is misquoting it.

⚠ **A second, narrower blind spot, from the same cause:** the attack is almost
always the last select of a turn, so **the cost of a bad attack lands in
`boundary`, never in the attributable stream.** The boundary ranking is
dominated by `TO_ACTIVE` — the *forced* promotion after being knocked out —
which is the opponent's turn arriving, not a choice of ours.

### 🔴 Three instrument defects found on the way, and none of them reached a number

The pattern of §8bc: everything below was caught by a control before it was
published, and each is now a guard in the script.

1. **The IRLS diverged on one turn bucket** — slope **74,173**, every state
   pinned to 0/1, and the reliability table is what caught it. Fixed with a
   standardised fit, a backtracking line search and a ridge; the script now
   prints a per-bucket `saturated` column that would have shown it immediately.
2. 🔴 **`evalfn` is UNDEFINED during setup and returns −8.2 on an empty board.**
   It scores prizes as `6 - len(prize)` *taken*, so between one player's prize
   pile being dealt and the other's it reads a six-prize deficit. Seven of the
   27 losses showed an identical spurious "−0.202 at turn 0" until a turn-0
   guard went in, and the same garbage was in the calibration corpus.
   ✅ **It has never touched a shipped number:** the only live consumers are
   `planner`/`sequencer`, which call `evaluate` at MAIN selects from turn 1 on
   (and both are closed axes anyway). ⚠ **It IS inside §8l's early-game 0.685**,
   which sampled turn 0 — §8l already called turns 0–1 noise, so the reading
   survives, but the clean number is C1's 0.667.
3. ⚠ **My own first fix was a selection bias, and it was worse than the bug.**
   Guarding on "both actives non-empty" looked like a safe way to exclude setup;
   it deleted **158 of 177 damage deltas**, because the state right after a
   DAMAGE select has the defender's active empty exactly when we **knocked it
   out**. It silently removed the successful damage and left `DAMAGE` reading
   0.24/game instead of 2.07. Caught by diffing a context's event count against
   the previous run. **A guard that changes a denominator by 88% is a finding,
   not a fix.**

### What this closes, and what it does not

- 🔴 **The "1 bad decision cost us games" hypothesis is CLOSED for self-inflicted
  within-turn damage**, on three independent baselines plus a sizing 13× under
  the gate. It joins passive damage (§8bm), KO-setup (§8bp) and Petrel (§8br) —
  **four seams named by the user, four kills, no arena time spent on any of them.**
- ⚡ **It also re-confirms §8bm/§8bn's shape from a third angle:** the WP we lose
  is lost while the *opponent* is acting, and our own decision stream is quieter
  in our losses than in our wins. The gap is not a blunder rate.
- ⛔ **NOT closed: errors of omission**, which the discriminator proves this
  instrument cannot rank. The measurement that would close them is the
  **option-level counterfactual** — score every legal option at the same state,
  not the one realized trajectory. For damage placement that is pure arithmetic
  (the damage is measured off the board, `p72`'s mapping is verified 839/840) and
  needs no engine. **That is the next build if this axis is reopened, and it is
  the honest version of "WP regret" the day-27 proposal was reaching for.**
- ⚠ **Do not re-run the trajectory version at a different threshold hoping for
  candidates.** The 0.20 line is not what killed it — the three baselines in the
  control table are, and they are threshold-free.

## 8br. 🔴 PETREL'S FETCH IS NOW INSTRUMENTED — and the whole policy gap sizes at 0.29 fetches/game, under the gate (2026-08-09, day 26)

The last seam the user named, and the only one nothing in this repo had ever
looked at: `p70_perturn_sweep` measures whether a card is **played** per
available turn; **what `Team Rocket's Petrel` (1219, ×4) FETCHES was never
measured at all.** `scripts/p76_petrel_fetch.py` closes that.

### 🔴 Two mapping bugs, and the control that caught the second

The first table this script produced said Buddy-Buddy Poffin was offered **9
times in 76 games**. E11 measured a Poffin *gap* of 0.80 plays/game, so 9 was
impossible — the table was nonsense. Cause: I filtered options on
`opt["area"] == HAND`, but **a PLAY option (type 7) carries no `area` at all**;
it is a bare index into the hand. Every card play in the corpus was invisible.
⇒ The script now takes the card id from **`optfeat.option_features`, the same
extractor that built the training data**, so it cannot disagree with the net
about what an option is.

⚠ The first positive control was itself too weak to catch this: requiring the
*next* record to be our seat fires on ~2% of the corpus (**n=10**, 100%). Scanning
to our next record instead gives **1331/1375 = 96.8%** — and the residual is
expected, because we may draw another copy of the same id between two records,
masking the decrement. **A control with n=10 licenses nothing.** `--verify`
re-runs it.

### What Petrel actually fetches

Petrel is the second-most-held card in our hand (2,748 records) and resolves
**121 times in 76 games — 1.59 plays/game.** Against 501 games from three
current Grimmsnarl pilots (`Raihan Ramadistra`, `flg`, `Sixth Sense`; §8bq),
692 fetches. Take rate is conditioned on the card actually being in the deck at
the time — the rule-21-correct unit:

| fetched card | our avail | our rate | their avail | their rate | gap |
|---|---|---|---|---|---|
| **Unfair Stamp** | 54 | **63.0%** | 379 | 45.1% | **+17.8%** |
| **Night Stretcher** | 106 | **32.1%** | 633 | 19.6% | **+12.5%** |
| **Spikemuth Gym** | 118 | **5.1%** | 669 | 13.2% | **−8.1%** |
| Rare Candy | 120 | 5.0% | 646 | 10.4% | −5.4% |
| Tool Scrapper | 87 | 6.9% | 383 | 3.1% | +3.8% |
| Boss's Orders | 107 | 7.5% | 605 | 10.4% | −2.9% |
| Buddy-Buddy Poffin | 116 | 0.9% | 671 | 3.1% | −2.3% |
| Lillie's Determination | 111 | 12.6% | 645 | 13.5% | −0.9% |

**There is a real, consistent difference: we over-fetch Unfair Stamp and Night
Stretcher and under-fetch Spikemuth Gym and Rare Candy.** Neither side ever
fetches Pokégear or Dawn, and neither fetches a second Petrel.

### ⛔ Rule 14 kills it anyway — and the tempting sizing is the wrong one

⚠ **Do not add up the take-rate gaps.** They are conditional rates over
overlapping denominators and sum to a share of nothing. The share of fetches
that would change if we adopted their policy **exactly** is the total variation
distance between the two fetched distributions:

```
total variation = 18.3%   ×   1.59 Petrel plays/game   =   0.29 fetches/game
```

⛔ **0.29/game against the 0.5 gate.** And that is the **ceiling** — it is the
entire distribution realigned at once. Any single-card rule is a fraction of it:
the largest, Unfair Stamp at +17.8% of 54 available fetches, is **0.13/game**.

### What this closes

⛔ **The Petrel fetch seam is closed by sizing.** Both seams the user named on
day 26 — passive-damage targeting and Petrel — are now measured, and **both die
at the same gate**: passive damage at 0.09/0.20 (§8bm) and 0.04 (§8bp), Petrel
at 0.29 as an upper bound.

⚠ **What is NOT claimed:** that our fetch policy is *correct*. It differs from
three stronger pilots in a stable direction, and 18.3% of fetches differ. The
claim is only that **no rule written on this seam can move enough decisions to
be measurable against a 63-point noise floor** — which is rule 14 doing exactly
the job it was written for, and the third time this session it has saved an
arena run.

### Addendum — why Petrel barely ever fetches Tool Scrapper (user question, same day)

```powershell
python -X utf8 scripts/p76_petrel_fetch.py --scrapper `
    --dir replays/submission_v5_s2 `
    --vs replays/2026-08-03 replays/2026-08-04 replays/2026-08-05 `
         replays/2026-08-06 replays/2026-08-07 `
    --vs-us "Raihan Ramadistra" --vs-us flg --vs-us "Sixth Sense"
```

The table above reads "Tool Scrapper 6.9% vs 3.1%" as one of the few cards we
take **more** often than the experts. Decomposing the denominator says the
opposite, and the mechanism is three multiplied constraints, not a judgement:

| | us | experts |
|---|---|---|
| Petrel fetches | 121 | 1,067 |
| Scrapper still in the deck (it is a **1-of**, `decks/grimmsnarl.py:30`) | 87 (71.9%) | 609 (57.1%) |
| …**and an opposing tool was on the board** | **14 (16.1%)** | **196 (32.2%)** |
| taken *with* a target | **1 / 14 = 7.1%** | 7 / 196 = 3.6% |
| taken with **no tool anywhere** | **5 / 73** | 9 / 409 |

🔴 **Five of our six Scrapper fetches happened with no tool on either board** —
the card could not do anything the moment we got it. So the +3.8% is not tech
awareness; conditioned on a target existing we take it **1 time in 14**. The
honest summary is that neither side prioritises it: with a legal target on the
board both are under 8%, because Unfair Stamp and Night Stretcher win the slot.

⚠ **And the two corpora do not face the same field.** Tools seen on board: ours
**Hero's Cape 10 / Air Balloon 3**; theirs **Air Balloon 178 / Hero's Cape 87**.
Our opponents are a spread of real ladder names (not mirror-heavy — checked),
so this is a *height* difference, not a matchup artefact: the experts play
against §8bq's top-of-ladder field where Mega decks are 26% combined, and Mega
decks carry Air Balloon. **Unverified but directly testable** by censusing tool
cards against `avg_score` band.

⛔ **Nothing to build.** "Never fetch Scrapper into a bare board" fires **5 in 76
games = 0.066/game**; "fetch it when a tool is out and we didn't" is **13 in 76 =
0.17/game**. Both far under the 0.5 gate, and the second is a tradeoff (rule 11).

### Addendum — HOW the fetch is decided: the option vector carries no board at all

Worth writing down because it is structural and would otherwise be re-derived.
**No rule touches a Petrel fetch.** `bcagent.act` ranks with `chip_target` /
`drag_target` / `boss_converts` *before* the net and promotes with `boss_veto` /
`poffin_force` / `counter_source` / `energy_spread` *after* it — every one of
them is bound to a damage-counter, Boss, or attach select. At `ctx=7` the choice
is `net.choose(obs)` verbatim: sort by logit, take `k` from `min/maxCount`
(here 0/1, so declining is a live option and the top logit wins).

🔴 **And a fetch option's own feature vector contains nothing about the board.**
Dumping `option_features` over a real 24-option fetch, the only non-zero entries
are: the type-3 one-hot, "this option is mine", `dense[v+11]` = the card's
**index in the deck list** (pure disambiguation so two copies aren't identical
vectors), and the v6 `cardType` one-hot. **The whole v3 target block (`hp`,
damage taken, `dies to 30`, energy count, `best_damage`, `we can KO it now`) is
identically zero** — it resolves a Pokémon at `(player, area, index)`, and the
deck is not an in-play area. The attack embedding is zero and the target
embedding is `card_emb[0]`.

⇒ **One option differs from another only by its card embedding and card type.**
Everything situational must arrive through the shared state vector `srepr`,
which is concatenated to *every* option identically and can only discriminate
through the head MLP's interaction term. The tool counts Scrapper needs *are*
in there (`features.py:96-97`, ours and theirs, capped at 4) — so the net is not
blind, it simply does not use them: **1 take in 14 with a target on board.**

⚠ **It is not a static priority list either.** Ranking cards by marginal take
rate and asking how often we took the top-ranked *available* card gives
**65/121 = 53.7%** (experts 469/1067 = 44.0%, though their number is depressed
by rare-but-high-rate cards like Hero's Cape sitting atop the derived order —
treat the two as "both roughly half", not as a ranking). The deviations are
lawful-looking, not random: Unfair Stamp beats Poké Pad and Boss's Orders
**100%/0%** when both are in the deck, and beats Night Stretcher 89%/11%, while
the contested pairs sit at 69–76%. So the fetch is *modulated* by the spot —
just not by the one fact Scrapper depends on.

⚠ **This does weaken the standing case for cutting Tool Scrapper** from the 60
(`decks/grimmsnarl_budew.py`, `grimmsnarl_boss.py`), which rests on 0.13
plays/game and **0.00 per mirror game**. §8aj's mirror number was always rule-16
compromised — our own list runs no tools, so the mirror answers by construction.
The first non-mirror number now exists: against the field the strongest pilots
actually face, **a scrappable tool is on the board 32.2% of the time.** The card
is not dead there. It is still our thinnest slot; it is no longer *provably* dead.

## 8bq. THE TOP OF THE LADDER, 1,972 FRESH GAMES — our archetype is the most-played deck there and wins 47.9%, and "Mega Lucario is the best deck" is one player (2026-08-09, day 26)

`scripts/p75_day_census.py` over the five newly-mined days (08-03…08-07).
`p9_field_census` censuses *from a seat*, which a mined day does not have — this
walks the `visualize` stream once and attributes boards to **both** players, then
joins `manifest.csv`'s `avg_score`.

```
python -X utf8 scripts/p75_day_census.py \
    --dir replays/2026-08-03 replays/2026-08-04 replays/2026-08-05 \
          replays/2026-08-06 replays/2026-08-07
```

**1,972 games, 3,944 seat-appearances, 36 archetypes, 1 episode dropped** (null
reward — an errored game has no winner and is dropped whole, never scored as a
loss).

### ⚠ The censoring floor has RISEN since §8i

`avg_score`: **min 1100, median 1166, max 1296**, all 1,972 joined. §8i put the
floor at ~1055. **This dump is ~140 points above our own 1027.2.** Everything
below describes the top of the ladder. ⛔ Still never an anchor.

### The field up there

| archetype | seats | share | WR | mean rating |
|---|---|---|---|---|
| **Marnie's Grimmsnarl ex (ours)** | 933 | **23.7%** | **47.9%** | 1161 |
| Mega Lucario ex | 730 | 18.5% | 62.6% | 1205 |
| Dudunsparce | 522 | 13.2% | 49.0% | 1165 |
| Alakazam | 456 | 11.6% | 45.2% | 1160 |
| Mega Lopunny ex | 294 | 7.5% | 33.0% | 1169 |
| Dragapult ex | 263 | 6.7% | 51.0% | 1163 |

⚠ Win rates here are population statistics over both seats, so they average to
50% by construction; read them as *relative to the field*, not as skill.

### 🔴 The control that rewrote the second headline

At this level a deck can be **one pilot wearing a deck's name** — the §8bn
mistake. So: top pilot's share of each archetype, their WR, and everyone else's.

| archetype | top pilot | their share | their WR | **everyone else** |
|---|---|---|---|---|
| Mega Lucario ex | Majkel1337 | **84.2%** | 66.2% | **43.5%** |
| Marnie's Grimmsnarl ex | Raihan Ramadistra | 50.6% | 49.6% | **46.2%** |
| Hariyama | Majkel1337 | 88.3% | 79.2% | 71.4% |
| Dudunsparce | LiamK | 28.9% | 52.3% | 47.7% |

⚡ **"Mega Lucario ex is the best deck at 62.6%" is FALSE.** Majkel1337 is 84.2%
of its games at 66.2%; **every other Lucario pilot wins 43.5%.** The deck is
below average and one player is carrying it. Had this gone unchecked it would
have argued for a deck switch on the strength of one opponent's skill.

🔴 **Our archetype's 47.9% survives the same control: 46.2% excluding its top
pilot.** That is a *deck* property, not a pilot artefact. **The most-played deck
at the top of the ladder wins slightly under half**, and Grimmsnarl's 23.7%
share means a large mirror fraction pulling it toward 50 — so its non-mirror
record is worse than 47.9%. ⚠ This does not say switch decks (the deck axis is
closed, and §8ba priced a deck term at +0.140 in a *specific* matchup, not in
general). It says the ceiling of the current archetype in this band is real and
should not be assumed away when reading a 1027.

### For the Petrel work: the demonstrator shortlist

933 Grimmsnarl seat-appearances. By pilot, with volume enough to mine:

| pilot | games | WR | mean episode rating |
|---|---|---|---|
| Raihan Ramadistra | 472 | 49.6% | 1177 |
| Sixth Sense | 127 | 50.4% | 1147 |
| **flg** | **90** | **55.6%** | 1144 |
| @kdcyberdude | 53 | 43.4% | 1145 |

⚡ **`李秉叡（ntumlnoob）` has switched off Grimmsnarl** — 215 games in this window,
now maining **Dudunsparce**, and at 35.8%. The expert corpus in §8bo is that
player on the old deck, which does not invalidate it (it was a like-for-like
Grimmsnarl comparison) but does mean **they are no longer the demonstrator to
mine for current Grimmsnarl play.** ⇒ `flg` is the best win-rate-per-game
Grimmsnarl source; `Raihan Ramadistra` is the volume.

## 8bp. 🔴 E13 DIES AT THE SIZING GATE — the KO-setup band is empty for the EXPERTS TOO, so §8bo's gap is not KO manufacture (2026-08-09, day 26)

Pre-registered in `docs/experiments/E13-ko-setup.md` at **50a6344, before
`p74_ko_setup_sizing.py` existed**. Prediction 1 named the sizing gate as the
most likely killer, and it was.

### The frozen condition, sized

`p74` implements E13's five clauses exactly: a damage-placement select, all
options opponent-side, their Active on offer and not already taken by the net,
their Active already damaged, and `best_damage` in the band `hp-30 ≤ A < hp` —
the placement is what converts a survivable Active into a lethal one.

```
python -X utf8 scripts/p74_ko_setup_sizing.py --dir replays/submission_v5_s2 --arch grimmsnarl
python -X utf8 scripts/p74_ko_setup_sizing.py --dir replays/ntumlnoob_31-07-2026 \
    --us "李秉叡（ntumlnoob）" --arch grimmsnarl
```

| funnel stage | us (24 mirror games) | ntumlnoob (149) |
|---|---|---|
| damage-placement selects | 303 (12.6/game) | 1497 (10.1/game) |
| Active on offer | 218 | 891 |
| …and not already taken | 209 | 772 |
| …and Active already damaged | **97** | **406** |
| **…and in the KO-setup band** | **1 (0.04/game)** | **7 (0.05/game)** |

⛔ **0.04 firings/game against a 0.5 gate — 12× under.** The rule is not written.
Same gate that closed Morgrem (0.2), Pokégear (0.27), Archaludon (0.187) and
both halves of E12 (0.09, 0.20).

### ⚡ The finding is not "the rule is too small" — it is that the MECHANISM WAS WRONG

E13's stated mechanism was that the 1150+ pilots concentrate chip damage to
*manufacture* KOs. **They do not.** Their KO-setup band is 7 of 406 damaged-Active
decisions — **1.7%**, against our 1.0%. The band is nearly empty for both sides,
so it cannot be carrying a 22.4%-vs-7.0% behavioural difference. Whatever drives
§8bo's gap, it is **not** converting the Active into range of our own attack, and
`best_damage`-shaped arithmetic cannot reach it.

⚠ This retires the reading of §8bo's "KO-available 1.2% vs 6.3%" as evidence of
deliberate KO manufacture. That coherence story was mine, and it is now measured
false. The 6.3% has some other cause — most likely their attacks landing more
often, which is downstream of play quality, not of target choice.

### A confound §8bo did NOT control for, now checked and clean

The funnel exposes a variable the direct standardisation never held fixed:
whether our attack **already** kills their Active, in which case a counter spent
there is wasted and declining it is correct rather than timid.

| | already lethal | share of damaged-Active decisions |
|---|---|---|
| us | 44 / 97 | **45.4%** |
| ntumlnoob | 186 / 406 | **45.8%** |

Within 0.4 points. **§8bo's conclusion survives** — the ~86%-behavioural gap is
not an artefact of us facing already-dead Actives more often. The gap is real and
still unexplained; only this explanation for it is dead.

### What is closed, and what is not

⛔ **Closed: the KO-setup rule form.** Do not revive it at a different chip value
— E13 pre-registered that as a separate experiment, and the reason it died (an
empty band on *both* sides) does not move with the threshold.

⚠ **Not closed: the §8bo gap itself.** The plain "prefer a damaged Active"
variant sizes at 2.6–4.0 firings/game, well over the gate. It was pre-registered
as a separate experiment precisely so it could not be reached for as a fallback
the moment clause 5 failed, and **that is exactly the situation now — so it is
not being reached for.** It is a 100%-forcing rule against a 22.4% behaviour,
which is E11's error verbatim (0.487). If it is ever run it needs its own
pre-registration and its own argument for why E11 does not apply.

## 8bo. ⚡ THE NET ALREADY BRANCHES ITS TARGETING ON THE MATCHUP — 4.1% → 90.9% "hit the Active", learned, with every rule OFF (2026-08-08, day 26)

**User hypothesis** (day 26): *"depending on who we're facing, the policy could
require changes — target the Active against Crustle, target the weakest against
something else."* ⇒ measured before building anything (rule 14).

### Why §8bm could not answer this

§8bm sized the **dominated** half of targeting. A KO is on the table in **1.2%**
of these decisions; the other **98.8% are tradeoffs**, where damage must go
somewhere and arithmetic cannot say where. **A matchup-conditional policy lives
entirely in that 98.8%, and §8bm measured none of it.** ⚠ So §8bm's "the net is
99.1% correct" means *on the arithmetic subset* and must never be quoted as
"the net targets correctly" — `p73` exists to stop exactly that over-read.

### 🔴 The result: the branch the user proposed is already there, and it is huge

`p73_target_policy.py`, tradeoff regime only, our 76 ladder games:

| opponent | decisions | picks lowest HP | **chose Active** | mean HP hit |
|---|---|---|---|---|
| Marnie's Grimmsnarl (mirror) | 256 | 72.3% | **4.1%** | 77 |
| Alakazam | 77 | 71.4% | **14.8%** | 61 |
| Mega Lucario ex | 61 | 41.0% | **39.0%** | 121 |
| Teal Mask Ogerpon ex | 23 | 56.5% | **61.9%** | 147 |
| Archaludon ex | 40 | 55.0% | **62.1%** | 119 |
| **Crustle** | 75 | 44.0% | **90.9%** | 126 |

⚡ **A 22× swing in "hit the Active" across archetypes — 4.1% in the mirror to
90.9% against Crustle — from a net with `chip_targeting`, `energy_spread` and
`counter_source` all FALSE.** The policy the user described is not missing; it
was learned from the corpus and it is already conditional, in the direction
proposed, including the "target the weakest" behaviour in the mirror and against
Alakazam (72.3% / 71.4% lowest-HP).

⇒ **This retro-explains `chip_target`.** That rule imposed ONE static ranking
("dies to 30, then most prizes, then lowest HP") across every matchup, and
§8c measured it at **−0.126 vs Crustle** while `chip_wall_defer` recovered
**+0.104** by *handing the select back to the net*. The recovery was never a
better rule — **it was switching the rule off in the matchup where the net's own
branch was better**, which is what the shipped config now does everywhere.

### 📋 Housekeeping: `chip_wall_defer` is dead code in the shipped agent

`bcagent.py:244` consults it only inside `if self.chip_targeting:`, and the
shipped bundle sets `chip_targeting=False`. **B3 instance 1 cannot fire in
production.** Harmless — the net handles the wall matchup unaided and §8an
measured all three nets *higher* against the repaired Crustle pilot — but it
should not be cited as a live rule.

### ⚠ The one difference from the 1150s, and its base rate

Same script on ntumlnoob's 330 games from **their** seat (#2, 1162.8):

| matchup | metric | us (~1027) | ntumlnoob (1162.8) |
|---|---|---|---|
| mirror | **chose Active** | **4.1%** (256) | **14.0%** (1,263) |
| mirror | picks lowest HP | 72.3% | 67.1% |
| mirror | mean HP hit | 77 | 95 |
| Crustle | chose Active | 90.9% (75) | 63.9% (263) |
| Alakazam | chose Active | 14.8% | 14.1% |
| all | KO was available | **1.2%** | **6.3%** |

🔴 **We hit the Active in the mirror 3.4× less often than the #2 player**, on
256 vs 1,263 decisions — real, sized and ordering-free.
⛔ **And it is a TRADEOFF, which is 0 for 5 in this project.** It is the same
shape as §8bj (Munkidori timing) and §8bl (Poffin) — both real, both sized, both
confound-checked, both worth **zero Elo**. ⚠ **The 1.2% vs 6.3% KO-availability
gap is the more interesting number and the least trustworthy**: it is not a
behavioural measure at all, being a joint function of damage output, opponent
strength and game length, and it is **not confound-checked**. Nothing here
licenses a rule; it licenses a pre-registered A/B at most.

### ✅ THE CONFOUND CHECK — the gap is BEHAVIOURAL, not board mix (run before any rule)

E11's one procedural win was checking the confound *before* pre-registering, so
the same was done here. Mirror decisions only, tradeoff regime, Active on offer:
**194 of ours, 794 of theirs.** Bucketed by the two situational features that
make the Active attractive independent of any policy:

| Active is lowest HP | Active is top prize | our n | our rate | their n | their rate |
|---|---|---|---|---|---|
| no | no | 98 | **1.0%** | 449 | **12.2%** |
| no | yes | 85 | **1.2%** | 279 | **16.8%** |
| yes | no | 11 | 54.5% | 65 | 13.8% |

**Direct standardisation — our per-bucket rates re-weighted to THEIR board mix:
4.1% → 5.5%, against their raw 14.1%.** The mix explains ~1.4 pp of a ~10 pp
gap; **~86% of it is behaviour.** ⇒ unlike a mix artefact, there is something
here to repair. Option-count distributions are also near-identical (6-option
selects 40% vs 45%), so the second situational axis does not rescue the mix
story either.

⚡ **And the policies differ in KIND, not just rate.** We essentially never hit
the Active unless it is the weakest thing on the board (1.0–1.2% when it is not
the lowest HP); they hit it 12–17% regardless. Where the Active *is* the lowest
HP the ordering **inverts** (us 54.5%, them 13.8%) — ⚠ on n=11, six events, so
that cell is fragile and is reported rather than relied on.

### 🔴 The condition that separates their Active hits — and it is rule-shaped

| cut | us (n) | ntumlnoob (n) |
|---|---|---|
| Active **already damaged** | **7.0%** (86) | **22.4%** (441) |
| Active undamaged | 1.9% (108) | 3.7% (353) |
| Active at **50–75% HP** | **7%** (14) | **49%** (112) |
| Active at 75–100% HP | 1% (145) | 7% (549) |

⇒ **Both sides condition on the same variable and we do it ~3–6× more weakly.**
The 1150s pile damage onto a *wounded* Active — concentrating toward a KO —
where we top up the lowest-HP bench target instead. That is coherent with the
KO-availability gap (**1.2% of our decisions vs 6.3% of theirs**): concentration
*manufactures* the KO opportunities that §8bm then measures us taking 99.1% of.

⚠ **Still a TRADEOFF (concentrate vs spread), and tradeoff rules are 0 for 5.**
⚠ **Our 50–75% cell is n=14 (one event).** The `already damaged` cut (86 vs 441)
is the sized one and is what any pre-registration must be built on.
⛔ **No rule written.** E11 forced a soft 70% preference to 100% and lost; any
intervention here must reproduce a *conditional rate*, and its threshold must be
frozen before the cell runs, not tuned after.

```powershell
python -X utf8 scripts/p73_target_policy.py --dir replays/submission_v5_s2
python -X utf8 scripts/p73_target_policy.py --dir replays/ntumlnoob_31-07-2026 --us "李秉叡（ntumlnoob）"
python -X utf8 scripts/p73_target_policy.py --dir replays/submission_v5_s2 \
    --confound replays/ntumlnoob_31-07-2026 "李秉叡（ntumlnoob）" --arch grimmsnarl
```

## 8bn. ⚡ THE FIELD AT ~1027 IS NOT THE FIELD §8ac PROJECTED — and our per-matchup win rates are not visibly worse than the #2 player's (2026-08-08, day 26)

`p9_field_census.py` on the user-supplied `replays/submission_v5_s2` — **76
games of `55326513` (`policy_v5_s2`)**, the highest-rated agent this project has
run, censused from our own seat.

### The field, and where the 27 losses actually are

| archetype | share | our WR | losses |
|---|---|---|---|
| **Marnie's Grimmsnarl ex** (mirror) | 31.6% | 58.3% | **10** |
| **Alakazam** | 23.7% | 77.8% | 4 |
| Crustle | 9.2% | 85.7% | 1 |
| Mega Lucario ex | 9.2% | 57.1% | 3 |
| **Teal Mask Ogerpon ex** | 6.6% | **40.0%** | 3 |
| Archaludon ex | 5.3% | 100.0% | 0 |
| Meowth ex · Dudunsparce · Slowking | 7.8% | 33% · 0% · 0% | 5 |

**Overall 49/76 = 64.5%.** Top 5 archetypes = 80.3% of the field; 68 distinct
teams over 76 games, so this is a field sample, not a few repeat opponents.

🔴 **The mirror is 31.6%, not the 71.4% §8ac projected for play above rating
1000.** §8ac measured opponent-rating *bands* inside dumps taken at ~955; this
is the whole field at ~1027 and the mirror share is less than half its
projection. ⚠ **Every "the climb runs through the mirror" framing — E10's frame,
§2.6's premise — rests on the projection, not on this.** The mirror is still the
largest single bucket and still owns the most loss mass (10 of 27), so the
*ordering* survives; the **weight does not**, and any weighted anchor verdict
computed from §8ac's shares is quoting a share this dump contradicts.
⚡ **Alakazam is back at 23.7%** after being dropped from planning entirely
since day 9 — and it is our second-best matchup (77.8%), so it is loss-mass
cheap but it re-prices the anchor weights.

### 🔴 The control that reframes the gap: the same census from ntumlnoob's seat

`p9 --us "李秉叡（ntumlnoob）"` over their 330 games (**#2 on the LB, 1162.8**):

| matchup | us, ~1027 (n) | ntumlnoob, 1162.8 (n) |
|---|---|---|
| mirror | 58.3% (24) | 57.0% (149) |
| Alakazam | 77.8% (18) | 70.8% (48) |
| Crustle | 85.7% (7) | 57.6% (33) |
| **Teal Mask Ogerpon** | **40.0%** (5) | **37.5%** (8) |
| overall | **64.5%** | **64.5%** |

⚡ **Teal Mask Ogerpon beats the #2 player at the same rate it beats us** ⇒ it
is a **matchup property of the archetype, not a piloting failure**, and it comes
off the "we are misplaying it" list before anything was spent on it.
⚠ **The overall 64.5%/64.5% coincidence must NOT be read as equal strength** —
their opponents average ~130 rating points stronger (§8ac: the pool is a
function of your own rating), so equal win rates against a harder field means
they are genuinely better. What it does say is that **the 135-point gap is not
concentrated in a matchup we could close by targeting better**, which is the
same conclusion §8bj and §8bl reached from the behavioural side.
⚠ Our per-matchup n's are 5–24 games (±20 pp on the mirror). These are
directional, not resolved.

## 8bm. 🔴 E12 — THE PASSIVE-DAMAGE SEAM IS SIZED ON OUR OWN LADDER GAMES AND BOTH HALVES DIE AT THE GATE (2026-08-08, day 26)

**User-directed** ("optimize how we target with our passive damages … a few
games where 1 bad decision cost us games"), and answered with an autopsy of the
**76 real ladder games of the shipped agent** the user supplied in
`replays/submission_v5_s2` — `55326513`, `policy_v5_s2`, at ~1027.

### Why this asked a different question from F1 and E11

`p66` ranked disagreement per decision (wrong unit, rule 21); `p70` ranked it
per turn (right unit, found Poffin). **Both rank by RATE, both found tradeoffs,
and tradeoff rules are 0 for 5.** Rule 11's winning class (3/3) is the
**dominated** option, and for passive damage dominance is *arithmetic* — it
needs no expert corpus at all. So this asks: **was an option available that
dominates the one we took?**

### ✅ The positive control, run before any number was read

The option→Pokémon mapping is the whole experiment, and the obvious route is
wrong: `steps[i][seat]["action"]` matched the Pokémon that actually lost HP
**11 of 48 times**. The miner's route — `steps[0][0].visualize`, `v["selected"]`
— matches **839 of 840**. `p72 --verify` re-runs this, and it is the first thing
to check if any number here looks wrong. ⚠ **A fifth confidently-wrong script
was avoided by running the control first, not by noticing afterwards.**

### The denominator (rule 13) — these are real, frequent decisions

| context | selects | ≥2 options | real share |
|---|---|---|---|
| 13 `DAMAGE_COUNTER` | 532 | 527 | 99.1% |
| 15 `DAMAGE` | 278 | 245 | 88.1% |
| 16 `REMOVE_DAMAGE_COUNTER` (Adrena-Brain source) | 532 | 387 | 72.7% |

**10.2 damage-placement choices and 5.09 source choices per game** — both an
order of magnitude above the 0.5/game sizing gate. ⇒ **the seam is not being
dismissed for want of opportunities.** ⛔ And all three are made by the **net
alone**: the shipped bundle reads
`AGENT_KWARGS = {'chip_targeting': False, 'energy_spread': False, 'counter_source': False}`.

### 🔴 Half 1 — "a KO was available and we spread instead": 0.09/game

Same damage, measured off the board rather than assumed, applied to every other
legal option in the same select:

| | |
|---|---|
| events | **7** in 76 games = **0.09/game** |
| share of real damage choices | **0.9%** |
| in games we LOST | 2 of 7 |
| distinct lost games touched | **2 of 27** |

**Below the gate that killed Morgrem (0.2), Pokégear (0.27) and the Archaludon
rule (0.187)** — and the loss-attribution the hypothesis rests on is 2 of 27
losses, i.e. it cannot be what is costing us games. ⚡ The complement is the
real finding: **the net takes the available KO 99.1% of the time with no rule
helping it**, which is §8's p2_lethal result (316/316 lethals taken) one level
down. 4 of the 7 are "hit Crustle, passed a Dwebble KO" — the wall matchup
`chip_wall_defer` already owns, and 5 of 7 are in games we won.

### 🔴 Half 2 — the Adrena-Brain SOURCE pick: 0.20/game, 1.5% of the ability

The source **caps** how much the activation moves ("up to 3 counters"), so a
source carrying 1 counter when another carries 3 moves 10 damage instead of 30 —
strictly less healing *and* strictly less damage, same action, same cost.
**Dominated by construction, and it needs no counterfactual.**

| | |
|---|---|
| under-moved events | **15** in 76 games = **0.20/game** (3.9% of real picks) |
| damage actually moved | 10,750 |
| damage available | 10,910 |
| **left on the table** | **160 = 1.5%, or 2 damage per game** |

Gap sizes are 10 damage in 14 of 15 cases. ⇒ **the net picks the source
correctly 96.1% of the time unaided**, which independently explains why
`counter_source` is switched off and harmless rather than merely unproven.

### ⚠ What was NOT measured, and why it is recorded rather than dropped

The **"would it have survived"** counterfactual on the source pick — did we heal
the wrong Pokémon and lose one that the counters would have saved — **cannot be
computed from these replays.** Log type 16 carries an uncensored `value`, but it
reconciles with the observed board delta only **434 of 699 times (38%
mismatch)**, so the damage stream will not carry a survival claim. ⛔ **Not
built on it.** The dominated test above was chosen *because* it needs no such
claim, not because the counterfactual was uninteresting.

⇒ **Both halves of the passive-damage family close on SIZING, before any rule
was written** — rule 14 working as designed, for the price of one script and no
A/B. The user's observation stands (single decisions do lose games); what this
measures is that **these** decisions are not the ones, and the clone is already
near-perfect at the arithmetic part of targeting.

```powershell
python -X utf8 scripts/p72_loss_autopsy.py --verify
python -X utf8 scripts/p72_loss_autopsy.py --dir replays/submission_v5_s2
```

## 8bl. 🔴 E11 — THE CLONE REALLY DOES UNDER-DEVELOP ITS BENCH, AND FORCING IT TO STOP DOES NOT PAY (2026-08-07, day 25, 3rd session)

Pre-registered in `docs/experiments/E11-poffin.md`, frozen in **`a50a240`
before the rule was written**, bar and predictions included.

### How it was found — and why F1 had already missed it

§8bj closed F1 after classifying the clusters that `p66_mirror_disagree` ranked
**per decision** — and §8bj's own conclusion was that the per-decision unit is
wrong (rule 21). The corrected unit was then applied only to the two clusters
the wrong unit had already selected. 🔴 **The ranking, not just the sizing, has
to run per turn.** `scripts/p70_perturn_sweep.py` does that over **every** option
class with no pre-selection, and it surfaces something the per-decision view
cannot see, because the clone here is never *confidently wrong* — it simply
never gets round to the play:

**`Buddy-Buddy Poffin` (PLAY)** — *"search your deck for up to 2 Basic Pokémon
with 70 HP or less and put them onto your Bench."* Share of AVAILABLE TURNS in
which it is actually played, **conditioned on our own board occupancy**:

| board size | expert turns | expert plays | our turns | our plays | gap |
|---|---|---|---|---|---|
| 1 | 94 | 98.9% | 18 | 100.0% | +1.1% |
| 2 | 63 | 96.8% | 20 | 85.0% | −11.8% |
| 3 | 76 | 84.2% | 26 | 69.2% | −15.0% |
| **4** | **114** | **70.2%** | **51** | **29.4%** | **−40.8%** |
| **5** | **147** | **46.9%** | **69** | **7.2%** | **−39.7%** |

**0.80 fewer plays per game**, ordering-free, over the 0.5 sizing gate — and the
confound is checked: **both sides decline at the same mean board occupancy**
(4.46 vs 4.45), so it is the behaviour that differs, not the mix of situations.
⚡ **The first candidate this project has produced in which WE are the weaker
player at something specific, sized and ordering-free.**

### ✅ The positive control, run BEFORE the A/B

A rule that silently never fires produces a null that means nothing (the §8be
family). 40 recorded games with the rule on, mined and measured by the same
instrument:

| board | rule OFF | rule ON |
|---|---|---|
| 1–4 | 100 / 85 / 69 / 29% | **100% (forced, by construction)** |
| 5 | 7.2% | 13.2% (rule deliberately does not fire) |
| overall | 39.7%, 0.91 plays/game | **61.6%, 1.32 plays/game** |

⇒ the intervention is real and it is the one specified.

### 🔴 The result: n=2,800, byte-identical net both arms, rule toggled

| cell | score | 95% CI | verdict |
|---|---|---|---|
| `poffin` ON vs OFF | 🔴 **0.487** | [0.469, 0.506] | **FAILS the 0.53 bar; does not resolve; point estimate slightly NEGATIVE** |

`[health] OK fallbacks=0 net_missing=0`, 575,270 net calls, seats balanced.
Because the weight file is byte-identical on both sides, **the ±13 Elo seed
nuisance (§8bk) cancels exactly** — this is one of the cleanest cells in the
repo, and it says no.

⇒ **The rule does not ship. `EVIDENCE` keeps the measurement; the agent keeps
its behaviour.** Tradeoff rules are now **0 for 5**.

### What this does and does NOT license

- ✅ **The measurement stands**: our clone develops its bench less than the
  1150+ pilots do, by 0.80 plays/game, conditioned on identical boards.
- ✅ **The repair is refuted in the form tested**: forcing the play whenever
  ≥2 bench slots are free is worth −0.013, i.e. nothing or slightly worse.
- ⚠ **NOT shown: that matching the experts' *conditional* policy would fail.**
  They play it 70.2% at board 4; the rule forced **100%**, overshooting the
  behaviour it was copying. A milder variant is unproven, not disproven (rule 4).
  ⛔ **And it is not being run** — E11 pre-registered that a different threshold
  is "a separate experiment, not a knob to tune after seeing the result", and
  tuning a rule downward until the number improves is precisely how a project
  manufactures a winner at α=0.05.
- ⚡ **The most useful reading is about the CLONE, not the card.** A behavioural
  difference from the 1150s that is real, sized, ordering-free and confound-
  checked still converts to **zero** Elo. Taken with §8bj (their extra actions
  are timing) this is the sharpest statement of the project's thesis: **the gap
  to 1150 is not a list of moves our clone fails to make.**

### Scoring the pre-registration

- **Prediction 1 — "lands in [0.50, 0.53] and does NOT resolve"**: half right.
  It did not resolve ✅; the point estimate landed **below** the band (0.487) ❌.
- **Prediction 2 — "if it resolves at all, it resolves positive"**: untested, it
  did not resolve. Must not be quoted as confirmed.
- **Prediction 3 — the named way it could be wrong** (the search whiffs for want
  of Basics in deck): **not measured, and still not measured.** It remains the
  most likely mechanism for a forced play being worthless, and it is recorded
  here rather than quietly dropped.

```powershell
python -X utf8 scripts/p70_perturn_sweep.py --top 30
python -X utf8 scripts/arena.py play "bc:poffinON,net=out/policy_v5_s2.npz,noChip,noSpread,noSrc,poffin" "bc:poffinOFF,net=out/policy_v5_s2.npz,noChip,noSpread,noSrc" --matches 1400 --deck-a grimmsnarl --deck-b grimmsnarl --archive out/arena/p71_e11_ab.jsonl
```

## 8bk. 🔬 F2 STEP 2 — TEN SEEDS OF ONE RECIPE, AND THE SPREAD IS A RANGE, NOT A NUISANCE TERM (2026-08-07, day 25, 3rd session)

Pre-registered in `docs/experiments/E10-final-push.md` (F2 steps 2–3), frozen in
`ad7d29f`. Six new seeds (5–10) trained on the **byte-identical** v5 recipe —
same corpus `pds_v4`, same architecture, same hyperparameters, same 12 epochs,
`--pool --opt-cols 37` — and each screened identically against `policy_v5_s1`
in the shipped configuration.

### The screens: mirror, DIRECT, seat-balanced, `--no-rules` both arms, n=1,400/cell

| seed | score vs `s1` | 95% CI | Elo | resolves? |
|---|---|---|---|---|
| **s7** | **0.528** | [0.502, 0.554] | **+19.5** | ✅ yes (barely) |
| s2 (fresh, §8bh) | 0.510 | [0.484, 0.536] | +7.0 | no |
| s5 | 0.504 | [0.478, 0.530] | +2.8 | no |
| s9 | 0.504 | [0.477, 0.530] | +2.8 | no |
| s8 | 0.501 | [0.475, 0.527] | +0.7 | no |
| s4 | 0.480 | [0.454, 0.506] | −13.9 | no |
| s10 | 0.472 | [0.446, 0.499] | −19.5 | 🔴 yes, worse |
| v5 | 0.469 | (from §8bf) | −21.6 | — |
| s3 | 0.465 | [0.439, 0.491] | −24.4 | 🔴 yes, worse |
| s6 | 0.458 | [0.432, 0.484] | −29.3 | 🔴 yes, worse |

`[health] OK fallbacks=0 net_missing=0` on all six new cells.

### 🔴 The variance decomposition, on ten unselected draws

Observed sd **0.0232**; per-cell sampling sd √(0.25/1400) = **0.0134**;
⇒ **between-seed sd 0.0190 ≈ 13.2 Elo**, χ² 95% CI (9 df) **[9.1, 24.1]**.
Two random seeds of one recipe differ by ~**18.7 Elo** typically.

🔴 **And the max minus the min over these ten reads 48.7 Elo** — §8bg's headline
"50 Elo", reproduced almost exactly on a distribution whose sd is 13. ⇒ **the
number was never wrong as a range and was always wrong as a nuisance term: a
range grows with the number of draws, a standard deviation does not.** Every
doc quoting "the seed is worth 50 Elo" should say **"a lucky seed beats an
unlucky one by ~50 Elo when you look at ten of them; the seed-to-seed sd is
~13."**

⚡ **A second-order fact worth the report:** the *screen* ranking is itself
mostly noise. Of ten seeds, **three resolve worse than `s1` and exactly one
resolves better** — and that one (`s7`) clears the lower bound by 0.002. On a
13-Elo sd with a 13.4-Elo measurement, a screen is a weak instrument and the
only thing it is licensed to do is **nominate**, which is precisely what the
pre-registration restricts it to.

### 🔴 The confirmation — `s7` FAILS THE BAR AND THE HARVEST SHIPS NOTHING

`s7` (the single screen winner) vs the incumbent **`policy_v5_s2`**, shipped
configuration, mirror direct, **n=2,800**. Ship bar, pre-registered before the
cell: **point ≥ 0.53 AND CI excluding 0.50.**

| cell | score | 95% CI | verdict |
|---|---|---|---|
| `policy_v5_s7` vs `policy_v5_s2` | 🔴 **0.487** | [0.468, 0.505] | **FAILS — below 0.53, and does not even resolve above 0.500** |

`[health] OK fallbacks=0 net_missing=0`, 569,642 net calls, seats balanced
(P0 699/701, P1 663/735). ⇒ **Branch taken: keep `policy_v5_s2`. Nothing ships.
`55326513` stands.**

🔴 **The winner's curse, measured a second time and slightly larger.** Naive
transitivity through `s1` predicted `s7` vs `s2` ≈ **0.518** (0.528 and 0.510
against the same reference). Measured **0.487** — a give-back of **0.031**,
against §8bh's 0.027 on the previous instance. **Two instances, 0.027 and 0.031,
both in the direction the pre-registration expected.**

### ⚡ The finding that outlives the null: this instrument CANNOT harvest this effect

Between-seed sd is **0.0190**; a 1,400-game screen measures a seed with sd
**0.0134**. Those are the same size. **Selecting the max of ten draws under
measurement error that large selects mostly for measurement error** — which is
exactly what happened: `s7` topped the screens and then lost to a seed that
screened 0.018 lower. To make the screen a real instrument the per-cell error
would have to fall to ~half the signal, i.e. **≈5,100 games per screen** (10
seeds ≈ 51,000 games, ~3.5 h of arena), and the prize for all of it is the
**≈+20 Elo** an E[max of 12] buys over the median — **on a ladder whose noise
floor is 63 points (§8ak).**

⇒ **F2 closes as a null with a quantitative reason, not a shrug: the seed
lottery is real, it is ~13 Elo of sd, and it is not harvestable with screens
whose error equals the effect.** ⚠ It is *not* shown that a better-powered
harvest would fail — that is unrun, not refuted (rule 4).

### Scoring the pre-registration, item by item

- **"The screen distribution will span ~50 Elo again"** — ✅ correct (48.7), and
  §8bh had already narrowed *why* that is the wrong statistic to quote.
- **"The winner's fresh-game confirmation vs `s2` shrinks toward ~0.51–0.53 and
  the bar is a coin flip to clear"** — 🔴 **wrong, and wrong in the direction
  that matters**: it landed at **0.487**, below the whole predicted band. The
  prediction under-modelled the give-back by treating one screen's selection as
  the only bias; the confirmation also changes *opponent*, and the incumbent
  `s2` is itself the 2nd-best of the ten draws.
- **"Either outcome is fine: the bundle already holds a selected seed"** — ✅ and
  it is now the load-bearing sentence: the shipped net ranks **2 of 10**.

```powershell
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 --bs 1024 --loss listwise --state-h 512,256 --head-h 256,128 --pool --opt-cols 37 --seed 7 --out out/policy_v5_s7.npz
python -X utf8 scripts/arena.py play "bc:v5s7ship,net=out/policy_v5_s7.npz,noChip,noSpread,noSrc" "bc:v5s1ship,net=out/policy_v5_s1.npz,noChip,noSpread,noSrc" --matches 700 --deck-a grimmsnarl --deck-b grimmsnarl --archive out/arena/p68_seed_s7.jsonl
```

## 8bj. 🔴 F1 — THE BIGGEST DISAGREEMENT WITH THE 1150s IN THE MIRROR IS *WHEN*, NOT *WHETHER*: an on-policy control shows identical rates, and F1 closes as a chapter (2026-08-07, day 25, 3rd session)

Pre-registered in `docs/experiments/E10-final-push.md` (F1 steps 2–4) with the
kill criterion written before any extraction: *"no cluster passes sizing AND
classifies as dominated → F1 closes as a chapter."*

### Step 2 — the extraction

`policy_v5_s2` scored over all **21,785** single-choice expert decisions in the
**257** mirror games (§8bi), `--equiv` on so two copies of one card in one role
cannot manufacture a disagreement (§8x):

| | count | rate |
|---|---|---|
| disagreements with the expert | 7,318 | 33.6% of decisions |
| equivalent-option ties dropped | 386 | — |
| **confident** (clone margin ≥ 0.25) | **4,785** | 22.0%, **18.6/game** |

Margin quantiles among disagreements: p50 0.370, p90 0.823. **519 clusters**
by (context, what the clone wanted, what the expert took); **3 pass the
pre-registered 0.5 firings/game gate.** Ranked first by a wide margin: the clone
wanting **Munkidori** — 2,167 confident disagreements, **8.4/game, in 253 of 257
games.**

### 🔴 Step 3 — and the ranking is an ARTIFACT of counting per DECISION

This is §8ai's detector defect in a new costume. That day's empty-bench detector
counted a pilot as declining to bench when it benched **later in the same turn**,
and it made a clean anchor (`rule:archaludon`) look broken. Munkidori's
Adrena-Brain is **"Once during your turn"** — so an agent that fires it at
action 1 and an expert who fires it at action 4 disagree on **every decision in
between** and do **exactly the same thing**.

`scripts/p67_option_rate.py` asks the ordering-free question instead. Munkidori
[ABILITY], expert mirror corpus:

| | per decision | per turn |
|---|---|---|
| available | 16.2/game | 3.77/game |
| **expert used it** | **38.5%** of available decisions | **93.8%** of available turns |
| **clone's top-1 was it** | **75.1%** of available decisions | **96.9%** of available turns |

**The 2× per-decision gap is a 3-point per-turn gap.** Residual "clone wanted it
in a turn the expert never used it": **0.19/game — below the 0.5 gate**, and in
the same band as Morgrem (0.2) and the Archaludon rule (0.187), both killed on it.

### ✅ And the on-policy control settles it, because the counterfactual could not

The clone column above is off-policy. So the shipped agent was made to **play**
80 mirror games in the shipped configuration (`harness.Recorder` via
`p20_record_games.py`, §8ad — the recorder whose output is byte-compatible with
Kaggle replays), those games were mined into a corpus with the **same** miner,
and the **same** instrument was run on them:

| Munkidori [ABILITY] | our shipped agent (on-policy, 1 seat, 80 games) | 1150+ experts (257 mirror games) |
|---|---|---|
| available turns / game | 3.71 | 3.77 |
| **uses / game** | **6.42** | **6.23** |
| uses per available turn | 1.73 | 1.65 |
| % of available turns used | 99.3% | 93.8% |

🔴 **We use Munkidori's ability at the same rate the 1150+ pilots do. The entire
cluster is sequencing** — we take it at the first opportunity (82.8% of offered
decisions), they take it later in the turn (38.5%) — **and sequencing is a closed
axis.** ⚡ **The instrument's own positive control comes free**: on our own
recorded games the "demonstrator took it" and "clone's top-1" columns are
identical (514/514, ratio 1.00), which exercises the reconstruction, the option
matching, the width-slicing and the net scoring end to end.

### Step 4 — the one ordering-free difference, and the discriminator says NO RULE

**Spikemuth Gym** (a Stadium: *"Once during each player's turn, that player may
search their deck for a Marnie's Pokémon…"*). Here the per-turn gap survives:

| Spikemuth Gym search | our agent | experts |
|---|---|---|
| available turns / game | 6.08 | 5.84 |
| **% of available turns used** | **95.7%** | **72.7%** |
| uses / game | 6.34 | 4.79 |
| turns where the clone wants it and the expert never used it | — | **0.77/game ✅ passes sizing** |

**The separating variable is the TURN NUMBER.** Averaged over available turns,
the experts use it at turn **6.56** and decline at turn **9.73**; we use it at
7.47 and decline (rarely) at 11.24. ⇒ **the 1150s stop searching once their
board is built; we keep searching to the end of the game.**

⚡ **Two follow-ups make the classification evidential rather than a judgement
call.** First, conditioning on the late game sharpens the split — at **turn ≥
10** we search in **146 of 162** available turns (**90.1%**) and the experts in
**236 of 471** (**50.1%**). 🔴 **A coin flip is not a rule.** If declining late
were dominated the 1150s would sit near 0%, not at 50% — *they* are deciding
case by case, which is what a tradeoff looks like from the outside. Second, the
searched card is tracked to see whether it is ever **played**:

| Spikemuth search → is the fetched card later played? | ours | experts |
|---|---|---|
| all searches | 345/390 = **88%** | 265/304 = **87%** |
| late searches (turn ≥ 10) | 48/80 = **60%** | 20/44 = **45%** |

⇒ **our extra late searches are not waste**: the card comes down 60% of the
time, a *higher* rate than the experts manage on their own late searches. What
they fetch is the same three cards in the same order (Grimmsnarl ex, Morgrem,
Impidimp). **Nothing here is dominated in either direction.**

🔴 **Classified as a TRADEOFF, so no rule is built (rule 11; tradeoff rules are
0 for 4).** The action is a free once-per-turn search with one mechanical cost —
one fewer card in the deck — and that cost only pays out in a deck-out, while
these games end at turn 11–13 with the deck nowhere near empty. **"Decline it
late" is therefore not provable by arithmetic from the board**, which is exactly
the bar the discriminator sets. ⚠ It is *also* not shown to be harmless; what is
shown is that it is **not the class rules have ever repaired.**

### ⇒ F1's kill criterion is met, and F1 closes as a chapter

*"No cluster passes sizing AND classifies as dominated."* Three passed sizing:
one dissolved into sequencing under an on-policy control, one is a tradeoff, one
(below) is encoding-shaped and on a closed axis. **The finding that goes in the
report is stronger than the rule would have been: what separates a 990 clone
from a 1150 rule-agent in our own matchup is not a menu of moves it fails to
find — it takes the same actions at the same rates — it is WHEN and WHETHER-late,
and that is precisely the class a per-decision clone cannot be repaired toward
with a rule.**

### 🔴 The third sized cluster looked like a REPRESENTATIONAL DEFECT for about ten minutes, and it is not one

The third cluster is `TO_HAND` with **both** sides unnamed, 1.41/game: in
**22.3% of TO_HAND rows every option carries `opt_card == 0`**, and those rows
**survive `--equiv`**, so they are not identical options either. Written up
first as "the net is choosing a search target whose identity is not in its
features" — an encoding-shaped defect in the B1 lineage. **That was wrong, and
the raw replays say so.** Option areas across 592 TO_HAND selects in 25 expert
mirror games:

| area | selects | card id resolvable? |
|---|---|---|
| 1 = DECK (`sel.deck` present) | 366 (61.8%) | ✅ yes — `_card_at` reads `sel["deck"][index]` |
| **6 = PRIZE** | **161 (27.2%)** | ⛔ **no, and correctly no** |
| 3 = DISCARD | 49 (8.3%) | ✅ yes |
| 12 = LOOKING | 16 (2.7%) | mostly yes |

🔴 **The unnamed options are PRIZE CARDS. They are face down.** No agent, human
or otherwise, knows which prize it is taking, and the one feature that separates
them is exactly the one that should: `opt_dense[36]`, the index disambiguator
(**the only differing column in all 615 rows, confirmed bitwise**). ⇒ **there is
no blind spot here, there is a hidden-information choice**, our net and a 1150+
pilot are both guessing, and a "disagreement" on a coin flip is not a defect in
either of them. ⛔ **Nothing to price and nothing to open.**

⚡ **Worth recording as a method note rather than buried**: the defect-shaped
reading survived the sizing gate, the `--equiv` filter and the clustering, and
died to *one look at the raw option dicts*. The encoding axis stayed closed by
inspection, not by an A/B — which is the cheapest way it has ever been defended.

### Scoring the pre-registration

- **"≥1 cluster passes the frequency bar"** — ✅ correct, 3 did.
- **"Most clusters classify as tradeoffs and F1 most likely ships nothing while
  producing the report's strongest §7b addendum"** — ✅ correct on both halves.
- **"If a dominated class exists, my guess is targeting/bench-management"** —
  untested: no dominated class was found anywhere, so the guess neither hit nor
  missed. Recorded as unfalsified, not as a hit.
- **Not predicted, and the actual result:** that the largest cluster would be an
  artifact of the counting unit. E10 wrote the sizing gate in firings/game and
  still let the *ranking* run per decision. ⇒ **the §8ai lesson needs to be a
  standing rule, not a remembered anecdote: size AND rank per turn.**

```powershell
python -X utf8 scripts/p66_mirror_disagree.py --net out/policy_v5_s2.npz --ds artifacts/pds_mirror_exp --equiv --margin 0.25 --dump out/logs/f1_disagreements.jsonl
python -X utf8 scripts/p20_record_games.py --a "bc:v5s2ship,net=out/policy_v5_s2.npz,noChip,noSpread,noSrc" --b "bc:v5s2ship_b,net=out/policy_v5_s2.npz,noChip,noSpread,noSrc" --deck-a grimmsnarl --games 80 --swap --out out/replays/f1_ours_mirror
python -X utf8 scripts/p67_option_rate.py --card Munkidori --type ABILITY --ds artifacts/pds_mirror_exp
python -X utf8 scripts/p67_option_rate.py --card Munkidori --type ABILITY --ds artifacts/pds_ours_mirror1
```

## 8bi. ✅ F1's SIZING GATE PASSES (257 mirror games, 22,665 expert decisions) and 🔴 F3's COVERAGE LEVER IS KILLED — the games that would fix the corpus DO NOT EXIST at any band we can mine (2026-08-07, day 25, 3rd session)

Both gates pre-registered in `docs/experiments/E10-final-push.md`, frozen in
`ad7d29f`. One new instrument answers both, because both questions are about a
**matchup** and every census this repo has run labels only one seat.

### The instrument: `scripts/p65_archetype_census.py`

Labels **seat 0 and seat 1 separately**, each from the *other* seat's
observation frames — the same lower-bound reconstruction
`p9_field_census.analyse` uses, so the labels are comparable with every share
already published. Two defects it caught in its own first run, recorded because
both are the §8ay/§8ax family (a label that silently means something else):

- 🔴 **`_signature` returns `Marnie's Grimmsnarl ex`, not `Grimmsnarl ex`.** The
  first version hardcoded the short name and reported **0.0% mirror** on a dump
  that is 45% mirror. A zero that looks like a finding.
- 🔴 **The demonstrator is `李秉叡（ntumlnoob）`, not `ntumlnoob`.** Exact-match
  on `--player` returned a zero-game corpus — rule 9's failure mode, caught here
  by the mirror count being non-zero while the player count was zero.

### ✅ F1 — the mirror corpus is large, and it is larger than the gate asked for

| dump | games | mirror games | expert decisions in mirror |
|---|---|---|---|
| `replays/ntumlnoob_31-07-2026` | 330 | **148** (44.8%) | **12,406** |
| `replays/sixth_sense_31-07-2026` | 228 | **112** (49.3%) | **10,259** |
| **unique** (3 episodes appear in both dumps) | 555 | **257** | **22,665** |

**Gate was ≥100 mirror games. Measured 257, and 22,665 expert decisions inside
them** — a corpus 1.75× the size of the held-out split every agreement number in
this repo is computed on (12,939). ⇒ **F1 proceeds to step 2 with no widening
and no top-band mining.**

⚠ **E10's stated reason for expecting the pass was wrong, and the distinction
matters for anything else that reads these dumps.** It predicted *"their band is
>70% mirror"*. **71.7% is the share of SEATS playing Grimmsnarl; the share of
GAMES that are mirror is 44.8%** — because a Grimmsnarl seat facing an Alakazam
seat contributes one Grimmsnarl seat and zero mirror games. The two numbers
differ by a factor of 1.6 and the prediction quoted the wrong one. Recorded as a
**pass on the gate, a miss on the prediction.**

### 🔴 F3 — the corpus-coverage lever, killed on AVAILABILITY

`PARKED-corpus-coverage.md`'s probe, run over the exact four dumps that built
`artifacts/pds_v4` (`replays/2026-07-2{6,7,8,9}`, 1,603 games, manifest
`avg_score` **1057–1223, mean 1171**). Decisions belonging to a **Grimmsnarl
seat**, by what it was facing:

| opponent | field weight (§8ac) | corpus decisions | share |
|---|---|---|---|
| mirror | 33.3% | 88,676 | **56.9%** |
| Crustle | 6.7% | 15,560 | 10.0% |
| Team Rocket's Spidops | — | 11,990 | 7.7% |
| Teal Mask Ogerpon ex | — | 9,136 | 5.9% |
| Dragapult ex | 5.3% | 8,097 | 5.2% |
| Alakazam | 22.0% | 7,668 | 4.9% |
| Cynthia's Garchomp ex | 6.7% | 923 | 0.6% |
| **Archaludon ex** | **8.0%** | **0** | **0.00%** |
| **Mega Lucario ex** | **4.0%** | **0** | **0.00%** |

🔴 **Zero games. Not "discarded by the miner" — the miner has no archetype
filter at all** (`build_policy_dataset` clones both seats of every game unless
`--player` is given, and `pds_v4` was built without it). **The games are not in
the dumps.** And they cannot be got: these dumps are the *top* of Kaggle's
episode feed, and §8i already measured that the feed **stops at avg_score
~1055** while the archetypes in question live below it (§8ac: Archaludon and
Mega Lucario are **0 of 47** in games above opponent rating 900). ⇒ **the repair
this lever proposes requires training data that does not exist on the platform,
at the band where the decks are actually played.** Killed, not declined.

⚡ **And the diagnosis was pointing the wrong way anyway.** The corpus is 56.9%
mirror against a field that is 33.3% mirror *at the rating we held when §8ac was
measured* and **71.4% mirror above 1000** — where we now sit (`55326513` reads
1004.5). **The "over-representation" of the mirror is the corpus being aimed at
the field we are climbing INTO**, and the under-represented archetypes are
exactly the ones §8ac says vanish as we climb. ⇒ **`PARKED-corpus-coverage.md`
is answered and closes: the gap is real, unfixable with obtainable data, and
shrinking on its own.**

⚠ The one thing this does **not** license: nothing here says the mirror-heavy
corpus is *optimal*, only that its mismatch cannot be repaired by mining. A
different corpus weighting is a different (closed) axis, not this one.

```powershell
python -X utf8 scripts/p65_archetype_census.py --dir replays/ntumlnoob_31-07-2026 --player ntumlnoob
python -X utf8 scripts/p65_archetype_census.py --dir replays/2026-07-26 --dir replays/2026-07-27 --dir replays/2026-07-28 --dir replays/2026-07-29 --top 16
```

## 8bh. 🔴 THE SELECTION DEBT IS PAID AND IT COST 0.027 — `s2`'s edge is +7 Elo, not +26, and §8bg's "50 Elo seed spread" was mostly the MAX of a selected set (2026-08-07, day 25, 3rd session)

**Pre-registered** in `docs/experiments/E10-final-push.md` (F2 step 1), frozen in
`ad7d29f` before either cell ran. §8bg closed by recording a debt in plain words:
*"`s2` won a screen of THREE seeds, so 0.537 is inflated by selection … a
confirmation run on fresh games does not exist yet."* It exists now.

### Mirror, DIRECT, seat-balanced, `--no-rules` BOTH arms, n=1,400/cell, all vs `policy_v5_s1`

| cell | score | 95% CI | Elo | verdict |
|---|---|---|---|---|
| `policy_v5_s2` — **screen** (§8bg, selected best of 3) | 0.537 | [0.511, 0.563] | +25.8 | ✅ resolved |
| `policy_v5_s2` — **fresh games** (this cell) | 🔴 **0.510** | [0.484, 0.536] | **+7.0** | **does NOT resolve** |
| `policy_v5_s4` — first screen | 0.480 | [0.454, 0.506] | −13.9 | does not resolve |

`[health] OK fallbacks=0 net_missing=0` on both new cells; 282,543 and 289,009
net calls, no fallback, no missing weights.

🔴 **0.537 → 0.510 on the same comparison, same configuration, same weight
files, different games.** The two readings are *statistically compatible*
(Δ=0.027, SE 0.019, p≈0.15) — this is not a contradiction and §8bg is not
retracted. But only one of them is an unbiased estimate of `s2`'s level: the
screen was **conditioned on being the maximum of three**, and the confirmation
was not. ⇒ **the honest number for the shipped net is 0.510, and the +25.8 Elo
in every doc that quotes §8bg should read +7.0.**

### 🔴 What this does to the "50 Elo seed spread"

§8bg's headline was `s2` 0.537 vs `s3` 0.465 — a **max minus a min over three
draws**, which is an order statistic, not a standard deviation. The four seed
offsets against `s1` that are *not* selected on:

| net | vs `s1` | source |
|---|---|---|
| `policy_v5` | 0.469 | §8bf (0.531 with the seats swapped) |
| `policy_v5_s2` | 0.510 | this section, fresh games |
| `policy_v5_s3` | 0.465 | §8bg |
| `policy_v5_s4` | 0.480 | this section |

Observed sd across the four is 0.0203; the per-cell sampling sd is
√(0.25/1400)=0.0134, so the **between-seed** sd is
√(0.0203² − 0.0134²) ≈ **0.015 in win rate ≈ 11 Elo**, and two random seeds
differ by √2× that ≈ **15 Elo typically** — not 50. ⚠ **On n=4 that estimate is
itself very loose** (3 df: a factor of ~2 either way is inside the interval).
The claim that survives is directional and it is enough to act on: **the seed
nuisance is real, it is smaller than §8bg said, and §8bg said it was that large
because it read a selected extreme as a spread.**

> ⚡ **UPDATED the same session with six more seeds — and the update is the
> cleanest possible demonstration of the point.** F2 step 2 screened seeds 5–10
> identically (§8bk), giving **ten** unselected offsets against `s1`: 0.469,
> 0.510, 0.465, 0.480, 0.504, 0.458, **0.528**, 0.501, 0.504, 0.472. Observed sd
> 0.0232, sampling sd 0.0134 ⇒ **between-seed sd 0.0190 ≈ 13.2 Elo** (χ², 9 df:
> **[9.1, 24.1]**). 🔴 **And the max-minus-min over those ten draws reads 48.7
> Elo** — §8bg's "50 Elo", reproduced almost exactly, **on a distribution whose
> standard deviation is 13.** ⇒ **the original number was never wrong as a
> range; it was wrong as a description of the nuisance.** A range grows with the
> number of draws; a sd does not.

### ⚠ What does NOT change

- **The seed is still pure nuisance and still comparable to real interventions.**
  11 Elo of sd sits between v5's +14 and nothing; a 2-seed A/B still measures its
  intervention on top of it, so §8aw's warning and the ≥3-seed rule stand.
- **Nothing about the ensembling verdicts** (§8bf, §8bg): those were
  net-vs-net at fixed weight files, with no selection step.
- **Nothing shipped moves.** `55326513` is `policy_v5_s2`, and this cell says it
  is a **median seed**, not a lucky one — which is a demotion of the ship's
  provenance and an *increase* in the headroom F2 is hunting.

### 🔴 And it re-prices F2 before F2 spends its budget

E10 justified the seed harvest on σ≈25: *"best-of-12 ≈ +35–40 Elo over the
median seed"*. At sd≈11 the same order statistic (E[max of 12] ≈ 1.63σ) is
**≈ +18 Elo over the median**, and this section measures the winner's curse that
eats into it on its only instance so far: **0.027 of win rate, ~19 Elo, lost
between a screen and its confirmation.** ⇒ **the harvest is still positive-EV
but roughly half the size E10 priced it at, and the pre-registered structure —
screens select, only a fresh-game confirmation ships — is what stops it from
being a bias-manufacturing machine.** The protocol is unchanged; the expectation
is halved, and it is halved *before* the ship decision rather than after.

⚡ **Third time a number has fallen when a debt was paid rather than quoted**:
B8's estimate went down at 4× the data (§8ao), the E5 "scaling curve" flattened
when its independent variable was checked (§8bb), and now a screen winner gives
back 0.027 on fresh games. **The pattern is not that this project is unlucky —
it is that selected numbers are the ones that get published unless something
forces a re-draw.**

### Scoring the pre-registration

- **E10's F2 prediction — "the winner's fresh-game confirmation shrinks toward
  ~0.51–0.53"** — was written about a *future* screen winner and lands exactly on
  the incumbent's own re-draw: **0.510.** Recorded as a hit on the mechanism,
  untested on the object it named.
- **E10 assumed the screen distribution "will span ~50 Elo again."** On the
  evidence above it should be expected to span ~30, and the F2 write-up must not
  reuse the 50.

```powershell
python -X utf8 scripts/arena.py play "bc:v5s2ship,net=out/policy_v5_s2.npz,noChip,noSpread,noSrc" "bc:v5s1ship,net=out/policy_v5_s1.npz,noChip,noSpread,noSrc" --matches 700 --deck-a grimmsnarl --deck-b grimmsnarl --archive out/arena/p65_s2_confirm.jsonl
python -X utf8 scripts/arena.py play "bc:v5s4ship,net=out/policy_v5_s4.npz,noChip,noSpread,noSrc" "bc:v5s1ship,net=out/policy_v5_s1.npz,noChip,noSpread,noSrc" --matches 700 --deck-a grimmsnarl --deck-b grimmsnarl --archive out/arena/p65_s4_screen.jsonl
```

## 8bg. 🔴 THE TRAINING SEED IS WORTH 50 Elo — more than the best feature intervention this project ever found, and it is pure nuisance (2026-08-07, day 25)

> 🔴 **NARROWED BY §8bh (same day, 3rd session): the "50 Elo" below is a MAX
> MINUS A MIN over three draws, and its maximum was selected.** `s2` re-measured
> on fresh games reads **0.510, not 0.537**; the between-seed sd over four
> unselected offsets is **≈11 Elo**, so two random seeds differ by ~15 Elo, not
> 50. The *nuisance* finding stands; the *magnitude* in this section's title does
> not. Read §8bh before quoting any number here.

**Decision rule pre-registered** in `docs/experiments/E9b-which-net-ships.md`,
frozen in commit `e214c66` **before the ens5 cell reported**, including two
predictions and the branch that was actually taken.

Three fresh seeds of the v5 recipe (seeds 2/3/4 — same corpus `pds_v4`, same
architecture, same hyperparameters, same listwise loss, same 12 epochs; the
**only** difference is the integer passed to `--seed`), each screened against
`policy_v5_s1` in the **shipped** configuration.

### Mirror, DIRECT, seat-balanced, `--no-rules` BOTH arms, n=1,400/cell

| cell | score | 95% CI | Elo | verdict |
|---|---|---|---|---|
| `policy_v5_s2` vs `policy_v5_s1` | **0.537** | [0.511, 0.563] | +25.8 | ✅ resolved BETTER |
| `policy_v5_s3` vs `policy_v5_s1` | **0.465** | [0.439, 0.491] | −24.4 | 🔴 resolved WORSE |
| ens5 vs `policy_v5_s1` | 0.522 | [0.496, 0.548] | +15.3 | null — does not resolve |

`[health] OK fallbacks=0 net_missing=0` on all three.

🔴 **`s2` and `s3` are 0.072 apart — a ~50 Elo spread produced by nothing but
the random seed.** Set that against the interventions this project has paid for:
the v4 state block, found by systematic feature enumeration and the
second-largest confirmed win in the repo, was **+37 Elo**. The v5 pooled block
was **+14**. ⇒ **Choosing a lucky seed is worth more than the best feature
engineering we ever did, and it is not a lever — it is a nuisance term that
every previous A/B was measured on top of.**

⚡ **This is an independent reproduction, not a new anomaly.** §8be noted in
passing that two same-recipe nets differing only in `--seed` swing **0.073**
against each other, and filed it as a complaint about instrument noise. Measured
here as an effect in its own right, in the shipped configuration, at n=1,400:
**0.072.** ⇒ **§8aw's "two seeds under-resolves every anchor we own" was more
right than it knew** — a 2-seed budget measures an intervention against a ±25
Elo nuisance, which is larger than most interventions tested.

### The 5-net vote does NOT rescue it

ens5 averages `v5`, `s1`, `s2`, `s3`, `s4` — and the screen shows that set is
**one good member (`s2`), one middling (`s1`), and two weak (`v5`, `s3`)**. It
reads 0.522 [0.496, 0.548]: **it does not resolve above 0.500 against a member
it contains.** ⇒ **a vote is bounded by its members; three mediocre nets cannot
be averaged into a good one.** Taken with §8bf's ens2 null, **the ensembling
axis has now failed to beat its own best member twice, at two different member
counts, both in the shipped configuration.**

### Scoring the pre-registration, including where it was wrong

- **Prediction 1 — "X lands in [0.52, 0.56]": correct on the point estimate
  (0.522) and at the very bottom edge of the band.** The part that mattered —
  whether it resolved — went the other way. Recorded as a weak hit, not a hit.
- **Prediction 2 — "ens5 will not beat `s2` by a resolved margin": untested.**
  That cell was deliberately not run (the rule forbade it), so this stands
  unfalsified rather than confirmed. It must not be quoted as evidence.
- **Branch taken: 2** (X did not resolve above 0.500) ⇒ **`policy_v5_s2` ships
  alone.** Submitted as **`55326513`**, evicting `55169114` (918.5) and retaining
  `55321893` (934.7).

⚠ **THE DEBT, RECORDED RATHER THAN HIDDEN: `s2` won a screen of three seeds, so
0.537 is inflated by selection by an unknown amount.** It survives a Bonferroni
correction for three comparisons (p≈0.016), so this is a caveat and not a
retraction — but **the honest estimate of `s2`'s edge is below 0.537 and nobody
should quote that number as `s2`'s level until a confirmation run on fresh games
exists.** It does not exist yet.

⛔ **DO NOT vote only the members that won the screen.** That selects on the
screening data and compounds the same bias across the whole ensemble. If it is
worth testing it is worth testing on fresh games.

```powershell
python -X utf8 scripts/arena.py play "bc:v5s2ship,net=out/policy_v5_s2.npz,noChip,noSpread,noSrc" "bc:v5s1ship,net=out/policy_v5_s1.npz,noChip,noSpread,noSrc" --matches 700 --deck-a grimmsnarl --deck-b grimmsnarl --archive out/arena/p64_seed_screen.jsonl
```

## 8bf. 🔴 E9 RE-MEASURED IN THE SHIPPED CONFIGURATION: the VOTE is a null against its better member, and the live bundle's whole gain is the SEED SWAP (2026-08-07, day 25)

§8be closed with the configuration trap named but not paid: arena `bc` defaults
to `chip_targeting`/`energy_spread`/`counter_source` **ON**, the submission
builds `--no-rules` with all three **OFF**, and **every E9 cell ran rules-on**.
Day 24 launched the re-measurement, archived 1,404 games to
`out/arena/p62_ship_config.jsonl`, and **never scored them.** Scored here, plus
the two cells that were missing.

### Mirror, DIRECT, seat-balanced, fixed weight files both sides, `--no-rules` on BOTH arms

| cell | shipped config (rules OFF) | §8be (rules ON) |
|---|---|---|
| ens2 vs `policy_v5` (the old shipped net) | **0.559** [0.533, 0.585] | 0.541 [0.519, 0.563] |
| ens2 vs `policy_v5_s1` (the better member) | 🔴 **0.505** [0.479, 0.531] | 0.531 [0.510, 0.553] |
| `policy_v5_s1` vs `policy_v5` (the seed swap) | **0.531** [0.505, 0.557] | 0.549 [0.527, 0.571] |

n=1,400/cell (1,404 for row 1), `[health] OK fallbacks=0 net_missing=0` on all
three — so no arm silently played the index-order fallback or the stale lw2
singleton (§8ax defects 3 and 4).

🔴 **Read rows 2 and 3 together.** The seed swap resolves (lower bound 0.505,
barely). The vote **does not beat the member it is built on**. Those are
consistent in exactly one way: **the live bundle's entire measured gain over the
old shipped net is the seed swap, and the vote contributes nothing measurable on
top of it.** Shipping `policy_v5_s1` as a single file would, on this evidence,
have bought the same thing for half the inference and none of the fail-soft
machinery.

⚠ **This is NOT a refutation of §8be, and must not be written up as one.**
0.505 [0.479, 0.531] and 0.531 [0.510, 0.553] overlap — the rules-on point
estimate sits exactly on the rules-off upper bound. Two readings this close
cannot separate "the configuration changed the answer" from "these are two
samples of one quantity". The claim that survives is narrower and still
decision-relevant: **the vote is unproven in the configuration we ship, not
disproven.** ⇒ **rule 4 applies — the mechanism is retracted for the shipped
configuration, not falsified.**

🔴 **And it undercuts the comparison that PICKED ens2.** §8be preferred ens2 over
the seedswap on ΔW +0.0289 (6/7 anchors) vs +0.0215 (4/7). **All fourteen of
those anchor cells were rules-on.** The two candidates it was choosing between
are, in the shipped configuration, separated by one null. ⇒ **no weighted anchor
verdict in §8be describes the shipped agent**, and re-running that table is the
only way to restore it.

### ⚡ AND THE DECORRELATION LESSON WAS ATTRIBUTED TO THE WRONG VARIABLE

`scripts/p63_net_agreement.py` makes §8be's one-off agreement check a standing
gate and reproduces both published numbers independently, over **12,939
held-out decisions** rather than §8be's 1,471: `policy_v5c_s1` vs
`policy_v5_s1` **100.0%** (exact), and `policy_v5` vs `policy_v5c_s0` **88.4%**
against the published 87.5%. Then three fresh seeds of the v5 recipe:

| pair | agree | pair | agree |
|---|---|---|---|
| `v5` vs `v5_s1` | 80.8% | `v5_s1` vs `s2` | 81.2% |
| `v5` vs `s2` | 81.4% | `v5_s1` vs `s3` | 80.6% |
| `v5` vs `s3` | 81.2% | `s2` vs `s3` | 80.9% |

🔴 **Every independent seed pair lands in 80.6–81.4% — a 0.8-point spread across
six pairs.** The 88.4% outlier is `v5c_s0`, which is **not a v5-recipe seed**.
⇒ **§8be's "members must be decorrelated" is right, but its implied cause is
wrong: seed variation does not produce correlated members, mixing RECIPES does.**
Training more seeds cannot fail the decorrelation gate, so "check agreement
before adding a member" is cheap insurance rather than the binding constraint
day 24 took it for.

⚠ **The gate default is 85%, and that number is a MIDPOINT GUESS stated as
such.** One pair helped at 80.8% and one hurt at 88.4%; nothing has measured
where between them the boundary is. HANDOFF's "~90%" was too loose — it would
have waved through the exact pair that made ens3 lose.

```powershell
python -X utf8 scripts/arena.py play "bc:ens2ship,net=out/policy_v5.npz+out/policy_v5_s1.npz,noChip,noSpread,noSrc" "bc:v5s1ship,net=out/policy_v5_s1.npz,noChip,noSpread,noSrc" --matches 700 --deck-a grimmsnarl --deck-b grimmsnarl --archive out/arena/p62_ship_config.jsonl
python -X utf8 scripts/p63_net_agreement.py --nets out/policy_v5.npz,out/policy_v5_s1.npz,out/policy_v5_s2.npz,out/policy_v5_s3.npz
```

## 8be. ⚡ E9 — SEED ENSEMBLING WORKS, AND THE SHIPPED NET WAS THE WEAKER OF TWO WE ALREADY HAD (2026-08-07, day 24)

> ⚠ **NARROWED BY §8bf (day 25) — read that first.** Every cell in this section
> ran **rules-ON** while the submission ships **rules-OFF**. Re-measured in the
> shipped configuration, the **seed swap survives (0.531)** and **the vote does
> not beat its better member (0.505, null)**. What stands: the shipped net was
> the weaker seed, and ensembling beats the *old shipped net*. What does **not**
> stand unqualified: the title's "ensembling works" (it does not resolve against
> the best member where it counts), and the weighted-anchor comparison below
> that **chose** ens2 over the seedswap — all fourteen of those cells are
> rules-on. Also: the "correlated members" diagnosis is right but its cause was
> misattributed — see §8bf, fresh seeds are uniformly ~81% apart.

**Pre-registered** in `docs/experiments/E9-ensemble.md` before any cell ran,
including two predictions that were wrong.

**Why this is not the closed capacity axis (§8w).** §8w scaled ONE net 2.6× and
8.2× and bought 2 decisions of 12,939, then lost 43 — the features bind a single
fitted function. Ensembling averages functions fitted **independently**. ⚡ And
the precondition was already measured, filed as a warning about our instrument:
§5.6/E8 found two same-recipe nets differing only in `--seed` swinging **0.073**
against each other. Re-measured directly here: `policy_v5` and `policy_v5_s1`
**disagree on 23.0% of 1,471 real ladder decisions** — about as far apart as we
are from human demonstrators. Nothing in this repo had ever tried a vote.

### Mirror, DIRECT head-to-head, fixed weight files both sides, n=2,000/cell

| cell | score | 95% CI | verdict |
|---|---|---|---|
| 🔴 `policy_v5` (SHIPPED) vs `policy_v5_s1` | **0.451** | [0.429, 0.473] | **the shipped net is the WEAKER seed** |
| ens2 vs `policy_v5` | **0.541** | [0.519, 0.563] | resolved |
| ens2 vs `policy_v5_s1` | **0.531** | [0.510, 0.553] | resolved — beats the better member too |
| ens3 vs `policy_v5_s1` | 0.491 | [0.469, 0.513] | null |
| ens3 vs ens2 | 0.495 | [0.473, 0.517] | null |

🔴 **Prediction 1 was wrong: arm C is not a null.** `policy_v5_s1` has sat in
`out/` since 08-01 and is ≈ +34 Elo on the net we ship. **Every A/B this project
ever ran "against v5" used the weaker of two available nets.**

### 🔴 MORE MEMBERS IS NOT BETTER — correlated members actively hurt

There are **four** v5-recipe nets on disk but only **three policies**:
`policy_v5c_s1` is **100.0% decision-identical** to `policy_v5_s1` (different
md5, same function). Voting with both would give that policy two of four votes —
a weighted vote with weights nobody chose, flattering the result because the
doubled member is the stronger one. And the honest 3-net vote **still lost**:
`policy_v5c_s0` agrees with `policy_v5` on **87.5%** of decisions, so ens3 is
effectively two votes for the v5-ish policy against one for the stronger `s1`.
⇒ **Ensemble members must be decorrelated; 87.5% agreement is enough to hurt.**
`build_submission.py` now refuses byte-identical members outright.

### Weighted anchor confirmation — 90.6% of the field, n=1,500/cell, one session

| anchor | w | incumbent | seedswap | ens2 | Δ swap | Δ ens2 |
|---|---|---|---|---|---|---|
| mirror (direct) | 0.320 | — | — | — | **+0.049** | **+0.041** |
| alakazam5 | 0.253 | 0.797 | 0.799 | 0.822 | +0.002 | +0.025 |
| archaludon | 0.080 | 0.714 | 0.707 | 0.755 | −0.007 | +0.041 |
| crustle_v1 | 0.080 | 0.764 | 0.811 | 0.803 | +0.047 | +0.039 |
| garchomp | 0.067 | 0.641 | 0.714 | 0.688 | +0.073 | +0.047 |
| v10 | 0.053 | 0.636 | 0.591 | 0.621 | −0.045 | −0.015 |
| dragapult | 0.053 | 0.812 | 0.805 | 0.825 | −0.007 | +0.013 |
| **WEIGHTED** | | | | | **+0.0215** | 🔴 **+0.0289** |

Resolution ±0.0115 (game sampling ⊕ §8ay's ±0.003 weight uncertainty):
**seedswap 1.9× outside, positive on 4/7; ens2 2.5× outside, positive on 6/7.**
⇒ **ens2 is the candidate** — larger, far more consistent, and it beats the
seedswap head-to-head. **The largest confirmed gain since the v4 state block.**

⚠ **`bc:garchomp` read 0.641 where §8ap recorded 0.857.** Its archived
fingerprint is `#a25b904d` — the stale width-496 `lw2` singleton, not v5. All
three arms met that identical build back-to-back, so the **deltas are unaffected**
(§8an's argument); the **level** has moved a long way and nobody should quote the
absolute row until it is explained. Possible sixth instance of anchor drift.

⚠ **Both candidates are worse against `rule:v10`** — the one archetype the corpus
contains **zero** games of (§8au). Coherent, not noise.

### 🔴 THE CONFIGURATION TRAP, found while building the bundle

Arena `bc` defaults to `chip_targeting`/`energy_spread`/`counter_source` **ON**;
the submission ships `--no-rules`, all three **OFF** (§8f: 0.427 with a v3 net).
**Every cell above ran rules-ON, both arms alike** — so the ΔW is internally
valid but describes a rules-on pair, while the bundle is rules-off. ⇒ **the
shipped configuration was never the measured one**, and this likely applies to
earlier verdicts too. Re-measured in the shipped configuration:
`p62_ship_config.jsonl`.

```powershell
python -X utf8 scripts/p60_ensemble.py --matches 1000 --arms C,A,B
python -X utf8 scripts/p61_ens_anchors.py --matches 750
```

## 8bd. 🔴 E3's NEAR-TIE BAND IS INDIFFERENT — and measuring it RETRACTS §8am's cliff, which was a property of the deviation's DEPTH, not its RATE (2026-08-07, day 23)

**Pre-registered** in `docs/experiments/beyond-bc/E3b-near-tie-gate.md`, frozen in
commit `675d09c` **before any arm ran**, including the two predictions below that
turned out wrong.

**Why it exists.** E3 (uncertainty-gated DAgger) is parked at its teacher gate: no
qualified reviewer, and the planner is disqualified as an automatic teacher
(§8bb). But E3's *premise* — that decisions where the clone's selected/unselected
boundary is nearly tied are worth relabelling — needs no teacher to test. **Take
the other side of the boundary and play the games.**

### The intervention, and why its geometry matters

`bc,flip<τ>` swaps the lowest-scored **selected** option for the highest-scored
**unselected** one whenever their logit gap is under τ, excluding bitwise-equivalent
pairs (§8x free ties). Since `choose` returns the top-k by logit, that is exactly
a swap of ranks *k* and *k+1*: 🔴 **the deviation is ONE RANK DEEP at every τ, by
construction.** τ moves only *how often* it fires. Hold that thought.

### Sizing first (rule 14), `p43 --dump-margins`, 115 ladder games, 19,573 decisions

| | |
|---|---|
| rankable candidates | **8,963** (77.9/game) |
| bitwise-equivalent free ties, excluded | 799 (4.1% of decisions; **8.2%** of those with a boundary) |
| median boundary margin | **1.479** logits |
| **the 160-item human review queue** | margins `[0.0001, 0.1316]` — **the bottom 10%** |

⇒ **The review queue is the extreme tail, not the band**, and free-by-construction
ties are far too few to be what makes near-ties cheap.

### The sweep — mirror, DIRECT, **the same weight file on both sides**, n=1,400/arm

| τ | decisions flipped | our score | 95% CI | W/D/L |
|---|---|---|---|---|
| **0** (control) | **0.0%** | **0.495** | [0.469, 0.521] | 693/0/707 |
| 0.10 | 7.0% | **0.494** | [0.467, 0.520] | 691/0/709 |
| **0.50** | **21.8%** | **0.487** | [0.461, 0.513] | 681/1/718 |
| 1.00 | 34.7% | **0.455** | [0.429, 0.481] | 636/1/763 |
| 2.00 | 51.2% | **0.356** | [0.332, 0.382] | 499/0/901 |

⚡ **This is the only experiment in this repo with no training-seed term.** §5.6's
"our A/Bs measured two networks, not one intervention" cannot apply: both arms
load `out/policy_v5.npz`, fingerprinted `#dc1c9acc` on both sides of every row.
The printed interval is the whole interval.
✅ **Harness control passed:** τ=0 fires 0 flips in 111,529 eligible decisions and
reads 0.495 — the pre-registered 0.500.

### 🔴 Result 1 — the band E3 targets is indifferent

The review queue's 160 items all sit at margin ≤ 0.1316, i.e. **strictly inside
the τ=0.10 band**. Flipping **every** decision in that band — 7.0% of all
decisions, an intervention far more aggressive than relabelling 160 of them —
measures **0.494 [0.467, 0.520]** against a ±0.026 resolution. At τ=0.50, which
covers 21.8% of decisions, still null: **0.487 [0.461, 0.513]**.

⇒ **No systematic re-ranking of the near-tie band pays.** Whatever E3 could
recover, it is not "the clone ranks these backwards".

⚠ **This does NOT kill E3, exactly as pre-registered.** The flip measures
**|E[effect]|**; a teacher's value is bounded by **E[|effect|]**, and §8am's own
reading is that an indifferent-on-average band is precisely where some choices are
better and some worse. 🔴 **The day-23 plan's claim that a null here "kills E3
without a reviewer" is therefore withdrawn — by the pre-registration that
predicted the null, not after seeing it.** What the null does establish is that
E3's entire value rests on case-by-case correctness, and that the *average*
movement available in its band is below what our best instrument resolves.

### 🔴 Result 2 — the pre-registration MISSED on two arms, and the misses are the finding

| arm | predicted | measured | verdict |
|---|---|---|---|
| τ=0 | 0.500 | 0.495 | ✅ met |
| τ=0.10 | null | 0.494 | ✅ met |
| τ=0.50 | null | 0.487 | ✅ met |
| **τ=1.00** | **≲0.40** | **0.455** | 🔴 **direction right, magnitude wrong** |
| **τ=2.00** | **≲0.20** | **0.356** | 🔴 **direction right, magnitude wrong** |

Both predictions came from §8am's temperature probe, matched on deviation rate.
Set the two instruments side by side at matched rate:

| deviation rate | **this probe** (one rank deep) | §8am (softmax, n=200) |
|---|---|---|
| ~21% | **0.487** [0.461, 0.513] | 0.520 [0.451, 0.588] |
| ~31–35% | **0.455** [0.429, 0.481] | **0.315** [0.255, 0.382] |
| ~44–51% | **0.356** [0.332, 0.382] | **0.055** [0.031, 0.096] |

**The bottom two rows are disjoint, the last one by a mile** — at roughly half of
all decisions deviated, sampling costs ≈ −494 Elo and one-rank flipping costs
≈ −103.

⇒ 🔴 **§8am's headline — "the first ~20% of deviations are free and the next 10%
cost ~150 Elo", a CLIFF in the deviation rate — is retracted as stated.** Raising
a softmax temperature raises the deviation **rate** and the deviation **depth**
together; §8am attributed the whole cost to the rate because it never varied them
separately. This probe pins depth at exactly one rank and finds **no cliff at
all** in margin units: 0.495 → 0.494 → 0.487 → 0.455 → 0.356, monotone 5/5, with
the incremental cost of each added band rising smoothly (−0.001, −0.007, −0.032,
−0.099) as the bands reach decisions the net is more confident about.

⚡ **Same shape as §8bb** (E5's "compute curve" that never scaled) and as §8ax
(the anchor that changed deck as well as pilot): **an effect attributed to the
variable that was named rather than the one that moved.** Third instance in three
sessions, and the first one caught by an experiment designed to isolate a
variable rather than by an audit afterwards.

⚠ **What this does NOT separate.** The flip differs from softmax in *two* ways:
depth (one rank vs unbounded) and targeting (only near-ties vs any decision).
Both push the same direction, so the comparison bounds their combined effect and
does not apportion it. Separating them is one more sweep (sample from the top-2
only, at matched rate) and is **not run**.

✅ **What survives of §8am:** its chosen τ=0.5 and B8's exploration budget are
untouched — a 20% deviation rate is confirmed free here too, at 7× the sample
size. What changes is the *reason*, and therefore what it licenses: the free band
is not "a fifth of selects are genuine near-ties", it is "moving one rank is cheap
wherever you do it, and moving further is what costs".

```powershell
python -X utf8 scripts/p43_dagger_queue.py --dump-margins out/logs/p43_margin_sizing.txt
python -X utf8 scripts/p59_e3_flip.py --matches 700 --taus 0,0.10,0.50,1.00,2.00
```
Archive `out/arena/p59_e3_flip.jsonl` (rows carry `run`; three invocations are
pooled in that file and **must be split on it** — the day-22 schema, load-bearing
on the very next experiment after it shipped). Logs `out/logs/p59_e3_*.txt`.

## 8bc. 🔴 THE REST OF THE VALIDATION FLOW, AUDITED: five more instrument defects, none of which ever touched a published number (2026-08-06, day 22)

**Why this exists.** §8ax and §8ay are two findings of one audit; this is the
other five, logged here because they were previously recorded only in `HANDOFF`
and the report chapter (§5.7) has to trace somewhere. **The audit was not
prompted by a wrong result.** Nothing looked broken. The question asked was the
one nobody had asked in twenty-two days: *what does the local validation flow do
that no one has ever checked?*

⇒ **The answer, and it is the finding rather than any single defect: every one
of the five lives in a part of the flow that produces no number a human reads.**
An instrument nobody quotes is an instrument nobody checks.

### Defect 1 — `arena.py elo` was numerically divergent for fifteen days

`fit_elo` took a **fixed `lr=8.0` step on an unnormalised batch gradient**. The
gradient sums over a player's games, so its curvature grows with *n* while the
step did not: past **~175 games per player the iteration is divergent** and
oscillates instead of converging.

| player | games | behaviour |
|---|---|---|
| `rule:crustle` | 1,320 | **8,586 Elo** swing between consecutive iterations |
| " | " | −3632 / +258 / +3397 / −3275 at iterations 499 / 500 / 501 / 502 |
| a 30-game anchor | 30 | still swung 200+ |

**Every rating it ever printed was an arbitrary sample of an oscillation**, and
which sample you got depended on the iteration cap. ✅ **Nothing published rests
on it** — every Elo figure in this file is a win-rate conversion — **which is
exactly why it survived: an unused instrument is never checked.** (Rule 9, one
level up: a metric that never prints is not a metric that passed.)

**Fix:** damped diagonal-Newton step (gradient over the summed per-game
curvature `p(1-p)·ln10/400`, damping 0.8), converging to `1e-4`; a 2-game prior
against a phantom at the anchor rating so an unbeaten player's rating cannot run
away; the anchor shift applied **once after** convergence rather than every pass.
✅ **Positive control:** it reproduces the bc-vs-crustle head-to-head **0.652 →
0.652**. `cmd_elo` now **refuses to print** a fit whose final step exceeds 0.5
Elo, and flags the **12 agents with no game path to the anchor** — their
*difference* is identified, their *level* is prior, not evidence.

### Defect 2 — a `net=` that failed the load guard silently played the singleton

`policynet.load` returns **None** rather than raising, and `PolicyAgent.__call__`
fell back to the module singleton — the old width-496 `policy_lw2`. So a net that
failed its own dimension guard was accepted by `build_agent`, **archived under
the requested net's name**, and would have played 496-wide lw2 against a 708-wide
control while printing an ordinary score.

Demonstrated deliberately, with a v7 net whose vocab map was one entry short —
§8aw's exact "stale map" hazard. ✅ **No past result is affected** — re-checked on
day 23 by loading every `*net*.npz` in the tree through the hardened loader:
**35 of 35 policy nets load**, and the single rejection is `sa/value_net.npz`, a
*value* net that `policynet.load` is right to refuse. Now an explicit `net=` that
does not load is a hard error.

### Defect 3 — the degradation counters were wired into the submission only

Day 15 built `bcagent.STATS` + `health_line()` to catch the silent index-order
fallback (§8g had to infer it indirectly from a 40.7% index-0 rate) and called it
*"the highest value-per-byte thing to log"* — then wired it into Kaggle's
`main.py` and **not into the arena**, the instrument day 17 called *"the ONLY
instrument"*. Worse: `p57` ran arena with `capture_output=True` and printed
stderr **only on non-zero exit**, so tracebacks from every *successful* run were
discarded — a run in which one arm fell back on every decision would have
returned a score and no complaint.

**Fix:** `arena.py play` prints `[health]` per invocation (counters zeroed per
run); p56/p57/p58 surface stderr on success and **hard-stop on DEGRADED**.

### Defect 4 — `bc` with no `net=` is an unversioned identity

**1,218 games in `out/arena/games.jsonl` under the bare name `bc`, spanning
2026-07-28 → 07-31** (rows where either seat is exactly `bc`), across which
`agents/sa/policy_net.npz` was a moving target. That is rule 19 — *an anchor is a
file* — one seat over: the same defect on **our own** side of the board.

**Fix:** agent names carry `#<md5-8 of the weight bytes>`, so a retrain that
reuses a path archives as a **new agent** instead of pooling into the old one.
The shipped v5 fingerprints `#dc1c9acc`, matching the bundle md5 already on
record.

### Defect 5 — archives append, and a re-run was invisible

`out/arena/p57_e8.jsonl` holds **3,000 games per v5c control cell against 1,500
per treatment cell**: the control was re-run for the v7pad pass into the same
file. ✅ **Published numbers are safe** — the drivers parse the printed score
line, not the archive — but anyone re-deriving E8 from that file gets a control
that was never the published one, and nothing in the file said so.

**Fix:** rows carry `run` (a per-invocation id, `SCHEMA = 2`) and `play`
announces out loud when the target file already holds that exact cell. ⚠ Rows
written before the bump have no `run` key and are one undifferentiated pool.

### What the five have in common

| defect | what it printed | who read it |
|---|---|---|
| divergent Elo fit | ratings | **nobody** — every published Elo is a win-rate conversion |
| silent net fallback | an ordinary score | drivers, which cannot tell |
| health counters | nothing (arena), stderr-on-failure (p57) | **nobody** |
| bare `bc` identity | a name | the archive, which pooled it |
| append-only archives | more rows | re-derivations, of which there had been none until §6.1's |

⇒ **Four of the five are invisible by construction, and the fifth is visible only
to a reader who was not there.** ✅ **No verdict in this repo changes** — every
published difference ran both arms back-to-back against one instrument, which is
the property that saved them, and it was adopted (§8ai's "a stored anchor score
is not a control") for an unrelated reason.

```powershell
python -X utf8 scripts/arena.py elo                 # refuses an unconverged fit
python -X utf8 scripts/arena.py play --help         # `[health]`, run ids, @deck
```

## 8aq. 🔴 AN ANCHOR CHANGED AFTER ITS LAST MEASUREMENT, AND EVERY DOC QUOTED THE OLD NUMBER — the shipped Crustle pilot is 0.755, not 0.866 (2026-08-02, day 18)

> 🔴 **CORRECTED BY §8ax (day 22). The 0.866 and the 0.755 were measured on
> DIFFERENT DECKS**, and the deck term is **+0.140** against the −0.111 this
> section attributes to the tie-break. The "which Pokémon it benches matters more
> than whether it benches" headline below is **retracted**; same-deck, the
> tie-break is +0.027 and the guard −0.038. ✅ The section's *method* finding —
> an instrument modified 26 minutes after its calibration, and rule 19 — stands
> and is if anything strengthened: the same audit missed a second axis of drift
> sitting in the same runs.

**How it surfaced.** `p34_matchup_liveness.py` (built today for the deck design)
prints each anchor's arena score beside the liveness table as a **cross-check** —
rule 18's "compute the headline a second way", installed deliberately because
this project has now shipped five analysis scripts that produced plausible wrong
numbers. On its first real use it read `rule:crustle` at **0.735 [0.670, 0.791]**
over 200 games against §8ap's published **0.866 [0.850, 0.880]**. Disjoint.

**The cause is not a bug in either script. The anchor is a different agent.**

| pilot | what changed | v5 scores | archive |
|---|---|---|---|
| **v1** | original import, no empty-bench guard | 0.7680 [0.749, 0.786] | `p20_v5_vs_crustle_v1` |
| **v2** | guard (flat 90000) + bench-anything default | 0.8700 [0.855, 0.884] | `p27_v5_vs_crustle_v2` |
| **v3** | guard (flat 90000), default restored | 0.8660 [0.850, 0.880] | `p28_v5_vs_crustle_v3` |
| 🔴 **v4** ⭐ **the one in the repo** | guard breaks ties toward Dwebble (`90000 + 5000`) | **0.7550** [0.735, 0.773] | `p35_v5_vs_crustle_v4` |

`83daa48` landed at **17:48:55**. The last game of the 0.866 run finished at
**17:22**. **The instrument was modified 26 minutes after its calibration and
three documents kept quoting the calibration.** The commit verified it on **six
recorded games** and said so honestly — but n=6 is rule 1, and nothing forced the
n=2,000 re-read.

### ⛔ RETRACTED (§8ax) — ~~the finding underneath it: the tie-break is worth more than the repair~~

⚠ **Everything in this subsection compares two cells that were run on different
decks.** It is kept as written because the reasoning was sound given what the
archive appeared to say, and because the way it failed is the point: the pilot
column below is correct and complete, and the confound is in a column that was
not printed at all. Same-deck, the tie-break is **+0.027** and the guard
**−0.038**.

v3 → v4 differ in **one term** — whether the empty-bench guard prefers Dwebble.
Everything else, including the guard itself, is identical. That tie-break is
worth **0.111** of win rate to the pilot (0.866 → 0.755), **larger than the whole
v1 → v3 empty-bench repair (+0.098) and in the opposite direction.**

⇒ **WHICH Pokémon a pilot benches on an empty bench matters more than WHETHER it
benches at all.** Under v3's flat 90000 every Pokémon scored identically, so the
choice fell to option order; Dwebble is the engine (it evolves into Crustle), and
picking it deliberately is the difference.

### ✅ Two published claims are corrected by this, and one survives

- 🔴 **§8an's Result 2 is no longer true of the shipped pilot.** It concluded
  *"the repaired pilot is a WORSE anchor than the broken one"* — at **0.755**, v4
  is the **best-resolving Crustle we have ever had**, better than the broken v1
  (0.768) *and* it keeps the guard, so it is not throwing games. **The repair did
  not have to cost resolution; the flat 90000 did.**
- 🔴 **§8an's live hypothesis is answered, in the direction it guessed.** It
  proposed that `90000` *"dominates every other option and hijacks the turn"* and
  that a smaller value would help — ⛔ correctly declining to tune it. The fix
  that shipped kept 90000 and added a tie-break, and that was enough.
- ✅ **§8ap's headline SURVIVES.** *"40.7% of the weighted verdict sits on anchors
  above 0.75"* is unchanged: 0.755 is still above 0.75. Its **Crustle row**
  ("⛔ near ceiling") is wrong; its conclusion is not.

### ⚠ What this does NOT say

**No verdict in this repo is retracted.** Every net-vs-net comparison carrying a
Crustle term was measured with *both* nets against the *same* pilot version, and
§8an established the pilot shift is a **level** shift that cancels in
differences. This is a stale *anchor table*, not a stale *verdict*.

### 🔧 The methods lesson, and it is a new one rather than a fifth instance

The five recent errors (§8ad, §8ae, §8af, §8ah, §8an's seat bug) were all **a
buggy script producing a plausible number**. This one is different: **both
scripts were correct and the world changed between them.** An anchor is a *file*,
and a number quoted from it is only valid for the version that produced it.

⇒ **RULE 19** (HANDOFF §2): **before quoting an anchor's score, check that its
source file is older than the archive you are quoting.** One command:

```powershell
git log -1 --format='%cd' -- agents/agentkit/rulebased/sources/<pilot>.py
python -X utf8 -c "import json;rows=[json.loads(l) for l in open('out/arena/<run>.jsonl')];import time;print(time.ctime(rows[-1]['ts']))"
```

✅ **Run over all seven anchors today: Crustle is the only one that drifted.**
Every other pilot and deck file predates its last A/B.

```powershell
python -X utf8 scripts/arena.py play "bc:v5,net=out/policy_v5.npz" "rule:crustle" `
    --deck-a grimmsnarl --deck-b crustle_v1 --matches 1000 `
    --archive out/arena/p35_v5_vs_crustle_v4.jsonl
```

## 8ap. 🔴 BOTH MISSING ANCHORS ARE CLOSED — and measuring them found that our anchor set's INFORMATIVENESS runs INVERSELY to its REPRESENTATIVENESS (2026-08-02, day 17)

**The gap.** §8ac named two archetypes that together outrank Crustle + Mega
Lucario and had **no anchor at all**: Cynthia's Garchomp ex **6.7%** and
Dragapult ex **5.3%**. Both are now closed, and neither cost what was expected.

- ⚡ **`rule:dragapult` ALREADY EXISTED.** Wired to `dragapult_ex` in
  `DECK_MODULE`, importable, functional — and **never once used**: no arena
  archive, no `EVIDENCE` reference, mentioned in exactly one `arena.py` docstring
  line. It was audited (`p24`) over 6 recorded games: **EXPOSED 0.000, 0
  empty-bench losses**. A working anchor sat unused for nine days.
- ✅ **Garchomp was BUILT** (`decks/cynthia_garchomp.py`) from the consensus 60
  in `out/meta/pre_shift_0722_0724.txt` (that exact list seen **159×**). §8af's
  exposure filter was run **first**: **all 20 distinct card ids are in the
  corpus, 0 of 60 copies untrained**, so our own net can pilot it and no rule
  pilot was needed — ROADMAP's *"hold the pilot constant and vary the 60"*.

### The complete anchor table for v5, sorted by how hard the anchor is

| anchor | v5 score, n=2,000 | §8ac field share | resolution |
|---|---|---|---|
| **mirror** | **~0.500** | **33.3%** (51.1% above rating 900) | ⚡ **best** |
| `rule:v10` (Mega Lucario) | 0.569 [0.547, 0.591] | 4.0% — **0 of 47 above 900** | good |
| `rule:archaludon` | 0.671 [0.650, 0.691] | 8.0% — **0 of 47 above 900** | fair |
| `rule:crustle` (v1, broken) | 0.768 [0.749, 0.786] | 6.7% | poor |
| `rule:alakazam5` | 0.789 [0.771, 0.807] | 22.0% | poor |
| **`rule:dragapult`** ⭐ new | **0.809** [0.791, 0.826] | **5.3%** | poor |
| **`bc:garchomp`** ⭐ new | **0.857** [0.841, 0.872] | **6.7%** | ⛔ near ceiling |
| ~~`rule:crustle` (v3, guard)~~ 🔴 **superseded — see §8aq** | ~~0.866 [0.850, 0.880]~~ → the shipped pilot is **v4** at **0.755** [0.735, 0.773] | 6.7% | ~~⛔ near ceiling~~ → **fair** |

### 🔴 The finding, and it is not the one this work was started to get

**Sort the anchors by resolution and you have also sorted them by
UNrepresentativeness.** The two anchors where we sit closest to 0.5 — and can
therefore actually separate two of our nets — are `rule:v10` (**4.0%** of the
field) and `rule:archaludon` (**8.0%**), and §8ac measured **both at 0 of 47
games above opponent rating 900**: they model a band we have left. Every anchor
that represents the field we now play is one we beat **77–87%** of the time,
where a difference between two nets compresses toward the ceiling.

⇒ 🔴 **§8ac's re-weighting was correct and had a side effect nobody noticed: it
moved weight ONTO the anchors that cannot resolve a difference and OFF the ones
that can.** At §8ac's weights, **40.7%** of the weighted verdict now sits on
anchors scoring above 0.75.

⚡ **The mirror is the only anchor that is both, and it is carrying the set.**
33.3% of the field, 51.1% above rating 900, 71.4% above 1000 — and it sits at
0.500 where resolution is best. **It does nearly all the discriminating work in
every weighted table in this repo.** ✅ Which is also the retrospective
justification for measuring §8ao's B8 A/B in the mirror: it was the right
instrument, not merely the convenient one.

⛔ **Adding more representative anchors does NOT fix this**, and today is the
proof: both new anchors landed at 0.809 and 0.857, i.e. among the least
informative in the set. **More coverage bought honesty, not sensitivity.**

### ⚠ And the net-piloted anchor is weak in a SECOND way

`bc:garchomp` is our own net holding someone else's 60. It therefore measures
**deck × how well OUR net pilots that deck**, and there is no reason to think it
pilots Garchomp as well as it pilots the list it was tuned on. So **0.857 is an
upper bound on our true win rate against the archetype**, biased optimistic by an
unmeasured amount. ⇒ **The "point our net at their deck" recipe produces anchors
that are both uninformative and flattering.** Better than having no Garchomp
opponent; **not** a substitute for a tuned pilot, and not to be quoted as "we
beat Garchomp 86% of the time."

### 🔧 A correction, made in the session that made the error

Off a **6-game** smoke (dragapult won 2/6) this was written up mid-session as
*"dragapult is far more competitive than Crustle"* and *"the new anchors may be
materially better instruments."* **At n=2,000 dragapult reads 0.809** — the
second-least informative anchor in the set, and the opposite of the direction
claimed. **Rule 1 exists for exactly this and it was violated by the person who
maintains it.** The n=6 figure should never have been characterised at all.

## 8aw. 🔴 THE EMBEDDING DEFECTS WERE REAL, THE FIXES ARE CORRECT, AND BOTH ARE NULLS — plus 92% of the tables are free to delete (2026-08-06, day 21)

Full record: `docs/experiments/embeddings/E8-vocab-remap.md`. Nets
`out/policy_v7_s{0,1}.npz` (remap+UNK+pad) and `out/policy_v7pad_s{0,1}.npz`
(pad only) against the same-session control `out/policy_v5c_s{0,1}.npz`.
Driver `scripts/p57_e8_arena.py`; logs `out/emb/e8_v7.log`, `out/emb/e8_v7b.log`.

### The two defects, measured

1. **90% of every table ships untrained.** `slot_emb` 104/1300 rows ever got a
   gradient, `bag_emb` 134/1300, `card_emb` 135/1300, `atk_emb` **57/1600**.
   88,000 embedding parameters ship; ~6,880 (7.8%) were ever trained. The
   untrained rows are **not** inert: their norms (3.908–3.953) are
   indistinguishable from trained rows' (3.970–4.068), so a card the corpus
   never contained arrives as a confident arbitrary identity rather than as
   "unknown".
2. **Row 0 is overloaded across 25.5% of all slot lookups** — empty slot, out of
   range, no stadium, no effect, no `padding_idx`. The net drove
   `|slot_emb[0]|` to **2.337** against a 3.958 table mean (11th smallest of
   1,300) — it taught itself what `padding_idx=0` gives exactly and for free.

### The fix and the sizing gate that priced it first

`train_policy.py --vocab` collapses each table to row 0 = PAD (`padding_idx`),
row 1 = a shared UNK, rows 2.. = the seen ids, per-table. **88,000 → 6,960
parameters (−92.1%)**, every layer width unchanged (`state_in` 708 on both
arms). Verified firing: UNK hits 6/6 v10 Pokémon, 2/4 archaludon, **0/4
crustle, 0/6 mirror**.

🔴 **Rule 14, run before the arena: UNK can only bite on ~12% of the weighted
field** (v10 4.0% + archaludon 8.0%); everything else is ≥78% in vocabulary.
Recorded in advance so a positive out-of-vocab arm could not later be quoted as
a field-wide result.

### Result — both interventions are nulls

| arm | opponent | weight | v7 Δ | v7pad Δ |
|---|---|---|---|---|
| A | mirror, **direct** | 33.3% | 0.487 [0.470, 0.506] | 0.4875 [0.470, 0.506] |
| E | rule:alakazam5 | 22.0% | −0.009 | — |
| D | rule:archaludon | 8.0% | −0.028 | — |
| B | rule:crustle | 6.7% | −0.003 | −0.014 |
| C | rule:v10 | 4.0% | +0.021 | +0.015 |
| | **weighted** | | ~~−0.0099~~ **−0.0078** (74%) | **−0.0047** (44%) |

> 🔴 **−0.0099 was an arithmetic error, corrected day 22 (§8ay).** The mirror row
> is a **score**, not a Δ, and the rest of the column is Δs: 0.487 − 0.5 =
> **−0.0128**, which the archive confirms at 0.4872 pooled over 3,000 games.
> Recomputing this table with its own weights gives **−0.0078**; the published
> total implies a mirror Δ of −0.019 that appears nowhere. With §8ay's corrected
> field shares it is −0.0078 as well — the weight changes cancel here.
> ✅ **The verdict is unchanged**: still inside ±0.025, still an unresolved null
> with the point estimate on the wrong side.

n=1500 games/cell/seed, 2 seeds. Two-cell 95% resolution **±0.036 per seed,
±0.025 pooled**; arm A direct is √2× tighter.

⚡ **The one arm whose seeds agreed does not survive attribution.** v7's arm C
(+0.018 / +0.023) was the entire case for UNK. But **`v7pad` has no UNK row** —
it keeps all 1,300 rows including the untrained ones and changes only
`padding_idx` — and it scored **+0.034** on that same arm at seed 0, *higher*
than v7-with-UNK, then −0.005 at seed 1. ⇒ both readings of arm C were
single-seed artefacts and the gain cannot be attributed to the mechanism it was
built to test. The decomposition also separates nothing: v7 and v7pad pool to
the same 0.487 on the mirror, so the mirror cost is not the capacity cut.

### ⚡ What is NOT a null: capacity is now bounded from both directions

**92% of the embedding parameters — 11.5% of the whole net — can be deleted for
0.0018 of corpus fit on one seed and 0.0003 on the other**, with no anchor
moving outside noise. Read with §8w (8.2× the parameters bought **−43
decisions**), the same net has now been measured to be insensitive to capacity
*added* and capacity *removed*. **Nothing in this project has ever been
capacity-limited.**

### 🔴 The seed floor does not carry to anchors, and now not even to the mirror

The day-20 box already warned that §8z's ±0.019 floor is a mirror-direct number
(`rule:v10` control seeds read 0.616 / 0.571). E8 adds two instances where the
between-seed swing **exceeds what sampling can produce**:

| arm | seed 0 → seed 1 | swing | 95% sampling |
|---|---|---|---|
| v7 D (archaludon) | +0.018 → −0.073 | **0.091** | ±0.051 |
| v7pad A (**mirror, direct**) | 0.524 → 0.451 | **0.073** | ±0.036 |

⇒ the escape hatch that the *direct* mirror arm is trustworthy at 2 seeds is
now closed too. **Two seeds × 1,500 games under-resolves every anchor we own**,
and single-seed anchor readings elsewhere in this repo are worth less than their
printed intervals claim. ⚠ This does not retract §8z or §8aa — both ran at
n=2,000 with replicated seeds — but it does say their intervals were optimistic.

### Rule 15, third instance — and three errors caught inside this experiment

A fourth candidate defect (*"`mode="mean"` erases copy counts on 57.4% of
decisions"*) was **retracted before anything was built**: the bag flats keep
duplicates and `EmbeddingBag(mode="mean")` divides by bag LENGTH, so the pool is
a count-weighted average `Σ (count_c/n)·e_c` — multiplicity survives. The
`--bagsum` arm was dropped; `sum = n·mean` and `n` is already dense.

Also corrected in-flight: (i) the driver printed **one cell's** width for a
two-cell delta, understating resolution by 41%; (ii) a **dose-response reading
across four arms** was published from single-seed cells each inside its own
interval and destroyed by the next data point — *rule 1 applies to patterns
across arms, not only to individual arms*; (iii) the driver printed "SUPPORTS
the UNK mechanism" for a net with no UNK row.

### ✅ The repair is KEPT, not shelved — user decision 2026-08-06

The user's call, recorded verbatim in intent: *the embeddings should be fixed
regardless of their impact on Elo.* Shipping 88,000 parameters of which 92% are
untrained noise is indefensible on its own terms whatever the scoreboard says.
So `--vocab` is **permanent, supported machinery on `main`**, not an experiment
branch:

- `train_policy.py --vocab out/emb/vocab.json` (implies `--pad`) and
  `--pad` alone; census from `scripts/p53_emb_vocab.py`.
- The map travels inside the npz as `vocab_<table>`; `policynet.load` refuses
  any net whose row count ≠ `2 + len(vocab)`, so tables and map cannot drift.
- Guards added where raw ids are fed straight to the tables:
  `scripts/context_accuracy.py` and `scripts/p54_emb_ablate.py` both refuse a
  v7 net by name rather than mis-index it.
- Nets: `out/policy_v7_s{0,1}.npz`, `out/policy_v7pad_s{0,1}.npz`.

⚠ **What shipping it would cost, stated so the decision is not made by
accident.** v5 holds a settled ladder position; v7 measures ~~−0.0099~~
**−0.0078 weighted** (arithmetic corrected day 22, §8ay; the verdict is the same)
— *not* resolved as a loss (every arm's interval spans zero, and the one arm
outside it is the one where seed variance exceeds sampling variance), but the
point estimate is on the wrong side and the LB's 63.2-point floor cannot
adjudicate the difference. **The correctness gain is real and the strength gain
is measured to be zero**, so this is a judgement call, not an optimisation.
⇒ **Default: v5 keeps shipping.** To ship v7 instead, the honest framing is the
standing rule — *name the agent the submission would evict first*.

⚠ **Maintenance note the remap introduces:** the vocabulary is derived from a
specific corpus. Rebuild the corpus and the census changes, so a net's map is
only valid for the census it was trained under. It travels in the npz precisely
so this cannot go wrong silently, but a corpus change means retraining, not
remapping.

⇒ **`dist/submission.tar.gz` is unchanged** (v5, md5 `dc1c9acc5ead16e5`).

---

## 8av. 🔴 CARD ATTRIBUTES ARE A CLEAN NULL — the §8au diagnosis survives, this repair for it does not (2026-08-05, day 20)

> ⚠ **Renumbered 2026-08-06.** This section was filed as §8aq and §8aq was
> already taken (*"an anchor changed after its last measurement"*, day 18), as
> §8ap was by the E6 section below. Both originals are cross-referenced across
> HANDOFF and EVIDENCE, so the NEW sections moved: E6 §8ap → **§8au**, E7 §8aq →
> **§8av**. Letters in use through §8at; append from §8au.

Full record and pre-registration:
`docs/experiments/embeddings/E7-card-attributes.md`, written before any arena
number existed.

**The intervention.** `--attr` replaces identity-by-embedding-row with
identity-by-attribute: 276 state columns (energyType, weakness, ability,
resistance, weak-to-facing-type across 12 slots) plus a `cardType` one-hot and
two target flags on the option vector, all read from the card DB, which covers
**all 1,267 cards** and therefore describes cards the corpus never contained.
Corpus `artifacts/pds_v6`, 248,985 rows, with `dense`/`xdense`/`opt_dense[:,:37]`
verified byte-identical to `pds_v4` — the control trains on **identical rows**.

**Sized before building** (`p55_attr_sizing.py`, rule 14): `cardType` at
`opt_card` was the strongest thing found anywhere (7 distinct, modal 0.416,
H/Hmax 0.780). The gate also **killed `aceSpec`** (one value corpus-wide) and
**`pokemonType`/`evolutionType`** — the six stage/ex flags already encoded give
12 distinct signatures and **none maps to more than one value of either**.

**The pre-registered prediction was an ASYMMETRY:** gain against `rule:v10`
(0/6 of its Pokémon in vocabulary) must exceed gain against `rule:crustle`
(4/4), because a uniform gain would be a better feature block rather than
evidence for the out-of-vocabulary mechanism.

| arm | n | result |
|---|---|---|
| **A** mirror, direct, pooled 2 seeds | 600 | **0.510 [0.470, 0.550]** — includes 0.500 |
| **C** `rule:v10,noS`, pooled 2 seeds | 2,000/cell | **+0.005 [−0.017, +0.027]** — includes 0 |

🔴 **v6 does not promote, and per rule 4 the out-of-vocabulary story is
RETRACTED for this intervention.** It is not falsified either: the two seeds
disagreed in *sign* at both sample sizes (−0.030/+0.073 at n=300;
−0.014/+0.024 at n=2,000). What is established is that **card attributes as
implemented do not recover the identity channel** — not that identity is
unimportant. §8ap stands untouched.

⚡ **And the clone fit the corpus BETTER on both seeds** (`val_top1` 0.7190 vs
0.7163, 0.7152 vs 0.7139) **while buying no strength** — rule 3's fourth
independent confirmation.

### 🔴 A DESIGN ERROR, and it is the most reusable thing here

The screening arms B/C/D are a **difference of two independent cells**, so at
n=300 each delta carries a 95% half-width of **±0.080**. Every screened delta
was inside it. Those rows were **uninformative, not null** — and arm C's
sign-flipping 0.103 seed swing is exactly what that resolution predicts.
**Arm A's direct head-to-head is 2× tighter for the same number of games**
(±0.040 at n=600), which is what `p33_anchor_resolution.py`'s `direct` flag has
been saying about the mirror all along. Screen on the direct arm; take a
two-cell anchor delta to n≥2,000 or do not quote it.

### 🔧 A correction made inside the session that made the error

After **seed 0 alone** this was written up mid-session as *"the pre-registered
prediction is falsified… fails on direction."* **Seed 1 reversed the sign and
that characterisation was wrong.** Same rule-1 failure §8an records from an n=6
smoke, committed by the person applying the rule. One seed of a two-cell delta
at n=300 licenses nothing.

### ⚠ A seed-variance caution the next design must carry

Against `rule:v10` at n=2,000 the two **control** seeds read **0.616 and 0.571**
(spread 0.045) while the two treatment seeds read 0.602 and 0.595 (spread
0.007). §8z's ±0.019 seed floor was measured **mirror-direct**; nothing licenses
assuming it carries to a third-party anchor. Two seeds cannot characterise a
variance — this is a flag, not a number to quote.

### What it leaves

The corpus contains **zero** Lucario games. E7 tried to repair an unseen
archetype by re-encoding cards and it did not work. The untested and simpler
explanation remains: the fix for an archetype never observed is **training data
containing it**. That is a data question, not an embedding question. `--drop-a`
sub-attribution is deliberately **not run** — five retrains against a null block
is wasted compute.

## 8au. 🔴 THE IDENTITY CHANNEL IS WORTH A QUARTER OF THE WIN RATE — and against Mega Lucario it is ALREADY DEAD (2026-08-04, day 20)

Full record: `docs/experiments/embeddings/E6-identity-channel.md`. No retraining
anywhere in this section — every arm is the frozen `out/policy_v5.npz`
(sha256 `26c681c4845a7eb0…`, byte-identical to the `sa/policy_net.npz` inside
`dist/submission.tar.gz`) with **embedding rows permuted**.

**Why permutation and not zeroing.** Zeroing a table moves the input
distribution the downstream layers were trained against, so degradation from
"identity destroyed" cannot be separated from degradation from "activations off
their training scale". Permutation feeds the identical multiset of row vectors
and scrambles only the card→row assignment. Row 0 is never touched:
`slot_emb[0]` is the empty slot and training drove it to norm **2.337** against
a 3.95 table mean, so it encodes "nothing here", not a card.

**The vocabulary, per table** (`scripts/p53_emb_vocab.py`) — rows that ever
received a gradient, out of the rows we ship:

| table | rows | ever looked up | share |
|---|---|---|---|
| `slot_emb` | 1300 | 104 | 8.0% |
| `bag_emb` | 1300 | 134 | 10.3% |
| `card_emb` | 1300 | 135 | 10.4% |
| `atk_emb` | 1600 | **57** | **3.6%** |

**The gate.** Permuting all four tables, mirror, direct head-to-head:
**0.997 [0.981, 0.999]**, W299/D0/L1, n=300. Not a broken net — the permuted
policy still beats `random` at **0.867 [0.803, 0.912]**, and the `--mode copy`
round-trip control is tensor-identical, so the serialisation path adds nothing.

**The finding.** Holding our own 19 card ids fixed and scrambling only cards we
can see but do not own (`atk_emb` excluded — `opt_attack` carries only *our*
attacks, so permuting it is a self-inflicted wound):

| opponent | opp **Pokémon** in vocabulary | v5 | identity scrambled | Δ |
|---|---|---|---|---|
| `rule:crustle` | **4 / 4** | 0.838 [0.792, 0.876] | 0.587 [0.530, 0.641] | **−0.251**, disjoint |
| `rule:v10,noS` | **0 / 6** | 0.625 [0.569, 0.678] | 0.607 [0.550, 0.660] | **−0.018**, overlapping |

Scoping control: the same net scores **0.550 [0.493, 0.605]** in the mirror,
CI spanning 0.500, because both decks are our 19 and the permuted rows are
never looked up.

🔴 **Knowing which Pokémon the opponent has is worth roughly a quarter of the
win rate where we can do it. Against Mega Lucario we cannot do it at all** —
Makuhita, Hariyama, Lunatone, Solrock, Riolu and Mega Lucario ex are *all six*
out of vocabulary, so scrambling that matchup costs nothing. There was nothing
left to destroy. Both scrambled arms land at ~0.59–0.61 regardless of opponent,
which is what "no opponent read" plays like: generic-good Pokémon, winning on
raw deck strength alone.

### ⚠ What this does NOT license

**Not** a claim that a fix recovers the 0.251. Permutation measures *correct*
identity against *scrambled* identity; an unseen card is neither — it is a
fixed random vector that at least stays consistent within a game. 0.251 is an
upper bound on a repair, not an estimate of one.

**Not** an attribution of the Mega Lucario weakness to vocabulary alone. The
corpus contains **no Lucario games at all**, and "never trained on the matchup"
is a sufficient explanation by itself. The vocabulary gap and the data gap are
the same absence seen twice, and no ablation separates them — only an
intervention can (E7).

### 🔧 A methods note this section nearly got wrong

The first Crustle number came out **0.838** against a recorded **0.755** and
briefly looked like an improvement. It is not: `83daa48` changed the Crustle
anchor on 08-02, and the *same* net with the *same* flags reads **0.767 before
/ 0.867 after** (n=2,000 / 4,000) — **+0.100 of apparent gain from changing
nothing on our side**, reproducing §8an's `CRUSTLE_CALIB` pair to three
decimals. `arena.build_agent` archives anchors as `rule:<name>` with **no
version**, so all 49,320 Crustle games pool two different opponents under one
identity, and `arena.py elo` fits over the whole archive. **Every Δ above is
valid only because both arms ran back-to-back in one session against one build.
A stored anchor score is not a control.**

## 8ao. 🔴 B8 FAILS ITS PRE-REGISTERED BAR — the RL fine-tune is a CLEAN NULL, and the control arm is what makes it clean (2026-08-02, day 17)

**The pre-registration, written before any code** (ROADMAP §2.5 B8, HANDOFF day-17
box): *if the fine-tuned net does not beat its byte-identical control by a margin
whose CI excludes the seed-only null at n≥2000, B8 DIES and becomes a report
chapter.* The seed-only null is **0.482** (§8z), a deviation of 0.018, so the
excluded band is **[0.482, 0.518]** and the numeric bar — computed and recorded
**before the A/B ran** — is **≥ 0.541 at n=2000**.

### The two arms

| A/B | score | 95% CI | verdict |
|---|---|---|---|
| **`b8` (advantage) vs `b8_ctrl`** | **0.512** | **[0.491, 0.534]** | 🔴 **FAILS the 0.541 bar** |
| `b8_ctrl` vs `v5` | 0.480 | [0.458, 0.502] | ≈ the seed floor |

**0.512 is ≈ +8 Elo.** The CI contains 0.5 and overlaps the excluded band almost
entirely. **This is a null, not a loss.** ⇒ **B8 dies by its own criterion.**

### ⚡ The second arm is not decoration — it is what licenses the word "clean"

It was **not** in the pre-registration; it was added because a bare null is
ambiguous between *"the outcome signal added nothing"* and *"we degraded both
arms equally and compared two damaged nets."* The control — same init, same
frozen 82.8% of parameters, same rows, same epochs, **advantage weighting off** —
reads **0.480 against v5**, which is within a noise-width of the measured
seed-only null of 0.482. ⇒ **The fine-tuning procedure itself costs nothing
detectable. Both arms sit where v5 sits, and the advantage weighting added
nothing on top of that.**

⚠ **Seat balance, as a sanity check on the instrument:** the treatment reads
**P0 0.529 / P1 0.496**, the ~1.8% first-player edge measured this morning over
4,000 mirror games, showing up exactly where it should — in an A/B between two
near-identical policies. The arena is behaving.

### The configuration that died, stated precisely so the verdict is not overread

| knob | value |
|---|---|
| corpus | **4,000** self-play games at τ=0.5, 702,138 rows |
| anchor | `pds_v4`, 248,985 rows at weight 1.0 |
| AWR temperature β | 1.0 ⇒ a **2.7×** win/loss weight ratio |
| gate | `--margin-max 1.0` ⇒ 300,181 rows (42.8%) re-weighted |
| effective sample size | **91.3%** of RL rows |
| trainable parameters | **120,577 of 702,913 (17.2%)** — head only |
| epochs / lr / seed | 3 / 2e-4 / 0 |

🔴 **What is killed is THIS configuration, and saying so is not hedging — it is
the difference between a measured result and an overreach.** A deliberately
gentle reweighting of a fifth of the parameters, on 4,000 games, failing to move
a clone, is not evidence that outcome signals are worthless. §8am independently
established that the 20% band of genuine near-ties **exists** and has somewhere
for such a signal to go.

### 🔴 And the null is consistent with §8ae's OWN unpriced caveat

§8ae passed RL's sizing probe on the arithmetic that a 1% effect at a context
recurring 20×/game needs **960 games** against a 5.5M-game budget. **That number
priced ONE context in isolation**, and §8ae said so in its own caveat 2:
*"shared parameters — so contexts are not estimated independently and the
effective sample size is smaller, unmeasured."*

**4,000 games is 0.073% of the budget it priced.** The binding constraint was
**memory, not compute**: 951,123 rows already occupy ~2.4 GB on a 7.3 GB machine
and `Data` materialises the whole corpus. ⇒ **The null does not distinguish "the
method does not work" from "the method was not given enough data", and no
honest reading of 0.512 resolves that.** ⚠ **This is exactly the sizing probe's
own warning arriving on schedule — §8ae wrote "a sizing probe that fails to kill
is not evidence that a thing works", and it was right.**

⇒ **Authorised follow-up (user, day 17): ONE rerun at more data, identical
config, identical bar.** ⛔ **Not a β sweep. Not a gate sweep.** Running several
configurations and reporting the best is shopping, and this project has a rule
about screening on the wrong metric that cost it two sessions.

### 🔴 THE RERUN RAN AT 4× THE DATA AND THE ESTIMATE WENT DOWN — B8 IS CLOSED

**16,000 games** (`--keep-margin 1.0`, 1,211,887 rows of 2,830,848 decisions),
**everything else identical**: same init, same frozen 82.8%, same β=1.0, same
gate, same 3 epochs / 2e-4 / seed 0 / `--export-last`.

| corpus | treatment vs its control | control vs v5 |
|---|---|---|
| 4,000 games | 0.512 [0.491, 0.534] | 0.480 [0.458, 0.502] |
| **16,000 games** | **0.506 [0.484, 0.528]** | 0.491 [0.469, 0.513] |

**Both fail the 0.541 bar. Both controls sit at v5.** And the point estimate
moved **0.512 → 0.506**, i.e. *down*.

⚡ **The decision rule was written down BEFORE this reported** (`out/logs/b8_prereg.txt`,
committed while the arena was still running), precisely because *"run it bigger"*
is the most available excuse after any null:

> (a) 16k materially **above** 0.512 ⇒ data is moving it, the 40,000-game run is
> justified. (b) 16k **at or below** 0.512 ⇒ 4× moved nothing, 10× is not
> indicated, and the axis closes on the **method**, not the budget.

**Branch (b). ⛔ No 40,000-game run. B8 is closed and becomes a report chapter.**

### ⚡ And a parameter-level diagnostic says what the null MEANS

Taken before the A/B reported, so it could not be fitted to the answer:

| | total abs Δ over the trained head |
|---|---|
| treatment vs control | **455.6** |
| control vs v5 | 1349.1 |

**The advantage weighting moved the head by ~34% of the distance the fine-tune
itself moved it from v5.** ⇒ **The null is not "the signal never reached the
parameters". The parameters moved substantially and the win rate did not.**
That is a stronger negative than a bare 0.506.

⚠ **One dimension remains genuinely untested and it is named rather than
buried: β.** Both runs used β=1.0 (a 2.7× win/loss weight ratio). A sweep was
declined **by instruction and by rule** — reporting the best of several
configurations is exactly the shopping this project has a rule against — so
"a stronger reweighting might work" is **unfalsified, not refuted**. It goes in
the report as an open question, not as a defeat and not as a hope.

✅ **What IS established, on 20,000 self-play games across two corpus sizes:
advantage-weighting a clone on its own recorded outcomes, at the smallest honest
scale, does not beat the same fine-tune without the outcome signal.**

Archives: `out/arena/p32_b8big_vs_ctrl.jsonl`, `out/arena/p32_bigctrl_vs_v5.jsonl`.

```powershell
python -X utf8 scripts/arena.py play "bc:b8,net=out/policy_b8.npz,noChip,noSpread,noSrc" `
    "bc:b8ctrl,net=out/policy_b8_ctrl.npz,noChip,noSpread,noSrc" `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 `
    --archive out/arena/p29_b8_vs_ctrl.jsonl
```
Archives: `out/arena/p29_b8_vs_ctrl.jsonl`, `out/arena/p29_ctrl_vs_v5.jsonl`.

## 8an. 🔴 THE CRUSTLE RE-RUN: the repair made the anchor EASIER, the alarm is ANSWERED, and the fixed pilot is a WORSE INSTRUMENT than the broken one (2026-08-02, day 17)

**The standing item.** §8ah found `sources/crustle.py:338` scoring every Pokémon
but Dwebble at −5000 for a bench play with no empty-bench guard, so the pilot
played on an empty bench and lost to the first KO. It filed the consequence as
*"every verdict carrying a Crustle term is suspect"* and *"our arena reads 0.663
against a 57.1% real win rate — this is a mechanism for part of it"*, i.e. **the
expectation was that our numbers were OPTIMISTIC.** The re-run was left
unauthorised for a day and is authorised now.

> 🔴 **THE SENTENCE BELOW IS FALSE FOR HALF THIS SECTION'S OWN TABLE, and it is
> the load-bearing one. CORRECTED BY §8ax (day 22).** The v1 column ran on
> `crustle_v1` (`p10`/`p19`/`p20`); the v2 and v3 columns ran on `crustle`
> (`p27`/`p28`) — a **20-of-60-slot** different deck, worth **+0.140** measured
> with the pilot held fixed. The +0.087…+0.102 attributed here to the
> empty-bench guard is mostly that. Same-deck, the repair is worth **≈ −0.04**,
> which is the sign §8ah predicted and this section reports as a surprise.
> ✅ **Result 1 (the §8ah alarm is retired) survives** — it rests on differences
> cancelling, not on this attribution. **Result 2 and the mechanism narrowing
> below are about a term five times smaller than the one nobody controlled.**

**The instrument.** Identical agent specs, identical decks (`grimmsnarl` vs
`crustle_v1`), n=2,000 each, seats alternating. **Only the pilot's bench logic
differs** — the deck file was never touched, so old and new archives are
directly comparable.

| net | vs **broken** pilot (v1) | vs **repaired** pilot (v2) | Δ | turns |
|---|---|---|---|---|
| v3 | 0.7700 [0.751, 0.788] | **0.8565** [0.840, 0.871] | **+0.087** | 14.3 → 16.3 |
| v4 | 0.7885 [0.770, 0.806] | **0.8880** [0.873, 0.901] | **+0.100** | 14.2 → 15.7 |
| v5 | 0.7680 [0.749, 0.786] | **0.8700** [0.855, 0.884] | **+0.102** | 14.2 → 15.8 |

**All three disjoint, all three the same sign, and the expected sign was the
other one.**

### ✅ Result 1 — the alarm is answered, and nothing needs rewriting

**The shift is a LEVEL shift, not a differential one.** +0.087 / +0.100 / +0.102
across three nets is one number within noise of itself. Every weighted verdict
in this repo is a **difference between nets**, and a common offset on one
anchor cancels in a difference:

| verdict's Crustle term | broken | repaired | change |
|---|---|---|---|
| v4 − v3 | +0.0185 | +0.0315 | **+0.013** |
| v5 − v4 | −0.0205 | −0.0180 | **+0.003** |

At §8ac's weight for Crustle (**6.7%**) those move the weighted totals by
**+0.0009 and +0.0002**. ⇒ **§8ah's consequence is retired: the defect did not
bias any published verdict by an amount any of them turn on.** The re-run cost
~21 minutes, not the "hours" the item assumed.

### 🔴 Result 2 — and the repaired pilot is a WORSE anchor than the broken one

It now sits at **0.86–0.89 against us**. An anchor we beat 89% of the time has
almost no room left to separate two of our nets — the same compression the
ROADMAP warns about for an anchor that is too *strong* (the 0.911 Crispin
anchor), arriving from the other end. **Fixing the pilot made it a more correct
agent and a less informative instrument.**

### 🔴 The mechanism was hypothesised, TESTED, and the hypothesis was WRONG

**The user asked the right question — "I thought we were only making sure it
benches when the bench is empty"** — and the diff says otherwise. `b7869d2`
replaced one line with three rules, and only one was the authorised repair:

| change | authorised | effect |
|---|---|---|
| bench-full → −5000 | — | no behavioural change |
| **empty bench → 90000** | ✅ **the fix** | benches when the bench is empty |
| **non-Dwebble default −5000 → 12000** | 🔴 **no** | benches *anything, always* |

**The hypothesis written here first was that the third line did the damage:**
our list wins on Shadow Bullet snipe and Munkidori passive damage (§9), so
telling a Crustle pilot to fill its bench hands us targets and prize sources.
The line was narrowed to the guard alone (`p28`, pilot v3) and re-measured.

| net | **v1** broken (no guard) | **v2** guard + bench-anything | **v3** guard only | v1 → v3 |
|---|---|---|---|---|
| v3 | 0.7700 [0.751, 0.788] | 0.8565 [0.840, 0.871] | **0.8670** [0.851, 0.881] | **+0.097** |
| v4 | 0.7885 [0.770, 0.806] | 0.8880 [0.873, 0.901] | **0.8750** [0.860, 0.889] | **+0.087** |
| v5 | 0.7680 [0.749, 0.786] | 0.8700 [0.855, 0.884] | **0.8660** [0.850, 0.880] | **+0.098** |

🔴 **REFUTED, on all three nets.** The minimal repair reproduces the whole shift
(**+0.087 … +0.098**, mean +0.094). And the v2−v3 difference — the *entire*
contribution of the unauthorised default inversion — is **+0.011, −0.013,
−0.004**: it **flips sign between nets** and averages to −0.002. That is noise,
measured three times. **The empty-bench guard is the whole effect and the
bench-anything default did nothing.**

✅ **The net ORDERING is unchanged by any of it** — v4 first under both the
broken pilot (0.7885) and the shipped guard-only one (0.8750) — which is the
second, independent reason no published verdict moves.

⚠ **The narrowing was still right, for a reason that is not this measurement:**
a defect fix may restore an author's intent, it may not install a new strategy.
It simply was not what moved the number.

### ⚡ The live hypothesis, and it is about the guard's MAGNITUDE

> 🔴 **ANSWERED ON DAY 18, IN THE DIRECTION THIS SECTION GUESSED — and it
> reverses Result 2 above.** `83daa48` shipped a fourth pilot (guard keeps 90000
> but breaks ties toward Dwebble) **26 minutes after the 0.866 run finished**, and
> it was never measured above n=6. At n=2,000 it reads **0.755 [0.735, 0.773]**.
> **That single tie-break is worth 0.111 to the pilot — larger than the whole
> +0.098 empty-bench repair and in the opposite direction.** So *"the repaired
> pilot is a WORSE instrument than the broken one"* is **not true of the pilot we
> actually ship**: v4 resolves better than the broken v1 (0.768) *and* keeps the
> guard. **WHICH Pokémon it benches matters more than WHETHER it benches.**
> `EVIDENCE` §8aq.
>
> ⛔ **AND THE ANSWER WAS ITSELF WRONG — RETRACTED BY §8ax (day 22).** The 0.866
> and the 0.755 are two different **decks**, 20 of 60 slots apart, worth
> **+0.140**. Same-deck the tie-break is **+0.027** and the guard **−0.038**, so
> the last sentence above is reversed and this section's *original* Result 2 —
> the one the box was written to overturn — is closer to right than its
> correction was. ⚠ **Two successive corrections, each confident, on one anchor.**

`return 90000` **dominates every other option in the set** — Dwebble as a wanted
card is 25000 and nothing else comes close. So whenever the bench is empty the
pilot benches, **at the cost of whatever else it would have done that select**.
Over 6 recorded games (`out/replays/audit_crustle_v3`) the pilot met an empty
bench at **54 decision points, ~9 per game**, against `rule:alakazam5`'s 19.

⇒ **A guard that was meant to fire on the brink of a loss is firing nine times a
game and hijacking the turn each time.** A value just above Dwebble's 25000
would remove the catastrophic failure without overriding the pilot's own plan.
⛔ **Not done — that is anchor tuning, and it is not authorised by this section.**
✅ **What IS verified is that the narrowed pilot still fixes the original
defect**: over the same 6 games, **EXPOSED = 0.000 and declines = 0** (`p24`).
Its 2 of 4 empty-bench losses are games with no Pokémon in hand to bench, which
is losing rather than a defect — the over-corrected pilot showed the same 2/10.

⛔ **Rule 12, one level up: no version of this pilot has been validated as
"plays Crustle well", only as "does not instantly lose."** Do not read 0.888 as
"we beat Crustle 89% of the time" — 6.7% of the real field plays this archetype
and §8i measured **76.9% over 13 real ladder games** against the broken-pilot
arena's 0.770, which was the *accurate* pairing.

⇒ **Standing consequence: quote the v1 (broken-pilot) numbers for anything that
compares against the real field, and the v2 numbers only for net-vs-net
differences, where the level cancels.** A third option — retune the pilot until
it matches the 76.9% real win rate — is real deck-and-anchor work and is **not**
authorised by this section.

```powershell
python -X utf8 scripts/arena.py play "bc:v5,net=out/policy_v5.npz,noChip,noSpread,noSrc" `
    "rule:crustle" --deck-a grimmsnarl --deck-b crustle --matches 1000 `
    --archive out/arena/p27_v5_vs_crustle_v2.jsonl
```

### 🔧 A methods note: this section was nearly published with the seat bug §8ae documents

The first pass scored the archives with `winner == 0` as "agent A won" and
reported **v3 0.489, v4 0.510, v5 0.502** — three numbers hugging 0.5, from
which the draft concluded the repair "moved the Crustle term in both directions,
none of it significant". **`agent0`/`agent1` are seat-indexed and the arena
alternates seats every game**, so that computation averaged each net together
with its opponent, which is why every result landed near 0.5.

⚡ **What caught it was not a review — it was `arena.py`'s own printed summary
saying `score=0.888` where the ad-hoc script said 0.510.** §8ae described this
exact bug five days ago and it was committed again anyway, in a throwaway
analysis script rather than in the archived tooling. **The lesson is narrower
and more useful than "be careful": do not re-derive a statistic the tool already
prints.** Every arena run emits the seat-corrected score; reading the archive by
hand re-implements that and can only introduce error.

## 8am. ⚡ B8's EXPLORATION TEMPERATURE, SIZED BEFORE ANY TRAINING — and it finds a CLIFF at 20% (2026-08-02, day 17)

**Why this had to be measured first.** B8 fine-tunes on our own recorded
outcomes, and a policy playing argmax against a copy of itself produces
near-deterministic games with nothing to attribute the outcome to. So actions
are sampled from `softmax(logits/tau)` — and `tau` is a design parameter, which
under rule 14 gets **sized, not guessed**. Too cold and there is no exploration;
too hot and the trajectories describe a policy we do not ship, which §8u
measured is exactly how a net gets weaker (distance from the field's modal
policy cost 0 → −55 → −92 Elo, monotonically).

**The instrument** (`scripts/p26_selfplay_gen.py --probe`): the sampling agent
plays the **argmax** agent — same net, same weights, seats alternating — so the
only difference between the two sides is the temperature. 200 games per arm.

| tau | off-argmax selects | score vs argmax | 95% CI | ≈ Elo |
|---|---|---|---|---|
| 0.25 | 14.1% | 0.465 | [0.397, 0.534] | −24 (null) |
| **0.50** | **20.4%** | **0.520** | **[0.451, 0.588]** | **+14 (null)** |
| 1.00 | 30.5% | **0.315** | [0.255, 0.382] | **−135** |
| 2.00 | 44.0% | **0.055** | [0.031, 0.096] | **−494** |

### ⛔ NARROWED BY §8bd (day 23) — the finding below is the shape, and the shape belongs to a DIFFERENT variable

> 🔴 **The cliff is not a property of the deviation RATE.** Raising a softmax
> temperature raises how *often* the agent leaves the argmax **and how far down
> the ranking it goes**; this probe never separated them and credited the rate.
> §8bd deviates on the same decisions but always by **exactly one rank**, so τ
> controls the rate alone — and at ~51% of decisions deviated it scores
> **0.356 [0.332, 0.382]** where the row below, at 44%, reads **0.055**. Disjoint,
> and there is **no cliff at all** in the one-rank curve (0.495 → 0.494 → 0.487 →
> 0.455 → 0.356, monotone 5/5).
> ✅ **What survives:** τ=0.5 as B8's exploration setting, and "≈20% deviation is
> free" as a *measurement* (independently reproduced at 7× the sample size).
> ⛔ **What does not:** the sentence below beginning *"The net's selects are
> sharply stratified"*. The free band is not "a fifth of selects are genuine
> near-ties"; it is **"one rank is cheap wherever you move it, and depth is what
> costs"**. B8's headroom argument rested on the retracted reading — see §8bd.

**The first ~20% of deviations are FREE and the next 10% cost ~150 Elo.** From
tau 0.25 → 0.50, off-argmax rises 6.3 pp for **no measurable change**. From 0.50
→ 1.00 it rises 10.1 pp and the agent loses **0.520 → 0.315**. There is no
gradual degradation: the exploration budget has a **cliff**.

⇒ **The net's selects are sharply stratified. About a fifth of them are genuine
near-ties where the choice does not matter, and essentially every deviation
beyond that band is a real mistake.**

⚡ **Two things this connects to, and one of them re-prices B8.**

1. **It corroborates §8x from a completely different direction.** §8x bounded
   the encoding by counting *bitwise-identical* option pairs and found the ties
   that exist are "two copies of one card in one role, i.e. free choices". This
   probe finds ~20% of selects are free in a behavioural sense — randomise them
   and measure nothing. Two instruments, no shared machinery, same story.
2. 🔴 **It sizes B8's headroom, and the reading is CAUTIOUSLY GOOD rather than
   bad.** The 20% band is free *on average*, which is not the same as every
   decision in it being irrelevant — a band of indifferent-on-average choices is
   precisely where some are better and some worse, netting to zero. **That is
   the population an outcome signal exists to sort, and it is the only
   population where it can act**: outside the band, deviating loses. It is also
   an independent justification for `--margin-max`, which was designed from §8u
   before this was run.

⚠ **What it does not say.** That the 20% band contains *recoverable* Elo. It
says the band exists and that the signal has somewhere to go. **B4 had a correct
diagnosis too and died anyway** (§8v).

**tau = 0.50 is the pick**: the largest temperature whose CI still covers 0.5.

```powershell
python -X utf8 scripts/p26_selfplay_gen.py --probe --games 200 --taus 0.25,0.5,1.0,2.0
```
Log: `out/logs/p26_tau_probe.txt`.

## 8az. 🔴 E1 MULTI-TASK REPRESENTATION LEARNING IS A THREE-ARM NULL (2026-08-03, day 19)

> ⚠ **Renumbered from §8au before merging to `main` (day 22).** `main` used
> 8au/8av/8aw for E6/E7/E8 while this branch used them for E1/E2/E5, and the two
> files auto-merge **cleanly** into six sections under three numbers with no
> conflict marker. `main` carried 22 cross-references to those numbers and this
> branch carried none, so the branch renumbers: **8au→8az, 8av→8ba, 8aw→8bb.**
>
> ✅ **AND THE WHOLE E-PROGRAM IS UNAFFECTED BY `main`'s DAY-22 VALIDATION
> AUDIT — checked, not assumed.** Every beyond-BC arena run is either a
> **mirror direct head-to-head** (E1 ×3, E2 mirror, E5 ×4) or
> **vs `rule:alakazam5` on `alakazam5`, its own tuned deck** (E2 ×2). So:
> **no `rule:crustle` anywhere ⇒ no exposure to §8ax's deck confound**; **no
> weighted `W = Σ wᵢΔᵢ` anywhere ⇒ no exposure to §8ay's corrected field
> shares**; and `arena.py elo` was never used. All six branch checkpoints in
> `out/e1/results/` and `out/e2/` load under `main`'s hardened
> `policynet.load()`, so the strict `net=` guard rejects none of them.
> ⇒ **No experiment on this branch needs re-running.** Two needed their
> write-ups corrected (§8ba's Alakazam arm, §8bb's mechanism) and that is a
> documentation fix, not arena time.

**Hypothesis.** The v5 policy encoder might learn a stronger state
representation if it also predicted terminal outcome and the fraction of legal
options selected. Unlike B8, the policy imitation target was unchanged: these
were small auxiliary gradients on a shared encoder, each weighted at 0.1.

All four arms used the same 248,985-row corpus, seed 0, 12 epochs, exact v5
architecture, and final-epoch export. Auxiliary modules were initialized after
all policy modules, preserving exact seeded policy initialization. Each
treatment was screened against its seed-matched policy-only control in 1,000
paired Grimmsnarl matches:

| treatment | score | 95% CI | W/D/L |
|---|---:|---:|---:|
| outcome | **0.505** | [0.484, 0.527] | 1011/0/989 |
| count | **0.507** | [0.486, 0.529] | 1015/0/985 |
| outcome + count | **0.500** | [0.478, 0.522] | 1000/0/1000 |

**Verdict: all three are null.** No arm advances to the weighted anchors and v5
remains the frozen baseline. The combined arm improved final held-out top-1
from **0.7134 → 0.7199** and then produced exactly **1,000 wins / 1,000
losses**. That independently repeats §8z/§8aa's warning: a better supervised
diagnostic is not evidence of a stronger agent. The outcome head overfit after
its first epoch; the count head learned a low-error target, but neither gradient
produced measurable playing strength. This is narrower than B8's null: B8
tested outcome-signed policy optimization; E1 tested whether the same terminal
signal helps when used only as an auxiliary representation target. Both routes
are now negative, and E4 does not inherit a validated value representation.

Record: `docs/experiments/beyond-bc/E1-multitask.md`. Archives:
`out/arena/e1_outcome_vs_control_seed0.jsonl`,
`out/arena/e1_count_vs_control_seed0.jsonl`, and
`out/arena/e1_both_vs_control_seed0.jsonl`.

## 8ba. 🔴 E2 OBSERVABLE MATCHUP ADAPTERS FAIL THEIR MIRROR SCREEN — and the Alakazam arm is UNINFORMATIVE, not null (2026-08-03, day 19; corrected day 22)

**Hypothesis.** Hard-routing residual adapters on visible opponent Grimmsnarl
and Alakazam lines could improve those matchups without moving the frozen v5
clone when neither line is visible.

The router used only opponent active, bench, and discard card ids. Route audit
on the fresh rating-977 trajectories recovered **100%** of true Grimmsnarl and
Alakazam census games. Both arms warm-started from v5, froze the base, and
exported after three epochs; the control kept adapters present but forced off.

Held-out diagnostics: general-route agreement stayed exactly **0.7137**, mirror
rose **0.7300 → 0.7340**, overall top-1 rose **0.7201 → 0.7221**. Strength:

| screen | score | 95% CI |
|---|---:|---:|
| treatment vs control, grimmsnarl mirror, n=1,000 | **0.521** | [0.490, 0.552] |
| treatment vs `rule:alakazam5`, n=1,000 | **0.782** | [0.756, 0.807] |
| control vs `rule:alakazam5`, n=1,000 | **0.792** | [0.766, 0.816] |

**Verdict: the adapters fail their screen and are not promoted; v5 remains the
shipping baseline.** The mirror arm is a **direct** head-to-head and carries the
verdict on its own: 0.521 [0.490, 0.552] includes 0.5, so three epochs of
specialist residual bought nothing where the router fires most.

🔴 **But "null" was too strong for the Alakazam arm, and this is corrected on day
22.** Treatment 0.782 and control 0.792 are **two independent cells against a
third party**, so the delta's resolution is √2× a single cell's:
**Δ = −0.0100 against ±0.0359 at n=1,000/cell.** The observed gap is 3.6× inside
the interval. ⇒ **uninformative, not null** — the same error `main`'s §8aq made
and the day-21 E8 box names by number. Reading "1.0 pp worse than control" as
evidence of anything is reading noise. Resolving it would need n≈2,000/cell, and
it is **not worth buying**: the mirror arm already decides the promotion, and no
value of the Alakazam delta changes it.

The useful negative is architectural and survives intact: a correct observable
router that provably protects the general path (agreement unmoved at 0.7137) is
still not enough when the specialist residual cannot buy a clear arena delta.

Record: `docs/experiments/beyond-bc/E2-routing.md`. Archives:
`out/arena/e2_mirror.jsonl`,
`out/arena/e2_vs_alakazam5_treatment.jsonl`,
`out/arena/e2_vs_alakazam5_control.jsonl`.

## 8bb. 🔴 E5 PLANNING IS CLOSED — and the "scaling curve" never scaled: realized compute was FLAT across the three arms that opened the gate (2026-08-04, day 20; corrected day 22)

**Hypothesis.** The repaired B4 turn sequencer (`seq,reply`) might recover with
more hidden-state averaging under Round-2-scale budgets. B4 had lost at
`0.375 [0.311, 0.444]` with a 1.0 s cap; ROADMAP §3.5 named a scaling-curve
probe as the only affordable local answer.

Fixed axis, no retuning: `K=8`, reply on, only `M` and a proportional cap change.
Control is frozen v5 without sequencing. Each cell is n=200 grimmsnarl mirror.

| arm | M | cap | score | 95% CI | s/completed plan |
|---|---:|---:|---:|---:|---:|
| low | 4 | 1.0 s | **0.380** | [0.316, 0.449] | 0.319 |
| medium | 8 | 2.0 s | **0.420** | [0.354, 0.489] | 0.427 |
| high | 16 | 4.0 s | **0.515** | [0.446, 0.583] | 0.724 |
| confirm | 32 | 8.0 s | **0.230** | [0.177, 0.293] | 1.331 |

The first three point estimates were non-decreasing and realized work per plan
rose, so the continue gate opened one preregistered confirmation cell. Confirm
then failed both fail rules: score below high, and upper bound below 0.5.

**Verdict: E5 is closed.** Do not promote a sequencer configuration, do not
invent a fifth compute point, and do not distill planner labels. The low cell
matches B4's repaired loss against v5 (`0.380` vs the older `0.375`).

### 🔴 Corrected day 22: the independent variable did not move across low/medium/high

Read back from the archives' own latency summaries, the **realized** cost per
decision was:

From `out/e5/manifest.json` and the archives' own latency summaries:

| arm | nominal cap | **total planning s** | mean ms/decision | plans/game | **firing rate** | overrule | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 1.0 s | **652** | 37 | 10.2 | **10.8%** | 58% | 0.380 |
| medium | 2.0 s | **616** | 36 | 7.2 | **7.4%** | 61% | 0.420 |
| high | 4.0 s | **606** | 32 | 4.2 | **4.2%** | 60% | 0.515 |
| confirm | 8.0 s | **8,288** | 471 | 31.1 | **35.0%** | 59% | 0.230 |

**The nominal budget went 1 → 2 → 4 s and total planning compute went
652 → 616 → 606 s — flat, slightly falling.** ⇒ **The three cells that opened the
pre-registered confirmation gate are three draws at essentially constant realized
compute**, n=200 each against a two-cell resolution of ±0.098. Pooled they are
**0.4383 [0.399, 0.478] over 600 games** — already a clean loss.

⇒ 🔴 **"Higher compute collapses the curve" attributes an effect to a variable
that did not vary.** This is `main`'s day-21 rule verbatim — *rule 1 applies to
patterns across arms, not only to single arms* — and it is the same shape as the
Crustle deck confound (§8ax on `main`): right conclusion, wrong cause.

### ⚡ What E5 actually measured: an ENGAGEMENT dose-response, monotone 4 for 4

The variable that genuinely moved is **how often the planner fires**, and
sorting the same four cells by it is monotone:

| firing rate | 4.2% | 7.4% | 10.8% | 35.0% |
|---|---:|---:|---:|---:|
| **score** | **0.515** | 0.420 | 0.380 | **0.230** |

**Every time the sequencer engages more, it plays worse — 4 arms out of 4, over
an 8× range**, while overruling the clone at a near-constant **58–61%** in every
regime. The arm that fires least is the only one indistinguishable from not
planning at all.

⚠ **Honest limit:** four cells at n=200, no repeat, adjacent pairs inside
±0.098. The **ordering** is 4/4 and the extremes (0.515 vs 0.230) are far
outside it; the adjacent steps individually are not resolved. This is a
direction, not a fitted slope.

⇒ This corroborates `main`'s §8w gate for RL from a new angle: the sequencer
reads the **same feature vectors** as the clone, so it cannot break a tie the
representation cannot express — it can only overrule a better-calibrated prior
with a worse one, and the damage scales with how often it does so.

✅ **All four cells were HEALTHY** — `errors: 0` and `budget_aborts: 0` in every
arm, including confirm, so the 0.230 is a real result and not a degraded agent.
⚠ **This was nearly recorded the other way.** A day-22 pass looked for the
sequencer counters in `out/logs/`, did not find them, and was about to file
confirm as unauditable — they are in `out/e5/manifest.json`, which is the better
place for them. **A counter you cannot find is one you will assume the worst
about**; the manifest is what made the defence possible.

⛔ **Do not re-run E5 to recover the curve.** The verdict is unchanged and now
rests on 600 pooled games rather than a three-point pattern; a fifth compute
point would be the same mistake with more budget.

Record: `docs/experiments/beyond-bc/E5-planning.md`. Archives:
`out/arena/e5_low_vs_control.jsonl`,
`out/arena/e5_medium_vs_control.jsonl`,
`out/arena/e5_high_vs_control.jsonl`,
`out/arena/e5_confirm_vs_control.jsonl`.

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
- ⚡ **FOUR variants are now measured (was one), and strength falls MONOTONICALLY
  with distance from the consensus 60** (§8aj, §8al). Against a same-deck control
  of **0.4980 [0.483, 0.513]**: 1 swap **0.4911** (null, p=0.54) · 2 swaps
  **0.4757** (**−17 Elo**, p=0.021) · 4 swaps **0.4637** (**−25 Elo**, p=0.0004).
  **Every deviation is at or below the floor and none is above it.**
- 🔴 **The consensus 60 behaves like a LOCAL OPTIMUM and our net is tuned to it.**
  With §8o (the deck is not the bottleneck) this answers Track C's
  experimentation half: **measured, and we kept the list.**
- ⚠ **All four A/Bs are MIRROR-ONLY.** That flatters variants which cut
  mirror-dead tech (Tool Scrapper is 0.00 plays/game there) and they lost anyway
  — but a variant aimed at a non-mirror matchup **cannot be judged by these runs
  at all.** Any future deck programme must be matchup-stratified from the start.
- ⛔ **Single-card guesses are retired as a method.** Three user-proposed edits
  produced one null and two significant losses; the next deck work needs a
  systematic search design, not another suggestion.
- ✅ **AND THAT SEARCH WAS THEN BUILT AND RUN — the entry above is closed**
  (§8ar, §8as, §8at). 11 candidates **frozen in a committed file before any
  variant deck existed**; two-stage, top-1-only promotion, so k variants could
  not manufacture a winner. **Stage 1: all 11 at or below the same-deck control**,
  six of eight mirror candidates losing significantly. **Stage 2: candidate G
  over all seven anchors, 57,600 games, ΔW = −0.0155** (§8ay's corrected weights;
  −0.0140 at the weights published on the day) against a design resolution of
  ±0.0059 — **negative on 7 of 7**. ⇒ **The kill line was not met, the search is
  over per its own pre-registration, and the consensus 60 stands.**
- ⚡ **What the design bought that a single A/B could not:** Ultra Ball held fixed
  across **six** different cut slots lost in all six (0.439–0.488), which
  separates "we cut six good slots" from "the added card is wrong"; and the cheap
  stage-1 screen predicted the expensive confirmation almost exactly (mirror
  **0.501** on 4,000 games → **0.500** on 15,800).
- 🔴 **§8af's exposure filter is NECESSARY BUT NOT SUFFICIENT, and this is the
  most reusable finding of the deck programme.** Ultra Ball sits at **5.59×** the
  training exposure of our weakest card and lost every slot; Energy Switch sits at
  3.61× and the net played it **1 time in 28 offers**. **Card-level exposure is
  not the binding constraint; card × DECK-CONTEXT is**, and nothing in this repo
  measures that.
- ⚠ **A five-card bundle is not attributable**: the community-list revision's
  −0.073 splits into **−0.040** for Xerosic ×2 alone and −0.033 for the other
  three (§8at). Bundled deck changes are no longer run. 🔴 And even the isolated
  −0.040 does **not** convict the card — "the card is wrong here" and "our clone
  misplays it" predict the same number, and the replays show the net firing
  Xerosic at opponent hand size 4 while nine offers at 7 went by.
- ✅ **The same-deck variance floor for deck A/Bs is 0.4980 [0.483, 0.513]** —
  the deck-side analogue of the seed-only null for nets. Any future decklist
  claim must clear it, and it sits essentially on 0.500, so the harness is
  unbiased.
- ⚠ **First player is worth ~1 point of win rate** (P0 0.510/0.513 vs P1
  0.486/0.470 across two 4,000-game arms). `arena.py` alternates seats; anything
  that does not is reading a ~2 pp bias as a result.
- 🔴 **Tool Scrapper is our thinnest slot on utilisation (0.13 plays/game) and is
  NOT cuttable on that evidence** — it is played **0.00 times per mirror game**
  because our list runs no tools, so a mirror test measures the matchup rather
  than the card. It needs a tool-running anchor, which §8ah's repair gates.

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
