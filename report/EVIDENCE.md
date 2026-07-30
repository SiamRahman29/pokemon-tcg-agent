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
`55072063` sat at 970.1.** See `HANDOFF.md` §3.0 — this is the live question.

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

### P4c — count opportunities, not turns

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
