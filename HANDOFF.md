# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** win the public LB. Top is **1179.6** (`flg`). Our best *scored*
submission reads **958.2**; a newer one (`55077709`, P6a) is pending and is the
first thing to check. Deadline **2026-08-16**, then ~2 weeks of continued play.
Kaggle CLI is authenticated, user has entered.

**Read §2 before trusting any number. §3 is the live plan.**
**This file must always end with a live plan, never a summary.**

---

## 1. Where we are (day 7, 2026-07-29 pm)

**Both targeting rules are shipped and live. P4b is our best submission.**

| submission | what | LB |
|---|---|---|
| `55077709` | **+ `counter_source` (P6a)** — submitted this session | **600.0 = the μ=600 start, not a score. Read it first.** |
| `55072063` | clone v2 + `chip_target` + `energy_spread` (P4b) — live | **958.2** ⏳ |
| `55054446` | clone v2 + `chip_target` — live | 916.8 → 936.0 → 979 → 901.6 → **905.2** |
| `55048039` | clone v2, no targeting | 752 → 758.6 (settled) |
| `55049206` | `rule:iono` sample agent | ~700–716 (settled) |

⏳ **958.2 is ONE reading — rule 2 is not satisfied yet.** Two readings ≥1 h
apart this session were identical, which is consistent with "not updated"
rather than "converged". Re-read before treating it as fact.

⚠ **And read `55054446` as a warning about rule 2.** Day 6 recorded it at
"916.8 → 936.0 → **979**, three readings, trending up" and reasoned from the
trend. It is now **905.2** — *below* its own first reading, ~74 points off the
value the plan was written against, and still moving (901.6 → 905.2 within one
hour). Three agreeing readings did not pin it down. **Treat a rising score as
unconverged, not as momentum**, and never let a plan depend on the third
decimal of an LB number that is still playing episodes.

The +184-point claim for `chip_target` survives this: 901.6 vs 758.6 for the
same clone without `targeting.py` is still +143, same sign, same conclusion.

That jump came from **one missing feature**, not from more training. `optfeat`
gives the net no HP and no damage, so it could not represent "this one dies to
30" and aimed chip damage at chance (25.7% lowest-HP picks). `agents/sa/targeting.py`
overrides those selects with a rule. Arena said `bc` vs `bc:noChip` = 0.577
[0.555, 0.599] n=2000 (+54 Elo) and `bc` vs `rule:v10,noS` = 0.418 → **0.537**
[0.506, 0.568] n=1000, flipping us past the public LB-950 agent. The LB agreed.

**This is the project's central method, confirmed end-to-end:**
> Find decisions the features cannot express, and write a rule for them.
> Three separate axes of *more training* bought nothing. One missing feature
> bought ~150 LB points.

**But day 7 sharpened it, and the sharpening is the important part.** Six rules
have now been A/B'd. The two that won both delete an option that is *provably
worthless*; the four that did nothing all pick a side in a *tradeoff* — and
every one of those four moved its audit rate exactly as designed first. It is
not enough for the net to be blind to a decision; the decision has to have a
right answer that arithmetic can prove. **Rule 11 in §2 is the test.**

### What the top of the board does

Nothing strong here is learned. `notebooks/` has three checked-in reference
agents: `strong-start-baseline-agent-v10-lb-950` (LB 950+, hand-written
deck-specific scoring, ~350 readable lines), `rule-based-not-psychic-alakazam-best-5th`
(**5th place, pure rules, no ML, no search**), and
`a-sample-archaludon-75-wr-vs-my-1300-starmie` (author reports 1300+;
matchup rules with grid-searched thresholds). The competition rewards **deck
expertise + matchup rules + damage arithmetic**.

---

## 2. How not to fool yourself

Every rule below was paid for. Rules 1, 2 and 8 have each invalidated real work.

1. **n=24 is noise.** A BC game costs ~0.17 s — n=1000 is 17 s of CPU. **Never
   accept an n<100 strength claim for anything cheap to measure.** ~2pp effects
   need n≈2000.
2. **One LB score is not a result.** `55049206` read 743.0 → 697.4 → 704.1.
   **Require two readings ≥1 h apart that agree.** Only the latest 2 submissions
   play episodes, so older scores are frozen, not converged.
3. **Validation metrics do not predict playing strength here — five times.**
   Value-net loss, policy top-1 ×3, `--winners-only`. **Judge every net in the
   arena, head-to-head**, never by val accuracy.
4. **Compare nets head-to-head, not through a third opponent.**
   `bc:<tag>,net=<path>` runs two nets in one process.
5. **A cross-deck arena score is mostly a DECK MATCHUP, not agent skill.**
   `rule:lucario` scores 0.781 vs `rule:iono`; the ~104-Elo-stronger
   `rule:v10,noS` scores 0.788 — indistinguishable. The pilot is invisible
   through that anchor. Head-to-head they are 0.646 [0.616, 0.675].
   **Measure skill in near-mirror matchups only.**
6. **CPU contention distorts wall-clock-budgeted agents** (`search:*`,
   `rule:v10` without `noS`). BC and `rule:*,noS` are untimed, so cross-run
   comparison is valid for them.
7. **This machine gives ~1.4 cores of real throughput** (Ryzen 5500U, 15 W).
   Run 2–3 jobs, not 4+.
8. **Frequency is not correctness, and per-turn binary audits hide
   multiplicity.** `munkidori_adrena_brain` read 99.4% per *turn* and 96.9% per
   *opportunity* — because with two Munkidori a turn offers two activations and
   `any()` scores one as 100%. **Count opportunities, not turns**
   (`MULTIPLICITY` in `opportunity_audit.py`).
9. **A metric that never prints is not a metric that passed.** `drag_target`
   was in the audit for days reading zero rows: it was keyed on `TO_ACTIVE`, but
   Boss's Orders drags through **`SWITCH`**. `TO_ACTIVE` is our own post-KO
   promotion, so every option was on our side and the opponent-only filter
   dropped them all, silently. **Check that each row has a non-zero
   denominator before believing the table.**
10. **Moving an audit rate is not winning games.** The P4a rules took the drag
    from 85/99 to 99/99 and the conversion turns from 36.9% to 100%, and then
    measured 0.489 and 0.493 in the arena. Rule 3 one level up, in the *rule*
    pipeline instead of the training one. **Arena-A/B every rule, no
    exceptions** — and prefer rules that delete a *dominated* option over rules
    that pick a side in a *tradeoff* (§3 P4a).
11. **Prefer rules that delete a DOMINATED option; distrust rules that pick a
    side in a TRADEOFF.** This is now the project's most reliable predictor,
    at **3 for 3 and 0 for 4**:

    | rule | class | arena |
    |---|---|---|
    | `chip_target` | dominated | **0.577** → +~150 LB |
    | `energy_spread` | dominated | **0.702** |
    | `counter_source` | dominated | **0.534** |
    | `drag_target` (aim the drag) | tradeoff | 0.489 |
    | `boss_converts` (force the play) | tradeoff | 0.493 |
    | `boss_veto` (suppress the play) | tradeoff | 0.493 |
    | `drag_target` high-HP tiebreak | tradeoff | 0.490 |

    The net has watched 2,810 games of humans making those trades and is
    already as good at them as our arithmetic. What it *cannot* do is see HP,
    damage or attached energy, so it loses to pure arithmetic every time the
    answer is arithmetic. **Before writing a rule, ask which column it is in.**
12. **The local arena is ONE opponent deck.** Everything routine is measured
    against `rule:v10,noS` on `lucario_v10`. A pattern the user watched in a
    real game can be genuinely absent locally without being absent on the LB.
    **When the user reports something from a live game and the local audit says
    it never happens, measure it on `replays/submission_replay_2026-07-29/`
    before closing it** — `scripts/p5a_replays.py` is the worked example, and it
    reads our real selects against 54 distinct LB opponents.
13. **Check the denominator is a real CHOICE, not just a real count.** P5a read
    "the rule takes the best target 26/26" — but 90 of its 95 pooled-KO rows
    offered only one prize value, so there was nothing to get wrong and the
    row could not fail. The honest denominator was **5**. Same disease as
    rule 8/9 one step further in: a rate over forced moves measures nothing.

---

## 3. THE PLAN (day 7)

**P4b shipped (`55072063`, 958.2). P6a shipped (`55077709`, pending). All of P5
is closed, and the whole Boss's Orders lever is closed four interventions
deep.** What is left is P1/P2.

### 0. Do this first: read `55077709`'s score

**P6a was submitted 2026-07-29 ~15:21 local. It validated COMPLETE and read
600.0 — that is the μ=600 every new submission starts at (§7), not a result.**
It climbs as it plays episodes; `55054446` went 916.8 → 936 → 979 → 905 over
two days. It is a bare `bc` with chip + spread + counter-source, all four
Boss's Orders rules off, built by:

```powershell
python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc --nets policy
```

Expect it to land near `55072063`'s 958.2 and, if the arena is telling the
truth, above it — the arena said +0.033 against `rule:v10,noS`. **Two readings
≥1 h apart (rule 2), and remember what `55054446` did**: it read 979 on day 6
and 905.2 today. If P6a comes in below 958.2, that is not automatically a
refutation at one reading; it is a reason to read it again.

### The board

| | item | state | arena |
|---|---|---|---|
| — | ship P4b | **DONE — `55072063` live at 958.2** | 0.702 n=4000 |
| **P6a** | **`counter_source` — Adrena-Brain's source pick** | **WON, default ON, SUBMITTED as `55077709`** | **0.534 n=2000; 0.626 vs `rule:v10`** |
| P6b | post-KO promotion (`TO_ACTIVE`) | sized: 9 misses in 120 games — **too small, closed** | — |
| P6c | how many counters to move (`..._COUNT`) | **closed — already 100% max** | — |
| P5a/b/c | the three live-game findings | **all three closed**, see below | — |
| P1 | re-rank decks against `rule:v10` | not started | ~20 min |
| P2 | MAIN-decision rules for the chosen deck | not started — **the real remaining mass** | days |
| P3 | abomasnow / Crustle lockdowns | not started, fold into P2 | hours |
| P4 | all three items | **closed**, see below and §6 | — |

Replays of the live agent vs real opponents are at
**`replays/submission_replay_2026-07-29/`** (user-supplied, 55 games, 54
distinct LB opponents — team name `Scio` is us, one game is the self-play
validation episode). These are the only games we have against the *actual* LB
field rather than our six local rule agents — use them for diagnosis, not
training (§6: more imitation data is dead). `scripts/p5a_replays.py` shows how
to read our own selects out of them.

---

### P6 — hunt the blind selects systematically

The two rules that won both patched the same defect: `optfeat` gives the net no
HP, no damage and no attached-energy count, so any select whose right answer is
that arithmetic is decided at chance. P5 looked for those by watching live
games. P6 enumerates them instead: **`scripts/p6_recon.py --matches 120`**
buckets every select the agent faces by (context, whose options they are) and
reports how many are a real decision. That table is the menu.

What it found, in descending size:

| select | share | verdict |
|---|---|---|
| `MAIN` | 47.7% | **P2.** The remaining mass, and it is not one decision. |
| `TO_HAND` ours | 15.3% | already at demonstrator parity (§6) |
| `DAMAGE_COUNTER` theirs | 5.6% | `chip_target` owns it |
| `ATTACH_FROM` ours | 5.5% | 2–4 options over 2 areas; the deck runs only {D}, so the cards are identical and the choice is *where from* — a resource tradeoff, so rule 11 says distrust. Unexamined. |
| `REMOVE_DAMAGE_COUNTER_COUNT` | 5.2% | **closed — already 100% max** |
| `TO_ACTIVE` ours | 3.9% | **closed — 91.2% right already** |
| `DAMAGE` theirs | 3.7% | `chip_target` owns it |
| `REMOVE_DAMAGE_COUNTER` ours | 2.9% | **P6a — the one live finding** |

#### P6a — `counter_source`: Adrena-Brain was moving 10 where 30 was available

Adrena-Brain moves "up to 3 damage counters" from one of our Pokemon to one of
theirs, and **the source is its own select** — `REMOVE_DAMAGE_COUNTER`, all
options ours, which is exactly the case `chip_target` declines by design. How
many counters then move is capped by what the source carries: the follow-up
`REMOVE_DAMAGE_COUNTER_COUNT` select offers "1,2,3" off a source with 3+ but
only "1,2" off a source with 2.

**The clone takes the maximum on that second select 100% of the time** (n=481),
so all of the loss is one select earlier, in the source pick — which is exactly
where the features go blind.

**Measured (120 games, 291 source selects with ≥2 options): in 59 of them
(20.3%) it picked a source that moves fewer counters than an available
alternative** — 10 or 20 damage where 30 was on the table.

`targeting.counter_source` (`bc:<label>,src`) takes that to **0**, and full
3-counter moves go from 67.1% to 76.5% of activations.

**arena: `bc:s,src` vs `bc` = 0.534 [0.513, 0.556], n=2000, grimmsnarl
mirror.** The interval clears 0.5, and it is balanced across seats (534/466 as
P0, 535/465 as P1), so it is not a seat artifact. Third dominated-option rule,
third win.

**Confirmed against an independent opponent (rule 4's spirit, and the §7
submission bar): `bc:s,src` vs `rule:v10,noS` = 0.626 [0.604, 0.647], n=2000**,
against **0.593 [0.562, 0.623] n=1000** for a bare `bc`. +0.033 there vs +0.034
in the mirror — two different measurements, same size, same sign.

**`counter_source` is therefore ON by default** in `PolicyAgent` and in
`arena.build_agent`, so a bare `bc` ships it. `bc:<label>,noSrc` turns it off.
It is the first rule to go on by default since `energy_spread`.

**Why this one is worth believing more than P5b was:** it is a *dominated*
option, not a tradeoff (rule 11). The heavily-damaged source is better in both
directions at once — it transfers more damage **and** it heals the Pokemon that
actually needed healing — so there is no trade to get wrong and no judgment of
the net's to override. Same shape as `energy_spread`, which scored 0.702.
Same minimalism too: it never changes *whether* counters move, only which of
our Pokemon they come off, and among sources that pay the full 3 it keeps the
net's own preference.

**Still: rule 10. The A/B decides, not the audit rate.**

---

### P5 — the three live-game findings, all three now closed

#### P5b — Boss's Orders: CLOSED NEGATIVE, and it closes the whole card

The veto was day 6's "best open lever". It is null, and so is the last
untested variant:

| intervention | arena, grimmsnarl mirror | verdict |
|---|---|---|
| `drag_target` — aim the drag | 0.489 [0.467, 0.511] n=2000 | null |
| `boss_converts` — force the play when it converts | 0.493 [0.471, 0.515] n=2000 | null |
| **`boss_veto` — suppress the play when it converts nothing** | **0.493 [0.471, 0.515] n=2000** | **null** |
| **`drag_target` high-HP tiebreak** | **0.490 [0.469, 0.512] n=2000** | **null** |
| both P4a rules together | 0.452 [0.435, 0.470] n=3000 | negative |

The veto was well-founded on paper — the user watched a dragged Pokemon get
evolved into their main attacker, and 32.4% of our plays had nothing KO-able on
their bench — and the mechanism engaged exactly as designed (`p5b_check.py`:
fires on 57.6% of the plays the net wants, and the fallback is safe — of the 50
vetoed plays that fell through to END, **50/50 had no attack available**, so it
never threw a turn away). It still won nothing.

The high-HP tiebreak likewise **decides 56.5% of drag selects** and moved the
score by 0.001.

**Four interventions, four nulls, on a card we play 38% of legal turns. Stop
writing Boss's Orders rules.** The read from rule 11: every one of them picks a
side in a trade the net has already seen thousands of humans make. All the code
is in `targeting.py` behind default-off flags — leave it there as the record.

#### P5a — pooled Adrena-Brain budget: CLOSED, and the instrument was broken

Day 6 sized this at "26 pooled-KO selects in 200 games, 26/26 correct". **Both
halves of that were wrong, and it still closes.**

1. **The budget was off by one activation.** `p5_audit` computed
   `left = max(1, len(armed - used))`, but the Munkidori being activated *right
   now* is already in `used` — the MAIN select that fires the ability precedes
   the DAMAGE_COUNTER select. So `armed - used` is what remains *afterwards*,
   and the pool is this activation **plus** those. It read `left == 1` on every
   row ever measured, including all 54 real-replay rows with two armed
   Munkidori, so **a 60-point pool was never once representable** — the audit
   was structurally incapable of detecting the exact scenario the user
   described. Fixed to `left = 1 + len(armed - used)`; the local pooled-KO
   denominator roughly tripled (26 → 89-95 per 200 games).
2. **The 26/26 was mostly forced moves** (rule 13). Of 95 pooled KOs, **90 had
   only one prize value among the candidates** — nothing to get wrong. The real
   denominator is ~5-7 per 200 games, and across three runs the rule missed
   2 of ~19.
3. **On the real replays it is rarer still.** `scripts/p5a_replays.py` over the
   55 live games: 266 chip selects, 6 pooled KOs, and **all 6 had a single
   prize value — not one real choice in 55 games against the actual field.**

So the mechanism the user described is real and the arithmetic is right, but it
is worth ~0.5 decisions per 200 games. Not a lever. **Do not rebuild it** — but
note the fixed budget term matters to anything else that reasons about
Adrena-Brain's per-turn output.

#### P5c — "never end a turn without attacking" — CLOSED, nothing to fix

Unchanged from day 6 and re-confirmed twice: the clone attacked on **3,683 of
3,683** turns where an END option and a payable ATTACK option were both on the
table. Detail below.

---

---

### What P4 settled (day 5, all closed)

- **P4b — spread {D} across two Munkidori: 0.702 [0.687, 0.715], n=4000.** The
  biggest effect this project has measured. Details below.
- **P4a — Boss's Orders forcing and aiming: closed negative.** Details below.
- **P4c — the audit counts opportunities, not turns.** Details below.

### P4b — Spread {D} across two Munkidori — DONE, +148 Elo

**The user's reading was right, and it is the biggest effect measured here yet.**
`targeting.energy_spread`, `bc:noSpread` turns it off.

**arena: `bc` vs `bc:noSpread` = 0.702 [0.687, 0.715], n=4000, grimmsnarl
mirror.** For comparison the +184-LB-point chip-targeting fix scored 0.577.

Four facts, all verified in-engine (`probe_adrena.py` pattern, 40 games with a
wrapper that greedily takes every Munkidori ability):

1. Adrena-Brain is **once per Pokemon**, not once per turn. We activated it
   twice in a turn 35 times; a slot that had used it was never re-offered.
2. The {D} condition is a **threshold, not a cost** — energy after use was
   unchanged 138 times out of 138.
3. **Munkidori is not a "Marnie's Pokemon"** (card 112 is plain `Munkidori`;
   the others are `Marnie's Impidimp/Morgrem/Grimmsnarl ex`). Punk Up cannot
   attach to it: in 40 games every attach option targeting a Munkidori came
   from the hand, i.e. the 1-per-turn manual attach. That is what makes the
   wasted attach expensive.
4. **A second {D} on a Munkidori is dead, full stop.** Munkidori's only attack
   is Mind Bend, cost {P}{C}, and this deck runs zero Psychic energy — so it
   cannot even be attack setup.

So: two Munkidori at 1 {D} each move 6 damage counters a turn (a 60-point swing,
since Adrena-Brain both heals us and damages them); one Munkidori at 2 {D}
moves 3. The clone chose the wasted attach **143 times to 94** — worse than a
coin flip, because `optfeat` gives it no attached-energy count. The rule takes
that to 0 and lifts Adrena-Brain activations from 1.26 to 1.60 per turn.

### P4a — Boss's Orders: CLOSED NEGATIVE. Do not redo.

**User observation:** *"we had the chance to play Boss's Orders to bring out a
weaker benched Pokémon to the active spot and knock it out with Shadow Bullet
but we didn't."* The observation was accurate; the fix was not.

| what | arena, grimmsnarl mirror | verdict |
|---|---|---|
| both rules | 0.452 [0.435, 0.470] n=3000 | **negative** |
| `drag_target` alone | 0.489 [0.467, 0.511] n=2000 | null |
| `boss_converts` alone | 0.493 [0.471, 0.515] n=2000 | null |

Two rules, both in `targeting.py`, both now default False and opt-in via
`bc:drag` / `bc:boss`:

- `drag_target` — rank the drag by (dies to our attack, prizes, lowest HP).
  Small lever: the clone already took the best available KO 85 times out of 99.
- `boss_converts` — **play** Boss's Orders when our attack would not KO the
  opponent's Active but would KO something on their bench. Big lever on paper:
  157 such turns in 300 games, and the clone played it on 36.9% of them (vs
  25.7% of all other legal turns — so it does discriminate, barely). The rule
  takes that to 100%.

**The lesson, and it is the important output of this item.** Both rules did
exactly what they were written to do — the drag went 85/99 → 99/99, the
conversion turns went 36.9% → 100% — and **neither bought a single game.**
Together they lost. So:

> **Rule 10: moving an audit rate is not winning games.** This is rule 3
> (val accuracy ≠ strength) reappearing one level up, in the *rule* pipeline
> rather than the training one. Every rule gets an arena A/B, no exceptions,
> and a rule that only measures well stays off.

**And the discriminator between P4b and P4a, which is what to steer by next:**
P4b overrides an option that is **provably worthless** — a second {D} on a
Munkidori does literally nothing, no judgment involved. P4a overrides a
**tradeoff** — Supporter for a prize, this KO versus that KO — and the clone's
implicit judgment there was already as good as our arithmetic. Prefer rules
that delete a dominated option. Be suspicious of rules that pick a side in a
trade; the net has seen 2,810 games of humans making that trade.

Note the pair being worse than either alone (0.452 vs ~0.49) is only marginally
outside the intervals, so read it as "no evidence of benefit, some evidence of
harm" rather than as a precise interaction estimate.

The isolation runs were launched under the older opt-out defaults; the repro
commands today are `bc:drag` vs `bc:base` and `bc:boss` vs `bc:base`.

### P4c — Count opportunities, not turns — DONE

**User observation:** *"I think we are not using Adrena-Brain at every chance."*
The instrument was wrong, as suspected — but the corrected number is small.

`opportunity_audit.py` now declares a `MULTIPLICITY` per line and prints an
`opps` column beside `turns`. `munkidori_adrena_brain` reads **99.4% per turn
but 96.9% per opportunity** (452 opportunities over 359 turns, 150 games). Real,
but a ~3% miss — the activation itself was never the lever. The lever was
upstream, in P4b: getting a second Munkidori armed at all.

Only `munkidori_adrena_brain` is a `"count"` line today, because it is the only
one whose copies are countable **on both sides** (one ABILITY option per
Munkidori, live and in the shards). Items are repeatable too, but a Rare Candy
option carries no target, so counting options would invent a denominator. The
docstring explains this; do not widen `"count"` without a real target count.

The audit also gained a live-only allocation metric (bare vs loaded Munkidori)
and its `drag_target` row now works at all — see §2 rule 9.

**Still open from this item:** the P2b parity verdicts were only re-derived for
`munkidori_adrena_brain`. The others are once-per-turn lines and so unaffected,
but the demonstrator-corpus side of the new `opps` column has not been run
(`--corpus artifacts/pds_v2`); `artifacts/` is gitignored and may need the
rebuild in §5.

### P1 — Re-rank decks against `rule:v10` (cheap, still unstarted)

The old sweep ranked decks by how well our clone beat `rule:iono`, which rule 5
kills. `mega_lucario_ex` came **last** there yet is the deck the LB-950 agent
plays. `scripts/deck_sweep.ps1` now defaults to `rule:v10,noS` / `lucario_v10`
and covers all 7 decks — run with no arguments (§7 PowerShell gotcha).
~20 min. It answers "what should P2 be written for", **not** "which deck is
strongest".

⚠ **Read the result with a thumb on the scale for grimmsnarl.** `chip_target`
and `energy_spread` are both grimmsnarl-specific (Shadow Bullet's snipe,
Munkidori's Adrena-Brain), so the sweep now measures "our agent + two
grimmsnarl-only rules" against decks that get neither. That is the right
question if you are asking *what to ship*; it is the wrong question if you are
asking *which deck has the higher ceiling*. For the latter, re-run with
`bc:plain,noChip,noSpread`.

### P2 — Write a real agent for one deck — **now the main line**

Where the leaderboard is, and after P5/P6 it is what is left. Explicit attack
planning (attacker/target/attack index with weakness and prize arithmetic),
per-option scoring, matchup branches. No search (§6). **Bar: 0.5 against
`rule:v10,noS` in a near-mirror, n≥500.** `rule:v10` is a working readable
template and is already in-process. The clone stays as fallback for every
select the rules do not cover — a hybrid starts strictly above whichever
component is better per decision class.

`scripts/context_accuracy.py` says **MAIN holds 3,930 of the net's 6,424 misses**
(18,924 rows, 33.9% miss), and `p6_recon` says MAIN is **47.7% of all selects
with ≥2 options**. Every other bucket is now either owned by a rule, measured
at parity, or measured too small — see the P6 table. **MAIN is the remaining
mass and it is what P2 is for.**

⚠ **Carry rule 11 into it.** MAIN is mostly tradeoffs (which Supporter, spend
the attach now or later, evolve or develop), which is precisely the class where
four straight rules did nothing. The parts of MAIN that are *arithmetic* — "can
I KO the Active this turn, and with which attacker" — are where P2's edge
should come from, and they are the parts `rule:v10` is explicitly built around.
Do not expect a general MAIN scorer to beat the clone; expect a lethal-detector
to.

### P3 — The abomasnow hole (open)

0.360 vs 0.475–0.519 elsewhere (pre-P2c, re-measure), and our selects/turn
collapse from 12.5–16.6 to **8.6** with shorter games — a lockdown, not subtle
misplay. Replay a loss with `SA_DEBUG=1` and read the actual select options.

**Related and untested: Crustle.** `Mysterious Rock Inn` (card 345) prevents all
damage from opponent {ex} attacks, and Grimmsnarl ex is `ex=True`, so **it deals
literally zero to Crustle** — attacking into it wastes every turn. **There is no
Crustle deck in the repo**, so `attack_into_ex_immune_active` (already in the
audit) has never fired. The out: Adrena-Brain and Freezing Shroud *move/place
damage counters*, which is not "damage done by attacks", so they should bypass
the prevention — **verify in-engine.** `dashimaki360/beating-the-day-1-1-crustle-bot`
is a public notebook on this matchup; V10 hardcodes 344/345 as "the crustle wall".

---

## 4. What ships

`agents/sa/bcagent.py` `PolicyAgent` + `agents/sa/policy_net.npz`
(= `policy_lw2`, listwise, 2,810-game corpus, val top-1 0.6755) +
`agents/sa/targeting.py`. ~1 ms/move, uses 0.1 s of the 600 s pool.

### Code map (`agents/sa/`)

- `bcagent.py` — **what we ship.** `net_path` pins an npz; `chip_targeting`
  toggles the override (`bc:noChip` in the arena). Default True.
- `targeting.py` — **all the rule overrides.** Each has its own `PolicyAgent`
  flag and its own `bc:` arena switch, so any one can be A/B'd alone.
  **Every new rule belongs here.**

  | function | select | switch | arena |
  |---|---|---|---|
  | `chip_target` | DAMAGE / DAMAGE_COUNTER(_ANY) | `noChip` | 0.577, n=2000 → **+~150 LB** |
  | `energy_spread` | MAIN, {D} ATTACH onto a Munkidori | `noSpread` | **0.702, n=4000** |
  | `counter_source` | REMOVE_DAMAGE_COUNTER (ours) | `noSrc` | **0.534, n=2000** |
  | `drag_target` | SWITCH (Boss's Orders' drag) | `drag`, **off** | 0.489 n=2000 — null |
  | `drag_target(prefer_high_hp)` | ditto, KO-able tiebreak | `dragHi`, **off** | 0.490 n=2000 — null |
  | `boss_converts` | MAIN, plays Boss's Orders | `boss`, **off** | 0.493 n=2000 — null |
  | `boss_veto` | MAIN, suppresses Boss's Orders | `veto`, **off** | 0.493 n=2000 — null |

  Three shapes, and the shape predicts the result (rule 11):
  - **Replace the whole ranking** — `chip_target`, `drag_target`. Fire only
    when *every* option is an opponent's Pokemon.
  - **Redirect the net's own pick** — `energy_spread`, `counter_source`. Never
    create or suppress an action, only change its target. Both need
    `full_rank(net, obs)` … `counter_source` does, because MAIN and
    REMOVE_DAMAGE_COUNTER selects have `maxCount == 1`, so `choose()` returns a
    single index with no runner-up to fall back to.
  - **Force or suppress an action outright** — `boss_converts`, `boss_veto`.
    Both null. Both tradeoffs.
- `policynet.py` — numpy inference. `SA_PNET_PATH` env override; **dim guard**
  (stale net → `None` → fallback; never remove it).
- `features.py` (v2, DENSE_DIM=242, PER_SLOT=18) / `optfeat.py` — shared by
  trainer and inference. **Any npz trained pre-v2 fails the dim guard.**
  Adding an HP/damage feature here bumps `VERSION` and retrains every net —
  a serious candidate, but `targeting.py` has been the cheaper path so far.
- `evalfn.py` + `textdmg.py` — handcrafted eval / expected damage.
  `targeting.best_damage` wraps `textdmg.estimate` with weakness and energy
  payability and is what every damage-vs-HP rule should call. Approximate in
  general, **exact for this deck** — every attack grimmsnarl can pay for is flat
  damage. Same object as V10's `evaluate_state`; read both together.
- `agent.py` (`SearchAgent`), `planner.py`, `timemgr.py` — search path, §6.
- `worlds.py`, `tracker.py`, `fastsearch.py`, `deck_library.json`.
- Both agents never raise: fallback = `list(range(minCount))`.

### The arena's real opponent: `rule:v10`

`scripts/import_v10_agent.py` lifts the LB-950 notebook into
`agents/agentkit/rulebased/sources/v10.py` plus `decks/lucario_v10.py` (its own
retuned 60 — *not* `decks/mega_lucario_ex.py`). Idempotent. Flags: `noS`
disables its MCTS, `tb<sec>` sets its budget — **in practice both are no-ops
because the MCTS never runs (§6)**; pass `noS` anyway so the archived name
records intent. `rule:v10x` makes the search reachable (still falls back).

---

## 5. Commands

```powershell
# LB / submission status
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(s.ref, s.date, s.status, s.public_score, '|', str(s.description)[:60]) for s in a.competition_submissions('pokemon-tcg-ai-battle')[:5]]"

# Skill measurement: near-mirror head-to-head (rule 5). The only kind that counts.
python -X utf8 scripts/arena.py play "rule:v10,noS" rule:lucario `
    --deck-a lucario_v10 --deck-b mega_lucario_ex --matches 500

# Against the real bar
python -X utf8 scripts/arena.py play bc "rule:v10,noS" `
    --deck-a grimmsnarl --deck-b lucario_v10 --matches 500

# A/B a rule override against the pure clone (how every targeting.py rule is judged).
# Off-switches: noChip, noSpread. Opt-in (default off): src, drag, dragHi, boss, veto.
# Isolate ONE rule per run: the P4a pair measured 0.452 while each alone was null.
# NOTE the first token after `bc:` is a LABEL, not a flag -- `bc:veto` is a plain
# `bc` named "veto" (§7). Write `bc:<label>,<flag>`.
python -X utf8 scripts/arena.py play "bc:s,src" bc `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000

# Net A/B, two nets in one process (~5 min, n=2000)
python -X utf8 scripts/arena.py play "bc:new,net=out/policy_X.npz" bc `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 --archive out/arena/ab_X.jsonl

powershell -File scripts/deck_sweep.ps1        # P1; no args (see gotcha)
python -X utf8 scripts/tally.py "<agent>" "out/arena/foo_*.jsonl"

# Audits — run these BEFORE writing any rule
python -X utf8 scripts/opportunity_audit.py --matches 100        # our games
python -X utf8 scripts/opportunity_audit.py --corpus artifacts/pds_v2   # demonstrators
python -X utf8 scripts/context_accuracy.py                       # per-context top-1
python -X utf8 scripts/p5_audit.py --matches 200   # sizes the three P5 findings
python -X utf8 scripts/p6_recon.py --matches 120   # EVERY select, bucketed -- the menu
python -X utf8 scripts/p5a_replays.py              # the same counters on 55 REAL games
python -X utf8 scripts/p5b_check.py --matches 150  # does a rule actually fire? (rule 9)

# Train (12 epochs; artifacts/pds_v2 is the shipped corpus)
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v2 --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 --out out/policy_X.npz

# Build + submit (smoke-tests the bundle the way Kaggle loads it)
python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc --nets policy
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_submit('dist/submission.tar.gz','msg','pokemon-tcg-ai-battle')"

# Import public notebook agents
python -X utf8 scripts/import_v10_agent.py     # rule:v10 + decks/lucario_v10
python -X utf8 scripts/import_rule_agents.py   # the four sample agents

# Find new public notebooks (this is how V10 was found — redo periodically)
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(k.ref,'|',k.title) for k in a.kernels_list(competition='pokemon-tcg-ai-battle',sort_by='voteCount',page_size=30)]"

# Rebuild data only (more data is NOT a lever — §6)
python -X utf8 scripts/fetch_top_episodes.py --date 2026-07-26 --max 400
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds/d26 replays/2026-07-26
```

### Data on disk

`replays/`: 07-17..07-22, 07-24, 07-26, 07-27 = 400 each; 07-23 = 175,
07-25 = 268, 07-16 = 115; 07-13/14/15 = 0. Plus 366 old-repo replays at
`E:\Kaggle\pokemon-tcg-simulation\replay_miner\replays\2026-07-06..12`.
**`replays/submission_replay_2026-07-29/` = our live 936 agent vs the real
field** (user-supplied; the only non-local opposition we have).

`artifacts/**` is gitignored. `artifacts/pds/` = 4,010 games (the *rejected*
lw3 corpus); `artifacts/pds_v2/` = 2,810 (the shipped v2 corpus) and exists
only on this disk. `pds_v2` is `pds` minus the three days that made lw3 worse:

```powershell
foreach ($d in @('old','d21','d22','d23','d24','d25','d26','d27')) {
  New-Item -ItemType Directory -Force "artifacts/pds_v2/$d" | Out-Null
  Copy-Item "artifacts/pds/$d/shard_*.npz" "artifacts/pds_v2/$d/" -Force
}
```

If `pds` itself is lost, rebuild shards from `replays/` with
`build_policy_dataset.py`.

---

## 6. Settled — do not redo

**The clone is plateaued. Three training axes are negative** (head-to-head,
n=2000, vs the previously shipped net):

| net | corpus | val top-1 | vs prev shipped |
|---|---|---|---|
| v1 (BCE) | 2,410 | 0.6596 | — |
| **v2 `policy_lw2`** | **2,810** | **0.6755** | **0.524 — SHIPPED** |
| `policy_lw3` (more data) | 4,010 | 0.6933 | 0.491 |
| `policy_win` (`--winners-only`) | 2,810 | 0.6410 | **0.375 — decisive** |

More data, more val accuracy, and winners-only all fail. `--winners-only` is
12pp *worse* — cloning the losing side is **helping**. Note lw3 has the best val
accuracy and lost: rule 3 in action. `--loss listwise` beats pointwise BCE and
reaches in 1 epoch what BCE took 4 to reach.

**Search is out, ours and the field's.** Ours: `search:M,noV,roll,mo,mc20,pb0.15`
vs `bc` = 0.323, n=31 — a terminal rollout returns 0/1, so a mean over 12
determinizations has SE ≈ 0.14 and the max over ~9 rivals sits ~0.21–0.28 above
truth by chance; it overruled the clone on 52% of decisions. More
determinizations and the value net were also negative. **V10's shallow MCTS has
never once executed** — two independent bugs (its candidate set comes from
`choose()` truncated to `select.maxCount`, which is 1 for every MAIN select
measured 70/70; and `search_begin(obs, your_deck=yd)` passes 1 of 7 required args
and raises `TypeError` into a bare `except`). Confirmed by timing: 200 games in
11.8 s. **So LB 950+ is 100% handcrafted policy, and nothing in this competition
has ever demonstrated search is worth anything.** Loose end if ever revisited:
`agents/sa/worlds.py`'s `World` is exactly the `search_begin` argument bundle.

**Ruled out by measurement:**

- **Decklist changes** — +2 Boss's Orders / -1 Tool Scrapper / -1 Spikemuth Gym
  scored 0.490 [0.468, 0.512], n=2000. We already play Boss's Orders on 38.2% of
  legal turns vs demonstrators' 31.4%; **Team Rocket's Petrel (4x) tutors any
  Trainer**, so access is already there. Spikemuth Gym is played ~100% by both
  sides — do not cut it. The list is an exact 60 seen 290× in one day's top
  episodes and the net is trained on it, so variants are off-distribution too.
- **TO_HAND duplicate-avoidance** — demonstrators fetch a duplicate 5.8%
  (n=57,053), we fetch one 5.8% (n=482). Already correct.
- **REMOVE_DAMAGE_COUNTER** — lowest lift on the board, but demonstrators are
  themselves inconsistent (Active 33.6%, max-prize 60.6%, ~2.8 options,
  n=9,911). **A low lift can mean a noisy label, not a blind feature.**
- **Self-play RL — dropped.** Days of work on 1.4 cores to maybe reach where
  hand-written rules already sit. Nothing at the top of this board is learned.
- **Boss's Orders, ALL FOUR interventions — the card is closed.** Aim the drag
  0.489, force the play 0.493, **suppress the play 0.493**, **high-HP drag
  tiebreak 0.490**, all n=2000; the P4a pair together 0.452 n=3000. Every one
  of them moved its audit rate exactly as designed (the drag 85/99 → 99/99, the
  conversion turns 36.9% → 100%, the veto firing on 57.6% of wanted plays with
  a verified-safe fallback, the tiebreak deciding 56.5% of drags) and not one
  won a game. See §3 P5b, rule 10 and rule 11. **Do not write a fifth.**
- **Adrena-Brain pooled budget (P5a)** — real mechanism, ~0.5 real decisions
  per 200 games, and zero in 55 games against the actual field. §3 P5a.
- **`REMOVE_DAMAGE_COUNTER_COUNT`** — the clone already moves the maximum
  offered **100%** of the time (n=481, 120 games). Nothing to fix; all of the
  loss was one select earlier, in the source pick (§3 P6a).
- **Post-KO promotion (`TO_ACTIVE`)** — promotes a Pokemon that cannot attack
  while an attacker was benched **9 times in 120 games** (91.2% right on the
  102 selects where it mattered). Too small, and a tradeoff besides.
- **"Never end a turn without attacking" (P5c)** — the clone already attacks on
  **100.0%** of the 3,659 turns where an END option and a payable ATTACK option
  were both on the table (200 games). Nothing to fix.

**Do not resurrect:** the arena→LB ladder anchored on `rule:iono`; the old deck
sweep's ranking; "the clone is comfortably above the rule baseline"; every n=24
number and every strength claim dated before 2026-07-27 pm (measured through
stale nets silently rejected by dim guards, a compute knob that could not bind,
and a mirror matchup compared against cross-deck runs); "3× compute made it
worse" (`SA_SPEND_MULT` only grants time, and time was never binding).

⚠ **Per §3c, the P2b "already at demonstrator parity" verdicts
(`munkidori_adrena_brain` 99.3%, `rare_candy_play` 82.0%,
`evolve_impidimp_to_morgrem` 91.6%, `dark_energy_to_munkidori` 78.3%) are only
valid for once-per-turn lines.** Re-derive them after the P4c instrument fix.

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
  ⚠ **If the machine sleeps mid-run, one game eats the whole nap** and
  `arena.py` prints `WOULD TIME OUT ON KAGGLE` off that single game. Check the
  distribution before believing it: in `ab_spread.jsonl` the worst pool was
  −3606.9 s and the *next* worst was 599.2 s, median 599.9 s, p99 latency 1.6 ms.
- **Submission:** `.tar.gz`, `main.py` + `deck.csv` at TOP level (+ `cg/`, `sa/`).
  Cap 197.7 MiB. 5/day, **latest 2 active**. New submissions start μ=600.
  Validation episode is self-play first — a crash there means Error.
  `kaggle competitions submit` may 400 despite a 100% upload; the Python client
  works, and that call **submits** — it is not a dry run.
- Kaggle Python API returns **snake_case** (`public_score`, `team_name`);
  `competition_leaderboard_view` paginates at 20 rows.
- **The first token after `bc:` is a LABEL, not a flag.** `bc:veto` silently
  builds a plain `bc` named "veto" — the flag parser starts at token 1, so the
  A/B compares the clone against itself. Write `bc:<label>,<flag>`
  (`bc:p5b,veto`). `arena.py` now raises on an unrecognised flag, which is what
  caught it, but the label slot itself still swallows anything.
- **PowerShell `-File script.ps1 -Days a,b,c` does not bind an array.** Edit the
  script default and launch with no args. **Never name a param `$Matches`** —
  collides with the automatic regex variable.
- Windows: `python -X utf8` everywhere. Run from repo root; `sys.path` needs
  `src/`, `agents/`, root. Launch long jobs with `Start-Process` (detached) and
  pass `-u` or python block-buffers redirected stdout.
- Some replays download truncated (exactly 3 MiB) and fail JSON parse; builders
  skip them (`errors=N`). Delete + re-fetch to recover.
- Old repo `E:\Kaggle\pokemon-tcg-simulation` = failed pure-RL attempts. Take its
  replays, not its approach.
- **Submission discipline:** submit only what has won head-to-head at n≥500
  against `rule:v10,noS`. Always `--nets`-pin the config. Rebuild rather than
  trusting an old tarball in `dist/`.
- Commit style: fine-grained, one-line semantic messages + Claude co-author
  trailer.
