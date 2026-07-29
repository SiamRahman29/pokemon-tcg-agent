# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** win the public LB. Top is **1179.8** (`Majkel1337`; `flg` is 2nd at
1173.7). Our best agent reads **970.1**. Deadline **2026-08-16**, then ~2 weeks
of continued play. Kaggle CLI is authenticated, user has entered.

**Read §2 before trusting any number. §3 is the live plan.**
**This file must always end with a live plan, never a summary.**

> ⛔ **DO NOT SUBMIT ANYTHING until §3.0 is resolved.** Only the **latest 2**
> submissions play episodes. The active pair is `55077709` (762.2, on trial)
> and `55072063` (**970.1, our best**). A third submission today **evicts
> `55072063` from active play** and leaves the unproven agent as our best live
> entry. There is no way to un-evict it except resubmitting and waiting out
> another climb from μ=600.

---

## 1. Where we are (day 7, 2026-07-29, end of session)

**P4b is our best agent at 970.1 and still rising. P6a shipped this session and
is reading 762.2 — unresolved, and §3.0 is about resolving it.**

| submission | what | LB |
|---|---|---|
| `55077709` | **+ `counter_source` (P6a)** — shipped this session | 600 → 762.2 → **746.4** ⚠ **falling** |
| `55072063` | clone v2 + `chip_target` + `energy_spread` (P4b) | 958.2 → **970.1** ✅ **our best** |
| `55054446` | clone v2 + `chip_target` | 916.8 → 936 → 979 → 901.6 → **905.2** (inactive) |
| `55048039` | clone v2, no targeting | 752 → 758.6 (settled) |
| `55049206` | `rule:iono` sample agent | ~700–716 (settled) |

**What those numbers mean.** A μ=600 start climbing is normal — `55072063`
needed ~4+ h to reach 958 — so a low *early* reading proves nothing on its own.
Two things make this one worrying anyway:

1. **It fell.** 762.2 at 10:22 UTC → **746.4 at 10:27 UTC**. A submission
   climbing toward 958 does not track downward; one settling near its true
   rating does.
2. **`55072063` rose 958.2 → 970.1 over the same window, against the same
   field.** So the meta is not dragging our agents down generically, and the
   gap is now ~220 points between two agents that differ by **one flag**.

That is still only ~1.5 h of episodes and two readings minutes apart, so it is
**suggestive, not settled** (rule 2 wants ≥1 h between readings). **The next
session's first job is to read it again** — §3.0 — and nothing else should
happen first.

⚠ **`55054446` is the standing warning about rule 2.** Day 6 recorded it at
"916.8 → 936.0 → **979**, three readings, trending up" and wrote the plan
against that. It is now **905.2** — *below* its own first reading and ~74
points off what the plan assumed. **Treat a rising score as unconverged, not as
momentum.** The same caution now cuts the other way for `55077709`: a low
reading on a young submission is equally uninformative.

The `chip_target` claim survives all of this: 905.2 vs 758.6 for the same clone
without `targeting.py` is still ~+147, same sign, same conclusion.

### ⚠ The meta has shifted (user-reported, 2026-07-29)

The user reports the field has moved, and the top of the board has reshuffled
(`Majkel1337` 1179.8 now leads; day 6 had `flg` at 1179.6 in first). **This
matters more than any single rule**, because *every* routine measurement in
this repo is taken against **one** opponent: `rule:v10,noS` piloting
`lucario_v10` (rule 12). If the real field no longer looks like that, our local
bar is measuring against a ghost, and a rule can clear it while losing games on
the actual ladder — which is a candidate explanation for §3.0 that does not
require `counter_source` to be a bad rule at all.

**Re-anchoring the arena on the current meta is now the highest-value
infrastructure work in the project.** See §3.1.

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
    | `counter_source` | **half dominated, half tradeoff — see §3.0** | 0.534, **LB unresolved** |
    | `drag_target` (aim the drag) | tradeoff | 0.489 |
    | `boss_converts` (force the play) | tradeoff | 0.493 |
    | `boss_veto` (suppress the play) | tradeoff | 0.493 |
    | `drag_target` high-HP tiebreak | tradeoff | 0.490 |

    The net has watched 2,810 games of humans making those trades and is
    already as good at them as our arithmetic. What it *cannot* do is see HP,
    damage or attached energy, so it loses to pure arithmetic every time the
    answer is arithmetic. **Before writing a rule, ask which column it is in.**

    ⚠ **And be strict about it — "dominated" is easy to talk yourself into.**
    `counter_source` was filed as dominated because the heavily-damaged source
    is better *both* on damage transferred and on healing. The first is real
    arithmetic; the second is a judgment (a heal is only worth it if the
    Pokemon is savable) that was **asserted, not measured**. It then won the
    arena and read 762 on the LB. **A rule is only in the dominated column if
    EVERY dimension it moves is arithmetic — if one of them is a judgment, it
    belongs in the tradeoff column no matter how good the other one looks.**
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

All of P5 is closed and the whole Boss's Orders lever is closed four
interventions deep. P6a won locally and shipped. **The open questions are now
(0) is P6a actually good, (1) are we even measuring against the right
opponents, and (2) Crustle.**

### 0. FIRST: resolve `55077709` (P6a / `counter_source`)

`counter_source` won both local bars — 0.534 [0.513, 0.556] n=2000 mirror, and
0.626 [0.604, 0.647] n=2000 vs `rule:v10,noS` against a bare `bc`'s 0.593 — and
then read **762.2 → 746.4** on the LB while the otherwise-identical
`55072063` sat at **970.1**. Resolve that before anything else.

**The prior going in should be "this rule is probably bad", not "the LB is
noisy".** Two agents differing by one flag, ~220 points apart, with the newer
one trending down rather than up. If step 1 confirms it, the local arena
produced a confident false positive — and understanding *why* is worth more
than the rule was, because every other rule was validated the same way.

**Step 1: read it again.** Two readings ≥1 h apart (rule 2), and compare
against `55072063`'s **contemporaneous** score, never against a remembered one.
Both are active and playing the same field, so the comparison is fair only if
both numbers are read at the same time.

```powershell
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(s.ref, s.date, s.status, s.public_score) for s in a.competition_submissions('pokemon-tcg-ai-battle')[:3]]"
```

**Step 2, if it stays well below `55072063`:** do **not** immediately resubmit
(see the ⛔ box at the top — a third submission evicts our best agent). Instead
work out *why*, because there are two very different candidate causes and they
have opposite fixes:

**Cause A — the local anchor is stale (§3.1).** `counter_source` was measured
against exactly one opponent deck. If the meta moved, it can be a fine rule
that we validated against the wrong field. Fix: re-anchor, then re-measure.

**Cause B — my dominance argument for it was half wrong, and this is the more
likely of the two.** I classified `counter_source` as a *dominated-option* rule
on the grounds that the heavily-damaged source is better in both directions at
once. **Only one of those directions actually holds:**

- *Damage transferred* — genuinely dominated. A source with 3+ counters moves
  30; a source with 1 moves 10. No judgment, pure arithmetic. This half is
  sound.
- *Healing* — **this is a tradeoff, and I asserted it rather than measured
  it.** Moving 30 off our most-damaged Pokemon is only the best heal if that
  Pokemon is *savable*. If it dies next turn regardless, the 30 is wasted, and
  healing a Pokemon you can actually keep alive would have been better. The
  clone may well have been making that judgment correctly with information the
  rule throws away.

So `counter_source` is **not** a clean member of the 3-for-3 dominated column;
it is a dominated half welded to a tradeoff half. If it is genuinely hurting,
this is where to look, and the fix is a narrower rule: **only redirect when the
net's pick is strictly worse on transfer AND not obviously the better heal** —
e.g. leave the net alone when its chosen source is the Active, or when the
max-counter source is already beyond saving (its HP ≤ incoming damage). That
variant preserves the arithmetic win and returns the judgment to the net.

**Step 3, the rollback if needed.** A bare `bc` with `counter_source` off is
one flag: `PolicyAgent(..., counter_source=False)` in `agents/sa/bcagent.py`,
or rebuild from the `55072063` tree. Do this only once a submission slot is
genuinely free (i.e. you are willing to lose `55054446`'s slot, not
`55072063`'s).

### 1. Re-anchor the arena on the CURRENT meta — the highest-value work here

**Everything routine in this repo is measured against `rule:v10,noS` piloting
`lucario_v10`, and the user reports the field has moved.** Rule 12 says one
opponent deck is not the field; a shifted meta makes that far worse, because
now it may not even be a *representative* deck. Every number in §3, §6 and the
rule table in §4 was earned against that single anchor.

**Do this:**
1. Fetch the newest top episodes and mine what is actually being played —
   `scripts/mine_meta.py` exists for exactly this, and `fetch_top_episodes.py`
   pulls the days. The last fetched day is **2026-07-27**; get 07-28 and 07-29.
2. Rank the decks by frequency **among high-rated players**, not overall.
3. Import or reconstruct the top 2–3 as arena opponents. `decks/crustle.py`
   (below) is a worked example of reconstructing a decklist straight from
   replays — the same method generalises.
4. **Then re-run the A/Bs that decide what ships**, at minimum `bc` vs
   `bc:x,noSrc` and `bc` vs `bc:x,noChip`/`noSpread`, against each new anchor.
   A rule that wins against all of them is real; one that wins only against
   `rule:v10` was never measured properly.

⚠ **Do not treat a cross-deck score as skill** (rule 5) — use each new anchor
the way `rule:v10` is used: as a fixed opponent for A/B *deltas*, where both
sides of the A/B face the identical opponent.

### 2. Crustle — the deck is now in the repo, unpiloted

**`decks/crustle.py` exists** (user-supplied this session), reconstructed as the
most common exact 60 across replays containing card 345. It resolves in the
arena: `arena.resolve_deck('crustle')` → 60 cards. Notable contents: Dwebble ×4
/ Crustle ×3, Cornerstone Mask Ogerpon ex, Mega Kangaskhan ex ×2, Crushing
Hammer ×4, Boss's Orders ×4.

**Two things are missing and both matter:**

1. **No pilot.** A decklist alone cannot reproduce the lockdown — the wall only
   works if the pilot sets it up and sits behind it. `bc` would play it
   off-distribution and `rule:v10` is Lucario-specific scoring. Options: find
   the public Crustle bot (`dashimaki360/beating-the-day-1-1-crustle-bot`
   implies one exists), or write a minimal rule pilot. **A weak pilot will
   under-read the matchup and make the hole look smaller than it is.**
2. **The `crustle-replays/` directory the decklist docstring cites is not in
   the repo.** Only the decklist survived. Ask the user for it if the source
   games are needed.

**The user's idea, recorded but NOT committed to (their instruction):** lean
harder into passive damage — Munkidori's Adrena-Brain and Froslass's Freezing
Shroud — to beat Crustle, either by (a) running more copies or (b) prioritising
those Pokemon when fetching from deck/discard.

Facts already established for it:

- **Munkidori is already at 4 — the copy cap. There is no room to add any.**
  Only the Froslass line (2 Snorunt / 2 Froslass) can grow. That substantially
  weakens option (a).
- Option (a) has a measured headwind besides: the last decklist variant scored
  0.490 [0.468, 0.512] n=2000, and the net is trained on this exact 60, so any
  change is off-distribution for the policy too (§6).
- **Option (b) has the better prior.** It changes no cards, and *conditional on
  the Crustle matchup* "fetch the Pokemon whose damage actually goes through"
  is close to a dominated choice rather than a tradeoff — the column that is
  3-for-3. It would be a matchup-branch rule, which is what the top of the
  board is built out of. It lands on `TO_HAND` (15.3% of selects; only the
  duplicate-avoidance question there has been closed, not this one).

**⚠ VERIFY THE PREMISE FIRST — one probe game, before any of the above.** The
whole idea rests on "Adrena-Brain and Freezing Shroud *move/place* damage
counters, which is not damage from an attack, so Mysterious Rock Inn should not
prevent them." **That has never been checked in-engine**, and our card db
carries no ability text for card 345 (`abilities: None`), so it cannot be
settled by reading — only by playing. Use the `probe_adrena.py` pattern that
settled P4b's four mechanics. **If counters do not bypass the prevention, the
entire passive-damage line is dead and no decklist work should happen.**

Also still unverified from day 5: that Grimmsnarl ex really deals **zero** to
Crustle. `attack_into_ex_immune_active` has been in `opportunity_audit.py` for
days and **has never fired**, purely because there was no Crustle deck to fire
against (rule 9). It can fire now.

### The board

| | item | state | arena |
|---|---|---|---|
| **§3.0** | **is `55077709` (P6a) actually good?** | **OPEN — do this first, submit nothing until it resolves** | 0.534; 0.626 vs `rule:v10` |
| **§3.1** | **re-anchor the arena on the current meta** | **OPEN — highest-value work, the meta shifted** | — |
| **§3.2** | **Crustle** | deck now in repo, **no pilot**; premise unverified | — |
| P6b | post-KO promotion (`TO_ACTIVE`) | sized: 9 misses in 120 games — **too small, closed** | — |
| P6c | how many counters to move (`..._COUNT`) | **closed — already 100% max** | — |
| P5a/b/c | the three live-game findings | **all three closed**, see below | — |
| P1 | re-rank decks | **superseded by §3.1** — do that instead | ~20 min |
| P2 | MAIN-decision rules — start with the **lethal audit** | not started — the real remaining mass | days |
| P4 | all three items | **closed**, see below and §6 | — |

**Replays of a live agent vs real opponents** are at
**`replays/submission_replay_2026-07-29/`** (user-supplied, 55 games, 54
distinct LB opponents — team name `Scio` is us, one game is the self-play
validation episode). `scripts/p5a_replays.py` shows how to read our own selects
out of them. Use them for diagnosis, not training (§6: more imitation data is
dead).

⚠ **The folder's date is misleading and it cost a conclusion.** Despite the
`2026-07-29` name these are **`55054446`'s games — the chip-only agent, before
`energy_spread`** — not the 958/970 P4b agent. The user has said future dumps
will be named to avoid this. **Two consequences:**

- Any measurement about Munkidori that depends on *two armed* Munkidori is
  understated in this corpus, because `energy_spread` is exactly the rule that
  arms the second one. The P5a replay result below (6 pooled KOs, 0 real
  choices) is confounded in that direction. **P5a's closure rests on the local
  numbers, which do include `energy_spread`** — the replay figure corroborates
  but cannot carry it.
- **Before using this folder for anything, check which agent produced it.**

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

**`counter_source` is ON by default** in `PolicyAgent` and in
`arena.build_agent`, so a bare `bc` ships it. `bc:<label>,noSrc` turns it off.

🔴 **AND THEN IT READ 762.2 ON THE LB. This rule is on trial — see §3.0.** Both
"independent" confirmations above share the same opponent deck
(`lucario_v10`) and the same era of the meta, so they are less independent than
they look. The mirror A/B and the `rule:v10` A/B can both be right about that
opponent and both irrelevant to the current field. §3.0 has the two candidate
causes and the narrower rule that would keep the arithmetic half.

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

### P1 — Re-rank decks — **superseded by §3.1, but the caveats still apply**

`scripts/deck_sweep.ps1` defaults to `rule:v10,noS` / `lucario_v10` and covers
all 7 decks (now 8 with `crustle`) — run with no arguments (§7 PowerShell
gotcha), ~20 min. **Do §3.1 first**: re-ranking decks against a stale anchor
answers the wrong question, and that anchor is what §3.1 is replacing.

⚠ **Read any sweep with a thumb on the scale for grimmsnarl.** `chip_target`,
`energy_spread` and `counter_source` are all grimmsnarl-specific (Shadow
Bullet's snipe, Munkidori's Adrena-Brain), so the sweep measures "our agent +
three grimmsnarl-only rules" against decks that get none of them. That is the
right question if you are asking *what to ship*; it is the wrong question if
you are asking *which deck has the higher ceiling*. For the latter, re-run with
`bc:plain,noChip,noSpread,noSrc`.

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

**Concrete first step — the lethal audit.** Do not write a MAIN scorer. Measure
one thing: *given the board, does a payable attack KO the opponent's Active,
and did we take it?* Missing an available KO is arithmetic, not judgment, and
`textdmg.estimate` is exact for this deck (§4), so the measurement is
trustworthy. **Size it in two separate cuts and do not merge the
denominators** — they are different classes and rule 11 predicts different
outcomes:

1. **Our current Active has a payable attack that KOs, and we chose a different
   attack.** Dominated. High prior. If this reads near zero the way
   `REMOVE_DAMAGE_COUNTER_COUNT` did (100% already correct), it closes cheaply
   and you have lost an hour, not a day.
2. **The KO needs a different attacker promoted first.** That costs a retreat
   and a turn of setup, so it is a tradeoff. Lower prior. Measure separately.

`scripts/p6_recon.py` is the template for this kind of counter, and
`p5b_check.py` is the template for confirming a rule actually fires (rule 9)
before spending an A/B on it.

### P3 — The abomasnow hole (open)

0.360 vs 0.475–0.519 elsewhere (pre-P2c, re-measure), and our selects/turn
collapse from 12.5–16.6 to **8.6** with shorter games — a lockdown, not subtle
misplay. Replay a loss with `SA_DEBUG=1` and read the actual select options.

**Crustle has moved to §3.2 — the deck now exists.** Summary of the mechanics
claim, which is still unverified: **Mysterious Rock Inn is an ABILITY on
Crustle itself** (card 345; 344 is Dwebble — earlier drafts of this file wrote
it as if it were a separate stadium card, and our own card db exposes no
ability text for it at all, `abilities: None`). It prevents damage from
opponent {ex} attacks, and Grimmsnarl ex is `ex=True`, so it should take
**zero** from Shadow Bullet. The proposed out is that Adrena-Brain and Freezing
Shroud *move/place damage counters*, which is not "damage done by an attack".
**None of this has been confirmed in-engine — see §3.2, which is where the
probe is specified.** `dashimaki360/beating-the-day-1-1-crustle-bot` is a
public notebook on this matchup; V10 hardcodes 344/345 as "the crustle wall".

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

# The current leaderboard (top is Majkel1337 1179.8; paginates at 20)
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(i, r.team_name, r.score) for i, r in enumerate(a.competition_leaderboard_view('pokemon-tcg-ai-battle')[:20], 1)]"

# §3.1 re-anchor: what is the field ACTUALLY playing now? (last fetched: 07-27)
python -X utf8 scripts/fetch_top_episodes.py --date 2026-07-29 --max 400
python -X utf8 scripts/mine_meta.py

# Rebuild data only (more data is NOT a lever — §6)
python -X utf8 scripts/fetch_top_episodes.py --date 2026-07-26 --max 400
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds/d26 replays/2026-07-26
```

### Data on disk

`replays/`: 07-17..07-22, 07-24, 07-26, 07-27 = 400 each; 07-23 = 175,
07-25 = 268, 07-16 = 115; 07-13/14/15 = 0. **Nothing newer than 07-27 — and
the meta has shifted since (§3.1).** Plus 366 old-repo replays at
`E:\Kaggle\pokemon-tcg-simulation\replay_miner\replays\2026-07-06..12`.

**`replays/submission_replay_2026-07-29/` = `55054446` (the chip-only agent)
vs the real field**, 55 games, user-supplied. ⚠ The folder name is a date, not
an agent — it is **not** the P4b agent despite being dated the same day. See
the warning in §3.

`decks/crustle.py` was reconstructed from a `crustle-replays/` directory that
is **not in the repo** — only the decklist survived (§3.2).

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

⚠ **Everything in this section was measured against ONE opponent
(`rule:v10,noS` / `lucario_v10`) in the pre-shift meta.** The negatives are
probably safe — a rule that could not win against that anchor is unlikely to be
a hidden gem — but the *positives* are the ones to re-check after §3.1, and
`counter_source` is already under suspicion for exactly this reason (§3.0).

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
- **⛔ "Latest 2 active" is a TRAP, not a footnote.** Submitting a third agent
  today silently **evicts your best one from active play** — it stops playing
  episodes and its score freezes wherever it was. This is exactly the position
  at the end of day 7: `55072063` (970.1, our best) is only active because
  nothing has been submitted after `55077709`. **Before every submission, list
  the active pair and name which one you are willing to lose.** A rollback is
  itself a submission and pays this cost too.
- **A young submission reads low and it means nothing.** Everything starts at
  μ=600 and climbs for hours (`55072063` took ~4+ h to reach 958). Never
  compare a fresh submission against a mature one, and never compare either
  against a remembered number — **read both scores in the same call** (§3.0).
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
