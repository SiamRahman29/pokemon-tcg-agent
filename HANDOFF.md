# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** public LB **and** the Strategy Category. Sim deadline **2026-08-17**,
then ~2 weeks continued play; strategy report due **2026-09-14**. Kaggle CLI is
authenticated.

**Standing (read 2026-08-01 10:17 UTC, full LB — now 6,088 rows):** we are
**`Scio`, rank 268 of 6,088, 923.0** — our best live number, and ⚠ **still
climbing, so NOT a settled reading** (rule 2). Top is `Majkel1337` 1251.3, then
`Sixth Sense` 1181.7, `keidroid` 1174.3, `flg` 1163.1. The board grew
3,000 → 5,000 → 6,024 → **6,088** and the top reshuffles constantly — treat any
ranking as a snapshot.

✅ **Getting the LB no longer needs 300 paginated calls:**
`competition_leaderboard_download('pokemon-tcg-ai-battle', path='out/lb')`
returns **all 6,024 rows as a zipped CSV in ONE call** (columns: `Rank`,
`TeamId`, `TeamName`, `Score`, `SubmissionCount`, …). The `page_token` walk in §5
still works but is obsolete. **This is what made the day-10 analysis possible** —
it lets any team name in a replay be joined to its rating.

> # 🔴 READ THIS FIRST — B7 RAN ON DAY 11 AND IS CLOSED (2026-07-31 night)
>
> **Day 10 measured; day 11 trained, and both arms lost.** The pre-registered
> bar was **+50 Elo weighted over the five anchors**. Neither arm was close, and
> neither was a null — both are losses against the net they were built to beat:
>
> | net | what it clones | miss vs **the field** | miss vs **ntumlnoob** | mirror vs v3, n=2,000 |
> |---|---|---|---|---|
> | **v3** (live) | the ~50-pilot mixture | **30.2%** | 40.1% | — control |
> | `rw25` | mixture, weighted to LB rating | 32.0% | **40.2%** | **0.421** [0.400, 0.443] ≈ **−55 Elo** |
> | `b7_ntum` | one 1163-rated expert | **36.2%** | 19.4% *(32.8% held out)* | **0.370** [0.349, 0.391] ≈ **−92 Elo** |
>
> 🔴 **Read the first and last columns together — the ordering is exact.** Field
> disagreement 30.2 → 32.0 → 36.2; Elo 0 → −55 → −92. **Every step away from the
> field's modal policy costs strength, and the net that best imitates the #2
> player is the weakest agent this project has built.** `EVIDENCE` §8t, §8u.
>
> ⚡ **And §8q's headline was NARROWED by a much harder test — this is the part
> to carry forward.** Over **87 same-deck, same-week demonstrators the net has
> never trained on a single row of**, agreement **peaks at 1050–1100 (76.1%) and
> falls in BOTH directions** — 66.7% below 900, 70.9% at 1100–1150, 65.6% at
> 1152, 59.9% at 1163. **Agreement measures distance from the fitted mode, not
> skill.** A 780-rated player is 33% unpredictable to us too, and nobody wants to
> clone them. §8r.
>
> ✅ **Covariate shift is RULED OUT** (it was §8q's one unanswered objection):
> policy-vs-policy disagreement is **26.7% on our own states vs 31.9% on theirs**
> — near-symmetric, so the expert clone is a genuinely different policy, not one
> policy measured off-support. The test carries a **1.7% positive control** (the
> v3 net reproducing its own submission's replays). §8s.
>
> ⛔ **Do not build a third demonstrator-selection variant.** Five axes of
> more/better training have now measured null or negative; **exactly one
> intervention ever worked and it was representational** (§8f). Spend the
> remaining days accordingly.
>
> # ⚡ AND TWO MORE CLOSED THE SAME NIGHT — read these before planning anything
>
> **1. ✅ THE CLONE IS NOT CAPACITY-BOUND. The ~30% residual is the ENCODING.**
> Identical corpus, identical recipe, only the width changed:
>
> | net | params | vs v3 | misses of 12,939 | agreement |
> |---|---|---|---|---|
> | **v3** (live) | 594,369 | 1.0× | **3,902** | 69.8% |
> | `cap_big` | 1,559,489 | **2.6×** | **3,900** | 69.9% |
> | `cap_xl` | 4,865,985 | **8.2×** | **3,945** | 69.5% |
>
> **2.6× buys two decisions out of 12,939; 8.2× loses 43.** Both big nets drive
> train loss far below v3's while validation peaks early and declines — they
> already have more capacity than the features can use. §8w.
>
> 🔴 **This is the gate result for RL and it cuts against it.** A policy gradient
> reads the **same** feature vectors. Where two options are bitwise-identical
> inputs with different right answers (§8f's exact finding), their **gradients
> are identical too** — exploration cannot break a tie the representation cannot
> express. **RL inherits this ceiling rather than escaping it.** ⇒ **The feature
> audit is a PREREQUISITE for the RL program, not a competing priority.**
>
> **2. 🔴 B4 IS DEAD, and the manner of death is the lesson.** The §8n design
> diagnosis was **right** — end-of-our-turn myopia was most of the rout — and
> simulating the opponent's reply moved it **0.075 → 0.375 [0.311, 0.444] n=200**,
> the largest movement any B4 change produced. **Still ≈ −89 Elo** against a
> clone costing 1 ms, and below the 0.40 line set before the run. **A correct
> explanation of a failure does not entitle you to a fix.** §8v.
> ✅ **The time-budget confound is RESOLVED, not caveated.** At matched budget,
> `seq,sb1.0` **without** reply scores **0.165 [0.120, 0.223]** against the reply
> arm's **0.375 [0.311, 0.444]**, n=200 each — **disjoint CIs**. So the design fix
> is worth **≈ +0.21 on its own** (≈ +193 Elo) and the extra time ≈ +154 Elo.
> **The diagnosis was confirmed by a controlled experiment and B4 died anyway.**
> 🔴 **Consequence for NNUE: B4 was its only consumer that is not the dead
> game-tree search (§2), so an incremental evaluator buys nothing here until some
> consumer exists.** Do not build one on spec.

<details><summary>Day 10's headline box — the restore and the expert dumps (superseded in part by §8r–§8u, kept for the reasoning)</summary>

> # 🔴 TWO THINGS CLOSED ON DAY 10 (2026-07-31 pm)
>
> **1. ✅ THE P4b RESTORE IS ANSWERED BY THE LADDER: the 952 was a BOARD-SIZE
> ARTIFACT, not a better agent.** The identical tarball was resubmitted as
> `55129730` and read **833.9 at 4.0 h of play**, where the original read
> **958.2 at ~4 h** — same code, same age, **−124 points**, board ~4,000 →
> **6,024**. Our three agents now read **833.9 / 818.1 / 841.5** live.
> **§8k ("everything we own is within 36 Elo") is confirmed on the LB itself.**
> ⛔ **Do not reopen the restore. Do not chase 952.** `EVIDENCE` §8p.
> ⚠ One reading only — re-read ≥1 h later before quoting 833.9 (rule 2). The
> matched-age comparison does not depend on convergence.
>
> **2. ⚡ WE NOW HAVE EXPERT DEMONSTRATIONS OF OUR EXACT DECK, AND THEY SAY
> SOMETHING UNCOMFORTABLE.** 227 games from **Sixth Sense (#3, 1152.4)** and 330
> from **李秉叡（ntumlnoob）(#2, 1162.8)** — both playing `decks/grimmsnarl.py`
> **card-for-card**. Scoring the live v3 net against their actual choices:
>
> | demonstrator | rating | rows | miss rate |
> |---|---|---|---|
> | 48 other grimmsnarl pilots (mostly 1090–1140) | ~1110 | 10,088 | **27.2%** |
> | Sixth Sense (#3) | 1152 | 18,296 | **34.4%** |
> | ntumlnoob (#2) | 1163 | 25,775 | **40.1%** |
>
> **Agreement falls monotonically as the demonstrator gets better.** And the two
> top players diverge from us in *different* contexts (Sixth Sense almost entirely
> in `TO_HAND`, −31.5 pp; ntumlnoob broadly in MAIN/damage/switch) — **so there is
> no single "expert move" to copy.**
>
> 🔴 **The structural fact behind it: our corpus is ALREADY elite and already
> concentrated.** flg (1125) **527 seats**, Dries @ Tufa Labs (1102) **490**,
> James Cox (1166) **414**, Dominic Peel (1136) 238, LiamK (1128) 216. **We clone
> 1100–1166 play and score 833.9.** More/better demonstrators is therefore NOT
> the obvious lever; **what has never been tried is cloning ONE policy instead of
> a ~50-pilot mixture.** `EVIDENCE` §8q.
>
> ⚠ **The alternative explanation is NOT ruled out and must be stated first in
> any write-up: COVARIATE SHIFT.** Agreement is measured on the demonstrator's own
> trajectory distribution, so part of that 40% is BC's compounding-error problem,
> not a policy we can copy. **Low agreement does not prove they play better. Only
> an arena A/B can.**
>
> ✅ **ANSWERED day 11 — covariate shift is ruled out (§8s), the monotone trend
> is narrowed to a PEAK at 1050–1100 (§8r), and both interventions this box
> motivated LOST (§8t, §8u).** The box is kept because its reasoning was sound
> and its bar was set honestly; the arena is what changed the answer.

</details>

<details><summary>Day 9's headline box — the "−130 regression" artifact (superseded by §8p, kept for the reasoning)</summary>

>

> **⚠ An earlier version of this box, written mid-session, said the gap was
> SOLVED and that v3's loss was reproduced locally. That was written after 2 of 5
> anchors and it was wrong. Corrected here after the full sweep.**
> `report/EVIDENCE.md` §8i.
>
> **1. There is a real v3 weakness, and the retired anchor is what found it.**
> Both agents vs a fixed `rule:v10,noS` on `lucario_v10`, n=2000, CIs disjoint:
> **P4b 0.576 [0.554, 0.598] vs v3 0.505 [0.483, 0.527]** — v3 is ≈ **−50 Elo**
> against **12.8%** of the field. B1 could not have seen this; the anchor had been
> retired two days before.
>
> **2. But it does NOT explain the regression — it points the other way.**
> Weighted by field share over the 61.4% now measured, **the arena says v3 is
> ≈ +35 Elo BETTER** than P4b (mirror head-to-head **0.657**, Crustle **+91 Elo**,
> Alakazam a dead heat, Lucario **−50**).
>
> **3. 🔴 And the −130 was a comparison rule 2 forbids.** `55072063`'s **952.0 is
> FROZEN** — earned 07-29 against a ~4,000-entrant board; the board is now
> **6,000**. The only same-time, both-active comparison is:
>
> | submission | agent | read 2026-07-31 |
> |---|---|---|
> | `55116557` | **v3, rules off** | **819.8** |
> | `55077709` | P6a (lw2 + chip + spread + `counter_source`) | **845.0** ⚠ *still climbing*: 824.9 → 837.5 → 845.0 |
>
> **−25 points, against an agent that has not converged — well inside the LB's
> own ±50–100 swing.** So the honest status is **"v3 and P6a are indistinguishable
> on the ladder"**, not "v3 lost 130 points".
>
> **⇒ §8g's "the arena is systematically wrong, n=2" is WEAKENED.** Both its
> instances compared against non-comparable numbers — `counter_source` against a
> converging score, B1 against a frozen one. **There may be no systematic arena
> bias to explain.** Do not spend more days explaining one.
>
> **🔴 The genuinely load-bearing finding of day 9 is the sampling frame, and it
> is independent of all of the above.**
> `fetch_top_episodes.py` mines the **top** episodes by `avg_score`, and Kaggle's
> daily datasets **bottom out at 1055** — buckets 800–900 and 900–1000 contain
> **zero** episodes. We play at **825–952**. **No amount of episode mining can
> ever describe the field we face.** §8b's "Lucario is 0% of the meta" was true
> at 1150+; in our own 109 real games it is **12.8%** — tied for the largest deck
> we play against.
>
> **Our own submission replays are the only evidence about our own opponents.**
>
> ✅ **Fixed the same day:** `scripts/p9_field_census.py` names the real field,
> and `scripts/import_field_agents.py` imported the two missing anchors
> (`rule:alakazam5`, `rule:archaludon`). Anchor coverage **39.4% → 71.6%**.
>
> ⚠ **v3 is NOT refuted as a net.** It wins big on 26.6% of the field and loses
> on 12.8%; the other 60.6% is unmeasured. **Do not discard it and do not reship
> it** until the 5-anchor sweep finishes (▶ item 2).

</details>

**Read §2 before trusting any number. §3 is the live plan. This file must always
end with a live plan, never a summary.**

⚠ **Day 9 note on reading this file:** several load-bearing claims dated
2026-07-30 were **narrowed, not deleted** — the meta-shift table (§1), rule 12's
"`lucario_v10` is 0% of the meta", and rule 16's "the arena does not measure
ladder strength". Each is now prefixed with what it is actually true *of*. If a
statement in here about "the meta" does not say **which score band** it describes,
distrust it: mined episodes are the top-1150 band, and
`scripts/p9_field_census.py` on our own replays is ours (`EVIDENCE` §8i).

### ▶ START HERE — DAY 16 (in progress 2026-08-02; user-directed. The goal is to WIN)

> # 🔴 THE DAY-16 HEADLINE: THE USER WATCHED ONE REPLAY AND FOUND AN ANCHOR THAT WAS THROWING GAMES
>
> **Day-15 item 6 said the five anchors carry 71.5% of every weighted verdict
> here, were imported rather than written by us, and had never been watched. The
> user watched `out/replays/anchor_vs_anchor/game000` and reported the Crustle
> pilot never benched a second Pokémon and lost when its active was KO'd. It was
> correct.** `EVIDENCE` §8ah.
>
> `sources/crustle.py:338` scored **every Pokémon except Dwebble at −5000** for a
> bench play, with **no empty-bench guard**, so once the Dwebbles were gone it
> played on an empty bench until the first KO ended the match. ✅ **Fixed** —
> bench-full −5000, **empty bench 90000**, otherwise Dwebble 25000 / any 12000.
>
> | agent | games | EXPOSED turn-ends | empty-bench losses |
> |---|---|---|---|
> | **`rule:crustle` before** | 3 | **0.667/game** | **2 of 2 losses** |
> | **`rule:crustle` after** | 12 | **0.000** | 2 of 10 |
> | `rule:archaludon` | 12 | 0.000 | 1 of 3 |
> | `rule:alakazam5` | 18 | 0.000 | 0 of 11 |
> | `rule:lucario` | 12 | 0.000 | 0 of 5 |
> | **`bc:v5` (ours)** | 51 | 0.000 | **0 of 23** |
>
> 🔴 **CONSEQUENCE — EVERY VERDICT CARRYING A CRUSTLE TERM IS SUSPECT and must be
> re-run against the repaired pilot.** Our arena reads **0.663** vs `rule:crustle`
> against a **57.1%** real win rate; §8i filed that as "the arena reads
> optimistic" and this is a mechanism for part of it. ⚠ **NOT YET DONE — user
> has not authorised the re-run** (hours of compute, rewrites published numbers).
> At §8ac weights Crustle is 6.7% of the field, so the dilution is real.
>
> ⚡ **The methods lesson is about OUR detector, not the pilot.** The obvious
> screen — "bench empty, bench play offered, chose something else" — **overcounts
> and is not an error rate**: a pilot that plays three items and *then* benches
> scores three declines and did nothing wrong. On it, `rule:archaludon` looked
> **worse than Crustle** (1.333/game) and **is clean**. The sharp detector adds
> *"...and it ATTACKED or ENDED THE TURN anyway"*. **Fourth confident-but-wrong
> script in three days** (§8ad, §8ae, §8af, this) — ⚡ but the first one caught
> *before* reporting, by asking **"what is the benign reading of this count?"**
>
> ⛔ **AND THE RULE IT SUGGESTED FOR OUR OWN AGENT DIED BY SIZING** (§8ai). Empty
> bench is the most *dominated* option there is (rules go 3/3 there), so it was
> promoted as the best-shaped candidate in days — then sized: over 75 real ladder
> games and 7,094 decisions it fires **0.187/game** and matches **1 of 22
> losses**. The Morgrem out died at ~0.2 (§8e), Pokégear at 0.27 (§8ag). ~1.3% of
> games against an n=2000 A/B that resolves 2.1%. **Not built. Third sizing
> closure in three days.**
>
> #### 📋 The health-check submission — `55169114`, and it is decision-identical to v5
>
> ⚠ **It was submitted 2026-08-01 18:42 UTC and recorded in NO doc until now.**
> `dist/submission_bc-grimmsnarl-netspolicy_20260802-004209.tar.gz`. `diff -rq`
> against the v5 bundle: **only `main.py` and `sa/bcagent.py` differ**, and only
> by the health counters + one `print`. **Weights, deck, engine, every other
> module byte-identical. No decision path changed.**
>
> | submission | what it is | age | read 05:08 UTC | read 05:25 UTC |
> |---|---|---|---|---|
> | `55160229` | **v5** | 18.8 h | 956.5 | **951.0** |
> | `55169114` | **v5 + stdout counters** | 10.7 h | **874.8** | **874.8** |
> | `55156480` | v4 (frozen, evicted) | 22.2 h | 910.5 | 910.5 |
>
> 🔴 **≈ −80 points between two agents that make the same move in the same
> state**, both past the 4 h convergence window, read in the same call. If it
> holds it is **the project's first measured LB null**, and it is **double** the
> +40.5 that day 15 called "rule 2 satisfied on a net pair". ⚠ **Two readings
> only 17 min apart — rule 2 wants ≥1 h. A settling read is armed for ~07:25 UTC
> and §8aj is NOT to be written until it lands.**
>
> ✅ **THE SUBMISSION-LOG TRACK IS CLOSED, and only one of its four goals paid.**
> Three episode logs collected:
> - ✅ **pool usage settled** — 318 selects, **1.12 s of a 1,800 s budget**, worst
>   call 0.153 s (startup). The "0.1 s of 600 s" claim is real.
> - ✅ **net is live, two ways in one file** — 86–98 net calls at **1.2–1.6 ms**
>   against 7–10 fallback-shaped calls at **23–37 µs**, a 50× separation.
> - 🔴 **Kaggle starts a FRESH PROCESS PER EPISODE** — all three logs read
>   `calls=1`, so the cumulative counter line can never print. The heartbeat fix
>   would cost a submission slot.
> - ⛔ **Nothing further is worth a slot.** The crash alarm already fires on the
>   next select (it does *not* wait for the heartbeat) and has reported zero over
>   3 games; `net_missing` belongs in a **build-time assertion**, not a log; and
>   the margin-distribution idea is obtainable **offline from replays we already
>   download**. Leave the counters in as passengers; never submit for logging.
>
> #### The day-16 order of work
>
> 1. ✅ `crustle.py:338` fixed · ✅ anchor pathology audit (`p24`) · ✅ empty-bench
>    sizing closed · ✅ EVIDENCE §8ah/§8ai · ✅ STRATEGY §4f written and §6's stale
>    day-9 shares corrected.
> 2. ⏳ **Ladder settling read (~07:25 UTC) → §8aj**, the identical-agent null.
> 3. 🃏 **DECK WORK IS THE USER'S PRIORITY AND IS NEXT.** ⛔ **Anchor-based deck
>    A/Bs are BLOCKED** until the repaired pilot is re-validated (rule 12: build
>    the anchor before the deck opinion). ⚡ **The MIRROR is not blocked** — both
>    seats are our own net on our own 60, so only the list differs — and it is
>    **33.3% of the field, 51.1% above rating 900, 71.4% above 1000**. Start
>    there. §8af says swaps inside the **134 corpus cards** are low-risk; §8ai
>    notes we run **10 basics in 60** but caps that lead at ~1.3% of games.
> 4. ⏸ Re-run the Crustle-term verdicts against the fixed pilot — **user's call**.

<details><summary>Day 15's headline box (2026-08-01) — the rating-dependent field; still live, superseded only in its item-6 status</summary>

> # 🔴 THE DAY-15 HEADLINE: THE "META SHIFT" IS OUR OWN CLIMB, AND IT RE-PRICES EVERY WEIGHTED VERDICT IN THIS REPO
>
> ⚡ **RANK 185 / 6,103 AT 955.1 — our best ever, and ✅ RULE 2 IS SATISFIED FOR
> THE FIRST TIME ON A NET PAIR.** Two readings 61 minutes apart, and they agree:
>
> | submission | 16:06 UTC | 17:07 UTC | age at 2nd |
> |---|---|---|---|
> | **`55160229` v5** | **955.1** | **955.1** | 6.5 h |
> | `55156480` v4 | 914.9 | **914.6** | 9.9 h |
>
> **Both converged, both active, same time — the only comparison rule 2 permits,
> and this project has never had one for two nets before.** v5 is **+40.5**.
>
> 🔴 **It still does not adjudicate §8aa, and saying so is the point.** The arena
> put v5 at **+14 Elo** (**+13.8 re-weighted**, §8ac), and rule 2's second clause
> is that **the LB cannot resolve an effect that size at all** — it confirmed
> `chip_target` at ~150 points and could never have adjudicated `counter_source`
> at ~12. The ladder agreeing in *sign* at ~3× the magnitude is consistent with
> §8i's calibration (the arena ranks matchups right and reads optimistic) and
> with §8ab's compression caveat; it is **not** independent confirmation.
> ⚠ Minor: v5 read **exactly** 955.1 twice, which more likely means few rated
> games in that hour than a perfectly stable rating. v4 moved 0.3.
>
> **The supplied replays answered day-15 item 2, and the answer was not the one
> the item expected.** Pooled over v4+v5 (75 games) the field looks transformed
> since day 9 — the **mirror 13.8% → 33.3%** (Fisher **p=0.002**), **Mega Lucario
> 12.8% → 4.0%**, win rate 63.0% → 70.7%. **But hold the opponent-rating band
> fixed and every era difference vanishes** (all Fisher p ≥ 0.065, n=181 games
> over four dumps). What actually moved is **us**: mean opponent rating **799 →
> 867**, tracking our own 820 → 955.
>
> | archetype | opp <800 | 800–900 | 900–1000 | 1000+ |
> |---|---|---|---|---|
> | **mirror** | 5.3% | 18.6% | **42.4%** | **71.4%** |
> | Alakazam | 13.3% | 28.8% | 33.3% | 14.3% |
> | Crustle | 16.0% | 5.1% | 9.1% | 7.1% |
> | **Mega Lucario** | 17.3% | 6.8% | **0.0%** | **0.0%** |
> | **Archaludon** | 10.7% | 15.3% | **0.0%** | **0.0%** |
>
> 🔴 **The opponent pool is not a population we sample — it is a function of our
> own rating, and it moves when we do.** Every anchor weight in this repo carries
> an invisible parameter: the score we held when the census was taken. **Rule 16's
> sampling-frame trap, committed a second time, on our own data.** `EVIDENCE` §8ac.
>
> **Re-weighted, measurements untouched — only the shares change:**
>
> | verdict | day-9 weights | **day-15 weights** |
> |---|---|---|
> | §8i `v3 − P4b` | +35.6 | **+62.1** |
> | §8j **rules ON − OFF** | **+0.8** | **−18.1** 🔴 **sign flip** |
> | §8z `v4 − v3` | +23.4 | +24.8 |
> | §8aa `v5 − v4` | +10.2 | +13.8 |
>
> ✅ **Nothing shipped has to change** — the v5 bundle already pins
> `chip_targeting/energy_spread/counter_source = False` (verified by reading
> `main.py` out of the tarball). **§8j's "the rules are worth nothing" was the
> right call for ~18× weaker reasons than the true ones.**
>
> ⚡ **AND IT RESOLVES A STANDING CONTRADICTION INSTEAD OF CREATING ONE.** §8b
> (mined ≥1144 band) said **52.1% of seats play our archetype**; §8i (our games at
> ~820) said the mirror is **13.8%**; day 9 filed these as irreconcilable. **They
> are two points on one monotone curve and we have been walking up it.**
>
> #### What this changes for the remaining 16 days
>
> 1. ⛔ **Track C's Archaludon lead is CLOSED BY SIZING, before it was built**
>    (rule 14). Promoted on "10.1% of the field and our worst matchup"; it is
>    **8.0% overall and 0 of 47 games above rating 900**. Same for B3's Mega
>    Lucario instance (**4.0%**) and, more mildly, Crustle (6.7%).
> 2. ⚡ **THE MIRROR IS THE MATCHUP THAT MATTERS AND GETS MORE SO AS WE CLIMB** —
>    33.3% now, **51.1% above 900**, 71.4% above 1000. It is also what our
>    head-to-head net A/Bs already measure, so **our most sensitive instrument is
>    now also our most representative one.** Weight it accordingly.
> 3. ⚠ **Two archetypes have no anchor and now outrank two that do:** Cynthia's
>    Garchomp ex **6.7%** + Dragapult ex **5.3%** = 12.0%, against Crustle +
>    Lucario's 10.7%. `decks/dragapult_ex.py` already exists.
>
> ✅ **ITEM 4 IS BUILT AND VERIFIED: `harness.Recorder`.** Optional recorder on
> `play_game`; `visualize_data()` output is byte-compatible with Kaggle replays,
> so recorded local games are read **unmodified** by `p9_field_census.py`,
> `build_policy_dataset.py` et al., and watchable in `notebooks/visualizer.html`.
> `scripts/p20_record_games.py` (CLI), `scripts/p20_recorder_equivalence.py`
> (12/12 exact checks). 🔴 **Its first version was a test that could not have
> failed** — it demanded identical games run-to-run, but `battle_start` takes no
> seed. **Before trusting an equivalence test, ask what would have counted as
> success.** `EVIDENCE` §8ad.
> 📼 **Ready to watch:** `out/replays/v5_vs_alakazam`, `out/replays/anchor_vs_anchor`
> (⚠ one anchor-vs-anchor game ran **39 turns** against 11 and 11 — first thing to
> explain in the item-6 audit).
>
> ⚡ **ITEM 5 RAN, AND RL SURVIVED ITS OWN KILL CRITERION.** The probe cost zero
> new games — four archives already carry pairs of known separation.
> **Throughput 5.96 games/s per process ⇒ ~5.5M games to the deadline.**
> Detecting §8z's +37 Elo from outcomes takes **800 games (0.015% of budget)**;
> resolving a **1-percentage-point** effect at a single select's context takes
> **960 games** if that context recurs ~20×/game (201 selects/game).
> 🔴 **So the credit-assignment objection — the last one standing after §8x
> narrowed the encoding argument — does not bind. It dies with a NUMBER, which
> is what §2 never had.** `EVIDENCE` §8ae.
> ⛔ **This is NOT a licence to build.** B4 passed all three of its kill criteria
> and then died at n=200. The model prices one context in isolation and ignores
> non-stationarity and shared parameters; training cost is not priced at all;
> and the nearest real measurement (`--winners-only` 0.375) still points the
> wrong way. **The next step is the smallest real thing: fine-tune a SMALL
> parameter set on our own recorded outcomes, A/B at n≥2000 vs a byte-identical
> control with the seed floor carried in.**
> ⚠ **Its first run was garbage in an instructive way:** arena archives are
> **seat-indexed and the seats swap every game**, so reading seat 0 as agent A
> averaged both agents together. It reported +37 as undetectable and +14 as
> detectable. **A bug that biases everything toward the null looks like a
> finding, not a crash.**

</details>

> ## 📍 THE SITUATION AT THE TOP OF DAY 15
>
> - ⏸ **DAY 14 WAS DELIBERATELY IDLE — nothing ran, by user instruction.** Both
>   submissions were left to play for 8–9 h so their replays could be downloaded.
>   **Day 14's item B (the centred option encoding) and item C (Track C deck
>   work) were NOT executed and are NOT cancelled** — they are parked below.
> - 📥 **THE USER IS SUPPLYING THE INPUTS.** Expect at the start of the session:
>   **(a) the replays of `55160229` (v5) and `55156480` (v4)** from their 8–9 h of
>   play. ⚠ That is our own-opponent census data (`p9_field_census.py --us`) for
>   two agents at once, and the **first replay set for the v4/v5 feature blocks.**
> - 🔴 **A NEGATIVE RESULT WAS RETRACTED — READ IT BEFORE PLANNING RL.**
>   "Self-play RL" has been struck from the settled-negative list in all four
>   docs **and from the assistant's memory file**: it was **never run.** No code,
>   no `n`, no CI — a **compute prior inherited from the search result**, filed
>   beside the measured negatives for twelve days. **Rule 15, third instance, and
>   this time the unmeasured claim was living inside `EVIDENCE.md` itself.**
>   ✅ Verified against the old repo too (`E:\Kaggle\pokemon-tcg-simulation`):
>   **no RL code, no training script, no reward function** outside `.venv`.
>   ⇒ **The status is "never attempted", not "dead".** `EVIDENCE` §2's box.
> - ⚡ **AND §8w'S GATE AGAINST RL IS SUBSTANTIALLY SATISFIED.** Its argument —
>   a policy gradient reads the same vectors, so bitwise-identical options get
>   identical gradients — was **narrowed by §8x the next day**: the tie ceiling is
>   **95.6%** against a clone at **71%**, so the encoding binds **at most 4.4 pp**,
>   and the ties that exist are two copies of one card in one role (free choices).
>   §8w named the feature audit as RL's **prerequisite**; it has now been done
>   **twice** (§8y/§8z, §8ab).
> - 🔴 **SO THE LIVE OBJECTION IS NEITHER COMPUTE NOR EXPRESSIVENESS — IT IS
>   CREDIT-ASSIGNMENT VARIANCE**, the same term that killed search (terminal 0/1
>   ⇒ SE ≈ 0.14; the max over ~9 rivals sits 0.21–0.28 above truth by chance).
>   One binary reward over a ~40-turn game with hundreds of selects.
>   **Rule 14 binds: SIZE IT BEFORE BUILDING IT.** The nearest measurement is
>   unfriendly (`--winners-only` **0.375**, §1) but is **not the same mechanism** —
>   that filtered *other people's* games by outcome and discarded half the corpus;
>   a gradient signed on *our own* trajectories does neither.
> - ⚡ **THE PLUMBING FOR ITEMS 2–4 MAY ALREADY EXIST, IN THE OLD REPO.**
>   `notebooks/how-to-output-local-battle-as-json-and-view.ipynb` +
>   `notebooks/visualizer.html`: the engine's own **`cg.game.visualize_data()`**
>   emits a replay the **official viewer** (`ptcgvis.heroz.jp`) renders, and the
>   notebook captures an **obs log + action log** in the *same*
>   `battle_start`/`battle_select` loop **our `harness.py:48-75` already runs**.
>   ⇒ **One optional recorder on `play_game` yields BOTH the human-watchable
>   replay AND the RL/exploration trajectories.** This is a contained change to
>   one function, not a build.
>
> **Deadlines: sim closes 2026-08-17 (16 days). Report due 2026-09-14 (44 days).**
> **Rubric: Model 70% (LB is ONE bullet of five) + Deck 20% + writing 10%.**

#### The day-15 order of work (user-set; items 1–4 are theirs, 5–6 are the parked engineering)

**1. 📈 READ THE LADDER FIRST, TWICE, ≥1 h APART — and this time it is a real
   question, not a ritual.** After 8–9 h **both** submissions are converged, so
   `55160229` (v5) vs `55156480` (v4) is finally a **same-time, both-active,
   both-settled** comparison — the only kind rule 2 permits. ⚠ **It still does
   not adjudicate §8aa** (+7.3 weighted, far under the LB's ±50–100), but it is
   the first honest live read on the pooled block.

**2. 📥 INGEST THE SUPPLIED REPLAYS.** `p9_field_census.py --us` on each
   submission separately. Two questions worth more than the census: **has our
   field composition moved** (it drives every anchor weight in this repo), and
   **does v5 face a different field than v4** (it should not — same deck; if it
   does, the census is measuring rating band, not deck choice).

**3. 🔬 SUBMISSION LOGS — instrument the agent.** ✅ **SUB-ITEM 1 IS BUILT AND
   READS CLEAN.** `sa/bcagent.STATS` + `health_line()`: counters for calls,
   catch-all `fallbacks`, `net_missing`, and the first traceback verbatim.
   Free on the happy path (one dict increment against ~1 ms of decision time),
   one line per game, never per-decision spam. **Measured locally over 733
   selects: `OK calls=396 fallbacks=0 net_missing=0` (v5) and `calls=337`
   (v4)** — 🔴 **the first DIRECT confirmation the net is live**, where §8g
   could only argue it from a 40.7% index-0 rate against the 100% a real
   fallback would show.
   ⚠ **Still to do: nothing prints it yet on Kaggle.** `build_submission.py`
   must emit `health_line()` once per game for the log to exist, and that costs
   a submission slot to verify — **the user's call, not mine.**
   🔴 **SUB-ITEM 2 IS MOOT AND SHOULD BE STRUCK:** "rule firing rates in the
   wild" cannot be measured, because the shipped bundle pins
   `chip_targeting/energy_spread/counter_source = False`. **There are no rules
   firing.** §8ac's re-weighting makes that pinning look better, not worse
   (rules are −18 Elo at the real weights), so the sub-item dies rather than
   becoming urgent.
   3. **pool usage** — we claim 0.1 s of 600 s; confirm it on the real harness;
   4. ⚠ **"cite a reason per action" is the expensive one** — ~1 ms/move ×
      thousands of selects is a lot of stdout, and **the log size cap and
      retention are UNKNOWN and must be checked before designing a format.**
      Cheap 90%: log the net's **top-1 logit margin** + a one-byte code for who
      decided (net / which rule / fallback), aggregated per game. That also buys
      the **margin distribution on real ladder states vs arena states** — a
      covariate-shift instrument for free.
   ⚠ **Design rule: compact per-game summary + rare event lines, never
   per-decision spam.**

**4. 🎬 THE TRAJECTORY RECORDER — build this before 5 and 6; it unblocks both,
   plus the user's own inspection.** 🔴 **`arena.py` archives ONE SUMMARY ROW PER
   GAME** — winner/turns/selects/latency/pool (`scripts/arena.py:281-294`).
   **No observations, no actions, no trajectories.** So today there is nothing to
   watch and nothing to learn from. Add an optional recorder to
   `harness.play_game` that (a) accumulates `obs` + chosen action per select and
   (b) calls `game.visualize_data()` before `battle_finish()`. **Port the old
   repo's `visualizer.html` so the user can watch games in the official viewer.**
   ⚠ Keep it **opt-in** — the A/B path must stay byte-identical and fast, and per
   §8aa's methods rule, **if this is meant to be a no-op for existing runs, prove
   it with an equivalence test, not with the arena.**

**5. 🤖 RL — SIZE THE VARIANCE FIRST, DO NOT BUILD.** The user's framing is
   **fine-tuning an already-decent clone on its own outcomes**, which is a
   different cost regime from the league self-play §0 declined — and §0 only ever
   considered the from-scratch version. ⚠ **Also correct the record with the
   user's own recollection**: RL did not fail against rule agents; **what matches
   that description is `search`.** The pre-registered probe, before any training
   code: **with the item-4 recorder, measure how many games the terminal-outcome
   signal needs to separate two policies of KNOWN Elo separation** (we have
   several — v4 vs `v4ctrl` at +37, the `no3` ablation at −36, and a **measured
   seed-only null at ±13**). **Kill criterion: if separation needs more games
   than ~1.4 cores can produce in the remaining days, RL dies for a few CPU-hours
   instead of a week — and it dies with a NUMBER**, which is the thing §2 never
   had. ⚠ **And whatever the probe says, it is a report chapter**: a retracted
   negative, re-derived honestly, is exactly §5's material.

**6. 🕵️ AUDIT THE ARENA OPPONENTS (user wants to watch first, then I analyze).**
   The five anchors carry **71.5% of every weighted verdict in this repo** and
   they were **imported, not written by us — nobody on this project has ever
   watched one play.** Gated on item 4. ⚠ Relevant to item 5 too: our anchors are
   rule pilots and our own nets, so **which opponent we generate exploration
   against decides what the data can teach.**

**PARKED FROM DAY 14 — not cancelled, and item B has a closure condition:**
   - **B. The centred option encoding** — append `opt_enc − mean(opt_enc)` to
     each **option** rather than pooling into the state (§8aa pooled on the
     *state* side, where the summary must survive the state MLP before it can
     affect a ranking; centring puts the comparison directly in the vector the
     head scores). ~20 min of compute. **If it lands ≤ +15 Elo, DECLARE THE
     FEATURE AXIS CLOSED** and write it up as a three-generation
     diminishing-returns curve (+115 → +37 → +14) — a better chapter than a
     fourth null.
   - **C. Track C deck work — 20% of the rubric and NOW FOUR SESSIONS UNTOUCHED.**
     Only one decklist variant has ever been A/B'd (0.490, null).
     🔴 **ITS CONCRETE LEAD DIED ON DAY 15, BY SIZING, BEFORE ANYTHING WAS
     BUILT — and that is rule 14 working, not a setback.** The lead was:
     Archaludon runs **Full Metal Lab ×4 (card 1244)**, we run **Spikemuth Gym
     ×4 (1259)**, and `WALL_POKEMON = {345}` models neither. It was promoted on
     *"Archaludon is 10.1% of the field and our worst real matchup"*. **At our
     current rating Archaludon is 8.0% of the field and 0 of 47 games above
     rating 900** (§8ac) — the tech would serve a band we are leaving. ⛔ Do not
     build it. **Same sizing kills B3's Mega Lucario instance (4.0%, also 0/47
     above 900).**
     ⚡ **What Track C should aim at instead, in order:**
     1. **The MIRROR — 33.3% of our field, 51.1% above rating 900, and rising
        with every point we gain.** A deck edge in the mirror is worth more than
        one anywhere else on the board, and it is the matchup our A/Bs measure
        best. Nobody has ever asked what beats our own 60.
     2. **Cynthia's Garchomp ex (6.7%) and Dragapult ex (5.3%)** — 12.0%
        together, more than Crustle + Lucario, and **neither has an anchor**.
        `decks/dragapult_ex.py` already exists; the pilot notebook is in
        `notebooks/`. Build the anchor before the deck opinion (rule 12).
     3. **The stewardship write-up is owed either way** — "we measured a change
        and kept the list" is deck analysis, and this closure is exactly that.
   - **D. `report/STRATEGY.md` — one edit per session, minimum.** §6 (opponent
     modelling) is still *in progress*; §8 needs the v5 entry. ⚡ **Day 14 already
     handed it a chapter for free: the self-play retraction belongs in §5's
     process-failure section**, next to the three failures already written there.

<details><summary>Day 14's plan as it was set at the end of day 13 (superseded — the day was idle by instruction; B and C are parked above)</summary>

> ## 📍 THE SITUATION
>
> - ⚡ **RANK 268 / 6,088 AT 923.0 — our best live number ever, and it is still
>   climbing.** `55156480` (the v4 state block) read **489.3 → 853.4 → 822.3 →
>   894.7 → 923.0** over its first 3 hours. **We were 465/6,075 at 864.1
>   yesterday.** Top is Majkel1337 1251.3, then Sixth Sense 1181.7.
>   ⚠ **923.0 IS NOT A SETTLED NUMBER** — rule 2 wants two *agreeing* readings
>   ≥1 h apart and these disagree because the agent is still converging. The
>   P4b restore took 4 h. **Re-read before quoting it.**
> - Active pair: **{`55156480` v4 923.0, `55129730` P4b 836.4}**. v3 is evicted
>   and frozen at 864.1. **v4 has now beaten every number this project has
>   produced except the original P4b's board-inflated 952** (§8p).
> - 🔴 **AND THE LADDER DOES NOT ADJUDICATE §8z** — v4 was +16.5 Elo weighted,
>   far below the ±50–100 the LB resolves. **A 60-point climb is not evidence
>   the block works**; the arena at n=4,000 with a seed control already answered
>   that, and this is a board that moved 3,000 → 6,088 entrants in a week.
> - ⚡ **v5 WAS SUBMITTED AS `55160229`** (`dist/submission_bc-grimmsnarl-netspolicy_20260801-163829.tar.gz`,
>   `NET_OK opt_in=37 state_in=708` — 536 + the 172-wide pool, so the block is
>   live in the bundle and not silently sliced off; sha verified against
>   `out/policy_v5.npz`). Active pair becomes **{`55160229` v5 climbing from
>   μ=600, `55156480` v4}**; **P4b is evicted.**
> - 🔴 **AND THE REASONING WAS CORRECTED MID-SESSION — read this, it is a
>   decision-framing error, not a new measurement.** The first verdict was "do
>   not submit: +7.3 weighted, negative on 2 anchors of 5, wrong shape". That
>   answers **"is v5 better than v4?"** (no) when the question a submission
>   actually asks is **"is v5 better than what it EVICTS?"** Eviction is by
>   recency, so v5 displaced **P4b — 836.4 against v4's 908–923, dominated on
>   the displayed score (best ACTIVE) and last of everything we own in the
>   arena (§8k).** v4 keeps its rating and stays active throughout, so the
>   displayed score cannot fall. ⇒ **The +50 bar was written when slots were
>   scarce and every submission evicted something valuable; the user relaxed
>   exactly that premise, and the bar was still being applied to the old one.**
>   ⚠ **Standing correction: before quoting the bar, name the agent the
>   submission would EVICT.** A candidate that loses to our best can still
>   dominate our worst.
>
> **Deadlines: sim closes 2026-08-17 (16 days). Report due 2026-09-14 (44 days).**
> **Rubric: Model 70% (LB is ONE bullet of five) + Deck 20% + writing 10%.**

#### ✅ Done on day 13 — two results, and together they are the best report material the project has

- **§8aa — the v5 pooled option-set block: the deep-sets fix, and it BARELY
  PAYS.** Every option is scored independently against one shared state vector,
  so the net has never seen the option *set*. Mean/max pool of the option
  encodings + count scalars, appended after the v4 block (`--pool`).
  **Agreement 71.0% → 72.7% — 214 more correct decisions of 12,939, the largest
  agreement gain this project has ever produced — for +14 Elo pooled over two
  seeds**, one noise-width, mixed-sign across the anchors.
- 🔴 **THE PAIR IS THE FINDING, and it is now measured in both directions:**

  | intervention | Δ agreement (of 12,939) | Δ Elo | Elo per decision |
  |---|---|---|---|
  | **v4 state block** (§8z) | **+8** | **+37** | 4.6 |
  | **v5 pooled option set** (§8aa) | **+214** | **+14** | **0.07** |

  **The exchange rate between fit and strength differs 70× between two
  interventions run a day apart on the same corpus.** ⇒ **`val_top1` is not a
  screening metric in either direction. Nothing may be promoted or killed on
  it.** This is rule 3 with both signs paid for.
- ⚡ **§8ab — the v4 ablation, and it validates the METHOD rather than the
  block.** `--drop-x` zeroes a member's columns (identical arch, params, init,
  rows, seed; `x_mask` stored in the npz so inference matches training):
  - **Drop any ONE of `turnActionCount` / stadium / effect card → within noise**
    (0.527, 0.526, 0.483 vs full v4).
  - 🔴 **Drop all THREE → 0.449 [0.427, 0.470], −36 Elo, disjoint.** They are
    **mutually redundant and jointly necessary** — and they are essentially the
    whole +37.
  - ⚡ **The five leftover members alone are WORSE THAN NO BLOCK AT ALL**
    (0.469 vs `v4ctrl`, −22 Elo, disjoint). **The three that went through §8y's
    sizing step carry everything; the five that skipped it are negative.**
    ⇒ **Derive and size. Do not bundle.**
- ⚠ **A caveat that touches every weighted table in this repo:** head-to-head
  Elo among these nets **orders consistently but compresses ~23 points over two
  hops** (v4−ctrl +37, ctrl−no3 +22, v4−no3 measured **+36** against an additive
  +59). **Weighted five-anchor totals are ordinal, not arithmetic.**
- 🔧 **A methods rule bought the hard way (§8aa's last section).** The refactor
  that enabled the pool moves the option encoding ahead of the state MLP *for
  every net*. A regression A/B read **0.503** where §8z had 0.567 — which would
  have meant the live net was broken. **The arena cannot settle that: it is not
  deterministic run to run.** A direct equivalence test (load the pre-edit
  module from git, same observations, compare scores) said **max |old − new| =
  0.000e+00 over 588 selects**. ⚡ **When a refactor is supposed to be a no-op,
  prove it with an equivalence test, not the noisy end-to-end instrument.**
- **Report:** `STRATEGY.md` §4c (audit-by-enumeration + the ablation), §4d
  (§8z's decoupling), §4e (§8aa's converse). Three new chapters.

#### The day-14 order of work

**A. 📈 READ THE LADDER FIRST, TWICE, ≥1 h APART.** v4 is mid-climb at 923.0.
   The question is where it settles — **against 864.1, which is what v3 reached
   on the same board size.** ⚠ Do not quote 923.0 as converged, and do not read
   a rank change as evidence about the v4 block (rule 2).

**B. 🔬 THE FEATURE AXIS IS NARROWING — one concrete lead left, then stop.**
   Three generations: option binding **+115 Elo** (§8f), state block **+37**
   (§8z), option-set pool **+14 and mixed-sign** (§8aa). **The returns are
   falling by roughly 3× a generation and the next one lands under the noise
   floor.** The single untested variant worth one day:
   1. ⚡ **The CENTRED option encoding** — append `opt_enc − mean(opt_enc)` to
      each **option** rather than pooling into the state. §8aa pooled on the
      *state* side, where the summary must survive the state MLP before it can
      affect a ranking; centring puts the comparison directly in the vector the
      head scores. **Different mechanism, same cheap append-and-slice, ~20 min
      of compute.** If it also lands ≤ +15 Elo, **declare the feature axis
      closed and write it up as a three-generation diminishing-returns curve** —
      which is a better chapter than a fourth null.
   ⛔ **Do NOT re-open capacity (§8w), demonstrator selection (§8u), data volume
   (§1) or search (§2).** Six axes are dead; this is the seventh probe of the
   one that lives, and it is nearly spent.

**C. 🃏 TRACK C DECK WORK — now the largest untouched item on the board.**
   20% of the rubric, and **only ONE decklist variant has ever been A/B'd**
   (0.490, null). It has not been reached on days 12 or 13. **Promote it to
   first item if B's lead does not land.** Concrete starting point found on
   day 13: **Archaludon runs Full Metal Lab ×4 (card 1244), a stadium**, and we
   run **Spikemuth Gym ×4 (1259)**. Playing ours removes theirs, and
   `WALL_POKEMON = {345}` does not model Full Metal Lab's damage reduction at
   all. **Audit before rule (rule 14):** how often do we hold Spikemuth Gym
   while Full Metal Lab is in play? Archaludon is 10.1% of the field, our worst
   real matchup (45.5% over 11 games), and the anchor v4 *and* v5 both barely
   move (+7, −6).

**D. 📝 `report/STRATEGY.md` — one edit per session, minimum.** §6 (opponent
   modelling) is still marked *in progress* and §8's negative-results list now
   needs the v5 entry. Day 13's three chapters are written.

</details>

<details><summary>Day 13's plan and situation (superseded — all four items ran; A/B/D done, C not reached and re-stated above)</summary>

> ## 📍 THE SITUATION (as of day 13's start)
>
> - **Rank 465 / 6,075, score 864.1** (two readings 12:02 and 13:03 BST: 869.7
>   then 864.1 — agreeing, rule 2 satisfied). Top is **Majkel1337 at 1300.6**,
>   which is 135 points clear of the old top and is new.
> - 🔴 **THE ACTIVE-PAIR FACTS IN THE OLD BOX WERE STALE AND BACKWARDS.** Live
>   per-submission scores read 08-01: **`55116557` v3 = 864.1** (our best, and
>   still climbing) and **`55129730` P4b-restore = 824.3**. ✅ **That is §8i's
>   arena prediction confirmed on the ladder** — the sweep said v3 +36 Elo over
>   P4b, the ladder says +40, both active, both converged. **The day-9 "B1 lost
>   130 points" story is now fully inverted.**
> - ⚡ **A NET WAS SUBMITTED ON DAY 12 — the first since 07-31: `55156480`**,
>   the v4 state block (`dist/submission_bc-grimmsnarl-netspolicy_20260801-131057.tar.gz`,
>   `NET_OK opt_in=37 state_in=536`, sha verified against `out/policy_v4.npz`).
>   It **evicts `55116557` (v3, 864.1)**; the active pair becomes {`55156480`
>   climbing from μ=600, `55129730` P4b 824.3}, so **the displayed score will
>   DROP to ~824 for ~4 h** — expected, not a regression. It read **489.3** at
>   ~5 minutes old, which is **7 games of TrueSkill and means nothing** (the P4b
>   restore went 600 → 715.9 → 833.9 over 4 h). §8z.
> - ✅ **DAY 12 BROKE THE PLATEAU, and it did it on the one axis that has ever
>   worked.** The v4 state block beats its own byte-identical control
>   **0.567 [0.545, 0.588] n=2000**, replicates at a second seed
>   (**0.539 [0.518, 0.561]**), against a **measured seed-only null
>   (0.482 [0.460, 0.504])**. Pooled **≈ +37 Elo**. Better on **5 anchors of 5**.
> - 🔴 **AND IT MOVED HELD-OUT AGREEMENT BY EIGHT DECISIONS OUT OF 12,939.**
>   Rule 3's converse, measured for the first time: **the agreement metric the
>   whole B7 programme rested on is blind to a 37-Elo intervention** (§8z).
> - ✅ **The deck is NOT the bottleneck** (§8o); "clone better demonstrators" is
>   not the lever (§8u); capacity is not the lever (§8w). **State features are.**
>
> **Deadlines: sim closes 2026-08-17 (17 days). Report due 2026-09-14 (45 days).**
> **Rubric: Model 70% (LB is ONE bullet of five) + Deck 20% + writing 10%.**
> **Winning is not the same as ranking.** Read §0 of `ROADMAP.md` before deciding
> that a rank point is worth more than a report chapter — the competition
> description says outright that a mid-tier LB with deep analysis can win.

#### ✅ Done on day 12 (read before planning day 13)

- **§8x — the encoding ceiling, computed rather than argued.** Bitwise-identical
  options get identical logits from any net, so `Σ(1/g)/N` bounds top-1 for this
  layout: **95.6%, against the clone's 69.8%.** So §8w's "the residual is the
  encoding" **cannot** mean the answer is inexpressible — un-expressibility is at
  most 4.4 of the 30.2 points. ✅ And every tie is **two copies of one card in one
  role**, so `context_accuracy.py --equiv` now counts those as hits: honest
  agreement is **71.0%**, TO_HAND **67.1%** not 61.2%.
- **§8y — the feature audit BY ENUMERATION** (`p18_missing_state_audit.py`), and
  it **retracted the candidate list three files had carried since day 10**: turn
  number, prizes and both hand counts are all encoded already (`features.py`
  88–99). **Rule 15, second instance — caught before anything was built.** Two
  more died on sizing (`remainDamageCounter` constant at 100% of decisions,
  `remainEnergyCost` at 99.1%).
- **§8z — the v4 state block: BUILT, MEASURED, REPLICATED, SUBMITTED.**
  `turnActionCount` + the select's **effect card** + the **stadium** + `retreated`
  / `stadiumPlayed` + tool counts + bench cap + pool size. Corpus `pds_v4` is
  **byte-identical to `pds_v3r`** on every pre-existing array, and `--no-extra` is
  the control on those identical rows.
- **A noise floor exists now.** `train_policy.py --seed`; two identical-recipe
  controls at different seeds measure **0.482 [0.460, 0.504]** — a null. **Every
  net-vs-net number in this repo previously had an unmeasured confound.**
- **Report:** `STRATEGY.md` §4b (the ceiling, new), §8's capacity bullet narrowed.

#### The day-13 order of work

**A. 📈 READ THE LADDER FIRST, TWICE, ≥1 h APART.** `v4` needs ~4 h to converge.
   ⚠ **The expected path is DOWN then UP**: displayed drops to ~824 (P4b) while
   v4 climbs from 600. **Do not react to the dip.** The question that matters is
   where v4 settles against **864.1**, the number v3 reached.
   🔴 **And whatever it reads, it does not adjudicate §8z** — +16.5 Elo weighted
   is far below the ladder's ±50–100 (rule 2). It was submitted because it is
   better on 5/5 anchors and slots are not scarce, not because the LB can see it.

**B. 🔬 THE FEATURE AXIS IS LIVE AGAIN — WORK IT, it is the only one that has
   ever paid, and it has now paid TWICE (§8f, §8z).** Two concrete leads, both
   derived rather than guessed:
   1. ⚡ **THE NET NEVER SEES THE OPTION SET.** Every option is scored
      independently against a shared state vector, so it cannot know whether it
      is choosing among 3 Trainers or 40 deck cards. That is *why* the effect
      card paid (§8y). **The direct fix is a pooled summary of the option
      encodings (mean/max) concatenated into the state** — deep-sets, one extra
      block, the same append-and-slice trick. **This is the declined appendix's
      Set-Transformer plank in its cheapest possible form**, and §8z is the first
      evidence it would pay.
   2. **The v4 block was shipped whole; nobody knows which member did the work.**
      An ablation (drop `turnActionCount` alone, drop the effect card alone) is
      two trainings and two A/Bs, and it is a report table either way.
   ⚠ **Carry the noise floor in**: ±13 Elo between seeds. **Any ablation arm must
   clear that**, so run each at n=2000 and prefer two seeds.

**C. 🃏 Track C deck work FOR DECK SCORE (20% of the rubric, still untouched).**
   Only **one** decklist variant has ever been A/B'd (0.490, null). Re-aim at
   **Archaludon** — it is the one anchor where v4 barely moved (+7 Elo) and our
   worst real matchup (45.5% over 11 games). Full Metal Lab is a second
   damage-reduction effect `WALL_POKEMON = {345}` does not model.

**D. 📝 `report/STRATEGY.md` — one edit per session, minimum.** Day 12 handed it
   two strong chapters that are **not** written yet: **§8z's decoupling** (an
   intervention worth 37 Elo that moves agreement by 0.06 pp — this is the
   sharpest statement of rule 3 the project has) and **§8y's method** (a feature
   audit done by diffing the observation against the code, which retracted a list
   three files were asserting).

> ✅ **A, B and D RAN ON DAY 13; C did not and is re-stated as day 14's item C.**
> A: the ladder was read four times and v4 climbed to **923.0, rank 268/6,088**.
> B: **both** leads ran — the pooled option set (§8aa, **+14 Elo for the largest
> agreement gain in the project**) and the ablation (§8ab, **the three derived
> members are jointly necessary and the five unsized extras are negative**).
> D: three chapters written (§4c, §4d, §4e).
> ⚠ **Item B's framing was right and its prediction was wrong in a useful way**:
> it called the pooled summary "the first evidence it would pay", citing §8z.
> It moved the *fit* more than anything ever has and bought a noise-width of
> strength — **which is exactly the decoupling §8z had just demonstrated, applied
> in the opposite direction and not anticipated.**

</details>

<details><summary>Day 11's order of work (superseded — all five items ran)</summary>

#### ✅ Done on day 11

- **Item A shipped: every corpus row carries its demonstrator's LB rating.**
  `build_policy_dataset.py --ratings` (+ `--exclude`, `--aliases`), 94–98% seat
  coverage. ⚠ **The first build silently lost 24.6% of d26 seats and 182 of the
  198 misses were ONE team — the LB's #1, appearing as `James Cox`, as
  `zoroark190` (a member username) and as the merged `James Cox & Henry Chao`.**
  Fixed by exact member-username matching plus a hand-verified
  `replays/team_aliases.tsv`. **A census keyed on a display name splits your
  most valuable demonstrator into three.**
- **`p15_rating_curve.py`** (agreement vs rating, with a `--seen-from` exposure
  control) and **`p16_policy_disagree.py`** (the covariate-shift discriminator).
- **`p9_field_census.py --us / --emit-players`** — censuses any named seat's
  opponents and writes the same-deck ones to a `--players-file`. **This makes
  the day-10 control population reproducible**; the original 48-name list was
  never on disk.
- **`train_policy.py --rating-temp / --rating-min / --init`** — per-row weighted
  listwise loss (ESS reported before training) and warm-start fine-tuning.
- **Three concluded experiments: §8r, §8s, §8t/§8u.** Both B7 arms killed
  against a bar set before the first run.

#### The day-12 order of work

**A. 📝 `report/STRATEGY.md` IS NOW THE HIGHEST-VALUE WORK IN THE PROJECT, and
   that is a measured claim rather than a consolation.** The LB is inside its
   own noise band (§8k, confirmed on the ladder by §8p), five training axes are
   dead, and the report is 30%+ of the rubric before counting the soundness /
   consistency / robustness bullets inside Model's 70%. **Day 11 alone handed it
   a genuinely publishable result**: *agreement with a demonstrator measures
   distance from the fitted mode, not skill* — with a peak, a zero-exposure
   control group, a symmetry test against covariate shift, a positive control at
   1.7%, and two pre-registered interventions that failed in the predicted
   direction. **Write §7b.1/§7b.2 up properly and start §6 (opponent modelling)
   from §8r.** One edit per session remains the floor, not the target.

**B. 🔬 THE FEATURE AUDIT IS NOW THE WHOLE ENGINEERING TRACK, and §8w promoted
   it from "the only axis that ever paid" to "the only axis with a live
   mechanism".** Capacity is ruled out (8.2× params, no gain), demonstrator
   selection is ruled out (§8u), data volume was already dead (§1) — **by
   elimination the residual is what the option encoding cannot bind.**
   The day-10 list is still unworked and is still the best candidate set:
   read `agents/sa/optfeat.py` and `features.py` against
   `context_accuracy.py`'s **MAIN misses (3,930 of 6,424)** and ask §8f's
   question — is the input **absent** (informational) or **present but
   unbindable** (representational)? Only the second has ever paid.
   ⛔ **The cheap-candidate list this item carried is RETRACTED (§8y): opponent
   hand size, prizes remaining and turn number are ALL already encoded**
   (`features.py` lines 88–99); only the stadium was really absent.
   `scripts/p18_missing_state_audit.py` now derives the list by diffing the
   observation against what `featurize()` reads, and sizes each candidate.
   ⚡ **§8s gives this a new instrument**: `p16_policy_disagree.py` names the
   contexts where a stronger policy actually diverges from ours — **MAIN 45.8%,
   DAMAGE_COUNTER 30.5%, SWITCH 27.6%** — so the feature audit now has a
   ranked target list derived from a 1163-rated policy rather than from guessing.

**C. 🃏 Track C deck work FOR DECK SCORE (20% of the rubric, and untouched).**
   §8o closed it as a rank lever, which is exactly why it should now be done
   honestly as deck analysis: only **one** decklist variant has ever been A/B'd
   (0.490, null). Re-aim it at our real worst matchups — **Archaludon (45.5% of
   11 real games)** and **Mega Lucario (50%)** — both of which now have anchors.

**D. ⛔ DO NOT SUBMIT.** Nothing clears **+50 Elo weighted over the five
   anchors**; the two newest nets are **−55** and **−92**. Every submission
   evicts a live agent (§8h). ⚠ **And do not submit `b7_ntum` "to see what the
   ladder says"** — that trade spends a slot and evicts a live agent to test a
   net the arena puts 92 Elo down, against an instrument that resolves ±50–100.

> ✅ **ALL FOUR RAN ON DAY 12.** A: two chapters written (§4b + the §8 narrowing).
> B: the audit was done and it **retracted its own candidate list** (§8y), then
> the derived replacement measured **+37 Elo pooled** (§8z). C: not reached —
> re-stated as day 13's item C. D: **deliberately broken, with the reasoning
> written down in §8z** — v4 is below the +50 bar at +16.5 weighted, and was
> submitted anyway because it is better on 5/5 anchors and the user relaxed
> submission scarcity. **The bar was re-priced, not the evidence.**

</details>

<details><summary>Day 10's order of work (superseded — item B ran and is closed by §8u)</summary>

#### ✅ Done on day 10

- **`55129730` P4b restore: 833.9 at 4.0 h vs the original's 958.2 at ~4 h.**
  Question closed by experiment, for the price of one submission. §8p.
- **Two expert dumps ingested and verified** (`replays/sixth_sense_31-07-2026`
  227 games, `replays/ntumlnoob_31-07-2026` 330 games). Both teams play our exact
  60. `selected` is present for third-party seats, so **the BC pipeline works on
  them unchanged**.
- **`build_policy_dataset.py --player / --players-file`** — builds a corpus from
  named seats only. Corpora on disk: `artifacts/pds_expert` (Sixth Sense, 19,107
  rows), `artifacts/pds_ntum` (ntumlnoob, 27,318), `artifacts/pds_grimm_ctrl`
  (48 other grimmsnarl pilots, 10,498 — **the same-deck control**).
- **`context_accuracy.py --all-rows`** — scores every row, not the trainer's
  `gid % 20` split. **Required for a corpus the net never trained on**; without it
  you silently score 5% of the data.
- **The agreement-vs-rating result** (§8q), plus two explanations killed:
  familiarity (`haggle`: 0 corpus seats, 75% agreement) and one-team idiosyncrasy
  (the ntumlnoob dump was fetched to break exactly that tie).

#### The day-11 order of work

**A. ⚡ TAG EVERY CORPUS ROW WITH ITS DEMONSTRATOR'S LB RATING — do this first,
   it gates B and C.** Team names are in every replay's `info.TeamNames`; the
   full leaderboard is one `competition_leaderboard_download` call. Store a
   per-row `rating` (and `submissionId` where the dump has `episodes_meta.json`).
   Immediate payoff: turn day 10's three points into a **proper agreement-vs-
   rating curve over all 1,603 corpus games** — a report figure either way.

**B. 🔬 THE TWO TRAINING EXPERIMENTS, in this order.** Bar for both, pre-
   registered: **+50 Elo weighted across the five anchors or it is a chapter, not
   a submission** (§8k).
   1. **Rating-weighted clone** on the full corpus — keeps all 248,985 rows while
      fixing mode-averaging. **Most likely of the two to work**, and original
      enough to be its own report chapter.
   2. **Single-expert clone** — fine-tune v3 on `artifacts/pds_ntum`, then on
      ntum + Sixth Sense. ⚠ 27k rows against a 249k corpus: **underfitting is the
      expected failure mode** — fine-tune, do not train from scratch, and early-
      stop on a held-out expert split.
   ⚠ **Carry the standing prior in**: three axes of more/better training measured
   null or negative (§1), and the only thing that ever moved the clone was
   representational (§8f). This is a different axis — demonstrator *selection* —
   but the prior is not friendly.

**C. 🔍 THE ALTERNATIVE EXPLANATION, which a good report must address first:
   covariate shift.** Some of the 40% miss is BC's compounding-error problem, not
   a copyable policy. **Cheapest discriminator: score the expert net on OUR
   agent's trajectories and ours on theirs** — if disagreement is symmetric it is
   policy difference; if it collapses when the states are ours, it was shift.

**D. 📝 `report/STRATEGY.md` — one edit per session, minimum.** §7b already
   argues "the ceiling is the clone, not the deck"; day 10 gave it a much better
   instrument (identical 60 at +310 rating) and a new figure (the curve). Also
   worth a methods line: **`--all-rows` and the `--player` filter are the kind of
   detail the soundness bullet rewards.**

**E. ⛔ DO NOT SUBMIT** unless something clears **+50 Elo weighted over the five
   anchors**. Every submission evicts. The restore is closed; do not spend
   another slot re-testing a settled question.

> ✅ **ALL FIVE ITEMS RAN ON DAY 11.** A shipped; **B killed both arms** (−55 and
> −92 Elo, §8t/§8u); **C answered — covariate shift ruled out** (§8s); D done
> (§7b.1 rewritten, §7b.2 added); E honoured — **nothing was submitted.**
> ⚠ Note item B's prediction was **wrong in an instructive way**: it called the
> rating-weighted arm "most likely to work" and expected underfitting to sink the
> expert arm. The rating-weighted arm moved expert agreement by **0.1 pp** — it
> did nothing at all — and the expert arm imitated *successfully* (+7.3 pp held
> out) and lost anyway. **The standing prior in the same item was the part that
> held.**

</details>

<details><summary>Day 9's close and the day-10 order of work (kept for the record; A and C below are still live and were re-stated in the day-11 list)</summary>

#### ✅ Done at the end of day 9 (read before planning day 10)

- ~~**`55129730` — THE P4b RESTORE IS LIVE AND CLIMBING** (600 → 715.9 → …).~~
  ✅ **SETTLED day 10: it converged to 833.9 at 4.0 h against the original's
  958.2 at ~4 h — the same code, 124 points lower, on a board 2,000 entrants
  bigger** (§8p). The reasoning below was sound and the experiment was worth its
  submission: it converted a three-day argument into a measurement.
  Active pair is `{55129730, 55116557 v3 818.1}`; **P6a is evicted, frozen 841.5**.
  **The reasoning reversed an earlier recommendation of mine and the user was
  right to push:** the LB said P4b 952 vs P6a 846 for three days — a 105-point
  gap at/above the resolution limit — and "the board grew so it is not
  comparable" was a weaker argument than I presented. Risked 33 points (P6a →
  v3's floor) to chase ~100.
- ❌ **Boss's Orders rule #5 (`bossPrize`) — NULL, and the card is now properly
  closed.** All three anchors overlap; weighted **+6 Elo**. The user's
  observation was correct and the rule fixes exactly the defect they described
  (fires on 28.6% of plays vs a 29% measured misplay rate, corroborated three
  independent ways) — **it just decides ~0.09 prizes per game, 1.5% of a 6-prize
  game, against an A/B that resolves 0.021.** ⚠ **Rule 14 was violated: built
  first, sized after. The sizing takes two minutes and predicts the null
  exactly.** `EVIDENCE` §6.
- 🔧 **`scripts/p14_prize_audit.py`** — automates the misplay hunting that used
  to require the user watching games. It re-found the Boss's Orders defect
  independently. ⚠ **Its "did not attack" bucket is NOT trustworthy** (14.9–27.7%
  against P5c's established 3,683/3,683); `_available()` prices 180 damage
  without checking our Active can legally attack. **Fix that before using it.**

#### The day-10 order of work

**A. ⚡ HUNT THE NEXT REPRESENTATIONAL DEFECT — the only proven rank lever.**
   B1 is the single largest effect this project has produced, and it was found by
   **reading the feature code against a premise nobody had checked** (§8f), not
   by guessing. Do that again, deliberately:
   - `python -X utf8 scripts/context_accuracy.py` — the per-context miss table.
     **MAIN holds 3,930 of 6,424 misses.** That is the target.
   - Read `agents/sa/optfeat.py` and `features.py` **against** the top miss
     contexts and ask the §8f question for each: is the input **absent**
     (informational) or **present but unbindable** (representational)? Only the
     second kind has ever paid.
   - ⛔ **This list was WRONG and was carried for two days — see §8y.** Hand
     size, turn number, prizes and the opponent's discard are all encoded
     already (`features.py` 88-99, and `opp_discard` is an id bag). Use
     `p18_missing_state_audit.py`, which derives the list instead of recalling
     it.
   - **Bar: any candidate must clear +50 Elo weighted across the five anchors
     before it is worth a submission** (§8k). Below that it is a report chapter.

**B. 📝 `report/STRATEGY.md` — the highest EV per hour in the project, and it is
   NOT a consolation prize.** Deck 20% + writing 10% + the soundness /
   consistency / robustness bullets inside Model's 70% dwarf the LB's one bullet,
   **and the LB is stuck inside its own noise band while the report is not.**
   Day 9 alone produced three strong chapters — the censored sampling frame
   (§8i), matchup-conditional rules at **+47 / −51 Elo** (§8j), and the
   everything-is-within-36-Elo result (§8k). ⚠ **One edit per session, minimum.**

**C. 🔬 B4 — DECIDE, do not drift.** The prototype exists and **loses 0.075
   [0.026, 0.199] n=40** (§8n). Two bugs eliminated; the live hypothesis is a
   *design* flaw — maximising end-of-OUR-turn value cannot see the opponent's
   reply. **Either** test the one-ply-reply fix (small change to `_rollout`'s
   terminal evaluation) **or kill it and write it up.** It is already a good
   negative-result chapter either way. Do not let it consume day 10.

**D. 🃏 Track C deck work — reframe it, then do it FOR DECK SCORE.** §8o proved
   it is not a rank lever, so stop selling it as the counter-meta fix. It is 20%
   of the rubric on its own and **only ONE decklist variant has ever been A/B'd**
   (0.490, null). Also **re-aim it**: Track C is written against Crustle, but our
   worst matchups are **Archaludon (45.5% over 11 real games)** and **Mega
   Lucario (50%)**, and anchors for both now exist.

**E. ⛔ DO NOT SUBMIT** unless something clears **+50 Elo weighted over the five
   anchors**. Every submission evicts, and the pair {P6a, v3} is already the
   arena's top two (§8k). The restore of P4b is **closed — do not reopen it.**

</details>

<details><summary>Day 9's completed items (kept for the record)</summary>

**Day 9 answered the question day 8 ended on.** The blocking problem — "no arena
number in this repo predicts ladder strength" — is **closed** (§8i): the arena
predicts fine, the anchor set was wrong, and it is now fixed.

0. ✅ **THE 5-ANCHOR SWEEP IS COMPLETE** (n=2000 per cell, 71.5% of the field).
   **Weighted by field share, v3 is +36 Elo over P4b** — it wins four anchors and
   loses only Mega Lucario. Δ Elo is `elo(v3 vs anchor) − elo(P4b vs anchor)`;
   the mirror row is a head-to-head, so its Δ is `elo(0.657)` directly — **do not
   compute that one as `elo(v) − elo(1−v)`, which doubles it.**

   | anchor | share | P4b | v3 | Δ Elo | weighted |
   |---|---|---|---|---|---|
   | `rule:alakazam5` | **22.0%** | 0.727 [0.707, 0.746] | 0.731 [0.711, 0.750] | **+4** dead heat | +0.8 |
   | mirror, head-to-head | 13.8% | (0.343) | **0.657 [0.636, 0.677]** | **+113** | +15.6 |
   | `rule:crustle` | 12.8% | 0.663 | 0.770 | **+92** | +11.8 |
   | `rule:v10` | 12.8% | 0.576 [0.554, 0.598] | 0.505 [0.483, 0.527] | **−50** 🔴 | −6.4 |
   | `rule:archaludon` | 10.1% | 0.621 [0.599, 0.642] | 0.669 [0.648, 0.690] | **+36** | +3.7 |
   | | **71.5%** | | | | **+36 Elo** |

   **And the ladder agrees, once compared honestly:** v3 819.8 vs P6a 845.0
   (both active, same time) = **−25**, against an agent still climbing, inside
   the LB's ±50–100. **Arena +36, ladder −25, instrument ±75: these are not in
   conflict.** The apparent 130-point contradiction came from comparing against
   a frozen 07-29 score (§8i).

   ⚠ **The one place v3 is genuinely worse is `rule:v10` (−50 Elo on 12.8%).**
   That is the live engineering lead — see item 3.

<details><summary>The sweep as it was being run (superseded, kept for the reasoning)</summary>

   **⚡ FINISH THE 5-ANCHOR SWEEP. Nothing should be submitted before it.** Two of
   the four runs are done; `p9_field_census.py`'s top 5 covers 71.6% of the field:

   | anchor | share | `bc:p4b,noSrc` | `bc:v3off,…` |
   |---|---|---|---|
   | `rule:v10` / `lucario_v10` | 12.8% | **0.576 [0.554, 0.598]** | **0.505 [0.483, 0.527]** |
   | **`rule:alakazam5`** / `alakazam5` | **22.0%** | **0.727 [0.707, 0.746]** | **0.731 [0.711, 0.750]** |
   | `rule:archaludon` / `archaludon_ex` | 10.1% | ⏳ running | ⏳ TODO |
   | `rule:crustle` / `crustle_v1` | 12.8% | 0.663 (§8c) | 0.770 (§8f) |
   | mirror (`grimmsnarl` v `grimmsnarl`) | 13.8% | — head-to-head — | ⏳ **running** |

   **v3 − P4b so far: Alakazam +0.004 (dead heat, CIs overlap), Crustle +0.107,
   Lucario −0.071.**

   ```powershell
   python -X utf8 scripts/arena.py play "bc:v3off,net=out/policy_b1_v3.npz,noChip,noSpread,noSrc" `
       rule:archaludon --deck-a grimmsnarl --deck-b archaludon_ex --matches 1000 `
       --archive out/arena/p9_v3off_vs_archaludon.jsonl
   ```

   ⚠ **Weight each anchor by its share before concluding anything.** A rule that
   wins 22% of the field and loses 12.8% is not "2 anchors to 1" — it is +9.2 pp
   of the field. That arithmetic is the whole point of the census, and it is the
   thing rule 12 was missing.

   🔴 **AND CHECK BOTH ARMS ARE THE SAME COMPARISON — this nearly went wrong.**
   §8f's mirror number (**0.661**) is v3 vs `out/policy_b1_ctrl.npz`, a
   **v2-feature net trained on the same `pds_v3` corpus**. That is *not* P4b
   (`lw2` net, `pds_v2` corpus, chip + spread rules **ON**). Dropping 0.661 into
   the column above as "v3 is +0.161 in the mirror" mixes a **feature ablation**
   with an **agent comparison**, and it flips the weighted verdict: done naively
   it totals ≈ +0.045 for v3, while the ladder says v3 is **132 points worse**.
   **The honest cell is `bc:v3off` vs `bc:p4b,noSrc` head to head — which had
   never been run in this project.**

   ✅ **It landed at 0.657 against the 0.661 that was being reused, so the reuse
   was harmless — but it was harmless by luck.** Run the cell you are actually
   weighting; it cost 12 minutes.

</details>

1. ⛔ **DO NOT RESTORE P4b — ANSWERED 2026-07-31 IN THE ARENA, FOR ZERO
   SUBMISSIONS** (§8k). All three agents swept across all five anchors, n=2000
   per cell, Elo relative to P4b:

   | agent | weighted | note |
   |---|---|---|
   | **v3** (rules off, `55116557`) | **+36** | active |
   | **P6a** (`55077709`) | **+7** | active, and our best LIVE score (845.0) |
   | **P4b** (`55072063`) | **0** | frozen 952.0 — **the arena ranks it LAST** |

   **The entire spread is 36 Elo and the LB resolves ±50–100** (§1). So restoring
   P4b would cost a submission, **evict `55077709` (845.0, our best active and
   still climbing)**, and restart at μ=600 for ~4 h — to install the agent the
   arena ranks last, on the strength of a **frozen** score earned on a board
   2,000 entrants smaller. **The active pair {v3, P6a} is already the arena's
   top two. No action needed.**

   ✅ Also settled: **`counter_source` is vindicated a second time** — P6a beats
   P4b by +24 Elo in the mirror and +7 weighted, independent of §8c's +0.052.

   🔴 **The strategic consequence, and it should steer the remaining 17 days:**
   if our best and worst agents differ by 36 Elo and the LB cannot see 36 Elo,
   **no further rule-sized improvement can move the rank.** The only levers big
   enough to clear the band are a materially better net or **ROADMAP B4**
   (turn-level sequencing — we use 0.1 s of the 600 s pool). **Another targeting
   rule is a report chapter, not a rank.**

<details><summary>The open-question framing this replaced (kept — the reasoning is report material)</summary>

   **⚠ THE P4b RESTORE IS NOW A GENUINELY OPEN QUESTION — DO NOT DO IT ON
   AUTOPILOT.** Every earlier version of this item assumed "952 > 837.5, so
   restoring is free value". **That premise is a frozen-vs-live comparison,
   which is exactly what §8i retracted.** The evidence now points both ways:

   | for restoring P4b | against restoring P4b |
   |---|---|
   | It *did* read **952.0**, the highest number this project has produced | That 952.0 was earned 07-29 on a **~4,000**-entrant board; the board is now **6,000**, and a frozen rating is not comparable to a live one (rule 2) |
   | The LB is the real referee and it liked P4b best | The **arena now covers 71.5% of the field** and says **v3 is +36 Elo over P4b** — and the arena's credibility was the only reason to doubt it |
   | | A restore **costs a submission and evicts** (`55077709`, 845.0). It restarts at μ=600 and needs ~4 h |
   | | We would be evicting the agent the arena ranks **highest** to install the one it ranks lower |

   **My read: this is now finely balanced and it is the user's call, not a
   default.** ⚠ **Do not treat "restore P4b" as settled just because three
   earlier versions of this file said so** — all three were written before the
   anchor set covered the field.

</details>

   **The decision that actually binds is item 2 (what is ACTIVE on 08-17), and
   there is time.** Nothing is at risk of being lost:
   - **Kaggle's copy of `55072063` is permanent** and keeps showing 952.0.
   - **P4b is rebuildable from git even without `dist/`** — `dist/**` is
     gitignored, but `agents/sa/policy_net.npz` **is tracked** and is the same
     lw2 net (`sha256 bba02a42…`), and the code is in history.
   - **You CANNOT re-activate an old submission.** The API has `competition_submit`
     and nothing like select/activate — "latest 2" is recency, not a choice. So a
     restore is always a *new* submission climbing from μ=600, whenever it happens.

   **What actually binds: the best agent must be ACTIVE at the 08-17 close and
   through the 08-31 continued-play window** (§8h). Two mild arguments against
   leaving it to the deadline: the climb takes ~4 h, and the field is growing fast
   (3,000 → 5,000 → **6,000** entrants in 3 days), so a late restore may not land
   on 952. ⚠ **And note that same growth is the reason 952 is not comparable to
   819.8** — it cuts both ways.

   ⚠ **Real gap worth closing cheaply: `out/policy_b1_v3.npz` and
   `artifacts/pds_v3/` are gitignored and exist ONLY on this disk.** Losing them
   means re-running the 4-day shard rebuild plus a 12-epoch train to get v3 back.
   **This is the one part of item 1 that is unambiguously worth doing now.**

   **The cheapest way to settle the restore question without spending a
   submission:** the arena already ranks P4b vs v3 (+36 Elo to v3, 71.5%
   coverage). The missing arm is **P6a** — the agent a restore would evict, and
   the only one whose live score (845.0) is comparable to v3's. Run
   `bc` (= P6a's exact config) against all five anchors and weight it. **If P6a
   ≈ v3, the restore evicts nothing and the only question is whether P4b beats
   both — which the arena says it does not.**

   ⛔ **SETTLED 2026-07-31 (day 10): the restore was DONE, and it read 833.9 at
   4.0 h against the original's 958.2 at ~4 h** (`EVIDENCE` §8p). Everything below
   is kept only as the record of how the decision was made. **Do not run it
   again.**

   If you do restore anyway:

   ```powershell
   python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_submit('dist/submission_bc-grimmsnarl-netspolicy_20260729-103819.tar.gz','P4b restore: lw2 + chip_target + energy_spread (the 952 agent)','pokemon-tcg-ai-battle')"
   ```

   - That tarball is **verified** to be `55072063`'s exact code (flags: chip +
     spread, `counter_source` absent from the signature) and **smoke-tested**
     (`scripts/restore_smoke.py`). §3.0's table.
   - **Eviction arithmetic (updated 2026-07-31): it evicts `55077709` at
     845.0 — which is now our BEST ACTIVE**, not our worst, and it is still
     climbing. Active becomes {P4b-restored, v3 819.8}. ⚠ **This is the reverse
     of what earlier drafts said** and it is the main new argument against.
   - ⚠ It restarts at **μ=600** and needs ~4 h to reach ~950; displayed dips to
     819.8 meanwhile. That is the cost, and it is unavoidable — a rating cannot be
     restored, only re-earned (§8h).
   - ⚠ **Read the LB before and after** (§5) and confirm with **two readings ≥1 h
     apart** (rule 2).

2. ~~**Build anchors for the REAL field.**~~ ✅ **DONE 2026-07-31 — this was the
   blocker on everything and it is gone** (§8i). `scripts/p9_field_census.py`
   names the field from our own 109 ladder games; `scripts/import_field_agents.py`
   imported the two anchors we lacked. Coverage **39.4% → 71.6%**. The remaining
   work is item 0's sweep, not more infrastructure.

   ⚠ **The one thing to internalise from it:** the 63% "other" was **not** an
   exotic tail. It was four ordinary archetypes plus a classifier with four
   hardcoded card ids. **Before believing a bucket called "other", check whether
   the classifier or the field is the thing that is small.**

3. ✅ **THE MEGA LUCARIO LEAD IS SIZED AND THE ANSWER IS "NOT WORTH SHIPPING
   ALONE"** (§8j, 2026-07-31). **The cause is not the v3 net — it is the rules
   being off.** v3+rules reads **0.572** vs `rule:v10` against rules-off's
   **0.505** (disjoint), i.e. level with P4b's 0.576.

   **But turning them on globally is worth +1 Elo** — the mirror loses exactly
   what Lucario gains (−7.1 vs +6.0 weighted), and Alakazam/Crustle/Archaludon
   are dead heats. **A Lucario-only branch sizes at ~+8 Elo, below the LB's
   ±50–100 resolution** (rule 2), so by rule 14 it is a **bundle candidate, not a
   solo submission, and not urgent.** The branch machinery already exists
   (`wall_defer` is the template) if it is ever bundled.

   ⚠ **Two traps this closed, both worth carrying:**
   - **"Rule X is harmful vs anchor Y" expires when X is modified.** We predicted
     rules-on would lose badly to Crustle (`chip_target` measured −0.126 there);
     it was a dead heat, because `wall_defer` has been ON by default since 07-30.
   - **Do not sum dead heats.** Only 2 of 5 cells have disjoint CIs. Branching
     wherever Δ>0 scores +12 Elo, of which **+4 is noise** from three overlapping
     cells.

3b. **B4 (turn-level sequencing) — probed GREEN, prototyped RED.** ⚠ **Superseded
   by §8n: the prototype LOSES 0.075 [0.026, 0.199] n=40.** Kept because the
   probe numbers are sound and are report material; see item C above for the
   decision. The probe said (`EVIDENCE` §8l, §8m):
   - **62% of our turns** have ≥2 real selects (rule 13 passed).
   - Space is median **98M** — exhaustive dead — but `fs.step` runs at
     **7,698/s**, so **~78,000 candidate sequences per turn** are affordable.
   - `evalfn` ranks **within** a turn: split-half top-1 agreement **62.0% vs
     6.2% chance** over 93 turns. Not determinization luck.
   - ⚠ **A pre-registered kill criterion was corrected mid-probe** (SNR at M=1
     answers a question B4 never asks). §8m documents it in full — **read that
     before trusting the verdict.**

   **Next: prototype + arena A/B at n≥1000 vs all five anchors, with pool-usage
   logging.** ⚠ Estimated gain is small and upward-biased (0.099 eval units per
   turn, a max over 16), and the 600 s pool is a real risk (§7). By rule 3 this
   is licensed as an experiment, not as an expected win.

3c. **The original Mega Lucario framing, kept because the two-instrument
   agreement still holds and still points at this matchup:**
   - **Arena:** v3 is **−50 Elo** vs `rule:v10` — the only anchor of five it
     loses, and the only negative term in the weighted table.
   - **Ladder:** we won **36.4% of 11 real games** against Mega Lucario
     opponents averaging **735**, i.e. **85 points below us** (§8i's calibration
     table). Losing to weaker players is not a matchup tax, it is a defect.

   It is **12.8% of the field**, we have a **real LB-950 pilot** for it already
   (`rule:v10,noS`), and `replays/submission_optv3` holds 11 real games to read.
   **Start with the audit, not a rule** (rule 14): what does Mega Lucario ex
   actually do to us — 340 HP, and `decks/lucario_v10.py` runs Gravity Mountain
   (bench HP?) plus Premium Power Pro ×4. Size the effect before writing
   anything, and check whether `chip_target`/`energy_spread` are net-negative
   here the way `chip_target` was against Crustle (§8c is the template).

   ✅ **The 5-anchor rule sweep is DONE — that is what item 3 above reports.**

4. **Re-mine the meta?** ⛔ **NO — and this is now a permanent rule, not a
   scheduling note.** Kaggle's daily episode datasets bottom out at `avg_score`
   **1055**; we play at 825–952; the 800–1000 buckets are **empty**. Mining
   produces an accurate picture of a band we never meet, and acting on it is what
   retired `rule:v10` — the anchor that turned out to be our worst matchup
   (§8i). Mining is still useful for
   **decklist consensus** (§1's "our 60 is the field's 60" is real Deck Score
   evidence) and for **Track B report figures about the top of the ladder** — but
   **never again as the input to an anchor decision.** Use
   `p9_field_census.py` on our own replays for that, and re-run it after every
   submission dump.

5. **Fix the two measured defects — but as questions, not licences** (§6 closed
   Boss's Orders after four null rules; all four were on the **lw2** net with the
   other rules on, so they do not settle this net):
   - **Boss's Orders: 9 of 31 real drags were misplays (29%)**, 5 of them throwing
     away a **double KO** (Shadow Bullet is 180 to the Active **plus 30 to a
     bench** — a ≤30 HP bench sitter means attacking takes two prizes).
   - **Froslass: 7 of 63 (11.1%)** evolves happened with more ability Pokemon on
     our side **and no armed Munkidori** — pure self-damage. ⚠ The other 19
     "we have more" rows are the intended engine (Shroud loads, Adrena-Brain
     ships); do not "fix" those.
6. **The Alakazam matchup is a strategy question nobody has asked.** It is 22% of
   the field — the biggest single thing we play against — and its attack is
   **Powerful Hand: 20 damage per card in the attacker's hand.** Nothing in
   `targeting.py` or the feature set reasons about the opponent's hand size, and
   the whole deck is a draw engine (Kadabra/Alakazam Psychic Draw, Dudunsparce
   Run Away Draw, Fezandipiti ex Flip the Script). **Size it before building it**
   (rule 14): how often is their hand large enough to matter, and is there any
   action of ours that shrinks it? ⚠ We already win this matchup 66.7%, so the
   headroom is small — check that first.

</details>

### The B1 arena result — kept because the CONTRAST with the ladder is the finding

> ⚠ **Read this as the specimen, not as a plan — and note that §8i has since
> explained it.** `optfeat` v3 beat the shipped agent **0.661 [0.640, 0.681]
> n=2000** in the mirror (≈ +115 Elo) and **0.770 vs `rule:crustle`**
> (shipped: 0.663) — two anchors, one adversarial, both agreeing, the first effect
> in the project larger than the LB's own resolution. With v3 features the hand
> rules measured **harmful** (`v3+rules` vs `v3 alone` = **0.427**), which is why
> it shipped with rules off.
>
> **It then read 825 against P4b's 952** (§8g). Nothing above was miscomputed and
> nothing above is retracted — every number reproduces from `out/arena/b1_*.jsonl`.
> **What was wrong was the coverage, not the measurement**: those two anchors are
> 26.6% of the field, and against the third-largest deck (`rule:v10`, 12.8%) the
> same v3 agent scores **0.505 vs P4b's 0.576** (§8i). The mirror's +0.161 and
> Lucario's −0.071 are both real; only one of them was in the anchor set.
> Nets: `out/policy_b1_v3.npz` (treatment), `out/policy_b1_ctrl.npz` (control).
> Corpus: `artifacts/pds_v3`.

<details><summary>The v3 bundle as built and shipped (kept for reproducibility)</summary>

**Built, smoke-tested, and SUBMITTED as `55116557` on 2026-07-30 18:14 UTC.**

   ```
   dist/submission_bc-grimmsnarl-netspolicy_20260731-000752.tar.gz  (4.0 MiB)
   dist/submission.tar.gz  <- same file (the `latest` copy)
   ```

   Built with, and this exact command is the reproducer:

   ```powershell
   python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc `
       --nets policy --policy-net out/policy_b1_v3.npz --no-rules
   ```

   **Verified, not assumed:**
   - `NET_OK opt_in=37` — the **v3** net is live in the extracted bundle. ⚠ This
     check is new and it matters: a net that fails the dim guard makes the agent
     play `list(range(minCount))` — **random-legal, and it still "runs"**. The
     builder now fails the build instead (`--policy-net` runs `policynet.load`),
     and the smoke asserts `NET_OK`.
   - `FLAGS chip=False spread=False src=False` — the rules are off, pinned in
     `main.py` as `AGENT_KWARGS`. ⚠ **Global defaults deliberately NOT flipped** —
     they remain correct for `lw2`, which is what is live right now. The
     `(net, flags)` **pair** is pinned at build time; `wall=True` is inert
     because `chip_target` never runs.
   - **sha256 of `sa/policy_net.npz` == `out/policy_b1_v3.npz`** — the packaged
     net is byte-identical to the one that measured 0.661.
   - `agent_pool_left=599.9s lat_max=0.04s` — 0.1 s of the 600 s pool.
   - Layout `main.py` + `deck.csv` + `cg/` + `sa/` at top level; 4.0 MiB of the
     197.7 MiB cap; smoke `exec`s the source with **no `__file__`** (the §7 gotcha
     that killed `55028078`).

   ⚠ **The packaging was all correct — and it did not save the result.** Every
   check above passed and the agent still lost ~130 points. **Build hygiene
   protects against shipping a broken bundle; it cannot protect against shipping
   a worse agent.** The thing that failed was the *decision*, and the decision
   came from the arena.

</details>

### Closed earlier on day 8 (kept for the record)

1. ~~**Size, then build, the Morgrem out**~~ ✅ **SIZED AND CLOSED 2026-07-30 —
   do not build it** (`EVIDENCE` §8e, `out/logs/p7_morgrem_200.txt`). The veto
   would fire **~0.2× per game**; the *free* version of the same out (post-KO
   promotion into a wall) is **already taken 95.4%** of the time; and the trade is
   *60 onto a wall they heal 22.5% off* vs *30 onto a 70-HP Dwebble that dies to
   it + 220 more HP of body* — a **tradeoff**, rule 11's 0-for-4 column. The
   effect is ~2.6% of our damage output in this matchup, which **an n=2000 A/B
   cannot resolve** (±0.021), so no A/B was spent. **Also corrected a load-bearing
   claim:** "our attacker deals 0 into theirs" is true of their **Active only** —
   Shadow Bullet's bench snipe lands **unprevented on Dwebble (82 events, mean
   73.9, 0 zeroed)** and kills the Crustle line's basics.
2. ⛔ **A pilot for `crispin_toolbox` DOES NOT EXIST PUBLICLY — searched
   2026-07-30, and the public-notebook well is dry for competitive pilots of any
   deck.** All **272** public notebooks for this competition were enumerated
   (4 sort orders × 3 pages). No Crispin/toolbox pilot at all. Three candidates
   whose titles claimed high ratings were pulled and **all three refuted against
   the 4,000-row LB** (rule 10, the same trap as the "1084.5 baseline"):

   | notebook (claim) | author's actual standing |
   |---|---|
   | `soutasakurai/max-elo-1208-libraryout-w-crustle-great-tusk` ("Max Elo 1208") | **`SOUTA Sakurai`, rank 3439/4000, 605.0** — *below the μ=600 start* |
   | `prvsiyan/ptcg-ai-battle-static-deck-tusk-1208-v24` ("Tusk 1208") | `prvsiyan`, rank 1083, 789.1 |
   | `pcxxxxxx/explainable-ptcg-agent-with-legal-ogerpon-deck` | `pcxxxxxx`, rank 2454, 686.6 |

   Every other verifiable notebook author also sits **below us**: `kokinnwakashuu`
   832.9, `jazivxt` 816.3, `pllinas` 739.1, `penguin069` 689.8, `naoto714` 633.0.
   **The top 10 (1187–1147) have published nothing.** So there is no public agent
   stronger than ours to import, and this avenue is closed — not deferred.

   **Consequence, and it is good news:** rule 12's bar (**≥2 anchors, one
   adversarial**) is *already met* by the mirror + `rule:crustle`, and
   `rule:crustle` is competitive on our own measurement (we score 0.663, not a
   0.911 blowout — a real number beats any notebook title). **Writing a Crispin
   pilot ourselves is NOT recommended:** a 5-attacker multi-type toolbox with
   Crispin tutoring is far harder to pilot than Crustle's single lockdown line,
   and a weak self-written pilot reproduces the 0.911 no-resolving-power failure.
   By rule 14, size that before building it.
3. **Re-mine the meta — BLOCKED UNTIL 07-31.** 07-30's episodes publish the
   following day (the current day always 403s) and 07-29 is already mined, so
   there is nothing new to fetch today. On 07-31: confirm the Crustle/Crispin
   shares and build the **deck matchup win-rate matrix** among high-rated players
   (ROADMAP Track B/C figure). ⚠ This also gates the Crispin-anchor question —
   check Crispin's share is still ~17% before spending any work on it.
4. ~~**ROADMAP B1** (feature-augmented retrain)~~ ✅ **DONE AND WON 2026-07-30/31
   — see the green box at the top.** `EVIDENCE` §8f. Follow-ups it opened, in
   value order:
   - **Retrain v3 on a bigger corpus.** v3 won on **1,603 games vs the shipped
     net's 2,810** — 43% less data. The pruned days are re-fetchable from
     `replays/manifests/` (12 days of episode ids). ⚠ But §1 says more data is
     *not* a lever, so treat this as a cheap check, not an expected gain.
   - **Re-A/B each rule against the v3 net individually.** We know the three
     together are harmful (0.427); we do **not** know whether one of them is
     still positive. `noChip` / `noSpread` / `noSrc` one at a time.
   - **The v3 features make `wall_defer`'s hardcoded `WALL_POKEMON = {345}`
     obsolete in principle** — "our damage into this target" is now feature 34,
     so the wall condition is readable off the board for *any* prevention
     ability. Only matters if a second wall deck appears.
5. ~~**Do not submit yet.**~~ ⚠ **SUPERSEDED BY B1 (item 0).** That advice was
   written when the best candidate was a ~12-Elo rule, which the LB cannot
   resolve. **B1 measures ≈ +115 Elo on two anchors — above the instrument's
   precision** — so the reasoning that said "wait and bundle" now says "submit
   this one". The bundle it was waiting for exists.
6. ~~**`report/STRATEGY.md` does not exist yet**~~ ✅ **CREATED 2026-07-31** after
   slipping ~4 sessions. §1–5 and §8 are written from concluded experiments;
   §6–7 are outlined against work in flight. **Standing rule: one edit per
   session, however small** — it is 30%+ of the rubric against the LB's one
   bullet of five, and it was the only deliverable with no same-day feedback
   loop to force it. See ROADMAP's doc-discipline audit.

### The five files, and what each owns

| file | owns |
|---|---|
| **`HANDOFF.md`** (this) | live state, the live engineering plan, the anti-self-deception rules, commands, gotchas |
| **`ROADMAP.md`** | the strategy-competition plan — what the engineering is *for*, the breakthrough hunt, the calendar |
| **`report/EVIDENCE.md`** | the hypothesis log: every concluded experiment with n, CI, verdict. **All closed-experiment detail lives there, not here.** |
| **`report/STRATEGY.md`** | **the report itself** — the deliverable due 09-14. ⚠ **One edit per session, however small.** It slipped ~4 sessions because it is the only file with no same-day feedback loop |
| **`competition_details_and_rubric.md`** | the rubric, verbatim |

**End of every session: update HANDOFF (plan), ROADMAP (calendar), EVIDENCE
(any experiment that CONCLUDED) and STRATEGY (one edit, however small) together.**

⚠ **And when you retract a claim, `grep` it across all five files in the same
commit.** Updates here have been additive — HANDOFF went 135 → 1,579 lines in
5 days and carries 27 retraction markers — so a wrong claim survives in whatever
copied it. "`lucario_v10` is 0% of the meta" propagated to four places and cost
us the anchor that would have caught B1. Rule 15 warns about this; we did it
anyway. ROADMAP's doc-discipline audit has the numbers.

> **Submission state (2026-07-31). ⚠ The previous version of this box was WRONG
> on the one point that mattered — see the ✅ below.**
>
> - **Daily quota: 5/day.** Never the binding constraint.
> - **Only the latest 2 submissions play episodes.** Active pair right now
>   (2026-07-31 10:46 UTC): **`55129730` (P4b restored, 833.9) + `55116557`
>   (v3, 818.1)**. `55077709` (P6a) is **evicted, frozen at 841.5**; `55072063`
>   (**952.0**) has been frozen since 07-30.
> - 🔴 **The 952.0 is not a target and never was one.** Re-running that exact
>   agent on today's board produced **833.9** (§8p). **Every one of our agents
>   reads 818–842 when played concurrently.**
> - ✅ **ANSWERED (was "the open question that decides the endgame"): the
>   displayed score is the best ACTIVE submission, NOT the best ever.** Proof:
>   best-ever is `55072063` at 952.0, best-active is 837.5, and **the board shows
>   us at 837.5 / rank 605.** We fell 224 → 605 on the eviction alone.
> - 🔴 **So "freezing is cheaper than it sounds" was FALSE and is retracted.** A
>   frozen score counts for nothing. **The best agent MUST be in the active pair
>   at the 08-17 close and through the 08-31 continued-play window.**
> - 🔴 **Every submission is therefore a real risk, not a free option.** It
>   evicts, and the evicted score stops counting the moment it does.
>
> **The bar on submitting is "do we expect this to beat the best agent we would
> be evicting" — and 🔴 as of 2026-07-31 NOTHING WE HAVE CLEARS IT.** All three
> agents are within **36 Elo** and the LB resolves **±50–100** (§8k), so no
> current candidate is distinguishable from what it would evict. **The active
> pair {v3 819.8, P6a 845.0} is the arena's top two; leave it alone.**
>
> ⚠ The rollback argument ("952 > 837.5") is **retracted**: 952.0 is frozen, was
> earned on a ~4,000-entrant board, and the arena ranks P4b **last** of the three.

---

## 1. Where we are (day 8 end, 2026-07-30)

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

### ⚠ The meta shift (2026-07-30) — TRUE, but about a band we never play in

> 🔴 **READ THIS BEFORE THE TABLE (added 2026-07-31, `EVIDENCE` §8i).** Every row
> below was mined from the **top 400 episodes by `avg_score`**, and Kaggle's daily
> datasets **contain nothing below `avg_score` 1055**. We play at **825–952**;
> the 800–900 and 900–1000 buckets are **empty**. So this table is an accurate
> description of the **top of the ladder** and says nothing about our opponents.
>
> **Acting on it cost ~130 LB points.** Row 1 below retired `rule:v10` as "0% of
> the meta"; in our own 109 real games Mega Lucario is **12.8% of the field**, and
> it is the anchor that would have caught B1. **For what we actually face, use
> `scripts/p9_field_census.py` on our own replay dumps — never this table.**
>
> What it IS still good for: the decklist-consensus finding (item 3 below, real
> Deck Score evidence) and Track B report figures about the top of the board.

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

1. ❌ ~~**`lucario_v10` — the single opponent every routine number in this repo is
   measured against — is 0 of 400 games.** Our arena bar has been measuring a
   deck that has left the meta entirely. This is rule 12's worst case,
   realised.~~ 🔴 **RETRACTED 2026-07-31 — this is the most expensive wrong
   sentence in the project.** It is 0 of 400 games **at avg_score ≥ 1144**. In
   our own 109 real games Mega Lucario is **12.8% of the field**, and it is the
   matchup we lose worst (36.4% over 11 games, against opponents rated **85
   points below us**). Retiring that anchor is what let B1 ship unseen. **The
   error was not the measurement — it was reading a top-band sample as "the
   meta".** §8i.
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
12. **A single-anchor arena will eventually lie to you, and on 2026-07-30 it
    did.** Everything routine was measured against `rule:v10,noS` on
    `lucario_v10` — ⚠ **which we then wrongly wrote off as "0% of the meta"; it
    is 12.8% of the field we actually play (§8i), and this rule's own example
    below is the good part of the story, not that clause.** Re-anchored on
    `rule:crustle`, **`chip_target` — the rule that bought ~150 LB points and
    defined the project's whole method — measures −0.126, i.e. actively
    harmful**, while it is worth +0.077 in the mirror (`EVIDENCE` §8c).
    **An arithmetic rule encodes an objective, and an objective is only correct
    while the strategic context holds.** So:
    - **Every rule A/B needs ≥2 anchors, one of them adversarial**, and every
      archived number carries an anchor label.
    - **A rule that wins on one anchor is a matchup branch candidate, not a
      shipped rule**, until a second anchor agrees.
    - Also: a pattern the user watched in a real game can be genuinely absent
      locally. **When the local audit says it never happens, measure it on
      `replays/submission_replay_2026-07-29/`** — `scripts/p5a_replays.py` reads
      our real selects against 54 distinct LB opponents.
    - ⚠ **An anchor must be COMPETITIVE to resolve anything.** `bc` piloting an
      off-distribution deck gave a 0.911 blowout — a ceiling that squeezes any
      rule delta to nothing. Import a real pilot (`import_crustle_agent.py`).
13. **Check the denominator is a real CHOICE, not just a real count.** P5a read
    "the rule takes the best target 26/26" — but 90 of its 95 pooled-KO rows
    offered only one prize value, so nothing could go wrong. The honest
    denominator was **5**. A rate over forced moves measures nothing.
14. **SIZE BEFORE YOU BUILD: a dramatic per-instance number says nothing about
    frequency, and frequency is where rules die.** "Morgrem deals 60 through the
    wall while Grimmsnarl ex deals 0" was filed as *the biggest known lever in the
    matchup*. Sized: the rule would fire **~0.2 times per game**, the free version
    of the same out was **already taken 95.4%** of the time, and the effect
    (~2.6% of our damage output) was **smaller than an n=2000 A/B can resolve**.
    Closed for the price of one probe and no A/B (`EVIDENCE` §8e). This is rule 10
    one stage earlier: **moving an audit rate is not winning games, and counting
    an opportunity is not finding one.** Ask "how often, and how big per
    instance?" *before* writing code — and check whether the cheap version of the
    behaviour already happens.

    ⚠ **A corollary that caught this one:** state the rule's *alternative*
    explicitly and measure it too. The whole argument rested on the alternative
    being worth zero; it was not (the bench snipe kills their basics), and nobody
    would have noticed without asking what the other branch actually does.
15. **RE-READ THE CODE THAT THE WHOLE METHOD RESTS ON. The project's founding
    premise was false for eight days and nobody checked.** "The net cannot see
    HP" was written in `targeting.py`, repeated in `HANDOFF`, and used to justify
    every rule — while `features.py` had been feeding the net per-slot HP,
    damage, energy and prize value since v1. The true gap was one line's worth:
    `opt["index"]` was never encoded, so two options naming two copies of the
    same card were **bitwise identical inputs with different right answers**.
    Fixing that measured **0.878** and made the rules harmful (`EVIDENCE` §8f).
    **A premise repeated in three files is not thereby verified — it is just
    load-bearing.** When a claim about the code justifies weeks of work, open the
    file and confirm it, especially if it has never been questioned.

    ⚠ **The general form, and the thing to carry forward:** ask whether a blind
    spot is **informational** (the input is absent) or **representational** (the
    input is present but cannot be bound to the decision). They look identical
    from the outside — the agent gets it wrong at chance — and they have opposite
    cures. Four hand rules cured the symptom; 12 features cured the cause and
    dominated them.
16. **AN ARENA RESULT IS A WEIGHTED AVERAGE OVER YOUR ANCHOR SET, AND NOTHING
    ELSE. State the weights before you read the score.** ✅ **Resolved
    2026-07-31** (`EVIDENCE` §8i) — the earlier version of this rule said the
    arena does not measure ladder strength and treated that as the project's
    central problem. **That was wrong, and believing it would have cost far more
    than the original mistake.** The arena predicts fine. Both LB "contradictions"
    were the same error: the anchor set did not span the field.

    - v3 measured **0.661 in the mirror** and **0.505 vs `rule:v10`** (P4b:
      0.576). Both are true. Only the first was in the anchor set, and the
      ladder averaged over both.
    - **Weight every anchor by its measured share before concluding anything.**
      "Wins 2 anchors, loses 1" is not a verdict; "+0.16 on 13.8% and −0.07 on
      12.8%" is the start of one. `p9_field_census.py` supplies the shares.

    ⚠ **And the deeper trap, which is the one to actually carry forward:**
    **CHECK WHERE YOUR POPULATION DATA COMES FROM BEFORE YOU LET IT RETIRE AN
    ANCHOR.** `fetch_top_episodes.py` mines the **top** episodes by `avg_score`,
    and Kaggle's daily datasets are **censored below `avg_score` 1055** — the
    800–1000 buckets are literally empty. We play at 825–952. So the mined meta
    was a perfectly accurate description of a population we never meet, and it
    said `lucario_v10` was **0% of the field** when in our own games it is
    **12.8%**. Retiring that anchor on that evidence is what let B1 ship.

    **The general form: a sampling frame you did not choose is a hypothesis, not
    a fact.** Ask what the data-generating process excludes — not whether the
    numbers are right.

    ✅ **The positive control still holds, and it is why the arena is trusted
    again:** the arena predicted 0.770 vs `rule:crustle` and we won **76.9% of 13
    real Crustle games**. The arena is accurate exactly where the anchor
    resembles the opponent — which is now most of the field (71.6%), and was
    26.6% when B1 was decided.

    **Standing requirement: measure against the top-5 anchors, weighted, and
    never again let a mirror A/B alone decide whether to turn a rule off** — the
    mirror is 13.8% of reality.
17. **HELD OUT BY GAME IS NOT HELD OUT BY PLAYER — and the identity you are
    holding out is not stable.** Day 11 tried to explain §8q's rating trend and
    found that the obvious confound could not even be *tested* with the corpus as
    built: the trainer splits on `gid % 20`, so **every demonstrator in the
    held-out split also appears in the training split**, and the "0 exposure"
    bucket was empty. A same-week, same-deck dump of **87 players the net had
    never seen a row of** was needed to answer it (the answer: exposure buys
    nothing, 73.6% unseen vs 69.3% for the most-trained-on). **Before believing
    "the net generalises to player X", check whether X is in the training set —
    the split does not do it for you.**

    ⚠ **And the key is a display name, which means it is not a key.** Naive
    matching left 24.6% of one day's seats unrated; **182 of those 198 misses
    were the LEADERBOARD'S #1 TEAM**, appearing as `James Cox`, as `zoroark190`
    (a member username) and as the post-merge `James Cox & Henry Chao`. §8q hit
    the same bug on `Sixth Sense` / `Raja Biswas`. **Teams rename and merge
    mid-competition, so any per-player statistic silently splits your most
    valuable demonstrator into three and reports it as sparse data.** Resolve on
    `teamId` from the episode sidecar where you have one; match member usernames
    exactly; keep verified renames in `replays/team_aliases.tsv`; and **print the
    match rate and the biggest unmatched names every time** (rule 9).

    ⚠ **A third form, same session: a control population built from "the
    opponents of X" contained US.** `Scio` is on that list because we played
    them, and our own agent's selects are exactly what the net was fitted to —
    left in, it scores ~98% against itself and inflates the control.
    **`--exclude` yourself from any population you intend to treat as
    independent.**

---

## 3. THE PLAN (day 9 → day 10)

**Day 9 closed the one question the whole project was blocked on** (§3.4): the
arena/ladder gap is an anchor-coverage problem, not an instrument problem, and
the anchor set is now rebuilt to 71.6% of the field. **The arena is trustworthy
again, with the weighting discipline in rule 16.**

**Day 8 closed all three of day 7's open questions** (§3.0 `counter_source` is
good and stays; §3.1 we were *not* measuring against the right opponents, and the
correction found a harmful rule; §3.2 the Crustle premise is verified) **and
shipped a fix (§3.3).** It also killed ROADMAP B2. The live work is now the
▶ START HERE list at the top of this file; §2.5 of `ROADMAP.md` holds the ranked
breakthrough candidates (B1, B3–B5) that run alongside.

### 3.0 ✅ RESOLVED (2026-07-30): `counter_source` stays

Re-measured against the new meta anchor: **`counter_source` is worth +0.052 vs
`rule:crustle`** (0.559 with, 0.507 without, n=2000 each) — *more* than it was
worth in the mirror (+0.034) or vs `lucario_v10` (+0.033). The LB scare was an
artifact of reading a ±75-point instrument at 12-Elo precision (§1). **Keep the
rule; no rollback.** `EVIDENCE` §8c. History below, kept because the reasoning is
report material.

### 3.0b The original write-up — "unresolvable on the LB"

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

**Preserved builds — labels VERIFIED and both smoke-tested 2026-07-31**, so
nothing needs rebuilding under time pressure:

| tarball in `dist/` | bundled rule flags | = submission | restores? |
|---|---|---|---|
| `...20260729-103819.tar.gz` | chip, spread (**no `counter_source` in the signature at all**) | **`55072063` — the 950.2 agent** | ✅ `NET_OK`, full game, 0.1 s pool |
| `...20260729-152103.tar.gz` | + `counter_source` | `55077709` (824.9) | ✅ `NET_OK`, full game |
| `...20260730-151057.tar.gz` | + `chip_wall_defer` | **never submitted** (day-8 wall branch) | — |
| `...20260731-000752.tar.gz` | **rules OFF + v3 net** | **not submitted yet** (item 0) | ✅ `NET_OK opt_in=37` |

All three lw2 bundles carry the same net (`sha256 bba02a42…` = `out/policy_lw2.npz`
= the live `agents/sa/policy_net.npz`) and **their own copies of `sa/` and `cg/`**,
so later repo changes cannot break them — the 07-29 bundles still report
`opt_in=n/a` because they predate that property, and they run fine.

⚠ **"Restorable" is NOT "recoverable to 950."** Re-submitting the P4b bundle
restarts it at **μ=600** and it must climb 4+ h; the 950.2 rating itself cannot be
restored, only re-earned. So the insurance is against *losing the code*, which we
have not, and never against a bad submission decision.

### 3.1 ⚠ Re-anchored (2026-07-30) — and re-anchored AGAIN on 07-31, see §3.4

> 🔴 **The premise below is retracted.** Day 8 rebuilt the anchor set because
> `lucario_v10` was "0% of the meta" — true only of the top-1150 band. **Adding
> `rule:crustle` was right; dropping `rule:v10` was the mistake**, and §3.4 put
> it back. The current anchor set is §4's five-deck table. Kept because the
> Crustle work in it is sound and the reasoning is report material.

Every number in §3, §6 and `EVIDENCE.md` was earned against `lucario_v10`, which
was believed to be **0% of the meta** (§1). So the bar itself has to be rebuilt.

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

✅ **The pilot blocker is SOLVED for Crustle: `rule:crustle` now exists.**
`scripts/import_crustle_agent.py` lifts the public `pixiux/ptcg-crustle-v1-submit`
agent (409 lines of readable option scoring) into
`agents/agentkit/rulebased/sources/crustle.py` + `decks/crustle_v1.py` (its own
tuned 60), registered in `DECK_MODULE`. Idempotent. It plays the real lockdown:
bench Dwebble → evolve → arm 3 energies incl. Grass → Hero's Cape → Battle Cage
stadium → heal with Jumbo Ice Cream / Cook at damage ≥50 → retreat to a ready
Crustle → Superb Scissors (479, **120 damage**, ×2 into Grass weakness).

```powershell
python -X utf8 scripts/arena.py play bc rule:crustle `
    --deck-a grimmsnarl --deck-b crustle_v1 --matches 1000
```

⚠ **Two Crustle decks, and the difference matters.** `crustle_v1` is the pilot's
own list — use it when you want the strongest Crustle we can run locally.
`crustle` is the **field consensus** list (77×-seen). The pilot scores ~20 of the
consensus list's cards through a generic fallback, so it plays them legally but
badly; early n=20 probes read 0.620 on its own list vs 0.700 on the consensus
one, in the direction that confirms this.

⛔ **`crispin_toolbox` has no pilot and CANNOT GET ONE from public code — the
search is complete, not pending (2026-07-30).** All 272 public notebooks were
enumerated; there is no Crispin/toolbox pilot, and **no public author outranks
us** (details and the refuted-title table are in the ▶ START HERE item 2 above).
The first attempt already showed why a substitute won't do: `bc` piloting it
scored 0.089 — **we beat it 0.911 [0.898, 0.923] at n=2000**, and an anchor we
beat 91% of the time has almost no resolving power for a rule worth ~1 pp because
the ceiling squeezes the delta. **A `bc`-piloted anchor is not good enough; do not
spend A/B time on one.** Rule 12's ≥2-anchor bar is met by the mirror +
`rule:crustle` in the meantime.

**Public notebooks worth mining (pulled to `notebooks/pulled/`, 2026-07-30):**

| ref | why |
|---|---|
| `pixiux/ptcg-crustle-v1-submit` | ✅ imported — `rule:crustle`. **Its competitiveness rests on our own number (we score 0.663), not on the title** — `pixiux` does not appear on the LB at all |
| ~~`makthanithin/pokemon-tcg-ai-battle-1084-5-baseline`~~ | ⚠ **DO NOT TRUST THE TITLE.** "1084.5" is the author's self-report. Checked against the full LB: they are **`Nithin maktha`, rank 750, 819.1** — **hundreds of places below us**, and no `makthanithin` appears at all. **A notebook title is not a measurement** (rule 10). Kept only as a lesson |
| ~~`soutasakurai/max-elo-1208-libraryout-w-crustle-great-tusk`~~ | ⚠ **THE SAME TRAP, SECOND TIME.** "Max Elo 1208" — the author is **rank 3439/4000 at 605.0, below the μ=600 start.** Pulled and rejected 2026-07-30 |
| ~~`prvsiyan/ptcg-ai-battle-static-deck-tusk-1208-v24`~~, ~~`pcxxxxxx/explainable-ptcg-agent-with-legal-ogerpon-deck`~~ | ⚠ also pulled, also refuted: 789.1 (rank 1083) and 686.6 (rank 2454) |
| `jazivxt/crustle-counter-al220-v29-agents-only` | someone else's *anti-Crustle* agent — directly Track C |
| `kokinnwakashuu/ptcg-lucario-public-lab-anti-crustle-log` | anti-Crustle analysis + logs |
| `prvsiyan/ptcg-ai-battle-control-v11-meta-portfolio` | "meta router"/portfolio = ROADMAP B3 (archetype detection → matchup branches) |
| `busyaprime/what-actually-wins-on-the-ladder`, `myso1987/...deck-meta-by-score-band` | independent meta analyses to cross-check our mining against |

⚠ **Do not treat a cross-deck score as skill** (rule 5) — use each new anchor the
way `rule:v10` was used: a fixed opponent for A/B *deltas*, both sides facing the
identical opponent. And **archive the per-anchor tables**; they are the rubric's
consistency/robustness exhibit and go into the report verbatim.

⚠ **Do not treat a cross-deck score as skill** (rule 5) — use each new anchor the
way `rule:v10` is used: a fixed opponent for A/B *deltas*, both sides facing the
identical opponent.

Also: **archive the per-anchor A/B tables.** They are the rubric's
consistency/robustness exhibit and go into the report verbatim.

### 3.2 ✅ Crustle — **this is the meta now**, piloted, and the premise is verified

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

### 3.3 ✅ FIXED AND SHIPPED (2026-07-30): the matchup branch

`chip_target` now defers to the net when the opponent's Active is a wall
(`targeting.WALL_POKEMON = {345}`), **ON by default**, `bc:<label>,noWall` to
disable.

| variant | vs `rule:crustle`, n=2000 |
|---|---|
| `bc` before (unconditional) | 0.559 [0.537, 0.581] |
| **`bc` now (branch on)** | **0.663 [0.642, 0.684]** |
| `bc:x,noChip` (ceiling) | 0.685 [0.665, 0.705] |

**Recovers 82% of the −0.126**, and the mirror control reads 0.521 [0.490, 0.552]
n=1000 (contains 0.5 — no bleed, and none is possible by construction).

⚠ **Do NOT submit this alone.** It is worth ~+10–15 Elo overall (a +0.10 swing in
18% of the field), which is **below the LB's resolution** (§1). Bundle it.
Remaining headroom: a wall-aware *ranker* instead of deferral, worth at most the
0.663 → 0.685 gap. **Next, bigger, and in the same matchup: the Morgrem out
below.**

### 3.3b The original diagnosis (kept — it is the report's argument)

**The measured defect** (`EVIDENCE` §8c): vs `rule:crustle`, `bc` scores **0.559**
and `bc:x,noChip` scores **0.685** — **our founding rule costs us 12.6 points of
score in 18% of the field.** In the mirror (52% of the field) it is worth +0.077
head-to-head, so **do not delete it — branch it.**

**Why it fails, measured:** `chip_target` ranks "dies to 30 first, most prizes
among those, then lowest HP", which against Crustle farms **Dwebble** (a 1-prize
basic) while the immune wall sits untouched. Counter-placement events onto Dwebble
drop **235 → 24** when the rule is off, and events onto Crustle rise **1,386 →
1,583** at a higher mean (12.9 → 15.0).

**The rule to write** — and note it is a *dominated-option* rule by rule 11, which
is the 3-for-3 column: **when the opponent's Active cannot be damaged by our
attacks, damage counters are the only way to remove it, so concentrate them
there** rather than spending them on a killable basic. The condition is factual,
not a judgment: we can test "would our attack deal 0 to this target" directly
(that is what `best_damage` / the census measures), so this is arithmetic, not a
guess about what matters.

**Design sketch (implement in `targeting.py`, default OFF until A/B'd):**

1. Detect the immune-wall condition per target, not per archetype: for the
   opponent's Active, `best_damage(our_active, ...) == 0` while a counter effect
   is available. That generalises past Crustle to any prevention ability, and
   needs no archetype classifier — **so it is cheaper than B3 and should be tried
   first.**
2. When it holds, rank counter targets by "damage that actually lands, most on
   the blocker" instead of by killability.
3. A/B against **all three** anchors: `rule:crustle`, the grimmsnarl mirror, and
   `rule:v10` (for continuity with the archived numbers). It must not bleed the
   mirror.

⚠ **Also test the cheap alternative first:** simply switching `chip_target` off
when the opponent's Active is undamageable is a one-line version of the same idea
and already has a measured +0.126 upper bound in this matchup. **Measure the
one-liner before building the ranker.**

~~🆕 **And a second, independent out from `EVIDENCE` §8d: Marnie's Morgrem
(non-ex) deals 60 through the wall while Grimmsnarl ex deals 0.**~~
❌ **CLOSED BY SIZING 2026-07-30 — do not build it** (`EVIDENCE` §8e,
`scripts/p7_morgrem.py`, `out/logs/p7_morgrem_200.txt`, 3× 200 games).

| measurement | result |
|---|---|
| turns the evolve-veto would actually fire | **38 / 49 / 53 per 200 games** = ~0.2/game |
| Morgrem Active vs a wall but **cannot pay {D}{D}** | 66% of such turns |
| **post-KO promotion into a wall** — the *free* route, no retreat cost | **288/302 = 95.4% already promote the Morgrem** |
| damage healed back off their Crustle | **22.5%** — the 60 is worth ~47 net |
| attack damage onto their **Dwebble** | **82 events, mean 73.9, 0 prevented** |

**Three reasons, any one sufficient.** (1) ~0.2 firings/game × ~47 net damage
against the ~352/game we already land = **~2.6%**, and an n=2000 A/B resolves
±0.021 — **the instrument cannot see it** (§1, now applied to the arena, not the
LB). (2) The cheap version of the out is already taken 95.4% of the time — the
"316/316 lethals, all forced" shape. (3) It is a **tradeoff**, not a dominated
option: 60 onto a healing 150-HP wall vs 30 onto a 70-HP Dwebble that *dies* to it
plus 220 more HP of body. Prizes are a genuine tie (1 per hit either way: ex = 2
prizes and survives exactly two 240s; Morgrem = 1 prize and dies to one), which is
what made it look dominated on paper — but "which target matters" is a judgment,
and rule 11's ⚠ clause is explicit that one judgment is enough.

⚠ **And it corrected a load-bearing sentence.** "Our main attacker deals 0 into
theirs" is true of their **Active only**. Shadow Bullet's 30 bench snipe is
**unprevented**, and onto a 70-HP Dwebble it kills the Crustle line's basics. Any
future anti-wall play is measured against *that*, not against zero.

⚠ **Not closed:** the retreat/promotion route — 451 turns per 200 games (2.3/game)
where Grimmsnarl ex attacks a wall for zero *with a Morgrem benched*. 10× the
denominator, but Grimmsnarl ex's retreat cost is **2** (the whole attack
investment), so it is a worse trade than it looks. Filed, not recommended.

### 3.4 ✅ RESOLVED (2026-07-31): the arena/ladder gap was anchor coverage

**The finding, in one line: the arena is accurate, and we retired the anchor that
would have caught B1 two days before B1 was decided.** Full write-up
`EVIDENCE` §8i; the numbers are in the top box of this file.

Three things came out of it, in decreasing order of how much they change:

1. **🔴 The public episode data cannot describe our opponents, ever.** Kaggle's
   daily datasets stop at `avg_score` **1055**; we play at **825–952**. This is
   censorship in the data-generating process, not a sampling choice we can tune.
   **`replays/submission_*` is the only evidence about our own field**, which
   makes those dumps the repo's most valuable asset and makes pulling replays
   after every submission a standing task.
2. **The anchor set is rebuilt to 71.6%** (§4's table) — `rule:alakazam5` and
   `rule:archaludon` imported, `rule:v10` reinstated.
3. **Rule 16 is rewritten** from "the arena does not measure ladder strength" to
   "an arena result is a weighted average over your anchor set" — with the
   sampling-frame warning as the general lesson.

**What is NOT resolved and is now item 0:** whether v3 is better than P4b once
all five anchors are weighted. Two of four runs are in; v3 loses Lucario and
wins the mirror and Crustle.

### The board

| | item | state |
|---|---|---|
| **§3.0** | is `55077709` (P6a) actually good? | ✅ **RESOLVED — yes, keep it.** +0.052 vs the new anchor |
| **§3.1** | re-anchor the arena on the current meta | ✅ **SUPERSEDED BY §3.4.** Day 8 re-anchored on the *mined* meta and that is what broke it: the mined meta is the top-1150 band, not ours. Day 9 re-anchored on **our own replays** — 5 anchors, 71.6% coverage. ⛔ `crispin_toolbox` stays pilot-less and is now **low priority: 0 appearances in 109 real games** |
| **§3.4** | why did the arena disagree with the LB? | ✅ **RESOLVED — anchor coverage, not the instrument.** v3 reads 0.505 vs `rule:v10` against P4b's 0.576, CIs disjoint (`EVIDENCE` §8i) |
| **§3.2** | Crustle premise probe | ✅ **VERIFIED — counters bypass the wall, AND a non-ex attacker gets through.** Track C steps 3–4 unblocked |
| **§3.3** | `chip_target` is HARMFUL vs Crustle (−0.126) | ✅ **FIXED AND SHIPPED** — the `wall_defer` branch recovers +0.104 |
| **§3.3b** | the Morgrem out (the non-ex attacker) | ❌ **CLOSED BY SIZING 2026-07-30 — do not build.** ~0.2 firings/game, the free route is already 95.4% right, and it is a tradeoff (`EVIDENCE` §8e) |
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

  ⚠ **EVERY NUMBER IN THIS TABLE IS CONDITIONAL ON THE `lw2` NET (2026-07-31).**
  Against the **v3** net the same three rules measure **0.427 together — actively
  harmful** (`EVIDENCE` §8f). They are *proxies for the option→target binding*, so
  once the features supply it the rules override a better-informed net with cruder
  arithmetic. **Read this table as "what the rules are worth to a net that cannot
  see its options' targets", not as a property of the rules.**

  **Two anchors per row now (rule 12).** Mirror = head-to-head vs the variant;
  Crustle = this variant's score against a fixed `rule:crustle`, so its rule
  value is the *difference from `bc`'s 0.559* (`EVIDENCE` §8c).

  | function | select | switch | mirror | vs Crustle |
  |---|---|---|---|---|
  | `chip_target` | DAMAGE / DAMAGE_COUNTER(_ANY) | `noChip` | 0.577 → +~150 LB | −0.126 unconditional 🔴 |
  | ↳ `wall_defer` branch | ditto, when their Active is a wall | `noWall` | no effect by construction (0.521 control) | **+0.104 recovered** ✅ |
  | `energy_spread` | MAIN, {D} ATTACH onto a Munkidori | `noSpread` | **0.702** n=4000 | **+0.193** ✅ |
  | `counter_source` | REMOVE_DAMAGE_COUNTER (ours) | `noSrc` | 0.534 n=2000 | **+0.052** ✅ |
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
- `features.py` (v2, DENSE_DIM=242, PER_SLOT=18) / `optfeat.py` (**v3 as of
  2026-07-30, OPT_DENSE 25 → 37**) — shared by trainer and inference.

  ⚠ **The project's stated blind spot was MISDIAGNOSED until 2026-07-30**
  (`EVIDENCE` §8f). "The net cannot see HP" is **false** — `features.py` has always
  given it per-slot HP, damage, energy and prize value for all 12 slots. The real
  gap: the v2 per-option vector encoded position only as *area* flags and **never
  encoded `opt["index"]`**, so two options naming two different bench slots were
  identical vectors — and two options naming **two copies of the same card were
  bitwise identical with different right answers.** That is exactly
  `energy_spread` (bare vs loaded Munkidori, and note it is the largest effect
  ever measured here, 0.702) and `chip_target`. **The rules restore a missing
  BINDING, not missing arithmetic.**

  **v3 appends 12 target-state features** (target HP, maxHP, damage fraction,
  dies-to-30, prize, energy count, own-type energy, ours/theirs, our damage into
  it, can-KO, and the **slot index**). ⚠ **Appended, never inserted** — dims 0..24
  are byte-identical to v2, and `policynet.Net.opt_in` derives each net's width
  from `head_in` and slices. **That is what lets a v2 and a v3 net run in ONE
  process for a head-to-head A/B (rule 4) across a feature change** — and it is
  also what stops a dim bump from silently falling the shipped net back to
  `list(range(minCount))`. **Do not replace `opt_in` with the global constant.**
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

### The anchor set — five decks, 71.6% of the field (rebuilt 2026-07-31)

Shares and our win rates are from **our own 109 ladder games**
(`scripts/p9_field_census.py`, `out/logs/p9_field_census_pooled.txt`), which is
the only source that describes the band we play in (`EVIDENCE` §8i).

| anchor | deck | share | our WR | pilot |
|---|---|---|---|---|
| `rule:alakazam5` | `alakazam5` | **22.0%** | 66.7% | author reports **5th place**, pure rules |
| mirror: `bc` v `bc` | `grimmsnarl` | 13.8% | 60.0% | ourselves |
| `rule:crustle` | `crustle_v1` | 12.8% | 57.1% | `pixiux/ptcg-crustle-v1-submit` |
| `rule:v10,noS` | `lucario_v10` | 12.8% | **50.0%** | the LB-950 notebook |
| `rule:archaludon` | `archaludon_ex` | 10.1% | **45.5%** ⚠ | `a-sample-archaludon-75-wr…` |

⚠ **Weight by share, always.** Every A/B in this repo before day 9 is a number
against *one* of these — usually `rule:v10` (pre-07-30) or the mirror + Crustle
(07-30/31). **A pre-day-9 number is not wrong, it is partial**; check which
anchor produced it before reusing it.

⚠ **Two anchors are new and their per-rule deltas are unmeasured.** In
particular `chip_target`'s wall branch hardcodes `WALL_POKEMON = {345}`
(Crustle), and **Archaludon's Full Metal Lab is a second damage-reduction effect
it has never seen** (−30 into any Metal Pokemon, and Hero's Cape puts Archaludon
ex at 400 HP). That is the most likely reason we lose that matchup.

⚠ **`crispin_toolbox` remains pilot-less and is now also low priority** — it did
not appear once in 109 real games, which is consistent with §1's box: it was
16.9% *of the top-1150 band*.

#### `rule:v10` — retired on 07-30, reinstated on 07-31

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

# The leaderboard, top 20
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); [print(i, r.team_name, r.score) for i, r in enumerate(a.competition_leaderboard_view('pokemon-tcg-ai-battle')[:20], 1)]"

# ⚡ THE FULL LEADERBOARD, ONE CALL (found 2026-07-31) -- USE THIS, not the
# pagination walk below. Writes out/lb/pokemon-tcg-ai-battle.zip containing one
# CSV of ALL 6,024 rows: Rank, TeamId, TeamName, LastSubmissionDate, Score,
# SubmissionCount, TeamMemberUserNames. Joining TeamName -> Score is what let
# day 10 rate every demonstrator in the training corpus (EVIDENCE §8q).
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_leaderboard_download('pokemon-tcg-ai-battle', path='out/lb')"

# ⛔ SUPERSEDED by the one-liner above; kept because it still works and the
# reasoning is report material. The client PRINTS "Next Page Token = ..." rather
# than returning it, so capture stdout and feed it back via page_token. This is
# how "1084.5 baseline" was refuted (its author is rank 750 at 819.1).
python -X utf8 -c "
from kaggle.api.kaggle_api_extended import KaggleApi
import io, contextlib
a=KaggleApi(); a.authenticate(); rows=[]; tok=None
for _ in range(40):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        batch=a.competition_leaderboard_view('pokemon-tcg-ai-battle', page_size=100, page_token=tok)
    if not batch: break
    rows+=batch; tok=None
    for line in buf.getvalue().splitlines():
        if 'Next Page Token' in line: tok=line.split('=',1)[1].strip()
    if not tok: break
print('rows', len(rows))
for i,r in enumerate(rows,1):
    if 'Scio' in (r.team_name or ''): print('RANK', i, r.score, r.team_name)
"

# Skill measurement: near-mirror head-to-head (rule 5). The only kind that counts.
python -X utf8 scripts/arena.py play "rule:v10,noS" rule:lucario `
    --deck-a lucario_v10 --deck-b mega_lucario_ex --matches 500

# Against the real bar
python -X utf8 scripts/arena.py play bc "rule:v10,noS" `
    --deck-a grimmsnarl --deck-b lucario_v10 --matches 500

# A/B a rule override against the pure clone (how every targeting.py rule is judged).
# Off-switches: noChip, noSpread, noSrc, noWall. Opt-in (default off): drag, dragHi, boss, veto.
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
# ⚡ --equiv: count a hit when the argmax option is BITWISE IDENTICAL to the
# chosen one. Those are two copies of ONE card in one role (two Trainers in the
# deck, two energies in hand onto the same target) -- picking either produces the
# same game, so plain top-1 charges the net for a coin flip. 30.2% -> 29.0%
# corpus-wide, TO_HAND 61.2% -> 67.1%. Use it for any agreement claim. EVIDENCE 8x.
python -X utf8 scripts/context_accuracy.py --net out/policy_b1_v3.npz `
    --ds artifacts/pds_v3r --equiv

# ── DAY 12: is the residual the ENCODING? Two probes, neither needs a net ──
# The CEILING. Bitwise-identical options get identical logits from ANY net, so
# sum(1/g)/N bounds top-1 for this layout. It is 95.6% and the clone gets 69.8%,
# i.e. un-expressibility explains at most 4.4 of the 30.2 points. --opt-cols 25
# reruns it against the v2 layout (the 8f control). EVIDENCE 8x.
python -X utf8 scripts/p17_encoding_ceiling.py --ds artifacts/pds_v3r

# THE FEATURE AUDIT, BY ENUMERATION. Diffs the observation against what
# featurize() actually reads, then SIZES each dropped field (rule 14: an absent
# input that is constant where the decisions happen explains nothing).
# ⛔ Use this instead of the remembered candidate list -- 3 of the 4 items that
# list carried for two days were already encoded. EVIDENCE 8y.
python -X utf8 scripts/p18_missing_state_audit.py --games 300

# The v4 state block: rebuild the corpus (byte-identical to pds_v3r plus
# xdense/xslots), then treatment and control on the IDENTICAL rows.
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds_v4/d26 `
    --ratings out/lb/pokemon-tcg-ai-battle.zip replays/2026-07-26   # ...d27-d29
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 --bs 1024 `
    --loss listwise --state-h 512,256 --head-h 256,128 --out out/policy_v4.npz
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 --bs 1024 `
    --loss listwise --state-h 512,256 --head-h 256,128 `
    --no-extra --out out/policy_v4ctrl.npz        # control: the v3 state vector

# ⚡ THE NOISE FLOOR, and every net-vs-net number in this repo needed it.
# Two IDENTICAL-recipe nets differing only in --seed measure 0.482 [0.460, 0.504]
# against each other -- a null, i.e. run-to-run variance is ~±13 Elo. Any A/B
# claiming less than that is claiming nothing. EVIDENCE 8z.
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v4 --epochs 12 --bs 1024 `
    --loss listwise --state-h 512,256 --head-h 256,128 --seed 1 `
    --no-extra --out out/policy_v4ctrl_s1.npz
python -X utf8 scripts/p6_recon.py --matches 120   # EVERY select, bucketed -- the menu
python -X utf8 scripts/p5_audit.py --matches 200   # sizes the three P5 findings
python -X utf8 scripts/p5a_replays.py              # the same counters on 55 REAL games

# What the SHIPPED v3 agent did against the real field (the arena's reality check).
# Reports the archetype mix, the Boss's Orders drag audit and the Froslass timing
# audit, all with honest denominators. EVIDENCE 8g.
# NOTE its archetype table uses a 4-card hardcoded classifier and buckets 63% as
# "other" -- use p9 below for the field, and p8 only for the two audits.
python -X utf8 scripts/p8_optv3_replays.py --dir replays/submission_optv3

# ⚡ WHAT THE FIELD ACTUALLY IS. The ONLY honest source -- our own games. Mining
# public episodes CANNOT answer this (they stop at avg_score 1055; we play at
# 825-952). Names every archetype by evolution LINE, ignores 1-of techs, and
# reconstructs each deck's card list. Pass both dumps to pool them. EVIDENCE 8i.
# ⚠ RE-RUN THIS AFTER EVERY SUBMISSION REPLAY DUMP -- the mix moves.
python -X utf8 scripts/p9_field_census.py `
    --dir replays/submission_optv3 replays/submission_replay_2026-07-29

# The two anchors the census said we were missing (idempotent; from notebooks/).
# rule:alakazam5 = the field's #1 deck (22.0%), a 5th-place pure-rules pilot.
# rule:archaludon = our worst matchup (45.5% WR over 11 real games).
python -X utf8 scripts/import_field_agents.py
python -X utf8 scripts/arena.py play bc rule:alakazam5 `
    --deck-a grimmsnarl --deck-b alakazam5 --matches 1000
python -X utf8 scripts/arena.py play bc rule:archaludon `
    --deck-a grimmsnarl --deck-b archaludon_ex --matches 1000

# Can a preserved bundle still be restored? (run from inside an extracted tarball)
python -X utf8 scripts/restore_smoke.py
python -X utf8 scripts/p5b_check.py --matches 150  # does a rule actually fire? (rule 9)

# Mine the TOP of the ladder. On disk: 07-26..07-30.
# ⚠ The CURRENT day 403s -- episodes publish the following day, so mine yesterday.
# 🔴 THIS IS NOT OUR FIELD. These datasets contain nothing below avg_score 1055
# and we play at 825-952. Use it for decklist consensus and report figures about
# the top of the board -- NEVER to decide which anchors to keep (EVIDENCE 8i).
python -X utf8 scripts/fetch_top_episodes.py --date 2026-07-30 --max 400
python -X utf8 scripts/mine_meta.py replays/2026-07-29    # takes dirs as arguments
powershell -File scripts/fetch_days.ps1        # several days; edit $Dates default (§7)

# Crustle: the counter-meta anchor (import once; idempotent)
python -X utf8 scripts/import_crustle_agent.py
python -X utf8 scripts/arena.py play bc rule:crustle `
    --deck-a grimmsnarl --deck-b crustle_v1 --matches 1000

# Is damage even landing? (the wall/counter census -- and the log-reading template)
python -X utf8 scripts/p2_lethal.py --matches 200          # lethal audit (closed)
python -X utf8 scripts/p3_crustle_probe.py --matches 60    # attack vs counter damage

# SIZE a rule before building it (rule 14). p7 is the per-TURN template -- resolve
# a decision once per turn, not once per select, or multiplicity inflates it.
python -X utf8 scripts/p7_morgrem.py --matches 200         # the Morgrem out (closed)

# Train (12 epochs; artifacts/pds_v2 is the shipped corpus)
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v2 --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 --out out/policy_X.npz

# ROADMAP B1: the feature A/B. artifacts/pds_v3 = 1,603 games at 37 opt-cols,
# rebuilt from the 4 raw replay days on disk. The CONTROL is the SAME rows
# truncated to the v2 layout (--opt-cols 25) -- so features are the only
# difference. `--opt-cols` exists for exactly this and nothing else.
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v3 --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 `
    --opt-cols 25 --out out/policy_b1_ctrl.npz        # control (v2 features)
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v3 --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 `
    --out out/policy_b1_v3.npz                        # treatment (v3 features)
python -X utf8 scripts/arena.py play "bc:v3,net=out/policy_b1_v3.npz" `
    "bc:ctrl,net=out/policy_b1_ctrl.npz" `
    --deck-a grimmsnarl --deck-b grimmsnarl --matches 1000 `
    --archive out/arena/b1_v3_vs_ctrl.jsonl

# Rebuild shards from raw replays (more data is NOT a lever -- EVIDENCE §1)
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds/d30 replays/2026-07-30

# ── ROADMAP B7 / day 11: WHO is demonstrating, and does it matter? (§8r-§8u) ──
# The full 6,024-row leaderboard in ONE call -- this is what makes any of it work.
python -X utf8 -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.competition_leaderboard_download('pokemon-tcg-ai-battle', path='out/lb')"

# Tag every row with the demonstrator's LB score. ⚠ ALWAYS read the coverage line
# it prints: the first run silently lost 24.6% of d26, and 92% of that was ONE
# team under three names (display name, member username, post-merge name). The
# fix is exact member matching (automatic) + replays/team_aliases.tsv (by hand).
python -X utf8 scripts/build_policy_dataset.py --out artifacts/pds_v3r/d26 `
    --ratings out/lb/pokemon-tcg-ai-battle.zip replays/2026-07-26   # ...d27-d29

# Agreement vs demonstrator rating. ⚠ DEFAULTS to the trainer's held-out split
# on purpose -- scoring all rows of a corpus the net trained on manufactures the
# very correlation being tested. --seen-from gives a real zero-exposure bucket.
python -X utf8 scripts/p15_rating_curve.py --net out/policy_b1_v3.npz `
    --ds artifacts/pds_v3r

# A REPRODUCIBLE control population: census any third-party dump from its
# owner's seat and emit the opponents on our archetype. ⚠ --exclude Scio, or the
# control contains our own agent and scores ~98% against itself.
python -X utf8 scripts/p9_field_census.py --dir replays/sixth_sense_31-07-2026 `
    --us "Sixth Sense" --us "Raja Biswas" --emit-players out/ctrl_players.txt

# Covariate shift: compare the two POLICIES to each other, not to human labels.
# Symmetric disagreement = a real policy difference; collapse on our own states
# = it was shift. artifacts/pds_ours doubles as the 1.7% positive control.
python -X utf8 scripts/p16_policy_disagree.py --a out/policy_b1_v3.npz `
    --b out/policy_b7_ntum.npz --ds artifacts/pds_ours artifacts/pds_ntum_r

# The two B7 nets, both KILLED (-55 and -92 Elo). Kept as reproducers only.
python -X utf8 scripts/train_policy.py --ds artifacts/pds_v3r --epochs 12 `
    --loss listwise --state-h 512,256 --head-h 256,128 `
    --rating-temp 25 --out out/policy_b7_rw25.npz     # ESS 41% of rows
python -X utf8 scripts/train_policy.py --ds artifacts/pds_ntum_r --epochs 30 `
    --lr 2e-4 --loss listwise --state-h 512,256 --head-h 256,128 `
    --init out/policy_b1_v3.npz --out out/policy_b7_ntum.npz

# Build + submit (smoke-tests the bundle the way Kaggle loads it)
python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc --nets policy

# ... with a CANDIDATE net + its rule flags pinned as a PAIR (the v3 config).
# --policy-net runs the dim guard at build time: a net this code cannot feed
# would otherwise ship happily and play random-legal on Kaggle. --no-rules is
# REQUIRED with a v3 net (the three rules measure 0.427 against it, EVIDENCE 8f).
python -X utf8 scripts/build_submission.py --deck grimmsnarl --agent bc `
    --nets policy --policy-net out/policy_b1_v3.npz --no-rules
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

  🔴 **These two dumps are the ONLY data in existence about the field we play.**
  Kaggle's public episode datasets are censored below `avg_score` 1055 and we sit
  at 825–952, so `replays/2026-07-*` cannot substitute (`EVIDENCE` §8i).
  **`replays/submission_*` must never be pruned**, and every future submission
  should have its replays pulled and fed to `p9_field_census.py`. ⚠ Each dump is
  ~50 games from **one agent at one rating**, so the mix moves between them
  (Lucario 20% vs 5%, Alakazam 13% vs 31%) — pool them, and treat any single
  archetype share as ±8 pp.
- **`replays/submission_optv3/`** — 56 files, **54 usable** (2 are bare
  step-arrays, not replays — skip anything where the JSON root is a list).
  **These are `55116557`'s games: the optfeat-v3 agent with every rule OFF.**
  The single most valuable diagnostic asset in the repo right now, because it is
  the only record of what our agent does against the **real field** rather than
  against our two anchors. Analysed by `scripts/p8_optv3_replays.py`
  (`out/logs/p8_optv3_replays.txt`); findings in `EVIDENCE` §8g.
  **Archetype mix — the number that invalidates the arena: 63% "other", Crustle
  24%, mirror 9%.**
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
  by timing). ~~Self-play RL dropped on the same evidence.~~ 🔴 **RETRACTED day
  14 — self-play RL was NEVER RUN.** It was a compute prior inherited from the
  search result and filed beside the measured negatives in four files for twelve
  days; there is no RL code in this repo or the old one. **Status is "never
  attempted", not "dead"** — the live objection is credit-assignment variance,
  and it must be SIZED before anything is built. `EVIDENCE` §2.
- **Boss's Orders — all four interventions null, the card is closed. Do not write
  a fifth.** `EVIDENCE` §6.
- **The Morgrem out is closed by SIZING, not by an A/B** — ~0.2 firings/game, the
  free route already 95.4% right, and a tradeoff besides. It also corrected "our
  attacker deals 0 into theirs": true of their **Active only**. `EVIDENCE` §8e.
- **Closed cheaply and correctly:** P5c never-end-without-attacking (3,683/3,683),
  `REMOVE_DAMAGE_COUNTER_COUNT` (100% already), post-KO promotion (9 misses/120
  games), `TO_HAND` duplicate-avoidance (parity), the decklist variant (0.490),
  P5a pooled Adrena-Brain (~0.5 real decisions per 200 games). `EVIDENCE` §8.
- **Do not resurrect:** the `rule:iono` arena→LB ladder; the old deck sweep's
  ranking; "the clone is comfortably above the rule baseline"; every n=24 number
  and every strength claim dated before 2026-07-27 pm; "3× compute made it
  worse". `EVIDENCE` §10.

⚠ **Everything above was measured against ONE opponent** — `rule:v10` on
`lucario_v10`. **That is far better news than day 8 thought.** Day 8 read it as
"measured against a dead deck" and discounted it; day 9 measured the actual field
and `lucario_v10` is **12.8% of it**, tied for the largest deck we face
(`EVIDENCE` §8i). So these results are *narrow*, not *stale* — they are one
genuine slice of the field, and the missing slices are the other four anchors,
not a replacement for this one.

The negatives are probably safe (a rule that does nothing against a real opponent
rarely becomes a winner against another). The **positives** still need the other
four anchors before they are treated as general.

⚠ **Open loose end:** the P2b "already at demonstrator parity" verdicts were only
re-derived for `munkidori_adrena_brain` after the P4c multiplicity fix; the
demonstrator side of the `opps` column has never been run
(`--corpus artifacts/pds_v2`). `EVIDENCE` §8.

---

## 7. Gotchas (all paid for)

- 🔴 **`cmd | tee log | grep ...` REPORTS THE EXIT CODE OF `grep`, NOT OF `cmd`.**
  The day-11 capacity run **crashed with a CPU OOM at epoch 8 of 12** and the
  harness reported **"completed (exit code 0)"**, because the last stage of the
  pipeline succeeded. The filtered view showed eight tidy epochs and no error;
  only reading the *unfiltered* log revealed the traceback. **A truncated run
  looks exactly like a finished one when you only read the grep.** Redirect
  (`> log 2>&1`) and echo `$?` when the exit status matters, and never conclude
  from a filtered log — this is rule 9 ("a metric that never prints is not a
  metric that passed") applied to the runner instead of the metric.
- ⚠ **MEMORY, not CPU, is this machine's binding constraint on model size**, and
  it bit twice in one hour. A 1.5M-param net (`--state-h 1024,512`) at
  `--bs 1024` OOMs mid-training on the 249k-row corpus; at `--bs 512` it OOMs
  during *data loading* — 231 MiB for one `(1633243, 37)` array — **whenever an
  arena process is running alongside it.** The `Data` class holds every shard in
  RAM plus per-row bag lists, so the load is a hard spike before a single epoch
  starts. **Do not run a large train and an arena concurrently** (rule 7 said
  2–3 jobs for CPU reasons; for the big nets it is 1), and **size this before
  planning any RL run** — a policy+value pair plus a replay buffer has to fit in
  what is left.
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
- ⚠ **A REJECTED NET DOES NOT CRASH — IT PLAYS RANDOM-LEGAL.** `policynet.load`
  returns `None` on a feature-dim mismatch and `PolicyAgent` falls back to
  `list(range(minCount))`, so a mis-built bundle smoke-tests "fine", uploads
  fine, and quietly scores ~600. Since 2026-07-31 `--policy-net` runs the dim
  guard at build time and the smoke asserts `NET_OK`. **Never ship a bundle whose
  build log lacks that line.**
- ⚠ **`dist/submission.tar.gz` is whatever was built LAST.** As of 2026-07-31 it
  is the **v3 + rules-off** candidate, not the live `lw2` agent. Check the
  timestamped filename before uploading.
- Kaggle Python API returns **snake_case** (`public_score`, `team_name`);
  `competition_leaderboard_view` paginates at 20 rows.
- **`obs["logs"]` is a per-observation DELTA, not a cumulative game log.**
  Observed lengths across our own selects: `[0, 0, 48, 14, 3, 1, ...]` —
  non-monotonic. **Never index into it as if it held the whole game**; concatenate
  deltas, or (better) tally events without needing offsets. This produced a probe
  that read 0.0 damage in every bucket including ones that cannot be zero
  (`EVIDENCE` §8d). Useful entry types: **`type 15`** = an attack
  (`cardId`, `attackId`, `playerIndex`); **`type 16`** = an HP change
  (`playerIndex` = the owner of the changed Pokemon, `cardId`, `value` negative
  for damage / positive for healing, and **`putDamageCounter`** True for
  placed/moved counters vs False for attack damage). ⚠ **A PREVENTED attack logs
  as `value: 0`**, so a filter of `value < 0` silently drops exactly the events
  that prove a prevention ability exists.
- **Third-party replay dumps (`replays/<team>_<date>/`) are a different animal
  from mined episodes, and three things about them bite:**
  - ⚠ **`info.TeamNames` is the display name AT EPISODE TIME and teams rename.**
    The Sixth Sense dump reports "Raja Biswas" on 113 games and "Sixth Sense" on
    30 — **one team, teamId 16452116**. A census keyed on the name splits one
    demonstrator in two. **Join on `teamId` from `episodes_meta.json`.**
  - ⚠ **A dump spans several of that team's SUBMISSIONS**, i.e. several different
    agents. `episodes_meta.json` carries `submissionId` per seat — use it, and
    tag rows so the weaker agent can be ablated out.
  - **The sidecar is not an episode.** `build_policy_dataset.py` now skips any
    file whose stem is not all digits; before that, `episodes_meta.json` was
    parsed as a replay and counted as `errors=1`.
- ⚠ **A player filter that matches nothing used to build a corpus of EVERYTHING.**
  An empty `--players-file`, or a CJK name off by one homoglyph
  (李秉**叡** vs 李秉**睿**), silently produced an unfiltered corpus under an
  expert corpus's name — the `bc:` label trap in a new place. **Both cases now
  `SystemExit`.** Take exact team names from `episodes_meta.json`, never retype
  them.
- ⚠ **`context_accuracy.py` scores the `gid % 20` val split by default.** On a
  corpus the net never trained on that silently measures **5% of your data**.
  Pass **`--all-rows`** for any external/expert corpus.
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
