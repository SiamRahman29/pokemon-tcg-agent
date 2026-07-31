# HANDOFF — PTCG AI Battle (Kaggle `pokemon-tcg-ai-battle`)

**Mission:** public LB **and** the Strategy Category. Sim deadline **2026-08-17**,
then ~2 weeks continued play; strategy report due **2026-09-14**. Kaggle CLI is
authenticated.

**Standing (read 2026-07-31 ~03:50 UTC, full LB — now 5,000 rows):** we are
**`Scio`, rank 605 of 5,000, 837.5** — down from **rank 224 at 950.2**, and
**none of that drop is play strength.** It is the eviction: see the box below.
Top is now `Brahim` 1162.4, then `James Cox & Henry Chao` 1155.3, `Raja Biswas`
1146.7. The board grew from 3,000 → 5,000 entrants in ~2 days and the top
reshuffles constantly — treat any ranking as a snapshot, and **paginate:
`competition_leaderboard_view` returns 20 rows and a `Next Page Token`; pass it
back via `page_token` to walk all 5,000** (§5).

> # 🔴 READ THIS FIRST — THE "−130 REGRESSION" WAS LARGELY AN ARTIFACT (day 9, 2026-07-31)
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

**Read §2 before trusting any number. §3 is the live plan. This file must always
end with a live plan, never a summary.**

⚠ **Day 9 note on reading this file:** several load-bearing claims dated
2026-07-30 were **narrowed, not deleted** — the meta-shift table (§1), rule 12's
"`lucario_v10` is 0% of the meta", and rule 16's "the arena does not measure
ladder strength". Each is now prefixed with what it is actually true *of*. If a
statement in here about "the meta" does not say **which score band** it describes,
distrust it: mined episodes are the top-1150 band, and
`scripts/p9_field_census.py` on our own replays is ours (`EVIDENCE` §8i).

### ▶ START HERE — the next actions, in order (set 2026-07-31, day 9 pm)

**Day 9 answered the question day 8 ended on.** The blocking problem — "no arena
number in this repo predicts ladder strength" — is **closed** (§8i): the arena
predicts fine, the anchor set was wrong, and it is now fixed. Everything below is
ordinary work again.

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

1. **⚠ THE P4b RESTORE IS NOW A GENUINELY OPEN QUESTION — DO NOT DO IT ON
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

3. **⚡ THE LIVE ENGINEERING LEAD: the Mega Lucario matchup.** Two independent
   instruments agree, which nothing else in this project has managed:
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

   ⚠ **And run the same 5-anchor sweep for the three rule flags.** We know the
   rules are harmful to v3 *in the mirror* (0.427); we do not know their sign
   against Lucario or Archaludon. **Option (b) — v3 with the rules back on —
   is still untested and is the cheapest thing on this list.**

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
6. **`report/STRATEGY.md` does not exist yet** — the only Track B deliverable not
   started. `report/EVIDENCE.md` is backfilled and ready to draft from.

### The four files, and what each owns

| file | owns |
|---|---|
| **`HANDOFF.md`** (this) | live state, the live engineering plan, the anti-self-deception rules, commands, gotchas |
| **`ROADMAP.md`** | the strategy-competition plan — what the engineering is *for*, the breakthrough hunt, the calendar |
| **`report/EVIDENCE.md`** | the hypothesis log: every concluded experiment with n, CI, verdict. **All closed-experiment detail lives there, not here.** |
| **`competition_details_and_rubric.md`** | the rubric, verbatim |

**End of every session: update HANDOFF (plan), ROADMAP (calendar), and
EVIDENCE (any experiment that concluded) together.**

> **Submission state (2026-07-31). ⚠ The previous version of this box was WRONG
> on the one point that mattered — see the ✅ below.**
>
> - **Daily quota: 5/day.** Never the binding constraint.
> - **Only the latest 2 submissions play episodes.** Active pair right now:
>   **`55116557` (v3, 824.6) + `55077709` (P6a, 837.5)**. `55072063` (**952.0**)
>   was **evicted** by the v3 submission and is frozen.
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
> be evicting" — and a rollback now DOES qualify**, because the thing it restores
> (952) is more than the thing it evicts (837.5). That is the reverse of what this
> box said yesterday, and the reason is §8h, not a change of heart.

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

# The FULL leaderboard (3,000 rows) -- to find our own rank or check a claim.
# The client PRINTS "Next Page Token = ..." rather than returning it, so capture
# stdout and feed it back via page_token. This is how "1084.5 baseline" was
# refuted (its author is rank 750 at 819.1).
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
  by timing). Self-play RL dropped on the same evidence. `EVIDENCE` §2.
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
