# E29 — mine + archetype census: is the mirror still our field, and is there a second deck worth cloning? Pre-registered.

**Status: PRE-REGISTERED 2026-08-12 (day 31), before any pull and before any
statistic.** Frozen at the commit adding this file. Round context:
`ROUND-2026-08-12.md`. **Zero arena games** — this pulls replays and counts them.

**Why it goes first:** longest lead time in the round, and it **gates E30**
(§4). Nothing has been mined since `replays/2026-08-07/`.

---

## 1. What this cell is for, and the one thing it must not do

Three questions. The first two are the charter's; **the third is carried in
because it is nearly free once the mine lands, and because E30 cannot start
without it** (§4).

| | question | why it is not already answered |
|---|---|---|
| **Q1** | is the mirror still **71.4%** of our field above rating 1000? | that figure is **10 of 14 games**, Wilson CI **[40%, 83%]**, measured **08-01** before the top reshuffled, and quoted as hard in five places |
| **Q2** | what are the **current** top archetypes, and do we hold enough high-rating games of any of them to **clone**? | sizes the **08-15 two-deck decision** instead of guessing it |
| **Q3** | do **fresh** dumps contain Grimmsnarl-seat games against the archetypes the corpus is missing? | **E30's sizing gate.** The same gate already ran on **old** data and **failed** — §4 |

⛔ **Mining NEVER picks an anchor.** That is what retired `rule:v10`, which was
12.8% of our field. This cell emits **counts and shares**, never a deck choice
and never a training target. The 08-15 decision is the user's; E29 sizes it.

⛔ **No policy picks change in this cell**, so charter §3's realised-changed-
picks obligation does not bind. Recorded so its absence is not read as an
omission. ⛔ `val_top1` appears nowhere here.

---

## 2. ⛔ Step 0 — the two availability gates, before any statistic is computed

### 0a. A fresh leaderboard, because nothing can be banded without one

Every share in Q1 is **conditioned on opponent rating**, and the only rating
source is the LB join. The snapshots we hold are `out/lb_snapshot.json` (07-31)
and `out/lb_snapshot_0801pm.json` (08-01) — **both predate the reshuffle**, and
§8p's rule ("never compare a score across board sizes") makes a stale snapshot
actively misleading, not merely old.

⇒ Download the full LB in one call before anything else
(`competition_leaderboard_download`, HANDOFF §5). ⛔ **If the fresh LB does not
land, Q1 is VOID on instrument, not null** — a banded share computed against an
08-01 rating table is a measurement of where we played *then*.

### 0b. Does a fresh population exist at all?

Pull **manifests broadly, episodes selectively** (standing directive 1): fetch
`manifest.csv` only, for every dated dataset **2026-08-08 … 2026-08-12**, and
report per date: episode count, `avg_score` min/median/max.

⛔ **If no dated dataset exists past `2026-08-07`, Q2 and Q3 report VOID ON
AVAILABILITY, not null.** "We could not look" and "we looked and found nothing"
are different verdicts and the second one closes an axis. ⚠ Kaggle's feed stops
at `avg_score` ~1055 while we play at ~976 (§8i), so **every fresh dump
describes a band ABOVE ours** — that is a property of the instrument, not a
finding, and it is why Q1 may not be answered from these dumps (§3).

---

## 3. ⚠ The two frames — fixed before the pull, because conflating them is the whole trap

Three numbers about "the mirror share" are already on the record and **they do
not contradict each other; they are different denominators**:

| | frame | measured | value |
|---|---|---|---|
| §8ac | archetype of the opponents **we faced**, our own ladder replays, opponent rating ≥1000 | 08-01, **n=14 games** | **71.4%** |
| §8bq | both-seat composition of the **published top-episode feed**, `avg_score` ≥1100 | 08-03…08-07 | **23.7%** our archetype |
| day 26 | opponents **we faced**, `replays/submission_v5_s2`, **all bands pooled** | 08-08, n=76 at ~1027 | **31.6%** |

⛔ **Only the §8ac frame can answer Q1.** A number from the feed frame may never
be quoted as an answer to Q1, in either direction. ⚠ And the day-26 31.6% is
**not** a refutation of 71.4% either — it pools every opponent rating, and
§8ac's own table is monotone in that variable (5.3% → 18.6% → 42.4% → 71.4%).
**Comparing an unbanded share to a banded one re-commits rule 16's sampling-
frame trap**, which this repo has now committed twice.

⇒ Q1 is answered by **banding our own replays and nothing else**:
`p9_field_census.py --us Scio --lb <fresh> --dir replays/submission_v5_s2
--dir replays/ours_mirror_rec`, reported **per opponent-rating band** with the
same cut points as §8ac (`<800 / 800–900 / 900–1000 / 1000+`).

---

## 4. 🔴 What Q3 must not be narrated as: E30's gate already ran once, and failed

`PARKED-corpus-coverage.md` is titled `~~PARKED~~ CLOSED`. Its gate ran on
**day 25 (F3)** over the four dumps that built `pds_v4` (1,603 games,
`avg_score` 1057–1223): **Archaludon and Mega Lucario appear in ZERO of them**,
the miner discards nothing, and it was **killed on availability, not declined**
(`EVIDENCE` §8bi, `ROADMAP` §2.6).

⇒ **E30 is not an untouched lever. It is a closed one, and the ONLY thing that
can re-open it is that the fresh dumps are a different population** — which is
exactly the new fact the charter names (the top reshuffled completely). Q3 is
therefore a **re-run of a failed gate on new data**, and it must be written up
that way whichever answer it returns.

**Pre-registered gate thresholds**, per target archetype, on Grimmsnarl-seat
games only (the seat whose decisions would enter the corpus):

- **≥50 games AND ≥500 seat decisions** ⇒ the matchup is **buildable**; E30 may
  proceed to pre-registration for that archetype;
- **below either** ⇒ **unbuildable**, E30 closes at the gate for that archetype
  with **zero arena time spent**, exactly as §8bm/§8bp/§8br/§8bs closed.

⚠ **If every target archetype is unbuildable, E30 closes entirely and the round
has no arena cell.** That is a legitimate outcome and it is cheaper than the
alternative; §8ac already predicts it (those archetypes are **0 of 47** games
above opponent rating 900). ⛔ Do not rescue E30 by relaxing this threshold
after seeing the counts.

---

## 5. The instrument — existing scripts, no new estimator

⚡ **Nothing here needs to be built.** `p65_archetype_census.py` already labels
**both** seats of every replay from the *other* seat's frames, and already
reports matchup pairs, mirror share, band filtering (`--band`), and
**decisions by opponent archetype** — which is Q3's gate statistic exactly.

| question | command |
|---|---|
| **Q1** | `p9_field_census.py --us Scio --lb out/lb/<fresh> --dir replays/submission_v5_s2 --dir replays/ours_mirror_rec` |
| **Q2** | `p65_archetype_census.py --dir replays/2026-08-XX …` per fresh date, then pooled, with `--band` at the feed's own quartiles |
| **Q3** | the same `p65` run, read off its **DECISIONS BY OPPONENT ARCHETYPE** block (no `--player` ⇒ every Grimmsnarl seat) |

⚠ **Two known traps in these scripts, both already paid for and both live here:**
`MIRROR` must stay the full `Marnie's Grimmsnarl ex` (the short name silently
reports 0.0% on a 45%-mirror dump), and player matching is **substring**, not
equality (the demonstrator appears as `李秉叡（ntumlnoob）`).

⚠ **Labels are lower bounds by construction** — a card that never left the deck
is invisible, so every count reads "at least N". Shares are comparable to every
share already published *because* they share this bias; they are **not**
comparable to a decklist.

---

## 6. Reading rules — keyed on the COMPARISON, written before the pull

⚠ Charter §3 rule 2: key the branch on the comparison, not on one arm. Q1's
comparison is **a two-proportion test against 10/14**, never a point read
against 0.714 — with n=14 the baseline itself spans [40%, 83%].

| Q1 result (banded ≥1000, our own replays) | verdict |
|---|---|
| CI **excludes 71.4% from below** | 🔴 **the premise is refuted.** Five quotations are overclaims; the 08-15 decision is re-priced against the *measured* field, and §1 of the charter gains a sixth entry |
| CI **contains 71.4%** and also contains 31.6% | ⚠ **UNDERPOWERED, not confirmed.** Report the interval, quote neither endpoint as hard, and say so in every place the number is used |
| CI **excludes 71.4% from above** | the mirror is *more* dominant than believed ⇒ the two-deck hedge gets *more* expensive, not less |
| banded **n < 30** | ⛔ **Q1 is UNDERPOWERED BY SIZING.** Report the count and stop. ⛔ Do not pool bands to reach n — that is the trap being tested |

| Q2 result | consequence for the 08-15 decision |
|---|---|
| a current top archetype clears **≥50 games / ≥500 decisions** at high rating | directive 2 becomes a **sized** question: name the archetype and its count, hand it to the user |
| none does | directive 2 **stays a variance hedge** and is priced exactly as §8al already prices it (~17–25 Elo) |

⚠ An archetype with **<30 seats** in the fresh feed is listed as **"seen, not
sized"** and gets no share quoted. Small-denominator shares are what produced
the 71.4% problem this cell exists to fix.

---

## 7. ⚠ Limits, stated before the result

- **The feed is censored at `avg_score` ~1055 and we play at ~976.** Q2
  describes a band above ours. It is the right frame for "is there an archetype
  worth cloning" (we clone *up*) and the **wrong** frame for "what will we
  face" (Q1's frame).
- **Q1's denominator is our own ladder games, which are a function of our own
  rating** (§8ac / rule 16). A re-census at ~976–1027 answers *where we play
  now*, and it will move again if we move.
- **Both censuses are lower-bound reconstructions** (§5), so a rare tech card is
  systematically under-counted relative to a deck's core.
- ⛔ **Q3 sizes availability, not value.** "The games exist" does not imply
  training on them buys Elo — E6→E7 is exactly that trap, and
  `PARKED-corpus-coverage.md` already carries the warning: *a compelling
  diagnosis is not a working repair.*

---

# ▶ RESULT — 2026-08-12

Instruments: `scripts/p90_e29_band.py` (Q1), `fetch_top_episodes.py` +
`p65_archetype_census.py` (Q2/Q3).

## R0. ✅ Both Step-0 gates pass — and the feed's ceiling has MOVED

**0a. Fresh LB, 6,771 rows.** ⚠ Board grew 6,483 → 6,771 and **the top
reshuffled again**: `Luca` 1217.2 is now #1 (it was #7 on 07-31) and `LiamK`,
#1 on 08-07 at 1166.0, is #8 at 1162.2.

📌 **We read `Scio`, rank 254, 933.1** — against rank 129 / 990.7 on 08-07.
⚠ **Stated with the repo's own two rules attached and NOT as a regression:**
a −57.6 move is **inside the 63-point noise floor** (§8ak), and rank is not
comparable across board sizes (§8p). **Nothing has gone wrong and nothing is
confirmed.** It is flagged because it is an input to the 08-15 decision.

**0b. Fresh population exists.** Datasets published for **08-08 … 08-11**
(08-12 not yet up, 403):

| date | episodes | `avg_score` min / median / max | ≥1100 |
|---|---|---|---|
| 08-08 | 4,669 | 988 / 1027 / **1300** | 617 |
| 08-09 | 4,668 | 982 / 1021 / 1211 | 418 |
| 08-10 | 4,603 | 984 / 1025 / 1210 | 420 |
| 08-11 | 4,622 | 984 / 1027 / **1252** | 657 |

**18,562 fresh episodes, 2,112 at ≥1100.**

⚡ **§8i's censoring claim is stale and this matters for E30.** The feed was
recorded as stopping at `avg_score` **~1055**; it now reaches **1300**. That is
the single fact that could legitimately re-open the availability kill of §4 —
and it is why Q3 is worth running rather than assumed.

## R1. 🔴 Q1 — the 71.4% does not reproduce, and the MONOTONE CURVE is the bigger casualty

76 of our ladder games (`submission_v5_s2`), opponents joined to the fresh LB
(4 unmatched), banded on §8ac's own cut points:

| opponent band | games | mirror | share | 95% Wilson | **§8ac (08-01)** |
|---|---|---|---|---|---|
| <800 | 13 | 0 | **0.0%** | [0.0%, 22.8%] | 5.3% |
| 800–900 | 14 | 4 | **28.6%** | [11.7%, 54.6%] | 18.6% |
| 900–1000 | 24 | 13 | **54.2%** | [35.1%, 72.1%] | 42.4% |
| **1000+** | **21** | **6** | **28.6%** | **[13.8%, 50.0%]** | **71.4%** |

### ⛔ The formal verdict is the pre-registered one: UNDERPOWERED BY SIZING

The 1000+ band holds **21 games against the frozen ≥30 floor**, so by §6 the
verdict is **UNDERPOWERED**, bands were **not** pooled to reach n, and neither
endpoint may be quoted as hard. That rule stands as written.

### ⚠ But the frozen gate was set on the wrong quantity, and the data is decisive anyway

Recorded because suppressing it would be its own distortion. At the realised n
the two-proportion comparison against 10/14 reads:

**28.6% vs 71.4% — difference −42.9 pp, 95% CI [−73.4, −12.3], z = −2.75.**
The interval **excludes zero**.

✅ **Robust to the one confound available to check.** Banding 08-08 games by a
08-12 LB dates opponents late; re-banding with `lb_snapshot_0801pm.json` gives
**27.8% (5/18)** against 28.6% (6/21). **The choice of snapshot does not move
it.**

⚡ **The lesson is E28's, one day old and in a different cell: a gate on a fixed
`n` is a gate on the wrong variable — power depends on the effect size.** A
−42.9 pp effect is detectable at n=21; the ≥30 floor was written to guard
against small-denominator overclaiming (the very disease 10/14 has) and it
mis-fires when the effect is large.

⇒ **Practical reading, stated with its status: 71.4% should not be quoted as a
hard number anywhere.** The formal re-census is underpowered; every check that
can be run points the same way; ⛔ and it is still **not** a licence to quote
28.6% as hard either.

⚡ **And the feed frame agrees, from a completely independent sample** (§R2):
our archetype is **10.5%** of 3,200 fresh top-band seats. Two different frames,
two different denominators, both saying the mirror is not what it was.

### 🔴 The finding that outranks the number: the monotone curve is not there

§8ac's planning claim was *"the mirror is the matchup that matters and **gets
more so as we climb**"* — 5.3 → 18.6 → 42.4 → 71.4, monotone. The re-census
reads **0.0 → 28.6 → 54.2 → 28.6**: it **rises and then falls**, peaking at
900–1000, which is *our own band*, and **diversifying above it**.

**The 1000+ band, in full (n=21):**

| archetype | games | share |
|---|---|---|
| **Marnie's Grimmsnarl ex** | 6 | **28.6%** |
| **Alakazam** | 6 | **28.6%** |
| Mega Lucario ex | 2 | 9.5% |
| Meowth ex | 2 | 9.5% |
| Dragapult ex | 2 | 9.5% |
| Teal Mask Ogerpon ex / Dudunsparce / Slowking | 1 each | 4.8% each |

⇒ **At the top the mirror is not dominant — it is tied with Alakazam**, which
day 26 already flagged as "back at 23.7% after being dropped from planning since
day 9". ⚠ Every verdict re-weighted by §8ac's monotone shares inherits this.

### ⛔ Why this cannot be powered up from the mine

n can only rise with **more of our own ladder games**, and those are not
API-reachable — `submission_v5_s2` was **supplied by the user** (day 26). ⛔ The
fresh feed is the **wrong frame** (§3) *and* would bias the sample: the feed
publishes only high-`avg_score` episodes, so our games in it are selected for
strong opponents. **Q1 is blocked on a fresh dump of our own games, which is a
user action, not a Kaggle call.**

## R2. Q2 — the current top, from 1,600 fresh episodes (`avg_score` 1102–1300, mean 1153)

Top-400 by `avg_score` from each of 08-08…08-11; 3,200 seats, both seats
labelled by lower-bound reconstruction.

| archetype | seats | share | | archetype | seats | share |
|---|---|---|---|---|---|---|
| **Dragapult ex** | 581 | **18.2%** | | Teal Mask Ogerpon ex | 228 | 7.1% |
| **Mega Lucario ex** | 389 | **12.2%** | | Hydrapple ex | 179 | 5.6% |
| **Alakazam** | 358 | **11.2%** | | Slowking | 151 | 4.7% |
| **Marnie's Grimmsnarl ex (us)** | 337 | **10.5%** | | Mega Lopunny ex | 96 | 3.0% |
| Dudunsparce | 239 | 7.5% | | Meganium | 94 | 2.9% |

🔴 **The top is a four-way field and we are fourth in it.** ⚠ Against §8bq's
**23.7%** for our archetype in the ≥1100 feed of 08-03…08-07, this reads
**10.5%** — the share **more than halved in one week**, on the same frame.

🔴 **The mirror is 1.1% of games (17 of 1,600)** in this band, and **11.6% of
all Grimmsnarl-seat decisions**. ⚡ Set against the training corpus's **63.9% of
opponent board slots**, the composition mismatch is **worse than
`PARKED-corpus-coverage.md` measured**, not better.

⚡ **Dragapult ex is the story of the week.** It is 18.2% of the top band and
the single most common matchup pair in it (`Dragapult vs Mega Lucario`, 4.8% of
all games). It held **5.3%** field weight in §8ac and **2.74%** of corpus slots.

### ⇒ What Q2 hands the 08-15 decision

Per §6's branch: **no current top archetype clears ≥50 games / ≥500 decisions**
(§R3), so **directive 2 stays a variance hedge and is priced exactly as §8al
prices it (~17–25 Elo).** ⛔ Q2 does **not** name a second deck, and mining
never picks an anchor.

⚠ **But the input to that decision has changed** and the user should have it:
our archetype is no longer the top band's largest bucket in either frame, and
the deck reads Tier 2 outside the sim. Nothing here says a second deck would
score better — only that *"the mirror is where we live"* is no longer the
premise it was on 08-01.

## R3. 🔴 Q3 / E30's gate — no target archetype clears it, and one is still absent entirely

Grimmsnarl-seat games in the 1,600 fresh episodes, against
`PARKED-corpus-coverage.md`'s own target set — **under-represented by MORE than
3×** (ratio < 0.33), which is the set that defines the lever:

| archetype | field | corpus | ratio | **target?** | games | decisions | **gate** |
|---|---|---|---|---|---|---|---|
| **Alakazam** | 22.0% | 3.39% | **0.15×** | **yes** | 37 | 2,687 | 🔴 **FAIL** (games) |
| **Archaludon ex** | 8.0% | 0.02% | **0.00×** | **yes** | **0** | **0** | 🔴 **FAIL** (absolute) |
| **Cynthia's Garchomp ex** | 6.7% | 0.40% | **0.06×** | **yes** | 5 | 449 | 🔴 **FAIL** (both) |
| **Mega Lucario ex** | 4.0% | 0.00% | **0.00×** | **yes** | 38 | 3,116 | 🔴 **FAIL** (games) |
| Dragapult ex | 5.3% | 2.74% | 0.52× | no | 55 | 4,777 | passes — but never a target |
| Crustle | 6.7% | 6.26% | 0.93× | no | 10 | 1,257 | — |

⛔ **Nothing in the target set passes.** The one archetype clearing both
thresholds is the one the parked file explicitly excluded: *"the 0.93× and 0.52×
rows are not load-bearing; the 0.00× / 0.06× / 0.15× rows are unambiguous."*
⛔ **E30 is not rescued by building on Dragapult** — that answers a question
nobody asked — and the threshold is not relaxed after seeing the counts (§4).

### ⚡ But the day-25 kill's REASON is now false, and the distinction matters

| | day 25 (F3) | **today** |
|---|---|---|
| Mega Lucario ex | **0 games**, "cannot be obtained" | **38 games, 3,116 decisions** |
| Archaludon ex | **0 games** | **0 games** |
| feed ceiling (§8i) | `avg_score` ~1055 | **1300** |

⇒ **The kill reason shifts from "these games do not exist" to "they exist and
are still too thin to clone."** For **Archaludon** the original kill stands and
is now confirmed on a second independent population — **0 of 1,600 top-band
games.** ⚠ Recorded precisely because "killed on availability" has been quoted
as settled, and half of it no longer is.

### ⛔ The failure is a PULL-DEPTH limit, not an availability limit

We pulled **1,600 of 18,562** available fresh episodes — **8.6%**. Alakazam
needs **1.35×** and Mega Lucario **1.32×** the current depth to clear 50 games.
**Deepening the pull MEETS the threshold; it does not relax it**, and mining is
this cell's own remit. A pull to top-900/date is running.

⚠ **Deeper pulls lower the `avg_score` cutoff**, moving the sample toward
~1050–1100 — still above our ~933 band, and *closer* to it. That is a change in
the sampled population and must be reported with any count it produces.
