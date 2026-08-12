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
