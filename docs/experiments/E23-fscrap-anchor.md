# E23 — E21's VOID arm, re-run where the condition can exist. Pre-registered.

**Status: PRE-REGISTERED 2026-08-11 (day 30), before the first arena game of any
cell.** Frozen at the commit that adds this file. Numbered E23 because E21 is the
Petrel fetch experiment this one repairs and E22 is the pessimistic-lookahead
pre-registration.

---

## Why this exists: a VOID cell is a debt, not a result

§8cc ran `bc:e21,fscrap` in the mirror and it read **0.5175 with
`fetch=0/3082 (0.0%)`** — the rule never fired once. `decks/grimmsnarl.py` runs
**zero Pokémon Tools**, so "a Tool is on THEIR board" is unsatisfiable in the
mirror *by construction*, and the score is a statement about wiring, not about
the hypothesis. §8cc's own verdict was **"size the condition IN THE MATCHUP THE
CELL WILL RUN IN"**. This is that verdict executed.

**H-scrap (unchanged from E21):** at the one select whose option vector carries
no board at all (§8br), supplying the board fact directly — fetch Tool Scrapper
when, and only when, a Tool is on the opponent's board — beats the clone.

## The sizing, run first and ON-POLICY (`scripts/p89_fscrap_sizing.sh`)

The flag is ON in every sizing run, so `fetch_fired` is what the rule really
does, not an off-policy replay estimate. §8cc measured that replay sizing
**under-predicts** (0.72 realized vs 0.461 sized, 1.6×), so this is the honest
direction to be wrong in.

200 games per anchor, `bc:p89,fscrap` vs each rule pilot **on its own tuned 60**
(rule 20). Tool counts are from the card DB (`cardType == 2`):

| anchor | deck | Tools in the 60 | fetches seen | fired | **per game** |
|---|---|---|---|---|---|
| `rule:v10,noS` | `lucario_v10` | 1 (Hero's Cape) | 276 | 60 | **0.300** |
| `rule:lucario` | `mega_lucario_ex` | 1 | 258 | 54 | 0.270 |
| `rule:archaludon` | `archaludon_ex` | 1 | 306 | 45 | **0.225** |
| `rule:dragapult` | `dragapult_ex` | 1 (Lucky Helmet) | 262 | 40 | 0.200 |
| `rule:crustle` | `crustle_v1` | 1 | 349 | 23 | 0.115 |
| — mirror (§8cc) | `grimmsnarl` | **0** | 3,082 | **0** | **0.000** |

⚡ **The firing rate is real but the treatment is what matters, and they are not
the same number.** A firing that agrees with the net is a no-op for the A/B.
`bcagent.STATS` now carries **`fetch_diff`** — firings whose pick differs from
`net.choose` — measured at **10 of 11** in a 30-game probe vs `lucario_v10`,
consistent with §8br's off-policy "conditioned on a target existing we take
Scrapper 1 time in 14". ⇒ treatment ≈ **0.30/game** against the primary anchor.

⚠ **Under the 0.5 gate, and run anyway — the same reason E21 gave, stated again
rather than assumed.** Rule 14 gates what is worth **building**; the rule is
already built and a 4,000-game anchor cell costs ~9 minutes here. What the gate
*does* still bind is the **ship** decision, and this cell cannot reach one.

## ⛔ The scope limit, written before the result

Our 60 runs **no Pokémon Tool**, so this rule is **structurally incapable of
firing in the mirror** — and the mirror is **71.4% of our field above rating
1000** (§8ac). ⇒ **Even a clean win here is matchup-specific tech, not a ladder
lever.** It would be worth at most 0.30 firings/game across the ~29% of the
field that runs Tools. That is stated now so that a positive result cannot be
narrated into something bigger later.

## 🔴 The prior, recorded before the result — and it is against H-scrap

E21's doc filed `fscrap` as *"closer to the dominated column"*. **That was
wrong, and the correction matters more than the cell.** The discriminator (§3)
separates rules that **delete a dominated option** (3/3) from rules that **pick a
side in a tradeoff** (0/6). As implemented, `fscrap` *promotes* Tool Scrapper
over Unfair Stamp and Night Stretcher — all three are live cards we chose to
run — so it is a **tradeoff rule**, exactly like `fstad`, which lost at z=−5.36.

The genuinely dominated-class version is the opposite rule: **delete Tool
Scrapper from the fetch options when no Tool is on either board** — fetching a
card that cannot do anything is strictly dead. §8br sized that at **0.066
plays/game**, and the mirror version at the net's 5.1% take rate over 1.59
fetches/game is **~0.08/game**. **Both are an order of magnitude under what
n=4,000 can resolve**, so the class the discriminator says WINS is unmeasurable
here and the class it says LOSES is the only one large enough to test. That
asymmetry is the finding this cell will produce regardless of its own sign.

⇒ **My prediction, written first: `fscrap` reads NULL** (Δ CI contains 0), making
tradeoff rules 0-for-7.

## The cells

Both arms use the same net (`out/policy_v5_s2.npz`), the shipped rule config
(`chip`, `spread`, `src`, `wall` at their PolicyAgent defaults = `bc:base`), and
differ **only** by the flag. The anchor is identical in both arms, so this is a
**two-cell delta**: its interval is √2× a single cell's (§8aw), and the driver
must print that width, not a single cell's.

| cell | A | B (anchor) | deck-b | n |
|---|---|---|---|---|
| **E23a (primary)** | `bc:e23,fscrap` | `rule:v10,noS` | `lucario_v10` | 4,000 |
| **E23a-ctl** | `bc:base` | `rule:v10,noS` | `lucario_v10` | 4,000 |
| **E23b (exploratory)** | `bc:e23,fscrap` | `rule:archaludon` | `archaludon_ex` | 2,000 |
| **E23b-ctl** | `bc:base` | `rule:archaludon` | `archaludon_ex` | 2,000 |

**E23b is reported separately and never pooled with E23a** — different anchor,
different firing rate, and pooling two matchups is how §8i had to be retracted.

## The bars, written before any result exists

Δ = score(`fscrap` arm) − score(`base` arm) against the same anchor.
At n=4,000/cell with scores near 0.68, SE per cell ≈ 0.0074 ⇒ **SE(Δ) ≈ 0.0105**.

| branch | condition | reading |
|---|---|---|
| ✅ **screen passes** | Δ ≥ **+0.030** AND 95% CI excludes 0 | a candidate, **not a ship** — needs a mirror-neutrality cell (it cannot fire there, so the prediction is exactly 0.500) and the five-anchor sweep (rule 16) |
| 🔴 **KILL** | CI contains 0 | H-scrap refuted at this n for this matchup; tradeoff rules go 0-for-7 and the Petrel seam closes a third time |
| ⚠ **harmful** | Δ ≤ **−0.030**, CI excludes 0 | audit the condition before anything else, as E21a's harmful branch required |

⛔ **No threshold tuning after reading a result** — E21's binding clause, carried
verbatim. "Fetch it only when *two* tools are out" is a separate experiment, not
a knob.

## Controls

1. 🔴 **VOID condition, and it is the whole reason this experiment exists.** The
   treatment arm must print **`fetch_diff` ≥ 0.10/game** (sized 0.30). Below
   that the cell is **VOID, not a null** — E21b's exact failure, and a null with
   no treatment is a statement about wiring (§8bz).
2. **The control arm must print no `fetch=` field at all**, which is what
   `bc:base` does when both flags are off — proof the arms differ only in the
   flag.
3. **Rule 20:** each anchor runs on its own `DECK_MODULE` deck, so no generic
   fallback is in play (worth +0.140 in the one case measured, §8ax).
4. **Rule 18:** pool from arena's own printed `A=` W/D/L lines, never by
   re-deriving from the archives. Every shard writes its **own** `--archive`
   under `out/arena/e23/`; nothing touches `out/arena/games.jsonl`.
5. `rule:v10` runs **`noS`** — its MCTS budgets on wall-clock and the shards
   contend for cores, which would make the anchor itself a function of load.

## ➕ Added 2026-08-11 AFTER cell b read, BEFORE the replication ran

Cell a (primary) read **+0.0041 [−0.0168, +0.0250]** — the pre-registered KILL
branch, controls clean (0.261 changed picks/game). Cell b (exploratory) read
**−0.0317 [−0.0596, −0.0039], z = −2.24**, which trips the harmful branch by
0.0017 with the CI clearing zero by 0.0039, and whose stated response is
*"audit the condition before anything else"*.

**The audit is a REPLICATION, and its reading rule is frozen here before it
runs** (cell `b2`: same anchor, same deck, same n, fresh games, nothing tuned):

| outcome | reading |
|---|---|
| b2's CI contains 0 **and** the pooled b+b2 delta's CI contains 0 | cell b was a sampling artifact. Verdict for the whole experiment is the primary's: **KILL**, no matchup-specific harm claimed |
| b2 reproduces ≤ −0.030 with CI excluding 0 | the harm is real and archaludon-specific ⇒ report it as such, and it is a **second** instance of a board-fact rule losing where the clone's blindness was better |
| b2 positive with CI excluding 0 | the two cells disagree at n=2,000 ⇒ **the anchor cell is under-powered for this effect size** and neither b nor b2 is quotable |

⚠ **Why I do not believe cell b as it stands, written before b2 exists.** The
implied per-changed-pick effect is **−0.0317 / 0.168 = −0.19 win probability per
fetch**, against cell a's **+0.0041 / 0.261 = +0.016**. A 12× disagreement in
magnitude and a disagreement in sign, between two cells of the same rule, is the
signature of noise rather than of a matchup. It is also the second of two cells
and carries no multiplicity correction. **z = −2.24 is not a finding; it is a
reason to spend ten more minutes.**

## ⛔ Void / out of scope

* **Nothing here may be submitted**, whatever it reads. E20/E22 own the
  submission slots this window and a screen never ships (§8bh).
* **The mirror is not re-run.** §8cc already established the rule cannot fire
  there; re-running it would buy another accidental C0 and nothing else.
