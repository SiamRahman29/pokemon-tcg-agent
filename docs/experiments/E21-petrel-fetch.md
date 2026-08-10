# E21 — board facts in Petrel's fetch. Pre-registered.

**Status: PRE-REGISTERED 2026-08-11 (day 30), before the first arena game of
either cell.** Frozen at the commit that adds this file. User-directed: *"test
in the arena to see if the experiments pan out instead of comparing if the
experiments make us similar to the experts, because we are trying to overtake
the experts."*

---

## The hypothesis

§8br's structural addendum: **a Petrel fetch option's feature vector contains
nothing about the board.** The whole v3 target block (`hp`, damage taken,
dies-to-30, energy count, `best_damage`) is identically zero, because it
resolves a Pokémon at `(player, area, index)` and the deck is not an in-play
area. One fetch option differs from another **only by its card embedding and
card type**. Everything situational must arrive through `srepr`, concatenated
identically to every option, discriminating only through the head MLP's
interaction term.

§8cb then measured the behavioural consequence on a different card: Unfair
Stamp is played on **56 of 56** legal turns — not a worse judgement, the
*absence* of one.

**H-fetch:** supplying the board fact directly, at the one select that cannot
see it, beats the clone.

## ⚠ These rules are NOT derived from expert behaviour, deliberately

§8u measured that agreement with the **field** predicts strength while
agreement with the **expert** anti-predicts it (rating-weighted clone −55 Elo,
single-expert clone −92). E11 then took a sized, ordering-free, confound-checked
expert gap (0.80 plays/game) and measured **0.487**. So "match what the 1150s
fetch" is a discredited generator. Both conditions below come from **card text
plus the board**:

* **`fstad`** — Spikemuth Gym is the only Stadium in the 60, and our entire
  evolution line is Marnie's, so it is a repeatable engine tutor for us and much
  weaker for most opponents. Fetch it when **no Stadium is in play, or the one
  in play is the opponent's** — the state where playing it both starts our
  engine and removes theirs.
* **`fscrap`** — Tool Scrapper does nothing unless a Tool is attached. Fetch it
  only when a Tool is on **their** board.

## Sizing, run before either rule was written (rule 14)

Over our 76 ladder games (`replays/submission_v5_s2`, 121 Petrel fetches):

| | condition holds | rule would CHANGE the pick | per game |
|---|---|---|---|
| `fstad` | 41 | **35** | **0.461** |
| `fscrap` | 14 | 13 | 0.171 |

`fstad` is the largest firing rate anything in this seam has produced — 15× the
Unfair Stamp decline rule (0.031) and 2.7× `fscrap`.

⚠ **`fscrap` is under the 0.5 gate and is run anyway**, with the reason stated
rather than hidden: an n=2,000 mirror A/B costs **~6.5 min** here (measured:
20 games in 3.9 s). The 0.5 gate was calibrated when the ladder was the
instrument and an A/B was expensive. **Rule 14 gates what is worth BUILDING;
it is not a law about what is affordable to measure.** `fscrap` is labelled
exploratory and cannot ship on this cell alone.

## 🔴 The prior, recorded before the result — and it is against H-fetch

By this project's own discriminator (§3): **rules that delete a *dominated*
option win 3/3; rules that pick a side in a *tradeoff* lose 0/5.** Choosing
Spikemuth Gym over Unfair Stamp or Night Stretcher is a **tradeoff** — every
option in a Petrel fetch is a card we chose to run. `fscrap` is closer to the
dominated column (Scrapper with no Tool on board is a strictly dead fetch), but
it is the arm with the weaker sizing.

⇒ **My prediction, written first: `fstad` reads NULL.** If it does, that is the
sixth tradeoff rule to fail and the discriminator gets its strongest test yet —
because this is the largest firing rate the class has ever had, so "it was too
rare to matter" is unavailable as an excuse.

## The cells

Mirror, **byte-identical weights on both sides** (`net=out/policy_v5_s2.npz`),
`--deck-a grimmsnarl --deck-b grimmsnarl`, rules identical except the flag under
test. The ±13 Elo seed nuisance (§8bg/§8bk) cancels exactly in a flag-toggled
A/B, so no seed budget is needed.

| cell | A | B | n |
|---|---|---|---|
| **E21a (primary)** | `bc:e21,fstad` | `bc:base` | 2,000 |
| **E21b (exploratory)** | `bc:e21,fscrap` | `bc:base` | 2,000 |

## The bars, written before any result exists

| branch | condition | reading |
|---|---|---|
| ✅ **screen passes** | point ≥ **0.530** AND CI excludes 0.500 | a candidate, **not a ship** — §8bh's `s7` screened 0.528 and read 0.487 on 2,800 fresh games. Confirmation on fresh games + the five-anchor set would be required |
| 🔴 **KILL** | CI contains 0.500 | H-fetch refuted at this n for this arm. The rule stays in the tree, OFF, and becomes a report chapter |
| ⚠ **harmful** | point ≤ 0.470, CI excludes 0.500 | the injected fact is actively wrong ⇒ audit the condition before anything else |

At n=2,000, SE ≈ 0.0112, so 0.530 is a **2.7σ** bar.

## Controls

1. 🔴 **The rule must fire, and at the rate it was sized at.** `bcagent.STATS`
   now carries `fetch_seen` / `fetch_fired`, printed in `health_line`. **A null
   with a zero firing count is a statement about the wiring, not about
   H-fetch** (§8bz). Expected ≈ 29% of fetches for `fstad`.
2. **Both arms archive to a dedicated `--archive` file**, never to
   `out/arena/games.jsonl` — HANDOFF's standing rule, and `_net_fp` puts the
   weights' fingerprint plus the flag in the archived identity (rule 19/20).
3. **Shipped config on both sides** apart from the flag: `chip`, `spread`,
   `src`, `wall` at their PolicyAgent defaults, exactly as `bc:base`.

## ⛔ Void / out of scope

* **Nothing here may be submitted**, whatever it reads. E20 owns the submission
  slots this window, a screen never ships (§8bh), and the five-anchor cell
  (rule 16) has not been run.
* **No threshold tuning after reading a result.** E11 pre-registered that a
  different threshold is *"a separate experiment, not a knob to tune after
  seeing the result"*, and that binds here: if `fstad` reads 0.52, the answer is
  a null, not a narrower stadium condition.
