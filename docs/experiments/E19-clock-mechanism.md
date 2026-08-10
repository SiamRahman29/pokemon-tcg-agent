# E19 — why didn't the clock's per-decision value become wins? Two cells, pre-registered

**Status: PRE-REGISTERED 2026-08-10 (day 29), before the first game of either
cell.** User-directed, both cells.

E18 (§8bz) established the puzzle rather than a verdict: the search **works**
(picks the genuinely best arm 40/60 = 67%, z=5.5; 68% of its overrules are real
improvements worth +0.035 each) and the win rate read **0.4764 [0.4281,
0.5252]**. E19 tests the two live explanations, one per cell.

---

## The two hypotheses

| | claim | what it predicts |
|---|---|---|
| **H-compound** | The rollout value is real but **only for a ONE-STEP deviation**. It is Q^π(s,a) with the *clone* continuing; E18's agent deviated **3.32 times a game**, so it played a different policy than the one it evaluated. "The search starts a plan the net does not follow." | capping at **one** overrule per game delivers the measured **+0.035** intact |
| **H-fusion** | The rollout value is **biased and never transferred**. We average over 20 determinized worlds and inside each the simulation acts as if the hidden cards were known; a line that only works because the *clone* plays it out scores well and gains nothing. Supporting sign: **37% of overrules take an option the net scores >3 worse.** | one overrule per game reads **≈ 0** |

**These are cleanly separated by cell A**, which is why it is worth its compute:
at cap=1 the one-step assumption is *exactly* satisfied, so H-compound's
prediction is a point value, not a direction.

---

## Cell A — one overrule per game (`bc:cap,orc,od1`)

Everything else is E18's configuration. `od1` caps overrules at **one per
game**, reset on the deck-registration select — ⚠ `arena.py` builds the agent
**once** and plays every match through it, so a cap that failed to reset would
fire once per *shard* and produce a null by construction.

* **n = 1,600**, mirror, byte-identical net both sides.
* Cost measured at **10 s/game** (the cap stops all searching once used).
* Power: SE ≈ 0.0125 ⇒ H-compound's **+0.035 is a 2.8σ effect**. This cell is
  adequately powered for the thing it is testing, which E18 was not.

| branch | condition | reading |
|---|---|---|
| ✅ **H-compound supported** | score ≥ 0.525, CI excludes 0.500 | the per-decision value is REAL and multi-deviation play destroys it ⇒ the fix is consistency (policy iteration), and §2.7's R2 candidate is the right home |
| 🔴 **H-fusion supported** | CI contains 0.500 | the one-step value never transferred. ⛔ **No firing policy can rescue it** — this kills the clock as a Round-1 axis outright, cell B included |
| ⚠ **neither** | score ≤ 0.475, CI excludes 0.500 | one deviation *costs* — an instrument finding, and the rollout's orientation gets re-audited before anything else |

## Cell B — fire only when we are LOSING (`bc:wp,orc,om99,ow0.5,os12`)

The user's trigger proposal, **half of it**. Measured offline against E17's
stored values, control-corrected, with win probability taken from a **disjoint**
half of the replicates:

| trigger | gain/decision | fires | value/game |
|---|---|---|---|
| fire everywhere | +0.0150 [+0.0024, +0.0277] | 100% | 0.708 |
| **`wp < 0.50` alone** | **+0.0150 [+0.0040, +0.0259]** | **22%** | **0.705** |
| `margin < 1.5` alone | +0.0132 [+0.0014, +0.0250] | 67% | 0.622 |
| margin<1.5 **AND** wp<0.50 | +0.0108 [+0.0004, +0.0212] | 16% | 0.510 |
| option count ≤ 5 (E18's gate) | +0.0118 [+0.0011, +0.0225] | 36% | 0.557 |

🔴 **"We are losing" is validated; "the net is confused" is REFUTED.** Adding the
margin condition *lowers* the gain (+0.0150 → +0.0108). The autopsy says why:
the wins come from options the net scored **>3 worse** — where it was confident
and **wrong** — not where it was torn. A small margin means the options really
are close, so there is little to win. ⇒ cell B runs the win-probability half
alone.

* **n = 600**, `os12` (E17: Q4 is +0.0143 at R_sel=10 vs +0.0163 at 30 — nearly
  flat, so this costs ~12% of the per-decision gain and saves a third of the
  compute).
* ⚠ **The realized fire rate is DOUBLE the idealized one** — 20.5/game against
  the offline 10.5 — because the 10-rollout probe has SE ≈ 0.16, so positions
  with true wp > 0.5 pass the noisy gate. The offline table used 25 replicates.
  **Cell B therefore tests a noisier trigger than the table describes.**
* Power: SE ≈ 0.0204. ⛔ **Underpowered by design** — it detects +0.03 at 1.5σ.
  **A null in cell B is uninformative and will not be reported as a kill.**

| branch | condition | reading |
|---|---|---|
| ✅ | score ≥ 0.540, CI excludes 0.500 | better targeting alone recovers the value ⇒ take it to a confirmation |
| 🟡 | anything else | uninformative at this n; **read cell A instead** |

## Ordering and dependency

**Cell A gates cell B's interpretation.** If A supports H-fusion, the
per-decision value never transferred and no trigger can help — B's result is
then a curiosity, not a lead. Both are launched together only because the
machine has the cores; **A is the one that decides anything.**

## Controls (both cells)

Unchanged from E18 and all still binding: byte-identical nets, `--deck-a/--deck-b
grimmsnarl` (⛔ never `sample` — the trigger collapses to 0.7% firing there),
the 45 s pool reserve as a void condition, the firing counters (a component that
never fires reads as a null about the wiring), and `errors`/`first_error`.
⚠ Cell A adds one: **`skip_capped` must be non-zero and `overruled` must not
exceed the game count**, or the cap is not doing what its name says.
