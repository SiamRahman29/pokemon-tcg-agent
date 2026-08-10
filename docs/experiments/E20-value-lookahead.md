# E20 — a learned value function + ONE-PLY engine lookahead. Pre-registered.

**Status: PRE-REGISTERED 2026-08-11 (day 30), before V is trained and before
the first arena game of any cell.** User-directed ("let's go", 30 h GPU, multiple
submissions authorised). Frozen at the commit that adds this file.

---

## The hypothesis, and why it is not a retry of anything dead

Every search this project has built died, and the docs file three separate
deaths. **They share a cause: none of them ever had a state evaluator worth
searching with.**

| axis | evaluator it used | result |
|---|---|---|
| §2 determinized search | terminal 0/1 rollout | 0.323, n=31, rollout SE ≈ 0.14 |
| B4 / E5 within-turn sequencing | handcrafted `evalfn` | −89 Elo (§8v) |
| the clock, E17–E19 | determinized rollout to terminal | biased; +0.035/overrule bought **0.4963** (§8ca) |

E19's own post-mortem names the survivor: *"a learned value function trained on
real outcomes rather than determinized simulations is untouched by this
result."* **E20 is that component.**

**H-eval:** the evaluator was the broken part. A V(s) fitted to *realized game
outcomes*, consumed by a **one-ply expansion in the real engine**, beats the
clone.

Why each prior death does not transfer:

* **Not `evalfn`** — V is learned from outcomes, not hand-written.
* **Not terminal rollouts** — the estimate is one engine step plus a function
  call. There is no 0/1 terminal draw, so §2's SE ≈ 0.14 variance term, which
  §2 says *"any future search must attack first"*, does not exist here.
* **Not §1's value net** — that was fitted on **human replays**, before the v3
  option binding (+115 Elo), the v4 state block (+37) and v5, and it was judged
  by **validation loss**, which rule 3 says predicts nothing. E20 fits the v5
  representation to **self-play outcomes** and is judged only in the arena.
* **Not the clock** — the clock's bias is most likely determinization (strategy
  fusion) compounding over ~100 simulated steps. One ply cannot compound.

⚠ **H-eval is falsifiable and its refutation is a real finding.** If V + 1-ply
reads ≈ 0.500, then "the evaluator was the broken part" is wrong, and the
synthesis that unifies three deaths in this repo fails with it.

### 🔴 The adverse prior, recorded before the build rather than discovered after

**§8az (E1) already trained an outcome head on this encoder and it OVERFIT
AFTER ITS FIRST EPOCH.** That section's closing sentence is the one that binds:
*"E4 does not inherit a validated value representation."*

E20 must not pretend that finding away. What differs, precisely:

| | E1 (§8az) | E20 |
|---|---|---|
| data | human BC corpus, 248,985 rows | **self-play `won`**, ~720k rows / 20k games |
| role | auxiliary gradient, weight 0.1, shared encoder | **the decision-maker** |
| what was measured | the *policy's* strength | V's own ranking, end-to-end |
| label distribution | outcomes of *human* games | outcomes of games **our policy actually played** — the on-policy value the search needs |

⇒ E1 is a null about *auxiliary multi-task training*, not about whether a value
function can evaluate a position. But its overfitting warning transfers
directly: **V gets early stopping on a `gid`-disjoint split, and the export rule
is pinned in advance** (rule 18's `--export-last` corollary — an arm that picks
its own checkpoint is measuring training length).

⚠ If V cannot separate won from lost states on held-out self-play at all, E20
stops at the orientation check and reports §8az's finding as replicated on
self-play data. That is a cheap, honest outcome and it is written here first.

---

## Cost — this is the reason E20 is a Round-1 candidate at all

§2.7's **LARGE-OR-NOTHING** blocker is about agents that spend wall-clock per
move. From §2.7's own measured table (`fs.step` 0.197 ms, `net.choose` 0.808 ms):

| agent | per decision | per game | n=2,000 A/B |
|---|---|---|---|
| clone (shipped) | ~1 ms | 0.17 s | ~5 min |
| **E20, W=4 worlds** | **~70 ms** | **~3.4 s** | **~1.9 core-hours** |
| the clock (E18/E19) | ~3,000 ms | 153 s | 85 core-hours |

⇒ **E20 validates like a clone, not like the clock**, and spends ~3 s of the
600 s pool, so `timeout = loss` is structurally out of reach.

---

## The build

**V(s)** — a scalar sigmoid head on the **v5 state representation only**
(`features.featurize` → `dense`(242) + `slots` + bags). Deliberately *not*
select-conditional: at inference V scores a **successor observation** returned
by `fs.step`, so its input must be a pure function of state.

* Data: B8's existing self-play shards (`artifacts/rl_v5_t05*`, ~720k rows /
  20k games) carry a **`won`** column — the one field the BC corpus has always
  had and nothing has ever trained on — plus newly generated games.
* Label: `won` from the **acting seat's** view; `seat` is in the shards and the
  orientation gets a printed check, not an assumption.
* Split **by `gid`** (rule 17 — a row-wise split leaks a game across both sides).

**The agent, `bc:<label>,vlp`** — reuses `oracle.py`'s fork verbatim and swaps
only the evaluator: `fs.begin(...)` → `fs.step(root, candidate)` → featurize the
successor → V → argmax, averaged over **W = 4** sampled worlds.

🔴 **W is FROZEN AT 4 BY THIS DOCUMENT.** It is not sized first and then chosen.
E17's trigger was *"post-hoc among four"* and E19 spent its compute discovering
what that was worth. If the measured cost exceeds budget, that is a **void
condition, not a tuning opportunity.**

🔴 **NO TRIGGER.** E20 fires at every legal qualifying MAIN decision (47.1/game,
§8by) because the cost permits it. Every trigger this project has deployed was
selected post-hoc from a sweep.

---

## The bars, written before any result exists

**Primary cell — mirror, byte-identical policy net on both sides, rules
identical on both sides, `--deck-a grimmsnarl --deck-b grimmsnarl`, n = 2,000.**
The seed nuisance (§8bg/§8aw) cancels by construction in a flag-toggled A/B, so
this needs no seed budget.

| branch | condition | reading |
|---|---|---|
| ✅ **screen passes** | point ≥ **0.530** AND CI excludes 0.500 | go to confirmation. ⛔ **A screen never ships** (§8bh: `s7` screened 0.528 and read 0.487 on 2,800 fresh games) |
| 🔴 **KILL** | CI contains 0.500 | H-eval is refuted at this n. V + 1-ply is dead as a Round-1 axis and becomes a report chapter |
| ⚠ **harmful** | point ≤ 0.470, CI excludes 0.500 | the lookahead is actively worse than the clone ⇒ audit V's orientation before anything else |

At n=2,000, SE ≈ 0.0112, so **0.530 is a 2.7σ bar.**

**Confirmation cell** — fresh games, n = 2,000, same configuration. Ships only
if the CI excludes 0.500.

**Anchor cell, required before any submission** — the weighted five-anchor set
(rule 16). A mirror-only verdict has never been allowed to decide a ship here,
and §8i is the reason.

---

## Controls, and the one E19 bought

1. **C0 — identical arms must read 0.500.** Already run as the Kaggle
   instrument's commissioning test.
2. **The component must fire.** A per-game count of expansions and of overrules
   (V's argmax ≠ the net's pick) is printed; **a null with a zero firing count
   is a statement about the wiring, not about H-eval** (§8bz).
3. **Rule 3.** V's held-out AUC/calibration is a *wiring check only*.
4. 🔴 **E19's lesson, written in as a standing constraint: no internal gate
   licenses anything.** Every internal control in E18 passed — C0 99.8%,
   C1 100%, identical arms at zero, selection verified at z=5.5 — and the
   instrument still measured the wrong quantity. **Internal validity is not
   external validity.** The only reading that counts is the end-to-end A/B.
   ⛔ Specifically: E20 must not publish a per-decision value figure as a
   headline the way E17 did.
5. **Void conditions:** worst remaining 600 s pool < 45 s reserve; evaluation
   error rate > 1%; or V's orientation check failing (V must score won states
   above lost ones on held-out self-play, a sign test, not a threshold).

## Diagnostics recorded but NOT gates

Overrule rate per game; the pre-search net's own score of what V took (E18's
autopsy shape); the win-probability band each overrule sits in. These make a
null interpretable. **None of them can license a build.**
