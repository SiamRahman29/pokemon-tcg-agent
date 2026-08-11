# E22 — pessimism + coverage, against a MEASURED mechanism. Pre-registered.

**Status: PRE-REGISTERED 2026-08-11 (day 30); renumbered E21→E22 on day 30 because `E21-petrel-fetch.md` already owned that number (§8cc). Content unchanged from the frozen version at `527e26a`, before the V-ensemble finished
training and before the first arena game.** Frozen at the commit adding this
file.

---

## What E20 established, and why this is not "tuning until it passes"

E20 shipped a value net and a one-ply argmax over it. It lost **0/10** in the
smoke with every health counter green, and `p87` measured why over 1,106 live
decisions:

| reading | value |
|---|---|
| live AUC, V at the decision vs eventual result | **0.7042** (training 0.8270) |
| V spread ACROSS positions (sd) | 0.3174 |
| V spread WITHIN a position (sd over siblings) | 0.0682 |
| 🔴 **V's argmax == the clone's pick** | **6.1%** vs a chance rate of **20.3%** |

**Agreement three times BELOW chance is not noise** — noise lands at 20%. It is
systematic anti-selection, and it has a standard name: **one-sided
extrapolation error under a max.** V is fitted to states the clone visits; the
successors of options the clone would not take are off-distribution, V is
uncalibrated there, and `max` over one on-distribution state plus several
uncalibrated ones selects an uncalibrated outlier nearly every time. The
clone's pick *is* the on-distribution option, so argmax avoids it.

⚠ **Two things this does NOT license.** It does not license calling V broken —
AUC 0.70 through the live path says the plumbing is sound and the evaluator
ranks games. And it does not license a sweep. E22 changes **two** things, both
of which are the textbook response to this exact mechanism and both of which
were named *before* any variant was run.

⚠ **E20's own design error, recorded.** E20 froze "no trigger, all options"
specifically to avoid E17's post-hoc arm selection. The clock's `arms=3` was
doing real work — it kept successors near-distribution — and removing it
dodged one bias into another. E22 restores it as a *coverage* constraint with a
stated reason, not as a tuned parameter.

---

## The two changes

**1. Pessimism (LCB).** Five value nets, identical recipe, differing only in
init and batch order (`--seed`), sharing one game split (`--split-seed 0`) so
their spread is epistemic and not a difference in held-out games. Score a
successor as

```
score(s') = mean_i V_i(s')  -  K * sd_i V_i(s')
```

An off-distribution successor is exactly where independently-initialised
members disagree, so the penalty is largest precisely where E20 failed. **K is
frozen at 1.0 by this document** — one standard deviation, the conventional
LCB. ⛔ K is not swept. A K sweep is the "post-hoc among four" failure E17 paid
for and E19 priced.

**2. Coverage.** Candidates are the policy net's **top-3** options (the clock's
`arms`), plus the agent's own pick as arm 0. Successors of options the clone
ranks last are the ones V has never seen; excluding them is the constraint,
not an optimisation.

Everything else is E20's configuration unchanged: W=4 worlds, one `fs.step`,
no value trigger, `grimmsnarl` both sides, byte-identical policy net both
sides so the seed nuisance cancels.

---

## Bars, written before the ensemble exists

**Primary cell — mirror, n = 2,000**, `bc:e22,vlp,vlcb1.0,varm3,vnet=<5 nets>`
vs `bc:base`. SE ≈ 0.0112.

| branch | condition | reading |
|---|---|---|
| ✅ **screen passes** | point ≥ **0.530**, CI excludes 0.500 | go to fresh-games confirmation, then weighted anchors. ⛔ a screen never ships (§8bh) |
| 🟡 **alive but short** | CI excludes 0.500, point in [0.505, 0.530) | the mechanism is real and the remedy is partial ⇒ **the indicated next step is DATA (E23 value iteration), not a larger K** |
| 🔴 **KILL** | CI contains 0.500 | pessimism + coverage does not rescue it. **H-eval is refuted in every cheap form**, and the axis closes with a report chapter |
| ⚠ **harmful** | point ≤ 0.470, CI excludes 0.500 | audit before interpreting |

**The diagnostic that decides what a null MEANS**, recorded but not a gate:
`p87`'s agreement statistic re-run under E22. If argmax-agreement moves from
**6.1%** up toward or past the **20.3%** chance rate, the mechanism was
correctly identified even if the win rate does not follow — and that sends the
work to data coverage rather than to more inference-time patching. If agreement
stays below chance, the diagnosis itself is wrong and gets re-opened.

## Controls

Unchanged and still binding: identical-arms C0 reads 0.500 (commissioned on
Kaggle at 0.5082 [0.4897, 0.5267]); the component must fire, with a printed
count, or a null is a statement about wiring; `--deck-a/--deck-b grimmsnarl`;
the 45 s reserve as a void condition; every value net's bytes in the archived
identity (rule 20). ⚠ And E19's standing constraint: **no internal gate
licenses anything — only the end-to-end A/B does.**

## Dependency

**E23 (value iteration — generate outcome data AT the successors the search
selects, retrain, repeat) is conditional on this cell.** It is the family
ROADMAP §2.7 names as *"declined for cost, never tested"*, and E20's result is
its motivation: a measured demonstration that on-policy value data does not
cover the action space a search explores. ⛔ It is not started until E21 reads.
