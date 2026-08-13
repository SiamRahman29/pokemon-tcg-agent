# E32 — a coherent from-scratch policy, not more repairs. Pre-registered.

**Status: 🔴 CLOSED 2026-08-13 (day 32) by this document's own rung-1 rule.**
`plan:pure` **0.035 [0.021, 0.058]** against a ≥0.25 bar; `plan` hybrid
**0.068 [0.047, 0.096]** against ≥0.45. n=400 each, mirror, shipped config.
**Both controls passed** (`share` 0.783 / 0.780 against a ≥0.30 floor;
`fallbacks=0`), so the cell is valid and the null is about the policy.
⛔ **No tuning round, no rung 3, no second architecture.** Full write-up:
`report/EVIDENCE.md` §8ck.

⚡ **The mechanism is measured, not guessed** — the policy declines an available
attack at **168 of 284 offers, with its own attacker armed in 168 of 168**, and
assembles Grimmsnarl ex **0.84 times per game** while dying in 11.4 turns. The
proximate cause is a priority error in `_score_play` (Munkidori 20000 outranks
Impidimp 19000 outranks Rare Candy 18000), so the bench fills with the support
engine and the win condition is never built. **Recorded so the question is
answerable later, not so it is answered now**: re-tuning constants after seeing
the score is the shopping §8ao declined a β-sweep for.

---

**Originally: PRE-REGISTERED 2026-08-13 (day 32), BEFORE any arena game against `bc`.**
Frozen at the commit adding this file. The agent (`agents/sa/planagent.py`) and
its `plan:` spec (`scripts/arena.py`) were written first; the only games run
before this document existed are the two smoke runs against `random` recorded
under "instrument checks" below, which spend no comparison.

---

## The claim

Every hand-written thing this project has shipped is a **patch on the clone**.
`chip_target`, `energy_spread`, `counter_source`, the wall branch, the Petrel
rules — each answers *"can I override the net at THIS select?"*, each was sized
and measured alone. The discriminator that came out of that programme (rule 11:
rules deleting a **dominated** option win 3/3, rules picking a side in a
**tradeoff** lose 0/4) is therefore a statement about **patches**, and it has
never been tested against a **policy**.

**E32 builds the object that is absent from §2.5 entirely:** one plan per turn,
and every option scored by whether it advances that plan.

## Why this is not a re-run of rule 11

Three standing measurements, none of which was available when the rule
programme closed:

1. **§8ch (E26) — the licence.** A trained policy's deviations cost
   **f = 0.758 [0.703, 0.814]**; rate/depth/location-matched random ones cost
   **0.12**. ⇒ the local optimum forbids **jumps, not paths**. A from-scratch
   coherent policy is a path, and it is the only member of that family never
   built. ⚠ E26 cannot separate sequential coherence from per-decision
   plausibility; the composite is what is being leaned on.
2. **§8bj (F1) — the shape of the target.** The 1145–1166 agents' mirror edge
   dissolved into *timing*: Munkidori usage 6.42/game against our 6.23,
   identical behaviour, different ordering. The one ordering-free difference
   (Spikemuth: they stop searching at turn ~9.7, we never stop) classified as a
   **tradeoff**, so no patch could take it. **A patch cannot express "stop
   searching after turn 9." A plan can** — and `planagent._score_play` does.
3. **§2.8 / §0 — the ceiling is observed, not hoped for.** "The 1145–1166
   agents are not at a better point in our landscape, they are in a different
   one", and nothing at the top of this board is learned. `v10.py` — a real
   competitor's agent holding LB 950+ with its MCTS provably never executing —
   is one `AttackPlan` per turn plus an option ladder conditioned on it.

## 🔴 The honest case AGAINST, first

Named here so the result cannot be narrated past them:

1. **Rule 11 is 0-for-4 on tradeoffs and this policy is nothing but
   tradeoffs.** The bands in `_score_play` pick sides constantly. The defence
   is that a *coherent* set of side-picks is a different object from four
   independent ones — but that defence is exactly what is on trial, and if E32
   nulls, rule 11 generalises from patches to policies and that is the finding.
2. **The clone is a 2,810-game fit to the field mode; this is one afternoon of
   hand-tuning.** §8al measured strength falling monotonically with distance
   from the consensus 60; the same could be true of distance from consensus
   *play*, in which case the hybrid arm beats the pure arm and both lose.
3. **The plan layer's knowledge is hardcoded** because the card DB exposes no
   ability text (`abilities` is `None` for all 19 cards in our 60, verified
   2026-08-13). Punk Up, Adrena-Brain and Freezing Shroud are constants. Any of
   them wrong is a silent, uniform bias.

## Design

`plan[:label][,pure][,net=<path>]`, `agents/sa/planagent.py`.

**The plan object is a KO SCHEDULE, not an attack plan.** B2 killed the lethal
audit because Marnie's Grimmsnarl ex has **one payable attack** (316/316
lethals taken, all forced) — so v10's attacker×attack×target search is empty
for us. What is not empty is *which six prizes, in what order, and which 30s
make them cheap*: `STRATEGY.md` §7c.1 ("the deck wins by making the opponent's
board fragile") and §7d (**~84% of mirror takes are single-prize on both
sides** — a six-single-prize grind, not the canonical 2-2-2 map).

`build_plan()` computes, per MAIN: the attack's effective damage (0 into a
`WALL_POKEMON`), the per-turn chip budget (armed Munkidori × 30, capped by
counters actually on our own board), the KO schedule sorted by cost, the attack
target (bench only when Boss's Orders **changes the prize count**), and the two
30-placements. Every `_score_*` reads that object and nothing else.

**Two arms, and the difference between them is the experiment:**

| arm | declined selects go to | measures |
|---|---|---|
| `plan` | the clone (plain `PolicyAgent`, shipped config) | the hybrid that could ship |
| `plan:x,pure` | index order, deliberately | the PLAN, with no second policy hiding behind it |

## Controls — both must pass or the cell VOIDs

1. **Coverage is printed, per run, and it is the treatment size.** `[plan]
   share=` in `arena.py`'s summary. A plan policy that quietly deferred every
   select would score exactly like the clone and read as a clean null — the
   §8be failure one architecture up. **VOID if `share` < 0.30.**
2. **`fallbacks=0`.** Non-zero means the catch-all fired and the agent was
   playing index order under an exception; §8g had to detect exactly that
   indirectly once and it must never be inferred again.

⚠ **Rule 20:** the fallback net's bytes are in the archived identity
(`plan:...#4790c469` for `out/policy_v5_s2.npz`); a pure run archives `#pure`.
`agents/sa/policy_net.npz` is a **stale pre-v5 net** and must never be the
implicit fallback.

## The ladder, with bars written before the runs

Each rung gates the next. **Every cell in shipped config, mirror, `grimmsnarl`
both sides, paired seats.**

| rung | cell | bar | on failure |
|---|---|---|---|
| **0** | `plan:pure` vs `random`, n=20 | ≥0.85, `fallbacks=0` | the instrument is broken, not the idea |
| **1** | `plan:pure` vs `bc` (v5_s2), n=400 | **≥0.25** | below this the policy is not a policy; rebuild or close |
| **2** | `plan` (hybrid) vs `bc`, n=400 | **≥0.45** | the plan layer is net-harmful where it fires; close |
| **3** | whichever arm won rung 1/2, vs `bc`, **n≥2,000** | **point ≥0.53 AND CI excluding 0.500** | 🔴 **E32 dies with a number and becomes the report's chapter on why coherence is not enough** |

⚠ **Rung 3's bar is the ship bar, deliberately, and it is the same one F2's
seed harvest failed.** A weaker bar cannot be honoured: §8ak measured the
ladder's noise floor at **63.2 points**, so the arena is the only instrument,
and §8aw forces **≥3 seeds** on anything decided by a net's draw. The plan arm
has no seed — it is deterministic given the board — which is the one place this
experiment is *cheaper* to resolve than every net A/B before it.

⚠ **Rung 1's 0.25 is not a typo and not a moving target.** §8u's single-expert
clone — a *successful* imitation of a 1163-rated player — scored **0.370**.
A first-generation hand-built policy scoring 0.25–0.40 in the mirror is
consistent with a viable architecture that is undertuned; scoring 0.05 is not.
**The bar is what separates "undertuned" from "wrong object", and it is set
here so that a bad result cannot be re-described afterwards as a good one.**

## What each outcome buys

- **Rung 3 clears** → a submission candidate by 08-15 (last safe day), and the
  first non-clone agent this project has ever shipped.
- **Rung 1 or 2 clears but rung 3 does not** → the architecture works and is
  undertuned. That is the **Round-2 program**, where it matters far more than
  here: §3.5.6 measured that the expanded card pool damages *our* architecture
  most (134 distinct card ids in the corpus, every new card outside them, and
  "rules generalise to unseen cards and clones structurally cannot"). A plan
  policy reading facts off the card DB is posture (b) of that section, built.
- **Rung 1 fails** → rule 11 generalises from patches to policies, the
  "different landscape" reading of §2.8 is wrong or unreachable by hand, and
  E32 is a chapter. ⛔ No rung 2, no tuning round, no second architecture.

## Instrument checks already run (no comparison spent)

- `plan:smoke,pure` vs `random`, n=10 and n=20, `grimmsnarl` both sides:
  **0.900** and **0.950**, `fallbacks=0`, `share=0.738`, `plans=930`. Rung 0
  passes. Archived to scratch, not to `games.jsonl`.

### 🔬 Two defects found and fixed BEFORE the first comparison cell reported

Both were caught by re-reading the ladder while rungs 1–2 were in flight; the
runs were **killed and restarted** rather than reported, and their partial
archives deleted. Carrying the §8cj lesson forward: a control (or a band
ordering) found wrong *after* its cell reports is a retraction, found *before*
it is just engineering.

1. **`END` scored −1000 while declined options scored −1**, so the policy
   preferred playing a Boss's Orders it had explicitly declined to ending its
   turn. `v10.py` scores END at **0** and declines at −1 — **the ordering is the
   contract, not the magnitudes.** Fixed; "attack with 0 damage" moved to −50,
   also below END.
2. **`attacker_slot` was never assigned**, so the retreat band (`6000.0 if
   plan.attacker_slot > 0`) was dead code and a Munkidori stranded in the Active
   spot — a body that cannot pay for its only attack — would simply pass the
   turn with an armed Grimmsnarl on the bench. Now set in `build_plan`. ⚠ This
   is rule 11's **winning** class (a dominated position), not the tradeoff
   class, which is why it is in v1 at all.

⚠ **Neither fix was tuned against a score**, because no comparison score
existed yet. That ordering is deliberate: it is the only way a hand-built policy
can be debugged without the debugging becoming the fit.
