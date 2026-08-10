# E18 — does the clock win GAMES? The arena A/B of the rollout oracle

**Status: PRE-REGISTERED 2026-08-10 (day 29), before the first A/B game.**
Agent: `agents/sa/oracle.py` (`bc:<label>,orc`). Licensed by **E17** (§8by),
which measured the per-decision value; **E18 is the only thing that can turn
that into a win rate**, because per-decision win-probability gains do not add.

---

## 1. The cell

```
A  bc:orc,orc,net=out/policy_v5_s2.npz      the oracle, E17's measured design
B  bc:base,net=out/policy_v5_s2.npz         the shipped agent, byte-identical net
   --deck-a grimmsnarl --deck-b grimmsnarl
```

**Same weights on both sides.** The only difference is whether the oracle runs,
so the seed nuisance (§8bg, ±25 Elo) cancels — this is the byte-identical
rule-toggled A/B the repo's discipline prefers, and it is available here because
the oracle is a wrapper, not a retrain.

⛔ **`--deck-a/--deck-b grimmsnarl` is NOT optional.** Arena defaults to
`sample`, and on `sample` the option-count distribution is unrecognisable —
**79% of decisions carry ≥12 options against 19.7% on our real deck**, so the
oracle's free trigger (`option count ≤ 5`) fired on **0.7%** of decisions
instead of 24%. A `sample` run would have measured a component that barely
fires and reported a null. This is §8ax / rule 20 one seat over, caught by the
option-count histogram in the health line.

## 2. What is already verified about the agent (not to be re-litigated)

| check | reading |
|---|---|
| runs in live games without falling back | `fallbacks=0 net_missing=0` over 22 games |
| the live fork does not corrupt the game | `planner.py`/`sequencer.py` already do it; 0 harness errors |
| it releases the ROOT search, never `fs.end()` | `fs.end()` frees ALL search memory mid-game |
| it actually fires | **5.25 fires/game, 3.1 overrules/game** (E17 predicted ~4.7) |
| the option-count trigger matches E17's population | n∈[3,5] = 24% live vs 36% in replays |
| the 600 s pool is real in the arena | `harness.py` mirrors and decrements it; worst game used **90.7 s of 600** |

## 3. 🔴 The pre-registered decision rule

**Primary: A's score against B, n = 400 games (200 matches, seat-balanced).**

| branch | condition | consequence |
|---|---|---|
| 🔴 **KILL** | score ≤ **0.470** (CI excludes 0.50 downward) | the clock LOSES games. Closed, written up, and E17's per-decision gain is a measured lesson about why per-decision value does not compose |
| 🟡 **INCONCLUSIVE** | CI contains 0.50 — **the expected outcome** | see §4. n=400 carries SE ≈ 0.025 and **cannot resolve the effect E17 predicts** |
| ✅ **PROMISING** | score ≥ **0.530** and CI excludes 0.50 | extend to n≈1,200 for the ship decision |

⚠ **The honest statement of power, written down first.** E17's per-decision
gain is +0.0120 at ~4.7 firings/game. There is **no model** in this repo that
converts that to a win rate, but nothing suggests it is large. At n=400,
SE = 0.025, so:

- a true +0.03 (0.530) is detected with probability ≈ **0.34**
- a true +0.05 (0.550) is detected with probability ≈ **0.71**

⇒ **A null at n=400 is uninformative and must not be reported as a kill**, and
a *win* at n=400 is the F2 trap: `s7` screened at 0.528 and read **0.487** on
2,800 fresh games (§8bh). **The screen may kill; it may not ship.**

## 4. 🔴 The user's decision, recorded as a deviation

The user directed: *"Build the agent and do the n=400 run. If it produces good
result, we submit it."* That is a **screen that ships**, which §8bh's
pre-registration forbids on the grounds that the screen's error equals the
effect's size. The concern was raised once and the direction stands, so:

- **n=400 runs first**, exactly as asked.
- If it reads ≥0.530, the cost of the confirmation (n≈1,200, ~10 h) is put to
  the user **with the interval in hand**, and the ship decision is theirs.
- **Whatever ships, this file records that the screen was not a confirmation.**

## 5. Controls

- **`skip_trigger` / `nopt` histogram** — a component that never fires reads as
  a null (E15's lesson: `sym8` was only interpretable because it could be shown
  firing on 8.36% of selects). The health line carries both.
- **`errors` and `first_error`** — a swallowed exception is a silent null. ⚠ An
  intermittent rollout error rate of **6.9%** was seen in one 8-game run and
  **0%** in others; it is under diagnosis and its message is now captured. If
  the A/B runs with a nonzero error rate, that rate is reported with the result.
- **`pool0`** — every archived row carries the remaining 600 s pool. **If any
  game drops below the 45 s reserve the run is void**, because `timeout = loss`
  is a failure mode this project has never met and must not meet on the ladder.
- **Seat balance** — arena plays both seats; report as P0 and as P1 separately.

## 6. ⚠ What E18 cannot settle even if it wins

**The mirror flatters this agent by construction.** The oracle scores options by
rolling out with the clone piloting both seats; in a mirror A/B the opponent
*is* the clone, so its opponent model is exactly correct. On the real ladder it
is not — 33.3% of our games are the mirror at rating 955, rising with rating
(§8ac), so the flattery is partial rather than total, but it is real and it
points one way. **A mirror win is a necessary, not a sufficient, condition for
shipping.**
