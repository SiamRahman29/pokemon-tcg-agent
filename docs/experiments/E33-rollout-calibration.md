# E33 — is the rollout estimator CALIBRATED? Isolating the cause E19 named and never tested. Pre-registered.

---

# 🔴 RETRACTED 2026-08-13, SAME DAY, BY THE CELL THAT SUCCEEDED IT. DO NOT CITE THE VERDICT BELOW.

**The defect.** `p94`'s observer took its rollout continuation policy from
`pnet.get()`. That singleton returns the repo default
`agents/sa/policy_net.npz` **#ce97c732 — the v2 clone**, while both seats of the
real game were played by `out/policy_v5_s2.npz` **#4790c469**. They are
different networks, verified by fingerprint.

⇒ **This cell's central claim is FALSE as run.** It says:

> *"Rollout and reality differ in exactly ONE thing: the world. Same net, same
> position, same continuation policy on both seats."*

They also differed in the continuation policy — by three generations of clone.
**The measured −0.0083 is therefore the sum of determinization bias and a
policy mismatch, and two errors that could cancel cannot be reported as one
error that is absent.**

🔴 **Consequences, stated without softening:**
- **§8ca's diagnosis is NOT retracted after all.** The retraction rested on this
  number. It goes back to open.
- **The estimator audited here is not the estimator that ships.** `oracle.py` is
  constructed as `RolloutOracle(..., net=self.net)` from `bcagent`, so the live
  clock uses the correct net. This cell audited a rollout nobody runs.
- The **dispersion ratio 1.1445** inherits the same defect and is withdrawn with
  it.

⚠ **The warning was already written down and I did not read it.** `p82`
(E17's collection script) carries a comment headed *"🔴 PIN THE NET IN THE
SCRIPT, not in the environment"*, explaining that the default is *"the v2 clone
— three generations behind the agent that played `submission_v5_s2`"*, and
recording that E17's C0 control separated the two at **67.3% vs 99.8%**. E33
had no equivalent control, which is exactly why nothing caught it here.

⚠ **The tell was visible in E34's pilot rows and I did not recognise it for an
hour**: 23% of sampled decisions had the agent's own pick ranked below the
net's argmax, which is impossible with rules off (`oracle.py`: verified
106/106). With the net pinned it is **0%**.

**Status: ⏹ STOPPED MID-RUN 2026-08-13 BY USER DECISION. NO VERDICT. §8ca stays
OPEN and undiagnosed.**

The corrected run reached **8,117 decisions** of a planned ~15,000 before being
stopped to move resources onto the from-scratch policy track. ⛔ **The partial
rows are archived and deliberately NOT analysed.** The pre-registration sets
`n` by a half-width bar of ≤0.02; a partial read cannot clear it, so computing
one now could only produce an UNRESOLVED number that would sit in the record
looking like a result. ⚠ The stop was **not** outcome-driven — nobody had seen
a statistic — but the rows stay unread regardless, because "we stopped and then
looked" is indistinguishable from optional stopping to anyone reading this
later.

⇒ **If this axis is ever reopened, the data is in `out/logs/e33/e33fix_*.jsonl`
and the cell resumes rather than restarts.** The instrument is now correct
(net pinned, identity asserted, rows streamed per game), which is the durable
part of this round.

---

**Original (SUPERSEDED) status: ✅ CONCLUDED 2026-08-13 (day 32) on the CALIBRATED branch.**
**BIAS = −0.0083 [−0.0278, +0.0111]** over **1,484 games / 15,271 decisions**
(nine fully-verified shards), half-width **0.0195** against the pre-registered
≤0.020. All ten shards read −0.0100 [−0.0284, +0.0083]. Controls: `none = 0.0%`
of 90,000+ rollouts, `returned_inner == calls`, AUC **0.7657**.

🔴 **⇒ §8ca's stated cause is RETRACTED.** *"The rollout estimate is biased,
most likely by determinization"* is not supported: the estimator is
level-calibrated, and the predicted **positive** sign did not appear either.
⚠ **E19's 0.4963 stands** — what is retracted is the explanation, not the
measurement, and the clock does not reopen as built.

⚠ **Post-hoc:** the estimator IS over-dispersed — ratio **1.1445 [1.1139,
1.1759]**, ~14% in variance ≈ 7% in sd — which is an order of magnitude too
small to turn a predicted +0.035 into 0.

🔬 **Successor named: E34, the randomized-overrule design.** This cell rolls out
**one** action (the clone's own) and therefore tests the *level*, not the *gap
between arms* that E17 and E19 actually spend.

Full write-up: `report/EVIDENCE.md` §8cm.

---

**Originally: PRE-REGISTERED 2026-08-13 (day 32), BEFORE the first measured game.** Frozen
at the commit adding this file. User-directed.

---

## The question, and why it is the gate on the whole search family

§8ca closed the clock on a diagnosis it did not test:

> at **one overrule per game** — where the one-step assumption every rollout
> rests on is *exactly* satisfied — it reads **0.4963 [0.4719, 0.5207]** at
> n=1,608, against a pre-registered point prediction of **0.535**. A +0.035 gain
> in win probability at one decision *is* +0.035 on that game's win rate **if the
> estimate is unbiased**. **It is not there ⇒ the rollout estimate is biased**,
> most likely by determinization (strategy fusion), **which is named but not
> isolated.**

Everything downstream is denominated in that estimator: E17's **+0.0139**/
decision, §8bx's dispersion, §8bw's **+0.120** scale bar, and §2.7's entire
sizing framework. **If the currency is bad, none of it buys games, and no amount
of Round-2 hardware repairs it.** If the currency is good, then §8ca's own
diagnosis is wrong and the search family's failure is somewhere else entirely.
Either way this is one measurement that moves a family, which is why it is
worth running before any build.

⚡ **And the mirror makes the isolation clean.** `oracle.py`'s docstring states
the model mismatch precisely: *"The value is win probability under
**clone-vs-clone continuation**. In a mirror A/B the opponent **is** the clone,
so the rollout model is exactly right there."* E19 cell A ran in the mirror.
⇒ **continuation-policy mismatch is already excluded for that cell**, and the
only remaining suspect is the one §8ca named: **determinization** — the sampled
world distribution, not the policy that plays it out.

## The design

A **pure observer**, wrapped around the shipped clone. At sampled MAIN
decisions it takes the clone's own pick, rolls that pick out to terminal `R`
times over freshly determinized worlds, records the mean as `p̂`, **and returns
the clone's pick unchanged.** The game is then played to its real conclusion by
the same clone on both seats, and the realized outcome `y ∈ {0, 0.5, 1}` is
recorded from the deciding seat's view.

⇒ **The rollout and the reality differ in exactly one thing: the world.** The
policy is identical (same net, both seats, both places), the position is
identical, the continuation is identical in distribution. **Any gap between
`p̂` and `y` is determinization bias and nothing else.** That is the isolation
E19 could not perform from a win rate.

⛔ **The observer must not change play.** It calls the clone, measures, and
returns the clone's answer. `plan`/`orc`-style overruling is absent by
construction, so these are ordinary `bc`-vs-`bc` games.

## Statistics, chosen for what they are robust to

**Primary: `bias = mean(p̂) − mean(y)`, with SE clustered by GAME.**

⚠ **Clustering is not optional and the repo has already paid for learning it.**
§8bw: *"three runs of one cell read +0.130/+0.107/+0.120 against a nominal
±0.017, because pairs are clustered inside positions; clustering widens it 4.1×
and all three then agree."* Decisions inside one game share an outcome, so a
naive SE here would be wrong in the same way.

⚠ **The primary is deliberately the MEAN and not the calibration slope.** `p̂`
is a finite-`R` estimate, so binning or regressing on it suffers **regression
dilution** — a noisy predictor flattens the fitted slope even when the
estimator is perfectly calibrated. The mean bias is immune: `E[p̂] = p` and
`E[y] = p` under calibration, whatever `R` is. **The slope is reported as
secondary with the attenuation stated, never as the headline.**

**Secondary:** the calibration curve by `p̂` decile; and the primary restricted
to `p̂ ∈ [0.15, 0.85]`, which is where E17 measured all of the value (57% of
decisions sit above 0.85 and are worth +0.0015).

## Controls — all three must pass or the cell VOIDs

1. **Positive control: `p̂` must resolve something.** AUC of `p̂` against `y`
   must clearly exceed 0.5. If the estimator carries no signal at all, this cell
   is about a broken fork, not about bias. (§8bw already showed the fork can
   fail silently — it *"silently accepts a decklist the seat is not playing"*.)
2. **Rollout health.** The `None` rate (fork or step failure) is reported. ⚠ It
   must not differ across the `p̂` range in a way that could manufacture the
   result. **VOID above 10%.**
3. **Play is unperturbed.** The observer returns the clone's pick at 100% of
   decisions; measured and asserted, not assumed.

## Pre-registered readings

| branch | condition | what it means |
|---|---|---|
| 🔴 **BIASED** | CI on `bias` excludes 0 | §8ca's diagnosis is **confirmed and isolated to determinization**. The rollout currency does not buy games ⇒ E17/§8bx/§8bw/§2.7's numbers are all denominated in it and must carry the caveat. **The fix is an information-set-aware evaluator or a V trained on real outcomes — NOT more rollouts and NOT more hardware.** |
| ✅ **CALIBRATED** | CI contains 0 **and** half-width ≤ 0.02 | 🔴 **§8ca's diagnosis is WRONG and gets retracted.** The estimator is fine and E19's null has another cause — selection noise at the margin, or per-decision gains genuinely not adding across a game. The search family stays open but its problem is re-specified. |
| ⚠ **UNRESOLVED** | CI contains 0, half-width > 0.02 | underpowered; report the width, do not narrate a null. Extend n or close on cost. |

**The sign matters and is predicted in advance:** if determinization is the
culprit, `p̂` should be **too optimistic** (`bias > 0`) — sampled worlds are
easier for the roller than the true world, because a mis-specified hidden-card
distribution removes surprises the real game delivers. **A significant negative
bias would falsify the mechanism even while confirming miscalibration**, and
that distinction is written here so it cannot be smoothed afterwards.

## Sizing

Run as a pilot first (rule 14): the pilot fixes `R`, the sampling rate, the
per-game cost and the `None` rate, and only then is `n` set. Recorded in the
outcomes section below **before** the full run.

Cost model going in: a rollout is ~101 ms (§8bw); at `R` rollouts × `k`
sampled decisions per game, the measurement adds `0.1·R·k` s to a ~0.2 s game.
`scripts/shard.py` (§8cl) makes the full run ~3.1× cheaper in wall clock.

    python -X utf8 scripts/p94_rollout_calibration.py --games 40 --rollouts 6 --every 4

### ▶ PILOT RAN 2026-08-13 (30 games, R=6, every 4th eligible decision)

**All three controls pass.** `none = 0.0%` of 1,860 rollouts (control 2);
`returned_inner = 3203/3203` (control 3 — play provably unperturbed);
**AUC(p̂, y) = 0.7648** (control 1 — the estimator clearly resolves).

| statistic | pilot |
|---|---|
| decisions / games | 310 over 30 |
| mean `p̂` | 0.5360 |
| mean `y` | 0.5613 |
| **BIAS** | **−0.0253 [−0.1576, +0.1071]**, clustered SE 0.0675 |
| naive SE | 0.0261 — **clustering widens it 2.58×** |
| cost | 12.78 s/game |

⇒ **UNRESOLVED at this n, exactly as the third branch describes**, and the
pilot's job was never to answer the question. ⚡ **§8bw's clustering lesson
replicated at 2.58×** on an independent estimator — a naive SE here would have
read this as a confident null.

**Sizing, fixed here BEFORE the full run.** Clustered SE ≈ 0.37/√G. For the
≤0.02 half-width the CALIBRATED branch requires, SE ≤ 0.0102 ⇒ **G ≈ 1,320
games**. At 12.78 s/game that is ~4.7 h serial, run as **8 shards ≈ 1 h wall**.
`R` stays at **6** — it does not enter the primary's expectation at any value,
so it is not a knob worth turning after seeing a number.

🔬 **One pooling defect found and fixed before the full run:** every shard
counts its games from 0, so pooling by `--analyze` would have merged shard A's
game 0 with shard B's game 0, collapsing independent clusters and reporting an
SE that is too **tight**. Game keys are now namespaced by seed. ⚠ Same family as
§8bw, in the direction that flatters us.

⛔ **The pilot's calibration curve is NOT reported as a finding.** It shows a
strong S-shape (`p̂` 0.167 → `y` 0.467; `p̂` 0.830 → `y` 0.636) which looks like
overconfidence and **is exactly what finite-`R` attenuation produces under
perfect calibration** — binning on a noisy `p̂` selects the noise. This is why
the primary was pre-registered as the mean. The curve is diagnostic only, at any n.
