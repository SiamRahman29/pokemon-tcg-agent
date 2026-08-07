# E9 — seed ensembling: do independently-trained nets vote better than one?

**Pre-registered 2026-08-07 (day 24), before any arena cell ran.** Track A: this
is aimed at a stronger agent, not at a report chapter.

## Why this is not a closed axis

The closed axis is **capacity** (§8w): one net made 2.6× and 8.2× larger bought
**two** decisions out of 12,939 and then lost 43. That says the *features*, not
the parameter count, are binding for a single fitted function.

Ensembling is a different operation: it averages functions fitted
**independently**, which cancels the error each one owes to its own
initialisation rather than to the shared features. ⚡ **And we have already
measured that this idiosyncratic term is large** — for a different reason, and
we filed it as a warning:

* §5.6 / E8: two same-recipe nets differing only in `--seed` swung **0.073**
  against each other in a direct mirror head-to-head, against ±0.036 of sampling
  noise. Four of five arms flipped sign between seeds.
* Measured again today over 721 real ladder decisions: `policy_v5` and
  `policy_v5_s1` **disagree on 23.0%** of them (77.0% agreement) — comparable to
  our 27–30% miss rate against *human* demonstrators.

**Two nets we have been calling "the same agent" are as far apart as we are from
the field.** That is precisely the condition under which averaging pays, and
nothing in this repo has ever tried it (grepped: no ensemble, vote, or
logit-averaging code anywhere).

⚠ **The standing risk, stated first:** this project has measured six times that
better *fit* does not imply better *play*, at exchange rates differing 70×
(§8z: +8 decisions → +37 Elo; §8aa: +214 decisions → +14). An ensemble is a
fit-improving operation by construction. **It may improve agreement and move no
games.** That is the null we expect to have to report.

## Mechanism

`Ensemble.scores` softmaxes each member over the option set, then averages —
one vote each. ⚠ **Not a raw-logit mean:** a listwise loss pins the ranking, not
the scale, so two equally good nets can differ by a constant logit factor and a
raw mean would silently weight them. The **count** (how many options to take)
comes from member 0's existing rule, so "which options" and "how many" are not
confounded. Inference is ~1.2 ms per member against an 1,800 s pool — free.

Spec: `bc:<label>,net=out/policy_v5.npz+out/policy_v5_s1.npz`. The archived
agent name carries an md5 over **all members in order**, so swapping a member
is a new agent (rule 19).

## Arms — all mirror, DIRECT head-to-head, n=2,000, same weight files both sides

| arm | cell | what it answers |
|---|---|---|
| **A** | `ens(s0,s1)` vs **`policy_v5`** | **the shipping question** — does voting beat the net we actually ship? |
| **B** | `ens(s0,s1)` vs `policy_v5_s1` | does it beat the *other* member, or only the weaker one? |
| **C** | `policy_v5` vs `policy_v5_s1` | 🔴 **the cheap alternative** — if one seed is simply better, SWAP THE NET and skip the ensemble entirely |

⚡ **Arm C is the one to run first and it is not a control — it is a rival
hypothesis.** The 0.073 seed swing means the shipped net may not be the better
of the two we already have. If C is decisive, the intervention is "ship the
other file", which costs nothing and needs no new code.

⚠ **No seed term in any of these.** Every arm is a head-to-head between fixed
weight files, so the interval printed is the whole interval. This does **not**
generalise to "ensembling works" — that would need several independent pairs —
but it does answer "is this artefact stronger than that artefact", which is what
a shipping decision needs.

## Pre-registered bars

1. **Arm C first.** If `policy_v5_s1` beats `policy_v5` with a CI excluding 0.5,
   the shipped net is the wrong file and that is the finding; ensembling is then
   measured against the *better* member, not the incumbent.
2. **Promote the ensemble only if arm A's CI excludes 0.5 AND arm B does not
   lose.** Beating one member while losing to the other is a weighted coin, not
   an improvement.
3. **Any promotion is then confirmed on the weighted anchor set at n≥2,000**
   before a submission is even discussed — the mirror is 32% of the field and
   §8ar showed it is mirror-blind on exactly the slots that differ by matchup.
4. ⛔ **No submission on this result alone.** A submission evicts `55160229`
   (990.7, our displayed rank), so the bar is not "better than nothing" but
   "better by enough to be worth a certain eviction".

## Predictions, on the record

* Arm C: **null** — 0.073 was a swing between *arms*, not evidence that either
  seed is better; I expect the members to be indistinguishable at n=2,000.
* Arm A: **small positive, probably not resolvable** — 0.51–0.53, CI likely
  containing 0.5.
* Arm B: same as A.
* ⚠ If A lands **above 0.54**, that is larger than every intervention this
  project has produced since the v4 state block, and the first thing to suspect
  is a bug in the vote, not a win. The check would be: does the ensemble's
  advantage survive with `--raw` (logit mean), and does it disappear when both
  members are the *same file* (a degenerate ensemble must score exactly 0.5).
