# E30 — corpus composition: does WHO WE TRAINED AGAINST cost us anything? Pre-registered.

**Status: PRE-REGISTERED 2026-08-12 (day 31), before any corpus is built and
before any arena game.** Frozen at the commit adding this file. Round context:
`ROUND-2026-08-12.md`. Sizing gate: `E29-mine-and-archetype-census.md` §R3.

**This is the round's only arena cell and the only thing that can ship by
08-17.**

---

## 1. ✅ The gate passed — for ONE archetype, and not the one it was parked on

E29 §R3, over 1,969 fresh top-band games (`avg_score` 1086–1300):

| target (>3× under-represented) | field | corpus | ratio | games | decisions | gate (≥50 / ≥500) |
|---|---|---|---|---|---|---|
| **Alakazam** | **22.0%** | 3.39% | **0.15×** | **77** | **5,775** | ✅ **PASSES** |
| Mega Lucario ex | 4.0% | 0.00% | 0.00× | 38 | 3,116 | 🔴 fails (games) |
| Cynthia's Garchomp ex | 6.7% | 0.40% | 0.06× | 5 | 449 | 🔴 fails (both) |
| **Archaludon ex** | 8.0% | 0.02% | 0.00× | **0** | **0** | 🔴 **fails absolutely** |

⇒ **E30 runs on Alakazam and only Alakazam.** ⛔ The other three are **not** in
this cell; the thresholds are not relaxed to admit them (E29 §4).

⚡ **Alakazam is the right one to have survived.** It carries the **largest
field weight of any target (22.0%)**, it is the most under-represented row that
is not literally zero (**0.15×**), and E29 §R1 measured it at **28.6% of our own
1000+ band — tied with the mirror.** ⛔ **Archaludon's original kill stands and
is now confirmed on a second independent population** (0 of 1,969).

⚠ **A further deepening of the mine is currently blocked**: Kaggle returned
**HTTP 429** after ~2,000 downloads today. Mega Lucario needs ~1.3× more depth
and may become admissible later; **it is not admitted retroactively into this
cell** — it would need its own registration.

---

## 2. The hypothesis, and the two axes it is NOT

The corpus holds **63.9%** of opponent board slots as the mirror against a
**33.3%** field weight, and Alakazam at **3.39%** against **22.0%**. E29 §R2
makes the mismatch **worse** than `PARKED-corpus-coverage.md` measured: in the
current top band the mirror is **1.1% of games** and **13.4% of Grimmsnarl-seat
decisions**.

⚠ **Not the dead "more data" axis** — that was **volume**; this is **which
games**, at held-constant volume (§4).
⚠ **Not the dead demonstrator-selection axis** — rating-weighting (**−55**) and
single-expert (**−92**) reweighted **who plays**; this reweights **who they play
against**. The demonstrator distribution is held fixed by construction (§4).

⛔ **It is not an encoding change.** E7/E8 both tried to fix an unseen archetype
by re-encoding cards and both measured nothing; the encoding axis is closed and
nothing here touches `features.py`.

---

## 3. ⚠ The instrument problem, stated BEFORE the build — this is why the design has two arms

§8ac justified the mirror A/B as *"our most sensitive instrument is now also our
most representative one."* 🔴 **E29 killed the second half of that sentence**:
the mirror is **1.1%** of the top band and **28.6%** of our own 1000+ band, not
71.4%.

⇒ **A corpus reweighted AWAY from the mirror is expected to lose a
mirror-vs-mirror A/B even if it is better on the ladder.** A mirror-only design
would read that as a kill, and it would be wrong. **This is the one failure mode
that makes the cell close to unfalsifiable, so it is designed out rather than
noted afterwards.**

---

## 4. ⛔ The build — composition is the ONLY thing that varies

**One pooled game set**, both arms drawn from it: every Grimmsnarl-seat game we
hold across `replays/2026-07-26` … `2026-08-11` plus the expert dumps.

| | treatment | control |
|---|---|---|
| game pool | identical | identical |
| demonstrator set | identical | identical |
| **opponent-archetype composition** | **reweighted toward E29 §R2's measured field** | **the CURRENT corpus's composition** (mirror-heavy, 63.9% slots) |
| total decisions | **capped equal** | **capped equal** |
| training config, seeds, epochs, init | byte-identical | byte-identical |

⚠ **Equal size is not optional.** Holding composition fixed while size varies
re-runs the dead volume axis and would confound the two. The cap is set by what
the *reweighted* arm can support — the binding constraint is Alakazam at 5,775
decisions.

⛔ **BUILD-SIZE GATE, before training:** if the equal cap falls below **60% of
`pds_v4`'s decision count**, the arms are too small to be compared against a net
trained at full size, and the cell reports **VOID ON BUILD SIZE** rather than a
null. ⚠ Pre-registered because a small-corpus null is indistinguishable from a
composition null, and this is exactly the confound §8bh's screen/confirm lesson
was about.

**Mechanics.** `build_policy_dataset.py` selects by *player*, not by opponent
archetype, so the two compositions are produced by **staging the selected
replays into two directories** and running the unmodified builder on each. ⇒ the
builder is not touched and cannot differ between arms.

---

## 5. The arms — pre-registered, in priority order, with the cut rule frozen

≥3 seeds per arm, **n ≥ 2000** per arm, shipped config (`--no-rules`), against a
**byte-identical control**. ⚠ A treatment-minus-control interval is **√2×** a
single cell's.

| | arm | deck-b | why |
|---|---|---|---|
| **A** | mirror | `grimmsnarl` | the sensitive instrument, and the one §3 predicts may read DOWN |
| **B** | **vs Alakazam** | `alakazam` | **the direct mechanism test** — the archetype whose representation actually changed |
| **C** | vs Dragapult | `dragapult_ex` | generalisation control: 18.2% of the top band, and **not** a target, so it should NOT move if the effect is composition-specific |

⛔ **If runway forces a cut, C is cut first, then nothing.** A and B are both
required. **Frozen here so a post-hoc cut cannot be narrated as a design.**

---

## 6. Reading rules — keyed on the COMPARISON, written before the first game

⚠ Charter §3 rule 2: key the branch on the comparison, never on one arm. E25's
own branch condition fired correctly for a reason its condition did not
establish.

| arm A (mirror) | arm B (vs Alakazam) | verdict |
|---|---|---|
| **down** | **up**, CI excludes 0 | ⚡ **INSTRUMENT ARTEFACT, NOT A KILL** — precisely §3's predicted case. The lever works and the mirror A/B cannot see it. ⇒ ship decision moves to arm B + C, and **§8ac's instrument justification is formally retired** |
| ~ | **up**, CI excludes 0 | ✅ **The lever works**, and cheaply — ship candidate |
| **up** | **up** | ✅ strongest outcome; ship candidate |
| ~ | ~ | 🔴 **NULL — the axis closes.** The last untested lever is spent, and `PARKED-corpus-coverage.md` closes for the second and final time |
| **down** | **down** | 🔴 **NULL, and worse than null** — reweighting away from the mirror costs us on both. Closes. |
| any | **up**, but **C also up by ≥ the same margin** | ⚠ **NOT composition-specific.** Suspect a general strength gain (or a nuisance draw, §8bg); do not attribute it to composition |

⛔ **`val_top1` may neither promote nor kill** (charter §3 rule 3). It is a
conformity metric; a perfect score is compatible with being modal, and modal is
~1000.

⚠ **Ship bar, frozen:** point **≥ 0.53** AND CI excluding 0.500 on a
**confirmation run against the incumbent `v5_s2` on FRESH games**. ⛔ Screens
never ship — that is F2's paid-for lesson (`s7` screened 0.528 and confirmed
0.487).

---

## 7. Obligations this cell inherits from charter §3

- ⚠ **Report the REALISED changed-picks/game from the run's own logs** — a knob
  is not a variable (three instances in three sessions). A corpus change is a
  knob; the realised deviation rate is the variable.
- ⚠ **Place the result on the E25/E26 cost law** and report
  `f = (method − rate-matched coin flip) / (0.500 − coin flip)` against
  `f_eval = 0.12` and `f_coh = 0.758`.
- ⚠ Any matched control samples from the **treatment's own per-option-count
  histogram** — "rate-matched" was twice found not matched (E26, E27).

## 8. ⚠ Limits, stated before the result

- **Availability is not value.** The games existing does not imply training on
  them buys Elo — E6→E7 is that trap, and the parked file's own warning is *a
  compelling diagnosis is not a working repair.*
- **One archetype.** This tests Alakazam. It does not test the 40.7% claim as a
  whole, and a null here does not prove composition is irrelevant in general —
  only that fixing the largest reachable slice of it bought nothing.
- **Fresh games are a band above ours** (`avg_score` 1086–1300 vs our ~933), so
  the treatment corpus is also a slightly *stronger* demonstrator pool. ⚠ The
  equal-size, same-pool control is what keeps that from becoming the treatment.
- ⛔ **Two-agent repo:** this cell writes only this file, its staged corpora, and
  its arena logs. It does not edit `HANDOFF.md`, `ROADMAP.md` or
  `report/EVIDENCE.md`.

---

# ▶ RESULT — 2026-08-12. 🔴 VOID ON BUILD SIZE. The gate fired before a single game was played.

⛔ **§4's build-size gate, frozen ~30 minutes before it was run, fires.** Zero
arena games, zero training runs, zero corpus staged.

| | decisions |
|---|---|
| `artifacts/pds_v4` (8 shards) | **248,985** |
| pre-registered floor (60% of it) | **149,391** |
| **E30's reweighted cap** — Alakazam-bound, 5,775 decisions at a 22% target | **26,250** |

**26,250 is 10.5% of `pds_v4`, against a 60% floor.** ⇒ **VOID ON BUILD SIZE**,
exactly as written.

## ⚡ And the arithmetic closes the axis at EVERY composition, not just this one

The gate is not a threshold that a different target weight would sneak past.
Holding the corpus at the 149,391-decision floor, the largest Alakazam share
5,775 decisions can support is

```
5,775 / 149,391 = 3.87%
```

⇒ **Alakazam can be lifted from its current 3.39% to at most 3.87% — half a
percentage point — against a 22.0% field weight.** 🔴 **The lever cannot be
pulled far enough to matter at any setting that leaves a trainable corpus.**
That is a stronger statement than the pre-registered VOID and it does not depend
on where the floor was set.

## What this does and does not close

- 🔴 **`PARKED-corpus-coverage.md` closes for the second time, on a NEW reason.**
  Day 25 killed it on **availability** ("the games do not exist"). E29 §R3
  showed that reason is now half false — Mega Lucario went 0 → 38 games, the
  feed ceiling rose 1055 → 1300, and Alakazam cleared the availability gate at
  77 games. **It dies instead on ARITHMETIC: the games exist and there are
  nowhere near enough of them to change a 249k-decision corpus's composition.**
- ⛔ **The round has no arena cell.** Nothing ships from E30 by 08-17, and the
  08-15 submission is unaffected by it.
- ⚠ **One variant survives and is NOT run here: upsampling** — repeating the
  5,775 Alakazam decisions to reach a target share rather than capping the
  corpus. ⛔ It is a **different and weaker intervention** (it raises exposure,
  not coverage) and would repeat 77 games ~9.5×, which is a strong overfitting
  risk on a sample that small. **It needs its own pre-registration and it is not
  licensed by this cell.**
- ⚡ **The cheap-gate discipline paid again.** This is the fifth axis closed at a
  sizing gate before a build (§8bm / §8bp / §8br / §8bs, now E30). The cost of
  the whole cell was one census and one `numpy` row count.

⚠ **What would re-open it: materially more Alakazam data, not a different
weighting.** At the current 5,775 decisions the answer is fixed; roughly **25×**
more would be needed to hold 22% at full corpus size. Kaggle's 429 blocks
further mining today, and the ceiling on what the feed can supply is unknown.
