# E3 — uncertainty-gated DAgger

Status: paused at teacher gate; the initial human-review pilot is audit-only.

## Hypothesis

The frozen v5 clone still reaches decisions on its own trajectories where two
semantically distinct actions have nearly equal logits. Correcting a small,
diverse set of those states with state-only human labels should reduce
on-policy error without moving the policy away from the field-modal behavior
that B7 showed is protective.

This is not more expert-state behavior cloning and not outcome reweighting.
States come from the clone's own live trajectories; the treatment differs from
a self-distillation control only in the reviewed action label.

## Frozen inputs

- Policy: `out/policy_v5.npz`
- SHA-256:
  `26c681c4845a7eb017def4ee5d353bbedd128767bfc034cb7091e95e0949849e`
- Anchor corpus: `artifacts/pds_v4`, 248,985 decisions from 1,603 games
- Trajectories:
  `replays/submission_v5_003/submission_977/`, 115 live ladder games
- Acting team: `Scio`
- The replay population was collected at rating 977 and is kept separate from
  the rating-897 population.

The existing planner is not a teacher for E3: its repaired form still scored
`0.375 [0.311, 0.444]` against BC. Planner output may be displayed later as an
advisory diagnostic, but it cannot create an automatic label.

## Queue construction

Command:

```powershell
python -X utf8 scripts/p43_dagger_queue.py
```

Eligibility:

1. the observation belongs to `Scio` and has at least two legal options;
2. the decision is not forced and has at least one selected and unselected
   option under v5;
3. replay action and a fresh v5 inference agree;
4. the selected/unselected boundary options are not bitwise-equivalent under
   the complete v3 option encoding.

Rank key:

1. ascending logit margin between the lowest-scored selected option and the
   highest-scored unselected option;
2. descending normalized softmax entropy;
3. stable item id.

The queue caps each replay at three decisions and each `(select type, context)`
bucket at 24 on the first pass. This prevents one long game or common context
from consuming the review budget.

Result:

```text
E3_QUEUE_OK 160 items from 115 replays
policy fidelity: 99.797% (20/9,831 mismatches)
rankable candidates: 8,963
queued margin range: [0.0001, 0.1316]
```

Artifacts:

```text
out/e3/review_queue.jsonl
out/e3/review_queue.manifest.json
```

## Planned human-label protocol

Do not run this with the current reviewer. If a qualified reviewer becomes
available, run:

```powershell
python -X utf8 scripts/p44_dagger_review.py
```

The local UI shows the acting player's observable state and legal options.
Game outcome and logged action are absent. Clone choices and logits are hidden
until explicitly revealed to reduce anchoring.

For each decision the reviewer:

- selects a legal action;
- records high or low confidence;
- may add a rationale;
- may abstain.

Only high-confidence labels train. The gate is at least 100 such labels; low
confidence and skipped rows remain audit evidence but contribute zero gradient.
Labels are saved atomically to `out/e3/reviews.json` after every decision.

## Human-review pilot

The intended reviewer disclosed that they have no Pokémon TCG expertise. The
pilot stopped after 15 saved reviews (7 high confidence, 4 low confidence, 4
skips). Five of the seven high-confidence labels contradicted v5.

Inspection showed that several contradictions selected functionally equivalent
copies or options that the UI did not distinguish adequately. One
high-confidence rationale also depended on using an attack before an ability,
although attacking ends the turn. These labels therefore cannot be treated as
teacher actions.

The pilot is retained as a queue/UI audit in `out/e3/reviews.json`, but
contributes zero training rows. E3 is not a policy null: it stopped because no
qualified human or independently validated stronger automated teacher is
available. The existing planner cannot fill that role because it already lost
to BC at `0.375 [0.311, 0.444]`.

## Frozen treatment/control design

After a qualified review:

```powershell
python -X utf8 scripts/p45_dagger_export.py
```

The exporter holds out one fifth of high-confidence labels by stable item hash.
The remaining states produce two corpus-format datasets:

1. treatment — reviewed human actions;
2. control — frozen v5 actions.

Their states, features, game ids, order, and training recipe are identical.
Only `opt_chosen` differs. This control separates the value of corrections from
extra self-distillation on hand-picked states.

The first fine-tune is preregistered as:

- warm start from v5;
- train the policy head only;
- listwise loss;
- three epochs, learning rate `2e-4`, batch size 1,024, seed 0;
- final-epoch export;
- original `pds_v4` corpus supplies 90% of supervised loss mass;
- curated E3 rows supply 10% through `--primary-mass 0.1`.

No result-dependent tuning of label mass, epochs, or learning rate is allowed
before the treatment/control screen.

## Gates

1. **Integrity:** queue/replay policy fidelity must remain at least 95%;
   treatment and control exports must differ only in `opt_chosen`.
2. **Teacher signal:** at least 100 high-confidence labels and a non-zero
   correction rate over v5 are required. Otherwise E3 closes as "no actionable
   disagreement," not as a policy null.
3. **Mechanism:** treatment must improve agreement on the untouched human
   holdout relative to the self-distillation control without reducing frozen
   corpus validation agreement by more than 0.5 percentage points.
4. **Strength screen:** treatment plays the control in 1,000 paired
   Grimmsnarl matches. A point estimate at or below 0.5 closes E3; a positive
   result must have a 95% interval excluding 0.5 to advance.
5. **Promotion:** confirm any survivor over the current weighted anchor set at
   `n >= 2,000`. Approximately `+50 Elo` weighted remains the shipping bar.

