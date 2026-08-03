# Beyond-BC run handoff

Last updated: 2026-08-03, E5 first scale curve continues; confirmation pending.

This is the handoff for the beyond-behavioral-cloning experiment program.
Future sessions should start here instead of using the repository-wide
`HANDOFF.md` as their task list. The common handoff remains historical context.

## Branch and safety

- Active branch: `experiments/beyond-bc`
- Do not commit or push to `main`.
- Do not push this branch, submit to Kaggle, or merge without user approval.
- Preserve unrelated user changes.
- Large checkpoints, replay dumps, and arena archives remain ignored.
- Tracked experiment records live in `docs/experiments/beyond-bc/`.

## Session decomposition

Run one stage per session where practical:

1. **S1 / E1:** auxiliary outcome and count representation learning.
2. **S2 / E3:** uncertainty collector, review UI/queue, and DAgger fine-tune.
3. **S3 / E2:** census-selected observable routing and residual adapters.
4. **S4 / E4:** conservative value/Q policy improvement, only if E1/E3 pass
   their prerequisites.
5. **S5 / E5:** Round-2 BC-guided planning scaling and distillation.

At each session end, update this file with exact state, active commands,
artifacts, blockers, and the first command for the next session.

## User inputs received

### E1 GPU results

Input archive:

```text
out/e1/e1_results.zip
```

Validated and extracted by:

```powershell
python -X utf8 scripts/p42_e1_intake.py
```

Result:

```text
E1_INTAKE_OK cuda 12 epochs seed=0
```

The user changed only the Kaggle extraction cell because Kaggle automatically
unzipped the attached dataset. This did not change training. All four logs
contain the exact preregistered commands, and all checkpoint hashes match
`manifest.json`.

Extracted artifacts:

```text
out/e1/results/
  control_seed0.npz
  outcome_seed0.npz
  count_seed0.npz
  both_seed0.npz
  corresponding logs
  manifest.json
```

### Fresh v5 ladder replays

The two submissions are decision-identical v5 policies; the lower-rated one
only adds logging. Keep their censuses separate because opponent populations
depend strongly on our rating.

```text
replays/submission_v5_003/submission_897/   86 games
replays/submission_v5_003/submission_977/  115 games
```

At rating 897:

- overall 57/86 = 66.3%;
- Alakazam 19.8%;
- mirror 15.1%;
- Archaludon 9.3%;
- Crustle 5.8%;
- Dragapult 4.7%.

At rating 977:

- overall 73/115 = 63.5%;
- mirror 37.4%;
- Alakazam 18.3%;
- Crustle 11.3%;
- Archaludon 7.0%;
- Garchomp 5.2%;
- Dragapult 5.2%.

Dunsparce variants are only about 2–3% in both current populations. Do not
request a targeted Dunsparce dump. The existing mirror, Alakazam, Crustle,
Archaludon, Garchomp, and Dragapult anchors cover 84.3% of the rating-977 field.

## Baseline frozen

- `out/policy_v5.npz`
- SHA-256:
  `26c681c4845a7eb017def4ee5d353bbedd128767bfc034cb7091e95e0949849e`
- `artifacts/pds_v4`: 248,985 decisions, 1,603 games, eight hashed shards
- Held-out single-choice agreement: 71.6% over 12,939 rows
- Same-checkpoint arena smoke completed without fallback

Full hashes and preregistration are in `E1-multitask.md`.

## E1 settled result

E1 is closed. No arm advances to weighted anchors:

```text
outcome  0.505 [0.484, 0.527]
count    0.507 [0.486, 0.529]
both     0.500 [0.478, 0.522]
```

## Active S2 / E3 state

E3 remains paused at the teacher gate. The 15-label pilot is audit-only and
must not train. Needs a qualified TCG reviewer or an automated teacher proven
stronger than v5. The existing planner is ineligible on the unrepaired small
setting; E5 is testing whether more compute changes that.

## E2 settled result

E2 is closed as a null. Adapters are not promoted.

```text
mirror treatment vs control   n=1,000  0.521 [0.490, 0.552]
vs rule:alakazam5 treatment   n=1,000  0.782 [0.756, 0.807]
vs rule:alakazam5 control     n=1,000  0.792 [0.766, 0.816]
```

## Active S5 / E5 state

The first three preregistered scale cells are complete. Archives each have 200
games and match `out/e5/manifest.json`.

```text
low     M=4   sb=1.0  0.380 [0.316, 0.449]  0.319 s/plan  aborts=0
medium  M=8   sb=2.0  0.420 [0.354, 0.489]  0.427 s/plan  aborts=0
high    M=16  sb=4.0  0.515 [0.446, 0.583]  0.724 s/plan  aborts=0
```

Gate reading:

- integrity and mechanism pass;
- scores are non-decreasing;
- high point estimate is above 0.5, so E5 does **not** close locally;
- high CI still includes 0.5, so high is **not** promoted;
- one higher confirmation cell is preregistered: `confirm` = `M=32`, `sb=8.0`.

Do not pick the best of low/medium/high. Do not distill yet.

First command for the next session, preferably outside Cursor:

```powershell
cd E:\Kaggle\pokemon-tcg-simulation-2
python -X utf8 scripts/p51_e5_scale.py --arms confirm
```

After it finishes, apply the confirmation gate in `E5-planning.md` and update
this file. `out/policy_v5.npz` remains the frozen shipping baseline.
