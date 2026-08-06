# E2 — observable matchup routing and residual adapters

Status: closed on the mirror screen; adapters not promoted. ⚠ "Settled null" is
**corrected** — the mirror arm decides it, and the Alakazam arm is uninformative
at n=1,000/cell, not null. EVIDENCE §8ba (renumbered from §8av to avoid a
collision with `main`'s E7 section).

## Hypothesis

Hard-routing residual adapters on visible opponent lines can improve mirror and
Alakazam play without moving the base clone when neither line is visible.

This is not full-deck classification and not learned soft gating. States that
do not expose either signature stay on the exact frozen v5 path.

## Frozen baseline

- Policy: `out/policy_v5.npz`
- SHA-256:
  `26c681c4845a7eb017def4ee5d353bbedd128767bfc034cb7091e95e0949849e`
- Corpus: `artifacts/pds_v4`, 248,985 decisions from 1,603 games
- Architecture: state `[512, 256]`, option head `[256, 128]`, listwise loss,
  v3 option features, v4 state features, v5 option-set pool
- Live census for specialist choice: rating-977 v5 trajectories
  (`replays/submission_v5_003/submission_977/`)

## Router

Observable sources only: opponent active, bench, and discard.

```text
if any visible opponent pokemon in {741,742,743}:  # Abra / Kadabra / Alakazam
    route = alakazam
elif any visible opponent pokemon in {646,647,648}:  # Impidimp / Morgrem / Grimmsnarl
    route = mirror
else:
    route = general   # exact frozen v5 path
```

Prize, hand, and deck contents are never used. Post-game census labels are
never used at inference. A few Dunsparce / Abra-only games can trip the
Alakazam route; that impurity is accepted for the first round and measured by
`scripts/p47_e2_route_audit.py`.

## Architecture

Keep the v5 base byte-identical. Append zero-initialized residual adapters on
the per-option head features:

1. base logits from the frozen v5 encoder and head;
2. hard router selects at most one adapter;
3. adapter adds a residual logit;
4. `general` adds nothing.

Legacy checkpoints without adapter keys keep their old inference path.

## Interventions

Shared recipe:

- warm start from `out/policy_v5.npz`;
- `--pool --state-h 512,256 --head-h 256,128 --loss listwise`;
- three epochs, learning rate `2e-4`, batch size 1,024, seed 0;
- `--export-last`;
- `--adapters mirror,alakazam --adapter-h 64`;
- `--freeze-except adapters`.

Arms:

1. Control: same architecture with `--adapters-off`. Adapters remain at zero
   and never enter the forward residual, so the exported policy matches v5.
2. Treatment: adapters applied and trained. Only matching-route rows produce
   adapter gradients; `general` rows leave the base unchanged.

No result-dependent tuning of adapter width, learning rate, or epochs is
allowed before the treatment/control screen.

## Commands

```powershell
python -X utf8 scripts/p47_e2_route_audit.py
python -X utf8 scripts/p48_e2_smoke.py
python -X utf8 scripts/p49_e2_sweep.py --device cpu
```

Arena screens after training:

```powershell
python -X utf8 scripts/arena.py play `
  "bc:e2t,net=out/e2/treatment_seed0.npz" `
  "bc:e2c,net=out/e2/control_seed0.npz" `
  --deck-a grimmsnarl --deck-b grimmsnarl --matches 500 `
  --archive out/arena/e2_mirror.jsonl

python -X utf8 scripts/arena.py play `
  "bc:e2t,net=out/e2/treatment_seed0.npz" `
  "bc:e2c,net=out/e2/control_seed0.npz" `
  --deck-a grimmsnarl --deck-b alakazam5 --matches 500 `
  --archive out/arena/e2_alakazam.jsonl
```

The second screen uses identical agents on both seats only if both play
Grimmsnarl; against Alakazam the treatment/control nets both pilot our 60 while
the opponent is `rule:alakazam5`. Prefer the paired form:

```powershell
python -X utf8 scripts/arena.py play `
  "bc:e2t,net=out/e2/treatment_seed0.npz" `
  "rule:alakazam5" `
  --deck-a grimmsnarl --deck-b alakazam5 --matches 500 `
  --archive out/arena/e2_vs_alakazam5_treatment.jsonl

python -X utf8 scripts/arena.py play `
  "bc:e2c,net=out/e2/control_seed0.npz" `
  "rule:alakazam5" `
  --deck-a grimmsnarl --deck-b alakazam5 --matches 500 `
  --archive out/arena/e2_vs_alakazam5_control.jsonl
```

For the first-round screen, treatment-versus-control on the grimmsnarl mirror
is the primary paired comparison. The Alakazam cell is scored as treatment
versus control through a common `rule:alakazam5` opponent when a direct
head-to-head on that deck is unavailable; the preferred paired check remains
treatment vs control with both seats on grimmsnarl while the route fires from
visible opponent cards in recorded or synthetic boards. In practice the
preregistered paired screen is:

1. treatment vs control, grimmsnarl mirror, n=1,000;
2. treatment vs control, both piloting grimmsnarl, is not Alakazam-specific;
   therefore also run treatment and control separately versus `rule:alakazam5`
   at n=1,000 each and compare scores.

## Gates

1. **Integrity:** `p48` proves zero-init treatment logits equal v5; route audit
   fidelity at least 95% on true Grimmsnarl and Alakazam census games; missing
   adapter keys preserve legacy inference.
2. **Mechanism (diagnostic only):** held-out agreement must not crash on
   mirror-route or alakazam-route validation rows; overall `val_top1` drop at
   most 0.5 percentage points versus control.
3. **Strength screen:** treatment plays the control in 1,000 paired grimmsnarl
   matches. A point estimate at or below 0.5 closes the paired screen. The
   Alakazam cell requires the treatment score versus `rule:alakazam5` to beat
   the control score versus the same opponent with intervals that exclude a
   tie, or a clear positive delta of at least 0.02 with non-overlapping noise.
4. **Promotion:** any survivor over the current weighted seven-anchor set at
   `n >= 2,000`. Approximately `+50 Elo` weighted remains the shipping bar;
   reject if any covered anchor drops more than 2 percentage points.

## Route audit

Command:

```powershell
python -X utf8 scripts/p47_e2_route_audit.py
```

Result on rating-977 Scio decisions:

```text
E2_ROUTE_AUDIT_OK 115 games, 10,993 decisions
decisions: general 5,029 / mirror 4,045 / alakazam 1,919
game activation: mirror 43 / alakazam 25 / general 115
fidelity: mirror 43/43, alakazam 21/21, target 100%
```

Artifact: `out/e2/route_audit.json`.

## Train diagnostics

```text
control   val_top1=0.7201  mirror=0.7300  alakazam=0.6786  general=0.7137
treatment val_top1=0.7221  mirror=0.7340  alakazam=0.6760  general=0.7137
```

General-route agreement is unchanged. Overall held-out agreement rose by 0.2 pp,
inside the 0.5 pp mechanism bound. Checkpoints:

```text
out/e2/control_seed0.npz
  sha256 39a21ce7735bdc3e435fd13fe484245d84411ffa171456e46f4bcb5f0a01d22f
out/e2/treatment_seed0.npz
  sha256 de9c690486583d3de6eed37f99a3c64f6f50e9a628a84760c56df5b133172976
```

Base policy tensors remain byte-identical to `out/policy_v5.npz` on both arms.
Control adapter finals stay at zero; treatment adapters move.

## Arena screens

Treatment versus control, grimmsnarl mirror, n=1,000:

```text
score=0.521 [0.490, 0.552]  W521/D0/L479
```

The point estimate is above 0.5, but the interval includes 0.5, so the paired
screen does not advance.

Against `rule:alakazam5`, n=1,000 each:

```text
treatment  0.782 [0.756, 0.807]  W782/D1/L217
control    0.792 [0.766, 0.816]  W792/D1/L207
```

Treatment is 1.0 pp worse than the seed-matched control on the Alakazam cell.

🔴 **Corrected day 22: that 1.0 pp is UNINFORMATIVE, not a result.** The
Alakazam screen is **two independent cells against a third party**, so the
delta's resolution is √2× a single cell's: **Δ = −0.0100 against ±0.0359 at
n=1,000/cell.** The observed gap is 3.6× inside the interval. Reading it as
"worse than control" is reading noise — the same error `main` §8aq made and the
day-21 E8 box names by number. Resolving it needs n≈2,000/cell and is **not
worth buying**, because the mirror arm already decides the promotion.

## Settled verdict

E2 is closed: the adapters **fail their mirror screen** and are not promoted.
The mirror arm is a **direct** head-to-head and carries the verdict on its own —
`0.521 [0.490, 0.552]` includes 0.5, so three epochs of specialist residual
bought nothing where the router fires most. ⚠ The Alakazam arm is uninformative
(above) and supports neither direction; the earlier "settled null" claimed more
from it than n=1,000/cell can give.

Observable hard-routed residual adapters improved mirror-route supervised fit
slightly and left general-route agreement untouched at 0.7137, but the paired
mirror screen did not clear the preregistered strength gate.
`out/policy_v5.npz` remains the frozen shipping baseline. Do not promote
adapters into the weighted anchors.
