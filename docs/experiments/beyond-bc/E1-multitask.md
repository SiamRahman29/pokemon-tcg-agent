# E1 — shared encoder with auxiliary learning

Status: settled null; no auxiliary arm promoted.

## Hypothesis

The policy encoder may become more useful if it must also predict game outcome
and the number of options selected. This tests representation learning, not the
closed B8 intervention: B8 directly reweighted policy imitation by terminal
outcome, whereas E1 keeps the listwise BC target unchanged and applies small
auxiliary losses to the shared state representation.

## Frozen baseline

- Corpus: `artifacts/pds_v4`
- Rows: 248,985
- Games: 1,603
- Train/validation rows: 235,314 / 13,671
- Checkpoint: `out/policy_v5.npz`
- Checkpoint SHA-256:
  `26c681c4845a7eb017def4ee5d353bbedd128767bfc034cb7091e95e0949849e`
- Architecture: state `[512, 256]`, option head `[256, 128]`, listwise loss,
  option pool enabled, 702,913 parameters
- Held-out single-choice agreement: 9,263 / 12,939 = 71.6%
- Baseline arena smoke: 20 same-checkpoint games completed without fallback;
  score 0.550 `[0.342, 0.742]`. This is a health check, not a strength result.

Dataset shard SHA-256 values:

```text
9dc3865546ca13f319d56aad8ff1d8f35a31dd52fa0806d8a99da4b9ac74e670 d26/shard_000.npz
a5ec4bbe5557491c9dc4a0c9f0f113ef8efe545033c573ee2a52e5079a3aa4b0 d26/shard_001.npz
27711bda799f8f23ccb0f001e4a08a220656ca7de331a1ee58de274a237f9926 d27/shard_000.npz
4e5495b9bbe8e1ee6e1a4803d4026b6097182f195cabb5a52b87f1d79c513997 d27/shard_001.npz
286e9dd40968b3b490f7d776f5e1dd078579cbb0b45617e542c26419f374a906 d28/shard_000.npz
6df5e206bbdb4872395fdd4bc58f7bb6b4dac58d9fda98d8d944d583bee3906c d28/shard_001.npz
93b9683aba54623607ff8e7a923934253ca8bacb158c92774263ab14ae4ea070 d29/shard_000.npz
70ddd31287061fb848f767d5a65959c732a23f7b0b83e57921b34d0163f73c49 d29/shard_001.npz
```

## Interventions

All four arms use seed 0, 12 epochs, batch size 1,024, the same split,
`--export-last`, and the exact v5 architecture.

1. Control: policy loss only.
2. Outcome: policy + `0.1 × BCE(win_logit, won)`.
3. Count: policy + `0.1 × soft_BCE(count_logit, selected_fraction)` on
   variable-count rows.
4. Both: policy + both auxiliary terms at weight 0.1.

Weight 0.1 makes each auxiliary term roughly five to six percent of the initial
total objective. It is large enough to train the head without allowing either
label to dominate listwise imitation.

## Implementation invariants

- Auxiliary modules are initialized after all policy modules, so adding them
  does not change seeded policy initialization.
- Legacy checkpoints have no auxiliary keys and retain their old inference.
- Policy ranking remains the default even for multitask checkpoints.
- Learned count selection is opt-in through `SA_COUNT_MODE=learned`.
- Auxiliary treatments require `--export-last`.
- CUDA training exports CPU NumPy arrays for the existing inference runtime.

## Smoke results

`scripts/p39_multitask_smoke.py` verifies:

- exact initial policy-parameter and logit equivalence;
- count-target values and masking;
- legacy and multitask NPZ export/load;
- legacy warm-start behavior.

A one-epoch full-corpus integration run completed in 48 seconds:

```text
params=703427
val_top1=0.6449
val_out_bce=0.6587
val_out_acc=0.6046
val_count_mae=0.0830
```

The resulting checkpoint loaded and completed a four-game arena health check.
Neither number is a strength verdict.

## Arena screens

Outcome auxiliary head versus the seed-matched control:

```text
n=2,000  score=0.505 [0.484, 0.527]  W1011/D0/L989
P0: 517-483  P1: 494-506
```

Verdict: null. The interval includes 0.5 and the point estimate is far below
the First Round shipping bar. Do not promote the outcome-only arm to the
weighted anchors.

Count auxiliary head versus the seed-matched control:

```text
n=2,000  score=0.507 [0.486, 0.529]  W1015/D0/L985
P0: 548-452  P1: 467-533
```

Verdict: null. The large seat split is balanced by the paired design; the
combined estimate includes 0.5 and does not justify weighted-anchor promotion.

Combined outcome and count heads versus the seed-matched control:

```text
n=2,000  score=0.500 [0.478, 0.522]  W1000/D0/L1000
P0: 537-463  P1: 463-537
```

Verdict: exact null. The paired seats cancel exactly, and the interval gives no
evidence that combining the two auxiliary gradients improves play.

## Settled verdict

E1 is closed as a three-arm null. Outcome-only (`0.505`), count-only (`0.507`),
and combined (`0.500`) auxiliary learning all failed the preregistered
head-to-head screen, so none advances to the weighted anchors and
`out/policy_v5.npz` remains the frozen shipping baseline.

The combined arm achieved the best final held-out agreement (`0.7199` versus
control `0.7134`) and exactly 0.500 playing strength. This independently repeats
the project's central warning: a better supervised diagnostic is not evidence
of a stronger agent. The outcome head also overfit after its first epoch.
Selected-count prediction learned a low-error target, but neither auxiliary
gradient improved the policy under this corpus and objective. E4 therefore does
not inherit a validated value representation from E1.

## Gates

1. Reject a broken arm if its auxiliary metric is degenerate or the exported
   checkpoint fails the smoke/arena health checks.
2. Use validation metrics only to confirm learning, never to select the winner.
3. Screen surviving arms head-to-head against the seed-matched control.
4. Confirm a promoted arm over the current seven-anchor weighted set at
   `n >= 2,000`.
5. Approximately `+50 Elo` weighted is the First Round shipping bar. Smaller
   repeated gains are retained as evidence but do not automatically replace v5.
