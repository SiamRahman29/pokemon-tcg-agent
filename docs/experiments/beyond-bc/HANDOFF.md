# Beyond-BC run handoff

Last updated: 2026-08-04, E5 closed as local near-miss; First Round beyond-BC
program has no open runnable arms.

This is the handoff for the beyond-behavioral-cloning experiment program.
Future sessions should start here instead of using the repository-wide
`HANDOFF.md` as their task list. The common handoff remains historical context.

## Branch and safety

- ✅ **Merged into `main` on day 22 (`ba32c45`).** This program's code and
  records now live on `main`; the branch is retained but is no longer ahead.
  ⚠ **Nothing has been pushed** — the no-push / no-submit rule below still
  stands and needs user approval.
- Active branch: `experiments/beyond-bc`
- Do not commit or push to `main`.
- Do not push this branch, submit to Kaggle, or merge without user approval.
- Preserve unrelated user changes.
- Large checkpoints, replay dumps, and arena archives remain ignored.
- Tracked experiment records live in `docs/experiments/beyond-bc/`.

## Baseline frozen

- `out/policy_v5.npz`
- SHA-256:
  `26c681c4845a7eb017def4ee5d353bbedd128767bfc034cb7091e95e0949849e`

## Settled program state

| experiment | status | headline |
|---|---|---|
| E1 | settled null | outcome/count/both all ~0.50–0.51 at n=2,000 |
| E2 | closed on the mirror screen | 0.521 [0.490, 0.552]; ⚠ Alakazam arm uninformative, not null |
| E3 | paused | teacher-blocked; pilot audit-only |
| E4 | blocked | needs E1/E3 prerequisites |
| E5 | closed | firing rate, not compute, is the axis that moved |

## E5 result, corrected day 22

```text
arm      nominal cap   total planning s   firing rate   score
low          1.0 s            652            10.8%      0.380 [0.316, 0.449]
medium       2.0 s            616             7.4%      0.420 [0.354, 0.489]
high         4.0 s            606             4.2%      0.515 [0.446, 0.583]
confirm      8.0 s          8,288            35.0%      0.230 [0.177, 0.293]
```

🔴 **The cap went 1 → 2 → 4 s and realized compute went 652 → 616 → 606 s.** The
three cells that opened the gate are three draws at constant compute; pooled
they are **0.4383 [0.399, 0.478] over 600 games**, already a loss. The variable
that actually moved is the planner's **firing rate**, and score is monotone
decreasing in it across all four arms over an 8× range.

✅ `errors: 0`, `budget_aborts: 0` in all four arms — confirm is a healthy cell.
No planner promotion. No distillation. No fifth compute point.

## Day-22 reconciliation with `main`

- EVIDENCE sections renumbered **8au→8az (E1), 8av→8ba (E2), 8aw→8bb (E5)**:
  `main` uses 8au/8av/8aw for E6/E7/E8 and the two files auto-merge *cleanly*
  into six sections under three numbers with no conflict marker. `main` had 22
  cross-references to those numbers, this branch had none.
- ✅ **No beyond-BC experiment needs re-running after `main`'s validation
  audit.** Every arena run here is mirror-direct or vs `rule:alakazam5` on its
  own tuned deck, so there is no exposure to §8ax (the Crustle deck confound) or
  §8ay (corrected field-share weights), and nothing here computes a weighted
  `W = Σ wᵢΔᵢ`. All six checkpoints under `out/e1/results/` and `out/e2/` load
  under `main`'s hardened `policynet.load()`.
- ✅ **MERGED INTO `main` ON DAY 22** (`ba32c45`, fast-forward). `main` and this
  branch are now the same commit. Both instrument sets are live in `arena.py`:
  `[health]` says whether the AGENT degraded, the sequencer line says whether
  the PLANNER did. All four smokes pass (`p39`, `p48`, `p46`) and 38/38
  checkpoints load.
- 🔴 **TWO INTEGRATION BREAKS THAT WERE NOT CONFLICTS**, both found by the
  smokes rather than by git — this is the merge's real lesson:
  1. **`p45_dagger_export.py` was writing unloadable shards.** `main`'s day-20
     `Writer` gained an `attr` column; the E3 export never passed it, so
     `np.stack([None, ...])` produced an OBJECT array and every DAgger shard
     failed `allow_pickle=False`. Beyond loading, it broke E3's core invariant:
     treatment and control must differ ONLY in the reviewed label, so the shard
     has to be column-identical to a `build_policy_dataset.py` one.
  2. **`optfeat.OPT_DENSE` grew 37 → 46 on `main`** (day 21) and `--opt-cols`
     defaults to it, so anything warm-starting from `policy_v5.npz` (708 wide)
     now builds a 726-wide state and is refused. `--opt-cols 37` is pinned in
     `p46_dagger_smoke.py` and `p49_e2_sweep.py`. ⚠ **Any future script that
     `--init`s from a pre-day-21 net must pin it too.**
- The five code conflicts and how they were resolved, for the record:
  - **`.gitignore` — take the UNION.** This branch added `out/e1/ out/e2/
    out/e3/ out/e5/`; `main` added `out/emb/` during E8. Neither side ignores
    the other's scratch, which is how a `git add -A` on this branch swept 11 MB
    of `main`'s E8 checkpoints into a docs commit (caught and removed).
  - **`arena.py` — take BOTH instrumentation blocks.** This branch's sequencer
    stats (`completed/overruled/errors/budget_aborts`) and `main`'s `[health]`
    line are complementary; E5 above is the argument for keeping both.
  - **`policynet.py` / `bcagent.py` — take both hunks.** E1's `outcome_w`/
    `count_w` and `main`'s v7 vocab LUT are both append-only optional-key reads
    and compose; verified by loading all six branch checkpoints under `main`'s
    `policynet.load()`.
  - **`train_policy.py` is the only real semantic merge** — E1's multitask heads
    (+359 lines) against `main`'s `--vocab` remap, both in the training loop.
    Re-run `p39_multitask_smoke.py` after resolving it.

## Next session

No runnable First Round beyond-BC arm remains without a new user input:

1. leave v5 frozen as the shipping baseline; or
2. resume E3 only if a qualified TCG reviewer / stronger validated teacher
   appears.

Round-2 paper design for the report is still owed as dossier material; do not
rebuild planning into the First Round agent on the strength of E5.
