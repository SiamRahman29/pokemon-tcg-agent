# Beyond-BC run handoff

Last updated: 2026-08-04, E5 closed as local near-miss; First Round beyond-BC
program has no open runnable arms.

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

## Baseline frozen

- `out/policy_v5.npz`
- SHA-256:
  `26c681c4845a7eb017def4ee5d353bbedd128767bfc034cb7091e95e0949849e`

## Settled program state

| experiment | status | headline |
|---|---|---|
| E1 | settled null | outcome/count/both all ~0.50–0.51 at n=2,000 |
| E2 | settled null | adapters fail mirror and Alakazam screens |
| E3 | paused | teacher-blocked; pilot audit-only |
| E4 | blocked | needs E1/E3 prerequisites |
| E5 | settled near-miss | confirm 0.230 [0.177, 0.293] vs high 0.515 |

## E5 confirmation result

```text
low      0.380 [0.316, 0.449]
medium   0.420 [0.354, 0.489]
high     0.515 [0.446, 0.583]
confirm  0.230 [0.177, 0.293]   M=32, sb=8.0, n=200
```

Confirm fails the preregistered gate: below high, and upper bound below 0.5.
No planner promotion. No distillation. No fifth compute point.

## Next session

No runnable First Round beyond-BC arm remains without a new user input:

1. leave v5 frozen as the shipping baseline; or
2. resume E3 only if a qualified TCG reviewer / stronger validated teacher
   appears.

Round-2 paper design for the report is still owed as dossier material; do not
rebuild planning into the First Round agent on the strength of E5.
