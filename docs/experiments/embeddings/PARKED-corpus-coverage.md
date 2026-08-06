# PARKED — the training corpus does not match the field we play

Found 2026-08-05 while chasing why Mega Lucario is 0/6 out of vocabulary.
**Parked deliberately. Not an embedding problem — do not work it under the
embedding component.**

Opponent board slots only (`slots[:, 6:12]`, `artifacts/pds_v4`, 1,109,220
lookups). Field weights from `p33_anchor_resolution.ANCHORS`.

| archetype | field weight | share of opponent slots in training | ratio |
|---|---|---|---|
| Grimmsnarl (mirror) | 33.3% | **63.9%** | 1.92x |
| Alakazam | 22.0% | 3.39% | **0.15x** |
| Archaludon | 8.0% | 0.02% | **0.00x** |
| Crustle | 6.7% | 6.26% | 0.93x |
| Garchomp | 6.7% | 0.40% | **0.06x** |
| Dragapult | 5.3% | 2.74% | 0.52x |
| Mega Lucario | 4.0% | 0.00% | **0.00x** |

**40.7% of the field is under-represented by more than 3x; the mirror is
over-represented 2x.** Lucario was never the special case — it is only the one
where the symptom was visible enough to trip over.

## Caveats, so this is not overread later

- Archetype ids are name-matched. The 0.93x and 0.52x rows are not load-bearing;
  the **0.00x / 0.06x / 0.15x rows are unambiguous**.
- Field weights are from §8ac, which itself warns they are a function of *our*
  rating band, and we have moved.
- This counts board *presence*, not games.
- ⚠ **A compelling diagnosis is not a working repair.** E6 → E7 is exactly that
  trap, one day old. Nothing here shows that fixing coverage buys Elo.

## The cheap first probe, when this is picked up

Do the replays we already hold *contain* games against the missing archetypes
that the miner discards, or do those games not exist at the demonstrator's
rating band? That is a sizing gate, not a build, and it either opens the lever
or kills it before anything is trained.

Measured by the inline script in the day-20 session; re-derivable from
`artifacts/pds_v4` + `p33_anchor_resolution.ANCHORS`.
