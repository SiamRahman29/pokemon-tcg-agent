# E6 — does the card-identity channel carry decision weight?

Status: **settled positive, and it localises a live blind spot.** No promotion
yet; this is a diagnosis, not a fix.

## Why permutation and not zeroing

Zeroing an embedding table is the obvious ablation and the wrong one: it moves
the input distribution the downstream layers were trained against, so
degradation from "identity destroyed" cannot be separated from degradation
from "activations off their training scale".

Permutation feeds the identical multiset of row vectors and scrambles only the
card -> row assignment. `perm_seen` restricts the shuffle to rows that ever
received a gradient (the p53 census), holding *trainedness* constant too.

Row 0 is never permuted in any mode. `slot_emb[0]` is the empty/unresolved
slot and training drove it to norm 0.835 against a 3.95 table mean, so it
encodes "nothing here" rather than a card; scrambling it would confound the
result with board occupancy.

## Frozen baseline

- Policy `out/policy_v5.npz`, sha256 `26c681c4845a7eb0...` (matches the E2
  frozen baseline, and is byte-identical to the `sa/policy_net.npz` inside
  `dist/submission.tar.gz` — the shipped net)
- Corpus `artifacts/pds_v4`, 248,985 decisions
- All arms are the same net weights with rows permuted. **Nothing is retrained.**

## Vocabulary census (`scripts/p53_emb_vocab.py`)

| table | source columns | rows | ever looked up | share |
|---|---|---|---|---|
| `slot_emb` | `slots`, `xslots` | 1300 | 104 | 8.0% |
| `bag_emb` | `bag_*_flat` | 1300 | 134 | 10.3% |
| `card_emb` | `opt_card`, `opt_target` | 1300 | 135 | 10.4% |
| `atk_emb` | `opt_attack` | 1600 | 57 | 3.6% |

Every other row ships at its random N(0, 1) initialisation.

## Validity controls

| control | result | reads |
|---|---|---|
| `--mode copy` round-trip | all tensors identical, keys equal | the `dict(np.load) -> np.savez` path introduces nothing |
| permuted net vs `random` | 0.867 [0.803, 0.912], n=150 | still a coherent policy, **not** a dim-guard fallback to random-legal |
| `v5` vs `random` | 1.000 [0.975, 1.000], n=150 | reference |
| opponent-scoped perm, **mirror** | 0.550 [0.493, 0.605], n=300 | CI spans 0.500 — the permuted rows are genuinely never looked up when both decks are our 19 |

## Arm 1 — global identity ablation (gate)

All four tables, `perm_seen`, mirror, direct head-to-head:

```
bc:v5  vs  bc:perm_seen      0.997 [0.981, 0.999]   W299/D0/L1   n=300
```

The identity channel is not decoration. Destroying it while leaving every
dense mechanical feature intact costs essentially every game.

This over-states the *live* problem, though: it scrambles our own 19 cards
too, and our deck is 19/19 in vocabulary with every row heavily trained.

## Arm 2 — opponent-side identity only

`--exclude-deck grimmsnarl --tables slot_emb,bag_emb,card_emb` holds our own
card ids fixed and scrambles only cards we can see but do not own. `atk_emb`
is excluded because `opt_attack` only ever carries *our* active's attack, so
permuting it would be a self-inflicted wound, not an opponent-identity effect.

| opponent | opponent **Pokémon** in vocabulary | `v5` | opponent-identity scrambled | Δ |
|---|---|---|---|---|
| `rule:crustle` | **4 / 4** | 0.838 [0.792, 0.876] | 0.587 [0.530, 0.641] | **−0.251**, CIs disjoint |
| `rule:v10,noS` | **0 / 6** | 0.625 [0.569, 0.678] | 0.607 [0.550, 0.660] | **−0.018**, CIs overlap |

n=300 per cell, seat-swapped.

## Verdict

Knowing *which* Pokémon the opponent has on board is worth roughly a quarter
of the win rate against Crustle (~−220 Elo uncompressed).

Against Mega Lucario that knowledge is **already absent**: all six of its
Pokémon — Makuhita, Hariyama, Lunatone, Solrock, Riolu, Mega Lucario ex —
are out of vocabulary, so their `slot_emb` rows are untrained random vectors
at inference. Scrambling the remainder (energy and generic trainers) costs
0.018, because there was nothing left to destroy. We do not play that matchup
blind by accident of tuning; we play it blind by construction.

Note both scrambled arms land at ~0.59–0.61 regardless of opponent. That is
what "no opponent read" looks like: generic-good Pokémon play, winning on raw
deck strength alone.

## What this does and does not license

Licensed: the identity channel matters, and the out-of-vocabulary decks are a
real, measured deficit rather than a theoretical one.

**Not** licensed: any claim that a fix recovers the 0.251. Permutation
measures the value of *correct* identity against *scrambled* identity. An
unseen card is not scrambled — it is a fixed random vector that at least
stays consistent within a game. The recoverable amount is bounded above by
this number, not equal to it.

Also unlicensed: attributing the known Mega Lucario weakness to this alone.
It is consistent with it (v5 scores 0.625 there against 0.838 vs Crustle),
but the corpus contains no Lucario games either, and "never trained on the
matchup" is a sufficient explanation on its own. Untangling the two needs an
intervention, not another ablation.

## Commands

```powershell
python -X utf8 scripts/p53_emb_vocab.py
python -X utf8 scripts/p54_emb_ablate.py --mode perm_seen
python -X utf8 scripts/p54_emb_ablate.py --mode perm_seen --exclude-deck grimmsnarl `
    --tables slot_emb,bag_emb,card_emb --out out/emb/v5__oppcards.npz
python -X utf8 scripts/arena.py play "bc:v5,net=out/policy_v5.npz" `
    "bc:oppcards,net=out/emb/v5__oppcards.npz" --matches 150 `
    --deck-a grimmsnarl --deck-b grimmsnarl
```

Archives: `out/arena/p54_*.jsonl`.
