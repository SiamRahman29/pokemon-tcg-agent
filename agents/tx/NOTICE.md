# Notice

Parts of this package are adapted from **ptcg-population-rl**
(https://github.com/ymgaq/ptcg-population-rl, commit `35707d6`), licensed under
the Apache License 2.0. A copy of that license is in `LICENSE` in this directory.

| File here | Upstream file | Changes |
|---|---|---|
| `contract.py` | `src/agents/mega_lucario/transformer/contract.py` | None |
| `model.py` | `src/agents/mega_lucario/transformer/model.py` | Import path only |
| `features.py` | `src/agents/mega_lucario/gbdt/features.py` and `src/agents/mega_lucario/transformer/features.py` | Deck-specific constants come from a `DeckProfile`. The Mega Lucario 12-column block is re-derived from the engine's attack data. Only the generic feature row is built. Column offsets are computed per deck. |
| `train.py` | `src/agents/mega_lucario/transformer/train.py` | Reads compact sparse shards and builds batches on demand. Checkpoints bind to a deck profile. Adds a time limit and in-epoch logging. Loss and optimisation are unchanged. |
| `checkpoint.py` | `src/agents/mega_lucario/transformer/checkpoint.py` | Validates a deck profile and contract id |
| `ppo.py` | `src/agents/mega_lucario/ppo/rollout.py` and `src/agents/mega_lucario/ppo/train.py` | Rollouts are sampled by the NumPy model in our arena and recorded sparsely at decision time. The behaviour audit checks a sample of decisions. The loss, GAE and KL calibration are unchanged. |

`deck.py`, the `scripts/tx_*.py` drivers and `scripts/ppo_session.py` / `scripts/ppo_chain.py` are new in this repository.
