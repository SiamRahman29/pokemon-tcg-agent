# data/

Pasted manually; everything here is gitignored (the engine is licensed and must
not be republished). Current contents:

```
data/
  EN_Card_Data.csv        card metadata (English) -- 1267 cards
  JP_Card_Data.csv        card metadata (Japanese, same content)
  Card_ID List_EN.pdf     card id reference (English)
  Card_ID List_JP.pdf     card id reference (Japanese)
  ptcg_engine/            the C++ engine SOURCE (reference only; not built here)
  sample_submission/
    sample_submission/
      main.py             the reference agent (defines the agent contract)
      deck.csv            a legal 60-card deck
      cg/                 the Python engine wrapper + compiled libs
        api.py            Observation/Option/State dataclasses + search API
        game.py          local battle harness (battle_start/select/finish)
        sim.py           ctypes bindings; loads cg.dll / libcg.so / .dylib
        cg.dll, libcg.so, libcg.dylib, libcg-arm64.so
```

The code finds `cg` automatically (`ptcg.config.find_sdk_dir` globs for
`cg/api.py`). The same card id in `EN_Card_Data.csv` is what the simulator and
your `deck.csv` use, and the engine also exposes it via `all_card_data()`.

`EN_Card_Data.csv` columns: Card ID, Card Name, Expansion, Collection No.,
Stage/Type, Rule, Category, Previous stage, HP, Type, Weakness, Resistance,
Retreat, Move Name, Cost, Damage, Effect Explanation.
