"""Per-table embedding vocabulary census over the training corpus.

Each of the four embedding tables is indexed by a different set of columns in
`artifacts/pds_v4`, so "which rows ever received a gradient" is a per-table
question, not a global one:

    slot_emb  <- slots, xslots        (board slots + stadium/effect ids)
    bag_emb   <- bag_*_flat           (hand / own discard / opp discard)
    card_emb  <- opt_card, opt_target (an option's card and its target)
    atk_emb   <- opt_attack

Rows outside these sets are still exported into the shipped npz at their
random N(0, 1) initialisation. Writes `out/emb/vocab.json` so the ablation
script and any later analysis share one definition of "seen".

    python -X utf8 scripts/p53_emb_vocab.py [--pds artifacts/pds_v4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

# table -> npz keys that index it
SOURCES = {
    "slot_emb": ("slots", "xslots"),
    "bag_emb": ("bag_my_hand_flat", "bag_my_discard_flat",
                "bag_opp_discard_flat"),
    "card_emb": ("opt_card", "opt_target"),
    "atk_emb": ("opt_attack",),
}


def census(pds: Path) -> dict:
    shards = sorted(pds.rglob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {pds}")
    counts = {t: {} for t in SOURCES}
    rows = 0
    for sh in shards:
        z = np.load(sh)
        keys = set(z.keys())
        rows += int(z["dense"].shape[0]) if "dense" in keys else 0
        for table, srcs in SOURCES.items():
            for k in srcs:
                if k not in keys:
                    continue
                ids, n = np.unique(z[k].ravel(), return_counts=True)
                for i, c in zip(ids.tolist(), n.tolist()):
                    counts[table][i] = counts[table].get(i, 0) + c
    return {
        "pds": str(pds),
        "shards": len(shards),
        "rows": rows,
        "tables": {t: {str(k): v for k, v in sorted(c.items())}
                   for t, c in counts.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pds", default="artifacts/pds_v4")
    ap.add_argument("--out", default="out/emb/vocab.json")
    args = ap.parse_args()

    rep = census(ROOT / args.pds)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")

    print(f"corpus {rep['pds']}  shards={rep['shards']}  rows={rep['rows']:,}")
    for t, c in rep["tables"].items():
        ids = sorted(int(k) for k in c)
        tot = sum(c.values())
        print(f"  {t:9s} distinct={len(ids):5d}  lookups={tot:12,d}  "
              f"max_id={max(ids) if ids else -1}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
