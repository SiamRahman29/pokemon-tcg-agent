"""Rule-14 sizing gate for the candidate card-attribute features.

E6 showed that identifying the opponent's Pokemon is worth ~0.25 win rate
where the corpus supports it, and worth ~0 against Mega Lucario because all
six of its Pokemon are out of vocabulary. The proposed repair is to stop
relying on a per-card embedding row for identity and feed the card's
*attributes* instead, which transfer to cards the corpus never contained.

Before building ~290 new dense columns, measure whether they carry signal at
a decision. A feature whose modal value covers ~everything is dead weight,
and this repo has already paid for five such columns (EVIDENCE 8ab: the v4
leftovers scored -22 Elo against having no block at all).

Reported per candidate, at the two decision-relevant lookups (opponent active
slot, and an option's target):

    distinct   values actually observed
    modal      share held by the most common value
    H/Hmax     normalised entropy

Slots are ordered `my active, my bench x5, opp active, opp bench x5`, so
column 6 is the opponent's active.

    python -X utf8 scripts/p55_attr_sizing.py
"""
from __future__ import annotations

import argparse
import collections
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

MY_ACTIVE, OPP_ACTIVE = 0, 6

# attribute -> how to read it off a card dict (None-safe, 0 == "not applicable")
ATTRS = {
    "cardType": lambda c: c.get("cardType"),
    "energyType": lambda c: c.get("energyType"),
    "weakness": lambda c: c.get("weakness") or 0,
    "resistance": lambda c: c.get("resistance") or 0,
    "pokemonType": lambda c: c.get("pokemonType"),
    "evolutionType": lambda c: c.get("evolutionType"),
    "aceSpec": lambda c: int(bool(c.get("aceSpec"))),
    "hasAbility": lambda c: int(bool(c.get("skills"))),
}


def stats(counter: collections.Counter) -> tuple[int, float, float]:
    tot = sum(counter.values())
    if not tot:
        return 0, 1.0, 0.0
    modal = max(counter.values()) / tot
    h = -sum((n / tot) * math.log2(n / tot) for n in counter.values() if n)
    hmax = math.log2(len(counter)) if len(counter) > 1 else 1.0
    return len(counter), modal, h / hmax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pds", default="artifacts/pds_v4")
    args = ap.parse_args()

    from ptcg.env import sdk
    sdk.load()
    from sa import cards as cdb
    card = cdb.card

    shards = sorted((ROOT / args.pds).rglob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {args.pds}")

    counts = {("opp_active", a): collections.Counter() for a in ATTRS}
    counts.update({("opt_target", a): collections.Counter() for a in ATTRS})
    # the derived cross-term the whole idea rests on: do we hit the thing in
    # front of us for weakness?
    weak_hit = collections.Counter()
    rows = 0

    for sh in shards:
        z = np.load(sh)
        slots = z["slots"]
        rows += slots.shape[0]
        opp = slots[:, OPP_ACTIVE]
        mine = slots[:, MY_ACTIVE]
        for a, fn in ATTRS.items():
            ids, n = np.unique(opp, return_counts=True)
            for i, c in zip(ids.tolist(), n.tolist()):
                counts[("opp_active", a)][fn(card(i))] += c
        tgt = z["opt_target"].ravel()
        tgt = tgt[tgt > 0]
        for a, fn in ATTRS.items():
            ids, n = np.unique(tgt, return_counts=True)
            for i, c in zip(ids.tolist(), n.tolist()):
                counts[("opt_target", a)][fn(card(i))] += c
        for m, o in zip(mine.tolist(), opp.tolist()):
            if not m or not o:
                weak_hit["no board"] += 1
                continue
            w = card(o).get("weakness")
            weak_hit[bool(w) and w == card(m).get("energyType")] += 1

    print(f"corpus {args.pds}  rows={rows:,}\n")
    for site in ("opp_active", "opt_target"):
        print(f"=== {site} ===")
        print(f"  {'attribute':14s} {'distinct':>8s} {'modal':>8s} {'H/Hmax':>8s}")
        for a in ATTRS:
            d, modal, hn = stats(counts[(site, a)])
            flag = "  <- dead" if (d < 2 or modal > 0.97) else ""
            print(f"  {a:14s} {d:8d} {modal:8.3f} {hn:8.3f}{flag}")
        print()

    tot = sum(weak_hit.values())
    print("=== derived: opponent active is weak to OUR active's type ===")
    for k in (True, False, "no board"):
        n = weak_hit.get(k, 0)
        print(f"  {str(k):9s} {n:9,d}  {n / tot:6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
