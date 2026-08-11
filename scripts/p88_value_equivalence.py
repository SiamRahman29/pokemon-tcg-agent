#!/usr/bin/env python
"""Does `valuenet.Net` reproduce the TRAINED model on identical inputs?

🔴 This is the check that should have run BEFORE any arena game, and did not.
`train_value.py` builds its input in torch; `sa/valuenet.py` rebuilds it in
numpy. Two implementations of one function is exactly the situation HANDOFF
rule 18 says to resolve by computing the number a second way and reconciling
*before* writing a word -- and E20 instead spent 2,000 games on the assumption
that they agree.

One suspect is already known by inspection: for an EMPTY card bag the trainer
pads with row 0 and takes a mean over that one row (`bag_emb[0]`), while
`valuenet.Net` substitutes `zeros`. Those are different vectors unless row 0
happens to be zero.

    python -X utf8 scripts/p88_value_equivalence.py --vnet out/value_v1.npz
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "."):
    sys.path.insert(0, str(ROOT / sub))

from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa import valuenet as vnet  # noqa: E402

BAGS = ("my_hand", "my_discard", "opp_discard")


def shipped_forward(net, dense, slot_ids, bags):
    """Calls the SHIPPED scoring path, not a copy of it. `valuenet.Net.forward`
    is exactly what plays, so an equivalence result here is about the agent."""
    b = dict(bags)
    b["slots"] = slot_ids
    return float(net.forward(dense, b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vnet", default="out/value_v1.npz")
    ap.add_argument("--data", default="artifacts/rl_v5_t05/w0")
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    net = vnet.load(args.vnet)
    if net is None:
        sys.exit(f"dim guard rejected {args.vnet}")

    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_value import ValueNet, Data

    paths = sorted(Path(p) for p in glob.glob(f"{args.data}/shard_*.npz"))[:1]
    data = Data(paths)
    n = min(args.n, len(data))
    idx = np.arange(n)

    model = ValueNet()
    z = np.load(args.vnet)
    sd = model.state_dict()
    sd["slot_emb.weight"] = torch.from_numpy(z["slot_emb"])
    sd["bag_emb.weight"] = torch.from_numpy(z["bag_emb"])
    sd["fc1.weight"] = torch.from_numpy(z["w1"])
    sd["fc1.bias"] = torch.from_numpy(z["b1"])
    sd["fc2.weight"] = torch.from_numpy(z["w2"])
    sd["fc2.bias"] = torch.from_numpy(z["b2"])
    sd["fc3.weight"] = torch.from_numpy(z["w3"][None, :])
    sd["fc3.bias"] = torch.from_numpy(np.array([z["b3"]], dtype=np.float32))
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        dense, slots, bf, bo, _y = data.batch(idx, torch.device("cpu"))
        ref = torch.sigmoid(model(dense, slots, bf, bo)).numpy()

    got, n_empty = [], 0
    for i in range(n):
        bags = {}
        for b in BAGS:
            off, flat = data.bag_off[b], data.bag_flat[b]
            bags[b] = flat[off[i]:off[i + 1]]
        if any(len(bags[b]) == 0 for b in BAGS):
            n_empty += 1
        got.append(shipped_forward(net, data.dense[i], data.slots[i], bags))
    got = np.array(got)

    print(f"rows compared              {n}")
    print(f"rows with >=1 empty bag    {n_empty} ({n_empty / n:.1%})")
    d = np.abs(got - ref)
    print("\nsa/valuenet.py Net.forward  vs  the trained torch model")
    print(f"  max |diff|   {d.max():.8f}")
    print(f"  mean |diff|  {d.mean():.8f}")
    print(f"  corr         {np.corrcoef(got, ref)[0, 1]:.8f}")
    ok = d.max() < 1e-4
    print("  " + ("✅ EQUIVALENT -- the agent scores what was trained"
                  if ok else "🔴 NOT EQUIVALENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
