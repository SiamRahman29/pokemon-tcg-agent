"""Numpy-only inference for the trained value net (see scripts/train_value.py).

If agents/sa/value_net.npz is absent, `get()` returns None and callers fall
back to the handcrafted eval.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .features import featurize

# `SA_VNET_PATH` mirrors `policynet.SA_PNET_PATH`. It exists so an A/B can pin
# WHICH value net played (rule 20: the identity a result is filed under must
# contain everything that can change the result) instead of silently using
# whatever `value_net.npz` happens to be on disk -- the exact defect rule 19
# describes, and `_net_fp` in arena.py is what records the bytes.
_PATH = Path(os.environ.get("SA_VNET_PATH")
             or Path(__file__).resolve().parent / "value_net.npz")
_BAGS = ("my_hand", "my_discard", "opp_discard")

_net = None
_tried = False


def load(path: str | Path) -> "Net | None":
    """Load a specific value net, bypassing the module-level singleton.

    Two `ValueLookahead` instances in one process (a head-to-head A/B, rule 4)
    must be able to hold different value nets, which a singleton cannot do.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        net = Net(np.load(p))
    except Exception:
        return None
    from .features import DENSE_DIM
    expect = (DENSE_DIM + 12 * net.slot_emb.shape[1]
              + 3 * net.bag_emb.shape[1])
    return net if net.w1.shape[1] == expect else None


class Net:
    def __init__(self, z):
        self.slot_emb = z["slot_emb"]
        self.bag_emb = z["bag_emb"]
        self.w1 = z["w1"]
        self.b1 = z["b1"]
        self.w2 = z["w2"]
        self.b2 = z["b2"]
        self.w3 = z["w3"]
        self.b3 = z["b3"]

    def win_prob(self, state: dict, me: int) -> float:
        dense, bags = featurize(state, me)
        parts = [dense, self.slot_emb[bags["slots"]].reshape(-1)]
        for name in _BAGS:
            b = bags[name]
            parts.append(self.bag_emb[b].mean(axis=0) if len(b)
                         else np.zeros(self.bag_emb.shape[1],
                                       dtype=np.float32))
        x = np.concatenate(parts)
        h = np.maximum(self.w1 @ x + self.b1, 0.0)
        h = np.maximum(self.w2 @ h + self.b2, 0.0)
        logit = float(self.w3 @ h + self.b3)
        return 1.0 / (1.0 + np.exp(-logit))


def get() -> Net | None:
    global _net, _tried
    if not _tried:
        _tried = True
        if _PATH.exists():
            try:
                net = Net(np.load(_PATH))
                from .features import DENSE_DIM
                expect = (DENSE_DIM + 12 * net.slot_emb.shape[1]
                          + 3 * net.bag_emb.shape[1])
                if net.w1.shape[1] == expect:
                    _net = net
            except Exception:
                _net = None
    return _net
