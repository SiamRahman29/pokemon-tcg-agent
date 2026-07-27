"""Numpy-only inference for the trained value net (see scripts/train_value.py).

If agents/sa/value_net.npz is absent, `get()` returns None and callers fall
back to the handcrafted eval.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import featurize

_PATH = Path(__file__).resolve().parent / "value_net.npz"
_BAGS = ("my_hand", "my_discard", "opp_discard")

_net = None
_tried = False


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
            _net = Net(np.load(_PATH))
    return _net
