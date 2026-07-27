"""Numpy inference for the cloned policy (see scripts/train_policy.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import featurize
from .optfeat import option_features

_PATH = Path(__file__).resolve().parent / "policy_net.npz"
_BAGS = ("my_hand", "my_discard", "opp_discard")
SEL_DENSE = 14

_net = None
_tried = False


def _sel_features(sel: dict) -> np.ndarray:
    v = np.zeros(SEL_DENSE, dtype=np.float32)
    t = sel.get("type") or 0
    if t < 11:
        v[t] = 1.0
    v[11] = sel.get("minCount", 0) / 5.0
    v[12] = sel.get("maxCount", 0) / 5.0
    v[13] = (sel.get("context") or 0) / 50.0
    return v


class Net:
    def __init__(self, z):
        self.slot_emb = z["slot_emb"]
        self.bag_emb = z["bag_emb"]
        self.card_emb = z["card_emb"]
        self.atk_emb = z["atk_emb"]
        self.ws = z["ws"]
        self.bs = z["bs"]
        self.w1 = z["w1"]
        self.b1 = z["b1"]
        self.w2 = z["w2"]
        self.b2 = z["b2"]
        self.count_frac = z["count_frac"] if "count_frac" in z else None

    def scores(self, obs: dict) -> np.ndarray:
        """Logit per option of obs['select']."""
        state = obs["current"]
        sel = obs["select"]
        me = state["yourIndex"]
        dense, bags = featurize(state, me)
        parts = [dense, self.slot_emb[bags["slots"]].reshape(-1)]
        for name in _BAGS:
            b = bags[name]
            parts.append(self.bag_emb[b].mean(axis=0) if len(b)
                         else np.zeros(self.bag_emb.shape[1],
                                       dtype=np.float32))
        parts.append(_sel_features(sel))
        x = np.concatenate(parts)
        srepr = np.maximum(self.ws @ x + self.bs, 0.0)

        opts = sel.get("option") or []
        n = len(opts)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        feats = np.empty((n, self.w1.shape[1]), dtype=np.float32)
        sw = len(srepr)
        for i, o in enumerate(opts):
            od, cid, aid, tid = option_features(obs, o)
            feats[i, :sw] = srepr
            b = sw + len(od)
            feats[i, sw:b] = od
            feats[i, b:b + 16] = self.card_emb[cid]
            feats[i, b + 16:b + 32] = self.atk_emb[aid]
            feats[i, b + 32:] = self.card_emb[tid]
        h = np.maximum(feats @ self.w1.T + self.b1, 0.0)
        return (h @ self.w2.T + self.b2).reshape(-1)

    def choose(self, obs: dict) -> list[int]:
        """Rank options by logit; how MANY to take comes from the data-derived
        per-(selectType, context) count-fraction table."""
        sel = obs["select"]
        mn = sel.get("minCount", 0)
        mx = sel.get("maxCount", 0)
        sc = self.scores(obs)
        order = list(np.argsort(-sc))
        k = mx
        if mx > mn:
            frac = 1.0
            if self.count_frac is not None:
                t = min(sel.get("type") or 0, 10)
                ctx = min(sel.get("context") or 0, 63)
                frac = float(self.count_frac[t, ctx])
            k = mn + int(round(frac * (mx - mn)))
            k = max(mn, min(k, mx))
        return [int(i) for i in order[:k]]


def get() -> Net | None:
    global _net, _tried
    if not _tried:
        _tried = True
        if _PATH.exists():
            try:
                net = Net(np.load(_PATH))
                from .features import DENSE_DIM
                from .optfeat import OPT_DENSE
                expect_state = (DENSE_DIM + 12 * net.slot_emb.shape[1]
                                + 3 * net.bag_emb.shape[1] + SEL_DENSE)
                expect_head = (net.ws.shape[0] + OPT_DENSE
                               + 2 * net.card_emb.shape[1]
                               + net.atk_emb.shape[1])
                if (net.ws.shape[1] == expect_state
                        and net.w1.shape[1] == expect_head):
                    _net = net
            except Exception:
                _net = None
    return _net
