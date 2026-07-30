"""Numpy inference for the cloned policy (see scripts/train_policy.py)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .features import featurize
from .optfeat import option_features

# SA_PNET_PATH lets an arena run score a candidate net without overwriting the
# shipped one. Kaggle sets no env vars, so there it is always the bundled npz.
_PATH = Path(os.environ.get("SA_PNET_PATH")
             or Path(__file__).resolve().parent / "policy_net.npz")
# how many options to take on a variable-count select; see Net.choose
COUNT_MODE = os.environ.get("SA_COUNT_MODE", "table")
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
        # Layers are stored generically (`sfc{i}_w` / `head{i}_w`) so the net
        # can be made deeper without touching this file. Nets exported before
        # that change used fixed ws/w1/w2 names -- still loadable.
        if "n_sfc" in z:
            self.state_layers = [(z[f"sfc{i}_w"], z[f"sfc{i}_b"])
                                 for i in range(int(z["n_sfc"][0]))]
            self.head_layers = [(z[f"head{i}_w"], z[f"head{i}_b"])
                                for i in range(int(z["n_head"][0]))]
        else:
            self.state_layers = [(z["ws"], z["bs"])]
            self.head_layers = [(z["w1"], z["b1"]), (z["w2"], z["b2"])]
        self.count_frac = z["count_frac"] if "count_frac" in z else None

    @property
    def state_in(self) -> int:
        return self.state_layers[0][0].shape[1]

    @property
    def state_out(self) -> int:
        return self.state_layers[-1][0].shape[0]

    @property
    def head_in(self) -> int:
        return self.head_layers[0][0].shape[1]

    @property
    def opt_in(self) -> int:
        """How many per-option dense features THIS net was trained on.

        Derived rather than read from `optfeat.OPT_DENSE`, because a v2 net
        (25) and a v3 net (37) have to be able to run in the same process for a
        head-to-head A/B across the feature change (HANDOFF rule 4). The v3 block
        is appended, so slicing to this width gives a v2 net byte-identical input
        to what it was trained on."""
        return (self.head_in - self.state_out
                - 2 * self.card_emb.shape[1] - self.atk_emb.shape[1])

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
        srepr = x
        for w, b in self.state_layers:      # every state layer is relu'd
            srepr = np.maximum(w @ srepr + b, 0.0)

        opts = sel.get("option") or []
        n = len(opts)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        feats = np.empty((n, self.head_in), dtype=np.float32)
        sw = len(srepr)
        ow = self.opt_in
        for i, o in enumerate(opts):
            od, cid, aid, tid = option_features(obs, o)
            feats[i, :sw] = srepr
            b = sw + ow
            # Slice to the width this net was trained at -- the v3 target block
            # is appended, so a v2 net simply does not see it.
            feats[i, sw:b] = od[:ow]
            feats[i, b:b + 16] = self.card_emb[cid]
            feats[i, b + 16:b + 32] = self.atk_emb[aid]
            feats[i, b + 32:] = self.card_emb[tid]
        h = feats
        for j, (w, b) in enumerate(self.head_layers):
            h = h @ w.T + b
            if j < len(self.head_layers) - 1:   # last layer is the raw logit
                h = np.maximum(h, 0.0)
        return h.reshape(-1)

    def choose(self, obs: dict) -> list[int]:
        """Rank options by logit; how MANY to take is the harder half.

        `table` (default): a data-derived per-(selectType, context) mean count
        fraction -- one number for the whole bucket, so it is wrong on every
        select whose true count is bimodal.
        `expect`: sum the per-option sigmoids, i.e. the model's own expected
        number of chosen options. Only meaningful for a net trained with a
        pointwise (BCE) term -- a pure listwise net's logits are not calibrated
        probabilities, only a valid ranking.
        """
        sel = obs["select"]
        mn = sel.get("minCount", 0)
        mx = sel.get("maxCount", 0)
        sc = self.scores(obs)
        order = list(np.argsort(-sc))
        k = mx
        if mx > mn:
            if COUNT_MODE == "expect":
                probs = 1.0 / (1.0 + np.exp(-np.clip(sc, -30.0, 30.0)))
                k = int(round(float(probs.sum())))
            else:
                frac = 1.0
                if self.count_frac is not None:
                    t = min(sel.get("type") or 0, 10)
                    ctx = min(sel.get("context") or 0, 63)
                    frac = float(self.count_frac[t, ctx])
                k = mn + int(round(frac * (mx - mn)))
            k = max(mn, min(k, mx))
        return [int(i) for i in order[:k]]


def load(path) -> Net | None:
    """Load a specific npz, returning None unless it matches the CURRENT
    feature dims. The guard is what stops a stale net from being used
    silently after a feature change -- never remove it."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        net = Net(np.load(path))
        from .features import DENSE_DIM
        from .optfeat import KNOWN_OPT_DENSE
        expect_state = (DENSE_DIM + 12 * net.slot_emb.shape[1]
                        + 3 * net.bag_emb.shape[1] + SEL_DENSE)
        # The option width is now per-net (see Net.opt_in), so the guard asks
        # whether it is a width we still know how to feed rather than whether it
        # equals today's OPT_DENSE. An unrecognised width is a stale net.
        if net.state_in == expect_state and net.opt_in in KNOWN_OPT_DENSE:
            return net
    except Exception:
        pass
    return None


def get() -> Net | None:
    global _net, _tried
    if not _tried:
        _tried = True
        _net = load(_PATH)
    return _net
