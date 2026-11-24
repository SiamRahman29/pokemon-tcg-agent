"""NumPy-only Transformer inference: no torch, so it runs in Kaggle's episode runner.

Adapted from ptcg-population-rl (Apache-2.0, commit 35707d6),
`mega_lucario/transformer/numpy_policy.py`. The forward pass and the greedy
multi-selection loop are upstream's. Changed: weights load from the `.npz`
that `scripts/tx_export.py` writes (named float32 tensors plus a JSON header
holding the contract and deck profile) instead of the C++ `.bin`, and the
features come from a `tx.features.Encoder`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from tx.contract import SCORE_TIE_TOLERANCE
from tx.features import WIDTH, Encoder

DIM, HEADS, LAYERS, FF = 128, 4, 6, 256
HEAD_DIM = DIM // HEADS
EPS = 1e-5


def load_npz(path: Path | str) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
    """Named float32 tensors and the header dict from an exported `.npz`."""
    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["__header__"]))
        tensors = {k: z[k].astype(np.float32) for k in z.files if k != "__header__"}
    if header.get("format") != "scio-tx-npz-v1":
        raise ValueError("not an exported Transformer .npz")
    for name, a in tensors.items():
        if not np.isfinite(a).all():
            raise ValueError(f"Non-finite weights: {name}")
    return tensors, header


def _erf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Abramowitz-Stegun 7.1.26 error function (|error| < 1.5e-7); numpy has no erf."""
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return sign * (1.0 - poly * np.exp(-a * a))


def _gelu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Exact (erf) GELU, matching torch's default `F.gelu`."""
    x64 = x.astype(np.float64)
    return (0.5 * x64 * (1.0 + _erf(x64 / math.sqrt(2.0)))).astype(np.float32)


class NumpyTransformer:
    """Prefix-independent encoder plus pointer policy and 51-bin value head."""

    def __init__(self, path: Path | str, encoder: Encoder | None = None) -> None:
        """Load an exported `.npz`; check it against `encoder` when given."""
        self.w, self.header = load_npz(path)
        if encoder is not None and self.header["contract_id"] != encoder.contract_id:
            raise ValueError("model .npz contract does not match this encoder/deck")
        self.support = np.linspace(-1, 1, 51, dtype=np.float32)

    def _linear(self, name: str, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply `nn.Linear` weights stored under `name`."""
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def _norm(self, name: str, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply `nn.LayerNorm` weights stored under `name`."""
        mean = x.mean(-1, keepdims=True)
        var = ((x - mean) ** 2).mean(-1, keepdims=True)
        return (x - mean) / np.sqrt(var + EPS) * self.w[name + ".weight"] + self.w[name + ".bias"]

    def _stem(self, name: str, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Two-layer token projection followed by LayerNorm."""
        return self._norm(
            name + ".norm",
            self._linear(name + ".linear2", _gelu(self._linear(name + ".linear1", x))),
        )

    def encode(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the final-normed hidden states; independent of the selection flags."""
        if x.ndim != 2 or x.shape[1] != WIDTH or not 20 <= len(x) <= 120:
            raise ValueError("Transformer supports 20-120 tokens of width 198")
        valid = x[:, 0] != 0
        kind = x[:, 1].astype(np.int64)
        table = self.w["deck_embedding.weight"]

        def embedding(slot: NDArray[np.float32]) -> NDArray[np.float32]:
            """Embed known deck slots and zero unknown cards."""
            index = np.clip(slot, 0, None).astype(np.int64)
            return table[index] * (slot >= 0)[:, None]

        identity, extra = embedding(x[:, 4]), embedding(x[:, 5])
        counts, evolution = x[:, 102:134] @ table, x[:, 134:166] @ table
        generic, result = x[:, 6:102], x[:, 166:198]
        h = (
            self._stem("global_stem", np.concatenate([generic, identity, extra], -1))
            * (kind == 0)[:, None]
        )
        h = (
            h
            + self._stem("board_stem", np.concatenate([generic, identity, counts, evolution], -1))
            * (kind == 1)[:, None]
        )
        h = (
            h
            + self._stem("zone_stem", np.concatenate([generic[:, :48], counts], -1))
            * (kind == 2)[:, None]
        )
        option = self._stem("option_stem", np.concatenate([generic, identity, result], -1))
        h = h + (option + self._linear("result_delta", result)) * (kind == 3)[:, None]
        h = h + self.w["stop"] * (kind == 4)[:, None] + self.w["token_type.weight"][kind]
        h = h.astype(np.float32)
        n = len(x)
        for i in range(LAYERS):
            p = f"blocks.{i}"
            qkv = self._linear(p + ".qkv", self._norm(p + ".norm1", h))
            q, k, v = qkv.reshape(n, 3, HEADS, HEAD_DIM).transpose(1, 2, 0, 3)
            scores = (q @ k.transpose(0, 2, 1)) / np.float32(math.sqrt(HEAD_DIM))
            scores = np.where(valid[None, None, :], scores, -np.inf)
            scores = np.exp(scores - scores.max(-1, keepdims=True))
            attention = scores / scores.sum(-1, keepdims=True)
            h = h + self._linear(p + ".out", (attention @ v).transpose(1, 0, 2).reshape(n, DIM))
            h = h + self._linear(
                p + ".ff2", _gelu(self._linear(p + ".ff1", self._norm(p + ".norm2", h)))
            )
        return self._norm("norm", h).astype(np.float32)

    def evaluate(
        self, h: NDArray[np.float32], x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float]:
        """Return masked pointer logits and expected Value for tokens `x` encoded as `h`."""
        valid = x[:, 0] != 0
        selected = x[:, 3] * valid
        summed = (h * selected[:, None]).sum(0)
        step = min(int(selected.sum()), 5)
        query = self._linear("policy_query", h[0] + summed + self.w["step_embedding.weight"][step])
        logits = (self._linear("policy_key", h) * query).sum(-1) / np.float32(
            math.sqrt(DIM)
        ) + self._linear("policy_bias", h)[:, 0]
        logits = np.where(valid & (x[:, 2] != 0), logits, -np.inf).astype(np.float32)
        value_logits = self._linear("value2", _gelu(self._linear("value1", h[0])))
        weights = np.exp(value_logits - value_logits.max())
        value = float((weights / weights.sum() * self.support).sum())
        return logits, value


def decide(model: NumpyTransformer, obs: dict[str, Any], enc: Encoder) -> list[int]:
    """Greedy multi-selection, as upstream's C++ `decide`: pick the best legal
    candidate (first within the tie tolerance) until STOP or maxCount."""
    action: list[int] = []
    tokens = hidden = None
    while len(action) < obs["select"]["maxCount"]:
        options = enc.options_for(obs, action)
        if not options:
            raise ValueError("No legal Transformer candidate")
        chosen = 0
        if len(options) > 1:
            if tokens is None:
                tokens = enc.tokenize(obs)
                hidden = model.encode(tokens)
            current = enc.with_prefix(tokens, obs, action)
            logits, _ = model.evaluate(hidden, current)
            scores = [float(logits[i]) for i in enc.token_indices(obs, options)]
            maximum = max(scores)
            while scores[chosen] < maximum - SCORE_TIE_TOLERANCE:
                chosen += 1
        best = options[chosen]
        if best == -1:
            break
        action.append(best)
    return action


def scores_for(model: NumpyTransformer, obs: dict[str, Any], prefix: Sequence[int],
               enc: Encoder) -> tuple[list[int], list[float], float]:
    """Return legal options, their logits and Value after `prefix`; used by parity checks."""
    tokens = enc.tokenize(obs)
    logits, value = model.evaluate(model.encode(tokens), enc.with_prefix(tokens, obs, prefix))
    options = enc.options_for(obs, prefix)
    return options, [float(logits[i]) for i in enc.token_indices(obs, options)], value
