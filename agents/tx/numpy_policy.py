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


_A = [np.float32(c) for c in (0.254829592, -0.284496736, 1.421413741, -1.453152027,
                                1.061405429)]


def _gelu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Exact (erf) GELU, matching torch's default `F.gelu`, in float32.

    erf is Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7); numpy has no erf.
    Float32 throughout (upstream: float64), which moves logits by ~1e-6.
    """
    a = np.abs(x) * np.float32(1 / math.sqrt(2.0))
    t = np.float32(1) / (np.float32(1) + np.float32(0.3275911) * a)
    poly = t * (_A[0] + t * (_A[1] + t * (_A[2] + t * (_A[3] + t * _A[4]))))
    erf = np.float32(1) - poly * np.exp(-a * a)
    return np.float32(0.5) * x * (np.float32(1) + np.copysign(erf, x))


class NumpyTransformer:
    """Prefix-independent encoder plus pointer policy and 51-bin value head."""

    def __init__(self, path: Path | str, encoder: Encoder | None = None) -> None:
        """Load an exported `.npz`; check it against `encoder` when given."""
        self.w, self.header = load_npz(path)
        if encoder is not None and self.header["contract_id"] != encoder.contract_id:
            raise ValueError("model .npz contract does not match this encoder/deck")
        self.support = np.linspace(-1, 1, 51, dtype=np.float32)
        # Contiguous transposes of every Linear weight, made once.
        self.wt = {k[: -len(".weight")]: np.ascontiguousarray(v.T) for k, v in self.w.items()
                   if k.endswith(".weight") and v.ndim == 2 and k[: -len(".weight")] + ".bias"
                   in self.w}

    def _linear(self, name: str, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply `nn.Linear` weights stored under `name`."""
        return x @ self.wt[name] + self.w[name + ".bias"]

    def _norm(self, name: str, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply `nn.LayerNorm` weights stored under `name`."""
        d = x - x.mean(-1, keepdims=True)
        var = np.einsum("ij,ij->i", d, d)[:, None] * np.float32(1 / x.shape[-1])
        d *= 1 / np.sqrt(var + np.float32(EPS))
        d *= self.w[name + ".weight"]
        d += self.w[name + ".bias"]
        return d

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

        # Each stem runs only on its own token kind (upstream ran all four
        # stems on every token and masked); row-wise, so the result is the same.
        generic, result = x[:, 6:102], x[:, 166:198]
        h = np.zeros((len(x), DIM), np.float32)
        r = np.flatnonzero(kind == 0)
        h[r] = self._stem("global_stem", np.concatenate(
            [generic[r], embedding(x[r, 4]), embedding(x[r, 5])], -1))
        r = np.flatnonzero(kind == 1)
        h[r] = self._stem("board_stem", np.concatenate(
            [generic[r], embedding(x[r, 4]), x[r, 102:134] @ table, x[r, 134:166] @ table], -1))
        r = np.flatnonzero(kind == 2)
        h[r] = self._stem("zone_stem", np.concatenate(
            [generic[r, :48], x[r, 102:134] @ table], -1))
        r = np.flatnonzero(kind == 3)
        h[r] = (self._stem("option_stem", np.concatenate(
            [generic[r], embedding(x[r, 4]), result[r]], -1))
                + self._linear("result_delta", result[r]))
        h[kind == 4] = self.w["stop"]
        h += self.w["token_type.weight"][kind]
        n = len(x)
        all_valid = bool(valid.all())
        for i in range(LAYERS):
            p = f"blocks.{i}"
            qkv = self._linear(p + ".qkv", self._norm(p + ".norm1", h))
            q, k, v = qkv.reshape(n, 3, HEADS, HEAD_DIM).transpose(1, 2, 0, 3)
            scores = (q @ k.transpose(0, 2, 1)) / np.float32(math.sqrt(HEAD_DIM))
            if not all_valid:
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
