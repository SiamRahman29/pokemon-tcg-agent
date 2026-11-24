"""Arena/Kaggle agent that plays an exported Transformer `.npz` greedily."""

from __future__ import annotations

from typing import Any

from tx.deck import DeckProfile
from tx.features import Encoder
from tx.numpy_policy import NumpyTransformer, decide


class TxAgent:
    """Plays the deck its model was trained for; refuses any other deck."""

    def __init__(self, deck: list[int], model_path: str,
                 cards: dict[int, dict[str, Any]]):
        self.model = NumpyTransformer(model_path)
        p = self.model.header["profile"]
        profile = DeckProfile(
            name=p["name"], counts={int(k): v for k, v in p["counts"].items()},
            ids=tuple(sorted(int(k) for k in p["counts"])), attack_ids=tuple(p["attack_ids"]),
            attacks={int(k): (d, tuple(c)) for k, (d, c) in p["attacks"].items()},
            energy_type=p["energy_type"])
        profile.check_deck(deck)
        self.enc = Encoder(profile, cards)
        if self.enc.contract_id != self.model.header["contract_id"]:
            raise ValueError("model contract does not match this source tree")
        self.deck = list(deck)

    def __call__(self, obs: dict[str, Any]) -> list[int]:
        sel = obs.get("select")
        if sel is None:
            return list(self.deck)
        if not sel.get("option") or sel.get("maxCount", 0) == 0:
            return []
        return decide(self.model, obs, self.enc)
