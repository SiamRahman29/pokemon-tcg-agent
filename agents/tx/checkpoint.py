"""Load a distilled checkpoint after checking it matches the expected contract.

Adapted from ptcg-population-rl (Apache-2.0, commit 35707d6),
`mega_lucario/transformer/checkpoint.py`: binds to a deck profile and
contract id instead of Mega Lucario's deck counts and feature names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tx.model import ARCHITECTURE, ObservationTransformer


def load_checkpoint(path: Path, contract_id: str | None = None,
                    profile_id: str | None = None) -> tuple[ObservationTransformer, dict[str, Any]]:
    """Validate contracts and finite weights, then return the model in eval mode."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != "scio-tx-checkpoint-v1"
            or payload["architecture"] != ARCHITECTURE
            or (contract_id is not None and payload["contract_id"] != contract_id)
            or (profile_id is not None and payload["profile_id"] != profile_id)):
        raise ValueError("incompatible Transformer checkpoint")
    model = ObservationTransformer(payload["contract"]["width"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("non-finite checkpoint weights")
    return model.eval(), payload
