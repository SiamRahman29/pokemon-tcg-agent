"""Policy-only agent: pure behavioral clone, near-instant decisions."""
from __future__ import annotations

import sys
import traceback

from . import policynet


class PolicyAgent:
    def __init__(self, decklist: list[int]):
        self.decklist = list(decklist)

    def __call__(self, obs: dict) -> list[int]:
        try:
            if obs.get("select") is None:
                return list(self.decklist)
            sel = obs["select"]
            n = len(sel.get("option") or [])
            mn = sel.get("minCount", 0)
            mx = sel.get("maxCount", 0)
            if n == 0 or mx == 0:
                return []
            if mn == mx == n:
                return list(range(n))
            net = policynet.get()
            if net is None:
                return list(range(mn))
            return net.choose(obs)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            try:
                return list(range((obs.get("select") or {}).get("minCount", 0)))
            except Exception:
                return []
