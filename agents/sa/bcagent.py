"""Policy-only agent: pure behavioral clone, near-instant decisions."""
from __future__ import annotations

import sys
import traceback

from . import policynet, targeting


class PolicyAgent:
    def __init__(self, decklist: list[int], net_path: str | None = None,
                 chip_targeting: bool = True):
        self.decklist = list(decklist)
        # An explicit net lets two candidate policies play each other inside
        # ONE arena process. Comparing them via a third opponent instead needs
        # ~2x the games for the same resolution, and the module-level
        # policynet.get() singleton cannot hold two nets at once.
        self.net = policynet.load(net_path) if net_path else None
        # The net cannot see option HP at all (see targeting.py), so it aims
        # chip damage at chance. Per-instance so the two sides of an A/B can
        # differ inside one process.
        self.chip_targeting = chip_targeting

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
            if self.chip_targeting:
                order = targeting.chip_target(obs)
                if order is not None:
                    return order[:max(min(mn, mx, n), 1)]
            net = self.net or policynet.get()
            if net is None:
                return list(range(mn))
            return net.choose(obs)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            try:
                return list(range((obs.get("select") or {}).get("minCount", 0)))
            except Exception:
                return []
