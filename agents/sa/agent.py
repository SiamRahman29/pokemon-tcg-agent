"""The search agent: Kaggle contract callable."""
from __future__ import annotations

import sys
import traceback

from .planner import Planner, candidate_combos, ROOT_CAP
from .timemgr import TimeManager
from .tracker import Tracker


class SearchAgent:
    def __init__(self, decklist: list[int], no_pnet: bool | None = None,
                 no_vnet: bool | None = None, max_worlds: int | None = None,
                 rollout: bool = False, main_cap: float | None = None,
                 minor_cap: float | None = None,
                 prior_bonus: float | None = None):
        self.decklist = list(decklist)
        self.tracker = Tracker()
        self.planner = Planner(decklist, no_pnet=no_pnet, no_vnet=no_vnet,
                               max_worlds=max_worlds, rollout=rollout,
                               prior_bonus=prior_bonus)
        self.tm = TimeManager(main_cap=main_cap, minor_cap=minor_cap)

    def __call__(self, obs: dict) -> list[int]:
        try:
            if obs.get("select") is None:
                self.tracker.reset()
                return list(self.decklist)
            self.tracker.maybe_reset(obs)
            self.tracker.update(obs)
            sel = obs["select"]
            combos = candidate_combos(sel, ROOT_CAP, self.planner.rng)
            if len(combos) == 1:
                return combos[0]
            budget = self.tm.budget(obs, len(combos))
            return self.planner.decide(obs, self.tracker.known_opp_hand(),
                                       budget, combos)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            try:
                # minCount indices are always a legal reply
                return list(range((obs.get("select") or {}).get("minCount", 0)))
            except Exception:
                return []
