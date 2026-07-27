"""Prove the engine loads and agents play a full game. Run this first.

    python scripts/sdk_smoke.py

Loads the cg engine, prints card/attack counts, then plays the `iono` sample
rule-based agent against a random agent for a few games via the local harness.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg import config          # noqa: E402
from ptcg.env import harness, sdk  # noqa: E402


def random_agent(deck: list[int]):
    def agent(obs_dict: dict) -> list[int]:
        obs = sdk.api().to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck)
        sel = obs.select
        k = min(max(sel.minCount, 1 if sel.maxCount else 0), sel.maxCount,
                len(sel.option))
        return random.sample(range(len(sel.option)), k)

    return agent


def _deck() -> list[int]:
    """The iono rule agent's own deck (its card counting assumes this list)."""
    from decks import iono

    return [cid for cid, cnt in iono.DECKLIST.items() for _ in range(cnt)]


def main() -> int:
    cg = sdk.load()  # noqa: F841
    print("engine loaded.")
    print("cards:", len(sdk.card_db()), "| attacks:", len(sdk.attack_db()))

    from agentkit.rulebased import make_rule_agent

    deck = _deck()
    res = harness.evaluate(make_rule_agent("iono", deck), random_agent(deck),
                           deck, games=20)
    print("rule:iono vs random (20 games):", res)
    if res["win_rate"] <= 0.5:
        print("WARN: the rule agent is not beating random -- worth investigating.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
