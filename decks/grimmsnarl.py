"""Marnie's Grimmsnarl ex / Munkidori -- the dominant top-LB deck.

Mined from 2026-07-26 top episodes (exact list seen 290x; flg, Dries, Luca...).
"""
from __future__ import annotations

try:
    from .base import Deck
except ImportError:
    from base import Deck  # noqa: F401

DECKLIST: dict[int, int] = {
    7: 10,     # Basic {D} Energy
    112: 4,    # Munkidori
    646: 4,    # Marnie's Impidimp
    1086: 4,   # Buddy-Buddy Poffin
    1152: 4,   # Poke Pad
    1219: 4,   # Team Rocket's Petrel
    1227: 4,   # Lillie's Determination
    1259: 4,   # Spikemuth Gym
    647: 3,    # Marnie's Morgrem
    648: 3,    # Marnie's Grimmsnarl ex
    1079: 3,   # Rare Candy
    1097: 3,   # Night Stretcher
    104: 2,    # Froslass
    860: 2,    # Snorunt
    1182: 2,   # Boss's Orders
    1080: 1,   # Unfair Stamp
    1122: 1,   # Pokegear 3.0
    1137: 1,   # Tool Scrapper
    1231: 1,   # Dawn
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
