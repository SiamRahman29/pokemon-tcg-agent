"""Crispin multi-energy toolbox (Kangaskhan/Raging Bolt/Ogerpon/Meowth).

Mined from 2026-07-26 top episodes (exact list 182x; James Cox #2 @1218,
61.9% win rate in top play).
"""
from __future__ import annotations

try:
    from .base import Deck
except ImportError:
    from base import Deck  # noqa: F401

DECKLIST: dict[int, int] = {
    1: 9,      # Basic {G} Energy
    1116: 4,   # Energy Switch
    1121: 4,   # Ultra Ball
    1198: 4,   # Crispin
    1250: 4,   # Area Zero Underdepths
    96: 3,     # Teal Mask Ogerpon ex
    756: 3,    # Mega Kangaskhan ex
    1071: 3,   # Meowth ex
    3: 2,      # Basic {W} Energy
    4: 2,      # Basic {L} Energy
    6: 2,      # Basic {F} Energy
    63: 2,     # Raging Bolt ex
    184: 2,    # Latias ex
    1097: 2,   # Night Stretcher
    1098: 2,   # Glass Trumpet
    1182: 2,   # Boss's Orders
    1197: 2,   # Xerosic's Machinations
    1205: 2,   # Cyrano
    5: 1,      # Basic {P} Energy
    108: 1,    # Wellspring Mask Ogerpon ex
    140: 1,    # Fezandipiti ex
    272: 1,    # Lillie's Clefairy ex
    978: 1,    # Passimian
    1088: 1,   # Prime Catcher
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
