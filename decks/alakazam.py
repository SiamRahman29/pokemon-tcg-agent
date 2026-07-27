"""Alakazam / Telepath Psychic Energy (Yushin Ito's list, 2026-07-26 variant)."""
from __future__ import annotations

try:
    from .base import Deck
except ImportError:
    from base import Deck  # noqa: F401

DECKLIST: dict[int, int] = {
    19: 4,     # Telepath Psychic Energy
    741: 4,    # Abra
    742: 4,    # Kadabra
    743: 4,    # Alakazam
    1081: 4,   # Enhanced Hammer
    1086: 4,   # Buddy-Buddy Poffin
    1152: 4,   # Poke Pad
    1225: 4,   # Hilda
    1231: 4,   # Dawn
    305: 3,    # Dunsparce
    1079: 3,   # Rare Candy
    1182: 3,   # Boss's Orders
    1197: 3,   # Xerosic's Machinations
    5: 2,      # Basic {P} Energy
    66: 2,     # Dudunsparce
    1266: 2,   # Nighttime Mine
    13: 1,     # Enriching Energy
    140: 1,    # Fezandipiti ex
    343: 1,    # Shaymin
    1097: 1,   # Night Stretcher
    1129: 1,   # Sacred Ash
    1184: 1,   # Lana's Aid
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
