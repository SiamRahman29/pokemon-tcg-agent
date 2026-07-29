"""Crustle deck reconstructed from `crustle-replays/*.json`.

The decklist is the most common exact 60-card multiset observed across the
episodes in `crustle-replays/` that include card ID `345` (Crustle).
"""

from __future__ import annotations

try:
    from .base import Deck
except ImportError:  # pragma: no cover
    from base import Deck  # noqa: F401


DECKLIST: dict[int, int] = {
    11: 4,    # Mist Energy
    14: 4,    # Spiky Energy
    18: 4,    # Grow Grass Energy
    20: 2,    # Rock Fighting Energy

    117: 1,   # Cornerstone Mask Ogerpon ex

    344: 4,   # Dwebble
    345: 3,   # Crustle

    756: 2,   # Mega Kangaskhan ex

    1086: 2,  # Buddy-Buddy Poffin

    1120: 4,  # Crushing Hammer
    1121: 2,  # Ultra Ball
    1122: 4,  # Pokégear 3.0
    1123: 1,  # Switch

    1147: 4,  # Jumbo Ice Cream
    1159: 1,  # Hero's Cape

    1182: 4,  # Boss's Orders
    1197: 1,  # Xerosic’s Machinations

    1212: 1,  # Cook
    1219: 4,  # Team Rocket's Petrel
    1225: 3,  # Hilda
    1227: 4,  # Lillie's Determination

    1257: 1,  # Team Rocket's Factory
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size

