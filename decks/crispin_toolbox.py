"""Crispin toolbox -- the other new meta deck of 2026-07-29.

Mined as the most common exact 60 among the 400 top episodes of 2026-07-29
(`out/meta/post_shift_0729.txt`): **135 of 800 seats (16.9%) at a 58.5% win
rate** -- the highest win rate of any archetype in the sample -- and this exact
list all 135 times. Pre-shift (07-22/24) the archetype appeared in **2 of 1,600
seats**.

⚠ **One team accounts for all 135 games** (`James Cox & Henry Chao`, 1205.1), so
treat the 58.5% as one strong pilot's result, not as a broad field average. It is
still a top-rated deck we will meet, and it is a *different* threat shape from
Crustle: a multi-type toolbox (Teal Mask Ogerpon ex, Mega Kangaskhan ex, Meowth
ex, Raging Bolt ex, Latias ex) tutored by Crispin x4 and Energy Switch x4, with
Area Zero Underdepths x4 as the stadium -- i.e. it contests Spikemuth Gym, which
both grimmsnarl mirrors play ~100% of the time.

Distinct from `decks/crispin_box.py`, which is an older/different reconstruction.
"""

from __future__ import annotations

try:
    from .base import Deck
except ImportError:  # pragma: no cover
    from base import Deck  # noqa: F401


DECKLIST: dict[int, int] = {
    1: 9,     # Basic {G} Energy
    3: 2,     # Basic {W} Energy
    4: 2,     # Basic {L} Energy
    5: 1,     # Basic {P} Energy
    6: 2,     # Basic {F} Energy

    63: 2,    # Raging Bolt ex
    96: 3,    # Teal Mask Ogerpon ex
    108: 1,   # Wellspring Mask Ogerpon ex
    140: 1,   # Fezandipiti ex
    184: 2,   # Latias ex
    272: 1,   # Lillie's Clefairy ex
    756: 3,   # Mega Kangaskhan ex
    978: 1,   # Passimian
    1071: 3,  # Meowth ex

    1088: 1,  # Prime Catcher
    1097: 2,  # Night Stretcher
    1098: 2,  # Glass Trumpet
    1116: 4,  # Energy Switch
    1121: 4,  # Ultra Ball

    1182: 2,  # Boss's Orders
    1197: 2,  # Xerosic's Machinations
    1198: 4,  # Crispin
    1205: 2,  # Cyrano

    1250: 4,  # Area Zero Underdepths
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
