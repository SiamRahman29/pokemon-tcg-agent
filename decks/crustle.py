"""Crustle -- the deck the top of the board is winning with, as of 2026-07-29.

**This is the current meta list**, mined as the most common exact 60 among the
400 top episodes of 2026-07-29 (`out/meta/post_shift_0729.txt`): the archetype
appears in **145 of 800 seats (18.1%) at a 56.6% win rate**, this exact list
**77 times**, and both of the LB's top two players are on it (`flg` 1205.7,
`Majkel1337` 1186.4). Pre-shift, on 2026-07-22/24, it appeared in **1 of 1,600
seats** -- see `out/meta/pre_shift_0722_0724.txt`. That is the whole meta shift
in two numbers.

⚠ **Replaced the earlier reconstruction** (mined from a `crustle-replays/`
directory that is not in the repo), which differed by 12 card slots -- most
importantly it ran **4x Crushing Hammer**, which the current list does not, and
it lacked Colress's Tenacity / Basic {G} Energy / Tool Scrapper / Battle Cage.
Measuring against the stale version would have answered the wrong matchup, which
is the same error rule 12 warns about one level up.

Mysterious Rock Inn is an ABILITY on Crustle itself (345; 344 is Dwebble) and
prevents damage from opponent {ex} attacks -- Marnie's Grimmsnarl ex is
`ex=True`, so Shadow Bullet should deal zero. **Unverified in-engine**; see
HANDOFF §3.2 for the probe that has to run before any deck work.
"""

from __future__ import annotations

try:
    from .base import Deck
except ImportError:  # pragma: no cover
    from base import Deck  # noqa: F401


DECKLIST: dict[int, int] = {
    1: 2,     # Basic {G} Energy
    11: 4,    # Mist Energy
    14: 4,    # Spiky Energy
    18: 4,    # Grow Grass Energy
    20: 2,    # Rock Fighting Energy

    117: 1,   # Cornerstone Mask Ogerpon ex

    344: 4,   # Dwebble
    345: 3,   # Crustle

    756: 2,   # Mega Kangaskhan ex

    1086: 2,  # Buddy-Buddy Poffin

    1121: 2,  # Ultra Ball
    1122: 4,  # Pokégear 3.0
    1123: 1,  # Switch

    1137: 1,  # Tool Scrapper
    1147: 4,  # Jumbo Ice Cream
    1159: 1,  # Hero's Cape

    1182: 4,  # Boss's Orders
    1194: 2,  # Colress's Tenacity
    1197: 1,  # Xerosic's Machinations

    1219: 4,  # Team Rocket's Petrel
    1225: 2,  # Hilda
    1227: 4,  # Lillie's Determination

    1257: 1,  # Team Rocket's Factory
    1264: 1,  # Battle Cage
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
