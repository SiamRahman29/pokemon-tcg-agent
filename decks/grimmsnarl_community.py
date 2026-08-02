"""The community-updated Marnie's Grimmsnarl ex list (user-supplied, day 18).

⚠ **Provenance differs from every other deck here and that matters.** Every
other list in `decks/` was mined from this board's own episodes. This one comes
from the wider TCG community, so it may be optimised for a card pool this engine
does not share -- `Special Red Card` is in the community list and **is not
implemented here at all** (searched every card id; no name contains "Red Card").

**Six of the 60 were not specified and are the user's judgement calls:**
  +1 Night Stretcher, +1 Rare Candy, +1 Marnie's Impidimp,
  +1 Marnie's Grimmsnarl ex, +1 Basic {D} Energy
plus two substitutions for cards this engine lacks or the user replaced:
  Special Red Card -> Unfair Stamp, Air Balloon -> Tool Scrapper
That is 59. 🔴 **The 60th is MINE, not the user's: +1 Buddy-Buddy Poffin (2->3).**
Reason: the community list cuts Poffin 4->2 while ADDING two new basics (Budew,
Yveltal), so basic-search demand rises as the search count falls. It also avoids
touching the Morgrem-2 / Rare-Candy tradeoff the user reasoned about explicitly.
⚠ If this list is ever quoted, quote that one slot as unattributed.

Net difference from `decks/grimmsnarl.py` is **5 cards**, not the 18 a raw diff
against the 54-card core would suggest:
  OUT: Poffin 4->3, Petrel 4->3, Morgrem 3->2, Pokegear 3.0 x1, Dawn x1
  IN : Grimmsnarl ex 3->4, Boss's Orders 2->3, Budew, Yveltal, Energy Switch
"""
from __future__ import annotations

try:
    from .base import Deck
except ImportError:
    from base import Deck  # noqa: F401

DECKLIST: dict[int, int] = {
    7: 10,     # Basic {D} Energy      (9 given + 1 user)
    112: 4,    # Munkidori
    646: 4,    # Marnie's Impidimp     (3 given + 1 user)
    648: 4,    # Marnie's Grimmsnarl ex (3 given + 1 user)
    1152: 4,   # Poke Pad
    1227: 4,   # Lillie's Determination
    1259: 4,   # Spikemuth Gym
    1086: 3,   # Buddy-Buddy Poffin    (2 given + 1 ASSISTANT)
    1219: 3,   # Team Rocket's Petrel
    1182: 3,   # Boss's Orders
    1097: 3,   # Night Stretcher       (2 given + 1 user)
    1079: 3,   # Rare Candy            (2 given + 1 user)
    647: 2,    # Marnie's Morgrem
    860: 2,    # Snorunt
    104: 2,    # Froslass
    235: 1,    # Budew                 NEW
    689: 1,    # Yveltal               NEW
    1116: 1,   # Energy Switch         NEW
    1080: 1,   # Unfair Stamp          (substituted for Special Red Card)
    1137: 1,   # Tool Scrapper         (substituted for Air Balloon)
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
