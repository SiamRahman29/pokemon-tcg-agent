"""Community list, user-revised (day 18): Budew and Yveltal out, Xerosic's x2 in.

**The user's reasoning, recorded because it is the hypothesis being tested:**
Budew is an easy snipe target, and with Grimmsnarl ex as the attacker Yveltal is
mostly an unused sub -- so both come out. Xerosic's Machinations goes in as hand
disruption, played when the opponent is holding a lot of cards.

⚠ **It is a SUPPORTER (cardType 3), not an Item.** This list already runs 10
supporters (4 Lillie's + 3 Petrel + 3 Boss's) and only one may be played per
turn; these two copies take it to 12 and compete with all of them. The card does
what the user says -- "your opponent discards cards from their hand until they
have 3 cards in their hand" -- it simply pays for it in a contested slot. Flagged
before the run, not after.

⚠ **Energy Switch x1 is carried over from the community list and is a KNOWN dead
slot in our net's hands**: over the recorded games in out/replays/community_vs_stock
it was **offered 25 times and played 0 times**. It is retained because the user
did not ask for it to change, and it is named here so the result is not read as
if all 60 slots were live.

Exposure (§8af, p22): Xerosic's Machinations is 7,779 as our own option, **2.76x**
the weakest card we already play -- low risk on the card-level filter. ⚠ Which
today's search showed is necessary and NOT sufficient: Ultra Ball sat at 5.59x
and lost all six slots it was tested in.

Five-card difference from decks/grimmsnarl.py:
  OUT: Poffin 4->3, Petrel 4->3, Morgrem 3->2, Pokegear 3.0 x1, Dawn x1
  IN : Grimmsnarl ex 3->4, Boss's Orders 2->3, Energy Switch x1, Xerosic's x2
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
    648: 4,    # Marnie's Grimmsnarl ex
    1152: 4,   # Poke Pad
    1227: 4,   # Lillie's Determination      SUPPORTER
    1259: 4,   # Spikemuth Gym
    1086: 3,   # Buddy-Buddy Poffin
    1219: 3,   # Team Rocket's Petrel        SUPPORTER
    1182: 3,   # Boss's Orders               SUPPORTER
    1097: 3,   # Night Stretcher
    1079: 3,   # Rare Candy
    647: 2,    # Marnie's Morgrem
    860: 2,    # Snorunt
    104: 2,    # Froslass
    1197: 2,   # Xerosic's Machinations      SUPPORTER  <-- NEW
    1116: 1,   # Energy Switch               ⚠ known dead slot, 0 of 25
    1080: 1,   # Unfair Stamp
    1137: 1,   # Tool Scrapper
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
