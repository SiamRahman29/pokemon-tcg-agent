"""ISOLATION test: our stock 60, with Xerosic's Machinations x2 and nothing else.

`grimmsnarl_xerosic` measured **0.431 [0.415, 0.446]** against the stock list's
own control of 0.504 -- about -51 Elo, the largest deck loss measured here. ⚠ But
that deck was a FIVE-card bundle (Xerosic x2, Energy Switch, Morgrem 3->2,
Petrel 4->3, Poffin 4->3, Grimmsnarl ex 3->4, Boss's 2->3), so §8ab's
"derive and size, do not bundle" says it cannot attribute the loss to any one of
them -- the Morgrem cut is as plausible a culprit as the Xerosic.

This list changes exactly TWO cards from `decks/grimmsnarl.py`:
  OUT: Dawn x1, Pokegear 3.0 x1   -- the two lowest-liveness non-mirror-blind
       slots in §8ar's matrix (0.40 and 0.47 weighted plays/game)
  IN : Xerosic's Machinations x2

⚠ It is still 2 cards, not 1, because Xerosic is being tested at the count the
user asked for. Supporters go 10 -> 12 and only one may be played per turn.
"""
from __future__ import annotations

try:
    from .base import Deck
except ImportError:
    from base import Deck  # noqa: F401

DECKLIST: dict[int, int] = {
    7: 10, 112: 4, 646: 4, 1086: 4, 1152: 4, 1219: 4, 1227: 4, 1259: 4,
    647: 3, 648: 3, 1079: 3, 1097: 3,
    104: 2, 860: 2, 1182: 2,
    1197: 2,   # Xerosic's Machinations  <-- the only addition
    1080: 1, 1137: 1,
    # removed vs stock: 1122 Pokegear 3.0 x1, 1231 Dawn x1
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
