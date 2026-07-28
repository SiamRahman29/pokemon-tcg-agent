"""Grimmsnarl variant: +2 Boss's Orders, -1 Tool Scrapper, -1 Spikemuth Gym.

Proposed from replay-watching (2026-07-28): we attack whatever is Active
instead of dragging up a cheap benched Pokemon and taking the prize sooner.
More Boss's Orders is the direct way to buy that option.

The audit (scripts/opportunity_audit.py) supports cutting Tool Scrapper -- over
the whole demonstrator corpus it is legal on only 318 turns against Boss's
Orders' 10,677 -- but does NOT support cutting Spikemuth Gym, which top players
play on 95.6% of the turns it is legal. The Gym cut is included anyway because
the change needs a second slot and this is the pair the user proposed; if the
A/B loses, re-test with the Gym left at 4 and a different slot cut.

Measure against `grimmsnarl`, same net both sides:

    python scripts/arena.py play bc bc --deck-a grimmsnarl_boss \
        --deck-b grimmsnarl --matches 1000

Caveat: `agents/sa/policy_net.npz` was cloned from replays of the *standard*
list (seen 290x in one day's top episodes), so a modified list is off
distribution for the net. A loss here is evidence against the deck OR against
the net's ability to pilot it, and the two are not separable with this pilot.
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
    1259: 3,   # Spikemuth Gym          (was 4)
    647: 3,    # Marnie's Morgrem
    648: 3,    # Marnie's Grimmsnarl ex
    1079: 3,   # Rare Candy
    1097: 3,   # Night Stretcher
    104: 2,    # Froslass
    860: 2,    # Snorunt
    1182: 4,   # Boss's Orders          (was 2)
    1080: 1,   # Unfair Stamp
    1122: 1,   # Pokegear 3.0
    1231: 1,   # Dawn
    # 1137 Tool Scrapper: cut (was 1)
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
