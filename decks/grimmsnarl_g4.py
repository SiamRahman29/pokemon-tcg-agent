"""Track C variant #2: `Dawn x1 -> 4th Marnie's Grimmsnarl ex`.

**Chosen by measurement, not by argument** -- which is the point, because the one
decklist variant this project had A/B'd before (0.490, null) was picked by
reasoning alone. `scripts/p25_deck_slot_audit.py` sized every slot in our 60 over
75 real ladder games (7,094 of our own selects):

  * **Dawn is our thinnest genuine slot** -- 0.29 plays/game in the mirror and
    0.25 outside it. Low in BOTH populations, so its weakness is a property of
    the card rather than of the matchup we happen to test in.
  * **Marnie's Grimmsnarl ex is our most-played card** -- 9.12 plays/game in the
    mirror, 8.96 outside -- and it sits at **3 copies of a legal 4**.

⛔ **Tool Scrapper was the obvious cut and is NOT tested here.** It is played
**0.00 times per game across 24 mirror games**, and it has to be: our list runs
no tools, so there is nothing to scrap. A mirror A/B would return "cutting it is
free" *by construction* -- the matchup would be producing the answer, not the
card (rule 16 in deck clothing). Tool Scrapper is anti-tool tech and can only be
judged against a tool-running anchor, which the crustle repair (§8ah) currently
blocks.

⚠ **The prior, written down before the run: expect a null.** Our 60 is
card-for-card the consensus list seen **353x** among the field's strongest
players. "We measured a change and kept the list" is deck analysis (ROADMAP
Track C stewardship); an unmeasured opinion is not.

Sizing of the swap itself (§8af): both cards are inside the corpus's 134 known
ids, so neither side is off-distribution for the net.
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
    648: 4,    # Marnie's Grimmsnarl ex   <-- 3 -> 4 (the add)
    1079: 3,   # Rare Candy
    1097: 3,   # Night Stretcher
    104: 2,    # Froslass
    860: 2,    # Snorunt
    1182: 2,   # Boss's Orders
    1080: 1,   # Unfair Stamp
    1122: 1,   # Pokegear 3.0
    1137: 1,   # Tool Scrapper
    # 1231: 1, # Dawn                     <-- CUT
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
