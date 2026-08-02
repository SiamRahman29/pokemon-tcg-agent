"""Track C variant #4 (user-directed): the four-card reconfiguration.

From the current 60:

    OUT  1x Team Rocket's Petrel   (4 -> 3)
         2x Buddy-Buddy Poffin     (4 -> 2)
         1x Tool Scrapper          (1 -> 0)
    IN   1x Marnie's Grimmsnarl ex (3 -> 4)
         1x Boss's Orders          (2 -> 3)
         2x Budew                  (0 -> 2)

**The thesis:** trade search redundancy for threat density and disruption. Petrel
tutors any Trainer and Poffin fetches Basics, so both are *enablers*; the cards
coming in are *win conditions* (a 4th 320 HP attacker), *reach* (a 3rd gust) and
*tempo* (the turn-one item lock).

⚠ **THIS IS A BUNDLE OF FOUR CHANGES AND THE ATTRIBUTION PROBLEM IS REAL.**
§8ab measured exactly this failure mode on the net side: the v4 state block's
three *derived* members carried the entire +37 Elo while the five *unsized*
extras were **worse than having no block at all** (−22 Elo), and the standing
lesson written from it was **"derive and size, do not bundle."** A win here does
not tell us which card won, and a loss does not tell us which card lost.
⇒ **If this arm moves, it must be ablated before anything is shipped.**
It is run as an exploratory configuration test, which is a legitimate thing to
do first -- but not a substitute for the ablation.

⚡ **One of the four is ALREADY MEASURED and it was a null.** §8aj tested
`Dawn x1 -> 4th Marnie's Grimmsnarl ex` at n=4,000: **0.4911 [0.476, 0.507]**
against a same-deck control of **0.4980 [0.483, 0.513]**, difference
−0.0069 ± 0.0112. So the 4th Grimmsnarl ex, *paid for out of a thin slot*, does
nothing detectable. Here it is paid for out of Petrel/Poffin instead, which is a
different trade -- but the prior on that single component is "neutral".

⚠ **Pre-registered risks, so they are not rediscovered as excuses:**
  1. 🔴 **Poffin 4 -> 2 halves our bench development.** §8ai found that in **269
     of 283** empty-bench decisions we had no benchable Basic in hand at all, and
     Poffin is the card that fixes that. Adding 2 Budew raises Basics 10 -> 12 and
     adds two more Poffin targets (30 HP <= 70), which cuts the other way -- but
     the net effect on bench development is genuinely unknown and is the most
     likely way this arm loses.
  2. **Petrel 4 -> 3** reduces Trainer access, and the 3rd Boss's Orders is itself
     a Trainer that Petrel would have fetched. Partly self-cancelling.
  3. **Two 30 HP Budew on the bench are free prizes** to Boss's Orders -- and this
     list runs a 3rd copy, so the mirror版 of this deck gusts them out more often.
  4. Prior remains a null: the consensus 60 (seen 353x) runs none of this.

Exposure (§8af): Budew is **4,375 occurrences as OUR OWN option**, 1.55x Tool
Scrapper's, so nothing here is off-distribution for the net.

Run at n=4,000 matches (8,000 games) against `grimmsnarl`; the same-deck control
floor is **0.4980 [0.483, 0.513]** (§8aj).
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
    235: 2,    # Budew                    <-- NEW (Itchy Pollen, 0 energy item lock)
    1086: 2,   # Buddy-Buddy Poffin       <-- 4 -> 2
    1152: 4,   # Poke Pad
    1219: 3,   # Team Rocket's Petrel     <-- 4 -> 3
    1227: 4,   # Lillie's Determination
    1259: 4,   # Spikemuth Gym
    647: 3,    # Marnie's Morgrem
    648: 4,    # Marnie's Grimmsnarl ex   <-- 3 -> 4
    1079: 3,   # Rare Candy
    1097: 3,   # Night Stretcher
    104: 2,    # Froslass
    860: 2,    # Snorunt
    1182: 3,   # Boss's Orders            <-- 2 -> 3
    1080: 1,   # Unfair Stamp
    1122: 1,   # Pokegear 3.0
    1231: 1,   # Dawn                     <-- kept (only Stage 1 + Stage 2 fetch)
    # 1137: 1, # Tool Scrapper            <-- CUT (0.00 plays per mirror game)
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
