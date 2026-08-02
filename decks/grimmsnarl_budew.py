"""Track C variant #3 (user-directed): `Buddy-Buddy Poffin x1 + Tool Scrapper x1 -> Budew x2`.

**The mechanism, verified in-engine rather than assumed from the paper card.**
Budew (235) has **no ability here** (`skills: []`); its item lock is an *attack*:

    Itchy Pollen -- energies: [], damage 10
    "During your opponent's next turn, they can't play any Item cards from their hand."

**It costs ZERO energy**, so Budew's Grass typing is irrelevant in our Dark list --
it fires with nothing attached. That is the whole case for the card.

**Exposure (§8af): the net already knows this card.** Budew appears **4,375 times
as OUR OWN option** and 30,107 times overall in the 2,810-game corpus -- **1.55x
the exposure of Tool Scrapper, which we play today**. Both encoding channels are
populated, so this is not a teaching problem. We are handing back a card the net
was already trained on and that our list simply did not run.

**Why these two cuts (user-directed, and the reasoning is sound):**
  * **Tool Scrapper x1 -> 0.** Our thinnest slot on utilisation (0.13 plays/game)
    and it is played **0.00 times per mirror game** (§8aj) -- our list runs no
    tools, so in the mirror there is nothing to scrap. It is pure anti-tool tech
    being cut from a mirror-facing list.
  * **Buddy-Buddy Poffin 4 -> 3.** 0.67 plays/game, 0.167 per copy -- the thinnest
    per-copy Item we run.
  * ⛔ **Dawn is deliberately KEPT.** It is the only card in the 60 that fetches
    the evolved line: *"Search your deck for a Basic Pokemon, a Stage 1 Pokemon,
    and a Stage 2 Pokemon ... into your hand."* Buddy-Buddy Poffin searches
    **Basics with <=70 HP onto the Bench** and cannot do this. §8aj's null on
    `Dawn -> 4th Grimmsnarl ex` says the slot is *contestable*, not that the card
    is dead -- and its unique effect is the reason to keep it.

**Sizing, pre-registered.** Itchy Pollen needs Budew ACTIVE, which in practice
means opening with it. Two copies are in the opening seven in
**1 - C(58,7)/C(60,7) = 22.1%** of games. Our items are played **3.60x/game**,
peaking at **0.88 on the first turn** (§8aj), so a one-turn lock denies roughly
one item play at the moment it is worth most. For a 22% -> visible effect the
swing inside those games must be ~7 pp at n=8,000, so **this arm is run at
n=4,000 matches (8,000 games), double the standard**, or the effect cannot clear
the noise floor whatever its true size.

⚠ **Pre-registered costs, so they are not rediscovered as excuses afterwards:**
  1. A 30 HP basic on the bench is a **free prize** to Boss's Orders (the mirror
     runs 2) plus any attack.
  2. Cutting a Poffin weakens basic-fetch, and Poffin *can* fetch Budew (30 <= 70)
     -- but onto the **Bench**, which does not set up Itchy Pollen.
  3. Prior remains a null: the consensus 60 is seen **353x** among the field's
     strongest players and does not run Budew.

Control: the same-deck floor measured at **0.4980 [0.483, 0.513]**, n=4,000 (§8aj).
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
    1086: 3,   # Buddy-Buddy Poffin       <-- 4 -> 3
    1152: 4,   # Poke Pad
    1219: 4,   # Team Rocket's Petrel
    1227: 4,   # Lillie's Determination
    1259: 4,   # Spikemuth Gym
    647: 3,    # Marnie's Morgrem
    648: 3,    # Marnie's Grimmsnarl ex
    1079: 3,   # Rare Candy
    1097: 3,   # Night Stretcher
    104: 2,    # Froslass
    860: 2,    # Snorunt
    1182: 2,   # Boss's Orders
    1080: 1,   # Unfair Stamp
    1122: 1,   # Pokegear 3.0
    1231: 1,   # Dawn                     <-- KEPT: only card that fetches Stage 1 + Stage 2
    # 1137: 1, # Tool Scrapper            <-- CUT (0.00 plays per mirror game)
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
