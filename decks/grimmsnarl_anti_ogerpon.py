"""Grimmsnarl tech vs Teal Mask Ogerpon — item lock + hand strip + snipe tools.

Built from p101 loss autopsy on bc:all@grimmsnarl vs bc:oger@teal_mask_ogerpon.
Ogerpon lists are item-heavy (Energy Search, Bug Catching Set, Judge, retrieval)
and rely on Hero's Cape / Tera Orb. Changes vs decks/grimmsnarl.py:

  + Budew x2          Itchy Pollen — 0-energy item lock for one opponent turn
  + Xerosic x1        hand strip when Ogerpon is drawing off Teal Dance
  + Boss's Orders 3   more bench snipes on low-HP Teal Mask lines
  + Tool Scrapper 2   strip Cape / Tera Orb
  - Buddy-Buddy Poffin 4->3
  - Night Stretcher 3->2
  - Dawn x1           Rare Candy + Poffin cover basics; slot freed for tech
  - Pokegear 3.0 x1

⚠ Screen result (p101, n=1000): WR=0.084 vs Ogerpon — WORSE than stock grimmsnarl
  (0.155). Do not ship this list; use policy fine-tune instead (policy_v5_s2_oger_ft2).
⚠ Mirror cost is unknown until screened — run p101 with --screen.
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
    235: 2,    # Budew
    1086: 3,   # Buddy-Buddy Poffin
    1152: 4,   # Poke Pad
    1219: 4,   # Team Rocket's Petrel
    1227: 4,   # Lillie's Determination
    1259: 4,   # Spikemuth Gym
    647: 3,    # Marnie's Morgrem
    648: 3,    # Marnie's Grimmsnarl ex
    1079: 3,   # Rare Candy
    1097: 2,   # Night Stretcher
    104: 2,    # Froslass
    860: 1,    # Snorunt
    1182: 3,   # Boss's Orders
    1080: 1,   # Unfair Stamp
    1137: 2,   # Tool Scrapper
    1197: 1,   # Xerosic's Machinations
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, DECK.size
