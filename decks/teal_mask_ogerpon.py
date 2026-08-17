"""Teal Mask Ogerpon ex — the field's consensus Grass list.

**Source.** `out/meta/day_0802.txt` (and identical on `day_0801.txt`): the
"Basic {G} Energy / Teal Mask Ogerpon ex" archetype, this exact 60 seen
**79×** among 08-02 top episodes (58× on 08-01). Our own ladder census
(`out/logs/p9_v5s2_field_census.txt`) names Teal Mask Ogerpon ex as
**4.8% of the field** (16/330 games) — 6th after Grimmsnarl / Alakazam /
Crustle / Dragapult / Mega Lucario.

This is a one-attacker Tera Grass engine (Teal Dance attach + draw, Myriad
Leaf Shower scaling with energy on both Actives). It is **not** the
Cornerstone Mask Ogerpon 1-of inside `decks/crustle.py`.

⚠ Same band caveat as `cynthia_garchomp.py`: mined episodes describe
`avg_score` ≥1055. The 4.8% share is from our own replays; the 60 is the
consensus list among strong players.

🔴 A `bc`-piloted Ogerpon agent measures **the deck × how well the shared
clone pilots it**, not a real Ogerpon player. No rule pilot exists.
"""
from __future__ import annotations

try:
    from .base import Deck
except ImportError:  # allow `python decks/teal_mask_ogerpon.py`
    from base import Deck  # noqa: F401

Basic_Grass_Energy = 1
Grow_Grass_Energy = 18
Teal_Mask_Ogerpon_ex = 96
Bug_Catching_Set = 1094
Energy_Retrieval = 1118
Energy_Search = 1119
Pokegear_3_0 = 1122
Tera_Orb = 1127
Tool_Scrapper = 1137
Jumbo_Ice_Cream = 1147
Hero_Cape = 1159
Boss_Orders = 1182
Briar = 1201
Judge = 1213
N_Plan = 1221
Harlequin = 1223
Lillie_Determination = 1227
Lively_Stadium = 1251

DECKLIST: dict[int, int] = {
    Basic_Grass_Energy: 18,
    Teal_Mask_Ogerpon_ex: 4,
    Bug_Catching_Set: 4,
    Energy_Search: 4,
    Judge: 4,
    Lillie_Determination: 4,
    Pokegear_3_0: 3,
    Boss_Orders: 3,
    Grow_Grass_Energy: 2,
    Energy_Retrieval: 2,
    Tera_Orb: 2,
    Jumbo_Ice_Cream: 2,
    Harlequin: 2,
    Lively_Stadium: 2,
    Tool_Scrapper: 1,
    Hero_Cape: 1,
    Briar: 1,
    N_Plan: 1,
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, f"expected 60 cards, got {DECK.size}"


if __name__ == "__main__":
    print(DECK)
    for entry in DECK:
        print(f"{entry.count:>2}x  {entry.card.name}")
