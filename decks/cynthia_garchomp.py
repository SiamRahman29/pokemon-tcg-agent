"""Cynthia's Garchomp ex — the field's consensus list, for the missing anchor.

**Why this exists (day 17).** §8ac re-weighted the field by opponent rating and
found two archetypes that outrank Crustle + Mega Lucario combined and have **no
anchor at all**: Cynthia's Garchomp ex at **6.7%** and Dragapult ex at **5.3%**.
`rule:dragapult` turned out to already exist and merely never to have been used;
Garchomp had nothing, and this is the deck half of closing that.

**Source: `out/meta/pre_shift_0722_0724.txt`, the "Basic {F} Energy / Rock
Fighting Energy" archetype — this exact 60 seen 159×** in the mined top-episode
snapshot. Transcribed with card ids, not names, and asserted to 60 below.

⚠ **The band caveat still binds (ROADMAP, revised 2026-07-31).** Mined episodes
describe `avg_score` ≥1055 and we play well below that, so this is *the consensus
list among strong players*, not necessarily what a 900-rated Garchomp pilot runs.
That is acceptable for a DECK — the list is the archetype — and it is exactly
what is **not** acceptable for choosing which archetypes to anchor, which is why
the 6.7% share above comes from `p9_field_census.py` on our own replays instead.

✅ **Exposure checked before building (§8af's filter): all 20 distinct card ids
are in the training corpus, 0 of 60 copies untrained.** So our own net can pilot
this list, which is what makes an anchor possible today without writing a rule
pilot — ROADMAP's "hold the pilot constant and vary the 60".

🔴 **And the caveat that goes with that, stated so the number cannot lie:** a
`bc:v5`-piloted Garchomp anchor measures **the deck × how well OUR net pilots
it**, not the strength of a real Garchomp player. It is a better instrument than
having no Garchomp opponent at all and a worse one than a tuned rule pilot.
Rule 12: it has not been validated as playing the archetype well.
"""
from __future__ import annotations

try:
    from .base import Deck
except ImportError:  # allow `python decks/cynthia_garchomp.py`
    from base import Deck  # noqa: F401

Basic_Fighting_Energy = 6
Rock_Fighting_Energy = 20
Cynthia_Roselia = 341
Cynthia_Roserade = 342
Cynthia_Gible = 379
Cynthia_Gabite = 380
Cynthia_Garchomp_ex = 381
Cynthia_Spiritomb = 387
Unfair_Stamp = 1080
Buddy_Buddy_Poffin = 1086
Night_Stretcher = 1097
Fighting_Gong = 1142
Poke_Pad = 1152
Cynthia_Power_Weight = 1173
Boss_Orders = 1182
Xerosic_Machinations = 1197
Surfer = 1203
Hilda = 1225
Lillie_Determination = 1227
Forest_of_Vitality = 1261

DECKLIST: dict[int, int] = {
    Basic_Fighting_Energy: 5,
    Rock_Fighting_Energy: 4,
    Cynthia_Roselia: 4,
    Cynthia_Gible: 4,
    Cynthia_Gabite: 4,
    Buddy_Buddy_Poffin: 4,
    Fighting_Gong: 4,
    Poke_Pad: 4,
    Lillie_Determination: 4,
    Cynthia_Roserade: 3,
    Cynthia_Garchomp_ex: 3,
    Cynthia_Power_Weight: 3,
    Hilda: 3,
    Cynthia_Spiritomb: 2,
    Night_Stretcher: 2,
    Boss_Orders: 2,
    Forest_of_Vitality: 2,
    Unfair_Stamp: 1,
    Xerosic_Machinations: 1,
    Surfer: 1,
}

DECK = Deck(DECKLIST)
assert DECK.size == 60, f"expected 60 cards, got {DECK.size}"


if __name__ == "__main__":
    print(DECK)
    for entry in DECK:
        print(f"{entry.count:>2}x  {entry.card.name}")
