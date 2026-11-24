"""Deck profile: everything deck-specific that the Transformer features read.

The upstream features (ptcg-population-rl, `mega_lucario/gbdt/features.py`)
hard-code Mega Lucario's card ids, attack ids, damage numbers and Fighting
energy. Here those come from the deck list plus the engine's card and attack
data, so the same code serves Grimmsnarl now and any deck later.

A profile is identified by a hash of its JSON form. Every artifact built from
it (distillation shards, checkpoints, the C++ header) records that hash, and
loading refuses a mismatch.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

MAX_SLOTS = 32     # deck-local card vocabulary (Transformer count blocks)
MAX_ATTACKS = 8    # attack one-hot width in the option feature row


@dataclass(frozen=True)
class DeckProfile:
    name: str
    counts: dict[int, int]            # card id -> copies, 60 total
    ids: tuple[int, ...]              # distinct card ids, sorted: slot order
    attack_ids: tuple[int, ...]       # MAX_ATTACKS long, padded with -1
    attacks: dict[int, tuple[int, tuple[int, ...]]]  # id -> (damage, energy cost)
    energy_type: int                  # the deck's attacking type (most common basic energy)

    def to_json(self) -> dict[str, Any]:
        return dict(
            name=self.name,
            counts={str(k): v for k, v in sorted(self.counts.items())},
            attack_ids=list(self.attack_ids),
            attacks={str(k): [d, list(c)] for k, (d, c) in sorted(self.attacks.items())},
            energy_type=self.energy_type,
        )

    @property
    def profile_id(self) -> str:
        raw = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def check_deck(self, deck: list[int]) -> None:
        if Counter(deck) != Counter(self.counts):
            raise ValueError(f"deck does not match the {self.name} profile")


def build(name: str, deck: list[int], cards: dict[int, dict[str, Any]],
          attack_data: dict[int, dict[str, Any]]) -> DeckProfile:
    """Derive a profile from a 60-card list and the engine's card/attack tables."""
    counts = Counter(int(c) for c in deck)
    if sum(counts.values()) != 60:
        raise ValueError("a deck has 60 cards")
    ids = tuple(sorted(counts))
    if len(ids) > MAX_SLOTS:
        raise ValueError(f"{len(ids)} distinct cards; the Transformer has {MAX_SLOTS} slots")
    attack_ids = sorted({a for c in ids for a in cards[c]["attacks"]})
    if len(attack_ids) > MAX_ATTACKS:
        raise ValueError(f"{len(attack_ids)} attacks; the feature row has {MAX_ATTACKS}")
    attacks = {a: (int(attack_data[a]["damage"]), tuple(int(e) for e in attack_data[a]["energies"]))
               for a in attack_ids}
    energy = Counter()
    for c, n in counts.items():
        if cards[c]["cardType"] == 5 and cards[c]["energyType"]:   # basic energy
            energy[cards[c]["energyType"]] += n
    if not energy:
        raise ValueError("deck has no basic energy to define its type")
    return DeckProfile(
        name=name, counts=dict(counts), ids=ids,
        attack_ids=tuple(attack_ids) + (-1,) * (MAX_ATTACKS - len(attack_ids)),
        attacks=attacks, energy_type=energy.most_common(1)[0][0])


def engine_tables() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Card and attack tables from the Playground engine, as plain dicts.

    Field names follow the engine's CardData. Note `energyType` on a Pokemon
    card is its elemental type (Grimmsnarl ex: 7, Darkness).
    """
    from ptcg.env import sdk

    fields = ("cardId", "name", "cardType", "energyType", "weakness", "resistance",
              "attacks", "ex", "megaEx", "stage1", "stage2", "basic", "hp", "evolvesFrom")
    cards = {cid: {f: getattr(c, f) for f in fields} for cid, c in sdk.card_db().items()}
    attacks = {aid: dict(damage=a.damage, energies=list(a.energies))
               for aid, a in sdk.attack_db().items()}
    return cards, attacks


def for_deck_module(name: str) -> DeckProfile:
    """Profile for a deck in `decks/`, using the engine's tables."""
    from arena import resolve_deck

    canonical, deck = resolve_deck(name)
    cards, attacks = engine_tables()
    return build(canonical, list(deck), cards, attacks)
