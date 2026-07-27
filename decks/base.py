"""Deck: an ordered {card_id: count} 60-card list with CSV helpers."""
from __future__ import annotations

import csv as _csv
from dataclasses import dataclass
from pathlib import Path

_DATA = Path(__file__).resolve().parents[1] / "data" / "EN_Card_Data.csv"


@dataclass(frozen=True)
class CardInfo:
    card_id: int
    name: str
    category: str
    stage: str
    hp: str
    type: str


_CARD_CACHE: dict[int, CardInfo] | None = None


def card_table() -> dict[int, CardInfo]:
    global _CARD_CACHE
    if _CARD_CACHE is None:
        _CARD_CACHE = {}
        if _DATA.exists():
            with _DATA.open(encoding="utf-8-sig") as fh:
                for row in _csv.DictReader(fh):
                    try:
                        cid = int(row["Card ID"])
                    except (KeyError, ValueError):
                        continue
                    _CARD_CACHE[cid] = CardInfo(
                        cid, row.get("Card Name", "?"),
                        row.get("Category", ""), row.get("Stage/Type", ""),
                        row.get("HP", ""), row.get("Type", ""))
    return _CARD_CACHE


@dataclass(frozen=True)
class DeckEntry:
    card: CardInfo
    count: int


class Deck:
    def __init__(self, decklist: dict[int, int]):
        self.decklist = dict(decklist)

    @property
    def size(self) -> int:
        return sum(self.decklist.values())

    def cards(self) -> list[int]:
        return [cid for cid, cnt in self.decklist.items() for _ in range(cnt)]

    def to_csv(self) -> str:
        return "\n".join(str(c) for c in self.cards()) + "\n"

    def __iter__(self):
        table = card_table()
        for cid, cnt in self.decklist.items():
            info = table.get(cid, CardInfo(cid, f"#{cid}", "", "", "", ""))
            yield DeckEntry(info, cnt)

    def __repr__(self) -> str:
        return f"Deck({self.size} cards, {len(self.decklist)} distinct)"
