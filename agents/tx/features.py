"""Candidate feature rows and Transformer entity tokens, from public observations.

Adapted from ptcg-population-rl (Apache-2.0, commit 35707d6):
`mega_lucario/gbdt/features.py` and `mega_lucario/transformer/features.py`.
Changes from upstream (see NOTICE.md):

  - Deck-specific values come from a `DeckProfile` instead of Mega Lucario
    constants: the card-id vocabulary, the 8 attack-id slots, the energy type
    used for weakness/resistance flags, and the 12-column "deck block" (attack
    damage, KO, readiness), now derived from the engine's attack data.
  - Only the generic row is built. The upstream 4,568 "rich" GBDT columns are
    not read by the Transformer, and we have no GBDT.
  - Column positions after the per-zone count blocks depend on the deck's
    number of distinct cards, so they are computed (`Layout`) rather than the
    upstream literals 346/354/356/357.

Token layout (unchanged): row 0 global, rows 1-12 board (6 per side), rows
13-18 zones, then one row per original option, then STOP. Columns: 0 valid,
1 type, 2 legal, 3 selected, 4 card slot, 5 context slot, 6-101 generic
(sign-log1p), 102-133 deck-slot counts, 134-165 evolution counts, 166-197
option result features.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from tx.contract import SCORE_TIE_TOLERANCE
from tx.deck import MAX_ATTACKS, DeckProfile

WIDTH = 198
MAX_TOKENS = 120
MAX_OPTIONS = 100
CARD_FIELDS = 15
ZONES = 10


class Layout:
    """Column offsets of the generic feature row for a deck with `n` distinct cards."""

    def __init__(self, n_ids: int):
        self.cards = 30                       # 4 x card_values
        self.context = self.cards + 4 * CARD_FIELDS   # 90
        self.kind = self.context + 48         # 138
        self.area = self.kind + 18            # 156
        self.attack = self.area + 12          # 168
        self.zones = self.attack + MAX_ATTACKS  # 176
        self.deck = self.zones + ZONES * n_ids  # the 12-column deck block
        self.width = self.deck + 12


class Encoder:
    """Feature rows and tokens for one deck profile and one engine card table."""

    def __init__(self, profile: DeckProfile, cards: dict[int, dict[str, Any]]):
        self.profile = profile
        self.cards = cards
        self.ids = profile.ids
        self.slots = {card: i for i, card in enumerate(profile.ids)}
        self.layout = Layout(len(profile.ids))
        self.contract = dict(
            version="scio-entity-transformer-v1",
            upstream="lucario-entity-transformer-v2",
            deck_profile=profile.profile_id,
            width=WIDTH, max_tokens=MAX_TOKENS, max_options=MAX_OPTIONS,
            board_slots=12, zone_slots=6, deck_slots=32,
            feature_width=self.layout.width,
            normalization="generic sign-log1p float32; card counts / 60",
            score_tie_tolerance=SCORE_TIE_TOLERANCE,
        )
        self.contract_id = hashlib.sha256(
            json.dumps(self.contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    # ---- public card resolution (upstream, unchanged) --------------------

    @staticmethod
    def card_at(obs: dict[str, Any], area: int, index: int | None,
                player: int) -> dict[str, Any] | None:
        """Resolve an observable card; hidden zones never reveal card identities."""
        if area not in (1, 2, 3, 4, 5, 6, 7, 12):
            return None
        if player not in (0, 1) or index is None or index < 0:
            raise ValueError("Invalid card reference")
        current, selection = obs["current"], obs["select"]
        if area == 2 and player != current["yourIndex"]:
            return None
        ps = current["players"][player]
        zones = {1: selection.get("deck"), 2: ps.get("hand"), 3: ps["discard"],
                 4: ps["active"], 5: ps["bench"], 6: ps["prize"],
                 7: current["stadium"], 12: current.get("looking")}
        cards = zones[area]
        if cards is None:
            return None
        return cards[index]

    def candidate(self, obs: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve the public card associated with a legal action option."""
        kind = option["type"]
        actor = obs["current"]["yourIndex"]
        if kind == 7:
            return self.card_at(obs, 2, option["index"], actor)
        if kind in (8, 9, 10, 11):
            return self.card_at(obs, option["area"], option["index"], actor)
        if kind in (3, 4, 5, 6):
            return self.card_at(obs, option["area"], option["index"], option["playerIndex"])
        if kind == 13:
            return next(iter(obs["current"]["players"][actor]["active"]), None)
        return None

    # ---- feature row ------------------------------------------------------

    def card_values(self, card: dict[str, Any] | None) -> list[float]:
        """Public card properties. Weakness/resistance flags are against the
        deck's attacking type (upstream: hard-coded Fighting)."""
        if not card:
            return [0.0] * CARD_FIELDS
        master = self.cards[card["id"]]
        hp, maximum = card.get("hp", 0), card.get("maxHp", 0)
        prizes = 3 if master["megaEx"] else 2 if master["ex"] else 1
        t = self.profile.energy_type
        return [1, int("hp" in card), hp, maximum, maximum - hp,
                len(card.get("energies", [])), len(card.get("tools", [])),
                int(master["stage1"]), int(master["stage2"]), int(master["ex"]),
                int(master["megaEx"]), int(master.get("weakness") == t),
                int(master.get("resistance") == t),
                int(card.get("appearThisTurn", False)), prizes]

    def _ready(self, c: dict[str, Any], extra: int) -> bool:
        """Can this public card pay for one of its attacks with `extra` more
        deck-type energy? Colorless (0) cost is paid by any energy."""
        have = Counter(c.get("energies", []))
        have[self.profile.energy_type] += extra
        total = len(c.get("energies", [])) + extra
        for a in self.cards[c["id"]]["attacks"]:
            if a not in self.profile.attacks:
                continue
            cost = self.profile.attacks[a][1]
            typed = Counter(e for e in cost if e != 0)
            if total >= len(cost) and all(have[e] >= n for e, n in typed.items()):
                return True
        return False

    def _damage(self, attack_id: int | None, my_active: dict | None,
                op_active: dict | None) -> int:
        """Printed base damage with generic weakness (x2) and resistance (-30).
        Approximate: text effects, tools and abilities are not modelled."""
        if attack_id not in self.profile.attacks:
            return 0
        damage = self.profile.attacks[attack_id][0]
        if damage and op_active and my_active:
            attacker = self.cards[my_active["id"]]["energyType"]
            master = self.cards[op_active["id"]]
            if master.get("weakness") == attacker:
                damage *= 2
            elif master.get("resistance") == attacker:
                damage = max(0, damage - 30)
        return damage

    def feature_row(self, obs: dict[str, Any], option_index: int,
                    prefix: Sequence[int]) -> list[float]:
        """One candidate's generic feature row from the acting player's view."""
        cur, sel = obs["current"], obs["select"]
        actor = cur["yourIndex"]
        me, opponent = cur["players"][actor], cur["players"][1 - actor]
        stop = option_index == -1
        option = {} if stop else sel["option"][option_index]
        kind = option.get("type", -1)
        card = None if stop else self.candidate(obs, option)
        target = None
        if kind in (8, 9):
            target = self.card_at(obs, option["inPlayArea"], option["inPlayIndex"], actor)
        elif kind in (3, 4, 5, 6):
            target = card
        my_active = next(iter(me["active"]), None)
        op_active = next(iter(opponent["active"]), None)
        if kind == 13:
            target = op_active
        area = option.get("area", -1)
        target_area = option.get("inPlayArea", area)
        owner = option.get("playerIndex", actor)
        row = [
            cur["turn"], int(cur["firstPlayer"] == actor), int(cur["supporterPlayed"]),
            int(cur["energyAttached"]), me["handCount"], me["deckCount"], len(me["prize"]),
            opponent["handCount"], opponent["deckCount"], len(opponent["prize"]),
            len(me["bench"]), len(opponent["bench"]), int(me["asleep"]),
            int(me["paralyzed"]), sel["minCount"], sel["maxCount"], len(sel["option"]),
            sel["remainDamageCounter"], int(sel.get("deck") is not None),
            int(cur.get("looking") is not None), len(prefix), sel["maxCount"] - len(prefix),
            int(stop), option.get("number", 0), int(card is not None and owner == actor),
            int(area == 4), int(area == 5), int(target_area == 4), int(target_area == 5),
            int(target is not None and (owner != actor or kind == 13)),
        ]
        for value in (card, target, my_active, op_active):
            row.extend(self.card_values(value))
        row.extend(int(sel["context"] == i) for i in range(48))
        row.extend(int(kind == i) for i in range(18))
        row.extend(int(area == i) for i in range(1, 13))
        row.extend(int(option.get("attackId") == a) for a in self.profile.attack_ids)
        zones = [me["hand"], me["discard"], me["active"], me["bench"], sel.get("deck") or [],
                 [card], [target], [self.candidate(obs, sel["option"][i]) for i in prefix],
                 [sel.get("effect")], [sel.get("contextCard")]]
        for cards in zones:
            counts = Counter(c["id"] for c in cards if c)
            row.extend(counts[i] for i in self.ids)
        # Deck block. Upstream semantics, deck-derived values:
        #   multi-prize attacker in play, # of them, # evolved, bench ready now,
        #   bench ready with +1 energy, deck-type energy in discard, energy on
        #   board, min(3, discard energy), attack damage, damage - HP, KO, KO
        #   that takes the last prizes.
        board = [c for c in me["active"] + me["bench"] if c]
        multi = sum(1 for c in board if self.cards[c["id"]]["ex"] or self.cards[c["id"]]["megaEx"])
        evolved = sum(1 for c in board if self.cards[c["id"]]["stage1"] or self.cards[c["id"]]["stage2"])
        etype = self.profile.energy_type
        in_discard = sum(1 for c in me["discard"]
                         if c and self.cards[c["id"]]["cardType"] == 5
                         and self.cards[c["id"]]["energyType"] == etype)
        damage = self._damage(option.get("attackId") if kind == 13 else None,
                              my_active, op_active)
        op_hp = (op_active or {}).get("hp", 0)
        ko = damage > 0 and op_hp > 0 and damage >= op_hp
        op_prize = self.card_values(op_active)[-1]
        row.extend([
            int(multi > 0), multi, evolved,
            sum(self._ready(c, 0) for c in me["bench"] if c),
            sum(self._ready(c, 1) for c in me["bench"] if c),
            in_discard, sum(len(c.get("energies", [])) for c in board), min(3, in_discard),
            damage, damage - op_hp if damage else 0, int(ko),
            int(ko and len(me["prize"]) <= op_prize),
        ])
        if len(row) != self.layout.width:
            raise ValueError(f"feature width {len(row)} != {self.layout.width}")
        return row

    # ---- tokens -----------------------------------------------------------

    @staticmethod
    def options_for(obs: dict[str, Any], prefix: Sequence[int]) -> list[int]:
        """Remaining legal option indices, then STOP (-1) when allowed."""
        sel = obs["select"]
        if len(prefix) > sel["maxCount"] or len(set(prefix)) != len(prefix):
            raise ValueError("Invalid selection prefix")
        if any(i < 0 or i >= len(sel["option"]) for i in prefix):
            raise ValueError("Prefix is outside legal options")
        if len(prefix) == sel["maxCount"]:
            return []
        values = [i for i in range(len(sel["option"])) if i not in prefix]
        if len(prefix) >= sel["minCount"]:
            values.append(-1)  # STOP follows real candidates, including on ties.
        return values

    @staticmethod
    def token_indices(obs: dict[str, Any], options: Sequence[int]) -> list[int]:
        """Original option indices and STOP -> token rows."""
        return [19 + i if i >= 0 else 19 + len(obs["select"]["option"]) for i in options]

    def tokenize(self, obs: dict[str, Any], prefix: Sequence[int] = ()) -> np.ndarray:
        """Public state and option tokens; rejects rather than truncates overflow."""
        L = self.layout
        cur, sel = obs["current"], obs["select"]
        actor = cur["yourIndex"]
        count = len(sel["option"])
        if count > MAX_OPTIONS:
            raise ValueError(f"Transformer option overflow: maximum {MAX_OPTIONS}")
        legal = self.options_for(obs, prefix)
        x = np.zeros((20 + count, WIDTH), np.float32)
        x[:, 4:6] = -1

        def slot(c: dict[str, Any] | None) -> int:
            return self.slots.get((c or {}).get("id"), -1)

        def counts(cards: Sequence[dict[str, Any] | None]) -> np.ndarray:
            a = np.zeros(32, np.float32)
            for c in cards:
                s = slot(c)
                if s >= 0:
                    a[s] += 1
            return a

        def generic(row: int, values: Sequence[float]) -> None:
            v = np.asarray(values, np.float32)
            x[row, 6:6 + len(v)] = np.sign(v) * np.log1p(np.abs(v))

        base = self.feature_row(obs, -1, [])
        x[0, 0] = 1
        x[0, 4] = slot(sel.get("effect"))
        x[0, 5] = slot(sel.get("contextCard"))
        generic(0, base[:20] + base[L.context:L.kind] + base[L.deck:L.deck + 8])
        me = cur["players"][actor]
        op = cur["players"][1 - actor]
        known = list(me["hand"]) + list(me["discard"])
        for side, ps in enumerate((me, op)):
            board = list(ps["active"])[:1] + [None] * (not ps["active"]) + list(ps["bench"])
            if len(board) > 6:
                raise ValueError("Board exceeds 6 slots per side")
            for j in range(6):
                r = 1 + 6 * side + j
                x[r, 1] = 1
                c = board[j] if j < len(board) else None
                if not c:
                    continue
                x[r, 0] = 1
                x[r, 4] = slot(c)
                generic(r, [side, int(j == 0)] + self.card_values(c))
                attached = c.get("energyCards", []) + c.get("tools", [])
                evolution = c.get("preEvolution", [])
                x[r, 102:134] = counts(attached) / 60
                x[r, 134:166] = counts(evolution) / 60
                if side == 0:
                    known += [c] + attached + evolution
        # Only publicly identified own cards are subtracted. The unknown pool is
        # split between deck and prizes; hidden identities are never read.
        known += [c for c in cur.get("stadium", []) if c and c.get("playerIndex") == actor]
        remaining = np.zeros(32, np.float32)
        for card, n in self.profile.counts.items():
            remaining[self.slots[card]] = n
        remaining = np.maximum(0, remaining - counts(known))
        total = me["deckCount"] + len(me["prize"])
        exact = sel.get("deck")
        exact_ok = (exact is not None and len(exact) == me["deckCount"]
                    and all(c for c in exact) and np.all(counts(exact) <= remaining))
        deck = counts(exact) if exact_ok else remaining * (me["deckCount"] / max(1, total))
        prize = remaining - deck if exact_ok else remaining * (len(me["prize"]) / max(1, total))
        zones = [counts(me["hand"]), counts(me["discard"]), deck, prize,
                 counts(exact or []), counts(op["discard"])]
        for z, values in enumerate(zones):
            r = 13 + z
            x[r, 0] = int(z != 4 or exact is not None)
            x[r, 1] = 2
            generic(r, [int(z == i) for i in range(6)]
                    + [float(values.sum()), int(z in (2, 3)), int(exact_ok and z in (2, 3))])
            x[r, 102:134] = values / 60
        for i, o in enumerate(sel["option"]):
            r = 19 + i
            x[r, 0] = 1
            x[r, 1] = 3
            x[r, 2] = int(i in legal)
            x[r, 3] = int(i in prefix)
            x[r, 4] = slot(self.candidate(obs, o))
            f = self.feature_row(obs, i, [])
            # 22..59: option state tail + the candidate's and target's card
            # values (upstream f[22:60]); then kind/area/attack one-hots.
            generic(r, f[22:L.cards + 2 * CARD_FIELDS] + f[L.kind:L.zones]
                    + f[L.deck:L.deck + 12])
            # Approximate public damage, explicitly uncertain (no text effects).
            result = np.zeros(32, np.float32)
            if o["type"] == 13:
                result[0] = 1
                result[3] = 1
                result[29] = f[L.deck + 8] / 600
                result[30] = f[L.deck + 10]
                result[31] = f[L.deck + 11]
            x[r, 166:198] = result
        x[-1, 0] = 1
        x[-1, 1] = 4
        x[-1, 2] = int(-1 in legal)
        return x

    def with_prefix(self, tokens: np.ndarray, obs: dict[str, Any],
                    prefix: Sequence[int]) -> np.ndarray:
        """Encoder inputs are prefix-independent; only the pointer flags change."""
        result = tokens.copy()
        result[:, 2:4] = 0
        result[self.token_indices(obs, self.options_for(obs, prefix)), 2] = 1
        if prefix:
            result[[19 + i for i in prefix], 3] = 1
        return result
