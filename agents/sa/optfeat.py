"""Per-option features for the policy net.

Resolves what card/attack an option refers to and produces:
  * dense per-option vector (type one-hot + misc scalars)
  * card id (for embedding), attack id (for embedding)
Layout must stay in sync with policy trainer/inference. Bump VERSION on change.
"""
from __future__ import annotations

import numpy as np

VERSION = 2

N_OPTION_TYPES = 17
OPT_DENSE = N_OPTION_TYPES + 8
N_ATTACK_IDS = 1600  # option_features returns (dense, card_id, attack_id, target_id)

# AreaType ints
_DECK, _HAND, _DISCARD, _ACTIVE, _BENCH, _PRIZE, _STADIUM = 1, 2, 3, 4, 5, 6, 7
_LOOKING = 12


def _card_at(state: dict, sel: dict, player: int, area: int,
             index: int) -> int:
    """Best-effort card id at (player, area, index); 0 if unknown."""
    try:
        if area == _LOOKING:
            look = state.get("looking")
            if look and index < len(look) and look[index]:
                return look[index]["id"]
            return 0
        if area == _DECK:
            deck = sel.get("deck")
            if deck and index < len(deck) and deck[index]:
                return deck[index]["id"]
            return 0
        pl = state["players"][player]
        if area == _HAND:
            hand = pl.get("hand")
            if hand and index < len(hand) and hand[index]:
                return hand[index]["id"]
            return 0
        if area == _DISCARD:
            if index < len(pl["discard"]):
                return pl["discard"][index]["id"]
            return 0
        if area == _ACTIVE:
            act = pl["active"]
            if act and act[0] is not None:
                return act[0]["id"]
            return 0
        if area == _BENCH:
            if index < len(pl["bench"]) and pl["bench"][index] is not None:
                return pl["bench"][index]["id"]
            return 0
        if area == _PRIZE:
            pr = pl["prize"]
            if index < len(pr) and pr[index] is not None:
                return pr[index]["id"]
            return 0
        if area == _STADIUM:
            st = state.get("stadium") or []
            if st:
                return st[0]["id"]
            return 0
    except (KeyError, IndexError, TypeError):
        return 0
    return 0


def option_features(obs: dict, opt: dict) -> tuple[np.ndarray, int, int, int]:
    """-> (dense vector, card_id, attack_id, target_id) for one option.
    target_id = the in-play Pokemon an ATTACH/EVOLVE points at."""
    state = obs["current"]
    sel = obs["select"]
    me = state["yourIndex"]

    dense = np.zeros(OPT_DENSE, dtype=np.float32)
    t = opt.get("type") or 0
    if t < N_OPTION_TYPES:
        dense[t] = 1.0
    x = N_OPTION_TYPES

    card_id = 0
    attack_id = 0
    target_id = 0

    area = opt.get("area")
    index = opt.get("index") or 0
    player = opt.get("playerIndex")
    player = me if player is None else player

    if t == 7:  # PLAY: index into my hand
        card_id = _card_at(state, sel, me, _HAND, opt.get("index") or 0)
    elif t in (8, 9):  # ATTACH / EVOLVE: card at (area,index) onto target
        card_id = _card_at(state, sel, me, area or _HAND, index)
        target_id = _card_at(state, sel, me, opt.get("inPlayArea") or 0,
                             opt.get("inPlayIndex") or 0)
    elif t in (3, 10, 11):  # CARD / ABILITY / DISCARD
        card_id = _card_at(state, sel, player, area or 0, index)
    elif t == 13:  # ATTACK
        attack_id = opt.get("attackId") or 0
        act = state["players"][me]["active"]
        if act and act[0] is not None:
            card_id = act[0]["id"]
        opp_act = state["players"][1 - me]["active"]
        if opp_act and opp_act[0] is not None:
            target_id = opp_act[0]["id"]
    elif t == 15:  # SKILL
        card_id = opt.get("cardId") or 0
    elif t in (4, 5, 6):  # TOOL_CARD / ENERGY_CARD / ENERGY on a pokemon
        card_id = _card_at(state, sel, player, area or 0, index)

    dense[x + 0] = 1.0 if player == me else 0.0
    dense[x + 1] = (opt.get("number") or 0) / 10.0
    dense[x + 2] = 1.0 if area == _ACTIVE else 0.0
    dense[x + 3] = 1.0 if area == _BENCH else 0.0
    dense[x + 4] = 1.0 if area == _HAND else 0.0
    dense[x + 5] = (opt.get("energyIndex") or 0) / 5.0
    ipa = opt.get("inPlayArea")
    dense[x + 6] = 1.0 if ipa == _ACTIVE else 0.0
    dense[x + 7] = 1.0 if ipa == _BENCH else 0.0

    if attack_id >= N_ATTACK_IDS:
        attack_id = 0
    return dense, card_id, attack_id, target_id
