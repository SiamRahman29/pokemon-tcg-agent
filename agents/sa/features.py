"""State featurization shared by the value-net trainer and the agent.

`featurize(state, me)` -> (dense float32 vector, id-bag int32 arrays).
The id bags are embedded by the net (sum of embedding rows), so the feature
layout here and the net architecture must move together. Bump VERSION when
changing either.
"""
from __future__ import annotations

import numpy as np

from . import cards as cdb
from .textdmg import best_estimated_damage

VERSION = 2
N_CARD_IDS = 1300          # card id space (ids are 1..1267 today)
N_SLOTS = 12               # my active, my bench x5, opp active, opp bench x5
PER_SLOT = 18
N_GLOBAL = 26
DENSE_DIM = N_GLOBAL + N_SLOTS * PER_SLOT

# id bags: per-slot card id (12), my hand, my discard, opp discard, opp known
BAG_NAMES = ("slots", "my_hand", "my_discard", "opp_discard")


def _slot_feats(pk: dict | None, mypl: dict, oppl: dict, out: np.ndarray,
                base: int) -> int:
    """Write PER_SLOT features for one pokemon slot; return its card id."""
    if pk is None:
        return 0
    c = cdb.card(pk["id"])
    hp = pk["hp"]
    max_hp = pk["maxHp"] or 1
    out[base + 0] = 1.0
    out[base + 1] = hp / 300.0
    out[base + 2] = max_hp / 300.0
    out[base + 3] = 1.0 - hp / max_hp
    out[base + 4] = 1.0 if c.get("basic") else 0.0
    out[base + 5] = 1.0 if c.get("stage1") else 0.0
    out[base + 6] = 1.0 if c.get("stage2") else 0.0
    out[base + 7] = 1.0 if c.get("ex") else 0.0
    out[base + 8] = 1.0 if c.get("megaEx") else 0.0
    out[base + 9] = 1.0 if c.get("tera") else 0.0
    out[base + 10] = min(len(pk["energies"]), 6) / 6.0
    out[base + 11] = (c.get("retreatCost") or 0) / 4.0
    out[base + 12] = 1.0 if pk["appearThisTurn"] else 0.0
    out[base + 13] = min(best_estimated_damage(pk, mypl, oppl), 400) / 400.0
    out[base + 14] = len(pk["tools"]) / 1.0 if pk["tools"] else 0.0
    out[base + 15] = cdb.prize_value(pk["id"]) / 3.0
    out[base + 16] = _cost_satisfaction(pk)
    own = c.get("energyType")
    out[base + 17] = min(sum(1 for e in pk["energies"]
                             if e == own or e >= 10), 4) / 4.0
    return pk["id"]


def _cost_satisfaction(pk: dict) -> float:
    """How close the attached energy comes to paying this Pokemon's cheapest
    attack (1.0 = can attack now)."""
    best = 0.0
    have = pk["energies"]
    for aid in cdb.card(pk["id"]).get("attacks") or []:
        a = cdb.attacks().get(aid)
        if not a:
            continue
        cost = a["energies"]
        if not cost:
            return 1.0
        if cdb.energy_satisfied(cost, have):
            return 1.0
        best = max(best, min(len(have), len(cost)) / len(cost))
    return best


def featurize(state: dict, me: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    opp = 1 - me
    mypl = state["players"][me]
    oppl = state["players"][opp]

    dense = np.zeros(DENSE_DIM, dtype=np.float32)
    g = 0

    def put(v):
        nonlocal g
        dense[g] = v
        g += 1

    put(min(state["turn"], 40) / 40.0)
    put(1.0 if state["firstPlayer"] == me else 0.0)
    put((6 - len(mypl["prize"])) / 6.0)      # my prizes taken
    put((6 - len(oppl["prize"])) / 6.0)
    put(len(mypl["prize"]) / 6.0)
    put(len(oppl["prize"]) / 6.0)
    put(min(mypl["deckCount"], 60) / 60.0)
    put(min(oppl["deckCount"], 60) / 60.0)
    put(1.0 if mypl["deckCount"] == 0 else 0.0)
    put(1.0 if oppl["deckCount"] == 0 else 0.0)
    put(min(mypl["handCount"], 12) / 12.0)
    put(min(oppl["handCount"], 12) / 12.0)
    put(1.0 if state["supporterPlayed"] else 0.0)
    put(1.0 if state["energyAttached"] else 0.0)
    for pl in (mypl, oppl):
        put(1.0 if pl["poisoned"] else 0.0)
        put(1.0 if pl["burned"] else 0.0)
        put(1.0 if pl["asleep"] else 0.0)
        put(1.0 if pl["paralyzed"] else 0.0)
        put(1.0 if pl["confused"] else 0.0)
    put(sum(1 for pk in mypl["bench"] if pk is not None) / 5.0)
    put(sum(1 for pk in oppl["bench"] if pk is not None) / 5.0)
    assert g == N_GLOBAL, g

    slot_ids = np.zeros(N_SLOTS, dtype=np.int32)
    slots = ([mypl["active"][0] if mypl["active"] else None]
             + [(mypl["bench"][i] if i < len(mypl["bench"]) else None)
                for i in range(5)]
             + [oppl["active"][0] if oppl["active"] else None]
             + [(oppl["bench"][i] if i < len(oppl["bench"]) else None)
                for i in range(5)])
    for si, pk in enumerate(slots):
        side_my, side_opp = (mypl, oppl) if si < 6 else (oppl, mypl)
        cid = _slot_feats(pk, side_my, side_opp,
                          dense, N_GLOBAL + si * PER_SLOT)
        slot_ids[si] = cid if cid < N_CARD_IDS else 0

    def bag(cards_list) -> np.ndarray:
        return np.asarray([c["id"] for c in cards_list
                           if c is not None and c["id"] < N_CARD_IDS],
                          dtype=np.int32)

    bags = {
        "slots": slot_ids,
        "my_hand": bag(mypl["hand"] or []),
        "my_discard": bag(mypl["discard"]),
        "opp_discard": bag(oppl["discard"]),
    }
    return dense, bags
