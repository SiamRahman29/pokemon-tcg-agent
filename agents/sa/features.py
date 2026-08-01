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

VERSION = 3
N_CARD_IDS = 1300          # card id space (ids are 1..1267 today)
N_SLOTS = 12               # my active, my bench x5, opp active, opp bench x5
PER_SLOT = 18
N_GLOBAL = 26
DENSE_DIM = N_GLOBAL + N_SLOTS * PER_SLOT

# --- the v4 block (day 12), APPENDED at the very end of the state vector ----
# `scripts/p18_missing_state_audit.py` enumerated every field of the
# observation that `featurize()` never reads and measured how much each varies
# at a real decision point. It killed three candidates that had been on the
# plan for two days -- opponent hand size, prizes remaining and turn number are
# ALL already encoded above (lines "put(...)" below) -- and two it found
# itself: `remainDamageCounter` is 0 at 100% of decisions and `remainEnergyCost`
# at 99.1%, so neither can explain a single miss.
#
# What survived, with the miss mass it targets (v3 net, 12,939 held-out rows):
#   * turnActionCount -- 20 distinct values, modal share 17% in MAIN.
#     MAIN is 2,629 of 3,902 misses. The net re-scores a barely-changed board
#     several times per turn with no idea how deep into the turn it is.
#   * the select's EFFECT card -- which card caused this select. Modal share
#     26% in TO_HAND (674 misses): the same context means "tutor a Trainer"
#     (Petrel), "take a Supporter" (Poke Pad), "recover from discard" (Night
#     Stretcher) or "search anything" (Ultra Ball), and the net scores each
#     option INDEPENDENTLY, so it never sees the option set that would reveal
#     which. This is the one input that tells it what kind of choice it is.
#   * the stadium -- 7 distinct, 61% Spikemuth Gym, and Area Zero Underdepths
#     changes the bench size. Absent entirely, including from every id bag.
#   * `retreated` / `stadiumPlayed` -- the two missing members of the
#     once-per-turn quartet whose other two (`supporterPlayed`,
#     `energyAttached`) have been encoded since v1. `retreated` is 43%
#     non-modal in SWITCH.
#
# ⚠ APPENDED, NEVER INSERTED -- and the append lands after `seld`, which is the
# LAST block of the state vector (see policynet.scores / train_policy.forward).
# A v3 net simply slices to its own `state_in` and reads byte-identical input,
# which is what lets v3 and v4 run head-to-head in one process (rule 4).
N_EXTRA = 8                # dense scalars
N_XSLOT = 2                # card ids embedded through the existing slot table:
#                            (stadium in play, the select's effect card)


def extra_feats(state: dict, sel: dict,
                me: int) -> tuple[np.ndarray, np.ndarray]:
    """The v4 block: (dense scalars, card ids to embed). See the note above."""
    x = np.zeros(N_EXTRA, dtype=np.float32)
    mypl, oppl = state["players"][me], state["players"][1 - me]
    stad = state.get("stadium") or []

    def tools(pl) -> int:
        n = 0
        for pk in ([pl["active"][0] if pl["active"] else None]
                   + list(pl["bench"])):
            if pk:
                n += len(pk.get("tools") or [])
        return n

    x[0] = min(int(state.get("turnActionCount") or 0), 24) / 24.0
    x[1] = 1.0 if state.get("retreated") else 0.0
    x[2] = 1.0 if state.get("stadiumPlayed") else 0.0
    x[3] = 1.0 if stad else 0.0
    x[4] = (mypl.get("benchMax") or 5) / 8.0
    x[5] = min(tools(mypl), 4) / 4.0
    x[6] = min(tools(oppl), 4) / 4.0
    x[7] = min(len(sel.get("deck") or []), 60) / 60.0

    eff = sel.get("effect")
    ids = np.zeros(N_XSLOT, dtype=np.int32)
    ids[0] = stad[0]["id"] if stad else 0
    ids[1] = (eff or {}).get("id", 0) if isinstance(eff, dict) else 0
    ids[ids >= N_CARD_IDS] = 0
    return x, ids

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
