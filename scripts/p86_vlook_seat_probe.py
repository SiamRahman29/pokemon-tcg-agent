#!/usr/bin/env python
"""Is `players[me]` still OUR board after `fs.step`? E20's seat probe.

The `vlp` smoke read **0 wins in 10 games** with errors=0 and a 77% overrule
rate -- the signature `vlook.py`'s docstring predicts for a sign-inverted V.
Before changing anything, establish the fact: after one `fs.step`, does the
successor observation index players ABSOLUTELY (so our pre-step `me` stays
valid) or RELATIVE to whoever is now to move (so it does not)?

Decides it structurally, with no reference to V: our own Active Pokemon's card
id should not change because we played one option, whereas if the array is
re-oriented `players[me]` becomes the OPPONENT's board and the id jumps to
theirs.

    python -X utf8 scripts/p86_vlook_seat_probe.py --games 6
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "."):
    sys.path.insert(0, str(ROOT / sub))

from ptcg.env import sdk  # noqa: E402

sdk.load()

from ptcg.env import harness  # noqa: E402
from sa import fastsearch as fs  # noqa: E402
from sa import policynet as pnet  # noqa: E402
from sa.worlds import determinize  # noqa: E402

MAIN = 0
STATS: Counter = Counter()


def active_id(state, p):
    a = state["players"][p]["active"]
    return a[0]["id"] if a else None


class Probe:
    def __init__(self, decklist):
        self.decklist = list(decklist)
        self.rng = random.Random(0)

    def __call__(self, obs):
        net = pnet.get()
        sel = obs.get("select")
        if sel is None:
            return list(self.decklist)
        picked = net.choose(obs) if net else list(range(sel.get("minCount", 1)))
        try:
            cur = obs.get("current") or {}
            if (sel.get("context") == MAIN and obs.get("search_begin_input")
                    and cur.get("result", -1) == -1
                    and sel.get("maxCount", 1) == 1
                    and len(sel.get("option") or []) >= 2):
                me = cur["yourIndex"]
                my_before = active_id(cur, me)
                opp_before = active_id(cur, 1 - me)
                if my_before is None:
                    return picked
                w = determinize(obs, self.decklist, [], self.rng)
                root, o = fs.begin(
                    obs["search_begin_input"],
                    [] if sel.get("deck") is not None else w.my_deck,
                    w.my_prize, w.opp_deck, w.opp_prize, w.opp_hand,
                    w.opp_active)
                try:
                    _sid, o2 = fs.step(root, [picked[0]])
                    c2 = o2.get("current")
                    if c2 is not None and c2.get("result", -1) == -1:
                        STATS["steps"] += 1
                        if c2["yourIndex"] != me:
                            STATS["mover_changed"] += 1
                        # THE TEST: with absolute indexing, players[me] is
                        # still us, so our Active id is unchanged (we did not
                        # necessarily switch). With relative indexing it
                        # becomes the opponent's board.
                        my_after = active_id(c2, me)
                        if my_after == my_before:
                            STATS["me_still_mine"] += 1
                        elif my_after == opp_before:
                            STATS["me_became_opponent"] += 1
                        else:
                            STATS["me_changed_other"] += 1
                        if c2["yourIndex"] != me:
                            a = active_id(c2, c2["yourIndex"])
                            if a == opp_before:
                                STATS["mover_slot_is_opp"] += 1
                            elif a == my_before:
                                STATS["mover_slot_is_mine"] += 1
                finally:
                    fs.end()
        except Exception as e:
            STATS["errors"] += 1
            print("ERR", type(e).__name__, e, file=sys.stderr)
        return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    args = ap.parse_args()

    import importlib
    deckmod = importlib.import_module("decks.grimmsnarl")
    deck = [cid for cid, n in deckmod.DECKLIST.items() for _ in range(n)]

    for g in range(args.games):
        harness.play_game(Probe(deck), Probe(deck), deck, deck)
        print(f"game {g}: steps={STATS['steps']} "
              f"mine={STATS['me_still_mine']} opp={STATS['me_became_opponent']}",
              flush=True)

    s = STATS
    n = max(s["steps"], 1)
    print("\n=== seat semantics after one fs.step ===")
    print(f"steps observed            {s['steps']}")
    print(f"mover changed seat        {s['mover_changed']} "
          f"({s['mover_changed']/n:.0%})")
    print(f"players[me] still OURS    {s['me_still_mine']} "
          f"({s['me_still_mine']/n:.0%})")
    print(f"players[me] is OPPONENT   {s['me_became_opponent']} "
          f"({s['me_became_opponent']/n:.0%})")
    print(f"players[me] some other id {s['me_changed_other']} "
          f"({s['me_changed_other']/n:.0%})")
    print(f"errors                    {s['errors']}")
    print("\nVERDICT: " + (
        "ABSOLUTE -- pre-step `me` stays valid; the 0/10 is NOT the seat"
        if s["me_still_mine"] > s["me_became_opponent"] else
        "RELATIVE -- pre-step `me` indexes the OPPONENT after control passes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
