"""Frequency of the three P5 defects, before anyone writes a rule for them.

All three came from the user watching live games. This does not fix anything --
it sizes each one, so the next session knows which are worth an arena A/B.

    python -X utf8 scripts/p5_audit.py --matches 200

P5a  Adrena-Brain aims at the cheapest target, not the most valuable one it
     could finish with the turn's POOLED budget. Two Munkidori = two
     activations = 60 damage; `targeting.chip_target` hardcodes CHIP_DAMAGE=30
     and evaluates each select alone, so a 60-HP Lucario ex (2 prizes, and
     their attacker) reads as unkillable and it takes the 50-HP Budew instead.
     Counted here: selects where nothing dies to 30, something dies to the
     pooled budget, and that something is worth more prizes than our pick.

P5b  Boss's Orders played with NOTHING on their bench we can KO. The drag then
     just promotes a Pokemon they wanted promoted -- the user watched one get
     evolved into their main attacker. Note P4a tested *forcing* the play and
     *aiming* the drag, both null; it never tested VETOING the play.

P5c  Ending a turn with a payable attack unused. Impidimp's Filch is {C} for 0
     damage and draws a card, so passing instead of Filching is strictly worse.
     Split the finding: this is the free half. Investing fresh energy into a
     fragile Morgrem to enable Corkscrew Punch is a tradeoff, not a dominated
     option, and per rule 10 it should not become a rule without its own A/B.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from ptcg.env import harness  # noqa: E402
from sa import cards as cdb  # noqa: E402
from sa.targeting import best_damage  # noqa: E402
import arena  # noqa: E402

MAIN, DAMAGE_COUNTER, DAMAGE_COUNTER_ANY = 0, 13, 14
OPT_PLAY, OPT_ABILITY, OPT_ATTACK, OPT_END = 7, 10, 13, 14
_HAND, _ACTIVE, _BENCH = 2, 4, 5
MUNKIDORI, DARK_TYPE, BOSS_ORDERS = 112, 7, 1182
PER_ACTIVATION = 30  # 3 damage counters


def _mine(pl):
    out = {}
    act = pl["active"]
    if act and act[0] is not None:
        out[(_ACTIVE, 0)] = act[0]
    for i, pk in enumerate(pl["bench"]):
        if pk is not None:
            out[(_BENCH, i)] = pk
    return out


class Probe:
    def __init__(self, inner):
        self.inner = inner
        self.game = 0
        self.a = Counter()   # P5a
        self.a2 = Counter()  # P5a: was there a choice at all?
        self.b = Counter()   # P5b
        self.c = Counter()   # P5c
        self.c_dmg = Counter()
        self._turn = None
        self._used = set()

    def __call__(self, obs):
        picked = self.inner(obs)
        try:
            sel = obs.get("select") or {}
            state = obs.get("current") or {}
            if not sel or not state:
                return picked
            me = state["yourIndex"]
            mypl, oppl = state["players"][me], state["players"][1 - me]
            ctx = sel.get("context")
            opts = sel.get("option") or []
            chosen = list(picked)[:1]

            key = (self.game, state["turn"], me)
            if key != self._turn:
                self._turn, self._used = key, set()

            if ctx == MAIN:
                self._main(sel, opts, chosen, mypl, oppl, me, state)
            elif ctx in (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY):
                self._chip(sel, opts, chosen, mypl, oppl, me, state)
        except Exception as e:  # noqa: BLE001
            self.a[f"ERR {type(e).__name__}: {e}"] += 1
        return picked

    # -- P5b and P5c ------------------------------------------------------
    def _main(self, sel, opts, chosen, mypl, oppl, me, state):
        # remember which Munkidori have already activated this turn
        for i in chosen:
            if 0 <= i < len(opts) and opts[i].get("type") == OPT_ABILITY:
                slot = (opts[i].get("area"), opts[i].get("index") or 0)
                pk = _mine(mypl).get(slot)
                if pk and pk["id"] == MUNKIDORI:
                    self._used.add(slot)

        active = mypl["active"][0] if mypl.get("active") else None
        hand = mypl["hand"]

        # P5b: did we play Boss's Orders with no KO-able bench target?
        boss = [i for i, o in enumerate(opts)
                if o.get("type") == OPT_PLAY
                and 0 <= (o.get("index") or 0) < len(hand)
                and hand[o.get("index") or 0]
                and hand[o.get("index") or 0]["id"] == BOSS_ORDERS]
        if boss and any(i in chosen for i in boss):
            koable = any(
                pk and best_damage(active, mypl, oppl, pk) >= pk["hp"]
                for pk in (oppl.get("bench") or []))
            self.b["played, a KO-able target existed" if koable
                   else "played, NOTHING on their bench was KO-able"] += 1

        # P5c: did we end the turn with a payable attack on the table?
        ends = [i for i, o in enumerate(opts) if o.get("type") == OPT_END]
        atks = [i for i, o in enumerate(opts) if o.get("type") == OPT_ATTACK]
        if ends and atks:
            took_end = any(i in chosen for i in ends)
            self.c["ended the turn with an attack available" if took_end
                   else "attacked"] += 1
            if took_end and active:
                opp_act = oppl["active"][0] if oppl.get("active") else None
                dmg = (best_damage(active, mypl, oppl, opp_act)
                       if opp_act else 0.0)
                name = cdb.card(active["id"]).get("name", "?")
                self.c_dmg[f"{name} could have done {int(dmg)}"] += 1

    # -- P5a --------------------------------------------------------------
    def _chip(self, sel, opts, chosen, mypl, oppl, me, state):
        cand = {}
        for i, o in enumerate(opts):
            if o.get("playerIndex") in (None, me):
                return                       # mixed select: not our case
            pk = self._at(state, 1 - me, o.get("area"), o.get("index") or 0)
            if pk is None or pk.get("hp") is None:
                return
            cand[i] = pk
        if len(cand) < 2:
            return

        mine = _mine(mypl)
        armed = {s for s, pk in mine.items()
                 if pk["id"] == MUNKIDORI
                 and any(e == DARK_TYPE for e in (pk.get("energies") or []))}
        # The Munkidori being activated RIGHT NOW is already in `_used`: the
        # MAIN select that fired the ability precedes this DAMAGE_COUNTER
        # select. So `armed - _used` is what is left AFTERWARDS, and the pool
        # is this activation plus those. The old `max(1, len(armed - _used))`
        # read 1 on every row ever measured -- including all 54 replay rows
        # with two armed Munkidori -- so a 60-point pool was never once
        # representable, and P5a's headline "0 misses" was measuring a budget
        # that could not exceed 30. Rule 9: check the denominator.
        left = 1 + len(armed - self._used)
        # you can only move counters you actually have on your own board
        own = sum(max(0, (pk.get("maxHp") or 0) - pk["hp"]) for pk in
                  mine.values())
        budget = min(PER_ACTIVATION * left, own)

        dies_now = [i for i, pk in cand.items()
                    if pk["hp"] <= PER_ACTIVATION]
        if dies_now:
            self.a["something already dies to one activation"] += 1
            return
        pooled = [i for i, pk in cand.items() if pk["hp"] <= budget]
        if not pooled:
            self.a["nothing dies even with the pooled budget"] += 1
            return
        best = max(cdb.prize_value(cand[i]["id"]) for i in pooled)
        c = chosen[0] if chosen and chosen[0] in cand else None
        got = cdb.prize_value(cand[c]["id"]) if c is not None else 0
        # A "miss" needs a CHOICE: two pooled targets whose prize values
        # differ. With one pooled candidate `best == got` by construction and
        # the row is vacuous, so count the live denominator separately --
        # otherwise "80/80 correct" is 80 forced moves, not 80 good ones.
        if len({cdb.prize_value(cand[i]["id"]) for i in pooled}) > 1:
            self.a2["pooled KOs where the prize values DIFFER"] += 1
        else:
            self.a2["only one prize value among the pooled KOs "
                    "(no choice to get wrong)"] += 1
        if c is not None and c in pooled and got >= best:
            self.a["pooled KO taken, best prizes"] += 1
        elif best > got:
            self.a[f"MISS: pooled {best}-prize KO available, took "
                   f"{got}-prize"] += 1
        else:
            self.a["pooled KO available, not taken"] += 1

    @staticmethod
    def _at(state, player, area, index):
        try:
            pl = state["players"][player]
            if area == _ACTIVE:
                act = pl["active"]
                return act[0] if act and act[0] is not None else None
            if area == _BENCH:
                b = pl["bench"]
                return b[index] if 0 <= index < len(b) else None
        except (KeyError, IndexError, TypeError):
            return None
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="bc")
    ap.add_argument("--opponent", default="rule:v10,noS")
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--deck-b", default="lucario_v10")
    ap.add_argument("--matches", type=int, default=200)
    args = ap.parse_args()

    _, deck_a = arena.resolve_deck(args.deck)
    _, deck_b = arena.resolve_deck(args.deck_b)
    _, agent_a = arena.build_agent(args.agent, deck_a)
    _, agent_b = arena.build_agent(args.opponent, deck_b)
    probe = Probe(agent_a)

    for m in range(args.matches):
        probe.game = m
        if m % 2 == 0:
            harness.play_game(probe, agent_b, list(deck_a), list(deck_b))
        else:
            harness.play_game(agent_b, probe, list(deck_b), list(deck_a))

    def show(title, counter, note=""):
        tot = sum(counter.values())
        print(f"\n{title}  (n={tot})")
        for k, v in counter.most_common():
            print(f"  {k:<52}{v:>7}{v / tot:>8.1%}" if tot else f"  {k}")
        if note:
            print(f"  {note}")

    print(f"\n=== {args.matches} games, {args.agent} [{args.deck}] "
          f"vs {args.opponent} [{args.deck_b}] ===")
    show("P5a  Adrena-Brain target vs the turn's POOLED budget", probe.a)
    show("P5a  ... and how many of those pooled KOs were a real choice",
         probe.a2)
    show("P5b  Boss's Orders plays", probe.b)
    show("P5c  turns with a payable attack on the table", probe.c)
    show("P5c  what the unused attack would have done", probe.c_dmg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
