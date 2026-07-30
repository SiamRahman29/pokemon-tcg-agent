"""Does the "Morgrem out" decision actually EXIST? (HANDOFF §3.3b, EVIDENCE §8d)

**The claim to size.** Against a Crustle board our main attacker deals **zero**
(Mysterious Rock Inn prevents damage from opponent {ex} attacks; verified at
n=224, 93.3% zeroed) while **Marnie's Morgrem -- a NON-ex attacker -- deals 60**
(verified: the only 15 attack-damage events that landed on a Crustle were all
Morgrem's Corkscrew Punch, 60 each). We already run 3 Morgrem as the evolution
stage, so a play-priority rule "do not evolve Morgrem into Grimmsnarl ex into a
Crustle board" would cost no decklist change.

**Before writing that rule, count its denominator (rule 13).** A rate over moves
that were forced measures nothing: P2's lethal audit read "316/316 lethals taken"
and every one of the 316 was the only attack offered, honest denominator 0. So
this script asks, in order:

  1. How many of our MAIN selects happen while their Active is a wall at all?
     (Known-in-advance bucket, per rule 9: vs `rule:crustle` this must be LARGE
     -- their whole plan is to sit behind it. If it prints small, the probe is
     broken, not the meta.)
  2. Of those, how often is our Active a Morgrem that the engine is offering an
     ATTACK option for -- i.e. the 60 is actually available to us right now?
  3. Of THOSE, how often is an evolve-onto-that-Morgrem option also on the menu?
     **That intersection is the rule's whole denominator.** No evolve option =
     nothing to suppress; no attack option = nothing to suppress it for.
  4. What does the clone do when both are offered? (evolve / attack / neither)
  5. The counterfactual size: how many turns do we spend attacking a wall with
     Grimmsnarl ex for zero, and how often was a Morgrem sitting on the bench
     while we did it? That bounds a *promotion* rule rather than an evolve-veto
     rule -- a different intervention with the same out.

⚠ **The decision is resolved PER TURN, not per select** (rule 8, multiplicity).
One turn hands us many MAIN selects -- play a card, use an ability, attach, then
attack -- and the evolve-or-attack question stays open across all of them. An
earlier version of this probe scored each select separately and so reported
"did something else (ABILITY)" as if using Munkidori's ability mid-turn were a
resolution of the choice. It is not. A turn is counted once: eligible if at any
select the Active was an armed Morgrem with the evolution on offer, and resolved
by what the turn actually did with it.

The option shapes read here (`cg.api.OptionType`):

    {'type': 13, 'attackId': 936}                       # ATTACK, Corkscrew Punch
    {'type': 9, 'area': 2, 'index': i,                  # EVOLVE: card i in hand
     'inPlayArea': 4, 'inPlayIndex': 0}                 # ... onto the Active

    python -X utf8 scripts/p7_morgrem.py --matches 120
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
import arena  # noqa: E402

MAIN = 0
TO_ACTIVE = 4        # post-KO promotion -- a FREE route to the same out
OPT_EVOLVE, OPT_ATTACK = 9, 13
LOG_TURN_START, LOG_ATTACK, LOG_HP = 2, 15, 16
CORKSCREW_PUNCH, SHADOW_BULLET = 936, 937
_HAND, _ACTIVE, _BENCH = 2, 4, 5

IMPIDIMP, MORGREM, GRIMMSNARL_EX = 646, 647, 648
CRUSTLE = 345
WALLS = frozenset({CRUSTLE})
RARE_CANDY = 1079

NAMES = {646: "Marnie's Impidimp", 647: "Marnie's Morgrem",
         648: "Marnie's Grimmsnarl ex", 112: "Munkidori", 104: "Froslass",
         860: "Snorunt", 345: "Crustle", 344: "Dwebble"}


def _name(cid) -> str:
    return NAMES.get(cid, f"#{cid}")


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


def _hand_id(state, me, index) -> int:
    try:
        card = state["players"][me]["hand"][index]
        return card["id"] if card else 0
    except (KeyError, IndexError, TypeError):
        return 0


class MorgremProbe:
    def __init__(self, inner):
        self.inner = inner
        self.wall = Counter()      # what our Active is while facing a wall
        self.decision = Counter()  # per TURN: attack with Morgrem, or evolve it away
        self.evolves = Counter()   # every evolve we actually take, by target/context
        self.bench = Counter()     # per TURN: bench Morgrem while Grimmsnarl ex hits a wall
        self.promo = Counter()     # post-KO promotion vs a wall -- the free route
        self.attacks = Counter()   # from the engine log
        self.heal = Counter()      # do they just heal our 60 back off?
        self.notes = Counter()
        self._turn: dict | None = None   # the turn currently being accumulated

    def __call__(self, obs):
        picked = self.inner(obs)
        try:
            self._look(obs, picked)
        except Exception as exc:  # noqa: BLE001
            self.notes[f"ERR {type(exc).__name__}: {exc}"] += 1
        return picked

    # --- per-turn accumulation (rule 8: one turn is one opportunity) --------
    def _flush(self) -> None:
        t = self._turn
        self._turn = None
        if not t:
            return
        if t["eligible"]:
            if t["evolved_away"]:
                self.decision["took the EVOLVE, deleting the 60 "
                              "(what a rule would veto)"] += 1
            elif t["morgrem_attacked"]:
                self.decision["attacked with Morgrem (already right)"] += 1
            else:
                self.decision["neither -- turn ended without using either"] += 1
        elif t["morgrem_armed"]:
            self.decision["[not eligible] armed Morgrem Active, "
                          "no evolution in hand"] += 1
        elif t["morgrem_active"]:
            self.decision["[not eligible] Morgrem Active but unarmed "
                          "(cannot pay {D}{D})"] += 1
        if t["zero_attack_turn"]:
            self.bench["a Morgrem was on our bench" if t["bench_morgrem"]
                       else "no Morgrem on our bench"] += 1

    def game_over(self) -> None:
        self._flush()

    def _look(self, obs, picked) -> None:
        state = obs.get("current") or {}
        me = state.get("yourIndex")
        if me is None:
            return

        # ⚠ obs['logs'] is a DELTA since our last observation (never index into
        # it as a whole-game log). Concatenating deltas sees each event once.
        for e in obs.get("logs") or []:
            if not isinstance(e, dict):
                continue
            # Healing on THEIR Crustle -- the main threat to the whole argument.
            # A heal is value > 0; a PREVENTED attack is value == 0, so it must
            # not be lumped in here (the p3 probe made exactly that mistake).
            if e.get("type") == LOG_HP and e.get("playerIndex") != me \
                    and e.get("cardId") == CRUSTLE:
                v = e.get("value") or 0
                if v > 0:
                    self.heal["healed back off their Crustle"] += v
                elif v < 0:
                    self.heal["damage that landed on their Crustle"] += -v
            if e.get("playerIndex") != me:
                continue
            if e.get("type") == LOG_ATTACK:
                self.attacks[f"{_name(e.get('cardId'))} atk {e.get('attackId')}"] += 1
            elif e.get("type") == LOG_TURN_START:
                self._flush()

        sel = obs.get("select") or {}

        # ---- the FREE route: after our Active is KO'd we must promote someone,
        # and promotion costs nothing (unlike retreating Grimmsnarl ex, whose
        # retreat cost is 2 -- i.e. the whole attack investment). Into a wall, a
        # Morgrem promotion attacks for 60 where a Grimmsnarl ex attacks for 0.
        if sel.get("context") == TO_ACTIVE:
            opts = sel.get("option") or []
            opp = _at(state, 1 - me, _ACTIVE, 0)
            if len(opts) >= 2 and opp is not None and opp.get("id") in WALLS:
                ids = []
                for o in opts:
                    pk = _at(state, me, o.get("area"), o.get("index") or 0)
                    ids.append(pk.get("id") if pk else None)
                pick = picked[0] if picked else None
                got = ids[pick] if pick is not None and 0 <= pick < len(ids) else None
                if MORGREM in ids:
                    self.promo[f"Morgrem was available -> promoted "
                               f"{_name(got)}"] += 1
                else:
                    self.promo["no Morgrem available to promote"] += 1
            return

        if sel.get("context") != MAIN:
            return
        options = sel.get("option") or []
        if not options:
            return
        if self._turn is None:
            self._turn = {"eligible": False, "evolved_away": False,
                          "morgrem_attacked": False, "morgrem_active": False,
                          "morgrem_armed": False, "zero_attack_turn": False,
                          "bench_morgrem": False}
        t = self._turn

        pick = picked[0] if picked else None
        chosen = options[pick] if pick is not None and 0 <= pick < len(options) else {}
        our_active = _at(state, me, _ACTIVE, 0)
        active_is_morgrem = bool(our_active) and our_active.get("id") == MORGREM

        # Record every evolve on the menu that we actually take, wall or not.
        if chosen.get("type") == OPT_EVOLVE:
            into = _hand_id(state, me, chosen.get("index") or 0)
            onto = _at(state, me, chosen.get("inPlayArea"),
                       chosen.get("inPlayIndex") or 0)
            where = "ACTIVE" if chosen.get("inPlayArea") == _ACTIVE else "bench"
            self.evolves[f"{_name(onto['id']) if onto else '?'} ({where})"
                         f" -> {_name(into)}"] += 1

        opp_active = _at(state, 1 - me, _ACTIVE, 0)
        if opp_active is None or opp_active.get("id") not in WALLS:
            return

        self.wall[f"our Active = "
                  f"{_name(our_active['id']) if our_active else 'none'}"] += 1

        # The engine's own answer to "can this Pokemon attack right now" -- an
        # ATTACK option on the menu. No energy reconstruction to get wrong.
        atk = [i for i, o in enumerate(options) if o.get("type") == OPT_ATTACK]
        # The evolution that deletes the non-ex attacker, onto the Active.
        evo = [i for i, o in enumerate(options)
               if o.get("type") == OPT_EVOLVE
               and o.get("inPlayArea") == _ACTIVE
               and _hand_id(state, me, o.get("index") or 0) == GRIMMSNARL_EX]

        # ---- the bench/promotion variant: Grimmsnarl ex Active into a wall, so
        # this turn's attack is worth 0 on the Active, while a Morgrem is benched.
        if our_active and our_active.get("id") == GRIMMSNARL_EX and atk:
            t["zero_attack_turn"] = True
            if any(pk and pk.get("id") == MORGREM
                   for pk in (state["players"][me].get("bench") or [])):
                t["bench_morgrem"] = True

        # ---- the evolve-veto denominator
        if active_is_morgrem:
            t["morgrem_active"] = True
            if atk:
                t["morgrem_armed"] = True
                if evo:
                    t["eligible"] = True
        if chosen.get("type") == OPT_ATTACK \
                and chosen.get("attackId") == CORKSCREW_PUNCH:
            t["morgrem_attacked"] = True
        if active_is_morgrem and pick in evo:
            t["evolved_away"] = True


def _table(title: str, data: Counter) -> None:
    tot = sum(data.values())
    print(f"\n=== {title} (n={tot}) ===")
    if not tot:
        print("  (nothing recorded -- check the denominator before believing it)")
        return
    for k, v in data.most_common(12):
        print(f"  {k:<58}{v:>7}{v / tot:>8.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=120)
    ap.add_argument("--agent", default="bc")
    ap.add_argument("--opponent", default="rule:crustle")
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--deck-b", default="crustle_v1")
    args = ap.parse_args()

    _, deck_a = arena.resolve_deck(args.deck)
    _, deck_b = arena.resolve_deck(args.deck_b)
    _, agent_a = arena.build_agent(args.agent, deck_a)
    _, agent_b = arena.build_agent(args.opponent, deck_b)
    probe = MorgremProbe(agent_a)

    for m in range(args.matches):
        if m % 2 == 0:
            harness.play_game(probe, agent_b, list(deck_a), list(deck_b))
        else:
            harness.play_game(agent_b, probe, list(deck_b), list(deck_a))
        probe.game_over()

    print(f"\n{args.agent} [{args.deck}] vs {args.opponent} [{args.deck_b}], "
          f"{args.matches} games")
    _table("MAIN selects while THEIR Active is a wall -- who is our Active",
           probe.wall)
    _table("PER TURN vs a wall -- THE RULE'S DENOMINATOR is the 'eligible' rows",
           probe.decision)
    _table("PER TURN: Grimmsnarl ex attacking a wall for ZERO -- was a Morgrem "
           "benched?", probe.bench)
    _table("POST-KO PROMOTION into a wall (free -- no retreat cost)", probe.promo)
    _table("Damage vs healing on their Crustle (totals, not counts)", probe.heal)
    _table("Every evolve we chose (any matchup state)", probe.evolves)
    _table("Attacks we used (engine log)", probe.attacks)
    if probe.notes:
        print("\nerrors:")
        for k, v in probe.notes.most_common(5):
            print(f"  {v:>5}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
