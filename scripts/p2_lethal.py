"""Does the clone take an available lethal? (HANDOFF P2 / ROADMAP B2, step 1)

MAIN is 47.7% of all selects with >= 2 options and holds 3,930 of the net's
6,424 misses, but it is not one decision -- and rule 11 says most of it
(which Supporter, attach now or later, evolve or develop) is *tradeoff*, the
class that went 0-for-4. So do not write a MAIN scorer. Measure the one part of
MAIN that is pure arithmetic first: **given the board, does a payable attack KO
the opponent's Active, and did we take it?**

ATTACK options arrive on the MAIN select itself carrying `attackId`, and the
engine only offers attacks the Active can pay for -- so "payable" needs no
modelling here. `textdmg.estimate` is approximate in general but **exact for
this deck** (every attack grimmsnarl can pay for is flat damage), so the
measurement is trustworthy.

**Two cuts, NEVER merged** (they are different classes and rule 11 predicts
opposite outcomes):

  1. SAME-ATTACKER -- our current Active was offered an attack that KOs and we
     did something else. Dominated. High prior. If this reads ~0 the way
     `REMOVE_DAMAGE_COUNTER_COUNT` did, B2 closes for an hour instead of a day.
  2. NEEDS-PROMOTION -- nothing we were offered KOs, but a benched Pokemon
     could KO if it were Active. Costs a retreat and a turn of setup, so it is
     a tradeoff. Lower prior. Measured separately.

Rule 13: the denominator must be a real CHOICE. A turn is only counted once, at
its **final** MAIN select -- the moment we actually commit -- because the
opponent's Active HP moves during our own turn (Adrena-Brain counters), so a
lethal that existed early can be irrelevant, or unnecessary, by the end.

    python -X utf8 scripts/p2_lethal.py --matches 200
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
from sa.textdmg import estimate  # noqa: E402
import arena  # noqa: E402

MAIN = 0
OPT_ATTACK, OPT_END, OPT_RETREAT = 13, 14, 12
_ACTIVE, _BENCH = 4, 5


def _damage(attack_id: int, attacker: dict, mypl: dict, oppl: dict,
            target: dict) -> float:
    """Damage `attack_id` from `attacker` would do to `target`, with weakness.

    Same arithmetic as `targeting.best_damage`, but for one named attack
    instead of the best one, because here we need to compare the attack we
    *chose* against the attacks we were *offered*."""
    atk_type = cdb.card(attacker["id"]).get("energyType")
    weak = cdb.card(target["id"]).get("weakness")
    mult = 2.0 if (weak is not None and weak == atk_type) else 1.0
    return estimate(attack_id, attacker, mypl, oppl) * mult


def _best_payable(pk: dict, mypl: dict, oppl: dict, target: dict) -> float:
    """Best damage `pk` could do to `target` with the energy it holds now."""
    best = 0.0
    for aid in cdb.card(pk["id"]).get("attacks") or []:
        a = cdb.attacks().get(aid)
        if a and cdb.energy_satisfied(a["energies"], pk.get("energies") or []):
            best = max(best, _damage(aid, pk, mypl, oppl, target))
    return best


class LethalProbe:
    """Records one row per turn, at the turn's final MAIN select."""

    def __init__(self, inner):
        self.inner = inner
        self.same = Counter()      # cut 1, all lethal-offered turns
        self.choice = Counter()    # cut 1, was declining it even possible?
        self.real = Counter()      # cut 1 restricted to real choices (rule 13)
        self.promo = Counter()     # cut 2
        self.game_end = Counter()  # rows the game ended on -- not gradable
        self.turns = 0
        self.errors = Counter()
        self.misses: list[dict] = []
        self._pending: dict | None = None   # last MAIN select seen this turn
        self._turn: int | None = None       # state["turn"] that row belongs to
        self.no_commit = 0                  # turns that never chose ATTACK/END

    # -- the row is only scored once we know the turn is over ---------------
    def _commit(self, game_ended: bool = False) -> None:
        row = self._pending
        self._pending = None
        if row is None:
            return
        self.turns += 1

        # ⚠ The row a game ENDS on is not a decision we can grade. Adrena-Brain
        # moves counters onto any of their Pokemon, so the winning turn often
        # takes the last prize via the ABILITY and never reaches an attack --
        # which looks identical to "ended the turn with a lethal on the table"
        # if you don't separate it. Scoring these as misses read 5.5% missed
        # lethals on the first run of this audit; all 19 were wins.
        if game_ended:
            self.game_end["game ended on this turn -- not gradable"] += 1
            return

        if row["lethal_offered"]:
            # Rule 13: "took the lethal" is only a real result if declining it
            # for ANOTHER ATTACK was possible. With one payable attack offered,
            # the row cannot fail -- P5c already showed the clone always attacks
            # when it can (3,683/3,683), so such a row measures nothing new.
            self.choice["a non-lethal attack was also offered (real choice)"
                        if row["nonlethal_alt"] else
                        "the lethal was the ONLY attack offered (forced)"] += 1
            if row["nonlethal_alt"]:
                self.real["took the lethal" if row["took_lethal"] else
                          "MISSED -- chose the non-lethal attack"
                          if row["attacked"] else
                          "MISSED -- ended the turn"] += 1
            if row["took_lethal"]:
                self.same["took the lethal"] += 1
            elif row["attacked"]:
                self.same["MISSED -- attacked with a non-lethal attack"] += 1
                self.misses.append(row)
            else:
                self.same["MISSED -- ended the turn without attacking"] += 1
                self.misses.append(row)
            return

        # No offered attack KOs. Could a benched Pokemon have done it?
        if row["bench_lethal"]:
            key = ("bench KO was available and retreat was legal"
                   if row["retreat_legal"] else
                   "bench KO existed but retreat was NOT legal")
            self.promo[key] += 1
        else:
            self.promo["no KO was available from anywhere"] += 1

    def __call__(self, obs):
        picked = self.inner(obs)
        try:
            sel = obs.get("select") or {}
            if sel.get("context") != MAIN:
                return picked
            opts = sel.get("option") or []
            state = obs.get("current") or {}
            me = state.get("yourIndex")
            if me is None or not opts:
                return picked
            # Turn boundaries are exact -- the state carries a turn counter --
            # so a turn that ends without an explicit ATTACK/END choice is
            # still scored, and counted (rule 9: know your denominator).
            turn = state.get("turn")
            if turn != self._turn:
                if self._pending is not None:
                    self.no_commit += 1
                self._commit()
                self._turn = turn

            mypl, oppl = state["players"][me], state["players"][1 - me]
            active = mypl["active"][0] if mypl.get("active") else None
            opp = oppl["active"][0] if oppl.get("active") else None
            if not active or not opp or opp.get("hp") is None:
                return picked

            offered = [(i, o["attackId"]) for i, o in enumerate(opts)
                       if o.get("type") == OPT_ATTACK and o.get("attackId")]
            hp = opp["hp"]
            lethal_ids = {aid for _, aid in offered
                          if _damage(aid, active, mypl, oppl, opp) >= hp}

            pick = picked[0] if picked else None
            chosen = opts[pick] if (pick is not None
                                    and 0 <= pick < len(opts)) else {}
            attacked = chosen.get("type") == OPT_ATTACK
            chosen_aid = chosen.get("attackId") if attacked else None

            bench_lethal = False
            if not lethal_ids:
                for pk in mypl.get("bench") or []:
                    if pk and _best_payable(pk, mypl, oppl, opp) >= hp:
                        bench_lethal = True
                        break

            self._pending = {
                "lethal_offered": bool(lethal_ids),
                "n_attacks": len(offered),
                "nonlethal_alt": bool(lethal_ids) and any(
                    aid not in lethal_ids for _, aid in offered),
                "took_lethal": chosen_aid in lethal_ids if lethal_ids else False,
                "attacked": attacked,
                "ended": chosen.get("type") == OPT_END,
                "bench_lethal": bench_lethal,
                "retreat_legal": any(o.get("type") == OPT_RETREAT
                                     for o in opts),
                "opp_id": opp["id"], "opp_hp": hp,
                "offered": [aid for _, aid in offered],
                "chosen_aid": chosen_aid,
                "chosen_type": chosen.get("type"),
                "best_offered": max((_damage(aid, active, mypl, oppl, opp)
                                     for _, aid in offered), default=0.0),
            }
            # Attacking or ending finishes the turn, so this select was final.
            if attacked or chosen.get("type") == OPT_END:
                self._commit()
        except Exception as exc:  # noqa: BLE001
            self.errors[f"{type(exc).__name__}: {exc}"] += 1
        return picked

    def game_over(self) -> None:
        """A game can end mid-turn (last prize taken); that row is not gradable."""
        self._commit(game_ended=True)
        self._turn = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=200)
    ap.add_argument("--agent", default="bc")
    ap.add_argument("--opponent", default="rule:v10,noS")
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--deck-b", default="lucario_v10")
    args = ap.parse_args()

    _, deck_a = arena.resolve_deck(args.deck)
    _, deck_b = arena.resolve_deck(args.deck_b)
    _, agent_a = arena.build_agent(args.agent, deck_a)
    _, agent_b = arena.build_agent(args.opponent, deck_b)
    probe = LethalProbe(agent_a)

    for m in range(args.matches):
        if m % 2 == 0:
            harness.play_game(probe, agent_b, list(deck_a), list(deck_b))
        else:
            harness.play_game(agent_b, probe, list(deck_b), list(deck_a))
        probe.game_over()

    print(f"\n{args.agent} vs {args.opponent} ({args.deck} vs {args.deck_b}), "
          f"{args.matches} games, {probe.turns} of our turns scored")

    n1 = sum(probe.same.values())
    print(f"\n=== CUT 1: SAME ATTACKER (dominated) -- a KO was on the table at "
          f"the turn's final MAIN select (n={n1}) ===")
    for k, v in probe.same.most_common():
        print(f"  {k:<48}{v:>7}{v / n1:>8.1%}" if n1 else f"  {k}")
    miss1 = sum(v for k, v in probe.same.items() if k.startswith("MISSED"))
    if n1:
        print(f"  --> missed {miss1}/{n1} = {miss1 / n1:.1%} of turns where a "
              f"lethal was offered")
        print(f"  --> {miss1 / probe.turns:.2%} of all our turns "
              f"(n={probe.turns})")

    nc = sum(probe.choice.values())
    print(f"\n--- rule 13: was declining the lethal even possible? (n={nc}) ---")
    for k, v in probe.choice.most_common():
        print(f"  {k:<48}{v:>7}{v / nc:>8.1%}" if nc else f"  {k}")
    nr = sum(probe.real.values())
    print(f"\n--- CUT 1 on the HONEST denominator: lethal AND a non-lethal "
          f"attack both offered (n={nr}) ---")
    if nr:
        for k, v in probe.real.most_common():
            print(f"  {k:<48}{v:>7}{v / nr:>8.1%}")
    else:
        print("  n=0 -- the deck never offers a choice between two attacks, so")
        print("  there is no dominated decision here to win. NOT 'we are")
        print("  perfect': there is nothing to get wrong (rule 13).")

    n2 = sum(probe.promo.values())
    print(f"\n=== CUT 2: NEEDS PROMOTION (tradeoff) -- nothing offered KOs "
          f"(n={n2}) ===")
    for k, v in probe.promo.most_common():
        print(f"  {k:<48}{v:>7}{v / n2:>8.1%}" if n2 else f"  {k}")

    ng = sum(probe.game_end.values())
    print(f"\n(excluded: {ng} turns the game ended on -- not gradable)")

    if probe.misses:
        print(f"\n=== first 12 cut-1 misses (diagnostic) ===")
        for row in probe.misses[:12]:
            what = ("END" if row["ended"]
                    else f"attack {row['chosen_aid']}"
                    if row["chosen_aid"] else f"type {row['chosen_type']}")
            print(f"  opp {row['opp_id']} at {row['opp_hp']} hp; offered "
                  f"{row['offered']} (best {row['best_offered']:.0f}); "
                  f"we chose {what}")
    if probe.errors:
        print("\nerrors:")
        for k, v in probe.errors.most_common(5):
            print(f"  {v:>5}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
