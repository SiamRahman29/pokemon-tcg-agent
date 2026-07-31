"""What did the optfeat-v3 agent actually do wrong on the ladder? (55116557)

**Why this exists.** v3 won every local A/B -- 0.661 vs the shipped agent in the
mirror, 0.770 vs `rule:crustle` -- and then read **819.8 on the LB against P4b's
952.0**. That is a ~130-point contradiction, *far* larger than the instrument's
±50-100 (HANDOFF §1), so unlike `counter_source` this one is real and the arena
is what is wrong. These are 54 games against the actual field, which is the only
place the disagreement can be diagnosed.

Two failures the user watched, both audited here, plus the good news:

  1. **Boss's Orders drags away a Pokemon we could have KO'd.** Shadow Bullet
     does 180 to their Active *and* 30 to a benched Pokemon, so dragging is only
     right when the drag beats "KO the Active (+ maybe snipe a 30-HP bench)".
     ⚠ Note `drag_target`/`boss_veto` were both measured NULL (0.489/0.493) --
     but that was against the **lw2** net, and this agent ships with **every
     rule off**, so the card is decided by the net alone for the first time.
  2. **Freezing Shroud is symmetric and we are its biggest victim.** "put 1
     damage counter on each Pokemon that has an Ability (both yours and your
     opponent's), except any Froslass" -- and OUR ability Pokemon are Munkidori
     (x4) and **Marnie's Grimmsnarl ex (Punk Up, x3)**, i.e. our own main
     attacker. The clock only pays if Munkidori is there to move the counters
     across. This counts the board at the moment we CHOOSE to evolve into her.

Denominators first, per rule 13 -- a rate over forced moves measures nothing, so
every audit below reports the cases where a real alternative existed.

    python -X utf8 scripts/p8_optv3_replays.py --dir replays/submission_optv3
"""
from __future__ import annotations

import argparse
import json
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

from sa import cards as cdb  # noqa: E402
from sa.targeting import best_damage  # noqa: E402

US = "Scio"
MAIN, SWITCH = 0, 3
OPT_EVOLVE, OPT_ATTACK = 9, 13
LOG_ATTACK, LOG_HP = 15, 16
_HAND, _ACTIVE, _BENCH = 2, 4, 5

MUNKIDORI, FROSLASS, SNORUNT = 112, 104, 860
IMPIDIMP, MORGREM, GRIMMSNARL_EX = 646, 647, 648
CRUSTLE, DWEBBLE = 345, 344
DARK_TYPE = 7
SHADOW_BULLET_SNIPE = 30      # Shadow Bullet's bench damage
BOSS_ORDERS = 1182


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


def _in_play(pl):
    out = []
    act = pl.get("active")
    if act and act[0] is not None:
        out.append(act[0])
    out += [pk for pk in (pl.get("bench") or []) if pk is not None]
    return out


def _has_ability(cid: int) -> bool:
    return bool(cdb.card(cid).get("skills"))


def _shroud_targets(pl) -> int:
    """Pokemon on this side that Freezing Shroud would put a counter on."""
    return sum(1 for pk in _in_play(pl)
               if _has_ability(pk["id"]) and pk["id"] != FROSLASS)


def _name(cid) -> str:
    return str(cdb.card(cid).get("name") or f"#{cid}")


def _archetype(pl_cards: set[int]) -> str:
    if CRUSTLE in pl_cards or DWEBBLE in pl_cards:
        return "Crustle"
    if GRIMMSNARL_EX in pl_cards or MORGREM in pl_cards:
        return "Grimmsnarl (mirror)"
    if 96 in pl_cards or 63 in pl_cards or 184 in pl_cards:
        return "Crispin toolbox"
    if 108 in pl_cards or 117 in pl_cards:
        return "Ogerpon"
    return "other"


class Report:
    def __init__(self):
        self.games = Counter()
        self.arch = Counter()
        self.arch_win = Counter()
        self.boss = Counter()
        self.boss_egs: list[str] = []
        self.shroud = Counter()
        self.shroud_egs: list[str] = []
        self.good = Counter()
        self.errs = Counter()


def analyse(path: Path, rep: Report) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        rep.errs["file is a bare step-array, not a replay"] += 1
        return
    names = (d.get("info") or {}).get("TeamNames") or []
    if US not in names:
        rep.errs["no Scio seat"] += 1
        return
    ours = {i for i, n in enumerate(names) if n == US}
    me_seat = min(ours)
    result = d["rewards"][me_seat]
    won = result > 0
    rep.games["win" if result > 0 else ("loss" if result < 0 else "draw")] += 1

    opp_cards: set[int] = set()
    vis = d["steps"][0][0].get("visualize") or []

    for v in vis:
        obs = v.get("obs")
        if not obs or not obs.get("current") or not obs.get("select"):
            continue
        state, sel = obs["current"], obs["select"]
        if state["result"] != -1:
            continue
        me = state["yourIndex"]
        if me not in ours:
            continue
        try:
            mypl, oppl = state["players"][me], state["players"][1 - me]
        except (KeyError, IndexError, TypeError):
            continue
        for pk in _in_play(oppl):
            opp_cards.add(pk["id"])

        action = v.get("selected")
        if action is None:
            action = v.get("action")
        if not isinstance(action, list):
            continue
        opts = sel.get("option") or []
        picked = [i for i in action if isinstance(i, int) and 0 <= i < len(opts)]
        if not picked:
            continue
        pick = picked[0]
        ctx = sel.get("context")

        our_act = _at(state, me, _ACTIVE, 0)
        their_act = _at(state, 1 - me, _ACTIVE, 0)

        # ---- 1. Boss's Orders: the drag (a SWITCH select over THEIR bench) ---
        if ctx == SWITCH and len(opts) >= 2 and their_act and our_act:
            tgts = {}
            ok = True
            for i, o in enumerate(opts):
                if o.get("playerIndex") in (None, me) or o.get("area") != _BENCH:
                    ok = False
                    break
                pk = _at(state, 1 - me, _BENCH, o.get("index") or 0)
                if pk is None or pk.get("hp") is None:
                    ok = False
                    break
                tgts[i] = pk
            if ok and len(tgts) >= 2 and their_act.get("hp") is not None:
                dmg_active = best_damage(our_act, mypl, oppl, their_act)
                kills_active = dmg_active >= their_act["hp"]
                got = tgts.get(pick)
                kills_got = (got is not None
                             and best_damage(our_act, mypl, oppl, got) >= got["hp"])
                # Shadow Bullet also snipes 30 onto a bench Pokemon, so a
                # <=30 HP bench sitter means "attack" is a DOUBLE KO.
                snipeable = [pk for pk in tgts.values()
                             if pk["hp"] <= SHADOW_BULLET_SNIPE]
                if not kills_active:
                    rep.boss["their Active was NOT KO-able -- drag is free"] += 1
                else:
                    # A real choice existed: we could have attacked instead.
                    p_act = cdb.prize_value(their_act["id"])
                    p_got = cdb.prize_value(got["id"]) if got else 0
                    if snipeable:
                        rep.boss["MISS: could KO Active + SNIPE a <=30hp bench "
                                 "(double KO) -- dragged instead"] += 1
                        if len(rep.boss_egs) < 8:
                            rep.boss_egs.append(
                                f"{path.stem} t{state['turn']}: could KO "
                                f"{_name(their_act['id'])} hp={their_act['hp']}"
                                f" ({p_act}p) + snipe "
                                f"{_name(snipeable[0]['id'])} hp="
                                f"{snipeable[0]['hp']}; dragged "
                                f"{_name(got['id']) if got else '?'}")
                    elif not kills_got:
                        rep.boss["MISS: dragged something we CANNOT KO, "
                                 "abandoning a KO-able Active"] += 1
                        if len(rep.boss_egs) < 8:
                            rep.boss_egs.append(
                                f"{path.stem} t{state['turn']}: could KO "
                                f"{_name(their_act['id'])} hp={their_act['hp']}"
                                f" ({p_act}p); dragged "
                                f"{_name(got['id']) if got else '?'} hp="
                                f"{got['hp'] if got else '-'} which survives")
                    elif p_got < p_act:
                        rep.boss[f"MISS: dragged a {p_got}-prize KO over a "
                                 f"{p_act}-prize KO"] += 1
                    else:
                        rep.boss["drag traded up or equal -- defensible"] += 1

        # ---- 2. Froslass: the moment we CHOOSE to bring her out -------------
        if ctx == MAIN and opts[pick].get("type") == OPT_EVOLVE:
            try:
                hand = mypl["hand"]
                into = hand[opts[pick].get("index") or 0]
                into_id = into["id"] if into else 0
            except (KeyError, IndexError, TypeError):
                into_id = 0
            if into_id == FROSLASS:
                mine_t = _shroud_targets(mypl)
                theirs_t = _shroud_targets(oppl)
                armed = sum(1 for pk in _in_play(mypl)
                            if pk["id"] == MUNKIDORI
                            and any(e == DARK_TYPE
                                    for e in (pk.get("energies") or [])))
                if theirs_t > mine_t:
                    rep.shroud["GOOD: they have more ability Pokemon than us"] += 1
                elif theirs_t == mine_t:
                    rep.shroud["neutral: equal ability Pokemon"] += 1
                else:
                    key = ("BAD: WE have more ability Pokemon -- the clock "
                           "hurts us more")
                    rep.shroud[key + (" (and NO armed Munkidori)"
                                      if not armed else " (armed Munkidori)")] += 1
                    if len(rep.shroud_egs) < 8:
                        rep.shroud_egs.append(
                            f"{path.stem} t{state['turn']}: ours={mine_t} "
                            f"theirs={theirs_t} armedMunkidori={armed}")

        # ---- 3. the good news ----------------------------------------------
        if ctx == MAIN and opts[pick].get("type") == OPT_ATTACK:
            if our_act and our_act["id"] == MORGREM:
                rep.good["Morgrem attacked"] += 1
                if (their_act and their_act.get("hp") is not None
                        and their_act["id"] in (CRUSTLE,)):
                    rep.good["  ... into a Crustle (the non-ex out)"] += 1

    rep.arch[_archetype(opp_cards)] += 1
    if won:
        rep.arch_win[_archetype(opp_cards)] += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="replays/submission_optv3")
    args = ap.parse_args()
    rep = Report()
    for path in sorted(Path(args.dir).glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            analyse(path, rep)
        except Exception as exc:  # noqa: BLE001
            rep.errs[f"{type(exc).__name__}: {exc}"] += 1

    tot = sum(rep.games.values())
    print(f"\n=== {args.dir}: {tot} games ===")
    print(f"  {dict(rep.games)}  win rate = "
          f"{(rep.games['win'] + 0.5 * rep.games['draw']) / max(tot,1):.3f}")

    print("\n=== opponent archetype (and our win rate vs each) ===")
    for k, v in rep.arch.most_common():
        print(f"  {k:<24}{v:>5}  won {rep.arch_win[k]:>3}  "
              f"({rep.arch_win[k]/v:.1%})")

    def table(title, c, egs=None, note=""):
        t = sum(c.values())
        print(f"\n=== {title} (n={t}) ===")
        if not t:
            print("  (no rows -- check the denominator before believing it)")
            return
        if note:
            print(f"  {note}")
        for k, v in c.most_common():
            print(f"  {k:<62}{v:>5}{v/t:>7.1%}")
        for e in (egs or []):
            print(f"    eg {e}")

    table("BOSS'S ORDERS -- the drag, only where attacking was a real option",
          rep.boss, rep.boss_egs,
          note="rows below the first are MISPLAYS: the Active was KO-able "
               "and we dragged anyway")
    table("FROSLASS -- the board when we chose to evolve into her",
          rep.shroud, rep.shroud_egs)
    table("THE GOOD", rep.good)
    if rep.errs:
        print("\nerrors/skips:")
        for k, v in rep.errs.most_common(6):
            print(f"  {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
