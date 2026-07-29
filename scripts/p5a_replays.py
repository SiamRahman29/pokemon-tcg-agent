"""P5a on REAL games: does Adrena-Brain misaim against the actual LB field?

Rule 11: the local arena is one opponent deck. `p5_audit.py --matches 200` said
the pooled-budget miss never happens (26 pooled-KO selects, best-prize target
taken 26/26) -- but that was `rule:v10,noS` on `lucario_v10`, and the user saw
the failure against the real field. So re-run the same counter over
`replays/submission_replay_2026-07-29/`, which is our live agent vs 54 distinct
LB opponents.

    python -X utf8 scripts/p5a_replays.py

Identical arithmetic to `p5_audit.Probe._chip`, but reading `obs`/`selected`
out of the replay instead of driving the engine. Our seat is the one whose
`info.TeamNames` entry is `Scio`; a self-play validation episode has us on both
sides and is counted on both.
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

MAIN, DAMAGE_COUNTER, DAMAGE_COUNTER_ANY = 0, 13, 14
OPT_ABILITY = 10
_ACTIVE, _BENCH = 4, 5
MUNKIDORI, DARK_TYPE = 112, 7
PER_ACTIVATION = 30
US = "Scio"


def _mine(pl):
    out = {}
    act = pl["active"]
    if act and act[0] is not None:
        out[(_ACTIVE, 0)] = act[0]
    for i, pk in enumerate(pl["bench"]):
        if pk is not None:
            out[(_BENCH, i)] = pk
    return out


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
    ap.add_argument("--dir", default="replays/submission_replay_2026-07-29")
    args = ap.parse_args()

    a = Counter()
    choice = Counter()
    misses = []
    n_games = n_err = n_sel = 0
    opponents = set()

    for path in sorted(Path(args.dir).glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            rep = json.loads(path.read_text(encoding="utf-8"))
            names = rep["info"]["TeamNames"]
            ours = {i for i, n in enumerate(names) if n == US}
            if not ours:
                continue
            opponents.update(n for n in names if n != US)
            n_games += 1
            vis = rep["steps"][0][0].get("visualize") or []
            turn_key, used = None, set()
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
                action = v.get("selected")
                if action is None:
                    action = v.get("action")
                if not isinstance(action, list):
                    continue
                opts = sel.get("option") or []
                chosen = [i for i in action
                          if isinstance(i, int) and 0 <= i < len(opts)][:1]
                mypl = state["players"][me]
                ctx = sel.get("context")

                key = (path.stem, state["turn"], me)
                if key != turn_key:
                    turn_key, used = key, set()

                if ctx == MAIN:
                    for i in chosen:
                        if opts[i].get("type") != OPT_ABILITY:
                            continue
                        slot = (opts[i].get("area"), opts[i].get("index") or 0)
                        pk = _mine(mypl).get(slot)
                        if pk and pk["id"] == MUNKIDORI:
                            used.add(slot)
                    continue
                if ctx not in (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY):
                    continue
                if len(opts) < 2:
                    continue

                n_sel += 1
                cand = {}
                bad = False
                for i, o in enumerate(opts):
                    if o.get("playerIndex") in (None, me):
                        bad = True       # mixed select: not our case
                        break
                    pk = _at(state, 1 - me, o.get("area"), o.get("index") or 0)
                    if pk is None or pk.get("hp") is None:
                        bad = True
                        break
                    cand[i] = pk
                if bad or len(cand) < 2:
                    continue

                mine = _mine(mypl)
                armed = {s for s, pk in mine.items()
                         if pk["id"] == MUNKIDORI
                         and any(e == DARK_TYPE
                                 for e in (pk.get("energies") or []))}
                # see p5_audit._chip: the activating Munkidori is already in
                # `used`, so the pool is THIS activation plus the ones left
                left = 1 + len(armed - used)
                own = sum(max(0, (pk.get("maxHp") or 0) - pk["hp"])
                          for pk in mine.values())
                budget = min(PER_ACTIVATION * left, own)

                if any(pk["hp"] <= PER_ACTIVATION for pk in cand.values()):
                    a["something already dies to one activation"] += 1
                    continue
                pooled = [i for i, pk in cand.items() if pk["hp"] <= budget]
                if not pooled:
                    a["nothing dies even with the pooled budget"] += 1
                    continue
                best = max(cdb.prize_value(cand[i]["id"]) for i in pooled)
                c = chosen[0] if chosen and chosen[0] in cand else None
                got = cdb.prize_value(cand[c]["id"]) if c is not None else 0
                # see p5_audit: one pooled candidate means best == got by
                # construction, so the row cannot fail. Only rows with two
                # different prize values are a real test of the aim.
                choice[("prize values DIFFER -- a real choice"
                        if len({cdb.prize_value(cand[i]["id"])
                                for i in pooled}) > 1
                        else "one prize value among them -- nothing to get "
                             "wrong")] += 1
                if c is not None and c in pooled and got >= best:
                    a["pooled KO taken, best prizes"] += 1
                elif best > got:
                    a[f"MISS: pooled {best}-prize KO available, "
                      f"took {got}-prize"] += 1
                    tgt = max(pooled,
                              key=lambda i: cdb.prize_value(cand[i]["id"]))
                    misses.append(
                        f"{path.stem} t{state['turn']}: armed={len(armed)} "
                        f"used={len(used)} budget={budget} own={own} | took "
                        f"{cdb.card(cand[c]['id'])['name'] if c is not None else '-'}"
                        f" hp={cand[c]['hp'] if c is not None else '-'} "
                        f"({got}p) over {cdb.card(cand[tgt]['id'])['name']} "
                        f"hp={cand[tgt]['hp']} ({best}p)")
                else:
                    a["pooled KO available, not taken"] += 1
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            if n_err <= 5:
                print(f"  {path.name}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)

    tot = sum(a.values())
    print(f"\n=== P5a over {n_games} real games "
          f"({len(opponents)} distinct opponents), errors={n_err} ===")
    print(f"chip selects seen: {n_sel}; opponent-only with >=2 options: {tot}")
    for k, v in a.most_common():
        print(f"  {k:<52}{v:>6}{v / tot:>8.1%}" if tot else f"  {k}")
    tot2 = sum(choice.values())
    print(f"\nof the pooled KOs, how many were a real choice (n={tot2})")
    for k, v in choice.most_common():
        print(f"  {k:<52}{v:>6}{v / tot2:>8.1%}")
    if misses:
        print("\nthe misses:")
        for m in misses[:40]:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
