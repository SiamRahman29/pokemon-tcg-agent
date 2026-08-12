"""E28 readings 2 and 3 — commitment switches, and realised prize maps.

Pre-registered in `docs/experiments/E28-replay-trace-audit.md` (frozen at
`aeb530b`). ⚠ **Reading 1 is VOID** (§R3/§R5: its positive control missed the
frozen 0.90 bar under both registered constructions). §4's verdict branches are
joint conditions on readings 1 AND 2, so **this script cannot issue a verdict on
the copycat hypothesis** -- it reports reading 2 as a standalone measurement and
reading 3 as report-track vocabulary.

    python -X utf8 scripts/p89_e28_reading2.py

**Reading 2 — commitment switches** (§N.4.1, HANDOFF probe 1), per game AND per
decision, same denominators both sides:

  * **attacker switch** -- our active changes while the OLD active is still
    alive (on our bench with hp > 0). A switch forced by a KO is not a
    commitment change and is excluded by that liveness test.
  * **target switch** -- the opponent Pokemon we are damaging changes before the
    previous one is KO'd (it is still on their board with hp > 0).

Damage is detected the way `p72_loss_autopsy.verify` detects it -- an hp DROP
between consecutive records -- because that route is the one with a positive
control behind it (839/840 vs 11/48 for the obvious `steps[i][seat].action`).

**Reading 3 — prize maps.** Classify each game's realised KO sequence by the
prize counts taken in order: 2-2-2, 1-1-2-2, 2-1-2-1, other. ⛔ Descriptive
only; not a feature and not a rule (the encoding axis is closed).
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from p9_field_census import _cid, _signature  # noqa: E402
from p87_e28_pairs import SIDES, MIRROR  # noqa: E402
from sa import cards as cdb  # noqa: E402


def _board(pl: dict) -> dict[int, int]:
    """serial -> hp for one player's active+bench."""
    out: dict[int, int] = {}
    for pk in (pl.get("active") or []) + (pl.get("bench") or []):
        if pk and pk.get("serial") is not None:
            out[pk["serial"]] = pk.get("hp", 0)
    return out


def _active(pl: dict):
    a = pl.get("active") or []
    return a[0] if a and a[0] else None


def _prize_left(pl: dict) -> int | None:
    for k in ("prize", "prizes", "prizeCount", "remainPrize"):
        v = pl.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, list):
            return len(v)
    return None


def scan(rep: dict):
    """(labels, names, per-seat stats). One parse per replay."""
    if not isinstance(rep, dict):
        return None
    info = rep.get("info")
    names = [str(x) for x in ((info or {}).get("TeamNames") or [])] \
        if isinstance(info, dict) else []
    try:
        vis = rep["steps"][0][0].get("visualize") or []
    except (KeyError, IndexError, TypeError):
        return None

    poke = [Counter(), Counter()]
    maxc = [defaultdict(int), defaultdict(int)]
    st_seq: list[tuple[int, dict]] = []      # (deciding seat, state)

    for v in vis:
        ob = v.get("obs") or {}
        st = ob.get("current")
        if not st:
            continue
        me = st.get("yourIndex")
        if me not in (0, 1):
            continue
        if ob.get("select"):
            st_seq.append((me, st))
        try:
            op = st["players"][1 - me]
        except (KeyError, IndexError, TypeError):
            continue
        g = 1 - me
        here: Counter = Counter()
        a0 = op.get("active")
        if a0 and a0[0]:
            here[_cid(a0[0])] += 1
            for c in (a0[0].get("cards") or []):
                here[_cid(c)] += 1
        for pk in (op.get("bench") or []):
            if pk:
                here[_cid(pk)] += 1
                for c in (pk.get("cards") or []):
                    here[_cid(c)] += 1
        for c in (op.get("discard") or []):
            here[_cid(c)] += 1
        for cid, n in here.items():
            if n > maxc[g][cid]:
                maxc[g][cid] = n
            if cdb.is_pokemon(cid):
                poke[g][cid] += 1

    labels = [_signature(poke[i], maxc[i]) for i in (0, 1)]

    stats = {}
    for seat in (0, 1):
        mine = [st for s, st in st_seq if s == seat]
        if len(mine) < 2:
            continue
        att_sw = tgt_sw = dec = 0
        last_active = None
        last_target = None
        prizes: list[int] = []
        prev_left = None

        for i, st in enumerate(mine):
            try:
                me_pl = st["players"][seat]
                op_pl = st["players"][1 - seat]
            except (KeyError, IndexError, TypeError):
                continue
            dec += 1

            # --- attacker switch: active changed, old one still ALIVE ---
            act = _active(me_pl)
            cur = act.get("serial") if act else None
            if (last_active is not None and cur is not None
                    and cur != last_active):
                bd = _board(me_pl)
                if bd.get(last_active, 0) > 0:      # not a KO-forced switch
                    att_sw += 1
            if cur is not None:
                last_active = cur

            # --- target switch: who we damaged changed before a KO ---
            if i + 1 < len(mine):
                try:
                    nxt_op = mine[i + 1]["players"][1 - seat]
                except (KeyError, IndexError, TypeError):
                    nxt_op = None
                if nxt_op:
                    b0, b1 = _board(op_pl), _board(nxt_op)
                    hurt = [s for s in b0 if s in b1 and b1[s] < b0[s]]
                    if len(hurt) == 1:
                        t = hurt[0]
                        if (last_target is not None and t != last_target
                                and b1.get(last_target, 0) > 0):
                            tgt_sw += 1
                        last_target = t

            # --- reading 3: prize takes, in order ---
            left = _prize_left(op_pl if False else me_pl)
            if left is not None:
                if prev_left is not None and left < prev_left:
                    prizes.append(prev_left - left)
                prev_left = left

        stats[seat] = {"att": att_sw, "tgt": tgt_sw, "dec": dec,
                       "prizes": prizes}
    return labels, names, stats


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run(side: str) -> dict:
    tot = {"att": 0, "tgt": 0, "dec": 0, "games": 0}
    per_game_att: list[int] = []
    per_game_tgt: list[int] = []
    maps: Counter = Counter()
    takes: Counter = Counter()
    n_takes: list[int] = []
    seen: set[str] = set()

    for dname, team in SIDES[side]:
        d = ROOT / dname
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.json")):
            if path.name == "manifest.json" or path.stem in seen:
                continue
            seen.add(path.stem)
            try:
                got = scan(json.loads(path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
            if got is None:
                continue
            labels, names, stats = got
            idxs = ([i for i, nm in enumerate(names)
                     if any(t.lower() in nm.lower() for t in team)]
                    if team else [0, 1])
            for i in idxs:
                if labels[i] != MIRROR or i not in stats:
                    continue
                s = stats[i]
                tot["att"] += s["att"]
                tot["tgt"] += s["tgt"]
                tot["dec"] += s["dec"]
                tot["games"] += 1
                per_game_att.append(s["att"])
                per_game_tgt.append(s["tgt"])
                seq = s["prizes"]
                maps["-".join(map(str, seq)) if seq else "none"] += 1
                for k in seq:
                    takes[k] += 1
                n_takes.append(len(seq))
    tot["att_pg"] = tot["att"] / max(tot["games"], 1)
    tot["tgt_pg"] = tot["tgt"] / max(tot["games"], 1)
    tot["att_pd"] = tot["att"] / max(tot["dec"], 1)
    tot["tgt_pd"] = tot["tgt"] / max(tot["dec"], 1)
    tot["maps"] = maps
    tot["takes"] = takes
    tot["mean_takes"] = sum(n_takes) / max(len(n_takes), 1)
    return tot


def main() -> int:
    print("⚠ Reading 1 is VOID (E28 §R3/§R5). §4's branches are JOINT "
          "conditions on readings 1 and 2, so NO verdict on the copycat\n"
          "  hypothesis is available from this run. Reading 2 is reported as a "
          "standalone measurement.\n")
    res = {s: run(s) for s in ("us", "them")}

    print(f"{'=' * 72}\n=== READING 2 — commitment switches ===\n")
    print(f"{'side':<7}{'seats':>7}{'decisions':>11}"
          f"{'attacker sw':>13}{'/game':>8}{'/dec':>9}"
          f"{'target sw':>11}{'/game':>8}{'/dec':>9}")
    for s in ("us", "them"):
        r = res[s]
        print(f"{s:<7}{r['games']:>7}{r['dec']:>11}"
              f"{r['att']:>13}{r['att_pg']:>8.2f}{r['att_pd']:>9.4f}"
              f"{r['tgt']:>11}{r['tgt_pg']:>8.2f}{r['tgt_pd']:>9.4f}")

    u, t = res["us"], res["them"]
    for nm, ku, nu, kt, nt in (
            ("attacker switches / decision", u["att"], u["dec"], t["att"], t["dec"]),
            ("target switches / decision", u["tgt"], u["dec"], t["tgt"], t["dec"])):
        lu, hu = wilson(ku, nu)
        lt, ht = wilson(kt, nt)
        d = ku / nu - kt / nt
        se = math.sqrt(ku / nu * (1 - ku / nu) / nu + kt / nt * (1 - kt / nt) / nt)
        print(f"\n  {nm}")
        print(f"    us   {ku/nu:.4f}  [{lu:.4f}, {hu:.4f}]")
        print(f"    them {kt/nt:.4f}  [{lt:.4f}, {ht:.4f}]")
        print(f"    us - them = {d:+.4f}  [{d - 1.96 * se:+.4f}, "
              f"{d + 1.96 * se:+.4f}]   z = {d / se if se else 0:+.2f}")

    print(f"\n{'=' * 72}\n=== READING 3 — realised prize maps (descriptive) ===\n")
    print("🔴 TRUNCATION, stated before the numbers: the `visualize` stream "
          "ends at the LAST DECISION, so the\n"
          "   final KO(s) are not observed. A game seat 0 WON was seen with 3 "
          "prizes still on its board.\n"
          "   ⇒ full-sequence map classification (2-2-2 vs 1-1-2-2) is NOT "
          "reliable from this extraction.\n"
          "   The take-SIZE composition below is robust to losing the tail; "
          "the sequences are shown only\n"
          "   as a shape, and mildly under-count 2-takes because a finishing "
          "KO on an ex is the likeliest\n"
          "   take to be truncated away.\n")
    print(f"{'side':<7}{'takes seen':>12}{'1-prize':>10}{'2-prize':>10}"
          f"{'%2-prize':>10}{'mean takes/game':>18}")
    for s in ("us", "them"):
        r = res[s]
        one, two = r["takes"][1], r["takes"][2]
        n = sum(r["takes"].values())
        print(f"{s:<7}{n:>12}{one:>10}{two:>10}"
              f"{(two / n if n else 0):>10.1%}{r['mean_takes']:>18.2f}")
    print()
    for s in ("us", "them"):
        print(f"  {s} — observed prefix shapes:")
        for m, c in res[s]["maps"].most_common(6):
            print(f"    {m:<16}{c:>6}  {c / max(res[s]['games'], 1):6.1%}")
    print("\n⛔ Reading 3 is vocabulary for STRATEGY.md, not a feature and not "
          "a rule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
