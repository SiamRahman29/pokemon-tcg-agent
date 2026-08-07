"""Rank EVERY option class by its per-TURN gap — rule 21 applied properly.

`p66_mirror_disagree` ranks clusters per DECISION, and §8bj proved that unit
inflates a within-turn ordering difference by ~25x. `p67_option_rate` fixes the
unit but only for one class you already suspected — and the class you suspect
comes from the per-decision ranking, so the bad unit still chooses what gets
looked at.

This closes that loop: for every (card, option type) class in the corpus,
compute the share of AVAILABLE TURNS in which each side actually used it, on
both an expert corpus and our own on-policy corpus, and rank by the gap times
its exposure. Nothing is pre-selected.

    python -X utf8 scripts/p70_perturn_sweep.py \\
        --expert artifacts/pds_mirror_exp --ours artifacts/pds_ours_mirror1

⚠ Both corpora must be ONE SEAT per game or the per-game rates differ by 2x
(§8bj: `pds_ours_mirror` counts both seats, `pds_ours_mirror1` does not).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from cg.api import OptionType  # noqa: E402
from sa.optfeat import N_OPTION_TYPES  # noqa: E402
from sa import cards as cdb  # noqa: E402

TYPE_NAME = {int(getattr(OptionType, n)): n
             for n in dir(OptionType) if n.isupper()}


def _nm(cid: int) -> str:
    if cid <= 0:
        return "-"
    try:
        return str(cdb.card(int(cid)).get("name") or f"#{cid}")
    except Exception:  # noqa: BLE001
        return f"#{cid}"


def rates(ds: Path) -> tuple[dict, int]:
    """class -> (turns available, turns used); plus the game count."""
    avail: dict = defaultdict(set)      # class -> {(gid, turn)}
    used: dict = defaultdict(set)
    games: set = set()
    for path in sorted(ds.rglob("shard_*.npz")):
        z = np.load(path)
        off, gid = z["opt_off"], z["gid"]
        card = z["opt_card"].astype(int)
        chosen = z["opt_chosen"]
        od = z["opt_dense"]
        otype = np.argmax(od[:, :N_OPTION_TYPES], axis=1)
        turn = np.rint(z["dense"][:, 0] * 40.0).astype(int)
        names = np.array([_nm(c) for c in card])
        for r in range(len(off) - 1):
            a, b = off[r], off[r + 1]
            ch = chosen[a:b]
            if ch.sum() != 1:
                continue
            g = int(gid[r])
            games.add(g)
            key_turn = (g, int(turn[r]))
            seen = set()
            for i in range(a, b):
                k = (names[i], TYPE_NAME.get(int(otype[i]), str(otype[i])))
                if k not in seen:
                    seen.add(k)
                    avail[k].add(key_turn)
            k_ch = (names[a + int(np.argmax(ch))],
                    TYPE_NAME.get(int(otype[a + int(np.argmax(ch))]),
                                  str(otype[a + int(np.argmax(ch))])))
            used[k_ch].add(key_turn)
    return {k: (len(v), len(used.get(k, ()))) for k, v in avail.items()}, len(games)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", default="artifacts/pds_mirror_exp")
    ap.add_argument("--ours", default="artifacts/pds_ours_mirror1")
    ap.add_argument("--min-turns", type=int, default=40,
                    help="hide classes rarer than this on either side")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    ex, ng_ex = rates(ROOT / args.expert)
    ou, ng_ou = rates(ROOT / args.ours)
    print(f"expert {args.expert}: {ng_ex} games, {len(ex)} option classes")
    print(f"ours   {args.ours}: {ng_ou} games, {len(ou)} option classes")

    rows = []
    for k in set(ex) & set(ou):
        ae, ue = ex[k]
        ao, uo = ou[k]
        if ae < args.min_turns or ao < args.min_turns:
            continue
        re_, ro = ue / ae, uo / ao
        # exposure in OUR turns per game x the rate gap = extra uses per game
        gap_per_game = (ro - re_) * (ao / ng_ou)
        rows.append((abs(gap_per_game), gap_per_game, k, ae, re_, ao, ro))
    rows.sort(reverse=True)

    print(f"\n=== EVERY option class, ranked by PER-TURN gap x exposure ===")
    print(f"  {'card':<26}{'type':<10}{'exp turns':>10}{'exp use':>9}"
          f"{'our turns':>10}{'our use':>9}{'extra/game':>12}")
    for _, gap, k, ae, re_, ao, ro in rows[:args.top]:
        mark = "✅" if abs(gap) >= 0.5 else "  "
        print(f"{mark}{k[0][:25]:<26}{k[1]:<10}{ae:>10}{re_:>9.1%}"
              f"{ao:>10}{ro:>9.1%}{gap:>+12.2f}")

    over = [r for r in rows if r[0] >= 0.5]
    print(f"\n{len(over)} of {len(rows)} classes exceed the 0.5 uses/game gate "
          f"on the PER-TURN measure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
