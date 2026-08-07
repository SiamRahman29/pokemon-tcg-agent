"""Size ONE option class per-turn, not per-decision — the §8ai correction.

`p66_mirror_disagree` ranks disagreements by DECISION. That is the detector
§8ai warned about: our first empty-bench detector counted a pilot as "declining
to bench" when it benched later in the same turn, and `rule:archaludon` looked
broken because of it. A clone that wants Munkidori's ability at action 1 and an
expert who uses it at action 4 disagree on **four decisions** and do the **same
thing** — and sequencing is a closed axis (E10).

So this asks the ordering-free question: over the turns where the option class
is AVAILABLE, how often does the expert use it at all, and how often would the
clone? Plus the rate question the per-decision view cannot answer: what is the
denominator?

    python -X utf8 scripts/p67_option_rate.py --card Munkidori --type ABILITY \\
        --net out/policy_v5_s2.npz --ds artifacts/pds_mirror_exp

⚠ The clone column is OFF-POLICY: it is the clone's top-1 evaluated in the
EXPERT's state. After the first divergence the states are no longer comparable,
so read it as "would have wanted it here", never as "would have played this
game this way".
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
from context_accuracy import BAGS, CTX_NAME, Net, bag_means  # noqa: E402
from sa.features import N_EXTRA  # noqa: E402
from sa.optfeat import N_OPTION_TYPES, pool_scalars  # noqa: E402
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="out/policy_v5_s2.npz")
    ap.add_argument("--ds", default="artifacts/pds_mirror_exp")
    ap.add_argument("--card", required=True, help="card name, exact")
    ap.add_argument("--type", default=None,
                    help="restrict to this OptionType (e.g. ABILITY, PLAY); "
                         "default: any type")
    args = ap.parse_args()

    want_t = None
    if args.type:
        want_t = int(getattr(OptionType, args.type.upper()))

    net = Net(ROOT / args.net)
    paths = sorted((ROOT / args.ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / args.ds}")

    rows_avail = rows_expert = rows_clone = 0
    n_rows = 0
    games: set[int] = set()
    # (gid, turn) -> [available, expert used, clone wanted]
    turns: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
    per_game_expert: dict[int, int] = defaultdict(int)
    per_game_clone: dict[int, int] = defaultdict(int)
    by_ctx: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])

    for path in paths:
        z = np.load(path)
        gid, off = z["gid"], z["opt_off"]
        n = len(gid)
        val = np.arange(n)
        width = net.bag_emb.shape[1]
        means = [bag_means(z, nm, n, width, net.bag_emb) for nm in BAGS]
        xd = z["xdense"] if "xdense" in z else None
        xs = z["xslots"].astype(np.int64) if "xslots" in z else None
        if net.x_mask is not None and xd is not None:
            xd = xd * net.x_mask[:N_EXTRA]
            xs = np.where(net.x_mask[N_EXTRA:] > 0, xs, 0)
        ctx = np.rint(z["seld"][:, 13] * 50.0).astype(int)
        turn = np.rint(z["dense"][:, 0] * 40.0).astype(int)
        opt_dense, chosen = z["opt_dense"], z["opt_chosen"]
        card = z["opt_card"].astype(np.int64)
        atk = z["opt_attack"].astype(np.int64)
        tgt = (z["opt_target"] if "opt_target" in z
               else np.zeros_like(card)).astype(np.int64)
        otype = np.argmax(opt_dense[:, :N_OPTION_TYPES], axis=1)
        names = np.array([_nm(c) for c in card])
        hit_opt = (names == args.card)
        if want_t is not None:
            hit_opt &= (otype == want_t)

        pool = None
        if net.n_pool:
            ow = net.opt_in
            oenc = np.concatenate([opt_dense[:, :ow], net.card_emb[card],
                                   net.atk_emb[atk], net.card_emb[tgt]], axis=1)
            pool = np.zeros((n, net.n_pool), dtype=np.float32)
            d = oenc.shape[1]
            for row in range(n):
                a, b = off[row], off[row + 1]
                if b <= a:
                    continue
                blk = oenc[a:b]
                pool[row, :d] = blk.mean(axis=0)
                pool[row, d:2 * d] = blk.max(axis=0)
                pool[row, 2 * d:] = pool_scalars(b - a)
        srepr = net.state_repr(z["dense"], z["slots"].astype(np.int64),
                               means, z["seld"], xd, xs, pool)

        for row in val:
            a, b = off[row], off[row + 1]
            ch = chosen[a:b]
            if ch.sum() != 1:
                continue
            n_rows += 1
            g = int(gid[row])
            games.add(g)
            avail = hit_opt[a:b]
            key = (g, int(turn[row]))
            if not avail.any():
                continue
            rows_avail += 1
            turns[key][0] = 1
            c = int(ctx[row])
            by_ctx[c][0] += 1

            ex = int(np.argmax(ch))
            if avail[ex]:
                rows_expert += 1
                turns[key][1] = 1
                per_game_expert[g] += 1
                by_ctx[c][1] += 1

            logits = net.option_logits(
                np.repeat(srepr[row][None, :], b - a, axis=0),
                opt_dense[a:b, :net.opt_in], card[a:b], atk[a:b], tgt[a:b])
            am = int(np.argmax(logits))
            if avail[am]:
                rows_clone += 1
                turns[key][2] = 1
                per_game_clone[g] += 1
                by_ctx[c][2] += 1

    ng = max(len(games), 1)
    what = f"{args.card}" + (f" [{args.type.upper()}]" if args.type else "")
    print(f"\n{what} over {args.ds} — {n_rows} decisions, {len(games)} games\n")

    print("=== PER DECISION (what p66 ranks on) ===")
    print(f"  decisions where it is AVAILABLE   {rows_avail:>7}"
          f"  {rows_avail / ng:6.2f}/game")
    print(f"  expert took it                    {rows_expert:>7}"
          f"  {rows_expert / max(rows_avail, 1):6.1%} of available")
    print(f"  clone's top-1 was it              {rows_clone:>7}"
          f"  {rows_clone / max(rows_avail, 1):6.1%} of available")

    n_t = len(turns)
    t_ex = sum(v[1] for v in turns.values())
    t_cl = sum(v[2] for v in turns.values())
    both = sum(1 for v in turns.values() if v[1] and v[2])
    only_cl = sum(1 for v in turns.values() if v[2] and not v[1])
    only_ex = sum(1 for v in turns.values() if v[1] and not v[2])
    print("\n=== PER TURN (ordering-free — the §8ai correction) ===")
    print(f"  turns where it is AVAILABLE       {n_t:>7}"
          f"  {n_t / ng:6.2f}/game")
    print(f"  expert used it in the turn        {t_ex:>7}"
          f"  {t_ex / max(n_t, 1):6.1%} of those turns")
    print(f"  clone wanted it in the turn       {t_cl:>7}"
          f"  {t_cl / max(n_t, 1):6.1%} of those turns")
    print(f"  BOTH                              {both:>7}")
    print(f"  clone only (the real gap)         {only_cl:>7}"
          f"  {only_cl / ng:6.2f}/game")
    print(f"  expert only                       {only_ex:>7}"
          f"  {only_ex / ng:6.2f}/game")

    ex_pg = sum(per_game_expert.values()) / ng
    cl_pg = sum(per_game_clone.values()) / ng
    print(f"\n=== USES PER GAME ===")
    print(f"  expert {ex_pg:5.2f}   clone (off-policy) {cl_pg:5.2f}   "
          f"ratio {cl_pg / max(ex_pg, 1e-9):4.2f}x")

    print("\n=== BY CONTEXT (available / expert took / clone wanted) ===")
    for c, v in sorted(by_ctx.items(), key=lambda kv: -kv[1][0])[:10]:
        print(f"  {CTX_NAME.get(c, str(c)):<26}{v[0]:>7}{v[1]:>9}{v[2]:>9}"
              f"    expert {v[1] / max(v[0], 1):5.1%}  clone "
              f"{v[2] / max(v[0], 1):5.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
