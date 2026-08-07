"""F1 — where do the 1150+ pilots CONFIDENTLY diverge from our clone, in the MIRROR?

⛔ **This is an AUDIT of experts, not training on them.** B7 trained on expert
actions and lost −55/−92 Elo (§8t/§8u); E3's near-tie relabelling is closed
(§8bd). The object here is the OPPOSITE end of the margin distribution from
E3's: decisions where the clone is **confident** and the expert did something
else anyway. Output is a rule candidate (rule 11: dominated options only) or a
priced defect — never a training target.

Method: score the net over an expert mirror corpus, keep rows where the expert's
action is not the clone's top-1 **and** the clone's probability margin is large,
then cluster by (context, what the clone wanted, what the expert took) and size
each cluster in firings per game — the gate that killed Morgrem (0.2), Pokegear
(0.27) and the Archaludon rule (0.187).

    python -X utf8 scripts/p66_mirror_disagree.py --net out/policy_v5_s2.npz \\
        --ds artifacts/pds_mirror_exp --equiv --margin 0.25

⚠ `--equiv` is not optional in practice: 7.8% of rows corpus-wide (32.4% of
TO_HAND) offer bitwise-identical options, and charging the expert for that coin
flip manufactures a disagreement cluster out of two copies of one card (§8x).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from context_accuracy import BAGS, CTX_NAME, Net, bag_means  # noqa: E402
from sa.features import N_EXTRA  # noqa: E402
from sa.optfeat import pool_scalars  # noqa: E402
from sa import cards as cdb  # noqa: E402


def _nm(cid: int) -> str:
    if cid <= 0:
        return "-"
    try:
        return str(cdb.card(int(cid)).get("name") or f"#{cid}")
    except Exception:  # noqa: BLE001
        return f"#{cid}"


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="out/policy_v5_s2.npz")
    ap.add_argument("--ds", default="artifacts/pds_mirror_exp")
    ap.add_argument("--equiv", action="store_true")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="keep a disagreement only if p(clone top-1) - "
                         "p(expert action) exceeds this")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--dump", default=None,
                    help="write every kept disagreement to this JSONL "
                         "(the input to step 5, watching the clusters)")
    args = ap.parse_args()

    net = Net(ROOT / args.net)
    paths = sorted((ROOT / args.ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / args.ds}")

    games: set[int] = set()
    n_rows = n_dis = n_conf = n_equiv = 0
    by_ctx: Counter = Counter()             # confident disagreements per ctx
    ctx_rows: Counter = Counter()
    clusters: Counter = Counter()
    cl_margin: dict[tuple, float] = defaultdict(float)
    cl_games: dict[tuple, set] = defaultdict(set)
    margins: list[float] = []
    dump: list[dict] = []

    for path in paths:
        z = np.load(path)
        gid, off = z["gid"], z["opt_off"]
        n = len(gid)
        val = np.arange(n)          # never trained on: score EVERY row (§8p)
        width = net.bag_emb.shape[1]
        means = [bag_means(z, nm, n, width, net.bag_emb) for nm in BAGS]
        xd = z["xdense"][val] if "xdense" in z else None
        xs = z["xslots"][val].astype(np.int64) if "xslots" in z else None
        if net.x_mask is not None and xd is not None:
            xd = xd * net.x_mask[:N_EXTRA]
            xs = np.where(net.x_mask[N_EXTRA:] > 0, xs, 0)
        ctx = np.rint(z["seld"][:, 13] * 50.0).astype(int)
        opt_dense, chosen = z["opt_dense"], z["opt_chosen"]
        card = z["opt_card"].astype(np.int64)
        atk = z["opt_attack"].astype(np.int64)
        tgt = (z["opt_target"] if "opt_target" in z
               else np.zeros_like(card)).astype(np.int64)

        pool = None
        if net.n_pool:
            ow = net.opt_in
            oenc = np.concatenate([opt_dense[:, :ow], net.card_emb[card],
                                   net.atk_emb[atk], net.card_emb[tgt]], axis=1)
            pool = np.zeros((len(val), net.n_pool), dtype=np.float32)
            d = oenc.shape[1]
            for k, row in enumerate(val):
                a, b = off[row], off[row + 1]
                if b <= a:
                    continue
                blk = oenc[a:b]
                pool[k, :d] = blk.mean(axis=0)
                pool[k, d:2 * d] = blk.max(axis=0)
                pool[k, 2 * d:] = pool_scalars(b - a)
        srepr = net.state_repr(z["dense"][val], z["slots"][val].astype(np.int64),
                               [m[val] for m in means], z["seld"][val], xd, xs,
                               pool)

        keys = None
        if args.equiv:
            raw = np.ascontiguousarray(np.concatenate(
                [np.ascontiguousarray(x).view(np.uint8).reshape(len(card), -1)
                 for x in (opt_dense, card, atk, tgt)], axis=1))
            keys = raw.view([("k", np.void, raw.shape[1])]).reshape(-1)

        for k, row in enumerate(val):
            a, b = off[row], off[row + 1]
            ch = chosen[a:b]
            if ch.sum() != 1:
                continue
            n_rows += 1
            games.add(int(gid[row]))
            # ⚠ Slice to the width this net was TRAINED at. `optfeat.OPT_DENSE`
            # grew 37 -> 46 on the merge and `policy_v5*` is a 37-column net, so
            # feeding the corpus's full row is a 350-vs-341 matmul error at
            # best and a silently shifted feature vector at worst.
            logits = net.option_logits(
                np.repeat(srepr[k][None, :], b - a, axis=0),
                opt_dense[a:b, :net.opt_in], card[a:b], atk[a:b], tgt[a:b])
            am = int(np.argmax(logits))
            ex = int(np.argmax(ch))
            if am == ex:
                continue
            if keys is not None and keys[a + am] == keys[a + ex]:
                n_equiv += 1          # the same card in the same role (§8x)
                continue
            n_dis += 1
            p = _softmax(logits)
            m = float(p[am] - p[ex])
            margins.append(m)
            c = int(ctx[row])
            ctx_rows[c] += 1
            if m < args.margin:
                continue
            n_conf += 1
            by_ctx[c] += 1
            key = (CTX_NAME.get(c, str(c)),
                   f"{_nm(card[a + am])}"
                   + (f"->{_nm(tgt[a + am])}" if tgt[a + am] > 0 else ""),
                   f"{_nm(card[a + ex])}"
                   + (f"->{_nm(tgt[a + ex])}" if tgt[a + ex] > 0 else ""))
            clusters[key] += 1
            cl_margin[key] += m
            cl_games[key].add(int(gid[row]))
            if args.dump is not None:
                dump.append({"gid": int(gid[row]), "ctx": key[0],
                             "clone": key[1], "expert": key[2],
                             "margin": round(m, 4), "n_opts": int(b - a)})

    ng = max(len(games), 1)
    print(f"\nnet {args.net} over {args.ds}: {n_rows} single-choice decisions "
          f"in {len(games)} mirror games")
    print(f"  disagreements               {n_dis:>7}  "
          f"{n_dis / max(n_rows, 1):6.1%} of decisions")
    if args.equiv:
        print(f"  (equivalent-option ties dropped: {n_equiv})")
    print(f"  CONFIDENT (margin >= {args.margin:.2f}) {n_conf:>7}  "
          f"{n_conf / max(n_rows, 1):6.1%} of decisions, "
          f"{n_conf / ng:.2f}/game")
    if margins:
        q = np.quantile(margins, [0.5, 0.75, 0.9, 0.99])
        print(f"  margin quantiles  p50={q[0]:.3f} p75={q[1]:.3f} "
              f"p90={q[2]:.3f} p99={q[3]:.3f}")

    print(f"\n=== CONFIDENT DISAGREEMENTS BY CONTEXT (n={n_conf}) ===")
    print(f"  {'context':<26}{'rows':>8}{'confident':>11}{'per game':>10}")
    for c, cnt in by_ctx.most_common(args.top):
        print(f"  {CTX_NAME.get(c, str(c)):<26}{ctx_rows[c]:>8}{cnt:>11}"
              f"{cnt / ng:>10.2f}")

    print(f"\n=== CLUSTERS, ranked by firings x mean margin "
          f"(gate: >= 0.50 firings/game = {0.5 * ng:.0f}) ===")
    print(f"  {'context':<22}{'clone wanted':<26}{'expert took':<26}"
          f"{'n':>6}{'/game':>7}{'margin':>8}{'games':>7}")
    ranked = sorted(clusters, key=lambda k: -(clusters[k] * cl_margin[k]
                                              / max(clusters[k], 1)))
    for key in ranked[:args.top]:
        cnt = clusters[key]
        mark = "✅" if cnt / ng >= 0.5 else "  "
        print(f"{mark}{key[0]:<22}{key[1]:<26}{key[2]:<26}{cnt:>6}"
              f"{cnt / ng:>7.2f}{cl_margin[key] / cnt:>8.3f}"
              f"{len(cl_games[key]):>7}")

    passed = [k for k in clusters if clusters[k] / ng >= 0.5]
    print(f"\n{len(passed)} of {len(clusters)} clusters pass the 0.5 "
          f"firings/game sizing gate")

    if args.dump is not None:
        Path(args.dump).write_text(
            "\n".join(json.dumps(d) for d in dump) + "\n", encoding="utf-8")
        print(f"wrote {len(dump)} rows -> {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
