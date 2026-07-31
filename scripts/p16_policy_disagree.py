"""Is the expert clone a DIFFERENT POLICY, or just a clone measured off-support?

The covariate-shift objection to EVIDENCE §8q/§8r: agreement with a demonstrator
is always measured on the *demonstrator's own* trajectories, so a strong pilot
reaching board states our clone rarely occupies inflates disagreement without
either policy being better. Low agreement would then be behavior cloning's
compounding-error problem, not a copyable policy.

The discriminator is to stop comparing each policy against HUMAN labels and
compare the two policies against EACH OTHER, on both state distributions:

    python scripts/p16_policy_disagree.py --a out/policy_b1_v3.npz \\
        --b out/policy_b7_ntum.npz --ds artifacts/pds_ours artifacts/pds_ntum_r

* Roughly SYMMETRIC disagreement => the two policies genuinely differ, and the
  §8q gap is a real policy difference the arena can adjudicate.
* Disagreement that COLLAPSES on our own states => the "expert policy" is only
  distinguishable where we never go, i.e. it was covariate shift.

⚠ This says nothing about which policy is better. Only an arena A/B does.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from context_accuracy import Net, bag_means, BAGS, CTX_NAME  # noqa: E402


def argmax_per_row(net: Net, z, rows: np.ndarray) -> dict[int, int]:
    """row index -> the option this net would take."""
    off = z["opt_off"]
    n = len(z["gid"])
    width = net.bag_emb.shape[1]
    means = [bag_means(z, nm, n, width, net.bag_emb) for nm in BAGS]
    srepr = net.state_repr(z["dense"][rows], z["slots"][rows].astype(np.int64),
                           [m[rows] for m in means], z["seld"][rows])
    opt_dense = z["opt_dense"]
    card = z["opt_card"].astype(np.int64)
    atk = z["opt_attack"].astype(np.int64)
    tgt = (z["opt_target"] if "opt_target" in z
           else np.zeros_like(card)).astype(np.int64)
    out = {}
    for k, row in enumerate(rows):
        a, b = off[row], off[row + 1]
        logits = net.option_logits(
            np.repeat(srepr[k][None, :], b - a, axis=0),
            opt_dense[a:b], card[a:b], atk[a:b], tgt[a:b])
        out[int(row)] = int(np.argmax(logits))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="policy A (.npz)")
    ap.add_argument("--b", required=True, help="policy B (.npz)")
    ap.add_argument("--ds", nargs="+", required=True,
                    help="one or more corpora = one or more STATE "
                         "DISTRIBUTIONS to compare the policies on")
    ap.add_argument("--min-rows", type=int, default=200)
    args = ap.parse_args()

    a_net, b_net = Net(ROOT / args.a), Net(ROOT / args.b)
    print(f"A = {args.a}\nB = {args.b}\n")
    print(f"{'state distribution':<28}{'rows':>8}{'A!=B':>9}{'A!=human':>10}"
          f"{'B!=human':>10}")
    per_ctx: dict[str, dict[int, tuple[int, int]]] = {}
    for ds in args.ds:
        paths = sorted((ROOT / ds).rglob("shard_*.npz"))
        if not paths:
            raise SystemExit(f"no shards under {ROOT / ds}")
        n = diff = a_miss = b_miss = 0
        ctx_acc: dict[int, tuple[int, int]] = {}
        for path in paths:
            z = np.load(path)
            off, chosen = z["opt_off"], z["opt_chosen"]
            # single-choice rows only, so argmax is the whole decision
            rows = np.array([i for i in range(len(z["gid"]))
                             if chosen[off[i]:off[i + 1]].sum() == 1])
            if not len(rows):
                continue
            pa = argmax_per_row(a_net, z, rows)
            pb = argmax_per_row(b_net, z, rows)
            ctx = np.rint(z["seld"][:, 13] * 50.0).astype(int)
            for row in rows:
                a_i, b_i = pa[int(row)], pb[int(row)]
                lo = off[row]
                human = int(np.argmax(chosen[lo:off[row + 1]]))
                n += 1
                d = int(a_i != b_i)
                diff += d
                a_miss += int(a_i != human)
                b_miss += int(b_i != human)
                c = int(ctx[row])
                t, k = ctx_acc.get(c, (0, 0))
                ctx_acc[c] = (t + d, k + 1)
        print(f"{ds:<28}{n:>8}{diff / max(n, 1):>9.1%}"
              f"{a_miss / max(n, 1):>10.1%}{b_miss / max(n, 1):>10.1%}")
        per_ctx[ds] = ctx_acc

    print("\nwhere the two policies differ, by context "
          f"(>= {args.min_rows} rows):")
    ctxs = sorted({c for m in per_ctx.values() for c in m},
                  key=lambda c: -max(m.get(c, (0, 0))[1]
                                     for m in per_ctx.values()))
    head = f"{'context':<24}" + "".join(f"{Path(d).name[-14:]:>16}"
                                        for d in args.ds)
    print(head)
    for c in ctxs:
        cells = []
        big = False
        for d in args.ds:
            t, k = per_ctx[d].get(c, (0, 0))
            big |= k >= args.min_rows
            cells.append(f"{t / k:>10.1%}({k:>4})" if k else f"{'-':>16}")
        if big:
            print(f"{CTX_NAME.get(c, str(c)):<24}" + "".join(cells))
    print("\nsymmetric => a real policy difference; collapsing on our own "
          "states => it was covariate shift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
