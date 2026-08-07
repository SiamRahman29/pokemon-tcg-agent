"""E9 gate: how DECORRELATED are two nets, before either joins a vote?

§8be measured the thing this script exists to prevent. Four v5-recipe nets sat
on disk and there were only THREE policies -- `policy_v5c_s1` is 100.0%
decision-identical to `policy_v5_s1` (different md5, same function). And the
honest 3-net vote still LOST (0.491 vs the best member), because
`policy_v5c_s0` agrees with `policy_v5` on 87.5% of decisions: ens3 was
effectively two votes for the v5-ish policy against one for the stronger `s1`.

⇒ **Members must be decorrelated, and 87.5% agreement is already enough to
hurt.** `build_submission.py` refuses byte-identical members but cannot see the
87.5% case -- an md5 says nothing about a function. This does.

Run it before adding ANY member to a vote:

    python -X utf8 scripts/p63_net_agreement.py --nets out/policy_v5.npz,out/policy_v5_s1.npz
    python -X utf8 scripts/p63_net_agreement.py --nets "out/policy_v5*.npz"

`agree` is plain top-1 index agreement -- the number §8be quotes (v5 vs v5_s1 =
77.0%). `agree_eq` additionally counts a match when the two argmax options are
BITWISE IDENTICAL (same dense, card, attack and target): those are the same card
in the same role, so no net reading these inputs could tell them apart and the
disagreement is a coin flip that produces the same game (§8x). **agree_eq is the
honest measure of how differently two nets PLAY**; plain `agree` is kept because
it is what the published number is.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from context_accuracy import BAGS, Net, bag_means  # noqa: E402
from sa.optfeat import pool_scalars  # noqa: E402
from sa.features import N_EXTRA  # noqa: E402


def argmaxes(net: Net, paths: list[Path], all_rows: bool):
    """Top-1 option index (shard-global) per single-choice row, plus a key for
    each so bitwise-identical options can be collapsed."""
    picks: list[int] = []
    keys: list[bytes] = []
    for path in paths:
        z = np.load(path)
        gid, off = z["gid"], z["opt_off"]
        n = len(gid)
        val = (np.arange(n) if all_rows else np.flatnonzero((gid % 20) == 0))
        if not len(val):
            continue
        width = net.bag_emb.shape[1]
        means = [bag_means(z, nm, n, width, net.bag_emb) for nm in BAGS]
        xd = z["xdense"][val] if "xdense" in z else None
        xs = z["xslots"][val].astype(np.int64) if "xslots" in z else None
        if net.x_mask is not None and xd is not None:
            xd = xd * net.x_mask[:N_EXTRA]
            xs = np.where(net.x_mask[N_EXTRA:] > 0, xs, 0)
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

        raw = np.ascontiguousarray(np.concatenate(
            [np.ascontiguousarray(x).view(np.uint8).reshape(len(card), -1)
             for x in (opt_dense, card, atk, tgt)], axis=1))

        for k, row in enumerate(val):
            a, b = off[row], off[row + 1]
            if chosen[a:b].sum() != 1:   # top-1 is only defined single-choice
                continue
            logits = net.option_logits(
                np.repeat(srepr[k][None, :], b - a, axis=0),
                opt_dense[a:b], card[a:b], atk[a:b], tgt[a:b])
            am = a + int(np.argmax(logits))
            picks.append(am)
            keys.append(raw[am].tobytes())
    return np.array(picks), keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", required=True,
                    help="comma-separated paths or globs")
    ap.add_argument("--ds", default="artifacts/pds_v4",
                    help="shard dir. ⚠ NOT artifacts/pds_ours: that corpus is "
                         "v3-era and carries no xdense/xslots, so a v5 net's "
                         "state comes out 668 wide against the 708 it wants "
                         "and the matmul refuses. Any net you compare must "
                         "share a feature generation with the shards.")
    ap.add_argument("--all-rows", action="store_true",
                    help="score EVERY row instead of the trainer's held-out "
                         "gid%%20 split")
    ap.add_argument("--max-agree", type=float, default=85.0,
                    help="gate: flag any pair at or above this agreement. ⚠ "
                         "85 is a MIDPOINT GUESS, not a measured threshold. "
                         "All we know is one pair that helped (v5/v5_s1, 80.8% "
                         "here) and one that hurt (v5/v5c_s0, 88.4% here / "
                         "87.5% in §8be) -- the boundary is somewhere between "
                         "and nothing has measured where. HANDOFF's '~90%' is "
                         "too loose: it would have waved through the exact "
                         "pair that made ens3 lose.")
    args = ap.parse_args()

    names: list[str] = []
    for tok in args.nets.split(","):
        tok = tok.strip()
        if not tok:
            continue
        hits = sorted(glob.glob(str(ROOT / tok)))
        names.extend(hits if hits else [str(ROOT / tok)])
    names = list(dict.fromkeys(names))
    if len(names) < 2:
        raise SystemExit(f"need >=2 nets, matched {len(names)}: {names}")

    paths = sorted((ROOT / args.ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / args.ds}")

    print(f"{len(names)} nets over {args.ds} "
          f"({'all rows' if args.all_rows else 'val split'})\n")
    picks: dict[str, np.ndarray] = {}
    keys: dict[str, list[bytes]] = {}
    for p in names:
        label = Path(p).name
        picks[label], keys[label] = argmaxes(Net(Path(p)), paths, args.all_rows)
        print(f"  scored {label}: {len(picks[label])} single-choice decisions")

    n_rows = {len(v) for v in picks.values()}
    if len(n_rows) != 1:
        raise SystemExit(f"nets disagree on row count: {n_rows} -- one of them "
                         "is a different feature generation")

    print(f"\n{'pair':<52}{'agree':>9}{'agree_eq':>10}  verdict")
    flagged = []
    for a, b in itertools.combinations(picks, 2):
        same = picks[a] == picks[b]
        eq = same | np.array([ka == kb for ka, kb in zip(keys[a], keys[b])])
        pa, pe = same.mean() * 100, eq.mean() * 100
        if pe >= 99.99:
            verdict = "🔴 SAME POLICY -- never vote these together"
        elif pe >= args.max_agree:
            verdict = f"🔴 correlated (>={args.max_agree:.0f}%) -- hurts a vote"
        else:
            verdict = "✅ decorrelated enough to vote"
        if pe >= args.max_agree:
            flagged.append((a, b, pe))
        print(f"{a + ' vs ' + b:<52}{pa:>8.1f}%{pe:>9.1f}%  {verdict}")

    print(f"\n{n_rows.pop()} decisions scored. §8be: 87.5% agreement was "
          "already enough to make a 3-net vote LOSE against its own best "
          "member.")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
