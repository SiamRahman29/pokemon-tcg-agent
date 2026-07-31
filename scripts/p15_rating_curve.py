"""Does our clone agree less with better players? — the curve, not three points.

EVIDENCE §8q measured top-1 agreement against three demonstrator groups (~1110
control, #3 at 1152, #2 at 1163) and found it falls monotonically as the
demonstrator improves. Three points is a trend, not a curve, and the three
groups came from three different dumps -- so dump, deck and opponent pool are
all confounded with rating.

This measures the same thing INSIDE one corpus, where every row was collected
the same way: `build_policy_dataset.py --ratings` tags each row with the LB
score of the demonstrator who made that choice, and this buckets the net's
agreement by that rating.

    python scripts/p15_rating_curve.py --net out/policy_b1_v3.npz \\
        --ds artifacts/pds_v3r

⚠ DEFAULTS TO THE TRAINER'S HELD-OUT SPLIT (`gid %% 20 == 0`), and that is not
a detail: the net was trained on this corpus. In-sample agreement is inflated
most for the demonstrators with the most seats, and seat count correlates with
rating -- so scoring all rows would manufacture the very relationship this is
testing, with the sign flipped. `--all-rows` is for corpora the net never saw.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from context_accuracy import Net, bag_means, BAGS  # noqa: E402

DEFAULT_EDGES = (0, 900, 1000, 1050, 1100, 1150, 9999)


def wilson(hit: int, n: int) -> tuple[float, float]:
    """95% CI for a proportion. n is rows, which are not independent within a
    game -- treat these as indicative, the way §8q's per-cell n's are."""
    if n == 0:
        return (0.0, 0.0)
    z, p = 1.96, hit / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score_shard(net: Net, z, rows: np.ndarray):
    """-> (hit, rand, rating, team, gid) per scored single-choice row."""
    off = z["opt_off"]
    n = len(z["gid"])
    width = net.bag_emb.shape[1]
    means = [bag_means(z, nm, n, width, net.bag_emb) for nm in BAGS]
    srepr = net.state_repr(z["dense"][rows], z["slots"][rows].astype(np.int64),
                           [m[rows] for m in means], z["seld"][rows])
    opt_dense, chosen = z["opt_dense"], z["opt_chosen"]
    card = z["opt_card"].astype(np.int64)
    atk = z["opt_attack"].astype(np.int64)
    tgt = (z["opt_target"] if "opt_target" in z
           else np.zeros_like(card)).astype(np.int64)
    rating, team, gid = z["rating"], z["team_id"], z["gid"]
    sub = z["sub_id"] if "sub_id" in z else np.full(n, -1, dtype=np.int64)
    out = []
    for k, row in enumerate(rows):
        a, b = off[row], off[row + 1]
        ch = chosen[a:b]
        if ch.sum() != 1:          # top-1 is only defined for single-choice
            continue
        k_opts = b - a
        logits = net.option_logits(
            np.repeat(srepr[k][None, :], k_opts, axis=0),
            opt_dense[a:b], card[a:b], atk[a:b], tgt[a:b])
        out.append((float(ch[int(np.argmax(logits))] == 1), 1.0 / k_opts,
                    float(rating[row]), int(team[row]), int(gid[row]),
                    int(sub[row])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="out/policy_b1_v3.npz")
    ap.add_argument("--ds", default="artifacts/pds_v3r")
    ap.add_argument("--all-rows", action="store_true",
                    help="score every row. ONLY correct for a corpus the net "
                         "never trained on -- see the module docstring.")
    ap.add_argument("--min-rows", type=int, default=300,
                    help="hide per-demonstrator rows below this")
    ap.add_argument("--seen-from",
                    help="JSON {team_id: train_rows} describing what the net "
                         "TRAINED on. Required to get a real `0 (unseen "
                         "player)` bucket when --ds is a corpus the net never "
                         "saw: without it, exposure is read from --ds itself, "
                         "which has no unseen demonstrators by construction.")
    args = ap.parse_args()

    net = Net(ROOT / args.net)
    paths = sorted((ROOT / args.ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / args.ds}")

    recs: list[tuple[float, float, float, int, int]] = []
    seen_rows: dict[int, int] = {}   # per team, rows the net TRAINED on
    for path in paths:
        z = np.load(path)
        if "rating" not in z:
            raise SystemExit(f"{path.name} has no `rating` array -- rebuild "
                             "the corpus with `--ratings`")
        gid = z["gid"]
        rows = (np.arange(len(gid)) if args.all_rows
                else np.flatnonzero((gid % 20) == 0))
        tr = z["team_id"][(gid % 20) != 0]
        for t, c in zip(*np.unique(tr, return_counts=True)):
            seen_rows[int(t)] = seen_rows.get(int(t), 0) + int(c)
        if len(rows):
            recs.extend(score_shard(net, z, rows))
        print(f"  {path.parent.name}/{path.name}: {len(recs)} scored",
              file=sys.stderr)

    if not recs:
        raise SystemExit("no single-choice rows scored")
    hit = np.array([r[0] for r in recs])
    rnd = np.array([r[1] for r in recs])
    rating = np.array([r[2] for r in recs])
    team = np.array([r[3] for r in recs])
    gid = np.array([r[4] for r in recs])
    sub = np.array([r[5] for r in recs])
    known = ~np.isnan(rating)

    split = "ALL rows" if args.all_rows else "the held-out gid%20 split"
    print(f"\nnet {args.net} vs {args.ds} on {split}")
    print(f"{len(hit)} single-choice rows, {known.sum()} with a rated "
          f"demonstrator ({known.mean():.1%}), "
          f"{len(np.unique(gid))} games\n")

    print(f"{'demonstrator rating':<22}{'rows':>8}{'games':>7}{'top1':>8}"
          f"{'95% CI':>16}{'random':>8}{'miss':>8}")
    edges = DEFAULT_EDGES
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = known & (rating >= lo) & (rating < hi)
        n = int(m.sum())
        if not n:
            continue
        h = int(hit[m].sum())
        lo_ci, hi_ci = wilson(h, n)
        label = f"{lo}-{hi}" if hi < 9000 else f"{lo}+"
        print(f"{label:<22}{n:>8}{len(np.unique(gid[m])):>7}{h / n:>8.1%}"
              f"{f'[{lo_ci:.3f}, {hi_ci:.3f}]':>16}{rnd[m].mean():>8.1%}"
              f"{1 - h / n:>8.1%}")
    if (~known).sum():
        m = ~known
        print(f"{'(unrated)':<22}{int(m.sum()):>8}{len(np.unique(gid[m])):>7}"
              f"{hit[m].mean():>8.1%}{'':>16}{rnd[m].mean():>8.1%}"
              f"{1 - hit[m].mean():>8.1%}")

    # Per-demonstrator, which is what a scatter plot in the report needs.
    tnames: dict[str, str] = {}
    for tp in (ROOT / args.ds).rglob("teams.json"):
        tnames.update(json.loads(tp.read_text(encoding="utf-8")))
    # ⚠ THE CONFOUND, and it is the whole reason this table exists. The net
    # trained on this corpus, held out by GAME not by PLAYER -- so every
    # demonstrator below has other games of theirs in the training split.
    # If agreement tracks how much of a player the net has SEEN, then §8q's
    # rating trend may be a familiarity trend wearing a rating costume: the
    # two expert dumps are unseen players, and unseen is what they have in
    # common with each other and not with anyone here.
    if args.seen_from:
        raw = json.loads(Path(args.seen_from).read_text(encoding="utf-8"))
        seen_rows = {int(k): int(v) for k, v in raw.items()}
        print(f"\n(exposure read from {args.seen_from}: "
              f"{len(seen_rows)} trained-on demonstrators)")
    train = np.array([seen_rows.get(int(t), 0) for t in team])
    print("\nagreement vs how much of that demonstrator the net TRAINED on:")
    print(f"{'train rows seen':<22}{'rows':>8}{'teams':>7}{'rating':>8}"
          f"{'top1':>8}")
    for lo, hi in ((0, 1), (1, 500), (500, 2000), (2000, 8000),
                   (8000, 10 ** 9)):
        m = known & (train >= lo) & (train < hi)
        n = int(m.sum())
        if not n:
            continue
        label = "0 (unseen player)" if hi == 1 else f"{lo}-{hi}"
        print(f"{label:<22}{n:>8}{len(np.unique(team[m])):>7}"
              f"{rating[m].mean():>8.1f}{hit[m].mean():>8.1%}")

    print(f"\nper demonstrator (>= {args.min_rows} rows):")
    print(f"{'rating':>8}{'rows':>8}{'games':>7}{'seen':>8}{'top1':>8}"
          f"  demonstrator")
    order = []
    for t in np.unique(team[known]):
        m = known & (team == t)
        if int(m.sum()) < args.min_rows:
            continue
        order.append((float(rating[m][0]), int(t), int(m.sum()),
                      len(np.unique(gid[m])), float(hit[m].mean())))
    for r, t, n, g, acc in sorted(order):
        print(f"{r:>8.1f}{n:>8}{g:>7}{seen_rows.get(t, 0):>8}{acc:>8.1%}  "
              f"{tnames.get(str(t), str(t))}")
    shown = sum(o[2] for o in order)
    print(f"({len(order)} demonstrators, {shown} of {int(known.sum())} rated "
          f"rows = {shown / max(int(known.sum()), 1):.0%})")

    # A "demonstrator" is not a person, it is a SUBMISSION. Where the dump
    # carries an `episodes_meta.json` sidecar we can see the same team's two
    # agents separately -- and if agreement differs between them, then
    # "agreement with player X" is dated the moment X uploads a new agent.
    subs = [s for s in np.unique(sub) if s > 0]
    if len(subs) > 1:
        print("\nby SUBMISSION (the same team can be two different agents):")
        print(f"{'submissionId':>14}{'rating':>9}{'rows':>8}{'games':>7}"
              f"{'top1':>8}{'95% CI':>16}")
        for s in subs:
            m = sub == s
            n, h = int(m.sum()), int(hit[m].sum())
            lo_ci, hi_ci = wilson(h, n)
            r = rating[m]
            print(f"{int(s):>14}{np.nanmean(r):>9.1f}{n:>8}"
                  f"{len(np.unique(gid[m])):>7}{h / n:>8.1%}"
                  f"{f'[{lo_ci:.3f}, {hi_ci:.3f}]':>16}")

    if len(order) >= 3:
        # Row-weighted least squares of agreement on rating, over demonstrators.
        # ⚠ Descriptive only: rows within a demonstrator are not independent,
        # so the slope's uncertainty is NOT the naive regression one.
        x = np.array([o[0] for o in order])
        y = np.array([o[4] for o in order])
        w = np.array([o[2] for o in order], dtype=float)
        b, a = np.polyfit(x, y, 1, w=np.sqrt(w))
        xm, ym = np.average(x, weights=w), np.average(y, weights=w)
        cov = np.average((x - xm) * (y - ym), weights=w)
        r = cov / (np.sqrt(np.average((x - xm) ** 2, weights=w))
                   * np.sqrt(np.average((y - ym) ** 2, weights=w)))
        print(f"\n{len(order)} demonstrators, row-weighted fit: "
              f"top1 = {a:.3f} {b:+.5f} x rating  (r = {r:+.3f})")
        print(f"  => {b * 100:+.2f} pp of agreement per +100 rating")
    return 0


if __name__ == "__main__":
    sys.exit(main())
