"""E26 sizing: WHERE in a game does the expert clone want a different move?

Rule 14 -- size before you build. This costs no arena games and it decides the
DESIGN of E26, not its verdict:

* If the expert net's argmax differs from ours at a roughly uniform rate across
  the game, a phase-localized substitution is just a rate knob and E25 already
  priced rate knobs (deviating costs -0.389 at 23%).
* If disagreement CONCENTRATES in a phase, then phase is a real axis and the
  arms are worth running -- a phase where they disagree a lot is where their
  policy difference lives.

⚠ **This probe sets the DESIGN, never the control's rate.** The control is
matched to the treatment's *realised* rate measured in the arena (E25 cell B),
because offline sizing does not predict on-policy firing in a fixed direction --
it came in under for `fscrap` (0.261 vs 0.300) and 1.6x over for `fstad`
(EVIDENCE 8cc/8ce).

⚠ And it is measured on the state distribution named by --ds. Ours is the one
that matters: the agent meets its own trajectories, not the expert's (8s ruled
out covariate shift for the b7 arm, so both are informative, but they are not
the same question).

    python -X utf8 scripts/p91_phase_disagree.py \\
        --a out/policy_v5_s2.npz --b out/policy_b7_ntum.npz \\
        --ds artifacts/pds_ours_mirror artifacts/pds_ntum_r
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from context_accuracy import Net, bag_means, BAGS  # noqa: E402

# features.featurize writes dense[0] = min(turn, 40) / 40.0 (agents/sa/features.py:269).
TURN_COL, TURN_SCALE = 0, 40.0

# Frozen before any arena game. The boundaries are not tuned -- 1-4 is the setup
# phase (bench, attach, no attack for the player going first), 5-9 is the
# contested middle, and 10+ is where the 1150s were measured to STOP searching
# (EVIDENCE 8bj, Spikemuth Gym at turn ~9.7) while we never stop.
PHASES = (("early", 1, 4), ("mid", 5, 9), ("late", 10, 99))


def argmax_rows(net: Net, z, rows: np.ndarray) -> np.ndarray:
    """The option index each net would take, per row."""
    off = z["opt_off"]
    n = len(z["gid"])
    width = net.bag_emb.shape[1]
    means = [bag_means(z, nm, n, width, net.bag_emb) for nm in BAGS]
    srepr = net.state_repr(z["dense"][rows], z["slots"][rows].astype(np.int64),
                           [m[rows] for m in means], z["seld"][rows])
    opt_dense, card = z["opt_dense"], z["opt_card"].astype(np.int64)
    atk = z["opt_attack"].astype(np.int64)
    tgt = (z["opt_target"] if "opt_target" in z
           else np.zeros_like(card)).astype(np.int64)
    out = np.empty(len(rows), dtype=np.int64)
    for k, row in enumerate(rows):
        a, b = off[row], off[row + 1]
        out[k] = int(np.argmax(net.option_logits(
            np.repeat(srepr[k][None, :], b - a, axis=0),
            opt_dense[a:b], card[a:b], atk[a:b], tgt[a:b])))
    return out


def load_shards(ds: str):
    shards = sorted(Path(ds).glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {ds}")
    return shards


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="out/policy_v5_s2.npz", help="our net")
    ap.add_argument("--b", default="out/policy_b7_ntum.npz", help="expert clone")
    ap.add_argument("--ds", nargs="+", required=True)
    ap.add_argument("--max-rows", type=int, default=20000,
                    help="cap per dataset; rows are a contiguous prefix so "
                         "whole games stay intact")
    args = ap.parse_args()

    net_a, net_b = Net(args.a), Net(args.b)
    print(f"A = {args.a}\nB = {args.b}\n")

    for ds in args.ds:
        agree_n = np.zeros(len(PHASES), dtype=np.int64)
        total_n = np.zeros(len(PHASES), dtype=np.int64)
        opts_n = np.zeros(len(PHASES), dtype=np.int64)
        multi_n = np.zeros(len(PHASES), dtype=np.int64)
        for shard in load_shards(ds):
            z = np.load(shard, allow_pickle=True)
            off = z["opt_off"]
            n = len(z["gid"])
            rows = np.arange(min(n, args.max_rows))
            # A decision with one option is not a decision. Every rate here is
            # per REAL select -- 8cb's rule 21: pick the unit before reading
            # the number.
            sizes = off[1:] - off[:-1]
            rows = rows[sizes[rows] >= 2]
            if not len(rows):
                continue
            turns = np.rint(z["dense"][rows, TURN_COL] * TURN_SCALE).astype(int)
            pa, pb = argmax_rows(net_a, z, rows), argmax_rows(net_b, z, rows)
            for pi, (_, lo, hi) in enumerate(PHASES):
                m = (turns >= lo) & (turns <= hi)
                total_n[pi] += int(m.sum())
                agree_n[pi] += int((pa[m] == pb[m]).sum())
                opts_n[pi] += int(sizes[rows[m]].sum())
                multi_n[pi] += int((sizes[rows[m]] >= 3).sum())

        print(f"--- {ds} ---")
        print(f"{'phase':<8}{'selects':>9}{'disagree':>10}{'rate':>8}"
              f"{'chance':>9}{'opts':>7}")
        for pi, (name, lo, hi) in enumerate(PHASES):
            t = int(total_n[pi])
            if not t:
                print(f"{name:<8}{0:>9}")
                continue
            dis = t - int(agree_n[pi])
            mean_opts = opts_n[pi] / t
            # What two INDEPENDENT policies would disagree at, given the option
            # counts actually present -- so a phase with more options is not
            # read as "they differ more there".
            chance = 1.0 - 1.0 / mean_opts
            print(f"{name:<8}{t:>9d}{dis:>10d}{dis/t:>8.3f}"
                  f"{chance:>9.3f}{mean_opts:>7.2f}")
        tt, ta = int(total_n.sum()), int(agree_n.sum())
        print(f"{'ALL':<8}{tt:>9d}{tt-ta:>10d}{(tt-ta)/max(tt,1):>8.3f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
