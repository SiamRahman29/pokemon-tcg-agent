"""E27 step 3: write a per-decision TD advantage column into self-play shards.

    A_t = V(s_{t+1}) - V(s_t)          (seat-corrected)
    A_T = won_T      - V(s_T)          (terminal)

**Why TD and not the terminal outcome.** B8 (§8ao) weighted by the game result
and measured a clean null over 20,000 games; §8bv then measured that conditioned
on the board, winners and losers play the same (-0.0024 bits), which is close to
a direct statement that the terminal label carries little about the action. A TD
residual attributes credit to the TRANSITION instead of the game, so it is a
strictly finer signal built from the same labels.

**Why this is not E25's contaminated ranking.** E20/E22/E25 scored UNVISITED
successors of off-policy options, and §8cg found the score is contaminated in
proportion to how far off-policy the option is. Here V is evaluated only at
states the game ACTUALLY REACHED. That is the whole distinction, it is the
reason E27 is licensed at all, and it is an argument rather than a measurement --
see the E27 doc's hypothesis 3.

⚠ **Seat correction is not optional.** `won` and `dense` are written from the
ACTING seat's point of view, and consecutive rows alternate seats whenever the
turn passes. V is P(the acting seat wins), so in a zero-sum two-player game the
next state's value from OUR seat is `1 - V` whenever the mover changed. Getting
this backwards does not crash -- it silently negates the advantage on exactly
the transitions that cross a turn boundary, which is most of the informative
ones.

⚠ **The batched forward is reconciled against the path that PLAYS before any
column is written** (`valuenet.Net.forward`, itself reconciled with the trainer
at 2.7e-7 by `p88_value_equivalence.py`). E20 spent 2,000 games on a value path
that computed a different function than was trained; rule 18 is to compute it a
second way and reconcile FIRST.

    python -X utf8 scripts/p92_td_advantage.py --data artifacts/e27_r1 \\
        --value out/value_e27r1.npz
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "agents"):
    sys.path.insert(0, str(ROOT / _sub))

from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa import valuenet  # noqa: E402

BAGS = ("my_hand", "my_discard", "opp_discard")


def bag_means(z, name: str, n: int, emb: np.ndarray) -> np.ndarray:
    """Mean embedding per row, with an EMPTY bag mapped to ROW 0.

    🔴 Row 0, not zeros. `train_value.py` pads an empty bag with row 0, so the
    weights were fitted against `bag_emb[0]`; substituting zeros computes a
    different function on the ~7% of rows whose bag empties -- and hands empty
    exactly when they have been played out, so the error is structured, not
    noise. This is the defect that voided E20's first reading (§8cd).
    """
    flat = z[f"bag_{name}_flat"].astype(np.int64)
    off = z[f"bag_{name}_off"].astype(np.int64)
    lens = np.diff(off)
    out = np.empty((n, emb.shape[1]), dtype=np.float32)
    if len(flat):
        # ⚠ `reduceat` RAISES on a start index equal to len(flat), which is what
        # an empty bag on the LAST row produces -- so the rows have to be
        # clamped before the call, not zeroed after it. Every clamped row has
        # lens == 0 and is overwritten with `emb[0]` below, so the clamp cannot
        # leak a wrong value; it only keeps the call legal.
        starts = np.minimum(off[:-1], len(flat) - 1)
        sums = np.add.reduceat(emb[flat], starts, axis=0)
        sums[lens == 0] = 0.0
    else:
        sums = np.zeros((n, emb.shape[1]), dtype=np.float32)
    nz = lens > 0
    out[nz] = sums[nz] / lens[nz, None]
    out[~nz] = emb[0]
    return out


def v_batch(net, z, n: int) -> np.ndarray:
    """P(acting seat wins) for every row, batched."""
    parts = [z["dense"].astype(np.float32),
             net.slot_emb[z["slots"].astype(np.int64)].reshape(n, -1)]
    for name in BAGS:
        parts.append(bag_means(z, name, n, net.bag_emb))
    x = np.concatenate(parts, axis=1)
    h = np.maximum(x @ net.w1.T + net.b1, 0.0)
    h = np.maximum(h @ net.w2.T + net.b2, 0.0)
    logit = (h @ net.w3.T + net.b3).reshape(-1)
    return (1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))).astype(np.float32)


def reconcile(net, z, n: int, v: np.ndarray, k: int, rng) -> float:
    """Max |batched - the path that plays| over k random rows."""
    rows = rng.choice(n, size=min(k, n), replace=False)
    worst = 0.0
    for r in rows:
        bags = {"slots": z["slots"][r].astype(np.int64)}
        for name in BAGS:
            off = z[f"bag_{name}_off"].astype(np.int64)
            bags[name] = z[f"bag_{name}_flat"][off[r]:off[r + 1]].astype(np.int64)
        ref = net.forward(z["dense"][r].astype(np.float32), bags)
        worst = max(worst, abs(float(ref) - float(v[r])))
    return worst


def td_advantage(v: np.ndarray, gid: np.ndarray, seat: np.ndarray,
                 won: np.ndarray) -> np.ndarray:
    """A_t, seat-corrected, terminal row bootstrapped from the realised result.

    Rows are in play order within a game, which is how `p26_selfplay_gen`
    writes them.
    """
    n = len(v)
    adv = np.empty(n, dtype=np.float32)
    last_of_game = np.empty(n, dtype=bool)
    last_of_game[:-1] = gid[1:] != gid[:-1]
    last_of_game[-1] = True

    nxt = np.empty(n, dtype=np.float32)
    nxt[:-1] = v[1:]
    nxt[-1] = 0.0
    # Whose value `nxt` is expressed from. Flip when the mover changed.
    flipped = np.zeros(n, dtype=bool)
    flipped[:-1] = seat[1:] != seat[:-1]
    nxt = np.where(flipped, 1.0 - nxt, nxt)

    adv = (nxt - v).astype(np.float32)
    adv[last_of_game] = (won[last_of_game].astype(np.float32)
                         - v[last_of_game])
    return adv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--check-rows", type=int, default=1500,
                    help="rows reconciled against valuenet.Net.forward")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    net = valuenet.load(args.value)
    if net is None:
        # Same refusal as policynet's guard: a value net that fails to load
        # would otherwise write an advantage column of pure noise, and the
        # training run downstream would look perfectly ordinary.
        raise SystemExit(f"value net {args.value!r} failed valuenet.load's "
                         f"dimension guard -- refusing to write advantages")

    paths = sorted(p for d in args.data
                   for p in glob.glob(f"{d}/**/shard_*.npz", recursive=True))
    if not paths:
        raise SystemExit(f"no shards under {args.data}")
    rng = np.random.default_rng(0)
    print(f"value net {args.value}\n{len(paths)} shards")

    tot, worst_all = 0, 0.0
    stats = []
    for p in paths:
        z = dict(np.load(p, allow_pickle=True))
        n = len(z["gid"])
        if "seat" not in z:
            raise SystemExit(
                f"{p} has no `seat` column -- it was not written by "
                f"p26_selfplay_gen, and without seats the TD residual is "
                f"silently negated across every turn boundary")
        v = v_batch(net, z, n)
        worst = reconcile(net, z, n, v, args.check_rows, rng)
        worst_all = max(worst_all, worst)
        if worst > args.tol:
            raise SystemExit(
                f"{p}: batched V differs from valuenet.Net.forward by "
                f"{worst:.2e} > {args.tol:.0e}. The column is NOT written. "
                f"This is the E20 defect's shape (§8cd) and it must be "
                f"reconciled, not tolerated.")
        adv = td_advantage(v, z["gid"], z["seat"], z["won"])
        stats.append((float(v.mean()), float(adv.mean()), float(adv.std()),
                      float(np.abs(adv).mean())))
        tot += n
        if not args.dry_run:
            z["adv"] = adv
            np.savez_compressed(p, **z)
        print(f"  {Path(p).parent.name}/{Path(p).name}: {n:,} rows  "
              f"V={v.mean():.3f}  adv mean={adv.mean():+.5f} "
              f"sd={adv.std():.4f}  |adv|={np.abs(adv).mean():.4f}  "
              f"recon={worst:.2e}")

    vm = float(np.mean([s[0] for s in stats]))
    am = float(np.mean([s[1] for s in stats]))
    print(f"\n{tot:,} rows, reconciliation max |diff| {worst_all:.2e} "
          f"(tol {args.tol:.0e})")
    print(f"mean V {vm:.4f}; mean advantage {am:+.6f}")
    # A telescoping sum over a game must land on the result, so the MEAN
    # advantage is near zero by construction. A mean far from zero means the
    # seat correction or the row order is wrong -- it is the cheapest available
    # check on the part of this script most likely to be silently backwards.
    if abs(am) > 0.02:
        print(f"⚠ mean advantage {am:+.4f} is far from 0. A telescoping TD sum "
              f"should nearly vanish; suspect the seat correction or row order.")
    if args.dry_run:
        print("(dry run -- no column written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
