"""Where is the clone weak, and is it weak because the features are blind?

P2c found +54 Elo in one place the net could not *represent* the answer: it had
no HP feature, so it aimed 30-damage effects at chance. That was found by
watching replays. This finds the rest of them systematically.

It scores the shipped net over the held-out demonstrator split and breaks top-1
agreement down by SelectContext, next to the accuracy a uniform random pick
would get. A context with many rows and a small edge over random is a context
the net has not learned -- and the next question for each is always "can the
feature vector even express the right answer here?" (see `sa/optfeat.py`, whose
per-option features are a card-id embedding plus eight positional scalars).

    python scripts/context_accuracy.py
    python scripts/context_accuracy.py --net out/policy_lw3.npz --ds artifacts/pds

`lift` is (top1 - random) and `errors` is rows x (1 - top1): the ranking that
matters is by `errors`, since that is decisions lost per unit of play.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents"):
    sys.path.insert(0, str(ROOT / sub))

from ptcg.env import sdk  # noqa: E402

sdk.load()

from cg.api import SelectContext  # noqa: E402
from sa.optfeat import pool_scalars  # noqa: E402
from sa.features import N_EXTRA  # noqa: E402

CTX_NAME = {int(getattr(SelectContext, n)): n
            for n in dir(SelectContext) if n.isupper()}
BAGS = ("my_hand", "my_discard", "opp_discard")


class Net:
    """The inference forward pass of sa/policynet.py, fed from shard arrays."""

    def __init__(self, path: Path):
        z = np.load(path)
        self.slot_emb, self.bag_emb = z["slot_emb"], z["bag_emb"]
        self.card_emb, self.atk_emb = z["card_emb"], z["atk_emb"]
        self.state = [(z[f"sfc{i}_w"], z[f"sfc{i}_b"])
                      for i in range(int(z["n_sfc"][0]))]
        self.state_in = self.state[0][0].shape[1]
        self.head = [(z[f"head{i}_w"], z[f"head{i}_b"])
                     for i in range(int(z["n_head"][0]))]
        self.n_pool = int(z["n_pool"][0]) if "n_pool" in z else 0
        self.x_mask = z["x_mask"] if "x_mask" in z else None

    @property
    def opt_in(self) -> int:
        """Per-option dense width this net was trained at (see policynet)."""
        return (self.head[0][0].shape[1] - self.state[-1][0].shape[0]
                - 2 * self.card_emb.shape[1] - self.atk_emb.shape[1])

    def state_repr(self, dense, slots, bag_means, seld, xdense=None,
                   xslots=None, pool=None):
        parts = [dense, self.slot_emb[slots].reshape(len(slots), -1),
                 *bag_means, seld]
        # The v4 block is APPENDED (features.py), and the v5 pool after it, so
        # slicing to this net's own input width feeds a v3 net byte-identical
        # input. Same trick as the agent's policynet.scores -- keep in sync.
        if xdense is not None:
            parts += [xdense, self.slot_emb[xslots].reshape(len(xslots), -1)]
        if pool is not None:
            parts.append(pool)
        x = np.concatenate(parts, axis=1)[:, :self.state_in]
        for w, b in self.state:
            x = np.maximum(x @ w.T + b, 0.0)
        return x

    def option_logits(self, srepr_per_opt, opt_dense, card, atk, tgt):
        h = np.concatenate([srepr_per_opt, opt_dense, self.card_emb[card],
                            self.atk_emb[atk], self.card_emb[tgt]], axis=1)
        for j, (w, b) in enumerate(self.head):
            h = h @ w.T + b
            if j < len(self.head) - 1:
                h = np.maximum(h, 0.0)
        return h.reshape(-1)


def bag_means(z, name: str, n: int, width: int, emb) -> np.ndarray:
    flat = z[f"bag_{name}_flat"].astype(np.int64)
    off = z[f"bag_{name}_off"]
    out = np.zeros((n, width), dtype=np.float32)
    for i in range(n):
        a, b = off[i], off[i + 1]
        if b > a:
            out[i] = emb[flat[a:b]].mean(axis=0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="agents/sa/policy_net.npz")
    ap.add_argument("--ds", default="artifacts/pds_v2")
    ap.add_argument("--min-rows", type=int, default=200,
                    help="hide contexts rarer than this")
    ap.add_argument("--all-rows", action="store_true",
                    help="score EVERY row, not just the trainer's gid%%20 val "
                         "split. Correct -- and required -- for a corpus the "
                         "net never trained on (e.g. artifacts/pds_expert).")
    ap.add_argument("--equiv", action="store_true",
                    help="count a hit when the argmax option is BITWISE "
                         "IDENTICAL to the chosen one (same dense, card, "
                         "attack and target). Those options are the same card "
                         "in the same role -- two copies of one Trainer in the "
                         "deck, two identical energies in hand -- so no net "
                         "reading these inputs can tell them apart and picking "
                         "either produces the same game. Plain top-1 charges "
                         "the net for that coin flip: it is 7.8% of rows "
                         "corpus-wide and 32.4% of TO_HAND (§8x).")
    args = ap.parse_args()

    net = Net(ROOT / args.net)
    paths = sorted((ROOT / args.ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / args.ds}")

    hit: dict[int, int] = {}
    tries: dict[int, int] = {}
    rand: dict[int, float] = {}
    for path in paths:
        z = np.load(path)
        gid, off = z["gid"], z["opt_off"]
        n = len(gid)
        val = (np.arange(len(gid)) if args.all_rows
               else np.flatnonzero((gid % 20) == 0))   # the trainer's split
        if not len(val):
            continue
        width = net.bag_emb.shape[1]
        means = [bag_means(z, nm, n, width, net.bag_emb) for nm in BAGS]
        xd = z["xdense"][val] if "xdense" in z else None
        xs = (z["xslots"][val].astype(np.int64) if "xslots" in z else None)
        if net.x_mask is not None and xd is not None:   # an ablation arm
            xd = xd * net.x_mask[:N_EXTRA]
            xs = np.where(net.x_mask[N_EXTRA:] > 0, xs, 0)
        ctx = np.rint(z["seld"][:, 13] * 50.0).astype(int)
        opt_dense, chosen = z["opt_dense"], z["opt_chosen"]
        card = z["opt_card"].astype(np.int64)
        atk = z["opt_attack"].astype(np.int64)
        tgt = (z["opt_target"] if "opt_target" in z
               else np.zeros_like(card)).astype(np.int64)

        # The v5 pool is a summary of the row's own option set, so it has to be
        # built before the state (policynet.scores does the same reordering).
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
            if ch.sum() != 1:      # top-1 is only defined for single-choice
                continue
            k_opts = b - a
            logits = net.option_logits(
                np.repeat(srepr[k][None, :], k_opts, axis=0),
                opt_dense[a:b], card[a:b], atk[a:b], tgt[a:b])
            c = int(ctx[row])
            am = int(np.argmax(logits))
            ok = ch[am] == 1
            if keys is not None and not ok:
                ok = keys[a + am] == keys[a + int(np.argmax(ch))]
            tries[c] = tries.get(c, 0) + 1
            hit[c] = hit.get(c, 0) + int(ok)
            rand[c] = rand.get(c, 0.0) + 1.0 / k_opts

    print(f"\nnet {args.net} on the held-out split of {args.ds}\n")
    print(f"{'context':<28}{'rows':>8}{'top1':>8}{'random':>8}"
          f"{'lift':>8}{'errors':>9}")
    rows = sorted(tries, key=lambda c: -(tries[c] - hit[c]))
    total_err = 0
    for c in rows:
        n_c, h_c = tries[c], hit[c]
        err = n_c - h_c
        total_err += err
        if n_c < args.min_rows:
            continue
        top1, rnd = h_c / n_c, rand[c] / n_c
        print(f"{CTX_NAME.get(c, str(c)):<28}{n_c:>8}{top1:>8.1%}"
              f"{rnd:>8.1%}{top1 - rnd:>+8.1%}{err:>9}")
    print(f"\n{sum(tries.values())} single-choice rows, {total_err} misses "
          f"({1 - sum(hit.values()) / sum(tries.values()):.1%})")
    print("rank by `errors`; a small `lift` on many rows means the net has not "
          "learned that context -- check whether optfeat can express it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
