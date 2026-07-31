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
        self.head = [(z[f"head{i}_w"], z[f"head{i}_b"])
                     for i in range(int(z["n_head"][0]))]

    def state_repr(self, dense, slots, bag_means, seld):
        x = np.concatenate(
            [dense, self.slot_emb[slots].reshape(len(slots), -1),
             *bag_means, seld], axis=1)
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
        srepr = net.state_repr(z["dense"][val], z["slots"][val].astype(np.int64),
                               [m[val] for m in means], z["seld"][val])
        ctx = np.rint(z["seld"][:, 13] * 50.0).astype(int)
        opt_dense, chosen = z["opt_dense"], z["opt_chosen"]
        card = z["opt_card"].astype(np.int64)
        atk = z["opt_attack"].astype(np.int64)
        tgt = (z["opt_target"] if "opt_target" in z
               else np.zeros_like(card)).astype(np.int64)

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
            tries[c] = tries.get(c, 0) + 1
            hit[c] = hit.get(c, 0) + int(ch[int(np.argmax(logits))] == 1)
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
