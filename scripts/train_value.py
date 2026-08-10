#!/usr/bin/env python
"""Train V(s) -> P(win) on SELF-PLAY OUTCOMES. E20, pre-registered.

⚡ **`agents/sa/valuenet.py` has cited this script since day 1 and it has never
existed.** The inference side is already written, already dimension-guarded, and
already takes `me` explicitly -- so this trainer's only job is to produce an npz
in exactly that layout: `slot_emb`, `bag_emb`, `w1/b1`, `w2/b2`, `w3/b3`, with
torch's (out, in) weight orientation, because `valuenet.Net` computes `w @ x`.

**The input is the pure STATE**: `features.featurize` -> dense(242) + slot ids +
three card bags. Deliberately NOT `seld`/`xdense`, which are select-conditional
-- at play time V scores a SUCCESSOR observation returned by `fs.step`, so its
input must be a function of state alone or training and inference diverge and
produce a plausible number (rule 18).

**The label is `won`, from the acting seat's point of view** (p26 writes it that
way; draws are 0.5). This is the one column the BC corpus has always carried and
that nothing has ever trained on.

⚠ §8az's warning is why early stopping is not optional: E1's outcome head on the
HUMAN corpus overfit after its first epoch. Split is by `gid` -- a row-wise
split leaks a game across both sides (rule 17's shape).

    python -X utf8 scripts/train_value.py --data artifacts/rl_v5_t05 \\
        --out out/value_v1.npz --epochs 30
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "agents"):
    sys.path.insert(0, str(ROOT / _sub))

# `sa.features` -> `sa.cards` -> `cg.sim`, so the SDK has to be on the path
# before the first feature import. Same bootstrap as `train_policy.py`.
from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa.features import DENSE_DIM, N_CARD_IDS  # noqa: E402

EMB = 16
BAGS = ("my_hand", "my_discard", "opp_discard")


class ValueNet(nn.Module):
    """Mirrors `valuenet.Net` exactly. Any change here is a change there."""

    def __init__(self, h1: int = 256, h2: int = 128, dropout: float = 0.1):
        super().__init__()
        self.slot_emb = nn.Embedding(N_CARD_IDS, EMB)
        self.bag_emb = nn.EmbeddingBag(N_CARD_IDS, EMB, mode="mean",
                                       include_last_offset=True)
        in_dim = DENSE_DIM + 12 * EMB + len(BAGS) * EMB
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, dense, slots, bag_flat, bag_off):
        parts = [dense, self.slot_emb(slots).flatten(1)]
        for name in BAGS:
            parts.append(self.bag_emb(bag_flat[name], bag_off[name]))
        x = torch.cat(parts, dim=1)
        h = self.drop(torch.relu(self.fc1(x)))
        h = self.drop(torch.relu(self.fc2(h)))
        return self.fc3(h).squeeze(1)


class Data:
    def __init__(self, paths: list[Path]):
        d, s, w, g = [], [], [], []
        bags: dict[str, list] = {b: [] for b in BAGS}
        offs: dict[str, list] = {b: [] for b in BAGS}
        for p in paths:
            z = np.load(p)
            d.append(z["dense"])
            s.append(z["slots"])
            w.append(z["won"])
            g.append(z["gid"])
            for b in BAGS:
                bags[b].append(z[f"bag_{b}_flat"])
                offs[b].append(z[f"bag_{b}_off"])
        self.dense = np.concatenate(d)
        self.slots = np.concatenate(s)
        self.won = np.concatenate(w).astype(np.float32)
        self.gid = np.concatenate(g)
        # Offsets are per-shard and must be rebased before concatenation, or
        # every shard after the first indexes into the wrong card bag -- a
        # silent corruption that trains fine and evaluates to noise.
        self.bag_flat, self.bag_off = {}, {}
        for b in BAGS:
            flats, off_out, base, row = [], [], 0, 0
            for fl, of in zip(bags[b], offs[b]):
                flats.append(fl)
                off_out.append(of[:-1] + base if row else of[:-1])
                base += len(fl)
                row += 1
            self.bag_flat[b] = np.concatenate(flats).astype(np.int64)
            self.bag_off[b] = np.concatenate(off_out + [np.array([base])]
                                             ).astype(np.int64)
        n = len(self.won)
        for b in BAGS:
            assert len(self.bag_off[b]) == n + 1, (b, len(self.bag_off[b]), n)

    def __len__(self) -> int:
        return len(self.won)

    def batch(self, idx: np.ndarray, dev) -> tuple:
        dense = torch.from_numpy(self.dense[idx]).to(dev)
        slots = torch.from_numpy(self.slots[idx].astype(np.int64)).to(dev)
        bf, bo = {}, {}
        for b in BAGS:
            off, flat = self.bag_off[b], self.bag_flat[b]
            segs = [flat[off[i]:off[i + 1]] for i in idx]
            lens = np.array([len(s) for s in segs])
            # EmbeddingBag with mode="mean" divides by zero on an empty bag and
            # returns NaN; pad an empty bag with row 0, matching valuenet.Net's
            # explicit zeros() branch closely enough for a mean over one row.
            segs = [s if len(s) else np.zeros(1, dtype=np.int64) for s in segs]
            lens = np.maximum(lens, 1)
            bf[b] = torch.from_numpy(np.concatenate(segs).astype(np.int64)).to(dev)
            bo[b] = torch.from_numpy(
                np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)).to(dev)
        y = torch.from_numpy(self.won[idx]).to(dev)
        return dense, slots, bf, bo, y


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank AUC over decided rows only (draws carry no order)."""
    m = y != 0.5
    y, p = y[m], p[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = float((y == 1).sum()), float((y == 0).sum())
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(Path(p) for d in args.data
                   for p in glob.glob(f"{d}/**/shard_*.npz", recursive=True))
    if not paths:
        sys.exit(f"no shards under {args.data}")
    print(f"loading {len(paths)} shards ...", flush=True)
    data = Data(paths)

    # 🔴 SPLIT BY GAME. Rows from one game are near-duplicates of each other and
    # share a label exactly; a row-wise split puts both sides of the same game
    # in train and val and reports a val number that means nothing.
    gids = np.unique(data.gid)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(gids)
    n_val = max(1, int(len(gids) * args.val_frac))
    val_g = set(gids[:n_val].tolist())
    is_val = np.fromiter((g in val_g for g in data.gid), dtype=bool,
                         count=len(data))
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"rows {len(data):,}  games {len(gids):,}  "
          f"train {len(tr_idx):,} / val {len(va_idx):,}  "
          f"base rate {data.won.mean():.4f}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = ValueNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()

    best, best_state, bad = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            dense, slots, bf, bo, y = data.batch(idx, dev)
            opt.zero_grad()
            loss = lossf(model(dense, slots, bf, bo), y)
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)

        model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for i in range(0, len(va_idx), args.bs):
                idx = va_idx[i:i + args.bs]
                dense, slots, bf, bo, y = data.batch(idx, dev)
                ps.append(torch.sigmoid(model(dense, slots, bf, bo)).cpu().numpy())
                ys.append(y.cpu().numpy())
        p, y = np.concatenate(ps), np.concatenate(ys)
        vl = float(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean())
        print(f"ep {ep:2d}  train {tot/len(perm):.4f}  val {vl:.4f}  "
              f"AUC {auc(y, p):.4f}  {time.time()-t0:.0f}s", flush=True)

        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {ep} (best val {best:.4f})", flush=True)
                break

    # Export rule pinned in advance (E20 / rule 18's corollary): BEST val
    # logloss on the gid-disjoint split, patience 3. Recorded, not chosen after.
    model.load_state_dict(best_state)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for i in range(0, len(va_idx), args.bs):
            idx = va_idx[i:i + args.bs]
            dense, slots, bf, bo, y = data.batch(idx, dev)
            ps.append(torch.sigmoid(model(dense, slots, bf, bo)).cpu().numpy())
            ys.append(y.cpu().numpy())
    p, y = np.concatenate(ps), np.concatenate(ys)

    # ⚠ THE ORIENTATION CONTROL. A sign-flipped V is the single failure that
    # would produce a confident, plausible, exactly-wrong agent: it would play
    # to LOSE and the arena would report ~0.0 rather than an error.
    mw, ml = float(p[y == 1].mean()), float(p[y == 0].mean())
    print(f"\nORIENTATION  mean V | won = {mw:.4f}   mean V | lost = {ml:.4f}   "
          f"AUC {auc(y, p):.4f}")
    if not mw > ml:
        sys.exit("ORIENTATION FAILED: V does not score won states above lost")

    sd = model.state_dict()
    np.savez(
        args.out,
        slot_emb=sd["slot_emb.weight"].numpy(),
        bag_emb=sd["bag_emb.weight"].numpy(),
        w1=sd["fc1.weight"].numpy(), b1=sd["fc1.bias"].numpy(),
        w2=sd["fc2.weight"].numpy(), b2=sd["fc2.bias"].numpy(),
        w3=sd["fc3.weight"].numpy()[0], b3=sd["fc3.bias"].numpy()[0],
    )
    print(f"wrote {args.out}  (val logloss {best:.4f}, AUC {auc(y, p):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
