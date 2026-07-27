"""Train the value net on shards from build_dataset.py.

    python scripts/train_value.py --ds artifacts/ds --epochs 4 --out agents/sa/value_net.npz

Game-level validation split (episode id hash), dropout + weight decay.
Architecture (mirrored by sa/valuenet.py numpy inference):
    slot_emb:  Embedding(N_CARD_IDS, 16) over 12 board slots -> 192
    bag_emb:   EmbeddingBag(N_CARD_IDS, 16, mode=mean) for my_hand,
               my_discard, opp_discard -> 48
    mlp:       [dense + 192 + 48] -> 512 -> relu -> 256 -> relu -> 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents"):
    sys.path.insert(0, str(ROOT / sub))

from ptcg.env import sdk  # noqa: E402

sdk.load()  # sa.features needs the cg engine importable

from sa.features import DENSE_DIM, N_CARD_IDS  # noqa: E402

EMB = 16
BAGS = ("my_hand", "my_discard", "opp_discard")


class ValueNet(nn.Module):
    def __init__(self, dropout: float = 0.15):
        super().__init__()
        self.slot_emb = nn.Embedding(N_CARD_IDS, EMB)
        self.bag_emb = nn.EmbeddingBag(N_CARD_IDS, EMB, mode="mean",
                                       include_last_offset=True)
        in_dim = DENSE_DIM + 12 * EMB + len(BAGS) * EMB
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1))

    def forward(self, dense, slots, bag_flat, bag_off):
        parts = [dense, self.slot_emb(slots).flatten(1)]
        for name in BAGS:
            parts.append(self.bag_emb(bag_flat[name], bag_off[name]))
        return self.fc(torch.cat(parts, dim=1)).squeeze(1)


class Data:
    """All shards concatenated, with per-row bag slices."""

    def __init__(self, paths: list[Path]):
        dense, slots, y, gid = [], [], [], []
        bag_rows: dict[str, list] = {n: [] for n in BAGS}
        for p in paths:
            z = np.load(p)
            dense.append(z["dense"])
            slots.append(z["slots"])
            y.append(z["y"])
            gid.append(z["gid"] if "gid" in z
                       else np.zeros(len(z["y"]), dtype=np.int64))
            for n in BAGS:
                flat = z[f"bag_{n}_flat"]
                off = z[f"bag_{n}_off"]
                bag_rows[n].extend(
                    flat[off[i]:off[i + 1]].astype(np.int64)
                    for i in range(len(off) - 1))
        self.dense = np.concatenate(dense)
        self.slots = np.concatenate(slots).astype(np.int64)
        self.y = np.concatenate(y)
        self.gid = np.concatenate(gid)
        self.bags = bag_rows
        self.n = len(self.y)

    def batches(self, idx: np.ndarray, bs: int,
                rng: np.random.Generator | None):
        order = rng.permutation(idx) if rng is not None else idx
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            bag_flat, bag_off = {}, {}
            for n in BAGS:
                rows = [self.bags[n][k] for k in sel]
                off = np.zeros(len(rows) + 1, dtype=np.int64)
                np.cumsum([len(r) for r in rows], out=off[1:])
                bag_flat[n] = torch.from_numpy(
                    np.concatenate(rows) if off[-1]
                    else np.zeros(0, dtype=np.int64))
                bag_off[n] = torch.from_numpy(off)
            yield (torch.from_numpy(self.dense[sel]),
                   torch.from_numpy(self.slots[sel]),
                   bag_flat, bag_off,
                   torch.from_numpy(self.y[sel]),
                   sel)


def export_npz(model: ValueNet, path: Path):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(
        path,
        slot_emb=sd["slot_emb.weight"],
        bag_emb=sd["bag_emb.weight"],
        w1=sd["fc.0.weight"], b1=sd["fc.0.bias"],
        w2=sd["fc.3.weight"], b2=sd["fc.3.bias"],
        w3=sd["fc.6.weight"], b3=sd["fc.6.bias"])
    print(f"exported -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="artifacts/ds")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--out", default="agents/sa/value_net.npz")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads() - 1))
    paths = sorted((ROOT / args.ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / args.ds}")
    data = Data(paths)
    val_mask = (data.gid % 20) == 0  # ~5% of games held out
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    print(f"{len(paths)} shards, {data.n} rows "
          f"({len(train_idx)} train / {len(val_idx)} val, "
          f"{len(np.unique(data.gid))} games)")

    model = ValueNet(args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    best_val = 1e9

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = seen = 0.0
        for dense, slots, bf, bo, y, _ in data.batches(train_idx, args.bs,
                                                       rng):
            opt.zero_grad()
            loss = lossf(model(dense, slots, bf, bo), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            seen += len(y)
        model.eval()
        vtot = 0.0
        correct = np.zeros(3)
        counts = np.zeros(3)
        with torch.no_grad():
            for dense, slots, bf, bo, y, sel in data.batches(val_idx, args.bs,
                                                             None):
                out = model(dense, slots, bf, bo)
                vtot += lossf(out, y).item() * len(y)
                turn = data.dense[sel][:, 0] * 40.0
                pred = (out.numpy() > 0)
                truth = (y.numpy() > 0.5)
                decided = y.numpy() != 0.5
                for b, (lo, hi) in enumerate(((0, 6), (6, 14), (14, 99))):
                    m = decided & (turn >= lo) & (turn < hi)
                    correct[b] += (pred[m] == truth[m]).sum()
                    counts[b] += m.sum()
        val_loss = vtot / max(len(val_idx), 1)
        acc = correct / np.maximum(counts, 1)
        print(f"epoch {epoch}: train={tot / seen:.4f} val={val_loss:.4f} "
              f"acc(early/mid/late)={acc[0]:.3f}/{acc[1]:.3f}/{acc[2]:.3f} "
              f"({time.time() - t0:.0f}s)")
        if val_loss < best_val:
            best_val = val_loss
            export_npz(model, ROOT / args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
