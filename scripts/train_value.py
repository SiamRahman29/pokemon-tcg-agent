"""Train the value net on shards from build_dataset.py.

    python scripts/train_value.py --ds artifacts/ds --epochs 3 --out agents/sa/value_net.npz

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

from sa.features import DENSE_DIM, N_CARD_IDS  # noqa: E402

EMB = 16
BAGS = ("my_hand", "my_discard", "opp_discard")


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_emb = nn.Embedding(N_CARD_IDS, EMB)
        self.bag_emb = nn.EmbeddingBag(N_CARD_IDS, EMB, mode="mean",
                                       include_last_offset=True)
        in_dim = DENSE_DIM + 12 * EMB + len(BAGS) * EMB
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, dense, slots, bag_flat, bag_off):
        parts = [dense, self.slot_emb(slots).flatten(1)]
        for name in BAGS:
            parts.append(self.bag_emb(bag_flat[name], bag_off[name]))
        return self.fc(torch.cat(parts, dim=1)).squeeze(1)


class Shard:
    def __init__(self, path: Path):
        z = np.load(path)
        self.dense = z["dense"]
        self.slots = z["slots"]
        self.y = z["y"]
        self.bags = {}
        for name in BAGS:
            self.bags[name] = (z[f"bag_{name}_flat"].astype(np.int64),
                               z[f"bag_{name}_off"].astype(np.int64))
        self.n = len(self.y)

    def batches(self, bs: int, rng: np.random.Generator):
        order = rng.permutation(self.n)
        for i in range(0, self.n, bs):
            idx = order[i:i + bs]
            dense = torch.from_numpy(self.dense[idx])
            slots = torch.from_numpy(self.slots[idx].astype(np.int64))
            y = torch.from_numpy(self.y[idx])
            bag_flat, bag_off = {}, {}
            for name in BAGS:
                flat, off = self.bags[name]
                lens = off[idx + 1] - off[idx]
                new_off = np.zeros(len(idx) + 1, dtype=np.int64)
                np.cumsum(lens, out=new_off[1:])
                out = np.zeros(new_off[-1], dtype=np.int64)
                for j, k in enumerate(idx):
                    out[new_off[j]:new_off[j + 1]] = flat[off[k]:off[k + 1]]
                bag_flat[name] = torch.from_numpy(out)
                bag_off[name] = torch.from_numpy(new_off)
            yield dense, slots, bag_flat, bag_off, y


def export_npz(model: ValueNet, path: Path):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(
        path,
        slot_emb=sd["slot_emb.weight"],
        bag_emb=sd["bag_emb.weight"],
        w1=sd["fc.0.weight"], b1=sd["fc.0.bias"],
        w2=sd["fc.2.weight"], b2=sd["fc.2.bias"],
        w3=sd["fc.4.weight"], b3=sd["fc.4.bias"])
    print(f"exported -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="artifacts/ds")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="agents/sa/value_net.npz")
    ap.add_argument("--val-frac", type=float, default=0.05)
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads() - 1))
    paths = sorted((ROOT / args.ds).glob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / args.ds}")
    shards = [Shard(p) for p in paths]
    n_total = sum(s.n for s in shards)
    print(f"{len(shards)} shards, {n_total} rows")

    val_shard = shards[-1]  # holdout: last shard (different games)
    train_shards = shards[:-1] if len(shards) > 1 else shards

    model = ValueNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = seen = 0.0
        for si, shard in enumerate(train_shards):
            for dense, slots, bf, bo, y in shard.batches(args.bs, rng):
                opt.zero_grad()
                out = model(dense, slots, bf, bo)
                loss = lossf(out, y)
                loss.backward()
                opt.step()
                tot += loss.item() * len(y)
                seen += len(y)
        model.eval()
        correct = vtot = vseen = 0.0
        with torch.no_grad():
            for dense, slots, bf, bo, y in val_shard.batches(args.bs, rng):
                out = model(dense, slots, bf, bo)
                vtot += lossf(out, y).item() * len(y)
                mask = y != 0.5
                correct += (((out > 0) == (y > 0.5)) & mask).sum().item()
                vseen += mask.sum().item()
        print(f"epoch {epoch}: train_loss={tot / seen:.4f} "
              f"val_loss={vtot / max(vseen, 1):.4f} "
              f"val_acc={correct / max(vseen, 1):.4f} "
              f"({time.time() - t0:.0f}s)")

    export_npz(model, ROOT / args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
