"""Train the policy net (behavioral cloning of top players' selects).

    python scripts/train_policy.py --ds artifacts/pds --out agents/sa/policy_net.npz

Pointwise BCE on option chosen/not-chosen, with the state encoded once per row.
Architecture (mirrored by sa/policynet.py):
    state:  dense(218) + slot_emb(12x16) + 3 bag means(16) + seld(14) -> 320
            -> Linear 256 relu -> state_repr
    option: opt_dense(25) + card_emb(16) + atk_emb(16) -> 57
    score:  Linear([state_repr, option]) 128 relu -> 1
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

sdk.load()

from sa.features import DENSE_DIM, N_CARD_IDS  # noqa: E402
from sa.optfeat import OPT_DENSE, N_ATTACK_IDS  # noqa: E402

EMB = 16
SEL_DENSE = 14
STATE_H = 256
HEAD_H = 128
BAGS = ("my_hand", "my_discard", "opp_discard")


class PolicyNet(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.slot_emb = nn.Embedding(N_CARD_IDS, EMB)
        self.bag_emb = nn.EmbeddingBag(N_CARD_IDS, EMB, mode="mean",
                                       include_last_offset=True)
        self.card_emb = nn.Embedding(N_CARD_IDS, EMB)
        self.atk_emb = nn.Embedding(N_ATTACK_IDS, EMB)
        in_state = DENSE_DIM + 12 * EMB + len(BAGS) * EMB + SEL_DENSE
        self.state_fc = nn.Sequential(
            nn.Linear(in_state, STATE_H), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(STATE_H + OPT_DENSE + 2 * EMB, HEAD_H), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(HEAD_H, 1))

    def forward(self, dense, slots, bag_flat, bag_off, seld,
                opt_dense, opt_card, opt_atk, opt_row):
        parts = [dense, self.slot_emb(slots).flatten(1)]
        for name in BAGS:
            parts.append(self.bag_emb(bag_flat[name], bag_off[name]))
        parts.append(seld)
        srepr = self.state_fc(torch.cat(parts, dim=1))       # (B, H)
        per_opt = torch.cat([srepr[opt_row], opt_dense,
                             self.card_emb(opt_card),
                             self.atk_emb(opt_atk)], dim=1)  # (O, ...)
        return self.head(per_opt).squeeze(1)                 # (O,)


class Data:
    def __init__(self, paths: list[Path]):
        sd, slots, seld, gid, won = [], [], [], [], []
        od, oc, oa, om = [], [], [], []
        self.opt_rows: list[tuple[int, int]] = []  # (start,end) per row
        bag_rows: dict[str, list] = {n: [] for n in BAGS}
        base = 0
        for p in paths:
            z = np.load(p)
            n = len(z["gid"])
            sd.append(z["dense"])
            slots.append(z["slots"])
            seld.append(z["seld"])
            gid.append(z["gid"])
            won.append(z["won"])
            od.append(z["opt_dense"])
            oc.append(z["opt_card"])
            oa.append(z["opt_attack"])
            om.append(z["opt_chosen"])
            off = z["opt_off"]
            for i in range(n):
                self.opt_rows.append((base + off[i], base + off[i + 1]))
            base += off[-1]
            for nm in BAGS:
                flat = z[f"bag_{nm}_flat"]
                boff = z[f"bag_{nm}_off"]
                bag_rows[nm].extend(flat[boff[i]:boff[i + 1]].astype(np.int64)
                                    for i in range(n))
        self.dense = np.concatenate(sd)
        self.slots = np.concatenate(slots).astype(np.int64)
        self.seld = np.concatenate(seld)
        self.gid = np.concatenate(gid)
        self.won = np.concatenate(won)
        self.opt_dense = np.concatenate(od)
        self.opt_card = np.concatenate(oc).astype(np.int64)
        self.opt_atk = np.concatenate(oa).astype(np.int64)
        self.opt_chosen = np.concatenate(om)
        self.bags = bag_rows
        self.n = len(self.gid)

    def batches(self, idx: np.ndarray, bs: int,
                rng: np.random.Generator | None):
        order = rng.permutation(idx) if rng is not None else idx
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            bag_flat, bag_off = {}, {}
            for nm in BAGS:
                rows = [self.bags[nm][k] for k in sel]
                off = np.zeros(len(rows) + 1, dtype=np.int64)
                np.cumsum([len(r) for r in rows], out=off[1:])
                bag_flat[nm] = torch.from_numpy(
                    np.concatenate(rows) if off[-1]
                    else np.zeros(0, dtype=np.int64))
                bag_off[nm] = torch.from_numpy(off)
            spans = [self.opt_rows[k] for k in sel]
            opt_idx = np.concatenate([np.arange(a, b) for a, b in spans])
            opt_row = np.concatenate(
                [np.full(b - a, j) for j, (a, b) in enumerate(spans)])
            yield (torch.from_numpy(self.dense[sel]),
                   torch.from_numpy(self.slots[sel]),
                   bag_flat, bag_off,
                   torch.from_numpy(self.seld[sel]),
                   torch.from_numpy(self.opt_dense[opt_idx]),
                   torch.from_numpy(self.opt_card[opt_idx]),
                   torch.from_numpy(self.opt_atk[opt_idx]),
                   torch.from_numpy(opt_row),
                   torch.from_numpy(self.opt_chosen[opt_idx]),
                   spans)


def export_npz(model: PolicyNet, path: Path):
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez_compressed(
        path,
        slot_emb=sd["slot_emb.weight"], bag_emb=sd["bag_emb.weight"],
        card_emb=sd["card_emb.weight"], atk_emb=sd["atk_emb.weight"],
        ws=sd["state_fc.0.weight"], bs=sd["state_fc.0.bias"],
        w1=sd["head.0.weight"], b1=sd["head.0.bias"],
        w2=sd["head.3.weight"], b2=sd["head.3.bias"])
    print(f"exported -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="artifacts/pds")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--winners-only", action="store_true")
    ap.add_argument("--out", default="agents/sa/policy_net.npz")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads() - 1))
    paths = sorted((ROOT / args.ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / args.ds}")
    data = Data(paths)
    keep = np.ones(data.n, dtype=bool)
    if args.winners_only:
        keep &= data.won > 0.5
    val_mask = (data.gid % 20) == 0
    train_idx = np.where(keep & ~val_mask)[0]
    val_idx = np.where(keep & val_mask)[0]
    print(f"{len(paths)} shards, {data.n} rows -> {len(train_idx)} train / "
          f"{len(val_idx)} val ({len(np.unique(data.gid))} games)")

    model = PolicyNet()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    best = -1.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = seen = 0.0
        for batch in data.batches(train_idx, args.bs, rng):
            (dense, slots, bf, bo, seld, odn, ocd, oat, orow, om, _) = batch
            opt.zero_grad()
            out = model(dense, slots, bf, bo, seld, odn, ocd, oat, orow)
            loss = lossf(out, om)
            loss.backward()
            opt.step()
            tot += loss.item() * len(om)
            seen += len(om)
        # val: top-1 accuracy on single-choice rows
        model.eval()
        hit = tries = 0
        with torch.no_grad():
            for batch in data.batches(val_idx, args.bs, None):
                (dense, slots, bf, bo, seld, odn, ocd, oat, orow, om,
                 spans) = batch
                out = model(dense, slots, bf, bo, seld, odn, ocd, oat,
                            orow).numpy()
                om = om.numpy()
                pos = 0
                for a, b in spans:
                    k = b - a
                    sc = out[pos:pos + k]
                    ch = om[pos:pos + k]
                    pos += k
                    if ch.sum() == 1:
                        hit += ch[np.argmax(sc)] == 1
                        tries += 1
        acc = hit / max(tries, 1)
        print(f"epoch {epoch}: train={tot / seen:.4f} val_top1={acc:.4f} "
              f"({time.time() - t0:.0f}s)")
        if acc > best:
            best = acc
            export_npz(model, ROOT / args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
