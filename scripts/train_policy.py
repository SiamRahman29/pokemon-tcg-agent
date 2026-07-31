"""Train the policy net (behavioral cloning of top players' selects).

    python scripts/train_policy.py --ds artifacts/pds --out agents/sa/policy_net.npz

The state is encoded once per row, then scored against each option.

    state:  dense + slot_emb(12x16) + 3 bag means(16) + seld(14)
            -> MLP(--state-h) -> state_repr
    option: opt_dense + card_emb(16) + atk_emb(16) + tgt_emb(16)
    score:  MLP([state_repr, option], --head-h) -> 1

`--loss listwise` optimizes softmax cross-entropy over each select's option
set, which is what the agent actually does at inference (rank the options and
take the top k). `--loss bce` is the original pointwise objective; it treats
every option independently and does not model "which of these is best".

Layer sizes are exported generically (`sfc{i}_w` / `head{i}_w` + counts), so
sa/policynet.py mirrors any depth without a code change.
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
from sa.optfeat import OPT_DENSE, OPT_DENSE_V2, N_ATTACK_IDS  # noqa: E402

EMB = 16
SEL_DENSE = 14
BAGS = ("my_hand", "my_discard", "opp_discard")
# The band the LB's top ~40 teams sit in; `val_top1@1120+` says how well the
# net fits STRONG demonstrators as opposed to the mixture (ROADMAP B7).
VAL_HI_RATING = 1120.0


def load_init(model: PolicyNet, path: Path) -> None:
    """Warm-start from an exported .npz (the fine-tuning arm of B7). Refuses on
    any shape mismatch rather than partially loading -- a silently half-loaded
    net trains fine and measures like a fresh one."""
    z = np.load(path)
    with torch.no_grad():
        for name, emb in (("slot_emb", model.slot_emb), ("bag_emb",
                          model.bag_emb), ("card_emb", model.card_emb),
                         ("atk_emb", model.atk_emb)):
            w = z[name]
            if w.shape != tuple(emb.weight.shape):
                raise SystemExit(f"--init {path.name}: {name} is {w.shape}, "
                                 f"model wants {tuple(emb.weight.shape)}")
            emb.weight.copy_(torch.from_numpy(w))
        for prefix, seq in (("sfc", model.state_fc), ("head", model.head)):
            lins = [m for m in seq if isinstance(m, nn.Linear)]
            n = int(z[f"n_{prefix}"][0])
            if n != len(lins):
                raise SystemExit(f"--init {path.name}: {n} {prefix} layers, "
                                 f"model has {len(lins)}")
            for i, lin in enumerate(lins):
                w, b = z[f"{prefix}{i}_w"], z[f"{prefix}{i}_b"]
                if w.shape != tuple(lin.weight.shape):
                    raise SystemExit(
                        f"--init {path.name}: {prefix}{i}_w is {w.shape}, "
                        f"model wants {tuple(lin.weight.shape)}")
                lin.weight.copy_(torch.from_numpy(w))
                lin.bias.copy_(torch.from_numpy(b))
    print(f"warm-started from {path}")


def _mlp(sizes: list[int], dropout: float, out_dim: int | None) -> nn.Sequential:
    """ReLU MLP over `sizes` hidden widths; `out_dim` appends a linear head."""
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
    if out_dim is not None:
        layers.append(nn.Linear(sizes[-1], out_dim))
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    def __init__(self, state_h: tuple[int, ...] = (256,),
                 head_h: tuple[int, ...] = (128,), dropout: float = 0.1,
                 opt_cols: int = OPT_DENSE):
        super().__init__()
        self.opt_cols = opt_cols
        self.slot_emb = nn.Embedding(N_CARD_IDS, EMB)
        self.bag_emb = nn.EmbeddingBag(N_CARD_IDS, EMB, mode="mean",
                                       include_last_offset=True)
        self.card_emb = nn.Embedding(N_CARD_IDS, EMB)
        self.atk_emb = nn.Embedding(N_ATTACK_IDS, EMB)
        in_state = DENSE_DIM + 12 * EMB + len(BAGS) * EMB + SEL_DENSE
        self.state_fc = _mlp([in_state, *state_h], dropout, None)
        in_head = state_h[-1] + opt_cols + 3 * EMB
        self.head = _mlp([in_head, *head_h], dropout, 1)

    def forward(self, dense, slots, bag_flat, bag_off, seld,
                opt_dense, opt_card, opt_atk, opt_tgt, opt_row):
        parts = [dense, self.slot_emb(slots).flatten(1)]
        for name in BAGS:
            parts.append(self.bag_emb(bag_flat[name], bag_off[name]))
        parts.append(seld)
        srepr = self.state_fc(torch.cat(parts, dim=1))       # (B, H)
        # Slice to `opt_cols`. The v3 target block is APPENDED to the v2 layout,
        # so `--opt-cols 25` trains the exact v2-feature control on the identical
        # rows -- same games, same selects, same labels, only the features differ.
        # That is a cleaner control than comparing against the shipped net, which
        # also differs in corpus (2,810 games vs whatever is on disk now).
        per_opt = torch.cat([srepr[opt_row], opt_dense[:, :self.opt_cols],
                             self.card_emb(opt_card),
                             self.atk_emb(opt_atk),
                             self.card_emb(opt_tgt)], dim=1)  # (O, ...)
        return self.head(per_opt).squeeze(1)                 # (O,)


def listwise_loss(out: torch.Tensor, chosen: torch.Tensor,
                  opt_row: torch.Tensor, n_rows: int,
                  w: torch.Tensor | None = None) -> torch.Tensor:
    """Softmax cross-entropy within each select's option set, averaged over
    the chosen options of that select. This is the objective that matches
    inference: the agent ranks the options and takes the top k.

    `w` is an optional per-ROW weight (ROADMAP B7): the loss becomes a weighted
    mean, so a strong demonstrator's selects pull the mode further than a weak
    one's. Weights are normalised to mean 1 by the caller, which keeps the
    effective step size comparable to the unweighted control."""
    # log-softmax per row, computed with a segmented max for stability
    big = torch.full((n_rows,), -1e30, device=out.device)
    mx = big.scatter_reduce(0, opt_row, out, reduce="amax", include_self=True)
    ex = torch.exp(out - mx[opt_row])
    denom = torch.zeros(n_rows, device=out.device).index_add_(0, opt_row, ex)
    logp = out - mx[opt_row] - torch.log(denom + 1e-12)[opt_row]
    picked = torch.zeros(n_rows, device=out.device).index_add_(
        0, opt_row, logp * chosen)
    cnt = torch.zeros(n_rows, device=out.device).index_add_(0, opt_row, chosen)
    valid = cnt > 0
    per_row = -(picked[valid] / cnt[valid])
    if w is None:
        return per_row.mean()
    wv = w[valid]
    return (per_row * wv).sum() / wv.sum().clamp_min(1e-8)


class Data:
    def __init__(self, paths: list[Path]):
        sd, slots, seld, gid, won, rating = [], [], [], [], [], []
        od, oc, oa, ot, om = [], [], [], [], []
        self.opt_rows: list[tuple[int, int]] = []  # (start,end) per row
        # ⚠ Bags are kept FLAT (one array + one offset array per bag), exactly
        # as the shards store them. The previous version materialised one small
        # numpy array per row per bag -- 249k rows x 3 bags = ~750k objects --
        # and that allocation, not the model, is what OOM'd this 7.3 GB machine
        # on any net above ~1.5M params. Same semantics, ~1 GB less resident.
        bag_flats: dict[str, list] = {n: [] for n in BAGS}
        bag_lens: dict[str, list] = {n: [] for n in BAGS}
        base = 0
        for p in paths:
            z = np.load(p)
            n = len(z["gid"])
            sd.append(z["dense"])
            slots.append(z["slots"])
            seld.append(z["seld"])
            gid.append(z["gid"])
            won.append(z["won"])
            # Corpora built before `--ratings` have no per-row demonstrator.
            rating.append(z["rating"] if "rating" in z
                          else np.full(n, np.nan, dtype=np.float32))
            od.append(z["opt_dense"])
            oc.append(z["opt_card"])
            oa.append(z["opt_attack"])
            ot.append(z["opt_target"] if "opt_target" in z
                      else np.zeros_like(z["opt_card"]))
            om.append(z["opt_chosen"])
            off = z["opt_off"]
            for i in range(n):
                self.opt_rows.append((base + off[i], base + off[i + 1]))
            base += off[-1]
            for nm in BAGS:
                flat = z[f"bag_{nm}_flat"]
                boff = z[f"bag_{nm}_off"]
                bag_flats[nm].append(flat.astype(np.int64, copy=False))
                bag_lens[nm].append(np.diff(boff).astype(np.int64))
        self.dense = np.concatenate(sd)
        self.slots = np.concatenate(slots).astype(np.int64)
        self.seld = np.concatenate(seld)
        self.gid = np.concatenate(gid)
        self.won = np.concatenate(won)
        self.rating = np.concatenate(rating)
        self.w = np.ones(len(self.gid), dtype=np.float32)
        self.opt_dense = np.concatenate(od)
        self.opt_card = np.concatenate(oc).astype(np.int64)
        self.opt_atk = np.concatenate(oa).astype(np.int64)
        self.opt_tgt = np.concatenate(ot).astype(np.int64)
        self.opt_chosen = np.concatenate(om)
        # Flats concatenate in row order, so a cumsum over the per-row lengths
        # gives GLOBAL offsets across shards.
        self.bag_flat: dict[str, np.ndarray] = {}
        self.bag_off: dict[str, np.ndarray] = {}
        for nm in BAGS:
            lens = np.concatenate(bag_lens[nm])
            off = np.zeros(len(lens) + 1, dtype=np.int64)
            np.cumsum(lens, out=off[1:])
            self.bag_off[nm] = off
            self.bag_flat[nm] = (np.concatenate(bag_flats[nm]) if off[-1]
                                 else np.zeros(0, dtype=np.int64))
        self.n = len(self.gid)

    def batches(self, idx: np.ndarray, bs: int,
                rng: np.random.Generator | None):
        order = rng.permutation(idx) if rng is not None else idx
        for i in range(0, len(order), bs):
            sel = order[i:i + bs]
            bag_flat, bag_off = {}, {}
            for nm in BAGS:
                go = self.bag_off[nm]
                lens = go[sel + 1] - go[sel]
                off = np.zeros(len(sel) + 1, dtype=np.int64)
                np.cumsum(lens, out=off[1:])
                if off[-1]:
                    idx = np.concatenate([np.arange(go[k], go[k + 1])
                                          for k in sel if go[k + 1] > go[k]])
                    gathered = self.bag_flat[nm][idx]
                else:
                    gathered = np.zeros(0, dtype=np.int64)
                bag_flat[nm] = torch.from_numpy(gathered)
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
                   torch.from_numpy(self.opt_tgt[opt_idx]),
                   torch.from_numpy(opt_row),
                   torch.from_numpy(self.opt_chosen[opt_idx]),
                   spans,
                   torch.from_numpy(self.w[sel]),
                   sel)


def count_fraction_table(data: "Data", idx: np.ndarray) -> np.ndarray:
    """(11, 64) mean of (chosen-min)/(max-min) per (selectType, context) over
    variable-count selects. Unseen cells default to 1.0 (take the max), which
    matches top play for searches/benching."""
    num = np.zeros((11, 64))
    den = np.zeros((11, 64))
    for k in idx:
        a, b = data.opt_rows[k]
        seld = data.seld[k]
        t = int(np.argmax(seld[:11]))
        ctx = int(round(seld[13] * 50.0))
        ctx = min(ctx, 63)
        mn = int(round(seld[11] * 5.0))
        mx = int(round(seld[12] * 5.0))
        if mx <= mn:
            continue
        chosen = float(data.opt_chosen[a:b].sum())
        num[t, ctx] += (chosen - mn) / (mx - mn)
        den[t, ctx] += 1.0
    frac = np.where(den > 0, num / np.maximum(den, 1), 1.0)
    return frac.astype(np.float32)


def export_npz(model: PolicyNet, path: Path, count_frac: np.ndarray):
    """Export every Linear generically, so inference mirrors any depth."""
    out: dict[str, np.ndarray] = {
        "slot_emb": model.slot_emb.weight.detach().numpy(),
        "bag_emb": model.bag_emb.weight.detach().numpy(),
        "card_emb": model.card_emb.weight.detach().numpy(),
        "atk_emb": model.atk_emb.weight.detach().numpy(),
        "count_frac": count_frac,
    }
    for prefix, seq in (("sfc", model.state_fc), ("head", model.head)):
        n = 0
        for mod in seq:
            if isinstance(mod, nn.Linear):
                out[f"{prefix}{n}_w"] = mod.weight.detach().numpy()
                out[f"{prefix}{n}_b"] = mod.bias.detach().numpy()
                n += 1
        out[f"n_{prefix}"] = np.array([n], dtype=np.int64)
    np.savez_compressed(path, **out)
    print(f"exported -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="artifacts/pds")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--winners-only", action="store_true")
    ap.add_argument("--loss", choices=("bce", "listwise", "both"),
                    default="listwise")
    ap.add_argument("--state-h", default="256",
                    help="comma-separated hidden widths for the state MLP")
    ap.add_argument("--head-h", default="128",
                    help="comma-separated hidden widths for the scoring MLP")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--out", default="agents/sa/policy_net.npz")
    ap.add_argument("--rating-temp", type=float, default=0.0,
                    help="ROADMAP B7: weight each row by "
                         "exp((rating - max) / T), normalised to mean 1. Small "
                         "T = clone the best demonstrators only; 0 (default) = "
                         "uniform, the standing control. Needs a corpus built "
                         "with `--ratings`.")
    ap.add_argument("--rating-min", type=float, default=0.0,
                    help="drop rows whose demonstrator is below this LB score")
    ap.add_argument("--init",
                    help="warm-start from an exported .npz (fine-tuning). "
                         "Shapes must match the arch flags.")
    ap.add_argument("--opt-cols", type=int, default=OPT_DENSE,
                    help="per-option feature columns to use. Default = all "
                         f"({OPT_DENSE}). Pass {OPT_DENSE_V2} to train the "
                         "v2-feature CONTROL on identical rows (ROADMAP B1).")
    args = ap.parse_args()
    if not 1 <= args.opt_cols <= OPT_DENSE:
        raise SystemExit(f"--opt-cols must be in 1..{OPT_DENSE}")

    torch.set_num_threads(max(1, torch.get_num_threads() - 1))
    # Seeded so that a control/treatment pair (e.g. --opt-cols 25 vs 37, ROADMAP
    # B1) differs in its FEATURES and not in dropout masks or batch order. Weight
    # init still differs where the layer widths differ, which cannot be avoided.
    torch.manual_seed(0)
    paths = sorted((ROOT / args.ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ROOT / args.ds}")
    data = Data(paths)
    keep = np.ones(data.n, dtype=bool)
    if args.winners_only:
        keep &= data.won > 0.5
    rated = ~np.isnan(data.rating)
    if args.rating_min > 0:
        keep &= rated & (data.rating >= args.rating_min)
        print(f"--rating-min {args.rating_min}: {int(keep.sum())} of {data.n} "
              f"rows kept")
    if args.rating_temp > 0:
        if not rated.any():
            raise SystemExit("--rating-temp needs a corpus built with "
                             "`--ratings`; every row's rating is NaN")
        # An unrated demonstrator gets the MEDIAN rating's weight rather than
        # 0 or 1: dropping them silently changes the corpus, and weighting them
        # 1.0 would make the unknown teams the most-cloned ones once the
        # exponential pushes everyone else down.
        r = np.where(rated, data.rating, np.nanmedian(data.rating))
        w = np.exp((r - np.nanmax(data.rating)) / args.rating_temp)
        w = (w / w[keep].mean()).astype(np.float32)
        data.w = w
        ess = w[keep].sum() ** 2 / np.square(w[keep]).sum()
        print(f"--rating-temp {args.rating_temp}: weights "
              f"[{w[keep].min():.4f}, {w[keep].max():.3f}], "
              f"effective sample size {ess:,.0f} of {int(keep.sum()):,} rows "
              f"({ess / max(int(keep.sum()), 1):.1%})")
        print(f"  {int((~rated & keep).sum())} unrated rows held at the median "
              f"rating ({np.nanmedian(data.rating):.1f})")
    val_mask = (data.gid % 20) == 0
    train_idx = np.where(keep & ~val_mask)[0]
    val_idx = np.where(keep & val_mask)[0]
    print(f"{len(paths)} shards, {data.n} rows -> {len(train_idx)} train / "
          f"{len(val_idx)} val ({len(np.unique(data.gid))} games)")

    count_frac = count_fraction_table(data, train_idx)

    state_h = tuple(int(x) for x in args.state_h.split(","))
    head_h = tuple(int(x) for x in args.head_h.split(","))
    model = PolicyNet(state_h=state_h, head_h=head_h, dropout=args.dropout,
                      opt_cols=args.opt_cols)
    if args.init:
        load_init(model, ROOT / args.init)
    print(f"arch: state{list(state_h)} head{list(head_h)} loss={args.loss} "
          f"opt_cols={args.opt_cols}/{OPT_DENSE} "
          f"params={sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    bcef = nn.BCEWithLogitsLoss()
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(0)
    best = -1.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = seen = 0.0
        for batch in data.batches(train_idx, args.bs, rng):
            (dense, slots, bf, bo, seld, odn, ocd, oat, otg, orow, om,
             spans, rw, _sel) = batch
            opt.zero_grad()
            out = model(dense, slots, bf, bo, seld, odn, ocd, oat, otg, orow)
            loss = torch.zeros((), dtype=out.dtype)
            wrow = rw if args.rating_temp > 0 else None
            if args.loss in ("bce", "both"):
                if wrow is None:
                    loss = loss + bcef(out, om)
                else:   # the row's weight applies to each of its options
                    per = bce_none(out, om) * wrow[orow]
                    loss = loss + per.sum() / wrow[orow].sum().clamp_min(1e-8)
            if args.loss in ("listwise", "both"):
                loss = loss + listwise_loss(out, om, orow, len(spans), wrow)
            loss.backward()
            opt.step()
            tot += loss.item() * len(om)
            seen += len(om)
        # val: top-1 accuracy on single-choice rows
        model.eval()
        hit = tries = 0
        hit_hi = tries_hi = 0
        with torch.no_grad():
            for batch in data.batches(val_idx, args.bs, None):
                (dense, slots, bf, bo, seld, odn, ocd, oat, otg, orow, om,
                 spans, _rw, vsel) = batch
                out = model(dense, slots, bf, bo, seld, odn, ocd, oat, otg,
                            orow).numpy()
                om = om.numpy()
                pos = 0
                for j, (a, b) in enumerate(spans):
                    k = b - a
                    sc = out[pos:pos + k]
                    ch = om[pos:pos + k]
                    pos += k
                    if ch.sum() == 1:
                        ok = ch[np.argmax(sc)] == 1
                        hit += ok
                        tries += 1
                        # Same rows, restricted to strong demonstrators. Rule 3
                        # still holds -- neither number predicts strength; this
                        # one just says WHOSE policy is being fit.
                        if data.rating[vsel[j]] >= VAL_HI_RATING:
                            hit_hi += ok
                            tries_hi += 1
        acc = hit / max(tries, 1)
        hi = hit_hi / max(tries_hi, 1)
        print(f"epoch {epoch}: train={tot / seen:.4f} val_top1={acc:.4f} "
              f"val_top1@{VAL_HI_RATING:.0f}+={hi:.4f} (n={tries_hi}) "
              f"({time.time() - t0:.0f}s)")
        if acc > best:
            best = acc
            export_npz(model, ROOT / args.out, count_frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
