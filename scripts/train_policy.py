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
import json
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

from sa.features import (A_GROUPS, DENSE_DIM, N_ATTR,  # noqa: E402
                         N_CARD_IDS, N_EXTRA, N_XSLOT, X_GROUPS)
from sa.optfeat import (OPT_DENSE, OPT_DENSE_V2, N_ATTACK_IDS,  # noqa: E402
                        pool_width)
from sa.routing import (NAME_TO_ROUTE, ROUTE_NAMES,  # noqa: E402
                        routes_from_corpus)

EMB = 16
SEL_DENSE = 14
BAGS = ("my_hand", "my_discard", "opp_discard")
# The four embedding tables and the id space each is indexed by. Used only by
# --vocab; see build_remap.
EMB_TABLES = ("slot_emb", "bag_emb", "card_emb", "atk_emb")
PAD_IX, UNK_IX = 0, 1
# The band the LB's top ~40 teams sit in; `val_top1@1120+` says how well the
# net fits STRONG demonstrators as opposed to the mixture (ROADMAP B7).
VAL_HI_RATING = 1120.0


def build_remap(vocab_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-table id -> row map with a PAD row and a shared UNK row.

    The shipped tables are allocated over the RAW id space (1300 card ids, 1600
    attack ids) but the corpus only ever touches 104/134/135/57 of those rows.
    The other ~90% are exported at their random init -- harmless while they are
    never read, and NOT harmless at inference, where an out-of-vocabulary
    opponent card lands on a random unit-normal vector whose norm (3.91-3.95) is
    indistinguishable from a trained row's (3.97-4.07). The net cannot tell "a
    card I have never seen" from "a card I know", so it reads confident garbage.

    Remapping collapses each table to exactly the rows that got a gradient, and
    routes everything else to ONE row that is trained, by construction, to mean
    "unknown card". Row 0 is PAD (empty slot / no stadium / no effect); with
    `padding_idx=0` it is pinned at zero and takes no gradient, which is where
    the v5 net was heading on its own -- it drove |slot_emb[0]| to 2.337 against
    a 3.958 table mean, the 11th smallest of 1,300 rows, on 25.5% of lookups.

    ⚠ Per-table, not shared: a card seen in hand but never on an opponent's
    board is trained in `bag_emb` and untrained in `slot_emb`. One shared vocab
    would re-introduce the exact defect this removes, just for fewer rows.
    """
    from sa.features import N_CARD_IDS
    from sa.optfeat import N_ATTACK_IDS
    tabs = json.loads(vocab_path.read_text(encoding="utf-8"))["tables"]
    sizes = {"atk_emb": N_ATTACK_IDS}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t in EMB_TABLES:
        if t not in tabs:
            raise SystemExit(f"{vocab_path} has no census for {t}")
        ids = np.array(sorted(int(k) for k in tabs[t] if int(k) != 0),
                       dtype=np.int64)
        size = sizes.get(t, N_CARD_IDS)
        if ids.size and int(ids[-1]) >= size:
            raise SystemExit(f"{t}: census id {int(ids[-1])} >= {size}; the "
                             "vocab was built against a different id space")
        lut = np.full(size, UNK_IX, dtype=np.int64)
        lut[PAD_IX] = PAD_IX
        lut[ids] = np.arange(2, 2 + ids.size, dtype=np.int64)
        out[t] = (ids, lut)
    return out


def apply_remap(data: "Data", remap: dict[str, tuple[np.ndarray, np.ndarray]]
                ) -> None:
    """Rewrite every id column in place. Ids at or past a table's raw size
    cannot appear -- features.py already clamps them to 0 -- but clip anyway so
    a corpus built by an older builder fails to UNK rather than IndexError."""
    def m(t: str, a: np.ndarray) -> np.ndarray:
        lut = remap[t][1]
        return lut[np.clip(a, 0, len(lut) - 1)]
    data.slots = m("slot_emb", data.slots)
    data.xslots = m("slot_emb", data.xslots)
    for nm in BAGS:
        data.bag_flat[nm] = m("bag_emb", data.bag_flat[nm])
    data.opt_card = m("card_emb", data.opt_card)
    data.opt_tgt = m("card_emb", data.opt_tgt)
    data.opt_atk = m("atk_emb", data.opt_atk)
    for t in EMB_TABLES:
        ids = remap[t][0]
        print(f"  {t:9s} {len(remap[t][1]):5d} raw ids -> {ids.size + 2:4d} "
              f"rows (PAD + UNK + {ids.size} seen)")


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
        # E1 auxiliary heads are append-only. A plain v5 checkpoint has no
        # auxiliary tensors, so warm-starting it deliberately leaves these
        # heads at their seeded initialization while loading the policy
        # byte-for-byte. A later multitask checkpoint restores them as well.
        for prefix, head in (("outcome", model.outcome_head),
                             ("count", model.count_head)):
            if head is None or f"{prefix}_w" not in z:
                continue
            w, b = z[f"{prefix}_w"], z[f"{prefix}_b"]
            if w.shape != tuple(head.weight.shape):
                raise SystemExit(f"--init {path.name}: {prefix}_w is {w.shape}, "
                                 f"model wants {tuple(head.weight.shape)}")
            head.weight.copy_(torch.from_numpy(w))
            head.bias.copy_(torch.from_numpy(b))
        # E2 adapters are append-only. A plain v5 checkpoint has none, so
        # warm-starting leaves the zero-initialized residuals in place.
        if model.adapters is not None and "adapter_names" in z:
            names = [str(x) for x in z["adapter_names"].tolist()]
            for name in names:
                if name not in model.adapters:
                    raise SystemExit(
                        f"--init {path.name}: unknown adapter {name!r}")
                seq = model.adapters[name]
                lins = [m for m in seq if isinstance(m, nn.Linear)]
                n = int(z[f"adapter_{name}_n"][0])
                if n != len(lins):
                    raise SystemExit(
                        f"--init {path.name}: adapter {name} has {n} layers, "
                        f"model has {len(lins)}")
                for i, lin in enumerate(lins):
                    w, b = (z[f"adapter_{name}{i}_w"],
                            z[f"adapter_{name}{i}_b"])
                    if w.shape != tuple(lin.weight.shape):
                        raise SystemExit(
                            f"--init {path.name}: adapter_{name}{i}_w is "
                            f"{w.shape}, model wants "
                            f"{tuple(lin.weight.shape)}")
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


def _make_adapter(in_dim: int, hidden: int) -> nn.Sequential:
    """Residual logit MLP; final layer is zero-initialized for v5 equivalence."""
    seq = nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    nn.init.zeros_(seq[-1].weight)
    nn.init.zeros_(seq[-1].bias)
    return seq


class PolicyNet(nn.Module):
    def __init__(self, state_h: tuple[int, ...] = (256,),
                 head_h: tuple[int, ...] = (128,), dropout: float = 0.1,
                 opt_cols: int = OPT_DENSE, extra: bool = True,
                 pool: bool = False, outcome: bool = False,
                 count: bool = False, adapter_names: list[str] | None = None,
                 adapter_h: int = 64, adapters_off: bool = False,
                 attr: bool = False,
                 rows: dict[str, int] | None = None, pad: bool = False):
        super().__init__()
        self.opt_cols = opt_cols
        self.extra = extra
        self.pool = pool
        self.adapters_off = adapters_off
        self.adapter_h = adapter_h
        self.attr = attr
        # `rows` shrinks each table to its own vocabulary (--vocab); absent, the
        # tables span the raw id space exactly as v3-v6 did. Only the ROW count
        # changes -- EMB is untouched -- so every downstream width, and hence
        # every exported layer shape, is identical to the control's.
        r = rows or {}
        pi = PAD_IX if pad else None
        self.slot_emb = nn.Embedding(r.get("slot_emb", N_CARD_IDS), EMB,
                                     padding_idx=pi)
        self.bag_emb = nn.EmbeddingBag(r.get("bag_emb", N_CARD_IDS), EMB,
                                       mode="mean", include_last_offset=True,
                                       padding_idx=pi)
        self.card_emb = nn.Embedding(r.get("card_emb", N_CARD_IDS), EMB,
                                     padding_idx=pi)
        self.atk_emb = nn.Embedding(r.get("atk_emb", N_ATTACK_IDS), EMB,
                                    padding_idx=pi)
        in_state = DENSE_DIM + 12 * EMB + len(BAGS) * EMB + SEL_DENSE
        if extra:                       # the v4 block, appended (features.py)
            in_state += N_EXTRA + N_XSLOT * EMB
        if pool:                        # the v5 block, appended (optfeat.py)
            in_state += pool_width(opt_cols, EMB)
        if attr:                        # the v6 block, appended (features.py)
            in_state += N_ATTR
        self.state_fc = _mlp([in_state, *state_h], dropout, None)
        in_head = state_h[-1] + opt_cols + 3 * EMB
        self.head = _mlp([in_head, *head_h], dropout, 1)
        # Constructed AFTER every policy parameter. Resetting the seed therefore
        # gives a control and an auxiliary treatment identical policy weights;
        # only the treatment consumes additional RNG after that point.
        self.outcome_head = (nn.Linear(state_h[-1], 1) if outcome else None)
        self.count_head = (nn.Linear(state_h[-1], 1) if count else None)
        # E2 adapters are also append-only and zero-initialized, so an untrained
        # treatment matches the frozen base logits exactly.
        self.adapter_names = list(adapter_names or [])
        self.adapter_route_ids: dict[str, int] = {}
        if self.adapter_names:
            unknown = [n for n in self.adapter_names if n not in NAME_TO_ROUTE
                       or NAME_TO_ROUTE[n] == 0]
            if unknown:
                raise SystemExit(
                    f"adapters must be non-general route names; got {unknown}")
            self.adapter_route_ids = {n: NAME_TO_ROUTE[n]
                                      for n in self.adapter_names}
            self.adapters = nn.ModuleDict({
                n: _make_adapter(in_head, adapter_h)
                for n in self.adapter_names
            })
        else:
            self.adapters = None

    def forward(self, dense, slots, bag_flat, bag_off, seld,
                opt_dense, opt_card, opt_atk, opt_tgt, opt_row,
                xdense=None, xslots=None, attrs=None, routes=None,
                return_state: bool = False):
        # The per-option encoding is built FIRST, because the v5 pool feeds it
        # into the state. It is the same tensor the head consumes below, so the
        # pool costs one reduction and no extra embedding lookups.
        # Slice to `opt_cols`. The v3 target block is APPENDED to the v2 layout,
        # so `--opt-cols 25` trains the exact v2-feature control on the identical
        # rows -- same games, same selects, same labels, only the features differ.
        # That is a cleaner control than comparing against the shipped net, which
        # also differs in corpus (2,810 games vs whatever is on disk now).
        oenc = torch.cat([opt_dense[:, :self.opt_cols],
                          self.card_emb(opt_card),
                          self.atk_emb(opt_atk),
                          self.card_emb(opt_tgt)], dim=1)     # (O, D)
        parts = [dense, self.slot_emb(slots).flatten(1)]
        for name in BAGS:
            parts.append(self.bag_emb(bag_flat[name], bag_off[name]))
        parts.append(seld)
        # v4 goes LAST so that `--no-extra` reproduces the v3 state vector
        # byte-for-byte on the identical rows -- the same control discipline as
        # `--opt-cols 25` for the option block.
        if self.extra:
            parts.append(xdense)
            parts.append(self.slot_emb(xslots).flatten(1))
        # ...and v5 goes after v4, so `pool=False` reproduces the v4 state
        # vector byte-for-byte. Same discipline, third generation.
        if self.pool:
            parts.append(self._pool(oenc, opt_row, dense.shape[0]))
        # ...and v6 goes after v5, so `attr=False` reproduces the v5 state
        # vector byte-for-byte. Same discipline, fourth generation.
        if self.attr:
            parts.append(attrs)
        srepr = self.state_fc(torch.cat(parts, dim=1))       # (B, H)
        per_opt = torch.cat([srepr[opt_row], oenc], dim=1)   # (O, ...)
        logits = self.head(per_opt).squeeze(1)               # (O,)
        if (self.adapters is not None and not self.adapters_off
                and routes is not None):
            residual = torch.zeros_like(logits)
            row_route = routes[opt_row]
            for name, route_id in self.adapter_route_ids.items():
                mask = row_route == route_id
                if mask.any():
                    residual[mask] = self.adapters[name](
                        per_opt[mask]).squeeze(1)
            logits = logits + residual
        return (logits, srepr) if return_state else logits

    def _pool(self, oenc: torch.Tensor, opt_row: torch.Tensor,
              n_rows: int) -> torch.Tensor:
        """Segment mean/max of the option encodings + two count scalars.

        A permutation-invariant summary of the option SET, which is the one
        thing an independently-scored option can never carry. Empty selects
        (none exist in the corpus, but the arena can produce one) pool to zero
        rather than to -inf."""
        d = oenc.shape[1]
        idx = opt_row.unsqueeze(1).expand(-1, d)
        cnt = torch.zeros(n_rows, device=oenc.device).index_add_(
            0, opt_row, torch.ones_like(opt_row, dtype=oenc.dtype))
        mean = torch.zeros(n_rows, d, device=oenc.device).index_add_(
            0, idx[:, 0], oenc) / cnt.clamp_min(1.0).unsqueeze(1)
        mx = torch.full((n_rows, d), -1e30, device=oenc.device).scatter_reduce(
            0, idx, oenc, reduce="amax", include_self=True)
        nz = (cnt > 0).unsqueeze(1)
        mx = torch.where(nz, mx, torch.zeros_like(mx))
        scal = torch.stack([cnt.clamp_max(40.0) / 40.0,
                            torch.log1p(cnt) / float(np.log(41.0))], dim=1)
        return torch.cat([mean, mx, scal], dim=1)


def parse_episode_span(spec: str) -> tuple[float, float]:
    """Parse `--episode-span START:END` into inclusive-exclusive fractions."""
    if ":" not in spec:
        raise SystemExit("--episode-span needs START:END (e.g. 0:0.5)")
    a, b = (s.strip() for s in spec.split(":", 1))
    try:
        start, end = float(a), float(b)
    except ValueError as exc:
        raise SystemExit(f"--episode-span {spec!r}: {exc}") from exc
    if not (0.0 <= start < end <= 1.0):
        raise SystemExit("--episode-span requires 0 <= START < END <= 1")
    return start, end


def episode_span_mask(gid: np.ndarray, start: float, end: float) -> np.ndarray:
    """True for rows in [floor(n*start), floor(n*end)) within each gid.

    Row order is the order they appear in `gid` (shard-concat chronological
    order from build_policy_dataset). Odd-length games put the middle row in
    the second half when start=0.5 (floor splits)."""
    keep = np.zeros(len(gid), dtype=bool)
    # First pass: counts per gid in appearance order, without sorting the
    # whole array (gids are not contiguous across day dirs).
    order: dict[int, list[int]] = {}
    for i, g in enumerate(gid.tolist()):
        order.setdefault(g, []).append(i)
    for idxs in order.values():
        n = len(idxs)
        lo = int(n * start)
        hi = int(n * end)
        for j in idxs[lo:hi]:
            keep[j] = True
    return keep


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


def count_targets(seld: torch.Tensor, chosen: torch.Tensor,
                  opt_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target fraction and validity mask for variable-count selects.

    This is the row-level equivalent of `count_fraction_table`: the old table
    averages these targets within `(selectType, context)` buckets, while E1
    asks the shared state representation to predict each row separately.
    """
    n_rows = seld.shape[0]
    picked = torch.zeros(n_rows, dtype=chosen.dtype,
                         device=chosen.device).index_add_(0, opt_row, chosen)
    mn = seld[:, 11] * 5.0
    mx = seld[:, 12] * 5.0
    valid = mx > mn + 1e-6
    target = (picked - mn) / (mx - mn).clamp_min(1e-6)
    return target.clamp(0.0, 1.0), valid


class Data:
    def __init__(self, paths: list[Path]):
        sd, slots, seld, gid, won, rating = [], [], [], [], [], []
        xd, xs, at = [], [], []
        # B8. Present only in shards written by p26_selfplay_gen.py; a BC
        # corpus gets NaN, which is what `--advantage` refuses to weight.
        margin: list = []
        adv: list = []
        self.rows_per_path: list[int] = []
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
            # Corpora built before day 12 have no v4 block; zeros keep them
            # loadable, and `--extra` on such a corpus is refused in main().
            xd.append(z["xdense"] if "xdense" in z
                      else np.zeros((n, N_EXTRA), dtype=np.float32))
            xs.append(z["xslots"] if "xslots" in z
                      else np.zeros((n, N_XSLOT), dtype=np.int32))
            # Same contract for the v6 block: corpora built before day 20 get
            # zeros so they stay loadable, and `--attr` on such a corpus is
            # refused in main() rather than silently training on nothing.
            at.append(z["attr"] if "attr" in z
                      else np.zeros((n, N_ATTR), dtype=np.float32))
            gid.append(z["gid"])
            won.append(z["won"])
            self.rows_per_path.append(n)
            margin.append(z["margin"] if "margin" in z
                          else np.full(n, np.nan, dtype=np.float32))
            # E27: the per-decision TD residual written by p92_td_advantage.py.
            # NaN for every corpus built before it, which --advantage-col
            # refuses to weight rather than treating as zero.
            adv.append(z["adv"] if "adv" in z
                       else np.full(n, np.nan, dtype=np.float32))
            # Corpora built before `--ratings` have no per-row demonstrator.
            rating.append(z["rating"] if "rating" in z
                          else np.full(n, np.nan, dtype=np.float32))
            # Pre-v6 corpora store OPT_DENSE_V3 (37) cols; current builds store
            # OPT_DENSE (46). Pad the short layout with zeros so mixed --ds
            # unions concatenate; --opt-cols then slices the prefix it needs.
            od_arr = z["opt_dense"]
            w = int(od_arr.shape[1])
            if w < OPT_DENSE:
                od_arr = np.pad(od_arr, ((0, 0), (0, OPT_DENSE - w)))
            elif w > OPT_DENSE:
                od_arr = od_arr[:, :OPT_DENSE]
            od.append(od_arr)
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
        self.xdense = np.concatenate(xd)
        self.xslots = np.concatenate(xs).astype(np.int64)
        self.attr = np.concatenate(at)
        self.has_extra = all("xdense" in np.load(p) for p in paths)
        self.has_attr = all("attr" in np.load(p) for p in paths)
        self.gid = np.concatenate(gid)
        self.won = np.concatenate(won)
        self.rating = np.concatenate(rating)
        self.margin = np.concatenate(margin)
        self.adv = np.concatenate(adv)
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
        # E2 route labels from observable opponent slots + discard only.
        self.routes = routes_from_corpus(
            self.slots, self.bag_flat["opp_discard"],
            self.bag_off["opp_discard"])

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
                   sel,
                   torch.from_numpy(self.xdense[sel]),
                   torch.from_numpy(self.xslots[sel]),
                   torch.from_numpy(self.attr[sel]),
                   torch.from_numpy(self.routes[sel]))


def td_advantage_weights(data: "Data", is_rl: np.ndarray, beta: float,
                         anchor_w: float) -> np.ndarray:
    """E27: AWR over the PER-DECISION TD residual, not the game result.

    `w = exp(beta * A / sd(A))` on RL rows. **The normalisation by sd(A) is the
    part that has to be argued, and it is frozen in the pre-registration rather
    than tuned**: a TD residual has sd ~0.07 while B8's `won - baseline` has
    sd ~0.5, so passing the same beta to both would make this reweighting ~7x
    gentler than the one that already measured null (§8ao). Dividing by sd puts
    beta in units of "standard deviations of advantage", and **beta = 0.5
    reproduces B8's weight RATIO of e ~ 2.72 between a +1sd and a -1sd
    decision** -- so E27 differs from B8 in the SIGNAL, not in how hard it
    pushes. Tuning beta afterwards is the shopping B8 was denied.

    ⚠ Rows whose advantage is NaN (any corpus not passed through
    `p92_td_advantage.py`) are refused outright rather than silently weighted 1,
    because a half-annotated corpus would train mostly on unweighted rows and
    report a perfectly ordinary loss curve.
    """
    if not is_rl.any():
        raise SystemExit("--advantage-col needs RL shards from "
                         "p26_selfplay_gen.py; every row came from --anchor-ds")
    a = data.adv
    bad = int(np.isnan(a[is_rl]).sum())
    if bad:
        raise SystemExit(
            f"{bad:,} of {int(is_rl.sum()):,} RL rows have no `adv` column. "
            f"Run scripts/p92_td_advantage.py over the corpus first -- "
            f"training on a partly-annotated corpus is a silent null.")
    sd = float(np.std(a[is_rl]))
    if sd <= 0:
        raise SystemExit("advantage column has zero variance -- nothing to weight")
    w = np.ones(data.n, dtype=np.float32)
    z = (a[is_rl] - float(np.mean(a[is_rl]))) / sd
    # 🔴 Clip the STANDARDISED advantage at +/-2 sd, chosen on a property of
    # the data and not of any score (E25's discipline for picking tau). The
    # advantage distribution is a spike at zero plus a heavy tail -- the same
    # shape §8by found for rollout value -- so an unclipped exponent hands a
    # handful of outlier transitions enormous weight: at +/-4 the effective
    # sample size measured 50.8% against B8's 91.3%, i.e. half the corpus
    # thrown away to variance before any signal is asked for. At +/-2 the FULL
    # weight range is [e^-1, e^+1], which is exactly B8's win-row/loss-row
    # range end to end.
    w[is_rl] = np.exp(beta * np.clip(z, -2.0, 2.0))
    w[~is_rl] = anchor_w
    w = (w / w.mean()).astype(np.float32)
    print(f"--advantage-col {beta}: adv mean={np.mean(a[is_rl]):+.5f} "
          f"sd={sd:.4f}; weights [{w.min():.4f}, {w.max():.3f}]; "
          f"ratio(+1sd/-1sd)={np.exp(2 * beta):.2f}")
    print(f"  anchor rows {int((~is_rl).sum()):,} at {anchor_w}")
    ess = w[is_rl].sum() ** 2 / np.square(w[is_rl]).sum()
    print(f"  effective sample size {ess:,.0f} of {int(is_rl.sum()):,} RL rows "
          f"({ess / max(int(is_rl.sum()), 1):.1%})")
    return w


def advantage_weights(data: "Data", is_rl: np.ndarray, beta: float,
                      anchor_w: float, margin_max: float) -> np.ndarray:
    """B8: advantage-weighted regression over our OWN recorded outcomes.

    The weight on an RL row is `exp((won - baseline) / beta)`, so a select from
    a game we won is cloned harder than one from a game we lost. This is
    deliberately NOT a policy gradient with negative steps: an AWR weight is
    always positive, so the update can only ever re-weight behaviour the clone
    already produces. That is the property that makes it safe to run on a net
    worth ~942 on the ladder -- the downside is bounded by how far a reweighting
    can move it, not by an unbounded ascent direction.

    ⚠ **`--winners-only` is not this.** It scored 0.375 (§1) by filtering OTHER
    people's games and discarding half the corpus. Here nothing is discarded,
    the games are our own, and the losing rows still train -- at lower weight.
    The distinction is the whole reason B8 is not a repeat of that measurement.

    `anchor_w` is the weight held by corpus rows, which keeps the fine-tune
    tethered to the clone (rule: the thing being risked is a working agent).
    `margin_max` restricts the reweighting to selects where the net's top-1
    logit lead was small. §8u measured that agreement with the FIELD predicts
    strength, so a confident select is one we have a positive reason not to
    disturb; and an outcome signal can only change a decision that was close.
    Rows above the threshold fall back to weight 1 -- still trained, just not
    re-weighted.
    """
    if not is_rl.any():
        raise SystemExit("--advantage needs RL shards (a corpus built by "
                         "p26_selfplay_gen.py); every row came from --anchor-ds")
    won = data.won
    base = float(won[is_rl].mean())
    w = np.ones(data.n, dtype=np.float32)
    gate = is_rl.copy()
    if margin_max > 0:
        m = data.margin
        # NaN margins are BC rows; they are never gated here anyway.
        gate &= np.nan_to_num(m, nan=np.inf) <= margin_max
    w[gate] = np.exp((won[gate] - base) / max(beta, 1e-6))
    w[~is_rl] = anchor_w
    # Normalise over the rows that actually carry the objective, so the step
    # size stays comparable to the byte-identical control's (§8z's discipline).
    w = (w / w.mean()).astype(np.float32)
    n_gate = int(gate.sum())
    print(f"--advantage {beta}: baseline won={base:.4f}; "
          f"{n_gate:,} of {int(is_rl.sum()):,} RL rows re-weighted "
          f"({n_gate / max(int(is_rl.sum()), 1):.1%}"
          + (f", margin<={margin_max}" if margin_max > 0 else "") + ")")
    print(f"  weights [{w.min():.4f}, {w.max():.3f}]; "
          f"anchor rows {int((~is_rl).sum()):,} at {anchor_w}")
    ess = w[is_rl].sum() ** 2 / np.square(w[is_rl]).sum()
    print(f"  effective sample size {ess:,.0f} of {int(is_rl.sum()):,} RL rows "
          f"({ess / max(int(is_rl.sum()), 1):.1%})")
    return w


def apply_freeze(model: "PolicyNet", spec: str) -> None:
    """Train only the named top-level parameter groups; freeze the rest.

    B8's pre-registered form is "fine-tune a SMALL parameter set", and the
    reason is §8w: 8.2x the parameters bought -43 decisions, so capacity is not
    what is missing and a full-net update on a much smaller, much noisier
    corpus is the way to lose the clone rather than improve it.
    """
    keep = {s.strip() for s in spec.split(",") if s.strip()}
    known = {n.split(".", 1)[0] for n, _ in model.named_parameters()}
    unknown = keep - known
    if unknown:
        raise SystemExit(f"--freeze-except names {sorted(unknown)}; "
                         f"parameter groups are {sorted(known)}")
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        if name.split(".", 1)[0] in keep:
            n_train += p.numel()
        else:
            p.requires_grad_(False)
            n_frozen += p.numel()
    print(f"--freeze-except {sorted(keep)}: training {n_train:,} params, "
          f"froze {n_frozen:,} ({n_train / max(n_train + n_frozen, 1):.1%} live)")


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


def apply_x_drop(data: "Data", names: list[str]) -> np.ndarray:
    """Zero the named members of the v4 state block and return the mask.

    Zeroing rather than deleting keeps the layer widths, the parameter count and
    the weight init identical to the full-block run, so a drop-one arm differs
    from it in the CONTENT of a few columns and nothing else. An xslot set to 0
    is the same "no card" row the embedding already uses for an absent stadium.
    """
    mask = np.ones(N_EXTRA + N_XSLOT, dtype=np.float32)
    for nm in names:
        if nm not in X_GROUPS:
            raise SystemExit(f"--drop-x {nm}: known groups are "
                             f"{', '.join(X_GROUPS)}")
        for i in X_GROUPS[nm]:
            mask[i] = 0.0
    data.xdense = data.xdense * mask[:N_EXTRA]
    data.xslots = np.where(mask[N_EXTRA:] > 0, data.xslots, 0)
    print(f"--drop-x {','.join(names)}: zeroed xdense cols "
          f"{[i for i in range(N_EXTRA) if mask[i] == 0]} and xslot cols "
          f"{[i for i in range(N_XSLOT) if mask[N_EXTRA + i] == 0]}")
    return mask


def apply_a_drop(data: "Data", names: list[str]) -> np.ndarray:
    """Zero the named members of the v6 attribute block and return the mask.

    Same discipline as `apply_x_drop`: the block ships whole, so without a
    drop-one arm nothing would say WHICH of energyType / weakness / ability /
    resist / weakHit paid for the result. Zeroing keeps widths, parameter count
    and init identical, so an arm differs only in column content.
    """
    mask = np.ones(N_ATTR, dtype=np.float32)
    for nm in names:
        if nm not in A_GROUPS:
            raise SystemExit(f"--drop-a {nm}: known groups are "
                             f"{', '.join(A_GROUPS)}")
        for i in A_GROUPS[nm]:
            mask[i] = 0.0
    data.attr = data.attr * mask
    print(f"--drop-a {','.join(names)}: zeroed {int((mask == 0).sum())} of "
          f"{N_ATTR} attr columns")
    return mask


def export_npz(model: PolicyNet, path: Path, count_frac: np.ndarray,
               x_mask: np.ndarray | None = None,
               a_mask: np.ndarray | None = None,
               remap: dict[str, tuple[np.ndarray, np.ndarray]] | None = None):
    """Export every Linear generically, so inference mirrors any depth."""
    def arr(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    out: dict[str, np.ndarray] = {
        "slot_emb": arr(model.slot_emb.weight),
        "bag_emb": arr(model.bag_emb.weight),
        "card_emb": arr(model.card_emb.weight),
        "atk_emb": arr(model.atk_emb.weight),
        "count_frac": count_frac,
        # Width of the v5 pooled block, 0 if the net has none. Inference cannot
        # derive this from `state_in` alone (the v4 and v5 widths are both
        # legal), so it is recorded explicitly. Nets exported before day 13 have
        # no such key and are read as 0.
        "n_pool": np.array([pool_width(model.opt_cols, EMB) if model.pool
                            else 0], dtype=np.int64),
        # Width of the v6 attribute block, 0 if the net has none. Same reason as
        # n_pool: `state_in` alone no longer identifies the layout once three
        # optional blocks exist. Nets exported before day 20 lack this key and
        # are read as 0.
        "n_attr": np.array([N_ATTR if model.attr else 0], dtype=np.int64),
    }
    if remap is not None:
        # The raw ids this net's rows stand for, in row order after PAD and UNK.
        # Inference rebuilds the lookup from these, so the map can never drift
        # from the tables it was trained with -- they travel in one file.
        for t in EMB_TABLES:
            out[f"vocab_{t}"] = remap[t][0].astype(np.int64)
    if a_mask is not None:
        out["a_mask"] = a_mask
    if x_mask is not None:
        # Which members of the v4 block this net was actually shown. Inference
        # applies it, so an ablation arm can never be fed a column it never saw.
        out["x_mask"] = x_mask
    for prefix, seq in (("sfc", model.state_fc), ("head", model.head)):
        n = 0
        for mod in seq:
            if isinstance(mod, nn.Linear):
                out[f"{prefix}{n}_w"] = arr(mod.weight)
                out[f"{prefix}{n}_b"] = arr(mod.bias)
                n += 1
        out[f"n_{prefix}"] = np.array([n], dtype=np.int64)
    for prefix, head in (("outcome", model.outcome_head),
                         ("count", model.count_head)):
        if head is not None:
            out[f"{prefix}_w"] = arr(head.weight)
            out[f"{prefix}_b"] = arr(head.bias)
    if model.adapters is not None:
        out["adapter_names"] = np.asarray(model.adapter_names)
        out["adapter_h"] = np.array([model.adapter_h], dtype=np.int64)
        out["adapter_route_ids"] = np.asarray(
            [model.adapter_route_ids[n] for n in model.adapter_names],
            dtype=np.int64)
        for name, seq in model.adapters.items():
            n = 0
            for mod in seq:
                if isinstance(mod, nn.Linear):
                    out[f"adapter_{name}{n}_w"] = arr(mod.weight)
                    out[f"adapter_{name}{n}_b"] = arr(mod.bias)
                    n += 1
            out[f"adapter_{name}_n"] = np.array([n], dtype=np.int64)
    np.savez_compressed(path, **out)
    print(f"exported -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="artifacts/pds",
                    help="shard dir; comma-separated for several")
    ap.add_argument("--anchor-ds", default="",
                    help="B8: corpus shard dir(s) mixed in as the ANCHOR term, "
                         "so the fine-tune cannot drift off the clone it "
                         "started from. Rows from here are never "
                         "advantage-weighted.")
    ap.add_argument("--advantage-col", type=float, default=0.0,
                    help="E27: AWR beta over the per-decision TD residual "
                         "written by p92_td_advantage.py, in units of sd(adv). "
                         "0.5 reproduces B8's weight ratio (e ~ 2.72).")
    ap.add_argument("--advantage", type=float, default=0.0,
                    help="B8: AWR temperature. Weight = exp((won-baseline)/B) "
                         "on --ds rows. 0 disables (plain cloning).")
    ap.add_argument("--anchor-w", type=float, default=1.0,
                    help="weight held by --anchor-ds rows")
    ap.add_argument("--primary-mass", type=float, default=0.0,
                    help="E3: target fraction of total supervised loss assigned "
                         "to --ds rows, with --anchor-ds supplying the remaining "
                         "mass. For example 0.1 gives curated DAgger labels 10%% "
                         "and the frozen BC corpus 90%%. 0 disables.")
    ap.add_argument("--margin-max", type=float, default=0.0,
                    help="B8: only re-weight selects whose top1-top2 logit "
                         "margin was <= this. 0 = re-weight every RL row.")
    ap.add_argument("--export-last", action="store_true",
                    help="export the FINAL epoch instead of the best-val one. "
                         "Required on both arms of any A/B where one arm's "
                         "objective is not corpus fit (rule 3) -- otherwise "
                         "the arms export different epochs and the comparison "
                         "is confounded by training length.")
    ap.add_argument("--freeze-except", default="",
                    help="B8: comma-separated top-level parameter groups to "
                         "train; everything else is frozen (e.g. 'head')")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--winners-only", action="store_true")
    ap.add_argument("--episode-span", default="",
                    help="keep only a chronological fraction of each game's "
                         "rows: START:END in [0,1] (e.g. 0:0.5 = first half, "
                         "0.5:1 = last half). Split is by decision-row count "
                         "per gid, in shard-concat order. Empty = full episode.")
    ap.add_argument("--loss", choices=("bce", "listwise", "both"),
                    default="listwise")
    ap.add_argument("--state-h", default="256",
                    help="comma-separated hidden widths for the state MLP")
    ap.add_argument("--head-h", default="128",
                    help="comma-separated hidden widths for the scoring MLP")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                    help="training device. Default cpu preserves historical "
                         "recipes; use cuda for the private E1 GPU sweep.")
    ap.add_argument("--aux-outcome-w", type=float, default=0.0,
                    help="E1: weight of win/loss BCE on the shared state "
                         "representation. 0 disables the outcome head.")
    ap.add_argument("--aux-count-w", type=float, default=0.0,
                    help="E1: weight of soft-label BCE for the selected-count "
                         "fraction on variable-count rows. 0 disables the head.")
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
    ap.add_argument("--seed", type=int, default=0,
                    help="torch/numpy seed. Vary it to SIZE run-to-run "
                         "variance, which is the confound behind every "
                         "net-vs-net A/B in this repo (§8z).")
    ap.add_argument("--drop-x", default="",
                    help="comma-separated members of the v4 state block to "
                         "ABLATE (features.X_GROUPS: "
                         f"{','.join(X_GROUPS)}). The surviving mask is stored "
                         "in the npz and applied at inference.")
    ap.add_argument("--attr", action="store_true",
                    help="the v6 block: append per-slot CARD ATTRIBUTES "
                         "(energyType, weakness, ability, resistance, "
                         "weak-to-facing-type) to the STATE vector. These come "
                         "from the card DB, which covers all 1,267 cards, so "
                         "unlike an embedding row they transfer to cards the "
                         "corpus never contained (E6). Default off = the v5 "
                         "state vector, byte-for-byte, on identical rows.")
    ap.add_argument("--drop-a", default="",
                    help="comma-separated members of the v6 attribute block to "
                         "ABLATE (features.A_GROUPS: "
                         f"{','.join(A_GROUPS)}). The surviving mask is stored "
                         "in the npz and applied at inference.")
    ap.add_argument("--vocab", default="",
                    help="the v7 block: an out/emb/vocab.json census. Collapses "
                         "each embedding table to the rows the corpus actually "
                         "trained, with row 0 = PAD and row 1 = a shared UNK "
                         "that every out-of-vocabulary card routes to. Implies "
                         "--pad. Default off = the v3-v6 raw id space, i.e. an "
                         "unseen card reads a random untrained row.")
    ap.add_argument("--pad", action="store_true",
                    help="pin embedding row 0 to zero and give it no gradient "
                         "(padding_idx). Alone, this is the v7 block's SECOND "
                         "half only -- the arm that isolates 'id 0 is "
                         "overloaded' from 'unseen cards read noise'.")
    ap.add_argument("--pool", action="store_true",
                    help="the v5 block: append a mean/max pool of the option "
                         "encodings + two count scalars to the STATE vector "
                         "(optfeat.pool_width). Default off = the v4 state "
                         "vector byte-for-byte, i.e. the control.")
    ap.add_argument("--no-extra", action="store_true",
                    help="ignore the v4 state block (features.extra_feats). "
                         "This is the day-12 CONTROL: identical rows, "
                         "identical recipe, the v3 state vector byte-for-byte.")
    ap.add_argument("--adapters", default="",
                    help="E2: comma-separated residual adapter names "
                         "(mirror,alakazam). Append-only; zero-initialized so "
                         "an untrained treatment matches the frozen base.")
    ap.add_argument("--adapter-h", type=int, default=64,
                    help="E2: hidden width of each residual adapter MLP")
    ap.add_argument("--adapters-off", action="store_true",
                    help="E2 control: keep adapters in the checkpoint but do "
                         "not add their residual during forward/training")
    args = ap.parse_args()
    adapter_names = [s.strip() for s in args.adapters.split(",") if s.strip()]
    if not 1 <= args.opt_cols <= OPT_DENSE:
        raise SystemExit(f"--opt-cols must be in 1..{OPT_DENSE}")
    if args.aux_outcome_w < 0 or args.aux_count_w < 0:
        raise SystemExit("auxiliary loss weights must be non-negative")
    if args.adapter_h < 1:
        raise SystemExit("--adapter-h must be positive")
    if args.adapters_off and not adapter_names:
        raise SystemExit("--adapters-off requires --adapters")
    if not 0.0 <= args.primary_mass < 1.0:
        raise SystemExit("--primary-mass must be in [0, 1)")
    if args.primary_mass > 0 and not args.anchor_ds:
        raise SystemExit("--primary-mass needs --anchor-ds; otherwise there is "
                         "no anchor mass to preserve")
    if args.primary_mass > 0 and (args.advantage > 0 or args.rating_temp > 0):
        raise SystemExit("--primary-mass, --advantage, and --rating-temp each "
                         "define row weights; choose one")
    if (args.aux_outcome_w > 0 or args.aux_count_w > 0) and not args.export_last:
        raise SystemExit("E1 auxiliary treatments require --export-last so "
                         "control and treatment export the same epoch")
    if adapter_names and not args.export_last:
        raise SystemExit("E2 adapter arms require --export-last so control "
                         "and treatment export the same epoch")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() "
                         "is false")
    if device.type == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads() - 1))
    # Seeded so that a control/treatment pair (e.g. --opt-cols 25 vs 37, ROADMAP
    # B1) differs in its FEATURES and not in dropout masks or batch order. Weight
    # init still differs where the layer widths differ, which cannot be avoided.
    torch.manual_seed(args.seed)
    paths: list[Path] = []
    for d in args.ds.split(","):
        d = d.strip()
        if not d:
            continue
        got = sorted((ROOT / d).rglob("shard_*.npz"))
        if not got:
            raise SystemExit(f"no shards under {ROOT / d}")
        paths += got
    n_primary = len(paths)
    for d in args.anchor_ds.split(","):
        d = d.strip()
        if not d:
            continue
        got = sorted((ROOT / d).rglob("shard_*.npz"))
        if not got:
            raise SystemExit(f"no shards under {ROOT / d}")
        paths += got
    data = Data(paths)
    # Which rows came from --ds rather than --anchor-ds. Built from the
    # per-path row counts Data records, so it cannot drift from the actual
    # concatenation order.
    is_rl = np.zeros(data.n, dtype=bool)
    is_rl[:sum(data.rows_per_path[:n_primary])] = True
    if args.anchor_ds:
        print(f"--anchor-ds: {int(is_rl.sum()):,} primary rows + "
              f"{int((~is_rl).sum()):,} anchor rows")
    remap = None
    rows = None
    if args.vocab:
        vp = ROOT / args.vocab
        if not vp.exists():
            raise SystemExit(f"{vp} missing -- run scripts/p53_emb_vocab.py")
        remap = build_remap(vp)
        print(f"--vocab {args.vocab}:")
        apply_remap(data, remap)
        rows = {t: int(remap[t][0].size) + 2 for t in EMB_TABLES}
        args.pad = True
    x_mask = None
    if args.drop_x:
        if args.no_extra:
            raise SystemExit("--drop-x ablates the v4 block; --no-extra "
                             "already removes all of it")
        x_mask = apply_x_drop(data, [s.strip() for s in args.drop_x.split(",")
                                     if s.strip()])
    a_mask = None
    if args.drop_a:
        a_mask = apply_a_drop(data, [s.strip() for s in args.drop_a.split(",")
                                     if s.strip()])
    keep = np.ones(data.n, dtype=bool)
    if args.episode_span:
        start, end = parse_episode_span(args.episode_span)
        span = episode_span_mask(data.gid, start, end)
        keep &= span
        print(f"--episode-span {start:g}:{end:g}: {int(span.sum())} of "
              f"{data.n} rows kept ({int(span.sum()) / max(data.n, 1):.1%} "
              f"by decision count per gid)")
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
    if args.advantage_col > 0:
        if args.advantage > 0:
            raise SystemExit("--advantage-col and --advantage both set data.w; "
                             "pick the terminal-outcome signal or the TD one")
        if args.rating_temp > 0:
            raise SystemExit("--advantage-col and --rating-temp both set data.w")
        data.w = td_advantage_weights(data, is_rl, args.advantage_col,
                                      args.anchor_w)
    if args.advantage > 0:
        if args.rating_temp > 0:
            raise SystemExit("--advantage and --rating-temp both set data.w; "
                             "pick one")
        if args.winners_only:
            # §1's 0.375 result IS --winners-only. Running both would produce a
            # net that is neither intervention and would be reported as B8.
            raise SystemExit("--advantage subsumes --winners-only (it keeps "
                             "the losing rows at lower weight); refusing both")
        data.w = advantage_weights(data, is_rl, args.advantage,
                                   args.anchor_w, args.margin_max)
    if args.primary_mass > 0:
        n_primary_rows = int(is_rl.sum())
        n_anchor_rows = int((~is_rl).sum())
        if not n_primary_rows or not n_anchor_rows:
            raise SystemExit("--primary-mass needs non-empty primary and anchor "
                             "datasets")
        primary_w = (args.primary_mass / (1.0 - args.primary_mass)
                     * n_anchor_rows / n_primary_rows)
        data.w = np.ones(data.n, dtype=np.float32)
        data.w[is_rl] = primary_w
        print(f"--primary-mass {args.primary_mass:g}: "
              f"{n_primary_rows:,} primary rows at {primary_w:.3f} + "
              f"{n_anchor_rows:,} anchor rows at 1.0")
    val_mask = (data.gid % 20) == 0
    train_idx = np.where(keep & ~val_mask)[0]
    val_idx = np.where(keep & val_mask)[0]
    print(f"{len(paths)} shards, {data.n} rows -> {len(train_idx)} train / "
          f"{len(val_idx)} val ({len(np.unique(data.gid))} games)")

    count_frac = count_fraction_table(data, train_idx)

    state_h = tuple(int(x) for x in args.state_h.split(","))
    head_h = tuple(int(x) for x in args.head_h.split(","))
    if not args.no_extra and not data.has_extra:
        raise SystemExit(f"{args.ds} was built before the v4 state block; "
                         "rebuild it or pass --no-extra")
    if args.attr and not data.has_attr:
        raise SystemExit(f"{args.ds} was built before the v6 attribute block; "
                         "rebuild it with scripts/build_policy_dataset.py")
    if args.drop_a and not args.attr:
        raise SystemExit("--drop-a ablates the v6 block; it needs --attr")
    if args.pool and args.no_extra:
        raise SystemExit("--pool is the v5 block and is defined as v4 + pool; "
                         "inference only knows the v4 and v5 state widths")
    model = PolicyNet(state_h=state_h, head_h=head_h, dropout=args.dropout,
                      opt_cols=args.opt_cols, extra=not args.no_extra,
                      pool=args.pool, outcome=args.aux_outcome_w > 0,
                      count=args.aux_count_w > 0,
                      adapter_names=adapter_names or None,
                      adapter_h=args.adapter_h,
                      adapters_off=args.adapters_off,
                      attr=args.attr, rows=rows, pad=args.pad).to(device)
    if args.init:
        load_init(model, ROOT / args.init)
    if args.freeze_except:
        if not args.init:
            raise SystemExit("--freeze-except without --init trains a few "
                             "layers on top of RANDOM embeddings; that is not "
                             "a fine-tune of anything")
        apply_freeze(model, args.freeze_except)
    if args.adapters_off and model.adapters is not None:
        # Control: adapters exist for export shape parity but must not train.
        for p in model.adapters.parameters():
            p.requires_grad_(False)
    if adapter_names:
        route_counts = {
            ROUTE_NAMES[rid]: int((data.routes == rid).sum())
            for rid in sorted(set(data.routes.tolist()))
        }
        print(f"E2 routes: {route_counts} adapters={adapter_names} "
              f"h={args.adapter_h} off={args.adapters_off}")
    print(f"arch: state{list(state_h)} head{list(head_h)} loss={args.loss} "
          f"opt_cols={args.opt_cols}/{OPT_DENSE} "
          f"extra={not args.no_extra} pool={args.pool} attr={args.attr} "
          f"vocab={bool(args.vocab)} pad={args.pad} "
          f"emb_params={sum(e.weight.numel() for e in (model.slot_emb, model.bag_emb, model.card_emb, model.atk_emb)):,} "
          f"aux_outcome={args.aux_outcome_w:g} "
          f"aux_count={args.aux_count_w:g} "
          f"adapters={adapter_names or []} "
          f"device={device.type} "
          f"params={sum(p.numel() for p in model.parameters())}")
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
           if trainable else None)
    bcef = nn.BCEWithLogitsLoss()
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(args.seed)
    best = -1.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        tot = seen = 0.0
        for batch in data.batches(train_idx, args.bs, rng):
            (dense, slots, bf, bo, seld, odn, ocd, oat, otg, orow, om,
             spans, rw, _sel, xd, xs, at, routes) = batch
            dense, slots, seld = (dense.to(device), slots.to(device),
                                  seld.to(device))
            bf = {k: v.to(device) for k, v in bf.items()}
            bo = {k: v.to(device) for k, v in bo.items()}
            odn, ocd, oat = odn.to(device), ocd.to(device), oat.to(device)
            otg, orow, om = otg.to(device), orow.to(device), om.to(device)
            rw, xd, xs = rw.to(device), xd.to(device), xs.to(device)
            at = at.to(device)
            routes = routes.to(device)
            if opt is not None:
                opt.zero_grad()
            need_state = args.aux_outcome_w > 0 or args.aux_count_w > 0
            result = model(dense, slots, bf, bo, seld, odn, ocd, oat, otg,
                           orow, xd, xs, at, routes=routes,
                           return_state=need_state)
            if need_state:
                out, srepr = result
            else:
                out, srepr = result, None
            loss = torch.zeros((), dtype=out.dtype)
            wrow = rw if (args.rating_temp > 0 or args.advantage > 0
                          or args.primary_mass > 0) else None
            if args.loss in ("bce", "both"):
                if wrow is None:
                    loss = loss + bcef(out, om)
                else:   # the row's weight applies to each of its options
                    per = bce_none(out, om) * wrow[orow]
                    loss = loss + per.sum() / wrow[orow].sum().clamp_min(1e-8)
            if args.loss in ("listwise", "both"):
                loss = loss + listwise_loss(out, om, orow, len(spans), wrow)
            if args.aux_outcome_w > 0:
                pred = model.outcome_head(srepr).squeeze(1)
                target = torch.from_numpy(data.won[_sel]).to(
                    device=device, dtype=pred.dtype)
                per = bce_none(pred, target)
                aux = (per.mean() if wrow is None else
                       (per * rw).sum() / rw.sum().clamp_min(1e-8))
                loss = loss + args.aux_outcome_w * aux
            if args.aux_count_w > 0:
                pred = model.count_head(srepr).squeeze(1)
                target, valid = count_targets(seld, om, orow)
                if valid.any():
                    per = bce_none(pred[valid], target[valid])
                    aux = (per.mean() if wrow is None else
                           (per * rw[valid]).sum()
                           / rw[valid].sum().clamp_min(1e-8))
                    loss = loss + args.aux_count_w * aux
            if opt is not None and loss.requires_grad:
                loss.backward()
                opt.step()
            tot += float(loss.detach()) * len(om)
            seen += len(om)
        # val: top-1 accuracy on single-choice rows
        model.eval()
        hit = tries = 0
        hit_hi = tries_hi = 0
        route_hit = {name: 0 for name in ROUTE_NAMES.values()}
        route_tries = {name: 0 for name in ROUTE_NAMES.values()}
        out_bce = out_ok = out_n = 0.0
        count_abs = count_n = 0.0
        with torch.no_grad():
            for batch in data.batches(val_idx, args.bs, None):
                (dense, slots, bf, bo, seld, odn, ocd, oat, otg, orow, om,
                 spans, _rw, vsel, xd, xs, at, routes) = batch
                dense, slots, seld = (dense.to(device), slots.to(device),
                                      seld.to(device))
                bf = {k: v.to(device) for k, v in bf.items()}
                bo = {k: v.to(device) for k, v in bo.items()}
                odn, ocd, oat = odn.to(device), ocd.to(device), oat.to(device)
                otg, orow, om = otg.to(device), orow.to(device), om.to(device)
                xd, xs = xd.to(device), xs.to(device)
                at = at.to(device)
                routes = routes.to(device)
                need_state = args.aux_outcome_w > 0 or args.aux_count_w > 0
                result = model(dense, slots, bf, bo, seld, odn, ocd, oat, otg,
                               orow, xd, xs, at, routes=routes,
                               return_state=need_state)
                if need_state:
                    out, srepr = result
                else:
                    out, srepr = result, None
                if args.aux_outcome_w > 0:
                    pred = model.outcome_head(srepr).squeeze(1)
                    target = torch.from_numpy(data.won[vsel]).to(
                        device=device, dtype=pred.dtype)
                    out_bce += float(bce_none(pred, target).sum())
                    out_ok += float(((pred >= 0) == (target >= 0.5)).sum())
                    out_n += len(target)
                if args.aux_count_w > 0:
                    pred = model.count_head(srepr).squeeze(1)
                    target, valid = count_targets(seld, om, orow)
                    if valid.any():
                        count_abs += float(
                            (torch.sigmoid(pred[valid]) - target[valid])
                            .abs().sum())
                        count_n += float(valid.sum())
                out = out.cpu().numpy()
                om = om.cpu().numpy()
                routes_np = routes.cpu().numpy()
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
                        rname = ROUTE_NAMES.get(int(routes_np[j]), "general")
                        route_hit[rname] += int(ok)
                        route_tries[rname] += 1
                        # Same rows, restricted to strong demonstrators. Rule 3
                        # still holds -- neither number predicts strength; this
                        # one just says WHOSE policy is being fit.
                        if data.rating[vsel[j]] >= VAL_HI_RATING:
                            hit_hi += ok
                            tries_hi += 1
        acc = hit / max(tries, 1)
        hi = hit_hi / max(tries_hi, 1)
        aux_msg = ""
        if args.aux_outcome_w > 0:
            aux_msg += (f" val_out_bce={out_bce / max(out_n, 1):.4f}"
                        f" val_out_acc={out_ok / max(out_n, 1):.4f}")
        if args.aux_count_w > 0:
            aux_msg += f" val_count_mae={count_abs / max(count_n, 1):.4f}"
        if adapter_names:
            for rname in ("mirror", "alakazam", "general"):
                rt = route_tries[rname]
                if rt:
                    aux_msg += (f" val_top1_{rname}="
                                f"{route_hit[rname] / rt:.4f}(n={rt})")
        print(f"epoch {epoch}: train={tot / seen:.4f} val_top1={acc:.4f} "
              f"val_top1@{VAL_HI_RATING:.0f}+={hi:.4f} (n={tries_hi})"
              f"{aux_msg} "
              f"({time.time() - t0:.0f}s)")
        # 🔴 rule 3. The default here SELECTS THE CHECKPOINT BY `val_top1`, and
        # that metric is measured not to predict strength in either direction
        # (§8z moved it by 8 decisions for +37 Elo; §8aa moved it by 214 for
        # +14 -- a 70x exchange-rate difference). It is tolerable for a plain
        # clone, whose objective IS corpus fit. It is NOT tolerable for any arm
        # whose objective deliberately departs from corpus fit: an
        # advantage-weighted arm will fit the corpus worse ON PURPOSE, stop
        # improving `acc` earlier, and export an EARLIER EPOCH than its control
        # -- so the A/B would be comparing epoch counts, not the intervention.
        # `--export-last` is mandatory for both arms of such a pair.
        if args.export_last:
            if epoch == args.epochs - 1:
                export_npz(model, ROOT / args.out, count_frac, x_mask, a_mask,
                           remap)
        elif acc > best:
            best = acc
            export_npz(model, ROOT / args.out, count_frac, x_mask, a_mask,
                           remap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
