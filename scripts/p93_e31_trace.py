"""E31 Part A -- does the clone use "what I just did" MORE than its demonstrator?

Pre-registration: docs/experiments/E31-trace-reliance.md (frozen a15eab7,
amended e3006f3 before this file existed).

E28's reading 1 asked whether our actions are predictable from the previous
action and died on its own positive control, because the cue it was given (the
previous symbol ALONE) cannot represent the answer on the 48.7% of slots where
repeating is illegal. The repair is to give the predictor the option list, which
`p87_e28_pairs.py:118` was already extracting and reading 1 then discarded.

The statistic is deliberately NOT a level. Trace dependence is a real property
of this game -- turns are sequences and later actions depend on earlier ones --
so a net that uses the trace is not thereby defective. Copycat is AMPLIFICATION:

    delta        = accuracy(cue = available symbols + previous symbol)
                 - accuracy(cue = available symbols)
    STATISTIC    = delta_net - delta_expert          (identical rows, both)

computed against two label sources on the very same pairs. Positive and CI clear
of zero means the net leans on the trace beyond what the humans it copied do.

    python -X utf8 scripts/p93_e31_trace.py                 # A1 + controls
    python -X utf8 scripts/p93_e31_trace.py --a2            # + localization
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts"):
    sys.path.insert(0, str(ROOT / sub))

from ptcg.env import sdk  # noqa: E402

sdk.load()

from context_accuracy import Net, bag_means, BAGS  # noqa: E402
from sa import cards as cdb  # noqa: E402
from sa.optfeat import pool_scalars  # noqa: E402

BACKOFF_MIN = 20      # (prev, sym) availability count below which we back off
N_BOOT = 2000


# --- the alphabet (E28 3: decision kind x card class, coarse on purpose) -----

def card_class(cid: int) -> int:
    """Coarse class of an option's card. 0 = none/unknown."""
    if cid <= 0:
        return 0
    if cdb.is_basic_energy(cid):
        return 1
    if cdb.is_pokemon(cid):
        c = cdb.card(cid)
        if c.get("megaEx") or c.get("ex"):
            return 2
        return 3 if c.get("basic") else 4
    c = cdb.card(cid)
    for k, v in (("supporter", 5), ("item", 6), ("tool", 7), ("stadium", 8)):
        if c.get(k):
            return v
    return 9


_CLASS_CACHE: dict[int, int] = {}


def klass(cid: int) -> int:
    v = _CLASS_CACHE.get(cid)
    if v is None:
        v = _CLASS_CACHE[cid] = card_class(int(cid))
    return v


# --- corpus load + net argmax ------------------------------------------------

class Rows:
    """Flat per-decision arrays for the whole corpus, plus per-option symbols."""

    def __init__(self):
        self.gid = []
        self.tac = []
        self.ctx = []          # select type (seld one-hot 0..10)
        self.sym = []          # per row: np.ndarray of option symbols
        self.chosen = []       # index into that row's options
        self.argmax = {}       # arm name -> per-row argmax index


def _seltype(seld: np.ndarray) -> np.ndarray:
    """Select type from the 11-wide one-hot; 11 = none set."""
    oh = seld[:, :11]
    t = np.argmax(oh, axis=1)
    return np.where(oh.max(axis=1) > 0, t, 11).astype(np.int32)


def score_shard(net: Net, z, xd_over=None, bag_over=None,
                dense_over=None, chunk: int = 4096) -> np.ndarray:
    """Net argmax option index per row. Overrides let A2 perturb one channel."""
    n = len(z["gid"])
    off = z["opt_off"]
    width = net.bag_emb.shape[1]
    means = [bag_means(z, nm, n, width, net.bag_emb) for nm in BAGS]
    if bag_over is not None:
        means[BAGS.index("my_discard")] = bag_over
    dense = z["dense"] if dense_over is None else dense_over
    xd = z["xdense"] if xd_over is None else xd_over
    xs = z["xslots"].astype(np.int64)
    opt_dense, card = z["opt_dense"], z["opt_card"].astype(np.int64)
    atk = z["opt_attack"].astype(np.int64)
    tgt = (z["opt_target"] if "opt_target" in z
           else np.zeros_like(card)).astype(np.int64)

    ow = net.opt_in
    oenc = None
    if net.n_pool:
        oenc = np.concatenate([opt_dense[:, :ow], net.card_emb[card],
                               net.atk_emb[atk], net.card_emb[tgt]], axis=1)

    out = np.full(n, -1, dtype=np.int64)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        idx = np.arange(lo, hi)
        pool = None
        if net.n_pool:
            d = oenc.shape[1]
            pool = np.zeros((hi - lo, net.n_pool), dtype=np.float32)
            for k, row in enumerate(idx):
                a, b = off[row], off[row + 1]
                if b <= a:
                    continue
                blk = oenc[a:b]
                pool[k, :d] = blk.mean(axis=0)
                pool[k, d:2 * d] = blk.max(axis=0)
                pool[k, 2 * d:] = pool_scalars(b - a)
        srepr = net.state_repr(dense[idx], z["slots"][idx].astype(np.int64),
                               [m[idx] for m in means], z["seld"][idx],
                               xd[idx], xs[idx], pool)
        # one matmul for every option in the chunk, not one per row
        cnt = (off[lo + 1:hi + 1] - off[lo:hi]).astype(np.int64)
        a0, b0 = off[lo], off[hi]
        if b0 <= a0:
            continue
        # slice the per-option dense block to the width this net was TRAINED
        # at -- opt_dense is append-only too, so a newer corpus is wider than
        # an older net's head expects (policynet.opt_in).
        logits = net.option_logits(np.repeat(srepr, cnt, axis=0),
                                   opt_dense[a0:b0, :ow], card[a0:b0],
                                   atk[a0:b0], tgt[a0:b0])
        pos = 0
        for k, row in enumerate(idx):
            c = int(cnt[k])
            if c:
                # WITHIN-ROW index: shard-local absolute offsets restart at 0
                # in every shard, so anything absolute breaks on concatenation.
                out[row] = int(np.argmax(logits[pos:pos + c]))
            pos += c
    return out


def load(ds: str, net: Net, arms: dict) -> Rows:
    R = Rows()
    for name in arms:
        R.argmax[name] = []
    paths = sorted((ROOT / ds).rglob("shard_*.npz"))
    if not paths:
        raise SystemExit(f"no shards under {ds}")
    for p in paths:
        z = np.load(p)
        n = len(z["gid"])
        off = z["opt_off"]
        ch = z["opt_chosen"]
        card = z["opt_card"].astype(np.int64)
        st = _seltype(z["seld"])
        tac = np.rint(z["xdense"][:, 0] * 24).astype(np.int32)

        for name, fn in arms.items():
            R.argmax[name].append(fn(net, z))

        for row in range(n):
            a, b = off[row], off[row + 1]
            c = ch[a:b]
            if b <= a or c.sum() != 1:
                R.gid.append(-1)          # keep row alignment with argmax
                R.tac.append(-1)
                R.ctx.append(-1)
                R.sym.append(None)
                R.chosen.append(-1)
                continue
            R.gid.append(int(z["gid"][row]))
            R.tac.append(int(tac[row]))
            R.ctx.append(int(st[row]))
            R.sym.append(np.asarray([st[row] * 16 + klass(card[i])
                                     for i in range(a, b)], dtype=np.int32))
            R.chosen.append(int(np.argmax(c)))       # within-row, see above
        print(f"  {p.parent.name}/{p.name}: {n} rows", flush=True)
    R.gid = np.asarray(R.gid)
    R.tac = np.asarray(R.tac)
    R.ctx = np.asarray(R.ctx)
    R.chosen = np.asarray(R.chosen)
    for k in R.argmax:
        R.argmax[k] = np.concatenate(R.argmax[k])
    return R


# --- pairs -------------------------------------------------------------------

def pairs(R: Rows) -> np.ndarray:
    """Indices i such that (i-1, i) is a strict within-turn adjacency."""
    ok = (R.gid[1:] == R.gid[:-1]) & (R.tac[1:] - R.tac[:-1] == 1)
    ok &= (R.gid[1:] >= 0) & (R.gid[:-1] >= 0)
    return np.flatnonzero(ok) + 1


# --- the two symbol predictors ----------------------------------------------

class Predictor:
    """P(symbol chosen | symbol available), optionally conditioned on the
    previous symbol with a backoff to the unconditioned table."""

    def __init__(self):
        self.ch = defaultdict(int)      # sym -> times chosen
        self.av = defaultdict(int)      # sym -> times available
        self.tch = defaultdict(int)     # (prev, sym) -> times chosen
        self.tav = defaultdict(int)     # (prev, sym) -> times available

    def fit_one(self, avail: np.ndarray, chosen_sym: int, prev: int):
        for s in np.unique(avail):
            s = int(s)
            self.av[s] += 1
            self.tav[(prev, s)] += 1
        self.ch[chosen_sym] += 1
        self.tch[(prev, chosen_sym)] += 1

    def _base(self, s: int) -> float:
        a = self.av[s]
        return (self.ch[s] + 0.5) / (a + 1.0) if a else 0.0

    def _trace(self, prev: int, s: int) -> float:
        a = self.tav[(prev, s)]
        if a < BACKOFF_MIN:
            return self._base(s)
        return (self.tch[(prev, s)] + 0.5) / (a + 1.0)

    def predict(self, avail: np.ndarray, prev: int | None) -> int:
        best, bs = None, -1.0
        for s in np.unique(avail):
            s = int(s)
            v = self._base(s) if prev is None else self._trace(prev, s)
            if v > bs:
                bs, best = v, s
        return int(best)


def deltas(R: Rows, pr: np.ndarray, labels: np.ndarray,
           fit_mask: np.ndarray, prev_sym: np.ndarray):
    """Fit both predictors on fit rows, score both on the rest.

    Returns (hit_base, hit_trace, gid) over the SCORED pairs, aligned."""
    P = Predictor()
    for k in np.flatnonzero(fit_mask):
        i = pr[k]
        P.fit_one(R.sym[i], int(labels[k]), int(prev_sym[k]))
    sc = np.flatnonzero(~fit_mask)
    hb = np.zeros(len(sc), dtype=bool)
    ht = np.zeros(len(sc), dtype=bool)
    for j, k in enumerate(sc):
        i = pr[k]
        av = R.sym[i]
        hb[j] = P.predict(av, None) == labels[k]
        ht[j] = P.predict(av, int(prev_sym[k])) == labels[k]
    return hb, ht, R.gid[pr[sc]]


def boot_diff(hb_n, ht_n, hb_e, ht_e, gid, rng) -> tuple[float, float, float]:
    """Paired bootstrap over GAMES of (delta_net - delta_expert)."""
    games, inv = np.unique(gid, return_inverse=True)
    by = [np.flatnonzero(inv == g) for g in range(len(games))]
    point = (ht_n.mean() - hb_n.mean()) - (ht_e.mean() - hb_e.mean())
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, len(by), len(by))
        idx = np.concatenate([by[p] for p in pick])
        out[b] = ((ht_n[idx].mean() - hb_n[idx].mean())
                  - (ht_e[idx].mean() - hb_e[idx].mean()))
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    # The net that PLAYS, verified by hashing the npz inside dist/
    # submission.tar.gz -- NOT agents/sa/policy_net.npz, which is a pre-v5 net
    # (no pooling block) that the live bundle overrides via --policy-net.
    ap.add_argument("--net", default="out/policy_v5_s2.npz")
    ap.add_argument("--ds", default="artifacts/pds_v6")
    ap.add_argument("--a2", action="store_true", help="run localization too")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    net = Net(ROOT / args.net)
    rng = np.random.default_rng(args.seed)

    arms = {"net": lambda nt, z: score_shard(nt, z)}
    print(f"loading {args.ds} and scoring {Path(args.net).name} ...")
    R = load(args.ds, net, arms)

    pr = pairs(R)
    print(f"\nwithin-turn pairs: {len(pr)}")
    if len(pr) < 50000:
        raise SystemExit("VOID: below the pre-registered 50,000 pair floor")

    print("building symbols ...")
    prev_rows, cur_rows = pr - 1, pr
    prev_sym = np.asarray([int(R.sym[r][R.chosen[r]]) for r in prev_rows],
                          dtype=np.int32)
    lab_e = np.asarray([int(R.sym[r][R.chosen[r]]) for r in cur_rows],
                       dtype=np.int32)
    am = R.argmax["net"]
    lab_n = np.asarray([int(R.sym[r][am[r]]) for r in cur_rows],
                       dtype=np.int32)

    games = np.unique(R.gid[cur_rows])
    half = set(rng.permutation(games)[:len(games) // 2].tolist())
    fit = np.asarray([g in half for g in R.gid[cur_rows]])
    print(f"fit games {len(half)} / score games {len(games) - len(half)}")

    val = (R.gid[cur_rows] % 20) == 0

    def run(labels, mask=None, prev=None):
        p = prev_sym if prev is None else prev
        m = fit if mask is None else (fit | ~mask)
        return deltas(R, pr, labels, m, p)

    print("\n=== A1 primary (all pairs) ===")
    hb_e, ht_e, g_e = run(lab_e)
    hb_n, ht_n, _ = run(lab_n)
    _report("expert", hb_e, ht_e)
    _report("net   ", hb_n, ht_n)
    pt, lo, hi = boot_diff(hb_n, ht_n, hb_e, ht_e, g_e, rng)
    print(f"\nSTATISTIC  delta_net - delta_expert = {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    verdict(pt, lo, hi)

    print("\n=== controls ===")
    # positive: synthetic trace-follower
    lab_s = np.asarray([_follow(R, r, prev_sym[k])
                        for k, r in enumerate(cur_rows)], dtype=np.int32)
    hb_s, ht_s, g_s = run(lab_s)
    _report("synth ", hb_s, ht_s)
    ps, ls, hs = boot_diff(hb_s, ht_s, hb_e, ht_e, g_s, rng)
    print(f"  positive  delta_synth - delta_expert = {ps:+.4f} [{ls:+.4f}, {hs:+.4f}]"
          f"   {'PASS' if ls > 0 else 'FAIL -> VOID'}")
    # negative: permute prev symbol within select type
    perm = prev_sym.copy()
    ctx_cur = R.ctx[cur_rows]
    for c in np.unique(ctx_cur):
        m = np.flatnonzero(ctx_cur == c)
        perm[m] = prev_sym[rng.permutation(m)]
    hb_e2, ht_e2, g2 = run(lab_e, prev=perm)
    hb_n2, ht_n2, _ = run(lab_n, prev=perm)
    de = ht_e2.mean() - hb_e2.mean()
    dn = ht_n2.mean() - hb_n2.mean()
    print(f"  negative  delta_expert={de:+.4f}  delta_net={dn:+.4f}"
          f"   {'PASS' if max(abs(de), abs(dn)) < 0.01 else 'FAIL -> VOID'}")

    print("\n=== A1 robustness (net's val split only) ===")
    if val.sum() < 5000:
        print(f"  {val.sum()} pairs -- below the 5,000 floor, not reported")
    else:
        hb_e3, ht_e3, g3 = run(lab_e, mask=val)
        hb_n3, ht_n3, _ = run(lab_n, mask=val)
        p3, l3, h3 = boot_diff(hb_n3, ht_n3, hb_e3, ht_e3, g3, rng)
        print(f"  n={val.sum()}  delta_net - delta_expert = "
              f"{p3:+.4f} [{l3:+.4f}, {h3:+.4f}]")
        if np.sign(p3) != np.sign(pt):
            print("  SIGN DISAGREES WITH PRIMARY -> VOID (pre-registered)")

    if args.a2:
        a2(net, args.ds, R, rng)
    return 0


def _follow(R: Rows, row: int, prev: int) -> int:
    """The synthetic trace-follower's pick: repeat the previous symbol if it is
    available, else fall back to whatever the net picked."""
    av = R.sym[row]
    if (av == prev).any():
        return int(prev)
    return int(av[R.argmax["net"][row]])


def _report(name, hb, ht):
    print(f"  {name}  base {hb.mean():.4f}  +trace {ht.mean():.4f}  "
          f"delta {ht.mean() - hb.mean():+.4f}   n={len(hb)}")


def verdict(pt, lo, hi):
    if lo > 0:
        print("  => COPYCAT CONFIRMED: the net leans on the trace beyond its "
              "demonstrator. Part B opens.")
    elif hi < 0:
        print("  => NO COPYCAT PROBLEM (net is LESS trace-driven than the "
              "humans). Part B does not run.")
    else:
        print("  => NO COPYCAT PROBLEM: CI contains 0, the net inherits the "
              "trace dependence the data contains. Part B does not run.")


def a2(net: Net, ds: str, R: Rows, rng):
    """Localization: perturb one channel, measure the argmax change rate."""
    print("\n=== A2 localization (off-manifold; read only vs calibration) ===")
    paths = sorted((ROOT / ds).rglob("shard_*.npz"))
    chans = ["turnActionCount", "my_discard", "retreated", "stadiumPlayed",
             "dense_ctl"]
    moved = {c: 0 for c in chans}
    tot = 0
    for p in paths:
        z = np.load(p)
        n = len(z["gid"])
        st = _seltype(z["seld"])
        ref = score_shard(net, z)
        tot += int((ref >= 0).sum())
        for c in chans:
            if c == "my_discard":
                over = _start_of_turn_discard(net, z)
                cur = score_shard(net, z, bag_over=over)
            elif c == "dense_ctl":
                d = z["dense"].copy()
                col = _matched_col(z)
                d[:, col] = _resample(d[:, col], st, rng)
                cur = score_shard(net, z, dense_over=d)
            else:
                col = {"turnActionCount": 0, "retreated": 1,
                       "stadiumPlayed": 2}[c]
                xd = z["xdense"].copy()
                xd[:, col] = _resample(xd[:, col], st, rng)
                cur = score_shard(net, z, xd_over=xd)
            moved[c] += int(((cur != ref) & (ref >= 0)).sum())
        print(f"  {p.parent.name}/{p.name} done", flush=True)
    print(f"\n  channel                argmax changed  (n={tot})")
    for c in chans:
        print(f"  {c:<22} {moved[c] / max(tot, 1):8.4f}")


def _resample(col, strat, rng):
    out = col.copy()
    for s in np.unique(strat):
        m = np.flatnonzero(strat == s)
        out[m] = col[rng.permutation(m)]
    return out


def _matched_col(z) -> int:
    """The dense state column whose variance is closest to turnActionCount's."""
    v = z["xdense"][:, 0].var()
    dv = z["dense"].var(axis=0)
    return int(np.argmin(np.abs(dv - v)))


def _start_of_turn_discard(net: Net, z) -> np.ndarray:
    """Mean-pooled discard as it stood at the FIRST decision of each turn."""
    n = len(z["gid"])
    width = net.bag_emb.shape[1]
    cur = bag_means(z, "my_discard", n, width, net.bag_emb)
    tac = np.rint(z["xdense"][:, 0] * 24).astype(int)
    gid = z["gid"]
    out = cur.copy()
    anchor = 0
    for i in range(n):
        if tac[i] <= 1 or (i and gid[i] != gid[i - 1]):
            anchor = i
        out[i] = cur[anchor]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
