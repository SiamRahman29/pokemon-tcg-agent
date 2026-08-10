"""E17 — what does a rollout oracle over OUR OWN options buy, at a payable budget?

Pre-registered in `docs/experiments/E17-self-oracle-value.md`, frozen at `c0a2cc9`
before this file existed. The instrument and every control for it are
`p80_rollout_feasibility.py` (`EVIDENCE` §8bw); this script only runs E17.

**The question ROADMAP §2.7 has not asked.** Its sizing gate passed on E16's
dispersion (§8bx, 90th-percentile |gap| 0.1263 against a 0.10 line) — but that
was the gap between *our* pick and *the expert's* pick, at positions selected
for an expert having disagreed with us. **The clock has no expert.** It ranks
the net's own top-k options at positions selected by nothing, and §8bx's own
caveat (3) says the two gap distributions are "similar but not identical".

And the multiplier the whole build rests on — §8bx caveat (1)'s *"roughly half
survives at gap/SE ≈ 2"* — is an arithmetic guess. Selection on noisy estimates
is precisely what killed F2 (§8bh: the screen's error equalled the effect's sd,
so the max selected for measurement error). So measure the **realized** gain of
a budgeted oracle, not the ceiling.

    Q4(R_sel) = E_i[ δ_{i, j*} ]   j* = argmax of the SELECTION-split estimate
                                   δ  = the true gap, scored on a DISJOINT split

The disjoint split is the whole design: it is what makes the estimate unbiased
under selection noise, and the identical-arms control below is what proves it.

Controls, any of which voids the run:
  **C0**  the recorded action answers the observation it is paired with.
  **C1**  the forked position is bitwise the replay's.
  🔴 **C-identical** — the entire Q4 procedure with all three arms set to the
        SAME option. It must read **0**. A positive reading means the estimator
        is selecting on noise and scoring on that same noise, and every number
        in the run is void. This is the winner's-curse control.
  **scale bar** §8bw's clone top-vs-last = +0.120. Read every Δ against it.

⚠ Values are win probability under **clone-vs-clone continuation** (§8bw), and
per-decision gains **do not add** across a game. Neither this script nor any
other in the repo can turn Q4 into a win rate; only the A/B can, and the A/B is
§2.7's large-or-nothing blocker.

    # collect (shardable -- fastsearch has process-global state, so shard by
    # PROCESS, never by thread)
    python -X utf8 scripts/p82_e17_self_oracle.py --collect --positions 250 \
        --pairs 40 --shard 0 --shards 6
    python -X utf8 scripts/p82_e17_self_oracle.py --collect --control ...
    # analyse
    python -X utf8 scripts/p82_e17_self_oracle.py --analyze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from sa import policynet as pnet  # noqa: E402
from sa.worlds import determinize  # noqa: E402
from p80_rollout_feasibility import (  # noqa: E402
    c0_alignment, c1_fidelity, load_games, our_seat, positions, rollout,
    seat_decklist,
)

OUT = ROOT / "out/logs"
SCALE_BAR = 0.120          # §8bw, the clone's own top vs last
R_SEL_GRID = (5, 10, 20, 30, 40)
R_SEL_MODEL = (30, 60, 130, 200)
MIN_EVAL = 8               # a split with fewer evaluation replicates is dropped


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

def replay_files(dump: Path, games: int) -> list[Path]:
    return sorted(p for p in dump.glob("*.json")
                  if p.name != "episodes_meta.json")[:games]


def index_positions(dump: Path, games: int, net) -> list[dict]:
    """Pass 1 — enumerate our live MAIN decisions WITHOUT keeping any of them.

    🔴 This streams one replay at a time and drops it. The first version used
    `p80.load_games`, which parses all 76 files into one list: 385 MB of JSON
    becomes several GB of Python objects, and ten shards on a 7.9 GB box
    starved each other. **The failure was silent** — `load_games` catches every
    per-file exception and `net.scores` was wrapped in a bare `except`, so a
    memory-starved shard reported "0 positions" and a half-starved one dropped
    7% of the population without a word (3,562 vs 3,311 candidates across
    processes that should have been identical). Rule 18's shape, in the harness
    rather than the engine.
    """
    out: list[dict] = []
    skipped = {"parse": 0, "no_seat": 0, "no_deck": 0, "net_error": 0}
    first_err = None
    for f in replay_files(dump, games):
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:                 # loud, not silent
            skipped["parse"] += 1
            first_err = first_err or f"parse: {type(e).__name__}: {e}"
            continue
        seat = our_seat(rep)
        if seat is None:
            skipped["no_seat"] += 1
            del rep
            continue
        deck = seat_decklist(rep, seat)
        if deck is None:
            skipped["no_deck"] += 1
            del rep
            continue
        for i, o in positions(rep, seat):
            try:
                sc = np.asarray(net.scores(o), dtype=float)
            except Exception as e:
                skipped["net_error"] += 1
                first_err = first_err or f"scores: {type(e).__name__}: {e}"
                continue
            order = list(np.argsort(-sc))
            out.append({
                "gid": f.stem, "step": i, "seat": seat, "deck": deck,
                "turn": int(o["current"]["turn"]),
                "nopt": int(len(o["select"]["option"])),
                "order": [int(x) for x in order],
                "scores": [float(sc[int(x)]) for x in order[:4]],
            })
        del rep
    print(f"  positions: {len(out)}   skipped: {skipped}")
    if first_err:
        print(f"  ⚠ first skip reason: {first_err}")
    if sum(skipped.values()) > 0.02 * (len(out) + sum(skipped.values())):
        print("  🔴 more than 2% of candidates were skipped — this shard is "
              "NOT measuring the same population as its peers. Do not merge it.")
        return []
    return out


def attach_obs(sample: list[dict], dump: Path) -> list[dict]:
    """Pass 2 — re-read only the files the sample actually needs."""
    want: dict[str, list[dict]] = {}
    for r in sample:
        want.setdefault(r["gid"], []).append(r)
    out = []
    for f in replay_files(dump, 10 ** 9):
        rows = want.get(f.stem)
        if not rows:
            continue
        rep = json.loads(f.read_text(encoding="utf-8"))
        steps = rep.get("steps") or []
        for r in rows:
            o = steps[r["step"]][r["seat"]].get("observation")
            if o:
                out.append(r | {"obs": o})
        del rep
    return out


def collect(args: argparse.Namespace, net, fp: str) -> int:
    dump = ROOT / args.dump
    pos = index_positions(dump, args.games, net)
    if not pos:
        print("🔴 no positions")
        return 1

    rng = random.Random(20260810)
    sample = rng.sample(pos, min(args.positions, len(pos)))
    sample = attach_obs(sample[args.shard::args.shards], dump)
    tag = "ctl" if args.control else "trt"
    print(f"[{tag} shard {args.shard}/{args.shards}] {len(sample)} positions "
          f"× {args.arms} arms × {args.pairs} replicates")

    if args.shard == 0 and not args.control:
        c1 = [(r["step"], r["obs"], r["seat"], r["deck"]) for r in sample[:40]]
        if not c1_fidelity(c1, min(40, len(c1))):
            return 1

    recs: list[dict] = []
    t0 = time.perf_counter()
    for pi, r in enumerate(sample):
        order = r["order"]
        if args.control:
            # 🔴 the winner's-curse control: three arms, ONE option. The engine
            # is not reproducible on a fixed world (§8bw C2), so these are three
            # independent draws of the same quantity -- exactly the null the
            # selection procedure must return zero on.
            arms = [order[0]] * args.arms
        else:
            arms = order[:args.arms]
        vals: list[list[float]] = [[] for _ in arms]
        kept = 0
        for k in range(args.pairs):
            seed = 7_000_003 * (args.shard + 1) * (pi + 1) + k
            row = []
            for a in arms:
                w = determinize(r["obs"], r["deck"], [], random.Random(seed))
                v, _s, _c = rollout(r["obs"], w, [a], r["seat"], net=net)
                row.append(v)
            if any(v is None for v in row):
                continue           # drop the replicate for ALL arms: the shared
            for j, v in enumerate(row):   # world is the pairing, and a partial
                vals[j].append(float(v))  # replicate breaks it
            kept += 1
        if kept >= max(8, args.pairs // 3):
            recs.append({k: r[k] for k in
                         ("gid", "step", "turn", "nopt", "scores")} |
                        {"arms": [int(a) for a in arms], "vals": vals})
        if pi and pi % 10 == 0:
            el = time.perf_counter() - t0
            print(f"    {pi}/{len(sample)}  {el:.0f}s  "
                  f"({el / pi:.1f}s/position)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"p82_e17_{tag}_{args.shard}.json"
    p.write_text(json.dumps({
        "net": args.net, "fp": fp,
        "control": bool(args.control), "arms": args.arms, "pairs": args.pairs,
        "dump": args.dump, "records": recs,
    }), encoding="utf-8")
    print(f"  ⇒ {len(recs)} usable positions → {p.name}  "
          f"({time.perf_counter() - t0:.0f}s wall)")
    return 0


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def ci(xs: list[float], label: str, note: str = "") -> tuple[float, float, float]:
    """Clustered on the POSITION. The pair-level interval is 4.1× too narrow
    (§8bw) and the pre-registration forbids quoting it."""
    k = len(xs)
    if k < 3:
        print(f"  {label}: too few positions ({k})")
        return float("nan"), float("nan"), float("nan")
    m = statistics.fmean(xs)
    se = statistics.stdev(xs) / math.sqrt(k)
    lo, hi = m - 1.96 * se, m + 1.96 * se
    print(f"  {label:<38s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  k={k}{note}")
    return m, lo, hi


def gaps(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """-> (R×A matrix of per-replicate values, gap matrix vs arm 0)."""
    v = np.asarray(rec["vals"], dtype=float).T      # R × A
    return v, v[:, 1:] - v[:, :1]


def budgeted_gain(recs: list[dict], r_sel: int, splits: int,
                  rng: random.Random, pool: str = "all") -> list[float]:
    """Q4: select argmax on `r_sel` replicates, score on the DISJOINT rest.

    Returns one number per position (the cluster unit). `pool="second"`
    restricts the whole procedure to the back half of the replicates, leaving
    the front half free to estimate the position's win probability
    *independently* — §8bw: selecting positions on the same rollouts that
    measure the effect buys a regression-to-the-mean bias.
    """
    out, fire, keep = [], [], []
    for ri, rec in enumerate(recs):
        v, _ = gaps(rec)
        if pool == "second":
            v = v[v.shape[0] // 2:]
        R, A = v.shape
        if R - r_sel < MIN_EVAL:
            continue
        acc, dev = [], 0
        idx = list(range(R))
        for _ in range(splits):
            rng.shuffle(idx)
            s, e = idx[:r_sel], idx[r_sel:]
            d_sel = v[s].mean(axis=0) - v[s, 0].mean()      # arm 0 -> exactly 0
            j = int(np.argmax(d_sel))
            dev += j != 0
            acc.append(float(v[e, j].mean() - v[e, 0].mean()))
        if acc:
            out.append(statistics.fmean(acc))
            fire.append(dev / len(acc))
            keep.append(ri)
    budgeted_gain.last_fire = statistics.fmean(fire) if fire else float("nan")
    # ⚠ positions with too few evaluation replicates are DROPPED, so out[j] is
    # not recs[j]. p81 shipped that misalignment once; carry the index list.
    budgeted_gain.last_keep = keep
    return out


def gain_tau(recs: list[dict], r_sel: int, tau: float, splits: int,
             rng: random.Random) -> tuple[list[float], float]:
    """Q4 with a minimum estimated gap before the oracle is allowed to overrule.

    ⚠ **POST-HOC.** τ was not pre-registered and six values were swept, which is
    exactly the shopping day 18 declined a B8 β-sweep for. Reported as
    exploratory, and it needs its own pre-registered confirmation before any
    build leans on it. What makes it more than a lucky pick is that the
    *treatment* is nearly flat in τ while the *control* collapses — the shape a
    real bias concentrated in marginal calls would make, not the shape of noise.
    """
    out, fire = [], []
    for rec in recs:
        v, _ = gaps(rec)
        R, A = v.shape
        if R - r_sel < MIN_EVAL:
            continue
        idx = list(range(R))
        acc, dev = [], 0
        for _ in range(splits):
            rng.shuffle(idx)
            s, e = idx[:r_sel], idx[r_sel:]
            d = v[s].mean(axis=0) - v[s, 0].mean()
            d[0] = 0.0
            j = int(np.argmax(d))
            if d[j] <= tau:
                j = 0                      # not earned -> keep the net's pick
            dev += j != 0
            acc.append(float(v[e, j].mean() - v[e, 0].mean()))
        if acc:
            out.append(statistics.fmean(acc))
            fire.append(dev / len(acc))
    return out, (statistics.fmean(fire) if fire else float("nan"))


def corrected(trt_g: list[float], ctl_g: list[float]) -> tuple[float, float, float]:
    """Treatment minus the identical-arms control, with the interval combined.

    The control is the estimator's reading when there is nothing to find, so it
    is the right null to subtract — and at τ=0 it is a third of the treatment.
    """
    mt = statistics.fmean(trt_g)
    st = statistics.stdev(trt_g) / math.sqrt(len(trt_g))
    mc = statistics.fmean(ctl_g)
    sc = statistics.stdev(ctl_g) / math.sqrt(len(ctl_g))
    se = math.sqrt(st ** 2 + sc ** 2)
    return mt - mc, mt - mc - 1.96 * se, mt - mc + 1.96 * se


def order_effect(ctl: list[dict]) -> None:
    """Is arm 0 -- always rolled FIRST, and the baseline in every estimator --
    exchangeable with the others? Under the control all three arms are the same
    option, so any difference is an artifact of call order.

    ⚠ The obvious placebo (arm2 − arm1) does NOT discriminate: it reads zero
    under 'no effect' AND under 'arm 0 is low'. Only more positions resolve it.
    """
    pool, placebo = [], []
    for r in ctl:
        v = np.asarray(r["vals"], dtype=float)
        pool.append(float((v[1].mean() + v[2].mean()) / 2 - v[0].mean()))
        placebo.append(float(v[2].mean() - v[1].mean()))
    ci(pool, "mean(arm1,arm2) − arm0  [order effect]")
    ci(placebo, "arm2 − arm1  [placebo, uninformative]")


def deconvolve(recs: list[dict], arm: int) -> tuple[float, float, float]:
    """-> (mean gap, observed between-position sd, TRUE sd) for one arm."""
    ds, noise = [], []
    for rec in recs:
        v, g = gaps(rec)
        if g.shape[1] < arm:
            continue
        col = g[:, arm - 1]
        ds.append(float(col.mean()))
        if len(col) > 1:
            noise.append(float(col.std(ddof=1)) / math.sqrt(len(col)))
    if len(ds) < 3:
        return float("nan"), float("nan"), float("nan")
    obs = statistics.stdev(ds)
    meas = statistics.fmean(noise)
    true = math.sqrt(max(0.0, obs ** 2 - meas ** 2))
    return statistics.fmean(ds), obs, true


def model_curve(recs: list[dict], r_grid, trials: int = 20000):
    """Q5 — MODEL-BASED extension of Q4 beyond the empirical R.

    ⚠ NOT a measurement. Fits a multivariate normal to the deconvolved gap
    distribution and to the per-replicate noise, then simulates selection at
    budgets the R=40 collection cannot reach empirically. Quoted as a shape,
    never as a number that ships.
    """
    mus, sds, noises = [], [], []
    A = min(len(r["vals"]) for r in recs)
    for arm in range(1, A):
        m, _o, t = deconvolve(recs, arm)
        mus.append(m)
        sds.append(t)
    # per-replicate paired sd of a gap, pooled
    per = []
    for rec in recs:
        _v, g = gaps(rec)
        for c in range(g.shape[1]):
            if g.shape[0] > 1:
                per.append(float(g[:, c].std(ddof=1)))
    s_pair = statistics.fmean(per) if per else float("nan")

    rng = np.random.default_rng(20260810)
    out = {}
    # ⚠ arms are drawn INDEPENDENTLY, which overstates E[max] whenever the
    # per-position gaps are positively correlated across arms. That biases the
    # ceiling UP — i.e. against the KILL branch, which is the conservative
    # direction for a gate whose default is to build.
    delta = np.stack([rng.normal(mus[c], max(sds[c], 1e-9), trials)
                      for c in range(len(mus))], axis=1)   # trials × (A-1)
    ceiling = float(np.maximum(0.0, delta.max(axis=1)).mean())
    for R in r_grid:
        # the estimate of every gap shares arm-0's rollout noise, so the errors
        # are correlated ~0.5 across arms; generating them independently would
        # make selection look easier than it is.
        sv = s_pair / math.sqrt(2.0)
        e0 = rng.normal(0.0, sv / math.sqrt(R), (trials, 1))
        ej = rng.normal(0.0, sv / math.sqrt(R), delta.shape)
        est = delta + (ej - e0)
        j = est.argmax(axis=1)
        pick = np.where(est.max(axis=1) > 0.0,
                        delta[np.arange(trials), j], 0.0)
        out[R] = float(pick.mean())
    return out, ceiling, s_pair, mus, sds


def strat(recs: list[dict], gains: list[float], key, name: str,
          cuts: list[float]) -> None:
    """Q6 — is the gain predictable from something FREE at play time?"""
    vals = [key(r) for r in recs]
    if len(vals) != len(gains):
        print(f"  ⚠ {name}: length mismatch, skipped")
        return
    print(f"  by {name}:")
    edges = [-math.inf] + cuts + [math.inf]
    for a, b in zip(edges, edges[1:]):
        sel = [g for g, x in zip(gains, vals) if a <= x < b]
        if len(sel) < 10:
            continue
        m = statistics.fmean(sel)
        se = statistics.stdev(sel) / math.sqrt(len(sel))
        lab = f"[{a:g}, {b:g})".replace("-inf", "−∞").replace("inf", "∞")
        print(f"    {lab:>16s}  {m:+.4f} ±{1.96 * se:.4f}  "
              f"k={len(sel):3d} ({len(sel) / len(gains):.0%})")


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def load_shards(tag: str) -> list[dict]:
    recs, meta = [], None
    for p in sorted(OUT.glob(f"p82_e17_{tag}_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        meta = meta or d
        recs += d["records"]
    if meta:
        print(f"[{tag}] {len(recs)} positions from "
              f"{len(list(OUT.glob(f'p82_e17_{tag}_*.json')))} shards, "
              f"net #{meta['fp']}, {meta['arms']} arms × {meta['pairs']} reps")
    return recs


def analyze(args: argparse.Namespace) -> int:
    rng = random.Random(4242)
    ctl = load_shards("ctl")
    trt = load_shards("trt")
    if not trt:
        print("🔴 no treatment shards")
        return 1

    # ---- C-identical, FIRST: a positive reading voids everything ---------
    print("\n🔴 [C-identical] the winner's-curse control — the whole Q4 "
          "procedure with three copies of ONE option. Must read 0.")
    void = False
    if not ctl:
        print("  ⚠ NOT RUN. The treatment below is unvalidated.")
        void = True
    else:
        ran = 0
        for R in R_SEL_GRID:
            g = budgeted_gain(ctl, R, args.splits, rng)
            m, lo, hi = ci(g, f"control gain @ R_sel={R:>2d}",
                           f"  deviates from top-1 "
                           f"{budgeted_gain.last_fire:.0%} of the time")
            if math.isnan(lo):
                continue          # too few positions at this budget, not a fail
            ran += 1
            if not (lo <= 0 <= hi):
                print("     🔴 IDENTICAL ARMS DO NOT READ ZERO — the estimator "
                      "is selecting on noise and scoring the same noise. "
                      "Every treatment number below is VOID.")
                void = True
        if ran == 0:
            print("  ⚠ the control produced no usable budget — treatment "
                  "unvalidated.")
            void = True
        if not void:
            print("  ✅ identical arms read zero at every budget — the "
                  "split-sample estimator is unbiased under selection noise.")

    # ---- Q1 / Q2 ---------------------------------------------------------
    print("\n[Q1] is the net's own ranking right on average?")
    res: dict = {}
    for arm, nm in ((1, "top-2 − top-1"), (2, "top-3 − top-1")):
        per = []
        for rec in trt:
            _v, g = gaps(rec)
            if g.shape[1] >= arm:
                per.append(float(g[:, arm - 1].mean()))
        m, lo, hi = ci(per, f"Δ({nm})")
        res[f"q1_arm{arm}"] = [m, lo, hi]
    print(f"  scale bar (§8bw clone top vs last): {SCALE_BAR:+.3f}")

    print("\n[Q2] dispersion — the raw material of any oracle")
    for arm in (1, 2):
        m, obs, true = deconvolve(trt, arm)
        print(f"  arm top-{arm + 1}: mean {m:+.4f}  observed sd {obs:.4f}  "
              f"measurement noise {math.sqrt(max(0, obs**2 - true**2)):.4f}  "
              f"⇒ TRUE sd {true:.4f}  (typical |gap| {0.798 * true:.4f})")
        res[f"q2_arm{arm}"] = {"mean": m, "obs_sd": obs, "true_sd": true}
    print(f"  §8bx measured 0.0768 / 0.0866 for OUR-pick vs EXPERT-pick — "
          f"the comparison this experiment exists to make")

    # ---- Q4, the headline ------------------------------------------------
    print("\n⭐ [Q4] REALIZED gain of a BUDGETED oracle over the net's top-3 "
          "(split-sample, disjoint selection/evaluation)")
    gains30: list[float] = []
    recs30: list[dict] = []
    for R in R_SEL_GRID:
        g = budgeted_gain(trt, R, args.splits, rng)
        note = (f"  = {statistics.fmean(g) / SCALE_BAR:>4.0%} of the scale bar,"
                f" fires {budgeted_gain.last_fire:.0%}"
                if len(g) >= 3 else "")
        m, lo, hi = ci(g, f"Q4 @ R_sel={R:>2d} pairs/arm", note)
        res[f"q4_{R}"] = [m, lo, hi, len(g)]
        if R == 30:
            gains30 = g
            recs30 = [trt[i] for i in budgeted_gain.last_keep]

    # ---- Q3 / Q5 ---------------------------------------------------------
    print("\n[Q3/Q5] ceiling and the model-based budget curve  "
          "⚠ MODEL-BASED — not measurements")
    curve, ceiling, s_pair, mus, sds = model_curve(trt, R_SEL_MODEL)
    print(f"  per-replicate paired sd of a gap: {s_pair:.3f} "
          f"(§8bw measured 0.43 on the top-vs-last contrast)")
    print(f"  Q3 perfect-oracle ceiling E[max(0, δ₂, δ₃)] = {ceiling:+.4f}"
          f"  ({ceiling / SCALE_BAR:.0%} of the scale bar)")
    for R, v in curve.items():
        print(f"    modelled Q4 @ R_sel={R:>3d}: {v:+.4f}"
              f"   ({v / ceiling:.0%} of the ceiling)" if ceiling > 0 else "")
    res["q3_ceiling"] = ceiling
    res["q5_curve"] = curve

    # ---- Q6 --------------------------------------------------------------
    if gains30:
        print("\n[Q6] is the gain predictable from anything FREE at play time? "
              "(exploratory — cannot produce a design on its own)")
        strat(recs30, gains30,
              lambda r: r["scores"][0] - r["scores"][1], "score margin",
              [0.5, 1.5, 3.0])
        strat(recs30, gains30, lambda r: r["turn"], "turn", [4, 7, 11])
        strat(recs30, gains30, lambda r: r["nopt"], "option count", [4, 6, 9])
        # ⭐ the one stratification that changes the DESIGN, not just the story:
        # a budget manager should spend only where the game is still live.
        # WP from the FRONT half of the replicates, gain from the BACK half —
        # disjoint, so the selection cannot bias the effect (§8bw M3).
        r_sub = 15
        gsub = budgeted_gain(trt, r_sub, args.splits, rng, pool="second")
        kept = [trt[i] for i in budgeted_gain.last_keep]
        wp = []
        for r in kept:
            arr = np.asarray(r["vals"], dtype=float)      # A × R
            wp.append(float(arr[0, :arr.shape[1] // 2].mean()))
        print(f"  ⭐ by position win probability — WP from the FRONT half of "
              f"the replicates, Q4@{r_sub} from the BACK half (disjoint):")
        if len(gsub) == len(wp):
            res["q6_wp"] = {}
            for a, b in ((0.0, 0.15), (0.15, 0.5), (0.5, 0.85), (0.85, 1.01)):
                sel = [g for g, w in zip(gsub, wp) if a <= w < b]
                if len(sel) >= 10:
                    se = statistics.stdev(sel) / math.sqrt(len(sel))
                    print(f"    WP [{a:.2f}, {b:.2f})  "
                          f"{statistics.fmean(sel):+.4f} ±{1.96 * se:.4f}  "
                          f"k={len(sel)} ({len(sel) / len(gsub):.0%})")
                    res["q6_wp"][f"{a}-{b}"] = [statistics.fmean(sel),
                                                1.96 * se, len(sel)]
            live = [g for g, w in zip(gsub, wp) if 0.15 <= w <= 0.85]
            if len(live) >= 10:
                se = statistics.stdev(live) / math.sqrt(len(live))
                print(f"    ⇒ COMPETITIVE ONLY [0.15, 0.85]: "
                      f"{statistics.fmean(live):+.4f} ±{1.96 * se:.4f}  "
                      f"k={len(live)} ({len(live) / len(gsub):.0%} of decisions)"
                      f"  ⚠ at R_sel={r_sub}, half the primary's budget")
                res["q6_competitive"] = [statistics.fmean(live), 1.96 * se,
                                         len(live), len(gsub)]
        else:
            print(f"    ⚠ length mismatch {len(gsub)} vs {len(wp)}, skipped")

    # ---- the control as a NULL to subtract, and the τ sweep --------------
    if ctl and not void:
        print("\n[bias] is arm 0 exchangeable? (control only — all arms are "
              "the SAME option, so any gap is call order, not move quality)")
        order_effect(ctl)
        print(f"\n[Q4 corrected + τ sweep at R_sel=30]  ⚠ τ is POST-HOC — six "
              f"values swept, not pre-registered")
        print(f"  {'τ':>5} {'fires':>6} | {'treatment':>26} | "
              f"{'control':>18} | {'corrected':>26}")
        res["tau"] = {}
        for tau in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20):
            gt, ft = gain_tau(trt, 30, tau, args.splits, random.Random(7))
            gc, _f = gain_tau(ctl, 30, tau, args.splits, random.Random(7))
            if len(gt) < 3 or len(gc) < 3:
                continue
            mt = statistics.fmean(gt)
            st = 1.96 * statistics.stdev(gt) / math.sqrt(len(gt))
            mc = statistics.fmean(gc)
            sc = 1.96 * statistics.stdev(gc) / math.sqrt(len(gc))
            cm, clo, chi = corrected(gt, gc)
            print(f"  {tau:>5.2f} {ft:>5.0%} | {mt:+.4f} [{mt-st:+.4f}, "
                  f"{mt+st:+.4f}] | {mc:+.4f} ±{sc:.4f} | "
                  f"{cm:+.4f} [{clo:+.4f}, {chi:+.4f}]")
            res["tau"][f"{tau}"] = [cm, clo, chi, ft]

    # ---- the NARROW branch's test, with the control subtracted -----------
    # E17 §6: build only if a FREE, online-computable trigger selects ≥25% of
    # decisions at ≥ +0.030. The Q6 table above is uncorrected, and the control
    # is a third of the effect at τ=0, so the trigger must be judged corrected.
    if ctl and not void and gains30:
        print("\n⭐ [NARROW test] candidate triggers, CONTROL-CORRECTED "
              "(R_sel=30, τ=0). Gate: ≥25% of decisions at ≥ +0.030")
        gt_all = budgeted_gain(trt, 30, args.splits, random.Random(11))
        rt = [trt[i] for i in budgeted_gain.last_keep]
        gc_all = budgeted_gain(ctl, 30, args.splits, random.Random(11))
        rc = [ctl[i] for i in budgeted_gain.last_keep]
        res["narrow"] = {}
        for name, fn in (("option count ≤ 5", lambda r: r["nopt"] <= 5),
                         ("option count ≤ 3", lambda r: r["nopt"] <= 3),
                         ("score margin < 1.5",
                          lambda r: r["scores"][0] - r["scores"][1] < 1.5),
                         ("turn ≥ 11", lambda r: r["turn"] >= 11)):
            gt = [g for g, r in zip(gt_all, rt) if fn(r)]
            gc = [g for g, r in zip(gc_all, rc) if fn(r)]
            if len(gt) < 10 or len(gc) < 10:
                print(f"  {name:<20s} too few positions "
                      f"(trt {len(gt)}, ctl {len(gc)})")
                continue
            cm, clo, chi = corrected(gt, gc)
            share = len(gt) / len(gt_all)
            ok = "✅" if (share >= 0.25 and clo > 0 and cm >= 0.030) else "❌"
            print(f"  {ok} {name:<20s} {cm:+.4f} [{clo:+.4f}, {chi:+.4f}]  "
                  f"{share:.0%} of decisions  (trt k={len(gt)}, ctl k={len(gc)})")
            res["narrow"][name] = [cm, clo, chi, share]

    # ---- the pre-registered verdict --------------------------------------
    print("\n" + "=" * 72)
    q4_30, lo30, hi30 = res.get("q4_30", [float("nan")] * 3)[:3]
    print(f"PRE-REGISTERED DECISION RULE (E17 §6, frozen at c0a2cc9)")
    print(f"  primary   Q4 @ R_sel=30 = {q4_30:+.4f} [{lo30:+.4f}, {hi30:+.4f}]")
    print(f"  secondary Q3 ceiling    = {ceiling:+.4f}")
    if void:
        print("  ⇒ 🔴 VOID — the identical-arms control did not pass.")
    elif ceiling <= 0.015 or (q4_30 <= 0.010 and lo30 <= 0 <= hi30):
        print("  ⇒ 🔴 KILL. The clock is closed for Round 1: the value E16 "
              "found at expert-disagreement positions does not exist among "
              "our own options at a payable budget.")
    elif q4_30 >= 0.020 and lo30 > 0:
        print("  ⇒ ✅ BUILD, in order: batched rollouts + process parallelism, "
              "the online budget manager, the agent, the A/B.")
    else:
        print("  ⇒ 🟡 NARROW. Build only if Q6 shows a free online trigger "
              "selecting ≥25% of decisions at ≥ +0.030; else it is a chapter.")
    print("=" * 72)
    print("⚠ Per-decision WP gains DO NOT ADD across a game, and the value is "
          "win probability under clone-vs-clone continuation. Neither this "
          "script nor any other here converts Q4 into a win rate.")

    (OUT / "p82_e17_summary.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="out/policy_v5_s2.npz",
                    help="the SHIPPED 55326513 weights (#4790c469). Pinned "
                         "here rather than taken from SA_PNET_PATH.")
    ap.add_argument("--dump", default="replays/submission_v5_s2")
    ap.add_argument("--games", type=int, default=76)
    ap.add_argument("--positions", type=int, default=250)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--arms", type=int, default=3)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--splits", type=int, default=200,
                    help="random selection/evaluation splits per position")
    ap.add_argument("--control", action="store_true",
                    help="identical-arms winner's-curse control")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--c0", action="store_true", help="run C0 and exit")
    args = ap.parse_args()

    if args.analyze:
        return analyze(args)

    # 🔴 PIN THE NET IN THE SCRIPT, not in the environment. `pnet.get()` returns
    # whatever `SA_PNET_PATH` happens to say, and the repo default
    # (`agents/sa/policy_net.npz`, #a25b904d) is the *v2* clone — three
    # generations behind the agent that played `submission_v5_s2`. Scoring one
    # net's options against another net's recorded games returns a plausible
    # number, not an error: C0 read **67.3%** on the default and 99.8% here, and
    # that gap is the only thing that separated a valid run from a silent one.
    npz = ROOT / args.net
    if not npz.exists():
        print(f"🔴 net not found: {npz}")
        return 1
    net = pnet.load(str(npz))
    if net is None:
        print(f"🔴 could not load {npz}")
        return 1
    fp = hashlib.md5(npz.read_bytes()).hexdigest()[:8]
    print(f"net: {args.net}  #{fp}")
    if fp != "4790c469":
        print(f"  ⚠ this is NOT the shipped 55326513 net (#4790c469) that "
              f"§8bw/§8bx pinned. State it wherever the numbers are used.")

    if args.c0:
        return 0 if c0_alignment(
            load_games(ROOT / args.dump, 15), net) else 1
    if args.collect:
        return collect(args, net, fp)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
