"""E16 — is the expert's move actually better than ours? Paired-rollout move value.

Pre-registered in `docs/experiments/E16-counterfactual-move-value.md`. Feasibility
and every control for the underlying instrument are in `p80_rollout_feasibility.py`
(`EVIDENCE` §8bw); this script only runs the experiment.

**The question.** §8u cloned the #2 player *successfully* (agreement 59.9% →
67.2%) and measured **−92 Elo**, with covariate shift ruled out (§8s). §8r then
showed why nothing here can explain that: every rate-vs-experts metric is a
**conformity** metric. So ask the counterfactual directly — fork the expert's
real position, force **their** move in one arm and **our net's** move in the
other, let the clone play both seats to a terminal state, and difference.

    Δ = E[ value(expert's move) − value(our net's move) ]   from the expert's seat

⚠ **Δ is win probability under clone-vs-clone continuation** — the value of a
one-step deviation from our own policy. That is the right question for "should
our net have played their move" and the wrong one for "is this move good in the
abstract". The distinction is the whole point of arm C.

Controls, all of which can void the run:
  **C0** the recorded action must answer the observation it is paired with
        (`seat_action` is off by one; the same-step read is 20.1% right).
  **C1** the forked position must be bitwise the replay's position.
  **agreement control** where the expert's pick and our net's pick are the SAME
        action, the two arms are identical, so Δ must read 0. This is free and
        it is the strongest check on the whole pipeline.
  **scale bar** §8bw's top-vs-last = **+0.120**. Read every Δ against it.

⛔ Not a training target — demonstrator selection is closed twice (§8t, §8u).

    python -X utf8 scripts/p81_e16_move_value.py --positions 600 --pairs 30
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
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
    seat_action, seat_decklist,
)

OURS = "Scio"


# --------------------------------------------------------------------------

def lb_ratings(path: Path) -> dict[str, float]:
    """TeamName -> Score from the leaderboard CSV zip."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    zf = zipfile.ZipFile(path)
    name = zf.namelist()[0]
    for row in csv.DictReader(io.TextIOWrapper(zf.open(name), encoding="utf-8")):
        team = row.get("TeamName") or row.get("﻿TeamName")
        try:
            out[team] = float(row["Score"])
        except Exception:
            continue
    return out


def aliases(path: Path) -> dict[str, str]:
    """Verified renames (rule 17: a display name is not a key)."""
    out: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and not line.startswith("#"):
                out[parts[0]] = parts[1]
    return out


def collect(dump: Path, ratings: dict[str, float], alias: dict[str, str],
            min_rating: float, net, limit: int) -> tuple[list, list, Counter]:
    """-> (disagreements, agreements, diagnostics).

    Each item: (obs, seat, deck, expert_pick, net_pick, turn, rating, gid)
    """
    dis, agr = [], []
    diag: Counter = Counter()
    for f, rep in load_games(dump, limit):
        names = (rep.get("info") or {}).get("TeamNames") or []
        for seat in range(len(names)):
            nm = alias.get(names[seat], names[seat])
            if nm == OURS:
                diag["skip_us"] += 1          # rule 18: exclude ourselves
                continue
            rating = ratings.get(nm)
            if rating is None:
                diag["skip_unrated"] += 1
                continue
            if rating < min_rating:
                diag["skip_below_cut"] += 1
                continue
            deck = seat_decklist(rep, seat)
            if deck is None:
                diag["skip_no_deck"] += 1
                continue
            for i, o in positions(rep, seat):
                a = seat_action(rep, seat, i)
                if not (a and len(a) == 1):
                    diag["skip_multi_or_missing"] += 1
                    continue
                nopt = len(o["select"]["option"])
                if not 0 <= a[0] < nopt:
                    diag["skip_out_of_range"] += 1
                    continue
                try:
                    sc = net.scores(o)
                    pick = int(np.argmax(sc))
                except Exception:
                    diag["skip_net_error"] += 1
                    continue
                item = (o, seat, deck, a[0], pick, o["current"]["turn"],
                        rating, f.stem)
                if a[0] == pick:
                    agr.append(item)
                    diag["agree"] += 1
                else:
                    dis.append(item)
                    diag["disagree"] += 1
    return dis, agr, diag


def measure(items: list, pairs: int, cont_net, label: str,
            seed0: int = 0):
    """Paired rollouts at each position.

    -> (per-position mean Δ, pairs used, seconds, index into `items`)
    ⚠ The index list matters: positions with too few usable pairs are dropped,
    so `per_pos[j]` is NOT `items[j]`. Zipping them positionally is a silent
    misalignment -- it was one here until this returned `idxs`.
    """
    per_pos: list[float] = []
    ns: list[int] = []
    idxs: list[int] = []
    within: list[float] = []
    t0 = time.perf_counter()
    for pi, (o, seat, deck, a_pick, b_pick, _t, _r, _g) in enumerate(items):
        d = []
        for k in range(pairs):
            seed = 1_000_003 * (seed0 + pi) + k
            wa = determinize(o, deck, [], random.Random(seed))
            va, _s, _c = rollout(o, wa, [a_pick], seat, net=cont_net)
            wb = determinize(o, deck, [], random.Random(seed))
            vb, _s, _c = rollout(o, wb, [b_pick], seat, net=cont_net)
            if va is None or vb is None:
                continue
            d.append(va - vb)
        if len(d) >= max(5, pairs // 3):
            per_pos.append(statistics.fmean(d))
            ns.append(len(d))
            idxs.append(pi)
            within.append(statistics.stdev(d) if len(d) > 1 else float("nan"))
        if pi and pi % 50 == 0:
            print(f"    [{label}] {pi}/{len(items)} positions, "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    measure.last_within = within
    return per_pos, ns, time.perf_counter() - t0, idxs


def report(per_pos: list[float], ns: list[int], label: str) -> tuple[float, float, float]:
    """Clustered on the POSITION -- the pair-level interval is 4.1x too narrow
    (§8bw), and quoting it is forbidden by the pre-registration."""
    k = len(per_pos)
    if k < 2:
        print(f"  {label}: too few positions ({k})")
        return float("nan"), float("nan"), float("nan")
    m = statistics.fmean(per_pos)
    se = statistics.stdev(per_pos) / (k ** 0.5)
    lo, hi = m - 1.96 * se, m + 1.96 * se
    print(f"  {label}: Δ = {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
          f"k={k} positions × {statistics.fmean(ns):.0f} pairs "
          f"(clustered on position)")
    return m, lo, hi


def run(args: argparse.Namespace) -> int:
    import hashlib
    net = pnet.get()
    if net is None:
        print("🔴 no policy net")
        return 1
    npz = Path(pnet._PATH)
    print(f"net: {npz}  #{hashlib.md5(npz.read_bytes()).hexdigest()[:8]}")

    ratings = lb_ratings(ROOT / args.lb)
    alias = aliases(ROOT / "replays/team_aliases.tsv")
    print(f"leaderboard rows: {len(ratings)}   aliases: {len(alias)}")

    # ---- C0 / C1 on the population actually used ------------------------
    ours = load_games(ROOT / "replays/submission_v5_s2", 15)
    if not c0_alignment(ours, net):
        return 1

    dump = ROOT / args.dump
    dis, agr, diag = collect(dump, ratings, alias, args.min_rating, net,
                             args.games)
    print(f"\npopulation: {args.dump}, rating ≥ {args.min_rating}")
    for k, v in sorted(diag.items()):
        print(f"  {k:22s} {v}")
    tot = diag["agree"] + diag["disagree"]
    if tot:
        print(f"  ⇒ our net's top-1 differs from the expert on "
              f"{diag['disagree'] / tot:.1%} of their MAIN decisions")
    if not dis:
        print("🔴 no disagreement positions")
        return 1

    rng = random.Random(20260809)
    dis_s = rng.sample(dis, min(args.positions, len(dis)))
    agr_s = rng.sample(agr, min(args.agree_positions, len(agr))) if agr else []

    c1_pos = [(0, o, seat, deck) for (o, seat, deck, *_r) in dis_s[:40]]
    if not c1_fidelity(c1_pos, min(40, len(c1_pos))):
        return 1

    # ---- the agreement control (identical arms -> must read 0) ----------
    print(f"\n[agreement control] identical arms at {len(agr_s)} positions "
          f"— must read 0")
    ap, an, asec, _ai = measure(agr_s, args.pairs, net, "agree",
                                seed0=500_000)
    am, alo, ahi = report(ap, an, "agreement")
    if not (alo <= 0 <= ahi):
        print("  🔴 IDENTICAL ARMS DO NOT READ ZERO. The pipeline is broken; "
              "the treatment number below is void.")
        return 1
    print("  ✅ identical arms read zero — the pipeline is sound.")

    # ---- the treatment --------------------------------------------------
    print(f"\n[treatment] expert's move vs our net's move, "
          f"{len(dis_s)} positions × {args.pairs} pairs")
    tp, tn, tsec, ti = measure(dis_s, args.pairs, net, "treat")
    tm, tlo, thi = report(tp, tn, "Δ(expert − ours)")
    # 🔴 THE MEAN IS NOT THE HEADROOM. A null mean is compatible with a large
    # per-decision gap whose SIGN varies -- |E[x]| vs E[|x|], the same
    # distinction E15's pre-registration drew about the near-tie band. Decompose
    # the observed spread into true between-position variance and measurement
    # noise, because the TRUE dispersion is what an oracle could capture.
    w = [x for x in getattr(measure, "last_within", []) if x == x]
    if len(tp) > 2 and w:
        obs_sd = statistics.stdev(tp)
        meas = statistics.fmean(w) / (statistics.fmean(tn) ** 0.5)
        true_var = obs_sd ** 2 - meas ** 2
        true_sd = true_var ** 0.5 if true_var > 0 else 0.0
        print(f"  [dispersion] observed per-position sd {obs_sd:.4f}; "
              f"measurement noise {meas:.4f}; ⇒ TRUE sd ≈ {true_sd:.4f}")
        print(f"  ⇒ typical |gap| at a disagreement ≈ {0.798 * true_sd:.4f} "
              f"(E|X| for a normal). An oracle picking the better side gains "
              f"≈ {0.399 * true_sd:.4f}/decision over a coin flip.")
    print(f"  scale bar (§8bw, clone's own top vs last): +0.120")
    print(f"  wall: {tsec:.0f}s treatment + {asec:.0f}s control")

    if tlo > 0:
        print("  ⇒ 🔴 THE EXPERTS' MOVES ARE BETTER in our own continuation.")
    elif thi < 0:
        print("  ⇒ 🔴 THE EXPERTS' MOVES ARE WORSE under our continuation — "
              "the strongest form of H1 (partial copying breaks coherence).")
    else:
        print("  ⇒ NULL: their per-move advantage is invisible to a clone "
              "continuation. H1 and H2 both predict this ⇒ arm C decides.")

    # ---- exploratory, labelled -----------------------------------------
    print("\n[exploratory — cannot produce a rule on its own]")
    by_turn: dict[int, list[float]] = defaultdict(list)
    by_band: dict[str, list[float]] = defaultdict(list)
    for j, m in zip(ti, tp):
        o, _seat, _deck, _a, _b, turn, rating, _g = dis_s[j]
        by_turn[min(turn // 3, 4)].append(m)
        band = ("1100+" if rating >= 1100 else
                "1075-1100" if rating >= 1075 else "1050-1075")
        by_band[band].append(m)
    for b in sorted(by_turn):
        v = by_turn[b]
        if len(v) >= 5:
            print(f"  turns {b * 3}-{b * 3 + 2}: Δ={statistics.fmean(v):+.4f} "
                  f"k={len(v)}")
    print("  by rating band (pre-declared -- does the cut drive the answer?)")
    for b in sorted(by_band):
        v = by_band[b]
        if len(v) >= 5:
            sd = statistics.stdev(v) / (len(v) ** 0.5)
            print(f"    {b:10s} Δ={statistics.fmean(v):+.4f} "
                  f"±{1.96 * sd:.4f}  k={len(v)}")

    out = {
        "net": str(npz), "dump": args.dump, "min_rating": args.min_rating,
        "pairs": args.pairs,
        "agreement": {"delta": am, "lo": alo, "hi": ahi, "k": len(ap)},
        "treatment": {"delta": tm, "lo": tlo, "hi": thi, "k": len(tp)},
        "diag": dict(diag),
    }
    (ROOT / "out/logs/p81_e16.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="replays/mirror_experts")
    ap.add_argument("--games", type=int, default=257)
    ap.add_argument("--lb", default="out/lb/pokemon-tcg-ai-battle.zip")
    ap.add_argument("--min-rating", type=float, default=1100.0)
    ap.add_argument("--positions", type=int, default=600)
    ap.add_argument("--agree-positions", type=int, default=150)
    ap.add_argument("--pairs", type=int, default=30)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
