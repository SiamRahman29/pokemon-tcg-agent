"""Calibrate readiness handoff τ against the v5_s2 train-time 50% episode cut.

Two BC nets share a recipe and split by `--episode-span` (dev = 0:0.5,
mid = 0.5:1). Live play cannot see remaining select count, so
`sa.readiness.prefer_mid` proxies that cut with a score. This script finds τ
such that `score >= τ` lines up with the train-time label on the SAME games
that built `artifacts/pds_v4` / `policy_v5_s2_dev_v4` / `mid_v4`.

Train-time label (matches `episode_span_mask` with start=0.5): eligible
decisions in chronological vis order; for decision i of n, y=1 iff i >= n//2
(floor split — odd-length games put the middle row in the second half).

Eligibility is the live-select loop in `scripts/build_policy_dataset.py`
(result still live, ≥2 options, recorded action, both seats). Shards are
NOT used: `_board_dev` needs `obs["current"]`.

    python -X utf8 scripts/p99_calibrate_readiness.py
    python -X utf8 scripts/p99_calibrate_readiness.py --dirs replays/2026-07-26

Do NOT pass August dumps — those are a later corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa.readiness import readiness_from_obs  # noqa: E402

# Replay dirs that built artifacts/pds_v4. August dumps are a different corpus.
DEFAULT_DIRS = [
    "replays/2026-07-26",
    "replays/2026-07-27",
    "replays/2026-07-28",
    "replays/2026-07-29",
]


def eligible_scores(rep: dict) -> list[float]:
    """Readiness at each build_policy_dataset-eligible select, vis order."""
    rewards = rep["rewards"]
    if rewards[0] is None or rewards[1] is None:
        return []
    vis = (rep.get("steps") or [[{}]])[0][0].get("visualize") or []
    out: list[float] = []
    for v in vis:
        obs = v.get("obs")
        if not obs or not obs.get("current") or not obs.get("select"):
            continue
        state = obs["current"]
        if state["result"] != -1:
            continue
        sel = obs["select"]
        opts = sel.get("option") or []
        if len(opts) < 2:
            continue
        action = v.get("selected")
        if action is None:
            action = v.get("action")
        if not isinstance(action, list):
            continue
        picked = [a for a in action
                  if isinstance(a, int) and 0 <= a < len(opts)]
        if len(picked) != len(action):
            continue
        out.append(readiness_from_obs(obs))
    return out


def walk_replays(dirs: list[str]) -> tuple[list[np.ndarray], int, int]:
    """Return (per-game score arrays, n_errors, n_empty_or_skip)."""
    games: list[np.ndarray] = []
    n_err = n_skip = 0
    paths: list[Path] = []
    for d in dirs:
        p = Path(d)
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_dir():
            print(f"missing dir: {p}", file=sys.stderr)
            continue
        paths.extend(sorted(x for x in p.glob("*.json")
                            if x.name != "manifest.json" and x.stem.isdigit()))
    print(f"scanning {len(paths)} replay jsons in {len(dirs)} dirs")
    for i, path in enumerate(paths, 1):
        try:
            rep = json.loads(path.read_text(encoding="utf-8"))
            scores = eligible_scores(rep)
            if not scores:
                n_skip += 1
                continue
            games.append(np.asarray(scores, dtype=np.float64))
        except Exception as exc:
            n_err += 1
            if n_err <= 5:
                print(f"  {path.name}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
        if i % 500 == 0 or i == len(paths):
            n_dec = sum(len(g) for g in games)
            print(f"  {i}/{len(paths)} files  games={len(games)}  "
                  f"decisions={n_dec}  skip={n_skip}  err={n_err}",
                  flush=True)
    return games, n_err, n_skip


def first_crossing(scores: np.ndarray, tau: float) -> int:
    """Index of first s>=τ; n if the gate never fires (one-way)."""
    hit = np.flatnonzero(scores >= tau)
    return int(hit[0]) if hit.size else int(len(scores))


def row_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Accuracy and positive-class F1 of pred vs y."""
    acc = float(np.mean(pred == y))
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return acc, f1


def sweep(games: list[np.ndarray], taus: np.ndarray) -> dict:
    cuts = np.array([g[len(g) // 2] for g in games], dtype=np.float64)
    y_parts: list[np.ndarray] = []
    s_parts: list[np.ndarray] = []
    for g in games:
        n = len(g)
        y_parts.append((np.arange(n) >= (n // 2)).astype(np.int8))
        s_parts.append(g)
    y = np.concatenate(y_parts)
    s = np.concatenate(s_parts)

    rows = []
    for tau in taus:
        pred = (s >= tau).astype(np.int8)
        acc, f1 = row_metrics(y, pred)
        errs = np.array([abs(first_crossing(g, float(tau)) - (len(g) // 2))
                         for g in games], dtype=np.float64)
        never = sum(1 for g in games
                    if first_crossing(g, float(tau)) == len(g))
        rows.append({
            "tau": float(tau),
            "acc": acc,
            "f1": f1,
            "mae": float(errs.mean()),
            "never": never,
        })

    by_mae = min(rows, key=lambda r: (r["mae"], -r["f1"], abs(r["tau"] - float(np.median(cuts)))))
    by_f1 = max(rows, key=lambda r: (r["f1"], r["acc"]))
    by_acc = max(rows, key=lambda r: (r["acc"], r["f1"]))
    return {
        "n_games": len(games),
        "n_dec": int(len(y)),
        "cuts": cuts,
        "rows": rows,
        "by_mae": by_mae,
        "by_f1": by_f1,
        "by_acc": by_acc,
        "median_cut": float(np.median(cuts)),
        "mean_cut": float(np.mean(cuts)),
        "p25_cut": float(np.percentile(cuts, 25)),
        "p75_cut": float(np.percentile(cuts, 75)),
        "pos_rate": float(y.mean()),
    }


def fmt_row(r: dict) -> str:
    return (f"  τ={r['tau']:.2f}  acc={r['acc']:.4f}  f1={r['f1']:.4f}  "
            f"mae={r['mae']:.3f}  never={r['never']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                    help="replay dirs that built pds_v4 (default: Jul 26-29)")
    ap.add_argument("--tmin", type=float, default=0.20)
    ap.add_argument("--tmax", type=float, default=0.70)
    ap.add_argument("--tstep", type=float, default=0.01)
    args = ap.parse_args()

    games, n_err, n_skip = walk_replays(args.dirs)
    if not games:
        raise SystemExit("no eligible games — check replay dirs / sdk.load()")

    n_steps = int(round((args.tmax - args.tmin) / args.tstep)) + 1
    taus = np.round(args.tmin + args.tstep * np.arange(n_steps), 2)
    z = sweep(games, taus)

    print()
    print(f"games={z['n_games']}  decisions={z['n_dec']}  "
          f"skip={n_skip}  errors={n_err}")
    print(f"label pos-rate (mid half)={z['pos_rate']:.3f}  "
          f"(odd-n games put the middle row in mid)")
    print()
    print("readiness at the cut row i=n//2:")
    print(f"  median={z['median_cut']:.4f}  mean={z['mean_cut']:.4f}  "
          f"p25={z['p25_cut']:.4f}  p75={z['p75_cut']:.4f}")
    print()
    print("sweep  (s>=τ) vs y=1[i>=n//2];  mae = mean |first s>=τ − n//2|")
    print("        PhaseHandoff is per-select (not sticky); first-crossing")
    print("        is the 'when to switch' timing the user actually feels.")
    for r in z["rows"]:
        print(fmt_row(r))

    print()
    print("best per-row (F1):     " + fmt_row(z["by_f1"]).strip())
    print("best per-row (acc):    " + fmt_row(z["by_acc"]).strip())
    print("best timing (min mae): " + fmt_row(z["by_mae"]).strip())
    print(f"median-at-boundary τ ≈ {z['median_cut']:.4f}  "
          f"(nearest sweep {min(z['rows'], key=lambda r: abs(r['tau'] - z['median_cut']))['tau']:.2f})")

    rec = z["by_mae"]
    per_row = z["by_f1"]
    differ = abs(rec["tau"] - per_row["tau"]) > 1e-9
    print()
    print("PhaseHandoff is NOT sticky (re-evaluates every select).")
    if differ:
        print(f"  timing-error τ={rec['tau']:.2f} and per-row F1 τ="
              f"{per_row['tau']:.2f} DIFFER — freeze timing-error "
              f"(first-crossing) as DEFAULT.")
    else:
        print(f"  timing-error τ and per-row F1 τ agree at {rec['tau']:.2f}.")

    # Reference points the arena has already used.
    for mark in (0.40, 0.50):
        hit = next((r for r in z["rows"] if abs(r["tau"] - mark) < 1e-9), None)
        if hit:
            vs = "earlier" if mark < rec["tau"] else (
                "later" if mark > rec["tau"] else "equal")
            print(f"  τ={mark:.2f} vs calibrated: {vs}  {fmt_row(hit).strip()}")

    print()
    print(f"DEFAULT_THRESHOLD = {rec['tau']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
