"""Ogerpon TD specialist: V on matchup games -> per-decision adv -> AWR.

    python -X utf8 scripts/p101_oger_td.py run --matches 500

Uses existing pds_ogerpon_matchup + pds_ogerpon_r2. Copies to
artifacts/pds_oger_td, trains V, writes `adv`, then fine-tunes a policy with
--advantage-col. Local RAM cannot load all ~787k policy rows; training keeps
a random fraction of games (default 0.3 ≈ 230k rows).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from p101_ogerpon_resist import (  # noqa: E402
    B_SPEC, DECK_A, DECK_B, LOG_DIR, NET_ALL, POLICY_ARCH, SCORE_RE, SUMMARY,
    _a_spec, _run,
)

SRC = [
    ROOT / "artifacts" / "pds_ogerpon_matchup",
    ROOT / "artifacts" / "pds_ogerpon_r2",
]
TD_DIR = ROOT / "artifacts" / "pds_oger_td"
TRAIN_DIR = ROOT / "artifacts" / "pds_oger_td_train"
VALUE_OUT = ROOT / "out" / "value_oger_td.npz"
POLICY_OUT = "out/policy_v5_s2_oger_td.npz"


def _rc(cmd: list[str], log_path: Path | None = None) -> int:
    return int(_run(cmd, log_path).returncode)


def _copy_shards() -> list[Path]:
    if TD_DIR.exists():
        shutil.rmtree(TD_DIR)
    TD_DIR.mkdir(parents=True)
    paths: list[Path] = []
    idx = 0
    for src in SRC:
        for p in sorted(src.rglob("shard_*.npz")):
            dest = TD_DIR / f"shard_{idx:03d}.npz"
            shutil.copy2(p, dest)
            paths.append(dest)
            idx += 1
    if not paths:
        raise SystemExit("no source shards")
    print(f"copied {len(paths)} shards -> {TD_DIR}", flush=True)
    return paths


def _train_value(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "train_value.py"),
        "--data", str(TD_DIR),
        "--out", str(VALUE_OUT),
        "--epochs", str(args.value_epochs),
        "--bs", "1024",
        "--lr", "1e-3",
    ]
    return _rc(cmd, LOG_DIR / "p101_oger_td_value.log")


def _write_adv() -> int:
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "p92_td_advantage.py"),
        "--data", str(TD_DIR),
        "--value", str(VALUE_OUT),
    ]
    return _rc(cmd, LOG_DIR / "p101_oger_td_p92.log")


def _filter_for_train(keep_frac: float, seed: int) -> int:
    """Keep a random subset of games so policy Data fits RAM."""
    if TRAIN_DIR.exists():
        shutil.rmtree(TRAIN_DIR)
    TRAIN_DIR.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    # Collect all gids first
    all_gids: set[int] = set()
    paths = sorted(TD_DIR.glob("shard_*.npz"))
    for p in paths:
        all_gids.update(int(g) for g in np.load(p)["gid"].tolist())
    gids = np.array(sorted(all_gids))
    n_keep = max(1, int(len(gids) * keep_frac))
    keep = set(int(x) for x in rng.choice(gids, size=n_keep, replace=False))
    print(f"filter: {n_keep}/{len(gids)} games ({keep_frac:.0%})", flush=True)

    from p101_reward_ab import _filter_rows  # noqa: E402

    n_rows = 0
    for i, p in enumerate(paths):
        z = dict(np.load(p, allow_pickle=True))
        gid = np.asarray(z["gid"])
        mask = np.array([int(g) in keep for g in gid], dtype=bool)
        if not mask.any():
            continue
        out = _filter_rows(z, mask)
        n_rows += int(mask.sum())
        dest = TRAIN_DIR / f"shard_{i:03d}.npz"
        np.savez_compressed(dest, **out)
    print(f"train corpus -> {TRAIN_DIR}  rows={n_rows}", flush=True)
    return 0 if n_rows else 1


def _train_policy(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "train_policy.py"),
        "--ds", str(TRAIN_DIR),
        *POLICY_ARCH,
        "--advantage-col", str(args.beta),
        "--init", args.init,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--export-last",
        "--out", POLICY_OUT,
        "--device", args.device,
    ]
    return _rc(cmd, LOG_DIR / "p101_oger_td_policy.log")


def _arena(net: str, matches: int, tag: str, deck_b: str, b_spec: str
           ) -> tuple[float, float, float, int]:
    log = LOG_DIR / f"p101_oger_td_{tag}.txt"
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "arena.py"), "play",
        _a_spec(net), b_spec,
        "--deck-a", DECK_A, "--deck-b", deck_b,
        "--matches", str(matches),
        "--archive", str(ROOT / "out" / "arena" / f"p101_oger_td_{tag}.jsonl"),
    ]
    rc = _rc(cmd, log)
    if rc:
        raise SystemExit(f"arena {tag} failed rc={rc}")
    body = log.read_text(encoding="utf-8", errors="replace")
    for line in body.splitlines():
        m = SCORE_RE.search(line)
        if m:
            return (float(m.group(1)), float(m.group(2)),
                    float(m.group(3)), int(m.group(4)))
    raise SystemExit(f"no score in {log}")


def cmd_run(args: argparse.Namespace) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.skip_prep:
        pass
    elif args.from_p92:
        if not VALUE_OUT.exists() or not TD_DIR.exists():
            raise SystemExit("--from-p92 needs value net and TD_DIR")
        print("=== write TD adv ===", flush=True)
        rc = _write_adv()
        if rc:
            return rc
        print("=== filter train corpus ===", flush=True)
        rc = _filter_for_train(args.keep_frac, args.seed)
        if rc:
            return rc
    else:
        _copy_shards()
        print("=== train value ===", flush=True)
        rc = _train_value(args)
        if rc:
            return rc
        print("=== write TD adv ===", flush=True)
        rc = _write_adv()
        if rc:
            return rc
        print("=== filter train corpus ===", flush=True)
        rc = _filter_for_train(args.keep_frac, args.seed)
        if rc:
            return rc
    print("=== fine-tune policy ===", flush=True)
    rc = _train_policy(args)
    if rc:
        return rc
    print("=== arena vs Ogerpon ===", flush=True)
    sc, lo, hi, n = _arena(POLICY_OUT, args.matches, "vs_ogerpon",
                           DECK_B, B_SPEC)
    line_o = f"oger_td vs Ogerpon  WR={sc:.3f} [{lo:.3f},{hi:.3f}] n={n}"
    print(line_o, flush=True)
    print("=== mirror sanity ===", flush=True)
    sc_m, lo_m, hi_m, n_m = _arena(
        POLICY_OUT, max(200, args.matches // 2), "mirror",
        DECK_A, _a_spec(NET_ALL))
    line_m = (f"oger_td vs stock grimmsnarl  WR={sc_m:.3f} "
              f"[{lo_m:.3f},{hi_m:.3f}] n={n_m}")
    print(line_m, flush=True)
    # stock baseline cell for comparison
    print("=== stock vs Ogerpon (paired control) ===", flush=True)
    sc_s, lo_s, hi_s, n_s = _arena(NET_ALL, args.matches, "stock_vs_ogerpon",
                                   DECK_B, B_SPEC)
    line_s = (f"stock vs Ogerpon  WR={sc_s:.3f} [{lo_s:.3f},{hi_s:.3f}] n={n_s}  "
              f"delta_td={sc - sc_s:+.3f}")
    print(line_s, flush=True)
    text = "\n".join([
        "=== p101 Ogerpon TD specialist ===",
        line_o, line_m, line_s,
        f"value={VALUE_OUT}",
        f"policy={POLICY_OUT}",
        f"keep_frac={args.keep_frac} beta={args.beta}",
    ]) + "\n"
    (LOG_DIR / "p101_oger_td_summary.txt").write_text(text, encoding="utf-8")
    with SUMMARY.open("a", encoding="utf-8") as sf:
        sf.write(text)
    print(text, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--matches", type=int, default=500)
    p.add_argument("--init", default=NET_ALL)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--value-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=0.5,
                   help="--advantage-col beta in sd(adv) units")
    p.add_argument("--keep-frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--skip-prep", action="store_true",
                   help="reuse TD_DIR / TRAIN_DIR / value net")
    p.add_argument("--from-p92", action="store_true",
                   help="value already trained; start at p92 write")
    p.set_defaults(func=cmd_run)
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
