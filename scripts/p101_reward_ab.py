"""A/B terminal vs prize vs Teal Mask rewards on the existing Ogerpon corpus.

R0: G = won                         (win/loss AWR; scale-up of oger_ft)
R1: G = won + α·(prize_diff_T − prize_diff_t)
R2: R1 + bonus when opp Teal Mask count drops between our selects

    python -X utf8 scripts/p101_reward_ab.py write
    python -X utf8 scripts/p101_reward_ab.py run --arms r0,r1,r2 --matches 500

Does not regenerate games. Rewrites `won` in copied shards; original
`won_terminal` is kept. Training is AWR on that column, init from v5_s2_all.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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

TEAL_MASK = 96
SRC_DIRS = [
    ROOT / "artifacts" / "pds_ogerpon_matchup",
    ROOT / "artifacts" / "pds_ogerpon_r2",
]
OUT_ROOT = ROOT / "artifacts" / "pds_oger_rew"
ALPHA = 1.0
MASK_BONUS = 0.5
ARMS = ("r0", "r1", "r2")
BAGS = ("my_hand", "my_discard", "opp_discard")
ROW_KEYS = (
    "dense", "slots", "seld", "xdense", "xslots", "attr", "gid", "won",
    "won_terminal", "rating", "opp_rating", "team_id", "sub_id",
    "behav_logp", "margin", "seat", "adv",
)
OPT_KEYS = ("opt_dense", "opt_card", "opt_attack", "opt_target", "opt_chosen")


def _filter_rows(z: dict, keep: np.ndarray) -> dict:
    keep = np.asarray(keep, dtype=bool)
    if keep.all():
        return z
    out: dict = {}
    for k in ROW_KEYS:
        if k in z:
            out[k] = np.asarray(z[k])[keep]
    off = np.asarray(z["opt_off"])
    starts, ends = off[:-1][keep], off[1:][keep]
    for k in OPT_KEYS:
        chunks = [np.asarray(z[k])[s:e] for s, e in zip(starts, ends)]
        out[k] = (np.concatenate(chunks) if chunks
                  else np.zeros((0,) + np.asarray(z[k]).shape[1:],
                                dtype=np.asarray(z[k]).dtype))
    new_off = np.zeros(int(keep.sum()) + 1, dtype=np.int64)
    if len(starts):
        new_off[1:] = np.cumsum(ends - starts)
    out["opt_off"] = new_off
    for name in BAGS:
        boff = np.asarray(z[f"bag_{name}_off"])
        flat = np.asarray(z[f"bag_{name}_flat"])
        bstarts, bends = boff[:-1][keep], boff[1:][keep]
        bchunks = [flat[s:e] for s, e in zip(bstarts, bends)]
        out[f"bag_{name}_flat"] = (np.concatenate(bchunks) if bchunks
                                   else np.zeros(0, dtype=flat.dtype))
        bo = np.zeros(int(keep.sum()) + 1, dtype=np.int64)
        if len(bstarts):
            bo[1:] = np.cumsum(bends - bstarts)
        out[f"bag_{name}_off"] = bo
    for k, v in z.items():
        if k not in out:
            out[k] = v
    return out


def _shard_paths() -> list[Path]:
    paths: list[Path] = []
    for d in SRC_DIRS:
        paths.extend(sorted(d.rglob("shard_*.npz")))
    if not paths:
        raise SystemExit("no source shards; run p101 generate/iterate first")
    return paths


def _rewards(z: dict, alpha: float, mask_bonus: float
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    gid = np.asarray(z["gid"])
    won = np.asarray(z["won"], dtype=np.float32)
    dense = np.asarray(z["dense"])
    slots = np.asarray(z["slots"])
    n = len(gid)
    g0 = won.copy()
    g1 = np.empty(n, dtype=np.float32)
    g2 = np.empty(n, dtype=np.float32)
    pdiff = dense[:, 2] - dense[:, 3]
    mask_n = (slots[:, 6:12] == TEAL_MASK).sum(axis=1)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(gid.tolist()):
        groups[int(g)].append(i)
    n_ko = 0
    for idxs in groups.values():
        w = float(won[idxs[0]])
        p_t = float(pdiff[idxs[-1]])
        for j, i in enumerate(idxs):
            g1[i] = w + alpha * (p_t - float(pdiff[i]))
            bonus = 0.0
            if j + 1 < len(idxs) and int(mask_n[i]) > int(mask_n[idxs[j + 1]]):
                bonus = mask_bonus
                n_ko += 1
            g2[i] = g1[i] + bonus
    return g0, g1, g2, n_ko


def cmd_write(args: argparse.Namespace) -> int:
    paths = _shard_paths()
    for arm in ARMS:
        (OUT_ROOT / arm).mkdir(parents=True, exist_ok=True)
    n_rows = n_ko = n_games = 0
    g_sum = {a: 0.0 for a in ARMS}
    idx = 0
    for src in paths:
        z = dict(np.load(src, allow_pickle=True))
        g0, g1, g2, ko = _rewards(z, args.alpha, args.mask_bonus)
        n_ko += ko
        n_games += len(set(int(g) for g in np.asarray(z["gid"]).tolist()))
        packed = {"r0": g0, "r1": g1, "r2": g2}
        term = np.asarray(z["won"], dtype=np.float32)
        keep = (term > 0.5) if getattr(args, "winners_only", True) else None
        n_kept = int(keep.sum()) if keep is not None else len(g0)
        n_rows += n_kept
        for arm, g in packed.items():
            out = dict(z)
            out["won_terminal"] = term
            out["won"] = g
            if keep is not None:
                out = _filter_rows(out, keep)
            g_sum[arm] += float(np.asarray(out["won"]).mean()) * len(out["won"])
            dest = OUT_ROOT / arm / f"shard_{idx:03d}.npz"
            np.savez_compressed(dest, **out)
        idx += 1
        print(f"  {src.relative_to(ROOT)} -> shard_{idx-1:03d}  n={len(g0)}"
              f"{'' if keep is None else f' keep={n_kept}'}  "
              f"mask_ko_steps={ko}", flush=True)
    print(f"wrote {idx} shards x3  rows={n_rows}  gid-frags={n_games}  "
          f"mask_ko_steps={n_ko}", flush=True)
    for arm in ARMS:
        print(f"  {arm} mean G={g_sum[arm] / max(n_rows, 1):.4f}", flush=True)
    return 0


def _train(arm: str, args: argparse.Namespace) -> int:
    ds = str(OUT_ROOT / arm)
    out = f"out/policy_v5_s2_oger_{arm}.npz"
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "train_policy.py"),
        "--ds", ds, *POLICY_ARCH,
        "--advantage", str(args.advantage),
        "--margin-max", str(args.margin_max),
        "--init", args.init,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--export-last",
        "--out", out,
        "--device", args.device,
    ]
    rc = _run(cmd, LOG_DIR / f"p101_rew_{arm}_train.log").returncode
    if rc == 0:
        print(f"trained {arm} -> {out}", flush=True)
    return rc


def _test(arm: str, matches: int) -> tuple[float, float, float, int] | None:
    net = f"out/policy_v5_s2_oger_{arm}.npz"
    log = LOG_DIR / f"p101_rew_{arm}_vs_ogerpon.txt"
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "arena.py"), "play",
        _a_spec(net), B_SPEC,
        "--deck-a", DECK_A, "--deck-b", DECK_B,
        "--matches", str(matches),
        "--archive", str(ROOT / "out" / "arena" / f"p101_rew_{arm}_vs_ogerpon.jsonl"),
    ]
    proc = _run(cmd, log)
    if proc.returncode != 0:
        return None
    body = log.read_text(encoding="utf-8", errors="replace")
    for line in body.splitlines():
        m = SCORE_RE.search(line)
        if m:
            return (float(m.group(1)), float(m.group(2)),
                    float(m.group(3)), int(m.group(4)))
    return None


def cmd_run(args: argparse.Namespace) -> int:
    rc = cmd_write(args)
    if rc:
        return rc
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    lines = [f"=== p101 reward A/B matches={args.matches} ==="]
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}; expected {ARMS}")
        print(f"=== train {arm} ===", flush=True)
        rc = _train(arm, args)
        if rc:
            return rc
        print(f"=== test {arm} vs Ogerpon ===", flush=True)
        got = _test(arm, args.matches)
        if not got:
            return 1
        sc, lo, hi, n = got
        line = f"{arm}  WR={sc:.3f} [{lo:.3f},{hi:.3f}] n={n}"
        print(line, flush=True)
        lines.append(line)
    text = "\n".join(lines) + "\n"
    (LOG_DIR / "p101_reward_ab.txt").write_text(text, encoding="utf-8")
    with SUMMARY.open("a", encoding="utf-8") as sf:
        sf.write(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_w = sub.add_parser("write", help="rewrite shards with G in won")
    p_w.add_argument("--alpha", type=float, default=ALPHA)
    p_w.add_argument("--mask-bonus", type=float, default=MASK_BONUS)
    p_w.add_argument("--winners-only", action=argparse.BooleanOptionalAction,
                     default=True,
                     help="drop losing games so the corpus fits local RAM")
    p_w.set_defaults(func=cmd_write)

    p_r = sub.add_parser("run", help="write if needed, train, arena")
    p_r.add_argument("--arms", default="r0,r1,r2")
    p_r.add_argument("--matches", type=int, default=500)
    p_r.add_argument("--init", default=NET_ALL)
    p_r.add_argument("--epochs", type=int, default=5)
    p_r.add_argument("--lr", type=float, default=1e-3)
    p_r.add_argument("--advantage", type=float, default=0.5)
    p_r.add_argument("--margin-max", type=float, default=2.0)
    p_r.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p_r.add_argument("--alpha", type=float, default=ALPHA)
    p_r.add_argument("--mask-bonus", type=float, default=MASK_BONUS)
    p_r.add_argument("--winners-only", action=argparse.BooleanOptionalAction,
                     default=True)
    p_r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
