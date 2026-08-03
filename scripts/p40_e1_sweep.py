"""Run the frozen E1 control/outcome/count/both training sweep.

Designed for a private Kaggle or Colab GPU session:

    python -X utf8 scripts/p40_e1_sweep.py --device cuda

Each arm has the same seed, data, architecture, batches, epochs, and final-epoch
export. The only differences are the pre-registered auxiliary loss weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "control": (0.0, 0.0),
    "outcome": (0.1, 0.0),
    "count": (0.0, 0.1),
    "both": (0.1, 0.1),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_and_tee(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
    if code:
        raise SystemExit(f"arm failed with exit code {code}; see {log_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="artifacts/pds_v4")
    ap.add_argument("--out-dir", default="out/e1")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="control,outcome,count,both",
                    help="comma-separated subset of: " + ",".join(ARMS))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing arm checkpoint/log")
    args = ap.parse_args()

    names = [s.strip() for s in args.arms.split(",") if s.strip()]
    unknown = set(names) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "E1",
        "started": time.time(),
        "dataset": args.ds,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.bs,
        "seed": args.seed,
        "arms": names,
        "results": {},
    }
    for name in names:
        outcome_w, count_w = ARMS[name]
        net_path = out_dir / f"{name}_seed{args.seed}.npz"
        log_path = out_dir / f"{name}_seed{args.seed}.log"
        if not args.force and (net_path.exists() or log_path.exists()):
            raise SystemExit(f"{name}: output already exists; use --force only "
                             "after preserving the prior run")
        cmd = [
            sys.executable, "-X", "utf8", "scripts/train_policy.py",
            "--ds", args.ds,
            "--epochs", str(args.epochs),
            "--bs", str(args.bs),
            "--loss", "listwise",
            "--state-h", "512,256",
            "--head-h", "256,128",
            "--pool",
            "--seed", str(args.seed),
            "--device", args.device,
            "--aux-outcome-w", str(outcome_w),
            "--aux-count-w", str(count_w),
            "--export-last",
            "--out", str(net_path.relative_to(ROOT)),
        ]
        print(f"\n=== E1 {name} ===")
        run_and_tee(cmd, log_path)
        digest = sha256(net_path)
        manifest["results"][name] = {
            "outcome_weight": outcome_w,
            "count_weight": count_w,
            "checkpoint": str(net_path.relative_to(ROOT)),
            "log": str(log_path.relative_to(ROOT)),
            "sha256": digest,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {digest}  {net_path}")

    manifest["finished"] = time.time()
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nE1_SWEEP_OK {len(names)} arms -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
