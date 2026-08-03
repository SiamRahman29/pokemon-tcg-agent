"""Run the frozen E2 control/treatment adapter fine-tune.

    python -X utf8 scripts/p49_e2_sweep.py --device cpu

Both arms warm-start from v5, freeze the base, and export the final epoch.
The control keeps adapters present but forced off; the treatment trains them.
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
    "control": True,     # adapters_off
    "treatment": False,
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
    ap.add_argument("--init", default="out/policy_v5.npz")
    ap.add_argument("--out-dir", default="out/e2")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adapter-h", type=int, default=64)
    ap.add_argument("--arms", default="control,treatment")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    names = [s.strip() for s in args.arms.split(",") if s.strip()]
    unknown = set(names) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")
    init_path = ROOT / args.init
    if not init_path.exists():
        raise SystemExit(f"missing init checkpoint {init_path}")
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "E2",
        "started": time.time(),
        "dataset": args.ds,
        "init": args.init,
        "init_sha256": sha256(init_path),
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.bs,
        "lr": args.lr,
        "seed": args.seed,
        "adapter_h": args.adapter_h,
        "adapters": ["mirror", "alakazam"],
        "arms": names,
        "results": {},
    }
    for name in names:
        adapters_off = ARMS[name]
        net_path = out_dir / f"{name}_seed{args.seed}.npz"
        log_path = out_dir / f"{name}_seed{args.seed}.log"
        if not args.force and (net_path.exists() or log_path.exists()):
            raise SystemExit(f"{name}: output already exists; use --force only "
                             "after preserving the prior run")
        cmd = [
            sys.executable, "-X", "utf8", "scripts/train_policy.py",
            "--ds", args.ds,
            "--init", args.init,
            "--epochs", str(args.epochs),
            "--bs", str(args.bs),
            "--lr", str(args.lr),
            "--loss", "listwise",
            "--state-h", "512,256",
            "--head-h", "256,128",
            "--pool",
            "--seed", str(args.seed),
            "--device", args.device,
            "--adapters", "mirror,alakazam",
            "--adapter-h", str(args.adapter_h),
            "--freeze-except", "adapters",
            "--export-last",
            "--out", str(net_path.relative_to(ROOT)),
        ]
        if adapters_off:
            cmd.append("--adapters-off")
        print(f"\n=== E2 {name} ===")
        run_and_tee(cmd, log_path)
        digest = sha256(net_path)
        manifest["results"][name] = {
            "adapters_off": adapters_off,
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
    print(f"\nE2_SWEEP_OK {len(names)} arms -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
