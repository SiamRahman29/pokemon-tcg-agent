"""Validate and extract the returned private E1 sweep results."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("control", "outcome", "count", "both")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="out/e1/e1_results.zip")
    ap.add_argument("--dest", default="out/e1/results")
    args = ap.parse_args()
    source = ROOT / args.zip
    dest = ROOT / args.dest
    if not source.exists():
        raise SystemExit(f"missing {source}")
    if dest.exists():
        raise SystemExit(f"{dest} already exists; preserve it before re-intake")

    with zipfile.ZipFile(source) as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise SystemExit("result archive has no manifest.json")
        manifest = json.loads(zf.read("manifest.json"))
        expected = {
            "experiment": "E1",
            "device": "cuda",
            "epochs": 12,
            "batch_size": 1024,
            "seed": 0,
            "arms": list(ARMS),
        }
        bad = {k: (manifest.get(k), want) for k, want in expected.items()
               if manifest.get(k) != want}
        if bad:
            raise SystemExit(f"manifest does not match preregistration: {bad}")
        for arm in ARMS:
            net = f"{arm}_seed0.npz"
            log = f"{arm}_seed0.log"
            if net not in names or log not in names:
                raise SystemExit(f"{arm}: missing {net} or {log}")
            got = hashlib.sha256(zf.read(net)).hexdigest()
            want = manifest["results"][arm]["sha256"]
            if got != want:
                raise SystemExit(f"{arm}: sha256 {got} != manifest {want}")
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise SystemExit(f"unsafe archive member {member.filename!r}")
        zf.extractall(dest)

    print(f"E1_INTAKE_OK cuda 12 epochs seed=0 -> {dest}")
    for arm in ARMS:
        print(f"  {arm}: {manifest['results'][arm]['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
