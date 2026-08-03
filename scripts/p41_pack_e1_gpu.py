"""Build the small PRIVATE upload bundle for the E1 GPU sweep.

The archive includes the licensed engine and must never be published or
committed. Upload it only as a private Kaggle/Colab input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inputs() -> list[Path]:
    paths: list[Path] = []
    for base, pattern in (
        (ROOT / "artifacts" / "pds_v4", "shard_*.npz"),
        (ROOT / "data" / "sample_submission" / "sample_submission" / "cg", "*"),
        (ROOT / "src", "*.py"),
        (ROOT / "agents" / "sa", "*.py"),
    ):
        if not base.exists():
            raise SystemExit(f"missing bundle input: {base}")
        paths.extend(p for p in base.rglob(pattern)
                     if p.is_file() and "__pycache__" not in p.parts)
    paths.extend([
        ROOT / "scripts" / "train_policy.py",
        ROOT / "scripts" / "p39_multitask_smoke.py",
        ROOT / "scripts" / "p40_e1_sweep.py",
    ])
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("missing bundle files: " + ", ".join(map(str, missing)))
    return sorted(set(paths))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/e1/e1_gpu_bundle.zip")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} already exists; preserve or remove it first")

    files = inputs()
    manifest = {
        "private": True,
        "experiment": "E1",
        "created": time.time(),
        "files": {
            str(p.relative_to(ROOT)).replace("\\", "/"): {
                "bytes": p.stat().st_size,
                "sha256": digest(p),
            }
            for p in files
        },
    }
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for path in files:
            arc = str(path.relative_to(ROOT)).replace("\\", "/")
            zf.write(path, arc)
        zf.writestr("E1_BUNDLE_MANIFEST.json",
                    json.dumps(manifest, indent=2) + "\n")

    print(f"E1_PRIVATE_BUNDLE_OK {out.stat().st_size / 1024 / 1024:.1f} MB "
          f"{len(files)} files -> {out}")
    print("Do not publish: archive contains the licensed cg engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
