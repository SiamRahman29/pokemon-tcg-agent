#!/usr/bin/env python
"""Pack the minimal runnable repo into a private Kaggle Dataset.

The payload is everything `scripts/arena.py play` needs and nothing else:
`src/ptcg`, `scripts/`, `agents/`, `decks/`, the engine bundle under
`data/sample_submission/` (which is what `config.find_sdk_dir()` globs for),
and whichever `out/*.npz` nets a job names.

⛔ Deliberately excluded: `data/Card_ID List_EN.pdf` (132 MB, unread by any
runtime path), `replays/` (24 GB), `out/arena/` and `artifacts/` (a training
job gets its own dataset -- see --corpus).

    python -X utf8 scripts/kaggle/pack.py --push
    python -X utf8 scripts/kaggle/pack.py --corpus pds_v4 --push

Prints the payload manifest with sizes before it uploads, because an upload
that silently omits a net produces a kernel that falls back to
`list(range(minCount))` and returns a plausible number (rule 18).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "out" / "kaggle_pack"

USER = "siamrahman29"
CODE_SLUG = "ptcg-code-v1"

# ⛔ The payload contains the licensed `cg` engine. The dataset MUST stay
# private (the CLI defaults to private; never pass --public). Same constraint
# `p41_pack_e1_gpu.py` records for the E1 bundle.

# (source, dest) relative to ROOT. Directories are copied whole.
PAYLOAD = [
    ("src/ptcg", "src/ptcg"),
    ("scripts", "scripts"),
    ("agents", "agents"),
    ("decks", "decks"),
    ("data/sample_submission", "data/sample_submission"),
    ("data/EN_Card_Data.csv", "data/EN_Card_Data.csv"),
]

# Nets live in out/ and are named per job; ship the whole v5 family plus the
# shipped weights so a job spec can reference any of them without a re-pack.
NET_GLOBS = ["out/policy_v5*.npz", "out/policy_v4.npz", "out/policy_b7_ntum.npz"]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.log")


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def stage(corpus: str | None) -> list[tuple[str, int]]:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    manifest: list[tuple[str, int]] = []

    for src_rel, dst_rel in PAYLOAD:
        src = ROOT / src_rel
        if not src.exists():
            sys.exit(f"MISSING payload item: {src_rel}")
        dst = STAGE / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            shutil.copy2(src, dst)
        manifest.append((dst_rel, _size(dst)))

    (STAGE / "out").mkdir(exist_ok=True)
    nets = 0
    for pat in NET_GLOBS:
        for f in sorted(ROOT.glob(pat)):
            shutil.copy2(f, STAGE / "out" / f.name)
            nets += 1
            manifest.append((f"out/{f.name}", f.stat().st_size))
    if nets == 0:
        sys.exit("MISSING: no nets matched NET_GLOBS -- a kernel would run the fallback")

    if corpus:
        src = ROOT / "artifacts" / corpus
        if not src.exists():
            sys.exit(f"MISSING corpus: artifacts/{corpus}")
        shutil.copytree(src, STAGE / "artifacts" / corpus, ignore=IGNORE)
        manifest.append((f"artifacts/{corpus}", _size(STAGE / "artifacts" / corpus)))

    # 🔴 ONE zip with repo-relative paths inside, not `--dir-mode zip`.
    # The CLI's per-directory mode uploads N zips and then 400s on
    # CreateDataset; the single-archive shape is what `p41_pack_e1_gpu.py`
    # already proved works for this repo's engine bundle.
    payload = STAGE / "ptcg_code.zip"
    files = [f for f in STAGE.rglob("*") if f.is_file() and f != payload]
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            zf.write(f, str(f.relative_to(STAGE)).replace("\\", "/"))
    for f in files:
        f.unlink()
    for d in sorted((p for p in STAGE.rglob("*") if p.is_dir()), reverse=True):
        d.rmdir()
    manifest.append(("→ ptcg_code.zip (single archive, compressed)",
                     payload.stat().st_size))

    meta = {
        "title": "ptcg code v1",
        "id": f"{USER}/{CODE_SLUG}",
        "licenses": [{"name": "unknown"}],
    }
    (STAGE / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None,
                    help="also ship artifacts/<name> (for training kernels)")
    ap.add_argument("--push", action="store_true",
                    help="create or version the Kaggle dataset")
    ap.add_argument("--message", default="repack")
    args = ap.parse_args()

    manifest = stage(args.corpus)
    total = sum(s for _, s in manifest)
    print(f"=== payload staged at {STAGE} ===")
    for name, size in manifest:
        print(f"  {size/1e6:8.2f} MB  {name}")
    print(f"  {'-'*8}")
    print(f"  {total/1e6:8.2f} MB  TOTAL")

    if not args.push:
        print("\n(dry run -- pass --push to upload)")
        return

    listed = subprocess.run(
        ["kaggle", "datasets", "list", "-m", "-s", CODE_SLUG],
        capture_output=True, text=True)
    exists = f"{USER}/{CODE_SLUG}" in listed.stdout

    cmd = (["kaggle", "datasets", "version", "-p", str(STAGE), "-m", args.message]
           if exists else
           ["kaggle", "datasets", "create", "-p", str(STAGE)])
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout or r.stderr)
    if r.returncode != 0:
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
