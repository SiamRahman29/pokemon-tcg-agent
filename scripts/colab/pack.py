"""Pack a Colab training zip: trainer + engine + init net + named corpora.

    python -X utf8 scripts/colab/pack.py
    python -X utf8 scripts/colab/pack.py --no-anchor

Upload `out/colab/ptcg_colab.zip` to Drive, then open
`notebooks/ptcg-colab-train.ipynb` on Colab (GPU T4).

The zip includes the licensed `cg` engine. Keep the Drive folder private.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "out" / "colab"
ZIP_NAME = "ptcg_colab.zip"

PAYLOAD = [
    ("src/ptcg", "src/ptcg"),
    ("scripts/train_policy.py", "scripts/train_policy.py"),
    ("scripts/train_value.py", "scripts/train_value.py"),
    ("scripts/p92_td_advantage.py", "scripts/p92_td_advantage.py"),
    ("scripts/p101_oger_td.py", "scripts/p101_oger_td.py"),
    ("scripts/p101_reward_ab.py", "scripts/p101_reward_ab.py"),
    ("agents/sa", "agents/sa"),
    ("data/sample_submission", "data/sample_submission"),
    ("data/EN_Card_Data.csv", "data/EN_Card_Data.csv"),
]
NETS = ["out/policy_v5_s2_all.npz", "out/value_oger_td.npz"]
CORPORA = ["pds_oger_td", "pds_oger_td_train", "pds_all"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.log")


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-anchor", action="store_true",
                    help="omit artifacts/pds_all (smaller zip, no BC leash)")
    args = ap.parse_args()

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    manifest: list[tuple[str, int]] = []

    for src_rel, dst_rel in PAYLOAD:
        src, dst = ROOT / src_rel, STAGE / dst_rel
        if not src.exists():
            raise SystemExit(f"MISSING {src_rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            shutil.copy2(src, dst)
        manifest.append((dst_rel, _size(dst)))

    (STAGE / "out").mkdir(exist_ok=True)
    for rel in NETS:
        src = ROOT / rel
        if not src.exists():
            print(f"skip missing {rel}")
            continue
        shutil.copy2(src, STAGE / rel)
        manifest.append((rel, src.stat().st_size))

    corpora = [c for c in CORPORA if not (args.no_anchor and c == "pds_all")]
    for name in corpora:
        src = ROOT / "artifacts" / name
        if not src.exists():
            raise SystemExit(f"MISSING artifacts/{name}")
        shutil.copytree(src, STAGE / "artifacts" / name, ignore=IGNORE)
        manifest.append((f"artifacts/{name}", _size(STAGE / "artifacts" / name)))

    payload = STAGE / ZIP_NAME
    files = [f for f in STAGE.rglob("*") if f.is_file() and f != payload]
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            zf.write(f, str(f.relative_to(STAGE)).replace("\\", "/"))
    staged = payload.stat().st_size
    print(f"=== {payload} ===")
    for name, size in manifest:
        print(f"  {size/1e6:8.2f} MB  {name}")
    print(f"  {staged/1e6:8.2f} MB  {ZIP_NAME}")
    print("\nUpload this zip to Drive, then run notebooks/ptcg-colab-train.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
