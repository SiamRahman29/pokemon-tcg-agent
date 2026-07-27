"""Download episode replay JSONs from the daily top-episodes datasets.

    python scripts/fetch_top_episodes.py [--date 2026-07-26] [--max 400]

Files land in replays/<date>/<episode_id>.json. Idempotent: existing files are
skipped, so re-running with a larger --max only fetches the difference.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-07-26")
    ap.add_argument("--max", type=int, default=400)
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dataset = f"kaggle/pokemon-tcg-ai-battle-episodes-{args.date}"
    out_dir = ROOT / "replays" / args.date
    out_dir.mkdir(parents=True, exist_ok=True)

    # The daily datasets ship a manifest.csv with per-episode avg_score;
    # fetch it first and take the top-rated episodes.
    import csv

    manifest = out_dir / "manifest.csv"
    if not manifest.exists():
        api.dataset_download_file(dataset, "manifest.csv", path=str(out_dir),
                                  quiet=True)
        zpath = out_dir / "manifest.csv.zip"
        if zpath.exists():
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(out_dir)
            zpath.unlink()
    with manifest.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: -float(r["avg_score"]))
    names = [f"{r['episode_id']}.json" for r in rows[: args.max]]
    print(f"{dataset}: {len(rows)} episodes in manifest; fetching top "
          f"{len(names)} by avg_score (cutoff "
          f"{float(rows[min(args.max, len(rows)) - 1]['avg_score']):.0f}) -> {out_dir}")

    done = skipped = failed = 0
    for i, name in enumerate(names):
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        try:
            api.dataset_download_file(dataset, name, path=str(out_dir),
                                      quiet=True)
            zpath = out_dir / (name + ".zip")
            if zpath.exists():  # kaggle wraps single files in a zip sometimes
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(out_dir)
                zpath.unlink()
            done += 1
        except Exception as exc:
            failed += 1
            print(f"  {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(names)} (ok={done} skip={skipped} "
                  f"fail={failed})", flush=True)
    print(f"finished: ok={done} skip={skipped} fail={failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
