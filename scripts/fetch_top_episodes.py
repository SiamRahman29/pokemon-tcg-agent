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

    from concurrent.futures import ThreadPoolExecutor

    counts = {"ok": 0, "skip": 0, "fail": 0}

    def fetch(name: str) -> None:
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            counts["skip"] += 1
            return
        try:
            api.dataset_download_file(dataset, name, path=str(out_dir),
                                      quiet=True)
            zpath = out_dir / (name + ".zip")
            if zpath.exists():  # kaggle wraps single files in a zip sometimes
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(out_dir)
                zpath.unlink()
            counts["ok"] += 1
        except Exception as exc:
            counts["fail"] += 1
            print(f"  {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        n = sum(counts.values())
        if n % 50 == 0:
            print(f"  {n}/{len(names)} {counts}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(fetch, names))
    print(f"finished: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
