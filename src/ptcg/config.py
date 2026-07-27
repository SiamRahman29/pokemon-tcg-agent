"""Repo paths and SDK discovery."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DIST_DIR = ROOT / "dist"
OUT_DIR = ROOT / "out"


def find_sdk_dir() -> Path | None:
    """Directory that *contains* the `cg/` package (so it can go on sys.path)."""
    for api in sorted(DATA_DIR.glob("**/cg/api.py")):
        return api.parents[1]
    return None


def find_sample_deck() -> Path | None:
    for deck in sorted(DATA_DIR.glob("**/deck.csv")):
        return deck
    return None
