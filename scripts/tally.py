"""Combine arena archives and report A's score with a Wilson interval.

    python scripts/tally.py <agent-name> out/arena/foo_*.jsonl

Sharded runs write separate archives; a single 30-game shard cannot resolve a
15pp effect, so ALWAYS pool the shards before reading a result. Rows are
seat-indexed (agent0 played seat 0), so this works regardless of seat order.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    name = sys.argv[1]
    paths: list[str] = []
    for pat in sys.argv[2:]:
        paths.extend(sorted(glob.glob(pat)))
    w = d = ln = 0
    pool_min = 600.0
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            seat = 0 if r["agent0"] == name else 1 if r["agent1"] == name else None
            if seat is None:
                continue
            pool_min = min(pool_min, r.get(f"pool{seat}", 600.0))
            if r["winner"] == 2:
                d += 1
            elif r["winner"] == seat:
                w += 1
            else:
                ln += 1
    n = w + d + ln
    if not n:
        raise SystemExit(f"no games for {name!r} in {len(paths)} file(s)")
    score = (w + 0.5 * d) / n
    lo, hi = wilson(w + 0.5 * d, n)
    print(f"{name}: score={score:.3f} [{lo:.3f}, {hi:.3f}] "
          f"W{w}/D{d}/L{ln} over {n} games ({len(paths)} archive(s))")
    verdict = ("better" if lo > 0.5 else
               "worse" if hi < 0.5 else "NOT separated from 0.5")
    print(f"  vs even: {verdict}")
    if pool_min < 300.0:
        print(f"  min thinking pool left: {pool_min:.0f}s of 600s"
              + ("  <-- WOULD TIME OUT ON KAGGLE" if pool_min <= 0 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
