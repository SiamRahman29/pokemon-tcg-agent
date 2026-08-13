#!/usr/bin/env python
"""Pool sharded arena logs into one score + CI.

🔴 Reads the **`score=` line `arena.py` already prints**, and sums its W/D/L.
It does NOT recompute anything from the `.jsonl` archives. That is deliberate:
`agent0`/`agent1` are seat-indexed and `arena.py` alternates seats every game,
so re-deriving a score from an archive averages each agent with its own
opponent -- the exact bug HANDOFF rule 18 was written for, which produced
"0.489 / 0.510 / 0.502" for true values of 0.857 / 0.888 / 0.870.

Pooling W/D/L across shards is valid because the shards are independent games
of the identical configuration, and it inherits the arena's seat correction.

    python -X utf8 scripts/kaggle/score.py --job c0-ident --expect 0.500
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINE = re.compile(r"^(?P<who>A=\S+): score=(?P<sc>[0-9.]+) "
                  r"\[[^]]*\] W(?P<w>\d+)/D(?P<d>\d+)/L(?P<l>\d+) over (?P<n>\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--expect", type=float, default=None,
                    help="pre-registered value the CI must contain")
    # A sharded run does not have to have been on Kaggle. E22 ran as 5 local
    # processes writing out/logs/e22a_s*.txt, and the pooling argument is the
    # same one the docstring makes -- independent games of one configuration,
    # each scored by the arena's own seat-corrected line. `--dir` points at
    # those logs so a local run is read by this scorer rather than by a
    # one-off, which is where re-derivation creeps back in.
    ap.add_argument("--dir", default=None,
                    help="directory of shard logs (default out/kaggle_out/<job>)")
    args = ap.parse_args()

    root = Path(args.dir) if args.dir else ROOT / "out" / "kaggle_out" / args.job
    logs = sorted(p for p in root.rglob("*_s*.txt")
                  if args.dir is None or p.name.startswith(args.job))
    if not logs:
        sys.exit(f"no shard logs for job {args.job} under {root}")

    W = D = L = 0
    names, unhealthy = set(), []
    for f in logs:
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = [LINE.match(x) for x in txt.splitlines()]
        m = [x for x in m if x]
        if not m:
            unhealthy.append(f"{f.name}: NO SCORE LINE")
            continue
        g = m[-1]
        W += int(g["w"]); D += int(g["d"]); L += int(g["l"])
        names.add(g["who"])
        for h in re.findall(r"\[health\][^\n]*", txt):
            if "OK" not in h or "fallbacks=0" not in h or "net_missing=0" not in h:
                unhealthy.append(f"{f.name}: {h.strip()}")
        # E32: the plan layer prints its own line and it carries the same
        # obligation -- `fallbacks` non-zero means the catch-all fired and the
        # agent was playing index order under an exception, which no score can
        # reveal on its own (§8g had to infer exactly that indirectly once).
        for h in re.findall(r"\[plan\][^\n]*", txt):
            if "OK" not in h or "fallbacks=0" not in h:
                unhealthy.append(f"{f.name}: {h.strip()}")

    n = W + D + L
    if n == 0:
        sys.exit("no games pooled")
    score = (W + 0.5 * D) / n
    se = math.sqrt(0.25 / n)
    lo, hi = score - 1.96 * se, score + 1.96 * se

    print(f"job      {args.job}")
    print(f"arm A    {' | '.join(sorted(names))}")
    print(f"shards   {len(logs)}")
    print(f"pooled   W{W}/D{D}/L{L} over {n} games")
    print(f"score    {score:.4f}  [{lo:.4f}, {hi:.4f}]   SE {se:.4f}")
    if unhealthy:
        print("\n🔴 HEALTH:")
        for u in unhealthy:
            print("  " + u)
    else:
        print("health   OK on every shard (fallbacks=0, net_missing=0)")

    if len(names) > 1:
        print("\n🔴 SHARDS DISAGREE ON THE ARM IDENTITY -- do not pool these.")
        return 1
    if args.expect is not None:
        ok = lo <= args.expect <= hi
        print(f"\nexpect   {args.expect:.4f}  ->  "
              f"{'✅ CONTAINED' if ok else '🔴 NOT CONTAINED'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
