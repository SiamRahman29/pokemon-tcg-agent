"""Pool the E18 shards and read them against the PRE-REGISTERED branches.

`docs/experiments/E18-clock-arena.md`, frozen at `cc070b0` before the first
game. This script does not decide anything the pre-registration did not already
decide; it only computes the numbers the branches are written in terms of.

    python -X utf8 scripts/p83_e18_score.py
"""
from __future__ import annotations

import glob
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORC = "orc"                      # the treatment agent's tag substring
RESERVE_S = 45.0                 # timeout = loss; below this the run is void


def rows(pattern: str = "out/arena/e18/shard_*.jsonl") -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(ROOT / pattern))):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def wilson(w: float, n: int) -> tuple[float, float, float]:
    """Score interval -- the same one arena.py quotes."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = w / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def main() -> int:
    import sys
    pattern = sys.argv[1] if len(sys.argv) > 1 else "out/arena/e18/shard_*.jsonl"
    logs = sys.argv[2] if len(sys.argv) > 2 else "out/logs/p83_e18_*.txt"
    rs = rows(pattern)
    if not rs:
        print("no rows yet")
        return 1

    # Which seat is the oracle in each row? Never assume seat 0 -- arena swaps.
    score = 0.0
    n = 0
    by_seat = {0: [0.0, 0], 1: [0.0, 0]}
    pools: list[float] = []
    for r in rs:
        a0, a1 = r["agent0"], r["agent1"]
        if (ORC in a0) == (ORC in a1):
            continue                       # cannot tell the arms apart
        seat = 0 if ORC in a0 else 1
        w = r["winner"]
        s = 0.5 if w == 2 else (1.0 if w == seat else 0.0)
        score += s
        n += 1
        by_seat[seat][0] += s
        by_seat[seat][1] += 1
        pools.append(float(r.get(f"pool{seat}", 600.0)))

    p, lo, hi = wilson(score, n)
    print(f"{pattern} -- vs the shipped agent, mirror, byte-identical net")
    print(f"  n = {n} games")
    print(f"  score = {p:.4f} [{lo:.4f}, {hi:.4f}]   (Wilson)")
    for s in (0, 1):
        w, k = by_seat[s]
        if k:
            print(f"    as P{s}: {w / k:.4f}  ({w:.1f}/{k})")

    # ---- the void condition, checked BEFORE the verdict ----------------
    worst = min(pools) if pools else 600.0
    print(f"\n  worst remaining 600 s pool over all games: {worst:.1f} s")
    if worst < RESERVE_S:
        print(f"  🔴 A GAME DROPPED BELOW THE {RESERVE_S:.0f} s RESERVE. "
              f"`timeout = loss` on the ladder ⇒ THE RUN IS VOID and the "
              f"budget manager needs fixing before anything else.")
        return 1
    print("  ✅ every game stayed above the reserve.")

    # ---- did it fire? a component that never fires reads as a null -----
    fired = Counter()
    for lg in sorted(glob.glob(str(ROOT / logs))):
        t = open(lg, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"(\w+)=(\d+)", t):
            if m.group(1) in ("fired", "overruled", "probed", "skip_won",
                              "skip_trigger", "errors", "rollouts",
                              "aborted", "skip_thin", "skip_capped"):
                fired[m.group(1)] += int(m.group(2))
    if fired:
        print("\n  [firing] " + "  ".join(f"{k}={v}" for k, v in
                                          sorted(fired.items())))
        if n:
            print(f"    ⇒ {fired['fired'] / n:.2f} fires/game, "
                  f"{fired['overruled'] / n:.2f} overrules/game")
        if fired["rollouts"]:
            er = fired["errors"] / fired["rollouts"]
            print(f"    ⇒ rollout error rate {er:.1%}"
                  + ("  ⚠ report this with the result" if er > 0.01 else ""))
        if fired["fired"] == 0:
            print("    🔴 THE ORACLE NEVER FIRED — this is not a null about "
                  "the clock, it is a null about the wiring.")
            return 1

    # ---- the pre-registered branches -----------------------------------
    print("\n" + "=" * 68)
    print("PRE-REGISTERED DECISION RULE (E18 §3, frozen at cc070b0)")
    if hi < 0.50 or p <= 0.470:
        print(f"  ⇒ 🔴 KILL. The clock LOSES games ({p:.4f}). E17's per-decision "
              f"gain did not compose into a win rate.")
    elif p >= 0.530 and lo > 0.50:
        print(f"  ⇒ ✅ PROMISING ({p:.4f}). §3 says extend to n≈1,200 for the "
              f"ship decision — this screen may NOT ship on its own (§8bh: s7 "
              f"screened 0.528 and read 0.487 on fresh games).")
    else:
        print(f"  ⇒ 🟡 INCONCLUSIVE ({p:.4f}), which §3 named as the EXPECTED "
              f"outcome. n=400 has SE≈0.025 and detects a true +0.03 with "
              f"probability 0.34. **A null here is not a kill.**")
    print("=" * 68)
    print("⚠ The mirror flatters this agent: the oracle rolls out with the "
          "clone piloting both seats, and here the opponent IS the clone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
