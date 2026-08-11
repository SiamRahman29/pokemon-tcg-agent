"""Pool and score an E23 cell -- the two-cell delta, with its own width.

    python -X utf8 scripts/p90_e23_score.py a|b

Reads ONLY arena's printed `A=` and `[health]` lines out of
`out/logs/p90_e23_<cell>_<arm>_*.txt`. Rule 18: never re-derive a score from the
archives, which is where the seat-indexing bug lives.

Prints, per arm, the pooled W/D/L and a Wilson interval, then the delta
(treatment - control) with **SE = sqrt(SE_a^2 + SE_b^2)** -- E23's bars are on
the delta, and §8aw's lesson is that our own driver once printed a single cell's
width and understated a delta's resolution by 41%.

Also evaluates control 1, the VOID condition that this whole experiment exists
to avoid: the treatment arm must show `fetch_diff` >= 0.10 per game.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "out" / "logs"

SCORE_RE = re.compile(r"^A=.*: score=([0-9.]+) .* W(\d+)/D(\d+)/L(\d+) over (\d+) games")
HEALTH_RE = re.compile(r"\[health\] (\w+) .*deck=(\d+)")
FETCH_RE = re.compile(r"fetch=(\d+)/(\d+) \([\d.]+%\) diff=(\d+)")

BARS = {"screen": 0.030, "harm": -0.030, "void_diff_per_game": 0.10}


def wilson(w: float, n: int) -> tuple[float, float, float]:
    """Score, lo, hi -- draws count as half a win, as arena does."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = w / n
    z = 1.959963985
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def read_arm(cell: str, arm: str) -> dict:
    """Pool every shard log for one arm."""
    W = D = L = N = 0
    seen = fired = diff = 0
    games_with_agent = 0
    degraded = []
    files = sorted(LOGS.glob(f"p90_e23_{cell}_{arm}_*.txt"))
    if not files:
        raise SystemExit(f"no logs for cell {cell} arm {arm}")
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            m = SCORE_RE.match(line)
            if m:
                W += int(m.group(2)); D += int(m.group(3))
                L += int(m.group(4)); N += int(m.group(5))
            h = HEALTH_RE.search(line)
            if h:
                if h.group(1) != "OK":
                    degraded.append(f.name)
                games_with_agent += int(h.group(2))
            fm = FETCH_RE.search(line)
            if fm:
                fired += int(fm.group(1))
                seen += int(fm.group(2))
                diff += int(fm.group(3))
    score, lo, hi = wilson(W + 0.5 * D, N)
    se = math.sqrt(max(score * (1 - score), 1e-9) / N) if N else float("nan")
    return dict(files=len(files), W=W, D=D, L=L, n=N, score=score, lo=lo, hi=hi,
                se=se, seen=seen, fired=fired, diff=diff,
                games=games_with_agent, degraded=degraded)


def main() -> None:
    cell = (sys.argv[1] if len(sys.argv) > 1 else "a").lower()
    t = read_arm(cell, "fscrap")
    c = read_arm(cell, "base")

    print(f"=== E23 cell {cell} ===")
    for name, a in (("fscrap (treatment)", t), ("base   (control)  ", c)):
        print(f"{name}: score={a['score']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}] "
              f"W{a['W']}/D{a['D']}/L{a['L']} n={a['n']} ({a['files']} shards)")
        if a["degraded"]:
            print(f"  DEGRADED health in: {a['degraded']}")

    print()
    print("--- control 1: did the rule actually treat anything?")
    for name, a in (("fscrap", t), ("base  ", c)):
        per = a["diff"] / a["games"] if a["games"] else 0.0
        print(f"  {name}: fetches_seen={a['seen']} fired={a['fired']} "
              f"diff={a['diff']}  ->  {per:.3f} changed picks/game")
    t_per = t["diff"] / t["games"] if t["games"] else 0.0
    c_per = c["diff"] / c["games"] if c["games"] else 0.0
    void = t_per < BARS["void_diff_per_game"]
    if c["fired"] or c["seen"]:
        print("  X CONTROL ARM FIRED -- the arms are not flag-isolated.")

    print()
    delta = t["score"] - c["score"]
    se = math.sqrt(t["se"] ** 2 + c["se"] ** 2)
    lo, hi = delta - 1.959963985 * se, delta + 1.959963985 * se
    print(f"--- the two-cell delta (its width is sqrt(2)x a single cell's)")
    print(f"  delta = {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"SE={se:.4f}  z={delta / se if se else float('nan'):+.2f}")

    print()
    if void:
        print(f"  VOID: treatment is {t_per:.3f} changed picks/game, under the "
              f"pre-registered {BARS['void_diff_per_game']:.2f}. This is a "
              f"statement about wiring, not about H-scrap (E21b's failure).")
    elif lo > 0 and delta >= BARS["screen"]:
        print("  SCREEN PASSES: candidate, NOT a ship -- needs the mirror "
              "neutrality cell and the five-anchor sweep (rule 16).")
    elif hi < 0 and delta <= BARS["harm"]:
        print("  HARMFUL branch: audit the condition before anything else.")
    elif lo <= 0 <= hi:
        print("  KILL: the CI contains 0. H-scrap refuted at this n for this "
              "matchup; tradeoff rules go 0-for-7.")
    else:
        print("  Between the bars: CI excludes 0 but the point is under the "
              "pre-registered magnitude. Report as measured; no branch fires.")


if __name__ == "__main__":
    main()
