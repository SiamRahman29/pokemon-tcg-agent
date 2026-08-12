"""E29 Q1 — re-census the 71.4% mirror share, IN §8ac's OWN FRAME.

Pre-registered in `docs/experiments/E29-mine-and-archetype-census.md` (frozen at
`e1beeee`).

    python -X utf8 scripts/p90_e29_band.py --dir replays/submission_v5_s2

**The frame, and why it is the only admissible one** (E29 §3). §8ac's 71.4% is
*the archetype of the opponents WE FACED, in our own ladder replays, restricted
to opponent rating >= 1000* -- **n = 14 games**, Wilson CI **[40%, 83%]**. Two
other numbers on the record are NOT comparable to it and may not be quoted
against it:

  * §8bq's **23.7%** is the composition of the censored published FEED;
  * day 26's **31.6%** is our own games with **every band pooled**, and §8ac's
    table is monotone in exactly that variable (5.3 / 18.6 / 42.4 / 71.4%).

Comparing an unbanded share to a banded one re-commits rule 16's sampling-frame
trap. So: band, and compare like with like.

⛔ Pre-registered stop: **if the >=1000 band holds fewer than 30 games, Q1 is
UNDERPOWERED BY SIZING** -- report the count and stop. ⛔ Do not pool bands to
reach n; the pooling is the thing being tested.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from p65_archetype_census import MIRROR, scan  # noqa: E402

BANDS = [(0, 800, "<800"), (800, 900, "800-900"),
         (900, 1000, "900-1000"), (1000, 10 ** 9, "1000+")]
ANCHOR_N, ANCHOR_K = 14, 10          # §8ac: 10 of 14


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, dest="dirs")
    ap.add_argument("--us", default="Scio")
    ap.add_argument("--lb", default="out/lb_snapshot_e29.json")
    ap.add_argument("--min-band", type=int, default=30)
    args = ap.parse_args()

    lb = json.loads(Path(args.lb).read_text(encoding="utf-8"))

    per_band: dict[str, Counter] = defaultdict(Counter)
    band_n: Counter = Counter()
    unmatched = 0
    total = 0

    for dname in args.dirs:
        d = ROOT / dname
        if not d.is_dir():
            print(f"  (missing {dname})")
            continue
        for path in sorted(d.glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                got = scan(path)
            except Exception:  # noqa: BLE001
                continue
            if got is None:
                continue
            seats, names = got
            ours = [i for i, nm in enumerate(names)
                    if args.us.lower() in str(nm).lower()]
            if len(ours) != 1:
                continue                      # self-play or not our game
            me = ours[0]
            opp_name = str(names[1 - me])
            opp_label = seats[1 - me].label()
            total += 1
            row = lb.get(opp_name)
            if not row:
                unmatched += 1
                continue
            r = float(row[1])
            for lo, hi, nm in BANDS:
                if lo <= r < hi:
                    per_band[nm][opp_label] += 1
                    band_n[nm] += 1
                    break

    print(f"\n{total} of our ladder games; {unmatched} opponents unmatched to "
          f"the LB ({Path(args.lb).name})")
    print(f"\n=== MIRROR SHARE BY OPPONENT RATING BAND (§8ac's frame) ===")
    print(f"  {'band':<10}{'games':>7}{'mirror':>8}{'share':>9}"
          f"{'95% Wilson CI':>22}")
    for _, _, nm in BANDS:
        n = band_n[nm]
        if not n:
            print(f"  {nm:<10}{0:>7}{'-':>8}{'-':>9}")
            continue
        k = per_band[nm][MIRROR]
        lo, hi = wilson(k, n)
        print(f"  {nm:<10}{n:>7}{k:>8}{k / n:>9.1%}"
              f"{f'[{lo:.1%}, {hi:.1%}]':>22}")

    n = band_n["1000+"]
    k = per_band["1000+"][MIRROR]
    a_lo, a_hi = wilson(ANCHOR_K, ANCHOR_N)
    print(f"\n=== ⛔ Q1 — the pre-registered comparison ===")
    print(f"  §8ac anchor (08-01): {ANCHOR_K}/{ANCHOR_N} = "
          f"{ANCHOR_K / ANCHOR_N:.1%}  CI [{a_lo:.1%}, {a_hi:.1%}]")
    if n < args.min_band:
        print(f"  this re-census:      {k}/{n}")
        print(f"\n  ⛔ UNDERPOWERED BY SIZING — the 1000+ band holds {n} games, "
              f"below the pre-registered {args.min_band}.")
        print(f"  ⛔ Not pooling bands to reach n; the pooling is what is being "
              f"tested (E29 §3).")
        if n:
            lo, hi = wilson(k, n)
            print(f"  For the record only: {k / n:.1%} [{lo:.1%}, {hi:.1%}] — "
                  f"⚠ NOT a refutation and NOT a confirmation.")
        return 0

    lo, hi = wilson(k, n)
    p1, p2 = k / n, ANCHOR_K / ANCHOR_N
    se = math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / ANCHOR_N)
    d = p1 - p2
    print(f"  this re-census:      {k}/{n} = {p1:.1%}  CI [{lo:.1%}, {hi:.1%}]")
    print(f"  difference: {d:+.1%}  [{d - 1.96 * se:+.1%}, {d + 1.96 * se:+.1%}]"
          f"   z = {d / se if se else 0:+.2f}")
    if d + 1.96 * se < 0:
        print("  🔴 REFUTED FROM BELOW — the premise is an overclaim.")
    elif d - 1.96 * se > 0:
        print("  ⚡ the mirror is MORE dominant than believed.")
    else:
        print("  ⚠ UNDERPOWERED — interval contains the anchor. Quote neither "
              "endpoint as hard.")

    print(f"\n=== full archetype composition of the 1000+ band (n={n}) ===")
    for lb_, c in per_band["1000+"].most_common(10):
        print(f"  {lb_:<30}{c:>5}{c / n:>8.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
