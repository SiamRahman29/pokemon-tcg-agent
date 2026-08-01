"""Has our field MOVED, or did we just climb into a different one? (day 15)

**Why this exists.** `p9_field_census.py` names the field in one dump. Day 15
had four dumps spanning six days and ~130 rating points of our own climb, and
the pooled shares moved hard: the mirror went 9.3% -> 33.3% while Mega Lucario
went 20.4% -> 4.0%. Those shares are the weights on every anchor in this repo
(HANDOFF rule 16), so the question is not "did the numbers change" but **which
variable changed them**:

  (a) TIME -- the meta genuinely shifted under everyone, or
  (b) BAND -- we climbed ~130 points and TrueSkill handed us a different
      population that was always playing those decks.

They have opposite consequences. (a) means re-weight the anchors and expect to
re-weight them again. (b) means the weights are a function of OUR rating, so
they must be re-derived every time we move -- and the day-9 census was never
wrong, it was describing where we stood.

The discriminator is the same one §8s used for covariate shift: hold one
variable and look along the other. This script buckets every game we have by
**opponent rating** and reports archetype share within each bucket, then splits
the same games by **dump** (= date and our own rating). If share tracks the
rating bucket and not the dump, it is (b).

⚠ Opponent ratings come from a leaderboard snapshot taken NOW, not from the
time of the game. Older dumps therefore carry drifted ratings, which blurs the
bucket assignment; it does not manufacture a trend.

    python -X utf8 scripts/p19_field_drift.py --lb out/lb_snapshot_0801pm.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from p9_field_census import _signature, analyse  # noqa: E402

# our own dumps, oldest first, with the score WE were reading while they played
DUMPS = [
    ("replays/submission_replay_2026-07-29", "07-29 P6a", 845),
    ("replays/submission_optv3", "07-31 v3", 820),
    ("replays/submission_v4", "08-01 v4", 915),
    ("replays/submission_v5", "08-01 v5", 955),
]


def _fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact on [[a,b],[c,d]] -- no scipy dependency."""
    from math import comb
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p(x: int) -> float:
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    obs = p(a)
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * 1.000001))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb", default="out/lb_snapshot_0801pm.json")
    ap.add_argument("--top", type=int, default=6,
                    help="how many archetypes to track across the splits")
    ap.add_argument("--bands", default="0,800,900,1000,9999",
                    help="opponent-rating bucket edges")
    args = ap.parse_args()

    lb = json.loads(Path(args.lb).read_text(encoding="utf-8"))
    edges = [int(x) for x in args.bands.split(",")]

    errs: Counter = Counter()
    games = []          # (dump_label, archetype, opp_rating|None, won)
    for d, label, ourscore in DUMPS:
        if not Path(d).is_dir():
            print(f"  (skipping missing {d})")
            continue
        n = 0
        for path in sorted(Path(d).glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                g = analyse(path, errs)
            except Exception as exc:  # noqa: BLE001
                errs[f"{type(exc).__name__}: {exc}"] += 1
                continue
            if g is None:
                continue
            arch = _signature(g.poke, g.max_copies)
            rating = lb.get(g.opp, [None, None])[1]
            games.append((label, ourscore, arch, rating, g.result > 0))
            n += 1
        print(f"  {label:<12} {n:>4} games  (we read ~{ourscore})")

    tot = len(games)
    counts = Counter(a for _, _, a, _, _ in games)
    top = [a for a, _ in counts.most_common(args.top)]
    print(f"\n{tot} games over {len(set(g[0] for g in games))} dumps; "
          f"tracking {len(top)} archetypes")

    # ---- split 1: by dump (confounds time and our own rating together) ----
    print("\n=== SHARE BY DUMP (time AND our rating move together here) ===")
    labels = [l for _, l, _ in DUMPS if any(g[0] == l for g in games)]
    ns = {l: sum(1 for g in games if g[0] == l) for l in labels}
    print(f"  {'archetype':<26}" + "".join(f"{l:>13}" for l in labels))
    print(f"  {'(n games)':<26}" + "".join(f"{ns[l]:>13}" for l in labels))
    for a in top:
        row = f"  {a:<26}"
        for l in labels:
            k = sum(1 for g in games if g[0] == l and g[2] == a)
            row += f"{k:>6}{k/max(ns[l],1):>7.1%}"
        print(row)
    row = f"  {'-- our win rate --':<26}"
    for l in labels:
        w = sum(1 for g in games if g[0] == l and g[4])
        row += f"{w:>6}{w/max(ns[l],1):>7.1%}"
    print(row)

    # ---- split 2: by opponent rating band (pooled across dumps) ----
    print("\n=== SHARE BY OPPONENT RATING BAND (pooled over ALL dumps) ===")
    print("  If a share tracks THIS split and not the one above, the 'meta "
          "shift' is\n  our own climb: TrueSkill changed the population, not "
          "the population's decks.")
    rated = [g for g in games if g[3] is not None]
    print(f"  {len(rated)}/{tot} games matched to the LB snapshot")
    bands = []
    for lo, hi in zip(edges, edges[1:]):
        sel = [g for g in rated if lo <= g[3] < hi]
        if sel:
            bands.append((f"{lo}-{hi if hi < 9999 else '+'}", sel))
    print(f"  {'archetype':<26}" + "".join(f"{b:>13}" for b, _ in bands))
    print(f"  {'(n games)':<26}" + "".join(f"{len(s):>13}" for _, s in bands))
    for a in top:
        row = f"  {a:<26}"
        for _, sel in bands:
            k = sum(1 for g in sel if g[2] == a)
            row += f"{k:>6}{k/len(sel):>7.1%}"
        print(row)

    # ---- split 3: the decisive one -- hold the band, vary the era ----
    print("\n=== THE DISCRIMINATOR: hold the BAND fixed, compare ERAS ===")
    print("  Old = dumps before 08-01 (v3/P6a), New = the v4+v5 dumps.")
    old = {"07-29 P6a", "07-31 v3"}
    for bname, sel in bands:
        o = [g for g in sel if g[0] in old]
        n_ = [g for g in sel if g[0] not in old]
        if len(o) < 5 or len(n_) < 5:
            print(f"\n  band {bname}: {len(o)} old / {len(n_)} new "
                  f"-- too thin to read")
            continue
        print(f"\n  band {bname}: {len(o)} old vs {len(n_)} new games")
        print(f"    {'archetype':<26}{'old':>14}{'new':>14}{'Fisher p':>11}")
        for a in top:
            ko = sum(1 for g in o if g[2] == a)
            kn = sum(1 for g in n_ if g[2] == a)
            p = _fisher(ko, len(o) - ko, kn, len(n_) - kn)
            flag = "  <-- real" if p < 0.05 else ""
            print(f"    {a:<26}{ko:>6}{ko/len(o):>8.1%}{kn:>6}"
                  f"{kn/len(n_):>8.1%}{p:>11.3f}{flag}")

    # ---- headline: pooled old vs new, uncontrolled, for the record ----
    print("\n=== POOLED old vs new (UNCONTROLLED -- band and era both move) ===")
    o = [g for g in games if g[0] in old]
    n_ = [g for g in games if g[0] not in old]
    print(f"    {'archetype':<26}{'old':>14}{'new':>14}{'Fisher p':>11}")
    for a in top:
        ko = sum(1 for g in o if g[2] == a)
        kn = sum(1 for g in n_ if g[2] == a)
        p = _fisher(ko, len(o) - ko, kn, len(n_) - kn)
        flag = "  <-- real" if p < 0.05 else ""
        print(f"    {a:<26}{ko:>6}{ko/len(o):>8.1%}{kn:>6}"
              f"{kn/len(n_):>8.1%}{p:>11.3f}{flag}")
    ro = [g[3] for g in o if g[3] is not None]
    rn = [g[3] for g in n_ if g[3] is not None]
    import statistics
    print(f"\n  mean opponent rating: old {statistics.mean(ro):.0f} "
          f"(n={len(ro)})  new {statistics.mean(rn):.0f} (n={len(rn)})")
    print("  ^ if these are close, the pooled table above is NOT a band effect")

    if errs:
        print("\nerrors/skips:")
        for k, v in errs.most_common(5):
            print(f"  {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
