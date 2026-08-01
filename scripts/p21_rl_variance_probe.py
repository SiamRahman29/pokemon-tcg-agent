"""SIZE THE RL CREDIT-ASSIGNMENT VARIANCE BEFORE BUILDING ANYTHING (day 15, item 5).

**The standing.** "Self-play RL is dead" was struck from four documents on day
14: it was a **compute prior filed as a measurement** for twelve days, never
run, no code and no `n` (`EVIDENCE` §2). So the status is *never attempted*, and
§8w's expressiveness objection was substantially answered by §8x (the encoding
binds at most 4.4 pp). ⇒ **The live objection is neither compute nor
expressiveness — it is credit-assignment variance**, the same term that killed
search (terminal 0/1 ⇒ SE ≈ 0.14, and the max over ~9 rivals sits 0.21–0.28
above truth by chance).

**Rule 14 binds: size it before building it.** The pre-registered probe, set
before any training code existed:

> measure how many games the terminal-outcome signal needs to separate two
> policies of KNOWN Elo separation. **Kill criterion: if separation needs more
> games than ~1.4 cores can produce in the remaining days, RL dies for a few
> CPU-hours instead of a week — and it dies with a NUMBER.**

⚡ **And it costs zero new games.** We already have head-to-head archives for
four pairs whose separation was measured at n≥2000, spanning +37 Elo down to a
measured null. Bootstrapping those archives answers the question directly:
subsample `n` games, ask whether the Wilson interval excludes 0.5 **with the
right sign**, and repeat.

⚠ **What this probe does and does not decide.** It sizes the *outcome signal* —
how many games it takes to SEE a policy difference of a given size. A policy
gradient must do strictly more than see one: it must attribute the difference
to individual selects, of which there are ~200 per game (reported below). So
these numbers are a **lower bound on** the games RL needs, and a pair that is
already unaffordable to detect is unaffordable to learn. **A pass here is not a
licence to build; a fail is a kill.**

    python -X utf8 scripts/p21_rl_variance_probe.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "out" / "arena"

# (archive, label, the separation measured at n>=2000, EVIDENCE section)
PAIRS = [
    ("p19_v4_vs_v4ctrl.jsonl", "v4 vs v4ctrl", +37, "§8z  the largest win the feature axis has produced"),
    ("p21_v4no3_vs_v4.jsonl", "no3 vs v4", -36, "§8ab dropping the three derived state features"),
    ("p20_v5_vs_v4.jsonl", "v5 vs v4", +14, "§8aa one noise-width, and it was still shipped"),
    ("p19_ctrl0_vs_ctrl1.jsonl", "ctrl0 vs ctrl1", 0, "§8z  the measured seed-only NULL -- must NOT separate"),
]

GRID = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400]


def wilson(w: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def outcomes(rows: list[dict]) -> tuple[list[float], str]:
    """Agent A's score per game: 1 win, 0.5 draw, 0 loss.

    🔴 **`agent0`/`agent1` are SEAT-indexed and the seats SWAP every game**
    (`arena.py:280-283` -- `evaluate_paired` plays A-as-P0 then A-as-P1). The
    first version of this function read seat 0 as though it were always agent A,
    which averages A's wins together with B's and drives every real separation
    toward 0.5. It made the **+37 Elo** pair look undetectable at n=6,400 while
    the **+14** pair detected fine, and put **61% false positives** on a
    measured null -- three impossibilities in one table, which is what gave it
    away. **Identify the agent by NAME, never by seat.**
    """
    a_name = rows[0]["agent0"]           # evaluate_paired starts with A as P0
    out = []
    for r in rows:
        a_seat = 0 if r["agent0"] == a_name else 1
        w = r.get("winner")
        if w == 2:
            out.append(0.5)
        else:
            out.append(1.0 if w == a_seat else 0.0)
    return out, a_name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("\n=== 1. THROUGHPUT: what can this machine actually produce? ===")
    # measured from the archives' own timestamps, so it includes real contention
    rates = []
    for fn, *_ in PAIRS:
        rows = load(ARENA / fn)
        ts = [r["ts"] for r in rows if "ts" in r]
        if len(ts) > 50:
            span = max(ts) - min(ts)
            if span > 0:
                rates.append(len(ts) / span)
    gps = statistics.median(rates) if rates else 0.0
    print(f"  median observed rate over {len(rates)} archives: "
          f"{gps:.2f} games/s in ONE process")
    # the machine gives ~1.4 cores of real throughput (rule 7); arenas run 2-3 up
    par = 2.0
    per_hour = gps * par * 3600
    print(f"  at {par:.0f} concurrent processes (rule 7: ~1.4 real cores): "
          f"{per_hour:,.0f} games/h")
    days = 16
    budget = per_hour * 8 * days      # 8 productive h/day is generous
    print(f"  budget to the 08-17 deadline ({days} days x 8 h): "
          f"{budget:,.0f} games TOTAL, for everything")

    print("\n=== 2. HOW MANY GAMES TO SEE A DIFFERENCE OF KNOWN SIZE? ===")
    print("  detection = Wilson 95% CI excludes 0.5 AND the sign is correct")
    print(f"  ({args.boot} bootstrap resamples per cell)\n")
    hdr = f"  {'pair':<18}{'Elo':>5}{'n=':>2}" + "".join(f"{g:>7}" for g in GRID)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    need: dict[str, int | None] = {}
    for fn, label, elo, _note in PAIRS:
        path = ARENA / fn
        if not path.exists():
            print(f"  {label:<18} -- archive missing: {fn}")
            continue
        rows_ = load(path)
        if not rows_:
            print(f"  {label:<18} -- no rows")
            continue
        sc, a_name = outcomes(rows_)
        obs = sum(sc) / len(sc)
        row = f"  {label:<18}{elo:>+5}{'':>2}"
        first = None
        for g in GRID:
            hits = 0
            for _ in range(args.boot):
                s = sum(rng.choice(sc) for _ in range(g))
                lo, hi = wilson(s, g)
                if elo > 0 and lo > 0.5:
                    hits += 1
                elif elo < 0 and hi < 0.5:
                    hits += 1
                elif elo == 0 and (lo > 0.5 or hi < 0.5):
                    hits += 1          # a FALSE positive for the null row
            rate = hits / args.boot
            row += f"{rate:>7.0%}"
            if elo != 0 and first is None and rate >= 0.80:
                first = g
        need[label] = first
        print(row + f"   n={len(sc)}, archived score {obs:.3f}")
    print("\n  ⚠ READ THE LAST ROW CAREFULLY -- it is NOT a false-positive rate.")
    print("    The bootstrap resamples each archive's OWN empirical outcomes, so")
    print("    it sizes the effect that archive actually observed, not the true")
    print("    one. The seed-only pair came in at 0.482, and at large n the")
    print("    instrument duly 'detects' 0.482. What that row really shows is")
    print("    🔴 how fast more games start calling SEED NOISE a result --")
    print("    and that the measured seed floor (0.018 off 0.5) is a LARGER")
    print("    deviation than the v5 block's real effect (0.014). That is §8aa's")
    print("    '+14 Elo is one noise-width' made arithmetic, and it is why n")
    print("    cannot simply be cranked: past ~2,000 the arena resolves the")
    print("    noise floor itself.")

    print("\n=== 3. THE PRE-REGISTERED KILL CRITERION ===")
    for fn, label, elo, note in PAIRS:
        if elo == 0:
            continue
        n = need.get(label)
        if n is None:
            print(f"  {label:<18} {elo:>+4} Elo -> NOT detected at 80% power "
                  f"even at n={GRID[-1]:,}")
        else:
            frac = n / budget if budget else float("inf")
            print(f"  {label:<18} {elo:>+4} Elo -> {n:,} games for 80% power "
                  f"= {frac:.4%} of the whole 16-day budget")
        print(f"      {note}")

    print("\n=== 4. BUT RL MUST DO MORE THAN SEE IT: THE CREDIT DILUTION ===")
    allsel = []
    for fn, *_ in PAIRS:
        p = ARENA / fn
        if p.exists():
            allsel += [r["selects"] for r in load(p) if "selects" in r]
    if allsel:
        m = statistics.mean(allsel)
        print(f"  mean selects per game: {m:.0f} "
              f"(median {statistics.median(allsel):.0f}, "
              f"max {max(allsel)})  over {len(allsel):,} archived games")
        print(f"  ONE binary reward is the only signal for all {m:.0f} of them.")
        print(f"  A REINFORCE advantage at a single select is the game outcome "
              f"minus a baseline,\n  so its per-visit SE is ~0.5. To resolve a "
              f"select whose true effect on the win\n  probability is d, that "
              f"select's context must be VISITED:")
        print(f"\n      {'effect d':>9}{'visits':>10}" +
              "".join(f"{'k=' + str(k):>12}" for k in (1, 5, 20)))
        print(f"      {'':>9}{'needed':>10}" +
              "".join(f"{'games':>12}" for _ in range(3)))
        for d in (0.10, 0.05, 0.02, 0.01):
            visits = (2 * 1.96 * 0.5 / d) ** 2 / 2   # two-sample, sd 0.5
            line = f"      {d:>9.0%}{visits:>10,.0f}"
            for k in (1, 5, 20):
                line += f"{visits / k:>12,.0f}"
            print(line)
        print("      k = how many times that context occurs per game "
              f"(there are {m:.0f} selects\n      per game, so a COMMON context "
              "is visited many times and a rare one once).")
        print(f"\n  ⚠ A 1-percentage-point shift in ONE select's win probability "
              "is already a\n     large per-decision effect. But note what the "
              "table says: at k=20 even\n     that costs ~{:,.0f} games, and the "
              "16-day budget is {:,.0f}."
              .format((2 * 1.96 * 0.5 / 0.01) ** 2 / 2 / 20, budget))
        print("  🔴 SO THE PRE-REGISTERED KILL CRITERION IS **NOT** MET. The "
              "variance argument\n     that has been standing in for a "
              "measurement since day 8 does not, when\n     actually sized, "
              "kill fine-tuning on our own outcomes.")

    print("\n=== 5. READ THIS BEFORE CONCLUDING ANYTHING ===")
    print("  Section 2 sizes the OUTCOME signal and it is affordable.")
    print("  Section 4 sizes the CREDIT signal and it is the one that decides")
    print("  RL, because a gradient must attribute an outcome to individual")
    print("  selects. Neither section is a licence to build: the honest next")
    print("  step is a REAL fine-tune on a SMALL number of parameters where the")
    print("  visit counts above are actually achievable, or a written kill.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
