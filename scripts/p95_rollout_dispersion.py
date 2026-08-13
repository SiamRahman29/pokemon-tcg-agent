#!/usr/bin/env python
"""E33 follow-up: is the rollout estimator OVER-DISPERSED? ⚠ POST-HOC.

🔴 **Labelled post-hoc on purpose.** E33 pre-registered a MEAN-bias test and
that test answers exactly one question: is the rollout's win probability right
**on average**. It is. But a level-calibrated estimator can still be
over-dispersed — its spread inflated beyond what sampling noise supports — and
over-dispersion is precisely what would inflate the *gap* estimates E17 (+0.0139
per decision) and §8bw (+0.120 scale bar) are denominated in. This diagnostic
was written after seeing E33's mean and is not part of its pre-registration.

**The identity it rests on.** Let `p` be the true win probability from a
position under clone-vs-clone continuation, `y` the realized outcome, and
`p̂` the mean of `R` rollouts. Under the null that rollouts are draws from the
same conditional distribution as the real game:

    Cov(p̂, y) = Var(p)                       (noise in p̂ is independent of y)
    Var(y)    = E[sigma^2] + Var(p)
    Var(p̂)   = Var(p) + E[sigma^2] / R       <- the prediction under H0

    => predicted Var(p̂) = Cov(p̂,y) + (Var(y) - Cov(p̂,y)) / R

**Ratio = observed Var(p̂) / predicted Var(p̂).** Greater than 1 means the
rollout distribution is wider than the real process supports at that position —
the estimator manufactures confidence, which is the mechanism that would let a
per-decision "+0.035" fail to become +0.035 of win rate.

⚠ CI by bootstrap over GAMES, not decisions (§8bw: decisions inside a game
share an outcome; the naive interval was 2.67x too tight in E33 itself).

    python -X utf8 scripts/p95_rollout_dispersion.py --rows "out/logs/e33/*.jsonl"
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
import sys


def load(pattern: str) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(x) for x in fh if x.strip()]
    return [r for r in rows if r.get("y") is not None]


def ratio(rows: list[dict]) -> tuple[float, float, float, float]:
    ph = [r["phat"] for r in rows]
    y = [r["y"] for r in rows]
    n = len(rows)
    mp, my = sum(ph) / n, sum(y) / n
    var_ph = sum((a - mp) ** 2 for a in ph) / (n - 1)
    var_y = sum((b - my) ** 2 for b in y) / (n - 1)
    cov = sum((a - mp) * (b - my) for a, b in zip(ph, y)) / (n - 1)
    R = sum(r.get("r", 6) for r in rows) / n
    var_p = max(cov, 1e-9)                      # Var(p), floored
    pred = var_p + max(var_y - var_p, 0.0) / R
    return var_ph / pred if pred > 0 else float("nan"), var_ph, pred, var_p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="out/logs/e33/*.jsonl")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load(args.rows)
    if not rows:
        sys.exit("no rows")
    by_game: dict[str, list[dict]] = {}
    for r in rows:
        by_game.setdefault(str(r["game"]), []).append(r)
    games = list(by_game)

    point, var_ph, pred, var_p = ratio(rows)
    print(f"decisions            {len(rows)}  over {len(games)} games")
    print(f"Var(p_hat) observed  {var_ph:.5f}")
    print(f"Var(p_hat) predicted {pred:.5f}   (H0: rollouts ~ the real process)")
    print(f"  of which Var(p)    {var_p:.5f}   = Cov(p_hat, y)")
    print(f"RATIO                {point:.4f}   (1.00 = exactly as dispersed "
          f"as the real process supports)")

    rng = random.Random(args.seed)
    boots = []
    for _ in range(args.boot):
        samp: list[dict] = []
        for _ in range(len(games)):
            samp += by_game[games[rng.randrange(len(games))]]
        try:
            boots.append(ratio(samp)[0])
        except Exception:
            pass
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    print(f"  95% CI             [{lo:.4f}, {hi:.4f}]   "
          f"(bootstrap over GAMES, B={len(boots)})")

    print()
    if lo > 1.0:
        print("🔴 OVER-DISPERSED: the rollout spread exceeds what the real "
              "process supports.\n   Gap estimates built on it are inflated; "
              "that is a mechanism for E19's null.")
    elif hi < 1.0:
        print("⚠ UNDER-dispersed -- unexpected; the estimator is more "
              "conservative than reality.")
    else:
        print("✅ CONSISTENT with the real process: no evidence of "
              "over-dispersion.\n   E19's null is NOT explained by an "
              "inflated-spread evaluator either.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
