"""Autopsy: does `sa/oracle.py` actually pick the BETTER option?

A sub-0.500 A/B has two incompatible explanations — the clock does not help, or
the clock is inverted — and they demand opposite responses. This separates them
against ground truth the project already owns.

**The test.** E17 stored 50 paired rollouts for each of the net's top-3 options
at 300 real positions (`out/logs/p82_e17_trt_*.json`), so each option's value is
known to ±0.04. Replay those exact positions through the LIVE
`RolloutOracle.choose()` — its own fresh rollouts, its own arm ordering, its own
argmax — and score **the option it returns** against E17's stored means.

    true_gain = value(oracle's pick) − value(arm 0)     [E17's stored values]

  > 0  the live code selects better options: the implementation is sound and a
       null A/B is about the CLOCK, not the wiring.
  ≈ 0  it is choosing at chance — a real bug, or the budget is too small here.
  < 0  🔴 INVERTED. It systematically prefers the worse option, which is exactly
       what a sign error in the rollout's win/loss orientation would produce,
       and exactly what would drag an A/B under 0.500.

⚠ This validates SELECTION, not the live game plumbing (option indices,
`want`, the harness contract). Those are covered by the A/B's own health line.

    python -X utf8 scripts/p84_oracle_autopsy.py --positions 40
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from sa import policynet as pnet  # noqa: E402
from sa.oracle import STATS, RolloutOracle  # noqa: E402
from p82_e17_self_oracle import attach_obs  # noqa: E402


def records() -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(ROOT / "out/logs/p82_e17_trt_*.json"))):
        out += json.load(open(p, encoding="utf-8"))["records"]
    return out


def ci(xs: list[float], label: str) -> tuple[float, float, float]:
    k = len(xs)
    if k < 3:
        print(f"  {label}: too few ({k})")
        return float("nan"), float("nan"), float("nan")
    m = statistics.fmean(xs)
    se = statistics.stdev(xs) / math.sqrt(k)
    print(f"  {label:<44s} {m:+.4f} [{m - 1.96 * se:+.4f}, "
          f"{m + 1.96 * se:+.4f}]  k={k}")
    return m, m - 1.96 * se, m + 1.96 * se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=40)
    ap.add_argument("--net", default="out/policy_v5_s2.npz")
    ap.add_argument("--sel", type=int, default=20)
    args = ap.parse_args()

    npz = ROOT / args.net
    net = pnet.load(str(npz))
    if net is None:
        print(f"🔴 could not load {npz}")
        return 1

    recs = records()
    # E17 kept only gid/step/turn/nopt/scores/arms/vals; the observation has to
    # come back out of the replay.
    keyed = [r | {"seat": 0} for r in recs]
    for r in keyed:
        r["seat"] = r.get("seat", 0)
    # our own seat is recovered by attach_obs via (gid, step); seat 0/1 is
    # stored per record in the collection, so re-derive it from the replay.
    from p80_rollout_feasibility import load_games, our_seat  # noqa: E402
    seat_of: dict[str, int] = {}
    for f, rep in load_games(ROOT / "replays/submission_v5_s2", 10 ** 9):
        s = our_seat(rep)
        if s is not None:
            seat_of[f.stem] = s
    keyed = [r | {"seat": seat_of[r["gid"]]} for r in keyed
             if r["gid"] in seat_of]
    sample = keyed[:args.positions]
    withobs = attach_obs(sample, ROOT / "replays/submission_v5_s2")
    print(f"positions replayed through the LIVE oracle: {len(withobs)}")

    from decks.grimmsnarl import DECKLIST  # noqa: E402
    deck = [c for c, n in DECKLIST.items() for _ in range(n)]

    orc = RolloutOracle(deck, net=net, arms=3, probe=0, r_sel=args.sel,
                        max_opts=99, min_opts=2, tau=0.0,
                        decision_cap_s=60.0, reserve_s=0.0)

    gains, chance, agree, best_gain = [], [], 0, []
    changed, chg_gain, kept_gain = [], [], []
    for r in withobs:
        vals = np.asarray(r["vals"], dtype=float)          # A x R, E17's truth
        truth = {int(a): float(vals[j].mean())
                 for j, a in enumerate(r["arms"])}
        a0 = int(r["arms"][0])
        pick = orc.choose(r["obs"], [a0])
        chosen = a0 if pick is None else int(pick[0])
        if chosen not in truth:
            continue                       # picked outside E17's measured arms
        gains.append(truth[chosen] - truth[a0])
        best = max(truth, key=lambda k: truth[k])
        agree += chosen == best
        best_gain.append(truth[best] - truth[a0])
        # chance baseline: what a coin flip among the same arms would score
        chance.append(statistics.fmean([truth[a] - truth[a0] for a in truth]))
        # ---- what the extra time CHANGED --------------------------------
        # `scores` holds the net's score for arms[0..3] in ITS ranking order,
        # so scores[j] is the pre-search agent's own opinion of arms[j].
        if chosen != a0:
            j = list(r["arms"]).index(chosen)
            sc = r.get("scores") or []
            changed.append(float(sc[0] - sc[j]) if j < len(sc) else float("nan"))
            chg_gain.append(truth[chosen] - truth[a0])
        else:
            kept_gain.append(truth[best] - truth[a0])

    if not gains:
        print("🔴 no comparable decisions")
        return 1

    print(f"\n[selection quality, scored on E17's STORED values]")
    m, lo, hi = ci(gains, "true gain of the LIVE oracle's pick")
    ci(chance, "chance baseline (uniform over the same arms)")
    ci(best_gain, "perfect oracle over the same arms (ceiling)")
    print(f"  picked E17's best arm on {agree}/{len(gains)} = "
          f"{agree / len(gains):.0%} (chance = {100 / 3:.0f}%)")

    # ---- what the extra time CHANGED -----------------------------------
    n = len(gains)
    print(f"\n[what the search CHANGED vs the pre-search agent]")
    print(f"  different choice on {len(changed)}/{n} decisions "
          f"= {len(changed) / n:.0%}")
    if changed:
        good = sum(1 for g in chg_gain if g > 0)
        ci(chg_gain, "true gain WHEN it overruled")
        print(f"    of those overrules, {good}/{len(chg_gain)} "
              f"= {good / len(chg_gain):.0%} were actually improvements")
        fin = [c for c in changed if c == c]
        if fin:
            print(f"  how the PRE-SEARCH net scored the option search took:")
            print(f"    mean score margin below the net's own top-1: "
                  f"{statistics.fmean(fin):+.2f}")
            for lo_, hi_, lab in ((-1e9, 0.5, "<0.5  (net nearly agreed)"),
                                  (0.5, 1.5, "0.5-1.5"),
                                  (1.5, 3.0, "1.5-3.0"),
                                  (3.0, 1e9, ">3.0  (net strongly disagreed)")):
                k = sum(1 for c in fin if lo_ <= c < hi_)
                if k:
                    print(f"      {lab:<28s} {k:>3d}  ({k / len(fin):.0%})")
    if kept_gain:
        ci(kept_gain, "value LEFT ON THE TABLE when it kept the net's pick")

    # 🔴 The AGREEMENT RATE is the decisive test of "does selection work", not
    # the mean gain. The mean is diluted by every position where the arms are
    # genuinely close (most of them), so it is underpowered by construction;
    # picking the best of 3 arms at 67% against a 33% null is not.
    k3 = len(gains)
    p_hit = agree / k3
    z = (p_hit - 1 / 3) / math.sqrt((1 / 3) * (2 / 3) / k3)
    print(f"\n  selection test: {agree}/{k3} best-arm picks vs a 1/3 null "
          f"⇒ z = {z:.1f}")

    print("\n" + "=" * 70)
    if z > 3.0 and m > 0:
        print(f"✅ THE ORACLE SELECTS BETTER OPTIONS (z={z:.1f}). The "
              f"implementation is sound, so a null A/B is a fact about the "
              f"CLOCK — per-decision value not composing into a win rate — "
              f"not about the wiring.")
    elif hi < 0 or (z > 3.0 and m < 0):
        print("🔴 INVERTED. The live code systematically prefers the WORSE "
              "option. Check the rollout's win/loss orientation (`me` vs "
              "`current.result`) and the arm ordering before reading the A/B "
              "as evidence about the clock at all.")
    elif lo > 0:
        print("✅ THE ORACLE SELECTS BETTER OPTIONS. The implementation is "
              "sound, so a null A/B is a fact about the CLOCK — per-decision "
              "value not composing into a win rate — not about the wiring.")
    elif hi < 0:
        print("🔴 INVERTED. The live code systematically prefers the WORSE "
              "option. Check the rollout's win/loss orientation (`me` vs "
              "`current.result`) and the arm ordering before reading the A/B "
              "as evidence about the clock at all.")
    else:
        print("🟡 INDISTINGUISHABLE FROM CHANCE at this sample. Either the "
              "budget is too small to select here, or selection is broken. "
              "Raise --positions/--sel before concluding anything.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
