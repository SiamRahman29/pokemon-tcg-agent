"""R1 sizing gate: is there a "plan" in this corpus that a per-decision clone destroys?

**The hypothesis being sized, not built** (rule 14: size before you build).
§8u is the sharpest number in the repo on why we are not a top player: we cloned
the #2 player **successfully** -- held-out agreement 59.9% -> 67.2% -- and it
measured **-92 Elo**. The reading that survives is that you copy a strong
player's moves without their *plan*, and a partial imitation of a coherent
strategy is worse than a coherent average one. Our net is memoryless and modal,
so it may be averaging several incompatible LINES into an action that belongs to
none of them.

**⛔ Why this is not just "add history features".** §8x already measured the
encoding ceiling at **95.6%** against a clone sitting at **71%** -- we are
UNDERFITTING, not under-expressive, so a feature that merely distinguishes more
states is not where the gap is. The claim here is different in kind: not "the
net cannot tell these states apart" but "the corpus contains several coherent
policies and the net fits their mean".

**The gate, and it is falsifiable.** If demonstrators really run distinct lines,
then knowing which line a seat is on must tell you something about their next
action **that the board state does not already tell you**. So:

    MI(action ; plan-cluster | state-bucket)   vs   the same with plans SHUFFLED

⚠ The conditioning is the whole experiment. Different clusters take different
actions mostly because they are in different *situations*; without bucketing on
the state that difference is guaranteed and means nothing. And the shuffle
baseline is required because MI is biased upward at finite counts -- a raw MI of
0.05 bits is not evidence of anything until the shuffled control is subtracted.

**Kill criterion, pre-registered here:** if conditional MI does not exceed the
shuffled baseline by a clear margin, **the premise of R1 is dead** and no
conditioned net gets built.

    python -X utf8 scripts/p79_plan_audit.py --dir replays/2026-08-05 --games 400
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from p72_loss_autopsy import _nm, _records  # noqa: E402
from sa.optfeat import option_features  # noqa: E402

MAIN = 0


def _opt_card(obs: dict, opt: dict) -> int:
    try:
        return int(option_features(obs, opt)[1] or 0)
    except Exception:  # noqa: BLE001
        return 0


def _turn(st: dict) -> int:
    t = st.get("turn")
    return int(round(t)) if isinstance(t, (int, float)) else -1


def seat_trajectory(recs: list[dict], seat: int) -> tuple[np.ndarray, list[dict]] | None:
    """(line signature, that seat's MAIN decisions) for one seat of one game.

    The signature describes the LINE -- what this pilot built and how fast --
    not the position. Everything is normalised per game so a long game does not
    read as a different plan purely for being long.
    """
    mine = []
    played: Counter = Counter()
    energy_turns = set()
    evolved_turn = None
    bench_max = 0
    prize_pace = []
    n_turns = 0
    for rec in recs:
        st = rec["obs"]["current"]
        if st.get("yourIndex") != seat or st.get("result", -1) != -1:
            continue
        t = _turn(st)
        n_turns = max(n_turns, t)
        pl = st["players"][seat]
        opp = st["players"][1 - seat]
        bench_max = max(bench_max, sum(1 for b in pl["bench"] if b))
        prize_pace.append((t, 6 - len(pl["prize"]), 6 - len(opp["prize"])))
        if st.get("energyAttached"):
            energy_turns.add(t)
        sel = rec["sel"]
        if sel.get("context") != MAIN:
            continue
        opts = sel.get("option") or []
        if len(opts) < 2 or len(rec["picked"]) != 1:
            continue
        cid = _opt_card(rec["obs"], opts[rec["picked"][0]])
        if cid:
            played[cid] += 1
        mine.append({"rec": rec, "turn": t, "cid": cid,
                     "hand": pl["handCount"],
                     "prizes": 6 - len(pl["prize"]),
                     "opp_prizes": 6 - len(opp["prize"]),
                     "n_opts": len(opts)})
        if evolved_turn is None and cid in (1183, 1184):
            evolved_turn = t
    if len(mine) < 6 or n_turns < 3:
        return None

    tot = max(sum(played.values()), 1)
    # The most-played trainers in this archetype -- the levers a pilot chooses
    # between. Rates, so game length is divided out.
    keys = [1219, 1210, 1188, 1178, 1189, 1213, 1187, 1211]
    card_sig = [played.get(k, 0) / tot for k in keys]
    # 🔴 THE CARD RATES ARE CIRCULAR AND MUST BE SEPARABLE. The label this
    # audit predicts is "which card was played", so a cluster defined by card
    # play rates predicts the label partly BY CONSTRUCTION -- that is the same
    # construction as the estimator control, not a finding. `--sig nocards`
    # keeps only shape-of-the-game features, which share no variable with the
    # label, and that arm is the one that can license R1.
    shape_sig = [
        bench_max / 5.0,
        len(energy_turns) / max(n_turns, 1),
        (evolved_turn or n_turns) / max(n_turns, 1),
        prize_pace[-1][1] / 6.0,
        prize_pace[-1][2] / 6.0,
        min(n_turns, 30) / 30.0,
    ]
    return (np.array(card_sig, dtype=np.float64),
            np.array(shape_sig, dtype=np.float64), mine)


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), size=k, replace=False)].copy()
    lab = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = x[m].mean(0)
    return lab


def cond_mi(rows: list[tuple], n_plan: int) -> float:
    """MI(action ; plan | bucket) in bits, estimated by plug-in."""
    by_bucket: dict = defaultdict(list)
    for bucket, plan, act in rows:
        by_bucket[bucket].append((plan, act))
    total = len(rows)
    mi = 0.0
    for bucket, items in by_bucket.items():
        n = len(items)
        if n < n_plan * 4:
            continue                      # too thin to estimate anything
        ja: Counter = Counter(items)
        pa: Counter = Counter(a for _, a in items)
        pp: Counter = Counter(p for p, _ in items)
        acc = 0.0
        for (p, a), c in ja.items():
            pj = c / n
            acc += pj * math.log2(pj / ((pp[p] / n) * (pa[a] / n)))
        mi += (n / total) * acc
    return mi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["replays/2026-08-05"])
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sig", choices=["all", "nocards"], default="nocards",
                    help="'all' includes card play rates (CIRCULAR with the "
                         "label); 'nocards' is the arm that can license R1")
    args = ap.parse_args()

    sigs, seats = [], []
    n = 0
    for d in args.dir:
        for path in sorted((ROOT / d).glob("*.json")):
            if n >= args.games:
                break
            if path.name == "manifest.json":
                continue
            try:
                rep = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            n += 1
            recs = _records(rep)
            rw = rep.get("rewards") or [None, None]
            for seat in (0, 1):
                got = seat_trajectory(recs, seat)
                if got is None:
                    continue
                card_sig, shape_sig, mine = got
                won = None
                if rw[0] is not None and rw[1] is not None and rw[0] != rw[1]:
                    won = 1 if rw[seat] > rw[1 - seat] else 0
                sigs.append(np.concatenate([card_sig, shape_sig])
                            if args.sig == "all" else shape_sig)
                seats.append({"mine": mine, "won": won})
    if len(sigs) < 50:
        print("🔴 too few seats parsed")
        return 1
    x = np.array(sigs)
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    print(f"\n=== R1 PLAN AUDIT — {n} games, {len(sigs)} seats, "
          f"{sum(len(s['mine']) for s in seats)} MAIN decisions ===")

    # The state bucket: what the board already tells you. Plan may only add
    # information ON TOP of this.
    # ⚠ COARSE ON PURPOSE. The first version bucketed on
    # (turn, hand, prizes, opp_prizes, n_opts) and the plug-in MI bias swamped
    # everything -- the shuffled baseline ran to 0.36 bits and the positive
    # control came out NEGATIVE, i.e. the estimator could not tell a real label
    # from a shuffled one. Bias grows with (#buckets x #plans x #actions)/N, so
    # the bucket is the lever.
    def bucket(dec):
        return (min(dec["turn"] // 3, 4), dec["prizes"] - dec["opp_prizes"])

    rng = random.Random(args.seed)
    print(f"\n{'plans':>7}{'sizes':>28}{'MI(act;plan|state)':>21}"
          f"{'shuffled':>11}{'excess':>10}")
    for k in args.k:
        lab = kmeans(x, k, seed=args.seed)
        rows = [(bucket(dec), int(lab[i]), dec["cid"])
                for i, s in enumerate(seats) for dec in s["mine"]]
        real = cond_mi(rows, k)
        sh = []
        for t in range(args.shuffles):
            perm = list(lab)
            rng.shuffle(perm)
            rows_s = [(bucket(dec), int(perm[i]), dec["cid"])
                      for i, s in enumerate(seats) for dec in s["mine"]]
            sh.append(cond_mi(rows_s, k))
        sizes = np.bincount(lab, minlength=k)
        print(f"{k:>7}{str(list(sizes)):>28}{real:>21.4f}"
              f"{np.mean(sh):>11.4f}{real - np.mean(sh):>+10.4f}")

    # ESTIMATOR POSITIVE CONTROL. A label DERIVED FROM THE ACTIONS themselves
    # must show excess MI; if it does not, the estimator is broken and every
    # null above is uninformative. (The `won` control below is a hypothesis
    # control, not an estimator control -- winners and losers may genuinely
    # play alike in the same state, so it is allowed to read zero.)
    petrel = np.array([sum(1 for d in s["mine"] if d["cid"] == 1219)
                       / max(len(s["mine"]), 1) for s in seats])
    hi = (petrel > np.median(petrel)).astype(int)
    rows = [(bucket(dec), int(hi[i]), dec["cid"])
            for i, s in enumerate(seats) for dec in s["mine"]]
    real = cond_mi(rows, 2)
    sh = []
    for _ in range(args.shuffles):
        perm = list(hi)
        rng.shuffle(perm)
        rows_s = [(bucket(dec), int(perm[i]), dec["cid"])
                  for i, s in enumerate(seats) for dec in s["mine"]]
        sh.append(cond_mi(rows_s, 2))
    print(f"\n  ESTIMATOR CONTROL label = plays Petrel above the median rate")
    print(f"    MI {real:.4f}   shuffled {np.mean(sh):.4f}"
          f"   excess {real - np.mean(sh):+.4f}"
          + ("   ✅ estimator sees a real label"
             if real - np.mean(sh) > 0.01 else "   🔴 ESTIMATOR IS BLIND"))

    # Control: the SAME statistic with a plan label we know is real and
    # behaviourally meaningful -- the seat's eventual RESULT. If even this
    # reads ~0 excess, the estimator cannot see plan-like structure at all and
    # the nulls above are uninformative.
    lab_w = [s["won"] for s in seats]
    rows = [(bucket(dec), int(s["won"]), dec["cid"])
            for s in seats if s["won"] is not None for dec in s["mine"]]
    real = cond_mi(rows, 2)
    sh = []
    idx = [i for i, s in enumerate(seats) if s["won"] is not None]
    for t in range(args.shuffles):
        perm = [lab_w[i] for i in idx]
        rng.shuffle(perm)
        rows_s = [(bucket(dec), int(perm[j]), dec["cid"])
                  for j, i in enumerate(idx) for dec in seats[i]["mine"]]
        sh.append(cond_mi(rows_s, 2))
    print(f"\n  POSITIVE-CONTROL label = did this seat WIN")
    print(f"    MI(act ; won | state) {real:.4f}   shuffled {np.mean(sh):.4f}"
          f"   excess {real - np.mean(sh):+.4f}")
    print("    ⚠ this is the ceiling-ish reading for 'a game-level label that")
    print("      really does change behaviour'. Compare the clusters against it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
