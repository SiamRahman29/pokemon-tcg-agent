#!/usr/bin/env python
"""Does V work at INFERENCE, and does it discriminate WITHIN a position?

The `vlp` smoke read 0/10 with errors=0. `p86` ruled out the seat: indexing is
absolute. Two explanations remain and they demand opposite responses:

  **(a) plumbing** -- V is fine in training and broken through the live path
  (feature mismatch, bag construction, forked-state scaling). ⇒ fix it.
  **(b) no within-position signal** -- V ranks *games* (AUC 0.827 held out) but
  cannot rank the ~6 successors of ONE position, so argmax over siblings is
  noise, and overruling the clone on noise at ~7 decisions a game loses almost
  every game. ⇒ E20's H-eval is refuted, exactly as B4's probe warned:
  *"a high across-game AUC is compatible with zero within-turn discrimination."*

This separates them with three readings taken during real play:

  1. **live AUC** of V(state at the decision) against the eventual result. If
     the inference path is sound this must land near training's 0.827; near
     0.50 means (a).
  2. **sibling spread** -- the sd of V across the one-ply successors of the
     SAME position, against the across-position sd. This is the quantity B4
     named and never measured.
  3. **best-sibling lift** -- does the successor V ranks first actually win
     more often than the one the clone picked? Ground truth is the realized
     game result, so this is only readable when V's pick == the clone's pick,
     which is itself the honest limit of an on-policy probe and is reported.

    python -X utf8 scripts/p87_vlook_value_probe.py --games 20 \\
        --vnet out/value_v1.npz
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "."):
    sys.path.insert(0, str(ROOT / sub))

from ptcg.env import sdk  # noqa: E402

sdk.load()

from ptcg.env import harness  # noqa: E402
from sa import fastsearch as fs  # noqa: E402
from sa import policynet as pnet  # noqa: E402
from sa import valuenet as vnet  # noqa: E402
from sa.worlds import determinize  # noqa: E402

MAIN = 0
ROWS: list[dict] = []


def auc(y: np.ndarray, p: np.ndarray) -> float:
    m = y != 0.5
    y, p = y[m], p[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    r = np.empty(len(p), dtype=float)
    r[order] = np.arange(1, len(p) + 1)
    npos, nneg = float((y == 1).sum()), float((y == 0).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


class Probe:
    """Plays the pure clone. It only OBSERVES -- V never changes a move, so the
    realized outcome stays a valid label for the state V scored (an agent that
    acted on V would make its own label)."""

    def __init__(self, decklist, V, tag):
        self.decklist = list(decklist)
        self.V = V
        self.tag = tag
        self.rng = random.Random(0)
        self.mine: list[dict] = []

    def __call__(self, obs):
        net = pnet.get()
        sel = obs.get("select")
        if sel is None:
            return list(self.decklist)
        picked = net.choose(obs) if net else list(range(sel.get("minCount", 1)))
        try:
            cur = obs.get("current") or {}
            if not (sel.get("context") == MAIN and obs.get("search_begin_input")
                    and cur.get("result", -1) == -1
                    and sel.get("maxCount", 1) == 1
                    and cur.get("turn", 0) >= 2):
                return picked
            opts = sel.get("option") or []
            if not (2 <= len(opts) <= 12):
                return picked
            me = cur["yourIndex"]
            v_here = float(self.V.win_prob(cur, me))

            # one ply into every sibling, on ONE shared world
            w = determinize(obs, self.decklist, [], self.rng)
            vals = []
            for j in range(len(opts)):
                root = None
                try:
                    root, o = fs.begin(
                        obs["search_begin_input"],
                        [] if sel.get("deck") is not None else w.my_deck,
                        w.my_prize, w.opp_deck, w.opp_prize, w.opp_hand,
                        w.opp_active)
                    _s, o2 = fs.step(root, [j])
                    c2 = o2.get("current")
                    if c2 is None:
                        vals.append(None)
                    elif c2.get("result", -1) != -1:
                        r = c2["result"]
                        vals.append(0.5 if r == 2 else (1.0 if r == me else 0.0))
                    else:
                        vals.append(float(self.V.win_prob(c2, me)))
                except Exception:
                    vals.append(None)
                finally:
                    if root is not None:
                        try:
                            fs.end()
                        except Exception:
                            pass
            good = [v for v in vals if v is not None]
            if len(good) < 2:
                return picked
            self.mine.append({
                "v_here": v_here,
                "sib_sd": float(np.std(good)),
                "sib_range": float(max(good) - min(good)),
                "argmax_is_clone": int(
                    int(max(range(len(vals)),
                            key=lambda i: (vals[i] if vals[i] is not None
                                           else -1))) == int(picked[0])),
                "n_opts": len(good),
                "seat": me,
            })
        except Exception as e:
            print("ERR", type(e).__name__, e, file=sys.stderr)
        return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--vnet", default="out/value_v1.npz")
    args = ap.parse_args()

    V = vnet.load(args.vnet)
    if V is None:
        sys.exit(f"value net failed to load (or failed the dim guard): {args.vnet}")

    import importlib
    deckmod = importlib.import_module("decks.grimmsnarl")
    deck = [cid for cid, n in deckmod.DECKLIST.items() for _ in range(n)]

    for g in range(args.games):
        a = Probe(deck, V, "a")
        b = Probe(deck, V, "b")
        r = harness.play_game(a, b, deck, deck)
        for seat, pr in ((0, a), (1, b)):
            won = (0.5 if r.winner == 2 else (1.0 if r.winner == seat else 0.0))
            for row in pr.mine:
                row["won"] = won
                ROWS.append(row)
        print(f"game {g}: winner={r.winner} rows={len(ROWS)}", flush=True)

    y = np.array([r["won"] for r in ROWS], dtype=float)
    v = np.array([r["v_here"] for r in ROWS], dtype=float)
    sd = np.array([r["sib_sd"] for r in ROWS], dtype=float)
    rng_ = np.array([r["sib_range"] for r in ROWS], dtype=float)
    agree = np.array([r["argmax_is_clone"] for r in ROWS], dtype=float)
    nop = np.array([r["n_opts"] for r in ROWS], dtype=float)

    print(f"\n=== V through the LIVE inference path, {len(ROWS)} decisions ===")
    print(f"live AUC (V at the decision vs eventual result)   {auc(y, v):.4f}")
    print(f"  training AUC on held-out self-play games         0.8270")
    print(f"  ⇒ {'PLUMBING IS SOUND' if auc(y, v) > 0.70 else '🔴 PLUMBING IS BROKEN'}")
    print(f"\nV spread ACROSS positions (sd)                    {v.std():.4f}")
    print(f"V spread WITHIN a position (mean sd over siblings) {sd.mean():.4f}")
    print(f"  mean sibling range (max-min)                     {rng_.mean():.4f}")
    print(f"  ratio within/across                              {sd.mean()/max(v.std(),1e-9):.3f}")
    print(f"\nV's argmax == the clone's pick                    {agree.mean():.1%}"
          f"  (chance ≈ {(1/nop).mean():.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
