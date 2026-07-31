"""How fast can we actually simulate a turn? -- the second half of B4's probe.

`p12_b4_probe.py` sized the space (median ~10^8 sequences per multi-decision
turn). Whether that matters depends entirely on throughput, so measure it
instead of assuming it: this times `fastsearch.begin` + `fastsearch.step` on
REAL mid-game observations, which is the exact machinery a turn-sequencer would
run on.

⚠ Note what is being timed. `begin` pays for a determinization (it must invent
the opponent's hidden zones); `step` advances one select. A turn-sequencer pays
**one `begin` per turn** and **one `step` per action per candidate sequence**, so
those two costs enter the budget very differently.

    python -X utf8 scripts/p12b_step_bench.py --matches 6
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from cg.api import SelectContext  # noqa: E402
import arena  # noqa: E402
from sa import fastsearch as fs  # noqa: E402
from sa.worlds import determinize  # noqa: E402

MAIN = int(SelectContext.MAIN)


class Bench:
    def __init__(self, inner, decklist, per_game=6):
        self.inner = inner
        self.decklist = decklist
        self.per_game = per_game
        self.rng = random.Random(7)
        self.begin_ms: list[float] = []
        self.step_ms: list[float] = []
        self.errs: Counter = Counter()
        self._done_this_game = 0

    def new_game(self):
        self._done_this_game = 0

    def __call__(self, obs):
        picked = self.inner(obs)
        try:
            sel = obs.get("select") or {}
            if (sel.get("context") != MAIN
                    or self._done_this_game >= self.per_game
                    or "search_begin_input" not in obs):
                return picked
            state = obs.get("current") or {}
            if not state or state.get("result") != -1:
                return picked
            sbi = obs["search_begin_input"]
            deck_visible = sel.get("deck") is not None

            t0 = time.perf_counter()
            world = determinize(obs, self.decklist, [], self.rng)
            sid, _o = fs.begin(sbi,
                               [] if deck_visible else world.my_deck,
                               world.my_prize, world.opp_deck,
                               world.opp_prize, world.opp_hand,
                               world.opp_active)
            self.begin_ms.append((time.perf_counter() - t0) * 1000)
            self._done_this_game += 1

            # Walk this turn forward the way a sequencer would: take the
            # engine's first legal option repeatedly and time each advance.
            cur_sid, cur = sid, _o
            for _ in range(12):
                s = cur.get("select") or {}
                opts = s.get("option") or []
                st = cur.get("current") or {}
                if not opts or st.get("result", -1) != -1:
                    break
                t1 = time.perf_counter()
                try:
                    cur_sid, cur = fs.step(cur_sid, [0])
                except fs.SearchError:
                    break
                self.step_ms.append((time.perf_counter() - t1) * 1000)
            fs.release(sid)
        except Exception as exc:  # noqa: BLE001
            self.errs[f"{type(exc).__name__}: {str(exc)[:60]}"] += 1
        return picked


def _p(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--turns-per-game", type=float, default=9.9,
                    help="our turns per game, measured by p12_b4_probe")
    args = ap.parse_args()

    _, deck_a = arena.resolve_deck("grimmsnarl")
    _, deck_b = arena.resolve_deck("lucario_v10")
    _, agent_a = arena.build_agent("bc", deck_a)
    _, agent_b = arena.build_agent("rule:v10,noS", deck_b)

    bench = Bench(agent_a, list(deck_a))
    from ptcg.env import harness
    for _m in range(args.matches):
        bench.new_game()
        harness.play_game(bench, agent_b, list(deck_a), list(deck_b))
    fs.end()

    if not bench.step_ms:
        print("no steps timed")
        for k, v in bench.errs.most_common(5):
            print(f"  {v:>4}  {k}")
        return 1

    b, s = bench.begin_ms, bench.step_ms
    print(f"\n=== simulation throughput ({len(b)} begins, {len(s)} steps) ===")
    print(f"  fs.begin  median {statistics.median(b):7.2f} ms   "
          f"p90 {_p(b,0.9):7.2f}   (one per turn)")
    print(f"  fs.step   median {statistics.median(s):7.3f} ms   "
          f"p90 {_p(s,0.9):7.3f}   ({1000/statistics.median(s):,.0f}/s)")

    # ---- the budget arithmetic ---------------------------------------------
    pool, turns = 600.0, args.turns_per_game
    per_turn = pool / turns
    step_s = statistics.median(s) / 1000
    begin_s = statistics.median(b) / 1000
    actions = 6            # median real selects per turn (p12 probe)
    usable = max(per_turn - begin_s, 0.0)
    cands = usable / (actions * step_s)
    print(f"\n=== budget arithmetic (600 s pool, {turns:.1f} of our turns/game) ===")
    print(f"  time available per turn                {per_turn:8.1f} s")
    print(f"  minus one determinize+begin            {begin_s:8.3f} s")
    print(f"  cost of ONE candidate sequence "
          f"({actions} actions)  {actions*step_s:8.4f} s")
    print(f"  => candidate sequences affordable/turn  {cands:8,.0f}")
    for label, space in (("median turn", 98_122_752), ("p90 turn", 10**12)):
        print(f"  as a fraction of the {label:<12} "
              f"({space:,} seqs): {cands/space:.2e}")
    print("\n  ⚠ Reaching even this needs the FULL pool every turn, and Kaggle "
          "counts\n     the pool per GAME -- exhausting it is a loss (HANDOFF "
          "section 7).")

    if bench.errs:
        print("\nerrors:")
        for k, v in bench.errs.most_common(5):
            print(f"  {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
