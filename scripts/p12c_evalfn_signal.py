"""B4's REAL kill criterion: does `evalfn` rank end-of-turn states at all?

**Why this is the decisive probe, not the branching one.** `p12_b4_probe.py`
found the sequence space is ~10^8 per turn and `p12b_step_bench.py` found we can
afford ~78,000 candidates -- so exhaustive search is dead but a **beam is very
affordable**. Throughput is therefore NOT the binding constraint; ROADMAP §2.5
predicted the constraint would be **eval quality**, and this measures it.

A turn-sequencer picks the sequence whose end-of-turn state `evalfn` scores
highest. **If `evalfn` cannot separate good board states from bad ones, that
machinery just finds the state `evalfn` most misjudges** -- a more expensive way
to lose, and precisely the failure mode of the dead rollout search (`EVIDENCE`
§2: "it was selecting noise").

So: record `evaluate(state, me)` at the end of each of our turns, then check
whether it predicts the eventual result of that game.

  * **AUC** over (won, lost) pairs at matched turn numbers -- 0.5 is noise.
  * A per-turn-number breakdown, because late-game eval is trivially predictive
    (you are ahead because you are winning) while **early-game** discrimination
    is what a sequencer actually needs.

⚠ **Rule 3 applies with force here**: five times in this project a metric that
looked good failed to predict playing strength. AUC on won/lost games is a
*correlational* check and is the weakest form of evidence we accept -- it can
only KILL B4 cheaply, never bless it. A pass here means "run the real A/B", not
"build the thing".

    python -X utf8 scripts/p12c_evalfn_signal.py --matches 120
"""
from __future__ import annotations

import argparse
import statistics
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

from cg.api import SelectContext  # noqa: E402
import arena  # noqa: E402
from sa.evalfn import evaluate  # noqa: E402

MAIN = int(SelectContext.MAIN)
OPT_ATTACK = 13


class Collector:
    """Records (turn, eval) at the last select of each of our turns."""

    def __init__(self, inner):
        self.inner = inner
        self.games: list[tuple[list[tuple[int, float]], int]] = []
        self._cur: list[tuple[int, float]] = []
        self._last = None
        self._key = None
        self.errs: Counter = Counter()

    def start_game(self):
        self._cur, self._last, self._key = [], None, None

    def end_game(self, result: int):
        if self._last is not None:
            self._cur.append(self._last)
        if self._cur:
            self.games.append((self._cur, result))
        self._cur, self._last, self._key = [], None, None

    def __call__(self, obs):
        picked = self.inner(obs)
        try:
            state = obs.get("current") or {}
            sel = obs.get("select") or {}
            if not state or not sel or state.get("result", -1) != -1:
                return picked
            me = state.get("yourIndex")
            key = (state.get("turn"), me)
            if key != self._key:
                if self._last is not None:
                    self._cur.append(self._last)
                self._key = key
                self._last = None
            self._last = (int(state.get("turn") or 0),
                          float(evaluate(state, me)))
        except Exception as exc:  # noqa: BLE001
            self.errs[f"{type(exc).__name__}: {str(exc)[:60]}"] += 1
        return picked


def auc(pos: list[float], neg: list[float]) -> float:
    """P(random win-game eval > random loss-game eval); 0.5 = no signal."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, s = {}, 0, 0.0
    # average ranks for ties
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank[k] = r
        i = j + 1
    for k, (_v, lab) in enumerate(allv):
        if lab == 1:
            s += rank[k]
    n1, n0 = len(pos), len(neg)
    return (s - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=120)
    ap.add_argument("--deck-b", default="lucario_v10")
    ap.add_argument("--opp", default="rule:v10,noS")
    args = ap.parse_args()

    _, deck_a = arena.resolve_deck("grimmsnarl")
    _, deck_b = arena.resolve_deck(args.deck_b)
    _, agent_a = arena.build_agent("bc", deck_a)
    _, agent_b = arena.build_agent(args.opp, deck_b)

    col = Collector(agent_a)
    from ptcg.env import harness
    for m in range(args.matches):
        col.start_game()
        if m % 2 == 0:
            r = harness.play_game(col, agent_b, list(deck_a), list(deck_b))
            col.end_game(1 if r.winner == 0 else 0)
        else:
            r = harness.play_game(agent_b, col, list(deck_b), list(deck_a))
            col.end_game(1 if r.winner == 1 else 0)

    if not col.games:
        print("no games collected")
        return 1

    wins = sum(1 for _s, r in col.games if r == 1)
    print(f"\n=== evalfn signal: {len(col.games)} games, "
          f"{wins} won ({wins/len(col.games):.1%}) vs {args.opp} ===")

    by_turn: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))
    for seq, res in col.games:
        for t, v in seq:
            by_turn[t][0 if res == 1 else 1].append(v)

    print("\n--- AUC by turn number (0.50 = no signal; >0.5 = eval predicts "
          "the win) ---")
    print(f"  {'turn':>5}{'n(win)':>8}{'n(loss)':>9}{'mean win':>11}"
          f"{'mean loss':>11}{'AUC':>8}")
    rows = []
    for t in sorted(by_turn):
        p, n = by_turn[t]
        if len(p) < 12 or len(n) < 12:
            continue
        a = auc(p, n)
        rows.append((t, a, len(p), len(n)))
        print(f"  {t:>5}{len(p):>8}{len(n):>9}{statistics.mean(p):>11.2f}"
              f"{statistics.mean(n):>11.2f}{a:>8.3f}")

    if rows:
        early = [a for t, a, _, _ in rows if t <= 8]
        late = [a for t, a, _, _ in rows if t > 8]
        print("\n--- the number that decides B4 ---")
        if early:
            print(f"  EARLY turns (<=8):  mean AUC {statistics.mean(early):.3f}"
                  f"  over {len(early)} turn-numbers")
        if late:
            print(f"  late  turns  (>8):  mean AUC {statistics.mean(late):.3f}"
                  f"  over {len(late)} turn-numbers")
        print("\n  ⚠ Late-game AUC is nearly free -- a winning board IS a high "
              "eval.\n     A sequencer needs EARLY discrimination, where the "
              "turn it is choosing\n     still changes the outcome. Judge B4 on "
              "the early row.")

    if col.errs:
        print("\nerrors:")
        for k, v in col.errs.most_common(4):
            print(f"  {v:>5}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
