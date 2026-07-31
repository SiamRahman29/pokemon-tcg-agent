"""B4's cheap probe: how big is the WITHIN-TURN sequence space, really?

**The candidate.** We use **0.1 s of a 600 s pool**. B4 proposes spending it on
enumerating our own turn's action *sequences* and scoring each end-of-turn state
with `evalfn`/`textdmg` -- explicitly NOT the dead game-tree search (no rollouts,
no determinized opponent turns, so the terminal-0/1 variance that killed
`EVIDENCE` §2 does not arise).

**Rule 14 says size it before building it.** This probe answers, in order:

  1. **Is there a decision at all?** Rule 13: a rate over forced moves measures
     nothing. Count turns with >= 2 real (>=2-option) selects, not turns.
  2. **How big is the space?** The naive sequence count is the product of the
     option counts across a turn -- an upper bound, since playing a card changes
     what is legal later.
  3. **Can we afford it?** At the measured per-eval cost, does the p90 turn fit
     in the pool, given ~600 s must cover a whole game (~15-25 of our turns)?
  4. ⚠ **Would ORDER even matter?** The premise is that sequencing beats greedy
     one-select-at-a-time choice. If a turn's actions are commutative -- play
     supporters and attachments in any order, then attack -- then only the SET
     matters and the enumeration buys nothing. This reports how many turns end
     in an attack and how many decisions precede it.

**Kill criterion (pre-registered, from ROADMAP §2.5):** branching intractable
even with beam limits, OR the decision denominator is so small the effect cannot
beat an n=2000 A/B's +/-0.021. Either kills B4 for the price of this probe.

    python -X utf8 scripts/p12_b4_probe.py --matches 100
"""
from __future__ import annotations

import argparse
import statistics
import sys
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

MAIN = int(SelectContext.MAIN)
OPT_ATTACK = 13
CTX = {int(getattr(SelectContext, n)): n
       for n in dir(SelectContext) if n.isupper()}


class Turn:
    __slots__ = ("selects", "real", "product", "main_real", "ended_attack",
                 "ctxs")

    def __init__(self):
        self.selects = 0
        self.real = 0          # selects with >= 2 options
        self.product = 1       # naive sequence-space upper bound
        self.main_real = 0     # real selects that were MAIN
        self.ended_attack = False
        self.ctxs: Counter = Counter()


class Probe:
    """Wraps an agent callable and groups its selects into OUR turns."""

    def __init__(self, inner):
        self.inner = inner
        self.turns: list[Turn] = []
        self._cur: Turn | None = None
        self._key = None
        self.errs: Counter = Counter()

    def _flush(self):
        if self._cur is not None and self._cur.selects:
            self.turns.append(self._cur)
        self._cur = None

    def __call__(self, obs):
        picked = self.inner(obs)
        try:
            sel = obs.get("select") or {}
            state = obs.get("current") or {}
            opts = sel.get("option") or []
            if not sel or not state:
                return picked
            key = (state.get("turn"), state.get("yourIndex"))
            if key != self._key:
                self._flush()
                self._cur = Turn()
                self._key = key
            t = self._cur
            t.selects += 1
            ctx = sel.get("context")
            n = len(opts)
            t.ctxs[CTX.get(ctx, str(ctx))] += 1
            if n >= 2:
                t.real += 1
                # cap the product so one pathological turn cannot dominate
                t.product = min(t.product * n, 10 ** 12)
                if ctx == MAIN:
                    t.main_real += 1
            if picked and ctx == MAIN:
                i = picked[0]
                if 0 <= i < n and opts[i].get("type") == OPT_ATTACK:
                    t.ended_attack = True
        except Exception as exc:  # noqa: BLE001
            self.errs[f"{type(exc).__name__}: {exc}"] += 1
        return picked

    def done(self):
        self._flush()


def _pct(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=100)
    ap.add_argument("--agent", default="bc")
    ap.add_argument("--deck-b", default="lucario_v10")
    ap.add_argument("--opp", default="rule:v10,noS")
    args = ap.parse_args()

    _, deck_a = arena.resolve_deck("grimmsnarl")
    _, deck_b = arena.resolve_deck(args.deck_b)
    _, agent_a = arena.build_agent(args.agent, deck_a)
    _, agent_b = arena.build_agent(args.opp, deck_b)

    probe = Probe(agent_a)
    from ptcg.env import harness
    # Alternate seats: turn shape differs between going first and second
    # (the first player cannot attack on turn 1), so a one-seat sample would
    # understate the decision count.
    for m in range(args.matches):
        if m % 2 == 0:
            harness.play_game(probe, agent_b, list(deck_a), list(deck_b))
        else:
            harness.play_game(agent_b, probe, list(deck_b), list(deck_a))
    probe.done()

    ts = probe.turns
    if not ts:
        print("no turns captured -- check the select grouping")
        return 1

    print(f"\n=== B4 probe: {len(ts)} of OUR turns over {args.matches} games "
          f"vs {args.opp} ===")

    # ---- 1. is there a decision at all? (rule 13) ----------------------------
    forced = sum(1 for t in ts if t.real == 0)
    one = sum(1 for t in ts if t.real == 1)
    multi = [t for t in ts if t.real >= 2]
    print("\n--- 1. THE HONEST DENOMINATOR: turns where SEQUENCING could matter")
    print(f"  every option forced (0 real selects)      {forced:>6}"
          f"{forced/len(ts):>8.1%}")
    print(f"  exactly ONE real select                   {one:>6}"
          f"{one/len(ts):>8.1%}  <- greedy is optimal by definition")
    print(f"  TWO OR MORE real selects                  {len(multi):>6}"
          f"{len(multi)/len(ts):>8.1%}  <- B4's actual denominator")

    # ---- 2. how big is the space? -------------------------------------------
    prods = [t.product for t in multi]
    print("\n--- 2. NAIVE SEQUENCE-SPACE SIZE (product of option counts, "
          "UPPER bound)")
    if prods:
        print(f"  over the {len(prods)} multi-decision turns: "
              f"median {statistics.median(prods):,.0f}   "
              f"p90 {_pct(prods,0.90):,.0f}   "
              f"p99 {_pct(prods,0.99):,.0f}   max {max(prods):,.0f}")
        buckets = Counter()
        for p in prods:
            if p <= 10:
                buckets["<= 10"] += 1
            elif p <= 100:
                buckets["11-100"] += 1
            elif p <= 10_000:
                buckets["101-10k"] += 1
            elif p <= 1_000_000:
                buckets["10k-1M"] += 1
            else:
                buckets["> 1M"] += 1
        for k in ("<= 10", "11-100", "101-10k", "10k-1M", "> 1M"):
            if buckets[k]:
                print(f"    {k:<10}{buckets[k]:>6}{buckets[k]/len(prods):>8.1%}")

    reals = [t.real for t in ts]
    print(f"\n  real selects per turn: median {statistics.median(reals):.0f}  "
          f"p90 {_pct(reals,0.90)}  max {max(reals)}")
    mains = [t.main_real for t in ts]
    print(f"  of which MAIN:         median {statistics.median(mains):.0f}  "
          f"p90 {_pct(mains,0.90)}  max {max(mains)}")

    # ---- 3. would ORDER matter? ---------------------------------------------
    att = sum(1 for t in multi if t.ended_attack)
    print("\n--- 3. WOULD ORDER MATTER? (the premise, not the branching)")
    print(f"  multi-decision turns ending in an attack  {att:>6}"
          f"{att/max(len(multi),1):>8.1%}")
    print("  ⚠ Most pre-attack actions (draw supporters, attach, evolve, bench) "
          "are\n     largely COMMUTATIVE -- if so, only the SET matters and "
          "enumerating\n     orderings buys nothing. Sizing the space is NOT "
          "evidence that order pays.")

    print("\n--- select contexts inside multi-decision turns ---")
    agg: Counter = Counter()
    for t in multi:
        agg.update(t.ctxs)
    for k, v in agg.most_common(8):
        print(f"  {k:<28}{v:>7}")

    if probe.errs:
        print("\nerrors:")
        for k, v in probe.errs.most_common(4):
            print(f"  {v:>5}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
