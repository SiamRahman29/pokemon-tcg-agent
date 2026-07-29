"""Which selects does the clone decide BLIND, and how big is each?

The two rules that won (`chip_target`, `energy_spread`) both patched the same
thing: `optfeat` gives the net no HP, no damage and no attached-energy count,
so any select whose right answer is that arithmetic is decided at chance. Both
were found by naming one such select and counting it. This enumerates the rest.

For every select the agent faces, bucket by (context, are the options ours /
theirs / mixed) and report how many have >= 2 options -- i.e. how many are a
real decision. That is the denominator any future rule has to work with.

    python -X utf8 scripts/p6_recon.py --matches 60
"""
from __future__ import annotations

import argparse
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
from ptcg.env import harness  # noqa: E402
import arena  # noqa: E402

_ACTIVE, _BENCH = 4, 5
CTX = {int(getattr(SelectContext, n)): n
       for n in dir(SelectContext) if n.isupper()}


class Probe:
    def __init__(self, inner):
        self.inner = inner
        self.buckets = Counter()
        self.src = Counter()      # own-side damage-counter source picks

    def __call__(self, obs):
        picked = self.inner(obs)
        try:
            sel = obs.get("select") or {}
            state = obs.get("current") or {}
            opts = sel.get("option") or []
            if not sel or not state or len(opts) < 2:
                return picked
            me = state["yourIndex"]
            sides = {o.get("playerIndex") for o in opts}
            if sides == {me}:
                side = "ours"
            elif me not in sides and None not in sides:
                side = "theirs"
            elif sides == {None}:
                side = "no side"
            else:
                side = "mixed"
            ctx = CTX.get(sel.get("context"), str(sel.get("context")))
            self.buckets[f"{ctx:<26} {side}"] += 1

            # The Adrena-Brain SOURCE pick: which of OUR Pokemon the counters
            # come off. `chip_target` returns None here by design, so the net
            # decides it with no HP feature at all. Moving counters off a
            # Pokemon that only HAS one counter moves 10 damage instead of 30
            # -- a dominated option, the shape that won twice.
            if (sel.get("context") in (13, 14) and side == "ours"
                    and picked):
                cnts = []
                for o in opts:
                    pk = _at(state, me, o.get("area"), o.get("index") or 0)
                    cnts.append(0 if pk is None
                                else max(0, (pk.get("maxHp") or 0) - pk["hp"]))
                if len(set(cnts)) > 1:
                    got = cnts[picked[0]] if 0 <= picked[0] < len(cnts) else 0
                    best = max(cnts)
                    self.src["took the MOST-damaged source" if got >= best
                             else f"took {got} dmg when {best} was available"] += 1
                else:
                    self.src["all sources carry the same damage"] += 1
        except Exception as e:  # noqa: BLE001
            self.buckets[f"ERR {type(e).__name__}: {e}"] += 1
        return picked


def _at(state, player, area, index):
    try:
        pl = state["players"][player]
        if area == _ACTIVE:
            act = pl["active"]
            return act[0] if act and act[0] is not None else None
        if area == _BENCH:
            b = pl["bench"]
            return b[index] if 0 <= index < len(b) else None
    except (KeyError, IndexError, TypeError):
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=60)
    ap.add_argument("--agent", default="bc")
    args = ap.parse_args()

    _, deck_a = arena.resolve_deck("grimmsnarl")
    _, deck_b = arena.resolve_deck("lucario_v10")
    _, agent_a = arena.build_agent(args.agent, deck_a)
    _, agent_b = arena.build_agent("rule:v10,noS", deck_b)
    probe = Probe(agent_a)

    for m in range(args.matches):
        if m % 2 == 0:
            harness.play_game(probe, agent_b, list(deck_a), list(deck_b))
        else:
            harness.play_game(agent_b, probe, list(deck_b), list(deck_a))

    tot = sum(probe.buckets.values())
    print(f"\n=== selects with >=2 options, {args.matches} games (n={tot}) ===")
    for k, v in probe.buckets.most_common(30):
        print(f"  {k:<40}{v:>7}{v / tot:>8.1%}")
    t2 = sum(probe.src.values())
    print(f"\nAdrena-Brain SOURCE pick -- which of OURS loses the counters "
          f"(n={t2})")
    for k, v in probe.src.most_common():
        print(f"  {k:<40}{v:>7}{v / t2:>8.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
