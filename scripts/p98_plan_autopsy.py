#!/usr/bin/env python
"""Why does E32's policy play 7.9 cards a game when the clone plays ~16?

`p97` (rung 0) says `plan:pure` fails on exactly one metric -- `play` -- and
that against the real clone it takes **1.275 prizes** to `bc`'s ~3.8 while its
games last 5.4 of its turns. That is a symptom. This script finds the cause,
because the alternative is re-tuning constants by eye, which is the shopping
pattern §8ao already refused a beta-sweep for.

**What it records.** A subclass of `PlanAgent` that intercepts every MAIN
option score. For each option it logs the type, the card id and the score, then
aggregates:

  * per card: how often it was OFFERED, DECLINED (score <= -1) and CHOSEN;
  * the END autopsy -- every turn-ending select, with the list of cards that
    were sitting in the option list and got declined.

⇒ The second is the one that decides the rebuild. Every `_score_play` branch
returns **-1.0** when its condition fails, and END returns **0.0**, so a turn
ends the instant every remaining option is gated off. If the declines are
concentrated in a few conditions, the fix is those conditions. If they are
spread evenly, the ladder itself is wrong and the plan has to bind differently.

⛔ This measures the CURRENT agent. It is a diagnosis, not a comparison, and no
number here is evidence that any change is an improvement -- only the arena is.

    python -X utf8 scripts/p98_plan_autopsy.py --games 120
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "decks", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from ptcg.env import harness  # noqa: E402
from sa import cards  # noqa: E402
from sa.bcagent import PolicyAgent  # noqa: E402
from sa.planagent import PlanAgent  # noqa: E402

O_PLAY, O_ATTACH, O_EVOLVE, O_ABILITY = 7, 8, 9, 10
O_RETREAT, O_ATTACK, O_END = 12, 13, 14
TYPE_NAME = {O_PLAY: "play", O_ATTACH: "attach", O_EVOLVE: "evolve",
             O_ABILITY: "ability", O_RETREAT: "retreat", O_ATTACK: "attack",
             O_END: "END"}


def label(state, me, opt) -> str:
    t = opt.get("type")
    if t != O_PLAY and t != O_EVOLVE:
        return TYPE_NAME.get(t, f"t{t}")
    try:
        from sa.planagent import _card_at
        c = _card_at(state, opt, me)
        cid = c.get("id") if c else None
        if cid is None:
            return TYPE_NAME.get(t, "?")
        nm = (cards.card(cid) or {}).get("name") or str(cid)
        return f"{TYPE_NAME[t]}:{nm}"
    except Exception:
        return TYPE_NAME.get(t, "?")


class Autopsy(PlanAgent):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.offered = Counter()
        self.declined = Counter()
        self.chosen = Counter()
        self.end_with = Counter()      # cards declined at a turn-ending select
        self.ends = 0
        self.ends_all_declined = 0
        self._pending: list[tuple[str, float]] = []

    def _score_main(self, obs, state, me, plan, opt):
        s = super()._score_main(obs, state, me, plan, opt)
        try:
            nm = label(state, me, opt)
            self.offered[nm] += 1
            if s <= -1.0:
                self.declined[nm] += 1
            self._pending.append((nm, float(s)))
        except Exception:
            pass
        return s

    def _decide(self, obs, sel, options):
        self._pending = []
        out = super()._decide(obs, sel, options)
        try:
            # 🔴 `_decide` returns the FULL RANKED LIST, not a pick -- `__call__`
            # applies `_trim` afterwards. Guarding on `len(out) == 1` silently
            # dropped ~75% of the observations and made END look 3x rarer than
            # it is. The chosen option is always `out[0]`.
            if self._pending and out:
                i = int(out[0])
                if 0 <= i < len(options):
                    nm = label(obs.get("current") or {},
                               (obs.get("current") or {}).get("yourIndex"),
                               options[i])
                    self.chosen[nm] += 1
                    if options[i].get("type") == O_END:
                        self.ends += 1
                        dec = [n for n, sc in self._pending
                               if sc <= -1.0 and n != "END"]
                        for n in dec:
                            self.end_with[n] += 1
                        others = [sc for n, sc in self._pending if n != "END"]
                        if others and all(sc <= -1.0 for sc in others):
                            self.ends_all_declined += 1
        except Exception:
            pass
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--net", default="out/policy_v5_s2.npz")
    args = ap.parse_args()

    import grimmsnarl
    deck = []
    for cid, k in grimmsnarl.DECKLIST.items():
        deck += [cid] * k

    a = Autopsy(list(deck), fallback=None, pure=True)
    b = PolicyAgent(list(deck), args.net)
    for i in range(args.games):
        if i % 2 == 0:
            harness.play_game(a, b, list(deck), list(deck))
        else:
            harness.play_game(b, a, list(deck), list(deck))

    g = args.games
    print(f"\nplan:pure vs bc  --  {g} games\n")
    print(f"{'option':<28}{'offered':>9}{'declined':>10}{'decl%':>8}"
          f"{'chosen':>8}{'/game':>8}")
    print("-" * 71)
    for nm, off in sorted(a.offered.items(), key=lambda x: -x[1]):
        dec = a.declined[nm]
        ch = a.chosen[nm]
        print(f"{nm:<28}{off:>9}{dec:>10}{100.0 * dec / max(1, off):>7.0f}%"
              f"{ch:>8}{ch / g:>8.2f}")

    print(f"\nturn-ending selects   {a.ends}  ({a.ends / g:.2f}/game)")
    print(f"  of which EVERY other option was declined: "
          f"{a.ends_all_declined} ({100.0 * a.ends_all_declined / max(1, a.ends):.0f}%)")
    print("\ncards sitting DECLINED at the moment the turn ended:")
    for nm, n in a.end_with.most_common(14):
        print(f"  {nm:<30}{n:>7}   {n / g:>6.2f}/game")
    return 0


if __name__ == "__main__":
    sys.exit(main())
