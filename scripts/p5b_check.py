"""Does `boss_veto` actually fire, and what does it play instead?

Rule 9: a metric that never prints is not a metric that passed. `drag_target`
sat in the audit for days reading zero rows. Before spending an n=2000 A/B on
the veto, confirm the mechanism engages at the rate `p5_audit.py` predicted
(35 of 108 Boss's Orders plays = 32.4%) and look at what the net falls back to.

    python -X utf8 scripts/p5b_check.py --matches 60
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

from ptcg.env import harness  # noqa: E402
from sa import cards as cdb, targeting  # noqa: E402
import arena  # noqa: E402

OPT_NAMES = {7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY",
             13: "ATTACK", 14: "END"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=60)
    args = ap.parse_args()

    _, deck_a = arena.resolve_deck("grimmsnarl")
    _, deck_b = arena.resolve_deck("lucario_v10")
    _, agent_a = arena.build_agent("bc:p5b,veto", deck_a)
    _, agent_b = arena.build_agent("rule:v10,noS", deck_b)

    fired = Counter()
    instead = Counter()
    endrisk = Counter()
    real = targeting.boss_veto

    def spy(obs, chosen, rank):
        out = real(obs, chosen, rank)
        if out is None:
            # was the top pick a Boss's Orders play we let through?
            sel = obs.get("select") or {}
            state = obs.get("current") or {}
            opts = sel.get("option") or []
            if (sel.get("context") == targeting.MAIN and chosen
                    and 0 <= chosen[0] < len(opts)
                    and opts[chosen[0]].get("type") == targeting.OPT_PLAY
                    and state.get("yourIndex") is not None):
                cid = targeting._hand_card_id(
                    state, state["yourIndex"],
                    opts[chosen[0]].get("index") or 0)
                if cid == targeting.BOSS_ORDERS:
                    fired["let through (a KO-able target existed)"] += 1
            return None
        fired["VETOED (nothing on their bench was KO-able)"] += 1
        opts = (obs.get("select") or {}).get("option") or []
        if out and 0 <= out[0] < len(opts):
            o = opts[out[0]]
            t = OPT_NAMES.get(o.get("type"), str(o.get("type")))
            name = "?"
            if t == "PLAY":
                state = obs.get("current") or {}
                cid = targeting._hand_card_id(
                    state, state["yourIndex"], o.get("index") or 0)
                name = cdb.card(cid).get("name", "?") if cid else "?"
            instead[f"{t}" + (f" {name}" if name != "?" else "")] += 1
            # The dangerous fallback: if the runner-up is END while an attack
            # is still payable, the veto did not just skip a Supporter -- it
            # threw the turn away. P5c says the clone never does this on its
            # own, but the clone is not choosing here, the veto is.
            if t == "END":
                atk = any(x.get("type") == 13 for x in opts)
                endrisk["an ATTACK was still on the table" if atk
                        else "no attack available (turn was over anyway)"] += 1
        return out

    targeting.boss_veto = spy
    try:
        for m in range(args.matches):
            if m % 2 == 0:
                harness.play_game(agent_a, agent_b, list(deck_a), list(deck_b))
            else:
                harness.play_game(agent_b, agent_a, list(deck_b), list(deck_a))
    finally:
        targeting.boss_veto = real

    tot = sum(fired.values())
    print(f"\n=== bc:veto, {args.matches} games vs rule:v10,noS ===")
    print(f"\nBoss's Orders plays the net wanted  (n={tot})")
    for k, v in fired.most_common():
        print(f"  {k:<48}{v:>6}{v / tot:>8.1%}" if tot else f"  {k}")
    tot2 = sum(instead.values())
    print(f"\nWhat it played instead  (n={tot2})")
    for k, v in instead.most_common():
        print(f"  {k:<48}{v:>6}{v / tot2:>8.1%}")
    tot3 = sum(endrisk.values())
    print(f"\nOf the fallbacks that were END  (n={tot3})")
    for k, v in endrisk.most_common():
        print(f"  {k:<48}{v:>6}{v / tot3:>8.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
