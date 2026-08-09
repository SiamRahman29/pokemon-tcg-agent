"""Rule 14 sizing for E13's KO-setup chip target — does the frozen condition fire?

**The condition being sized was frozen in `docs/experiments/E13-ko-setup.md` at
commit 50a6344, before this file existed.** That ordering is the point: the gate
is allowed to kill the rule, and it cannot be widened afterwards to survive.

The condition (E13 §"The intervention"), all clauses required:

  1. select.context in CHIP_CONTEXTS (13/14/15)
  2. every option is a readable OPPONENT-side Pokemon (chip_target's own guard)
  3. an option names their Active, and the net did NOT already pick it
  4. their Active is already damaged (current hp < printed hp)
  5. A = best_damage(our active -> their active) satisfies
         A < hp          (the KO is not already there)
         A >= hp - 30    (this placement is what puts it in reach)

⛔ **Below 0.5 firings/game the rule is dead and is not written** — the gate that
closed Morgrem (0.2), Pokegear (0.27), Archaludon (0.187) and both halves of E12
(0.09 and 0.20).

    python -X utf8 scripts/p74_ko_setup_sizing.py --dir replays/submission_v5_s2

The funnel is printed clause by clause so a kill says WHERE the rule dies rather
than just that it did. Option->Pokemon mapping is `p72`'s, which carries its own
positive control (839/840 over 76 games).
"""
from __future__ import annotations

import argparse
import json
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

from p9_field_census import _signature, analyse  # noqa: E402
from p72_loss_autopsy import (  # noqa: E402
    AREA_ACTIVE, CHIP_CONTEXTS, _pk, _records,
)
from sa import cards, targeting  # noqa: E402

CHIP = targeting.CHIP_DAMAGE          # 30


def _printed_hp(pk: dict) -> int | None:
    hp = cards.card(int(pk["id"])).get("hp")
    return int(hp) if hp else None


def size(dirs: list[str], us: set[str],
         want_arch: str | None = None) -> tuple[Counter, int]:
    f: Counter = Counter()
    games = 0
    errs: Counter = Counter()
    for d in dirs:
        for path in sorted((ROOT / d).glob("*.json")):
            if path.name == "manifest.json":
                continue
            if want_arch:
                # same archetype classifier p73 conditioned its confound check
                # on -- best_damage is deck-specific, so a cross-deck funnel
                # would not be comparing the same arithmetic
                try:
                    g = analyse(path, errs, us)
                except Exception:  # noqa: BLE001
                    continue
                if g is None or want_arch.lower() not in _signature(
                        g.poke, g.max_copies).lower():
                    continue
            try:
                rep = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            names = (rep.get("info") or {}).get("TeamNames") or []
            seats = {i for i, n in enumerate(names) if n in us}
            if not seats:
                continue
            games += 1
            for a in _records(rep):
                state = a["obs"]["current"]
                me = state.get("yourIndex")
                if me not in seats:
                    continue

                # --- clause 1 -------------------------------------------------
                if a["sel"].get("context") not in CHIP_CONTEXTS:
                    continue
                opts = a["sel"].get("option") or []
                if len(opts) < 2 or len(a["picked"]) != 1:
                    continue
                f["c1_damage_select"] += 1

                # --- clause 2: opponent-side and readable ---------------------
                cand = []
                bad = False
                for j, o in enumerate(opts):
                    pi = o.get("playerIndex")
                    pk = _pk(state, o)
                    if pi is None or pi == me or pk is None or pk.get("hp") is None:
                        bad = True
                        break
                    cand.append((j, o, pk))
                if bad or len(cand) < 2:
                    f["x2_mixed_or_unreadable"] += 1
                    continue
                f["c2_opponent_side"] += 1

                # --- clause 3: Active on offer, net did not already take it ---
                act = [(j, o, pk) for j, o, pk in cand
                       if o.get("area") == AREA_ACTIVE]
                if not act or len(act) == len(cand):
                    f["x3_active_not_a_choice"] += 1
                    continue
                f["c3a_active_on_offer"] += 1
                if opts[a["picked"][0]].get("area") == AREA_ACTIVE:
                    f["x3_net_already_chose_active"] += 1
                    continue
                f["c3_net_chose_elsewhere"] += 1

                # --- clause 4: already damaged --------------------------------
                ap_ = act[0][2]
                printed = _printed_hp(ap_)
                if printed is None:
                    f["x4_no_printed_hp"] += 1
                    continue
                hp = int(ap_["hp"])
                if hp >= printed:
                    f["x4_active_undamaged"] += 1
                    continue
                f["c4_active_damaged"] += 1

                # --- clause 5: the KO-setup band ------------------------------
                try:
                    ours = state["players"][me]
                    theirs = state["players"][1 - me]
                    our_act = (ours.get("active") or [None])[0]
                    A = targeting.best_damage(our_act, ours, theirs, ap_)
                except Exception:  # noqa: BLE001
                    f["x5_unreadable"] += 1
                    continue
                if A >= hp:
                    f["x5_already_lethal"] += 1
                    continue
                if A < hp - CHIP:
                    f["x5_out_of_reach"] += 1
                    continue
                f["FIRES"] += 1
    return f, games


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["replays/submission_v5_s2"])
    ap.add_argument("--us", action="append", default=["Scio"])
    ap.add_argument("--gate", type=float, default=0.5)
    ap.add_argument("--arch", default=None,
                    help="restrict to games vs this archetype substring")
    args = ap.parse_args()

    f, games = size(args.dir, set(args.us), args.arch)
    if not games:
        print("no games matched --us; nothing to size")
        return 1

    def line(label: str, key: str, note: str = "") -> None:
        n = f[key]
        print(f"  {label:<34}{n:>7}{n/games:>10.2f}/game   {note}")

    print(f"\n=== E13 KO-setup sizing — {games} games "
          f"({', '.join(args.dir)}) ===")
    print(f"  {'stage':<34}{'count':>7}{'per game':>10}")
    line("1  damage-placement selects", "c1_damage_select", "the denominator")
    line("2  all options opponent-side", "c2_opponent_side")
    line("3a Active was on offer", "c3a_active_on_offer")
    line("3b net chose something else", "c3_net_chose_elsewhere",
         f"({f['x3_net_already_chose_active']} it already took)")
    line("4  their Active was damaged", "c4_active_damaged",
         f"({f['x4_active_undamaged']} at full HP)")
    line("5  FIRES (KO-setup band)", "FIRES",
         f"({f['x5_already_lethal']} already lethal, "
         f"{f['x5_out_of_reach']} out of reach)")

    rate = f["FIRES"] / games
    print(f"\n  firings/game = **{rate:.2f}**   gate = {args.gate}")
    if rate >= args.gate:
        print("  ✅ CLEARS the rule 14 gate — proceed to the rule, the positive "
              "control, then the A/B.")
    else:
        print("  ⛔ BELOW the gate. E13 dies here and no rule is written. Do NOT "
              "drop a clause to raise this number:\n     E13 pre-registered that "
              "a rule without clause 5, or a different chip value, is a SEPARATE "
              "experiment.")

    # Reported for the record only. E13's "Why this form" section pre-registered
    # that the clause-4-only rule is a different experiment, not a fallback.
    plain = f["c4_active_damaged"]
    print(f"\n  (for the record: the plain 'prefer a damaged Active' variant "
          f"would fire {plain} times, {plain/games:.2f}/game — E13 froze this as "
          f"a SEPARATE experiment, not a knob.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
