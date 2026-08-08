"""Does the net's damage TARGETING already adapt to the opponent's archetype?

**Why this exists.** §8bm sized the *dominated* half of passive-damage targeting
and killed it: when a KO is on the table the net takes it 99.1% of the time.
But a KO is on the table in ~1% of these decisions. **The other 99% are
tradeoffs** -- damage has to go somewhere and arithmetic alone cannot say where.
That is the regime a matchup-conditional policy ("hit the Active vs Crustle, hit
the weakest thing vs Alakazam") would live in, and §8bm says NOTHING about it.

So, before any rule: **does the net already condition on the matchup?**

- If its target distribution shifts by archetype, it has learned the branch and
  a hand-written one has little room -- which is what happened to `chip_target`,
  whose own docstring records the net concentrating counters correctly unaided.
- If it targets identically no matter who is across the table, that is a
  measured blindness, priced the way §8au priced the embedding defect, and it is
  the strongest case this project could have for the user's proposal.

⚠ **This is a DESCRIPTION, not a verdict.** "The net does X" never implies X is
wrong -- the field it cloned may do X for good reason. A difference here buys a
pre-registered A/B, not a rule (rule 14, and tradeoff rules are 0/5).

    python -X utf8 scripts/p73_target_policy.py --dir replays/submission_v5_s2

Option->Pokemon mapping is `p72`'s, which carries its positive control.
"""
from __future__ import annotations

import argparse
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

from p9_field_census import _signature, analyse  # noqa: E402
from p72_loss_autopsy import (  # noqa: E402
    AREA_ACTIVE, CHIP_CONTEXTS, _hp_map, _nm, _pk, _records,
)
from sa import cards  # noqa: E402

import json  # noqa: E402


def collect(dirs: list[str], us: set[str], want_arch: str | None) -> list[dict]:
    """One row per tradeoff-regime damage decision where the Active was on offer.

    Each row carries the SITUATION (properties of the board that make the Active
    attractive regardless of any policy) and the CHOICE. That separation is the
    whole point: a rate difference between two players is only behavioural if it
    survives holding the situation fixed.
    """
    errs: Counter = Counter()
    rows: list[dict] = []
    for d in dirs:
        for path in sorted((ROOT / d).glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                g = analyse(path, errs, us)
            except Exception:  # noqa: BLE001
                continue
            if g is None:
                continue
            arch = _signature(g.poke, g.max_copies)
            if want_arch and want_arch.lower() not in arch.lower():
                continue
            rep = json.loads(path.read_text(encoding="utf-8"))
            names = (rep.get("info") or {}).get("TeamNames") or []
            seats = {i for i, n in enumerate(names) if n in us}
            if not seats:
                continue
            recs = _records(rep)
            for a, b in zip(recs, recs[1:]):
                state = a["obs"]["current"]
                if state.get("yourIndex") not in seats:
                    continue
                if a["sel"].get("context") not in CHIP_CONTEXTS:
                    continue
                opts = a["sel"].get("option") or []
                if len(opts) < 2 or len(a["picked"]) != 1:
                    continue
                chosen_opt = opts[a["picked"][0]]
                chosen = _pk(state, chosen_opt)
                if not chosen:
                    continue
                s0, s1 = _hp_map(state), _hp_map(b["obs"]["current"])
                ser = chosen["serial"]
                if ser not in s0 or ser not in s1:
                    continue
                dmg = s0[ser] - s1[ser]
                if dmg <= 0:
                    continue
                cand = [(j, o, _pk(state, o)) for j, o in enumerate(opts)]
                cand = [(j, o, p) for j, o, p in cand if p]
                if len(cand) < 2:
                    continue
                if any(0 < p["hp"] <= dmg for _, _, p in cand):
                    continue                      # §8bm's regime, excluded
                act = [(j, o, p) for j, o, p in cand if o.get("area") == AREA_ACTIVE]
                if not act or len(act) == len(cand):
                    continue                      # Active not a live choice here
                ap_ = act[0][2]
                hps = [p["hp"] for _, _, p in cand]
                pv = [cards.prize_value(p["id"]) for _, _, p in cand]
                apv = cards.prize_value(ap_["id"])
                rows.append({
                    "arch": arch,
                    # --- the SITUATION ---
                    "act_is_lowest_hp": ap_["hp"] == min(hps),
                    "act_is_top_prize": apv == max(pv) and pv.count(max(pv)) == 1,
                    "n_options": len(cand),
                    # --- the CHOICE ---
                    "chose_active": chosen_opt.get("area") == AREA_ACTIVE,
                })
    return rows


def confound(ours: list[dict], theirs: list[dict], a_name: str,
             b_name: str) -> None:
    """Does the 'chose Active' gap survive holding the board situation fixed?"""
    def key(r: dict) -> tuple:
        return (r["act_is_lowest_hp"], r["act_is_top_prize"])

    def rate(rs: list[dict]) -> float:
        return sum(r["chose_active"] for r in rs) / len(rs) if rs else float("nan")

    print("\n" + "=" * 78)
    print("=== CONFOUND CHECK: is the 'chose Active' gap behaviour, or board mix? ===")
    print("=" * 78)
    print(f"  raw rates   {a_name}: {rate(ours):.1%} (n={len(ours)})"
          f"   |   {b_name}: {rate(theirs):.1%} (n={len(theirs)})")

    ga: dict = defaultdict(list)
    gb: dict = defaultdict(list)
    for r in ours:
        ga[key(r)].append(r)
    for r in theirs:
        gb[key(r)].append(r)

    print(f"\n  {'Active is lowest HP':<21}{'top prize':<11}"
          f"{'our n':>7}{'our rate':>10}{'their n':>9}{'their rate':>12}{'share us':>10}{'share them':>12}")
    keys = sorted(set(ga) | set(gb), key=lambda k: (-len(gb.get(k, [])),))
    for k in keys:
        A, B = ga.get(k, []), gb.get(k, [])
        print(f"  {str(k[0]):<21}{str(k[1]):<11}{len(A):>7}"
              f"{rate(A):>10.1%}{len(B):>9}{rate(B):>12.1%}"
              f"{len(A)/max(len(ours),1):>10.1%}{len(B)/max(len(theirs),1):>12.1%}")

    # Direct standardisation: our per-bucket rates, THEIR bucket mix. If this
    # lands on their raw rate, the gap was mix. If it stays at ours, behaviour.
    num = den = 0.0
    missing = 0
    for k, B in gb.items():
        A = ga.get(k, [])
        if not A:
            missing += len(B)
            continue
        num += rate(A) * len(B)
        den += len(B)
    adj = num / den if den else float("nan")
    print(f"\n  {a_name} rate re-weighted to {b_name}'s board mix: **{adj:.1%}**"
          f"   ({missing} of their decisions fell in buckets we never saw)")
    print(f"    vs our raw {rate(ours):.1%} and their raw {rate(theirs):.1%}")
    print(f"\n  ⇒ if the adjusted rate sits near OUR raw rate, the gap is "
          f"BEHAVIOURAL.\n    If it moves to THEIRS, the gap was the board mix "
          f"and there is nothing to repair.")

    print(f"\n  option-count mix (a second situational axis):")
    for nm, rs in ((a_name, ours), (b_name, theirs)):
        c = Counter(r["n_options"] for r in rs)
        tot = sum(c.values())
        print(f"    {nm:<12}" + "  ".join(
            f"{k}opt {c[k]/max(tot,1):.0%}" for k in sorted(c)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["replays/submission_v5_s2"])
    ap.add_argument("--us", action="append", default=["Scio"])
    ap.add_argument("--min-games", type=int, default=4,
                    help="hide archetypes rarer than this")
    ap.add_argument("--confound", nargs=2, metavar=("DIR", "TEAM"),
                    help="second dump + its owner's seat name; runs the "
                         "confound check instead of the per-archetype table")
    ap.add_argument("--arch", default="grimmsnarl",
                    help="archetype substring to condition on for --confound")
    args = ap.parse_args()

    if args.confound:
        d2, team = args.confound
        ours = collect(args.dir, set(args.us), args.arch)
        theirs = collect([d2], {team}, args.arch)
        print(f"conditioned on opponent archetype ~ {args.arch!r}: "
              f"{len(ours)} of our decisions, {len(theirs)} of theirs")
        confound(ours, theirs, "us", Path(d2).name[:12])
        return 0

    us = set(args.us)
    errs: Counter = Counter()
    # archetype label per replay path, from the census classifier
    label: dict[Path, str] = {}
    for d in args.dir:
        for path in sorted((ROOT / d).glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                g = analyse(path, errs, us)
            except Exception:  # noqa: BLE001
                continue
            if g is not None:
                label[path] = _signature(g.poke, g.max_copies)

    # per archetype: how the net aims when NO KO is available
    stat: dict[str, Counter] = defaultdict(Counter)
    games_of: Counter = Counter()

    for path, arch in label.items():
        rep = json.loads(path.read_text(encoding="utf-8"))
        names = (rep.get("info") or {}).get("TeamNames") or []
        seats = {i for i, n in enumerate(names) if n in us}
        if not seats:
            continue
        games_of[arch] += 1
        recs = _records(rep)
        for a, b in zip(recs, recs[1:]):
            state = a["obs"]["current"]
            if state.get("yourIndex") not in seats:
                continue
            if a["sel"].get("context") not in CHIP_CONTEXTS:
                continue
            opts = a["sel"].get("option") or []
            if len(opts) < 2 or len(a["picked"]) != 1:
                continue
            chosen_opt = opts[a["picked"][0]]
            chosen = _pk(state, chosen_opt)
            if not chosen:
                continue
            s0, s1 = _hp_map(state), _hp_map(b["obs"]["current"])
            ser = chosen["serial"]
            if ser not in s0 or ser not in s1:
                continue
            dmg = s0[ser] - s1[ser]
            if dmg <= 0:
                continue

            cand = [(j, _pk(state, o)) for j, o in enumerate(opts)]
            cand = [(j, p) for j, p in cand if p]
            if len(cand) < 2:
                continue
            # isolate the TRADEOFF regime: no KO available to anyone
            if any(0 < p["hp"] <= dmg for _, p in cand):
                stat[arch]["ko_available"] += 1
                continue
            stat[arch]["tradeoff"] += 1

            hps = sorted(p["hp"] for _, p in cand)
            rank = hps.index(chosen["hp"])          # 0 = lowest HP on the board
            stat[arch]["rank_lowest" if rank == 0 else
                       ("rank_highest" if rank == len(hps) - 1 else
                        "rank_middle")] += 1
            # active-vs-bench, only where BOTH were actually on offer
            areas = {o.get("area") for o in opts}
            if AREA_ACTIVE in areas and len(areas) > 1:
                stat[arch]["choice_act_or_bench"] += 1
                if chosen_opt.get("area") == AREA_ACTIVE:
                    stat[arch]["chose_active"] += 1
            stat[arch]["hp_sum"] += chosen["hp"]
            stat[arch]["hp_n"] += 1

    print(f"\n=== HOW THE NET AIMS PASSIVE DAMAGE, BY OPPONENT ARCHETYPE ===")
    print("  (tradeoff regime only: selects where NO option was KO-able)")
    print(f"\n  {'archetype':<28}{'games':>6}{'decisions':>10}"
          f"{'lowest HP':>11}{'highest':>9}{'mid':>7}"
          f"{'chose Active':>14}{'mean HP hit':>13}")
    rows = sorted(stat.items(), key=lambda kv: -kv[1]["tradeoff"])
    for arch, c in rows:
        if games_of[arch] < args.min_games:
            continue
        n = c["tradeoff"]
        if not n:
            continue
        ab = c["choice_act_or_bench"]
        print(f"  {arch[:27]:<28}{games_of[arch]:>6}{n:>10}"
              f"{c['rank_lowest']/n:>10.1%}{c['rank_highest']/n:>9.1%}"
              f"{c['rank_middle']/n:>7.1%}"
              f"{(c['chose_active']/ab if ab else float('nan')):>13.1%}"
              f"{(c['hp_sum']/max(c['hp_n'],1)):>13.0f}")

    tot = Counter()
    for c in stat.values():
        tot.update(c)
    n = tot["tradeoff"]
    print(f"\n  {'ALL':<28}{sum(games_of.values()):>6}{n:>10}"
          f"{tot['rank_lowest']/max(n,1):>10.1%}"
          f"{tot['rank_highest']/max(n,1):>9.1%}"
          f"{tot['rank_middle']/max(n,1):>7.1%}"
          f"{(tot['chose_active']/max(tot['choice_act_or_bench'],1)):>13.1%}"
          f"{(tot['hp_sum']/max(tot['hp_n'],1)):>13.0f}")
    print(f"\n  KO was available in {tot['ko_available']} more decisions "
          f"({tot['ko_available']/max(tot['ko_available']+n,1):.1%}) — "
          f"those are §8bm's regime and are excluded above.")
    print("\n⚠ Read the SPREAD across rows, not any single row. Identical rows "
          "= the net does not condition on the matchup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
