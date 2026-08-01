"""Is Pokégear 3.0 being played correctly, and is its decision even a CHOICE?

**The user's observation (day 15):** *"Pokégear lets us see 7 cards and pick
one. I see that we are picking cards but I don't think we have any mechanism of
using the knowledge."*

**The mechanism half is correct, and the card text bounds how much it costs.**

    "Look at the top 7 cards of your deck. You may reveal a Supporter card you
     find there and put it into your hand. Shuffle the other cards back into
     your deck."

  * `features.py`'s id bags are `slots`, `my_hand`, `my_discard`,
    `opp_discard`. **There is no "cards I have seen in my deck" bag**, and
    `optfeat` reads the `looking` zone only while the select is open. So once
    the select closes, everything we saw is gone. The observation is right.
  * ⚠ **But the card SHUFFLES the rest back**, so the ordering knowledge is
    destroyed by the card itself, not by our encoding. What a perfect player
    retains is only *"these 6 cards are in my deck, therefore not prized"* --
    real information, but far smaller than "I know my next 6 draws".

So the question worth money is not "do we retain the knowledge" but **rule 13:
is the pick a real CHOICE, and do we get it right?** A Pokégear that finds
exactly one Supporter is a forced move -- nothing can go wrong and no mechanism
could help. This script measures, on our own ladder replays:

  1. how often a Pokégear select actually happens (rule 14: size it);
  2. how many Supporters were among the options -- the honest denominator;
  3. whether we took a Supporter when one was there (a WHIFF is a clear misplay
     -- it is dominated, not a tradeoff, so rule 11 says a rule could fix it);
  4. when 2+ Supporters were available, which one we took.

    python -X utf8 scripts/p23_pokegear_audit.py --dir replays/submission_v5 replays/submission_v4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()
from sa import cards as cdb  # noqa: E402

US = "Scio"
POKEGEAR = 1122
LOOKING = 12          # AreaType of the "top 7 cards" zone
SUPPORTER = 3         # cardType; Items are 1 (verified: Dawn/Crispin/Cyrano=3,
#                       Pokégear/Rare Candy/Ultra Ball=1)

# ⚠ The first version of this script keyed on `trainerType`, which does not
# exist in this card db -- `.get("trainerType", ...)` returned None for every
# card, so NOTHING was ever a Supporter and the audit reported **100% whiffs
# over 39 selects**. That is rule 9 exactly ("a metric that never prints is not
# a metric that passed"): a 100.0% cell against a user who had just said they
# watch us picking cards should be read as a broken probe, not a finding.


def is_supporter(cid: int) -> bool:
    try:
        return cdb.card(cid).get("cardType") == SUPPORTER
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+",
                    default=["replays/submission_v5", "replays/submission_v4"])
    ap.add_argument("--us", default=US)
    args = ap.parse_args()

    # positive control: these MUST be Supporters or the probe is broken (rule 9)
    ctrl = {cdb.card(c).get("name"): is_supporter(c)
            for c in (1231, 1198, 1205, 1225)}
    print(f"  positive control (all must be True): {ctrl}")
    if not all(ctrl.values()):
        print("  🔴 Supporter detection is broken -- refusing to report numbers")
        return 1

    games = fires = 0
    sup_hist: Counter = Counter()
    whiffs = 0
    took_sup = 0
    declined = 0
    forced = 0
    real_choice = 0
    picked_names: Counter = Counter()
    passed_names: Counter = Counter()
    opt_counts: Counter = Counter()
    seen_nonsup = 0

    for d in args.dir:
        for path in sorted(Path(d).glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(doc, dict):
                continue
            names = (doc.get("info") or {}).get("TeamNames") or []
            if args.us not in names:
                continue
            ours = {i for i, n in enumerate(names) if n == args.us}
            games += 1
            for v in doc["steps"][0][0].get("visualize") or []:
                obs = v.get("obs")
                if not isinstance(obs, dict):
                    continue
                st = obs.get("current")
                sel = obs.get("select")
                if not st or not sel:
                    continue
                if st.get("yourIndex") not in ours:
                    continue
                eff = sel.get("effect")
                eff_id = eff.get("id") if isinstance(eff, dict) else None
                if eff_id != POKEGEAR:
                    continue
                fires += 1
                opts = sel.get("option") or []
                opt_counts[len(opts)] += 1
                look = st.get("looking") or []
                seen_nonsup += sum(1 for c in look
                                   if c and not is_supporter(c.get("id", 0)))
                # ⚡ THE ENGINE PRE-FILTERS: every option already names a
                # Supporter found in the top 7. So "did we take a Supporter"
                # is not the question -- "did we take ANYTHING, and which"
                # is (minCount is 0, so declining is legal).
                ids = []
                for o in opts:
                    ix = o.get("index")
                    cid = 0
                    if ix is not None and ix < len(look) and look[ix]:
                        cid = look[ix].get("id", 0)
                    ids.append(cid)
                sups = [c for c in ids if c and is_supporter(c)]
                sup_hist[len(sups)] += 1

                # the action is per-seat: [[seat0 picks], [seat1 picks]]
                act = v.get("action")
                mine = []
                if isinstance(act, list) and act:
                    me = st.get("yourIndex")
                    cand = act[me] if me is not None and me < len(act) else None
                    mine = cand if isinstance(cand, list) else []
                chosen = ids[mine[0]] if mine and mine[0] < len(ids) else None

                if not opts:
                    whiffs += 1
                    continue
                if chosen is None:
                    declined += 1
                else:
                    took_sup += 1
                if len(opts) == 1:
                    forced += 1
                else:
                    real_choice += 1
                    if chosen:
                        picked_names[cdb.card(chosen).get("name")] += 1
                        for c in ids:
                            if c and c != chosen:
                                passed_names[cdb.card(c).get("name")] += 1

    print(f"\n=== {games} of our games, {fires} Pokégear selects "
          f"({fires / max(games,1):.2f} per game) ===")
    if not fires:
        print("  no Pokégear selects found -- check the effect id / dump")
        return 0
    print(f"  option-count histogram: "
          f"{dict(sorted(opt_counts.items()))}")
    print(f"\n  ⚡ The engine PRE-FILTERS the options to Supporters found in "
          f"the top 7,\n     so every option is already a legal, sensible take. "
          f"minCount=0, i.e.\n     declining is legal.")
    print(f"\n  🔴 THE HONEST DENOMINATOR (rule 13):")
    print(f"     FORCED (1 Supporter offered, nothing can go wrong)  "
          f"{forced:>4} ({forced/fires:>5.1%})")
    print(f"     REAL CHOICE (2+ Supporters offered)                 "
          f"{real_choice:>4} ({real_choice/fires:>5.1%})")
    print(f"\n  took a Supporter: {took_sup}/{fires} = {took_sup/fires:.1%}"
          f"   declined: {declined}")
    if declined:
        print(f"  🔴 {declined} DECLINE(S) -- taking a free Supporter is "
              "DOMINATED (rule 11),\n     the class where a rule has gone 3 "
              "for 3. Worth a look.")
    else:
        print("  ✅ we never decline a free Supporter -- the dominated half "
              "of this card\n     is already played correctly, so no rule can "
              "buy anything there.")
    if picked_names:
        print(f"\n  when 2+ offered, TOOK:   {dict(picked_names.most_common())}")
        print(f"                  PASSED: {dict(passed_names.most_common())}")
    print(f"\n  === SIZING (rule 14) ===")
    print(f"  real Pokégear choices per game: "
          f"{real_choice / max(games,1):.2f}")
    print(f"  ⚠ compare: the Morgrem out was CLOSED WITHOUT AN A/B at ~0.2 "
          f"firings/game\n     (§8e). An n=2000 arena A/B resolves ~0.021 of "
          f"win rate.")
    print(f"\n  === THE KNOWLEDGE THAT IS ACTUALLY LOST ===")
    print(f"  non-Supporter cards seen and shuffled back: {seen_nonsup} "
          f"over {fires} looks\n  ({seen_nonsup/max(fires,1):.1f} per look). "
          f"The net sees these ONLY while the\n  select is open; no id bag "
          f"retains them (features.BAG_NAMES).")
    print(f"  ⚠ But the CARD shuffles them back, so their ORDER is destroyed "
          f"by the card,\n     not by our encoding. What a perfect player "
          f"keeps is only 'these are in\n     my deck, so not prized' -- much "
          f"less than 'I know my next draws'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
