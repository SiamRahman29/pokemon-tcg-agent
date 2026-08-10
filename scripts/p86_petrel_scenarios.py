"""Two named Petrel scenarios, measured on our own ladder games (day 29).

**Why this exists.** §8br closed the Petrel seam *by sizing* -- the whole fetch
policy differs from three stronger pilots by 0.29 fetches/game, under the 0.5
gate. ⚠ **That is a statement about MEASURABILITY, not about correctness**, and
§8br says so in as many words. The user named two concrete spots where the
policy might be not-merely-different but *wrong*, and neither is answered by a
marginal take rate:

  A. Their ACTIVE carries a Tool -- do we fetch Tool Scrapper?
     §8br's `--scrapper` bucket is "a tool anywhere on THEIR board", which pools
     the active (where scrapping changes the current attack/retreat maths) with
     the bench (where it usually does not). This splits them.

  B. We hold a strong hand and still fetch/play Unfair Stamp because something
     of ours was KO'd. Unfair Stamp shuffles BOTH hands away; we draw 5, they
     draw 2. So with a hand of H at the moment of the play our net card delta is
     `5 - (H-1)` and theirs is `2 - O`. Played on a big hand into a small one it
     is negative on BOTH axes -- a **dominated** play, which is rule 11's good
     column (3/3), not the tradeoff column (0/4).

⚠ **The mapping this rests on is p76's**, and it is controlled there
(`p76_petrel_fetch.py --verify`, 1331/1375 = 96.8%). Card ids come from
`optfeat.option_features`, the extractor that built the training data, so this
script cannot disagree with the net about what an option is.

    python -X utf8 scripts/p86_petrel_scenarios.py --scrapper
    python -X utf8 scripts/p86_petrel_scenarios.py --stamp
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

from p72_loss_autopsy import _nm  # noqa: E402
from p76_petrel_fetch import (  # noqa: E402  -- rule 18: reuse, do not re-derive
    FETCH_CTX, OPT_PLAY, PETREL, SCRAPPER, _opt_card, _walk,
)

STAMP = 1080          # Unfair Stamp (ACE SPEC, x1)
MAIN = 0


def _tools_split(state: dict, me: int) -> tuple[int, int, int, int]:
    """(their active, their bench, our active, our bench) tool counts.

    §8br pooled their active and bench into one "THEIR tool on board" bucket.
    The distinction is the whole question here: a Tool on the ACTIVE is the one
    that changes what is happening this turn.
    """
    out = []
    for seat in (1 - me, me):
        pl = state["players"][seat]
        act = sum(len((pk or {}).get("tools") or [])
                  for pk in (pl.get("active") or []) if pk)
        ben = sum(len((pk or {}).get("tools") or [])
                  for pk in (pl.get("bench") or []) if pk)
        out += [act, ben]
    return out[0], out[1], out[2], out[3]


def _fetch_selects(dirs: list[str], us: set[str]):
    """Yield (state_at_fetch, options, taken_id, available_ids) per resolution."""
    for _, seats, recs in _walk(dirs, us):
        for i, r in enumerate(recs):
            st = r["obs"]["current"]
            me = st.get("yourIndex")
            if me not in seats or len(r["picked"]) != 1:
                continue
            o = (r["sel"].get("option") or [])[r["picked"][0]]
            if (o.get("type") or 0) != OPT_PLAY or _opt_card(r["obs"], o) != PETREL:
                continue
            nxt = next((x for x in recs[i + 1:i + 6]
                        if x["obs"]["current"].get("yourIndex") == me
                        and x["sel"].get("context") == FETCH_CTX), None)
            if nxt is None:
                continue
            opts = nxt["sel"].get("option") or []
            avail = {_opt_card(nxt["obs"], oo) for oo in opts} - {0}
            taken = (_opt_card(nxt["obs"], opts[nxt["picked"][0]])
                     if nxt["picked"] else None)
            yield nxt["obs"]["current"], me, avail, taken


# --- A. the Tool Scrapper question -------------------------------------------

def scenario_a(dirs: list[str], us: set[str], label: str) -> None:
    buckets: Counter = Counter()
    taken: Counter = Counter()
    instead: Counter = Counter()
    for st, me, avail, took in _fetch_selects(dirs, us):
        if SCRAPPER not in avail:          # already drawn/discarded -- it is a 1-of
            buckets["(scrapper not in deck)"] += 1
            continue
        ta, tb, oa, ob = _tools_split(st, me)
        key = ("THEIR ACTIVE has a tool" if ta else
               "their BENCH only" if tb else
               "only OUR tools" if (oa or ob) else
               "no tool anywhere")
        buckets[key] += 1
        if took == SCRAPPER:
            taken[key] += 1
        elif key == "THEIR ACTIVE has a tool":
            instead[took] += 1

    print(f"\n=== A. THEIR ACTIVE HAS A TOOL — do we fetch Tool Scrapper? "
          f"[{label}] ===")
    print("  denominator: Petrel fetches where Scrapper was STILL IN THE DECK\n")
    print(f"  {'board state at the fetch':<32}{'fetches':>9}{'took Scrapper':>16}{'rate':>9}")
    order = ["THEIR ACTIVE has a tool", "their BENCH only", "only OUR tools",
             "no tool anywhere"]
    for k in order:
        n = buckets.get(k, 0)
        if not n:
            print(f"  {k:<32}{n:>9}{'-':>16}{'-':>9}")
            continue
        print(f"  {k:<32}{n:>9}{taken.get(k, 0):>16}{taken.get(k, 0)/n:>9.1%}")
    tot = sum(buckets.get(k, 0) for k in order)
    print(f"  {'-'*32}{tot:>9}{sum(taken.values()):>16}"
          f"{sum(taken.values())/max(tot,1):>9.1%}")
    print(f"  [{buckets.get('(scrapper not in deck)', 0)} fetches excluded: "
          f"Scrapper already out of the deck]")
    if instead:
        print("\n  what we fetched INSTEAD when their active had a tool:")
        for cid, n in instead.most_common(8):
            print(f"    {(_nm(cid) if cid else '(declined)')[:28]:<30}{n:>4}")


# --- B. the Unfair Stamp question --------------------------------------------

def scenario_b(dirs: list[str], us: set[str], label: str) -> None:
    """Two decisions: fetching Stamp, and PLAYING it.

    ⚠ Our own hand is fully visible in our observation; the opponent's `hand`
    list is empty by construction and only `handCount` is populated. Using
    `len(hand)` for them would score every opponent at 0 and manufacture the
    result. `handCount` is used for both seats.
    """
    # B0 -- the fetch
    fetch_h: list[int] = []
    pass_h: list[int] = []
    for st, me, avail, took in _fetch_selects(dirs, us):
        if STAMP not in avail:
            continue
        h = int(st["players"][me].get("handCount") or 0)
        (fetch_h if took == STAMP else pass_h).append(h)

    print(f"\n=== B0. FETCHING Unfair Stamp — does hand size gate it? [{label}] ===")
    print("  denominator: Petrel fetches where Stamp was legal AND in the deck")
    print("  (hand size is measured AT the fetch select, i.e. Petrel already played)\n")
    for lbl, xs in (("took Stamp", fetch_h), ("passed on Stamp", pass_h)):
        if not xs:
            print(f"  {lbl:<20} n=0")
            continue
        print(f"  {lbl:<20} n={len(xs):<4} mean hand {sum(xs)/len(xs):.2f}"
              f"   hand>=5: {sum(1 for x in xs if x >= 5)}/{len(xs)}"
              f"   dist {sorted(xs)}")

    # B1 -- the play.
    # ⚠ RULE 21. The obvious denominator -- every MAIN select where Stamp was
    # legal -- counts ONE turn many times and is contaminated by within-turn
    # ORDERING: we decline early in the turn holding 8 cards, play four of them,
    # then play Stamp holding 4. Per select that reads as "declines on big
    # hands, plays on small ones" when nothing about the judgement differs.
    # The ordering-free unit is the TURN, scored at its FIRST offer.
    turns: dict[tuple, dict] = {}
    for path, seats, recs in _walk(dirs, us):
        for r in recs:
            st = r["obs"]["current"]
            me = st.get("yourIndex")
            if me not in seats or r["sel"].get("context") != MAIN:
                continue
            opts = r["sel"].get("option") or []
            idx = [k for k, o in enumerate(opts)
                   if (o.get("type") or 0) == OPT_PLAY
                   and _opt_card(r["obs"], o) == STAMP]
            if not idx:
                continue
            key = (path.name, me, st.get("turn"))
            t = turns.setdefault(key, {
                "h0": int(st["players"][me].get("handCount") or 0),
                "o0": int(st["players"][1 - me].get("handCount") or 0),
                # a size-free proxy for "a strong hand": how many DISTINCT cards
                # in hand are legal plays right now, i.e. how much this hand can
                # actually do. Read off the option list, not guessed.
                "plays0": len({_opt_card(r["obs"], o) for o in opts
                               if (o.get("type") or 0) == OPT_PLAY}),
                "offers": 0, "played": False, "h_at_play": None,
                "o_at_play": None,
            })
            t["offers"] += 1
            if any(k in r["picked"] for k in idx) and not t["played"]:
                t["played"] = True
                t["h_at_play"] = int(st["players"][me].get("handCount") or 0)
                t["o_at_play"] = int(st["players"][1 - me].get("handCount") or 0)

    tl = list(turns.values())
    played = [(t["h_at_play"], t["o_at_play"]) for t in tl if t["played"]]
    declined = [(t["h0"], t["o0"]) for t in tl if not t["played"]]
    print("\n=== B1a. THE TURN-LEVEL DECISION (rule 21: ordering-free) ===")
    print(f"  turns where Stamp was legal at some MAIN select: {len(tl)}"
          f"   played that turn: {sum(1 for t in tl if t['played'])}"
          f"   ({sum(1 for t in tl if t['played'])/max(len(tl),1):.1%})")
    print(f"  (the per-SELECT denominator would have been "
          f"{sum(t['offers'] for t in tl)} -- "
          f"{sum(t['offers'] for t in tl)/max(len(tl),1):.1f} selects per turn)")
    print(f"\n  measured at the turn's FIRST offer, so ordering cannot move it:")
    print(f"  {'':<22}{'n':>5}{'mean hand':>11}{'mean legal plays':>18}"
          f"{'mean THEIR hand':>17}")
    for lbl, sub in (("played this turn", [t for t in tl if t["played"]]),
                     ("never played it", [t for t in tl if not t["played"]])):
        if not sub:
            continue
        print(f"  {lbl:<22}{len(sub):>5}"
              f"{sum(t['h0'] for t in sub)/len(sub):>11.2f}"
              f"{sum(t['plays0'] for t in sub)/len(sub):>18.2f}"
              f"{sum(t['o0'] for t in sub)/len(sub):>17.2f}")
    big = [t for t in tl if t["h0"] >= 7]
    if big:
        print(f"\n  🔎 turns opening with a BIG hand (H>=7): {len(big)}"
              f"   played anyway: {sum(1 for t in big if t['played'])}"
              f"  ({sum(1 for t in big if t['played'])/len(big):.1%})")
        small = [t for t in tl if t["h0"] <= 4]
        print(f"     turns opening with a SMALL hand (H<=4): {len(small)}"
              f"   played: {sum(1 for t in small if t['played'])}"
              f"  ({sum(1 for t in small if t['played'])/max(len(small),1):.1%})")

    print("\n=== B1b. PLAYING Unfair Stamp — the card maths at the moment we played ===")
    print("  'Each player shuffles their hand into their deck. Then you draw 5,")
    print("   your opponent draws 2.'  We hold H (Stamp included), they hold O.")
    print("   our net = 5 - (H-1)      their net = 2 - O\n")
    if not played:
        return
    print(f"\n  {'our hand H':>11}{'their hand O':>14}{'our net':>9}"
          f"{'their net':>11}{'n':>5}")
    agg = Counter(played)
    dominated = both_neg = 0
    for (h, o), n in sorted(agg.items()):
        ours_net, theirs_net = 5 - (h - 1), 2 - o
        flag = ""
        if ours_net < 0 and theirs_net >= 0:
            flag = "  🔴 DOMINATED — we lose cards, they gain"
            dominated += n
        elif ours_net < 0:
            both_neg += n
        print(f"  {h:>11}{o:>14}{ours_net:>+9}{theirs_net:>+11}{n:>5}{flag}")
    tot = len(played)
    print(f"\n  plays where WE lose cards (H-1 > 5):"
          f" {sum(n for (h, _), n in agg.items() if h - 1 > 5)}/{tot}")
    print(f"  plays that are DOMINATED (we lose cards AND they do not):"
          f" {dominated}/{tot}  = {dominated/tot:.1%}")
    print(f"  plays where both sides lose cards (a real trade):"
          f" {both_neg}/{tot}")
    # ⚠ "we lose cards" is not the same as "we lost the exchange". Stamp is a
    # DISRUPTION card: shedding 3 of our own to strip 11 of theirs is the play
    # working, not failing. The interpretable scalar is the DIFFERENTIAL.
    diffs = [(5 - (h - 1)) - (2 - o) for h, o in played]
    worse = sum(1 for d in diffs if d < 0)
    print(f"  mean card DIFFERENTIAL (our net - their net): "
          f"{sum(diffs)/len(diffs):+.2f}"
          f"   plays that lost the exchange on raw cards: {worse}/{tot}")
    hs = [h for h, _ in played]
    print(f"  mean hand when we played it: {sum(hs)/len(hs):.2f}"
          f"   (max {max(hs)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["replays/submission_v5_s2"])
    # 🔴 `action="append"` APPENDS TO THE DEFAULT rather than replacing it, so
    # `--us flg` silently means {"Scio", "flg"} and an "experts only" run is
    # contaminated with our own seat under the experts' label. Default None,
    # resolve after parsing.
    ap.add_argument("--us", action="append", default=None)
    ap.add_argument("--scrapper", action="store_true")
    ap.add_argument("--stamp", action="store_true")
    ap.add_argument("--vs", nargs="+", metavar="DIR",
                    help="expert dumps to run scenario A against")
    ap.add_argument("--vs-us", action="append", default=[],
                    help="seat name(s) in --vs; repeat for several pilots")
    args = ap.parse_args()
    us = set(args.us or ["Scio"])
    if not (args.scrapper or args.stamp):
        args.scrapper = args.stamp = True
    if args.scrapper:
        scenario_a(args.dir, us, "/".join(sorted(us)))
        # ⚠ "we do X badly" means nothing until the people beating us are shown
        # doing X well -- §8bj's on-policy control is the same lesson.
        if args.vs and args.vs_us:
            scenario_a(args.vs, set(args.vs_us), "/".join(sorted(args.vs_us)))
    if args.stamp:
        scenario_b(args.dir, us, "/".join(sorted(us)))
        if args.vs and args.vs_us:
            scenario_b(args.vs, set(args.vs_us),
                       "/".join(sorted(args.vs_us)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
