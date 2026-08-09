"""What does Petrel actually DO for us? — the last uninstrumented seam.

**Why this exists.** `p70_perturn_sweep` measures whether a card is *played* per
available turn. Nothing in this repo has ever looked at what `Team Rocket's
Petrel` (1219, x4, "search your deck for a Trainer") **fetches**, which is the
decision the user named. Petrel is the SECOND most-held card in our hand.

⚠ **The mapping is controlled before anything is counted.** E12's first
option->Pokemon mapping was wrong and only a positive control caught it (11/48
vs 55/55). Here the claim is "option {area:2, index:i} names hand[i]", and
`--verify` tests it the only way that cannot beg the question: **a hand card we
chose to play must LEAVE our hand by the next record.** A mapping that scores
low there is wrong, and every number below it is void.

    python -X utf8 scripts/p76_petrel_fetch.py --dir replays/submission_v5_s2 --verify
    python -X utf8 scripts/p76_petrel_fetch.py --dir replays/submission_v5_s2
"""
from __future__ import annotations

import argparse
import json
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

from p72_loss_autopsy import _nm, _records  # noqa: E402
from sa import cards as cdb  # noqa: E402
from sa.optfeat import option_features  # noqa: E402

PETREL = 1219
HAND = 2
MAIN = 0
OPT_PLAY = 7


def _opt_card(obs: dict, opt: dict) -> int:
    """The card an option names — via the NET'S OWN extractor, not a re-derivation.

    🔴 The hand-rolled version of this was wrong and the first table built on it
    was nonsense: it filtered on `opt["area"] == HAND`, but a **PLAY option
    (type 7) carries no `area` at all** — it is a bare index into the hand. So
    every card play in the corpus was invisible and Buddy-Buddy Poffin came back
    as "offered 9 times in 76 games", which E11's own measurement (0.80
    plays/game of GAP) contradicts outright.

    `optfeat.option_features` is what built the training data and what the net
    sees, so using it here means this script cannot disagree with the net about
    what an option *is*.
    """
    try:
        return int(option_features(obs, opt)[1] or 0)
    except Exception:  # noqa: BLE001
        return 0


def _hand_ids(state: dict, seat: int) -> list[int | None]:
    hand = (state["players"][seat].get("hand") or [])
    return [(h or {}).get("id") for h in hand]


def _walk(dirs: list[str], us: set[str]):
    """Yield (path, seat, index, records) for every game with our seat."""
    for d in dirs:
        for path in sorted((ROOT / d).glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                rep = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            names = (rep.get("info") or {}).get("TeamNames") or []
            seats = {i for i, n in enumerate(names) if n in us}
            if not seats:
                continue
            yield path, seats, _records(rep)


def verify(dirs: list[str], us: set[str]) -> int:
    """Does a hand card we PLAYED actually leave our hand by the next record?

    ⚠ Counts only cards whose copy count in hand should drop -- a duplicate held
    twice cannot distinguish "played one" from "played none" by presence, so the
    test is on the COUNT of that id, not on membership.
    """
    ok = bad = skip = 0
    for _, seats, recs in _walk(dirs, us):
        for i, a in enumerate(recs):
            st = a["obs"]["current"]
            me = st.get("yourIndex")
            if me not in seats or len(a["picked"]) != 1:
                continue
            # ⚠ The obvious "compare against recs[i+1]" only ever fires when the
            # very next record is OURS, which is ~2% of the corpus (n=10) -- far
            # too thin to license anything. Scan to our next record instead.
            nxt = None
            for b in recs[i + 1:]:
                if b["obs"]["current"].get("yourIndex") == me:
                    nxt = b["obs"]["current"]
                    break
            if nxt is None:
                skip += 1
                continue
            opts = a["sel"].get("option") or []
            o = opts[a["picked"][0]]
            if (o.get("type") or 0) != OPT_PLAY:
                continue
            cid = _opt_card(a["obs"], o)
            ids = _hand_ids(st, me)
            if not cid or cid not in ids:
                bad += 1
                continue
            before = ids.count(cid)
            after = _hand_ids(nxt, me).count(cid)
            if after < before:
                ok += 1
            else:
                bad += 1
    total = ok + bad
    print(f"\n=== POSITIVE CONTROL: a played hand card leaves the hand ===")
    print(f"  matched {ok}/{total} ({ok/max(total,1):.1%})"
          f"   [{skip} had no later record for our seat]")
    print("  ⚠ Not a clean 100% even when correct: between our two records we "
          "may DRAW\n     another copy of the same id, which masks the "
          "decrement. Read this as a\n     floor on the mapping's accuracy, not "
          "as an exact rate.")
    if total and ok / total < 0.9:
        print("  🔴 THE MAPPING IS WRONG. Every count in this script is void "
              "until it is fixed.\n     This is E12's 11/48 all over again.")
        return 1
    print("  ✅ mapping holds; the counts below rest on it.")
    return 0


FETCH_CTX = 7        # the search select Petrel opens; options are type 3 over area 1 (deck)


def fetches(dirs: list[str], us: set[str]) -> tuple[list[dict], int, int]:
    """One row per Petrel resolution: what was ON OFFER, and what we took.

    ⚠ `minCount` is 0 — **declining is legal**, and a decline is a real choice,
    not missing data. It is recorded as `taken=None` rather than dropped.
    """
    rows: list[dict] = []
    games = plays = 0
    for _, seats, recs in _walk(dirs, us):
        games += 1
        for i, r in enumerate(recs):
            st = r["obs"]["current"]
            me = st.get("yourIndex")
            if me not in seats or len(r["picked"]) != 1:
                continue
            o = (r["sel"].get("option") or [])[r["picked"][0]]
            if (o.get("type") or 0) != OPT_PLAY:
                continue
            if _opt_card(r["obs"], o) != PETREL:
                continue
            plays += 1
            nxt = next((x for x in recs[i + 1:i + 6]
                        if x["obs"]["current"].get("yourIndex") == me
                        and x["sel"].get("context") == FETCH_CTX), None)
            if nxt is None:
                continue
            opts = nxt["sel"].get("option") or []
            avail = Counter()
            for oo in opts:
                cid = _opt_card(nxt["obs"], oo)
                if cid:
                    avail[cid] += 1
            taken = None
            if nxt["picked"]:
                taken = _opt_card(nxt["obs"], opts[nxt["picked"][0]]) or None
            rows.append({"avail": avail, "taken": taken, "n": len(opts)})
    return rows, games, plays


def report_fetch(rows: list[dict], games: int, plays: int, label: str,
                 top: int) -> dict[int, tuple[int, int]]:
    print(f"\n=== WHAT PETREL FETCHES — {label} ===")
    print(f"  {games} games, {plays} Petrel plays ({plays/max(games,1):.2f}/game), "
          f"{len(rows)} resolved fetches")
    if not rows:
        return {}
    took = Counter(r["taken"] for r in rows)
    seen: Counter = Counter()
    for r in rows:
        for cid in r["avail"]:
            seen[cid] += 1
    print(f"\n  {'card':<30}{'fetched':>9}{'share':>8}"
          f"{'in deck when':>14}{'take|avail':>12}")
    for cid, n in took.most_common(top):
        if cid is None:
            print(f"  {'(declined — took nothing)':<30}{n:>9}"
                  f"{n/len(rows):>8.1%}{'-':>14}{'-':>12}")
            continue
        av = seen[cid]
        print(f"  {_nm(cid)[:28]:<30}{n:>9}{n/len(rows):>8.1%}"
              f"{av:>14}{(n/av if av else float('nan')):>12.1%}")
    misses = [(cid, seen[cid], took[cid]) for cid in seen
              if seen[cid] >= max(8, len(rows) // 12)]
    misses.sort(key=lambda t: -(t[1] - t[2]))
    print(f"\n  available but usually LEFT in the deck:")
    for cid, av, tk in misses[:8]:
        print(f"    {_nm(cid)[:28]:<30} available {av:>4}, taken {tk:>4}"
              f"  ({tk/av:.1%})")
    return {cid: (seen[cid], took[cid]) for cid in seen}


SCRAPPER = 1137      # "Choose up to 2 Pokemon Tools attached to Pokemon ... discard them"


def _board_tools(state: dict, me: int) -> tuple[int, int, Counter]:
    """(tools on THEIR board, tools on ours, ids seen) — from `pk["tools"]`."""
    theirs = ours = 0
    ids: Counter = Counter()
    for seat in (me, 1 - me):
        pl = state["players"][seat]
        for where in ("active", "bench"):
            for pk in (pl.get(where) or []):
                if not pk:
                    continue
                for t in (pk.get("tools") or []):
                    ids[(t or {}).get("id") if isinstance(t, dict) else t] += 1
                    if seat == me:
                        ours += 1
                    else:
                        theirs += 1
    return theirs, ours, ids


def scrapper(dirs: list[str], us: set[str], label: str) -> None:
    """Tool Scrapper's PRECONDITION at fetch time — is there a tool to scrap?

    ⚠ A take rate alone cannot say whether a fetch was right. Scrapper does
    nothing unless a Pokemon Tool is attached to something, so the only
    interpretable unit is "taken | a target existed". Reporting 6.9% without
    this split reads as tech awareness; the split says the opposite.
    """
    f: Counter = Counter()
    tool_ids: Counter = Counter()
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
            s2 = nxt["obs"]["current"]
            opts = nxt["sel"].get("option") or []
            avail = {_opt_card(nxt["obs"], oo) for oo in opts}
            taken = _opt_card(nxt["obs"], opts[nxt["picked"][0]]) if nxt["picked"] else None
            f["fetches"] += 1
            if SCRAPPER not in avail:
                f["scrapper already gone (it is a 1-of)"] += 1
                continue
            f["scrapper in deck"] += 1
            th, ou, ids = _board_tools(s2, me)
            tool_ids.update(ids)
            key = ("THEIR tool on board" if th else
                   "only OUR tool" if ou else "NO tool anywhere")
            f[key] += 1
            if taken == SCRAPPER:
                f["  ...taken: " + key] += 1
    print(f"\n=== TOOL SCRAPPER'S PRECONDITION — {label} ===")
    for k, v in f.items():
        print(f"  {k:<40}{v:>6}{v/max(f['fetches'],1):>9.1%}")
    ind, th = f.get("scrapper in deck", 0), f.get("THEIR tool on board", 0)
    if ind:
        print(f"  → a scrappable opposing tool existed {th}/{ind} = "
              f"{th/ind:.1%} of the fetches where Scrapper was still in the deck")
        print(f"  → taken WITH a target: {f.get('  ...taken: THEIR tool on board', 0)}/{th}"
              f"   |   taken with NO target: "
              f"{f.get('  ...taken: NO tool anywhere', 0)}/{f.get('NO tool anywhere', 0)}")
    print("  tools seen:", {_nm(k) if isinstance(k, int) else k: v
                            for k, v in tool_ids.most_common(8)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["replays/submission_v5_s2"])
    ap.add_argument("--us", action="append", default=["Scio"])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fetch", action="store_true",
                    help="the fetch analysis (what Petrel actually gets)")
    ap.add_argument("--vs", nargs="+", metavar="DIR",
                    help="second corpus of dumps to compare against")
    ap.add_argument("--vs-us", action="append", default=[],
                    help="seat name(s) in --vs; repeat for several pilots")
    ap.add_argument("--scrapper", action="store_true",
                    help="Tool Scrapper's precondition: was a tool ever on the board?")
    ap.add_argument("--top", type=int, default=22)
    args = ap.parse_args()
    us = set(args.us)

    if args.verify:
        return verify(args.dir, us)

    if args.scrapper:
        scrapper(args.dir, us, "/".join(sorted(us)))
        if args.vs and args.vs_us:
            scrapper(args.vs, set(args.vs_us), "/".join(sorted(args.vs_us)))
        return 0

    if args.fetch:
        rows, g, p = fetches(args.dir, us)
        a = report_fetch(rows, g, p, "/".join(sorted(us)), args.top)
        if args.vs and args.vs_us:
            rows2, g2, p2 = fetches(args.vs, set(args.vs_us))
            b = report_fetch(rows2, g2, p2, "/".join(sorted(args.vs_us)),
                             args.top)
            print(f"\n=== TAKE-RATE GAP, conditioned on the card being AVAILABLE ===")
            print("  (the rule-21-correct unit: per fetch where the card was "
                  "actually there)")
            print(f"  {'card':<30}{'our avail':>10}{'our rate':>10}"
                  f"{'their avail':>12}{'their rate':>12}{'gap':>9}")
            keys = sorted(set(a) | set(b),
                          key=lambda c: -(a.get(c, (0, 0))[0] + b.get(c, (0, 0))[0]))
            for cid in keys:
                av1, tk1 = a.get(cid, (0, 0))
                av2, tk2 = b.get(cid, (0, 0))
                if av1 < 8 or av2 < 8:
                    continue
                r1, r2 = tk1 / av1, tk2 / av2
                print(f"  {_nm(cid)[:28]:<30}{av1:>10}{r1:>10.1%}"
                      f"{av2:>12}{r2:>12.1%}{r1-r2:>+9.1%}")
            # --- RULE 14, done properly --------------------------------------
            # ⚠ The tempting sizing is "add up the take-rate gaps", and it is
            # WRONG: those are CONDITIONAL rates over overlapping denominators,
            # so they do not sum to a share of anything. The share of fetches
            # that would change if we adopted their policy exactly is the total
            # variation distance between the two FETCHED distributions.
            def dist(rs: list[dict]) -> dict[int | None, float]:
                c = Counter(r["taken"] for r in rs)
                return {k: v / len(rs) for k, v in c.items()} if rs else {}

            pa, pb = dist(rows), dist(rows2)
            tv = 0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0))
                           for k in set(pa) | set(pb))
            per_game = tv * (p / max(g, 1))
            print(f"\n  === SIZING (rule 14) ===")
            print(f"  total variation between the two fetch distributions: "
                  f"**{tv:.1%}**")
            print(f"  our Petrel plays/game: {p/max(g,1):.2f}"
                  f"  ⇒ adopting their policy WHOLESALE changes "
                  f"**{per_game:.2f} fetches/game**")
            print(f"  gate = 0.5/game  ⇒ "
                  f"{'✅ CLEARS' if per_game >= 0.5 else '⛔ BELOW THE GATE'}")
            print("  ⚠ And that is the ceiling: it is the whole distribution at "
                  "once. Any SINGLE\n     card rule is a fraction of it — size "
                  "the rule you would actually write,\n     not the sum of every "
                  "difference you can see.")
            print("\n  ⚠ A gap here is a DESCRIPTION. Rule 14 sizes it before "
                  "anything is built,\n     and rule 11 asks which column it is "
                  "in — Petrel's fetch is a tradeoff\n     unless one option "
                  "dominates on every dimension.")
        return 0

    games = 0
    held: Counter = Counter()        # records where the card sat in our hand
    offered: Counter = Counter()     # ... and was a legal option
    played: Counter = Counter()      # ... and we took it
    held_main: Counter = Counter()   # held specifically at a MAIN select
    seen_paths = set()

    for path, seats, recs in _walk(args.dir, us):
        if path not in seen_paths:
            seen_paths.add(path)
            games += 1
        for r in recs:
            st = r["obs"]["current"]
            me = st.get("yourIndex")
            if me not in seats:
                continue
            ids = _hand_ids(st, me)
            for cid in set(i for i in ids if i):
                held[cid] += 1
                if r["sel"].get("context") == MAIN:
                    held_main[cid] += 1
            opts = r["sel"].get("option") or []
            for k, o in enumerate(opts):
                if (o.get("type") or 0) != OPT_PLAY:
                    continue
                cid = _opt_card(r["obs"], o)
                if not cid:
                    continue
                offered[cid] += 1
                if k in r["picked"]:
                    played[cid] += 1

    print(f"\n=== HAND CARDS: held vs OFFERED vs played — {games} games ===")
    print("  'held' counts records where at least one copy sat in our hand.")
    print("  'offered' counts OPTIONS naming it; 'played' those we took.")
    print(f"\n  {'card':<30}{'held':>8}{'held@MAIN':>11}{'offered':>9}"
          f"{'played':>8}{'play|offer':>12}")
    for cid, n in held.most_common(args.top):
        off, pl = offered[cid], played[cid]
        mark = "⚡" if cid == PETREL else "  "
        print(f"{mark}{_nm(cid)[:28]:<30}{n:>8}{held_main[cid]:>11}{off:>9}"
              f"{pl:>8}{(pl/off if off else float('nan')):>12.1%}")

    off, pl, h = offered[PETREL], played[PETREL], held[PETREL]
    print(f"\n  ⚡ Petrel: held in {h} records ({held_main[PETREL]} at a MAIN "
          f"select), offered {off}, played {pl}.")
    if off <= 5:
        print("  🔴 Petrel is essentially NEVER a legal option in this corpus. "
              "That is a\n     statement about the ENGINE or the deck, not "
              "about the net's judgement —\n     there is no fetch decision to "
              "instrument until this is explained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
