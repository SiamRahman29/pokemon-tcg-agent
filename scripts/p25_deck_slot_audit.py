"""Which slot in our 60 is the weakest? Size every card BEFORE spending an A/B.

**Rule 14 applied to the deck.** Track C has exactly one decklist A/B to its name
(0.490, null) and it was chosen by argument. §8af removed the standing excuse --
a swap into one of the corpus's **134 known card ids** is no more
off-distribution than Tool Scrapper, which we already play -- so the question is
no longer *"is it safe to change a card"* but **"which card is worth changing"**,
and that is a measurement over our own games rather than an opinion.

An n=2000 arena A/B resolves ~0.021 of win rate. A card we are offered 0.3 times
per game cannot move that no matter how much better its replacement is. So the
sizing gate here is the same one that killed the Morgrem out (§8e, ~0.2
firings/game), Pokegear's real choices (§8ag, 0.27) and the empty-bench rule
(§8ai, 0.187): **a slot must be live often enough for a change to be visible.**

Per card in our 60, over our real ladder games:
  * **offered**   -- selects where the card appeared as one of OUR options
  * **taken**     -- selects where we actually chose it
  * **take rate** -- taken / offered; a low rate on a high offer count means the
                     card is being declined, which is the signature of a slot the
                     policy does not want
  * **dead**      -- games where it was still sitting in hand at the last
                     observed state (drawn, never used)
  * **unseen**    -- games where it never left the deck or prizes at all

⚠ **Option -> card resolution uses `optfeat.option_features`, the same resolver
the net itself uses.** Do not hand-roll it: option `type` decides whether `index`
means a hand slot (type 7 PLAY) or an (area,index) pair (8/9 ATTACH/EVOLVE), and
getting that wrong silently attributes decisions to the wrong card.

    python -X utf8 scripts/p25_deck_slot_audit.py
    python -X utf8 scripts/p25_deck_slot_audit.py --dir replays/submission_v5
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "decks", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()
from sa import cards as cdb, optfeat  # noqa: E402
from decks import grimmsnarl  # noqa: E402

US = "Scio"


def name_of(cid: int) -> str:
    return (cdb.card(cid) or {}).get("name", str(cid))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", nargs="*",
                    default=["replays/submission_v5", "replays/submission_v4"])
    ap.add_argument("--us", default=US)
    args = ap.parse_args()

    deck = dict(grimmsnarl.DECKLIST)
    offered, taken = Counter(), Counter()
    dead, unseen = Counter(), Counter()
    games = 0
    sel_total = 0

    for d in args.dir:
        for f in sorted(glob.glob(str(ROOT / d / "*.json"))):
            rep = json.loads(Path(f).read_text(encoding="utf-8"))
            names = (rep.get("info") or {}).get("TeamNames") or []
            if args.us not in names:
                continue
            seat = names.index(args.us)
            vis = rep["steps"][0][0]["visualize"]
            games += 1
            seen_this_game = set()
            played_this_game = set()
            last_state = None
            for v in vis:
                obs = v.get("obs")
                if not isinstance(obs, dict):
                    continue
                cur = obs.get("current") or {}
                if cur.get("yourIndex") != seat:
                    continue
                sel = obs.get("select") or {}
                opts = sel.get("option") or []
                if not opts:
                    continue
                last_state = cur
                sel_total += 1
                act = v.get("action")
                chosen = set((act[0] if isinstance(act, list) and act else []) or [])
                ids_here = set()
                for j, o in enumerate(opts):
                    try:
                        _, cid, _, _ = optfeat.option_features(obs, o)
                    except Exception:  # noqa: BLE001 -- one bad option must not kill the audit
                        continue
                    if not cid or cid not in deck:
                        continue
                    ids_here.add(cid)
                    if j in chosen:
                        taken[cid] += 1
                        played_this_game.add(cid)
                for cid in ids_here:
                    offered[cid] += 1
                    seen_this_game.add(cid)

            if last_state:
                me = last_state["players"][seat]
                held = {c["id"] for c in (me.get("hand") or []) if c}
                # drawn but never used: sitting in hand at the end AND we never
                # once chose it this game.
                for cid in held:
                    if cid in deck and cid not in played_this_game:
                        dead[cid] += 1
                for cid in deck:
                    if cid not in seen_this_game and cid not in held:
                        unseen[cid] += 1

    print("Our 60, sized over %d real games (%d of our selects), agent=%s\n"
          % (games, sel_total, args.us))
    print("⚠ 'offered' counts SELECTS the card appeared in, not copies, and one")
    print("  turn is many selects -- so a LOW take% is NOT evidence the card is")
    print("  bad. PLAYS/GAME is the number that caps what a swap can pay.\n")
    print("%-26s %3s %8s %7s %10s %10s %6s %6s" % (
        "card", "n", "offered", "taken", "plays/game", "per copy", "dead", "unseen"))
    print("-" * 84)
    rows = []
    for cid, n in deck.items():
        off, tk = offered[cid], taken[cid]
        rows.append((tk / max(games, 1) / n, cid, n, off, tk,
                     tk / max(games, 1), dead[cid], unseen[cid]))
    for percopy, cid, n, off, tk, pg, dd, uns in sorted(rows):
        flag = ""
        if pg < 0.30:
            flag = "  <== under every sizing floor we have used"
        elif percopy < 0.20:
            flag = "  <== thin per copy"
        print("%-26s %3d %8d %7d %10.2f %10.3f %6d %6d%s"
              % (name_of(cid)[:26], n, off, tk, pg, percopy, dd, uns, flag))

    print("\nSizing floors this project has already killed things at:")
    print("  Morgrem out ~0.2 firings/game (§8e) · Pokegear real choices 0.27")
    print("  (§8ag) · empty-bench rule 0.187 (§8ai). n=2000 resolves ~0.021 WR.")
    print("\n⚠ Utilisation caps the CEILING of a swap, it does not value the card.")
    print("  Boss's Orders is played rarely and wins games when it is; that is")
    print("  exactly the §8e trap. Read this table as 'how much room is there',")
    print("  never as 'this card is bad'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
