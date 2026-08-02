"""Audit the arena anchors for game-losing pathologies, by watching them play.

**Why this exists (day 15/16).** The five anchors carry **71.5% of every
weighted verdict in this repo** and they were imported, not written here. Day-15
item 6 said nobody on this project had ever watched one play. The user watched
`out/replays/anchor_vs_anchor/game000` and reported that the Crustle pilot never
benched a second Pokemon and lost when its active was KO'd. That was **correct**
and it had a one-line cause (`sources/crustle.py:338`, fixed 2026-08-02).

An anchor that throws games does not merely add noise -- it **biases every A/B
that uses it in our favour**, silently, in a direction that looks like progress.
So the check has to be mechanical and repeatable rather than a one-off reading.

Two detectors, both derived from that bug rather than guessed:

  1. **DECLINED BENCH** -- the bench is empty, the engine offered a `PLAY` option
     naming a basic Pokemon, and the agent chose something else. Filling an empty
     bench is the most *dominated* option in the game: skip it and the next KO
     ends the match on the spot. (`optfeat.py:224` is the ground truth for the
     encoding: option type 7 = PLAY, `index` indexes MY HAND.)
     ⚠ **This detector OVERCOUNTS and must not be read as an error rate.** A turn
     is many selects, so an agent that plays three items and *then* benches is
     counted as three declines and has done nothing wrong. It is a screen, not a
     verdict -- which is why detector 1b exists.

  1b. **EXPOSED TURN END** -- the bench is empty, a bench play was on offer, and
     the agent chose to ATTACK or to end the turn anyway. This is the sharp
     version: it hands the opponent a board where any KO wins, while holding the
     card that would have prevented it. Unlike detector 1 there is no benign
     reading, because the turn is over.
  2. **EMPTY-BENCH LOSS** -- the game ended with this seat holding no active and
     no bench **while the opponent still had prizes left**. The prize check is
     what makes it unambiguous: a `Result` log alone does not distinguish this
     from a normal prize-out win.

    python -X utf8 scripts/p24_anchor_pathology.py
    python -X utf8 scripts/p24_anchor_pathology.py --dir out/replays/audit_lucario
"""
from __future__ import annotations

import argparse
import glob
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

PLAY = 7      # optfeat.py:224 -- index is an index into MY hand
ATTACK = 13   # optfeat.py:234 -- attacking ends the turn
PASS = 14     # the terminal "done" option (seen with area/index both null)
ENDS_TURN = {ATTACK, PASS}


def name_of(cid: int) -> str:
    return (cdb.card(cid) or {}).get("name", str(cid))


def audit_seat(replay: dict, seat: int) -> dict:
    vis = replay["steps"][0][0]["visualize"]
    out = dict(dec=0, empty=0, declined=0, exposed=0, cards=Counter(),
               lost=False, empty_loss=False, turns=0)
    for v in vis:
        obs = v.get("obs")
        if not isinstance(obs, dict):
            continue
        cur = obs.get("current") or {}
        if cur.get("yourIndex") != seat:
            continue
        players = cur.get("players") or []
        if len(players) != 2 or not players[seat]:
            continue
        p = players[seat]
        hand = [c for c in (p.get("hand") or []) if c]
        bench = [c for c in (p.get("bench") or []) if c]
        opts = (obs.get("select") or {}).get("option") or []
        out["dec"] += 1
        out["turns"] = max(out["turns"], cur.get("turn") or 0)
        if bench:
            continue
        out["empty"] += 1

        def is_bench_play(o):
            if (o.get("type") or 0) != PLAY:
                return None
            i = o.get("index") or 0
            if 0 <= i < len(hand) and cdb.is_basic_pokemon(hand[i]["id"]):
                return hand[i]["id"]
            return None

        offered = [c for c in (is_bench_play(o) for o in opts) if c]
        if not offered:
            continue
        act = v.get("action")
        chosen = (act[0] if isinstance(act, list) and act else []) or []
        took = any(is_bench_play(opts[c]) for c in chosen if c < len(opts))
        if not took:
            out["declined"] += 1
            for cid in offered:
                out["cards"][name_of(cid)] += 1
            # the sharp version: it ended the turn anyway, so there is no
            # "it benched later in the same turn" reading left.
            if any((opts[c].get("type") or 0) in ENDS_TURN
                   for c in chosen if c < len(opts)):
                out["exposed"] += 1

    rw = replay.get("rewards") or []
    if len(rw) == 2 and rw[0] is not None and rw[1] is not None:
        out["lost"] = rw[seat] < rw[1 - seat]
        if out["lost"]:
            last = None
            for v in vis:
                c = v.get("current") or {}
                ps = c.get("players") or []
                if len(ps) == 2 and ps[seat]:
                    last = c
            if last:
                me, opp = last["players"][seat], last["players"][1 - seat]
                if (not [a for a in (me.get("active") or []) if a]
                        and not [b for b in (me.get("bench") or []) if b]
                        and len(opp.get("prize") or []) > 0):
                    out["empty_loss"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", nargs="*", default=None,
                    help="replay dirs (default: every dir under out/replays)")
    args = ap.parse_args()

    dirs = args.dir or sorted(
        str(p) for p in (ROOT / "out/replays").iterdir() if p.is_dir())

    agg: dict[str, dict] = {}
    for dpath in dirs:
        for f in sorted(glob.glob(str(Path(dpath) / "*.json"))):
            rep = json.loads(Path(f).read_text(encoding="utf-8"))
            names = (rep.get("info") or {}).get("TeamNames") or ["seat0", "seat1"]
            for seat in (0, 1):
                r = audit_seat(rep, seat)
                key = names[seat]
                a = agg.setdefault(key, dict(games=0, dec=0, empty=0, declined=0,
                                             exposed=0, losses=0, empty_losses=0,
                                             cards=Counter(), files=[]))
                a["games"] += 1
                for k in ("dec", "empty", "declined", "exposed"):
                    a[k] += r[k]
                a["losses"] += int(r["lost"])
                a["empty_losses"] += int(r["empty_loss"])
                a["cards"].update(r["cards"])
                if r["empty_loss"]:
                    a["files"].append(Path(f).name)

    print("%-30s %5s %7s %6s %8s %8s %9s %8s" % (
        "agent", "games", "dec", "empty", "declined", "EXPOSED", "exp/game", "EB-loss"))
    print("-" * 92)
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["declined"]):
        print("%-30s %5d %7d %6d %8d %8d %9.3f %4d/%-3d" % (
            k[:30], a["games"], a["dec"], a["empty"], a["declined"],
            a["exposed"], a["exposed"] / max(a["games"], 1),
            a["empty_losses"], a["losses"]))
        if a["cards"]:
            print("      declined: %s" % ", ".join(
                "%s x%d" % (c, n) for c, n in a["cards"].most_common(5)))
        if a["files"]:
            print("      empty-bench losses in: %s" % ", ".join(a["files"][:6]))
    print()
    print("declined = empty bench AND a bench play offered AND not taken (OVERCOUNTS:")
    print("           benching later in the same turn still counts here).")
    print("EXPOSED  = same, but the agent ATTACKED or ENDED THE TURN anyway. No benign reading.")
    print("EB-loss  = lost with no active, no bench, opponent still holding prizes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
