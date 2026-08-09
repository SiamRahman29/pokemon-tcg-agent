"""Census a MINED DAY dump — both seats, every game, joined to the episode rating.

`p9_field_census` censuses **from a seat** (`--us`), which is right for our own
ladder dumps and for a named third party. A mined day is neither: it is an
arbitrary slice of the whole ladder, and the question is what the FIELD is
playing and who the strong pilots are.

So this walks the `visualize` stream **once** and attributes boards to *both*
players (each entry observes the opponent of whoever is on turn), labels each
side's deck with `p9`'s `_signature`, and joins `manifest.csv`'s `avg_score`.

⚠ **§8i: the episode feed is band-censored** — nothing below ~1055 `avg_score`
reaches these dumps. Every share below describes the TOP of the ladder, not the
field our agent is matched against at 1027. Use it for demonstrators and for
"what do the strong decks look like", ⛔ **never as an anchor or a meta share.**

    python -X utf8 scripts/p75_day_census.py --dir replays/2026-08-0{3,4,5,6,7}
    python -X utf8 scripts/p75_day_census.py --dir replays/2026-08-07 --arch grimmsnarl
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
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

from p9_field_census import _cid, _signature  # noqa: E402
from sa import cards as cdb  # noqa: E402


def _ratings(d: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    m = d / "manifest.csv"
    if not m.exists():
        return out
    with m.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[int(row["episode_id"])] = float(row["avg_score"])
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _both_seats(path: Path) -> list[dict] | None:
    """One row per SEAT: name, deck label, result, from a single stream walk."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(d, dict):
        return None
    names = (d.get("info") or {}).get("TeamNames") or []
    rewards = d.get("rewards") or []
    if len(names) < 2 or len(rewards) < 2:
        return None
    # An errored episode carries a null reward. It has no winner, so it cannot
    # enter a win rate -- dropped whole rather than scored as a loss.
    if any(not isinstance(r, (int, float)) for r in rewards[:2]):
        return None

    poke: dict[int, Counter] = defaultdict(Counter)
    copies: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for v in d["steps"][0][0].get("visualize") or []:
        obs = v.get("obs")
        if not obs or not obs.get("current"):
            continue
        st = obs["current"]
        me = st.get("yourIndex")
        if me is None:
            continue
        them = 1 - me                      # the seat this entry can SEE
        try:
            op = st["players"][them]
        except (KeyError, IndexError, TypeError):
            continue
        here: Counter = Counter()
        act = op.get("active")
        if act and act[0]:
            here[_cid(act[0])] += 1
            for c in (act[0].get("cards") or []):
                here[_cid(c)] += 1
        for pk in (op.get("bench") or []):
            if pk:
                here[_cid(pk)] += 1
                for c in (pk.get("cards") or []):
                    here[_cid(c)] += 1
        for c in (op.get("discard") or []):
            here[_cid(c)] += 1
        for cid, n in here.items():
            if n > copies[them][cid]:
                copies[them][cid] = n
            if cdb.is_pokemon(cid):
                poke[them][cid] += 1

    rows = []
    for seat in (0, 1):
        if not poke.get(seat):
            continue                        # never observed: cannot label
        rows.append({
            "name": names[seat],
            "arch": _signature(poke[seat], copies[seat]),
            "win": float(rewards[seat]) > float(rewards[1 - seat]),
        })
    return rows or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", required=True)
    ap.add_argument("--min-games", type=int, default=8,
                    help="hide archetypes rarer than this")
    ap.add_argument("--arch", default="grimmsnarl",
                    help="archetype substring to list PILOTS for — the "
                         "demonstrator shortlist")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    seats: list[dict] = []
    rate_of: list[float] = []
    parsed = skipped = 0
    for d in args.dir:
        dp = ROOT / d
        rat = _ratings(dp)
        for path in sorted(dp.glob("*.json")):
            rows = _both_seats(path)
            if not rows:
                skipped += 1
                continue
            parsed += 1
            try:
                r = rat.get(int(path.stem))
            except ValueError:
                r = None
            if r is not None:
                rate_of.append(r)
            for row in rows:
                row["rating"] = r
                seats.append(row)

    if not seats:
        print("nothing parsed")
        return 1

    print(f"\n=== DAY CENSUS — {parsed} games parsed, {skipped} skipped, "
          f"{len(seats)} seat-appearances ===")
    print(f"  dumps: {', '.join(args.dir)}")
    if rate_of:
        print(f"  episode avg_score: min {min(rate_of):.0f}  "
              f"median {statistics.median(rate_of):.0f}  max {max(rate_of):.0f}"
              f"   ({len(rate_of)} of {parsed} joined to manifest.csv)")
        print("  ⚠ §8i: the feed is censored below ~1055 — this is the TOP of "
              "the ladder,\n     not the field at our 1027. Demonstrators only, "
              "never an anchor.")

    by_arch: dict[str, list[dict]] = defaultdict(list)
    for s in seats:
        by_arch[s["arch"]].append(s)
    print(f"\n  {'archetype':<34}{'seats':>7}{'share':>8}{'WR':>8}"
          f"{'mean rating':>13}")
    for arch, ss in sorted(by_arch.items(), key=lambda kv: -len(kv[1])):
        if len(ss) < args.min_games:
            continue
        rs = [s["rating"] for s in ss if s["rating"] is not None]
        print(f"  {arch[:33]:<34}{len(ss):>7}{len(ss)/len(seats):>8.1%}"
              f"{sum(s['win'] for s in ss)/len(ss):>8.1%}"
              f"{(statistics.mean(rs) if rs else float('nan')):>13.0f}")
    shown = sum(len(s) for s in by_arch.values() if len(s) >= args.min_games)
    print(f"  ({len(by_arch)} archetypes total; "
          f"{len(seats)-shown} seats in archetypes under --min-games)")

    # ⚠ The control that stops the archetype table being over-read. At the top
    # of the ladder a deck can be carried by ONE pilot, and then its "win rate"
    # is that pilot's statistic wearing a deck's name -- the §8bn mistake.
    print(f"\n=== PILOT CONCENTRATION — is that WR the deck, or one player? ===")
    print(f"  {'archetype':<28}{'seats':>7}{'top pilot':<26}"
          f"{'their share':>12}{'their WR':>10}{'everyone else':>15}")
    for arch, ss in sorted(by_arch.items(), key=lambda kv: -len(kv[1])):
        if len(ss) < args.min_games:
            continue
        c = Counter(s["name"] for s in ss)
        nm, n = c.most_common(1)[0]
        mine = [s for s in ss if s["name"] == nm]
        rest = [s for s in ss if s["name"] != nm]
        rest_wr = (sum(s["win"] for s in rest) / len(rest)) if rest else float("nan")
        flag = "🔴" if n / len(ss) >= 0.5 else "  "
        print(f"{flag}{arch[:27]:<28}{len(ss):>7}{nm[:25]:<26}"
              f"{n/len(ss):>12.1%}{sum(s['win'] for s in mine)/len(mine):>10.1%}"
              f"{rest_wr:>14.1%}"
              f"{'' if rest else ' (n=0)'}")
    print("  🔴 = one pilot is at least half the archetype's games. That row's "
          "win rate is\n     a PILOT statistic, not a deck statistic. Do not "
          "quote it as a deck's strength.")

    by_player: dict[str, list[dict]] = defaultdict(list)
    for s in seats:
        by_player[s["name"]].append(s)
    print(f"\n=== TOP {args.top} PILOTS BY APPEARANCES ===")
    print(f"  {'player':<34}{'games':>7}{'WR':>8}{'mean rating':>13}  main deck")
    for nm, ss in sorted(by_player.items(), key=lambda kv: -len(kv[1]))[:args.top]:
        rs = [s["rating"] for s in ss if s["rating"] is not None]
        main = Counter(s["arch"] for s in ss).most_common(1)[0][0]
        print(f"  {nm[:33]:<34}{len(ss):>7}"
              f"{sum(s['win'] for s in ss)/len(ss):>8.1%}"
              f"{(statistics.mean(rs) if rs else float('nan')):>13.0f}  "
              f"{main[:30]}")

    want = [s for s in seats if args.arch.lower() in s["arch"].lower()]
    print(f"\n=== {args.arch.upper()} PILOTS — the demonstrator shortlist "
          f"({len(want)} seat-appearances) ===")
    if not want:
        print("  none found in these dumps.")
        return 0
    wp: dict[str, list[dict]] = defaultdict(list)
    for s in want:
        wp[s["name"]].append(s)
    print(f"  {'player':<34}{'games':>7}{'WR':>8}{'mean rating':>13}")
    ranked = sorted(
        wp.items(),
        key=lambda kv: -(statistics.mean(
            [s["rating"] for s in kv[1] if s["rating"] is not None] or [0])))
    for nm, ss in ranked[:args.top]:
        rs = [s["rating"] for s in ss if s["rating"] is not None]
        print(f"  {nm[:33]:<34}{len(ss):>7}"
              f"{sum(s['win'] for s in ss)/len(ss):>8.1%}"
              f"{(statistics.mean(rs) if rs else float('nan')):>13.0f}")
    print("\n  ⚠ Rating here is the EPISODE's avg_score (both seats), not the "
          "pilot's own\n     rating — a strong player in a lopsided game reads "
          "low. Ranking, not a score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
