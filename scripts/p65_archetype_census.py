"""Label BOTH seats of every replay, so a dump can be sized by MATCHUP.

Every archetype census this repo has run (`p9_field_census`, `p19_field_drift`)
labels the OPPONENT of one named seat. That answers "what does our field play".
It cannot answer either of the day-25 sizing questions, both of which are about
a *pair*:

  F1 -- how many MIRROR games does an expert dump hold, and how many of the
        expert's own decisions live inside them?
  F3 -- does the TRAINING corpus contain games where a Grimmsnarl seat faced
        the archetypes `PARKED-corpus-coverage.md` found missing, or do those
        games simply not exist at the band the episodes were mined from?

So: label seat 0 and seat 1 separately, each from the OTHER seat's observation
frames -- the same lower-bound reconstruction `p9_field_census.analyse` uses, so
the labels are comparable to every share already published (a card that never
left the deck is invisible; counts are "at least N").

    python -X utf8 scripts/p65_archetype_census.py --dir replays/ntumlnoob_31-07-2026 \\
        --player ntumlnoob
    python -X utf8 scripts/p65_archetype_census.py --dir replays/2026-07-26 \\
        --dir replays/2026-07-27 --dir replays/2026-07-28 --dir replays/2026-07-29

⚠ A game whose two seats carry the SAME team name cannot be split by name; it is
counted and labelled by seat index, never by name.
"""
from __future__ import annotations

import argparse
import csv
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

from p9_field_census import _cid, _signature  # noqa: E402
from sa import cards as cdb  # noqa: E402

# ⚠ The label `_signature` gives OUR archetype. It is NOT "Grimmsnarl ex" --
# the deepest stage of the line is `Marnie's Grimmsnarl ex`, and hardcoding the
# short name silently reports 0.0% mirror on a dump that is 45% mirror.
MIRROR = "Marnie's Grimmsnarl ex"


def _load_avg_scores(d: Path) -> dict[int, float]:
    """episode_id -> avg_score, from the dump's manifest.csv if it has one."""
    out: dict[int, float] = {}
    mf = d / "manifest.csv"
    if not mf.is_file():
        return out
    for row in csv.DictReader(mf.read_text(encoding="utf-8-sig").splitlines()):
        try:
            out[int(row["episode_id"])] = float(row["avg_score"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


class Seat:
    """One seat of one game, as reconstructed from the other seat's frames."""

    def __init__(self) -> None:
        self.poke: Counter = Counter()
        self.max_copies: dict[int, int] = defaultdict(int)
        self.decisions = 0        # selects with >=2 options made BY this seat

    def label(self) -> str:
        return _signature(self.poke, self.max_copies)


def scan(path: Path) -> tuple[list[Seat], list[str]] | None:
    """Both seats of one replay. Seat i's cards come from seat (1-i)'s frames."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        return None
    names = (d.get("info") or {}).get("TeamNames") or []
    seats = [Seat(), Seat()]

    for v in d["steps"][0][0].get("visualize") or []:
        obs = v.get("obs")
        if not obs or not obs.get("current"):
            continue
        st = obs["current"]
        me = st.get("yourIndex")
        if me not in (0, 1):
            continue

        sel = obs.get("select") or {}
        if len(sel.get("option") or []) >= 2 and st.get("result") == -1:
            seats[me].decisions += 1

        try:
            op = st["players"][1 - me]
        except (KeyError, IndexError, TypeError):
            continue
        g = seats[1 - me]

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
            if n > g.max_copies[cid]:
                g.max_copies[cid] = n
            if cdb.is_pokemon(cid):
                g.poke[cid] += 1
    return seats, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, dest="dirs")
    ap.add_argument("--player", action="append", default=[],
                    help="count decisions for these team name(s) specifically "
                         "(F1: the expert whose mirror games we are sizing)")
    ap.add_argument("--band", type=float, default=None,
                    help="keep only episodes with manifest avg_score >= this")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--emit-mirror", default=None,
                    help="write the path of every MIRROR game to this file, "
                         "one per line (the input to the F1 mirror corpus)")
    args = ap.parse_args()
    emit: list[str] = []

    keep = set(args.player)
    errs: Counter = Counter()
    pairs: Counter = Counter()          # (seat0 label, seat1 label) sorted
    per_seat: Counter = Counter()
    mirror_games = 0
    mirror_dec_player = 0               # expert decisions inside mirror games
    all_dec_player = 0
    player_games = 0
    vs_arch_dec: Counter = Counter()    # opponent archetype -> our-side decisions
    vs_arch_games: Counter = Counter()
    bands: list[float] = []
    mirror_bands: list[float] = []
    n = 0

    for dname in args.dirs:
        d = Path(dname)
        if not d.is_dir():
            print(f"  (skipping missing {d})")
            continue
        avg = _load_avg_scores(d)
        nd = 0
        for path in sorted(d.glob("*.json")):
            if not path.stem.isdigit():
                continue
            score = avg.get(int(path.stem))
            if args.band is not None and (score is None or score < args.band):
                continue
            try:
                got = scan(path)
            except Exception as exc:  # noqa: BLE001
                errs[f"{type(exc).__name__}: {exc}"] += 1
                continue
            if got is None:
                errs["not a replay dict"] += 1
                continue
            seats, names = got
            labels = [s.label() for s in seats]
            for lb in labels:
                per_seat[lb] += 1
            pairs[tuple(sorted(labels))] += 1
            if score is not None:
                bands.append(score)
            is_mirror = labels[0] == labels[1] == MIRROR
            if is_mirror:
                mirror_games += 1
                emit.append(str(path))
                if score is not None:
                    mirror_bands.append(score)

            # seat-resolved accounting for the named player (F1) or, with no
            # --player, for every Grimmsnarl seat in the dump (F3)
            # ⚠ SUBSTRING, not equality: the demonstrator appears as
            # `李秉叡（ntumlnoob）`, so an exact `--player ntumlnoob` matches
            # zero seats and reports a zero-game corpus rather than an error.
            idxs = ([i for i, nm in enumerate(names)
                     if any(k.lower() in nm.lower() for k in keep)] if keep
                    else [i for i, lb in enumerate(labels) if lb == MIRROR])
            for i in idxs:
                if keep and labels[i] != MIRROR:
                    continue          # only size the expert's Grimmsnarl seats
                player_games += 1
                all_dec_player += seats[i].decisions
                opp = labels[1 - i]
                vs_arch_dec[opp] += seats[i].decisions
                vs_arch_games[opp] += 1
                if opp == MIRROR:
                    mirror_dec_player += seats[i].decisions
            n += 1
            nd += 1
        print(f"  {d.name:<28} {nd:>5} games")

    if not n:
        print("no games")
        return 1

    print(f"\n{n} games, {n * 2} seats"
          + (f"; manifest avg_score {min(bands):.0f}-{max(bands):.0f} "
             f"(mean {sum(bands) / len(bands):.0f})" if bands else ""))

    print(f"\n=== SEAT ARCHETYPE SHARE (both seats, n={n * 2}) ===")
    for lb, c in per_seat.most_common(args.top):
        print(f"  {lb:<28} {c:>6}  {c / (n * 2):6.1%}")

    print(f"\n=== MATCHUP PAIRS (n={n}) ===")
    for (a, b), c in pairs.most_common(args.top):
        print(f"  {a:<26} vs {b:<26} {c:>5}  {c / n:6.1%}")

    print(f"\n=== MIRROR ({MIRROR} both seats) ===")
    print(f"  mirror games            {mirror_games:>6}  {mirror_games / n:6.1%}")
    if mirror_bands:
        print(f"  their manifest avg_score {min(mirror_bands):.0f}-"
              f"{max(mirror_bands):.0f} (mean "
              f"{sum(mirror_bands) / len(mirror_bands):.0f})")

    who = "/".join(sorted(keep)) if keep else f"every {MIRROR} seat"
    print(f"\n=== DECISIONS BY OPPONENT ARCHETYPE, for {who} ===")
    print(f"  {'opponent':<28}{'games':>7}{'decisions':>11}{'dec/game':>10}")
    for opp, dec in vs_arch_dec.most_common(args.top):
        g = vs_arch_games[opp]
        print(f"  {opp:<28}{g:>7}{dec:>11}{dec / max(g, 1):>10.1f}")
    print(f"  {'TOTAL':<28}{player_games:>7}{all_dec_player:>11}"
          f"{all_dec_player / max(player_games, 1):>10.1f}")
    print(f"\n  MIRROR decisions: {mirror_dec_player} of {all_dec_player} "
          f"({mirror_dec_player / max(all_dec_player, 1):.1%})")

    if args.emit_mirror:
        Path(args.emit_mirror).write_text("\n".join(emit) + "\n",
                                          encoding="utf-8")
        print(f"\nwrote {len(emit)} mirror-game paths -> {args.emit_mirror}")

    if errs:
        print("\nerrors:")
        for k, c in errs.most_common(8):
            print(f"  {c:>5}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
