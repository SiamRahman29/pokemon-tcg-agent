"""What is the field ACTUALLY playing? -- decompose the 63% "other" (HANDOFF rule 16).

**Why this exists.** `p8_optv3_replays.py` reported the archetype mix of our 54
real ladder games against a **hardcoded four-archetype classifier**, so 63% of
opponents landed in a bucket called "other". That single number is the measured
cause of the arena/ladder divergence: our A/Bs are anchored on two decks that
cover ~33% of the field. You cannot build anchors for a bucket named "other".

This script names it. For every replay it reconstructs what the opponent
actually showed -- every Pokemon that reached play plus everything that reached
the discard -- and labels the deck by its signature Pokemon. Output:

  1. the archetype table (share, our win rate) with real names, no "other"
  2. for each archetype above a threshold, the reconstructed card list with
     observed copy counts -- the input to `decks/<name>.py`

⚠ Reconstruction is LOWER-BOUND by construction: a card that never left the
deck is invisible. Counts are the max simultaneously observed (in play +
discard + their hand is NOT visible), so treat them as "at least N".

    python -X utf8 scripts/p9_field_census.py --dir replays/submission_optv3
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

from sa import cards as cdb  # noqa: E402

US = "Scio"


def _cid(c) -> int:
    return c["id"] if isinstance(c, dict) else int(c)


def _name(cid: int) -> str:
    return str(cdb.card(cid).get("name") or f"#{cid}")


def _is_basic(cid: int) -> bool:
    return cdb.is_basic_pokemon(cid)


def _evolution_index() -> tuple[dict[int, int], dict[int, list[int]]]:
    """parent[cid] -> pre-evolution cid, and children[cid] -> evolutions.

    The card db stores `evolvesFrom` as a NAME, so this resolves names to ids.
    Needed because a short game may only ever show us the Kadabra -- naming
    that deck "Kadabra" and the next one "Alakazam" splits one archetype in
    two, which is exactly how a field census lies about concentration.
    """
    by_name: dict[str, list[int]] = defaultdict(list)
    for cid in cdb.cards():
        if cdb.is_pokemon(cid):
            by_name[str(cdb.card(cid).get("name") or "")].append(cid)
    parent: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    for cid in cdb.cards():
        if not cdb.is_pokemon(cid):
            continue
        pre = cdb.card(cid).get("evolvesFrom")
        if not pre:
            continue
        for pid in by_name.get(str(pre), []):
            parent[cid] = pid
            children[pid].append(cid)
            break
    return parent, children


_PARENT, _CHILDREN = _evolution_index()


def _root(cid: int) -> int:
    seen = set()
    while cid in _PARENT and cid not in seen:
        seen.add(cid)
        cid = _PARENT[cid]
    return cid


def _deepest(root: int, observed: set[int]) -> int:
    """The card that NAMES this line: its deepest stage, preferring one we saw."""
    best, best_key = root, (0, root in observed, cdb.prize_value(root), -root)
    stack = [(root, 0)]
    seen = {root}
    while stack:
        cid, d = stack.pop()
        key = (d, cid in observed, cdb.prize_value(cid), -cid)
        if key > best_key:
            best, best_key = cid, key
        for ch in _CHILDREN.get(cid, []):
            if ch not in seen:
                seen.add(ch)
                stack.append((ch, d + 1))
    return best


def _signature(poke: Counter, copies: dict[int, int]) -> str:
    """Name the deck by the engine Pokemon it is actually built around.

    ⚠ The obvious heuristic -- "the highest-prize Pokemon" -- is WRONG, and it
    was wrong here first: an Abra/Kadabra/Alakazam deck running a **single**
    Fezandipiti ex as a draw tech got labelled "Fezandipiti ex", which split
    the field's largest archetype across four names. So: **ignore 1-ofs.**
    A card the deck runs one copy of is a tech, not an identity.
    """
    if not poke:
        return "(no Pokemon seen)"
    observed = set(poke)

    # Collapse every observed Pokemon into its evolution LINE, and score the
    # line by the most copies any of its stages showed.
    lines: dict[int, dict] = {}
    for cid in poke:
        r = _root(cid)
        e = lines.setdefault(r, {"copies": 0, "obs": 0})
        e["copies"] = max(e["copies"], copies.get(cid, 0))
        e["obs"] += poke[cid]
    for r, e in lines.items():
        e["name_id"] = _deepest(r, observed)

    # A deck's engine is played in multiples; a 1-of is a tech, not an
    # identity. (This is what mislabelled an Alakazam deck "Fezandipiti ex".)
    cand = [r for r, e in lines.items() if e["copies"] >= 2] or list(lines)

    def best(pool: list[int]) -> str:
        pool.sort(key=lambda r: (-lines[r]["copies"], -lines[r]["obs"], r))
        return _name(lines[pool[0]]["name_id"])

    exs = [r for r in cand if cdb.prize_value(lines[r]["name_id"]) >= 2]
    if exs:
        return best(exs)
    evo = [r for r in cand if not _is_basic(lines[r]["name_id"])]
    if evo:
        return best(evo)
    return best(cand)


class Game:
    def __init__(self, path: Path, opp: str):
        self.path = path
        self.opp = opp
        self.result = 0
        self.poke: Counter = Counter()   # id -> observations in play
        self.max_copies: dict[int, int] = defaultdict(int)
        self.stadium: Counter = Counter()


def analyse(path: Path, errs: Counter, us: set[str] = frozenset()) -> Game | None:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        errs["file is a bare step-array, not a replay"] += 1
        return None
    names = (d.get("info") or {}).get("TeamNames") or []
    us = set(us) or {US}
    if not (us & set(names)):
        errs[f"no {'/'.join(sorted(us))} seat"] += 1
        return None
    ours = {i for i, n in enumerate(names) if n in us}
    opp_name = next((n for i, n in enumerate(names) if i not in ours), "?")
    g = Game(path, opp_name)
    g.result = d["rewards"][min(ours)]

    for v in d["steps"][0][0].get("visualize") or []:
        obs = v.get("obs")
        if not obs or not obs.get("current"):
            continue
        st = obs["current"]
        me = st.get("yourIndex")
        if me not in ours:
            continue
        try:
            op = st["players"][1 - me]
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
            if n > g.max_copies[cid]:
                g.max_copies[cid] = n
            if cdb.is_pokemon(cid):
                g.poke[cid] += 1
        stad = st.get("stadium")
        if stad:
            g.stadium[_cid(stad if not isinstance(stad, list) else stad[0])] += 1
    return g


def _rating_report(games, lb_path: Path, by_arch) -> None:
    """How strong were these opponents, really? Joins team names to an LB dump.

    ⚠ This is the check that stops the census being over-claimed. The census
    describes **the opponents our agent was matched against**, which is NOT the
    same as "our rating band" -- on the v3 dump the opponents average ~60-85
    points BELOW us. Read every share and win rate in this file as a property of
    that pool.
    """
    import statistics
    try:
        lb = json.loads(lb_path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"\n  (no LB snapshot: {exc})")
        return
    scored = [(g, float(lb[g.opp][1])) for g in games if g.opp in lb]
    print(f"\n=== OPPONENT STRENGTH ({len(scored)}/{len(games)} matched to "
          f"{lb_path.name}) ===")
    if not scored:
        print("  no team names matched -- is the snapshot current?")
        return
    sc = [s for _, s in scored]
    print(f"  rating: mean {statistics.mean(sc):.0f}  "
          f"median {statistics.median(sc):.0f}  "
          f"min {min(sc):.0f}  max {max(sc):.0f}")
    print("  ⚠ compare this to OUR score at the time. If the pool sits below "
          "us, the\n     shares below describe a weaker field than our own "
          "band, and a win rate\n     near 50% there is worse than it looks.")
    print(f"  {'archetype':<32}{'games':>6}{'mean opp rating':>17}{'our WR':>9}")
    for arch, gs in sorted(by_arch.items(), key=lambda kv: -len(kv[1])):
        rs = [float(lb[g.opp][1]) for g in gs if g.opp in lb]
        if len(rs) < 2:
            continue
        w = sum(1 for g in gs if g.result > 0)
        print(f"  {arch:<32}{len(gs):>6}{statistics.mean(rs):>17.0f}"
              f"{w/len(gs):>9.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["replays/submission_optv3"],
                    help="one or more replay dumps; pass several to pool them. "
                         "⚠ different dumps are different AGENTS at different "
                         "ratings -- pooling mixes two populations on purpose, "
                         "to show which archetypes survive both")
    ap.add_argument("--detail-min", type=int, default=3,
                    help="reconstruct decklists for archetypes seen >= N times")
    ap.add_argument("--lb", default=None,
                    help="JSON dump of the leaderboard ({team: [rank, score]}) "
                         "to report how strong these opponents actually were. "
                         "Build it with the full-LB command in HANDOFF section 5.")
    ap.add_argument("--us", action="append", default=[],
                    help=f"the seat to census FROM (default {US!r}). Repeat for "
                         "a team that renamed. Point it at a third-party dump's "
                         "owner to census THEIR opponents.")
    ap.add_argument("--emit-players",
                    help="write the opponent team names of one archetype to "
                         "this file, one per line -- a reproducible "
                         "`--players-file` for build_policy_dataset.py")
    ap.add_argument("--emit-archetype", default="grimmsnarl",
                    help="substring of the archetype label --emit-players "
                         "selects (default: our own deck)")
    args = ap.parse_args()

    errs: Counter = Counter()
    games: list[Game] = []
    src: dict[Path, str] = {}
    for dirname in args.dir:
        for path in sorted(Path(dirname).glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                g = analyse(path, errs, set(args.us))
            except Exception as exc:  # noqa: BLE001
                errs[f"{type(exc).__name__}: {exc}"] += 1
                continue
            if g is not None:
                src[g.path] = dirname
                games.append(g)

    by_arch: dict[str, list[Game]] = defaultdict(list)
    for g in games:
        by_arch[_signature(g.poke, g.max_copies)].append(g)

    tot = len(games)
    print(f"\n=== {', '.join(args.dir)}: {tot} games, "
          f"{len(by_arch)} distinct opponent archetypes ===")
    wins = sum(1 for g in games if g.result > 0)
    print(f"  overall win rate {wins}/{tot} = {wins/max(tot,1):.1%}")

    multi = len(args.dir) > 1
    print("\n=== THE REAL FIELD (signature Pokemon = highest-prize / evolved) ===")
    if multi:
        print("  ⚠ per-dump shares are shown too: an archetype that swings "
              "between dumps is NOT a stable anchor candidate")
    hdr = f"  {'archetype':<32}{'games':>6}{'share':>8}{'won':>6}{'WR':>8}"
    if multi:
        hdr += "".join(f"{Path(d).name[:14]:>16}" for d in args.dir)
    print(hdr)
    rows = sorted(by_arch.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for arch, gs in rows:
        w = sum(1 for g in gs if g.result > 0)
        line = (f"  {arch:<32}{len(gs):>6}{len(gs)/tot:>7.1%}{w:>6}"
                f"{w/len(gs):>8.1%}")
        if multi:
            for d in args.dir:
                n = sum(1 for g in gs if src[g.path] == d)
                tn = sum(1 for g in games if src[g.path] == d)
                line += f"{n:>9}{n/max(tn,1):>7.0%}"
        print(line)

    cum = 0
    print("\n=== COVERAGE: how many anchors buy how much of the field? ===")
    for i, (arch, gs) in enumerate(rows, 1):
        cum += len(gs)
        print(f"  top {i:>2} anchors ({arch:<28}) -> {cum/tot:>6.1%} of the field")
        if cum / tot > 0.85:
            break

    print("\n=== DISTINCT OPPONENT TEAMS (a repeat opponent is not a repeat deck) ===")
    opps = Counter(g.opp for g in games)
    print(f"  {len(opps)} distinct teams over {tot} games; "
          f"most frequent: {', '.join(f'{k} x{v}' for k, v in opps.most_common(4))}")

    if args.lb:
        _rating_report(games, Path(args.lb), by_arch)

    if args.emit_players:
        want = [a for a in by_arch if args.emit_archetype.lower() in a.lower()]
        names = sorted({g.opp for a in want for g in by_arch[a]})
        Path(args.emit_players).write_text(
            "\n".join(names) + "\n", encoding="utf-8")
        print(f"\n=== EMITTED {len(names)} team names playing "
              f"{args.emit_archetype!r} -> {args.emit_players} ===")
        print(f"  matched archetypes: {want}")
        print("  ⚠ these are the SEATS OPPOSITE the census subject, so the "
              "file is a\n     same-deck, same-window control population -- "
              "feed it to build_policy_dataset.py --players-file")

    for arch, gs in rows:
        if len(gs) < args.detail_min:
            continue
        print(f"\n=== RECONSTRUCTION: {arch}  ({len(gs)} games) ===")
        print("   'at least N' = max copies observed in any single game "
              "(hand + deck are invisible)")
        best: dict[int, int] = defaultdict(int)
        seen_in: Counter = Counter()
        for g in gs:
            for cid, n in g.max_copies.items():
                if n > best[cid]:
                    best[cid] = n
                seen_in[cid] += 1
        pk = [(c, n) for c, n in best.items() if cdb.is_pokemon(c)]
        rest = [(c, n) for c, n in best.items() if not cdb.is_pokemon(c)]
        for title, items in (("POKEMON", pk), ("TRAINERS / ENERGY", rest)):
            print(f"  -- {title} --")
            for cid, n in sorted(items, key=lambda t: (-seen_in[t[0]], -t[1])):
                print(f"     {n}x  {_name(cid):<34} #{cid:<5} "
                      f"seen in {seen_in[cid]}/{len(gs)} games")
        stad = Counter()
        for g in gs:
            stad.update(g.stadium)
        if stad:
            print("  -- STADIUM (observations) --")
            for cid, n in stad.most_common(5):
                print(f"     {_name(cid)} #{cid}  x{n}")

    if errs:
        print("\nerrors/skips:")
        for k, v in errs.most_common(6):
            print(f"  {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
