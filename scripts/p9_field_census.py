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

    🔴 **INDEXED BY NAME, NOT BY ID, AND THAT IS THE WHOLE POINT (day 22).**
    Until day 22 this resolved `evolvesFrom` to a single card id and `break`ed on
    the first match, so of the **106 basic printings that share a name with
    another**, only one got its evolutions attached -- **228 broken links**. A
    deck seen through Abra #741 was labelled "Abra" while the identical deck seen
    through Abra #109 was labelled "Alakazam": ONE archetype split by which
    reprint the opponent happened to draw. That is the precise failure this
    function exists to prevent, one level down, and it reached the anchors --
    **Riolu #677 and #974 both lost Mega Lucario ex**, and `rule:v10`'s field
    share is a published weight (§8ac, `p33.ANCHORS`).

    ⚠ **Linking every printing is not enough on its own.** `_signature` groups
    observed cards into lines by ROOT, and two printings of one basic are two
    roots, so a Riolu #974 in play still would not join its own Mega Lucario ex
    -- three Lucario games label as "Hariyama" that way. Names are the only
    stable key, so the index is names end to end.
    """
    parent: dict[str, str] = {}
    children: dict[str, set[str]] = defaultdict(set)
    for cid in cdb.cards():
        if not cdb.is_pokemon(cid):
            continue
        pre = cdb.card(cid).get("evolvesFrom")
        if not pre:
            continue
        parent.setdefault(_name(cid), str(pre))
        children[str(pre)].add(_name(cid))
    return parent, children


_PARENT, _CHILDREN = _evolution_index()
# one representative id per name, for prize_value / is_basic
_ID_BY_NAME: dict[str, int] = {}
for _c in cdb.cards():
    if cdb.is_pokemon(_c):
        _ID_BY_NAME.setdefault(_name(_c), _c)


def _root(name: str) -> str:
    seen = set()
    while name in _PARENT and name not in seen:
        seen.add(name)
        name = _PARENT[name]
    return name


def _deepest(root: str, observed: set[str]) -> str:
    """The stage that NAMES this line: the deepest one, preferring one we saw."""
    def key(n: str, d: int) -> tuple:
        rid = _ID_BY_NAME.get(n, 0)
        return (d, n in observed, cdb.prize_value(rid) if rid else 0, n)

    best, best_key = root, key(root, 0)
    stack = [(root, 0)]
    seen = {root}
    while stack:
        nm, d = stack.pop()
        k = key(nm, d)
        if k > best_key:
            best, best_key = nm, k
        for ch in _CHILDREN.get(nm, ()):
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

    🔴 **AND THE 1-OF GUARD WAS NOT ENOUGH, because `ex` outranked COPIES.** The
    old order tried every `ex` line first and only fell back to copy count within
    that group, so a **2-of** tech beat a 4/3/3 engine: five Abra/Kadabra/Alakazam
    games running 2x Dunsparce as tech label as "Dudunsparce ex" the moment the
    evolution index is repaired. The same trap as the Fezandipiti one, one copy
    higher, and it was masked by the orphan bug rather than fixed. **Copies now
    dominate and `ex`/evolved only break ties** -- which is what "a deck's engine
    is played in multiples" actually means. ⚠ A deck whose engine and its tech
    run the SAME count is still decided by ex-ness, and that is a real remaining
    ambiguity, not a solved case.
    """
    if not poke:
        return "(no Pokemon seen)"
    observed = {_name(cid) for cid in poke}

    # Collapse every observed Pokemon into its evolution LINE -- keyed by the
    # root's NAME, so two printings of one basic are one line -- and score the
    # line by the most copies any stage showed, the total copies across its
    # stages, and whether the deck ever actually EVOLVED it.
    lines: dict[str, dict] = {}
    for cid in poke:
        r = _root(_name(cid))
        e = lines.setdefault(r, {"copies": 0, "obs": 0, "mass": 0, "evo": 0})
        e["copies"] = max(e["copies"], copies.get(cid, 0))
        e["mass"] += copies.get(cid, 0)
        e["obs"] += poke[cid]
        if not _is_basic(cid):
            e["evo"] = 1
    for r, e in lines.items():
        e["label"] = _deepest(r, observed)

    # A deck's engine is played in multiples; a 1-of is a tech, not an
    # identity. (This is what mislabelled an Alakazam deck "Fezandipiti ex".)
    cand = [r for r, e in lines.items() if e["copies"] >= 2] or list(lines)

    def rank(r: str) -> tuple:
        e = lines[r]
        rid = _ID_BY_NAME.get(e["label"], 0)
        # A deck EVOLVES its engine and merely plays its support basics. That
        # separates 2x Mega Lucario ex (behind 3x Solrock) from the Solrock, and
        # it is the only signal here about deck ROLE rather than count. Then
        # total copies across the line -- a 4/3/3 Abra line outweighs a 3/2
        # Dunsparce tech even where their single largest stage ties.
        return (e["evo"], e["mass"], e["copies"],
                1 if rid and cdb.prize_value(rid) >= 2 else 0,
                e["obs"], r)

    return lines[max(cand, key=rank)]["label"]


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
