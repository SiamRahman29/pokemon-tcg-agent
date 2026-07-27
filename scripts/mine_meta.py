"""Mine deck meta from downloaded replay JSONs.

    python scripts/mine_meta.py <replay_dir> [<replay_dir> ...]

For each replay: team names, ratings (if the day's manifest.csv is next to the
files or in replays/<date>/), decks (60-card id lists), winner. Aggregates:

  * archetype win rates (archetype = the deck's Pokemon signature)
  * per-team deck usage for the highest-rated teams
  * exact decklists (most common per archetype), printed with card names
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decks.base import card_table  # noqa: E402

CARDS = card_table()


def card_name(cid: int) -> str:
    info = CARDS.get(cid)
    return info.name if info else f"#{cid}"


def archetype_of(deck: list[int]) -> str:
    """Signature: the highest-HP / most distinctive Pokemon names in the deck."""
    from collections import Counter as C

    poke = [cid for cid in deck
            if CARDS.get(cid) and CARDS[cid].category.strip().lower() == "pokémon"]
    if not poke:
        poke = deck
    counts = C(poke)
    # distinctive = evolved/ex names by count then id; take top 2 names
    names = []
    for cid, _ in counts.most_common():
        n = card_name(cid)
        base = n.replace(" ex", "").replace("Mega ", "")
        if any(base in x for x in names):
            continue
        stage = CARDS.get(cid).stage if CARDS.get(cid) else ""
        if "Basic" in stage and len(counts) > 3:
            # prefer evolutions/ex as signature, but keep basics as fallback
            continue
        names.append(n)
        if len(names) == 2:
            break
    if not names:
        names = [card_name(counts.most_common(1)[0][0])]
    return " / ".join(sorted(names))


def load_manifest_scores(rep_dir: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for cand in (rep_dir / "manifest.csv",
                 rep_dir.parent / "manifests" / rep_dir.name / "manifest.csv"):
        if cand.exists():
            with cand.open(encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    out[int(r["episode_id"])] = float(r["avg_score"])
            break
    return out


def main() -> int:
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        print(__doc__)
        return 1

    games = []  # (teams, decks, winner, avg_score)
    for rep_dir in dirs:
        scores = load_manifest_scores(rep_dir)
        for path in sorted(rep_dir.glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                teams = d["info"]["TeamNames"]
                rewards = d["rewards"]
                vis = d["steps"][0][0].get("visualize") or []
                decks = vis[0]["action"] if vis else None
                if not decks:  # fall back to step 1 actions
                    decks = [d["steps"][1][i]["action"] for i in range(2)]
                if rewards[0] is None or rewards[1] is None:
                    winner = 2
                else:
                    winner = (0 if rewards[0] > rewards[1]
                              else 1 if rewards[1] > rewards[0] else 2)
                ep = int(path.stem)
                games.append((teams, decks, winner, scores.get(ep, 0.0)))
            except Exception as exc:
                print(f"  skip {path.name}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)

    print(f"parsed {len(games)} games\n")

    # archetype stats
    arch_wdl = defaultdict(lambda: [0, 0, 0])   # w/d/l
    arch_users: dict[str, Counter] = defaultdict(Counter)
    arch_decks: dict[str, Counter] = defaultdict(Counter)
    team_best: dict[str, float] = defaultdict(float)
    team_arch: dict[str, Counter] = defaultdict(Counter)

    for teams, decks, winner, score in games:
        for seat in (0, 1):
            arch = archetype_of(decks[seat])
            res = 1 if winner == 2 else (0 if winner == seat else 2)
            arch_wdl[arch][res] += 1
            arch_users[arch][teams[seat]] += 1
            arch_decks[arch][tuple(sorted(decks[seat]))] += 1
            team_best[teams[seat]] = max(team_best[teams[seat]], score)
            team_arch[teams[seat]][arch] += 1

    print("=== archetypes by usage (win% excl. draws) ===")
    rows = sorted(arch_wdl.items(), key=lambda kv: -sum(kv[1]))
    for arch, (w, dr, l) in rows[:20]:
        n = w + l
        wr = w / n if n else 0.0
        print(f"  {w + dr + l:4d} games  {wr:5.1%}  {arch}  "
              f"({len(arch_users[arch])} teams)")

    print("\n=== top teams by best avg_score seen ===")
    for team, score in sorted(team_best.items(), key=lambda kv: -kv[1])[:25]:
        archs = ", ".join(f"{a} x{c}" for a, c in team_arch[team].most_common(2))
        print(f"  {score:7.1f}  {team}: {archs}")

    # dump the most common exact list of the top archetypes
    print("\n=== most common decklist per top archetype ===")
    for arch, (w, dr, l) in rows[:6]:
        decklist, cnt = arch_decks[arch].most_common(1)[0]
        print(f"\n--- {arch} (this exact list seen {cnt}x) ---")
        for cid, k in sorted(Counter(decklist).items(),
                             key=lambda kv: (-kv[1], kv[0])):
            print(f"  {k}x {card_name(cid)} [{cid}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
