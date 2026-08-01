"""Record local games as Kaggle-format replays -- watchable AND analysable.

**Why this exists (day 15).** `arena.py` archives one summary row per game
(winner / turns / selects / latency / pool, `arena.py:281-294`). After fifteen
days that means:

  - the **five anchors carry 71.5% of every weighted verdict in this repo** and
    nobody on this project has ever watched one play a turn;
  - there are **no trajectories**, so the RL variance probe has no data source;
  - and when an A/B says 0.768 there is no way to look at what went wrong.

`harness.Recorder` fixes all three at once, and the reason it is cheap is that
the format is not ours to invent: `cg.game.visualize_data()` emits exactly the
structure Kaggle puts at `steps[0][0]["visualize"]`, so a recorded local game is
read unmodified by **every replay tool already in this repo**:

    python -X utf8 scripts/p9_field_census.py --dir out/replays/mirror --us seat0
    python -X utf8 scripts/build_policy_dataset.py --out artifacts/x out/replays/...

and by the official viewer via `notebooks/visualizer.html` (open it, pick a
recorded .json, it POSTs to ptcgvis.heroz.jp).

    # watch our live net play the Alakazam anchor, 3 games
    python -X utf8 scripts/p20_record_games.py --a "bc:v5,net=out/policy_v5.npz" \
        --b rule:alakazam5 --deck-b alakazam5 --games 3 --out out/replays/alakazam

    # the anchors playing EACH OTHER -- day-15 item 6, nobody has seen this
    python -X utf8 scripts/p20_record_games.py --a rule:alakazam5 --deck-a alakazam5 \
        --b rule:crustle --deck-b crustle --games 4 --out out/replays/anchors

⚠ Recording is opt-in and is NOT how you measure strength -- it writes multi-MB
files per game. Use `arena.py play` for A/Bs; use this to look at them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import harness, sdk  # noqa: E402
sdk.load()  # puts the licensed `cg` engine on sys.path -- agents import it
from arena import build_agent, resolve_deck  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="agent spec for seat 0 "
                    "(same grammar as arena.py, e.g. 'bc:v5,net=out/policy_v5.npz')")
    ap.add_argument("--b", required=True, help="agent spec for seat 1")
    ap.add_argument("--deck-a", default="grimmsnarl")
    ap.add_argument("--deck-b", default=None,
                    help="defaults to --deck-a (a mirror)")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--out", default="out/replays/rec")
    ap.add_argument("--swap", action="store_true",
                    help="alternate seats between games, like the arena does")
    ap.add_argument("--no-obs", action="store_true",
                    help="drop the per-select observations -- much smaller "
                         "files that the VIEWER still renders, but that the "
                         "repo's replay tools can no longer read")
    args = ap.parse_args()

    name_a, deck_a = resolve_deck(args.deck_a)
    name_b, deck_b = resolve_deck(args.deck_b or args.deck_a)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"  seat0 {args.a!r} on {name_a}")
    print(f"  seat1 {args.b!r} on {name_b}")
    tally = {0: 0, 1: 0, 2: 0}
    for i in range(args.games):
        swap = args.swap and i % 2 == 1
        spec0, spec1 = (args.b, args.a) if swap else (args.a, args.b)
        d0, d1 = (deck_b, deck_a) if swap else (deck_a, deck_b)
        lbl0, agent0 = build_agent(spec0, list(d0))
        lbl1, agent1 = build_agent(spec1, list(d1))
        rec = harness.Recorder(names=(lbl0, lbl1), keep_obs=not args.no_obs)
        r = harness.play_game(agent0, agent1, list(d0), list(d1), recorder=rec)
        # normalise the tally to A's point of view regardless of seat
        if r.winner == 2:
            tally[2] += 1
        else:
            a_won = (r.winner == 1) if swap else (r.winner == 0)
            tally[0 if a_won else 1] += 1
        path = rec.dump(outdir / f"game{i:03d}.json")
        mb = path.stat().st_size / 1e6
        print(f"  game {i:>3}  winner={r.winner} ({lbl0 if r.winner==0 else lbl1 if r.winner==1 else 'draw'})"
              f"  turns={r.turns:>3} selects={r.selects:>4}"
              f"  -> {path} ({mb:.1f} MB)")

    print(f"\n  A ({args.a}) won {tally[0]}/{args.games}, "
          f"lost {tally[1]}, drew {tally[2]}")
    print(f"  open notebooks/visualizer.html and pick a file from {outdir} "
          "to watch one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
