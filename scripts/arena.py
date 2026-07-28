"""The arena: every strength claim in this project runs through here.

    python scripts/arena.py play A B [--matches N] [--deck-a SPEC] [--deck-b SPEC]
    python scripts/arena.py elo

`play` runs seat-swapped paired matches between two agent specs, appends one
JSONL row per game to `out/arena/games.jsonl` (the permanent archive), and
prints A's score with a Wilson 95% interval. `elo` fits Elo ratings over the
whole archive (`rule:iono` anchored at 1000).

Agent specs:

    rule:<name>  (sample rule-based agents: dragapult, iono, abomasnow, lucario)
    random       (uniform legal choice -- the floor to measure against)

A `rule:<name>` agent is bound to the deck it was tuned for, so pass its own
deck: `rule:iono --deck-a iono`, `rule:dragapult --deck-a dragapult_ex`,
`rule:abomasnow --deck-a mega_abomasnow_ex`, `rule:lucario --deck-a mega_lucario_ex`.

Deck specs: `sample` (the SDK sample deck), a `decks/` module name (`iono`,
`dragapult_ex`, ...), or a path to a headerless 60-line deck.csv.
"""
from __future__ import annotations

import argparse
import json
import random as _random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg import config              # noqa: E402
from ptcg.env import harness, sdk    # noqa: E402

ARENA_DIR = ROOT / "out" / "arena"
GAMES_PATH = ARENA_DIR / "games.jsonl"
SCHEMA = 1


# --- decks --------------------------------------------------------------------

def resolve_deck(spec: str) -> tuple[str, list[int]]:
    """Resolve a deck spec to (stable name, 60-card id list)."""
    if spec == "sample":
        path = config.find_sample_deck()
        if path is None:
            raise SystemExit("no sample deck found under data/")
        rows = path.read_text().split("\n")
        return "sample", [int(rows[i]) for i in range(60)]
    if spec.endswith(".csv"):
        rows = Path(spec).read_text().split("\n")
        return Path(spec).stem, [int(rows[i]) for i in range(60)]
    import importlib

    mod = importlib.import_module(f"decks.{spec}")
    counts: dict[int, int] = mod.DECKLIST
    return spec, [cid for cid, cnt in counts.items() for _ in range(cnt)]


# --- agents -------------------------------------------------------------------

def make_random_agent(deck: list[int], seed: int = 0) -> harness.Agent:
    """Uniform choice among the legal options -- the measurement floor."""
    rng = _random.Random(seed)

    def agent(obs_dict: dict) -> list:
        obs = sdk.api().to_observation_class(obs_dict)
        if obs.select is None:
            return list(deck)
        sel = obs.select
        k = min(max(sel.minCount, 1 if sel.maxCount else 0), sel.maxCount,
                len(sel.option))
        return rng.sample(range(len(sel.option)), k)

    return agent


def _flag_num(flags: set[str], prefix: str, cast):
    """Pull `<prefix><number>` out of a spec's flag set (e.g. `w48`, `mc12`)."""
    for f in flags:
        if f.startswith(prefix):
            rest = f[len(prefix):]
            try:
                return cast(rest)
            except ValueError:
                continue
    return None


def build_agent(spec: str, deck: list[int]) -> tuple[str, harness.Agent]:
    """Build (canonical name, agent). The name is what the archive records, so
    the same config always archives under the same identity."""
    kind = spec.split(":", 1)[0]
    if kind == "rule":
        # rule:<deck-name> -- a self-contained sample rule-based agent, bound to
        # the deck it was tuned for (pass the matching --deck so its card
        # counting is correct; see agentkit.rulebased.DECK_MODULE).
        from agentkit.rulebased import make_rule_agent

        rname = spec.split(":", 1)[1].split(",")[0]
        return f"rule:{rname}", make_rule_agent(rname, deck)
    if kind == "random":
        return "random", make_random_agent(deck)
    if kind == "search":
        # search[:tag][,noP][,noV][,w<N>][,roll][,mc<S>][,nc<S>] -- the sa
        # determinized-search agent. Every flag applies to THIS instance only,
        # so two configs can be A/B'd head-to-head in one process (the SA_*
        # env vars are module-level and would otherwise apply to both sides).
        from sa.agent import SearchAgent

        tag = spec.split(":", 1)[1] if ":" in spec else ""
        parts = tag.split(",")
        flags = {p.strip() for p in parts[1:]}
        no_pnet = True if "noP" in flags else None
        no_vnet = True if "noV" in flags else None
        # w<N>: determinizations per decision (the real compute knob)
        max_worlds = _flag_num(flags, "w", int)
        # roll: rollout to terminal instead of a 1.5-turn horizon + leaf eval.
        # mc/nc: main / minor per-decision time cap in seconds (rollouts need
        # more than the 4.5s default, and we use 3% of the pool today).
        rollout = "roll" in flags
        return ((f"search:{tag}" if tag else "search"),
                SearchAgent(deck, no_pnet=no_pnet, no_vnet=no_vnet,
                            max_worlds=max_worlds, rollout=rollout,
                            main_cap=_flag_num(flags, "mc", float),
                            minor_cap=_flag_num(flags, "nc", float)))
    if kind == "bc":
        # bc[:tag][,net=<path>] -- pure behavioral-clone policy agent. `net=`
        # pins this instance to a specific npz, so two candidate policies can
        # play each other head-to-head in one process.
        from sa.bcagent import PolicyAgent

        tag = spec.split(":", 1)[1] if ":" in spec else ""
        net_path = None
        for f in tag.split(",")[1:]:
            if f.strip().startswith("net="):
                net_path = f.strip()[4:]
        if net_path and not Path(net_path).exists():
            raise SystemExit(f"bc net not found: {net_path}")
        return (f"bc:{tag}" if tag else "bc"), PolicyAgent(deck, net_path)
    raise SystemExit(f"unknown agent spec: {spec!r}")


# --- play ---------------------------------------------------------------------

def cmd_play(args: argparse.Namespace) -> int:
    if config.find_sdk_dir() is None:
        print("cg engine not found (paste the sample submission into data/).")
        return 1
    sdk.load()

    deck_name_a, deck_a = resolve_deck(args.deck_a)
    deck_name_b, deck_b = resolve_deck(args.deck_b)
    name_a, agent_a = build_agent(args.a, deck_a)
    name_b, agent_b = build_agent(args.b, deck_b)

    games_path = Path(args.archive) if args.archive else GAMES_PATH
    games_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t_start = time.monotonic()
    # Append+flush per game. Buffering rows until the end meant a killed run
    # lost every game it had played, which made long runs un-abortable.
    archive_fh = games_path.open("a", encoding="utf-8")

    def on_game(match: int, a_seat: int, r: harness.GameResult) -> None:
        names = (name_a, name_b) if a_seat == 0 else (name_b, name_a)
        decks = ((deck_name_a, deck_name_b) if a_seat == 0
                 else (deck_name_b, deck_name_a))
        row = {
            "schema": SCHEMA, "ts": time.time(), "match": match,
            "agent0": names[0], "agent1": names[1],
            "deck0": decks[0], "deck1": decks[1],
            "winner": r.winner, "turns": r.turns, "selects": r.selects,
            "lat0": harness.latency_summary(r.decision_ms[0]),
            "lat1": harness.latency_summary(r.decision_ms[1]),
        }
        rows.append(row)
        archive_fh.write(json.dumps(row) + "\n")
        archive_fh.flush()
        print(f"  match {match} seat{a_seat}: winner={r.winner} "
              f"turns={r.turns} selects={r.selects}", flush=True)

    print(f"{name_a} [{deck_name_a}] vs {name_b} [{deck_name_b}], "
          f"{args.matches} paired matches ({2 * args.matches} games)...")
    try:
        res = harness.evaluate_paired(agent_a, agent_b, deck_a, deck_b,
                                      matches=args.matches, on_game=on_game)
    finally:
        if rows:  # archive whatever finished, even on interrupt
            with games_path.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")

    dt = time.monotonic() - t_start
    print(f"\nA={name_a}: score={res['score']:.3f} "
          f"[{res['wilson_low']:.3f}, {res['wilson_high']:.3f}] "
          f"W{res['wins']}/D{res['draws']}/L{res['losses']} over {res['games']} games")
    print(f"  as P0: W/D/L={res['a_as_p0_wdl']}  as P1: W/D/L={res['a_as_p1_wdl']}")
    print(f"  elapsed {dt:.1f}s; archived {len(rows)} rows -> {games_path}")
    if "sa.planner" in sys.modules:  # search internals, summed over both sides
        st = sys.modules["sa.planner"].STATS
        if st["decides"]:
            print(f"  planner: {st['decides']} decides, "
                  f"{st['worlds'] / st['decides']:.1f} worlds/decide, "
                  f"{100 * st['spent_s'] / max(st['budget_s'], 1e-9):.0f}% of "
                  f"budget used, {st['spent_s']:.0f}s total")
        if st["rollouts"]:
            print(f"  rollouts: {st['rollouts']} "
                  f"({100 * st['roll_terminal'] / st['rollouts']:.0f}% reached "
                  f"terminal), {st['roll_steps'] / st['rollouts']:.0f} steps each")
    return 0


# --- elo ----------------------------------------------------------------------

def _load_rows(path: Path = GAMES_PATH) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def fit_elo(rows: list[dict], anchor: str = "rule:iono",
            anchor_rating: float = 1000.0, iters: int = 500,
            lr: float = 8.0) -> dict[str, float]:
    """Batch-gradient Bradley-Terry fit (draw = half a win), all games weighted
    equally regardless of when they were played. Anchored so ratings are
    comparable across refits as the archive grows."""
    games = [(r["agent0"], r["agent1"], r["winner"]) for r in rows]
    players = sorted({p for a0, a1, _ in games for p in (a0, a1)})
    rating = {p: anchor_rating for p in players}
    for _ in range(iters):
        grad = {p: 0.0 for p in players}
        for a0, a1, winner in games:
            s0 = 1.0 if winner == 0 else 0.5 if winner == 2 else 0.0
            e0 = 1.0 / (1.0 + 10.0 ** ((rating[a1] - rating[a0]) / 400.0))
            grad[a0] += s0 - e0
            grad[a1] += (1.0 - s0) - (1.0 - e0)
        for p in players:
            rating[p] += lr * grad[p]
        if anchor in rating:  # re-anchor every pass
            shift = anchor_rating - rating[anchor]
            for p in players:
                rating[p] += shift
    return rating


def cmd_elo(args: argparse.Namespace) -> int:
    path = Path(args.archive) if args.archive else GAMES_PATH
    rows = _load_rows(path)
    if not rows:
        print(f"no archive at {path}; run `arena.py play` first.")
        return 1
    counts: Counter = Counter()
    lat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for seat in (0, 1):
            name = r[f"agent{seat}"]
            counts[name] += 1
            ls = r.get(f"lat{seat}") or {}
            if ls.get("n"):
                lat[name].append(ls["p99"])
    ratings = fit_elo(rows)
    print(f"Elo over {len(rows)} archived games (rule:iono anchored at 1000):\n")
    for name, elo in sorted(ratings.items(), key=lambda kv: -kv[1]):
        line = f"  {elo:7.1f}  {name}  ({counts[name]} games)"
        if lat[name]:
            line += f"  p99={sum(lat[name]) / len(lat[name]):.0f}ms"
        print(line)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="run paired matches and archive them")
    p.add_argument("a", help="agent spec for A")
    p.add_argument("b", help="agent spec for B")
    p.add_argument("--matches", type=int, default=10,
                   help="paired matches (2 games each; default 10)")
    p.add_argument("--deck-a", default="sample")
    p.add_argument("--deck-b", default="sample")
    p.add_argument("--archive", default=None,
                   help="archive path (default: out/arena/games.jsonl)")
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("elo", help="fit Elo over the archive")
    p.add_argument("--archive", default=None,
                   help="archive path (default: out/arena/games.jsonl)")
    p.set_defaults(fn=cmd_elo)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
