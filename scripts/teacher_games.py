"""Teacher games for the population-RL pipeline, played by our shipped agent.

The 24th-place pipeline (`~/Kaggle/ptcg-population-rl`) distils a Transformer
from games played by a GBDT teacher. Here the teacher is `policy_v5_s2` on
Grimmsnarl instead: the same agent as Playground submission 58661466
(`bc`, chip/spread/source rules off, greedy). The fork's GBDT teacher is
greedy too, so neither side samples.

Each game is written in the fork's `ptcg-replay-v1` record layout (header,
decision records, terminal). The changes from that layout are:

  - Decision records are written for the teacher's seat(s) only. The
    opponent's decisions are not training data, and dropping them about halves
    the file. A mirror game records both seats.
  - Each teacher decision also carries `teacher_scores`: the net's logit for
    every option in `observation.select.option`, the per-option targets that
    the fork's extractor otherwise reads from `gbdt_queries`. It also carries
    `net_agrees`, which says whether `action` is the net's top-k, ties allowed
    (it is not when a forced shortcut or the agent's catch-all answered).
  - There is no `view` field. The fork's Python features read `observation`
    only, and `observation` here is the official JSON from the same Playground
    engine, minus `search_begin_input` (see DROP).
  - Decision records carry `index` (position among this game's RECORDED
    decisions) rather than the fork's `step`, since the opponent's steps are
    not written.

Each game is appended as its own gzip member, so a shard killed mid-game keeps
every finished game readable. The card-info table the fork's extractor needs
(`card_metadata.json`) is the Playground's 1,431 cards and matches this engine
field for field; it is not written per shard.

Opponents come from `--mix` (a named cycle, see MIXES) or a single `--opp`.

    # one Kaggle shard: 7,500 games of the wide mix, ids from 1,000,000
    python -X utf8 scripts/teacher_games.py --mix wide --games 7500 \\
        --shard 1 --archive out/teacher/s1.jsonl.gz
    # one opponent
    python -X utf8 scripts/teacher_games.py --opp "rule:v10,noS" \\
        --deck-b lucario_v10 --games 4 --archive out/teacher/smoke.jsonl.gz
"""
from __future__ import annotations

import os

# One BLAS thread per process: shards run one per core, and numpy's default
# thread pool would oversubscribe them. Must precede the numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import collections  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import harness, sdk  # noqa: E402

sdk.load()

from arena import build_agent, resolve_deck  # noqa: E402

TEACHER = "v5s2_grimmsnarl"
RULES_OFF = "noChip,noSpread,noSrc"   # == build_submission.py --no-rules
SELF = "self"
# Not part of the fork's observation: the harness's clock, and the engine's
# encoded search snapshot (21% of an observation's bytes, read by nothing in
# the fork).
DROP = {"remainingOverageTime", "search_begin_input"}
_CALLS = itertools.count()            # global call order, to merge mirror seats

# A mix is a cycle of (opponent spec, opponent deck, games per cycle).
# `wide` (chosen 2026-11-24): 25% mirror, 75% split evenly over the 8 rule
# bots that run cleanly on the Playground engine. Cycle = 8 + 8*3 = 32 games.
MIXES = {
    "wide": [
        (SELF, "grimmsnarl", 8),
        ("rule:v10,noS", "lucario_v10", 3),
        ("rule:crustle", "crustle_v1", 3),
        ("rule:alakazam5", "alakazam5", 3),
        ("rule:archaludon", "archaludon_ex", 3),
        ("rule:dragapult", "dragapult_ex", 3),
        ("rule:iono", "iono", 3),
        ("rule:abomasnow", "mega_abomasnow_ex", 3),
        ("rule:lucario", "mega_lucario_ex", 3),
    ],
}


class Teacher:
    """The shipped agent, plus a record of its scores for every decision."""

    def __init__(self, spec: str, deck: list[int], deck_name: str):
        self.name, self.agent = build_agent(spec, list(deck), deck_name)
        self.net = self.agent.net
        if self.net is None:
            raise SystemExit(f"{spec}: net did not load")
        self.records: list[dict] = []

    def __call__(self, obs: dict) -> list[int]:
        action = [int(i) for i in self.agent(obs)]
        sel = obs.get("select")
        if sel is None:
            return action
        n = len(sel.get("option") or [])
        scores = None
        agrees = None
        if n >= 2:
            sc = np.asarray(self.net.scores(obs), dtype=np.float64)
            scores = [float(x) for x in sc]
            # Tie-aware: identical copies of a card score identically, and the
            # agent may take any of them. The action agrees with the net when
            # every picked option scores at least as high as every unpicked one.
            picked = set(action)
            rest = [sc[i] for i in range(n) if i not in picked]
            agrees = bool(not action or not rest
                          or min(sc[i] for i in action) >= max(rest))
        rec = {k: v for k, v in obs.items() if k not in DROP}
        # Serialised now, because the harness writes into `obs` after we return.
        self.records.append(dict(
            order=next(_CALLS),
            seat=int(obs["current"]["yourIndex"]),
            observation=json.dumps(rec, separators=(",", ":")),
            action=action, teacher_scores=scores, net_agrees=agrees))
        return action


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def schedule(args) -> list[tuple[str, str]]:
    """One cycle of (opponent spec, opponent deck), repeated over the games."""
    if args.mix:
        return [(spec, deck) for spec, deck, k in MIXES[args.mix] for _ in range(k)]
    return [(args.opp, args.deck_b or args.deck)]


def play_one(gid: int, tseat: int, spec: str, deck_a_name: str, deck_a: list[int],
             opp_spec: str, opp_deck: str, model: str) -> tuple[list[str], int, dict]:
    """Play one game; return its record lines, result and decision tally."""
    deck_b_name, deck_b = resolve_deck(opp_deck)
    teacher = Teacher(spec, deck_a, deck_a_name)
    mirror = opp_spec == SELF
    if mirror:
        opp = Teacher(spec, deck_b, deck_b_name)
        opp_name = teacher.name
    else:
        opp_name, opp = build_agent(opp_spec, list(deck_b), deck_b_name)
    # tseat is the teacher's seat; first player is worth ~1 pp (EVIDENCE §8aj).
    seats = (teacher, opp) if tseat == 0 else (opp, teacher)
    decks = (list(deck_a), list(deck_b)) if tseat == 0 else (list(deck_b), list(deck_a))
    names = [teacher.name, opp_name] if tseat == 0 else [opp_name, teacher.name]
    deck_names = ([deck_a_name, deck_b_name] if tseat == 0
                  else [deck_b_name, deck_a_name])
    r = harness.play_game(seats[0], seats[1], decks[0], decks[1])
    recorded = [0, 1] if mirror else [tseat]
    recs = (sorted(seats[0].records + seats[1].records, key=lambda x: x["order"])
            if mirror else teacher.records)
    header = dict(type="header", format="ptcg-replay-v1", game=gid,
                  agents=names, teacher_seats=recorded,
                  models=[model if s in recorded else None for s in (0, 1)],
                  decks=[decks[0], decks[1]], deck_names=deck_names)
    lines = [json.dumps(header, separators=(",", ":"))]
    tally = collections.Counter()
    for i, rec in enumerate(recs):
        lines.append('{"type":"decision","game":%d,"index":%d,"seat":%d,'
                     '"agent":%s,"observation":%s,"action":%s,'
                     '"teacher_scores":%s,"net_agrees":%s}' % (
                         gid, i, rec["seat"], json.dumps(names[rec["seat"]]),
                         rec["observation"], json.dumps(rec["action"]),
                         json.dumps(rec["teacher_scores"]),
                         json.dumps(rec["net_agrees"])))
        if rec["net_agrees"] is not None:
            tally["scored"] += 1
            tally["disagree"] += not rec["net_agrees"]
    lines.append(json.dumps(dict(
        type="terminal", game=gid, result=int(r.winner), turns=int(r.turns),
        steps=int(r.selects), agent_errors=0, engine_errors=0),
        separators=(",", ":")))
    return lines, int(r.winner), tally


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="out/policy_v5_s2.npz")
    ap.add_argument("--deck", default="grimmsnarl")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--mix", choices=sorted(MIXES), help="named opponent cycle")
    who.add_argument("--opp", help=f"one arena spec for the opponent, or `{SELF}`")
    ap.add_argument("--deck-b", default=None,
                    help="with --opp: the opponent's deck module (default: --deck)")
    ap.add_argument("--games", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0,
                    help="game ids start at shard * 1,000,000, so parallel "
                         "shards never share an id")
    ap.add_argument("--archive", required=True, help="output .jsonl.gz")
    ap.add_argument("--report", type=int, default=250)
    args = ap.parse_args()

    model = md5(ROOT / args.net)
    spec = f"bc:{TEACHER},net={args.net},{RULES_OFF}"
    deck_a_name, deck_a = resolve_deck(args.deck)
    cycle = schedule(args)
    out = ROOT / args.archive
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} exists; games are appended, so refusing to mix runs")
    base = args.shard * 1_000_000
    print(f"teacher={spec} md5={model} deck={deck_a_name}\n"
          f"cycle={len(cycle)} games over {len(set(cycle))} opponents, "
          f"games={args.games} ids from {base} -> {out}", flush=True)

    per_opp: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    total = collections.Counter()
    errors = 0
    t0 = time.time()
    for g in range(args.games):
        opp_spec, opp_deck = cycle[g % len(cycle)]
        # Alternates game to game AND flips every cycle, so each cycle position
        # (and therefore each opponent) gets both seats equally over two cycles.
        tseat = (g + g // len(cycle)) % 2
        try:
            lines, winner, tally = play_one(base + g, tseat, spec, deck_a_name,
                                            deck_a, opp_spec, opp_deck, model)
        except Exception as e:  # noqa: BLE001 -- an engine failure loses one game, not the shard
            errors += 1
            print(f"game {base + g} vs {opp_spec}: {type(e).__name__}: {e}", flush=True)
            continue
        with gzip.open(out, "ab", compresslevel=6) as f:
            f.write(("\n".join(lines) + "\n").encode())
        total.update(tally)
        per_opp[opp_spec][2 if winner == 2 else (0 if winner == tseat else 1)] += 1
        if (g + 1) % args.report == 0 or g + 1 == args.games:
            el = time.time() - t0
            print(f"{g + 1:>7}/{args.games} games  {(g + 1) / el * 3600:,.0f} games/h  "
                  f"{out.stat().st_size / (g + 1) / 1e3:.0f} KB/game  "
                  f"disagree={total['disagree']}/{total['scored']}  "
                  f"errors={errors}", flush=True)
    el = time.time() - t0
    print(f"\nteacher W/L/D by opponent (mirror counts both seats as one game):")
    for opp_spec, (w, l, d) in per_opp.items():
        n = w + l + d
        print(f"  {opp_spec:<22} {n:>6} games  score {(w + 0.5 * d) / max(n, 1):.3f}  "
              f"W/L/D={w}/{l}/{d}")
    print(f"done: {args.games - errors} games, {errors} engine errors, "
          f"{total['scored']} scored decisions, {total['disagree']} where the "
          f"action is not the net's top-k, {el / 60:.1f} min", flush=True)
    return 0 if errors == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
