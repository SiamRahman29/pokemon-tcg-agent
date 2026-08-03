"""E2: measure observable route coverage and purity on live v5 trajectories.

Compares the hard router (visible opponent active/bench/discard) against the
post-game census signature from `p9_field_census.py`. Census labels are an
audit target only; they are never used for training or inference.

    python -X utf8 scripts/p47_e2_route_audit.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from p9_field_census import _signature, analyse  # noqa: E402
from sa.routing import (NAME_TO_ROUTE, ROUTE_ALAKAZAM, ROUTE_GENERAL,  # noqa: E402
                        ROUTE_MIRROR, ROUTE_NAMES, route_from_obs)


CENSUS_MIRROR = "Marnie's Grimmsnarl ex"
CENSUS_ALAKAZAM = "Alakazam"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays",
                    default="replays/submission_v5_003/submission_977")
    ap.add_argument("--player", default="Scio")
    ap.add_argument("--out", default="out/e2/route_audit.json")
    args = ap.parse_args()

    root = ROOT / args.replays
    paths = sorted(p for p in root.glob("*.json") if p.name != "manifest.json")
    if not paths:
        raise SystemExit(f"no replays under {root}")

    decision_routes = Counter()
    game_routes = Counter()
    census_games = Counter()
    confusion = Counter()  # (census_arch, activated_routes_joined)
    true_mirror_hit = true_mirror_n = 0
    true_alakazam_hit = true_alakazam_n = 0
    transitions = Counter()
    errs: Counter = Counter()

    for path in paths:
        g = analyse(path, errs, {args.player})
        if g is None:
            continue
        arch = _signature(g.poke, g.max_copies)
        census_games[arch] += 1

        replay = json.loads(path.read_text(encoding="utf-8"))
        names = (replay.get("info") or {}).get("TeamNames") or []
        vis = replay["steps"][0][0].get("visualize") or []
        seen: set[str] = set()
        last = None
        for v in vis:
            obs = v.get("obs") or {}
            state = obs.get("current") or {}
            sel = obs.get("select") or {}
            me = state.get("yourIndex")
            if (me not in (0, 1) or me >= len(names)
                    or names[me] != args.player
                    or not (sel.get("option") or [])):
                continue
            route = route_from_obs(obs)
            name = ROUTE_NAMES[route]
            decision_routes[name] += 1
            seen.add(name)
            if last is not None and last != name:
                transitions[(last, name)] += 1
            last = name

        activated = "+".join(sorted(x for x in seen if x != "general")) or "general"
        confusion[(arch, activated)] += 1
        for name in seen:
            game_routes[name] += 1

        if arch == CENSUS_MIRROR:
            true_mirror_n += 1
            if "mirror" in seen:
                true_mirror_hit += 1
        if arch == CENSUS_ALAKAZAM:
            true_alakazam_n += 1
            if "alakazam" in seen:
                true_alakazam_hit += 1

    mirror_fidelity = (true_mirror_hit / true_mirror_n
                       if true_mirror_n else 1.0)
    alakazam_fidelity = (true_alakazam_hit / true_alakazam_n
                         if true_alakazam_n else 1.0)
    # Gate uses the two target archetypes jointly.
    target_n = true_mirror_n + true_alakazam_n
    target_hit = true_mirror_hit + true_alakazam_hit
    target_fidelity = target_hit / target_n if target_n else 1.0

    out = {
        "replays": args.replays,
        "player": args.player,
        "replay_files": len(paths),
        "decision_routes": dict(decision_routes),
        "decision_total": int(sum(decision_routes.values())),
        "game_activation": dict(game_routes),
        "census_games": dict(census_games),
        "confusion": [
            {"census": arch, "activated": activated, "games": n}
            for (arch, activated), n in sorted(
                confusion.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "transitions": [
            {"from": a, "to": b, "count": n}
            for (a, b), n in sorted(transitions.items(),
                                    key=lambda kv: -kv[1])
        ],
        "fidelity": {
            "mirror_games": true_mirror_n,
            "mirror_activated": true_mirror_hit,
            "mirror": mirror_fidelity,
            "alakazam_games": true_alakazam_n,
            "alakazam_activated": true_alakazam_hit,
            "alakazam": alakazam_fidelity,
            "target": target_fidelity,
        },
        "route_ids": {name: rid for name, rid in NAME_TO_ROUTE.items()},
        "errors": dict(errs),
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"E2_ROUTE_AUDIT {len(paths)} games, "
          f"{out['decision_total']} decisions")
    print(f"  decisions: {dict(decision_routes)}")
    print(f"  game activation: {dict(game_routes)}")
    print(f"  fidelity mirror={mirror_fidelity:.3%} "
          f"({true_mirror_hit}/{true_mirror_n}) "
          f"alakazam={alakazam_fidelity:.3%} "
          f"({true_alakazam_hit}/{true_alakazam_n}) "
          f"target={target_fidelity:.3%}")
    print(f"  wrote {out_path}")
    if target_fidelity < 0.95:
        raise SystemExit(
            f"target fidelity {target_fidelity:.3%} < 95%; "
            "route signatures are not safe to train")
    print("E2_ROUTE_AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
