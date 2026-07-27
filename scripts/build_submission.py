"""Assemble the sa search-agent submission and tar.gz it for Kaggle.

    python scripts/build_submission.py [--deck grimmsnarl] [--agent search|bc]

Bundle layout (Kaggle: .tar.gz, main.py at TOP level):
    main.py            entrypoint defining agent(obs) -> list[int]
    deck.csv           60 card ids, one per line
    cg/                engine (from the local SDK)
    sa/                agent package (+ value_net.npz / policy_net.npz /
                       deck_library.json if present)

Then smoke-runs the *extracted* bundle from a temp dir: full self-play game,
crash = build failure. Prints latency + time-pool stats.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg import config  # noqa: E402

SIZE_CAP_MIB = 197.7

MAIN_PY = '''\
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

AGENT_KIND = {agent_kind!r}


def _read_deck():
    path = os.path.join(_HERE, "deck.csv")
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as fh:
        return [int(line) for line in fh.read().split()[:60]]


_deck = _read_deck()

if AGENT_KIND == "bc":
    from sa.bcagent import PolicyAgent as _A
else:
    from sa.agent import SearchAgent as _A

_agent = _A(_deck)


def agent(obs):
    return _agent(obs)
'''

SMOKE = r'''
import sys, time
sys.path.insert(0, ".")
import main

deck = list(main._deck)
assert len(deck) == 60, len(deck)

import cg.game as game

class B:  # opposing agent: trivial legal
    def __call__(self, obs):
        if obs.get("select") is None:
            return deck
        return list(range(obs["select"]["minCount"]))

opp = B()
obs, _ = game.battle_start(deck, deck)
overage = [600.0, 600.0]
selects = 0
lat_max = 0.0
try:
    while True:
        st = obs.get("current")
        if st is not None and st["result"] != -1:
            print(f"RESULT={st['result']} turns={st['turn']} selects={selects} "
                  f"agent_pool_left={overage[0]:.1f}s lat_max={lat_max:.2f}s")
            break
        who = st["yourIndex"]
        obs["remainingOverageTime"] = overage[who]
        t0 = time.perf_counter()
        choice = main.agent(obs) if who == 0 else opp(obs)
        dt = time.perf_counter() - t0
        overage[who] -= dt
        if who == 0:
            lat_max = max(lat_max, dt)
        obs = game.battle_select([int(c) for c in choice])
        selects += 1
        assert overage[0] > 0, "agent exhausted its time pool"
        if selects > 6000:
            raise SystemExit("game did not terminate")
finally:
    game.battle_finish()
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--agent", default="search", choices=["search", "bc"])
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()

    sdk_dir = config.find_sdk_dir()
    if sdk_dir is None:
        raise SystemExit("cg engine not found under data/")

    build = config.DIST_DIR / f"_build_{args.agent}-{args.deck}"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)

    (build / "main.py").write_text(MAIN_PY.format(agent_kind=args.agent),
                                   encoding="utf-8")

    deck = importlib.import_module(f"decks.{args.deck}").DECK
    (build / "deck.csv").write_text(deck.to_csv(), encoding="utf-8")
    print(f"deck.csv: decks/{args.deck}.py ({deck.size} cards)")

    shutil.copytree(sdk_dir / "cg", build / "cg",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "agents" / "sa", build / "sa",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for extra in ("value_net.npz", "policy_net.npz", "deck_library.json"):
        state = "present" if (build / "sa" / extra).exists() else "MISSING"
        print(f"  sa/{extra}: {state}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = config.DIST_DIR / f"submission_{args.agent}-{args.deck}_{stamp}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for item in sorted(build.iterdir()):
            tar.add(item, arcname=item.name)
    shutil.rmtree(build)

    size_mib = out.stat().st_size / (1024 * 1024)
    print(f"built {out} ({size_mib:.1f} MiB)")
    if size_mib > SIZE_CAP_MIB:
        raise SystemExit(f"exceeds cap {SIZE_CAP_MIB} MiB")

    if not args.no_smoke:
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(out) as tar:
                tar.extractall(tmp)
            proc = subprocess.run([sys.executable, "-X", "utf8", "-c", SMOKE],
                                  cwd=tmp, capture_output=True, text=True,
                                  timeout=1800)
            ok = proc.returncode == 0 and "RESULT=" in proc.stdout
            print(f"  smoke: {'OK' if ok else 'FAILED'}")
            for line in proc.stdout.strip().splitlines()[-3:]:
                print(f"    {line}")
            if not ok:
                print(proc.stderr[-3000:])
                raise SystemExit("smoke test failed")

    latest = config.DIST_DIR / "submission.tar.gz"
    shutil.copy2(out, latest)
    print(f"latest -> {latest}\nupload: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
