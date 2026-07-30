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

# Kaggle loads this file with exec(code_object, env) -- `__file__` is NOT
# defined there (kaggle_environments/agent.py:get_last_callable). Never rely
# on it: fall back to the documented agent dir, then cwd.
_CANDS = []
try:
    _CANDS.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
_CANDS.append("/kaggle_simulations/agent")
_CANDS.append(os.getcwd())
_HERE = next((p for p in _CANDS
              if p and os.path.exists(os.path.join(p, "deck.csv"))), _CANDS[-1])
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

AGENT_KIND = {agent_kind!r}
# Explicit rule flags, pinned at BUILD time. The defaults in sa/bcagent.py are
# tuned for the lw2 net; an optfeat-v3 net wants them OFF (the three rules
# measure 0.427 against it -- report/EVIDENCE.md 8f). Pinning the (net, flags)
# PAIR here keeps both configurations shippable without a global default flip.
AGENT_KWARGS = {agent_kwargs!r}


def _read_deck():
    path = os.path.join(_HERE, "deck.csv")
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path) as fh:
        return [int(line) for line in fh.read().split()[:60]]


_deck = _read_deck()

if AGENT_KIND == "bc":
    from sa.bcagent import PolicyAgent as _A
    _agent = _A(_deck, **AGENT_KWARGS)
else:
    from sa.agent import SearchAgent as _A
    _agent = _A(_deck)


def agent(obs):
    return _agent(obs)
'''

SMOKE = r'''
import sys, time
sys.path.insert(0, ".")

# Load main.py THE WAY KAGGLE DOES: exec the source with no __file__ in globals
# (kaggle_environments/agent.py does exec(code_object, env)). `import main`
# would define __file__ and hide a whole class of crash -- it did once.
with open("main.py", "rb") as _fh:
    _src = _fh.read()
_env = {}
exec(compile(_src, "main.py", "exec"), _env)
assert "__file__" not in _env, "smoke must not leak __file__ into agent globals"


class _M:
    pass


main = _M()
main._deck = _env["_deck"]
main.agent = _env["agent"]

deck = list(main._deck)
assert len(deck) == 60, len(deck)

# The agent "runs" perfectly well with a rejected net -- it just plays
# list(range(minCount)), i.e. random-legal, and scores ~600. Kaggle sets no env
# vars, so the bundled npz is the only thing that can save it. Assert the net is
# actually live, and print the pinned rule flags so the build log records the
# exact configuration that was shipped.
_ag = _env.get("_agent")
if _ag is not None and type(_ag).__name__ == "PolicyAgent":
    from sa import policynet as _pn
    _live = _ag.net or _pn.get()
    assert _live is not None, "POLICY NET NOT LOADED -- agent would play random-legal"
    print(f"NET_OK opt_in={_live.opt_in} state_in={_live.state_in}")
    print("FLAGS chip=%s spread=%s src=%s wall=%s"
          % (_ag.chip_targeting, _ag.energy_spread, _ag.counter_source,
             _ag.chip_wall_defer))

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
    # Which nets to ship. The SA_NO_* kill-switches are env vars and Kaggle
    # sets none, so anything bundled is LIVE there. Omitting an npz makes the
    # matching net.get() return None and the agent fall back to the
    # handcrafted eval -- the only way to pin the config for the grader.
    ap.add_argument("--nets", default="both",
                    choices=["both", "policy", "value", "none"],
                    help="nets to include (default both)")
    # Ship a candidate net instead of agents/sa/policy_net.npz. The bundle is
    # what the grader runs, so the net and the rule flags must be pinned
    # TOGETHER -- a v3 net with the lw2 defaults is the 0.427 configuration.
    ap.add_argument("--policy-net", default=None,
                    help="path to an npz to ship as sa/policy_net.npz")
    ap.add_argument("--no-rules", action="store_true",
                    help="disable chip_targeting/energy_spread/counter_source. "
                         "Required with an optfeat-v3 net (EVIDENCE 8f).")
    args = ap.parse_args()
    if args.agent == "bc" and args.nets in ("value", "none"):
        raise SystemExit("--agent bc requires the policy net (--nets both|policy)")

    agent_kwargs: dict[str, bool] = {}
    if args.no_rules:
        agent_kwargs = {"chip_targeting": False, "energy_spread": False,
                        "counter_source": False}

    sdk_dir = config.find_sdk_dir()
    if sdk_dir is None:
        raise SystemExit("cg engine not found under data/")

    build = config.DIST_DIR / f"_build_{args.agent}-{args.deck}"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)

    (build / "main.py").write_text(
        MAIN_PY.format(agent_kind=args.agent, agent_kwargs=agent_kwargs),
        encoding="utf-8")
    print(f"agent kwargs: {agent_kwargs or '(bcagent.py defaults)'}")

    deck = importlib.import_module(f"decks.{args.deck}").DECK
    (build / "deck.csv").write_text(deck.to_csv(), encoding="utf-8")
    print(f"deck.csv: decks/{args.deck}.py ({deck.size} cards)")

    shutil.copytree(sdk_dir / "cg", build / "cg",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "agents" / "sa", build / "sa",
                    ignore=shutil.ignore_patterns("__pycache__"))
    keep_policy = args.nets in ("both", "policy")
    keep_value = args.nets in ("both", "value")
    for npz, keep in (("policy_net.npz", keep_policy),
                      ("value_net.npz", keep_value)):
        if not keep:
            (build / "sa" / npz).unlink(missing_ok=True)
    if args.policy_net:
        src = Path(args.policy_net)
        if not src.is_absolute():
            src = ROOT / src
        if not src.exists():
            raise SystemExit(f"--policy-net not found: {src}")
        if not keep_policy:
            raise SystemExit("--policy-net needs --nets both|policy")
        shutil.copy2(src, build / "sa" / "policy_net.npz")
        print(f"  sa/policy_net.npz <- {src}")
        # The dim guard silently returns None for a net whose feature layout
        # this code cannot feed, and the agent then falls back to
        # list(range(minCount)) -- i.e. a broken agent that still "runs".
        # Verifying HERE turns that into a build failure. (See policynet.load.)
        # sa.cards imports cg.sim, so the SDK has to be on the path first. Done
        # here rather than at module import to keep the builder light.
        from ptcg.env import sdk as _sdk
        _sdk.load()
        from sa import policynet as _pn
        if _pn.load(build / "sa" / "policy_net.npz") is None:
            raise SystemExit(f"{src} FAILS the dim guard -- it would silently "
                             "fall back to random-legal on Kaggle")
        print("  dim guard: net loads OK")
    for extra in ("value_net.npz", "policy_net.npz", "deck_library.json"):
        present = (build / "sa" / extra).exists()
        note = "present" if present else "excluded -> handcrafted fallback"
        print(f"  sa/{extra}: {note}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = (config.DIST_DIR /
           f"submission_{args.agent}-{args.deck}-nets{args.nets}_{stamp}.tar.gz")
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
            if args.agent == "bc" and "NET_OK" not in proc.stdout:
                ok = False
                print("  smoke: net was NOT live in the extracted bundle")
            print(f"  smoke: {'OK' if ok else 'FAILED'}")
            for line in proc.stdout.strip().splitlines()[-4:]:
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
