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
import os
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

# 🔴 Kaggle's EPISODE RUNNER is Python 3.11 -- `kaggle_environments` lives under
# `/usr/local/lib/python3.11/dist-packages/`. Kaggle *notebooks* are 3.12, and
# so is this dev box, so 3.12-only syntax parses everywhere we normally look and
# then raises SyntaxError at IMPORT on the grader. Submission 55489084 died that
# way: one PEP 701 multi-line f-string in `sa/policynet.py` (a logging nicety)
# stopped `from sa.bcagent import PolicyAgent`, both seats crashed in 0.04s, and
# the LB said only "Validation Episode failed." The smoke test CANNOT catch this
# -- it runs on the local interpreter, where the file is valid.
KAGGLE_PY = (3, 11)


def _find_kaggle_python() -> str | None:
    """An interpreter matching Kaggle's episode runner, or None.

    `SA_KAGGLE_PYTHON` wins; otherwise ask uv for a managed 3.11. Returning
    None is not fatal -- the caller falls back to a narrower static check.
    """
    pinned = os.environ.get("SA_KAGGLE_PYTHON")
    if pinned and Path(pinned).exists():
        return pinned
    ver = "%d.%d" % KAGGLE_PY
    try:
        proc = subprocess.run(["uv", "python", "find", ver],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    cand = proc.stdout.strip()
    return cand if proc.returncode == 0 and cand and Path(cand).exists() else None


def _scan_pep701(root: Path) -> list[str]:
    """Static fallback: f-strings that only parse on 3.12+ (PEP 701).

    Catches the multi-line replacement field that actually shipped. It is a
    subset of what a real 3.11 parse catches, so it is the fallback, never the
    primary check.
    """
    import io
    import tokenize
    fs_start = getattr(tokenize, "FSTRING_START", None)
    fs_end = getattr(tokenize, "FSTRING_END", None)
    if fs_start is None:  # pre-3.12 host: it would have failed outright
        return []
    hits = []
    for f in sorted(root.rglob("*.py")):
        try:
            toks = list(tokenize.generate_tokens(
                io.StringIO(f.read_text(encoding="utf-8")).readline))
        except Exception:  # noqa: BLE001  -- a parse failure is caught elsewhere
            continue
        depth, start = 0, None
        for t in toks:
            if t.type == fs_start:
                if depth == 0:
                    start = t
                depth += 1
            elif t.type == fs_end:
                depth -= 1
                if depth == 0 and start is not None:
                    quoted = start.string.lstrip("fFrRbB")
                    if t.end[0] != start.start[0] and quoted not in ('"""', "'''"):
                        hits.append(f"{f.relative_to(root)}:{start.start[0]}"
                                    " multi-line f-string (PEP 701, 3.12+)")
                    start = None
    return hits


def _check_kaggle_syntax(build: Path) -> None:
    """Compile every bundled .py the way the grader's interpreter will."""
    py = _find_kaggle_python()
    if py:
        proc = subprocess.run([py, "-m", "compileall", "-q", str(build)],
                              capture_output=True, text=True, timeout=600)
        for cache in build.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        if proc.returncode != 0:
            print(proc.stdout[-3000:] or proc.stderr[-3000:])
            raise SystemExit(
                "syntax check FAILED under Python %d.%d -- this bundle would "
                "crash at import on Kaggle, exactly like 55489084" % KAGGLE_PY)
        print("  kaggle syntax: OK under %d.%d (%s)" % (*KAGGLE_PY, py))
        return
    hits = _scan_pep701(build)
    if hits:
        for h in hits:
            print(f"    {h}")
        raise SystemExit(
            "bundle uses 3.12-only f-string syntax and Kaggle runs %d.%d"
            % KAGGLE_PY)
    print("  kaggle syntax: no %d.%d interpreter found (`uv python install "
          "%d.%d`); ran the PEP-701 scan only -- WEAKER" % (*KAGGLE_PY,
                                                            *KAGGLE_PY))

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
# Extra ensemble members, relative to the bundle root. Empty = single net, i.e.
# exactly the behaviour every submission before this one had. When non-empty the
# agent votes across sa/policy_net.npz + these (EVIDENCE: E9).
ENSEMBLE_EXTRA = {ensemble_extra!r}
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
    _agent = None
    if ENSEMBLE_EXTRA:
        # 🔴 FAIL-SOFT IS NON-NEGOTIABLE ON THE SHIPPED PATH. An explicit
        # `net=` is STRICT by design (day 22: a net that failed its guard used
        # to silently play a different one), and strict means it RAISES. In the
        # arena that is correct -- a bad cell must not print a score. Here it
        # would forfeit a live episode. So: try the vote, and on ANY failure
        # fall back to the single bundled net, which is member 0 -- i.e. the
        # agent that shipped before this change. Degrade, log, keep playing.
        try:
            _paths = [os.path.join(_HERE, "sa", "policy_net.npz")]
            _paths += [os.path.join(_HERE, p) for p in ENSEMBLE_EXTRA]
            _agent = _A(_deck, "+".join(_paths), **AGENT_KWARGS)
            print("[health] ENSEMBLE OK members=%d" % len(_paths), flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
            print("[health] ENSEMBLE FAILED TO LOAD -- falling back to the "
                  "single bundled net", flush=True)
            _agent = None
    if _agent is None:
        _agent = _A(_deck, **AGENT_KWARGS)
else:
    from sa.agent import SearchAgent as _A
    _agent = _A(_deck)


# --- health logging (day 15) -------------------------------------------------
# Kaggle keeps our own submission's stdout, and until now we printed NOTHING on
# the happy path, so the logs were empty. The thing worth the bytes is whether
# the NET IS ACTUALLY LIVE: `bcagent.__call__`'s catch-all returns
# `range(minCount)` -- index order -- and a submission running that on every
# decision still returns legal moves, still finishes games and still gets a
# rating, so it looks normal from outside. EVIDENCE 8g had to infer the net was
# live from a 40.7% index-0 rate; this makes it a direct read.
#
# Discipline: ONE line per game (the deck handshake fires once per game), plus a
# heartbeat every 1000 selects so a single long episode still reports, plus a
# one-shot line the first time the fallback ever fires. Never per-decision.
#
# EVERYTHING here is wrapped: a logging bug must never be able to break the
# agent, which would trade a real rating for a diagnostic.
#
# ⚠ MAIN_PY is a str.format() template -- every literal brace below is doubled.
_LOG = {{"next": 0, "failed": False}}


def _log_health(force=False):
    try:
        from sa import bcagent as _bc
        s = _bc.STATS
        if s["fallbacks"] and not _LOG["failed"]:
            _LOG["failed"] = True
            print("[health] FIRST FALLBACK -- net path raised; now playing "
                  "index order", flush=True)
            print(str(s["first_error"])[:1500], flush=True)
            return
        if force or s["calls"] >= _LOG["next"]:
            _LOG["next"] = s["calls"] + 1000
            print(_bc.health_line(), flush=True)
    except Exception:
        pass


def agent(obs):
    out = _agent(obs)
    _log_health(force=obs.get("select") is None)
    return out
'''

SMOKE = r'''
import os, sys, time
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
    # An ensemble that quietly loaded ONE member is a different agent shipping
    # under the measured one's name, and it would score like the single net
    # while the build log said nothing. Assert the member count explicitly.
    _want = int(os.environ.get("SMOKE_MEMBERS", "1"))
    _got = len(getattr(_live, "nets", []) or [1])
    print(f"MEMBERS={_got} want={_want}")
    assert _got == _want, f"ensemble has {_got} members, expected {_want}"
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
    # Extra ensemble members (EVIDENCE E9). Each is copied into the bundle as
    # sa/policy_net_<i>.npz and voted with sa/policy_net.npz. Member ORDER is
    # part of the agent's identity -- member 0 supplies the count rule.
    ap.add_argument("--ensemble-net", action="append", default=[],
                    help="extra npz to vote with the shipped policy net; "
                         "repeatable")
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

    ensemble_extra = [f"sa/policy_net_{i + 1}.npz"
                      for i in range(len(args.ensemble_net))]
    (build / "main.py").write_text(
        MAIN_PY.format(agent_kind=args.agent, agent_kwargs=agent_kwargs,
                       ensemble_extra=ensemble_extra),
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
    if args.ensemble_net:
        if not keep_policy:
            raise SystemExit("--ensemble-net needs --nets both|policy")
        from ptcg.env import sdk as _sdk2
        _sdk2.load()
        from sa import policynet as _pn2
        for i, extra_net in enumerate(args.ensemble_net):
            src = Path(extra_net)
            if not src.is_absolute():
                src = ROOT / src
            if not src.exists():
                raise SystemExit(f"--ensemble-net not found: {src}")
            dst = build / "sa" / f"policy_net_{i + 1}.npz"
            shutil.copy2(src, dst)
            if _pn2.load(dst) is None:
                raise SystemExit(f"{src} FAILS the dim guard")
            print(f"  sa/{dst.name} <- {src} (dim guard OK)")
        # The members must be DIFFERENT policies. Two files holding one policy
        # would give it two votes -- silently weighting a vote whose whole
        # point is one member one vote (E9: policy_v5c_s1 is 100% identical to
        # policy_v5_s1 despite a different md5).
        import hashlib
        seen: dict[str, str] = {}
        for f in ["policy_net.npz"] + [f"policy_net_{i + 1}.npz"
                                       for i in range(len(args.ensemble_net))]:
            digest = hashlib.md5((build / "sa" / f).read_bytes()).hexdigest()
            if digest in seen:
                raise SystemExit(
                    f"ensemble members {seen[digest]} and {f} are the SAME "
                    "bytes -- that is a weighted vote, not an ensemble")
            seen[digest] = f
        print(f"  ensemble: {len(seen)} distinct members")
    for extra in ("value_net.npz", "policy_net.npz", "deck_library.json"):
        present = (build / "sa" / extra).exists()
        note = "present" if present else "excluded -> handcrafted fallback"
        print(f"  sa/{extra}: {note}")

    # Before tarring: would the GRADER's interpreter even parse this? The smoke
    # below runs on the local one and cannot answer that (see KAGGLE_PY).
    _check_kaggle_syntax(build)

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
            env = dict(os.environ,
                       SMOKE_MEMBERS=str(1 + len(args.ensemble_net)))
            proc = subprocess.run([sys.executable, "-X", "utf8", "-c", SMOKE],
                                  cwd=tmp, capture_output=True, text=True,
                                  timeout=1800, env=env)
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
