"""Package an exported Transformer `.npz` (agents/tx) as a Playground submission.

    python -X utf8 scripts/build_tx_submission.py --model <model.npz | dir>

The bundle needs numpy only, with no torch and no engine copy:

    main.py            entry point: agent(obs) -> list[int]
    deck.csv           the 60 cards the model was trained for
    model.npz          the exported weights (header carries the deck profile)
    card_table.json    the engine card table (R2 pool) the features read
    scio_tx/           agents/tx, minus training code

Kaggle's runner puts other environments' directories on sys.path ahead of the
agent directory, one of them holding a module named `agents.py` (lux_ai_s3),
so the package gets a name nothing else uses and main.py moves the agent
directory to the front of sys.path.

After building, the extracted bundle is smoked in a separate process:
  - every .py compiles under the runner's Python (3.11),
  - main.py is exec'd the way Kaggle does (no `__file__`), behind decoy
    `agents.py` and `tx.py` modules,
  - it plays rule bots through the Playground engine with no fallbacks and
    within the time budget,
  - optionally (`--runner-python`, a Python with kaggle-environments) it
    self-plays in the real runner, which is Kaggle's validation episode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

PACKAGE = "scio_tx"
SOURCES = ("__init__.py", "agent.py", "contract.py", "deck.py", "features.py",
           "numpy_policy.py")
SIZE_CAP_MIB = 197.7

MAIN_PY = r'''"""Grimmsnarl entity Transformer (v5_s2 distillation -> PPO), NumPy inference."""

import json
import os
import sys
import traceback

# Kaggle execs this file without __file__; fall back to the agent directory, then cwd.
_CANDIDATES = []
try:
    _CANDIDATES.append(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
_CANDIDATES += ["/kaggle_simulations/agent", os.getcwd()]
HERE = next((p for p in _CANDIDATES if p and os.path.exists(os.path.join(p, "model.npz"))),
            _CANDIDATES[-1])
# Always first: the runner may list HERE already, but behind directories that shadow names.
while HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)

from scio_tx.agent import TxAgent  # noqa: E402

with open(os.path.join(HERE, "deck.csv")) as _f:
    DECK = [int(v) for v in _f.read().split()[:60]]
with open(os.path.join(HERE, "card_table.json")) as _f:
    CARDS = {int(k): v for k, v in json.load(_f).items()}
AGENT = TxAgent(DECK, os.path.join(HERE, "model.npz"), CARDS)
STATS = {"decisions": 0, "fallbacks": 0, "games": 0}
print("[health] model loaded: MODEL_ID", flush=True)


def _fallback(obs):
    """Legal default when inference raises: never forfeit a live episode."""
    sel = obs["select"]
    return list(range(max(sel.get("minCount", 0), min(1, sel["maxCount"], len(sel["option"])))))


def agent(obs, config=None):
    """The deck at the opening handshake, otherwise the Transformer's choice."""
    if obs.get("select") is None:
        STATS["games"] += 1
        print("[health] game=%d decisions=%d fallbacks=%d" % (
            STATS["games"], STATS["decisions"], STATS["fallbacks"]), flush=True)
        return list(DECK)
    STATS["decisions"] += 1
    try:
        return AGENT(obs)
    except Exception:
        STATS["fallbacks"] += 1
        if STATS["fallbacks"] <= 3:
            print("[health] FALLBACK", flush=True)
            traceback.print_exc()
        return _fallback(obs)
'''

SMOKE_PY = r'''
import json, os, sys, tempfile, time
bundle, root, games = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.chdir(bundle)
# Mimic Kaggle's runner: decoys that shadow common names sit first, the agent
# directory last. The repo paths are only for the engine and the rule bots.
decoy = tempfile.mkdtemp()
for name in ("agents.py", "tx.py"):
    with open(os.path.join(decoy, name), "w") as f:
        f.write("raise ImportError('decoy %s shadowed the bundle')\n" % name)
sys.path[:] = [p for p in sys.path if p not in ("", bundle)]
sys.path.insert(0, decoy)
sys.path += [os.path.join(root, "src"), os.path.join(root, "agents"), root, bundle]
env = {"__name__": "kaggle_agent"}
exec(compile(open(os.path.join(bundle, "main.py")).read(), "main.py", "exec"), env)
assert "__file__" not in env
assert env["AGENT"].__class__.__module__ == "scio_tx.agent", env["AGENT"].__class__.__module__

import importlib
from ptcg.env import sdk
sdk.load()
import cg.game as game
from agentkit.rulebased import make_rule_agent

# (rule bot, its deck module), the opponents tx_eval measures against.
OPPONENTS = [("dragapult", "dragapult_ex"), ("iono", "iono"),
             ("abomasnow", "mega_abomasnow_ex"), ("lucario", "mega_lucario_ex")]
deck = env["agent"]({"select": None})
assert sorted(deck) == sorted(env["DECK"]) and len(deck) == 60
report = {"games": 0, "wins": 0, "max_latency": 0.0, "max_game_seconds": 0.0}
for g in range(games):
    bot, deck_mod = OPPONENTS[g % len(OPPONENTS)]
    opp_deck = [c for c, n in importlib.import_module("decks." + deck_mod).DECKLIST.items()
                for _ in range(n)]
    opp = make_rule_agent(bot, opp_deck)
    seat = (g // len(OPPONENTS)) % 2
    decks = [deck, opp_deck] if seat == 0 else [opp_deck, deck]
    obs, _ = game.battle_start(decks[0], decks[1])
    spent, steps, pool = 0.0, 0, [600.0, 600.0]
    try:
        while True:
            cur = obs["current"]
            if cur["result"] != -1:
                break
            who = cur["yourIndex"]
            obs["remainingOverageTime"] = pool[who]
            t0 = time.perf_counter()
            choice = env["agent"](obs) if who == seat else opp(obs)
            dt = time.perf_counter() - t0
            pool[who] -= dt
            if who == seat:
                spent += dt
                report["max_latency"] = max(report["max_latency"], dt)
            obs = game.battle_select([int(c) for c in choice])
            steps += 1
            if steps > 6000:
                raise SystemExit("game did not terminate")
    finally:
        game.battle_finish()
    report["games"] += 1
    report["wins"] += int(cur["result"] == seat)
    report["max_game_seconds"] = max(report["max_game_seconds"], spent)
report.update({k: env["STATS"][k] for k in ("decisions", "fallbacks")})
print("SMOKE " + json.dumps(report), flush=True)
assert env["STATS"]["fallbacks"] == 0, "inference fell back"
assert report["max_game_seconds"] < 300, "too slow for the 600 s budget"
'''

RUNNER_PY = r'''
import os, sys
from kaggle_environments import make
bundle = sys.argv[1]
os.chdir(bundle)
main = os.path.join(bundle, "main.py")
# Kaggle's validation episode: the submission against itself in the real runner.
steps = make("cabt", debug=True).run([main, main])
statuses = [s["status"] for s in steps[-1]]
print("RUNNER statuses=%s rewards=%s steps=%d"
      % (statuses, [s["reward"] for s in steps[-1]], len(steps)))
assert statuses == ["DONE", "DONE"], statuses
'''


def resolve_model(path: Path) -> Path:
    """A .npz, or a directory holding exactly one model.npz."""
    if path.is_file():
        return path
    found = sorted(path.rglob("model.npz")) if path.is_dir() else []
    if len(found) != 1:
        raise SystemExit(f"{path}: expected an .npz or a directory with one model.npz")
    return found[0]


def find_python(version: str = "3.11") -> str:
    """An interpreter matching Kaggle's episode runner, else ours."""
    if os.environ.get("PTCG_KAGGLE_PYTHON"):
        return os.environ["PTCG_KAGGLE_PYTHON"]
    if sys.version_info[:2] == tuple(int(v) for v in version.split(".")):
        return sys.executable
    try:
        return subprocess.run(["uv", "python", "find", version], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return sys.executable


def build(model: Path, staging: Path) -> dict:
    """Write the bundle tree; return the model header (fails here, not on Kaggle)."""
    from arena import resolve_deck
    from tx.agent import TxAgent
    from tx.deck import engine_tables

    cards = engine_tables()[0]
    header = json.loads(str(__import__("numpy").load(model)["__header__"]))
    _, deck = resolve_deck(header["profile"]["name"])
    TxAgent(list(deck), str(model), cards)  # deck, profile and contract checks

    if staging.exists():
        shutil.rmtree(staging)
    (staging / PACKAGE).mkdir(parents=True)
    for name in SOURCES:
        text = (ROOT / "agents/tx" / name).read_text(encoding="utf-8")
        text = text.replace("from tx.", f"from {PACKAGE}.").replace("import tx.", f"import {PACKAGE}.")
        if "from tx " in text or "import tx\n" in text or "from tx." in text:
            raise SystemExit(f"unrewritten tx import in {name}")
        (staging / PACKAGE / name).write_text(text, encoding="utf-8")
    for name in ("LICENSE", "NOTICE.md"):
        shutil.copy2(ROOT / "agents/tx" / name, staging / PACKAGE / name)
    (staging / "deck.csv").write_text("\n".join(map(str, deck)) + "\n")
    shutil.copy2(model, staging / "model.npz")
    (staging / "card_table.json").write_text(
        json.dumps({str(k): v for k, v in sorted(cards.items())}, separators=(",", ":")))
    digest = hashlib.sha256(model.read_bytes()).hexdigest()[:12]
    tag = (f"{digest} ppo cycle {header['cycle']}" if "cycle" in header
           else f"{digest} distilled epoch {header.get('epoch')}")
    (staging / "main.py").write_text(MAIN_PY.replace("MODEL_ID", tag), encoding="utf-8")
    print(f"model {model} ({tag}); deck {header['profile']['name']}; cards {len(cards)}")
    return header


def smoke(archive: Path, games: int, python: str, runner_python: str | None) -> None:
    """Compile under the runner's Python and play the extracted bundle."""
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        with tarfile.open(archive) as tar:
            tar.extractall(bundle, filter="data")
        names = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*.py"))
        subprocess.run([python, "-m", "py_compile", *names], cwd=bundle, check=True)
        for cache in bundle.rglob("__pycache__"):
            shutil.rmtree(cache)
        print(f"  syntax OK under {python}")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        subprocess.run([python, "-X", "utf8", "-c", SMOKE_PY, str(bundle), str(ROOT),
                        str(games)], check=True, env=env)
        if runner_python:
            subprocess.run([runner_python, "-c", RUNNER_PY, str(bundle)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True,
                    help="exported .npz (e.g. a PPO candidate) or a directory with one model.npz")
    ap.add_argument("--out", type=Path, default=None, help="archive path (default dist/)")
    ap.add_argument("--games", type=int, default=8, help="smoke games against rule bots")
    ap.add_argument("--python", default=None, help="smoke interpreter (default: a 3.11)")
    ap.add_argument("--runner-python", default=None,
                    help="Python with kaggle-environments: also self-play in the real runner")
    ap.add_argument("--no-smoke", action="store_true")
    a = ap.parse_args()

    model = resolve_model(a.model.resolve())
    digest = hashlib.sha256(model.read_bytes()).hexdigest()[:8]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = a.out or ROOT / "dist" / f"submission_tx-{digest}_{stamp}.tar.gz"
    staging = out.parent / f"_build_tx-{digest}"
    build(model, staging)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for path in sorted(staging.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                tar.add(path, arcname=path.relative_to(staging).as_posix())
    shutil.rmtree(staging)
    size = out.stat().st_size / 2**20
    print(f"built {out} ({size:.1f} MiB)")
    if size > SIZE_CAP_MIB:
        raise SystemExit(f"exceeds the {SIZE_CAP_MIB} MiB cap")
    if not a.no_smoke:
        smoke(out, a.games, a.python or find_python(), a.runner_python)
        print("smoke: OK")
    print(f"upload: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
