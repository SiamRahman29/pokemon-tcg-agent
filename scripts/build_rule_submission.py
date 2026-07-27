"""Assemble a self-contained *rule-based* submission and zip it for Kaggle.

The four sample rule-based agents (`notebooks/a-sample-rule-based-agent-*.ipynb`)
are already complete Kaggle submissions: a `main.py` that loads `deck.csv` and
defines `agent(obs_dict) -> list[int]`, depending only on the `cg` engine. This
builder ships exactly that -- no agentkit, no torch:

    main.py    <- the notebook's `%%writefile main.py` cell, verbatim
    cg/        <- the engine (from the local SDK)
    deck.csv   <- the agent's own 60-card deck (from decks/<module>.py)

    python scripts/build_rule_submission.py [name]   # name defaults to `iono`

After zipping it verifies the archive is under the Kaggle size cap and smoke-runs
the *extracted* submission from a temp dir (a stand-in for
`/kaggle_simulations/agent/`) so we know it imports and plays a full legal game
standalone. Output: dist/submission_rule-<name>_<stamp>.zip (+ a stable
dist/submission.zip pointer).
"""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ptcg import config  # noqa: E402

SIZE_CAP_MIB = 197.7

# name -> (notebook filename, decks/ module with the exact deck it was tuned for)
RULE_SUBS = {
    "dragapult": ("a-sample-rule-based-agent-dragapult-ex-deck.ipynb", "dragapult_ex"),
    "iono": ("a-sample-rule-based-agent-iono-s-deck.ipynb", "iono"),
    "abomasnow": ("a-sample-rule-based-agent-mega-abomasnow-ex-deck.ipynb", "mega_abomasnow_ex"),
    "lucario": ("a-sample-rule-based-agent-mega-lucario-ex-deck.ipynb", "mega_lucario_ex"),
}


def notebook_main(nb_path: Path) -> str:
    """The verbatim `%%writefile main.py` cell body (magic line stripped)."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "writefile main.py" in src:
                lines = src.splitlines(keepends=True)
                return "".join(lines[1:])  # drop `%%writefile main.py`
    raise SystemExit(f"no `%%writefile main.py` cell in {nb_path.name}")


# Smoke test run *inside* the extracted submission dir: import main standalone
# (its own cg + deck.csv), then play one full game of main.agent vs a trivial
# legal opponent. Prints RESULT=<winner> on success; any exception fails the build.
SMOKE = r'''
import sys
sys.path.insert(0, ".")
import main                      # loads ./cg and ./deck.csv, defines agent()
import cg.game as game
import cg.api as api

deck = list(main.my_deck)
assert len(deck) == 60, f"deck.csv is {len(deck)} cards, not 60"

def opponent(obs_dict):
    obs = api.to_observation_class(obs_dict)
    if obs.select is None:
        return deck
    n = max(obs.select.minCount, 0)
    return list(range(n))         # first n options -- always legal, may be []

obs = game.battle_start(deck, deck)[0]
selects = 0
try:
    while True:
        o = api.to_observation_class(obs)
        state = o.current
        if state is not None and state.result != -1:
            print(f"RESULT={state.result} turns={state.turn} selects={selects}")
            break
        if o.select is None:
            print(f"RESULT={o.current.result if o.current else 2} (no-select end)")
            break
        who = state.yourIndex if state is not None else 0
        choice = main.agent(obs) if who == 0 else opponent(obs)
        obs = game.battle_select(list(choice))
        selects += 1
        if selects > 6000:
            raise SystemExit("game did not terminate")
finally:
    game.battle_finish()
'''


def smoke_test(zip_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        shutil.unpack_archive(str(zip_path), tmp)
        proc = subprocess.run([sys.executable, "-c", SMOKE], cwd=tmp,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(
                f"smoke test FAILED (extracted layout):\n{proc.stdout}\n{proc.stderr}")
        print(f"  smoke: {proc.stdout.strip()}")


def main() -> int:
    name = (sys.argv[1] if len(sys.argv) > 1 else "iono").lower()
    if name not in RULE_SUBS:
        raise SystemExit(f"unknown rule agent {name!r}; choose from {list(RULE_SUBS)}")
    nb_file, deck_mod = RULE_SUBS[name]

    sdk_dir = config.find_sdk_dir()
    if sdk_dir is None:
        raise SystemExit("cannot find the cg engine (need a cg/api.py under the repo).")

    build = config.DIST_DIR / f"_build_rule-{name}"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)

    (build / "main.py").write_text(
        notebook_main(ROOT / "notebooks" / nb_file), encoding="utf-8")
    print(f"main.py: notebooks/{nb_file} (verbatim)")

    shutil.copytree(sdk_dir / "cg", build / "cg",
                    ignore=shutil.ignore_patterns("__pycache__"))
    print(f"cg:      {sdk_dir / 'cg'}")

    deck = importlib.import_module(f"decks.{deck_mod}").DECK
    (build / "deck.csv").write_text(deck.to_csv(), encoding="utf-8")
    print(f"deck.csv:decks/{deck_mod}.py ({deck.size} cards)")

    for pyc in build.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = config.DIST_DIR / f"submission_rule-{name}_{stamp}"
    out = Path(shutil.make_archive(str(base), "zip", root_dir=build))
    shutil.rmtree(build)

    size_mib = out.stat().st_size / (1024 * 1024)
    print(f"built {out} ({size_mib:.1f} MiB)")
    if size_mib > SIZE_CAP_MIB:
        raise SystemExit(f"archive {size_mib:.1f} MiB exceeds cap {SIZE_CAP_MIB} MiB")
    print(f"  size OK (cap {SIZE_CAP_MIB} MiB)")

    smoke_test(out)

    latest = config.DIST_DIR / "submission.zip"
    shutil.copy2(out, latest)
    print(f"latest -> {latest}")
    print(f"\nupload: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
