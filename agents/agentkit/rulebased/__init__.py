"""Run the sample rule-based agents (imported from the notebooks) in-process.

Each `sources/<name>.py` is the notebook's `main.py` verbatim except the
deck.csv loader, replaced with `my_deck = list(MY_DECK)`. `make_rule_agent`
executes the source in a fresh namespace with MY_DECK injected and returns its
`agent` callable. Requires ptcg.env.sdk.load() to have made `cg` importable.
"""
from __future__ import annotations

from pathlib import Path

from ptcg.env import sdk

SOURCES = Path(__file__).resolve().parent / "sources"

# rule agent name -> decks/ module it was tuned for
DECK_MODULE = {
    "dragapult": "dragapult_ex",
    "iono": "iono",
    "abomasnow": "mega_abomasnow_ex",
    "lucario": "mega_lucario_ex",
}


def make_rule_agent(name: str, deck: list[int]):
    sdk.load()
    src_path = SOURCES / f"{name}.py"
    if not src_path.exists():
        raise ValueError(
            f"no rule agent source {src_path}; run scripts/import_rule_agents.py")
    ns: dict = {"MY_DECK": list(deck), "__name__": f"rulebased_{name}"}
    exec(compile(src_path.read_text(encoding="utf-8"), str(src_path), "exec"), ns)
    return ns["agent"]
