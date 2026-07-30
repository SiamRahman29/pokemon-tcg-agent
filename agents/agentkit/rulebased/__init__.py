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
    # the public LB 950+ baseline (scripts/import_v10_agent.py) -- the only
    # opponent we have that is measured well above the sample agents.
    # `v10x` is the same agent with its unreachable MCTS made reachable.
    "v10": "lucario_v10",
    "v10x": "lucario_v10",
    # The counter-meta pilot (scripts/import_crustle_agent.py). Crustle went
    # from 1 seat in 1,600 to 18.1% of the field at 56.6% WR between 07-24 and
    # 07-29 while `lucario_v10` fell to 0%, so this -- not `v10` -- is the
    # opponent that matters now. Its own list is `crustle_v1`; the field's
    # consensus list is `crustle` (see both decks' docstrings).
    "crustle": "crustle_v1",
}


def make_rule_agent(name: str, deck: list[int], overrides: dict | None = None):
    """`overrides` are module globals poked in after exec, for agents that
    expose knobs as module-level constants (v10's USE_SEARCH / time budget).
    Poking the namespace keeps the imported source byte-identical to the
    notebook, and keeps two instances in one process independent."""
    sdk.load()
    src_path = SOURCES / f"{name}.py"
    if not src_path.exists():
        raise ValueError(
            f"no rule agent source {src_path}; run scripts/import_rule_agents.py")
    ns: dict = {"MY_DECK": list(deck), "__name__": f"rulebased_{name}"}
    exec(compile(src_path.read_text(encoding="utf-8"), str(src_path), "exec"), ns)
    for key, value in (overrides or {}).items():
        if key not in ns:
            raise ValueError(f"rule:{name} has no module global {key!r}")
        ns[key] = value
    return ns["agent"]
