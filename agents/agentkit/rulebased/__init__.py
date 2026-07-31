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
    # The counter-meta pilot (scripts/import_crustle_agent.py). Crustle is
    # 12.8% of the field we actually play (`p9_field_census.py`, 109 real
    # ladder games). Its own list is `crustle_v1`; the field's consensus list
    # is `crustle` (see both decks' docstrings).
    #
    # ⚠ The comment that used to sit here said `lucario_v10` had "fallen to
    # 0%" and that Crustle -- "not v10" -- was the opponent that mattered now.
    # That was mined from the daily TOP-episode datasets, which bottom out at
    # avg_score 1055 while we play at 825-950. In our own 109 real games
    # Mega Lucario is **12.8% of the field**, tied with Crustle. `rule:v10` was
    # never stale; it was measured against the wrong population.
    "crustle": "crustle_v1",
    # The two anchors the census said we were missing
    # (scripts/import_field_agents.py). Between them they close the gap from
    # 39.4% to 71.6% of the real field.
    #
    #   `alakazam5` -- the LARGEST archetype, 22.0%. Author reports 5th place,
    #   pure rules, no ML, no search. Powerful Hand does 20 per card in hand,
    #   so its damage scales with hand size; nothing in targeting.py sees that.
    #
    #   `archaludon` -- 10.1%, and our WORST matchup at 45.5%. A second
    #   damage-reduction deck (Full Metal Lab: -30 into any Metal Pokemon),
    #   which `targeting.WALL_POKEMON = {345}` does not know about.
    "alakazam5": "alakazam5",
    "archaludon": "archaludon_ex",
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
