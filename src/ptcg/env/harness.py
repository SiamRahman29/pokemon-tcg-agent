"""Local battle harness: run agent-vs-agent games through cg.game.

An Agent is `Callable[[dict], list[int]]` -- the exact Kaggle contract: it gets
the observation dict and returns option indices (or the 60-card deck when
`select` is None).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ptcg.env import sdk

Agent = Callable[[dict], list[int]]

MAX_SELECTS = 6000  # runaway-game guard


class Recorder:
    """Capture a game in Kaggle's own replay format. OPT-IN, off by default.

    **Why this exists (day 15).** `scripts/arena.py` archives one summary row
    per game -- winner, turns, selects, latency, pool. No observations, no
    actions, no trajectories. So after fifteen days there was nothing to watch
    and nothing to learn from: the anchors that carry 71.5% of every weighted
    verdict in this repo had never been observed playing a single turn, and an
    RL probe had no data source at all.

    ⚡ **The payoff is that the format is not ours.** `cg.game.visualize_data()`
    emits exactly the structure Kaggle puts at `steps[0][0]["visualize"]` in a
    downloaded replay, and attaching `obs`/`action` per step is what the
    engine's own notebook does. So a recorded local game is readable by
    **every replay tool already in this repo** -- `p9_field_census.py`,
    `p5a_replays.py`, `build_policy_dataset.py`, `p16_policy_disagree.py` --
    with no adapter, and by the official viewer at `ptcgvis.heroz.jp` via
    `notebooks/visualizer.html`.

    ⚠ `visualize_data()` must be called BEFORE `battle_finish()`; the engine is
    a single-battle-at-a-time ctypes wrapper and finishing frees the buffer.

    ⚠ The `info.TeamNames` / `rewards` keys are what the replay tools key on to
    find "our" seat, so `dump()` writes them even though a local game has no
    teams -- pass `names=` to make a recording addressable by those tools.
    """

    def __init__(self, names: tuple[str, str] = ("seat0", "seat1"),
                 keep_obs: bool = True):
        self.names = names
        self.keep_obs = keep_obs
        # index 0 is the pre-battle state, which has no action -- the engine's
        # notebook seeds both logs with a placeholder and so do we, otherwise
        # every obs is off by one against the visualize stream.
        self.obs_log: list = [""]
        self.action_log: list = [None]
        self.seat_log: list = [None]
        self.vis: list | None = None
        self.result: GameResult | None = None

    def on_select(self, obs: dict, who: int, action: list[int]) -> None:
        if self.keep_obs:
            # `search_begin_input` is the engine's internal scratch and is by
            # far the largest key; the notebook drops it and nothing reads it.
            rec = {k: v for k, v in obs.items() if k != "search_begin_input"}
        else:
            rec = ""
        self.obs_log.append(rec)
        self.action_log.append(list(action))
        self.seat_log.append(who)

    def capture(self, game) -> None:
        """Pull the engine's visualize stream. Call before `battle_finish()`."""
        try:
            self.vis = json.loads(game.visualize_data())
        except Exception:  # noqa: BLE001 -- a recording must never fail a game
            self.vis = None

    def to_replay(self) -> dict:
        """The Kaggle-shaped replay dict: steps[0][0]['visualize'] + info."""
        vis = list(self.vis or [])
        for i in range(len(vis)):
            if i < len(self.obs_log):
                vis[i]["obs"] = self.obs_log[i]
                # the viewer expects one action per seat; the notebook writes
                # the same list twice and the renderer picks by seat.
                a = self.action_log[i]
                vis[i]["action"] = [a, a]
        r = self.result
        rewards = [0, 0]
        if r is not None and r.winner in (0, 1):
            rewards = [1, 0] if r.winner == 0 else [0, 1]
        return {
            "steps": [[{"visualize": vis}]],
            "rewards": rewards,
            "info": {"TeamNames": list(self.names)},
            "ptcg_local": {          # our own metadata, ignored by the viewer
                "winner": None if r is None else r.winner,
                "turns": None if r is None else r.turns,
                "selects": None if r is None else r.selects,
                "seats": self.seat_log,
            },
        }

    def dump(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_replay()), encoding="utf-8")
        return p


@dataclass
class GameResult:
    winner: int          # 0 / 1 / 2 (draw)
    turns: int
    selects: int
    decision_ms: tuple[list[float], list[float]] = field(
        default_factory=lambda: ([], []))
    # Thinking pool left per seat at game end. Kaggle turns an exhausted pool
    # into an instant LOSS; the harness does not, so a slow agent would look
    # fine here and lose every long game on the ladder. Always check this
    # before shipping anything that searches.
    pool_left: tuple[float, float] = (600.0, 600.0)


def play_game(agent0: Agent, agent1: Agent, deck0: list[int],
              deck1: list[int], recorder: "Recorder | None" = None
              ) -> GameResult:
    """One full game. agentN plays seat N. Decks are dealt by the engine.

    `recorder` is OPT-IN and defaults to None, in which case this function is
    byte-identical to its pre-day-15 form: the only additions are two `if
    recorder is not None` guards, so the A/B path pays one predictable branch
    per select and allocates nothing. Proven a no-op by
    `scripts/p20_recorder_equivalence.py`, not by the arena -- §8aa's methods
    rule (the arena is not deterministic run to run and cannot settle a
    question about whether a refactor changed anything).
    """
    api = sdk.api()
    game = sdk.game()

    # Initial deck selection happens outside the battle: agents get
    # select=None and must return their deck. We honor the contract.
    for agent, deck in ((agent0, deck0), (agent1, deck1)):
        ret = agent({"select": None, "logs": [], "current": None})
        if list(ret) != list(deck):
            deck[:] = [int(c) for c in ret]

    obs, _start = game.battle_start(deck0, deck1)
    if obs is None:
        raise RuntimeError("battle_start failed")

    times: tuple[list[float], list[float]] = ([], [])
    overage = [600.0, 600.0]  # mirror Kaggle's per-agent thinking pool
    selects = 0
    result: GameResult | None = None
    try:
        while True:
            state = obs.get("current")
            if state is not None and state["result"] != -1:
                result = GameResult(state["result"], state["turn"], selects,
                                    times, (overage[0], overage[1]))
                return result
            who = state["yourIndex"]
            agent = agent0 if who == 0 else agent1
            obs["remainingOverageTime"] = overage[who]
            t0 = time.perf_counter()
            choice = agent(obs)
            dt = time.perf_counter() - t0
            overage[who] -= dt
            times[who].append(dt * 1000.0)
            picked = [int(c) for c in choice]
            if recorder is not None:
                recorder.on_select(obs, who, picked)
            obs = game.battle_select(picked)
            selects += 1
            if selects > MAX_SELECTS:
                result = GameResult(2, state["turn"], selects, times,
                                    (overage[0], overage[1]))
                return result
    finally:
        # ⚠ ORDER IS LOAD-BEARING: `visualize_data()` reads a buffer that
        # `battle_finish()` frees, so the capture happens first or it returns
        # nothing.
        if recorder is not None:
            recorder.result = result
            recorder.capture(game)
        game.battle_finish()


def _wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def latency_summary(ms: list[float]) -> dict:
    if not ms:
        return {"n": 0}
    s = sorted(ms)
    return {
        "n": len(s),
        "mean": sum(s) / len(s),
        "p50": s[len(s) // 2],
        "p99": s[min(len(s) - 1, int(len(s) * 0.99))],
        "max": s[-1],
    }


def evaluate(agent_a: Agent, agent_b: Agent, deck: list[int],
             games: int = 20) -> dict:
    """A vs B on the same deck, alternating seats. Returns A's summary."""
    wins = draws = losses = 0
    for i in range(games):
        if i % 2 == 0:
            r = play_game(agent_a, agent_b, list(deck), list(deck))
            a_won = r.winner == 0
        else:
            r = play_game(agent_b, agent_a, list(deck), list(deck))
            a_won = r.winner == 1
        if r.winner == 2:
            draws += 1
        elif a_won:
            wins += 1
        else:
            losses += 1
    score = (wins + 0.5 * draws) / games
    lo, hi = _wilson(wins + 0.5 * draws, games)
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "win_rate": score, "wilson_low": lo, "wilson_high": hi}


def evaluate_paired(agent_a: Agent, agent_b: Agent, deck_a: list[int],
                    deck_b: list[int], matches: int = 10,
                    on_game=None) -> dict:
    """Seat-swapped paired matches: each match plays A-as-P0 then A-as-P1."""
    wins = draws = losses = 0
    wdl_p0 = [0, 0, 0]
    wdl_p1 = [0, 0, 0]
    games = 0
    for m in range(matches):
        for a_seat in (0, 1):
            if a_seat == 0:
                r = play_game(agent_a, agent_b, list(deck_a), list(deck_b))
            else:
                r = play_game(agent_b, agent_a, list(deck_b), list(deck_a))
            games += 1
            if r.winner == 2:
                draws += 1
                (wdl_p0 if a_seat == 0 else wdl_p1)[1] += 1
            elif r.winner == a_seat:
                wins += 1
                (wdl_p0 if a_seat == 0 else wdl_p1)[0] += 1
            else:
                losses += 1
                (wdl_p0 if a_seat == 0 else wdl_p1)[2] += 1
            if on_game is not None:
                on_game(m, a_seat, r)
    score = (wins + 0.5 * draws) / games if games else 0.0
    lo, hi = _wilson(wins + 0.5 * draws, games)
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "score": score, "wilson_low": lo, "wilson_high": hi,
            "a_as_p0_wdl": tuple(wdl_p0), "a_as_p1_wdl": tuple(wdl_p1)}
