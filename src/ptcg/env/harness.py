"""Local battle harness: run agent-vs-agent games through cg.game.

An Agent is `Callable[[dict], list[int]]` -- the exact Kaggle contract: it gets
the observation dict and returns option indices (or the 60-card deck when
`select` is None).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

from ptcg.env import sdk

Agent = Callable[[dict], list[int]]

MAX_SELECTS = 6000  # runaway-game guard


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
              deck1: list[int]) -> GameResult:
    """One full game. agentN plays seat N. Decks are dealt by the engine."""
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
    try:
        while True:
            state = obs.get("current")
            if state is not None and state["result"] != -1:
                return GameResult(state["result"], state["turn"], selects,
                                  times, (overage[0], overage[1]))
            who = state["yourIndex"]
            agent = agent0 if who == 0 else agent1
            obs["remainingOverageTime"] = overage[who]
            t0 = time.perf_counter()
            choice = agent(obs)
            dt = time.perf_counter() - t0
            overage[who] -= dt
            times[who].append(dt * 1000.0)
            obs = game.battle_select([int(c) for c in choice])
            selects += 1
            if selects > MAX_SELECTS:
                return GameResult(2, state["turn"], selects, times,
                                  (overage[0], overage[1]))
    finally:
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
