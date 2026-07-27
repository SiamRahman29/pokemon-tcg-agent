"""Turn-level determinized search.

For each sampled world (determinization) we search over OUR selects: try each
root candidate, then continue with a greedy 1-step-lookahead playout until the
opponent's next main select (or terminal / depth cap), and score the leaf with
the heuristic eval. Root values are averaged across worlds.

Opponent micro-decisions occurring inside our turn (e.g. their forced discards)
are played greedily *against* us (they minimize our eval).
"""
from __future__ import annotations

import itertools
import math
import random
import time

from . import fastsearch as fs
from . import policynet as pnet
from . import valuenet as vnet
from .evalfn import evaluate
from .worlds import determinize

MAIN = 0  # SelectType.MAIN

ROOT_CAP = 24        # max candidate combos at the root select
ROOT_POLICY_KEEP = 10  # root candidates kept when the policy net prunes
PLAYOUT_CAP = 8      # max candidates per playout decision
MAX_PLAYOUT_STEPS = 160
MAX_WORLDS = 12

DEBUG = bool(int(__import__("os").environ.get("SA_DEBUG", "0")))

STATS = {"decides": 0, "worlds": 0, "begin_fail": 0, "step_fail": 0,
         "playout_steps": 0, "budget_s": 0.0, "spent_s": 0.0}


def candidate_combos(sel: dict, cap: int, rng: random.Random | None = None) \
        -> list[list[int]]:
    """Enumerate (or sample) legal selection index lists."""
    n = len(sel["option"])
    mn = sel["minCount"]
    mx = min(sel["maxCount"], n)
    if mx <= 0:
        return [[]]
    combos: list[list[int]] = []
    if mn == 0:
        combos.append([])
    per_size = max(2, cap // max(1, mx - max(mn, 1) + 1))
    for k in range(max(mn, 1), mx + 1):
        total = math.comb(n, k)
        if total <= per_size:
            combos.extend(list(c) for c in itertools.combinations(range(n), k))
        else:
            seen = set()
            first = tuple(range(k))
            last = tuple(range(n - k, n))
            for c in (first, last):
                if c not in seen:
                    seen.add(c)
                    combos.append(list(c))
            r = rng or random
            while len(seen) < per_size:
                c = tuple(sorted(r.sample(range(n), k)))
                if c not in seen:
                    seen.add(c)
                    combos.append(list(c))
    return combos[:cap] if len(combos) > cap else combos


class Planner:
    def __init__(self, decklist: list[int]):
        self.decklist = list(decklist)
        self.rng = random.Random(0xC0FFEE)

    # ---- greedy playout ------------------------------------------------------

    def _playout(self, sid: int, obs: dict, me: int, deadline: float) -> float:
        """Greedy continuation: rest of our turn, the opponent's whole turn
        (played greedily against us), stopping at our next main select."""
        steps = 0
        opp_turn_seen = False
        while steps < MAX_PLAYOUT_STEPS:
            state = obs.get("current")
            sel = obs.get("select")
            if state is None or sel is None or state["result"] != -1:
                break
            acting = state["yourIndex"]
            if sel["type"] == MAIN:
                if acting != me:
                    opp_turn_seen = True
                elif opp_turn_seen:
                    break  # back to us: 1.5-turn horizon reached
            if time.perf_counter() > deadline:
                break
            cands = candidate_combos(sel, PLAYOUT_CAP, self.rng)
            if len(cands) == 1:
                try:
                    sid, obs = fs.step(sid, cands[0])
                except fs.SearchError:
                    break
                steps += 1
                continue
            net = pnet.get()
            if net is not None:
                # clone policy plays both sides: one step, no child evals
                try:
                    sid, obs = fs.step(sid, net.choose(obs))
                    steps += 1
                    continue
                except (fs.SearchError, Exception):
                    pass  # fall through to greedy
            best_v = None
            best = None
            for c in cands:
                try:
                    sid2, obs2 = fs.step(sid, c)
                except fs.SearchError:
                    continue
                steps += 1
                v = evaluate(obs2["current"], me)
                better = (best_v is None
                          or (v > best_v if acting == me else v < best_v))
                if better:
                    best_v = v
                    best = (sid2, obs2)
                if time.perf_counter() > deadline:
                    break
            if best is None:
                break
            sid, obs = best
        return self._leaf_value(obs, me)

    def _leaf_value(self, obs: dict, me: int) -> float:
        cur = obs.get("current")
        if cur is None:
            return 0.0
        if cur["result"] != -1:
            return evaluate(cur, me)  # terminal: +/-WIN or 0
        net = vnet.get()
        if net is not None:
            # scaled so terminals still dominate; comparable across leaves
            return 40.0 * (net.win_prob(cur, me) - 0.5)
        return evaluate(cur, me)

    # ---- root decision -------------------------------------------------------

    def decide(self, obs: dict, known_opp_hand: list[int], budget_s: float,
               combos: list[list[int]]) -> list[int]:
        me = obs["current"]["yourIndex"]
        sbi = obs["search_begin_input"]
        deck_visible = obs["select"].get("deck") is not None

        # policy prior: keep only the top root candidates the clone would
        # consider, plus its outright choice
        net = pnet.get()
        if net is not None and len(combos) > ROOT_POLICY_KEEP:
            try:
                sc = net.scores(obs)
                choice = net.choose(obs)
                ranked = sorted(
                    combos,
                    key=lambda c: -(sum(sc[i] for i in c) / max(len(c), 1)
                                    + 0.01 * len(c)))
                combos = ranked[:ROOT_POLICY_KEEP]
                if choice not in combos:
                    combos.append(choice)
            except Exception:
                pass

        t0 = time.perf_counter()
        deadline = t0 + budget_s
        totals = [0.0] * len(combos)
        counts = [0] * len(combos)
        alive = list(range(len(combos)))  # root candidates still in play

        worlds = 0
        while worlds < MAX_WORLDS and len(alive) >= 1:
            now = time.perf_counter()
            if worlds > 0 and (now > deadline or len(alive) == 1):
                break
            # keep at least one full world even if the budget is tiny
            world = determinize(obs, self.decklist, known_opp_hand, self.rng)
            try:
                root_sid, _root_obs = fs.begin(
                    sbi,
                    [] if deck_visible else world.my_deck,
                    world.my_prize, world.opp_deck, world.opp_prize,
                    world.opp_hand, world.opp_active)
            except (fs.SearchError, ValueError):
                STATS["begin_fail"] += 1
                break  # determinization rejected; bail to what we have
            for i in alive:
                try:
                    sid, o = fs.step(root_sid, combos[i])
                except fs.SearchError:
                    totals[i] += -1e6
                    counts[i] += 1
                    continue
                v = self._playout(sid, o, me, deadline)
                totals[i] += v
                counts[i] += 1
                if time.perf_counter() > deadline and worlds > 0:
                    break
            fs.end()
            worlds += 1
            # successive halving: drop clearly-worse candidates so later
            # worlds concentrate on the contenders
            if len(alive) > 3:
                scored = sorted(
                    (totals[i] / max(counts[i], 1), i) for i in alive)
                keep = max(3, len(alive) // 2) if worlds >= 2 else \
                    max(4, int(len(alive) * 0.7))
                alive = sorted(i for _, i in scored[-keep:])

        best_i = alive[0] if alive else 0
        best_v = None
        for i in (alive or range(len(combos))):
            if counts[i] == 0:
                continue
            v = totals[i] / counts[i]
            if best_v is None or v > best_v:
                best_v = v
                best_i = i

        STATS["decides"] += 1
        STATS["worlds"] += worlds
        STATS["budget_s"] += budget_s
        STATS["spent_s"] += time.perf_counter() - t0
        if DEBUG:
            import sys
            st = obs["current"]
            print(f"[sa] t{st['turn']} sel{obs['select']['type']}"
                  f"/{len(combos)}c w{worlds} "
                  f"best={best_v if best_v is not None else float('nan'):.2f}"
                  f" pick={combos[best_i]} bud={budget_s:.2f}s"
                  f" spent={time.perf_counter() - t0:.2f}s",
                  file=sys.stderr, flush=True)
        return combos[best_i]
