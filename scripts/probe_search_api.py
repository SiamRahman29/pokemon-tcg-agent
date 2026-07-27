"""Probe the engine's search API: correctness of branching + steps/sec.

Plays a scripted game (rule:iono vs rule:dragapult) until midgame, then from a
MAIN select runs search_begin with a naive determinization and measures:
  * search_step throughput (steps/sec)
  * whether branching (two search_steps from the same searchId) works
  * whether the search observation reveals the opponent's hand when they act
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402

sdk.load()
import cg.api as api  # noqa: E402
import cg.game as game  # noqa: E402

from agentkit.rulebased import make_rule_agent  # noqa: E402
from decks import iono, dragapult_ex  # noqa: E402

deck0 = [c for c, n in iono.DECKLIST.items() for _ in range(n)]
deck1 = [c for c, n in dragapult_ex.DECKLIST.items() for _ in range(n)]
a0 = make_rule_agent("iono", deck0)
a1 = make_rule_agent("dragapult", deck1)

obs, _ = game.battle_start(deck0, deck1)
target_obs = None
for i in range(400):
    st = obs["current"]
    if st["result"] != -1:
        print("game ended early", st["result"])
        sys.exit(1)
    who = st["yourIndex"]
    sel = obs["select"]
    # a midgame MAIN select for player 0 with several options
    if (who == 0 and st["turn"] >= 6 and sel["type"] == 0
            and len(sel["option"]) >= 4):
        target_obs = obs
        break
    choice = (a0 if who == 0 else a1)(obs)
    obs = game.battle_select(list(choice))

assert target_obs is not None, "no suitable midgame select found"
o = api.to_observation_class(target_obs)
st = o.current
me = st.yourIndex
opp = 1 - me
print(f"midgame reached: turn={st.turn} me={me} options={len(o.select.option)}"
      f" handCount(opp)={st.players[opp].handCount}"
      f" deckCount: mine={st.players[me].deckCount} opp={st.players[opp].deckCount}")

# --- naive determinization ---------------------------------------------------
my_deck_all = list(deck0)
known_mine = [c.id for c in st.players[me].hand or []]
known_mine += [c.id for c in st.players[me].discard]
for pk in (st.players[me].active + st.players[me].bench):
    if pk:
        known_mine.append(pk.id)
        known_mine += [c.id for c in pk.energyCards] + [c.id for c in pk.tools]
        known_mine += [c.id for c in pk.preEvolution]
pool = list(my_deck_all)
for c in known_mine:
    if c in pool:
        pool.remove(c)
random.shuffle(pool)
n_prize = len(st.players[me].prize)
my_prize = pool[:n_prize]
my_deck = pool[n_prize:]

opp_deck_all = list(deck1)
known_opp = [c.id for c in st.players[opp].discard]
for pk in (st.players[opp].active + st.players[opp].bench):
    if pk:
        known_opp.append(pk.id)
        known_opp += [c.id for c in pk.energyCards] + [c.id for c in pk.tools]
        known_opp += [c.id for c in pk.preEvolution]
pool2 = list(opp_deck_all)
for c in known_opp:
    if c in pool2:
        pool2.remove(c)
random.shuffle(pool2)
n_oh = st.players[opp].handCount
n_op = len(st.players[opp].prize)
opp_hand = pool2[:n_oh]
opp_prize = pool2[n_oh:n_oh + n_op]
opp_deck = pool2[n_oh + n_op:]

t0 = time.perf_counter()
root = api.search_begin(o, my_deck, my_prize, opp_deck, opp_prize, opp_hand, [])
t1 = time.perf_counter()
print(f"search_begin: {(t1 - t0) * 1000:.1f} ms; root searchId={root.searchId}")

# --- branching test ----------------------------------------------------------
s_a = api.search_step(root.searchId, [0])
s_b = api.search_step(root.searchId, [1])
print("branching ok:", s_a.searchId != s_b.searchId,
      f"(ids {s_a.searchId}, {s_b.searchId})")

# does opponent's hand become visible when they act in search?
def find_opp_turn(state, limit=200):
    cur = state
    for _ in range(limit):
        oo = cur.observation
        if oo.current.result != -1:
            return None
        if oo.current.yourIndex == opp:
            return oo
        k = max(oo.select.minCount, min(1, oo.select.maxCount))
        cur = api.search_step(cur.searchId, list(range(k)))
    return None

oo = find_opp_turn(s_a)
if oo is not None:
    print("opp acting: hand visible?", oo.current.players[opp].hand is not None)

# --- throughput --------------------------------------------------------------
rng = random.Random(0)
n_steps = 0
t0 = time.perf_counter()
cur = root
depths = 0
while time.perf_counter() - t0 < 3.0:
    oo = cur.observation
    if oo.current.result != -1 or oo.select is None:
        cur = root
        depths += 1
        continue
    sel = oo.select
    k = rng.randint(sel.minCount, sel.maxCount)
    pick = rng.sample(range(len(sel.option)), k)
    cur = api.search_step(cur.searchId, pick)
    n_steps += 1
t1 = time.perf_counter()
print(f"throughput: {n_steps / (t1 - t0):.0f} search_steps/sec "
      f"({n_steps} steps, {depths} restarts)")

api.search_end()
game.battle_finish()
print("done.")
