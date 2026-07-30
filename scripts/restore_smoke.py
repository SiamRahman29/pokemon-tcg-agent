"""Smoke a PRESERVED bundle -- can we actually restore it if a submission fails?

Version-agnostic on purpose: an older bundle carries its own `sa/policynet.py`
without the `opt_in` property added 2026-07-31, so the check for "is the net
live" has to work either way. What matters is the same either way: a net that
fails the dim guard does not crash, it plays random-legal.

    cd <extracted bundle> && python -X utf8 restore_smoke.py
"""
import sys
import time

sys.path.insert(0, ".")

# Load main.py the way Kaggle does: exec the source with no __file__ in globals.
with open("main.py", "rb") as fh:
    src = fh.read()
env = {}
exec(compile(src, "main.py", "exec"), env)
assert "__file__" not in env, "smoke must not leak __file__ into agent globals"

deck = list(env["_deck"])
assert len(deck) == 60, len(deck)

ag = env.get("_agent")
from sa import policynet as pn  # noqa: E402

live = getattr(ag, "net", None) or pn.get()
assert live is not None, "POLICY NET NOT LOADED -- would play random-legal"
print("NET_OK  opt_in=%s state_in=%s"
      % (getattr(live, "opt_in", "n/a"), live.state_in))
print("FLAGS   " + " ".join(
    f"{k}={getattr(ag, k)}" for k in
    ("chip_targeting", "energy_spread", "counter_source", "chip_wall_defer")
    if hasattr(ag, k)))

import cg.game as game  # noqa: E402


class Trivial:
    def __call__(self, obs):
        if obs.get("select") is None:
            return deck
        return list(range(obs["select"]["minCount"]))


opp = Trivial()
obs, _ = game.battle_start(deck, deck)
pool = [600.0, 600.0]
selects, lat_max = 0, 0.0
try:
    while True:
        st = obs.get("current")
        if st is not None and st["result"] != -1:
            print(f"RESULT={st['result']} turns={st['turn']} selects={selects} "
                  f"pool_left={pool[0]:.1f}s lat_max={lat_max:.3f}s")
            break
        who = st["yourIndex"]
        obs["remainingOverageTime"] = pool[who]
        t0 = time.perf_counter()
        choice = env["agent"](obs) if who == 0 else opp(obs)
        dt = time.perf_counter() - t0
        pool[who] -= dt
        if who == 0:
            lat_max = max(lat_max, dt)
        obs = game.battle_select([int(c) for c in choice])
        selects += 1
        assert pool[0] > 0, "agent exhausted its time pool"
        if selects > 6000:
            raise SystemExit("game did not terminate")
finally:
    game.battle_finish()
