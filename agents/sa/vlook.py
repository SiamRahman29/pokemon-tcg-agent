"""E20 — ONE-PLY lookahead scored by a learned value function.

**This is `oracle.py` with the evaluator swapped, and that swap is the whole
hypothesis.** The clock forked a position and rolled ~100 engine steps to a
terminal 0/1; E19 (§8ca) measured that estimate as biased and bought nothing.
Here the fork advances **exactly one step** and the resulting state is scored by
`valuenet`, trained on 20,000 self-play outcomes (`scripts/train_value.py`).

Why each prior death does not transfer:

  * §2's search died of variance — a terminal rollout returns 0/1, so a mean
    over 12 determinizations carries SE ~ 0.14 and the max over ~9 arms sits
    0.21-0.28 above truth by chance. **V returns a calibrated scalar. That
    variance term does not exist here.**
  * B4/E5 died with `evalfn`, a handcrafted scorer. **V is learned from
    realized outcomes.**
  * The clock died (most likely) of strategy fusion compounding over ~100
    simulated steps inside a determinized world. **One step cannot compound.**

⚠ Determinization is still present — `fs.begin` needs a full world — so fusion
is *reduced*, not eliminated. Averaging over `worlds` sampled worlds is what
marginalizes it, and W is frozen at 4 by the pre-registration.

🔴 **THE SEAT TRAP, and it is the one bug here that would look like a result.**
After `fs.step` the successor's `current["yourIndex"]` is frequently the
OPPONENT, because control passes. V must be read from **our** seat, captured
before the step. Reading `yourIndex` instead yields a V that is systematically
inverted exactly when control changes hands — an agent that plays to lose,
confidently, with every health counter green.
"""
from __future__ import annotations

import random
import time
from collections import Counter

import numpy as np

from . import fastsearch as fs
from . import policynet as pnet
from . import valuenet as vnet
from .worlds import determinize

MAIN = 0
RESERVE_S = 45.0

STATS: Counter = Counter()
ERR: dict = {}


def health_line() -> str:
    s = STATS
    ev = s["evals"]
    out = (f"[vlook] fired={s['fired']} overruled={s['overruled']} "
           f"evals={ev} skip_shape={s['skip_shape']} "
           f"skip_trigger={s['skip_trigger']} skip_noclock={s['skip_noclock']} "
           f"skip_thin={s['skip_thin']} skip_novnet={s['skip_novnet']} "
           f"errors={s['errors']} secs={s['secs_x1000'] / 1000.0:.1f}")
    if ev:
        out += f" ms/eval={s['secs_x1000'] / ev:.1f}"
    if "err" in ERR:
        out += f" first_error={ERR['err']!r}"
    o = s["overruled"]
    if o:
        # Same reasoning as oracle._change_line: a null is uninterpretable
        # unless we know whether V agreed with the net (nothing to change) or
        # overruled it constantly and still gained nothing.
        out += (f" | CHANGED {o}/{s['fired']} ({o / max(s['fired'], 1):.0%})"
                f" net-margin mean {s['margin_x1000'] / 1000.0 / o:+.2f}")
        buckets = [f"{b}:{s[f'netmargin_{b}']}" for b in
                   ("lt0p5", "lt1p5", "lt3", "ge3") if s.get(f"netmargin_{b}")]
        if buckets:
            out += " (" + " ".join(buckets) + ")"
        out += f" v-gap mean {s['gap_x1000'] / 1000.0 / o:+.3f}"
    return out


def reset_stats() -> None:
    STATS.clear()
    ERR.clear()


class ValueLookahead:
    """One ply into the real engine, scored by V. Returns None on any doubt."""

    def __init__(self, decklist: list[int], net=None, worlds: int = 4,
                 max_opts: int = 12, min_opts: int = 2, tau: float = 0.0,
                 decision_cap_s: float = 5.0, reserve_s: float = RESERVE_S,
                 min_turn: int = 2, seed: int = 0, vnet_path: str | None = None):
        self.decklist = list(decklist)
        self.net = net
        # Pinned per INSTANCE, not via the module singleton, so two agents in
        # one process can hold different value nets (rule 4's head-to-head).
        self.V = vnet.load(vnet_path) if vnet_path else None
        self.worlds = int(worlds)
        self.max_opts = int(max_opts)
        self.min_opts = int(min_opts)
        self.tau = float(tau)
        self.decision_cap_s = float(decision_cap_s)
        self.reserve_s = float(reserve_s)
        self.min_turn = int(min_turn)
        self.rng = random.Random(seed)

    def new_game(self) -> None:
        return None

    # -- the fork, ONE step, then V -------------------------------------
    def _evaluate(self, obs: dict, world, first: list[int], me: int, V):
        """-> V(successor) from `me`'s seat, or None.

        Frees with `fs.end()`, never `fs.release()`. §N.6's defect 2: `release`
        reclaims ONE search id while a fork creates one per step, which leaked
        1.68 GB in 8 minutes and probably caused an intermittent 6.9% error
        rate through allocation failure.
        """
        root = None
        try:
            sel = obs["select"]
            root, o = fs.begin(
                obs["search_begin_input"],
                [] if sel.get("deck") is not None else world.my_deck,
                world.my_prize, world.opp_deck, world.opp_prize,
                world.opp_hand, world.opp_active)
            _sid, o = fs.step(root, first)
            cur = o.get("current")
            if cur is None:
                return None
            r = cur.get("result", -1)
            if r != -1:
                # One ply ended the game: that is ground truth, not an estimate.
                return 0.5 if r == 2 else (1.0 if r == me else 0.0)
            return V.win_prob(cur, me)          # 🔴 `me`, never cur["yourIndex"]
        except Exception as e:
            STATS["errors"] += 1
            if "err" not in ERR:
                ERR["err"] = f"{type(e).__name__}: {e}"[:200]
            return None
        finally:
            STATS["evals"] += 1
            if root is not None:
                try:
                    fs.end()
                except Exception:
                    pass

    def choose(self, obs: dict, picked: list[int]) -> list[int] | None:
        t0 = time.perf_counter()
        try:
            sel = obs.get("select") or {}
            cur = obs.get("current") or {}
            if sel.get("context") != MAIN or not obs.get("search_begin_input"):
                STATS["skip_shape"] += 1
                return None
            if not (sel.get("minCount", 1) <= 1 <= sel.get("maxCount", 1)):
                STATS["skip_shape"] += 1
                return None
            if cur.get("result", -1) != -1 or cur.get("turn", 0) < self.min_turn:
                STATS["skip_shape"] += 1
                return None
            if not picked or len(picked) != 1:
                STATS["skip_shape"] += 1
                return None
            n = len(sel.get("option") or [])
            STATS[f"nopt_{min(n, 12):02d}"] += 1
            # ⚠ NOT a value trigger. E20 pre-registers "no trigger"; this is a
            # shape bound only -- fewer than 2 options is not a decision, and
            # the upper bound caps worst-case cost, not expected value.
            if not (self.min_opts <= n <= self.max_opts):
                STATS["skip_trigger"] += 1
                return None

            V = self.V or vnet.get()
            if V is None:
                STATS["skip_novnet"] += 1       # a missing npz must not read as a null
                return None
            net = self.net or pnet.get()
            if net is None:
                STATS["skip_shape"] += 1
                return None

            rem = float(obs.get("remainingOverageTime", 600.0))
            budget = min(self.decision_cap_s, rem - self.reserve_s)
            if budget <= 0.2:
                STATS["skip_noclock"] += 1
                return None
            deadline = t0 + budget

            me = cur["yourIndex"]               # OUR seat, captured before any step
            arms = list(range(n))               # every option; no arm selection
            base = self.rng.randrange(1 << 30)

            tot = [0.0] * n
            cnt = [0] * n
            for k in range(self.worlds):
                if time.perf_counter() > deadline:
                    break
                # A SHARED world per replicate — the only pairing available
                # (§8bw C2: the engine draws its own shuffles, so common random
                # numbers do not exist here). Worth rho ~ 0.53.
                seed = base + 100_003 + k
                for j in arms:
                    w = determinize(obs, self.decklist, [], random.Random(seed))
                    v = self._evaluate(obs, w, [j], me, V)
                    if v is not None:
                        tot[j] += v
                        cnt[j] += 1
            if min(cnt) < 1:
                STATS["skip_thin"] += 1
                return None

            means = [tot[j] / cnt[j] for j in arms]
            a0 = int(picked[0])
            best = int(max(arms, key=lambda j: means[j]))
            gap = means[best] - means[a0]
            STATS["fired"] += 1
            if best == a0 or gap <= self.tau:
                return None
            STATS["overruled"] += 1
            try:
                sc = np.asarray(net.scores(obs), dtype=float)
                margin = float(sc[a0] - sc[best])
                STATS["margin_x1000"] += int(margin * 1000)
                b = ("lt0p5" if margin < 0.5 else "lt1p5" if margin < 1.5
                     else "lt3" if margin < 3.0 else "ge3")
                STATS[f"netmargin_{b}"] += 1
            except Exception:
                pass
            STATS["gap_x1000"] += int(gap * 1000)
            return [best]
        except Exception as e:
            STATS["errors"] += 1
            if "err" not in ERR:
                ERR["err"] = f"choose {type(e).__name__}: {e}"[:200]
            return None
        finally:
            STATS["secs_x1000"] += int((time.perf_counter() - t0) * 1000)
