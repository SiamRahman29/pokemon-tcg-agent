"""E26: substitute a DIFFERENT POLICY's pick, or a rate- and rank-matched
random one, so that coherence is the only thing that differs between the arms.

**The question** (docs/experiments/E26-coherence-at-matched-rate.md, §N.3).
E25 closed "override the clone with a better *evaluator*": the evaluator's
confidence and its distance from the clone are the same variable, so the ranking
signal is the damage signal (§8cg). A different *policy* is not an evaluator --
it is on-distribution for itself and its deviations are chosen in sequence by a
net trained to make them. Whether that matters is H1 vs H2, and it has been the
project's named open question since day 28.

**The design.** Both arms run this same wrapper and pay the same second forward
pass -- the `sym1` / `flip0` discipline, which is what made §8bu's null readable:

* `xnet=<path>`  -- TREATMENT: play that net's pick.
* `xrnd<p>`      -- CONTROL: with probability `p`, play a random option drawn at
                    a rank sampled from `xrank`, else play our own pick.

⚠ **Rank matching is the part E25 wished it had.** §8cf recorded that ~20% of
V's apparent edge might have been net-margin rather than selection, because the
control deviated DEEPER than the treatment. Here the control's rank histogram is
the treatment's *measured* one, so depth is matched by construction and the
residual is direction alone.

⚠ **Only single-pick decisions are eligible.** "Rank" is meaningless for a
multi-select, and a treatment that replaces a whole pick list while the control
replaces one element would not be matched. Both arms fall through identically
on `want > 1`, and the counters say how often that happened.

⚠ `x_fired` is the denominator; `x_diff` is the treatment. A firing whose pick
the base net would have made anyway is NOT a treatment -- E24's lesson (§8ce),
where `fired` overstated the treated set by ~7%.
"""
from __future__ import annotations

import random

import numpy as np


def parse_rank_hist(spec: str) -> list[float] | None:
    """`"1:0.62,2:0.21,3:0.10"` -> a normalised weight per rank (index 0 unused).

    Ranks are 1-based because rank 0 is our own pick, i.e. no deviation at all.
    """
    if not spec:
        return None
    w: dict[int, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(":")
        r = int(k)
        if r < 1:
            raise ValueError(f"rank {r} < 1: rank 0 is our own pick, not a deviation")
        w[r] = w.get(r, 0.0) + float(v)
    if not w:
        return None
    hi = max(w)
    out = [0.0] * (hi + 1)
    for r, v in w.items():
        out[r] = v
    tot = sum(out)
    if tot <= 0:
        raise ValueError(f"rank histogram sums to {tot}")
    return [v / tot for v in out]


# Every live Substitute, so `health_line` can print the rank histogram the way
# oracle/vlook print theirs. The arena builds both seats in ONE process, so this
# is a list rather than a singleton -- a singleton would silently report only
# whichever agent was constructed last.
LIVE: list["Substitute"] = []


def health_line() -> str:
    return " | ".join(s.line() for s in LIVE) if LIVE else ""


class Substitute:
    """Holds whichever arm's machinery is in use, plus the shared counters."""

    def __init__(self, xnet=None, rand_rate: float = 0.0,
                 rank_hist: list[float] | None = None, seed: int = 26):
        LIVE.append(self)
        self.xnet = xnet
        self.rand_rate = float(rand_rate or 0.0)
        self.rank_hist = rank_hist
        self._rng = random.Random(seed)
        # rank of the played option under OUR net's ordering; index 0 = our own
        # pick, so `ranks[0]` counts the firings where nothing changed. This
        # histogram is the object cell B is matched to.
        self.ranks = [0] * 24
        self.rank_over = 0        # deviation deeper than the histogram is long
        self.clipped = 0          # sampled rank >= option count, clipped down

    # -- reporting -------------------------------------------------------
    def hist_spec(self) -> str:
        """The `xrank=` string cell B should be run with (deviations only)."""
        tot = sum(self.ranks[1:]) + self.rank_over
        if tot <= 0:
            return ""
        parts = [f"{r}:{self.ranks[r] / tot:.4f}"
                 for r in range(1, len(self.ranks)) if self.ranks[r]]
        return ",".join(parts)

    def line(self) -> str:
        tot = sum(self.ranks)
        dev = tot - self.ranks[0]
        mean_rank = (sum(r * c for r, c in enumerate(self.ranks)) / dev
                     if dev else 0.0)
        return (f"x_ranks={self.ranks[:8]} over={self.rank_over} "
                f"clipped={self.clipped} mean_dev_rank={mean_rank:.2f} "
                f"xrank={self.hist_spec()}")

    # -- the two arms ----------------------------------------------------
    def apply(self, base_net, obs: dict, picked: list[int],
              want: int, stats: dict) -> list[int]:
        """Return the pick this arm plays. Never raises: on any failure the
        base pick stands, because an experiment must not forfeit a live
        episode (the fail-soft rule the shipped agent is built on)."""
        options = (obs.get("select") or {}).get("option") or []
        if want != 1 or len(options) < 2 or len(picked) != 1:
            stats["x_skip"] += 1
            return picked
        stats["x_fired"] += 1
        try:
            sc = np.asarray(base_net.scores(obs), dtype=np.float64)
            order = list(np.argsort(-sc))
            rank_of = {int(o): r for r, o in enumerate(order)}

            if self.xnet is not None:
                xp = self.xnet.choose(obs)
                if len(xp) != 1 or not 0 <= int(xp[0]) < len(options):
                    return picked
                new = int(xp[0])
            elif self.rand_rate > 0.0:
                if self._rng.random() >= self.rand_rate:
                    self.ranks[0] += 1
                    return picked
                new = int(order[self._sample_rank(len(order))])
            else:
                return picked

            r = rank_of.get(new, 0)
            if r < len(self.ranks):
                self.ranks[r] += 1
            else:
                self.rank_over += 1
            if new != int(picked[0]):
                stats["x_diff"] += 1
            return [new]
        except Exception:  # noqa: BLE001
            stats["x_error"] += 1
            return picked

    def _sample_rank(self, n_opts: int) -> int:
        """A deviation rank >= 1, drawn from the treatment's histogram.

        ⚠ Clipping is COUNTED, not silent. If the treatment's histogram has
        mass at rank 5 and this decision offers 3 options, the draw lands on
        the deepest available option -- which makes the control marginally
        SHALLOWER than the treatment, in the direction that understates the
        control's cost. Stating the sign is the point of the counter.
        """
        h = self.rank_hist
        if not h:
            return self._rng.randrange(1, n_opts)
        r = self._rng.choices(range(len(h)), weights=h, k=1)[0]
        r = max(1, r)
        if r >= n_opts:
            self.clipped += 1
            r = n_opts - 1
        return r
