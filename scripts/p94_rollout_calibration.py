#!/usr/bin/env python
"""E33 — is the rollout estimator CALIBRATED against realized outcomes?

Pre-registered in `docs/experiments/E33-rollout-calibration.md`.

**The one thing this measures.** A PURE OBSERVER wraps the shipped clone. At
sampled MAIN decisions it takes the clone's own pick, rolls that pick out to
terminal `R` times over freshly determinized worlds, records the mean as `p̂`,
and **returns the clone's pick unchanged**. The real game then finishes with the
same clone on both seats and the realized outcome `y` is recorded from the
deciding seat's view.

⇒ **Rollout and reality differ in exactly ONE thing: the world.** Same net, same
position, same continuation policy on both seats. Any gap between `p̂` and `y`
is determinization bias — the cause §8ca named as *"named but not isolated"*.

⚠ **Primary statistic is the MEAN bias, clustered by game — not a calibration
slope.** `p̂` is a finite-`R` estimate, so regressing `y` on it suffers
regression dilution and flattens even under perfect calibration. `E[p̂] = p` and
`E[y] = p` make the mean immune to that, whatever `R` is. Clustering is §8bw's
lesson paid once already: decisions inside a game share an outcome, and a naive
SE was 4.1x too tight there.

    python -X utf8 scripts/p94_rollout_calibration.py --games 40 --rollouts 6 --every 4
    python -X utf8 scripts/p94_rollout_calibration.py --analyze "out/logs/e33/*.jsonl"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from ptcg.env import harness  # noqa: E402
from sa import fastsearch as fs  # noqa: E402
from sa import policynet as pnet  # noqa: E402
from sa.bcagent import PolicyAgent  # noqa: E402
from sa.worlds import determinize  # noqa: E402

MAIN = 0
ROLLOUT_CAP = 1500


class Observer:
    """Wraps an agent, measures, and returns the agent's answer untouched.

    🔴 The `returned_inner` counter is control 3 and is not decorative: an
    observer that silently altered one pick in a thousand would make these
    ordinary `bc`-vs-`bc` games into something else, and the calibration claim
    rests on them being ordinary.
    """

    def __init__(self, inner, decklist, rollouts: int, every: int,
                 seed: int, net=None, min_opts: int = 2):
        self.inner = inner
        # 🔴 THE ROLLOUT NET IS PASSED IN. The first version called
        # `pnet.get()`, which returns `agents/sa/policy_net.npz` (#ce97c732) --
        # the **v2** clone, three generations behind the
        # `out/policy_v5_s2.npz` (#75ebeabd) the seats actually play. That made
        # this cell's central claim ("rollout and reality differ in exactly ONE
        # thing: the world") FALSE: they also differed in the continuation
        # policy. p82 had already written the warning ("PIN THE NET IN THE
        # SCRIPT") after E17's C0 read 67.3% on the default vs 99.8% pinned.
        self.net = net
        # 🔴 The game key is namespaced by seed because shards are pooled by
        # `--analyze` and every shard counts its games from 0. Without this,
        # clustering would merge shard A's game 0 with shard B's game 0 --
        # collapsing independent clusters and reporting an SE that is too
        # TIGHT, which is the §8bw failure in the direction that flatters us.
        self.tag = str(seed)
        self.decklist = list(decklist)
        self.rollouts = rollouts
        self.every = every
        self.min_opts = min_opts
        self.rng = random.Random(seed)
        self.rows: list[dict] = []
        self.game = -1
        self._seen = 0
        self.stats = {"calls": 0, "returned_inner": 0, "measured": 0,
                      "rollouts": 0, "rollout_none": 0, "eligible": 0}

    def new_game(self) -> None:
        self.game += 1
        self._seen = 0

    def _rollout(self, obs: dict, first: list[int], me: int, net):
        """One clone-vs-clone playout to terminal. -> {0, 0.5, 1} or None."""
        sel = obs["select"]
        root = None
        try:
            world = determinize(obs, self.decklist, [], self.rng)
            root, o = fs.begin(
                obs["search_begin_input"],
                [] if sel.get("deck") is not None else world.my_deck,
                world.my_prize, world.opp_deck, world.opp_prize,
                world.opp_hand, world.opp_active)
            sid, o = fs.step(root, first)
            steps = 1
            while steps < ROLLOUT_CAP:
                cur, s2 = o.get("current"), o.get("select")
                if cur is None or s2 is None:
                    return None
                if cur["result"] != -1:
                    r = cur["result"]
                    return 0.5 if r == 2 else (1.0 if r == me else 0.0)
                sid, o = fs.step(sid, net.choose(o))
                steps += 1
            return None
        except Exception:
            return None
        finally:
            # `fs.end()` frees the whole arena. Safe here for oracle.py's
            # documented reason: the real game is driven through ptcg.env, not
            # fs, so no other search is live across this call.
            if root is not None:
                try:
                    fs.end()
                except Exception:
                    pass

    def __call__(self, obs: dict) -> list[int]:
        self.stats["calls"] += 1
        picked = self.inner(obs)
        self.stats["returned_inner"] += 1
        try:
            sel = obs.get("select") or {}
            cur = obs.get("current") or {}
            if sel.get("context") != MAIN or not obs.get("search_begin_input"):
                return picked
            if cur.get("result", -1) != -1:
                return picked
            if not (sel.get("minCount", 1) <= 1 <= sel.get("maxCount", 1)):
                return picked
            if len(sel.get("option") or []) < self.min_opts:
                return picked
            if not picked or len(picked) != 1:
                return picked
            self.stats["eligible"] += 1
            self._seen += 1
            if self._seen % self.every:
                return picked

            net = self.net
            if net is None:
                return picked
            me = cur["yourIndex"]
            vals = []
            for _ in range(self.rollouts):
                v = self._rollout(obs, list(picked), me, net)
                self.stats["rollouts"] += 1
                if v is None:
                    self.stats["rollout_none"] += 1
                else:
                    vals.append(v)
            if not vals:
                return picked
            self.stats["measured"] += 1
            self.rows.append({
                "game": f"{self.tag}:{self.game}", "seat": me, "turn": cur.get("turn"),
                "nopt": len(sel.get("option") or []),
                "phat": sum(vals) / len(vals), "r": len(vals), "y": None,
            })
        except Exception:
            pass
        return picked

    def close_game(self, result: int) -> None:
        """Stamp the realized outcome onto this game's rows."""
        for row in self.rows:
            if row["game"] == f"{self.tag}:{self.game}" and row["y"] is None:
                seat = row["seat"]
                row["y"] = (0.5 if result == 2
                            else (1.0 if result == seat else 0.0))


# --- analysis ----------------------------------------------------------------

def _clustered(rows: list[dict]) -> tuple[float, float, int, int]:
    """mean(phat - y) with SE clustered by game.

    Cluster-robust SE over G game-clusters: sd of per-game SUMS of the
    residual, scaled by the total count. Reduces to the iid SE when every
    cluster holds one observation.
    """
    by_game: dict[int, list[float]] = {}
    for r in rows:
        by_game.setdefault(r["game"], []).append(r["phat"] - r["y"])
    n = sum(len(v) for v in by_game.values())
    g = len(by_game)
    if n == 0 or g < 2:
        return 0.0, float("nan"), n, g
    mean = sum(sum(v) for v in by_game.values()) / n
    # sum over clusters of (sum of residual deviations)^2
    ss = sum((sum(x - mean for x in v)) ** 2 for v in by_game.values())
    var = ss * g / max(1, (g - 1)) / (n * n)
    return mean, math.sqrt(max(var, 0.0)), n, g


def _auc(rows: list[dict]) -> float:
    """AUC of phat against a WIN outcome; draws dropped."""
    pos = [r["phat"] for r in rows if r["y"] == 1.0]
    neg = [r["phat"] for r in rows if r["y"] == 0.0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def analyze(rows: list[dict]) -> int:
    rows = [r for r in rows if r.get("y") is not None]
    if not rows:
        sys.exit("no completed rows")
    bias, se, n, g = _clustered(rows)
    lo, hi = bias - 1.96 * se, bias + 1.96 * se
    mp = sum(r["phat"] for r in rows) / n
    my = sum(r["y"] for r in rows) / n
    auc = _auc(rows)

    print(f"decisions      {n}   over {g} games")
    print(f"mean phat      {mp:.4f}")
    print(f"mean y         {my:.4f}")
    print(f"BIAS           {bias:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"(clustered SE {se:.4f})")
    # the naive SE, printed only to show how much clustering matters (§8bw)
    resid = [r["phat"] - r["y"] for r in rows]
    mu = sum(resid) / n
    naive = math.sqrt(sum((x - mu) ** 2 for x in resid) / (n * (n - 1)))
    print(f"  naive SE     {naive:.4f}   (clustering widens it "
          f"{se / naive:.2f}x)")
    print(f"AUC(phat, y)   {auc:.4f}   <- control 1, must clearly exceed 0.5")

    mid = [r for r in rows if 0.15 <= r["phat"] <= 0.85]
    if mid:
        b2, s2, n2, g2 = _clustered(mid)
        print(f"\nrestricted to phat in [0.15, 0.85]   n={n2} over {g2} games")
        print(f"BIAS           {b2:+.4f}  [{b2 - 1.96 * s2:+.4f}, "
              f"{b2 + 1.96 * s2:+.4f}]")

    print("\ncalibration curve (secondary -- attenuated by finite R):")
    print(f"  {'bin':>12}  {'n':>5}  {'mean phat':>9}  {'mean y':>7}  {'gap':>7}")
    edges = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
    for a, b in zip(edges, edges[1:]):
        sub = [r for r in rows if a <= r["phat"] < b]
        if not sub:
            continue
        sp = sum(r["phat"] for r in sub) / len(sub)
        sy = sum(r["y"] for r in sub) / len(sub)
        print(f"  [{a:.2f},{b:.2f})  {len(sub):>5}  {sp:>9.3f}  {sy:>7.3f}  "
              f"{sp - sy:>+7.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--rollouts", type=int, default=6)
    ap.add_argument("--every", type=int, default=4,
                    help="measure every Nth eligible MAIN decision")
    ap.add_argument("--net", default="out/policy_v5_s2.npz")
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", default=None,
                    help="glob of jsonl row files; skips playing")
    args = ap.parse_args()

    if args.analyze:
        rows: list[dict] = []
        for f in sorted(glob.glob(args.analyze)):
            with open(f, encoding="utf-8") as fh:
                rows += [json.loads(x) for x in fh if x.strip()]
        return analyze(rows)

    sys.path.insert(0, str(ROOT / "decks"))
    deck_mod = __import__(args.deck)
    deck = []
    for cid, k in deck_mod.DECKLIST.items():
        deck += [cid] * k

    a = PolicyAgent(list(deck), args.net)
    b = PolicyAgent(list(deck), args.net)
    if a.net is None:
        sys.exit(f"🔴 PolicyAgent did not load {args.net}")
    import hashlib
    fp = hashlib.md5(Path(args.net).read_bytes()).hexdigest()[:8]
    print(f"net pinned to {args.net} #{fp} (rollouts use the seat's own net)")
    obs_a = Observer(a, deck, args.rollouts, args.every, args.seed, a.net)
    if obs_a.net is not a.net:
        sys.exit("🔴 rollout net is not the seat's net")

    out = Path(args.out) if args.out else ROOT / "out" / "logs" / "e33" / \
        f"e33_s{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = []
    # 🔴 Rows stream and flush per game rather than being written at the end.
    # A shard that buffers an hour of work into its last line loses all of it
    # to any late failure, and leaves a progress monitor unable to tell a
    # working run from a hung one.
    with out.open("w", encoding="utf-8") as fh:
        written = 0
        for i in range(args.games):
            obs_a.new_game()
            # Seats alternate so the observer measures from both sides; the
            # observed agent is byte-identical to the unobserved one either way.
            if i % 2 == 0:
                res = harness.play_game(obs_a, b, list(deck), list(deck))
            else:
                res = harness.play_game(b, obs_a, list(deck), list(deck))
            obs_a.close_game(res.winner)
            results.append(res.winner)
            while written < len(obs_a.rows):
                r = obs_a.rows[written]
                if r["y"] is None:
                    break
                fh.write(json.dumps(r) + "\n")
                written += 1
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{args.games} games, "
                      f"{written} decisions, {time.time() - t0:.0f}s",
                      flush=True)

    el = time.time() - t0
    s = obs_a.stats
    print(f"\nplayed {args.games} games in {el:.0f}s "
          f"({el / max(1, args.games):.2f}s/game)")
    print(f"eligible={s['eligible']} measured={s['measured']} "
          f"rollouts={s['rollouts']} none={s['rollout_none']} "
          f"({100.0 * s['rollout_none'] / max(1, s['rollouts']):.1f}%)")
    print(f"control 3: returned_inner={s['returned_inner']}/{s['calls']} "
          f"({'OK' if s['returned_inner'] == s['calls'] else 'ALTERED PLAY'})")

    print(f"wrote {written} rows to {out}")
    print()
    return analyze(obs_a.rows)


if __name__ == "__main__":
    sys.exit(main())
