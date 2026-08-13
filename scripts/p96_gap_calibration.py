#!/usr/bin/env python
"""E34 — is the rollout estimator's GAP BETWEEN ARMS calibrated? Randomized overrule.

Pre-registered in `docs/experiments/E34-gap-calibration.md`.

**The one thing this measures.** E33 showed the rollout's LEVEL is right for the
clone's own pick. Nothing this project spends is a level: E17's +0.0139, the
oracle's stage-2 argmax and E19's cashed overrule are all **differences between
two arms**, and a shared position-specific error cancels in the level while
surviving in the difference.

So: at one sampled MAIN decision per game, roll out arm A (the agent's own pick)
and arm B (the net's rank-2 option) on shared worlds, record the predicted gap
`d = p_B - p_A`, then **flip a fair coin and actually play B half the time**.
The `z=1` and `z=0` games are exchangeable, so their difference in mean outcome
is the realized gap -- measured, not assumed, on the same positions that
produced the prediction.

    miscal = [mean(y | z=1) - mean(y | z=0)]  -  mean(d)

⛔ **Arm B is fixed by `net.scores` BEFORE any rollout is read.** Selecting the
arm the rollout preferred would make the realized gap fall short of the
predicted one under perfect calibration -- the winner's curse, the same family
of error as E33's finite-R calibration curve.

⚠ **The primary is a difference of MEANS, not a regression on `d`.** `d` is a
finite-R estimate, so regressing realized on predicted attenuates toward zero
whatever the truth. The mean is immune to R.

⚡ **No clustering correction is needed and that is by design**: exactly one
randomized decision per game makes the game the independent unit, so §8bw's
lesson is paid by construction rather than by correction. `harness.play_game`
takes **no seed** (verified by inspection), so the two arms cannot share a
world -- this is an unpaired comparison and is sized as one.

    python -X utf8 scripts/p96_gap_calibration.py --games 60 --rollouts 20 --q 0.1
    python -X utf8 scripts/p96_gap_calibration.py --analyze "out/logs/e34/*.jsonl"
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

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


class Randomizer:
    """Wraps an agent. Fires at most ONCE per game, at random.

    🔴 The per-game fire cap is the whole design, not a budget saving. Every
    value the rollout produces is Q^pi(s,a) -- the worth of deviating ONCE with
    the clone continuing -- and E18 deviated 3.32 times a game, which is the
    confound E19 cell A was built to remove. One fire per game satisfies the
    one-step assumption exactly, so the predicted and realized quantities are
    the same estimand.
    """

    def __init__(self, inner, decklist, rollouts: int, q: float, seed: int,
                 net, min_opts: int = 2, min_turn: int = 2):
        self.inner = inner
        # 🔴 THE NET IS PASSED IN, NEVER FETCHED FROM THE SINGLETON, and the
        # first version of this file got it wrong. `pnet.get()` returns the
        # repo default `agents/sa/policy_net.npz` (#ce97c732) -- the **v2**
        # clone, three generations behind the `out/policy_v5_s2.npz`
        # (#75ebeabd) the agent actually plays. p82 documented this trap in
        # writing ("PIN THE NET IN THE SCRIPT") after E17's C0 control read
        # 67.3% on the default against 99.8% on the right one, and this cell
        # walked into it anyway: arm B was the OLD net's rank-2 and the
        # rollouts continued with the OLD net, so the measured gap belonged to
        # an estimator nobody ships.
        self.net = net
        self.tag = str(seed)
        self.decklist = list(decklist)
        self.rollouts = int(rollouts)
        self.q = float(q)
        self.min_opts = int(min_opts)
        self.min_turn = int(min_turn)
        self.rng = random.Random(seed)               # worlds
        # ⚠ The coin runs on its OWN stream. Sharing the world stream would tie
        # the treatment assignment to how many rollouts happened to be drawn,
        # which is a covariate -- and control 1 exists to catch exactly that.
        self.coin = random.Random(seed * 1_000_003 + 7)
        self.rows: list[dict] = []
        self.game = -1
        self.fired = False
        self.stats = {"calls": 0, "eligible": 0, "attempts": 0, "fired": 0,
                      "rollouts": 0, "rollout_none": 0, "thin": 0,
                      "kept_a": 0, "played_b": 0, "identity_ok": 0,
                      "double_fire": 0, "no_scores": 0}

    def new_game(self) -> None:
        self.game += 1
        self.fired = False

    # -- one clone-vs-clone playout to terminal -> {0, 0.5, 1} or None -----
    def _rollout(self, obs: dict, world, first: list[int], me: int, net):
        sel = obs["select"]
        root = None
        try:
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
            # `fs.end()` frees the whole arena; safe here for oracle.py's
            # documented reason -- the real game runs through ptcg.env, not fs.
            if root is not None:
                try:
                    fs.end()
                except Exception:
                    pass

    def __call__(self, obs: dict) -> list[int]:
        self.stats["calls"] += 1
        picked = self.inner(obs)
        try:
            out = self._maybe(obs, picked)
        except Exception:
            return picked
        if out is None:
            return picked
        return out

    def _maybe(self, obs: dict, picked) -> list[int] | None:
        sel = obs.get("select") or {}
        cur = obs.get("current") or {}
        if self.fired:
            return None
        if sel.get("context") != MAIN or not obs.get("search_begin_input"):
            return None
        if cur.get("result", -1) != -1 or cur.get("turn", 0) < self.min_turn:
            return None
        if not (sel.get("minCount", 1) <= 1 <= sel.get("maxCount", 1)):
            return None
        nopt = len(sel.get("option") or [])
        if nopt < self.min_opts or not picked or len(picked) != 1:
            return None
        self.stats["eligible"] += 1
        if self.coin.random() >= self.q:
            return None
        self.stats["attempts"] += 1

        net = self.net
        if net is None:
            return None
        try:
            sc = np.asarray(net.scores(obs), dtype=float)
        except Exception:
            self.stats["no_scores"] += 1
            return None
        a = int(picked[0])
        rest = [int(i) for i in np.argsort(-sc) if int(i) != a]
        if not rest:
            return None
        b = rest[0]

        me = cur["yourIndex"]
        base = self.rng.randrange(1 << 30)
        tot = [0.0, 0.0]
        cnt = [0, 0]
        for k in range(self.rollouts):
            for j, arm in enumerate((a, b)):
                # ⚠ Same seed for both arms at replicate k => the SAME
                # determinized world. This is the only pairing available
                # (§8bw C2: the engine draws its own shuffles beyond the
                # world), and it is worth rho~0.53 on the gap.
                w = determinize(obs, self.decklist, [], random.Random(base + k))
                v = self._rollout(obs, w, [arm], me, net)
                self.stats["rollouts"] += 1
                if v is None:
                    self.stats["rollout_none"] += 1
                else:
                    tot[j] += v
                    cnt[j] += 1
        need = max(1, self.rollouts // 3)
        if min(cnt) < need:
            # 🔴 A thin position is skipped WITHOUT consuming the game's one
            # fire. Consuming it would make the fired set depend on rollout
            # failure, which is a covariate we did not randomize over.
            self.stats["thin"] += 1
            return None

        pa, pb = tot[0] / cnt[0], tot[1] / cnt[1]
        z = 1 if self.coin.random() < 0.5 else 0
        if self.fired:                       # cannot happen; counted anyway
            self.stats["double_fire"] += 1
        self.fired = True
        self.stats["fired"] += 1
        self.rows.append({
            "game": f"{self.tag}:{self.game}", "seat": me,
            "turn": cur.get("turn"), "nopt": nopt,
            "arm_a": a, "arm_b": b,
            "phat_a": pa, "phat_b": pb, "d": pb - pa,
            "margin": float(sc[a] - sc[b]),
            "r_a": cnt[0], "r_b": cnt[1],
            "z": z, "y": None,
        })
        if z:
            self.stats["played_b"] += 1
            return [b]
        self.stats["kept_a"] += 1
        self.stats["identity_ok"] += 1
        return list(picked)

    def close_game(self, result: int) -> None:
        key = f"{self.tag}:{self.game}"
        for row in self.rows:
            if row["game"] == key and row["y"] is None:
                seat = row["seat"]
                row["y"] = (0.5 if result == 2
                            else (1.0 if result == seat else 0.0))


# --- analysis ----------------------------------------------------------------

def _point(rows: list[dict]) -> dict:
    y1 = [r["y"] for r in rows if r["z"] == 1]
    y0 = [r["y"] for r in rows if r["z"] == 0]
    d = [r["d"] for r in rows]
    if not y1 or not y0:
        return {}
    realized = sum(y1) / len(y1) - sum(y0) / len(y0)
    predicted = sum(d) / len(d)
    return {"realized": realized, "predicted": predicted,
            "miscal": realized - predicted, "n1": len(y1), "n0": len(y0)}


def _adjusted(rows: list[dict]) -> float:
    """Realized gap adjusted for the PRE-randomization covariate m=(pa+pb)/2.

    ANCOVA on a covariate fixed before the coin is flipped: it cannot bias the
    treatment effect, and it buys ~10% on SE (E33: Var(p) against Var(y)).
    Reported, not relied on.
    """
    m = np.array([(r["phat_a"] + r["phat_b"]) / 2.0 for r in rows])
    z = np.array([float(r["z"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    X = np.column_stack([np.ones_like(z), z, m - m.mean()])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[1])
    except Exception:
        return float("nan")


def _auc(rows: list[dict], key: str = "phat_a") -> float:
    pos = [r[key] for r in rows if r["y"] == 1.0]
    neg = [r[key] for r in rows if r["y"] == 0.0]
    if not pos or not neg:
        return float("nan")
    w = t = 0
    for x in pos:
        for v in neg:
            if x > v:
                w += 1
            elif x == v:
                t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))


def _boot(rows, fn, B, seed):
    rng = random.Random(seed)
    n = len(rows)
    out = []
    for _ in range(B):
        s = [rows[rng.randrange(n)] for _ in range(n)]
        try:
            v = fn(s)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out.append(v)
        except Exception:
            pass
    out.sort()
    if len(out) < 20:
        return float("nan"), float("nan")
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def _balance(rows) -> list[str]:
    lines = []
    for key in ("phat_a", "turn", "nopt", "margin"):
        a = [float(r[key]) for r in rows if r["z"] == 1]
        b = [float(r[key]) for r in rows if r["z"] == 0]
        if len(a) < 2 or len(b) < 2:
            continue
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
        vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
        se = math.sqrt(va / len(a) + vb / len(b))
        t = (ma - mb) / se if se > 0 else 0.0
        flag = "  🔴 IMBALANCED" if abs(t) > 3.0 else ""
        lines.append(f"  {key:>8}  z=1 {ma:>8.3f}   z=0 {mb:>8.3f}   "
                     f"t {t:+.2f}{flag}")
    return lines


def analyze(rows: list[dict], boot: int = 4000, seed: int = 0) -> int:
    rows = [r for r in rows if r.get("y") is not None]
    if len(rows) < 10:
        sys.exit(f"only {len(rows)} completed rows")
    p = _point(rows)
    if not p:
        sys.exit("one treatment arm is empty")
    n = len(rows)

    lo_m, hi_m = _boot(rows, lambda s: (_point(s) or {}).get("miscal"), boot, seed)
    lo_r, hi_r = _boot(rows, lambda s: (_point(s) or {}).get("realized"), boot, seed + 1)
    lo_d, hi_d = _boot(rows, lambda s: (_point(s) or {}).get("predicted"), boot, seed + 2)

    print(f"games (fired)      {n}   z=1 {p['n1']}  z=0 {p['n0']}   "
          f"P(z=1) {p['n1'] / n:.3f}")
    print(f"PREDICTED gap      {p['predicted']:+.4f}  "
          f"[{lo_d:+.4f}, {hi_d:+.4f}]   = mean(p_B - p_A)")
    print(f"REALIZED  gap      {p['realized']:+.4f}  "
          f"[{lo_r:+.4f}, {hi_r:+.4f}]   = mean(y|z=1) - mean(y|z=0)")
    print(f"MISCAL             {p['miscal']:+.4f}  "
          f"[{lo_m:+.4f}, {hi_m:+.4f}]   <- PRIMARY")

    adj = _adjusted(rows)
    lo_a, hi_a = _boot(rows, _adjusted, boot, seed + 3)
    print(f"  realized (adj)   {adj:+.4f}  [{lo_a:+.4f}, {hi_a:+.4f}]   "
          f"(ANCOVA on pre-randomization m)")

    if not (lo_d <= 0.0 <= hi_d):
        sh = p["realized"] / p["predicted"]
        lo_s, hi_s = _boot(
            rows, lambda s: ((_point(s) or {}).get("realized", 0.0)
                             / (_point(s) or {}).get("predicted", 1e-9)),
            boot, seed + 4)
        print(f"  shrinkage        {sh:+.3f}  [{lo_s:+.3f}, {hi_s:+.3f}]   "
              f"(1.0 = calibrated, 0.0 = illusory)")

    print("\ncontrol 1 -- balance across the coin:")
    for line in _balance(rows):
        print(line)
    a0 = _auc([r for r in rows if r["z"] == 0])
    print(f"\ncontrol 4 -- AUC(p_A, y) on the z=0 arm   {a0:.4f}   "
          f"(must clearly exceed 0.5)")

    print("\nfired-position profile (descriptive):")
    for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 99)):
        sub = [r for r in rows if lo <= (r["turn"] or 0) < hi]
        if sub:
            print(f"  turn [{lo:>2},{hi:>2})   n {len(sub):>5}   "
                  f"mean d {sum(x['d'] for x in sub) / len(sub):+.4f}")
    win = [r for r in rows if 3 <= r["nopt"] <= 5]
    if len(win) >= 10:
        q = _point(win)
        lo_w, hi_w = _boot(win, lambda s: (_point(s) or {}).get("miscal"),
                           boot, seed + 5)
        print(f"\nrestricted to oracle's window 3<=nopt<=5   n={len(win)}")
        print(f"  predicted {q['predicted']:+.4f}   realized {q['realized']:+.4f}"
              f"   MISCAL {q['miscal']:+.4f}  [{lo_w:+.4f}, {hi_w:+.4f}]")

    print()
    if lo_d <= 0.0 <= hi_d:
        print("⛔ VOID: the predicted gap is not bounded away from 0, so there "
              "is nothing\n   to check the calibration OF. Narrow the "
              "population (see sizing).")
    elif lo_m <= 0.0 <= hi_m and not (lo_m <= -p["predicted"] <= hi_m):
        print("✅ GAPS CALIBRATED: miscal contains 0 and EXCLUDES the "
              "fully-illusory\n   alternative. E19's null has no evaluator "
              "explanation left.")
    elif not (lo_m <= 0.0 <= hi_m):
        print("🔴 INFLATED (or deflated): miscal excludes 0. The currency is "
              "mispriced by\n   the shrinkage factor above; E17 / §8bx / §8bw "
              "/ §2.7 reprice.")
    else:
        print("⚠ UNRESOLVED: the interval contains both 0 and the "
              "fully-illusory\n   alternative. Report the width; do not "
              "narrate a null.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--q", type=float, default=0.1,
                    help="per-eligible-decision fire probability")
    ap.add_argument("--min-opts", type=int, default=2)
    ap.add_argument("--min-turn", type=int, default=2)
    ap.add_argument("--net", default="out/policy_v5_s2.npz")
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", default=None)
    args = ap.parse_args()

    if args.analyze:
        rows: list[dict] = []
        for f in sorted(glob.glob(args.analyze)):
            with open(f, encoding="utf-8") as fh:
                rows += [json.loads(x) for x in fh if x.strip()]
        return analyze(rows, args.boot, args.seed)

    sys.path.insert(0, str(ROOT / "decks"))
    deck_mod = __import__(args.deck)
    deck = []
    for cid, k in deck_mod.DECKLIST.items():
        deck += [cid] * k

    a = PolicyAgent(list(deck), args.net)
    b = PolicyAgent(list(deck), args.net)
    # 🔴 CONTROL 5, and it is an assert rather than a print because the failure
    # it guards is silent: the rollout net must be the OBJECT the seat plays
    # with, not a same-named reload and never the singleton.
    if a.net is None:
        sys.exit(f"🔴 PolicyAgent did not load {args.net}")
    fp = hashlib.md5(Path(args.net).read_bytes()).hexdigest()[:8]
    print(f"net pinned to {args.net} #{fp} (rollouts use the seat's own "
          f"net object)")
    gp = pnet.get()
    if gp is not None and gp is not a.net:
        print(f"  ⚠ the pnet.get() singleton is a DIFFERENT net and is "
              f"deliberately unused here")
    rz = Randomizer(a, deck, args.rollouts, args.q, args.seed, a.net,
                    args.min_opts, args.min_turn)
    if rz.net is not a.net:
        sys.exit("🔴 rollout net is not the seat's net")

    out = Path(args.out) if args.out else ROOT / "out" / "logs" / "e34" / \
        f"e34_s{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # 🔴 Rows are streamed and flushed per game, not written at the end. A
    # multi-hour shard that buffers its result until the last line loses
    # everything to any late failure, and E33 already lost a shard's control
    # counters to a file-lifetime accident.
    with out.open("w", encoding="utf-8") as fh:
        written = 0
        for i in range(args.games):
            rz.new_game()
            if i % 2 == 0:
                res = harness.play_game(rz, b, list(deck), list(deck))
            else:
                res = harness.play_game(b, rz, list(deck), list(deck))
            rz.close_game(res.winner)
            while written < len(rz.rows):
                r = rz.rows[written]
                if r["y"] is None:
                    break
                fh.write(json.dumps(r) + "\n")
                written += 1
            fh.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{args.games} games, {written} fired, "
                      f"{time.time() - t0:.0f}s", flush=True)

    el = time.time() - t0
    s = rz.stats
    print(f"\nplayed {args.games} games in {el:.0f}s "
          f"({el / max(1, args.games):.2f}s/game)")
    print(f"eligible={s['eligible']} attempts={s['attempts']} "
          f"fired={s['fired']} ({s['fired'] / max(1, args.games):.0%} of games) "
          f"thin={s['thin']} no_scores={s['no_scores']}")
    print(f"rollouts={s['rollouts']} none={s['rollout_none']} "
          f"({100.0 * s['rollout_none'] / max(1, s['rollouts']):.1f}%)  "
          f"<- control 3, VOID above 10%")
    print(f"control 2: played_b={s['played_b']} kept_a={s['kept_a']} "
          f"(sum {s['played_b'] + s['kept_a']} == fired {s['fired']}: "
          f"{'OK' if s['played_b'] + s['kept_a'] == s['fired'] else 'BROKEN'}) "
          f"double_fire={s['double_fire']}")

    print(f"wrote {written} rows to {out}\n")
    return analyze([r for r in rz.rows if r["y"] is not None],
                   args.boot, args.seed)


if __name__ == "__main__":
    sys.exit(main())
