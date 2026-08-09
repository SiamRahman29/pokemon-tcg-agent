"""WP-regret autopsy — score the losses with `evalfn` and rank the drops.

**Why this exists** (HANDOFF day-27 §3, proposal 1). Every seam this project has
mined was found by a **rate miner**: count how often we do X, count how often the
experts do X, subtract. That machinery is structurally blind to a blunder that
happens *once* — a frequency-1 mistake has no rate to compare. The user's named
seam, "a few games where 1 bad decision cost us games", is exactly that shape.

The instrument is `evalfn`, whose across-game discrimination was measured in §8l
(AUC **0.685** early / **0.901** late over 200 games). This script does not
assume that number transfers: **control C1 re-measures it on the corpus being
autopsied**, and if it collapses there is nothing below it worth reading.

**What is and is not attributable.** Consecutive decision records give a WP
delta. A pair inside one player's turn is caused by *that player's action*. A
pair spanning the handover carries their last action AND the whole opponent
turn, and is tagged `boundary` — reported, never attributed. ⛔ The attack is
almost always the last select of a turn, so **the cost of a bad attack lands in
`boundary`, not in the attributable stream.** Any conclusion here is about
decisions whose damage shows up before control passes.

**The control that makes the headline falsifiable.** A lost game *must* contain
a big drop — the trajectory ends at 0. So "our losses contain a −0.3 drop" is
not evidence of anything on its own. Three baselines are computed against the
identical instrument:
  * the same statistic in the games we **won** (the drop still exists there),
  * the **opponent's** attributable drops in the same games (they made decisions
    too, scored by the same `evalfn`, and in our 27 losses they won anyway),
  * the **concentration** share: how much of a lost game's total decline the
    single worst attributable decision carries. Diffuse bleed and a blunder are
    different diagnoses and this separates them.

    python -X utf8 scripts/p77_wp_regret.py --verify        # controls only
    python -X utf8 scripts/p77_wp_regret.py                 # full autopsy
    python -X utf8 scripts/p77_wp_regret.py --top 25        # deeper ranked list
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from p72_loss_autopsy import (  # noqa: E402
    CHIP_CONTEXTS, _hp_map, _nm, _pk, _records,
)
from sa.evalfn import evaluate  # noqa: E402
from sa.optfeat import option_features  # noqa: E402

CTX = {
    0: "MAIN", 1: "SETUP_ACTIVE", 2: "SETUP_BENCH", 3: "SWITCH",
    4: "TO_ACTIVE", 5: "TO_BENCH", 6: "TO_FIELD", 7: "TO_HAND", 8: "DISCARD",
    9: "TO_DECK", 10: "TO_DECK_BOTTOM", 11: "TO_PRIZE", 12: "NOT_MOVE",
    13: "DAMAGE_COUNTER", 14: "DAMAGE_COUNTER_ANY", 15: "DAMAGE",
    16: "REMOVE_DMG_COUNTER", 17: "HEAL", 18: "EVOLVES_FROM",
    19: "EVOLVES_TO", 20: "DEVOLVE", 21: "ATTACH_FROM", 22: "ATTACH_TO",
    23: "DETACH_FROM", 24: "LOOK", 25: "EFFECT_TARGET",
    26: "DISCARD_ENERGY_CARD", 27: "DISCARD_TOOL_CARD",
    28: "SWITCH_ENERGY_CARD", 29: "DISCARD_CARD_OR_ATTACHED", 30: "DISCARD_ENERGY",
    31: "TO_HAND_ENERGY", 32: "TO_DECK_ENERGY", 33: "SWITCH_ENERGY",
    34: "SKILL_ORDER", 35: "ATTACK", 36: "DISABLE_ATTACK", 37: "EVOLVE",
    38: "DRAW_COUNT", 39: "DAMAGE_COUNTER_COUNT",
    40: "REMOVE_DMG_CTR_COUNT", 41: "IS_FIRST", 42: "MULLIGAN", 43: "ACTIVATE",
    44: "FIRST_EFFECT", 45: "MORE_DEVOLVE", 46: "COIN_HEAD",
    47: "AFFECT_SPECIAL_COND", 48: "RECOVER_SPECIAL_COND",
}

# Turn buckets for calibration. `turn` in the observation counts player turns.
BUCKETS = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 99)]


def _bucket(turn: int) -> int:
    for b, (lo, hi) in enumerate(BUCKETS):
        if lo <= turn <= hi:
            return b
    return len(BUCKETS) - 1


def _turn(state: dict) -> int:
    t = state.get("turn")
    return int(round(t)) if isinstance(t, (int, float)) else -1


def _live(st: dict) -> bool:
    """Is `evalfn` even DEFINED on this state? Only from turn 1 on.

    🔴 It is not, during setup, and this was found the hard way. `evaluate`
    scores prizes as `6 - len(prize)` **taken**, so between the moment one
    player's prize pile is dealt and the moment the other's is, it reads a
    six-prize deficit and returns **-8.2 on a completely empty board**. Seven of
    the 27 losses showed an identical "-0.202 WP at turn 0" until this guard
    went in, and the same garbage was in the calibration corpus. `evalfn` is
    only ever called mid-game by the live agent (MAIN selects, turn >= 1), so
    nothing had exercised it here before.

    ⚠ The first version of this guard **also** demanded a non-empty active on
    both sides, and that was a selection bias, not a fix: the state right after
    a DAMAGE select has the defender's active EMPTY whenever we knocked it out,
    so it silently deleted **158 of our 177 damage deltas** (counted directly)
    — precisely the successful ones. Setup is a turn-0 phenomenon; the guard is
    a turn-0 guard and nothing more.
    """
    return _turn(st) >= 1


def _opt_card(obs: dict, opt: dict) -> int:
    """Card id an option names, via the NET'S OWN extractor (p76's lesson)."""
    try:
        return int(option_features(obs, opt)[1] or 0)
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------------------------- corpus
# ⚠ Replays run ~4 MB each and a single day directory is 1.7 GB, so nothing here
# holds more than one parsed replay at a time. Everything a later stage needs
# (including the human-readable description of a decision) is extracted during
# the streaming pass and the replay is dropped.
def stream_games(dirs: list[str], limit: int = 0):
    n_err = n = 0
    for d in dirs:
        for path in sorted((ROOT / d).glob("*.json")):
            if path.name == "manifest.json":
                continue
            if limit and n >= limit:
                return
            try:
                rep = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                n_err += 1
                continue
            n += 1
            yield path, rep
            del rep
    if n_err:
        print(f"  ({n_err} unreadable replays skipped)")


def our_seat(rep: dict, us: set[str]) -> int | None:
    names = (rep.get("info") or {}).get("TeamNames") or []
    seats = [i for i, n in enumerate(names) if n in us]
    return seats[0] if len(seats) == 1 else None


def outcome(rep: dict, seat: int) -> int | None:
    """+1 we won, -1 we lost, 0 draw, None unknown."""
    rw = rep.get("rewards") or [None, None]
    if rw[0] is None or rw[1] is None:
        return None
    if rw[seat] > rw[1 - seat]:
        return 1
    if rw[seat] < rw[1 - seat]:
        return -1
    return 0


def trajectory(rep: dict, me: int) -> list[dict]:
    """Every decision record, scored by `evalfn` from ONE fixed seat (`me`).

    Scoring from a fixed seat is what makes consecutive scores differenceable.
    The state dict carries the acting player's view, but `evalfn` reads only
    public quantities (hp, energy, prize COUNT, handCount, deckCount, status),
    so the same seat's score is well defined on either side's record.
    """
    out = []
    for k, rec in enumerate(_records(rep)):
        st = rec["obs"]["current"]
        if st.get("result", -1) != -1 or not _live(st):
            continue
        out.append({
            "idx": k,
            "obs": rec["obs"], "state": st, "sel": rec["sel"],
            "picked": rec["picked"], "actor": st.get("yourIndex"),
            "turn": _turn(st), "ev": float(evaluate(st, me)),
            "ctx": rec["sel"].get("context"),
            "n_opts": len(rec["sel"].get("option") or []),
        })
    return out


# ---------------------------------------------------------------- calibration
def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC. labels in {0,1}."""
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks over ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1, n0 = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 100,
                  lam: float = 1e-3) -> tuple[float, float]:
    """1-D logistic -> (a, b) with p = sigmoid(a*x + b).

    ⚠ The plain IRLS this started as **diverged** on the turn 6-8 bucket
    (a = 74,173, every state pinned to 0 or 1) and quietly destroyed the WP
    scale in that bucket before the reliability table caught it. Three things
    fix it and all three are load-bearing: standardise `x` first, damp the
    Newton step by backtracking on the penalised log-likelihood so an iterate
    can never make the fit worse, and carry a small ridge so a near-separable
    bucket cannot run the slope to infinity.
    """
    mu, sd = float(x.mean()), float(x.std()) or 1.0
    xs = (x - mu) / sd

    def nll(c0: float, c1: float) -> float:
        z = np.clip(c0 * xs + c1, -30, 30)
        return float(np.mean(np.logaddexp(0.0, z) - y * z)
                     + lam * (c0 * c0 + c1 * c1))

    a, b = 1.0, 0.0
    f = nll(a, b)
    for _ in range(iters):
        z = np.clip(a * xs + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1 - p), 1e-8, None)
        r = (y - p) / len(y)
        g = np.array([np.dot(r, xs) - 2 * lam * a, r.sum() - 2 * lam * b])
        h = np.array([[np.dot(w * xs, xs), np.dot(w, xs)],
                      [np.dot(w, xs), w.sum()]]) / len(y) + 2 * lam * np.eye(2)
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        for _ in range(30):                       # backtracking line search
            f2 = nll(a + t * step[0], b + t * step[1])
            if f2 <= f:
                break
            t *= 0.5
        else:
            break                                 # no improving step: converged
        a, b, prev = a + t * step[0], b + t * step[1], f
        f = f2
        if prev - f < 1e-12:
            break
    # back to the original x scale
    return float(a / sd), float(b - a * mu / sd)


class Calibrator:
    """eval score -> P(win), fitted per turn bucket on an INDEPENDENT corpus."""

    def __init__(self) -> None:
        self.par: dict[int, tuple[float, float]] = {}
        self.auc: dict[int, tuple[float, int]] = {}
        self.degen: dict[int, float] = {}

    def fit(self, rows: list[tuple[int, float, int]]) -> None:
        by_b: dict[int, list[tuple[float, int]]] = defaultdict(list)
        for b, ev, y in rows:
            by_b[b].append((ev, y))
        for b, xs in sorted(by_b.items()):
            x = np.array([e for e, _ in xs], dtype=np.float64)
            y = np.array([v for _, v in xs], dtype=np.float64)
            a, bb = _fit_logistic(x, y)
            self.par[b] = (a, bb)
            self.auc[b] = (_auc(x, y), len(x))
            # Divergence guard: the first version of this fit ran a slope to
            # 74,000 and pinned every state to 0/1. A fit that saturates its
            # own training states is not a calibration.
            p = 1.0 / (1.0 + np.exp(-np.clip(a * x + bb, -30, 30)))
            self.degen[b] = float(np.mean((p < 1e-3) | (p > 1 - 1e-3)))

    def wp(self, ev: float, turn: int) -> float:
        a, b = self.par.get(_bucket(turn), (0.35, 0.0))
        return float(1.0 / (1.0 + np.exp(-np.clip(a * ev + b, -30, 30))))


def calib_rows(dirs: list[str], limit: int = 0
               ) -> tuple[list[tuple[int, float, int]], int]:
    """(bucket, eval, won) for every decision record of every seat."""
    rows: list[tuple[int, float, int]] = []
    n = 0
    for _, rep in stream_games(dirs, limit):
        rw = rep.get("rewards") or [None, None]
        if rw[0] is None or rw[1] is None or rw[0] == rw[1]:
            continue
        for rec in _records(rep):
            st = rec["obs"]["current"]
            if st.get("result", -1) != -1 or not _live(st):
                continue
            seat = st.get("yourIndex")
            if seat not in (0, 1):
                continue
            rows.append((_bucket(_turn(st)), float(evaluate(st, seat)),
                         1 if rw[seat] > rw[1 - seat] else 0))
        n += 1
    return rows, n


# ------------------------------------------------------------------- controls
def controls(ours: list[dict], cal: Calibrator, cal_n: int) -> None:
    print("\n" + "=" * 74)
    print("=== CONTROLS — nothing below is readable if these fail ===")
    print("=" * 74)

    print(f"\nC1. Does `evalfn` separate winners from losers AT ALL here?")
    print(f"    Calibration corpus: {cal_n} games, fitted per turn bucket.")
    print(f"    {'turn bucket':<14}{'n states':>10}{'AUC':>9}{'a':>9}{'b':>9}"
          f"{'saturated':>11}")
    for b, (lo, hi) in enumerate(BUCKETS):
        if b not in cal.auc:
            continue
        auc, n = cal.auc[b]
        a, bb = cal.par[b]
        lab = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
        flag = " 🔴" if cal.degen[b] > 0.05 else ""
        print(f"    {lab:<14}{n:>10}{auc:>9.3f}{a:>9.3f}{bb:>9.3f}"
              f"{cal.degen[b]:>10.1%}{flag}")
    early = [cal.auc[b][0] for b in cal.auc if b <= 1]
    late = [cal.auc[b][0] for b in cal.auc if b >= 3]
    if early and late:
        print(f"    early(0-5) {np.mean(early):.3f} vs late(9+) {np.mean(late):.3f}"
              f"   [§8l on self-play read 0.685 / 0.901]")

    # C2: does the calibration transfer to OUR games?
    xs, ys = [], []
    for g in ours:
        if g["result"] == 0:
            continue
        y = 1 if g["result"] > 0 else 0
        for w in g["wps"]:
            xs.append(w)
            ys.append(y)
    xs_a, ys_a = np.array(xs), np.array(ys)
    brier = float(np.mean((xs_a - ys_a) ** 2))
    auc_ours = _auc(xs_a, ys_a)
    print(f"\nC2. Does that calibration TRANSFER to the corpus we are autopsying?")
    print(f"    {len(xs_a)} states over {len([g for g in ours if g['result']])} games:"
          f"  AUC {auc_ours:.3f}   Brier {brier:.4f}"
          f"   (base-rate Brier {np.mean((ys_a.mean()-ys_a)**2):.4f})")
    print(f"    reliability:  {'predicted':>10}{'actual':>9}{'n':>8}")
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for lo, hi in zip(edges, edges[1:]):
        m = (xs_a >= lo) & (xs_a < hi if hi < 1.0 else xs_a <= 1.0)
        if m.sum() < 20:
            continue
        print(f"      [{lo:.1f},{hi:.1f})  {xs_a[m].mean():>10.3f}"
              f"{ys_a[m].mean():>9.3f}{int(m.sum()):>8}")

    # C3: the noise floor of a single-decision delta
    deltas = np.array([d["dwp"] for g in ours for d in g["deltas"]
                       if d["attributable"]])
    print(f"\nC3. The NOISE FLOOR of one attributable delta ({len(deltas)} of them,"
          f" both seats, all games)")
    qs = [1, 5, 25, 50, 75, 95, 99]
    vals = np.percentile(deltas, qs)
    print("    percentile " + "".join(f"{q:>9}%" for q in qs))
    print("    dWP        " + "".join(f"{v:>+10.3f}" for v in vals))
    print(f"    mean {deltas.mean():+.4f}   sd {deltas.std():.4f}"
          f"   |  a 'blunder' must clear this scale, not merely be negative.")


# -------------------------------------------------------------------- autopsy
def _describe(rec: dict) -> str:
    """Human-readable rendering of one decision. Built during the streaming
    pass, because the observation it needs is far too big to keep."""
    obs, sel = rec["obs"], rec["sel"]
    opts = sel.get("option") or []
    pick = rec["picked"]
    ctx = CTX.get(sel.get("context"), f"ctx{sel.get('context')}")
    chosen = ", ".join(_nm(_opt_card(obs, opts[j])) for j in pick
                       if 0 <= j < len(opts)) or "?"
    alts = [_nm(_opt_card(obs, o)) for k, o in enumerate(opts)
            if k not in pick][:4]
    return (f"{ctx}({len(opts)} opts) chose {chosen}"
            + (f"  | alts: {', '.join(alts)}" if alts else ""))


def _dominated_ko(a: dict, b: dict) -> str | None:
    """§8bm's dominated test, re-run here as a DISCRIMINATOR on this instrument.

    Returns a description when the same damage, placed on a different legal
    option, would have knocked something out while what we hit survived. This
    is not a new measurement — `p72` owns it — it is the known-positive that
    tells us whether a WP-regret ranking can SEE an error of omission.
    """
    if a["sel"].get("context") not in CHIP_CONTEXTS or len(a["picked"]) != 1:
        return None
    opts = a["sel"].get("option") or []
    if len(opts) < 2:
        return None
    s0, s1 = _hp_map(a["state"]), _hp_map(b["state"])
    chosen = _pk(a["state"], opts[a["picked"][0]])
    if not chosen:
        return None
    ser = chosen["serial"]
    if ser not in s0 or ser not in s1:
        return None
    dmg = s0[ser] - s1[ser]
    if dmg <= 0 or s1[ser] <= 0:          # no damage, or the target did die
        return None
    alts = [pk for j, o in enumerate(opts) if j != a["picked"][0]
            for pk in [_pk(a["state"], o)]
            if pk and pk["serial"] != ser and 0 < pk["hp"] <= dmg]
    if not alts:
        return None
    return (f"dmg={dmg} hit {_nm(chosen['id'])}@{chosen['hp']}hp (survived)"
            f"  |  KO available: "
            + ", ".join(f"{_nm(p['id'])}@{p['hp']}hp" for p in alts))


def build(dirs: list[str], us: set[str], cal: Calibrator) -> list[dict]:
    """Per game: trajectory, WP, and the attributed delta stream."""
    out = []
    for path, rep in stream_games(dirs):
        me = our_seat(rep, us)
        if me is None:
            continue
        res = outcome(rep, me)
        if res is None:
            continue
        traj = trajectory(rep, me)
        if len(traj) < 4:
            continue
        for r in traj:
            r["wp"] = cal.wp(r["ev"], r["turn"])
        deltas = []
        for i, (a, b) in enumerate(zip(traj, traj[1:])):
            if b["idx"] != a["idx"] + 1:
                continue        # a dropped record sits between them: not a delta
            same = (a["actor"] == b["actor"] and a["turn"] == b["turn"])
            # ⚠ BOTH ends of a delta are scored with the FIRST record's turn
            # bucket. Scoring each end with its own bucket differences two
            # different logistic fits, which manufactured a spurious -0.202 at
            # every turn-0 handover in the first run of this script. Only
            # `boundary` pairs can cross a bucket, so this changes nothing in
            # the attributable stream and repairs the boundary one.
            wp0 = a["wp"]
            wp1 = cal.wp(b["ev"], a["turn"])
            deltas.append({
                "i": i, "actor": a["actor"], "turn": a["turn"],
                "ctx": a["ctx"], "n_opts": a["n_opts"],
                "wp0": wp0, "wp1": wp1,
                "dwp": wp1 - wp0,
                # signed from the ACTING player's point of view
                "dwp_actor": (wp1 - wp0) * (1 if a["actor"] == me else -1),
                "attributable": same,
                "what": _describe(a) if a["actor"] == me else "",
                "dom_ko": (_dominated_ko(a, b) if a["actor"] == me else None),
            })
        out.append({"gid": path.stem, "me": me, "result": res,
                    "opp": ((rep.get("info") or {}).get("TeamNames")
                            or ["?", "?"])[1 - me],
                    "wps": [r["wp"] for r in traj], "deltas": deltas})
    return out


def report(ours: list[dict], top: int, thresh: float) -> None:
    losses = [g for g in ours if g["result"] < 0]
    wins = [g for g in ours if g["result"] > 0]
    print("\n" + "=" * 74)
    print(f"=== THE AUTOPSY — {len(losses)} losses, {len(wins)} wins ===")
    print("=" * 74)

    def worst(g, actor_is_me=True):
        cand = [d for d in g["deltas"] if d["attributable"]
                and ((d["actor"] == g["me"]) == actor_is_me)]
        return min(cand, key=lambda d: d["dwp_actor"]) if cand else None

    # 1. per-loss table
    print("\n1. WORST ATTRIBUTABLE DECISION IN EACH LOSS")
    print("   (dWP is from OUR seat; `boundary` pairs — where the opponent's whole")
    print("    turn is folded in, including the reply to our attack — are excluded)")
    print(f"   {'game':<11}{'opponent':<22}{'turn':>5}{'worstdWP':>10}"
          f"{'wp before':>11}  what we chose")
    rows = []
    for g in losses:
        w = worst(g)
        if w is None:
            continue
        rows.append((w["dwp_actor"], g, w))
    for dwp, g, w in sorted(rows, key=lambda t: t[0]):
        print(f"   {g['gid']:<11}{g['opp'][:21]:<22}{w['turn']:>5}"
              f"{dwp:>+10.3f}{w['wp0']:>11.3f}  {w['what']}")

    # 2. ranked list across all losses
    allc = [(d["dwp_actor"], g, d) for g in losses for d in g["deltas"]
            if d["attributable"] and d["actor"] == g["me"]]
    allc.sort(key=lambda t: t[0])
    print(f"\n2. THE {top} WORST SINGLE DECISIONS ACROSS ALL {len(losses)} LOSSES")
    for dwp, g, d in allc[:top]:
        print(f"   {dwp:>+7.3f}  {g['gid']} t{d['turn']:<3}"
              f" wp {d['wp0']:.2f}->{d['wp1']:.2f}  {d['what']}")

    # 2b. the punish — the drop that lands AFTER our last decision of a turn
    print(f"\n2b. THE PUNISH — worst BOUNDARY swings in the losses (NOT attributable:")
    print( "    each folds our last action of the turn together with the opponent's")
    print( "    entire reply, and this is where the cost of a bad ATTACK lands)")
    bnd = [(d["dwp_actor"], g, d) for g in losses for d in g["deltas"]
           if not d["attributable"] and d["actor"] == g["me"]]
    bnd.sort(key=lambda t: t[0])
    for dwp, g, d in bnd[:top]:
        print(f"   {dwp:>+7.3f}  {g['gid']} t{d['turn']:<3}"
              f" wp {d['wp0']:.2f}->{d['wp1']:.2f}  last was {d['what']}")

    # 2c. where does the WP actually go? the full accounting
    print("\n2c. WHERE THE LOST WP GOES — decomposition of all negative flow")
    print(f"   {'stream':<44}{'losses':>12}{'wins':>12}")
    def flow(gs, pred):
        tot = sum(-min(0.0, d["dwp"]) for g in gs for d in g["deltas"])
        part = sum(-min(0.0, d["dwp"]) for g in gs for d in g["deltas"]
                   if pred(g, d))
        return part / max(tot, 1e-9)
    streams = (
        ("our own turn, attributable to our decision",
         lambda g, d: d["attributable"] and d["actor"] == g["me"]),
        ("boundary after OUR last decision (their reply)",
         lambda g, d: not d["attributable"] and d["actor"] == g["me"]),
        ("their own turn, attributable to their decision",
         lambda g, d: d["attributable"] and d["actor"] != g["me"]),
        ("boundary after THEIR last decision (our reply)",
         lambda g, d: not d["attributable"] and d["actor"] != g["me"]),
    )
    for lab, pred in streams:
        print(f"   {lab:<44}{flow(losses, pred):>11.1%}{flow(wins, pred):>12.1%}")
    print("   ⚠ Read this before anything above: a stream that carries almost none")
    print("     of the decline cannot hold the fix, however bad its worst event looks.")
    # ... and in absolute WP, because a share is not a size (rule 13)
    print("\n   In ABSOLUTE WP per game — this is the number the gate applies to:")
    print(f"   {'':<44}{'losses':>12}{'wins':>12}")
    for lab, pred in streams[:2]:
        v = [sum(-min(0.0, d["dwp"]) for d in g["deltas"] if pred(g, d))
             for g in losses]
        w = [sum(-min(0.0, d["dwp"]) for d in g["deltas"] if pred(g, d))
             for g in wins]
        print(f"   {lab:<44}{np.mean(v):>12.3f}{np.mean(w):>12.3f}")
    print("   ⇒ a policy that made every one of our within-turn decisions PERFECT")
    print("     could recover at most the first row, and only if every drop in it")
    print("     were avoidable — which the wins column says it is not.")

    # 3. THE CONTROL: is any of this specific to us, or to losing?
    print("\n3. THE CONTROLS — is a big drop a BLUNDER or just what losing looks like?")

    def stat(gs, mine):
        w = [worst(g, mine) for g in gs]
        w = [x for x in w if x]
        v = np.array([x["dwp_actor"] for x in w]) if w else np.array([0.0])
        return v

    a = stat(losses, True)
    b = stat(wins, True)
    c = stat(losses, False)   # the opponent, in the games they WON
    d_ = stat(wins, False)    # the opponent, in the games they LOST
    print(f"   {'stream':<40}{'n':>5}{'mean worst':>12}{'median':>9}{'min':>9}")
    for lab, v in (("US, in the 27 losses", a), ("US, in the 49 wins", b),
                   ("THEM, in the 27 games they won", c),
                   ("THEM, in the 49 games they lost", d_)):
        print(f"   {lab:<40}{len(v):>5}{v.mean():>+12.3f}"
              f"{np.median(v):>+9.3f}{v.min():>+9.3f}")
    print("   ⚠ If 'US in losses' ≈ 'THEM in the games they won', a big worst-drop")
    print("     is a property of the instrument and the game, not of our play.")

    # 4. concentration: one blunder, or a diffuse bleed?
    print("\n4. CONCENTRATION — does ONE decision carry the loss?")
    print(f"   {'stream':<28}{'n':>4}{'worst/our-neg':>15}{'worst/total':>13}")
    for lab, gs in (("losses", losses), ("wins", wins)):
        sh1, sh2 = [], []
        for g in gs:
            mine = [d for d in g["deltas"] if d["attributable"]
                    and d["actor"] == g["me"]]
            if not mine:
                continue
            neg = -sum(min(0.0, d["dwp_actor"]) for d in mine)
            tot = -sum(min(0.0, d["dwp_actor"]) for d in g["deltas"])
            w = -min(d["dwp_actor"] for d in mine)
            if neg > 1e-9:
                sh1.append(w / neg)
            if tot > 1e-9:
                sh2.append(w / tot)
        print(f"   {lab:<28}{len(sh1):>4}{np.mean(sh1):>15.1%}{np.mean(sh2):>13.1%}")
    print("   (worst/our-neg = share of all OUR negative attributable dWP carried by")
    print("    the single worst one; worst/total folds in the boundary pairs too.)")

    # 5. sizing gate
    print(f"\n5. SIZING — candidates at |dWP| >= {thresh:.2f}, against the 0.5/game gate")
    n_games = len(ours)
    for lab, gs in (("all games", ours), ("losses only", losses),
                    ("wins only", wins)):
        hits = [d for g in gs for d in g["deltas"]
                if d["attributable"] and d["actor"] == g["me"]
                and d["dwp_actor"] <= -thresh]
        print(f"   {lab:<16}{len(hits):>5} events over {len(gs):>3} games"
              f"  = {len(hits)/max(len(gs),1):.3f}/game")
    hits = [d for g in ours for d in g["deltas"]
            if d["attributable"] and d["actor"] == g["me"]
            and d["dwp_actor"] <= -thresh]
    if hits:
        print("\n   by context:")
        by = Counter(CTX.get(d["ctx"], f"ctx{d['ctx']}") for d in hits)
        for k, v in by.most_common(12):
            print(f"     {k:<24}{v:>5}  ({v/max(n_games,1):.3f}/game)")
        forced = sum(1 for d in hits if d["n_opts"] < 2)
        print(f"   of which FORCED (<2 options, no decision was made): {forced}"
              f" of {len(hits)}")

    # 7. THE DISCRIMINATOR: can this instrument see an error we KNOW is there?
    print("\n7. DISCRIMINATOR — does the WP ranking SEE §8bm's dominated events?")
    print("   `p72` found plays where the same damage would have KO'd a different")
    print("   legal target and what we hit survived. Those are known errors of")
    print("   OMISSION. If the ranking cannot surface them, its null is a null")
    print("   about self-inflicted damage only.")
    print(f"   {'game':<11}{'res':>5}{'turn':>5}{'dWP':>9}{'rank in game':>14}"
          f"  the dominated event")
    seen = 0
    for g in ours:
        mine = sorted([d for d in g["deltas"]
                       if d["attributable"] and d["actor"] == g["me"]],
                      key=lambda d: d["dwp_actor"])
        for d in g["deltas"]:
            if not d.get("dom_ko"):
                continue
            seen += 1
            rank = (mine.index(d) + 1) if d in mine else -1
            rk = f"{rank} of {len(mine)}" if rank > 0 else "not attributable"
            print(f"   {g['gid']:<11}{'LOSS' if g['result'] < 0 else 'win':>5}"
                  f"{d['turn']:>5}{d['dwp_actor']:>+9.3f}{rk:>14}"
                  f"  {d['dom_ko']}")
    if not seen:
        print("   (none found — check `p72` first, it owns this test)")

    # 6. mean attributable dWP by context — the seam view
    print("\n6. MEAN ATTRIBUTABLE dWP BY CONTEXT (our decisions, all games)")
    print("   ⛔ DO NOT read a negative row as 'this context is our leak'. The delta")
    print("      after a select is dominated by the MECHANICAL consequence of that")
    print("      context — a DAMAGE select spends our attack, so `evalfn` re-reads")
    print("      the threat terms whatever we targeted. Only differences BETWEEN")
    print("      options at the same state measure a choice, and that needs the")
    print("      option-level counterfactual this script does not compute.")
    agg: dict[str, list[float]] = defaultdict(list)
    for g in ours:
        for d in g["deltas"]:
            if d["attributable"] and d["actor"] == g["me"] and d["n_opts"] >= 2:
                agg[CTX.get(d["ctx"], f"ctx{d['ctx']}")].append(d["dwp_actor"])
    print(f"   {'context':<24}{'n':>7}{'/game':>8}{'mean dWP':>11}{'p05':>9}")
    for k, v in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        if len(v) < 10:
            continue
        arr = np.array(v)
        print(f"   {k:<24}{len(arr):>7}{len(arr)/max(n_games,1):>8.2f}"
              f"{arr.mean():>+11.4f}{np.percentile(arr,5):>+9.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["replays/submission_v5_s2"])
    ap.add_argument("--calib", nargs="+",
                    default=["replays/2026-08-05", "replays/2026-08-06",
                             "replays/2026-08-07"])
    ap.add_argument("--calib-games", type=int, default=600,
                    help="cap on calibration games (0 = all)")
    ap.add_argument("--us", action="append", default=["Scio"])
    ap.add_argument("--verify", action="store_true", help="controls only")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--thresh", type=float, default=0.20,
                    help="|dWP| that counts as a candidate blunder")
    args = ap.parse_args()

    print(f"fitting the INDEPENDENT calibration corpus "
          f"({', '.join(args.calib)}, cap {args.calib_games or 'all'}) ...",
          flush=True)
    rows, n_cal = calib_rows(args.calib, args.calib_games)
    print(f"  {n_cal} games, {len(rows)} states", flush=True)
    cal = Calibrator()
    cal.fit(rows)

    print(f"scoring the corpus under autopsy ({', '.join(args.dir)}) ...",
          flush=True)
    ours = build(args.dir, set(args.us), cal)
    print(f"  {len(ours)} games with our seat and a result", flush=True)

    controls(ours, cal, n_cal)
    if args.verify:
        return 0
    report(ours, args.top, args.thresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
