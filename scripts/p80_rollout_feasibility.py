"""Can we fork a REAL position out of a replay and score its options by rollout?

**Why this exists** (HANDOFF §N.4.0). Every evaluation instrument this project
owns is one of two things: a **conformity metric** (agreement with experts —
§8r proved that measures distance from the fitted mode, not skill) or a **weak
evaluator** (`evalfn`, AUC 0.667 early). Neither can answer the only question
left open by §8u: *"in THIS exact position, is their move better than ours?"*

A rollout instrument can. Fork a real position, play option A and option B from
it with the clone piloting **both** seats, and difference the win rates. No
corpus, no mode, no conformity. §N.4.0 flagged one blocking risk and this script
exists to resolve it before any design work:

    ⚠ `fastsearch.begin` has only ever been called on a `search_begin_input`
      captured **in the same process** by a live agent. Whether the engine
      accepts an sbi read out of a replay JSON is untested. If it does not,
      the whole probe is dead.

Controls, in the order they must pass — each one can kill the instrument:

  **C1 reconstruction fidelity.** The forked position must be the SAME position:
  identical option list (not merely the same count), same board on both sides,
  same turn, same acting seat. Anything less and every number downstream is
  measured on a different game.
  **C2 determinism.** Is a rollout reproducible given a fixed determinized
  world? This is what decides whether **common random numbers** — the variance
  reduction §N.4.0's design assumes — is available at all.
  **C3 resolution (the positive control that matters).** The clone's OWN
  top-ranked option, rolled out against its LAST-ranked option at the same
  position. If the instrument cannot see that gap it cannot see anything, and
  no sample size fixes it. ⚠ Note this is a much coarser contrast than §8bd's
  k-th vs (k+1)-th flip, which measured 0.494 and is *expected* to be a null —
  a near-tie band being indifferent says nothing about top-vs-worst.
  **C4 the wrong-deck NEGATIVE control.** `begin` takes the seat's hidden deck
  as an *argument*: it does not know, and cannot check, which 60 that seat is
  actually playing. Hand it the wrong decklist and it returns a plausible win
  rate rather than an error — rule 18's shape exactly. This control exists so
  the failure announces itself. ✅ **It is now defused rather than merely
  flagged:** `seat_decklist()` reads each seat's registered 60 out of the
  replay, so the fork is fed the deck that seat actually played. That matters
  more than it sounds — the population that looked safe by construction was
  not (only 18 of 50 `mirror_experts` seats run our exact 60).

Measurements that size the design if the controls pass: rollout cost by turn,
the paired-vs-unpaired sd (how much a shared world actually buys), the spread
of position win-probabilities (how many positions are resolvable at all), and
whether an EXPERT's seat reconstructs — which is the entire payoff, since it is
what lets us score *their* move against *ours* with no assumption that
agreement means skill.

    python -X utf8 scripts/p80_rollout_feasibility.py --verify   # C1+C2 only
    python -X utf8 scripts/p80_rollout_feasibility.py            # full probe
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
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

from sa import fastsearch as fs  # noqa: E402
from sa import policynet as pnet  # noqa: E402
from sa.worlds import determinize  # noqa: E402
from decks.grimmsnarl import DECKLIST  # noqa: E402

MAIN = 0
FLAT = [cid for cid, n in DECKLIST.items() for _ in range(n)]
OURS = "Scio"
ROLLOUT_CAP = 1500


# --------------------------------------------------------------------------
# position extraction
# --------------------------------------------------------------------------

def seat_decklist(rep: dict, seat: int) -> list[int] | None:
    """The 60 cards THAT SEAT actually registered, read out of the replay.

    🔴 This exists because C4 found the fork accepts any decklist silently, and
    then the "safe" population turned out not to be safe: over 25
    `mirror_experts` games, **only 18 of 50 seats run our exact 60** — the rest
    are 1–3 card variants of the same archetype. "Both seats are Grimmsnarl"
    is NOT "both seats are our 60", and determinizing a variant with our list
    would have mis-filled the hidden zones of 64% of expert seats without a
    single error.

    The registration action is a bare 60-int list at step 1. ✅ Positive
    control: on `submission_v5_s2` this returns `decks/grimmsnarl.py`'s list
    for our own seat **20/20**.
    """
    for step in (rep.get("steps") or [])[:3]:
        if seat >= len(step):
            continue
        a = step[seat].get("action")
        if isinstance(a, list) and len(a) == 60 and all(
                isinstance(x, int) for x in a):
            return list(a)
    return None


def our_seat(rep: dict) -> int | None:
    names = (rep.get("info") or {}).get("TeamNames") or []
    for i, n in enumerate(names):
        if n == OURS:
            return i
    return None


def positions(rep: dict, seat: int, min_turn: int = 2,
              min_opts: int = 3) -> list[tuple[int, dict]]:
    """Live MAIN decisions of `seat` that carry a usable sbi.

    Single-index picks only (`minCount <= 1 <= maxCount`), so an arm is one
    option and the A/B contrast is unambiguous.
    """
    out = []
    for i, step in enumerate(rep.get("steps") or []):
        if seat >= len(step):
            continue
        o = step[seat].get("observation") or {}
        cur, sel = o.get("current") or {}, o.get("select") or {}
        if not (o.get("search_begin_input") and cur and sel):
            continue
        if cur.get("result", -1) != -1 or cur.get("turn", 0) < min_turn:
            continue
        if sel.get("context") != MAIN:
            continue
        if not (sel.get("minCount", 1) <= 1 <= sel.get("maxCount", 1)):
            continue
        if len(sel.get("option") or []) < min_opts:
            continue
        out.append((i, o))
    return out


def load_games(d: Path, limit: int) -> list[tuple[Path, dict]]:
    files = sorted(p for p in d.glob("*.json") if p.name != "episodes_meta.json")
    out = []
    for f in files[:limit]:
        try:
            out.append((f, json.loads(f.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------
# the instrument
# --------------------------------------------------------------------------

def fork(o: dict, world) -> tuple[int, dict]:
    sel = o["select"]
    return fs.begin(o["search_begin_input"],
                    [] if sel.get("deck") is not None else world.my_deck,
                    world.my_prize, world.opp_deck, world.opp_prize,
                    world.opp_hand, world.opp_active)


def rollout(o: dict, world, first_pick: list[int] | None, me: int):
    """Clone pilots BOTH seats to a terminal state.

    -> (value in {0, 0.5, 1} from `me`'s view or None, steps, seconds)

    ⚠ The value is win probability **under clone-vs-clone continuation**, not
    game-theoretic value. That is the right question for "should our net have
    played their move" (a one-step deviation from our own policy) and the wrong
    one for "is this move good in the abstract". State it wherever it is used.
    """
    net = pnet.get()
    t0 = time.perf_counter()
    try:
        sid, obs = fork(o, world)
    except Exception:
        return None, 0, time.perf_counter() - t0
    steps = 0
    if first_pick is not None:
        try:
            sid, obs = fs.step(sid, first_pick)
        except Exception:
            fs.end()
            return None, 0, time.perf_counter() - t0
        steps = 1
    val = None
    while steps < ROLLOUT_CAP:
        cur, sel = obs.get("current"), obs.get("select")
        if cur is None or sel is None:
            break
        if cur["result"] != -1:
            r = cur["result"]
            val = 0.5 if r == 2 else (1.0 if r == me else 0.0)
            break
        try:
            sid, obs = fs.step(sid, net.choose(obs))
        except Exception:
            break
        steps += 1
    fs.end()
    return val, steps, time.perf_counter() - t0


def board_key(state: dict, who: int):
    pl = state["players"][who]

    def pk(p):
        if p is None:
            return None
        return (p["id"], p.get("damage"), len(p.get("energyCards") or []),
                len(p.get("tools") or []))
    return (tuple(pk(x) for x in pl["active"]),
            tuple(pk(x) for x in pl["bench"]),
            pl["deckCount"], pl["handCount"], len(pl["discard"]))


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------

def c1_fidelity(pos: list, n: int) -> bool:
    """The forked position must BE the position, option list included."""
    ok = bad = 0
    reasons: dict[str, int] = {}
    for k, (i, o, me, deck) in enumerate(pos[:n]):
        w = determinize(o, deck, [], random.Random(9000 + k))
        try:
            _sid, obs2 = fork(o, w)
        except Exception as e:
            bad += 1
            reasons[f"begin:{type(e).__name__}"] = \
                reasons.get(f"begin:{type(e).__name__}", 0) + 1
            continue
        cur, sel = o["current"], o["select"]
        c2, s2 = obs2.get("current") or {}, obs2.get("select") or {}
        checks = {
            "options": ([json.dumps(x, sort_keys=True) for x in sel["option"]]
                        == [json.dumps(x, sort_keys=True)
                            for x in (s2.get("option") or [])]),
            "context": s2.get("context") == sel.get("context"),
            "turn": c2.get("turn") == cur.get("turn"),
            "seat": c2.get("yourIndex") == cur.get("yourIndex"),
            "board_us": board_key(c2, cur["yourIndex"]) ==
            board_key(cur, cur["yourIndex"]),
            "board_opp": board_key(c2, 1 - cur["yourIndex"]) ==
            board_key(cur, 1 - cur["yourIndex"]),
        }
        fs.end()
        if all(checks.values()):
            ok += 1
        else:
            bad += 1
            for name, good in checks.items():
                if not good:
                    reasons[name] = reasons.get(name, 0) + 1
    tot = ok + bad
    rate = ok / tot if tot else 0.0
    print(f"[C1] forked position identical to the replay's: {ok}/{tot} "
          f"= {rate:.1%}")
    if reasons:
        print(f"     failures: {reasons}")
    if rate < 0.99:
        print("     🔴 RECONSTRUCTION IS BROKEN -- everything below is "
              "measured on a different game.")
        return False
    print("     ✅ an sbi captured in another process reconstructs exactly.")
    return True


def c4_wrong_deck(o: dict, me: int, n: int, deck: list[int]) -> None:
    """Negative control: hand the fork a decklist the seat is NOT playing.

    A tool that rejected it would make the instrument safe on any seat. It does
    not reject it — so the instrument is only valid where the seat's 60 is
    known, and every use of it must say which seat and which deck.
    """
    try:
        from decks.crustle_v1 import DECKLIST as WRONG
    except Exception:
        print("[C4] skipped (no crustle_v1 decklist)")
        return
    wrong = [c for c, k in WRONG.items() for _ in range(k)]
    out = {}
    for name, dl in (("correct", deck), ("wrong", wrong)):
        vals = []
        for k in range(n):
            w = determinize(o, dl, [], random.Random(500 + k))
            v, _s, _c = rollout(o, w, [0], me)
            if v is not None:
                vals.append(v)
        out[name] = (len(vals), statistics.fmean(vals) if vals else float("nan"))
    print(f"[C4] wrong-deck control: correct 60 -> {out['correct'][1]:.3f} "
          f"({out['correct'][0]}/{n} ok) · WRONG 60 -> {out['wrong'][1]:.3f} "
          f"({out['wrong'][0]}/{n} ok)")
    if out["wrong"][0] > 0:
        print("     🔴 THE FORK ACCEPTS A DECKLIST THE SEAT IS NOT PLAYING and "
              "returns a plausible number, not an error.")
        print("     ✅ defused: the 'correct' arm above is the seat's OWN "
              "registered 60, read from the replay by seat_decklist() — not "
              "our list assumed onto them. Any seat in a replay we hold is "
              "usable; a seat without a recovered decklist is skipped.")
    else:
        print("     ✅ the wrong decklist is rejected; any seat is safe.")


def c2_determinism(o: dict, me: int, repeats: int,
                   deck: list[int]) -> None:
    """Is a rollout reproducible given a fixed world? Decides whether CRN exists."""
    outs = []
    for _ in range(repeats):
        w = determinize(o, deck, [], random.Random(4242))
        v, st, _ = rollout(o, w, [0], me)
        outs.append((v, st))
    vals = sorted({v for v, _ in outs})
    steps = [s for _, s in outs]
    same = len(set(outs)) == 1
    print(f"[C2] same world, same pick, {repeats} repeats: "
          f"values={vals} steps={steps}")
    if same:
        print("     ✅ deterministic -> true common random numbers ARE available.")
    else:
        print("     🔴 NOT deterministic. The engine draws its own randomness "
              "(shuffles/coins) beyond the determinized world, so a shared "
              "world is the ONLY pairing available -- not CRN.")


# --------------------------------------------------------------------------
# main probe
# --------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    net = pnet.get()
    if net is None:
        print("🔴 no policy net loaded")
        return 1
    import hashlib
    npz = Path(pnet._PATH)
    fp = hashlib.md5(npz.read_bytes()).hexdigest()[:8]
    print(f"net: {npz}  #{fp}")

    games = load_games(ROOT / args.dump, args.games)
    print(f"games: {len(games)} from {args.dump}")

    pos: list = []
    per_game = []
    no_deck = 0
    for _f, rep in games:
        seat = our_seat(rep)
        if seat is None:
            continue
        deck = seat_decklist(rep, seat)
        if deck is None:
            no_deck += 1
            continue
        ps = positions(rep, seat)
        per_game.append(len(ps))
        for i, o in ps:
            pos.append((i, o, seat, deck))
    if no_deck:
        print(f"⚠ {no_deck} games skipped: no registered decklist recovered")
    print(f"live MAIN positions with >=3 single-pick options: {len(pos)} "
          f"({statistics.fmean(per_game):.1f}/game)" if per_game else "none")
    if not pos:
        return 1

    if not c1_fidelity(pos, args.fidelity_n):
        return 1

    rng = random.Random(20260809)
    sample = rng.sample(pos, min(args.positions, len(pos)))
    c2_determinism(sample[0][1], sample[0][2], args.repeats, sample[0][3])
    c4_wrong_deck(sample[0][1], sample[0][2], args.repeats * 5, sample[0][3])
    if args.verify:
        return 0

    # ---- C3 + M2: top vs last, paired on the world ----------------------
    print(f"\n[C3] positive control: the clone's TOP option vs its LAST, "
          f"{args.pairs} paired rollouts at each of {len(sample)} positions")
    diffs_all: list[float] = []
    top_all: list[float] = []
    last_all: list[float] = []
    per_pos = []
    cost: dict[int, list[float]] = {}
    t_start = time.perf_counter()
    for pi, (i, o, me, deck) in enumerate(sample):
        try:
            sc = net.scores(o)
        except Exception:
            continue
        order = sorted(range(len(sc)), key=lambda k: -sc[k])
        a, b = order[0], order[-1]
        d, ta, la = [], [], []
        for k in range(args.pairs):
            seed = 100000 * pi + k
            wa = determinize(o, deck, [], random.Random(seed))
            va, _s, secs = rollout(o, wa, [a], me)
            cost.setdefault(o["current"]["turn"], []).append(secs)
            wb = determinize(o, deck, [], random.Random(seed))
            vb, _s, _c = rollout(o, wb, [b], me)
            if va is None or vb is None:
                continue
            d.append(va - vb)
            ta.append(va)
            la.append(vb)
        if len(d) < 5:
            continue
        diffs_all += d
        top_all += ta
        last_all += la
        per_pos.append((statistics.fmean(d), len(d), o["current"]["turn"],
                        statistics.fmean(ta)))
    elapsed = time.perf_counter() - t_start

    if not diffs_all:
        print("     🔴 no usable pairs")
        return 1
    n = len(diffs_all)
    mean = statistics.fmean(diffs_all)
    sd_p = statistics.stdev(diffs_all) if n > 1 else 0.0
    se = sd_p / (n ** 0.5)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    # 🔴 THE NAIVE INTERVAL IS TOO NARROW AND WE CAUGHT IT BY REPLICATION.
    # Pairs are nested inside positions: two runs of this identical cell read
    # +0.130 and +0.107 against a nominal +/-0.017, which cannot both be true.
    # The unit of independent variation is the POSITION, not the pair -- the
    # per-position mean effect varies far more than within-position sampling
    # suggests. Cluster on the position and quote that.
    pos_means = [m for m, _c, _t, _p in per_pos]
    k = len(pos_means)
    se_cl = (statistics.stdev(pos_means) / (k ** 0.5)) if k > 1 else float("nan")
    mean_cl = statistics.fmean(pos_means)
    lo_cl, hi_cl = mean_cl - 1.96 * se_cl, mean_cl + 1.96 * se_cl
    print(f"     Δ(top − last) = {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
          f"n={n} pairs over {len(per_pos)} positions  (naive, pairs "
          f"treated as independent -- DO NOT QUOTE)")
    print(f"     Δ CLUSTERED BY POSITION = {mean_cl:+.4f} "
          f"[{lo_cl:+.4f}, {hi_cl:+.4f}]  k={k} positions  "
          f"(×{se_cl / se:.1f} wider) ← the honest interval")
    lo = lo_cl
    print(f"     top arm {statistics.fmean(top_all):.3f} · "
          f"last arm {statistics.fmean(last_all):.3f}")
    if lo > 0:
        print("     ✅ the instrument RESOLVES a contrast the clone itself "
              "ranks first vs last. It has real signal.")
    else:
        print("     🔴 the instrument cannot separate the clone's best option "
              "from its worst. No sample size rescues a design built on it.")

    # M2: how much did the shared world buy?
    sd_u = (statistics.pvariance(top_all) + statistics.pvariance(last_all)) ** 0.5
    rho = 1 - (sd_p ** 2 / sd_u ** 2) if sd_u else 0.0
    print(f"\n[M2] paired sd {sd_p:.3f} vs unpaired {sd_u:.3f} "
          f"-> shared-world correlation ρ≈{rho:.2f} "
          f"({rho * 100:.0f}% variance removed)")

    # M1: cost
    print("\n[M1] rollout cost by turn")
    for t in sorted(cost):
        v = cost[t]
        print(f"     turn {t:2d}: n={len(v):4d} mean {statistics.fmean(v) * 1000:.0f} ms")
    allc = [c for v in cost.values() for c in v]
    mean_ms = statistics.fmean(allc) * 1000
    print(f"     overall {mean_ms:.0f} ms/rollout, {len(allc)} rollouts, "
          f"{elapsed:.0f}s wall")
    # What does that buy? ⚠ Size on the CLUSTERED sd -- sizing on the pair-level
    # sd is the same mistake the naive CI above makes, and it understates the
    # budget by the same factor.
    per_pos_pairs = statistics.fmean(c for _m, c, _t, _p in per_pos)
    sd_pos = statistics.stdev(pos_means) if k > 1 else float("nan")
    need_pos = (1.96 * sd_pos / 0.02) ** 2
    print(f"     ⇒ ±0.020 on a pooled Δ needs ≈{need_pos:.0f} POSITIONS "
          f"(× {per_pos_pairs:.0f} pairs each) = "
          f"{need_pos * per_pos_pairs * 2 * mean_ms / 1000 / 60:.0f} min "
          f"on one core")
    print(f"     (sizing on the pair-level sd would have said "
          f"{(1.96 * sd_p / 0.02) ** 2:.0f} pairs — the same error as the "
          f"naive CI)")

    # M3: are positions resolvable at all?
    print("\n[M3] position win-probability under the clone (top arm)")
    band = sum(1 for _m, _n, _t, p in per_pos if 0.15 <= p <= 0.85)
    print(f"     {band}/{len(per_pos)} sampled positions sit in [0.15, 0.85]; "
          f"mean {statistics.fmean(p for *_x, p in per_pos):.3f}")
    print("     ⚠ a position the clone wins 95% of the time cannot show a "
          "large action-value gap -- it bounds what any instrument can see "
          "there, and it is a property of the POSITION, not the estimator.")

    # M4: expert seats
    print("\n[M4] do the EXPERTS' seats reconstruct? (the payoff use case)")
    for dump in args.expert_dumps:
        d = ROOT / "replays" / dump
        if not d.exists():
            print(f"     {dump}: MISSING")
            continue
        gs = load_games(d, args.expert_games)
        tot_pos = 0
        ok = tried = 0
        for _f, rep in gs:
            for seat in (0, 1):
                ps = positions(rep, seat)
                tot_pos += len(ps)
                if not ps:
                    continue
                sdl = seat_decklist(rep, seat)
                if sdl is None:
                    continue
                j, oo = ps[len(ps) // 2]
                w = determinize(oo, sdl, [], random.Random(5 + j))
                tried += 1
                try:
                    _sid, obs2 = fork(oo, w)
                    same = ([json.dumps(x, sort_keys=True)
                             for x in oo["select"]["option"]]
                            == [json.dumps(x, sort_keys=True)
                                for x in obs2["select"]["option"]])
                    fs.end()
                    ok += bool(same)
                except Exception:
                    pass
        print(f"     {dump}: {len(gs)} games, {tot_pos} positions, "
              f"reconstruct {ok}/{tried}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="replays/submission_v5_s2")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--positions", type=int, default=24)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--fidelity-n", type=int, default=60)
    ap.add_argument("--expert-dumps", nargs="*",
                    default=["mirror_experts", "ntumlnoob_31-07-2026"])
    ap.add_argument("--expert-games", type=int, default=8)
    ap.add_argument("--verify", action="store_true",
                    help="controls C1/C2 only")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
