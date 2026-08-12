"""E28 reading 1 — self-predictability: can a_t be predicted from a_{t-1} alone?

Pre-registered in `docs/experiments/E28-replay-trace-audit.md` (frozen at
`aeb530b`). Step 0 passed in `p87_e28_pairs.py` (us 26,318 / them 46,029 pairs,
11 strata at >=500 both sides).

    python -X utf8 scripts/p88_e28_reading1.py --controls   # controls only
    python -X utf8 scripts/p88_e28_reading1.py              # + the comparison

⛔ **The comparison is printed ONLY if both controls pass.** E28 §3: an
estimator gets a control or its number is not admissible (§8bv's rule). The
gate is enforced in code here, not left to the reader.

**Alphabet (coarse, pre-registered): decision kind x card class.**
`cardType` is a 7-value enum -- 0 Pokemon, 1 Item, 2 Tool, 3 Supporter,
4 Stadium, 5 Basic Energy, 6 Special Energy -- plus `-1` for an option naming
no card (a target, a pass). ⚠ **Coarse on purpose**: §8bv's plug-in MI positive
control read NEGATIVE under a too-fine conditioning bucket.

**The card an option names comes from `optfeat.option_features`, the NET'S OWN
extractor** -- so this script cannot disagree with the net about what an option
is. The hand-rolled version of that lookup was wrong once already (`p76`: a
PLAY option carries no `area` at all, so every card play was invisible).

**Stratification is mandatory** (E28 §3): a pair is assigned to the stratum of
the PREDICTED decision `b`, so "which decisions we face" cannot masquerade as
self-predictability. Within a stratum `ctx_b` is constant, so the target symbol
reduces to `b`'s card class and the cue is the full pair `(ctx_a, class_a)`.

**Train/test split is BY GAME, not by pair.** Two pairs from one game share a
trace; splitting by pair would leak it and inflate both sides.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from p9_field_census import _cid, _signature  # noqa: E402
from p87_e28_pairs import SIDES, MIRROR  # noqa: E402
from sa import cards as cdb  # noqa: E402
from sa.optfeat import option_features  # noqa: E402

SEED = 20260812
KEPT_STRATA = {0, 7, 13, 16, 40, 21, 15, 43, 22, 3, 30}   # from Step 0


def _klass(obs: dict, opt: dict) -> int:
    """Coarse card class of the option, via the net's own extractor."""
    try:
        cid = int(option_features(obs, opt)[1] or 0)
    except Exception:  # noqa: BLE001
        return -1
    if cid <= 0:
        return -1
    d = cdb.card(cid)
    if not d:
        return -1
    t = d.get("cardType")
    return int(t) if isinstance(t, int) else -1


def scan(rep: dict):
    """(seat labels, per-seat traces, team names). One parse per replay.

    A trace entry is (turn, tac, ctx, chosen_class, available_classes).
    """
    if not isinstance(rep, dict):
        return None
    info = rep.get("info")
    names = [str(x) for x in ((info or {}).get("TeamNames") or [])] \
        if isinstance(info, dict) else []
    try:
        vis = rep["steps"][0][0].get("visualize") or []
    except (KeyError, IndexError, TypeError):
        return None

    poke = [Counter(), Counter()]
    maxc = [defaultdict(int), defaultdict(int)]
    stream: list[dict] = []

    for v in vis:
        ob = v.get("obs") or {}
        st = ob.get("current")
        if not st:
            continue
        me = st.get("yourIndex")
        if me not in (0, 1):
            continue

        sel = ob.get("select")
        if sel:
            act = v.get("selected")
            if act is None:
                act = v.get("action")
            opts = sel.get("option") or []
            if (isinstance(act, list) and len(act) == 1 and opts
                    and isinstance(act[0], int) and 0 <= act[0] < len(opts)):
                avail = [_klass(ob, o) for o in opts]
                stream.append({
                    "seat": me, "turn": st.get("turn"),
                    "tac": st.get("turnActionCount"), "ctx": sel.get("context"),
                    "k": avail[act[0]], "avail": avail,
                })

        try:
            op = st["players"][1 - me]
        except (KeyError, IndexError, TypeError):
            continue
        g = 1 - me
        here: Counter = Counter()
        a0 = op.get("active")
        if a0 and a0[0]:
            here[_cid(a0[0])] += 1
            for c in (a0[0].get("cards") or []):
                here[_cid(c)] += 1
        for pk in (op.get("bench") or []):
            if pk:
                here[_cid(pk)] += 1
                for c in (pk.get("cards") or []):
                    here[_cid(c)] += 1
        for c in (op.get("discard") or []):
            here[_cid(c)] += 1
        for cid, n in here.items():
            if n > maxc[g][cid]:
                maxc[g][cid] = n
            if cdb.is_pokemon(cid):
                poke[g][cid] += 1

    return [_signature(poke[i], maxc[i]) for i in (0, 1)], stream, names


def collect(side: str) -> list[list[dict]]:
    """One trace per (game, mirror seat), deduped by episode id."""
    out: list[list[dict]] = []
    seen: set[str] = set()
    for dname, team in SIDES[side]:
        d = ROOT / dname
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.json")):
            if path.name == "manifest.json" or path.stem in seen:
                continue
            seen.add(path.stem)
            try:
                got = scan(json.loads(path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
            if got is None:
                continue
            labels, stream, names = got
            idxs = ([i for i, nm in enumerate(names)
                     if any(t.lower() in nm.lower() for t in team)]
                    if team else [0, 1])
            for i in idxs:
                if labels[i] != MIRROR:
                    continue
                tr = [r for r in stream if r["seat"] == i]
                if len(tr) >= 2:
                    out.append(tr)
    return out


def pairs(trace: list[dict]) -> list[tuple[dict, dict]]:
    """Consecutive within-turn pairs; boundaries EXCLUDED, not attributed."""
    return [(a, b) for a, b in zip(trace, trace[1:])
            if a["seat"] == b["seat"] and a["turn"] == b["turn"]
            and a["turn"] is not None and a["tac"] is not None
            and b["tac"] is not None and b["tac"] == a["tac"] + 1]


# ---------------------------------------------------------------- transforms
def t_identity(tr: list[dict]) -> list[dict]:
    return tr


def t_repeat(tr: list[dict], fallback: bool = True) -> list[dict]:
    """⬆ POSITIVE CONTROL -- repeat the previous action wherever still legal.

    "Legal" is checked against the slot's OWN option list: the synthetic agent
    re-picks its previous card class iff an option of that class is offered
    here. An estimator that cannot see this cannot see repetition at all.

    🔴 `fallback` is the correction re-registered in E28 §R4, and it exists
    because the first construction FAILED THE BAR FOR A STRUCTURAL REASON.
    **Repetition is legal at only 51.3% of within-turn pairs.** The original
    trace kept the REAL action on the other 48.7%, so it was only ~half
    synthetic and could not reach 0.90 however sensitive the estimator was --
    it read 0.8774. With `fallback=True` an illegal slot takes the
    **lowest-index option** instead, making the trace a deterministic function
    of (previous action, option list). ⇒ the >=0.90 bar becomes a statement
    about the ESTIMATOR, which is what it was always meant to be.

    ⛔ The 0.90 threshold is UNCHANGED and there is no third construction.
    """
    out = [dict(r) for r in tr]
    for i in range(1, len(out)):
        prev = out[i - 1]["k"]
        same_turn = out[i]["turn"] == out[i - 1]["turn"]
        if same_turn and prev in out[i]["avail"]:
            out[i]["k"] = prev
        elif fallback and out[i]["avail"]:
            out[i]["k"] = out[i]["avail"][0]
    return out


def t_shuffle(tr: list[dict], rng: random.Random) -> list[dict]:
    """⬇ NEGATIVE CONTROL -- actions shuffled within each turn.

    ⚠ Permuted within (turn x ctx), not within turn alone. Shuffling across
    decision kinds would put a card class into a slot where it cannot occur,
    which DEPRESSES accuracy below the base rate and lets the control pass
    trivially. Restricting the permutation to same-kind slots preserves each
    stratum's marginal exactly and destroys only the ordering -- which is the
    thing the control is meant to test. This is strictly the harder version.
    """
    out = [dict(r) for r in tr]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(out):
        buckets[(r["turn"], r["ctx"])].append(i)
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        ks = [out[i]["k"] for i in idxs]
        rng.shuffle(ks)
        for i, k in zip(idxs, ks):
            out[i]["k"] = k
    return out


# ------------------------------------------------------------------ estimator
def evaluate(traces: list[list[dict]], rng: random.Random) -> dict:
    """Held-out accuracy + plug-in MI, per stratum and pooled. Split BY GAME."""
    order = list(range(len(traces)))
    rng.shuffle(order)
    cut = len(order) // 2
    train_ix, test_ix = set(order[:cut]), set(order[cut:])

    # stratum -> cue -> Counter(target)
    tab: dict[int, dict[tuple, Counter]] = defaultdict(lambda: defaultdict(Counter))
    marg: dict[int, Counter] = defaultdict(Counter)
    for t in train_ix:
        for a, b in pairs(traces[t]):
            if b["ctx"] not in KEPT_STRATA:
                continue
            tab[b["ctx"]][(a["ctx"], a["k"])][b["k"]] += 1
            marg[b["ctx"]][b["k"]] += 1

    hit = tot = 0
    per: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    joint: dict[int, Counter] = defaultdict(Counter)
    for t in test_ix:
        for a, b in pairs(traces[t]):
            c = b["ctx"]
            if c not in KEPT_STRATA:
                continue
            cue = (a["ctx"], a["k"])
            cnt = tab[c].get(cue)
            pred = (cnt.most_common(1)[0][0] if cnt
                    else (marg[c].most_common(1)[0][0] if marg[c] else None))
            ok = int(pred == b["k"])
            hit += ok
            tot += 1
            per[c][0] += ok
            per[c][1] += 1
            joint[c][(cue, b["k"])] += 1

    # plug-in MI per stratum on the held-out half
    mis: dict[int, float] = {}
    for c, jc in joint.items():
        n = sum(jc.values())
        if n < 2:
            continue
        px: Counter = Counter()
        py: Counter = Counter()
        for (x, y), k in jc.items():
            px[x] += k
            py[y] += k
        mi = 0.0
        for (x, y), k in jc.items():
            pxy = k / n
            mi += pxy * math.log2(pxy / ((px[x] / n) * (py[y] / n)))
        mis[c] = mi

    base_hit = 0
    for t in test_ix:
        for a, b in pairs(traces[t]):
            if b["ctx"] in KEPT_STRATA and marg[b["ctx"]]:
                base_hit += int(marg[b["ctx"]].most_common(1)[0][0] == b["k"])
    return {
        "acc": hit / tot if tot else 0.0,
        "base": base_hit / tot if tot else 0.0,
        "n": tot,
        "per": {c: (v[0] / v[1], v[1]) for c, v in per.items() if v[1]},
        "mi": mis,
        "mi_mean": sum(mis.values()) / len(mis) if mis else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", action="store_true",
                    help="run the two controls and stop")
    args = ap.parse_args()

    rng = random.Random(SEED)
    print("loading traces (one parse per replay) ...")
    data = {s: collect(s) for s in ("us", "them")}
    for s, tr in data.items():
        print(f"  {s:<5} {len(tr):>5} seat-traces, "
              f"{sum(len(pairs(t)) for t in tr):>6} within-turn pairs")

    print(f"\n{'=' * 70}\n=== ⛔ ESTIMATOR CONTROLS (E28 §3) — both must pass ===\n")
    pos = evaluate([t_repeat(t) for t in data["us"]], random.Random(SEED))
    old = evaluate([t_repeat(t, fallback=False) for t in data["us"]],
                   random.Random(SEED))
    neg = evaluate([t_shuffle(t, random.Random(SEED + 1)) for t in data["us"]],
                   random.Random(SEED))
    obs = evaluate(data["us"], random.Random(SEED))

    print(f"⬆ positive (repeat-when-legal, deterministic fallback)"
          f"  held-out acc = {pos['acc']:.4f}"
          f"   (n={pos['n']}, MI={pos['mi_mean']:.4f})")
    print(f"   pre-registered: >= 0.90        "
          f"{'✅ PASS' if pos['acc'] >= 0.90 else '🔴 FAIL => CELL IS VOID'}")
    print(f"   [superseded construction, kept real action on the 48.7% of "
          f"slots where repetition is illegal: {old['acc']:.4f} — E28 §R3]")
    print(f"\n⬇ negative (within-turn shuffle) held-out acc = {neg['acc']:.4f}"
          f"   vs marginal base rate {neg['base']:.4f}"
          f"   (MI={neg['mi_mean']:.4f})")
    lift = neg["acc"] - neg["base"]
    ok_neg = abs(lift) < 0.02 and neg["mi_mean"] < 0.02
    print(f"   pre-registered: at base rate, MI ~ 0   (lift {lift:+.4f})  "
          f"{'✅ PASS' if ok_neg else '🔴 FAIL => ALPHABET LEAKS TURN POSITION, CELL IS VOID'}")

    print(f"\n   (reference, untransformed `us`: acc = {obs['acc']:.4f}, "
          f"base = {obs['base']:.4f}, MI = {obs['mi_mean']:.4f})")

    if not (pos["acc"] >= 0.90 and ok_neg):
        print(f"\n🔴 A CONTROL FAILED — the comparison is NOT computed and NOT "
              f"read. E28 §3 makes this VOID, not null.")
        return 1
    print("\n✅ BOTH CONTROLS PASS — the comparison may now be read.")
    if args.controls:
        return 0

    print(f"\n{'=' * 70}\n=== READING 1 — the comparison ===\n")
    them = evaluate(data["them"], random.Random(SEED))
    print(f"{'side':<8}{'held-out acc':>14}{'base rate':>12}{'lift':>9}"
          f"{'MI (mean)':>12}{'n pairs':>10}")
    for nm, r in (("us", obs), ("them", them)):
        print(f"{nm:<8}{r['acc']:>14.4f}{r['base']:>12.4f}"
              f"{r['acc'] - r['base']:>+9.4f}{r['mi_mean']:>12.4f}{r['n']:>10}")
    print(f"\n{'ctx':>5}{'us acc':>10}{'them acc':>10}{'us MI':>9}"
          f"{'them MI':>9}{'us n':>8}{'them n':>8}")
    for c in sorted(KEPT_STRATA):
        u, t = obs["per"].get(c), them["per"].get(c)
        if not u or not t:
            continue
        print(f"{c:>5}{u[0]:>10.4f}{t[0]:>10.4f}{obs['mi'].get(c, 0):>9.4f}"
              f"{them['mi'].get(c, 0):>9.4f}{u[1]:>8}{t[1]:>8}")
    print("\n⚠ Reading 2 (commitment switches) is required before any verdict: "
          "E28 §4 keys every branch on readings 1 AND 2 jointly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
