"""E28 Step 0 — the sizing gate: consecutive within-turn decision pairs per side.

Pre-registered in `docs/experiments/E28-replay-trace-audit.md` (frozen at
`aeb530b`). ⛔ **This script computes the GATE ONLY.** No self-predictability
statistic, no MI, no comparison — those are Step 1 and they may not be read
until both estimator controls pass (E28 §3).

    python -X utf8 scripts/p87_e28_pairs.py

**What a "pair" is, per the pre-registration.** Two decision records (a, b) made
by the SAME seat, inside the SAME turn, at CONSECUTIVE `turnActionCount`.

⚠ `turnActionCount` is a **global** within-turn counter, not a per-seat one: at
setup the stream reads `tac 2 seat 0 / tac 3 seat 1 / tac 4 seat 0`. Requiring
`same turn AND same seat AND b.tac == a.tac + 1` is therefore what excludes both
the turn handover *and* an interleaved opponent record. ⚠ **Boundary pairs are
EXCLUDED, not attributed** — `p77`'s rule: a pair spanning the handover carries
the opponent's whole reply.

**Sides.** `us` = our shipped net's own games; `them` = the same-deck expert
dumps. ⚠ Both sides are filtered to seats whose reconstructed archetype is
`Marnie's Grimmsnarl ex`, so the action alphabet matches (E28 §3). A seat that
is ours by *name* but not on our 60 is dropped and counted as dropped.
"""
from __future__ import annotations

import argparse
import json
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
from sa import cards as cdb  # noqa: E402

MIRROR = "Marnie's Grimmsnarl ex"

# Pre-registered sides. `teams=None` => take every MIRROR seat in the dump
# (the local self-play recordings carry no ladder team name).
#
# 🔴 THE THREE EXPERT DUMPS OVERLAP AND MUST BE DEDUPED BY EPISODE ID.
# `mirror_experts` is a re-cut of the other two, not an independent pull:
#     ntumlnoob ∩ mirror_experts = 148,  sixth_sense ∩ mirror_experts = 112,
#     ntumlnoob ∩ sixth_sense    =   4,  union = 555 against a naive sum of 816.
# Pooling them unguarded double-counts **261 games (32%)** and would inflate
# `them` past the Step-0 gate on duplicated data. `--dedupe` (default on) keys
# on the episode id, which is the file stem.
#
# ⚠ `sixth_sense_31-07-2026` is mostly **Raja Biswas** (158 seats) rather than
# `Sixth Sense` (69) -- naming the dump after one demonstrator and then matching
# on that name would silently drop 70% of it.
EXPERTS = ("ntumlnoob", "Raja Biswas", "Sixth Sense", "Dominic Peel")

SIDES: dict[str, list[tuple[str, tuple[str, ...] | None]]] = {
    "us": [
        ("replays/submission_v5_s2", ("Scio",)),
        ("replays/submission_optv3", ("Scio",)),
        ("replays/ours_mirror_rec", None),
    ],
    "them": [
        ("replays/ntumlnoob_31-07-2026", EXPERTS),
        ("replays/sixth_sense_31-07-2026", EXPERTS),
        ("replays/mirror_experts", EXPERTS),
    ],
}


def scan(rep: dict) -> tuple[list[str], list[dict], list[str]] | None:
    """(seat labels, decision records, team names) from ONE parse of a replay.

    Seat i's archetype is reconstructed from seat (1-i)'s observation frames --
    the same lower-bound reconstruction `p9_field_census.analyse` uses, so the
    labels are comparable to every share this repo has published.
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

    poke: list[Counter] = [Counter(), Counter()]
    maxc: list[dict[int, int]] = [defaultdict(int), defaultdict(int)]
    recs: list[dict] = []

    for v in vis:
        ob = v.get("obs") or {}
        st = ob.get("current")
        if not st:
            continue
        me = st.get("yourIndex")
        if me not in (0, 1):
            continue

        # --- decision record, same extraction as build_policy_dataset ---
        sel = ob.get("select")
        if sel:
            act = v.get("selected")
            if act is None:
                act = v.get("action")
            opts = sel.get("option") or []
            if (isinstance(act, list) and opts
                    and all(isinstance(a, int) and 0 <= a < len(opts)
                            for a in act)):
                recs.append({
                    "seat": me,
                    "turn": st.get("turn"),
                    "tac": st.get("turnActionCount"),
                    "ctx": sel.get("context"),
                    "nopt": len(opts),
                })

        # --- opponent board reconstruction, for the seat label ---
        try:
            op = st["players"][1 - me]
        except (KeyError, IndexError, TypeError):
            continue
        g = 1 - me
        here: Counter = Counter()
        act0 = op.get("active")
        if act0 and act0[0]:
            here[_cid(act0[0])] += 1
            for c in (act0[0].get("cards") or []):
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

    labels = [_signature(poke[i], maxc[i]) for i in (0, 1)]
    return labels, recs, names


def pairs_for(recs: list[dict], seat: int) -> list[tuple[dict, dict]]:
    """Consecutive within-turn pairs made by `seat`, boundaries excluded."""
    out = []
    for a, b in zip(recs, recs[1:]):
        if a["seat"] != seat or b["seat"] != seat:
            continue                      # handover / interleaved opponent
        if a["turn"] != b["turn"] or a["turn"] is None:
            continue                      # spans the turn boundary
        if a["tac"] is None or b["tac"] is None or b["tac"] != a["tac"] + 1:
            continue                      # not consecutive
        out.append((a, b))
    return out


def run_side(side: str, specs: list[tuple[str, tuple[str, ...] | None]],
             dedupe: bool = True) -> dict:
    per_ctx: Counter = Counter()          # stratum = context of the PREDICTED b
    per_ctx_a: Counter = Counter()        # secondary: context of the CUE a
    games = seats_kept = seats_dropped = 0
    boundary_dropped = 0
    dup_skipped = 0
    seen: set[str] = set()                # episode ids already counted
    errs: Counter = Counter()

    print(f"\n=== SIDE `{side}` ===")
    for dname, team in specs:
        d = ROOT / dname
        if not d.is_dir():
            print(f"  (missing {dname})")
            continue
        g = k = dup = 0
        for path in sorted(d.glob("*.json")):
            if path.name == "manifest.json":
                continue
            if dedupe:
                if path.stem in seen:
                    dup += 1
                    dup_skipped += 1
                    continue
                seen.add(path.stem)
            try:
                rep = json.loads(path.read_text(encoding="utf-8"))
                got = scan(rep)
            except Exception as exc:      # noqa: BLE001
                errs[f"{type(exc).__name__}: {exc}"] += 1
                continue
            if got is None:
                errs["not a replay dict"] += 1
                continue
            labels, recs, names = got
            games += 1
            g += 1

            # ⚠ SUBSTRING, not equality: the demonstrator appears as
            # `李秉叡（ntumlnoob）`, so exact matching finds zero seats.
            if team:
                idxs = [i for i, nm in enumerate(names)
                        if any(t.lower() in nm.lower() for t in team)]
            else:
                idxs = [0, 1]
            for i in idxs:
                if labels[i] != MIRROR:
                    seats_dropped += 1     # named, but not on our 60
                    continue
                seats_kept += 1
                k += 1
                mine = [r for r in recs if r["seat"] == i]
                pr = pairs_for(recs, i)
                boundary_dropped += max(len(mine) - 1, 0) - len(pr)
                for a, b in pr:
                    per_ctx[b["ctx"]] += 1
                    per_ctx_a[a["ctx"]] += 1
        print(f"  {Path(dname).name:<28} {g:>5} games, {k:>5} mirror seats"
              + (f"   ({dup} dup episodes skipped)" if dup else ""))

    tot = sum(per_ctx.values())
    print(f"  {'TOTAL':<28} {games:>5} games, {seats_kept:>5} seats kept, "
          f"{seats_dropped} dropped (not on our 60), "
          f"{dup_skipped} duplicate episodes skipped")
    print(f"  consecutive within-turn pairs: {tot}   "
          f"(adjacent-but-excluded, i.e. boundary/interleaved: "
          f"{boundary_dropped})")
    if errs:
        for e, c in errs.most_common(5):
            print(f"    ! {c:>4}  {e}")
    return {"per_ctx": per_ctx, "per_ctx_a": per_ctx_a, "total": tot}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-stratum", type=int, default=500,
                    help="pre-registered: a stratum below this on EITHER side "
                         "is dropped and said so")
    ap.add_argument("--min-pooled", type=int, default=2000,
                    help="pre-registered: pooled below this per side => VOID")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="⛔ diagnostic only -- the expert dumps overlap 32%%")
    args = ap.parse_args()

    res = {s: run_side(s, specs, dedupe=not args.no_dedupe)
           for s, specs in SIDES.items()}
    us, them = res["us"], res["them"]

    print(f"\n{'=' * 68}\n=== ⛔ STEP-0 GATE (pre-registered, E28 §2) ===")
    print(f"\n{'ctx':>5} {'us pairs':>10} {'them pairs':>12}   verdict")
    keep, drop = [], []
    for ctx in sorted(set(us["per_ctx"]) | set(them["per_ctx"]),
                      key=lambda c: -(us["per_ctx"][c] + them["per_ctx"][c])):
        u, t = us["per_ctx"][ctx], them["per_ctx"][ctx]
        ok = u >= args.min_stratum and t >= args.min_stratum
        (keep if ok else drop).append((ctx, u, t))
        print(f"{str(ctx):>5} {u:>10} {t:>12}   "
              f"{'KEEP' if ok else 'dropped (<%d)' % args.min_stratum}")

    print(f"\n  pooled pairs   us={us['total']}   them={them['total']}")
    print(f"  strata kept    {len(keep)}  {[c for c, _, _ in keep]}")
    print(f"  strata dropped {len(drop)}")
    kept_u = sum(u for _, u, _ in keep)
    kept_t = sum(t for _, _, t in keep)
    print(f"  pairs in KEPT strata   us={kept_u}   them={kept_t}")

    void = us["total"] < args.min_pooled or them["total"] < args.min_pooled
    print(f"\n  {'🔴 VOID — pooled below %d on a side' % args.min_pooled if void else '✅ GATE PASSES on pooled count'}")
    if not keep:
        print("  🔴 no stratum survives => nothing to compute in Step 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
