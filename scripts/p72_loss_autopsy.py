"""Autopsy our own losses for damage-placement decisions that were DOMINATED.

**Why this exists.** Every behavioural gap this project has mined so far was
found by comparing our *rate* of doing something against the experts' rate --
`p66` per decision (wrong unit, rule 21), `p70` per turn (right unit, found
Poffin). Both rank by DISAGREEMENT, and both produced nulls (§8bj, §8bl),
because a rate gap is a **tradeoff** and rule 11 says tradeoff rules lose (0/5).

This asks a different question, and it is the question rule 11 says wins (3/3):
**was an option available that DOMINATES the one we took?** For passive damage
the dominated case is arithmetic and needs no expert corpus at all -- if the
same damage, placed on a different legal option, would have KO'd a Pokemon and
what we actually hit did not die, the alternative took a prize and ours did not.

⚠ **This finds candidates, it does not judge them.** "A KO was available" is
not automatically dominated -- declining a bench KO to keep damage on the
Active can be correct. The script therefore reports the SPLIT (how many
alternatives, whether the chosen target died later anyway, whether we lost the
game) and leaves the classification to a human, per rule 14: size before you
build.

    python -X utf8 scripts/p72_loss_autopsy.py --dir replays/submission_v5_s2

⚠ Option->Pokemon mapping is via `steps[0][0].visualize`, NOT the per-step
`action` fields -- those do not align (verified: the visualize path matches the
observed hp drop 55/55, the per-step path 11/48). `--verify` re-runs that
positive control and is the first thing to check if a number here looks wrong.
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

from sa import cards as cdb  # noqa: E402

# Select contexts that place damage on the OPPONENT's board (targeting.py).
DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE = 13, 14, 15
CHIP_CONTEXTS = (DAMAGE_COUNTER, DAMAGE_COUNTER_ANY, DAMAGE)
# Adrena-Brain's SOURCE pick -- every option is one of OURS, and the counters
# come OFF it. Picking the wrong one is the "we healed the wrong Pokemon" case.
REMOVE_DAMAGE_COUNTER = 16

AREA_ACTIVE, AREA_BENCH = 4, 5


def _nm(cid: int) -> str:
    if not cid or cid <= 0:
        return "-"
    try:
        return str(cdb.card(int(cid)).get("name") or f"#{cid}")
    except Exception:  # noqa: BLE001
        return f"#{cid}"


def _pk(state: dict, o: dict) -> dict | None:
    """The Pokemon an option names, or None if the index is out of range."""
    try:
        pl = state["players"][o["playerIndex"]]
    except (KeyError, IndexError, TypeError):
        return None
    arr = pl["active"] if o.get("area") == AREA_ACTIVE else pl["bench"]
    k = o.get("index", 0)
    return arr[k] if 0 <= k < len(arr) else None


def _hp_map(state: dict) -> dict[int, int]:
    out: dict[int, int] = {}
    for pl in state["players"]:
        for p in pl["active"] + pl["bench"]:
            out[p["serial"]] = p["hp"]
    return out


def _records(rep: dict) -> list[dict]:
    """Decision records in order, same extraction as build_policy_dataset."""
    vis = rep["steps"][0][0].get("visualize") or []
    out = []
    for v in vis:
        ob = v.get("obs") or {}
        if not ob.get("current") or not ob.get("select"):
            continue
        sel = v["obs"]["select"]
        act = v.get("selected")
        if act is None:
            act = v.get("action")
        if not isinstance(act, list):
            continue
        opts = sel.get("option") or []
        if any(not isinstance(a, int) or not 0 <= a < len(opts) for a in act):
            continue
        out.append({"obs": ob, "sel": sel, "picked": act})
    return out


def verify(games: list[tuple[Path, dict]]) -> int:
    """Positive control: does the chosen option name the Pokemon that lost hp?"""
    hit = miss = amb = 0
    for _, rep in games:
        recs = _records(rep)
        for a, b in zip(recs, recs[1:]):
            if a["sel"].get("context") not in CHIP_CONTEXTS:
                continue
            if len(a["picked"]) != 1:
                continue
            s0, s1 = _hp_map(a["obs"]["current"]), _hp_map(b["obs"]["current"])
            drop = [s for s in s0 if s in s1 and s1[s] < s0[s]]
            if len(drop) != 1:
                amb += 1
                continue
            pk = _pk(a["obs"]["current"], a["sel"]["option"][a["picked"][0]])
            if pk and pk["serial"] == drop[0]:
                hit += 1
            else:
                miss += 1
    tot = hit + miss
    print(f"[verify] chosen option names the damaged Pokemon: {hit}/{tot}"
          f"  ({amb} ambiguous, excluded)")
    if tot and hit / tot < 0.99:
        print("  🔴 MAPPING IS BROKEN -- every number below is meaningless.")
        return 1
    print("  ✅ mapping holds; the autopsy below rests on it.")
    return 0


def autopsy(games: list[tuple[Path, dict]], us: set[str], verbose: int) -> None:
    n_games = n_lost = 0
    # denominators first (rule 13: a rate without its denominator is a vibe)
    sel_seen: Counter = Counter()
    sel_real: Counter = Counter()
    missed: list[dict] = []
    per_game_missed: Counter = Counter()
    lost_games: set = set()

    for path, rep in games:
        names = (rep.get("info") or {}).get("TeamNames") or []
        seats = {i for i, n in enumerate(names) if n in us}
        if not seats:
            continue
        rewards = rep.get("rewards") or [None, None]
        if rewards[0] is None or rewards[1] is None:
            continue
        gid = path.stem
        n_games += 1
        me = next(iter(seats))
        lost = rewards[me] < rewards[1 - me]
        if lost:
            n_lost += 1
            lost_games.add(gid)

        recs = _records(rep)
        for idx, (a, b) in enumerate(zip(recs, recs[1:])):
            state = a["obs"]["current"]
            if state.get("yourIndex") not in seats:
                continue
            ctx = a["sel"].get("context")
            if ctx not in CHIP_CONTEXTS:
                continue
            opts = a["sel"].get("option") or []
            sel_seen[ctx] += 1
            if len(opts) < 2 or len(a["picked"]) != 1:
                continue          # forced: no decision was made
            sel_real[ctx] += 1

            # Damage actually dealt, measured rather than assumed -- the amount
            # differs by source (Adrena-Brain 30, Shadow Bullet's snipe, ...)
            # and reading it off the board covers every case uniformly.
            s0, s1 = _hp_map(state), _hp_map(b["obs"]["current"])
            chosen = _pk(state, opts[a["picked"][0]])
            if not chosen:
                continue
            ser = chosen["serial"]
            if ser not in s0 or ser not in s1:
                continue
            dmg = s0[ser] - s1[ser]
            if dmg <= 0:
                continue
            chosen_died = s1[ser] <= 0

            # Would the SAME damage have KO'd a different legal option?
            alts = []
            for j, o in enumerate(opts):
                if j == a["picked"][0]:
                    continue
                pk = _pk(state, o)
                if not pk or pk["serial"] == ser:
                    continue
                if 0 < pk["hp"] <= dmg:
                    alts.append(pk)
            if not alts or chosen_died:
                continue

            missed.append({
                "gid": gid, "lost": lost, "ctx": ctx, "dmg": dmg,
                "chosen": chosen, "alts": alts,
                "chosen_area": opts[a["picked"][0]].get("area"),
                "chosen_hp": chosen["hp"],
                "turn": int(round(state.get("turn", -1)))
                if isinstance(state.get("turn"), (int, float)) else -1,
            })
            per_game_missed[gid] += 1

    print(f"\n=== {n_games} games, {n_lost} lost ({n_lost / max(n_games,1):.1%}) ===")
    print("\n=== DENOMINATOR: are these decisions even real? ===")
    print(f"  {'context':<26}{'selects':>9}{'>=2 options':>13}{'real share':>12}")
    for ctx in CHIP_CONTEXTS:
        if not sel_seen[ctx]:
            continue
        nm = {DAMAGE_COUNTER: "13 DAMAGE_COUNTER",
              DAMAGE_COUNTER_ANY: "14 DAMAGE_COUNTER_ANY",
              DAMAGE: "15 DAMAGE"}[ctx]
        print(f"  {nm:<26}{sel_seen[ctx]:>9}{sel_real[ctx]:>13}"
              f"{sel_real[ctx]/max(sel_seen[ctx],1):>11.1%}")
    tot_real = sum(sel_real.values())
    print(f"  {'TOTAL':<26}{sum(sel_seen.values()):>9}{tot_real:>13}"
          f"{tot_real/max(sum(sel_seen.values()),1):>11.1%}"
          f"   = {tot_real/max(n_games,1):.1f} real choices/game")

    print("\n=== THE DOMINATED CANDIDATE: a KO was available and we spread instead ===")
    n = len(missed)
    in_lost = sum(1 for m in missed if m["lost"])
    print(f"  events                          {n}")
    print(f"  per game                        {n / max(n_games,1):.2f}"
          f"   (sizing gate is 0.5/game)")
    print(f"  share of real damage choices    {n / max(tot_real,1):.1%}")
    print(f"  in games we LOST                {in_lost} of {n}"
          f"  ({in_lost / max(n,1):.1%}; losses are {n_lost/max(n_games,1):.1%} of games)")
    gl = len({m['gid'] for m in missed if m['lost']})
    print(f"  distinct LOST games touched     {gl} of {n_lost}")

    by_ctx = Counter(m["ctx"] for m in missed)
    print("\n  by context:", dict(by_ctx))
    print("  how many KOs were on offer:",
          dict(Counter(len(m["alts"]) for m in missed)))

    print("\n=== WHAT WE HIT INSTEAD (the tradeoff check) ===")
    print("  If we were hitting the ACTIVE while a bench KO went begging, that")
    print("  is a defensible tradeoff. If we were hitting another BENCH slot,")
    print("  it is much closer to dominated.")
    area = Counter()
    for m in missed:
        # the chosen Pokemon's own area, recovered from the option we took
        area["active" if m["chosen_area"] == AREA_ACTIVE else "bench"] += 1
    print("  ", dict(area))

    tgt = Counter(_nm(m["chosen"]["id"]) for m in missed)
    print("\n  chosen target card:", tgt.most_common(6))
    kod = Counter(_nm(p["id"]) for m in missed for p in m["alts"])
    print("  the KO we passed up:", kod.most_common(6))

    if verbose:
        print("\n=== SAMPLE EVENTS (lost games first) ===")
        for m in sorted(missed, key=lambda x: not x["lost"])[:verbose]:
            alts = ", ".join(f"{_nm(p['id'])}@{p['hp']}hp" for p in m["alts"])
            print(f"  {'LOSS' if m['lost'] else 'win '} {m['gid']} ctx{m['ctx']}"
                  f" dmg={m['dmg']:>3}  hit {_nm(m['chosen']['id'])}"
                  f"@{m['chosen_hp']}hp (survived)  |  KO available: {alts}")


def source_audit(games: list[tuple[Path, dict]], us: set[str], verbose: int) -> None:
    """Adrena-Brain's SOURCE pick -- the cleanest dominated test in the deck.

    The ability moves **up to 3** damage counters off one of ours and onto one
    of theirs, so the source CAPS how much damage the activation is worth. A
    source carrying 1 counter when another carries 3 moves 10 damage instead of
    30 -- strictly less healing AND strictly less damage, same action, same
    cost. That is dominated by construction and needs no counterfactual, which
    matters here because the replay's damage log does NOT reconcile with the
    board (38% mismatch) and cannot carry a "would it have survived" claim.

    `counter_source` in targeting.py is exactly this rule, it cleared its own
    A/B at 0.534, and the SHIPPED bundle has it OFF. So this measures what the
    net leaves on the table with no rule helping it.
    """
    n_games = 0
    sel_seen = sel_real = 0
    under: list[dict] = []
    moved_tot = best_tot = 0

    for path, rep in games:
        names = (rep.get("info") or {}).get("TeamNames") or []
        seats = {i for i, n in enumerate(names) if n in us}
        if not seats:
            continue
        rewards = rep.get("rewards") or [None, None]
        if rewards[0] is None or rewards[1] is None:
            continue
        me = next(iter(seats))
        lost = rewards[me] < rewards[1 - me]
        n_games += 1
        recs = _records(rep)

        for i, a in enumerate(recs):
            state = a["obs"]["current"]
            if state.get("yourIndex") not in seats:
                continue
            if a["sel"].get("context") != REMOVE_DAMAGE_COUNTER:
                continue
            sel_seen += 1
            opts = a["sel"].get("option") or []
            if len(opts) < 2 or len(a["picked"]) != 1:
                continue
            sel_real += 1

            chosen = _pk(state, opts[a["picked"][0]])
            if not chosen:
                continue
            # Counters actually moved, measured off the board. The source pick
            # resolves together with the TARGET pick, so the heal lands ~2
            # records later, not on the next one (verified: 65/65).
            ser, h0 = chosen["serial"], chosen["hp"]
            moved = 0
            for j in range(i + 1, min(i + 6, len(recs))):
                hj = _hp_map(recs[j]["obs"]["current"]).get(ser)
                if hj is None:
                    break
                if hj != h0:
                    moved = hj - h0
                    break
            if moved <= 0:
                continue

            # What the best available source was worth: min(3 counters, what it
            # carries) x 10.
            best, best_pk = moved, None
            for j, o in enumerate(opts):
                if j == a["picked"][0]:
                    continue
                pk = _pk(state, o)
                if not pk:
                    continue
                cap = min(MAX_MOVE * 10, pk["maxHp"] - pk["hp"])
                if cap > best:
                    best, best_pk = cap, pk
            moved_tot += moved
            best_tot += max(best, moved)
            if best > moved and best_pk is not None:
                under.append({"gid": path.stem, "lost": lost, "moved": moved,
                              "best": best, "chosen": chosen, "alt": best_pk})

    print("\n\n" + "=" * 70)
    print("=== ADRENA-BRAIN SOURCE PICK (ctx 16) — the dominated test ===")
    print("=" * 70)
    print(f"  selects                         {sel_seen}"
          f"   ({sel_seen / max(n_games,1):.1f}/game)")
    print(f"  with >=2 options (a real pick)  {sel_real}"
          f"   ({sel_real / max(n_games,1):.2f}/game)  <- sizing gate is 0.5")
    n = len(under)
    print(f"\n  UNDER-MOVED (a source carrying more was available)")
    print(f"    events                        {n}"
          f"   ({n / max(n_games,1):.2f}/game)")
    print(f"    share of real picks           {n / max(sel_real,1):.1%}")
    lost_n = sum(1 for m in under if m["lost"])
    print(f"    in games we LOST              {lost_n} of {n}")
    print(f"\n  damage actually moved           {moved_tot}")
    print(f"  damage available to move        {best_tot}")
    print(f"  LEFT ON THE TABLE               {best_tot - moved_tot}"
          f"   ({(best_tot - moved_tot) / max(best_tot,1):.1%} of the ability's value,"
          f" {(best_tot - moved_tot) / max(n_games,1):.0f} damage/game)")
    if under:
        gap = Counter(m["best"] - m["moved"] for m in under)
        print("  gap size histogram:", dict(sorted(gap.items())))
    if verbose and under:
        print("\n  === SAMPLE (lost games first) ===")
        for m in sorted(under, key=lambda x: not x["lost"])[:verbose]:
            print(f"    {'LOSS' if m['lost'] else 'win '} {m['gid']}"
                  f"  moved {m['moved']} off {_nm(m['chosen']['id'])}"
                  f" ({m['chosen']['hp']}/{m['chosen']['maxHp']})"
                  f"  |  {m['best']} was available off"
                  f" {_nm(m['alt']['id'])} ({m['alt']['hp']}/{m['alt']['maxHp']})")


MAX_MOVE = 3   # Adrena-Brain moves "up to 3 damage counters" (targeting.py)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", default=["replays/submission_v5_s2"])
    ap.add_argument("--us", action="append", default=["Scio"],
                    help="our seat's team name(s)")
    ap.add_argument("--verify", action="store_true",
                    help="run ONLY the option->Pokemon positive control")
    ap.add_argument("--verbose", type=int, default=12,
                    help="print N sample events (0 = none)")
    args = ap.parse_args()

    games: list[tuple[Path, dict]] = []
    n_err = 0
    for d in args.dir:
        for path in sorted((ROOT / d).glob("*.json")):
            try:
                games.append((path, json.loads(path.read_text(encoding="utf-8"))))
            except Exception:  # noqa: BLE001
                n_err += 1
    print(f"loaded {len(games)} replays ({n_err} unreadable)")

    rc = verify(games)
    if args.verify or rc:
        return rc
    autopsy(games, set(args.us), args.verbose)
    source_audit(games, set(args.us), args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
