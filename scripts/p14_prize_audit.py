"""Where did we leave PRIZES on the table? -- the general form of the Boss's Orders bug.

**Why this exists, and it is a process fix rather than another rule.** Every
defect this project has repaired was found the same way: the user watched games
and said "that looked wrong", then we measured it. That loop found real bugs --
the Boss's Orders double-KO throwaway (29% of drags) came from exactly one such
observation -- but it does not scale, and it only finds what someone happened to
notice.

This automates the noticing. Per turn it computes **the prizes a competent
player could have taken** with the board as it stood, compares that to what we
actually took, and buckets every shortfall by cause. It needs no opponent model
and no search: this deck has ONE payable attack (Shadow Bullet, 180 to the
Active plus 30 to a bench), so "what was available this turn" is arithmetic.

Lines it prices, all from the real board state:

  * **attack now**            -- KO the Active if 180 kills it, plus the snipe
                                 if any bench sitter is at <= 30 HP.
  * **drag then attack**      -- for each benched target we can KO with Boss's
                                 Orders, that target's prizes plus a snipe onto
                                 a DIFFERENT bench sitter.
  * **what we actually did**  -- read from the log (type 15 attacks, type 16 HP
                                 changes) and the prize counts.

🔴 **STATUS 2026-07-31: PARTIALLY VALIDATED. Do NOT quote the "did not attack"
bucket.** It reports 27.7% on real replays and 14.9% on arena games, and both
contradict a far more solid measurement: P5c audited never-end-without-attacking
at **3,683 / 3,683** (`EVIDENCE` §8). When a new instrument disagrees with an
established one, the instrument is the suspect (rule 9 -- `drag_target` read zero
rows for days because it was keyed on the wrong context). The arena cross-check
rules out replay frame coverage as the cause, since those frames are complete,
so the fault is in `_available()` over-reporting: it prices "180 to the Active"
from `best_damage` without establishing that our Active can legally attack
**this turn** (just promoted after a KO, first-player turn 1, ability lock).
**Fix that before believing any row.**

✅ **The bucket that IS corroborated** is "dragged when ATTACKING NOW priced
higher" -- 4.4% of live turns on replays, 8.5% in the arena. That is the same
defect `p8_optv3_replays.py` measured independently at 29% of drags and that
`targeting.boss_prize_veto` was written for, found here by a completely
different route.

⚠ Rule 13 governs the output: a shortfall only counts when a **better legal line
existed**, so turns where we took everything available, or where nothing was
available, are reported separately and are not misplays.

⚠ Rule 14 governs what you do with it: a shortfall is a SIZE, not a mandate.
Multiply frequency by prizes-per-instance before writing anything.

    python -X utf8 scripts/p14_prize_audit.py --matches 150
    python -X utf8 scripts/p14_prize_audit.py --replays replays/submission_optv3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from sa import cards as cdb  # noqa: E402
from sa.targeting import best_damage  # noqa: E402

MAIN, SWITCH = 0, 3
OPT_PLAY, OPT_ATTACK = 7, 13
SNIPE = 30
BOSS_ORDERS = 1182
US = "Scio"


def _active(pl):
    a = pl.get("active")
    return a[0] if a and a[0] is not None else None


def _prize_if_ko(attacker, mypl, oppl, tgt) -> int:
    if tgt is None or tgt.get("hp") is None:
        return 0
    if best_damage(attacker, mypl, oppl, tgt) < tgt["hp"]:
        return 0
    return cdb.prize_value(tgt["id"])


def _snipe_prize(oppl, exclude=None) -> int:
    best = 0
    for i, pk in enumerate(oppl.get("bench") or []):
        if pk is None or pk.get("hp") is None or i == exclude:
            continue
        if pk["hp"] <= SNIPE:
            best = max(best, cdb.prize_value(pk["id"]))
    return best


def _available(state, me):
    """(best prizes obtainable this turn, how). Arithmetic, not search."""
    try:
        mypl, oppl = state["players"][me], state["players"][1 - me]
    except (KeyError, IndexError, TypeError):
        return 0, "-"
    act = _active(mypl)
    opp_act = _active(oppl)
    if act is None or opp_act is None:
        return 0, "-"
    attack_now = _prize_if_ko(act, mypl, oppl, opp_act) + _snipe_prize(oppl)
    best, how = attack_now, "attack now"
    for i, pk in enumerate(oppl.get("bench") or []):
        if pk is None:
            continue
        got = _prize_if_ko(act, mypl, oppl, pk)
        if got:
            tot = got + _snipe_prize(oppl, exclude=i)
            if tot > best:
                best, how = tot, "drag then attack"
    return best, how


class Audit:
    def __init__(self):
        self.turns = Counter()
        self.short = Counter()
        self.lost = 0
        self.egs: list[str] = []
        self.errs = Counter()


def _walk(vis, ours, rep: Audit, tag: str):
    """One game: group selects into our turns, price each, compare."""
    cur = None
    for v in vis:
        obs = v.get("obs")
        if not obs or not obs.get("current") or not obs.get("select"):
            continue
        st, sel = obs["current"], obs["select"]
        if st.get("result", -1) != -1:
            continue
        me = st.get("yourIndex")
        if me not in ours:
            continue
        key = (st.get("turn"), me)
        if cur is None or cur["key"] != key:
            if cur is not None:
                _close(cur, rep, tag)
            avail, how = _available(st, me)
            cur = {"key": key, "avail": avail, "how": how, "took": 0,
                   "turn": st.get("turn"), "attacked": False, "boss": False,
                   "state": st, "me": me}
        act = v.get("selected")
        if act is None:
            act = v.get("action")
        opts = sel.get("option") or []
        if isinstance(act, list) and act:
            i = act[0]
            if isinstance(i, int) and 0 <= i < len(opts):
                o = opts[i]
                if sel.get("context") == MAIN and o.get("type") == OPT_ATTACK:
                    cur["attacked"] = True
                if sel.get("context") == SWITCH and len(opts) >= 2:
                    cur["boss"] = True
        # prizes taken this turn = drop in our remaining prize pile
        try:
            cur["took"] = max(cur["took"],
                              6 - len(st["players"][me]["prize"]))
        except Exception:  # noqa: BLE001
            pass
    if cur is not None:
        _close(cur, rep, tag)


def _close(cur, rep: Audit, tag: str):
    avail = cur["avail"]
    if avail <= 0:
        rep.turns["no prizes were available"] += 1
        return
    if not cur["attacked"]:
        rep.turns["prizes available, WE DID NOT ATTACK"] += 1
        rep.short["did not attack at all"] += 1
        rep.lost += avail
        if len(rep.egs) < 10:
            rep.egs.append(f"{tag} t{cur['turn']}: {avail}p available "
                           f"({cur['how']}) and we did not attack")
        return
    rep.turns["prizes available and we attacked"] += 1
    # We cannot read post-attack prize deltas reliably per turn, so the honest
    # signal is the SHAPE: did we take the line that priced highest?
    if cur["how"] == "attack now" and cur["boss"]:
        rep.short["dragged when ATTACKING NOW priced higher"] += 1
        rep.lost += 1
        if len(rep.egs) < 10:
            rep.egs.append(f"{tag} t{cur['turn']}: attack-now was worth "
                           f"{avail}p and we spent Boss's Orders instead")
    elif cur["how"] == "drag then attack" and not cur["boss"]:
        rep.short["did NOT drag when the drag priced higher"] += 1
        rep.lost += 1
        if len(rep.egs) < 10:
            rep.egs.append(f"{tag} t{cur['turn']}: a drag was worth {avail}p "
                           f"and we attacked into the Active instead")


def from_replays(d: Path, rep: Audit):
    for path in sorted(d.glob("*.json")):
        if path.name == "manifest.json":
            continue
        try:
            js = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            rep.errs["unparseable"] += 1
            continue
        if not isinstance(js, dict):
            rep.errs["bare step-array"] += 1
            continue
        names = (js.get("info") or {}).get("TeamNames") or []
        if US not in names:
            continue
        ours = {i for i, n in enumerate(names) if n == US}
        vis = js["steps"][0][0].get("visualize") or []
        _walk(vis, ours, rep, path.stem)


def from_arena(matches: int, agent: str, rep: Audit):
    import arena
    from ptcg.env import harness
    _, deck_a = arena.resolve_deck("grimmsnarl")
    _, deck_b = arena.resolve_deck("lucario_v10")
    _, inner = arena.build_agent(agent, deck_a)
    _, opp = arena.build_agent("rule:v10,noS", deck_b)

    frames: list[dict] = []

    class Rec:
        def __call__(self, obs):
            p = inner(obs)
            if obs.get("select") is not None:
                frames.append({"obs": obs, "selected": list(p) if p else []})
            return p

    for _m in range(matches):
        frames.clear()
        harness.play_game(Rec(), opp, list(deck_a), list(deck_b))
        _walk(frames, {0, 1}, rep, f"g{_m}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=150)
    ap.add_argument("--agent", default="bc")
    ap.add_argument("--replays", default=None,
                    help="audit a real replay dump instead of the arena")
    args = ap.parse_args()

    rep = Audit()
    if args.replays:
        from_replays(Path(args.replays), rep)
        src = args.replays
    else:
        from_arena(args.matches, args.agent, rep)
        src = f"{args.matches} arena games as {args.agent}"

    tot = sum(rep.turns.values())
    print(f"\n=== prize audit: {src} ({tot} of our turns) ===")
    for k, v in rep.turns.most_common():
        print(f"  {k:<44}{v:>6}{v/max(tot,1):>8.1%}")

    n = sum(rep.short.values())
    live = rep.turns["prizes available, WE DID NOT ATTACK"] + \
        rep.turns["prizes available and we attacked"]
    print(f"\n--- SHORTFALLS (turns where a better priced line existed) "
          f"n={n} of {live} live turns ---")
    if not n:
        print("  none -- every live turn took the highest-priced line")
    for k, v in rep.short.most_common():
        print(f"  {k:<44}{v:>6}{v/max(live,1):>8.1%}")
    for e in rep.egs:
        print(f"    eg {e}")

    if live:
        print(f"\n  prizes left on the table: ~{rep.lost} over {live} live "
              f"turns = {rep.lost/live:.2f}/turn")
        print("  ⚠ rule 14: multiply by frequency before writing a rule, and")
        print("     check the cheap version of the play is not already happening.")
    if rep.errs:
        print("\nerrors:")
        for k, v in rep.errs.most_common(4):
            print(f"  {v:>5}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
