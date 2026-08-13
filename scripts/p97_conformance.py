#!/usr/bin/env python
"""A BEHAVIOURAL CONFORMANCE HARNESS -- rung 0 for any from-scratch policy.

**Why this exists.** E32 (§8ck) built a coherent plan-conditioned policy and
found out it was broken only after a full arena ladder came back at **0.035**
against a bar of 0.25. Every number that diagnosed it afterwards was a per-game
behavioural statistic available in minutes: it declined the attack at 59% of the
selects that offered one, evolved its 320HP attacker **0.84 times a game**, and
ended 56% of its turns without attacking, running 11.4 turns / 146 selects
against a normal ~318. ⇒ **an agent that does that is broken on its face and
should never have reached the arena.** This is the two-stage funnel §8ak's
"the arena is the only instrument" implies but never had.

⛔ **THIS IS A FLOOR, NOT A TARGET, AND THE DISTINCTION IS THE WHOLE POINT.**
This project's graveyard is built on conformity metrics: §8r, and the standing
result that **agreement with the field never measured strength** (we match the
top players on every eval and sit ~976). A harness that scored "closeness to the
clone" would be one more of those. ⇒ **the band for each metric is the ENVELOPE
OF AGENTS THAT DEMONSTRABLY WORK**, not a neighbourhood of the clone. Passing
means "inside the range of behaviours competent agents actually exhibit". It
never means good, and no result here may be reported as evidence of strength.

**The controls are the instrument.** A band is only admissible if:

  ✅ every competent agent passes it -- `bc` on grimmsnarl plus FOUR independent
     rule agents on FOUR different decks, none of them written by us. A metric
     one of them fails is measuring style, and is dropped rather than widened
     after the fact.
  🔴 `random` fails it, and `plan:pure` (E32, known broken at 0.035) fails it.
     A harness that passes a known-broken agent has no power and is worthless.

⚠ **Metrics are DECK-AGNOSTIC on purpose.** No rule agent in the repo is tuned
for grimmsnarl (`DECK_MODULE` has none), so every cross-agent control plays a
different 60. Only structural quantities -- does it attack when it can, does it
evolve, does it attach, how long is a turn -- are comparable across decks, and
those are exactly the ones that caught E32.

    python -X utf8 scripts/p97_conformance.py --games 60
    python -X utf8 scripts/p97_conformance.py --games 60 --only "plan:pure"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", "decks", "."):
    p = str(ROOT / sub) if sub != "." else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402
sdk.load()

from ptcg.env import harness  # noqa: E402

MAIN = 0
O_PLAY, O_ATTACH, O_EVOLVE, O_ABILITY = 7, 8, 9, 10
O_DISCARD, O_RETREAT, O_ATTACK, O_END = 11, 12, 13, 14

# (spec, deck module, role). "ref" sets the bands; "ok" must pass; "bad" must fail.
ROSTER = [
    # 🔴 `net=` IS PINNED. A bare `bc` spec passes no net path, so PolicyAgent
    # falls back to the `policynet.get()` singleton -- `sa/policy_net.npz`
    # #a25b904d, the **v2** clone -- and the reference agent defining these
    # bands would be three generations behind the #4790c469 that actually
    # ships. Rule 20 records it in the archived name either way; pinning it
    # means the band describes the agent we care about.
    ("bc:v5,net=out/policy_v5_s2.npz", "grimmsnarl", "ref"),
    ("rule:v10",       "lucario_v10",      "ok"),
    ("rule:dragapult", "dragapult_ex",     "ok"),
    ("rule:lucario",   "mega_lucario_ex",  "ok"),
    ("rule:archaludon", "archaludon_ex",   "ok"),
    ("plan:pure",      "grimmsnarl",       "bad"),
    ("random",         "grimmsnarl",       "bad"),
]


class Probe:
    """Wraps an agent and counts what it DID. Returns its pick untouched."""

    def __init__(self, inner):
        self.inner = inner
        self.games: list[dict] = []
        self._reset()

    def _reset(self) -> None:
        self.c = {k: 0 for k in
                  ("selects", "main", "play", "attach", "evolve", "ability",
                   "retreat", "attack", "end", "end_attack_avail",
                   "turns", "turns_attack_offered", "turns_attacked")}
        self._turn = None
        self._t_offered = False
        self._t_attacked = False
        self._snap = None
        self._snaps: list[dict] = []

    @staticmethod
    def _read(cur: dict) -> dict | None:
        """Board state, in DECK-AGNOSTIC form.

        🔴 These four exist because E32's real failure was not visible in the
        option-type counts at all. It filled its bench with the support engine
        and never built the win condition -- it evolved its 320HP attacker
        **0.84 times a game** -- and "how much of my board is an evolution"
        plus "how much damage is on the opponent's board" say that directly,
        while `play`/`evolve` counts only say how many buttons were pressed.
        Damage is a FRACTION of printed HP so a 110HP deck and a 320HP deck
        are on one scale.
        """
        try:
            me, opp = cur["yourIndex"], 1 - cur["yourIndex"]
            mine, theirs = cur["players"][me], cur["players"][opp]
            board = list(mine.get("active") or []) + list(mine.get("bench") or [])
            ob = list(theirs.get("active") or []) + list(theirs.get("bench") or [])
            if not board or not ob:
                return None
            tot = sum(p.get("maxHp") or 0 for p in ob)
            dmg = sum((p.get("maxHp") or 0) - (p.get("hp") or 0) for p in ob)
            act = (mine.get("active") or [{}])[0]
            return {
                "bench_end": float(len(mine.get("bench") or [])),
                "evolved_frac": sum(1.0 for p in board
                                    if p.get("preEvolution")) / len(board),
                "opp_damage_frac": (dmg / tot) if tot else 0.0,
                "active_energy": float(len(act.get("energies") or [])),
                # 🔴 OUR OWN pile. A prize is drawn from the pile of the player
                # who scored the KO, so `theirs["prize"]` shrinking counts the
                # prizes taken AGAINST us -- the first version read that and
                # would have scored an agent on how well its opponent played.
                # Invisible in a mirror (symmetric) and glaring against random,
                # which is how it was caught.
                "prizes_taken": 6.0 - len(mine.get("prize") or []),
            }
        except Exception:
            return None

    def _close_turn(self) -> None:
        if self._turn is None:
            return
        self.c["turns"] += 1
        if self._t_offered:
            self.c["turns_attack_offered"] += 1
            if self._t_attacked:
                self.c["turns_attacked"] += 1
        # one snapshot per turn -- the LAST state we saw in it -- so turns with
        # many selects do not outvote turns with few.
        if self._snap is not None:
            self._snaps.append(self._snap)
        self._t_offered = self._t_attacked = False
        self._snap = None

    def new_game(self) -> None:
        self._reset()

    def close_game(self) -> None:
        self._close_turn()
        c = self.c
        s = self._snaps
        board = {k: (sum(x[k] for x in s) / len(s) if s else 0.0)
                 for k in ("bench_end", "evolved_frac", "opp_damage_frac",
                           "active_energy")}
        # prizes are cumulative, so the LAST turn's reading is the game's, not
        # the mean of the running total.
        board["prizes_taken"] = s[-1]["prizes_taken"] if s else 0.0
        self.games.append({
            **board,
            "selects": c["selects"],
            "turns": c["turns"],
            "selects_per_turn": c["selects"] / max(1, c["turns"]),
            "play": c["play"], "attach": c["attach"], "evolve": c["evolve"],
            "ability": c["ability"], "retreat": c["retreat"],
            "attack": c["attack"],
            "attack_per_turn": c["attack"] / max(1, c["turns"]),
            # 🔴 THE E32 DETECTOR. Of our turns where an attack was offered at
            # some MAIN select, what fraction ended without one being taken?
            "turns_offered": c["turns_attack_offered"],
            "attack_take_rate": (c["turns_attacked"] /
                                 max(1, c["turns_attack_offered"])),
            # The same failure seen at the select rather than the turn: how
            # often did we END with an attack sitting in the option list?
            "end_decline_rate": (c["end_attack_avail"] / max(1, c["end"])),
        })
        self._reset()

    def __call__(self, obs: dict):
        picked = self.inner(obs)
        try:
            self._observe(obs, picked)
        except Exception:
            pass
        return picked

    def _observe(self, obs: dict, picked) -> None:
        sel = obs.get("select") or {}
        cur = obs.get("current") or {}
        if not sel or cur.get("result", -1) != -1:
            return
        self.c["selects"] += 1
        if sel.get("context") != MAIN:
            return
        self.c["main"] += 1
        turn = cur.get("turn")
        if turn != self._turn:
            self._close_turn()
            self._turn = turn
        snap = self._read(cur)
        if snap is not None:
            self._snap = snap
        opts = sel.get("option") or []
        if any(o.get("type") == O_ATTACK for o in opts):
            self._t_offered = True
        if not picked or len(picked) != 1:
            return
        i = int(picked[0])
        if not (0 <= i < len(opts)):
            return
        t = opts[i].get("type")
        for key, code in (("play", O_PLAY), ("attach", O_ATTACH),
                          ("evolve", O_EVOLVE), ("ability", O_ABILITY),
                          ("retreat", O_RETREAT), ("attack", O_ATTACK),
                          ("end", O_END)):
            if t == code:
                self.c[key] += 1
        if t == O_ATTACK:
            self._t_attacked = True
        if t == O_END and any(o.get("type") == O_ATTACK for o in opts):
            self.c["end_attack_avail"] += 1


METRICS = ["selects", "turns", "selects_per_turn", "attack", "attack_per_turn",
           "attack_take_rate", "end_decline_rate", "evolve", "attach",
           "ability", "play", "retreat",
           # board-state metrics -- see Probe._read
           "bench_end", "evolved_frac", "opp_damage_frac", "active_energy",
           "prizes_taken"]
# Metrics where SMALL is broken vs where LARGE is broken is not assumed: the
# band is two-sided and comes from the observed envelope.

# 🔴 An absolute floor on the band half-width, per metric, and it is not
# cosmetic. The smoke run put `end_decline_rate` at exactly 0.000 for all five
# competent agents, so an envelope-derived band would have been [0, 0] -- and
# any new policy that ever ends one turn with an attack still on the table
# would "fail rung 0". A band that tight is a style metric wearing a health
# metric's name. Rates get 0.05, counts get 0.5.
FLOOR = {"selects": 8.0, "turns": 1.0, "selects_per_turn": 1.5,
         "attack": 0.5, "attack_per_turn": 0.05, "attack_take_rate": 0.05,
         "end_decline_rate": 0.05, "evolve": 0.5, "attach": 0.5,
         "ability": 0.5, "play": 0.5, "retreat": 0.5,
         "bench_end": 0.4, "evolved_frac": 0.05, "opp_damage_frac": 0.04,
         "active_energy": 0.4, "prizes_taken": 0.5}


def deck_ids(name: str) -> list[int]:
    mod = __import__(name)
    out: list[int] = []
    for cid, k in mod.DECKLIST.items():
        out += [cid] * k
    return out


def run(spec: str, deck_name: str, games: int, opponent: str = "self") -> dict:
    """`opponent="self"` = mirror. Any other spec = a fixed reference foe.

    🔴 **The mirror frame is weak and this argument exists because of it.**
    Two equally broken agents still take prizes off each other, still put
    damage on a board, still finish a game -- so E32's policy sits INSIDE the
    band on every board-state metric in self-play, and its real numbers
    (0.84 evolutions/game) were only ever visible against a competent
    opponent. ⚠ `bc` cannot be that opponent for the cross-deck controls,
    because it is strong on grimmsnarl and untested on the other four 60s,
    which would compare a candidate facing a strong foe against controls
    facing a weak one. `random` is uniformly weak on every deck, so it is the
    one reference that keeps the cross-deck controls admissible.
    """
    from arena import build_agent

    deck = deck_ids(deck_name)
    name, a = build_agent(spec, list(deck), deck_name)
    _, b = build_agent(opponent if opponent != "self" else spec,
                       list(deck), deck_name)
    pr = Probe(a)
    t0 = time.time()
    for i in range(games):
        pr.new_game()
        if i % 2 == 0:
            harness.play_game(pr, b, list(deck), list(deck))
        else:
            harness.play_game(b, pr, list(deck), list(deck))
        pr.close_game()
    out = {"spec": spec, "name": name, "deck": deck_name, "games": len(pr.games),
           "secs": round(time.time() - t0, 1)}
    for m in METRICS:
        vals = [g[m] for g in pr.games]
        n = max(1, len(vals))
        mu = sum(vals) / n
        out[m] = mu
        # The SE enters the band. Without it the envelope of five sampled
        # means is narrower than the envelope of the five true means, and the
        # harness would fail honest policies for being unlucky.
        var = sum((v - mu) ** 2 for v in vals) / max(1, n - 1)
        out[f"se_{m}"] = (var / n) ** 0.5
    out["_rows"] = pr.games
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--only", default=None, help="run one spec and score it")
    ap.add_argument("--opponent", default="self",
                    help="'self' for the mirror, or a fixed reference spec")
    ap.add_argument("--bands", default="out/conformance_bands.json")
    args = ap.parse_args()

    bands_path = ROOT / args.bands

    if args.only:
        if not bands_path.exists():
            sys.exit(f"no bands at {bands_path}; run without --only first")
        bands = json.loads(bands_path.read_text(encoding="utf-8"))["bands"]
        deck = "grimmsnarl"
        r = run(args.only, deck, args.games, args.opponent)
        print(f"\n{r['name']} on {deck}  ({r['games']} games, {r['secs']}s)\n")
        fails = []
        for m in METRICS:
            lo, hi = bands[m]
            ok = lo <= r[m] <= hi
            if not ok:
                fails.append(m)
            print(f"  {m:>18}  {r[m]:>9.3f}   band [{lo:.3f}, {hi:.3f}]   "
                  f"{'ok' if ok else '🔴 OUT'}")
        print()
        if fails:
            print(f"🔴 FAILS rung 0 on {len(fails)} metric(s): "
                  f"{', '.join(fails)}")
            return 1
        print("✅ passes rung 0. ⛔ This means NOT-OBVIOUSLY-BROKEN. It is not "
              "evidence of\n   strength and may not be reported as any.")
        return 0

    results = []
    for spec, deck, role in ROSTER:
        print(f"running {spec} on {deck} ...", flush=True)
        try:
            r = run(spec, deck, args.games, args.opponent)
        except Exception as e:
            print(f"  🔴 {spec} FAILED TO RUN: {type(e).__name__}: {e}")
            continue
        r["role"] = role
        results.append(r)
        print(f"  {r['name']:<22} {r['secs']:>6.1f}s", flush=True)

    good = [r for r in results if r["role"] in ("ref", "ok")]
    if len(good) < 3:
        sys.exit("too few competent agents ran to define an envelope")

    # The band is the envelope of agents that work, widened 20%. NOT a
    # neighbourhood of the clone -- see this file's docstring.
    bands = {}
    for m in METRICS:
        vals = [r[m] for r in good]
        lo, hi = min(vals), max(vals)
        se = max(r[f"se_{m}"] for r in good)
        # 25% of the observed range (five agents do not span the space of
        # competent play) + 10% of scale + 2 SE of sampling noise, never
        # below the per-metric floor.
        pad = max(0.25 * (hi - lo) + 0.10 * max(abs(lo), abs(hi)) + 2 * se,
                  FLOOR[m])
        bands[m] = [lo - pad, hi + pad]

    hdr = f"{'agent':<24}{'deck':<18}" + "".join(f"{m[:11]:>12}" for m in METRICS)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        mark = {"ref": "ref ", "ok": "ok  ", "bad": "BAD "}[r["role"]]
        print(f"{mark}{r['name']:<20}{r['deck']:<18}"
              + "".join(f"{r[m]:>12.3f}" for m in METRICS))

    print("\nband (envelope of competent agents, +20%):")
    for m in METRICS:
        print(f"  {m:>18}  [{bands[m][0]:>9.3f}, {bands[m][1]:>9.3f}]")

    print("\ncontrols:")
    ok_all = True
    for r in results:
        fails = [m for m in METRICS if not (bands[m][0] <= r[m] <= bands[m][1])]
        if r["role"] in ("ref", "ok"):
            good_ok = not fails
            ok_all &= good_ok
            print(f"  ✅ {r['name']:<20} passes"
                  if good_ok else
                  f"  🔴 {r['name']:<20} FAILS {fails} -- that metric is "
                  f"measuring STYLE, drop it")
        else:
            has_power = bool(fails)
            ok_all &= has_power
            print(f"  ✅ {r['name']:<20} caught on {fails}"
                  if has_power else
                  f"  🔴 {r['name']:<20} PASSES and is known broken -- "
                  f"the harness has NO POWER")

    out = {"games": args.games, "opponent": args.opponent, "bands": bands,
           "results": [{k: v for k, v in r.items() if k != "_rows"}
                       for r in results]}
    bands_path.parent.mkdir(parents=True, exist_ok=True)
    bands_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {bands_path}")
    print("\n" + ("✅ harness is admissible: every competent agent passes and "
                  "every known-broken\n   one is caught."
                  if ok_all else
                  "🔴 harness is NOT admissible as it stands -- see the failing "
                  "controls above."))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
