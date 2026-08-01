"""Prove the day-15 recorder is a faithful, side-effect-free tap -- exactly.

**Why this file exists.** §8aa nearly lost a day to a refactor that was supposed
to be a no-op, checked with an arena A/B that read 0.503 against a previous
0.567. The arena *cannot* settle that question, and the rule bought there was:

    ⚡ When a refactor is supposed to be a no-op, prove it with an equivalence
    test, not with the noisy end-to-end instrument.

🔴 **The first version of this script got that wrong in a new way**, and the
failure is worth recording because it is the same trap one level down. It
played each game twice -- once with `recorder=None`, once with a `Recorder` --
and demanded identical action streams. All four games "failed". They failed
because **`cg.game.battle_start` takes no seed**: the engine shuffles
internally and two consecutive games diverge by select 2 whatever you do. That
test could never have passed, and had the recorder genuinely been broken the
output would have looked exactly the same. **A test that cannot fail for the
reason you are testing is not evidence** -- it is rule 9 ("a metric that never
prints is not a metric that passed") wearing a different hat.

So this script tests the claims that CAN be made exact, one game at a time:

  1. **The tap is faithful.** Wrap `game.battle_select` and require the
     arguments the ENGINE actually received to equal `recorder.action_log`
     element for element. Nothing recorded that was not played; nothing played
     that was not recorded.
  2. **The tap has no side effects on the agent's input.** Serialise each `obs`
     immediately before and after `on_select` and require equality, so a
     recorded run cannot feed an agent anything a plain run would not.
  3. **The capture is complete.** observations == selects, and the engine's
     visualize stream is one step longer (its step 0 is the pre-battle state).
  4. **The artifact round-trips**: written, re-read, and parsed by the repo's
     own replay reader (`p9_field_census.analyse`) without special-casing.

⚠ What is NOT proven here, because no seed exists to prove it with: that
`recorder=None` produces the identical *game* to the pre-edit code. That path
differs from the original only by two `is not None` guards, one local
assignment and one hoisted list comprehension -- behaviour-preserving by
inspection. Item 5 below reports the distribution of turns/selects with the
recorder on and off as a *weak* corroboration, and it is labelled weak.

    python -X utf8 scripts/p20_recorder_equivalence.py --games 4
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import harness, sdk  # noqa: E402
sdk.load()
from arena import build_agent, resolve_deck  # noqa: E402


class CheckedRecorder(harness.Recorder):
    """A Recorder that also asserts it did not touch the observation."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.mutations = 0

    def on_select(self, obs, who, action):
        before = json.dumps(obs, sort_keys=True, default=str)
        super().on_select(obs, who, action)
        if json.dumps(obs, sort_keys=True, default=str) != before:
            self.mutations += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="bc:v5,net=out/policy_v5.npz")
    ap.add_argument("--b", default="bc:v4,net=out/policy_v4.npz")
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--games", type=int, default=4)
    args = ap.parse_args()

    _, deck = resolve_deck(args.deck)
    game = sdk.game()
    print(f"  {args.a} vs {args.b} on {args.deck}, {args.games} recorded games\n")

    fails = 0
    tmp = Path("out/replays/_equivalence")
    on_stats, off_stats = [], []

    for i in range(args.games):
        # --- tap the ENGINE so we know what it was really told ---------------
        seen: list[tuple] = []
        real_select = game.battle_select

        def spy(picked, _real=real_select, _seen=seen):
            _seen.append(tuple(int(c) for c in picked))
            return _real(picked)

        game.battle_select = spy
        try:
            _, a0 = build_agent(args.a, list(deck))
            _, a1 = build_agent(args.b, list(deck))
            rec = CheckedRecorder(names=("A", "B"))
            r = harness.play_game(a0, a1, list(deck), list(deck), recorder=rec)
        finally:
            game.battle_select = real_select

        recorded = [tuple(a) for a in rec.action_log[1:]]
        n_obs = sum(1 for o in rec.obs_log if o != "")
        vis = rec.vis or []

        checks = {
            "1 tap faithful (engine args == action_log)": recorded == seen,
            "2 no obs mutation": rec.mutations == 0,
            "3a observations == selects": n_obs == r.selects,
            "3b visualize steps == selects + 1": len(vis) == r.selects + 1,
        }
        path = rec.dump(tmp / f"eq{i:03d}.json")
        try:
            from p9_field_census import analyse
            g = analyse(path, Counter(), {"A"})
            checks["4 round-trips through p9_field_census"] = g is not None
        except Exception as exc:  # noqa: BLE001
            checks["4 round-trips through p9_field_census"] = False
            print(f"      reader raised: {type(exc).__name__}: {exc}")

        ok = all(checks.values())
        fails += not ok
        print(f"  game {i}: winner={r.winner} turns={r.turns} "
              f"selects={r.selects}  {'✅' if ok else '🔴'}")
        for name, good in checks.items():
            print(f"      {'ok  ' if good else 'FAIL'}  {name}")
        if recorded != seen:
            for j, (x, y) in enumerate(zip(recorded, seen)):
                if x != y:
                    print(f"      first mismatch at select {j}: "
                          f"recorded {x} vs engine {y}")
                    break
        on_stats.append((r.turns, r.selects))

    # --- 5. weak corroboration: the on/off distributions --------------------
    for _ in range(args.games):
        _, a0 = build_agent(args.a, list(deck))
        _, a1 = build_agent(args.b, list(deck))
        r = harness.play_game(a0, a1, list(deck), list(deck))
        off_stats.append((r.turns, r.selects))

    print(f"\n  5 (WEAK -- no seed exists, so this is distributional only)")
    for lab, st in (("recorder ON ", on_stats), ("recorder OFF", off_stats)):
        t = [x for x, _ in st]
        s = [y for _, y in st]
        print(f"      {lab}: turns mean {statistics.mean(t):5.1f}  "
              f"selects mean {statistics.mean(s):6.1f}  (n={len(st)})")
    print("      ^ these should look alike; they are NOT a proof and a "
          "difference\n        here would be evidence, not the agreement.")

    print(f"\n  {args.games - fails}/{args.games} games passed every exact check")
    if fails:
        print("  🔴 FAIL")
        return 1
    print("  ✅ PASS -- the recorder is a faithful, side-effect-free tap "
          "and its\n     output is readable by the repo's existing replay tools")
    return 0


if __name__ == "__main__":
    sys.exit(main())
