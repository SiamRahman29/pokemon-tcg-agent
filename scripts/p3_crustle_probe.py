"""Does damage actually reach Crustle? (HANDOFF §3.2, the premise probe)

**The claim that all Crustle deck work rests on, never once checked in-engine:**
Mysterious Rock Inn is an ABILITY on Crustle (345) that prevents damage from the
opponent's {ex} attacks. Marnie's Grimmsnarl ex is `ex=True`, so Shadow Bullet
should deal **zero**. The proposed out is that Adrena-Brain and Freezing Shroud
*move or place damage counters*, which is not "damage done by an attack", so
they should go through. Our card db carries no ability text for 345
(`abilities: None`), so this cannot be settled by reading -- only by playing.

**If counters do not bypass the prevention, the entire passive-damage line is
dead and no decklist work should happen** (ROADMAP Track C step 2).

This reads the engine's own event log rather than reconstructing state, because
the log makes the exact distinction the question needs:

    {'type': 15, 'playerIndex': 1, 'cardId': 345, 'attackId': 479}   # an attack
    {'type': 16, 'playerIndex': 0, 'cardId': 860, 'value': -120,
     'putDamageCounter': False}                                       # HP change

`type 16` carries **`putDamageCounter`** -- True for placed/moved counters, False
for attack damage -- `playerIndex` says whose Pokemon changed, `cardId` says
which, and `value` is negative for damage / positive for healing.

**This is a CENSUS, not an attribution**, and that is deliberate. The first
version of this probe snapshotted `len(logs)` when we committed an action and
attributed everything after it -- and read **0.0 damage in every bucket
including "no Crustle in play"**, which cannot be true of an agent that wins 56%
of these games. Cause: **`obs['logs']` is a per-observation DELTA, not a
cumulative log** (observed lengths `[0, 0, 48, 14, 3, 1, ...]`, non-monotonic),
so the offsets were nonsense. Rather than repair the attribution, note that the
question does not need it: "does attack damage onto a Crustle ever land, and do
damage counters onto a Crustle ever land" is answered by tallying the events
themselves. Fewer moving parts, and nothing to get subtly wrong (rule 9).

    python -X utf8 scripts/p3_crustle_probe.py --matches 60
"""
from __future__ import annotations

import argparse
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

from ptcg.env import harness  # noqa: E402
import arena  # noqa: E402

MAIN = 0
OPT_ABILITY, OPT_ATTACK = 10, 13
LOG_ATTACK, LOG_HP = 15, 16
_ACTIVE, _BENCH = 4, 5
CRUSTLE, DWEBBLE, MUNKIDORI, FROSLASS = 345, 344, 112, 104
GRIMMSNARL_EX = 648


class CrustleProbe:
    """Tallies every HP-change event by (whose Pokemon, which card, counter?)."""

    def __init__(self, inner):
        self.inner = inner
        # key -> list of damage values (positive = damage dealt to that Pokemon)
        self.events: dict[tuple, list[int]] = defaultdict(list)
        self.heals: dict[tuple, list[int]] = defaultdict(list)
        self.attacks_used = Counter()
        self.notes = Counter()

    def __call__(self, obs):
        try:
            # ⚠ obs['logs'] is the DELTA since our last observation, so every
            # event is seen exactly once by concatenating deltas -- never index
            # into it as if it were the whole game's log.
            me = (obs.get("current") or {}).get("yourIndex")
            for e in obs.get("logs") or []:
                if not isinstance(e, dict):
                    continue
                if e.get("type") == LOG_ATTACK:
                    who = "ours" if e.get("playerIndex") == me else "theirs"
                    self.attacks_used[
                        f"{who}: card {e.get('cardId')} attack {e.get('attackId')}"] += 1
                    continue
                if e.get("type") != LOG_HP:
                    continue
                owner = "ours" if e.get("playerIndex") == me else "theirs"
                key = (owner, e.get("cardId"), bool(e.get("putDamageCounter")))
                val = e.get("value") or 0
                # ⚠ A PREVENTED attack logs as value == 0, which is the single
                # most important event class here -- it must be counted as a
                # damage event of size 0, not filed with the heals. (An earlier
                # version routed `val >= 0` to heals and so reported "zeros
                # 0/1110" for every row, hiding exactly what we came to measure.)
                if val > 0:
                    self.heals[key].append(val)
                else:
                    self.events[key].append(-val)
        except Exception as exc:  # noqa: BLE001
            self.notes[f"ERR {type(exc).__name__}: {exc}"] += 1
        return self.inner(obs)

    def game_over(self) -> None:
        pass


def _at(state, player, area, index):
    try:
        pl = state["players"][player]
        if area == _ACTIVE:
            act = pl["active"]
            return act[0] if act and act[0] is not None else None
        if area == _BENCH:
            b = pl["bench"]
            return b[index] if 0 <= index < len(b) else None
    except (KeyError, IndexError, TypeError):
        return None
    return None


NAMES = {345: "Crustle", 344: "Dwebble", 117: "Cornerstone Ogerpon ex",
         756: "Mega Kangaskhan ex", 112: "Munkidori", 104: "Froslass",
         860: "Snorunt", 646: "Marnie's Impidimp", 647: "Marnie's Morgrem",
         648: "Marnie's Grimmsnarl ex"}


def _name(cid) -> str:
    return NAMES.get(cid, f"#{cid}")


def _table(title: str, data: dict[tuple, list[int]], owner: str) -> None:
    print(f"\n=== {title} ===")
    rows = [(k, v) for k, v in data.items() if k[0] == owner]
    if not rows:
        print("  (nothing recorded)")
        return
    print(f"  {'target':<26} {'kind':<16} {'n':>6} {'mean':>8} {'zeros':>12}")
    for (_, cid, is_ctr), vals in sorted(rows, key=lambda kv: -len(kv[1])):
        n = len(vals)
        zero = sum(1 for v in vals if v == 0)
        kind = "damage COUNTER" if is_ctr else "attack damage"
        print(f"  {_name(cid):<26} {kind:<16} {n:>6} "
              f"{sum(vals) / n:>8.1f} {f'{zero}/{n}':>12}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=60)
    ap.add_argument("--agent", default="bc")
    ap.add_argument("--opponent", default="rule:crustle")
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--deck-b", default="crustle_v1")
    args = ap.parse_args()

    _, deck_a = arena.resolve_deck(args.deck)
    _, deck_b = arena.resolve_deck(args.deck_b)
    _, agent_a = arena.build_agent(args.agent, deck_a)
    _, agent_b = arena.build_agent(args.opponent, deck_b)
    probe = CrustleProbe(agent_a)

    for m in range(args.matches):
        if m % 2 == 0:
            harness.play_game(probe, agent_b, list(deck_a), list(deck_b))
        else:
            harness.play_game(agent_b, probe, list(deck_b), list(deck_a))
        probe.game_over()

    print(f"\n{args.agent} [{args.deck}] vs {args.opponent} [{args.deck_b}], "
          f"{args.matches} games")
    _table("Damage landing on THEIR Pokemon (i.e. dealt by us)",
           probe.events, "theirs")
    _table("Damage landing on OUR Pokemon (i.e. dealt by them)",
           probe.events, "ours")

    print("\n--- attacks used ---")
    for k, v in probe.attacks_used.most_common(10):
        print(f"  {v:>6}  {k}")

    # The verdict, computed rather than eyeballed.
    atk_on_crustle = probe.events.get(("theirs", CRUSTLE, False), [])
    ctr_on_crustle = probe.events.get(("theirs", CRUSTLE, True), [])
    print("\n=== VERDICT ===")
    if atk_on_crustle:
        nz = sum(1 for v in atk_on_crustle if v > 0)
        z = len(atk_on_crustle) - nz
        print(f"  ATTACK damage events onto Crustle: n={len(atk_on_crustle)}, "
              f"ZEROED {z} ({z / len(atk_on_crustle):.1%}), "
              f"landed {nz} (mean of those "
              f"{(sum(atk_on_crustle) / nz if nz else 0):.1f})")
        print("  -> Mysterious Rock Inn "
              + ("PREVENTS every attack we made" if nz == 0
                 else f"prevents most but NOT all -- {nz} landed, so some "
                      "attacker is getting through (see 'attacks used': a "
                      "NON-ex attacker is not covered by an anti-ex ability)"))
    else:
        print("  no attack-damage events onto Crustle recorded")
    if ctr_on_crustle:
        nz = sum(1 for v in ctr_on_crustle if v > 0)
        print(f"  DAMAGE COUNTER events onto Crustle: n={len(ctr_on_crustle)}, "
              f"non-zero {nz} ({nz / len(ctr_on_crustle):.1%}), "
              f"mean {sum(ctr_on_crustle) / len(ctr_on_crustle):.1f}, "
              f"total {sum(ctr_on_crustle)}")
        print("  -> counters "
              + ("BYPASS the prevention: the passive-damage line is LIVE"
                 if nz else "are ALSO prevented: the passive-damage line is DEAD"))
    else:
        print("  no damage-counter events onto Crustle recorded")
    if probe.notes:
        print("\nerrors:")
        for k, v in probe.notes.most_common(5):
            print(f"  {v:>5}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
