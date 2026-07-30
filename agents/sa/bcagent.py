"""Policy-only agent: pure behavioral clone, near-instant decisions."""
from __future__ import annotations

import sys
import traceback

from . import policynet, targeting


class PolicyAgent:
    def __init__(self, decklist: list[int], net_path: str | None = None,
                 chip_targeting: bool = True, energy_spread: bool = True,
                 drag_target: bool = False, boss_converts: bool = False,
                 drag_high_hp: bool = False, boss_veto: bool = False,
                 counter_source: bool = True, chip_wall_defer: bool = True):
        self.decklist = list(decklist)
        # An explicit net lets two candidate policies play each other inside
        # ONE arena process. Comparing them via a third opponent instead needs
        # ~2x the games for the same resolution, and the module-level
        # policynet.get() singleton cannot hold two nets at once.
        self.net = policynet.load(net_path) if net_path else None
        # The net cannot see option HP at all (see targeting.py), so it aims
        # chip damage at chance. Per-instance so the two sides of an A/B can
        # differ inside one process.
        self.chip_targeting = chip_targeting
        # Same blindness on the other side of the board: no attached-energy
        # count per option, so it stacks a dead second {D} on one Munkidori.
        self.energy_spread = energy_spread
        # Boss's Orders: which benched Pokemon to drag, and when the drag is
        # worth the Supporter. Both need damage-vs-HP arithmetic, and both
        # default OFF: together they measured 0.452 [0.435, 0.470] over 3000
        # mirror games. Whatever the per-rule isolation says, `_A(_deck)` in
        # the submission's main.py takes these defaults -- so a rule turns on
        # here only once it has cleared 0.5 on its own.
        self.drag_target = drag_target
        self.boss_converts = boss_converts
        # `drag_high_hp` only reorders the KO-able group inside drag_target, so
        # it does nothing unless drag_target is on too.
        self.drag_high_hp = drag_high_hp
        # The third Boss's Orders intervention (P5b): suppress the play when
        # their bench holds nothing we can KO -- 32.4% of our plays. Off until
        # its own A/B clears 0.5, same discipline as the two above.
        self.boss_veto = boss_veto
        # Adrena-Brain's source pick: same HP blindness, and the source caps
        # how many counters the ability can move at all. ON by default -- it
        # cleared 0.5 alone (0.534 [0.513, 0.556] n=2000 mirror) and an
        # independent opponent agreed (0.626 [0.604, 0.647] vs rule:v10,noS
        # against 0.593 for a bare bc). `bc:<label>,noSrc` turns it off.
        self.counter_source = counter_source
        # The matchup branch (2026-07-30): `chip_target` is worth +0.077 in the
        # mirror and **-0.126 against `rule:crustle`**, because "kill what dies
        # to 30" farms Dwebbles while the undamageable wall survives. This defers
        # the select to the net whenever their Active is a wall.
        #
        # ON by default -- it cleared its bar on the anchor that motivated it
        # (0.663 [0.642, 0.684] vs 0.559 [0.537, 0.581] for unconditional
        # chip_target, n=2000 each vs `rule:crustle`) and it cannot fire in the
        # matchups where chip_target pays, so the mirror is untouched by
        # construction -- confirmed at 0.521 [0.490, 0.552] n=1000, containing
        # 0.5. `bc:<label>,noWall` turns it off. See report/EVIDENCE.md §8c.
        self.chip_wall_defer = chip_wall_defer

    def __call__(self, obs: dict) -> list[int]:
        try:
            if obs.get("select") is None:
                return list(self.decklist)
            sel = obs["select"]
            n = len(sel.get("option") or [])
            mn = sel.get("minCount", 0)
            mx = sel.get("maxCount", 0)
            if n == 0 or mx == 0:
                return []
            if mn == mx == n:
                return list(range(n))
            want = max(min(mn, mx, n), 1)
            if self.chip_targeting:
                order = targeting.chip_target(obs, self.chip_wall_defer)
                if order is not None:
                    return order[:want]
            if self.drag_target:
                order = targeting.drag_target(obs, self.drag_high_hp)
                if order is not None:
                    return order[:want]
            if self.boss_converts:
                order = targeting.boss_converts(obs)
                if order is not None:
                    return order[:want]
            net = self.net or policynet.get()
            if net is None:
                return list(range(mn))
            picked = net.choose(obs)
            if self.boss_veto:
                # lazy: the full ranking costs a second forward pass, and the
                # veto fires only when the net's top pick is Boss's Orders
                fixed = targeting.boss_veto(
                    obs, list(picked), lambda: targeting.full_rank(net, obs))
                if fixed is not None:
                    return fixed
            if self.counter_source:
                fixed = targeting.counter_source(
                    obs, list(picked), lambda: targeting.full_rank(net, obs))
                if fixed is not None:
                    return fixed
            if self.energy_spread:
                fixed = targeting.energy_spread(obs, list(picked))
                if fixed is not None:
                    return fixed
            return picked
        except Exception:
            traceback.print_exc(file=sys.stderr)
            try:
                return list(range((obs.get("select") or {}).get("minCount", 0)))
            except Exception:
                return []
