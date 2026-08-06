"""Policy-only agent: pure behavioral clone, near-instant decisions."""
from __future__ import annotations

import sys
import traceback

from . import policynet, targeting

# --- health counters (day 15) -------------------------------------------------
# 🔴 WHY THIS EXISTS. `__call__`'s catch-all returns `range(minCount)` -- the
# first N options in INDEX ORDER -- and prints a traceback to stderr. On Kaggle
# that traceback goes nowhere anyone reads, so **a submission can run the
# index-order fallback on EVERY decision and look completely normal from
# outside**: it still returns legal moves, still finishes games, still gets a
# rating. §8g had to detect exactly this indirectly, by arguing from a 40.7%
# index-0 rate over 4,682 selects against the 100% a real fallback would show.
#
# These counters make it a direct read. They are free on the happy path (one
# dict increment per select, against ~1 ms of decision time) and the summary is
# ONE LINE per game, never per-decision spam.
STATS = {
    "calls": 0,          # selects seen
    "fallbacks": 0,      # times the catch-all fired -- ANY non-zero value is a bug
    "net_missing": 0,    # net failed to load: also index-order, also silent
    "deck_returns": 0,   # the pre-battle deck handshake
    "first_error": None,  # the first traceback, verbatim, for the log
}


def health_line() -> str:
    """One-line health summary -- the highest value-per-byte thing to log.

    Print this once per game from the agent wrapper; a submission log built
    from it answers "was the net actually live?" without any per-move output.
    """
    s = STATS
    bad = s["fallbacks"] + s["net_missing"]
    status = "OK" if bad == 0 else "DEGRADED"
    line = (f"[health] {status} calls={s['calls']} fallbacks={s['fallbacks']} "
            f"net_missing={s['net_missing']} deck={s['deck_returns']}")
    if s["first_error"]:
        line += f" first_error={s['first_error'][:200]!r}"
    return line


def reset_stats() -> None:
    STATS.update(calls=0, fallbacks=0, net_missing=0, deck_returns=0,
                 first_error=None)


class PolicyAgent:
    def __init__(self, decklist: list[int], net_path: str | None = None,
                 chip_targeting: bool = True, energy_spread: bool = True,
                 drag_target: bool = False, boss_converts: bool = False,
                 drag_high_hp: bool = False, boss_veto: bool = False,
                 counter_source: bool = True, chip_wall_defer: bool = True,
                 boss_prize_veto: bool = False,
                 sequencer: bool = False, seq_k: int = 8, seq_dets: int = 4,
                 seq_budget: float = 0.35, seq_reply: bool = False):
        self.decklist = list(decklist)
        # The FIFTH Boss's Orders rule: suppress the play when attacking
        # right now takes strictly more prizes than any drag can. The
        # other four picked a side in a trade and all measured null; this
        # one deletes a dominated option (EVIDENCE 6 vs 8g). Opt-in until
        # it clears the five anchors.
        self.boss_prize_veto = boss_prize_veto
        # B4: turn-level lookahead (sequencer.py). OFF by default and opt-in
        # via `bc:<label>,seq` until it clears an arena A/B -- it is an
        # experiment, not a shipped component (EVIDENCE 8m).
        self.seq = None
        if sequencer:
            from .sequencer import Sequencer
            self.seq = Sequencer(decklist, k=seq_k, dets=seq_dets,
                                 budget_s=seq_budget, reply=seq_reply)
        # An explicit net lets two candidate policies play each other inside
        # ONE arena process. Comparing them via a third opponent instead needs
        # ~2x the games for the same resolution, and the module-level
        # policynet.get() singleton cannot hold two nets at once.
        self.net = policynet.load(net_path) if net_path else None
        # 🔴 AN EXPLICIT `net=` THAT DOES NOT LOAD MUST NEVER BECOME THE
        # SINGLETON. `policynet.load` returns None on any guard failure or
        # exception rather than raising, and `__call__` below falls back to
        # `policynet.get()` -- the tracked `sa/policy_net.npz`, which is the old
        # width-496 `policy_lw2`. So before day 22, `bc:v7,net=<a net that
        # fails the dim or vocab guard>` played lw2, archived under the name of
        # the net it was ASKED for, and printed a perfectly ordinary score. A
        # whole A/B could run that way and read as a result. Demonstrated with a
        # v7 net whose vocab map was one entry short -- exactly the "rebuild the
        # corpus and a net's map is stale" hazard §8aw names -- which loaded as
        # None, was accepted by `arena.build_agent` (it checks only that the
        # path EXISTS), and would have played 496-wide lw2 against a 708-wide
        # control. Fail loudly instead: this is the fifth "plausible number, not
        # a crash" in this repo (rule 18).
        #
        # ⚠ Only the `net=` path is strict. The submission never passes
        # `net_path` -- `build_submission.py` ships the candidate AS
        # `sa/policy_net.npz` and verifies it with `policynet.load` at build
        # time -- so the shipped agent keeps its fail-soft behaviour, where
        # degrading and logging beats forfeiting a live episode.
        if net_path and self.net is None:
            raise ValueError(
                f"net {net_path!r} exists but FAILED policynet.load's guard "
                f"(feature dims, n_pool/n_attr, or the v7 vocab row count). "
                f"Refusing to fall back to the tracked sa/policy_net.npz, "
                f"which is a different net and would have scored silently.")
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
        STATS["calls"] += 1
        try:
            if obs.get("select") is None:
                STATS["deck_returns"] += 1
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
                # index order, silently, forever -- the failure §8g had to
                # infer. Counted so it can be read instead.
                STATS["net_missing"] += 1
                return list(range(mn))
            picked = net.choose(obs)
            if self.boss_veto:
                # lazy: the full ranking costs a second forward pass, and the
                # veto fires only when the net's top pick is Boss's Orders
                fixed = targeting.boss_veto(
                    obs, list(picked), lambda: targeting.full_rank(net, obs))
                if fixed is not None:
                    return fixed
            if self.boss_prize_veto:
                fixed = targeting.boss_prize_veto(
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
                    picked = fixed
            # B4 last: it overrules the clone AND the rules, because it is the
            # only component that scores a whole turn rather than one option.
            # Returns None (fall through) whenever it cannot plan safely.
            if self.seq is not None:
                planned = self.seq.plan(obs, list(picked))
                if planned is not None:
                    return planned
            return picked
        except Exception:
            STATS["fallbacks"] += 1
            if STATS["first_error"] is None:
                STATS["first_error"] = traceback.format_exc()
            traceback.print_exc(file=sys.stderr)
            try:
                return list(range((obs.get("select") or {}).get("minCount", 0)))
            except Exception:
                return []
