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
    # E3 near-tie probe (day 23). An intervention whose firing rate is not
    # printed cannot be distinguished from one that never fired -- rule 9, and
    # the reason §8am reports "off-argmax selects" next to every score.
    "flip_eligible": 0,
    "flips": 0,
}


def _option_sig(obs: dict, option: dict) -> bytes:
    """Bitwise identity of an option under the shipped encoding (§8x)."""
    from .optfeat import option_features
    import numpy as _np

    dense, card_id, attack_id, target_id = option_features(obs, option)
    return (_np.asarray(dense, dtype=_np.float32).tobytes()
            + _np.asarray([card_id, attack_id, target_id],
                          dtype=_np.int32).tobytes())


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
    if s["flip_eligible"]:
        line += (f" flips={s['flips']}/{s['flip_eligible']}"
                 f" ({s['flips'] / s['flip_eligible']:.1%})")
    if s["first_error"]:
        line += f" first_error={s['first_error'][:200]!r}"
    return line


def reset_stats() -> None:
    STATS.update(calls=0, fallbacks=0, net_missing=0, deck_returns=0,
                 first_error=None, flip_eligible=0, flips=0)


class PolicyAgent:
    def __init__(self, decklist: list[int], net_path: str | None = None,
                 chip_targeting: bool = True, energy_spread: bool = True,
                 drag_target: bool = False, boss_converts: bool = False,
                 drag_high_hp: bool = False, boss_veto: bool = False,
                 counter_source: bool = True, chip_wall_defer: bool = True,
                 boss_prize_veto: bool = False,
                 sequencer: bool = False, seq_k: int = 8, seq_dets: int = 4,
                 seq_budget: float = 0.35, seq_reply: bool = False,
                 flip_margin: float | None = None,
                 poffin_force: bool = False):
        self.decklist = list(decklist)
        # E3's teacher-free gate (day 23). Take the OTHER side of a near-tie:
        # when the logit gap between the lowest-scored SELECTED option and the
        # highest-scored UNSELECTED one is below this, swap them. This is not a
        # candidate rule -- it is a probe that measures whether the band E3
        # wants a human to relabel is indifferent, using no teacher at all.
        # None = off, and `bc` with no flag is byte-identical in behaviour to
        # what it was before this existed.
        self.flip_margin = flip_margin
        # The FIFTH Boss's Orders rule: suppress the play when attacking
        # right now takes strictly more prizes than any drag can. The
        # other four picked a side in a trade and all measured null; this
        # one deletes a dominated option (EVIDENCE 6 vs 8g). Opt-in until
        # it clears the five anchors.
        self.boss_prize_veto = boss_prize_veto
        # B4: turn-level lookahead (sequencer.py). OFF by default and opt-in
        # via `bc:<label>,seq` until it clears an arena A/B -- it is an
        # experiment, not a shipped component (EVIDENCE 8m).
        # Load an explicit net before constructing the sequencer so simulated
        # continuations use the same policy as the owning agent. Falling back
        # to policynet.get() here would silently use the bundled checkpoint.
        # `net=a.npz+b.npz` loads an ENSEMBLE: the members vote per option
        # (softmax each, then average). One path behaves exactly as before.
        if net_path and "+" in net_path:
            parts = [p for p in net_path.split("+") if p]
            self.net = policynet.load_ensemble(parts)
        else:
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
        #
        # ⚠ MERGE NOTE (day 22): this guard sits immediately after the load and
        # BEFORE the sequencer is built, because the beyond-BC branch moved the
        # load earlier so `Sequencer` shares the agent's net. Failing here means
        # we never construct a Sequencer around a silently-null net, which is
        # strictly better than where the guard originally landed on `main`.
        if net_path and self.net is None:
            raise ValueError(
                f"net {net_path!r} exists but FAILED policynet.load's guard "
                f"(feature dims, n_pool/n_attr, or the v7 vocab row count). "
                f"Refusing to fall back to the tracked sa/policy_net.npz, "
                f"which is a different net and would have scored silently.")
        self.seq = None
        if sequencer:
            from .sequencer import Sequencer
            self.seq = Sequencer(decklist, k=seq_k, dets=seq_dets,
                                 budget_s=seq_budget, reply=seq_reply,
                                 net=self.net)
        # An explicit net lets two candidate policies play each other inside
        # ONE arena process. Comparing them via a third opponent instead needs
        # ~2x the games for the same resolution, and the module-level
        # policynet.get() singleton cannot hold two nets at once.
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
        # E11: play Buddy-Buddy Poffin when the bench has >=2 free slots. The
        # 1150+ pilots play it in 70.2% of available turns at board size 4 and
        # our clone in 29.4% -- 0.80 plays/game, ordering-free (rule 21). OFF by
        # default until its own A/B clears the bar, same discipline as every
        # rule above it.
        self.poffin_force = poffin_force
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

    def _flip_near_tie(self, net, obs: dict, picked: list[int]) -> list[int]:
        """Swap the boundary pair when their logit gap is under `flip_margin`.

        The definition is `p43_dagger_queue.make_candidate`'s, deliberately and
        exactly: same boundary pair (lowest selected vs highest unselected),
        same exclusion of bitwise-equivalent options -- two copies of one card
        in one role are a free tie by construction (§8x) and flipping them
        would dilute the treatment with a known no-op. If the two scripts ever
        disagree, the sizing no longer describes the intervention.

        ⚠ Costs a second forward pass (~1.2 ms). `bc` has no time-budgeted
        component and uses 1.12 s of a 1,800 s pool, so this is not a compute
        confound the way E5's planner was -- but it IS an asymmetry between the
        arms, and it is stated rather than assumed away.
        """
        options = (obs.get("select") or {}).get("option") or []
        chosen = set(picked)
        unchosen = set(range(len(options))) - chosen
        if not chosen or not unchosen:
            return picked
        STATS["flip_eligible"] += 1
        scores = net.scores(obs)
        low = min(chosen, key=lambda i: float(scores[i]))
        high = max(unchosen, key=lambda i: float(scores[i]))
        if float(scores[low]) - float(scores[high]) >= self.flip_margin:
            return picked
        if _option_sig(obs, options[low]) == _option_sig(obs, options[high]):
            return picked
        STATS["flips"] += 1
        return [high if i == low else i for i in picked]

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
            if self.flip_margin is not None:
                picked = self._flip_near_tie(net, obs, picked)
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
            if self.poffin_force:
                forced = targeting.poffin_force(obs, list(picked))
                if forced is not None:
                    return forced[:want]
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
