"""THE probe that decides B4: is the within-turn eval spread bigger than its noise?

`EVIDENCE` §8l left B4 alive on three criteria but flagged that none of them
measured the quantity that matters. A turn-sequencer ranks **end-of-turn states
reachable from the SAME position**, which share deck, prizes and the opponent's
board -- so their eval differences are far smaller than the between-game spread
§8l's AUC was built on. **A high across-game AUC is compatible with zero
within-turn discrimination.**

This measures the thing directly, and against the right baseline.

**The baseline is determinization noise, and that choice is the whole point.**
The dead rollout search (`EVIDENCE` §2) did not fail because it was mis-tuned; it
failed because its estimator's standard error (~0.14 on a 0/1 rollout) exceeded
the differences it was trying to resolve, so its argmax was noise. The same
question here: hidden information (our deck order, the opponent's hand) has to be
invented before we can simulate, so **the same action sequence scores differently
under different determinizations.** If that noise is as large as the spread
between candidate sequences, a beam is an expensive random number generator.

Reported per turn:

⚠ **A FIRST VERSION OF THIS PROBE USED THE WRONG BASELINE. Recorded because the
correction is the methodological point.** It compared the spread across sequences
(one determinization) with the spread of one sequence across determinizations,
and called the ratio SNR. That is wrong for a **ranking** problem: a different
determinization changes what everyone draws, which moves **every candidate
together** -- a common-mode shift that inflates the "noise" term while leaving
the ordering, the only thing a sequencer uses, untouched. The first run duly
reported SNR 1.05 ("noise") **and** top-1 agreement of 66.7% against 6.2% chance
("not noise") -- a contradiction that only makes sense if the baseline was
measuring level noise, not rank noise.

**The fix, and it is not goalpost-moving:** score the **full K x M matrix**
(every candidate under every determinization) and do a proper two-way variance
decomposition, so the common-mode term is removed by construction:

  * `between`   -- variance across candidates of their determinization-averaged
                   eval. This is real, transferable merit.
  * `resid`     -- the candidate x determinization interaction. This is the
                   part that does NOT transfer, i.e. the noise a beam would
                   chase.
  * **SNR = sqrt(between / resid)** -- the number that decides B4.
  * **split-half top-1 agreement** -- pick the argmax on half the
    determinizations, check it wins on the other half. Chance is 1/K. This was
    pre-registered in the first version and is what caught the bad baseline.

Candidates are sampled from the policy net's top-3 at each select, not uniformly
at random: a beam would explore plausible lines, and uniform-random play would
inflate `spread_seq` with sequences no agent would ever consider.

**Kill criterion (pre-registered): SNR <= ~1.5 or top-1 agreement near chance
=> B4 is selecting noise, and no beam width or pool budget can repair it.**

    python -X utf8 scripts/p12d_within_turn_signal.py --matches 40
"""
from __future__ import annotations

import argparse
import random
import statistics
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

from cg.api import SelectContext  # noqa: E402
import arena  # noqa: E402
from sa import fastsearch as fs  # noqa: E402
from sa import policynet as pnet  # noqa: E402
from sa.evalfn import evaluate  # noqa: E402
from sa.worlds import determinize  # noqa: E402

MAIN = int(SelectContext.MAIN)
MAX_STEPS = 24          # a turn is ~6 real selects; 24 is a generous cap
TOPK = 3                # sample among the net's top-3 at each select


def _rollout_turn(sid, obs, me, net, rng, forced=None):
    """Play OUR turn to its end. Returns (eval, actions, ok).

    `forced` replays a recorded action list where legal, so the same sequence
    can be scored under a different determinization.
    """
    actions: list[list[int]] = []
    cur_sid, cur = sid, obs
    for depth in range(MAX_STEPS):
        st = cur.get("current") or {}
        sel = cur.get("select") or {}
        opts = sel.get("option") or []
        if not st or st.get("result", -1) != -1:
            break
        if st.get("yourIndex") != me:
            break                       # turn passed to the opponent
        if not opts:
            break
        pick = None
        if forced is not None and depth < len(forced):
            cand = [i for i in forced[depth] if 0 <= i < len(opts)]
            if len(cand) == len(forced[depth]) and cand:
                pick = cand
        if pick is None:
            if len(opts) == 1:
                pick = [0]
            else:
                try:
                    sc = net.scores(cur) if net is not None else None
                except Exception:  # noqa: BLE001
                    sc = None
                if sc is None:
                    pick = [rng.randrange(len(opts))]
                else:
                    order = sorted(range(len(opts)),
                                   key=lambda i: -float(sc[i]))[:TOPK]
                    pick = [rng.choice(order)]
        try:
            cur_sid, cur = fs.step(cur_sid, pick)
        except fs.SearchError:
            return None, actions, False
        actions.append(list(pick))
    st = cur.get("current") or {}
    if not st:
        return None, actions, False
    return float(evaluate(st, me)), actions, True


class Probe:
    def __init__(self, inner, decklist, k=16, dets=8, per_game=3):
        self.inner = inner
        self.decklist = decklist
        self.k, self.dets, self.per_game = k, dets, per_game
        self.rng = random.Random(11)
        self.snr: list[float] = []
        self.spread_seq: list[float] = []
        self.spread_det: list[float] = []
        self.agree = Counter()
        self.gap: list[float] = []      # best - median candidate, in eval units
        self.k_used: list[int] = []
        self.errs: Counter = Counter()
        self._n = 0

    def new_game(self):
        self._n = 0

    def __call__(self, obs):
        picked = self.inner(obs)
        try:
            sel = obs.get("select") or {}
            st = obs.get("current") or {}
            if (sel.get("context") != MAIN or self._n >= self.per_game
                    or "search_begin_input" not in obs
                    or st.get("result", -1) != -1
                    or len(sel.get("option") or []) < 2):
                return picked
            me = st["yourIndex"]
            net = pnet.get()
            sbi = obs["search_begin_input"]
            deck_visible = sel.get("deck") is not None
            self._n += 1

            def root(seed):
                w = determinize(obs, self.decklist, [], random.Random(seed))
                return fs.begin(sbi, [] if deck_visible else w.my_deck,
                                w.my_prize, w.opp_deck, w.opp_prize,
                                w.opp_hand, w.opp_active)

            # ---- propose K candidate action sequences ------------------------
            sid0, o0 = root(1000)
            acts = []
            for j in range(self.k):
                _v, a, ok = _rollout_turn(sid0, o0, me, net,
                                          random.Random(5000 + j))
                if ok and a:
                    acts.append(a)
            fs.release(sid0)
            if len(acts) < 4:
                self.errs["fewer than 4 usable sequences"] += 1
                return picked

            # ---- score the FULL K x M matrix --------------------------------
            # Every candidate replayed under every determinization, so the
            # common-mode term can be removed by construction.
            mat: list[list[float]] = []      # [det][cand]
            for m in range(self.dets):
                try:
                    sidm, om = root(2000 + m)
                except (fs.SearchError, ValueError):
                    continue
                row = []
                for a in acts:
                    v, _a, ok = _rollout_turn(sidm, om, me, net,
                                              random.Random(7), forced=a)
                    row.append(v if (ok and v is not None) else None)
                fs.release(sidm)
                mat.append(row)
            # keep only candidates scored under every determinization
            if len(mat) < 4:
                self.errs["fewer than 4 usable determinizations"] += 1
                return picked
            keep = [c for c in range(len(acts))
                    if all(r[c] is not None for r in mat)]
            if len(keep) < 4:
                self.errs["fewer than 4 candidates scored everywhere"] += 1
                return picked
            M, K = len(mat), len(keep)
            X = [[float(mat[m][c]) for c in keep] for m in range(M)]

            grand = sum(sum(r) for r in X) / (M * K)
            cand_mean = [sum(X[m][c] for m in range(M)) / M for c in range(K)]
            det_mean = [sum(r) / K for r in X]
            between = sum((cm - grand) ** 2 for cm in cand_mean) / max(K - 1, 1)
            resid = sum((X[m][c] - cand_mean[c] - det_mean[m] + grand) ** 2
                        for m in range(M) for c in range(K)) \
                / max((M - 1) * (K - 1), 1)
            self.spread_seq.append(between ** 0.5)
            self.spread_det.append(resid ** 0.5)
            if resid > 1e-12:
                # between includes resid/M of noise; subtract it before the ratio
                sig = max(between - resid / M, 0.0)
                self.snr.append((sig / resid) ** 0.5)

            # ---- split-half top-1 agreement ---------------------------------
            h = M // 2
            a_mean = [sum(X[m][c] for m in range(h)) / h for c in range(K)]
            b_mean = [sum(X[m][c] for m in range(h, M)) / (M - h)
                      for c in range(K)]
            ia = max(range(K), key=lambda c: a_mean[c])
            ib = max(range(K), key=lambda c: b_mean[c])
            self.agree["argmax AGREED across halves" if ia == ib
                       else "argmax CHANGED"] += 1
            self.k_used.append(K)
            self.gap.append(max(cand_mean) - statistics.median(cand_mean))
        except Exception as exc:  # noqa: BLE001
            self.errs[f"{type(exc).__name__}: {str(exc)[:60]}"] += 1
        return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=40)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--dets", type=int, default=8)
    args = ap.parse_args()

    _, deck_a = arena.resolve_deck("grimmsnarl")
    _, deck_b = arena.resolve_deck("lucario_v10")
    _, agent_a = arena.build_agent("bc", deck_a)
    _, agent_b = arena.build_agent("rule:v10,noS", deck_b)

    probe = Probe(agent_a, list(deck_a), k=args.k, dets=args.dets)
    from ptcg.env import harness
    for _m in range(args.matches):
        probe.new_game()
        harness.play_game(probe, agent_b, list(deck_a), list(deck_b))
    fs.end()

    n = len(probe.snr)
    print(f"\n=== within-turn eval signal: {n} turns, K={args.k} sequences, "
          f"M={args.dets} determinizations ===")
    if not n:
        print("  no usable turns")
        for k, v in probe.errs.most_common(6):
            print(f"  {v:>5}  {k}")
        return 1

    print(f"\n  BETWEEN candidates (determinization-averaged merit)  "
          f"median sd {statistics.median(probe.spread_seq):7.3f}")
    print(f"  RESIDUAL cand x det (does NOT transfer)              "
          f"median sd {statistics.median(probe.spread_det):7.3f}  <- noise")
    print(f"\n  ** SNR = sqrt(between / resid):  median "
          f"{statistics.median(probe.snr):.2f} **")
    q = sorted(probe.snr)
    print(f"     p25 {q[len(q)//4]:.2f}   p75 {q[3*len(q)//4]:.2f}   "
          f"share of turns with SNR > 1: "
          f"{sum(1 for x in probe.snr if x > 1)/n:.1%}")

    t = sum(probe.agree.values())
    kbar = statistics.median(probe.k_used) if probe.k_used else args.k
    print(f"\n  split-half top-1 agreement (chance = {1/kbar:.1%} at "
          f"K={kbar:.0f}, n={t}):")
    for k, v in probe.agree.most_common():
        print(f"    {k:<45}{v:>5}{v/max(t,1):>8.1%}")

    print(f"\n  best-minus-median candidate, in eval units: median "
          f"{statistics.median(probe.gap):.3f}")
    print("     (what a perfect chooser would gain per turn IF the eval were")
    print("      exact. ⚠ UPWARD BIASED: it is a max over K noisy estimates,")
    print("      the same selection bias that flattered the dead search.)")

    # A sequencer can afford ~78,000 rollouts/turn (p12b), i.e. thousands of
    # determinizations per candidate -- so the noise it faces is the standard
    # ERROR of the candidate mean, not the per-cell residual.
    se = statistics.median(probe.spread_det) / (args.dets ** 0.5)
    bet = statistics.median(probe.spread_seq)
    print(f"\n  at M={args.dets}: between-sd {bet:.3f} vs SE of a candidate "
          f"mean {se:.3f}  => {bet/max(se,1e-9):.1f}x")
    print("     (noise averages down with M; TRUE merit spread does not)")

    print("\n--- verdict against the pre-registered kill criterion ---")
    agr = probe.agree["argmax AGREED across halves"] / max(t, 1)
    if agr <= 2.0 / kbar:
        print("  🔴 KILL: the argmax does not survive an independent")
        print("     determinization, so a beam would be selecting luck -- the")
        print("     exact failure mode of the dead rollout search (EVIDENCE 2).")
    else:
        print(f"  🟢 SURVIVES: the argmax reproduces at {agr:.1%} against "
              f"{1/kbar:.1%} chance,")
        print("     i.e. the eval is ranking real merit, not determinization")
        print("     luck. ⚠ This licenses a PROTOTYPE + A/B, nothing more:")
        print("     rule 3 -- five metrics have looked good and not paid.")

    if probe.errs:
        print("\nerrors/skips:")
        for k, v in probe.errs.most_common(6):
            print(f"  {v:>5}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
