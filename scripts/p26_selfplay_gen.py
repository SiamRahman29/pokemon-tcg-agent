"""B8: generate our OWN trajectories, with outcomes, in the corpus's own format.

**Why this is not `p20_record_games.py`.** That writes a full Kaggle replay per
game -- multi-MB, because it keeps every observation. B8 needs tens of thousands
of games, which would be tens of GB and then a second pass to featurize. This
script featurizes AT SELECT TIME and writes `shard_*.npz` directly, in the exact
layout `build_policy_dataset.py` produces, so **`train_policy.py` reads these
shards with no adapter at all** -- including the `won` column, which the BC
corpus has carried since day 1 and which nothing has ever trained on.

⚡ **That column is the whole point of B8.** The corpus records *what a human
did*, never *whether it worked*. `won` is the one field in these shards that is
not a re-parameterisation of information the clone already has (§8w, §8x).

**Two columns the BC corpus does NOT have, added here:**

  - `behav_logp` -- log-probability the behaviour policy assigned to the action
    it actually took. Without it an off-policy correction is not computable and
    the fine-tune is silently on-policy-only.
  - `margin` -- top-1 minus top-2 logit at that select. This is the gate: where
    the net is confident it is reproducing field-modal play, and §8u measured
    that agreement with the FIELD is what predicts strength. Outcome information
    can only usefully move a decision that was close.

**Exploration.** A pure argmax policy playing itself produces near-deterministic
games and no signal to attribute. Actions are sampled from `softmax(logits/tau)`
without replacement (Plackett-Luce), which keeps the multi-select semantics the
net was trained under and makes `behav_logp` exact rather than approximate.

⚠ `--probe` runs the SIZING STEP FIRST and writes nothing: it reports, per tau,
how far sampling moves the agent from argmax and what that costs in strength.
Rule 14 -- a temperature is a design parameter and it gets measured, not
guessed. Too cold and the trajectories carry no exploration; too hot and they
are off-distribution for the policy we actually ship.

    # step 1 -- pick tau by measurement, writes no data
    python -X utf8 scripts/p26_selfplay_gen.py --probe --games 40

    # step 2 -- generate
    python -X utf8 scripts/p26_selfplay_gen.py --tau 0.5 --games 4000 \
        --out artifacts/rl_v5_t05
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import harness, sdk  # noqa: E402

sdk.load()

from sa.features import extra_feats, featurize  # noqa: E402
from sa.optfeat import option_features, OPT_DENSE  # noqa: E402
from sa import policynet  # noqa: E402
from build_policy_dataset import Writer, sel_features  # noqa: E402
from arena import build_agent, resolve_deck  # noqa: E402


class RLWriter(Writer):
    """`Writer` plus the two columns an outcome-trained net needs.

    Subclassed rather than forked so the shard layout cannot drift from the BC
    corpus's -- a fine-tune that reads a differently-shaped array would train
    fine and measure like noise.
    """

    def reset(self):
        super().reset()
        self.blogp: list[float] = []
        self.margin: list[float] = []
        self.seat: list[int] = []

    def add_rl(self, *a, behav_logp: float, margin: float, seat: int, **kw):
        # ⚠ order matters: Writer.add() flushes when the row budget is hit, so
        # the extra columns must be appended BEFORE that call or a flush lands
        # with three arrays one row short.
        self.blogp.append(behav_logp)
        self.margin.append(margin)
        self.seat.append(seat)
        super().add(*a, **kw)

    def flush(self):
        if not self.sd:
            return
        blogp = np.asarray(self.blogp, dtype=np.float32)
        margin = np.asarray(self.margin, dtype=np.float32)
        seat = np.asarray(self.seat, dtype=np.int8)
        idx = self.idx
        out_dir = self.out_dir
        n = len(self.sd)
        super().flush()          # writes shard_{idx}.npz and resets
        # Re-open and append. Cheaper than duplicating Writer.flush's 20-line
        # dict, and it cannot fall out of sync with it.
        path = out_dir / f"shard_{idx:03d}.npz"
        # `allow_pickle=True` because the base writer now emits at least one
        # object-dtype column (the team/submission id strings). This path ran
        # clean for B8 on day 17 and raises today, so the corpus schema gained
        # an object array since -- and the failure is at RE-OPEN, after the
        # shard is already on disk, which is why `artifacts/<run>/shard_000.npz`
        # exists and looks fine while the RL columns were never appended.
        # Reading back a file we just wrote ourselves is not an untrusted load.
        z = dict(np.load(path, allow_pickle=True))
        assert len(blogp) == n == len(z["gid"]), "RL columns out of step"
        z["behav_logp"] = blogp
        z["margin"] = margin
        z["seat"] = seat
        np.savez_compressed(path, **z)


def pl_sample(logits: np.ndarray, k: int, tau: float,
              rng: np.random.Generator) -> tuple[list[int], float]:
    """Sample k options without replacement from softmax(logits/tau).

    Returns (picked, log-probability of that ordered draw). Plackett-Luce is
    the right family here because it IS the agent's inference rule in the
    tau->0 limit: rank by logit, take the top k (`policynet.Net.choose`). So
    the behaviour policy degenerates continuously to the shipped one and
    `behav_logp` stays exact for every k.
    """
    z = logits / max(tau, 1e-6)
    z = z - z.max()
    remaining = list(range(len(z)))
    picked: list[int] = []
    logp = 0.0
    for _ in range(min(k, len(remaining))):
        sub = z[remaining]
        sub = sub - sub.max()
        p = np.exp(sub)
        p /= p.sum()
        j = int(rng.choice(len(remaining), p=p))
        logp += float(np.log(max(p[j], 1e-12)))
        picked.append(remaining[j])
        remaining.pop(j)
    return picked, logp


class SamplingNetAgent:
    """The shipped v5 decision path with argmax replaced by a PL sample.

    ⚠ It reproduces `bcagent.PolicyAgent`'s early exits deliberately. The
    shipped bundle pins `chip_targeting`/`energy_spread`/`counter_source` to
    False (§8ac, verified by reading `main.py` out of the tarball), so with the
    rules off that path is: forced-select shortcuts, then `net.choose`. Anything
    else here would generate trajectories for a policy we do not ship.
    """

    def __init__(self, decklist, net, tau: float, rng, tap=None):
        self.decklist = list(decklist)
        self.net = net
        self.tau = tau
        self.rng = rng
        self.tap = tap          # called with the per-select record, or None

    def __call__(self, obs: dict) -> list[int]:
        if obs.get("select") is None:
            return list(self.decklist)
        sel = obs["select"]
        opts = sel.get("option") or []
        n = len(opts)
        mn = sel.get("minCount", 0)
        mx = sel.get("maxCount", 0)
        if n == 0 or mx == 0:
            return []
        if mn == mx == n:
            return list(range(n))       # forced: no decision exists
        sc = self.net.scores(obs)
        k = mx
        if mx > mn:
            frac = 1.0
            if self.net.count_frac is not None:
                t = min(sel.get("type") or 0, 10)
                ctx = min(sel.get("context") or 0, 63)
                frac = float(self.net.count_frac[t, ctx])
            k = mn + int(round(frac * (mx - mn)))
        k = max(mn, min(k, mx)) if mx > mn else mx
        k = max(k, 1)
        if self.tau <= 0.0:
            picked = [int(i) for i in np.argsort(-sc)[:k]]
            logp = 0.0
        else:
            picked, logp = pl_sample(sc, k, self.tau, self.rng)
        if self.tap is not None and n >= 2:
            order = np.argsort(-sc)
            margin = float(sc[order[0]] - sc[order[1]]) if n >= 2 else 0.0
            greedy = set(int(i) for i in order[:k])
            self.tap(obs, picked, logp, margin, set(picked) != greedy)
        return picked


def load_net(path: str):
    net = policynet.load(ROOT / path)
    if net is None:
        raise SystemExit(f"{path} did not load -- feature dims do not match "
                         "the current code (policynet.load's guard)")
    return net


def probe(args) -> int:
    """Sizing step: what does sampling cost, per tau? Writes nothing."""
    _, deck = resolve_deck(args.deck)
    net = load_net(args.net)
    print(f"  probe: {args.games} games per tau, deck={args.deck}, "
          f"net={args.net}")
    print(f"  {'tau':>6} {'off-argmax':>11} {'score vs argmax':>17} "
          f"{'95% CI':>16} {'turns':>7}")
    rows = []
    for tau in [float(t) for t in args.taus.split(",")]:
        rng = np.random.default_rng(args.seed)
        dev = [0, 0]
        wins = draws = 0
        turns = 0
        for g in range(args.games):
            def tap(obs, picked, logp, margin, deviated):
                dev[0] += int(deviated)
                dev[1] += 1
            sampler = SamplingNetAgent(deck, net, tau, rng, tap=tap)
            greedy = SamplingNetAgent(deck, net, 0.0, rng)
            # seats alternate: seat is worth ~1 pp (§8aj) and an unbalanced
            # probe would read that as a tau effect.
            if g % 2 == 0:
                r = harness.play_game(sampler, greedy, list(deck), list(deck))
                won = r.winner == 0
            else:
                r = harness.play_game(greedy, sampler, list(deck), list(deck))
                won = r.winner == 1
            turns += r.turns
            if r.winner == 2:
                draws += 1
            elif won:
                wins += 1
        n = args.games
        score = (wins + 0.5 * draws) / n
        lo, hi = harness._wilson(wins + 0.5 * draws, n)
        rate = dev[0] / max(dev[1], 1)
        print(f"  {tau:>6.2f} {rate:>10.1%} {score:>17.3f} "
              f"  [{lo:.3f}, {hi:.3f}] {turns / n:>7.1f}")
        rows.append((tau, rate, score, lo, hi))
    print("\n  Read this as a TRADE, not a maximum:")
    print("  * off-argmax ~0%  -> the trajectories carry no exploration and")
    print("    the outcome column has nothing to attribute variance to.")
    print("  * score well under 0.5 -> the behaviour policy is not the one we")
    print("    ship, so its outcomes describe a different agent (§8u's")
    print("    ordering: distance from the field's modal policy costs Elo).")
    print("  Pick the largest tau whose CI still covers ~0.5.")
    return 0


def generate(args) -> int:
    _, deck = resolve_deck(args.deck)
    net = load_net(args.net)
    rng = np.random.default_rng(args.seed)
    out = ROOT / args.out
    writer = RLWriter(out)
    opp_label = args.opp or f"self(tau={args.tau})"
    print(f"  net={args.net} tau={args.tau} deck={args.deck} opp={opp_label}")
    print(f"  -> {out}")

    n_rows = n_seen = 0
    tally = {0: 0, 1: 0, 2: 0}
    t0 = time.time()
    for g in range(args.games):
        pending: list[tuple] = []          # rows awaiting the game's outcome

        def make_tap(seat: int):
            def tap(obs, picked, logp, margin, deviated):
                state = obs["current"]
                sel = obs["select"]
                me = state["yourIndex"]
                opts = sel.get("option") or []
                dense, bags = featurize(state, me)
                od = np.zeros((len(opts), OPT_DENSE), dtype=np.float32)
                oc = np.zeros(len(opts), dtype=np.int32)
                oa = np.zeros(len(opts), dtype=np.int32)
                ot = np.zeros(len(opts), dtype=np.int32)
                for i, o in enumerate(opts):
                    od[i], oc[i], oa[i], ot[i] = option_features(obs, o)
                mask = np.zeros(len(opts), dtype=np.float32)
                mask[picked] = 1.0
                pending.append((dense, bags, sel_features(sel),
                                (od, oc, oa, ot), mask,
                                extra_feats(state, sel, me),
                                logp, margin, me))
            return tap

        # Seats alternate. §8aj measured first player at ~+1 pp, so a corpus
        # generated from one seat would bake that in as if it were policy.
        swap = g % 2 == 1
        a = SamplingNetAgent(deck, net, args.tau, rng, tap=make_tap(0))
        if args.opp:
            _, b = build_agent(args.opp, list(deck))
        else:
            b = SamplingNetAgent(deck, net, args.tau, rng, tap=make_tap(1))
        a0, a1 = (b, a) if swap else (a, b)
        r = harness.play_game(a0, a1, list(deck), list(deck))
        tally[r.winner if r.winner in (0, 1) else 2] += 1

        gid = args.gid_base + g
        for (dense, bags, seld, opts, mask, extra,
             logp, margin, me) in pending:
            n_seen += 1
            if args.keep_margin > 0 and margin > args.keep_margin:
                continue
            # `won` is from the acting seat's point of view -- the same
            # convention build_policy_dataset uses (`rewards[me] > rewards[1-me]`).
            # A draw is 0.5, which the BC corpus never had to represent because
            # a Kaggle replay with a null reward is skipped.
            if r.winner == 2:
                won = 0.5
            else:
                won = 1.0 if r.winner == me else 0.0
            writer.add_rl(dense, bags, seld, opts, mask, gid, won,
                          extra=extra, behav_logp=logp, margin=margin,
                          seat=me)
            n_rows += 1
        if (g + 1) % args.report == 0:
            el = time.time() - t0
            print(f"  {g + 1:>6}/{args.games} games  {n_rows:>8} rows  "
                  f"{(g + 1) / el:.2f} games/s  "
                  f"seat0={tally[0]} seat1={tally[1]} draw={tally[2]}")
    writer.flush()
    el = time.time() - t0
    print(f"\n  {args.games} games, {n_rows} rows, {el / 60:.1f} min "
          f"({args.games / max(el, 1e-9):.2f} games/s)")
    if args.keep_margin > 0:
        print(f"  --keep-margin {args.keep_margin}: kept {n_rows:,} of "
              f"{n_seen:,} decisions ({n_rows / max(n_seen, 1):.1%}), "
              f"{n_rows / max(args.games, 1):.1f} rows/game")
    print(f"  seat0={tally[0]} seat1={tally[1]} draw={tally[2]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="out/policy_v5.npz")
    ap.add_argument("--deck", default="grimmsnarl")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="sampling temperature; 0 = argmax (no exploration)")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default="artifacts/rl_v5")
    ap.add_argument("--opp", default=None,
                    help="arena spec for the opponent seat. Default: another "
                         "sampling copy of the same net (mirror self-play), "
                         "which is 33.3%% of our real field and 51.1%% above "
                         "opponent rating 900 (§8ac).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gid-base", type=int, default=0,
                    help="game ids start here. ⚠ The trainer's val split is "
                         "`gid %% 20 == 0`, so parallel workers MUST use "
                         "disjoint bases or the same game lands in both sides.")
    ap.add_argument("--keep-margin", type=float, default=0.0,
                    help="write ONLY rows whose top1-top2 logit margin is <= "
                         "this. 0 = keep everything. §8ao's rerun is memory- "
                         "bound rather than compute-bound (the trainer "
                         "materialises the whole corpus), and the rows above "
                         "the training gate are ones the AWR term never "
                         "re-weights anyway -- so dropping them at WRITE time "
                         "buys ~2.3x the games for the same RAM. ⚠ It is not "
                         "free: those rows trained at weight 1.0 as extra "
                         "cloning of our own confident play. The corpus anchor "
                         "(--anchor-ds) is what tethers the fine-tune, and it "
                         "is unaffected.")
    ap.add_argument("--report", type=int, default=50)
    ap.add_argument("--probe", action="store_true",
                    help="size the temperature and write nothing")
    ap.add_argument("--taus", default="0.25,0.5,1.0,2.0",
                    help="--probe: temperatures to sweep")
    args = ap.parse_args()
    return probe(args) if args.probe else generate(args)


if __name__ == "__main__":
    sys.exit(main())
