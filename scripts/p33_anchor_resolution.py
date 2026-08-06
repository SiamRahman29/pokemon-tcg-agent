"""Can a near-ceiling anchor resolve anything? §8ap said no. This is the arithmetic.

**Why this exists (day 18).** §8ap sorted the anchor set by how close our score
sits to 0.5 and found the order also sorts them by UNrepresentativeness: the two
anchors we can separate nets on (`rule:v10` 4.0%, `rule:archaludon` 8.0%) are the
two §8ac measured at **0 of 47 games above opponent rating 900**, while every
anchor that represents the field we now play is one we beat **77-87%** of the
time. It filed that as ⛔ *"near ceiling"* / *"cannot resolve a difference"* and
concluded that adding representative anchors *"bought honesty, not sensitivity"*.

**That conclusion is a blocker for the day-18 plan's item 1** -- the
matchup-stratified deck search -- because a stratified design measures deck
variants at exactly those anchors. So it gets checked before anything is built
(rule 14), and checking it is arithmetic, not compute.

The check turns on a distinction §8ap does not make:

  * **In ELO units** a near-ceiling anchor really is a worse instrument. A fixed
    Elo delta maps to a win-rate delta proportional to p(1-p), while the noise
    only falls as sqrt(p(1-p)). Net resolution therefore degrades as
    1/sqrt(p(1-p)) -- and the required n as 0.25/(p(1-p)).
  * **In WIN-RATE units it is a BETTER instrument**, because sqrt(p(1-p)) is
    smaller near the ceiling and the noise is all there is.

Which one is right depends on the objective, and for a deck the objective is not
in dispute: we pick the 60 that maximises the field-weighted win rate
`W = sum_i w_i p_i`. That is **linear in win rate**, so the near-ceiling terms
carry small effects AND small variance, and the ratio -- which is what a decision
turns on -- is what this script prints.

    python -X utf8 scripts/p33_anchor_resolution.py
    python -X utf8 scripts/p33_anchor_resolution.py --n 4000

⚠ Every score here is COPIED from the arena's own printed summary line via the
EVIDENCE section named beside it, never recomputed from `out/arena/*.jsonl`.
Rule 18: the archives are seat-indexed and `arena.py` alternates seats, so
re-deriving a score by hand is how §8an nearly published 0.510 for 0.888.
"""
from __future__ import annotations

import argparse
import math

# (label, v5 score, field share, direct?)
# Scores: EVIDENCE §8ap's complete anchor table, which takes each from the
# `score=` line arena.py printed for that run.
# 🔴 EXCEPT Crustle.  §8ap's table says 0.866 and names it "v3, guard"; the pilot
# in the repo is a FOURTH version (`83daa48`, committed 17:48 -- 26 minutes after
# the 0.866 run's last game at 17:22) that had never been measured above n=6.
# Measured at n=2,000 on day 18: **0.755 [0.735, 0.773]** (`p35`,
# out/arena/p35_v5_vs_crustle_v4.jsonl).  See §8aq.
# 🔴 AND THE 0.866 WAS A DIFFERENT DECK, not only a different pilot (§8ax, day
# 22).  The row below is `rule:crustle@crustle_v1` -- the pilot on its own 60.
# The same pilot on `decks/crustle.py` (the field-consensus 60) scores **0.893**,
# a **+0.140** deck term measured with the pilot held fixed.  Which one belongs
# here is a live question this script does NOT settle: 0.755 is the harder,
# better-resolving instrument, and 0.893 is the deck we are more likely to face.
# §8ac's 6.7% share was assigned before anyone noticed there were two.
# Shares: EVIDENCE §8ac, the band-aware census of our own ladder replays.
# `direct`: the mirror is the one anchor where variant-vs-control is a single
# head-to-head run (§8aj), so it costs n games instead of 2n AND its estimate is
# a difference measured within one experiment rather than between two.
ANCHORS = [
    ("mirror (grimmsnarl)", 0.500, 0.333, True),
    # ⚠ every non-mirror row is `rule:<name>@<its own tuned deck>`
    ("rule:v10 (M Lucario)", 0.569, 0.040, False),
    ("rule:archaludon",      0.671, 0.080, False),
    ("rule:alakazam5",       0.789, 0.220, False),
    ("rule:dragapult",       0.809, 0.053, False),
    ("bc:garchomp",          0.857, 0.067, False),
    ("rule:crustle (v4)",    0.755, 0.067, False),
]

# EVIDENCE §8an: three nets measured against two pilot versions of ONE anchor.
# 🔴 THE TWO COLUMNS ARE ALSO TWO DIFFERENT DECKS (§8ax, day 22): the left is
# `crustle_v1`, the right is `crustle`, and the deck term alone is +0.140. So
# this is still "the same three nets read through two instruments of different
# difficulty" -- which is all the compression calibration below needs, and the
# calibration is reported as UNDERPOWERED either way -- but it is NOT "two pilot
# versions", and nothing here may be quoted as a pilot effect.
CRUSTLE_CALIB = {
    #        v1 (broken)  v3 (guard only)
    "v3": (0.7700, 0.8670),
    "v4": (0.7885, 0.8750),
    "v5": (0.7680, 0.8660),
}
CRUSTLE_N = 2000

GAMES_PER_SEC = 5.96  # §8ae, measured, one process


def sigma(p: float, direct: bool) -> float:
    """Per-game SD of the estimated win-rate change at this anchor.

    Two arms of n games each: SE = sqrt(2p(1-p)/n), i.e. sigma = sqrt(2p(1-p)).
    One DIRECT head-to-head of n games: the score IS the variant's win rate
    against the control deck, so SE = sqrt(0.25/n) and sigma = 0.5.  The direct
    form is sqrt(2) more precise per game AND costs half as many games -- a 4x
    efficiency advantage that only the mirror can claim.
    """
    return 0.5 if direct else math.sqrt(2.0 * p * (1.0 - p))


def se_diff(p: float, n: int, direct: bool = False) -> float:
    """SE of (variant - control) with n games per arm at this anchor."""
    return sigma(p, direct) / math.sqrt(n)


def elo_slope(p: float) -> float:
    """d(win rate)/d(Elo) at score p, under the logistic model."""
    return p * (1.0 - p) * math.log(10.0) / 400.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8000,
                    help="games per arm per anchor (default 8000, what §8al used)")
    args = ap.parse_args()
    n = args.n

    print(f"\nANCHOR RESOLUTION AT n={n} GAMES PER ARM PER ANCHOR")
    print("  'min WR' = smallest win-rate change detectable at 95% (1.96 x SE of")
    print("  the two-arm difference).  'min Elo' = the same thing in Elo units.")
    print("  'n x vs mirror' = games needed for EQUAL Elo resolution, relative")
    print("  to the mirror.\n")
    hdr = ("%-22s %6s %6s %8s %8s %10s %10s"
           % ("anchor", "p", "share", "min WR", "min Elo", "nx vs mir", "w*SE"))
    print(hdr)
    print("-" * len(hdr))

    mirror_pq = 0.25
    var_sum = 0.0
    for label, p, w, direct in ANCHORS:
        se = se_diff(p, n, direct)
        min_wr = 1.96 * se
        min_elo = min_wr / elo_slope(p)
        n_mult = mirror_pq / (p * (1.0 - p))
        var_sum += (w * se) ** 2
        print("%-22s %6.3f %6.1f%% %8.4f %8.1f %10.2f %10.5f"
              % (label, p, 100 * w, min_wr, min_elo, n_mult, w * se))

    w_tot = sum(w for _, _, w, _ in ANCHORS)
    se_w = math.sqrt(var_sum)
    print("-" * len(hdr))
    print("%-22s %6s %6.1f%% %8.4f %8.1f"
          % ("WEIGHTED (all anchors)", "", 100 * w_tot, 1.96 * se_w,
             1.96 * se_w / elo_slope(0.5)))
    print("  ^ the field-weighted win rate W = sum_i w_i p_i, which is the")
    print("    quantity a deck choice is actually trying to maximise.")
    print(f"    {100 * (1 - w_tot):.1f}% of the field is not covered by any anchor")
    print("    and is assumed to move like the covered part.")

    # --- the cost of the full stratified sweep -------------------------------
    cells = len(ANCHORS)
    games = sum(n if direct else 2 * n for _, _, _, direct in ANCHORS)
    hrs = games / GAMES_PER_SEC / 3600
    print(f"\nCOST of one variant, all {cells} anchors, EQUAL n: {games:,} games")
    print(f"  = {hrs:.1f} h single-process at {GAMES_PER_SEC} games/s "
          f"({hrs / 2.5:.1f} h at the 2-3 jobs rule 7 allows)")

    # --- where the noise actually lives --------------------------------------
    print("\nSHARE OF THE WEIGHTED VARIANCE, per anchor (equal n):")
    for label, p, w, direct in ANCHORS:
        share = (w * se_diff(p, n, direct)) ** 2 / var_sum
        print("  %-22s %5.1f%%" % (label, 100 * share))
    print("  ⇒ this is the honest form of §8ap's complaint.  The mirror does not")
    print("    merely 'carry the set' -- it IS the set, in variance terms, and")
    print("    the near-ceiling anchors are cheap to add rather than useless.")
    print("    ⚠ which also means EQUAL n is the wrong design: 5 of the 7")
    print("      anchors are consuming most of the games for a few percent of")
    print("      the precision.")

    # --- the design: optimal allocation --------------------------------------
    # Neyman allocation with unequal costs: to minimise sum_i w_i^2 sigma_i^2/n_i
    # subject to a fixed game budget sum_i c_i n_i, put n_i proportional to
    # w_i sigma_i / sqrt(c_i).  This is the actual output of this script -- the
    # stratified design, in games per anchor.
    alloc = []
    for label, p, w, direct in ANCHORS:
        c = 1.0 if direct else 2.0
        alloc.append((label, w * sigma(p, direct) / math.sqrt(c), c, p, direct))
    # With n_i = k*a_i, the game budget is G = k * sum_i c_i a_i and the
    # variance is sum_i w_i^2 sigma_i^2 / n_i = (1/k) * sum_i w_i sigma_i
    # sqrt(c_i).  Solve for the G that reproduces the equal-n variance.
    cost_a = sum(a * c for _, a, c, _, _ in alloc)
    var_a = sum(w * sigma(p, d) * math.sqrt(c)
                for (_, _, c, p, d), (_, _, w, _) in zip(alloc, ANCHORS))
    budget = cost_a * var_a / var_sum
    scale = budget / cost_a
    print(f"\nTHE DESIGN -- optimal allocation for the SAME weighted precision")
    print(f"  (±{1.96 * math.sqrt(var_sum):.4f} on W), by Neyman allocation with costs:\n")
    print("  %-22s %10s %10s" % ("anchor", "n per arm", "games"))
    tot_g = 0
    for (label, a, c, p, d) in alloc:
        n_i = int(round(a * scale / 100.0) * 100)
        g = int(c * n_i)
        tot_g += g
        print("  %-22s %10s %10s" % (label, f"{n_i:,}", f"{g:,}"))
    print("  %-22s %10s %10s" % ("TOTAL", "", f"{tot_g:,}"))
    print(f"  = {tot_g / GAMES_PER_SEC / 3600:.1f} h single-process, "
          f"{tot_g / GAMES_PER_SEC / 3600 / 2.5:.1f} h at 2-3 jobs "
          f"-- {100 * tot_g / games:.0f}% of the equal-n cost for the same "
          f"precision.")
    print("  ⚠ These n are for ONE variant.  A search over k variants costs k")
    print("    times this, minus the control arms, which are shared.")

    # --- what stratification actually buys, stated honestly ------------------
    se_mirror_only = 0.5 / math.sqrt(tot_g)
    print("\nWHAT STRATIFICATION BUYS -- and it is NOT precision:")
    print(f"  Spend the same {tot_g:,} games mirror-only and Delta_mirror is")
    print(f"  measured to ±{1.96 * se_mirror_only:.4f}, i.e. BETTER than the")
    print(f"  stratified design's ±{1.96 * math.sqrt(var_sum):.4f} on W.")
    print("  Every game in one cell beats games spread over seven.")
    print("  ⇒ 🔴 The case for stratifying is entirely about BIAS.  Mirror-only")
    print("    estimates W by assuming Delta is the same in every matchup, and")
    print("    §8al showed that assumption failing by construction: Tool")
    print("    Scrapper is played 0.00 times per mirror game and drawn in 81% of")
    print("    real games, so a mirror A/B returns 'cutting it is free' however")
    print("    many games you buy.  A more precise estimate of the wrong")
    print("    quantity is worse, not better -- it is rule 16 with a tighter CI.")
    print("  ⇒ So a variant is stratified when its cards are matchup-specific,")
    print("    and measured mirror-only when they are not.  Deciding which is a")
    print("    LIVENESS question, per card per matchup, and that instrument does")
    print("    not exist yet.  It is the next build, not this one.")

    # --- the compression calibration -----------------------------------------
    print("\nDOES A REAL DIFFERENCE COMPRESS THE WAY THE ELO MODEL SAYS? (§8an)")
    print("  Same three nets, one anchor, two pilot versions of different")
    print("  difficulty.  If the Elo model holds, a net-vs-net difference should")
    print("  shrink by the ratio of p(1-p) between the two pilots.\n")
    pairs = [("v4-v3", "v3", "v4"), ("v5-v4", "v4", "v5")]
    print("  %-8s %10s %10s %10s %10s" % ("pair", "at p~.78", "predicted", "observed", "SE(obs)"))
    for name, lo, hi in pairs:
        d_broken = CRUSTLE_CALIB[hi][0] - CRUSTLE_CALIB[lo][0]
        d_fixed = CRUSTLE_CALIB[hi][1] - CRUSTLE_CALIB[lo][1]
        p_b = 0.5 * (CRUSTLE_CALIB[hi][0] + CRUSTLE_CALIB[lo][0])
        p_f = 0.5 * (CRUSTLE_CALIB[hi][1] + CRUSTLE_CALIB[lo][1])
        pred = d_broken * (p_f * (1 - p_f)) / (p_b * (1 - p_b))
        se_o = math.sqrt(p_f * (1 - p_f) / CRUSTLE_N + p_b * (1 - p_b) / CRUSTLE_N)
        print("  %-8s %10.4f %10.4f %10.4f %10.4f"
              % (name, d_broken, pred, d_fixed, se_o))
    print("\n  ⚠ READ THE LAST COLUMN BEFORE THE OTHERS.  Both observed values sit")
    print("    well inside one SE of both the prediction and of zero, so this")
    print("    CANNOT distinguish the Elo model from any other.  It is reported")
    print("    because it is the only empirical handle available and because")
    print("    'we checked and it was underpowered' is the honest verdict --")
    print("    not because it confirms anything.  The design below therefore")
    print("    rests on the arithmetic above, which needs no model of how")
    print("    effects compress: it is stated in win-rate units throughout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
