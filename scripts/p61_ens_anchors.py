"""E9 confirmation: the three shipping candidates over the weighted anchor set.

The mirror already decided the ORDER (direct head-to-head, n=2,000 each):
ens2 > policy_v5_s1 > policy_v5. The mirror is 32.0% of the field, so this asks
the question the mirror cannot: does the order survive the matchups we do not
play against ourselves?

Three arms, every cell run back-to-back in ONE session against ONE anchor build
(the standing rule -- a stored anchor score is not a control, EVIDENCE 8ai/8ax):

    incumbent  policy_v5.npz                 <- what ships today (md5 dc1c9acc)
    seedswap   policy_v5_s1.npz              <- free: one file, no new code
    ens2       policy_v5.npz+policy_v5_s1    <- best in the mirror, 2x inference

⚠ Anchor identities are `rule:<name>@<deck>` since day 22. `rule:crustle` is run
on `crustle_v1`, which is the cell `p33.ANCHORS` weights at 0.755 -- NOT the
`crustle` deck the E-series used, where we score 0.893 (EVIDENCE 8ax).

Weights are the §8ay-corrected field shares. ⚠ They come from 75 games and the
mirror's own 95% interval is [22.5%, 43.2%], so a weighted total carries about
±0.003 of weight uncertainty on top of the game sampling -- reported below.

    python -X utf8 scripts/p61_ens_anchors.py --matches 750
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

S0 = "out/policy_v5.npz"
S1 = "out/policy_v5_s1.npz"

CANDIDATES = [
    ("incumbent", f"bc:v5s0,net={S0}"),
    ("seedswap", f"bc:v5s1,net={S1}"),
    ("ens2", f"bc:ens01,net={S0}+{S1}"),
]

# (label, opponent spec, opponent deck, §8ay-corrected field weight)
ANCHORS = [
    # decks are each pilot's OWN tuned list, taken from
    # `agentkit.rulebased.DECK_MODULE` rather than guessed -- running a pilot
    # off its tuned deck is a different instrument worth +0.140 (§8ax), and
    # arena warns on stderr if any of these disagree with the map.
    ("alakazam5", "rule:alakazam5", "alakazam5", 0.253),
    ("archaludon", "rule:archaludon", "archaludon_ex", 0.080),
    ("crustle_v1", "rule:crustle", "crustle_v1", 0.080),
    ("garchomp", "bc:garchomp", "cynthia_garchomp", 0.067),
    ("v10", "rule:v10,noS", "lucario_v10", 0.053),
    ("dragapult", "rule:dragapult", "dragapult_ex", 0.053),
]
MIRROR_WEIGHT = 0.320

SCORE_RE = re.compile(
    r"score=([\d.]+) \[([\d.]+), ([\d.]+)\] W(\d+)/D(\d+)/L(\d+) over (\d+) games")


def run(a: str, b: str, deck_b: str, matches: int, archive: Path):
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "arena.py"),
           "play", a, b, "--matches", str(matches),
           "--deck-a", "grimmsnarl", "--deck-b", deck_b,
           "--archive", str(archive)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        print(out.stdout[-1500:])
        print(out.stderr[-1500:])
        raise SystemExit(f"arena failed: {' '.join(cmd)}")
    if out.stderr.strip():
        print("  ⚠ stderr:", out.stderr.strip()[-600:].replace("\n", " | "))
    m = None
    for line in out.stdout.splitlines():
        hit = SCORE_RE.search(line)
        if hit:
            m = hit
        if "[health]" in line and "DEGRADED" in line:
            raise SystemExit(f"🔴 DEGRADED cell: {' '.join(cmd)}")
    if m is None:
        print(out.stdout[-1500:])
        raise SystemExit("could not parse a score line")
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=750,
                    help="pairs per cell (games = 2x)")
    ap.add_argument("--log", default="out/logs/p61_ens_anchors.txt")
    args = ap.parse_args()

    archive = ROOT / "out" / "arena" / "p61_ens_anchors.jsonl"
    log = ROOT / args.log
    log.parent.mkdir(parents=True, exist_ok=True)
    # mirror terms come from p60's DIRECT head-to-heads, which are sqrt(2)
    # tighter than any two-cell delta and are already at n=2,000.
    mirror_delta = {"incumbent": 0.0, "seedswap": 0.049, "ens2": 0.041}
    scores: dict[tuple[str, str], float] = {}
    t0 = time.time()

    for aname, aspec, adeck, _w in ANCHORS:
        for cname, cspec in CANDIDATES:
            sc, lo, hi = run(cspec, aspec, adeck, args.matches, archive)
            scores[(cname, aname)] = sc
            print(f"  {cname:10} vs {aname:12} {sc:.3f} [{lo:.3f}, {hi:.3f}]",
                  flush=True)
        base = scores[("incumbent", aname)]
        print(f"  -> {aname}: seedswap {scores[('seedswap', aname)] - base:+.3f}"
              f"   ens2 {scores[('ens2', aname)] - base:+.3f}", flush=True)

        lines = ["E9 confirmation -- candidates over the weighted anchor set",
                 f"{args.matches} pairs/cell, {time.time() - t0:.0f}s so far",
                 "", f"{'anchor':>12} {'w':>6} {'incumb':>7} {'seedswap':>9} "
                 f"{'ens2':>7} {'d(swap)':>8} {'d(ens2)':>8}"]
        wsum = MIRROR_WEIGHT
        dsw = MIRROR_WEIGHT * mirror_delta["seedswap"]
        den = MIRROR_WEIGHT * mirror_delta["ens2"]
        lines.append(f"{'mirror*':>12} {MIRROR_WEIGHT:6.3f} {'--':>7} {'--':>9} "
                     f"{'--':>7} {mirror_delta['seedswap']:+8.3f} "
                     f"{mirror_delta['ens2']:+8.3f}")
        for an, _s, _d, w in ANCHORS:
            if (("incumbent", an)) not in scores:
                continue
            b = scores[("incumbent", an)]
            s = scores[("seedswap", an)]
            e = scores[("ens2", an)]
            wsum += w
            dsw += w * (s - b)
            den += w * (e - b)
            lines.append(f"{an:>12} {w:6.3f} {b:7.3f} {s:9.3f} {e:7.3f} "
                         f"{s - b:+8.3f} {e - b:+8.3f}")
        lines += ["", f"coverage {wsum:.1%} of the measured field",
                  f"WEIGHTED delta seedswap {dsw:+.4f}",
                  f"WEIGHTED delta ens2     {den:+.4f}",
                  "",
                  "* mirror terms are p60's DIRECT head-to-heads at n=2,000",
                  "  (seedswap 0.549 and ens2 0.541 vs the incumbent).",
                  "⚠ weights are 75-game estimates (§8ay): add ~±0.003 of",
                  "  weight uncertainty to each weighted total."]
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + log.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
