"""E7 arena driver: the four pre-registered arms, with paired controls.

Every treatment cell runs its control back to back in the same session against
the same opponent build. No cell is ever compared against a stored number --
`rule:crustle`'s score moved by +0.100 between two 08-02 runs with no change on
our side, and the archive recorded both under one unversioned name
(`arena.build_agent` used to write `rule:<name>` with no version). A stored
anchor score is therefore not a control, and this repo has the receipts.

🔴 **And day 22 found the +0.100 was mostly the DECK, not the code** (§8ax): the
two runs also swapped `crustle_v1` for `crustle`, a 20-of-60-slot change worth
**+0.140** measured with the pilot held fixed. Arm B below runs `crustle`. That
does not touch E7's delta -- both arms ran against this cell back-to-back, which
is exactly what this docstring is about -- but arm B is **not** the 0.755 cell
`p33.ANCHORS` weights. `arena` now archives `rule:<name>@<deck>` and warns on a
mismatch; the warning goes to stderr and `run()` below echoes it.

Arms (docs/experiments/embeddings/E7-card-attributes.md):

    A  mirror, DIRECT v6 vs control      33.3% field weight, one run not two
    B  rule:crustle       4/4 opp Pokemon in vocabulary
    C  rule:v10,noS       0/6            <- the hypothesis
    D  rule:alakazam5    21/23           22% field share

The pre-registered prediction is an ASYMMETRY: gain(C) > gain(B). A uniform
gain across anchors is a better feature block, not evidence for the
out-of-vocabulary mechanism.

    python -X utf8 scripts/p56_e7_arena.py --matches 150
    python -X utf8 scripts/p56_e7_arena.py --matches 1000 --arms A,C
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# arm -> (opponent spec, opponent deck, note)
ARMS = {
    "A": (None, "grimmsnarl", "mirror, direct"),
    "B": ("rule:crustle", "crustle", "4/4 opp Pokemon in vocab"),
    "C": ("rule:v10,noS", "lucario_v10", "0/6 opp Pokemon in vocab"),
    "D": ("rule:alakazam5", "alakazam5", "21/23 opp Pokemon in vocab"),
}

SCORE_RE = re.compile(
    r"score=([\d.]+) \[([\d.]+), ([\d.]+)\].*over (\d+) games")


def run(a: str, b: str, deck_a: str, deck_b: str, matches: int,
        archive: Path) -> tuple[float, float, float, int]:
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "arena.py"),
           "play", a, b, "--matches", str(matches),
           "--deck-a", deck_a, "--deck-b", deck_b, "--archive", str(archive)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        print(out.stdout[-2000:])
        print(out.stderr[-2000:])
        raise SystemExit(f"arena failed: {' '.join(cmd)}")
    # ⚠ stderr is read on SUCCESS too -- see the same block in p57. A cell whose
    # agent fell back to index order still exits 0 and still prints a score.
    if out.stderr.strip():
        print("  ⚠ arena wrote to stderr:")
        print("   ", out.stderr.strip()[-1500:].replace("\n", "\n    "))
    m = None
    for line in out.stdout.splitlines():
        hit = SCORE_RE.search(line)
        if hit:
            m = hit
        if "[health]" in line:
            print(f"  {line.strip()}")
            if "DEGRADED" in line:
                raise SystemExit(
                    "🔴 the agent ran its index-order fallback during this "
                    f"cell. The score is not a measurement of the net. "
                    f"Cell: {' '.join(cmd)}")
    if m is None:
        print(out.stdout[-2000:])
        raise SystemExit("could not parse an arena score line")
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)),
            int(m.group(4)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=150,
                    help="seat-swapped pairs per cell (games = 2x this)")
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--treat", default="out/policy_v6_s{s}.npz")
    ap.add_argument("--ctrl", default="out/policy_v5c_s{s}.npz")
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    archive = ROOT / "out" / "arena" / "p56_e7.jsonl"
    rows: list[tuple] = []

    for seed in seeds:
        t = args.treat.format(s=seed)
        c = args.ctrl.format(s=seed)
        for nm in (t, c):
            if not (ROOT / nm).exists():
                raise SystemExit(f"missing net: {nm}")
        for arm in arms:
            opp, deck_b, note = ARMS[arm]
            if arm == "A":
                # The mirror is the one anchor where treatment-vs-control is a
                # single head-to-head: n games instead of 2n, and the estimate
                # is a difference measured WITHIN one experiment.
                sc, lo, hi, n = run(f"bc:v6s{seed},net={t}",
                                    f"bc:v5cs{seed},net={c}",
                                    "grimmsnarl", "grimmsnarl",
                                    args.matches, archive)
                rows.append((seed, arm, note, sc, lo, hi, n, None, None))
                print(f"[seed {seed}] arm {arm} ({note}): v6 {sc:.3f} "
                      f"[{lo:.3f}, {hi:.3f}] n={n}", flush=True)
            else:
                ts, tlo, thi, tn = run(f"bc:v6s{seed},net={t}", opp,
                                       "grimmsnarl", deck_b, args.matches,
                                       archive)
                cs, clo, chi, cn = run(f"bc:v5cs{seed},net={c}", opp,
                                       "grimmsnarl", deck_b, args.matches,
                                       archive)
                rows.append((seed, arm, note, ts, tlo, thi, tn, cs, ts - cs))
                print(f"[seed {seed}] arm {arm} vs {opp} ({note}): "
                      f"v6 {ts:.3f} [{tlo:.3f}, {thi:.3f}] | "
                      f"ctrl {cs:.3f} [{clo:.3f}, {chi:.3f}] | "
                      f"delta {ts - cs:+.3f}  n={tn}", flush=True)

    print("\n=== E7 summary ===")
    print(f"{'arm':4s} {'opponent':16s} {'seed':>4s} {'v6':>7s} {'ctrl':>7s} "
          f"{'delta':>7s} {'n':>6s}")
    for seed, arm, note, sc, lo, hi, n, cs, d in rows:
        opp = ARMS[arm][0] or "mirror(direct)"
        print(f"{arm:4s} {opp:16s} {seed:4d} {sc:7.3f} "
              f"{('%.3f' % cs) if cs is not None else '     --':>7s} "
              f"{('%+.3f' % d) if d is not None else '     --':>7s} {n:6d}")

    # the pre-registered asymmetry, averaged over seeds
    def mean_delta(arm: str):
        ds = [r[8] for r in rows if r[1] == arm and r[8] is not None]
        return sum(ds) / len(ds) if ds else None

    b, c = mean_delta("B"), mean_delta("C")
    if b is not None and c is not None:
        print(f"\npre-registered asymmetry: gain(C, out-of-vocab) {c:+.3f} "
              f"vs gain(B, in-vocab) {b:+.3f}  ->  "
              f"{'SUPPORTS' if c > b else 'DOES NOT SUPPORT'} the mechanism")
        print("  (direction only -- per rule 4 the intervals must not overlap)")
    print("\nseed floor is +/-13 Elo ~ +/-0.019 win rate; a delta under the "
          "seed spread is a null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
