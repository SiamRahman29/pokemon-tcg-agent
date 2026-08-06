"""E3b: the teacher-free near-tie gate -- flip the boundary pair, play the games.

Pre-registration: docs/experiments/beyond-bc/E3b-near-tie-gate.md, frozen before
any arm ran. Sizing first (`p43_dagger_queue.py --dump-margins`), then this.

Each arm is `bc,flip<tau>` against plain `bc` in the mirror, DIRECT head-to-head,
with **the same weight file on both sides**. That makes this the one experiment
here with no training-seed term (report/STRATEGY.md 5.6): the arms are not two
networks, they are one network with one decision rule changed.

The tau=0 arm never fires -- a boundary margin is >= 0 by construction -- and is
a harness control that must read 0.500.

    python -X utf8 scripts/p59_e3_flip.py --matches 1000
    python -X utf8 scripts/p59_e3_flip.py --matches 50 --taus 0,0.5   # smoke
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCORE_RE = re.compile(
    r"score=([\d.]+) \[([\d.]+), ([\d.]+)\] W(\d+)/D(\d+)/L(\d+) over (\d+) games")
FLIP_RE = re.compile(r"flips=(\d+)/(\d+)")

# §8am, the same axis measured by temperature instead of margin. Printed beside
# our own numbers so the two instruments can be read against each other rather
# than remembered.
P26_TAU_PROBE = [
    (0.25, 14.1, 0.465), (0.50, 20.4, 0.520),
    (1.00, 30.5, 0.315), (2.00, 44.0, 0.055),
]


def run(tau: float, matches: int, net: str, archive: Path
        ) -> tuple[float, float, float, int, int, int, int, int]:
    """One arm. Returns (score, lo, hi, W, D, L, n, flips, eligible)."""
    a = f"bc:e3flip{tau:g},net={net},flip{tau:g}"
    b = f"bc:v5ctrl,net={net}"
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "arena.py"),
           "play", a, b, "--matches", str(matches),
           "--deck-a", "grimmsnarl", "--deck-b", "grimmsnarl",
           "--archive", str(archive)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        print(out.stdout[-2000:])
        print(out.stderr[-2000:])
        raise SystemExit(f"arena failed: {' '.join(cmd)}")
    # stderr on SUCCESS too (day 22, defect 3): a cell whose agent fell back to
    # index order on every decision still exits 0 and still prints a score.
    if out.stderr.strip():
        print("  ⚠ arena wrote to stderr:")
        print("   ", out.stderr.strip()[-1500:].replace("\n", "\n    "))
    score = flips = eligible = None
    for line in out.stdout.splitlines():
        hit = SCORE_RE.search(line)
        if hit:
            score = hit
        if "[health]" in line:
            print(f"  {line.strip()}")
            if "DEGRADED" in line:
                raise SystemExit(
                    "🔴 the agent ran its index-order fallback during this "
                    f"cell; the score is not a measurement. {' '.join(cmd)}")
            fh = FLIP_RE.search(line)
            if fh:
                flips, eligible = int(fh.group(1)), int(fh.group(2))
    if score is None:
        print(out.stdout[-2000:])
        raise SystemExit("could not parse an arena score line")
    return (float(score.group(1)), float(score.group(2)), float(score.group(3)),
            int(score.group(4)), int(score.group(5)), int(score.group(6)),
            int(score.group(7)), flips or 0, eligible or 0)


def _write_log(log: Path, rows: list, args, t0: float) -> str:
    out = ["E3b near-tie gate -- flip the boundary pair under tau, mirror, "
           f"direct, one net both sides ({args.net})",
           f"{args.matches} paired matches per arm, "
           f"{time.time() - t0:.0f}s so far", "",
           f"{'tau':>6} {'flipped':>8} {'score':>7} {'95% CI':>18} "
           f"{'W/D/L':>16} {'n':>6}"]
    for tau, rate, sc, lo, hi, w, d, ell, n in rows:
        out.append(f"{tau:6g} {rate:7.1f}% {sc:7.3f}  [{lo:.3f}, {hi:.3f}]  "
                   f"{f'{w}/{d}/{ell}':>16} {n:6d}")
    out += ["", "for comparison, §8am measured the same axis by TEMPERATURE "
            "(n=200/arm, sampling vs argmax):",
            f"{'tau':>6} {'off-argmax':>11} {'score':>7}"]
    for tau, dev, sc in P26_TAU_PROBE:
        out.append(f"{tau:6g} {dev:10.1f}% {sc:7.3f}")
    text = "\n".join(out) + "\n"
    log.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=1000,
                    help="seat-swapped pairs per arm (games = 2x this)")
    ap.add_argument("--taus", default="0,0.10,0.50,1.00,2.00")
    ap.add_argument("--net", default="out/policy_v5.npz")
    ap.add_argument("--log", default="out/logs/p59_e3_flip.txt")
    args = ap.parse_args()

    if not (ROOT / args.net).exists():
        raise SystemExit(f"missing net: {args.net}")
    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    archive = ROOT / "out" / "arena" / "p59_e3_flip.jsonl"
    rows = []
    t0 = time.time()
    log = ROOT / args.log
    log.parent.mkdir(parents=True, exist_ok=True)
    for tau in taus:
        sc, lo, hi, w, d, ell, n, flips, elig = run(
            tau, args.matches, args.net, archive)
        rate = 100.0 * flips / elig if elig else 0.0
        rows.append((tau, rate, sc, lo, hi, w, d, ell, n))
        print(f"tau={tau:<5g} flipped {rate:5.1f}% of decisions  "
              f"score={sc:.3f} [{lo:.3f}, {hi:.3f}] W{w}/D{d}/L{ell} n={n}",
              flush=True)
        # ⚠ Write after EVERY arm, not at the end. The first attempt at this
        # sweep was killed inside its first cell and took the whole log with
        # it; an arm that finished and was not recorded is an arm that has to
        # be paid for twice.
        _write_log(log, rows, args, t0)

    text = _write_log(log, rows, args, t0)
    print("\n" + text)
    print("Pre-registered: tau=0 -> 0.500 (harness control); 0.10 and 0.50 "
          "null; 1.00 <=0.40; 2.00 <=0.20.")
    print("⚠ A null at 0.50 does NOT kill E3 -- it measures |E[effect]|, and a "
          "teacher's value is bounded by E[|effect|]. See the pre-registration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
