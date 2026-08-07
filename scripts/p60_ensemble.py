"""E9: does a vote of independently-trained nets beat a single net?

Pre-registration: docs/experiments/E9-ensemble.md, frozen before any cell ran.

Arms, all mirror + DIRECT head-to-head with fixed weight files on both sides
(so there is no training-seed term and the printed interval is the whole one):

    C  policy_v5 vs policy_v5_s1      <- run FIRST; a rival hypothesis, not a
                                         control. If one seed is simply better,
                                         "ship the other file" beats ensembling
                                         and costs nothing.
    A  ens(s0,s1) vs policy_v5        <- the shipping question
    B  ens(s0,s1) vs policy_v5_s1     <- does it beat the OTHER member too?
    D  ens(s0,s0) vs policy_v5        <- degenerate: a net voting with itself
                                         MUST read 0.500. Bug check, cheap.

    python -X utf8 scripts/p60_ensemble.py --matches 1000 --arms C,A,B
    python -X utf8 scripts/p60_ensemble.py --matches 40 --arms D    # smoke
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

# arm -> (spec A, spec B, note)
ARMS = {
    "C": (f"bc:v5s0,net={S0}", f"bc:v5s1,net={S1}",
          "member vs member -- the rival hypothesis"),
    "A": (f"bc:ens01,net={S0}+{S1}", f"bc:v5s0,net={S0}",
          "ensemble vs the SHIPPED member"),
    "B": (f"bc:ens01,net={S0}+{S1}", f"bc:v5s1,net={S1}",
          "ensemble vs the other member"),
    "D": (f"bc:ensdegen,net={S0}+{S0}", f"bc:v5s0,net={S0}",
          "degenerate ensemble -- MUST read 0.500"),
}

SCORE_RE = re.compile(
    r"score=([\d.]+) \[([\d.]+), ([\d.]+)\] W(\d+)/D(\d+)/L(\d+) over (\d+) games")


def run(a: str, b: str, matches: int, archive: Path):
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
    if out.stderr.strip():
        print("  ⚠ arena wrote to stderr:")
        print("   ", out.stderr.strip()[-1200:].replace("\n", "\n    "))
    m = None
    for line in out.stdout.splitlines():
        hit = SCORE_RE.search(line)
        if hit:
            m = hit
        if "[health]" in line:
            print(f"  {line.strip()}")
            if "DEGRADED" in line:
                raise SystemExit("🔴 an agent fell back to index order; the "
                                 f"score is not a measurement. {' '.join(cmd)}")
    if m is None:
        print(out.stdout[-2000:])
        raise SystemExit("could not parse an arena score line")
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)),
            int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=1000)
    ap.add_argument("--arms", default="C,A,B")
    ap.add_argument("--log", default="out/logs/p60_ensemble.txt")
    args = ap.parse_args()

    for p in (S0, S1):
        if not (ROOT / p).exists():
            raise SystemExit(f"missing net: {p}")
    archive = ROOT / "out" / "arena" / "p60_ensemble.jsonl"
    log = ROOT / args.log
    log.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for arm in [a.strip().upper() for a in args.arms.split(",") if a.strip()]:
        a, b, note = ARMS[arm]
        sc, lo, hi, w, d, ell, n = run(a, b, args.matches, archive)
        rows.append((arm, note, sc, lo, hi, w, d, ell, n))
        verdict = "RESOLVED" if (lo > 0.5 or hi < 0.5) else "null (CI spans 0.5)"
        print(f"[{arm}] {note}\n     A={sc:.3f} [{lo:.3f}, {hi:.3f}] "
              f"W{w}/D{d}/L{ell} n={n}  -> {verdict}", flush=True)
        out = ["E9 ensemble -- mirror, direct, fixed weight files both sides",
               f"{args.matches} paired matches per arm, "
               f"{time.time() - t0:.0f}s so far", "",
               f"{'arm':>4} {'score':>7} {'95% CI':>18} {'W/D/L':>16} "
               f"{'n':>6}  note"]
        for r in rows:
            out.append(f"{r[0]:>4} {r[2]:7.3f}  [{r[3]:.3f}, {r[4]:.3f}]  "
                       f"{f'{r[5]}/{r[6]}/{r[7]}':>16} {r[8]:6d}  {r[1]}")
        log.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\nBars: C decides whether the shipped file is the wrong one; "
          "promote the ensemble only if A resolves above 0.5 AND B does not "
          "lose; then confirm on the weighted anchors before any submission "
          "talk (a submission evicts the 990.7).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
