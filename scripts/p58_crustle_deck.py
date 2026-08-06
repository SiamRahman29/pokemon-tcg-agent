"""Separate the Crustle DECK term from the Crustle PILOT term. One session, one pilot.

**Why this exists (day 22).** §8an and §8aq both attribute a ~0.11 swing in the
`rule:crustle` anchor to the pilot's bench logic. The archives say the deck moved
too, and it moved in lockstep with the score:

    deck crustle_v1  ->  0.768 (p20)  0.748 (p34)  0.755 (p35)  0.764 (p37 ctrl)
    deck crustle     ->  0.870 (p27)  0.866 (p28)

`decks/crustle.py` (field consensus) and `decks/crustle_v1.py` (the pilot's own
list) differ in **20 of 60 slots**, and `crustle_v1.py`'s own docstring says the
pilot "scores ~20 of the consensus list's cards through a generic fallback, so it
plays them legally but badly". HANDOFF §3.2 even carries an n=20 probe reading
**0.620 on its own list vs 0.700 on the consensus one** -- the same sign, and
most of the magnitude, that §8an credits to the empty-bench guard.

`arena.build_agent` archives the anchor as `rule:<name>` with no deck, so both
pool under one identity and nothing caught it. This measures the deck term
directly with the PILOT HELD FIXED at the one version in the repo today:

    python -X utf8 scripts/p58_crustle_deck.py --matches 1000

Two cells, one net (`out/policy_v5.npz`), one pilot, back-to-back in one session
-- §8aq's own rule. If the delta reproduces ~0.11, the deck explains the whole
swing and both pilot findings are confounded. If it is ~0, the pilot findings
stand and the deck coincidence needs another explanation.

⚠ Both cells are DIRECT runs of the same net against the same pilot, so each
carries a single-cell Wilson interval (arena prints it) and the DIFFERENCE
carries sqrt(2)x that -- +/-0.031 at n=2,000 per cell. Printed below; do not
quote a delta inside it.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NET = "out/policy_v5.npz"
# (deck module, what it is)
CELLS = [
    ("crustle_v1", "the pilot's own 60 -- what p20/p34/p35/p37 played"),
    ("crustle", "the field-consensus 60 -- what p27/p28/p54/p56/p57 played"),
]

SCORE_RE = re.compile(
    r"score=([\d.]+) \[([\d.]+), ([\d.]+)\].*over (\d+) games")


def run(deck_b: str, matches: int, archive: Path) -> tuple[float, float, float, int]:
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "arena.py"),
           "play", f"bc:v5,net={NET}", "rule:crustle", "--matches", str(matches),
           "--deck-a", "grimmsnarl", "--deck-b", deck_b,
           "--archive", str(archive)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if out.returncode != 0:
        print(out.stdout[-2000:])
        print(out.stderr[-2000:])
        raise SystemExit(f"arena failed: {' '.join(cmd)}")
    # ⚠ stderr is kept even on success: `bcagent.__call__`'s catch-all prints a
    # traceback there and returns index order, which looks like a normal score
    # from outside. Discarding it is how a degraded arm publishes cleanly.
    if out.stderr.strip():
        print("  ⚠ arena wrote to stderr:")
        print("   ", out.stderr.strip()[:1500].replace("\n", "\n    "))
    m = None
    for line in out.stdout.splitlines():
        hit = SCORE_RE.search(line)
        if hit:
            m = hit
        if line.startswith("[health]"):
            print(f"  {line.strip()}")
    if m is None:
        print(out.stdout[-2000:])
        raise SystemExit("could not parse an arena score line")
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)),
            int(m.group(4)))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", type=int, default=1000,
                    help="seat-swapped pairs per cell (games = 2x this)")
    ap.add_argument("--archive", default="out/arena/p58_crustle_deck.jsonl")
    args = ap.parse_args()

    archive = ROOT / args.archive
    rows = []
    for deck_b, note in CELLS:
        sc, lo, hi, n = run(deck_b, args.matches, archive)
        rows.append((deck_b, note, sc, lo, hi, n))
        print(f"[deck {deck_b:11s}] bc:v5 {sc:.4f} [{lo:.3f}, {hi:.3f}] "
              f"n={n}   ({note})", flush=True)

    print("\n=== the DECK term, pilot held fixed at the repo's v4 ===")
    print(f"{'deck':12s} {'score':>8s} {'95% CI':>18s} {'n':>6s}")
    for deck_b, _, sc, lo, hi, n in rows:
        print(f"{deck_b:12s} {sc:8.4f}  [{lo:.3f}, {hi:.3f}] {n:6d}")
    d = rows[1][2] - rows[0][2]
    n_cell = 2 * args.matches
    res = 1.96 * (2 * 0.25 / n_cell) ** 0.5
    print(f"\nDELTA (consensus - pilot's own) = {d:+.4f}   "
          f"two-cell 95% resolution +/-{res:.4f}")
    if abs(d) <= res:
        print("  ⇒ INSIDE the interval: uninformative, NOT a null (§8aq's own")
        print("    mistake). The pilot attribution is neither confirmed nor")
        print("    refuted by this run.")
    elif d > 0:
        print("  ⇒ The DECK alone moves the anchor. §8an's +0.087..+0.102 and")
        print("    §8aq's -0.111 were measured across a deck swap out and back,")
        print("    so neither isolates the pilot. Compare against 0.111.")
    else:
        print("  ⇒ The deck moves it the OTHER way, so it cannot explain the")
        print("    published swing. The pilot attribution survives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
