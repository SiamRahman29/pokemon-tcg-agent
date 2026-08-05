"""E8 arena driver: the v7 vocabulary remap (PAD + UNK) vs a v5 control.

Two interventions ship together in `--vocab`, and they have very different
reach, so the arms are chosen to separate them:

    pad   `padding_idx=0` pins the empty-slot row at zero and stops its
          gradient. FIELD-WIDE -- 25.5% of every game's slot lookups are id 0,
          in the mirror as much as anywhere. Arm A measures this.
    UNK   every card the corpus never contained routes to one trained row
          instead of a random untrained one. Concentrated on the two anchors
          whose Pokemon are out of vocabulary (arm C 0/6, arm D 2/4); arm B is
          4/4 in vocabulary and is the NEGATIVE CONTROL -- UNK cannot fire
          there, so its delta must be a null or the mechanism is not what the
          gain is.

`--treat`/`--ctrl` take a {s} placeholder for the seed. The pad-only arm runs
as its own treatment against the same control:

    python -X utf8 scripts/p57_e8_arena.py --matches 300
    python -X utf8 scripts/p57_e8_arena.py --matches 300 \
        --treat out/policy_v7pad_s{s}.npz --tag v7pad

⚠ Arms B/C/D are a DIFFERENCE OF TWO INDEPENDENT CELLS against a third party,
so their resolution is ~sqrt(2)x a single cell's: +/-0.080 at n=300 games,
+/-0.057 at n=600, +/-0.022 at n=2000. Arm A is a direct head-to-head and is
2x tighter for the same games. A screen at n=600 cannot resolve anything under
+/-0.057 -- that is uninformative, not null (EVIDENCE 8aq made exactly this
mistake).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# arm -> (opponent spec, opponent deck, note)
# Coverage is measured, not remembered: Pokemon-in-slot_emb / all-cards-in
# bag_emb against out/emb/vocab.json.
ARMS = {
    "A": (None, "grimmsnarl", "mirror direct; 6/6, 19/19 -- pad only"),
    "B": ("rule:crustle", "crustle", "4/4, 24/24 -- UNK cannot fire"),
    "C": ("rule:v10,noS", "lucario_v10", "0/6, 7/17 -- UNK fires hardest"),
    "D": ("rule:archaludon", "archaludon_ex", "2/4, 14/15"),
    # 22.0% of the field -- 5.5x arm C's weight. Only 2 of 9 Pokemon are out of
    # vocabulary, so if the C effect is real but does not appear here, the
    # mechanism is confined to the one archetype the corpus never contained and
    # is worth ~nothing weighted.
    "E": ("rule:alakazam5", "alakazam5", "7/9, 21/23 -- UNK fires weakly"),
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
    m = None
    for line in out.stdout.splitlines():
        hit = SCORE_RE.search(line)
        if hit:
            m = hit
    if m is None:
        print(out.stdout[-2000:])
        raise SystemExit("could not parse an arena score line")
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)),
            int(m.group(4)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=300,
                    help="seat-swapped pairs per cell (games = 2x this)")
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--treat", default="out/policy_v7_s{s}.npz")
    ap.add_argument("--ctrl", default="out/policy_v5c_s{s}.npz")
    ap.add_argument("--tag", default="v7", help="label for the treatment arm")
    ap.add_argument("--archive", default="out/arena/p57_e8.jsonl")
    args = ap.parse_args()

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    archive = ROOT / args.archive
    rows: list[tuple] = []

    for seed in seeds:
        t = args.treat.format(s=seed)
        c = args.ctrl.format(s=seed)
        for nm in (t, c):
            if not (ROOT / nm).exists():
                raise SystemExit(f"missing net: {nm}")
        for arm in arms:
            opp, deck_b, note = ARMS[arm]
            ta, ca = f"bc:{args.tag}s{seed},net={t}", f"bc:v5cs{seed},net={c}"
            if arm == "A":
                # The mirror is the one anchor where treatment-vs-control is a
                # single head-to-head: n games instead of 2n, and the estimate
                # is a difference measured WITHIN one experiment.
                sc, lo, hi, n = run(ta, ca, "grimmsnarl", "grimmsnarl",
                                    args.matches, archive)
                rows.append((seed, arm, note, sc, lo, hi, n, None, None))
                print(f"[seed {seed}] arm {arm} ({note}): {args.tag} {sc:.3f} "
                      f"[{lo:.3f}, {hi:.3f}] n={n}", flush=True)
            else:
                ts, tlo, thi, tn = run(ta, opp, "grimmsnarl", deck_b,
                                       args.matches, archive)
                cs, clo, chi, cn = run(ca, opp, "grimmsnarl", deck_b,
                                       args.matches, archive)
                rows.append((seed, arm, note, ts, tlo, thi, tn, cs, ts - cs))
                print(f"[seed {seed}] arm {arm} vs {opp} ({note}): "
                      f"{args.tag} {ts:.3f} [{tlo:.3f}, {thi:.3f}] | "
                      f"ctrl {cs:.3f} [{clo:.3f}, {chi:.3f}] | "
                      f"delta {ts - cs:+.3f}  n={tn}", flush=True)

    print(f"\n=== E8 summary ({args.tag} vs v5c) ===")
    print(f"{'arm':4s} {'opponent':16s} {'seed':>4s} {args.tag:>7s} "
          f"{'ctrl':>7s} {'delta':>7s} {'n':>6s}")
    for seed, arm, note, sc, lo, hi, n, cs, d in rows:
        opp = ARMS[arm][0] or "mirror(direct)"
        print(f"{arm:4s} {opp:16s} {seed:4d} {sc:7.3f} "
              f"{('%.3f' % cs) if cs is not None else '     --':>7s} "
              f"{('%+.3f' % d) if d is not None else '     --':>7s} {n:6d}")

    def mean_delta(arm: str):
        ds = [r[8] for r in rows if r[1] == arm and r[8] is not None]
        return sum(ds) / len(ds) if ds else None

    # Pooled mirror, both seeds -- the pad half, at the weight the field
    # actually gives it (0.333) and on the tightest instrument we own.
    ma = [r for r in rows if r[1] == "A"]
    if ma:
        sc = sum(r[3] * r[6] for r in ma) / sum(r[6] for r in ma)
        print(f"\narm A pooled (pad, field-wide): {sc:.3f} over "
              f"{sum(r[6] for r in ma)} games")
    b, c, d = mean_delta("B"), mean_delta("C"), mean_delta("D")
    if b is not None and c is not None:
        oov = [x for x in (c, d) if x is not None]
        # ⛔ This test is only meaningful for a treatment that HAS a UNK row.
        # `--tag v7pad` has none -- it keeps all 1300 rows and changes only
        # padding_idx -- so an out-of-vocab/in-vocab gap there is evidence about
        # something else entirely. The first version of this line printed
        # "SUPPORTS the UNK mechanism" under v7pad, which is unreadable nonsense
        # sitting in a permanent log.
        if "pad" in args.tag or not args.treat.count("v7_"):
            print(f"asymmetry: out-of-vocab {sum(oov) / len(oov):+.3f} vs "
                  f"in-vocab (B) {b:+.3f}  --  ⚠ NOT a UNK test: '{args.tag}' "
                  "has no UNK row. Interpret as a plain arm contrast.")
        else:
            print(f"pre-registered asymmetry: out-of-vocab "
                  f"{sum(oov) / len(oov):+.3f} vs in-vocab (B) {b:+.3f}  ->  "
                  f"{'SUPPORTS' if sum(oov) / len(oov) > b else 'DOES NOT SUPPORT'}"
                  " the UNK mechanism")
            print("  (direction only -- per rule 4 intervals must not overlap)")
    # Two INDEPENDENT cells, so the delta's se is sqrt(2)x one cell's -- the
    # earlier form printed one cell's width and understated this by 41%.
    n_games = 2 * args.matches
    per_seed = 1.96 * (2 * 0.25 / n_games) ** 0.5
    print(f"\n⚠ two-cell 95% resolution: +/-{per_seed:.3f} per seed, "
          f"+/-{per_seed / len(seeds) ** 0.5:.3f} pooled over {len(seeds)} "
          "seeds. A delta inside it is UNINFORMATIVE, not null. Arm A is a "
          "DIRECT head-to-head and is sqrt(2)x tighter. Seed floor +/-0.019.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
