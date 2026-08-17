"""One policy net, on OUR deck, against the whole available field.

Complements p100_multi_archetype_arena.py rather than repeating it: that matrix
puts the shared v5_s2_all net on SIX archetype decks against the field. This one
fixes the deck at grimmsnarl and varies the NET, which is what the EVIDENCE §1a
corpus-composition thread needs -- top10 / grimm20 / v5_s2 all pilot the same 60
cards, so any difference between them is the corpus and nothing else.

⚠ SEAT SEMANTICS, stated once because misreading them invalidates every cell:
`arena.py play A B` reports `A=...: score=`, which is **A's** win rate. Here A is
ALWAYS the net under test on `grimmsnarl`, so higher is always better for us.
Matches are seat-paired (N matches -> 2N games, both seats), so seat bias is
already controlled inside each cell.

    python -X utf8 scripts/p102_net_vs_field.py --net out/policy_v5_s2_grimm20.npz --tag grimm20
    python -X utf8 scripts/p102_net_vs_field.py --net out/policy_v5_s2_top10.npz  --tag top10
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "scripts" / "arena.py"
LOG_DIR = ROOT / "out" / "logs"
RULES_OFF = "noChip,noSpread,noSrc"
NET_ALL = "out/policy_v5_s2_all.npz"
NET_V5S2 = "out/policy_v5_s2.npz"
OUR_DECK = "grimmsnarl"

# (B spec, B deck, paired matches). Two families:
#   rule:*  -- handwritten pilots, the only non-learned opposition available
#   bc:*    -- the shared v5_s2_all net piloting an archetype deck (this is what
#              "the archetype BC agents" are; there is no per-archetype NET)
# ⚠ no rule pilot exists for garchomp or ogerpon (p100's own note), so those two
# archetypes appear only in their bc form.
OPPONENTS: list[tuple[str, str, int]] = [
    ("rule:v10,noS", "lucario_v10", 250),
    ("rule:crustle", "crustle_v1", 200),
    ("rule:alakazam5", "alakazam5", 200),
    ("rule:archaludon", "archaludon_ex", 200),
    ("rule:dragapult", "dragapult_ex", 200),
    (f"bc:all_luc,net={NET_ALL},{RULES_OFF}", "lucario_v10", 200),
    (f"bc:all_gar,net={NET_ALL},{RULES_OFF}", "cynthia_garchomp", 200),
    (f"bc:all_ala,net={NET_ALL},{RULES_OFF}", "alakazam5", 200),
    (f"bc:all_arch,net={NET_ALL},{RULES_OFF}", "archaludon_ex", 200),
    (f"bc:all_cru,net={NET_ALL},{RULES_OFF}", "crustle", 200),
    (f"bc:all_oger,net={NET_ALL},{RULES_OFF}", "teal_mask_ogerpon", 200),
    (f"bc:all_grimm,net={NET_ALL},{RULES_OFF}", "grimmsnarl", 200),
]

# The corpus-composition nets, played against each other on the same 60 cards.
# Every one of these is architecturally identical (verified byte-for-byte at
# export), so a cell here isolates the CORPUS and nothing else. A net is skipped
# when it would face itself -- self-play is 0.500 by construction, not a
# measurement.
NETS: dict[str, str] = {
    "v5s2": NET_V5S2,
    "hostall": "out/policy_v5_s2_hostall.npz",
    "top10": "out/policy_v5_s2_top10.npz",
    "grimm20": "out/policy_v5_s2_grimm20.npz",
    "grimm50": "out/policy_v5_s2_grimm50.npz",
}

SCORE_RE = re.compile(r"score=([0-9.]+) \[([0-9.]+), ([0-9.]+)\] "
                      r"W(\d+)/D(\d+)/L(\d+) over (\d+) games")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="the net under test (A)")
    ap.add_argument("--tag", required=True, help="short name for logs")
    ap.add_argument("--matches", type=int, default=0,
                    help="override paired matches per cell (0 = per-opponent)")
    ap.add_argument("--deck", default=OUR_DECK, help="the deck A pilots")
    args = ap.parse_args()

    if not (ROOT / args.net).exists():
        raise SystemExit(f"missing {args.net}")
    for net in (NET_ALL, NET_V5S2):
        if not (ROOT / net).exists():
            raise SystemExit(f"missing {net}")

    # Field + the sibling nets, minus self.
    opponents = list(OPPONENTS)
    for name, path in NETS.items():
        if Path(path).as_posix() == Path(args.net).as_posix():
            continue
        if not (ROOT / path).exists():
            raise SystemExit(f"missing {path}")
        opponents.append((f"bc:{name},net={path},{RULES_OFF}", "grimmsnarl", 300))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "out" / "arena").mkdir(parents=True, exist_ok=True)
    a_spec = f"bc:{args.tag},net={args.net},{RULES_OFF}"
    summary = LOG_DIR / f"p102_{args.tag}_field_summary.txt"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    head = [f"=== p102 {args.tag} vs field  {stamp} ===",
            f"A = {a_spec} @ {args.deck}   (score is A's; higher = better for us)",
            f"{len(opponents)} cells", ""]
    summary.write_text("\n".join(head) + "\n", encoding="utf-8")
    print("\n".join(head), flush=True)

    rows = []
    for b_spec, deck_b, n_default in opponents:
        n = args.matches or n_default
        b_short = b_spec.split(",")[0].replace(":", "")
        name = f"p102_{args.tag}_vs_{b_short}_{deck_b}"
        log_path = LOG_DIR / f"{name}.txt"
        print(f"--- {args.tag}@{args.deck} vs {b_spec.split(',')[0]}@{deck_b} "
              f"(n={n}) ---", flush=True)
        cmd = [sys.executable, "-X", "utf8", str(ARENA), "play",
               a_spec, b_spec,
               "--deck-a", args.deck, "--deck-b", deck_b,
               "--matches", str(n),
               "--archive", str(ROOT / "out" / "arena" / f"{name}.jsonl")]
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf,
                                  stderr=subprocess.STDOUT)
        body = log_path.read_text(encoding="utf-8", errors="replace")
        m = SCORE_RE.search(body)
        if proc.returncode != 0 or not m:
            print(f"  FAILED (exit={proc.returncode}) -- see {log_path.name}",
                  flush=True)
            print(body.strip().splitlines()[-5:] if body.strip() else "(empty)")
            rows.append((b_short, deck_b, None))
            continue
        sc, lo, hi, w, d, l, g = m.groups()
        health = "OK" if "fallbacks=0 net_missing=0" in body else "⚠ CHECK"
        line = (f"  {sc} [{lo}, {hi}]  W{w}/D{d}/L{l} over {g} games  "
                f"({time.time() - t0:.0f}s, health {health})")
        print(line, flush=True)
        rows.append((b_short, deck_b, (float(sc), float(lo), float(hi),
                                       int(w), int(d), int(l), int(g))))
        with summary.open("a", encoding="utf-8") as sf:
            sf.write(f"{b_short}@{deck_b}\n{line}\n")

    print(f"\n=== {args.tag} vs field ===", flush=True)
    print(f"{'opponent':32s} {'score':>7s}  {'95% CI':>16s}  {'n':>5s}")
    for b_short, deck_b, r in rows:
        key = f"{b_short}@{deck_b}"
        if r is None:
            print(f"{key:32s} {'FAILED':>7s}")
            continue
        sc, lo, hi, w, d, l, g = r
        print(f"{key:32s} {sc:7.3f}  [{lo:.3f}, {hi:.3f}]  {g:5d}")
    with summary.open("a", encoding="utf-8") as sf:
        sf.write(f"\ndone {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    print(f"\nsummary -> {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
