"""Evaluate a distilled Transformer against its teacher and the rule bots.

Play mode runs this shard's share of every matchup through `arena.py play`
(paired, seat-swapped), one archive per matchup. Summary mode reads every
archive under a directory and prints the Transformer's score per opponent with
a 95% interval, next to the teacher's own score from the teacher games, and
the distillation gate: score >= 0.45 against v5_s2 (frozen 2026-11-24).

    # one shard of four (Kaggle: one process per core)
    python -X utf8 scripts/tx_eval.py --model out/tx_model/model.npz \\
        --shard 0 --shards 4 --out out/tx_eval/s0
    python -X utf8 scripts/tx_eval.py --summarize out/tx_eval
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEACHER = "bc:v5s2,net=out/policy_v5_s2.npz,noChip,noSpread,noSrc"
GATE = 0.45
# (tag, opponent spec, opponent deck, paired matches = 2 games each)
MATCHUPS = [
    ("teacher", TEACHER, "grimmsnarl", 1000),
    ("v10", "rule:v10,noS", "lucario_v10", 200),
    ("crustle", "rule:crustle", "crustle_v1", 200),
    ("alakazam", "rule:alakazam5", "alakazam5", 200),
    ("archaludon", "rule:archaludon", "archaludon_ex", 200),
    ("dragapult", "rule:dragapult", "dragapult_ex", 200),
    ("iono", "rule:iono", "iono", 200),
    ("abomasnow", "rule:abomasnow", "mega_abomasnow_ex", 200),
    ("lucario", "rule:lucario", "mega_lucario_ex", 200),
]
# v5_s2's own score in teacher-v1 (30k games), for comparison.
TEACHER_SCORE = {"v10": 0.585, "crustle": 0.784, "alakazam": 0.805, "archaludon": 0.714,
                 "dragapult": 0.814, "iono": 0.742, "abomasnow": 0.472, "lucario": 0.650,
                 "teacher": 0.500}


def play(args) -> int:
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for tag, spec, deck, matches in MATCHUPS:
        n = matches // args.shards + (args.shard < matches % args.shards)
        if n == 0:
            continue
        cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts/arena.py"), "play",
               f"tx:{args.model}", spec, "--deck-a", "grimmsnarl", "--deck-b", deck,
               "--matches", str(n), "--archive", str(out / f"{tag}.jsonl")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = [l for l in r.stdout.splitlines() if "score=" in l]
        print(f"{tag}: {tail[-1] if tail else r.stderr[-400:]}", flush=True)
        rc |= r.returncode
    return rc


def summarize(args) -> int:
    tally: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for f in sorted((ROOT / args.summarize).rglob("*.jsonl")):
        tag = f.stem
        for line in f.open():
            r = json.loads(line)
            me = 0 if r["agent0"].startswith("tx:") else 1
            t = tally[tag]
            t[0] += 0.5 if r["winner"] == 2 else float(r["winner"] == me)
            t[1] += 1
    rows = []
    print(f"{'opponent':<12}{'games':>7}{'tx score':>10}   95% CI          teacher")
    for tag, *_ in MATCHUPS:
        if tag not in tally:
            continue
        s, n = tally[tag]
        p = s / n
        z = 1.96
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append(dict(opponent=tag, games=n, score=p, lo=centre - half, hi=centre + half))
        print(f"{tag:<12}{n:>7}{p:>10.3f}   [{centre - half:.3f}, {centre + half:.3f}]"
              f"   {TEACHER_SCORE.get(tag, float('nan')):.3f}")
    gate = next((r for r in rows if r["opponent"] == "teacher"), None)
    if gate:
        verdict = "PASS" if gate["score"] >= GATE else "FAIL"
        print(f"\ngate: score vs v5_s2 {gate['score']:.3f} over {gate['games']} games, "
              f"bar {GATE} -> {verdict}")
    (ROOT / args.summarize / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="exported .npz, repo-relative or absolute")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", help="this shard's archive directory")
    ap.add_argument("--summarize", help="directory of archives to summarize")
    a = ap.parse_args()
    if a.summarize:
        return summarize(a)
    if not (a.model and a.out):
        ap.error("--model and --out are required to play")
    return play(a)


if __name__ == "__main__":
    sys.exit(main())
