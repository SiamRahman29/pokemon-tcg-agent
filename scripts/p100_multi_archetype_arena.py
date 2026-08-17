"""Arena matrix: shared v5_s2_all net on named archetype decks.

Rules off (noChip,noSpread,noSrc). Opponents: field rule pilots + grimmsnarl
v5_s2 / v5_s2_all. No rule pilots for Garchomp or Teal Mask Ogerpon.

    python -X utf8 scripts/p100_multi_archetype_arena.py
    python -X utf8 scripts/p100_multi_archetype_arena.py --only arch,cru,oger
    python -X utf8 scripts/p100_multi_archetype_arena.py --matches 50   # smoke
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "scripts" / "arena.py"
LOG_DIR = ROOT / "out" / "logs"
NET_ALL = "out/policy_v5_s2_all.npz"
NET_V5S2 = "out/policy_v5_s2.npz"
RULES_OFF = "noChip,noSpread,noSrc"

AGENTS = [
    ("luc", "lucario_v10"),
    ("gar", "cynthia_garchomp"),
    ("ala", "alakazam5"),
    ("arch", "archaludon_ex"),
    ("cru", "crustle"),
    ("oger", "teal_mask_ogerpon"),
]

OPPONENTS = [
    ("rule:v10,noS", "lucario_v10", 250),
    ("rule:crustle", "crustle_v1", 200),
    ("rule:alakazam5", "alakazam5", 200),
    ("rule:archaludon", "archaludon_ex", 200),
    ("rule:dragapult", "dragapult_ex", 200),
    (f"bc:v5s2,net={NET_V5S2},{RULES_OFF}", "grimmsnarl", 200),
    (f"bc:all,net={NET_ALL},{RULES_OFF}", "grimmsnarl", 200),
]


def cell_name(tag: str, deck_a: str, b_spec: str, deck_b: str) -> str:
    b_short = b_spec.split(",")[0].replace(":", "").replace("/", "_")
    return f"p100_{tag}_{deck_a}_vs_{b_short}_{deck_b}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=0,
                    help="override paired matches for every cell (0 = per-opp)")
    ap.add_argument("--only", default="",
                    help="comma tags to run (luc,gar,ala,arch,cru,oger); "
                         "empty = all")
    ap.add_argument("--summary", default="",
                    help="summary log path (default: out/logs/p100_*_summary.txt)")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    summary = Path(args.summary) if args.summary else (
        LOG_DIR / (f"p100_{'_'.join(sorted(only))}_summary.txt"
                   if only else "p100_multi_archetype_summary.txt"))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for net in (NET_ALL, NET_V5S2):
        if not (ROOT / net).exists():
            raise SystemExit(f"missing {net}")

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines = [f"=== p100 multi-archetype arena start {stamp} ===",
             f"net={NET_ALL} rules={RULES_OFF}",
             "note: no rule:garchomp or rule:ogerpon pilots exist",
             f"agents={only or [t for t, _ in AGENTS]}", ""]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    for tag, deck_a in AGENTS:
        if only and tag not in only:
            continue
        a_spec = f"bc:{tag},net={NET_ALL},{RULES_OFF}"
        for b_spec, deck_b, n_default in OPPONENTS:
            n = args.matches or n_default
            name = cell_name(tag, deck_a, b_spec, deck_b)
            log_path = LOG_DIR / f"{name}_arena.txt"
            header = (f"=== {tag}@{deck_a} vs {b_spec}@{deck_b} "
                      f"matches={n} ===")
            print(header, flush=True)
            with summary.open("a", encoding="utf-8") as sf:
                sf.write(header + "\n")

            cmd = [
                sys.executable, "-X", "utf8", str(ARENA), "play",
                a_spec, b_spec,
                "--deck-a", deck_a, "--deck-b", deck_b,
                "--matches", str(n),
                "--archive", str(ROOT / "out" / "arena" / f"{name}.jsonl"),
            ]
            t0 = time.time()
            with log_path.open("w", encoding="utf-8") as lf:
                lf.write(header + "\n")
                lf.flush()
                proc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf,
                                      stderr=subprocess.STDOUT)
            elapsed = time.time() - t0
            # Pull the score / health lines into the summary.
            body = log_path.read_text(encoding="utf-8", errors="replace")
            score_lines = [ln for ln in body.splitlines()
                           if ln.startswith("A=") or "[health]" in ln
                           or ln.startswith("bc:") or ln.startswith("rule:")]
            tail = "\n".join(score_lines[-4:])
            status = f"exit={proc.returncode} elapsed={elapsed:.1f}s log={log_path.name}"
            print(tail, flush=True)
            print(status, flush=True)
            with summary.open("a", encoding="utf-8") as sf:
                sf.write(tail + "\n" + status + "\n\n")
            if proc.returncode != 0:
                print(f"FAILED cell {name}", flush=True)
                return proc.returncode

    print(f"done -> {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
