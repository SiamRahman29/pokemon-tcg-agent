#!/usr/bin/env python
"""Run one arena cell as N parallel local processes, then pool them.

🔴 **WHY THIS EXISTS: it is §2.7's named blocker, and the blocker turns out to
be missing infrastructure rather than missing compute.**

ROADMAP §2.7 priced the clock's validation cost and then wrote, verbatim:

    at 153 s/game an n=2,000 mirror A/B is ~85 core-hours, i.e. one overnight
    run *if* `arena.py` parallelises across the 6 local cores. **Measure that
    first -- it is the number the build decision turns on.**

**It was never measured. The answers are: `arena.py` does not parallelise at
all** (no Pool, no `--jobs`, one game after another in one process), **and this
box has 12 cores, not 6.** So the sentence that closed the axis -- "the honest
decision rule for this axis is LARGE OR NOTHING" -- rested on a serial harness
nobody had checked.

⇒ **This is not a new agent idea. It is the instrument that decides whether
expensive agent ideas can be adjudicated at all**, and at least four axes were
settled at n=200 (+/-0.07) for want of it:

| axis | what it could afford | what it needed |
|---|---|---|
| B4 within-turn sequencing (§8v) | n=200 | n=2,000 |
| the clock / E18 (§8ca) | n=408 | n=2,000 |
| E19 cell A | n=1,608, days of wall clock | -- |
| game-tree search (§2) | **n=31**, rollout SE ~0.14 | -- |

§8ak measured the ladder's noise floor at **63.2 points**, so the arena is not
the weaker instrument, it is *the only one*. Making it ~10x faster changes what
this project is able to know.

**The pooling half already existed and is not reimplemented here.**
`scripts/kaggle/score.py --dir` sums the seat-corrected `score=` line each shard
prints, refuses to pool shards whose arm identities disagree (rule 20), and
flags any shard whose health line is not clean. E22 already ran this way as five
hand-started local processes; this script is the launcher that was missing.

⚠ **Shards are independent because the engine draws its own shuffles** (§8bw:
"CRN is unavailable -- the engine draws its own shuffles"), and `arena.py` seeds
no global RNG. **The one exception is `random`**, whose agent is built with a
hardcoded `seed=0`, so sharding it would run N *correlated* copies and report a
CI that is too tight. That is guarded below rather than documented, because a
silently-correlated null is exactly the failure mode §8be is named for.

    python -X utf8 scripts/shard.py --job e33a \\
        --a "bc:x,net=out/policy_v5_s2.npz" --b "rule:v10,noS" \\
        --deck-a grimmsnarl --deck-b lucario_v10 --matches 1000 --jobs 10
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "scripts" / "arena.py"
POOLER = ROOT / "scripts" / "kaggle" / "score.py"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="run one arena cell across local cores and pool it")
    ap.add_argument("--job", required=True,
                    help="shard-set name; logs land in out/logs/shards/<job>/")
    ap.add_argument("--a", required=True, help="agent spec for A")
    ap.add_argument("--b", required=True, help="agent spec for B")
    ap.add_argument("--deck-a", default="grimmsnarl")
    ap.add_argument("--deck-b", default="grimmsnarl")
    ap.add_argument("--matches", type=int, required=True,
                    help="TOTAL paired matches across all shards (2 games each)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="processes (default: cpu_count - 2, min 1)")
    ap.add_argument("--expect", type=float, default=None,
                    help="pre-registered value the pooled CI must contain")
    ap.add_argument("--allow-correlated-random", action="store_true",
                    help="permit sharding a `random` spec (see the guard below)")
    args = ap.parse_args()

    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 2)

    # 🔴 The guard. `arena.make_random_agent(deck, seed=0)` is seeded per
    # PROCESS, so N shards of a `random` arm replay one RNG stream N times.
    # Pooling them reports sqrt(N) more precision than the games contain.
    if not args.allow_correlated_random:
        for spec in (args.a, args.b):
            if spec.split(":", 1)[0] == "random":
                sys.exit(
                    "🔴 `random` is built with a hardcoded seed=0, so shards of "
                    "it are CORRELATED and the pooled CI would be too tight.\n"
                    "   Run a random cell serially with arena.py, or pass "
                    "--allow-correlated-random if you have a reason.")

    per, extra = divmod(args.matches, jobs)
    if per == 0:
        jobs, per, extra = args.matches, 1, 0
    counts = [per + (1 if i < extra else 0) for i in range(jobs)]

    log_dir = ROOT / "out" / "logs" / "shards" / args.job
    log_dir.mkdir(parents=True, exist_ok=True)
    arch_dir = ROOT / "out" / "arena"
    arch_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to reuse a job name: shard logs are globbed by `*_s*.txt`, so a
    # second run under one name pools OLD games with NEW ones silently -- the
    # archive-append defect arena.py's own header warns about, one level up.
    stale = sorted(log_dir.glob(f"{args.job}_s*.txt"))
    if stale:
        sys.exit(f"🔴 {len(stale)} shard log(s) already exist under {log_dir}.\n"
                 f"   Pooling would mix runs. Pick a new --job or delete them.")

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    procs: list[tuple[subprocess.Popen, Path]] = []
    t0 = time.time()
    print(f"{args.job}: {args.matches} matches over {jobs} shards "
          f"({counts[0]}-{counts[-1]} each) on {os.cpu_count()} cores")
    for i, m in enumerate(counts):
        out = log_dir / f"{args.job}_s{i}.txt"
        cmd = [sys.executable, "-X", "utf8", str(ARENA), "play", args.a, args.b,
               "--deck-a", args.deck_a, "--deck-b", args.deck_b,
               "--matches", str(m),
               "--archive", str(arch_dir / f"{args.job}_s{i}.jsonl")]
        fh = out.open("w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                       env=env, cwd=str(ROOT)), out))

    bad = 0
    for p, out in procs:
        if p.wait() != 0:
            bad += 1
            print(f"🔴 shard {out.name} exited {p.returncode}")
    elapsed = time.time() - t0
    print(f"{args.job}: {jobs} shards done in {elapsed:.0f}s "
          f"({args.matches * 2} games)")
    if bad:
        print(f"🔴 {bad} shard(s) failed -- pooling anyway so the partial "
              f"result is visible, but DO NOT report it as the cell.")

    pool = [sys.executable, "-X", "utf8", str(POOLER),
            "--job", args.job, "--dir", str(log_dir)]
    if args.expect is not None:
        pool += ["--expect", str(args.expect)]
    print()
    rc = subprocess.run(pool, env=env, cwd=str(ROOT)).returncode
    return rc if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
