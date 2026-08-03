"""Run E5's preregistered planning-compute scale cells.

Each treatment uses frozen v5 and differs only in determinizations and the
proportional per-select cap. Arena output remains the source of score/CI truth.

    python -X utf8 scripts/p51_e5_scale.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NET = Path("out/policy_v5.npz")
NET_SHA256 = "26c681c4845a7eb017def4ee5d353bbedd128767bfc034cb7091e95e0949849e"
ARMS = {
    "low": {"k": 8, "dets": 4, "budget_s": 1.0},
    "medium": {"k": 8, "dets": 8, "budget_s": 2.0},
    "high": {"k": 8, "dets": 16, "budget_s": 4.0},
    # Preregistered after the first scale curve cleared its continue gate.
    # Do not change this after seeing confirmation outcomes.
    "confirm": {"k": 8, "dets": 32, "budget_s": 8.0},
}
SCORE_RE = re.compile(
    r"A=.*?: score=([0-9.]+) \[([0-9.]+), ([0-9.]+)\] "
    r"W(\d+)/D(\d+)/L(\d+) over (\d+) games"
)
SEQ_RE = re.compile(
    r"sequencer \(.*?\): (\d+) completed, (\d+) overrules, "
    r"(\d+) errors, (\d+) budget aborts, "
    r"([0-9.]+)s total \(([0-9.]+)s/completed\)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_and_tee(cmd: list[str], log_path: Path) -> str:
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + subprocess.list2cmdline(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            chunks.append(line)
        code = proc.wait()
    if code:
        raise SystemExit(f"arena failed with exit code {code}; see {log_path}")
    return "".join(chunks)


def parse_result(output: str) -> dict:
    score_match = SCORE_RE.search(output)
    seq_match = SEQ_RE.search(output)
    if score_match is None or seq_match is None:
        raise SystemExit("arena output missing score or sequencer summary")
    score, low, high, wins, draws, losses, games = score_match.groups()
    planned, overruled, errors, aborts, sim_s, per_plan = seq_match.groups()
    return {
        "score": float(score),
        "wilson_low": float(low),
        "wilson_high": float(high),
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "games": int(games),
        "sequencer": {
            "completed": int(planned),
            "overruled": int(overruled),
            "errors": int(errors),
            "budget_aborts": int(aborts),
            "sim_s": float(sim_s),
            "seconds_per_completed": float(per_plan),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="low,medium,high")
    ap.add_argument(
        "--matches",
        type=int,
        default=100,
        help="paired matches per arm; 100 produces the preregistered 200 games",
    )
    ap.add_argument("--out-dir", default="out/e5")
    ap.add_argument("--archive-dir", default="out/arena")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = set(names) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms: {sorted(unknown)}")
    if args.matches < 1:
        raise SystemExit("--matches must be positive")

    net_path = ROOT / NET
    if not net_path.exists():
        raise SystemExit(f"missing frozen baseline: {net_path}")
    got_hash = sha256(net_path)
    if got_hash != NET_SHA256:
        raise SystemExit(f"v5 hash changed: {got_hash} != {NET_SHA256}")

    out_dir = ROOT / args.out_dir
    archive_dir = ROOT / args.archive_dir
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "E5",
        "status": "scale_probe",
        "started": time.time(),
        "baseline": str(NET).replace("\\", "/"),
        "baseline_sha256": got_hash,
        "deck": "grimmsnarl",
        "paired_matches_per_arm": args.matches,
        "games_per_arm": 2 * args.matches,
        "arms": names,
        "fixed_candidate_count": 8,
        "reply": True,
        "results": {},
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "experiment": "E5",
            "baseline_sha256": got_hash,
            "paired_matches_per_arm": args.matches,
            "fixed_candidate_count": 8,
            "reply": True,
        }
        mismatches = {
            key: (prior.get(key), value)
            for key, value in expected.items()
            if prior.get(key) != value
        }
        if mismatches:
            raise SystemExit(f"existing E5 manifest is incompatible: {mismatches}")
        manifest["started"] = prior.get("started", manifest["started"])
        manifest["results"] = prior.get("results", {})
        manifest["arms"] = list(dict.fromkeys(prior.get("arms", []) + names))

    for name in names:
        config = ARMS[name]
        log_path = out_dir / f"{name}.log"
        archive_path = archive_dir / f"e5_{name}_vs_control.jsonl"
        if not args.force and (log_path.exists() or archive_path.exists()):
            raise SystemExit(
                f"{name}: output exists; preserve it or pass --force deliberately"
            )
        treatment = (
            f"bc:e5-{name},net={NET.as_posix()},seq,reply,"
            f"sk{config['k']},sd{config['dets']},sb{config['budget_s']}"
        )
        control = f"bc:e5-control,net={NET.as_posix()}"
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            "scripts/arena.py",
            "play",
            treatment,
            control,
            "--deck-a",
            "grimmsnarl",
            "--deck-b",
            "grimmsnarl",
            "--matches",
            str(args.matches),
            "--archive",
            str(archive_path.relative_to(ROOT)),
        ]
        if args.dry_run:
            print(subprocess.list2cmdline(cmd))
            continue

        print(f"\n=== E5 {name}: M={config['dets']} cap={config['budget_s']}s ===")
        output = run_and_tee(cmd, log_path)
        result = parse_result(output)
        result.update(config)
        result["log"] = str(log_path.relative_to(ROOT)).replace("\\", "/")
        result["archive"] = str(archive_path.relative_to(ROOT)).replace("\\", "/")
        manifest["results"][name] = result
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    if not args.dry_run:
        manifest["finished"] = time.time()
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nE5_SCALE_OK {len(names)} arms -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
