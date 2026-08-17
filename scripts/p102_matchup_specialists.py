"""R0 (winners-only) specialists: bc:all@grimmsnarl vs named opponents.

    python -X utf8 scripts/p102_matchup_specialists.py run
    python -X utf8 scripts/p102_matchup_specialists.py run --only mirror,crustle
    python -X utf8 scripts/p102_matchup_specialists.py baseline --only lucario

Each arm: generate matchup shards from v5_s2_all, fine-tune winners-only
(R0), arena specialist and stock vs the same opponent.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

NET_ALL = "out/policy_v5_s2_all.npz"
RULES_OFF = "noChip,noSpread,noSrc"
POLICY_ARCH = ["--opt-cols", "37", "--state-h", "512,256", "--head-h", "256,128",
               "--pool", "--loss", "listwise"]
DECK_US = "grimmsnarl"
LOG_DIR = ROOT / "out" / "logs"
SUMMARY = LOG_DIR / "p102_matchup_specialists.txt"
SCORE_RE = re.compile(
    r"score=([\d.]+) \[([\d.]+), ([\d.]+)\].*over (\d+) games")

# (tag, opponent deck module, gid base for generation)
MATCHUPS = [
    ("mirror", "grimmsnarl", 970000),
    ("crustle", "crustle", 980000),
    ("garchomp", "cynthia_garchomp", 990000),
    ("lucario", "lucario_v10", 1_000_000),
]


def _spec(net: str, tag: str = "grim") -> str:
    return f"bc:{tag},net={net},{RULES_OFF}"


def _opp_spec() -> str:
    return f"bc:opp,net={NET_ALL},{RULES_OFF}"


def _run(cmd: list[str], log_path: Path | None = None) -> int:
    print(" ".join(cmd), flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as lf:
            return subprocess.run(cmd, cwd=str(ROOT), stdout=lf,
                                  stderr=subprocess.STDOUT, check=False).returncode
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def _parse_score(log: Path) -> tuple[float, float, float, int]:
    body = log.read_text(encoding="utf-8", errors="replace")
    for line in body.splitlines():
        m = SCORE_RE.search(line)
        if m:
            return (float(m.group(1)), float(m.group(2)),
                    float(m.group(3)), int(m.group(4)))
    raise SystemExit(f"no score in {log}")


def _selected(only: str | None) -> list[tuple[str, str, int]]:
    if not only:
        return list(MATCHUPS)
    want = {x.strip() for x in only.split(",") if x.strip()}
    got = [m for m in MATCHUPS if m[0] in want]
    missing = want - {m[0] for m in got}
    if missing:
        raise SystemExit(f"unknown matchup(s) {sorted(missing)}; "
                         f"known={[m[0] for m in MATCHUPS]}")
    return got


def _generate(tag: str, deck_b: str, gid_base: int, args: argparse.Namespace) -> int:
    out = f"artifacts/pds_spec_{tag}"
    workers = max(1, args.workers)
    per = args.games // workers
    leftover = args.games - per * workers
    procs: list[subprocess.Popen] = []
    for i in range(workers):
        n = per + (leftover if i == workers - 1 else 0)
        shard_out = f"{out}/w{i}" if workers > 1 else out
        cmd = [
            sys.executable, "-X", "utf8",
            str(ROOT / "scripts" / "p26_selfplay_gen.py"),
            "--net", NET_ALL,
            "--deck", DECK_US,
            "--deck-b", deck_b,
            "--opp", _opp_spec(),
            "--tau", str(args.tau),
            "--games", str(n),
            "--seed", str(args.seed + 1000 * i),
            "--gid-base", str(gid_base + i * 10_000),
            "--out", shard_out,
        ]
        log = LOG_DIR / f"p102_gen_{tag}_w{i}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(" ".join(cmd), flush=True)
        lf = log.open("w", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf,
                                      stderr=subprocess.STDOUT))
    rc = 0
    for p in procs:
        rc |= p.wait()
    print(f"generate {tag} games={args.games} workers={workers} rc={rc} -> {out}",
          flush=True)
    return rc


def _finetune(tag: str, args: argparse.Namespace) -> int:
    ds = f"artifacts/pds_spec_{tag}"
    out = f"out/policy_v5_s2_spec_{tag}.npz"
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "train_policy.py"),
        "--ds", ds, *POLICY_ARCH,
        "--winners-only",
        "--init", NET_ALL,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--export-last",
        "--out", out,
        "--device", args.device,
    ]
    rc = _run(cmd, LOG_DIR / f"p102_train_{tag}.log")
    if rc == 0:
        print(f"trained {tag} -> {out}", flush=True)
    return rc


def _arena(tag: str, deck_b: str, net: str, matches: int, label: str
           ) -> tuple[float, float, float, int]:
    log = LOG_DIR / f"p102_{label}_{tag}_vs_{deck_b}.txt"
    archive = ROOT / "out" / "arena" / f"p102_{label}_{tag}_vs_{deck_b}.jsonl"
    cmd = [
        sys.executable, "-X", "utf8", str(ROOT / "scripts" / "arena.py"), "play",
        _spec(net, "grim"), _opp_spec(),
        "--deck-a", DECK_US, "--deck-b", deck_b,
        "--matches", str(matches),
        "--archive", str(archive),
    ]
    rc = _run(cmd, log)
    if rc:
        raise SystemExit(f"arena failed {label}/{tag} rc={rc}")
    return _parse_score(log)


def cmd_baseline(args: argparse.Namespace) -> int:
    lines = [f"=== p102 stock baseline matches={args.matches} ==="]
    for tag, deck_b, _ in _selected(args.only):
        sc, lo, hi, n = _arena(tag, deck_b, NET_ALL, args.matches, "stock")
        line = f"stock@{tag}  WR={sc:.3f} [{lo:.3f},{hi:.3f}] n={n}"
        print(line, flush=True)
        lines.append(line)
    text = "\n".join(lines) + "\n"
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("a", encoding="utf-8") as sf:
        sf.write(text)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    lines = [f"=== p102 R0 specialists games={args.games} matches={args.matches} ==="]
    rows: list[tuple[str, float, float]] = []
    for tag, deck_b, gid in _selected(args.only):
        print(f"\n######## {tag} vs {deck_b} ########", flush=True)
        if not args.skip_gen:
            rc = _generate(tag, deck_b, gid, args)
            if rc:
                return rc
        rc = _finetune(tag, args)
        if rc:
            return rc
        sc_s, lo_s, hi_s, n_s = _arena(tag, deck_b, NET_ALL, args.matches, "stock")
        net = f"out/policy_v5_s2_spec_{tag}.npz"
        sc_t, lo_t, hi_t, n_t = _arena(tag, deck_b, net, args.matches, "spec")
        delta = sc_t - sc_s
        line = (f"{tag:8s}  stock={sc_s:.3f} [{lo_s:.3f},{hi_s:.3f}]  "
                f"spec={sc_t:.3f} [{lo_t:.3f},{hi_t:.3f}]  "
                f"delta={delta:+.3f}  n={n_t}")
        print(line, flush=True)
        lines.append(line)
        rows.append((tag, sc_s, sc_t))
    lines.append("")
    lines.append("nets:")
    for tag, _, _ in _selected(args.only):
        lines.append(f"  {tag}: out/policy_v5_s2_spec_{tag}.npz")
    text = "\n".join(lines) + "\n"
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--only", default="",
                       help="comma tags: mirror,crustle,garchomp,lucario")
        p.add_argument("--matches", type=int, default=500)
        p.add_argument("--games", type=int, default=4000)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--tau", type=float, default=0.4)
        p.add_argument("--seed", type=int, default=102)
        p.add_argument("--epochs", type=int, default=5)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        p.add_argument("--skip-gen", action="store_true")

    p_r = sub.add_parser("run", help="generate + R0 fine-tune + arena")
    add_common(p_r)
    p_r.set_defaults(func=cmd_run)

    p_b = sub.add_parser("baseline", help="stock arenas only")
    add_common(p_b)
    p_b.set_defaults(func=cmd_baseline)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
