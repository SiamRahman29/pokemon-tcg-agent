"""Run the PPO chain: one Kaggle CPU session after another, each resuming the last.

Session k is kernel `ptcg-ppo-s{k:03d}`. Session 1 starts from the distilled
checkpoint (the `--init-kernel` output); session k > 1 attaches session k-1's
output and resumes from its `ppo_state.json`. After each session this pulls
only the small files (state, history, evaluation summaries, candidates) to
`out/kaggle_out/ppo-s{k:03d}/` and prints the cycles it ran.

Stops at `--until-cycle`, on a failed session, or when a checkpoint beats the
distilled model (a file appears under state/candidates/) if
`--stop-on-candidate` is set. Submitting is left to a person.

Runs locally and uses almost no memory: it only polls `kaggle kernels status`.

    python -X utf8 scripts/ppo_chain.py --init-kernel siamrahman29/ptcg-tx-train-v1 \\
        --first 1 --until-cycle 300
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "siamrahman29"
CODE_DS = f"{USER}/ptcg-teacher-code"


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def status(slug: str) -> str:
    out = sh(["kaggle", "kernels", "status", f"{USER}/{slug}"])
    for word in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if word in out:
            return word
    return "UNKNOWN: " + out.strip()[-200:]


def launch(k: int, source: str, fresh: bool, hours: float) -> str:
    job = f"ppo-s{k:03d}"
    flag = "--init" if fresh else "--resume"
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts/kaggle/launch.py"), "push",
           "--job", job, "--shards", "1", "--out-flag=--output", "--code-ds", CODE_DS,
           "--kernel-input", source,
           "--cmd", f"scripts/ppo_session.py {flag} /kaggle/input --output out/ppo{{i}} "
                    f"--max-hours {hours}",
           "--collect", "out/ppo0,out/logs"]
    print(sh(cmd).strip().splitlines()[-1], flush=True)
    return f"ptcg-{job}"


def pull(slug: str, k: int) -> dict:
    dest = ROOT / "out" / "kaggle_out" / f"ppo-s{k:03d}"
    dest.mkdir(parents=True, exist_ok=True)
    sh(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", str(dest),
        "--file-pattern", r"ppo_state\.json|history\.jsonl|summary\.json|candidates|\.log$|logs/"])
    states = list(dest.rglob("ppo_state.json"))
    return json.loads(states[0].read_text()) if states else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-kernel", default=f"{USER}/ptcg-tx-train-v1",
                    help="kernel whose output holds the distilled model.pt")
    ap.add_argument("--first", type=int, default=1, help="session number to start at")
    ap.add_argument("--until-cycle", type=int, default=300)
    ap.add_argument("--hours", type=float, default=11.0, help="per-session budget")
    ap.add_argument("--poll", type=int, default=300, help="seconds between status checks")
    ap.add_argument("--stop-on-candidate", action="store_true")
    a = ap.parse_args()
    k = a.first
    source = a.init_kernel if k == 1 else f"{USER}/ptcg-ppo-s{k - 1:03d}"
    while True:
        slug = launch(k, source, fresh=(k == 1), hours=a.hours)
        print(f"{time.strftime('%m-%d %H:%M')} session {k} launched ({slug})", flush=True)
        while (s := status(slug)) in ("RUNNING", "QUEUED") or s.startswith("UNKNOWN"):
            time.sleep(a.poll)
        state = pull(slug, k)
        cycle = state.get("cycle")
        print(f"{time.strftime('%m-%d %H:%M')} session {k} {s}: cycle {cycle}", flush=True)
        if s != "COMPLETE" or cycle is None:
            print("session failed; chain stopped", flush=True)
            return 1
        cands = list((ROOT / "out" / "kaggle_out" / f"ppo-s{k:03d}").rglob("candidates/*.npz"))
        if cands:
            print(f"CANDIDATE beats the distilled model: {[c.name for c in cands]}", flush=True)
            if a.stop_on_candidate:
                return 0
        if cycle >= a.until_cycle:
            print(f"reached cycle {cycle}; chain done", flush=True)
            return 0
        source = f"{USER}/{slug}"
        k += 1


if __name__ == "__main__":
    sys.exit(main())
