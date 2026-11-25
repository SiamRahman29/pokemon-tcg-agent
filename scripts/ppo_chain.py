"""Run the PPO chain on Kaggle CPU notebooks, from this machine (polling only).

Two modes:

  sessions (`--parallel 0`): session k is kernel `ptcg-ppo-s{k:03d}`, one
      notebook that collects and learns in a loop for ~11 h, then the next
      session resumes from its output.

  parallel (`--parallel K`): every cycle runs as K collector notebooks
      (`ptcg-ppo-col{j}`, each plays 1/K of the cycle's games from the current
      state) and then one learner notebook that attaches the state and all K
      rollouts and runs the update. Learners alternate between
      `ptcg-ppo-learn-a` and `-b`, so each attaches the other's output. The
      PPO algorithm is the same as in a session; only the games are spread
      over more cores.

After each session or learner this pulls only the small files (state,
history, evaluation summaries, candidates, logs) to `out/kaggle_out/` and
prints the cycle reached. Stops at `--until-cycle`, on a notebook that fails
twice, or (with `--stop-on-candidate`) when a checkpoint beats the distilled
model. Submitting is left to a person.

    python -X utf8 scripts/ppo_chain.py --parallel 4 \\
        --source siamrahman29/ptcg-ppo-s001 --until-cycle 300
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
SMALL = r"ppo_state\.json|history\.jsonl|summary\.json|candidates|\.log$|logs/"


def log(msg: str) -> None:
    print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def status(slug: str) -> str:
    out = sh(["kaggle", "kernels", "status", f"{USER}/{slug}"])
    for word in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if word in out:
            return word
    return "UNKNOWN: " + out.strip()[-200:]


def push(job: str, inputs: list[str], cmd: str) -> str:
    """Push a new version of kernel `ptcg-{job}`; retry transient API failures."""
    args = [sys.executable, "-X", "utf8", str(ROOT / "scripts/kaggle/launch.py"), "push",
            "--job", job, "--shards", "1", "--out-flag=--output", "--code-ds", CODE_DS,
            "--cmd", cmd, "--collect", "out/ppo0,out/logs"]
    for src in inputs:
        args += ["--kernel-input", src]
    for attempt in range(5):
        out = sh(args)
        if "successfully pushed" in out:
            return f"ptcg-{job}"
        log(f"push {job} failed (attempt {attempt + 1}): {out.strip()[-300:]}")
        time.sleep(60 * (attempt + 1))
    raise SystemExit(f"could not push {job}")


def wait(slugs: list[str], poll: int, seen_timeout: int = 900) -> dict[str, str]:
    """Wait until every kernel finishes. A just-pushed kernel can still report
    its PREVIOUS version's COMPLETE for a short while, so a final status only
    counts after the kernel has been seen QUEUED or RUNNING (or after
    `seen_timeout` seconds)."""
    t0 = time.time()
    seen = {s: False for s in slugs}
    final: dict[str, str] = {}
    while len(final) < len(slugs):
        for s in slugs:
            if s in final:
                continue
            st = status(s)
            if st in ("QUEUED", "RUNNING"):
                seen[s] = True
            elif st in ("COMPLETE", "ERROR", "CANCEL") and (
                    seen[s] or time.time() - t0 > seen_timeout):
                final[s] = st
        if len(final) < len(slugs):
            time.sleep(poll)
    return final


def pull(slug: str, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    sh(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", str(dest),
        "--file-pattern", SMALL])
    states = list(dest.rglob("ppo_state.json"))
    return json.loads(states[0].read_text()) if states else {}


def last_history(dest: Path) -> dict:
    hist = list(dest.rglob("history.jsonl"))
    lines = hist[0].read_text().splitlines() if hist else []
    return json.loads(lines[-1]) if lines else {}


def run_parallel(a) -> int:
    source = a.source
    k = a.parallel
    learner = 0
    if source.split("/")[-1].startswith("ptcg-ppo-learn-"):
        learner = 1 if source.endswith("-a") else 0
    st = wait([source.split("/")[-1]], a.poll, seen_timeout=0)[source.split("/")[-1]]
    if st != "COMPLETE":
        log(f"source {source} ended {st}; chain not started")
        return 1
    state = pull(source.split("/")[-1], ROOT / "out/kaggle_out/ppo-par/source")
    log(f"source {source} at cycle {state.get('cycle')}")
    while True:
        c0 = time.time()
        cycle = state.get("cycle")
        collect = (f"scripts/ppo_session.py --role collect --shards {k} --resume /kaggle/input "
                   f"--output out/ppo{{i}} --games {a.games} {a.extra}").strip()
        slugs = [push(f"ppo-col{j}", [source], collect.replace("--shards", f"--shard {j} --shards"))
                 for j in range(k)]
        log(f"cycle {cycle}: {k} collectors pushed")
        done = wait(slugs, a.poll)
        for retry in range(2):
            failed = [s for s, v in done.items() if v != "COMPLETE"]
            if not failed:
                break
            log(f"collectors failed: {failed}; retry {retry + 1}")
            for s in failed:
                j = int(s.rsplit("col", 1)[1])
                push(f"ppo-col{j}", [source],
                     collect.replace("--shards", f"--shard {j} --shards"))
            done.update(wait(failed, a.poll))
        if any(v != "COMPLETE" for v in done.values()):
            log(f"collectors failed twice: {done}; chain stopped")
            return 1
        t_col = time.time() - c0
        name = f"ppo-learn-{'ab'[learner]}"
        learn = (f"scripts/ppo_session.py --role learn --resume /kaggle/input "
                 f"--output out/ppo{{i}} --games {a.games} {a.extra}").strip()
        inputs = [source] + [f"{USER}/{s}" for s in slugs]
        slug = push(name, inputs, learn)
        st = wait([slug], a.poll)[slug]
        if st != "COMPLETE":
            log(f"{slug} ended {st}; retrying once")
            slug = push(name, inputs, learn)
            st = wait([slug], a.poll)[slug]
        dest = ROOT / "out/kaggle_out/ppo-par" / f"c{(cycle or 0) + 1:04d}"
        new = pull(slug, dest) if st == "COMPLETE" else {}
        if st != "COMPLETE" or new.get("cycle") is None:
            log(f"{slug} ended {st}; chain stopped")
            return 1
        h = last_history(dest)
        log(f"cycle {new['cycle']} done in {(time.time() - c0) / 60:.1f} min "
            f"(collect {t_col / 60:.1f}, learn {(time.time() - c0 - t_col) / 60:.1f}); "
            f"kl {h.get('heldout_global_approximate_kl', float('nan')):.4f} "
            f"lr {new.get('learning_rate')}")
        if "eval" in h:
            log("eval: " + json.dumps({o: v["score"] for o, v in h["eval"].items()}))
        # The state (with candidates/) carries over, so only this cycle's line counts.
        if h.get("candidate"):
            log(f"CANDIDATE beats the distilled model at cycle {new['cycle']}")
            if a.stop_on_candidate:
                return 0
        if new["cycle"] >= a.until_cycle:
            log(f"reached cycle {new['cycle']}; chain done")
            return 0
        state, source, learner = new, f"{USER}/{slug}", 1 - learner


def run_sessions(a) -> int:
    k = a.first
    source = a.init_kernel if k == 1 else f"{USER}/ptcg-ppo-s{k - 1:03d}"
    while True:
        flag = "--init" if k == 1 else "--resume"
        slug = push(f"ppo-s{k:03d}", [source],
                    f"scripts/ppo_session.py {flag} /kaggle/input --output out/ppo{{i}} "
                    f"--max-hours {a.hours} --games {a.games}")
        log(f"session {k} launched ({slug})")
        s = wait([slug], a.poll)[slug]
        dest = ROOT / "out" / "kaggle_out" / f"ppo-s{k:03d}"
        state = pull(slug, dest)
        cycle = state.get("cycle")
        log(f"session {k} {s}: cycle {cycle}")
        if s != "COMPLETE" or cycle is None:
            log("session failed; chain stopped")
            return 1
        if list(dest.rglob("candidates/*.npz")):
            log(f"CANDIDATE beats the distilled model at cycle {cycle}")
            if a.stop_on_candidate:
                return 0
        if cycle >= a.until_cycle:
            log(f"reached cycle {cycle}; chain done")
            return 0
        source = f"{USER}/{slug}"
        k += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", type=int, default=0,
                    help="collector notebooks per cycle (0 = one-notebook sessions)")
    ap.add_argument("--source", default="",
                    help="parallel: kernel whose output holds the state to continue from")
    ap.add_argument("--init-kernel", default=f"{USER}/ptcg-tx-train-v1",
                    help="sessions: kernel whose output holds the distilled model.pt")
    ap.add_argument("--first", type=int, default=1, help="sessions: session number to start at")
    ap.add_argument("--until-cycle", type=int, default=300)
    ap.add_argument("--games", type=int, default=5000, help="games per cycle")
    ap.add_argument("--hours", type=float, default=11.0, help="sessions: per-session budget")
    ap.add_argument("--poll", type=int, default=30, help="seconds between status checks")
    ap.add_argument("--stop-on-candidate", action="store_true")
    ap.add_argument("--extra", default="",
                    help="parallel: extra ppo_session.py flags for every notebook (smoke tests)")
    a = ap.parse_args()
    if a.parallel:
        if not a.source:
            ap.error("--parallel needs --source")
        return run_parallel(a)
    return run_sessions(a)


if __name__ == "__main__":
    sys.exit(main())
