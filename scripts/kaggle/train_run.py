#!/usr/bin/env python
"""Run the v5_s2-on-every-dataset notebook ON KAGGLE and pull back only the net.

Sibling of `launch.py`, and deliberately NOT part of it: that one packs the repo
into a private dataset and runs arena shards with `enable_internet: False`. This
one pushes a self-contained notebook that needs the internet (it fetches the
episode datasets itself) and needs no payload dataset at all.

    python -X utf8 scripts/kaggle/train_run.py push --probe    # ~5 min, CPU
    python -X utf8 scripts/kaggle/train_run.py push --gpu      # the real run
    python -X utf8 scripts/kaggle/train_run.py status
    python -X utf8 scripts/kaggle/train_run.py pull

⚡ The point of this script: the replay data NEVER touches this machine. The
~1.3 TB of episode JSON is downloaded by Kaggle, from Kaggle, inside the kernel;
`pull` brings back the trained `.npz` (~2.6 MB) and the corpus zip.

⚠ `--probe` rewrites the notebook's CONFIG cell to SMOKE mode before pushing.
Run it first. It costs five minutes and it is the only way to find out, without
spending a GPU session, whether internet is enabled for this account and whether
kagglehub resolves the episode datasets from inside a batch kernel.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "out" / "kaggle_jobs"
USER = "siamrahman29"                     # same account launch.py pushes to
NOTEBOOK = ROOT / "notebooks" / "ptcg-v5s2-hostall-kaggle.ipynb"
TRAIN_NB = ROOT / "notebooks" / "ptcg-v5s2-hostall-train.ipynb"
BUILD_KERNELS = ["siamrahman29/ptcg-v5s2-hostall-h1",
                 "siamrahman29/ptcg-v5s2-hostall-h2"]
SLUG = "ptcg-v5s2-hostall"
EPISODE_DS = "kaggle/pokemon-tcg-ai-battle-episodes-{d}"
CG_DS = "kiyotah/cg-lib"
# The corpus, now a durable private Dataset rather than kernel output.
CORPUS_DS = "siamrahman29/ptcg-hostall-corpus"
RATINGS_DS = "siamrahman29/ptcg-episode-ratings"
FIRST_DAY, LAST_DAY = "2026-06-16", "2026-08-13"
# ⚠ Measured cap: 50 dataset_sources push, 55 get a 400 from SaveKernel. 59 days
# therefore needs TWO kernels; --half picks which.
MAX_SOURCES = 45


def episode_days(half: int = 0) -> list[str]:
    import datetime as dt
    a, b, out = dt.date.fromisoformat(FIRST_DAY), dt.date.fromisoformat(LAST_DAY), []
    while a <= b:
        out.append(a.isoformat())
        a += dt.timedelta(days=1)
    if half == 0:
        return out
    mid = (len(out) + 1) // 2
    return out[:mid] if half == 1 else out[mid:]


def kernel_id(probe: bool, scale: bool = False, half: int = 0,
              train: bool = False, tag: str = "") -> str:
    if train:
        return f"{USER}/{SLUG}-train" + (f"-{tag}" if tag else "")
    if scale:
        return f"{USER}/{SLUG}-scale"
    if half:
        return f"{USER}/{SLUG}-h{half}"      # ⚠ distinct, or half 2 overwrites half 1
    return f"{USER}/{SLUG}-probe" if probe else f"{USER}/{SLUG}"


def stage_dir(probe: bool, half: int = 0, train: bool = False) -> Path:
    if train:
        return STAGE / f"{SLUG}-train"
    if half:
        return STAGE / f"{SLUG}-h{half}"
    return STAGE / (f"{SLUG}-probe" if probe else SLUG)


def load_notebook(train: bool = False) -> dict:
    nb = TRAIN_NB if train else NOTEBOOK
    if not nb.exists():
        sys.exit(f"{nb} missing -- run scripts/make_kaggle_notebook.py"
                 + (" --mode train" if train else ""))
    return json.loads(nb.read_text(encoding="utf-8"))


def set_config(nb: dict, **overrides: str) -> int:
    """Rewrite assignments in the notebook's CONFIG cell.

    ⚠ Textual, and it verifies every key it was asked to set actually matched.
    A silently-unapplied override is how a "probe" becomes a full 59-day run.
    """
    hits = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        if not any(ln.startswith("SMOKE") for ln in src):
            continue
        for i, ln in enumerate(src):
            key = ln.split("=", 1)[0].strip()
            if key in overrides:
                tail = ln.split("#", 1)
                comment = f"   # {tail[1].strip()}" if len(tail) > 1 else ""
                src[i] = f"{key:16s} = {overrides[key]}{comment}\n"
                hits += 1
        break
    missed = set(overrides) - {"__none__"}
    if hits < len(missed):
        sys.exit(f"CONFIG rewrite matched {hits} of {len(missed)} keys "
                 f"({sorted(missed)}); the notebook's CONFIG cell changed shape")
    return hits


def push(args: argparse.Namespace) -> None:
    nb = load_notebook(args.train)
    if args.train:
        # ⚠ TPU VM used for its HOST RAM (405.7 GB) and 224 cores, NOT its TPU:
        # --device cpu. That is what lets the 40.1M-row corpus load with the
        # trainer unchanged, instead of needing a streaming loader that would
        # change the shuffling and break comparability with policy_v5_s2.
        # --ds /kaggle/input: train_policy rglobs shard_*.npz, so it finds all
        # 701 shards wherever Kaggle mounted the two build kernels' outputs.
        set_config(nb, SMOKE="False", CORPUS='"/kaggle/input"',
                   SKIP_TRAIN="False",
                   DEVICE='"cuda"' if args.gpu else '"cpu"',
                   STREAM="True" if args.stream else "False",
                   STREAM_BUFFER=str(args.stream_buffer),
                   TOP_PCT=str(args.top_pct),
                   EPOCHS=str(args.epochs), MAX_HOURS=str(args.max_hours),
                   # ⚠ rule 20: a path is not an identity. --tag already splits
                   # the kernel and the pull directory, so it must split the net
                   # too -- otherwise a top10 run exports over the name the
                   # 40.1M-row day-35 net already carries and the only record of
                   # which corpus produced which weights is the kernel log.
                   NET_NAME=f'"policy_v5_s2_{args.tag or "hostall"}.npz"')
    elif args.scale_test:
        # ⚠ ONE day, EVERY episode in it, no training. This is the measurement
        # the last run skipped: it converts "0.15 s/episode on my laptop" into
        # a real Kaggle number before anything multi-hour is committed.
        set_config(nb, SMOKE="False", EPISODES_PER_DAY="0",
                   SKIP_TRAIN="True", THROUGHPUT_TEST="True", DEVICE='"cpu"')
    elif args.half:
        # A BUILD half: every episode of ~30 days, no training. Stage 3 needs a
        # streaming loader that does not exist yet, so these kernels exist to
        # produce the corpus and nothing else.
        set_config(nb, SMOKE="False", EPISODES_PER_DAY="0",
                   SKIP_TRAIN="True", THROUGHPUT_TEST="False", DEVICE='"cpu"')
    elif args.probe:
        # Two days, a dozen episodes, one epoch: enough to prove the datasets
        # resolve, the engine loads and the trainer exports -- and nothing more.
        set_config(nb, SMOKE="True", DEVICE='"cpu"')
    else:
        # ⚠ DEVICE must follow the accelerator we actually asked Kaggle for.
        # Leaving it at "cuda" on a CPU kernel works -- the train cell falls
        # back -- but the notebook then reads `cuda` while running `cpu`, and
        # the log is the only record of which numerics produced the net.
        over = {"DEVICE": '"cuda"' if args.gpu else '"cpu"'}
        if args.episodes:
            over["EPISODES_PER_DAY"] = str(args.episodes)
        set_config(nb, **over)

    d = stage_dir(args.probe, args.half, args.train)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "notebook.ipynb").write_text(json.dumps(nb, indent=1,
                                                 ensure_ascii=False) + "\n",
                                      encoding="utf-8")
    days = ([args.day] if args.day else episode_days(args.half)) or ["-"]
    # The cg engine rides along as a mount: the featurizer needs its card and
    # attack tables, and a runtime attach is impossible in a batch kernel.
    # A training run needs only the engine; the corpus arrives via kernel_sources.
    sources = [CG_DS, CORPUS_DS, RATINGS_DS] if args.train else (
        [CG_DS] + [EPISODE_DS.format(d=d) for d in days])
    if len(sources) > MAX_SOURCES:
        sys.exit(f"{len(sources)} dataset_sources exceeds the measured cap "
                 f"({MAX_SOURCES}); pass --half 1 or --half 2")
    meta = {
        "id": kernel_id(args.probe, args.scale_test, args.half, args.train, getattr(args, "tag", "")),
        "title": kernel_id(args.probe, args.scale_test, args.half, args.train, getattr(args, "tag", "")).split("/")[1],
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        # ⚠ Unlike every job launch.py pushes, this one MUST have the internet:
        # the kernel is what downloads the episode datasets.
        "enable_internet": True,
        "enable_gpu": bool(args.gpu and not args.probe),
        "enable_tpu": False,
        "dataset_sources": sources,
        "competition_sources": [],
        # ⚠ The corpus arrives as a DATASET now, not kernel output: kernel
        # output is destroyed by a re-push of the kernel that made it.
        "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2),
                                            encoding="utf-8")

    print(f"=== {meta['id']} ===")
    print(f"  probe={args.probe}  scale_test={args.scale_test}  "
          f"gpu={meta['enable_gpu']}")
    print(f"  {len(sources)} episode dataset(s) attached: "
          f"{days[0]}{'' if len(days) == 1 else ' .. ' + days[-1]}")
    print(f"  staged at {d}")
    if args.dry_run:
        print("\n(dry run -- nothing pushed)")
        return
    r = subprocess.run(["kaggle", "kernels", "push", "-p", str(d)],
                       capture_output=True, text=True)
    print(r.stdout or r.stderr)
    if r.returncode:
        sys.exit(r.returncode)
    print(f"\nwatch it:  python -X utf8 scripts/kaggle/train_run.py status"
          f"{' --probe' if args.probe else ''}")
    print(f"           https://www.kaggle.com/code/{meta['id'].split('/')[1]}")


def status(args: argparse.Namespace) -> None:
    r = subprocess.run(["kaggle", "kernels", "status", kernel_id(args.probe, getattr(args, "scale_test", False), getattr(args, "half", 0), getattr(args, "train", False))],
                       capture_output=True, text=True)
    print(r.stdout or r.stderr)


def logs(args: argparse.Namespace) -> None:
    """The kernel's stdout. This is where a 'missing dataset' or an OOM shows."""
    dest = ROOT / "out" / "kaggle_out" / (kernel_id(args.probe, getattr(args, "scale_test", False), getattr(args, "half", 0), getattr(args, "train", False)).split("/")[1])
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["kaggle", "kernels", "output", kernel_id(args.probe, getattr(args, "scale_test", False), getattr(args, "half", 0), getattr(args, "train", False)),
                        "-p", str(dest)], capture_output=True, text=True)
    print(r.stdout or r.stderr)
    for f in sorted(dest.glob("*.log")) + sorted(dest.glob("*.txt")):
        print(f"\n--- {f.name} ---")
        print("\n".join(f.read_text(encoding="utf-8",
                                    errors="replace").splitlines()[-40:]))


def pull(args: argparse.Namespace) -> None:
    dest = ROOT / "out" / "kaggle_out" / (kernel_id(args.probe, getattr(args, "scale_test", False), getattr(args, "half", 0), getattr(args, "train", False)).split("/")[1])
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["kaggle", "kernels", "output", kernel_id(args.probe, getattr(args, "scale_test", False), getattr(args, "half", 0), getattr(args, "train", False)),
                        "-p", str(dest)], capture_output=True, text=True)
    print(r.stdout or r.stderr)
    got = sorted(p for p in dest.rglob("*") if p.is_file())
    for f in got:
        print(f"  {f.stat().st_size / 1e6:9.2f} MB  {f.relative_to(dest)}")
    nets = [f for f in got if f.suffix == ".npz"]
    if not nets:
        # 🔴 rule 18's shape: a pull that returns 0 nets and exits 0 reads as
        # "the run is done" and is actually "the run produced nothing".
        print("\n🔴 NO .npz PULLED. The kernel did not export a net -- read the "
              "log above before re-pushing; do not assume it is still running.")
        sys.exit(1)
    print(f"\ncopy the net into out/ and A/B it against out/policy_v5_s2.npz:")
    for n in nets:
        print(f"  Copy-Item '{n}' out\\{n.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("push")
    p.add_argument("--probe", action="store_true",
                   help="push the SMOKE config as a separate CPU kernel first")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--episodes", type=int, default=0,
                   help="override EPISODES_PER_DAY (default: the notebook's)")
    p.add_argument("--scale-test", action="store_true",
                   help="one day, EVERY episode, no training -- the real "
                        "throughput measurement")
    p.add_argument("--day", default="",
                   help="attach exactly this one day (for --scale-test)")
    p.add_argument("--half", type=int, default=0, choices=(0, 1, 2),
                   help="1 or 2: which half of the 59 days to attach, since "
                        "only ~50 sources fit on one kernel")
    p.add_argument("--train", action="store_true",
                   help="train off the corpus the build kernels produced, on a "
                        "TPU VM (for its 405 GB host RAM), --device cpu")
    p.add_argument("--tag", default="",
                   help="suffix for the kernel id, so runs do not clobber")
    p.add_argument("--top-pct", type=int, default=0,
                   help="keep only the top N%% of episodes by avg_score")
    p.add_argument("--stream", action="store_true",
                   help="stream shards instead of loading the whole corpus")
    p.add_argument("--stream-buffer", type=int, default=8)
    p.add_argument("--max-hours", type=float, default=10.0,
                   help="stop cleanly at this many hours so the export is "
                        "committed; MUST be under Kaggle's 12 h cap")
    p.add_argument("--epochs", type=int, default=6,
                   help="epochs this session; chain more with --init later")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=push)

    for name, fn in (("status", status), ("pull", pull), ("logs", logs)):
        q = sub.add_parser(name)
        q.add_argument("--probe", action="store_true")
        q.add_argument("--scale-test", action="store_true")
        q.add_argument("--half", type=int, default=0, choices=(0, 1, 2))
        q.add_argument("--train", action="store_true")
        q.add_argument("--tag", default="")
        q.set_defaults(fn=fn)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
