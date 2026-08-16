"""Generate the self-contained Kaggle notebook that retrains the v5_s2 recipe
on EVERY daily episode dataset the hosts have released.

    python scripts/make_kaggle_notebook.py
    python scripts/make_kaggle_notebook.py --out notebooks/foo.ipynb

⚠ The notebook is GENERATED, never hand-edited. Every module and script it
carries is embedded VERBATIM from this repo at generation time (`%%writefile`),
so the corpus builder and the trainer that run on Kaggle are byte-identical to
the ones that produced `out/policy_v5_s2.npz`. Editing the .ipynb by hand
breaks that guarantee silently -- edit the source file and regenerate.

The recipe it runs is HANDOFF's day-25 command, unchanged:

    scripts/train_policy.py --ds <corpus> --epochs 12 --bs 1024 \
        --loss listwise --state-h 512,256 --head-h 256,128 --pool \
        --opt-cols 37 --seed 2

Only `--ds` differs from v5_s2: 4 days of top-400 replays (`artifacts/pds_v4`)
becomes every released day, still top-N by `avg_score`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Everything the corpus builder and the trainer import, in dependency order.
# `policynet.py` is not needed to train -- it is carried so the notebook can
# load its own export the way the arena loads it, before you download it.
#
# ⚠ This list must be the FULL import closure including LAZY imports. Leaving
# `targeting` out cost a whole smoke run: `optfeat.option_features` imports it
# from inside the function, so the builder caught one ModuleNotFoundError per
# episode, kept going, and wrote a corpus with 2% of its rows. It exited 0 and
# printed `games=12` while writing 28 rows. The `errors` guard in the fetch cell
# exists because of this, and it is why that guard is not optional.
EMBED = [
    "src/ptcg/__init__.py",
    "src/ptcg/config.py",
    "src/ptcg/env/__init__.py",
    "src/ptcg/env/sdk.py",
    "agents/sa/__init__.py",
    "agents/sa/cards.py",
    "agents/sa/textdmg.py",
    "agents/sa/targeting.py",
    "agents/sa/features.py",
    "agents/sa/optfeat.py",
    "agents/sa/routing.py",
    "agents/sa/policynet.py",
    "scripts/build_policy_dataset.py",
    "scripts/train_policy.py",
]

# The shipped net, for an exact architecture check on the notebook's own export.
REFERENCE_NET = "out/policy_v5_s2.npz"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


def writefile_cell(rel: str) -> dict:
    body = (ROOT / rel).read_text(encoding="utf-8")
    if not body.endswith("\n"):
        body += "\n"
    return code(f"%%writefile {rel}\n{body}")


HEADER = """\
# PTCG — the `v5_s2` recipe on **every** host-released episode dataset

This notebook rebuilds the shipped agent's policy net from scratch on Kaggle,
using the same behavioural-cloning recipe that produced `out/policy_v5_s2.npz`,
but over **every daily episode dataset the competition hosts have released**
(`kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD`, 2026-06-16 onwards) instead
of the four days `v5_s2` was trained on.

**Nothing about the agent changes except the corpus.** The trainer and the corpus
builder below are embedded verbatim from the repo, and the training command is
the day-25 recipe unchanged:

```
scripts/train_policy.py --ds <corpus> --epochs 12 --bs 1024 --loss listwise \\
    --state-h 512,256 --head-h 256,128 --pool --opt-cols 37 --seed 2
```

`--opt-cols 37` is load-bearing: the builder writes 46-wide option features and
the v6 attribute block is *appended*, so 37 slices exactly the v5 layout. A net
trained without it cannot share an ensemble with `policy_v5`.

---

## What it does, in order

1. Rebuilds the repo tree under `/kaggle/working/ptcg` from `%%writefile` cells.
2. Pulls the `cg` engine (`kiyotah/cg-lib`) — the featurizer needs its card and
   attack tables.
3. Probes every daily dataset from `START_DATE` to today, reads each
   `manifest.csv`, and takes the top `EPISODES_PER_DAY` episodes by `avg_score`.
4. Per day: download the replays → build shards → **delete the replays**, so
   peak disk stays ~2 GB instead of ~1.3 TB.
5. Trains, verifies the export loads through `sa/policynet.py`, and leaves the
   net + corpus in `/kaggle/working` to download.

## ⚠ "Every dataset" means every DAY, top-N per day. Read this before raising N.

Every day's dataset holds ~4,600 episodes and **~21.5 GB of JSON**. All 59 days
is **~1.3 TB and ~42 M training rows**. That is not a Kaggle-sized job and no
setting of the knobs below makes it one — the ceiling is RAM, not the hosts:

`train_policy.Data` concatenates the whole corpus into memory, and it holds the
per-shard arrays *and* the concatenated copy alive at the same time. Measured on
`artifacts/pds_all`: **~4.0 KB/row resident, ~7.8 KB/row at the load peak.**
42 M rows would need ~330 GB. Kaggle gives you ~29 GB.

| `EPISODES_PER_DAY` | episodes | rows | load peak | build | train (12 ep) |
|---|---|---|---|---|---|
| 150 | ~8,850 | ~1.3 M | ~10 GB | ~25 min | ~1.5 h CPU / ~35 min GPU |
| **300** (default) | ~17,700 | ~2.7 M | **~21 GB** | ~45 min | ~3 h CPU / ~1.2 h GPU |
| 400 | ~23,600 | ~3.5 M | ~28 GB | ~1 h | **OOM risk on a 29 GB box** |
| 600 | ~35,400 | ~5.3 M | ~41 GB | ~1.5 h | will not load |

For reference, `v5_s2` itself was trained on **4 days × 400 episodes ≈ 250 K
rows**. The default here is ~11× that corpus, drawn from 15× as many days.

Episodes are taken **top-N by `avg_score`**, which is the same selection rule
`scripts/fetch_top_episodes.py --max 400` uses for the repo's own dumps — so
this is the v5_s2 corpus's own rule applied to every released day, not a new one.

**Set `SMOKE = True` for a ~5 minute end-to-end rehearsal before committing a
multi-hour session.**

## Requirements

* **Internet ON** (Settings → Internet). kagglehub cannot reach the datasets
  without it, and the failure looks like every date being "missing".
* **GPU T4 x2** recommended (`DEVICE = "cuda"`). `DEVICE = "cpu"` reproduces
  `v5_s2`'s numerics exactly but takes ~3 h for the default corpus.
* Nothing needs to be attached as a data source, and **no API key or secret is
  needed** — on Kaggle, kagglehub authenticates from the token file the kernel
  already has (`KAGGLE_API_V1_TOKEN_PATH`).

⚠ The scaffold cell sets `DISABLE_KAGGLE_CACHE=1`. Do not remove it — see the
comment there; without it this notebook silently trains on an empty corpus.
"""

CONFIG = '''\
# ─────────────────────────────────────────────────────────────── CONFIG ──────
SMOKE            = False   # True = 2 days x 12 episodes x 1 epoch, ~5 min rehearsal

ROOT             = "/kaggle/working/ptcg"
START_DATE       = "2026-06-16"   # the first daily dataset the hosts published
END_DATE         = None           # None = today (UTC). Dates with no dataset are skipped.
EPISODES_PER_DAY = 0              # top-N of each day's manifest by avg_score. 0 = ALL.
                                  # ⚠ read the RAM table in the header before raising this
                                  # for a run that TRAINS; a build-only run is unconstrained.
BUILD_PROCS      = 4              # parallel builder processes (Kaggle gives 4 vCPU)
TOP_PCT          = 0              # 0 = every episode. 10 = keep only the top 10%
                                  # of episodes by avg_score. `gid` IS the
                                  # episode id, so this is a row mask over the
                                  # SAME corpus -- no rebuild.
KEEP_GIDS        = ""             # set by the cut cell below
STREAM           = False          # load shards a buffer at a time. REQUIRED above
                                  # ~4M rows: the in-RAM path wants 7.8 KB/row
                                  # (313 GB at 40.1M) against a 34 GB box.
                                  # ⚠ shuffling becomes buffer-local -- declare it.
STREAM_BUFFER    = 8              # shards resident per buffer under STREAM
MAX_HOURS        = 0.0            # >0: stop after the first epoch past this many
                                  # hours and exit 0. Kaggle DISCARDS output when
                                  # it kills a kernel at the 12 h cap, so a run
                                  # that might overrun must stop itself.
SKIP_TRAIN       = False          # True = build the corpus and stop (for scale tests
                                  # and for the build half of a split pipeline)
THROUGHPUT_TEST  = True           # measure mount read scaling before building

# ── the v5_s2 recipe, verbatim (HANDOFF, "THE DAY-25 PLAN") ──────────────────
EPOCHS    = 12
BS        = 1024
LOSS      = "listwise"
STATE_H   = "512,256"
HEAD_H    = "256,128"
OPT_COLS  = 37        # ⚠ load-bearing: slices the v5 option layout out of the 46-wide corpus
SEED      = 2
POOL      = True
# ─────────────────────────────────────────────────────────────────────────────

DEVICE       = "cuda"   # "cpu" == v5_s2's exact numerics; "cuda" is ~3x faster
CORPUS       = "artifacts/pds_hostall"
NET_NAME     = "policy_v5_s2_hostall.npz"
RATINGS_ZIP  = ""       # optional Kaggle LB export (.zip/.csv) -> enables val_top1@1120+.
                        # Attach it as a dataset and point at the file; leave "" to skip.

if SMOKE:
    EPISODES_PER_DAY, EPOCHS = 12, 1
    CORPUS, NET_NAME = "artifacts/pds_smoke", "policy_smoke.npz"

print(f"corpus  {CORPUS}   net  {NET_NAME}   episodes/day {EPISODES_PER_DAY}")
'''

SCAFFOLD = '''\
import os, sys, shutil, subprocess

# ⚠ Episodes come from ATTACHED datasets (/kaggle/input), never downloaded.
# Per-file kagglehub downloads are rate-limited to ~60 requests per ~20 min
# (429 on DownloadDataset, and it fails serial too -- it is a request budget,
# not a concurrency problem). A 17,700-file run measured 98.9% failures and
# still exited 0. Declare the days in kernel-metadata `dataset_sources`
# instead: Kaggle mounts them before the kernel starts, at zero disk cost and
# zero API calls. The cap is ~50 sources per kernel, so 59 days needs two.
# kagglehub is still used for the cg engine only -- one small request.

for sub in ("src/ptcg/env", "agents/sa", "scripts", "data", "replays",
            "artifacts", "out"):
    os.makedirs(f"{ROOT}/{sub}", exist_ok=True)
os.chdir(ROOT)                      # every %%writefile below is relative to here
# CORPUS is a repo-relative dir when this run BUILDS one, and an absolute
# /kaggle/input path when it trains off a corpus another kernel built.
CORPUS_DIR = CORPUS if os.path.isabs(CORPUS) else f"{ROOT}/{CORPUS}"
print("cwd:", os.getcwd())

# The repo puts `src` and `agents` on sys.path; the scripts do it themselves,
# but the verification cells at the bottom import from this kernel.
for sub in ("src", "agents", ""):
    p = os.path.join(ROOT, sub) if sub else ROOT
    if p not in sys.path:
        sys.path.insert(0, p)
'''

SDK = '''\
# ── the `cg` engine ──────────────────────────────────────────────────────────
# `sa/cards.py` reads the card and attack tables out of the engine, so the
# FEATURIZER needs it even though this notebook never plays a game.
# `kiyotah/cg-lib` is the package the host sample notebooks ship (same api.py,
# sim.py and libcg.so as the competition's sample_submission).
import glob, logging, kagglehub, kagglehub.clients as _khc

logging.getLogger("kagglehub").setLevel(logging.ERROR)   # 23,600 log lines otherwise


class _QuietBar:                # kagglehub draws a tqdm bar per FILE; that is
    def __init__(self, *a, **k): pass          # one per episode. Kill it or the
    def __enter__(self): return self           # notebook output eats the browser.
    def __exit__(self, *a): return False
    def update(self, *a, **k): pass


_khc.tqdm = _QuietBar

# ⚠ The engine is MOUNTED too, not downloaded. A runtime kagglehub attach dies
# with "New Datasets cannot be attached in non-interactive sessions" in a batch
# kernel, and the HTTP fallback spends the same 429 budget the episodes need.
# `kiyotah/cg-lib` is declared in dataset_sources alongside the episode days.
cg_dir = None
for _c in glob.glob("/kaggle/input/**/cg/api.py", recursive=True):
    cg_dir = os.path.dirname(os.path.dirname(_c))
    break
if cg_dir is None:
    # Last resort: one HTTP request, which the budget can afford.
    os.environ["DISABLE_KAGGLE_CACHE"] = "1"
    print("cg-lib not mounted; falling back to a single HTTP fetch")
    cg_dir = kagglehub.dataset_download("kiyotah/cg-lib")
# `ptcg.config.find_sdk_dir()` globs data/**/cg/api.py, so drop it where the
# repo already looks rather than special-casing sys.path.
dst = f"{ROOT}/data/cg_sdk"
if not os.path.exists(f"{dst}/cg/api.py"):
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(os.path.join(cg_dir, "cg"), f"{dst}/cg", dirs_exist_ok=True)
print("cg engine ->", dst)
'''

SDK_CHECK = '''\
# ── the featurizer must agree with the engine before anything else runs ──────
from ptcg.env import sdk
sdk.load()
from sa import cards as cdb
from sa.features import DENSE_DIM, N_ATTR, N_EXTRA
from sa.optfeat import OPT_DENSE

print(f"cards={len(cdb.cards())}  attacks={len(cdb.attacks())}")
print(f"DENSE_DIM={DENSE_DIM}  N_EXTRA={N_EXTRA}  N_ATTR={N_ATTR}  OPT_DENSE={OPT_DENSE}")
assert len(cdb.cards()) > 1000, "engine loaded but the card table is empty"
assert OPT_DENSE >= OPT_COLS, f"corpus writes {OPT_DENSE} option cols, recipe slices {OPT_COLS}"
'''

PLAN = """\
# ── which days are ATTACHED, and which episodes to take from each ────────────
# ⚠ Episodes are read from /kaggle/input mounts. Nothing is downloaded, so the
# 429 request budget is never touched. Declare the days in kernel-metadata
# `dataset_sources` (max ~50 per kernel; 59 days needs two kernels).
import csv, glob, re, time

IN = "/kaggle/input"
DAY_RE = re.compile(r"episodes-(\\d{4}-\\d{2}-\\d{2})")


def find_day_dirs():
    \"\"\"{date: mount dir}. Kaggle mounts some sources at /kaggle/input/<slug>
    and others under /kaggle/input/datasets/<owner>/<slug>, so SEARCH rather
    than assume -- launch.py's runner carries the same warning.\"\"\"
    out = {}
    for manifest in glob.glob(f"{IN}/**/manifest.csv", recursive=True):
        d = os.path.dirname(manifest)
        m = DAY_RE.search(d)
        if m:
            out.setdefault(m.group(1), d)
    return dict(sorted(out.items()))


days_dir = find_day_dirs()
print(f"{len(days_dir)} day mounts found under {IN}")
if not days_dir:
    raise SystemExit(
        "no episode datasets are attached. This notebook does NOT download "
        "them -- add them to kernel-metadata `dataset_sources` (or the "
        "notebook's Data panel) and re-run.")

plan, n_avail = {}, 0
for d, path in days_dir.items():
    with open(os.path.join(path, "manifest.csv"), encoding="utf-8-sig",
              newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: -float(r["avg_score"]))
    n_avail += len(rows)
    take = rows if EPISODES_PER_DAY <= 0 else rows[:EPISODES_PER_DAY]
    # ⚠ Only count episodes that are ACTUALLY on the mount. A manifest row
    # whose .json is missing would otherwise inflate every number below and
    # turn a short corpus into a silent one.
    ids = [r["episode_id"] for r in take
           if os.path.exists(os.path.join(path, r["episode_id"] + ".json"))]
    plan[d] = (path, ids)
    miss = len(take) - len(ids)
    print(f"  {d}  {len(rows):5d} in manifest -> take {len(ids):5d}"
          + (f"   ⚠ {miss} missing on mount" if miss else ""))

n_take = sum(len(v[1]) for v in plan.values())
print(f"\\n{len(plan)} days, {n_avail:,} episodes available, taking {n_take:,}")
est_rows = n_take * 150
print(f"estimate: ~{est_rows/1e6:.1f} M rows, ~{est_rows*7.8e-6:.0f} GB at the "
      f"corpus-load peak if this run also TRAINS")
"""

THROUGHPUT = """\
# ── does mount read bandwidth scale with processes? ──────────────────────────
# Single-core measured 45 MB/s / 10 episodes/s off a mount. If that is a
# PER-PROCESS rate, BUILD_PROCS workers multiply it; if it is the mount's
# ceiling, they do not, and the build must be split across more kernels.
# Measure it, do not assume it -- assuming is what cost the last run.
import json as _json
from concurrent.futures import ProcessPoolExecutor


def _read_bytes(f):
    with open(f, "rb") as fh:
        b = fh.read()
    _json.loads(b)
    return len(b)


if THROUGHPUT_TEST:
    d0, (p0, ids0) = next(iter(plan.items()))
    files = [os.path.join(p0, e + ".json") for e in ids0[:200]]
    for nproc in (1, 2, 4):
        sel = files[(nproc - 1) * 48:(nproc - 1) * 48 + 48]   # disjoint: no page cache reuse
        t = time.time()
        if nproc == 1:
            tot = sum(_read_bytes(f) for f in sel)
        else:
            with ProcessPoolExecutor(max_workers=nproc) as ex:
                tot = sum(ex.map(_read_bytes, sel, chunksize=2))
        dt_ = time.time() - t
        print(f"  {nproc} proc: {tot/1e6:6.0f} MB in {dt_:5.1f}s -> "
              f"{tot/1e6/dt_:6.0f} MB/s, {len(sel)/dt_:5.1f} episodes/s",
              flush=True)
"""

FETCH = r'''
# ── featurize, BUILD_PROCS-way parallel, straight off the mounts ─────────────
# ⚠ The builder is used UNMODIFIED. Parallelism comes from giving N copies of it
# N disjoint directories of SYMLINKS into the mount -- no episode is copied, no
# extra disk is used, and build_policy_dataset.py stays byte-identical to the
# one that produced v5_s2's corpus.
import shutil, subprocess
from concurrent.futures import ThreadPoolExecutor

RATINGS_ARGS = ["--ratings", RATINGS_ZIP] if RATINGS_ZIP else []


def build_day(date, path, ids):
    out_root = f"{CORPUS}/d{date.replace('-', '')}"
    link_root = f"{ROOT}/links/{date}"
    shutil.rmtree(f"{ROOT}/{out_root}", ignore_errors=True)
    shutil.rmtree(link_root, ignore_errors=True)
    nproc = max(1, min(BUILD_PROCS, len(ids)))
    for k in range(nproc):
        os.makedirs(f"{link_root}/p{k}", exist_ok=True)
    # manifest.csv is deliberately NOT linked: the builder skips non-digit
    # stems anyway, and linking it per shard dir only invites a re-read.
    for n, e in enumerate(ids):
        try:
            os.symlink(os.path.join(path, e + ".json"),
                       f"{link_root}/p{n % nproc}/{e}.json")
        except FileExistsError:
            pass

    def one(k):
        r = subprocess.run(
            [sys.executable, "-X", "utf8", "scripts/build_policy_dataset.py",
             "--out", f"{out_root}/p{k}", f"links/{date}/p{k}"] + RATINGS_ARGS,
            capture_output=True, text=True, cwd=ROOT)
        return k, r

    with ThreadPoolExecutor(max_workers=nproc) as ex:   # threads launch the
        results = list(ex.map(one, range(nproc)))        # real subprocesses
    shutil.rmtree(link_root, ignore_errors=True)

    games = rows = errs = 0
    for k, r in results:
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise RuntimeError(f"builder p{k} failed on {date}")
        line = next((l for l in r.stdout.splitlines()
                     if l.startswith("games=")), "")
        kv = dict(x.split("=", 1) for x in line.split() if "=" in x)
        games += int(kv.get("games", 0))
        rows += int(kv.get("rows", 0))
        errs += int(kv.get("errors", 0))
        if int(kv.get("errors", 0)):
            print(f"    p{k} stderr: " + " | ".join(r.stderr.splitlines()[:3]))
    return games, rows, errs


# 🔴 THE GUARD THAT WAS MISSING LAST TIME.
# The previous run lost 17,512 of 17,700 episodes and still exited 0, because
# the only check was on BUILDER errors and a day with zero input files reports
# `games=0 rows=0 errors=0`. Count what was PLANNED against what was BUILT, and
# stop on the day it diverges -- not at the end, and not as a warning.
MIN_YIELD = 0.98

t_all = time.time()
done_days = tot_games = tot_rows = 0
for i, (date, (path, ids)) in enumerate(plan.items()):
    t0 = time.time()
    games, rows, errs = build_day(date, path, ids)
    tot_games += games
    tot_rows += rows
    done_days += 1
    el = time.time() - t_all
    eta = el / done_days * (len(plan) - done_days) / 60
    print(f"[{i+1:2d}/{len(plan)}] {date}  {games:5d}/{len(ids):5d} eps  "
          f"{rows:8,d} rows  errors={errs}  {time.time()-t0:5.0f}s  "
          f"| elapsed {el/60:5.1f} min, ~{eta:.0f} min left", flush=True)
    if games < MIN_YIELD * len(ids):
        raise RuntimeError(
            f"{date}: featurized {games} of {len(ids)} planned episodes "
            f"({games/max(len(ids),1):.1%}). Stopping HERE rather than "
            f"building a corpus that is quietly short. Read the builder "
            f"stderr above.")

shutil.rmtree(f"{ROOT}/links", ignore_errors=True)
print(f"\ncorpus: {tot_games:,} episodes, {tot_rows:,} rows in "
      f"{(time.time()-t_all)/60:.1f} min")
'''

CENSUS = '''\
# ── what actually got built, and will it fit in RAM ──────────────────────────
import glob
import numpy as np

shards = sorted(glob.glob(f"{CORPUS_DIR}/**/shard_*.npz", recursive=True))
rows = opts = 0
games = set()
for p in shards:
    z = np.load(p)
    rows += len(z["gid"])
    opts += len(z["opt_chosen"])
    games.update(np.unique(z["gid"]).tolist())

disk = sum(os.path.getsize(p) for p in shards)
print(f"{len(shards)} shards under {CORPUS_DIR}")
print(f"{rows:,} decisions from {len(games):,} games, {opts:,} options "
      f"({opts / max(rows, 1):.1f} per decision)")
print(f"{disk / 1e6:.1f} MB on disk")

# ⚠ The kill happens while the corpus LOADS, not while it trains.
# `train_policy.Data` appends every shard's arrays to per-key lists, then
# concatenates -- and the lists stay referenced until __init__ returns, so both
# copies are resident at once. Measured on artifacts/pds_all: 4.0 KB/row
# resident, ~7.8 KB/row at that peak.
steady, peak = rows * 4.0e-6, rows * 7.8e-6
try:
    import psutil
    have = psutil.virtual_memory().total / 1e9
except Exception:
    have = float("nan")
print(f"\\ncorpus load peaks at ~{peak:.1f} GB (settling to ~{steady:.1f} GB); "
      f"this machine has {have:.0f} GB")
if peak > 0.80 * have:
    print("⚠ TOO TIGHT. Lower EPISODES_PER_DAY, delete the shard dirs for the "
          "days you want thinned, and re-run the fetch cell -- the trainer "
          "would otherwise be OOM-killed partway through loading, after you "
          "have already paid for the downloads.")
assert rows > 0, "empty corpus"
'''

CUT = '''\n# ── the rating cut: which episodes survive ───────────────────────────────────
# ⚡ The 0.440 result (EVIDENCE §1a) says UNFILTERED volume is negative and the
# suspect is composition: v5_s2 cloned the top ~400/day (cutoff ~1150) while the
# full corpus reaches down to avg_score ~700-900. This applies a quality bar to
# the SAME shards, so composition moves and nothing else does.
import csv as _csv, glob as _glob

if TOP_PCT and TOP_PCT > 0:
    cand = _glob.glob("/kaggle/input/**/episode_ratings.csv", recursive=True)
    if not cand:
        raise SystemExit("TOP_PCT set but episode_ratings.csv is not attached; "
                         "add siamrahman29/ptcg-episode-ratings")
    with open(cand[0], encoding="utf-8-sig", newline="") as fh:
        rows = [(int(r["episode_id"]), float(r["avg_score"]))
                for r in _csv.DictReader(fh)]
    rows.sort(key=lambda r: -r[1])
    k = max(1, int(len(rows) * TOP_PCT / 100.0))
    KEEP_GIDS = f"{ROOT}/keep_gids.txt"
    with open(KEEP_GIDS, "w", encoding="utf-8") as fh:
        # chr(10), not a backslash escape: this cell is embedded in the
        # generator as a string literal, where a newline escape is one more
        # level of quoting to get wrong.
        fh.write(chr(10).join(str(e) for e, _ in rows[:k]))
    print(f"TOP_PCT {TOP_PCT}%: {k:,} of {len(rows):,} episodes, "
          f"avg_score cutoff {rows[k-1][1]:.0f} -> {KEEP_GIDS}")
    print(f"  (for reference v5_s2's own corpus cut at ~1150)")
else:
    print("TOP_PCT 0: every episode in the corpus is kept")
'''


TRAIN = '''\
# ── train: the v5_s2 command, unchanged except --ds and --out ────────────────
import torch

device = DEVICE
if device == "cuda":
    # ⚠ `torch.cuda.is_available()` is NOT sufficient on Kaggle. A session that
    # gets a **Tesla P100** reports True and then fails on the first kernel
    # launch: the image ships torch 2.10+cu128, which supports sm_70..sm_120,
    # and the P100 is sm_60. Measured, not assumed -- a GPU probe kernel came
    # back "Tesla P100 ... is not compatible with the current PyTorch install".
    # Crashing here costs a corpus build; crashing an hour in costs the session.
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    if cap is None:
        print("⚠ no CUDA in this session; using cpu (which is what v5_s2 used)")
        device = "cpu"
    elif cap[0] < 7:
        name = torch.cuda.get_device_name(0)
        print(f"⚠ {name} is compute {cap[0]}.{cap[1]}; this image's torch needs "
              f">= 7.0. Using cpu. Pick the T4 x2 accelerator, not P100, if you "
              f"want the GPU.")
        device = "cpu"
    else:
        print(f"cuda: {torch.cuda.get_device_name(0)} (compute {cap[0]}.{cap[1]})")

cmd = [sys.executable, "-X", "utf8", "scripts/train_policy.py",
       "--ds", CORPUS,
       "--epochs", str(EPOCHS),
       "--bs", str(BS),
       "--loss", LOSS,
       "--state-h", STATE_H,
       "--head-h", HEAD_H,
       "--opt-cols", str(OPT_COLS),
       "--seed", str(SEED),
       "--max-hours", str(MAX_HOURS),
       *(["--stream", "--stream-buffer", str(STREAM_BUFFER)] if STREAM else []),
       *(["--keep-gids", KEEP_GIDS] if KEEP_GIDS else []),
       "--device", device,
       "--out", f"out/{NET_NAME}"]
if POOL:
    cmd.append("--pool")

if SKIP_TRAIN:
    print("SKIP_TRAIN: corpus is built; stopping before training.")
    raise SystemExit

print(" ".join(cmd), "\\n" + "-" * 78, flush=True)

log = []
with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT, text=True, bufsize=1) as pr:
    for line in pr.stdout:
        print(line, end="", flush=True)
        log.append(line)
    rc = pr.wait()

open(f"{ROOT}/out/{NET_NAME}.train.log", "w", encoding="utf-8").writelines(log)
assert rc == 0, f"trainer exited {rc}"
'''

VERIFY = '''\
# ── verify the export the way the ARENA loads it ─────────────────────────────
# ⚠ HANDOFF rule 20: a path is not an identity. This cell checks the export
# against the SHAPES of the shipped `out/policy_v5_s2.npz`, key by key -- that
# is what "exactly the v5_s2 agent, different corpus" has to mean at the
# artefact level -- and prints the fingerprint to carry into the arena.
import hashlib
import importlib
import numpy as np

import sa.policynet as pn
importlib.reload(pn)

# taken from out/policy_v5_s2.npz at notebook-generation time
V5_S2_SHAPES = {shapes}

path = f"{{ROOT}}/out/{{NET_NAME}}"
net = pn.load(path)
assert net is not None, "policynet.load() refused the export"

z = np.load(path)
got = {{k: list(z[k].shape) for k in z.files}}
params = sum(int(z[k].size) for k in z.files if z[k].dtype == np.float32)
digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]

print(f"{{NET_NAME}}  #{{digest}}  {{os.path.getsize(path) / 1e6:.2f}} MB")
print(f"state MLP input {{got['sfc0_w'][1]}}   head input {{got['head0_w'][1]}}   "
      f"n_pool={{int(z['n_pool'][0])}}   n_attr={{int(z['n_attr'][0])}}")
print(f"{{params:,}} float params in {{len(z.files)}} arrays")

missing = sorted(set(V5_S2_SHAPES) - set(got))
extra = sorted(set(got) - set(V5_S2_SHAPES))
diff = sorted(k for k in set(got) & set(V5_S2_SHAPES) if got[k] != V5_S2_SHAPES[k])
if missing or extra or diff:
    for k in missing:
        print(f"  MISSING {{k}} {{V5_S2_SHAPES[k]}}")
    for k in extra:
        print(f"  EXTRA   {{k}} {{got[k]}}")
    for k in diff:
        print(f"  SHAPE   {{k}}: {{got[k]}} != v5_s2's {{V5_S2_SHAPES[k]}}")
    raise SystemExit(
        "this net is NOT architecturally v5_s2. It cannot be A/B'd against "
        "out/policy_v5_s2.npz as a corpus change, and it cannot share an "
        "ensemble with policy_v5. Check --pool / --opt-cols / --state-h / "
        "--head-h in the CONFIG cell.")

print("\\n✅ byte-for-byte the same architecture as out/policy_v5_s2.npz "
      "(every array name and shape matches). The ONLY thing that differs "
      "between this net and the shipped one is the corpus it was fitted to.")
'''

COLLECT = '''\
# ── leave everything downloadable in /kaggle/working ─────────────────────────
OUT = "/kaggle/working"

shutil.copy(f"{ROOT}/out/{NET_NAME}", f"{OUT}/{NET_NAME}")
shutil.copy(f"{ROOT}/out/{NET_NAME}.train.log", f"{OUT}/{NET_NAME}.train.log")

# The corpus is small and expensive to rebuild -- ship it so a re-train (another
# seed, an ablation) does not spend another hour of downloads.
corpus_zip = shutil.make_archive(f"{OUT}/{os.path.basename(CORPUS)}", "zip",
                                 root_dir=f"{ROOT}/{CORPUS}")

# The loose shards are in the zip now; the engine stays so the cells above can
# be re-run (another seed, an ablation) without re-downloading it.
shutil.rmtree(f"{ROOT}/artifacts", ignore_errors=True)

for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    if os.path.isfile(p):
        print(f"{os.path.getsize(p) / 1e6:9.2f} MB  {f}")
'''

FOOTER = """\
---

## Back in the repo

Download `{net}` and `pds_hostall.zip` from the output pane, then:

```powershell
Copy-Item <downloads>\\{net} out\\{net}
Expand-Archive <downloads>\\pds_hostall.zip artifacts\\pds_hostall

# the A/B that matters: same rules, same deck, byte-identical config, net swapped
python -X utf8 scripts/arena.py play `
    "bc:hostall,net=out/{net},noChip,noSpread,noSrc" `
    "bc:v5s2ship,net=out/policy_v5_s2.npz,noChip,noSpread,noSrc" `
    --matches 2000 --deck-a grimmsnarl --deck-b grimmsnarl `
    --archive out/arena/hostall_vs_v5s2.jsonl
```

⚠ **Read the result against the noise floor, not against 0.500.** Two
identical-recipe nets differing only in `--seed` measure **0.482 [0.460, 0.504]**
against each other (EVIDENCE §8z), so anything inside roughly ±0.025 is a null
and *"more data helped"* is not a claim this A/B can support at n=2,000.

⚠ **`--device cuda` is not `v5_s2`'s numerics.** If the arena reads a difference
you want to attribute to the corpus, re-run the training with `DEVICE = "cpu"`
before believing it — otherwise the corpus change and the device change are
confounded.
"""


def reference_shapes() -> str:
    """`{name: shape}` of the shipped net, so the notebook can assert that what
    it trained is architecturally the same artefact."""
    import numpy as np

    p = ROOT / REFERENCE_NET
    if not p.exists():
        raise SystemExit(f"{REFERENCE_NET} missing -- the notebook's identity "
                         "check is generated from the shipped net itself")
    with np.load(p) as z:
        shapes = {k: list(z[k].shape) for k in sorted(z.files)}
    body = ",\n    ".join(f"{k!r}: {v}" for k, v in shapes.items())
    return "{\n    " + body + ",\n}"


def build(out_path: Path, mode: str = "full") -> int:
    """mode="full": build the corpus then train. mode="train": train off a
    corpus another kernel already built and that this one MOUNTS -- no
    downloads, no featurizing, and no 3 h rebuild every time a hyperparameter
    moves."""
    cells = [md(HEADER), code(CONFIG), code(SCAFFOLD)]

    cells.append(md("## The repo, verbatim\n\nEvery cell below is a byte-identical "
                    "copy of a file in the repo, embedded at generation time by\n"
                    "`scripts/make_kaggle_notebook.py`. **Do not edit them here** "
                    "— edit the source file and regenerate,\nor the net this "
                    "notebook trains stops being comparable to `policy_v5_s2.npz`.\n"))
    for rel in EMBED:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"missing source: {rel}")
        cells.append(writefile_cell(rel))

    cells.append(md("## The engine\n"))
    cells.append(code(SDK))
    cells.append(code(SDK_CHECK))
    cells.append(md("## The corpus\n"))
    if mode == "full":
        cells.append(code(PLAN))
        cells.append(code(THROUGHPUT))
        cells.append(code(FETCH))
    cells.append(code(CENSUS))
    cells.append(md("## Training\n"))
    cells.append(code(CUT))
    cells.append(code(TRAIN))
    cells.append(code(VERIFY.format(shapes=reference_shapes())))
    cells.append(code(COLLECT))
    cells.append(md(FOOTER.format(net="policy_v5_s2_hostall.npz")))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.13"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                       "isInternetEnabled": True, "isGpuEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    n_src = sum(len((ROOT / r).read_text(encoding="utf-8").splitlines())
                for r in EMBED)
    print(f"wrote {out_path} -- {len(cells)} cells, {len(EMBED)} embedded files "
          f"({n_src:,} lines of repo source)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notebooks/ptcg-v5s2-hostall-kaggle.ipynb")
    ap.add_argument("--mode", choices=("full", "train"), default="full")
    args = ap.parse_args()
    out = args.out
    if args.mode == "train" and out == "notebooks/ptcg-v5s2-hostall-kaggle.ipynb":
        out = "notebooks/ptcg-v5s2-hostall-train.ipynb"
    return build(ROOT / out, args.mode)


if __name__ == "__main__":
    sys.exit(main())
