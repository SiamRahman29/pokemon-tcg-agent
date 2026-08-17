"""Rewrite a policy corpus keeping only the rows of selected episodes.

WHY THIS EXISTS (EVIDENCE §1a follow-up). `train_policy.py --keep-gids` filters
rows *after* the loader has already materialised them, so under `--stream` every
epoch still opens and fully decompresses every shard that holds at least one
surviving row. A global top-10% cut is spread across all 59 days, so that is
essentially all 701 shards: the gradient steps drop to 10% but the I/O does not
move, and "10% of the data" does not buy 10% of the time.

It also quietly shrinks the shuffle. `StreamData` buffers `--stream-buffer`
whole shards and trains on whatever survives the mask inside them, so a 10% cut
turns a 458k-row shuffle pool into a ~46k-row one at the same `--stream-buffer`
-- a confound §1a's measured 0.0012 bound does NOT cover, because that was
measured with no mask in play.

This script does the cut ONCE, off the critical path, and repacks the survivors
into full-size shards. The training run then reads a corpus that is 10% of the
bytes and buffers the same number of trainable rows per buffer as the unfiltered
run did, so `--stream-buffer` keeps its old meaning.

    python -X utf8 scripts/filter_corpus.py \
        --ds artifacts/pds_hostall --keep-gids out/keep_gids.txt \
        --out artifacts/pds_top10

⚠ `gid` IS the episode id (EVIDENCE §1a), which is what makes this a row mask
over the existing shards rather than a rebuild.
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Kept in sync with train_policy.BAGS. Imported rather than re-declared would be
# better, but this script has to run inside a Kaggle kernel where only the
# embedded files exist, and it must not drag the trainer's torch import in.
BAGS = ("my_hand", "my_discard", "opp_discard")

# Ragged blocks: offset key -> value keys it indexes. Values are addressed by
# the offsets, not by row, so both sides have to be rebuilt under a mask.
RAGGED: dict[str, list[str]] = {
    "opt_off": ["opt_dense", "opt_card", "opt_attack", "opt_target",
                "opt_chosen"],
}
RAGGED.update({f"bag_{nm}_off": [f"bag_{nm}_flat"] for nm in BAGS})

# Every key the ragged rebuild owns. Anything else in the shard is per-row and
# is carried through generically -- ⚠ do NOT reintroduce a hardcoded row-key
# list here. The first version of this script had one and silently dropped
# `opp_rating`, `sub_id` and `team_id`, which is the join from a row back to
# its demonstrator's leaderboard rating.
_RAGGED_KEYS = set(RAGGED) | {v for vs in RAGGED.values() for v in vs}


def slice_shard(z, keep: np.ndarray) -> dict[str, np.ndarray]:
    """One shard's arrays, restricted to the rows `keep` selects."""
    n = len(keep)
    out: dict[str, np.ndarray] = {}
    for off_key, val_keys in RAGGED.items():
        if off_key not in z:
            continue
        off = z[off_key].astype(np.int64)
        lens = np.diff(off)[keep]
        total = int(lens.sum())
        # Global positions of every element belonging to a kept row, in row
        # order. `cumsum(lens) - lens` is the exclusive prefix sum, so this maps
        # each new element back to its original slot across the gaps the mask
        # opens up.
        starts = off[:-1][keep]
        idx = (np.repeat(starts - (np.cumsum(lens) - lens), lens)
               + np.arange(total, dtype=np.int64))
        new_off = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=new_off[1:])
        out[off_key] = new_off
        for vk in val_keys:
            if vk in z:
                if len(z[vk]) != off[-1]:
                    raise SystemExit(
                        f"{vk} has {len(z[vk]):,} elements but {off_key} ends "
                        f"at {off[-1]:,}; the shard's ragged block is "
                        f"inconsistent and slicing it would corrupt it")
                out[vk] = z[vk][idx]
    for k in z.files:
        if k in _RAGGED_KEYS:
            continue
        a = z[k]
        if a.ndim >= 1 and a.shape[0] == n:
            out[k] = a[keep]
        else:
            # rule 9: never write a corpus with a column we did not understand.
            raise SystemExit(
                f"shard key {k!r} has shape {a.shape}, which is neither ragged "
                f"nor per-row against {n:,} rows. Teach filter_corpus.py what "
                f"it is rather than dropping or mis-slicing it.")
    return out


def merge(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate sliced shards into one, rebuilding every offset array."""
    if len(parts) == 1:
        return parts[0]
    out: dict[str, np.ndarray] = {}
    for off_key, val_keys in RAGGED.items():
        if off_key not in parts[0]:
            continue
        lens = np.concatenate([np.diff(p[off_key]) for p in parts])
        off = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=off[1:])
        out[off_key] = off
        for vk in val_keys:
            present = [p[vk] for p in parts if vk in p]
            if present:
                out[vk] = np.concatenate(present)
    for k in parts[0]:
        if k not in _RAGGED_KEYS:
            out[k] = np.concatenate([p[k] for p in parts])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="corpus dir to filter")
    ap.add_argument("--keep-gids", required=True,
                    help="one episode id per line (leading column of a CSV ok)")
    ap.add_argument("--out", required=True, help="corpus dir to write")
    ap.add_argument("--rows-per-shard", type=int, default=57000,
                    help="target rows per output shard. The default matches the "
                         "hostall corpus's own 57,243, so --stream-buffer keeps "
                         "the meaning it had in the unfiltered run.")
    args = ap.parse_args()

    src = Path(args.ds)
    if not src.is_absolute():
        src = ROOT / src
    shards = sorted(glob.glob(f"{src}/**/shard_*.npz", recursive=True))
    if not shards:
        raise SystemExit(f"no shards under {src}")

    kp = Path(args.keep_gids)
    if not kp.is_absolute():
        kp = ROOT / kp
    want: set[int] = set()
    for ln in kp.read_text(encoding="utf-8-sig").splitlines():
        tok = ln.split(",", 1)[0].strip()
        if tok and tok.lstrip("-").isdigit():
            want.add(int(tok))
    if not want:
        raise SystemExit(f"{kp}: parsed zero episode ids")
    want_arr = np.fromiter(want, dtype=np.int64, count=len(want))

    dst = Path(args.out)
    if not dst.is_absolute():
        dst = ROOT / dst
    dst.mkdir(parents=True, exist_ok=True)

    print(f"{len(shards)} shards under {src}", flush=True)
    print(f"--keep-gids {kp.name}: {len(want):,} episode ids", flush=True)

    buf: list[dict[str, np.ndarray]] = []
    buf_rows = 0
    n_out = 0
    tot_in = tot_kept = 0
    eps: set[int] = set()
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, buf_rows, n_out
        if not buf:
            return
        merged = merge(buf)
        path = dst / f"shard_{n_out:04d}.npz"
        np.savez_compressed(path, **merged)
        n_out += 1
        buf, buf_rows = [], 0

    for i, p in enumerate(shards):
        with np.load(p) as z:
            gid = z["gid"]
            tot_in += len(gid)
            keep = np.isin(gid, want_arr)
            k = int(keep.sum())
            if not k:
                continue
            tot_kept += k
            eps.update(np.unique(gid[keep]).tolist())
            buf.append(slice_shard(z, keep))
            buf_rows += k
        if buf_rows >= args.rows_per_shard:
            flush()
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(shards)} shards, {tot_kept:,} rows kept "
                  f"({time.time() - t0:.0f}s)", flush=True)
    flush()

    if not tot_kept:
        raise SystemExit("kept ZERO rows; the id space is wrong (gid is the "
                         "episode id)")
    # rule 9: a filter that matches almost nothing must not look like a small
    # corpus. Report what survived in rows AND in episodes, against the ask.
    print(f"\n{tot_kept:,} of {tot_in:,} rows kept ({tot_kept / tot_in:.1%}), "
          f"{len(eps):,} of {len(want):,} requested episodes matched")
    print(f"wrote {n_out} shards to {dst} in {time.time() - t0:.0f}s")
    if len(eps) < len(want):
        print(f"⚠ {len(want) - len(eps):,} requested episodes are not in this "
              f"corpus at all")


if __name__ == "__main__":
    main()
