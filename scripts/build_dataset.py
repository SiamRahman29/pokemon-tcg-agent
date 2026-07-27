"""Build value-net training shards from replay JSONs.

    python scripts/build_dataset.py --out artifacts/ds --stride 2 <replay_dir>...

Each replay's `visualize` entries hold the acting player's observation at every
select. We featurize every `stride`-th state and label it with the final result
from the acting player's perspective (1 win / 0 loss / 0.5 draw).

Shards: artifacts/ds/shard_XXX.npz with
    dense (N, D) f32 | slots (N, 12) i32 | y (N,) f32
    bag_<name>_flat (i32) + bag_<name>_off (N+1, i32) for hand/discards
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa.features import featurize, DENSE_DIM  # noqa: E402

SHARD_ROWS = 150_000


class ShardWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0
        self.reset()

    def reset(self):
        self.dense: list[np.ndarray] = []
        self.slots: list[np.ndarray] = []
        self.y: list[float] = []
        self.gid: list[int] = []
        self.bags: dict[str, list[np.ndarray]] = {
            "my_hand": [], "my_discard": [], "opp_discard": []}

    def add(self, dense, bags, label, game_id: int):
        self.dense.append(dense)
        self.slots.append(bags["slots"])
        for k in self.bags:
            self.bags[k].append(bags[k])
        self.y.append(label)
        self.gid.append(game_id)
        if len(self.y) >= SHARD_ROWS:
            self.flush()

    def flush(self):
        if not self.y:
            return
        arrs = {
            "dense": np.stack(self.dense),
            "slots": np.stack(self.slots),
            "y": np.asarray(self.y, dtype=np.float32),
            "gid": np.asarray(self.gid, dtype=np.int64),
        }
        for k, lists in self.bags.items():
            off = np.zeros(len(lists) + 1, dtype=np.int64)
            for i, a in enumerate(lists):
                off[i + 1] = off[i] + len(a)
            flat = (np.concatenate(lists) if off[-1] else
                    np.zeros(0, dtype=np.int32))
            arrs[f"bag_{k}_flat"] = flat
            arrs[f"bag_{k}_off"] = off
        path = self.out_dir / f"shard_{self.idx:03d}.npz"
        np.savez_compressed(path, **arrs)
        print(f"  wrote {path.name}: {len(self.y)} rows")
        self.idx += 1
        self.reset()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="artifacts/ds")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()

    writer = ShardWriter(ROOT / args.out)
    n_games = n_states = n_err = 0
    for d in args.dirs:
        for path in sorted(Path(d).glob("*.json")):
            if path.name == "manifest.json":
                continue
            try:
                rep = json.loads(path.read_text(encoding="utf-8"))
                rewards = rep["rewards"]
                if rewards[0] is None or rewards[1] is None:
                    continue
                vis = rep["steps"][0][0].get("visualize") or []
                n_games += 1
                for vi, v in enumerate(vis):
                    if vi % args.stride:
                        continue
                    obs = v.get("obs")
                    if not obs or not obs.get("current"):
                        continue
                    state = obs["current"]
                    if state["result"] != -1:
                        continue
                    me = state["yourIndex"]
                    r_me, r_opp = rewards[me], rewards[1 - me]
                    label = 1.0 if r_me > r_opp else 0.0 if r_me < r_opp \
                        else 0.5
                    dense, bags = featurize(state, me)
                    try:
                        gid = int(path.stem)
                    except ValueError:
                        gid = hash(path.stem) & 0x7FFFFFFF
                    writer.add(dense, bags, label, gid)
                    n_states += 1
            except Exception as exc:
                n_err += 1
                if n_err <= 5:
                    print(f"  {path.name}: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
    writer.flush()
    print(f"games={n_games} states={n_states} errors={n_err} "
          f"dense_dim={DENSE_DIM}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
