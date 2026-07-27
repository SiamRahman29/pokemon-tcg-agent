"""Build policy-cloning shards from replay JSONs.

    python scripts/build_policy_dataset.py --out artifacts/pds/d26 replays/2026-07-26

One row per select with >=2 options: state features + per-option features +
multi-hot chosen mask (from the replay's actual action).
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

from sa.features import featurize  # noqa: E402
from sa.optfeat import option_features, OPT_DENSE  # noqa: E402

SHARD_ROWS = 60_000
SEL_DENSE = 14


def sel_features(sel: dict) -> np.ndarray:
    v = np.zeros(SEL_DENSE, dtype=np.float32)
    t = sel.get("type") or 0
    if t < 11:
        v[t] = 1.0
    v[11] = sel.get("minCount", 0) / 5.0
    v[12] = sel.get("maxCount", 0) / 5.0
    v[13] = (sel.get("context") or 0) / 50.0
    return v


class Writer:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0
        self.reset()

    def reset(self):
        self.sd, self.slots, self.seld, self.gid = [], [], [], []
        self.bags = {"my_hand": [], "my_discard": [], "opp_discard": []}
        self.od, self.ocard, self.oatk, self.otgt, self.chosen = [], [], [], [], []
        self.off = [0]
        self.won = []

    def add(self, dense, bags, seld, opts, chosen_mask, gid, won):
        self.sd.append(dense)
        self.slots.append(bags["slots"])
        for k in self.bags:
            self.bags[k].append(bags[k])
        self.seld.append(seld)
        od, oc, oa, ot = opts
        self.od.append(od)
        self.ocard.append(oc)
        self.oatk.append(oa)
        self.otgt.append(ot)
        self.chosen.append(chosen_mask)
        self.off.append(self.off[-1] + len(oc))
        self.gid.append(gid)
        self.won.append(won)
        if len(self.sd) >= SHARD_ROWS:
            self.flush()

    def flush(self):
        if not self.sd:
            return
        arrs = {
            "dense": np.stack(self.sd),
            "slots": np.stack(self.slots),
            "seld": np.stack(self.seld),
            "gid": np.asarray(self.gid, dtype=np.int64),
            "won": np.asarray(self.won, dtype=np.float32),
            "opt_dense": np.concatenate(self.od),
            "opt_card": np.concatenate(self.ocard),
            "opt_attack": np.concatenate(self.oatk),
            "opt_target": np.concatenate(self.otgt),
            "opt_chosen": np.concatenate(self.chosen),
            "opt_off": np.asarray(self.off, dtype=np.int64),
        }
        for k, lists in self.bags.items():
            off = np.zeros(len(lists) + 1, dtype=np.int64)
            for i, a in enumerate(lists):
                off[i + 1] = off[i] + len(a)
            arrs[f"bag_{k}_flat"] = (np.concatenate(lists) if off[-1]
                                     else np.zeros(0, dtype=np.int32))
            arrs[f"bag_{k}_off"] = off
        path = self.out_dir / f"shard_{self.idx:03d}.npz"
        np.savez_compressed(path, **arrs)
        print(f"  wrote {path.name}: {len(self.sd)} rows")
        self.idx += 1
        self.reset()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    writer = Writer(ROOT / args.out)
    n_games = n_rows = n_err = 0
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
                try:
                    gid = int(path.stem)
                except ValueError:
                    gid = hash(path.stem) & 0x7FFFFFFF
                n_games += 1
                for v in vis:
                    obs = v.get("obs")
                    if not obs or not obs.get("current") or not obs.get("select"):
                        continue
                    state = obs["current"]
                    if state["result"] != -1:
                        continue
                    sel = obs["select"]
                    opts = sel.get("option") or []
                    if len(opts) < 2:
                        continue
                    action = v.get("selected")
                    if action is None:
                        action = v.get("action")
                    if not isinstance(action, list):
                        continue
                    picked = [a for a in action
                              if isinstance(a, int) and 0 <= a < len(opts)]
                    if len(picked) != len(action):
                        continue
                    me = state["yourIndex"]
                    won = 1.0 if rewards[me] > rewards[1 - me] else 0.0
                    dense, bags = featurize(state, me)
                    od = np.zeros((len(opts), OPT_DENSE), dtype=np.float32)
                    oc = np.zeros(len(opts), dtype=np.int32)
                    oa = np.zeros(len(opts), dtype=np.int32)
                    ot = np.zeros(len(opts), dtype=np.int32)
                    for i, o in enumerate(opts):
                        od[i], oc[i], oa[i], ot[i] = option_features(obs, o)
                    mask = np.zeros(len(opts), dtype=np.float32)
                    mask[picked] = 1.0
                    writer.add(dense, bags, sel_features(sel),
                               (od, oc, oa, ot), mask, gid, won)
                    n_rows += 1
            except Exception as exc:
                n_err += 1
                if n_err <= 5:
                    print(f"  {path.name}: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
    writer.flush()
    print(f"games={n_games} rows={n_rows} errors={n_err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
