"""E3: export reviewed labels as treatment/control policy shards.

The treatment and control contain identical states, option features, game ids,
and row order.  Only `opt_chosen` differs:

* treatment: the reviewer's high-confidence action;
* control: the frozen v5 clone's original action.

One fifth of labels is held out by stable item id and written separately.

    python -X utf8 scripts/p45_dagger_export.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts", ""):
    p = str(ROOT / sub) if sub else str(ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)

from ptcg.env import sdk  # noqa: E402

sdk.load()

from sa.features import extra_feats, featurize  # noqa: E402
from sa.optfeat import OPT_DENSE, option_features  # noqa: E402
from build_policy_dataset import Writer, sel_features  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return str(shown).replace("\\", "/")


def load_queue(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def is_holdout(item_id: str, mod: int) -> bool:
    return int(hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:8], 16) % mod == 0


def unique_gid(item_id: str, used: set[int]) -> int:
    gid = int(hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:12], 16)
    # train_policy's historical validation split is gid % 20 == 0. E3 owns a
    # separate 20% holdout, so every exported training row must avoid that path.
    while gid in used or gid % 20 == 0:
        gid += 1
    used.add(gid)
    return gid


def add_row(writer: Writer, item: dict[str, Any], action: list[int],
            gid: int) -> None:
    obs = item["observation"]
    state = obs["current"]
    sel = obs["select"]
    options = sel.get("option") or []
    me = int(state["yourIndex"])
    dense, bags = featurize(state, me)
    od = np.zeros((len(options), OPT_DENSE), dtype=np.float32)
    oc = np.zeros(len(options), dtype=np.int32)
    oa = np.zeros(len(options), dtype=np.int32)
    ot = np.zeros(len(options), dtype=np.int32)
    for i, option in enumerate(options):
        od[i], oc[i], oa[i], ot[i] = option_features(obs, option)
    mask = np.zeros(len(options), dtype=np.float32)
    mask[action] = 1.0
    writer.add(
        dense, bags, sel_features(sel), (od, oc, oa, ot), mask, gid, 0.5,
        extra=extra_feats(state, sel, me))


def prepare_dir(path: Path, force: bool) -> None:
    shards = list(path.glob("shard_*.npz")) if path.exists() else []
    if shards and not force:
        raise SystemExit(f"{path} already contains shards; pass --force to replace")
    if shards:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default="out/e3/review_queue.jsonl")
    ap.add_argument("--reviews", default="out/e3/reviews.json")
    ap.add_argument("--out", default="artifacts/e3_dagger")
    ap.add_argument("--min-labels", type=int, default=100)
    ap.add_argument("--holdout-mod", type=int, default=5,
                    help="one in N stable item ids is held out (default 5)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.holdout_mod < 2:
        raise SystemExit("--holdout-mod must be at least 2")

    queue_path = ROOT / args.queue
    reviews_path = ROOT / args.reviews
    if not queue_path.exists() or not reviews_path.exists():
        raise SystemExit("queue/reviews missing; finish p44 review first")
    items = load_queue(queue_path)
    by_id = {item["id"]: item for item in items}
    review_doc = json.loads(reviews_path.read_text(encoding="utf-8"))
    if review_doc.get("queue_sha256") != digest(queue_path):
        raise SystemExit("reviews belong to a different queue")
    reviews = review_doc.get("reviews") or {}
    high = {
        item_id: value for item_id, value in reviews.items()
        if value.get("status") == "labeled"
        and value.get("confidence") == "high"
    }
    unknown = set(high) - set(by_id)
    if unknown:
        raise SystemExit(f"reviews contain unknown ids: {sorted(unknown)[:3]}")
    if len(high) < args.min_labels:
        raise SystemExit(
            f"only {len(high)} high-confidence labels; need {args.min_labels}")

    train_ids = sorted(
        item_id for item_id in high
        if not is_holdout(item_id, args.holdout_mod))
    holdout_ids = sorted(set(high) - set(train_ids))
    if not train_ids or not holdout_ids:
        raise SystemExit("stable split produced an empty train or holdout set")

    base = ROOT / args.out
    treatment_dir = base / "treatment"
    control_dir = base / "control"
    prepare_dir(treatment_dir, args.force)
    prepare_dir(control_dir, args.force)
    treatment = Writer(treatment_dir)
    control = Writer(control_dir)
    used_gids: set[int] = set()
    changed = 0
    for item_id in train_ids:
        item = by_id[item_id]
        human_action = sorted(set(int(x) for x in high[item_id]["action"]))
        clone_action = sorted(set(int(x) for x in item["clone_action"]))
        if not item["min_count"] <= len(human_action) <= item["max_count"]:
            raise SystemExit(f"{item_id}: human action violates select bounds")
        if any(x < 0 or x >= item["n_options"] for x in human_action):
            raise SystemExit(f"{item_id}: human action index out of range")
        gid = unique_gid(item_id, used_gids)
        add_row(treatment, item, human_action, gid)
        add_row(control, item, clone_action, gid)
        changed += set(human_action) != set(clone_action)
    treatment.flush()
    control.flush()

    holdout_path = base / "holdout.jsonl"
    base.mkdir(parents=True, exist_ok=True)
    with holdout_path.open("w", encoding="utf-8") as f:
        for item_id in holdout_ids:
            item = by_id[item_id]
            row = {
                "id": item_id,
                "human_action": sorted(set(high[item_id]["action"])),
                "clone_action": item["clone_action"],
                "observation": item["observation"],
                "note": high[item_id].get("note") or "",
            }
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    manifest = {
        "experiment": "E3",
        "queue": args.queue,
        "queue_sha256": digest(queue_path),
        "reviews": args.reviews,
        "high_confidence_labels": len(high),
        "train_labels": len(train_ids),
        "holdout_labels": len(holdout_ids),
        "train_corrections": changed,
        "train_correction_rate": changed / len(train_ids),
        "holdout_rule": f"sha1(item_id) % {args.holdout_mod} == 0",
        "treatment": display_path(treatment_dir),
        "control": display_path(control_dir),
        "only_difference": "opt_chosen: human action vs frozen clone action",
    }
    (base / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"E3_EXPORT_OK {len(train_ids)} train / {len(holdout_ids)} holdout")
    print(f"  corrections={changed}/{len(train_ids)} "
          f"({changed / len(train_ids):.1%})")
    print(f"  treatment={treatment_dir}")
    print(f"  control={control_dir}")
    print(f"  holdout={holdout_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
