"""Deterministic E3 queue/export contract smoke test."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    queue_path = ROOT / "out/e3/review_queue.jsonl"
    if not queue_path.exists():
        raise SystemExit("run p43_dagger_queue.py first")
    all_items = [json.loads(line) for line in queue_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    items = all_items[:40]
    assert len(items) == 40
    assert len({x["id"] for x in items}) == len(items)
    assert all("observation" in x and "winner" not in x for x in items)
    assert all(x["boundary_margin"] >= 0 for x in items)

    with tempfile.TemporaryDirectory(prefix="e3_smoke_") as td:
        tmp = Path(td)
        queue = tmp / "queue.jsonl"
        with queue.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        reviews = {}
        changed = False
        for item in items:
            action = list(item["clone_action"])
            if not changed and item["min_count"] == item["max_count"] == 1:
                alternatives = [i for i in range(item["n_options"])
                                if i not in action]
                if alternatives:
                    action = [alternatives[0]]
                    changed = True
            reviews[item["id"]] = {
                "status": "labeled",
                "confidence": "high",
                "action": action,
                "note": "smoke",
            }
        assert changed
        review_path = tmp / "reviews.json"
        review_path.write_text(json.dumps({
            "experiment": "E3",
            "queue_sha256": hashlib.sha256(queue.read_bytes()).hexdigest(),
            "reviews": reviews,
        }), encoding="utf-8")
        out = tmp / "export"
        cmd = [
            sys.executable, "-X", "utf8", str(ROOT / "scripts/p45_dagger_export.py"),
            "--queue", str(queue),
            "--reviews", str(review_path),
            "--out", str(out),
            "--min-labels", "20",
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        with np.load(out / "treatment/shard_000.npz") as tz, np.load(
                out / "control/shard_000.npz") as cz:
            assert set(tz.files) == set(cz.files)
            differences = []
            for key in tz.files:
                if not np.array_equal(tz[key], cz[key], equal_nan=True):
                    differences.append(key)
        assert differences == ["opt_chosen"], differences
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["train_corrections"] == 1
        assert manifest["holdout_labels"] > 0

        # Exercise E3's weighted anchor path through one tiny real training run.
        anchor = out / "anchor"
        anchor.mkdir()
        with np.load(out / "control/shard_000.npz") as z:
            arrays = {key: z[key] for key in z.files}
        arrays["gid"] = arrays["gid"].copy()
        arrays["gid"][0] -= arrays["gid"][0] % 20
        np.savez_compressed(anchor / "shard_000.npz", **arrays)
        trained = out / "trained.npz"
        train_cmd = [
            sys.executable, "-X", "utf8", str(ROOT / "scripts/train_policy.py"),
            "--ds", str(out / "treatment"),
            "--anchor-ds", str(anchor),
            "--primary-mass", "0.1",
            "--init", "out/policy_v5.npz",
            "--freeze-except", "head",
            "--state-h", "512,256",
            "--head-h", "256,128",
            "--pool",
            "--loss", "listwise",
            "--epochs", "1",
            "--bs", "32",
            "--lr", "2e-4",
            "--export-last",
            "--out", str(trained),
        ]
        subprocess.run(train_cmd, cwd=ROOT, check=True)
        with np.load(trained) as z:
            assert int(z["n_sfc"][0]) == 2
            assert int(z["n_head"][0]) == 3

    print("E3_SMOKE_OK queue, export isolation, and primary-mass fine-tune")
    return 0


if __name__ == "__main__":
    sys.exit(main())
