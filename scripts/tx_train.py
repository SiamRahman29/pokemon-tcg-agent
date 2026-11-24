"""Train the distilled Transformer from `scripts/tx_extract.py` shards.

    python -X utf8 scripts/tx_train.py --data out/tx_data --output out/tx_model \\
        --device cuda --max-hours 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agents"))

from tx.train import train  # noqa: E402


def find_data(root: Path) -> Path:
    """The one distillation dataset under `root` (e.g. /kaggle/input, whose
    mount layout Kaggle has changed before: never hardcode the slug)."""
    import json

    hits = [m.parent for m in root.rglob("manifest.json")
            if json.loads(m.read_text()).get("format") == "scio-tx-distill-v1"]
    if len(hits) != 1:
        raise SystemExit(f"expected one distillation manifest under {root}, found {hits}")
    print(f"data: {hits[0]}", flush=True)
    return hits[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--hard-weight", type=float, default=0.1)
    p.add_argument("--value-weight", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=204)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--max-updates", type=int)
    p.add_argument("--max-hours", type=float,
                   help="stop and keep the best checkpoint after this long")
    p.add_argument("--log-every", type=int, default=500)
    a = p.parse_args()
    if not (a.data / "manifest.json").exists():
        a.data = find_data(a.data)
    train(a.data, a.output, epochs=a.epochs, batch_size=a.batch_size,
          learning_rate=a.learning_rate, temperature=a.temperature,
          hard_weight=a.hard_weight, value_weight=a.value_weight, seed=a.seed,
          device=a.device, max_updates=a.max_updates, max_hours=a.max_hours,
          log_every=a.log_every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
