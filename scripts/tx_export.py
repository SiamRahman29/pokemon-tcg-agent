"""Export a distilled checkpoint (`model.pt`) to a torch-free `.npz`, and check it.

The `.npz` holds every state-dict tensor as float32 plus `__header__`, a JSON
string with the contract, deck profile and source checkpoint hash. After
writing, it replays `--check` decisions from a teacher `.jsonl.gz` through
both torch and NumPy and fails if any logit differs by more than `--atol` or
any greedy action differs.

    python -X utf8 scripts/tx_export.py --checkpoint out/tx_model/model.pt \\
        --output out/tx_model/model.npz --games out/teacher/s0.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "agents", "scripts"):
    sys.path.insert(0, str(ROOT / sub))

import torch  # noqa: E402

from tx.checkpoint import load_checkpoint  # noqa: E402
from tx.deck import DeckProfile, engine_tables  # noqa: E402
from tx.features import Encoder  # noqa: E402
from tx.model import select_logits  # noqa: E402
from tx.numpy_policy import NumpyTransformer, decide  # noqa: E402


def profile_from_json(p: dict) -> DeckProfile:
    return DeckProfile(
        name=p["name"], counts={int(k): v for k, v in p["counts"].items()},
        ids=tuple(sorted(int(k) for k in p["counts"])), attack_ids=tuple(p["attack_ids"]),
        attacks={int(k): (d, tuple(c)) for k, (d, c) in p["attacks"].items()},
        energy_type=p["energy_type"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--games", type=Path, required=True,
                    help="a teacher .jsonl.gz (or a directory holding some) to draw "
                         "parity decisions from")
    ap.add_argument("--check", type=int, default=500, help="decisions to compare")
    ap.add_argument("--atol", type=float, default=1e-4)
    a = ap.parse_args()

    if a.games.is_dir():
        found = sorted(a.games.rglob("*.jsonl.gz"))
        if not found:
            raise SystemExit(f"no teacher .jsonl.gz under {a.games}")
        a.games = found[0]
    model, payload = load_checkpoint(a.checkpoint)
    profile = profile_from_json(payload["profile"])
    if profile.profile_id != payload["profile_id"]:
        raise SystemExit("checkpoint profile does not hash to its profile_id")
    cards, _ = engine_tables()
    enc = Encoder(profile, cards)
    if enc.contract_id != payload["contract_id"]:
        raise SystemExit("this source tree's encoder does not match the checkpoint contract")
    header = dict(format="scio-tx-npz-v1", contract_id=payload["contract_id"],
                  contract=payload["contract"], profile=payload["profile"],
                  profile_id=payload["profile_id"], architecture=payload["architecture"],
                  teacher_model_md5=payload["teacher_model_md5"], epoch=payload["epoch"],
                  checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
    tensors = {k: v.detach().cpu().numpy().astype(np.float32)
               for k, v in model.state_dict().items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, __header__=np.array(json.dumps(header)), **tensors)

    npm = NumpyTransformer(a.output, enc)
    worst = 0.0
    n = action_diff = top_diff = 0
    with gzip.open(a.games, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] != "decision" or r["teacher_scores"] is None:
                continue
            obs = r["observation"]
            x = enc.tokenize(obs)
            xt = torch.as_tensor(x[None])
            with torch.no_grad():
                lt = model(xt, torch.ones(1, len(x), dtype=torch.bool))[0].numpy()
            ln, _ = npm.evaluate(npm.encode(x), x)
            fin = np.isfinite(lt)
            if not np.array_equal(fin, np.isfinite(ln)):
                raise SystemExit("legal masks differ between torch and NumPy")
            worst = max(worst, float(np.abs(lt[fin] - ln[fin]).max()))
            top_diff += int(select_logits(torch.as_tensor(lt[None]))[0]) != int(
                np.argmax(ln >= ln[fin].max() - 1e-5))
            decide(npm, obs, enc)          # exercises the full multi-select loop
            n += 1
            if n >= a.check:
                break
    print(json.dumps(dict(output=str(a.output), decisions=n, max_abs_logit_diff=worst,
                          top1_differs=top_diff, size_mb=round(a.output.stat().st_size / 1e6, 2))))
    if worst > a.atol or top_diff:
        raise SystemExit(f"NumPy/torch parity failed (max diff {worst}, top-1 differs {top_diff})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
