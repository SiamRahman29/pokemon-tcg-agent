"""Distil the teacher's policy and the games' outcomes into the Transformer.

Adapted from ptcg-population-rl (Apache-2.0, commit 35707d6),
`mega_lucario/transformer/train.py`. The loss, metrics and optimisation are
upstream's: KL to the teacher's softmax on each decision's first query, plus
`hard_weight` x the joint NLL of the full ordered selection (picks and STOP),
plus `value_weight` x cross-entropy of the 51-bin value head against the
acting seat's terminal outcome. AdamW, weight decay 1e-4, grad clip 0.5, the
best validation loss checkpoint is kept.

Changed:
  - Data comes from `scripts/tx_extract.py`'s compact shards. Tokens stay
    sparse in RAM and each padded batch is built on demand: 30k games are
    ~6 GB this way and ~72 GB dense, more than a Kaggle notebook has.
  - Checkpoints bind to a deck profile instead of Mega Lucario's schema.
  - `max_hours` stops cleanly so a hosted run always writes its checkpoint,
    and progress is logged within an epoch.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tx.features import WIDTH
from tx.model import ARCHITECTURE, ObservationTransformer, select_logits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(directory: Path, manifest: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    """Concatenate one split's shards, verifying checksums and alignment."""
    parts: dict[str, list[np.ndarray]] = {k: [] for k in (
        "pos", "val", "nnz", "size", "target", "cand_idx", "cand_score", "n_cand",
        "d_size", "d_outcome")}
    for shard in manifest["splits"][split]["shards"]:
        path = directory / shard["path"]
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"checksum mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as d:
            q, ds = len(d["size"]), d["d_size"]
            if (int(ds.sum()) != q or len(d["nnz"]) != q or len(d["target"]) != q
                    or int(d["nnz"].sum()) != len(d["pos"]) or len(d["pos"]) != len(d["val"])
                    or int(d["n_cand"].sum()) != len(d["cand_idx"])
                    or not np.isin(d["d_outcome"], [-1, 0, 1]).all()
                    or not np.isfinite(d["val"]).all() or not np.isfinite(d["cand_score"]).all()
                    or np.any(d["target"] >= d["size"]) or np.any(d["size"] < 20)):
                raise ValueError(f"invalid shard: {path.name}")
            for k in parts:
                parts[k].append(d[k])
    if not parts["size"]:
        raise ValueError(f"empty split: {split}")
    data = {k: np.concatenate(v) for k, v in parts.items()}
    data["nnz_ptr"] = np.r_[0, np.cumsum(data["nnz"], dtype=np.int64)]
    data["cand_ptr"] = np.r_[0, np.cumsum(data["n_cand"], dtype=np.int64)]
    data["d_ptr"] = np.r_[0, np.cumsum(data["d_size"], dtype=np.int64)]
    data["size"] = data["size"].astype(np.int64)
    data["target"] = data["target"].astype(np.int64)
    data["outcome"] = data.pop("d_outcome").astype(np.float32)
    del data["nnz"], data["n_cand"], data["d_size"]
    return data


def _gather(ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Element indices of the given rows of a ragged array, and each one's row."""
    starts, ends = ptr[rows], ptr[rows + 1]
    lens = ends - starts
    owner = np.repeat(np.arange(len(rows)), lens)
    offset = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens)
    return np.repeat(starts, lens) + offset, owner


def batches(data: dict[str, np.ndarray], batch_size: int, device: str,
            rng: np.random.Generator | None = None) -> Iterator[tuple[torch.Tensor, ...]]:
    """Padded batches of whole decisions, as upstream: (x, mask, teacher score,
    target, owner decision, first query of each decision, outcome)."""
    order = np.arange(len(data["outcome"]))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        decisions = order[start:start + batch_size]
        q, owners = _gather(data["d_ptr"], decisions)
        first = np.searchsorted(owners, np.arange(len(decisions)))
        sizes = data["size"][q]
        width = int(sizes.max())
        x = np.zeros(len(q) * width * WIDTH, np.float32)
        e, row = _gather(data["nnz_ptr"], q)
        x[row * (width * WIDTH) + data["pos"][e]] = data["val"][e]
        score = np.full((len(q), width), -np.inf, np.float32)
        c, crow = _gather(data["cand_ptr"], q)
        score[crow, data["cand_idx"][c]] = data["cand_score"][c]
        mask = np.arange(width)[None, :] < sizes[:, None]
        yield tuple(torch.as_tensor(a, device=device) for a in (
            x.reshape(len(q), width, WIDTH), mask, score, data["target"][q], owners,
            first, data["outcome"][decisions]))


def value_targets(outcomes: torch.Tensor) -> torch.Tensor:
    """Terminal outcomes as targets for the 51-bin value distribution (upstream)."""
    position = (outcomes.clamp(-1, 1) + 1) * 25
    lower = position.floor().long()
    upper = position.ceil().long()
    fraction = position - lower
    target = outcomes.new_zeros((len(outcomes), 51))
    target.scatter_add_(1, lower[:, None], (1 - fraction)[:, None])
    target.scatter_add_(1, upper[:, None], fraction[:, None])
    return target


def losses(logits, teacher, mask, target, temperature: float, hard_weight: float):
    """Teacher KL and hard-action loss over legal candidates (upstream)."""
    mask = mask & torch.isfinite(logits)
    probability = (teacher.masked_fill(~mask, -torch.inf) / temperature).softmax(-1)
    log_probability = logits.log_softmax(-1)
    hard = F.cross_entropy(logits, target)
    kl = (probability * (probability.clamp_min(1e-30).log()
                         - log_probability.masked_fill(~mask, 0.0))).sum(-1).mean()
    return hard_weight * hard + kl, kl


def decision_loss(model, b, temperature: float, hard_weight: float, value_weight: float):
    """Combined decision loss, its components, and the policy logits (upstream)."""
    x, mask, teacher, target, owners, first, outcome = b
    logits, values, value_logits = model.forward_details(x, mask)
    _, kl = losses(logits[first], teacher[first], mask[first], target[first],
                   temperature, hard_weight)
    nll = F.cross_entropy(logits, target, reduction="none")
    joint = logits.new_zeros(len(first)).index_add(0, owners, nll).mean()
    value_loss = -(value_targets(outcome) * value_logits[first].log_softmax(-1)).sum(-1).mean()
    loss = kl + hard_weight * joint + value_weight * value_loss
    return loss, dict(kl=kl, action_nll=joint, value_loss=value_loss,
                      value_mse=(values[first] - outcome).square().mean()), logits


@torch.no_grad()
def evaluate(model, data, batch_size: int, device: str, temperature: float,
             hard_weight: float, value_weight: float) -> dict[str, Any]:
    """Held-out policy and value metrics. `teacher_top1` counts queries where
    the model's pick (within the tie tolerance) is the teacher's target."""
    model.eval()
    total = correct = queries = 0
    sums = {k: 0.0 for k in ("loss", "kl", "action_nll", "value_loss", "value_mse")}
    for b in batches(data, batch_size, device):
        loss, metrics, logits = decision_loss(model, b, temperature, hard_weight, value_weight)
        n = len(b[-1])
        total += n
        queries += len(logits)
        correct += int((select_logits(logits) == b[3]).sum())
        for k, v in dict(loss=loss, **metrics).items():
            sums[k] += float(v) * n
    return dict(decisions=total, queries=queries, teacher_top1=correct / queries,
                **{k: v / total for k, v in sums.items()})


def train(directory: Path, output: Path, *, epochs: int = 5, batch_size: int = 256,
          learning_rate: float = 3e-4, temperature: float = 1.0, hard_weight: float = 0.1,
          value_weight: float = 0.25, seed: int = 204, device: str = "cuda",
          max_updates: int | None = None, max_hours: float | None = None,
          log_every: int = 500) -> dict[str, Any]:
    """Distil policy and value from game-split shards; keep the best checkpoint."""
    t0 = time.time()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    manifest = json.loads((directory / "manifest.json").read_text())
    digest = hashlib.sha256(json.dumps(manifest["contract"], sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    if digest != manifest["contract_id"]:
        raise ValueError("Transformer contract mismatch")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(seed)
    model = ObservationTransformer(WIDTH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_data = load_split(directory, manifest, "train")
    validation = load_split(directory, manifest, "validation")
    ram_gb = sum(a.nbytes for d in (train_data, validation) for a in d.values()) / 1e9
    print(json.dumps(dict(train_decisions=len(train_data["outcome"]),
                          train_queries=len(train_data["size"]),
                          validation_decisions=len(validation["outcome"]),
                          data_ram_gb=round(ram_gb, 2),
                          load_minutes=round((time.time() - t0) / 60, 1))), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(epochs=epochs, max_updates=max_updates, max_hours=max_hours,
                  batch_size=batch_size, learning_rate=learning_rate, temperature=temperature,
                  hard_weight=hard_weight, value_weight=value_weight,
                  value_target="terminal outcome per acting seat",
                  loss="KL(first query) + hard_weight * joint NLL + value_weight * distribution CE",
                  seed=seed, device=device, weight_decay=1e-4, gradient_clip=0.5,
                  hardware=torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
                  torch_version=str(torch.__version__))
    updates = 0
    history: list[dict[str, Any]] = []
    best = float("inf")
    out_of_time = False
    per_epoch = -(-len(train_data["outcome"]) // batch_size)
    for epoch in range(epochs):
        model.train()
        total = 0
        accumulated = 0.0
        for b in batches(train_data, batch_size, device, rng):
            if max_updates is not None and updates >= max_updates:
                break
            if max_hours is not None and time.time() - t0 > max_hours * 3600:
                out_of_time = True
                break
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = decision_loss(model, b, temperature, hard_weight, value_weight)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total += len(b[-1])
            accumulated += float(loss.detach()) * len(b[-1])
            if updates % log_every == 0:
                print(json.dumps(dict(epoch=epoch + 1, update=updates, of=per_epoch * epochs,
                                      train_loss=round(accumulated / total, 4),
                                      minutes=round((time.time() - t0) / 60, 1))), flush=True)
        if total == 0:
            break
        metrics = evaluate(model, validation, batch_size, device, temperature, hard_weight,
                           value_weight)
        history.append(dict(epoch=epoch + 1, updates=updates, train_loss=accumulated / total,
                            partial=out_of_time, validation=metrics,
                            minutes=round((time.time() - t0) / 60, 1)))
        print(json.dumps(history[-1]), flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(dict(
                format="scio-tx-checkpoint-v1", contract_id=manifest["contract_id"],
                contract=manifest["contract"], architecture=ARCHITECTURE,
                profile=manifest["profile"], profile_id=manifest["profile_id"],
                teacher_model_md5=manifest["teacher_model_md5"],
                dataset_sha256=sha256(directory / "manifest.json"), config=config,
                epoch=epoch + 1, validation=metrics,
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            ), output / "model.pt")
        if out_of_time or (max_updates is not None and updates >= max_updates):
            break
    payload = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    report = dict(config=config, updates=updates, history=history,
                  selected_epoch=payload["epoch"], stopped_on_time=out_of_time,
                  checkpoint_sha256=sha256(output / "model.pt"),
                  teacher_model_md5=manifest["teacher_model_md5"],
                  minutes=round((time.time() - t0) / 60, 1))
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(selected_epoch=payload["epoch"], validation=payload["validation"])),
          flush=True)
    return report
