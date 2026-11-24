"""Clipped PPO for the entity Transformer: rollout sampling, GAE and the update.

Adapted from ptcg-population-rl (Apache-2.0, commit 35707d6),
`mega_lucario/ppo/rollout.py` and `mega_lucario/ppo/train.py`. Kept as
upstream: decision-level joint log-probability over the sequential picks,
terminal-only reward, per-seat GAE, one pass of clipped PPO with a clipped
scalar value loss plus the 51-bin distribution loss, held-out global KL, and
learning-rate calibration toward a KL target.

Changed:
  - Rollouts are played by `Sampler` (NumPy, in our Python arena) and
    recorded at decision time in the same sparse layout as distillation, so
    there is no separate extraction pass.
  - The behaviour audit recomputes a SAMPLE of decisions (upstream: all of
    them, in float64, on one thread).
  - The update uses `threads` CPU threads (upstream pinned one).
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tx.features import WIDTH, Encoder
from tx.numpy_policy import NumpyTransformer
from tx.train import _gather

DEFAULTS = dict(gamma=1.0, gae_lambda=0.99, clip=0.2, value_clip=0.2, value_coefficient=0.25,
                entropy_coefficient=0.01, learning_rate=3e-5, weight_decay=1e-4,
                max_grad_norm=0.5, epochs=1, updates=128, batch_size=1024,
                micro_batch_size=128, temperature=1.0, holdout_decisions=10000,
                kl_target=0.004, kl_tolerance=0.001, seed=204)


# ---- rollout sampling -----------------------------------------------------

class Sampler:
    """Samples a policy at a temperature; optionally records PPO decisions.

    Mirrors upstream's sampled C++ decide loop: one encoder pass per decision,
    then one pointer query per pick until STOP or maxCount. A query with a
    single legal candidate is forced and is neither scored nor recorded (its
    log-probability is 0). Value is the first scored query's, which equals
    every query's: the value head reads only the global token, and the
    encoder ignores the selection flags.
    """

    def __init__(self, model: NumpyTransformer, enc: Encoder, deck: list[int],
                 rng: np.random.Generator, temperature: float = 1.0, record: bool = True):
        self.model, self.enc, self.deck = model, enc, list(deck)
        self.rng, self.t, self.record = rng, temperature, record
        self.decisions: list[dict[str, Any]] = []

    def __call__(self, obs: dict[str, Any]) -> list[int]:
        sel = obs.get("select")
        if sel is None:
            return list(self.deck)
        if not sel.get("option") or sel.get("maxCount", 0) == 0:
            return []
        enc, model = self.enc, self.model
        action: list[int] = []
        tokens = hidden = None
        value = None
        logp = entropy = 0.0
        queries: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        while len(action) < sel["maxCount"]:
            options = enc.options_for(obs, action)
            if not options:
                raise ValueError("no legal Transformer candidate")
            if len(options) == 1:
                best = options[0]
            else:
                if tokens is None:
                    tokens = enc.tokenize(obs)
                    hidden = model.encode(tokens)
                current = enc.with_prefix(tokens, obs, action)
                logits, v = model.evaluate(hidden, current)
                if value is None:
                    value = v
                idx = enc.token_indices(obs, options)
                z = logits[idx].astype(np.float64) / self.t
                lp = z - (z.max() + math.log(np.exp(z - z.max()).sum()))
                p = np.exp(lp)
                j = int(self.rng.choice(len(options), p=p / p.sum()))
                logp += float(lp[j])
                entropy += float(-(p * lp).sum())
                if self.record:
                    flat = current.reshape(-1)
                    nz = np.flatnonzero(flat)
                    queries.append((nz.astype(np.uint16), flat[nz], len(current), idx[j]))
                best = options[j]
            if best == -1:
                break
            action.append(best)
        if self.record and queries:
            self.decisions.append(dict(seat=int(obs["current"]["yourIndex"]), queries=queries,
                                       logp=logp, value=float(value), entropy=entropy))
        return action


def pack_episodes(episodes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Flatten recorded episodes into the sparse rollout arrays.

    Each episode: dict(episode=int, reward=float, decisions=[Sampler records]).
    The terminal reward sits on the episode's last decision (done=True).
    """
    pos, val, nnz, size, target, d_size = [], [], [], [], [], []
    scal = {k: [] for k in ("old_logp", "old_value", "old_entropy", "reward", "done",
                            "episode", "step")}
    for ep in episodes:
        n = len(ep["decisions"])
        for s, d in enumerate(ep["decisions"]):
            for p_, v_, sz, t in d["queries"]:
                pos.append(p_)
                val.append(v_)
                nnz.append(len(p_))
                size.append(sz)
                target.append(t)
            d_size.append(len(d["queries"]))
            scal["old_logp"].append(d["logp"])
            scal["old_value"].append(d["value"])
            scal["old_entropy"].append(d["entropy"])
            last = s == n - 1
            scal["reward"].append(ep["reward"] if last else 0.0)
            scal["done"].append(last)
            scal["episode"].append(ep["episode"])
            scal["step"].append(s)
    out = dict(pos=np.concatenate(pos).astype(np.uint16), val=np.concatenate(val).astype(np.float32),
               nnz=np.asarray(nnz, np.int32), size=np.asarray(size, np.int64),
               target=np.asarray(target, np.int64), d_size=np.asarray(d_size, np.int64))
    for k, v in scal.items():
        dt = (np.bool_ if k == "done" else np.int64 if k in ("episode", "step")
              else np.float64)
        out[k] = np.asarray(v, dt)
    return out


def merge(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate packed rollouts and derive the ragged pointers."""
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["nnz_ptr"] = np.r_[0, np.cumsum(data["nnz"], dtype=np.int64)]
    data["d_ptr"] = np.r_[0, np.cumsum(data["d_size"], dtype=np.int64)]
    return data


# ---- training-side statistics (upstream) -----------------------------------

def compute_gae(values, rewards, done, episodes, steps, gamma: float = 1.0,
                lam: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    """GAE and returns independently for complete seat-specific episodes (upstream)."""
    values = np.asarray(values, np.float64)
    rewards = np.asarray(rewards, np.float64)
    done = np.asarray(done, bool)
    episodes = np.asarray(episodes)
    steps = np.asarray(steps)
    advantage = np.zeros_like(values)
    order = np.lexsort((steps, episodes))
    bounds = np.flatnonzero(np.diff(episodes[order])) + 1
    for rows in np.split(order, bounds):
        if not done[rows[-1]] or done[rows[:-1]].any() or len(np.unique(steps[rows])) != len(rows):
            raise ValueError("incomplete or inconsistent episode")
        following_value = following_advantage = 0.0
        for row in rows[::-1]:
            continuation = 0.0 if done[row] else 1.0
            delta = rewards[row] + gamma * continuation * following_value - values[row]
            following_advantage = delta + gamma * lam * continuation * following_advantage
            advantage[row] = following_advantage
            following_value = values[row]
    return advantage, advantage + values


def batch(data: dict[str, np.ndarray], decisions: np.ndarray, device: str,
          dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    """Padded prefix tensors for whole decisions (upstream `batch`, sparse source)."""
    decisions = np.asarray(decisions)
    q, owners = _gather(data["d_ptr"], decisions)
    first = np.searchsorted(owners, np.arange(len(decisions)))
    sizes = data["size"][q]
    width = int(sizes.max())
    x = np.zeros(len(q) * width * WIDTH, np.float32)
    e, row = _gather(data["nnz_ptr"], q)
    x[row * (width * WIDTH) + data["pos"][e]] = data["val"][e]
    mask = np.arange(width)[None, :] < sizes[:, None]
    return dict(x=torch.as_tensor(x.reshape(len(q), width, WIDTH), device=device, dtype=dtype),
                mask=torch.as_tensor(mask, device=device),
                targets=torch.as_tensor(data["target"][q], device=device, dtype=torch.long),
                owners=torch.as_tensor(owners, device=device),
                first=torch.as_tensor(first, device=device), count=len(decisions))


def joint_stats(model, b: dict[str, Any], temperature: float, distribution: bool = False):
    """Sum per-query log-probabilities and entropies; Value from the first query (upstream)."""
    logits, values, value_logits = model.forward_details(b["x"], b["mask"])
    logp = (logits / temperature).log_softmax(-1)
    query_logp = logp.gather(1, b["targets"][:, None]).squeeze(1)
    query_entropy = -(logp.exp() * logp.masked_fill(~torch.isfinite(logits), 0.0)).sum(-1)
    total = logits.new_zeros(b["count"])
    stats = (total.index_add(0, b["owners"], query_logp), values[b["first"]],
             total.index_add(0, b["owners"], query_entropy))
    return (*stats, value_logits[b["first"]]) if distribution else stats


@torch.no_grad()
def audit_behavior(model, data, temperature: float, sample: int, seed: int) -> dict[str, Any]:
    """Recompute a sample of recorded log-probs/values/entropies in torch float32."""
    model.eval()
    n = len(data["old_logp"])
    rows = np.sort(np.random.default_rng(seed).choice(n, min(sample, n), replace=False))
    errors = dict(logp=0.0, value=0.0, entropy=0.0)
    for start in range(0, len(rows), 128):
        r = rows[start:start + 128]
        stats = joint_stats(model, batch(data, r, "cpu"), temperature)
        for name, pred in zip(errors, stats):
            errors[name] = max(errors[name], float(np.abs(pred.numpy() - data["old_" + name][r]).max()))
    return dict(max_abs_errors=errors, decisions=len(rows))


def clipped_loss(new_logp, value, entropy, old_logp, old_value, advantage, returns, config):
    """Clipped policy and value objectives with entropy regularisation (upstream)."""
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    policy = -torch.minimum(ratio * advantage,
                            ratio.clamp(1 - config["clip"], 1 + config["clip"]) * advantage).mean()
    clipped_value = old_value + (value - old_value).clamp(-config["value_clip"], config["value_clip"])
    value_loss = 0.5 * torch.maximum((value - returns).square(),
                                     (clipped_value - returns).square()).mean()
    loss = (policy + config["value_coefficient"] * value_loss
            - config["entropy_coefficient"] * entropy.mean())
    return loss, dict(loss=loss, policy_loss=policy, value_loss=value_loss,
                      entropy=entropy.mean(), approximate_kl=(ratio - 1 - log_ratio).mean(),
                      clip_fraction=((ratio - 1).abs() > config["clip"]).float().mean())


def distribution_targets(returns: torch.Tensor) -> torch.Tensor:
    """Returns interpolated onto the 51-bin value support (upstream)."""
    position = (returns.clamp(-1, 1) + 1) * 25
    lower = position.floor().long()
    upper = position.ceil().long()
    fraction = position - lower
    target = returns.new_zeros((len(returns), 51))
    target.scatter_add_(1, lower[:, None], (1 - fraction)[:, None])
    target.scatter_add_(1, upper[:, None], fraction[:, None])
    return target


def update(model, optimizer, data: dict[str, np.ndarray], config: dict[str, Any],
           threads: int = 4) -> dict[str, Any]:
    """One clipped-PPO pass over a fresh rollout (upstream `train`, in memory).

    `optimizer` is updated in place and must belong to `model`. Returns the
    report, including the held-out global KL that calibration targets.
    """
    torch.set_num_threads(threads)
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    advantages, returns = compute_gae(data["old_value"], data["reward"], data["done"],
                                      data["episode"], data["step"], config["gamma"],
                                      config["gae_lambda"])
    n = len(advantages)
    holdout_count = min(config["holdout_decisions"], max(1, n // 10))
    heldout = np.linspace(0, n - 1, holdout_count, dtype=int)
    training = np.setdiff1d(np.arange(n), heldout)
    advantages = (advantages - advantages[training].mean()) / max(float(advantages[training].std()), 1e-8)
    for group in optimizer.param_groups:
        group.update(lr=config["learning_rate"], weight_decay=config["weight_decay"])
    model.float().train()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    order = rng.permutation(training)
    reports = []
    used = 0
    t = config["temperature"]
    for start in range(0, len(order), config["batch_size"]):
        if len(reports) >= config["updates"]:
            break
        indices = order[start:start + config["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        record: dict[str, float] = {}
        micro = config["micro_batch_size"]
        for chunk in range(0, len(indices), micro):
            rows = indices[chunk:chunk + micro]
            *stats, value_logits = joint_stats(model, batch(data, rows, "cpu"), t, distribution=True)
            tensors = [torch.as_tensor(a[rows], dtype=torch.float32)
                       for a in (data["old_logp"], data["old_value"], advantages, returns)]
            loss, metrics = clipped_loss(*stats, *tensors, config)
            distribution_loss = -(distribution_targets(tensors[-1])
                                  * F.log_softmax(value_logits, -1)).sum(-1).mean()
            loss = loss + config["value_coefficient"] * distribution_loss
            metrics.update(loss=loss, distribution_loss=distribution_loss)
            if any(not torch.isfinite(v).all() for v in metrics.values()):
                raise ValueError("non-finite PPO loss")
            fraction = len(rows) / len(indices)
            (loss * fraction).backward()
            for k, v in metrics.items():
                record[k] = record.get(k, 0.0) + float(v.detach()) * fraction
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"],
                                              error_if_nonfinite=True)
        optimizer.step()
        record["gradient_norm"] = float(norm)
        reports.append(record)
        used += len(indices)
    model.eval()
    with torch.no_grad():
        ratios, log_deltas = [], []
        for start in range(0, len(heldout), config["micro_batch_size"]):
            rows = heldout[start:start + config["micro_batch_size"]]
            new_logp = joint_stats(model, batch(data, rows, "cpu"), t)[0]
            delta = new_logp - torch.as_tensor(data["old_logp"][rows], dtype=new_logp.dtype)
            ratios.extend((delta.exp() - 1 - delta).tolist())
            log_deltas.extend((-delta).tolist())
    state = model.state_dict()
    delta_l2 = float(sum((state[k].double() - before[k].double()).square().sum()
                         for k in state).sqrt())
    if not reports or not np.isfinite(delta_l2) or delta_l2 <= 0:
        raise ValueError("PPO did not make a finite weight update")
    return dict(heldout_global_approximate_kl=float(np.mean(ratios)),
                heldout_sampled_forward_kl=float(np.mean(log_deltas)),
                holdout_decisions=len(heldout),
                holdout_indices_sha256=hashlib.sha256(heldout.astype("<i8").tobytes()).hexdigest(),
                training_decisions=len(training), updates=len(reports), decisions_used=used,
                decisions_available=n, parameter_delta_l2=delta_l2,
                learning_rate=config["learning_rate"],
                mean_metrics={k: float(np.mean([r[k] for r in reports])) for k in reports[0]},
                last_metrics=reports[-1])


def export_npz(model, payload: dict[str, Any], path) -> None:
    """Write the current weights as the NumPy `.npz` the samplers and agents load."""
    header = dict(format="scio-tx-npz-v1", contract_id=payload["contract_id"],
                  contract=payload["contract"], profile=payload["profile"],
                  profile_id=payload["profile_id"], architecture=payload["architecture"],
                  teacher_model_md5=payload.get("teacher_model_md5"),
                  stage=payload.get("stage", "distill"), cycle=payload.get("cycle"))
    tensors = {k: v.detach().cpu().numpy().astype(np.float32)
               for k, v in model.state_dict().items()}
    np.savez(path, __header__=np.array(json.dumps(header)), **tensors)
