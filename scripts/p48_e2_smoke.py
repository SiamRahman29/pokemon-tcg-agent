"""Deterministic smoke checks for E2 residual adapters.

Verifies zero-initialized adapters preserve base logits, export/load round-trips,
legacy checkpoints ignore missing adapter keys, and freeze-except trains only
the adapter parameter group.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from train_policy import (BAGS, N_EXTRA, N_XSLOT, OPT_DENSE, PolicyNet,
                          apply_freeze, export_npz, load_init)
from sa import policynet
from sa.features import DENSE_DIM
from sa.routing import ROUTE_ALAKAZAM, ROUTE_GENERAL, ROUTE_MIRROR


def synthetic_batch():
    gen = torch.Generator().manual_seed(77)
    n_rows = 4
    dense = torch.randn(n_rows, DENSE_DIM, generator=gen)
    slots = torch.randint(0, 20, (n_rows, 12), generator=gen)
    bag_flat = {}
    bag_off = {}
    for j, name in enumerate(BAGS):
        # One id per row so EmbeddingBag offsets stay aligned with n_rows.
        bag_flat[name] = torch.tensor([1 + j, 2 + j, 3 + j, 4 + j])
        bag_off[name] = torch.tensor([0, 1, 2, 3, 4])
    seld = torch.zeros(n_rows, 14)
    seld[:, 0] = 1.0
    seld[:, 11] = torch.tensor([1, 1, 1, 1]) / 5.0
    seld[:, 12] = torch.tensor([1, 1, 1, 1]) / 5.0
    opt_row = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    chosen = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    n_opts = len(opt_row)
    opt_dense = torch.randn(n_opts, OPT_DENSE, generator=gen)
    opt_card = torch.randint(0, 20, (n_opts,), generator=gen)
    opt_atk = torch.randint(0, 10, (n_opts,), generator=gen)
    opt_tgt = torch.randint(0, 20, (n_opts,), generator=gen)
    xdense = torch.randn(n_rows, N_EXTRA, generator=gen)
    xslots = torch.randint(0, 20, (n_rows, N_XSLOT), generator=gen)
    routes = torch.tensor([ROUTE_GENERAL, ROUTE_MIRROR,
                           ROUTE_ALAKAZAM, ROUTE_GENERAL])
    return (dense, slots, bag_flat, bag_off, seld, opt_dense, opt_card,
            opt_atk, opt_tgt, opt_row, chosen, xdense, xslots, routes)


def main() -> int:
    batch = synthetic_batch()
    args = batch[:10] + batch[11:13]
    routes = batch[13]

    torch.manual_seed(123)
    control = PolicyNet(state_h=(32,), head_h=(16,), dropout=0.0, pool=True)
    torch.manual_seed(123)
    treatment = PolicyNet(state_h=(32,), head_h=(16,), dropout=0.0, pool=True,
                          adapter_names=["mirror", "alakazam"], adapter_h=8)
    # Policy base weights must match after identical seeding; adapters consume
    # RNG after the base, so only compare the shared parameter names.
    for name, value in control.state_dict().items():
        assert torch.equal(value, treatment.state_dict()[name]), name

    control.eval()
    treatment.eval()
    with torch.no_grad():
        base_logits = control(*args)
        # Zero-init adapters must leave logits unchanged on every route.
        treated = treatment(*args, routes=routes)
        assert torch.equal(base_logits, treated), "zero adapters changed logits"
        # adapters-off must also preserve logits even after a non-zero nudge.
        treatment.adapters["mirror"][-1].weight.fill_(0.1)
        treatment.adapters["mirror"][-1].bias.fill_(0.2)
        treatment.adapters_off = True
        off_logits = treatment(*args, routes=routes)
        assert torch.equal(base_logits, off_logits), "adapters-off leaked"
        treatment.adapters_off = False
        on_logits = treatment(*args, routes=routes)
        # Mirror rows (opt_row 1 -> options 2,3) must move; general must not.
        assert not torch.equal(base_logits, on_logits)
        assert torch.equal(base_logits[0:2], on_logits[0:2])
        assert torch.equal(base_logits[6:8], on_logits[6:8])
        assert not torch.equal(base_logits[2:4], on_logits[2:4])

    # Freeze contract: only adapters train.
    apply_freeze(treatment, "adapters")
    for name, p in treatment.named_parameters():
        if name.startswith("adapters."):
            assert p.requires_grad, name
        else:
            assert not p.requires_grad, name

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        table = np.ones((11, 64), dtype=np.float32)
        # Restore zero adapters for an equivalence export.
        torch.manual_seed(123)
        fresh = PolicyNet(state_h=(32,), head_h=(16,), dropout=0.0, pool=True,
                          adapter_names=["mirror", "alakazam"], adapter_h=8)
        with torch.no_grad():
            for name, value in control.state_dict().items():
                fresh.state_dict()[name].copy_(value)
        legacy_path = td / "legacy.npz"
        adapter_path = td / "adapters.npz"
        export_npz(control, legacy_path, table)
        export_npz(fresh, adapter_path, table)

        legacy = policynet.Net(np.load(legacy_path))
        adapted = policynet.Net(np.load(adapter_path))
        assert not legacy.adapters
        assert set(adapted.adapters) == {"mirror", "alakazam"}

        # Warm-start from legacy leaves adapters at zero init.
        torch.manual_seed(999)
        restored = PolicyNet(state_h=(32,), head_h=(16,), dropout=0.0,
                             pool=True, adapter_names=["mirror", "alakazam"],
                             adapter_h=8)
        before = restored.adapters["mirror"][-1].weight.detach().clone()
        load_init(restored, legacy_path)
        for name, value in control.state_dict().items():
            assert torch.equal(value, restored.state_dict()[name]), name
        assert torch.equal(before, restored.adapters["mirror"][-1].weight)

        # Round-trip adapter export restores residuals.
        load_init(restored, adapter_path)
        with torch.no_grad():
            assert torch.equal(control(*args),
                               restored(*args, routes=routes))

    print("E2_SMOKE_OK zero_init adapters_off freeze export warm_start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
