"""Deterministic smoke checks for E1 multitask policy checkpoints.

This intentionally uses synthetic tensors: it verifies architecture and export
contracts without loading the 249k-row corpus.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from train_policy import (BAGS, N_EXTRA, N_XSLOT, OPT_DENSE, PolicyNet,
                          count_targets, export_npz, load_init)
from sa import policynet
from sa.features import DENSE_DIM


def synthetic_batch():
    gen = torch.Generator().manual_seed(77)
    n_rows = 3
    dense = torch.randn(n_rows, DENSE_DIM, generator=gen)
    slots = torch.randint(0, 20, (n_rows, 12), generator=gen)
    bag_flat = {}
    bag_off = {}
    for j, name in enumerate(BAGS):
        bag_flat[name] = torch.tensor([1 + j, 2 + j, 3 + j, 4 + j])
        bag_off[name] = torch.tensor([0, 2, 3, 4])
    seld = torch.zeros(n_rows, 14)
    seld[:, 0] = 1.0
    # Row targets: 0.0, 0.5, then a fixed-count row which must be masked.
    seld[:, 11] = torch.tensor([1, 1, 1]) / 5.0
    seld[:, 12] = torch.tensor([2, 3, 1]) / 5.0
    opt_row = torch.tensor([0, 0, 1, 1, 1, 2])
    chosen = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.float32)
    n_opts = len(opt_row)
    opt_dense = torch.randn(n_opts, OPT_DENSE, generator=gen)
    opt_card = torch.randint(0, 20, (n_opts,), generator=gen)
    opt_atk = torch.randint(0, 10, (n_opts,), generator=gen)
    opt_tgt = torch.randint(0, 20, (n_opts,), generator=gen)
    xdense = torch.randn(n_rows, N_EXTRA, generator=gen)
    xslots = torch.randint(0, 20, (n_rows, N_XSLOT), generator=gen)
    return (dense, slots, bag_flat, bag_off, seld, opt_dense, opt_card,
            opt_atk, opt_tgt, opt_row, chosen, xdense, xslots)


def main() -> int:
    batch = synthetic_batch()
    args = batch[:10] + batch[11:]

    # Auxiliary modules are constructed after the complete policy. With a
    # shared seed, adding them must not perturb any policy parameter or logit.
    torch.manual_seed(123)
    control = PolicyNet(state_h=(32,), head_h=(16,), dropout=0.0, pool=True)
    torch.manual_seed(123)
    treatment = PolicyNet(state_h=(32,), head_h=(16,), dropout=0.0, pool=True,
                          outcome=True, count=True)
    for name, value in control.state_dict().items():
        assert torch.equal(value, treatment.state_dict()[name]), name
    control.eval()
    treatment.eval()
    with torch.no_grad():
        base_logits = control(*args)
        aux_logits, srepr = treatment(*args, return_state=True)
    assert torch.equal(base_logits, aux_logits), "aux heads changed policy logits"
    assert srepr.shape == (3, 32)

    target, valid = count_targets(batch[4], batch[10], batch[9])
    np.testing.assert_allclose(target.numpy(), [0.0, 0.5, 0.0], atol=1e-6)
    assert valid.tolist() == [True, True, False]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        table = np.ones((11, 64), dtype=np.float32)
        legacy_path = td / "legacy.npz"
        aux_path = td / "aux.npz"
        export_npz(control, legacy_path, table)
        export_npz(treatment, aux_path, table)

        legacy = policynet.Net(np.load(legacy_path))
        aux = policynet.Net(np.load(aux_path))
        assert legacy.outcome_head is None and legacy.count_head is None
        assert aux.outcome_head is not None and aux.count_head is not None

        # A legacy warm start restores the policy while deliberately leaving
        # newly introduced auxiliary heads at their seeded initialization.
        torch.manual_seed(999)
        restored = PolicyNet(state_h=(32,), head_h=(16,), dropout=0.0,
                             pool=True, outcome=True, count=True)
        before_aux = restored.outcome_head.weight.detach().clone()
        load_init(restored, legacy_path)
        for name, value in control.state_dict().items():
            assert torch.equal(value, restored.state_dict()[name]), name
        assert torch.equal(before_aux, restored.outcome_head.weight)

    print("MULTITASK_SMOKE_OK policy_equivalence targets export warm_start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
