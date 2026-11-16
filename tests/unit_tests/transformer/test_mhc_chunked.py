# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the native single-pass mHC call paths."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.hyper_connection import HyperConnectionModule, SinglePassMHCState


def module(device: str, hidden: int = 17) -> HyperConnectionModule:
    config = SimpleNamespace(
        num_residual_streams=4,
        hidden_size=hidden,
        mhc_single_pass=True,
        mhc_sinkhorn_iterations=4,
        mhc_epsilon=1e-6,
        mhc_init_gating_factor=0.1,
        layernorm_epsilon=1e-6,
        use_fused_mhc=False,
        sequence_parallel=False,
    )
    return HyperConnectionModule(config, 1).to(device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("recompute", [False, True])
def test_native_numerics(monkeypatch, dtype, batch, recompute):
    torch.manual_seed(41)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = module(device)
    raw = [
        torch.randn(11, batch, 68, device=device, dtype=dtype),
        torch.randn(11, batch, 4, device=device),
        torch.randn(11, batch, 17, device=device, dtype=dtype),
        torch.randn(17, device=device, dtype=dtype),
    ]
    results = []
    for size in [0, 7]:
        monkeypatch.setenv("MCORE_MHC_TOKEN_CHUNK_SIZE", str(size))
        m.zero_grad(set_to_none=True)
        inputs = [v.detach().clone().requires_grad_() for v in raw]

        def forward(hidden, mix, branch, bias):
            pre, post, res = m._compute_single_pass_mappings(hidden)
            aggregate = SinglePassMHCState(mix).contract(hidden, 4)
            residual = m._single_pass_bda(res, hidden, post, branch, bias, 0.0, True)
            final = SinglePassMHCState(pre).contract(residual, 4)
            return aggregate, residual, final

        output = checkpoint(forward, *inputs, use_reentrant=True) if recompute else forward(*inputs)
        loss = sum(v.float().square().mean() for v in output)
        loss.backward()
        results.append(
            (
                [v.detach().clone() for v in output],
                [v.grad.clone() for v in inputs],
                [p.grad.clone() for p in m.parameters()],
            )
        )
    for ref, got in zip(results[0], results[1]):
        for a, b in zip(ref, got):
            torch.testing.assert_close(
                b,
                a,
                rtol=0.025 if dtype == torch.bfloat16 else 2e-5,
                atol=0.004 if dtype == torch.bfloat16 else 2e-6,
            )


def test_bounded_square(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    class SquareProbe(TorchDispatchMode):
        def __init__(self):
            self.maximum = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.pow.Tensor_Scalar and args[1] == 2:
                self.maximum = max(self.maximum, args[0].numel())
            return func(*args, **(kwargs or {}))

    monkeypatch.setenv("MCORE_MHC_TOKEN_CHUNK_SIZE", "7")
    m = module("cpu")
    x = torch.randn(11, 3, 68, dtype=torch.bfloat16, requires_grad=True)
    probe = SquareProbe()
    with probe:
        output = m._compute_single_pass_mappings(x)
        sum(v.sum() for v in output).backward()
    assert probe.maximum <= 7 * 68, f"unbounded FP32 square: {probe.maximum} elements"
