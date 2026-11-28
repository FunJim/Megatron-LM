# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FP32 Indexer pointwise fusion with the original PyTorch head reduction.

GEMM, head reduction, masks and stable selection are intentionally not fused.
Only a forward-local GEMM result may be overwritten; graph-connected supervision
always uses the original out-of-place operations. No activation is cached.
"""

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _relu_weight_inplace(scores, weights, size, keys: tl.constexpr, BLOCK: tl.constexpr):
        index = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        value = tl.load(scores + index, index < size, other=0)
        weight = tl.load(weights + index // keys, index < size, other=0)
        rectified = tl.where(value < 0.0, 0.0, value)
        tl.store(scores + index, rectified * weight, index < size)


def indexer_fusion_enabled() -> bool:
    value = os.environ.get("MCORE_CSA2_INDEXER_FUSION", "0")
    if value not in ("0", "1"):
        raise ValueError("MCORE_CSA2_INDEXER_FUSION must be 0 or 1")
    return value == "1"


def indexer_sort_rows(default: int) -> int:
    """Optional Indexer-only sort batching; never changes Attention query chunks."""
    value = os.environ.get("MCORE_CSA2_INDEXER_SORT_ROWS")
    if value is None:
        return default
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise ValueError("MCORE_CSA2_INDEXER_SORT_ROWS must be a positive integer")
    return int(value)


def relu_weight_sum(scores: torch.Tensor, weights: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Consume a fresh FP32 GEMM result, preserving the native sum's reduction order."""
    eligible = (
        indexer_fusion_enabled()
        and triton is not None
        and not torch.is_grad_enabled()
        and not scores.requires_grad
        and scores.is_cuda
        and scores.dtype == torch.float32
        and weights.dtype == torch.float32
        and scores.device == weights.device
        and scores.is_contiguous()
        and tuple(weights.shape) == tuple(scores.shape[:-1])
        and scores.numel() >= 64 * 1024 * 1024
    )
    if not eligible:
        if (indexer_fusion_enabled() and not torch.is_grad_enabled()
                and not scores.requires_grad and scores.dtype == weights.dtype == torch.float32):
            return scores.relu_().mul_(weights.unsqueeze(-1)).sum(dim=head_dim)
        return (scores.relu() * weights.unsqueeze(-1)).sum(dim=head_dim)
    weight_values = weights.contiguous()
    _relu_weight_inplace[(triton.cdiv(scores.numel(), 1024),)](
        scores, weight_values, scores.numel(), scores.shape[-1],
        BLOCK=1024, num_warps=4, enable_fp_fusion=False,
    )
    return scores.sum(dim=head_dim)
