# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bound native CSA2 scores and sorting while preserving packed selection semantics."""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CSA2Indexer,
    CSA2IndexerSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    build_csa2_thd_layout,
)


class _Linear(nn.Linear):
    def __init__(self, input_size, output_size, config, **kwargs):
        super().__init__(input_size, output_size, bias=False, dtype=config.params_dtype)

    def forward(self, x):
        return super().forward(x), None


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class _FrequencyTable(nn.Module):
    def forward(self, length, packed_seq=False):
        angle = torch.arange(length).float()[:, None] * torch.tensor([1.0, 0.01])
        return torch.cat((angle, angle), -1)[:, None, None, :]


def _indexer(ratio: int) -> CSA2Indexer:
    config = SimpleNamespace(
        hidden_size=6,
        q_lora_rank=4,
        v_head_dim=8,
        qk_pos_emb_head_dim=4,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=6,
        dsa_indexer_topk=5,
        attention_backend="unfused",
        dsa_kernel_backend="none",
        dsa_indexer_precision="bf16",
        params_dtype=torch.float32,
        init_method=lambda weight: nn.init.normal_(weight, std=0.2),
        attention_latent_norm_epsilon=1e-6,
        apply_rope_fusion=False,
        rotary_interleaved=False,
        multi_latent_attention=True,
        mrope_section=None,
        csa_query_chunk_size=0,
        csa2_candidate_topk_blocks=3,
        csa2_candidate_block_size=2,
    )
    groups = ProcessGroupCollection()
    groups.tp = groups.cp = torch.distributed.ProcessGroup(0, 1)
    return CSA2Indexer(
        config, CSA2IndexerSubmodules(_Linear, _Linear, _RMSNorm, _Linear), ratio, groups
    )


def _layout(real: list[int], padded: list[int], tail: int = 0):
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, *accumulate(real)], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, *accumulate(real)], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, *accumulate(padded)], dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, *accumulate(padded)], dtype=torch.int32),
        max_seqlen_q=max(padded, default=0),
        max_seqlen_kv=max(padded, default=0),
        total_tokens=sum(padded) + tail,
        pad_between_seqs=True,
    )
    return build_csa2_thd_layout(params, sum(padded) + tail)


class _WorkspaceProbe(TorchDispatchMode):
    def __init__(self, keys: int):
        self.keys = keys
        self.max_sort_rows = 0
        self.max_score_rows = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.sort.stable, torch.ops.aten.sort.default) and args[0].ndim == 2:
            self.max_sort_rows = max(self.max_sort_rows, args[0].shape[0])
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.empty.memory_format, torch.ops.aten.new_empty.default):
            if result.ndim == 2 and result.shape[1] == self.keys and result.dtype == torch.float32:
                self.max_score_rows = max(self.max_score_rows, result.shape[0])
        return result


@pytest.mark.parametrize("shape", [(33, 101), (2, 33, 101), (1, 1), (0, 101), (33, 0)])
@pytest.mark.parametrize("topk", [0, 5, 200])
def test_chunked_selection_preserves_ties_mask_and_empty_inputs(shape, topk):
    torch.manual_seed(41)
    scores = torch.randint(-2, 3, shape).float()
    if scores.numel():
        scores[..., ::3] = -torch.inf
        scores[..., 0, :] = -torch.inf
    indexer = _indexer(1)
    indexer.topk = topk
    expected = indexer.select_indices(scores)
    for chunk in [1, 7, 64]:
        indexer.config.csa_query_chunk_size = chunk
        actual = indexer.select_indices(scores)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == torch.int32


def test_chunked_selection_bounds_sort_and_accepts_strided_scores():
    indexer = _indexer(1)
    scores = torch.randint(-2, 3, (101, 33)).float().T
    expected = indexer.select_indices(scores)
    indexer.config.csa_query_chunk_size = 7
    torch.testing.assert_close(indexer.select_indices(scores), expected, rtol=0, atol=0)
    probe = _WorkspaceProbe(101)
    with probe:
        indices = indexer.select_indices(torch.zeros(33, 101))
    assert 0 < probe.max_sort_rows <= 7
    torch.testing.assert_close(indices[0], torch.arange(5, dtype=torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("candidate_kind", ["none", "dense", "compact"])
@pytest.mark.parametrize("topk", [0, 5, 100])
@pytest.mark.parametrize("export_candidates", [False, True])
def test_native_packed_score_selection_matches_unchunked(
    ratio, candidate_kind, topk, export_candidates
):
    torch.manual_seed(41)
    layout = _layout([17, 16], [20, 20], tail=3)
    compressed = layout.for_compression(ratio)
    indexer = _indexer(ratio)
    indexer.topk = topk
    x = torch.randn(layout.total_tokens, 1, 6)
    qr = torch.randn(layout.total_tokens, 1, 4)
    qr[::4] = 0  # exact score ties exercise stable local-key order
    latent = torch.randn(compressed.capacity, 1, 8)
    candidates = None
    if candidate_kind == "dense":
        candidates = (torch.arange(compressed.capacity) % 3 != 1).expand(layout.total_tokens, -1)
    elif candidate_kind == "compact":
        candidates = CSA2CandidateBlocks(
            torch.tensor([[0, 2, -1]], dtype=torch.int32).expand(layout.total_tokens, -1),
            torch.full((layout.total_tokens,), 2, dtype=torch.int32),
            2,
        )
    kwargs = dict(
        candidates=candidates,
        thd_layout=layout,
        compressed_layout=compressed,
        return_candidates=export_candidates,
    )
    snapshots = [v.clone() for v in (x, qr, latent)]
    with torch.no_grad():
        expected = indexer.forward_with_scores(x, qr, latent, _FrequencyTable(), **kwargs)
        indexer.config.csa_query_chunk_size = 7
        probe = _WorkspaceProbe(compressed.capacity)
        with probe:
            actual = indexer.forward_with_scores(x, qr, latent, _FrequencyTable(), **kwargs)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert actual[1] is None
    if export_candidates:
        torch.testing.assert_close(actual[2].indices, expected[2].indices, rtol=0, atol=0)
        torch.testing.assert_close(actual[2].lengths, expected[2].lengths, rtol=0, atol=0)
    else:
        assert actual[2] is None
    assert 0 < probe.max_score_rows <= 7
    assert probe.max_sort_rows <= 7
    for value, before in zip((x, qr, latent), snapshots):
        torch.testing.assert_close(value, before, rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1, 2])
def test_supervised_native_path_keeps_scores_and_gradients(ratio):
    torch.manual_seed(42)
    layout = _layout([5, 9], [6, 10], tail=3)
    compressed = layout.for_compression(ratio)
    indexer = _indexer(ratio)
    raw = [
        torch.randn(layout.total_tokens, 1, 6),
        torch.randn(layout.total_tokens, 1, 4),
        torch.randn(compressed.capacity, 1, 8),
    ]
    results = []
    for chunk in [0, 7]:
        indexer.config.csa_query_chunk_size = chunk
        indexer.zero_grad(set_to_none=True)
        inputs = [v.clone().requires_grad_() for v in raw]
        selected, scores, _ = indexer.forward_with_scores(
            *inputs,
            _FrequencyTable(),
            thd_layout=layout,
            compressed_layout=compressed,
            return_scores=True,
        )
        assert scores is not None and scores.requires_grad
        scores.masked_fill(~scores.isfinite(), 0).square().mean().backward()
        results.append(
            (
                selected,
                [x.grad.clone() for x in inputs],
                [p.grad.clone() for p in indexer.parameters()],
            )
        )
    for expected, actual in zip(results[0], results[1]):
        if isinstance(expected, list):
            for ref, got in zip(expected, actual):
                torch.testing.assert_close(got, ref, rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
