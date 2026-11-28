# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Indexer-only dispatch and native numerical contracts, independent of process groups."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.fusions.fused_csa2_indexer import (
    indexer_fusion_enabled,
    indexer_sort_rows,
    relu_weight_sum,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2Indexer


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_native_pointwise_dispatch_is_exact(monkeypatch, dtype):
    monkeypatch.setenv('MCORE_CSA2_INDEXER_FUSION', '1')
    torch.manual_seed(41)
    values = torch.randn(17, 3, 19, dtype=dtype)
    weights = torch.randn(17, 3, dtype=torch.float32)
    original_weights = weights.clone()
    with torch.no_grad():
        expected = (values.relu() * weights.unsqueeze(-1)).sum(1)
        actual = relu_weight_sum(values.clone(), weights, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(weights, original_weights, rtol=0, atol=0)
    assert actual.dtype == torch.float32


def test_supervision_keeps_eager_inputs_and_gradients(monkeypatch):
    monkeypatch.setenv('MCORE_CSA2_INDEXER_FUSION', '1')
    values = torch.randn(17, 3, 19, requires_grad=True)
    weights = torch.randn(17, 3, requires_grad=True)
    original = values.detach().clone()
    reference = (values.relu() * weights.unsqueeze(-1)).sum(1)
    actual = relu_weight_sum(values, weights, 1)
    ref_grads = torch.autograd.grad(reference.square().mean(), (values, weights))
    got_grads = torch.autograd.grad(actual.square().mean(), (values, weights))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0)
    for got, expected in zip(got_grads, ref_grads):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    with torch.no_grad():
        relu_weight_sum(values, weights, 1)
    torch.testing.assert_close(values, original, rtol=0, atol=0)


def test_indexer_sort_override_preserves_config_and_ties(monkeypatch):
    monkeypatch.setenv('MCORE_CSA2_INDEXER_SORT_ROWS', '11')
    model = SimpleNamespace(config=SimpleNamespace(csa_query_chunk_size=7), topk=5)
    scores = torch.randint(-2, 3, (101, 33)).float().T
    scores[0] = 0
    scores[1] = -torch.inf
    before = scores.clone()
    actual = CSA2Indexer.select_indices(model, scores)
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :5].sort(dim=-1).values
    expected = ids.masked_fill(~scores.gather(-1, ids).isfinite(), -1).int()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores, before, rtol=0, atol=0)
    assert model.config.csa_query_chunk_size == 7
    assert torch.equal(actual[0], torch.arange(5, dtype=torch.int32))
    assert torch.equal(actual[1], torch.full((5,), -1, dtype=torch.int32))


def test_execution_switch_validation(monkeypatch):
    monkeypatch.delenv('MCORE_CSA2_INDEXER_FUSION', raising=False)
    monkeypatch.delenv('MCORE_CSA2_INDEXER_SORT_ROWS', raising=False)
    assert not indexer_fusion_enabled()
    assert indexer_sort_rows(7) == 7
    monkeypatch.setenv('MCORE_CSA2_INDEXER_FUSION', 'true')
    with pytest.raises(ValueError):
        indexer_fusion_enabled()
    for invalid in ('0', '-1', '2.5', '1; false'):
        monkeypatch.setenv('MCORE_CSA2_INDEXER_SORT_ROWS', invalid)
        with pytest.raises(ValueError):
            indexer_sort_rows(7)
