# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.utils import cat_with_oom_fallback


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('offset', [0, 17])
def test_adjacent_checkpoint_views_reuse_storage(device, offset):
    buffer = torch.arange(100, device=device, dtype=torch.float32)
    original = buffer[offset : offset + 48].view(8, 6)
    parts = list(original.chunk(2))
    with patch('torch.cat', side_effect=AssertionError('unexpected allocation')):
        merged = cat_with_oom_fallback(parts)
    assert merged.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
    assert merged.storage_offset() == offset
    assert torch.equal(merged, original)


@pytest.mark.parametrize('case', ['separate', 'reversed', 'gap', 'noncontiguous', 'mixed_dtype'])
def test_nonadjacent_checkpoint_views_use_cat(case):
    buffer = torch.arange(64, dtype=torch.float32).view(8, 8)
    first, second = buffer.chunk(2)
    if case == 'separate':
        second = second.clone()
    elif case == 'reversed':
        first, second = second, first
    elif case == 'gap':
        first = first[:2]
    elif case == 'noncontiguous':
        first, second = first.t(), second.t()
    elif case == 'mixed_dtype':
        second = second.double()
    expected = torch.cat([first, second])
    merged = cat_with_oom_fallback([first, second])
    assert torch.equal(merged, expected)
    assert merged.untyped_storage().data_ptr() != buffer.untyped_storage().data_ptr()


def test_device_oom_falls_back_to_cpu():
    parts = [torch.ones(2, 3, device='cuda'), torch.zeros(2, 3, device='cuda')]
    original_cat = torch.cat

    def fail_device_cat(tensors):
        if tensors[0].is_cuda:
            raise torch.cuda.OutOfMemoryError('test device OOM')
        return original_cat(tensors)

    with patch('torch.cat', side_effect=fail_device_cat):
        merged = cat_with_oom_fallback(parts)
    assert merged.device.type == 'cpu'
    assert torch.equal(merged, original_cat([tensor.cpu() for tensor in parts]))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_swiglu_factory_reuses_ddp_buffer(device):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.transformer.mlp import apply_swiglu_sharded_factory

    buffer = torch.arange(100, dtype=torch.float32, device=device)
    original = buffer[17:65].view(8, 6)
    sharded = ShardedTensor.from_rank_offsets('fc1.weight', original)
    factory = apply_swiglu_sharded_factory(sharded, ())
    pieces = factory.build()
    with patch('torch.cat', side_effect=AssertionError('unexpected allocation')):
        merged = factory.merge_fn([piece.data for piece in pieces])
    assert merged.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
    assert merged.storage_offset() == 17
    assert torch.equal(merged, original)
