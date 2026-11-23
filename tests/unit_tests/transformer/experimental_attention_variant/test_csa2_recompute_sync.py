# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU rendezvous placement and numerical parity for full Hybrid recomputation."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from megatron.core.models.hybrid.hybrid_stack_adapter import (
    HybridStackForwardContext,
    _synchronize_recompute_group,
    checkpointed_hybrid_forward,
)
from megatron.core.tensor_parallel import random as checkpoint_runtime


def _run(sync_group):
    config = SimpleNamespace(
        recompute_method='uniform',
        recompute_num_layers=1,
        fp8=None,
        fp4=None,
        quant_recipe=None,
        distribute_saved_activations=False,
    )
    layers = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(3)])
    with torch.no_grad():
        for index, layer in enumerate(layers):
            layer.weight.copy_(torch.arange(16).reshape(4, 4) / 32 + index / 8)
            layer.layer_number = index + 1
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4).requires_grad_()
    events = []

    def layer_forward(layer, hidden, context):
        events.append(('layer', layer.layer_number, torch.is_grad_enabled()))
        return layer(hidden).tanh()

    with (
        patch.object(checkpoint_runtime, '_get_all_rng_states', return_value=(None, None, None)),
        patch.object(checkpoint_runtime, '_set_all_rng_states'),
        patch(
            'megatron.core.models.hybrid.hybrid_stack_adapter._synchronize_recompute_group'
        ) as barrier,
    ):
        barrier.side_effect = lambda group, first, last: events.append(
            ('barrier', group, first, last)
        )
        output = checkpointed_hybrid_forward(
            config,
            layers,
            inputs,
            HybridStackForwardContext(),
            tp_group=None,
            quantization_context=lambda *args: nullcontext(),
            layer_forward=layer_forward,
            recompute_sync_group=sync_group,
        )
        assert barrier.call_count == 0
        output.sum().backward()
        return output.detach(), inputs.grad, [layer.weight.grad for layer in layers], events


def test_sync_runs_before_replay_only_and_preserves_gradients():
    group = object()
    baseline = _run(None)
    actual = _run(group)
    for expected, value in zip(baseline[:3], actual[:3]):
        if isinstance(expected, list):
            for left, right in zip(expected, value):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        else:
            torch.testing.assert_close(expected, value, atol=0, rtol=0)
    assert actual[3][3:] == [
        ('barrier', group, 3, 3),
        ('layer', 3, True),
        ('barrier', group, 2, 2),
        ('layer', 2, True),
        ('barrier', group, 1, 1),
        ('layer', 1, True),
    ]
    assert not any(event[0] == 'barrier' for event in baseline[3])


@pytest.mark.parametrize('frozen', [False, True])
def test_no_grad_forward_never_enters_sync(frozen):
    config = SimpleNamespace(
        recompute_method='uniform',
        recompute_num_layers=1,
        fp8=None,
        fp4=None,
        quant_recipe=None,
        distribute_saved_activations=False,
    )
    layers = nn.ModuleList([nn.Identity()])
    layers[0].layer_number = 1
    inputs = torch.ones(2, 4, requires_grad=not frozen)
    with (
        torch.no_grad(),
        patch.object(checkpoint_runtime, '_get_all_rng_states', return_value=(None, None, None)),
        patch(
            'megatron.core.models.hybrid.hybrid_stack_adapter._synchronize_recompute_group'
        ) as barrier,
    ):
        checkpointed_hybrid_forward(
            config,
            layers,
            inputs,
            HybridStackForwardContext(),
            tp_group=None,
            quantization_context=lambda *args: nullcontext(),
            layer_forward=lambda layer, hidden, context: layer(hidden),
            recompute_sync_group=object(),
        )
        barrier.assert_not_called()


def test_frozen_checkpoint_inputs_do_not_sync_original_forward():
    config = SimpleNamespace(
        recompute_method='uniform',
        recompute_num_layers=1,
        fp8=None,
        fp4=None,
        quant_recipe=None,
        distribute_saved_activations=False,
    )
    layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
    layers[0].layer_number = 1
    with patch(
        'megatron.core.models.hybrid.hybrid_stack_adapter._synchronize_recompute_group'
    ) as sync:
        output = checkpointed_hybrid_forward(
            config,
            layers,
            torch.ones(2, 4),
            HybridStackForwardContext(),
            tp_group=None,
            quantization_context=lambda *args: nullcontext(),
            layer_forward=lambda layer, hidden, context: layer(hidden),
            recompute_sync_group=object(),
        )
        output.sum().backward()
        sync.assert_not_called()
        assert layers[0].weight.grad is not None


def test_recompute_group_identity_mismatch_is_rejected():
    group = SimpleNamespace(size=lambda: 2)

    def gather(outputs, identity, group):
        outputs[0].copy_(identity)
        outputs[1].copy_(identity + 2)

    with patch('megatron.core.models.hybrid.hybrid_stack_adapter.dist.all_gather', gather):
        with pytest.raises(RuntimeError, match='different Hybrid recompute groups'):
            _synchronize_recompute_group(group, 67, 68)
