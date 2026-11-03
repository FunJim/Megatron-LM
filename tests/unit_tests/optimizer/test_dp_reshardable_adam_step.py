# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Preserve the checkpoint Adam step when loading nonpersistent bucket state."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject
from megatron.core.dist_checkpointing.state_dict_utils import load_preprocess
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


@pytest.mark.parametrize('precision_aware', [False, True])
@pytest.mark.parametrize('model_dtype', [torch.bfloat16, torch.float32])
def test_dp_reshardable_preserves_adam_step(precision_aware, model_dtype):
    """A template step must not replace the saved step or change the next Adam update."""
    model_param = torch.nn.Parameter(torch.ones(4, dtype=model_dtype))
    master = torch.nn.Parameter(torch.full((4,), 0.75))
    adam = torch.optim.AdamW([master], lr=1e-3, foreach=False, fused=False)
    adam.state[master] = {
        'step': torch.tensor(3.0),
        'exp_avg': torch.zeros_like(master),
        'exp_avg_sq': torch.zeros_like(master),
    }
    reference = torch.nn.Parameter(torch.full((4,), 0.75))
    reference_adam = torch.optim.AdamW([reference], lr=1e-3, foreach=False, fused=False)
    reference_adam.state[reference] = {
        'step': torch.tensor(3.0),
        'exp_avg': torch.full_like(reference, 0.02),
        'exp_avg_sq': torch.full_like(reference, 0.002),
    }

    wrapper = DistributedOptimizer.__new__(DistributedOptimizer)
    wrapper.config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware)
    if precision_aware:
        hdo = HybridDeviceOptimizer.__new__(HybridDeviceOptimizer)
        torch.optim.Optimizer.__init__(hdo, [model_param], {})
        hdo.state[model_param] = adam.state[master]
        hdo.state[model_param]['master_param'] = master
        wrapper.optimizer = hdo
    else:
        wrapper.optimizer = adam
    wrapper.model_param_group_index_map = {model_param: (0, 0)}
    dtype = (model_dtype, torch.float32)
    wrapper.gbuf_ranges = [{dtype: [{'param_map': {model_param: {}}}]}]
    tensors = {
        'param': reference.detach().clone(),
        'exp_avg': torch.full_like(master, 0.02),
        'exp_avg_sq': torch.full_like(master, 0.002),
        'step': LocalNonpersistentObject(torch.tensor(1.0)),
        'padding': LocalNonpersistentObject(False),
    }
    _, nonpersistent, _ = load_preprocess({'bucket': [tensors]})
    source = {**tensors, **nonpersistent['bucket'][0]}
    template_step = source['step']
    state = {0: {dtype: [[{'padding': True}, source]]}}

    wrapper.load_parameter_state_from_dp_reshardable(state)

    assert source['step'] is template_step and template_step.item() == 1
    assert adam.state[master]['step'].item() == 3
    if precision_aware:
        assert hdo.state[model_param] is adam.state[master]
    master.grad = torch.full_like(master, 0.1)
    reference.grad = master.grad.clone()
    adam.step()
    reference_adam.step()
    torch.testing.assert_close(master, reference, rtol=0, atol=0)
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(
            adam.state[master][key], reference_adam.state[reference][key], rtol=0, atol=0
        )
