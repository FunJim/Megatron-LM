# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate installed native CPU streaming SGD on one GPU without pytest.

Run with the newly built environment's Python and an allocated GPU:
    python validate_cpu_streaming_sgd.py --output streaming-sgd-validation.json
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.optim import SGD, AdamW

from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig


def validate() -> dict:
    """Check real CUDA transfers, FP32 master updates, scratch reuse and default Adam."""
    assert torch.cuda.is_available(), 'An allocated CUDA GPU is required'
    torch.manual_seed(1234)
    old_env = os.environ.get('MCORE_CPU_STREAMING_SGD')
    os.environ['MCORE_CPU_STREAMING_SGD'] = '1'
    try:
        config = OptimizerConfig(
            optimizer='sgd',
            sgd_momentum=0.0,
            weight_decay=0.0,
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=1.0,
            use_precision_aware_optimizer=True,
            use_distributed_optimizer=True,
            bf16=True,
            main_grads_dtype=torch.bfloat16,
        )
        assert config.cpu_streaming_sgd
        assert not OptimizerConfig(cpu_streaming_sgd=False).cpu_streaming_sgd
    finally:
        if old_env is None:
            os.environ.pop('MCORE_CPU_STREAMING_SGD')
        else:
            os.environ['MCORE_CPU_STREAMING_SGD'] = old_env

    # A small linear projection whose BF16 weight crosses the 64 MiB staging boundary.
    model = torch.nn.Linear(4096, 4097, bias=False, device='cuda', dtype=torch.bfloat16)
    scale = torch.nn.Parameter(torch.ones(4097, device='cuda', dtype=torch.float32))
    skipped = torch.nn.Parameter(torch.ones(17, device='cuda', dtype=torch.bfloat16))
    params = [model.weight, scale, skipped]
    masters = [torch.nn.Parameter(param.detach().float().cpu()) for param in params]
    groups = [{'params': params[:1], 'lr': 1e-3}, {'params': params[1:], 'lr': 2e-3}]
    reference = SGD(
        [{'params': masters[:1], 'lr': 1e-3}, {'params': masters[1:], 'lr': 2e-3}],
        momentum=0.0,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )
    optimizer = HybridDeviceOptimizer(
        groups,
        offload_fraction=config.optimizer_offload_fraction,
        cpu_optimizer_cls=SGD,
        gpu_optimizer_cls=SGD,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=False,
        cpu_streaming_sgd=config.cpu_streaming_sgd,
        momentum=config.sgd_momentum,
        weight_decay=config.weight_decay,
    )
    assert optimizer.defaults['cpu_streaming_sgd']
    assert optimizer.gpu_optimizer is None
    expected = model.weight.numel() + scale.numel()
    scratch_ptr = None
    updates = []
    inputs = torch.randn(2, 4096, device='cuda', dtype=torch.bfloat16)
    skipped_before = skipped.detach().clone()
    for step in range(1, 4):
        optimizer.zero_grad()
        loss = (model(inputs).float() * scale).square().mean()
        loss.backward()
        # Megatron precision-aware optimizers pass gradients in decoupled_grad.
        for param, master in zip(params, masters):
            param.decoupled_grad = param.grad
            param.grad = None
            gradient = param.decoupled_grad
            master.grad = None if gradient is None else gradient.detach().float().cpu()
        if step == 2:
            optimizer.param_groups[0]['lr'] = 5e-4
            reference.param_groups[0]['lr'] = 5e-4
        reference.step()
        optimizer.step()
        torch.cuda.synchronize()
        for param, master in zip(params, masters):
            native_master = optimizer.gpu_params_map_cpu_copy[param]
            torch.testing.assert_close(native_master, master, rtol=0, atol=0)
            torch.testing.assert_close(param.detach().cpu(), master.to(param.dtype), rtol=0, atol=0)
        assert not optimizer.cpu_copy_map_grad
        assert optimizer.streaming_updated_elements == expected
        assert optimizer.streaming_cpu_gradient_bytes == 64 * 1024 * 1024
        assert optimizer._streaming_sgd_scratch.is_pinned()
        if scratch_ptr is not None:
            assert optimizer._streaming_sgd_scratch.data_ptr() == scratch_ptr
        scratch_ptr = optimizer._streaming_sgd_scratch.data_ptr()
        updates.append(
            {
                'step': step,
                'loss': float(loss),
                'updated_elements': optimizer.streaming_updated_elements,
                'cpu_gradient_bytes': optimizer.streaming_cpu_gradient_bytes,
                'master_max_abs_error': 0.0,
            }
        )
    torch.testing.assert_close(skipped, skipped_before, rtol=0, atol=0)

    # A fresh HDO built from defaults retains the option (DistributedOptimizer does this).
    rebuilt = HybridDeviceOptimizer(groups, **optimizer.defaults)
    assert rebuilt.cpu_streaming_sgd
    assert rebuilt._streaming_sgd_scratch is None

    adam_param = torch.nn.Parameter(torch.ones(19, device='cuda'))
    adam_master = torch.nn.Parameter(adam_param.detach().cpu().clone())
    adam = HybridDeviceOptimizer(
        [adam_param],
        offload_fraction=1.0,
        cpu_optimizer_cls=AdamW,
        gpu_optimizer_cls=AdamW,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=False,
        lr=1e-3,
    )
    adam_reference = AdamW([adam_master], lr=1e-3)
    for _ in range(3):
        adam_param.grad = torch.ones_like(adam_param)
        adam_master.grad = torch.ones_like(adam_master)
        adam.step()
        adam_reference.step()
    torch.cuda.synchronize()
    torch.testing.assert_close(adam_param.detach().cpu(), adam_master, rtol=0, atol=0)
    assert not adam.cpu_streaming_sgd
    assert adam._streaming_sgd_scratch is None
    assert adam.cpu_copy_map_grad
    return {
        'status': 'passed',
        'torch': torch.__version__,
        'device': torch.cuda.get_device_name(),
        'native_hdo_source': str(Path(__import__(HybridDeviceOptimizer.__module__, fromlist=['x']).__file__).resolve()),
        'config_reader': 'OptimizerConfig.cpu_streaming_sgd default_factory reads MCORE_CPU_STREAMING_SGD=1',
        'updates': updates,
        'adam_default_control': 'passed',
        'missing_gradient_control': 'passed',
        'defaults_reconstruction': 'passed',
    }


def main() -> None:
    """Write validation evidence to JSON, exiting nonzero on any failed assertion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate()
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
