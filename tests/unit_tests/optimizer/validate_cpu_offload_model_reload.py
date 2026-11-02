# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Validate mixed-dtype model reload and native dense/expert CPU SGD on one GPU."""

import argparse
import gc
import inspect
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig


class MixedModel(torch.nn.Module):
    """BF16 dense/expert projections with native FP32 normalization parameters."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        size = config.hidden_size
        self.norm = torch.nn.LayerNorm(size, device='cuda', dtype=torch.float32)
        self.dense = torch.nn.Linear(size, size, bias=False, device='cuda', dtype=torch.bfloat16)
        self.expert = torch.nn.Linear(size, size, bias=False, device='cuda', dtype=torch.bfloat16)
        self.expert.weight.allreduce = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.expert(self.dense(self.norm(inputs.float()).to(inputs.dtype)))


def validate() -> dict:
    """Reload BF16/FP32 weights after optimizer creation, then check three native updates."""
    assert torch.cuda.is_available(), 'Run this script inside the allocated GPU environment'
    assert not torch.distributed.is_initialized(), 'Run in a separate subprocess'
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    torch.manual_seed(31)
    old_env = os.environ.get('MCORE_CPU_STREAMING_SGD')
    os.environ['MCORE_CPU_STREAMING_SGD'] = '1'
    initialized_parallel = False
    try:
        with tempfile.TemporaryDirectory(prefix='native-ddp-sgd-') as rendezvous:
            torch.distributed.init_process_group(
                'nccl',
                init_method=Path(rendezvous, 'store').as_uri(),
                rank=0,
                world_size=1,
                timeout=timedelta(minutes=3),
            )
            parallel_state.initialize_model_parallel(create_gloo_process_groups=False)
            initialized_parallel = True
            groups = ProcessGroupCollection.use_mpu_process_groups()
            config = TransformerConfig(
                num_layers=1,
                num_attention_heads=1,
                hidden_size=1024,
                bf16=True,
                params_dtype=torch.bfloat16,
            )
            model = MixedModel(config)
            before = {
                name: param.detach().cpu().clone() for name, param in model.named_parameters()
            }
            ddp = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    use_distributed_optimizer=True,
                    ddp_cpu_staging=True,
                    grad_reduce_in_fp32=False,
                    overlap_grad_reduce=False,
                    overlap_param_gather=False,
                ),
                model,
                pg_collection=groups,
            )
            assert ddp.ddp_config.ddp_cpu_staging
            for name, param in model.named_parameters():
                assert param.is_cuda
                torch.testing.assert_close(param.detach().cpu(), before[name], rtol=0, atol=0)
            assert model.norm.weight.dtype == torch.float32
            assert model.norm.bias.dtype == torch.float32
            assert model.dense.weight.dtype == torch.bfloat16
            assert model.expert.weight.dtype == torch.bfloat16
            assert not model.expert.weight.allreduce
            assert len(ddp.buffers) == 2, len(ddp.buffers)
            assert len(ddp.expert_parallel_buffers) == 1, len(ddp.expert_parallel_buffers)
            buffer_counts = {
                'dense': sum(len(buffer.params) for buffer in ddp.buffers),
                'expert': sum(len(buffer.params) for buffer in ddp.expert_parallel_buffers),
            }
            assert buffer_counts == {'dense': 3, 'expert': 1}, buffer_counts

            optimizer_config = OptimizerConfig(
                optimizer='sgd',
                lr=1e-3,
                sgd_momentum=0.0,
                weight_decay=0.0,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                optimizer_cpu_offload=True,
                optimizer_offload_fraction=1.0,
                use_precision_aware_optimizer=True,
                main_grads_dtype=torch.bfloat16,
                main_params_dtype=torch.float32,
                cpu_streaming_sgd=True,
                overlap_cpu_optimizer_d2h_h2d=False,
                clip_grad=1.0,
            )
            optimizer = get_megatron_optimizer(
                optimizer_config, [ddp], use_gloo_process_groups=False, pg_collection=groups
            )
            children = optimizer.chained_optimizers
            assert len(children) == 2, f'Expected dense/expert optimizer children: {len(children)}'
            param_names = {param: name for name, param in model.named_parameters()}
            model_shards = []
            child_kinds = []
            for child in children:
                assert isinstance(child, DistributedOptimizer)
                hdo = child.optimizer
                assert isinstance(hdo, HybridDeviceOptimizer)
                assert hdo.cpu_streaming_sgd and hdo.defaults['cpu_streaming_sgd']
                assert hdo.offload_fraction == 1.0 and hdo.gpu_optimizer is None
                assert all(isinstance(sub, torch.optim.SGD) for sub in hdo.cpu_optimizers)
                child_kinds.append(
                    'expert' if any(g['is_expert_parallel'] for g in hdo.param_groups) else 'dense'
                )
                for model_groups, shard_groups in (
                    (child.model_float16_groups, child.shard_float16_groups),
                    (child.model_fp32_groups, child.shard_fp32_groups),
                ):
                    for model_params, shard_params in zip(model_groups, shard_groups):
                        assert len(model_params) == len(shard_params)
                        for model_param, shard in zip(model_params, shard_params):
                            master = hdo.gpu_params_map_cpu_copy[shard]
                            assert master.device.type == 'cpu' and master.dtype == torch.float32
                            model_shards.append((model_param, shard, master))
            assert sorted(child_kinds) == ['dense', 'expert'], child_kinds
            assert len(model_shards) == len(param_names) == 4
            assert {param_names[item[0]] for item in model_shards} == set(before)

            # Loading weights after optimizer construction mirrors model-only DCP resume.
            loaded = {name: value.clone() for name, value in before.items()}
            loaded['norm.weight'].fill_(0.75)
            loaded['norm.bias'].fill_(0.25)
            loaded['dense.weight'].mul_(0.5)
            loaded['expert.weight'].mul_(0.75)
            model.load_state_dict(loaded)
            optimizer.reload_model_params()
            for model_param, shard, master in model_shards:
                name = param_names[model_param]
                expected_loaded = loaded[name].float().view(-1)
                actual_loaded = master.detach().view(-1)
                assert torch.equal(actual_loaded, expected_loaded), (
                    f'Stale CPU master after reload: {name}; '
                    f'max_error={float((actual_loaded - expected_loaded).abs().max())}'
                )
            before = loaded

            inputs = torch.randn(4, 1024, device='cuda', dtype=torch.bfloat16)
            target = torch.randn(4, 1024, device='cuda', dtype=torch.float32)
            updates = []
            scratch_ptrs = {}
            torch.cuda.reset_peak_memory_stats()
            for step in range(1, 4):
                ddp.zero_grad_buffer()
                optimizer.zero_grad()
                loss = (ddp(inputs).float() - target).square().mean()
                loss.backward()
                ddp.finish_grad_sync()
                assert math.isfinite(float(loss.detach()))
                for model_param, shard, master in model_shards:
                    assert torch.isfinite(model_param.main_grad).all()
                    assert model_param.numel() == shard.numel() == master.numel()
                reference_masters = {
                    id(master): master.detach().clone() for _, _, master in model_shards
                }
                successful, grad_norm, _ = optimizer.step()
                torch.cuda.synchronize()
                assert successful, f'Optimizer update failed at step {step}'
                assert grad_norm is not None and math.isfinite(float(grad_norm)), grad_norm

                # Native clipping has now updated the shard gradients. Use the same
                # gradients with independent torch SGD to verify every CPU/GPU value.
                for model_param, shard, master in model_shards:
                    reference = torch.nn.Parameter(reference_masters[id(master)])
                    reference.grad = shard.decoupled_grad.detach().float().cpu()
                    torch.optim.SGD(
                        [reference],
                        lr=optimizer_config.lr,
                        momentum=0.0,
                        weight_decay=0.0,
                        foreach=False,
                        fused=False,
                    ).step()
                    torch.testing.assert_close(master, reference, rtol=0, atol=0)
                    torch.testing.assert_close(
                        model_param.detach().cpu().view(-1),
                        reference.to(model_param.dtype).view(-1),
                        rtol=0,
                        atol=0,
                    )

                streamed = []
                for kind, child in zip(child_kinds, children):
                    hdo = child.optimizer
                    expected = sum(
                        p.numel()
                        for g in hdo.param_groups
                        for p in g['params']
                        if getattr(p, 'decoupled_grad', p.grad) is not None
                    )
                    assert expected == sum(p.numel() for g in hdo.param_groups for p in g['params'])
                    assert expected > 0 and hdo.streaming_updated_elements == expected
                    assert not hdo.cpu_copy_map_grad
                    assert hdo.streaming_cpu_gradient_bytes == 64 * 1024 * 1024
                    assert hdo._streaming_sgd_scratch.is_pinned()
                    pointer = hdo._streaming_sgd_scratch.data_ptr()
                    if kind in scratch_ptrs:
                        assert pointer == scratch_ptrs[kind]
                    scratch_ptrs[kind] = pointer
                    streamed.append(
                        {
                            'kind': kind,
                            'updated_elements': expected,
                            'cpu_gradient_bytes': hdo.streaming_cpu_gradient_bytes,
                        }
                    )
                assert sum(item['updated_elements'] for item in streamed) == sum(
                    p.numel() for p in model.parameters()
                )
                for model_param, shard, master in model_shards:
                    assert torch.isfinite(master).all()
                    torch.testing.assert_close(
                        model_param.detach().cpu().view(-1),
                        master.to(model_param.dtype).view(-1),
                        rtol=0,
                        atol=0,
                    )
                updates.append(
                    {
                        'step': step,
                        'loss': float(loss.detach()),
                        'grad_norm': float(grad_norm),
                        'successful': bool(successful),
                        'streaming_sgd': streamed,
                    }
                )
                print(json.dumps({'native_ddp_sgd_step': updates[-1]}), flush=True)

            changed = [
                name
                for name, param in model.named_parameters()
                if not torch.equal(before[name], param.detach().cpu())
            ]
            assert 'dense.weight' in changed and 'expert.weight' in changed, changed
            assert any(name.startswith('norm.') for name in changed), changed
            assert all(torch.isfinite(param).all() for param in model.parameters())
            result = {
                'status': 'passed',
                'passed': True,
                'checked_utc': datetime.now(timezone.utc).isoformat(),
                'environment': sys.prefix,
                'torch': torch.__version__,
                'device': torch.cuda.get_device_name(),
                'world_size': 1,
                'optimizer_children': child_kinds,
                'completed_steps': 3,
                'ddp_cpu_staging': ddp.ddp_config.ddp_cpu_staging,
                'cpu_streaming_sgd': optimizer_config.cpu_streaming_sgd,
                'optimizer_offload_fraction': optimizer_config.optimizer_offload_fraction,
                'main_grads_dtype': str(optimizer_config.main_grads_dtype),
                'clip_grad': optimizer_config.clip_grad,
                'changed_parameters': changed,
                'parameter_dtypes': {
                    name: str(param.dtype) for name, param in model.named_parameters()
                },
                'values_preserved_at_init': True,
                'loaded_mixed_dtype_masters_refreshed': True,
                'all_three_updates_match_torch_sgd': True,
                'ddp_buffer_counts': {
                    'dense': len(ddp.buffers),
                    'expert': len(ddp.expert_parallel_buffers),
                },
                'ddp_buffer_parameter_counts': buffer_counts,
                'model_parameter_elements': sum(p.numel() for p in model.parameters()),
                'updates': updates,
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                'sources': {
                    'ddp': inspect.getsourcefile(DistributedDataParallel),
                    'factory': inspect.getsourcefile(get_megatron_optimizer),
                    'hdo': inspect.getsourcefile(HybridDeviceOptimizer),
                    'config': inspect.getsourcefile(OptimizerConfig),
                },
                'scope': 'Single GPU native DDP/dense-expert optimizer integration; no multi-node acceptance',
            }
            del optimizer, children, model_shards, ddp, model, inputs, target, loss
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            return result
    finally:
        if initialized_parallel:
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if old_env is None:
            os.environ.pop('MCORE_CPU_STREAMING_SGD', None)
        else:
            os.environ['MCORE_CPU_STREAMING_SGD'] = old_env


def main() -> None:
    """Emit JSON evidence after successful validation; fail the process on any assertion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate()
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='', flush=True)


if __name__ == '__main__':
    main()
