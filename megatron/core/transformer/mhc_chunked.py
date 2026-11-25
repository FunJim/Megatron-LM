# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bound native single-pass mHC temporaries without changing FP32 arithmetic."""

import os
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

from megatron.core.component_nvtx import capture_execution, execution_phase, backward_scope


def token_chunk_size() -> int:
    """Read the opt-in maximum number of tokens per native mHC operation."""
    size = int(os.environ.get('MCORE_MHC_TOKEN_CHUNK_SIZE', '0'))
    if size < 0:
        raise ValueError('MCORE_MHC_TOKEN_CHUNK_SIZE must be nonnegative')
    return size


class _ChunkedMHC(torch.autograd.Function):
    """Save native inputs and recompute one token chunk at a time in backward."""

    @staticmethod
    def forward(ctx, function, chunk_size, token_args, *args):
        ctx.nvtx_execution = capture_execution()
        ctx.function = function
        ctx.chunk_size = chunk_size
        ctx.token_args = token_args
        ctx.tensor_positions = [i for i, value in enumerate(args) if value is not None]
        ctx.save_for_backward(*(args[i] for i in ctx.tensor_positions))
        ctx.arg_count = len(args)
        ctx.set_materialize_grads(False)
        shape = args[0].shape[:2]
        tokens = shape[0] * shape[1]
        flat = [
            value.reshape(tokens, *value.shape[2:]) if i < token_args else value
            for i, value in enumerate(args)
        ]
        result = None
        for start in range(0, tokens, chunk_size):
            end = min(start + chunk_size, tokens)
            values = function(*(v[start:end] if i < token_args else v for i, v in enumerate(flat)))
            if result is None:
                result = [v.new_empty((tokens, *v.shape[1:])) for v in values]
            for destination, value in zip(result, values):
                destination[start:end].copy_(value)
        return tuple(v.reshape(*shape, *v.shape[1:]) for v in result)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        args = [None] * ctx.arg_count
        for i, value in zip(ctx.tensor_positions, ctx.saved_tensors):
            args[i] = value
        needed = ctx.needs_input_grad[3:]
        tokens = args[0].shape[0] * args[0].shape[1]
        flat = [
            value.reshape(tokens, *value.shape[2:]) if i < ctx.token_args else value
            for i, value in enumerate(args)
        ]
        outputs = [None if v is None else v.reshape(tokens, *v.shape[2:]) for v in grad_outputs]
        gradients = [
            (
                torch.empty_like(v)
                if needed[i] and i < ctx.token_args
                else torch.zeros_like(v) if needed[i] else None
            )
            for i, v in enumerate(flat)
        ]
        for start in range(0, tokens, ctx.chunk_size):
            end = min(start + ctx.chunk_size, tokens)
            inputs = [
                (
                    None
                    if v is None
                    else (v[start:end] if i < ctx.token_args else v)
                    .detach()
                    .requires_grad_(needed[i])
                )
                for i, v in enumerate(flat)
            ]
            with torch.enable_grad():
                with execution_phase("backward_recompute", ctx.nvtx_execution):
                    values = ctx.function(*inputs)
                active_outputs = [
                    (v, g[start:end]) for v, g in zip(values, outputs) if g is not None
                ]
                active_inputs = [v for i, v in enumerate(inputs) if needed[i]]
                with backward_scope([v for v, _ in active_outputs]):
                    grads = torch.autograd.grad(
                        [v for v, _ in active_outputs],
                        active_inputs,
                        [g for _, g in active_outputs],
                        allow_unused=True,
                    )
            for i, grad in zip((i for i in range(len(args)) if needed[i]), grads):
                if i < ctx.token_args:
                    if grad is None:
                        gradients[i][start:end].zero_()
                    else:
                        gradients[i][start:end].copy_(grad)
                elif grad is not None:
                    gradients[i].add_(grad)
            del inputs, values, grads, active_outputs, active_inputs
        return (
            None,
            None,
            None,
            *(v.reshape_as(args[i]) if v is not None else None for i, v in enumerate(gradients)),
        )


def chunked_mhc(
    function: Callable[..., tuple[Tensor, ...]], chunk_size: int, token_args: int, *args
) -> tuple[Tensor, ...]:
    """Apply a deterministic native operation with bounded forward and backward temporaries."""
    return _ChunkedMHC.apply(function, chunk_size, token_args, *args)


def aggregate_chunk(hidden: Tensor, mix: Tensor) -> tuple[Tensor]:
    """Retain FP32 products and reduction before casting to the activation dtype."""
    streams = hidden.unflatten(-1, (mix.shape[-1], -1))
    return ((streams.float() * mix.float().unsqueeze(-1)).sum(-2).to(hidden.dtype),)


def bda_chunk(
    residual_mix: Tensor, residual: Tensor, post: Tensor, branch: Tensor, bias: Tensor | None
) -> tuple[Tensor]:
    """Compute the zero-dropout native residual merge with the original reduction order."""
    streams = residual.float().unflatten(-1, (post.shape[-1], -1))
    mixed = (residual_mix.float().unsqueeze(-1) * streams.unsqueeze(-2)).sum(-3)
    expanded = post.float().unsqueeze(-1) * branch.float().unsqueeze(-2)
    if bias is not None:
        expanded = expanded + post.float().unsqueeze(-1) * bias.float()
    return ((mixed + expanded).flatten(-2).to(branch.dtype),)
