# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in component attribution, adapted from NVIDIA Megatron-LM commit 27e1018dbc.

Decorators are identities unless MCORE_COMPONENT_NVTX is enabled before import.
Checkpoint callers explicitly label their first execution and subsequent replay;
gradient mode alone does not identify recomputation. Backward ranges use autograd
node hooks, on the executing thread, without changing tensors or gradients.
"""

import functools
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from threading import RLock

import torch

_value = os.environ.get("MCORE_COMPONENT_NVTX", "0").lower()
if _value not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
    raise ValueError(f"Invalid MCORE_COMPONENT_NVTX: {_value!r}")
ENABLED = _value in ("1", "true", "yes", "on")
_active = ENABLED
_step = None


@dataclass(frozen=True)
class _Execution:
    phase: str = "forward"
    step: int | None = None
    mb: int | None = None
    pp: int | None = None
    cp: int | None = None
    dp: int | None = None
    hf: int | None = None
    hybrid: int | None = None


_execution = ContextVar("component_nvtx_execution", default=_Execution())
_claimed = ContextVar("component_nvtx_claimed", default=())


def set_capture_context(active: bool, step: int | None = None) -> None:
    """Restrict enabled instrumentation to a profiler's capture window."""
    global _active, _step
    _active = ENABLED and active
    _step = step


def capture_execution() -> _Execution:
    """Snapshot host labels for a checkpoint or custom autograd function."""
    state = _execution.get()
    return replace(state, step=_step) if state.step is None else state


def _name(prefix: str, state: _Execution) -> str:
    fields = [f"phase={state.phase}"]
    fields.extend(
        f"{key}={value}"
        for key, value in vars(state).items()
        if key != "phase" and value is not None
    )
    return f"{prefix}:rank={os.environ.get('RANK', '0')}:" + ":".join(fields)


@contextmanager
def execution_phase(phase: str, state: _Execution | None = None) -> Iterator[None]:
    """Label execution using the checkpoint caller's saved host metadata."""
    token = _execution.set(replace(state or capture_execution(), phase=phase))
    try:
        yield
    finally:
        _execution.reset(token)


def _tensors(value: object) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _tensors(getattr(value, field.name))


class _BackwardRange:
    def __init__(self, prefix: str, nodes: set, state: _Execution) -> None:
        self.name = _name(prefix, replace(state, phase="backward"))
        self.count = len(nodes)
        self.tasks = {}
        self.lock = RLock()
        for node in nodes:
            node.register_prehook(self._pre)
            node.register_hook(self._post)

    def _close(self, task_id: int) -> None:
        with self.lock:
            task = self.tasks.pop(task_id, None)
            if task is not None and task[0] is not None:
                torch.cuda.nvtx.range_end(task[0])

    def _pre(self, grad_outputs):
        task_id = torch._C._current_graph_task_id()
        with self.lock:
            if task_id not in self.tasks:
                handle = torch.cuda.nvtx.range_start(self.name) if _active else None
                self.tasks[task_id] = [handle, 0]
                # Some tracked outputs may be unused. Close at engine completion
                # even when their nodes never execute; retain_graph can run again.
                torch.autograd.Variable._execution_engine.queue_callback(
                    lambda: self._close(task_id)
                )

    def _post(self, grad_inputs, grad_outputs):
        task_id = torch._C._current_graph_task_id()
        with self.lock:
            task = self.tasks[task_id]
            task[1] += 1
            if task[1] == self.count:
                self._close(task_id)


def _track_backward(
    prefix: str, outputs, first: int, claimed: set, state: _Execution, transparent: bool = False
) -> None:
    last = torch.autograd._get_sequence_nr()
    stack = [tensor.grad_fn for tensor in _tensors(outputs) if tensor.grad_fn is not None]
    seen, nodes = set(), set()
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if not first <= node._sequence_nr() < last:
            continue
        if transparent or node not in claimed:
            nodes.add(node)
        stack.extend(edge[0] for edge in node.next_functions)
    parents = _claimed.get()
    if not transparent and parents:
        parents[-1].update(claimed | nodes)
    if nodes:
        _BackwardRange(prefix, nodes, state)


def component(name: str) -> Callable:
    """Attribute a component's forward and owned backward nodes."""

    def decorate(fn):
        if not ENABLED:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _active:
                return fn(*args, **kwargs)
            state = capture_execution()
            track = torch.is_grad_enabled()
            first = torch.autograd._get_sequence_nr() if track else None
            claimed = set()
            token = _claimed.set((*_claimed.get(), claimed))
            torch.cuda.nvtx.range_push(_name(f"comp:{name}", state))
            try:
                outputs = fn(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
                _claimed.reset(token)
            if track:
                _track_backward(f"comp:{name}.bwd", outputs, first, claimed, state)
            return outputs

        return wrapper

    return decorate


def layer_scope(fn: Callable) -> Callable:
    """Tag the current Hybrid layer; V4.1 D/E pairs map to zero-based HF blocks."""
    if not ENABLED:
        return fn

    @functools.wraps(fn)
    def wrapper(layer, *args, **kwargs):
        if not _active:
            return fn(layer, *args, **kwargs)
        number = layer.layer_number
        hf = (number - 1) // 2 if getattr(layer.config, "dsv4_version", None) == "v4.1" else None
        state = replace(capture_execution(), hybrid=number, hf=hf)
        token = _execution.set(state)
        track = torch.is_grad_enabled()
        first = torch.autograd._get_sequence_nr() if track else None
        torch.cuda.nvtx.range_push(_name(f"layer:{number}", state))
        try:
            outputs = fn(layer, *args, **kwargs)
            if track:
                side_outputs = tuple(
                    kwargs[key] for key in ("mhc_state", "cross_layer_state") if key in kwargs
                )
                _track_backward(
                    f"layer:{number}.bwd", (outputs, side_outputs), first, set(), state, True
                )
            return outputs
        finally:
            torch.cuda.nvtx.range_pop()
            _execution.reset(token)

    return wrapper


def microbatch_scope(fn: Callable) -> Callable:
    """Save microbatch and explicit process-group coordinates for Hybrid forward."""
    if not ENABLED:
        return fn

    @functools.wraps(fn)
    def wrapper(stack, *args, **kwargs):
        if not _active:
            return fn(stack, *args, **kwargs)
        state = replace(
            capture_execution(),
            mb=getattr(stack.layers[0], "current_microbatch", None) if len(stack.layers) else None,
            **{
                key: getattr(stack.pg_collection, key).rank()
                for key in ("pp", "cp", "dp")
                if getattr(stack.pg_collection, key, None) is not None
            },
        )
        token = _execution.set(state)
        first = torch.autograd._get_sequence_nr() if torch.is_grad_enabled() else None
        torch.cuda.nvtx.range_push(_name("microbatch", state))
        try:
            outputs = fn(stack, *args, **kwargs)
            if first is not None:
                _track_backward("microbatch.bwd", outputs, first, set(), state, True)
            return outputs
        finally:
            torch.cuda.nvtx.range_pop()
            _execution.reset(token)

    return wrapper


def host_scope(name: str) -> Callable:
    """Mark communication or host work without scanning returned activation graphs."""

    def decorate(fn):
        if not ENABLED:
            return fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with forward_range(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def checkpoint_forward(fn: Callable) -> Callable:
    """Label the first checkpoint invocation as forward, later invocations as replay."""
    if not ENABLED:
        return fn
    state = capture_execution()
    first_call = True

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal first_call
        phase = state.phase if first_call else "recompute"
        first_call = False
        with execution_phase(phase, state):
            with forward_range("checkpoint.group"):
                return fn(*args, **kwargs)

    return wrapper


class forward_range:
    """Mark a coarse forward-only region, without per-operation events."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.active = False

    def __enter__(self):
        self.active = _active
        if self.active:
            torch.cuda.nvtx.range_push(_name(f"comp:{self.name}", capture_execution()))

    def __exit__(self, *exc):
        if self.active:
            torch.cuda.nvtx.range_pop()
