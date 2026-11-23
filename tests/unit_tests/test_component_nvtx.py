# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU autograd checks; runnable directly without Megatron's GPU conftest."""

import ast
import importlib.util
import os
import sys
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]


def load_nvtx(enabled=True):
    name = "component_nvtx_test_" + str(enabled)
    with patch.dict(os.environ, {"MCORE_COMPONENT_NVTX": "1" if enabled else "0"}):
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "megatron/core/component_nvtx.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


class Recorder:
    def __init__(self):
        self.events = []
        self.open = {}
        self.stack = []
        self.counter = 0

    def range_push(self, name):
        self.events.append(("push", name, threading.get_ident()))
        self.stack.append(name)

    def range_pop(self):
        self.events.append(("pop", self.stack.pop(), threading.get_ident()))

    def range_start(self, name):
        self.counter += 1
        self.open[self.counter] = name
        self.events.append(("start", name, threading.get_ident()))
        return self.counter

    def range_end(self, handle):
        self.events.append(("end", self.open.pop(handle), threading.get_ident()))


class ComponentNvtxTests(unittest.TestCase):
    def setUp(self):
        self.nvtx = load_nvtx()
        self.recorder = Recorder()
        self.mock = patch.object(torch.cuda, "nvtx", self.recorder)
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.assertEqual(self.recorder.open, {})
        self.assertEqual(self.recorder.stack, [])

    def test_disabled_decorators_are_identity(self):
        module = load_nvtx(False)

        def fn(x):
            return x.sin()

        for decorator in (
            module.component("x"),
            module.layer_scope,
            module.microbatch_scope,
            module.host_scope("x"),
            module.checkpoint_forward,
        ):
            self.assertIs(decorator(fn), fn)
        with module.forward_range("x"):
            fn(torch.ones(2))
        self.assertEqual(self.recorder.events, [])

    def test_invalid_flag_is_rejected(self):
        with patch.dict(os.environ, {"MCORE_COMPONENT_NVTX": "invalid"}):
            spec = importlib.util.spec_from_file_location(
                "bad_nvtx", ROOT / "megatron/core/component_nvtx.py"
            )
            with self.assertRaisesRegex(ValueError, "MCORE_COMPONENT_NVTX"):
                spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_nested_components_preserve_outputs_gradients_and_rng(self):
        inner = self.nvtx.component("inner")(lambda x: x.sin())

        @self.nvtx.component("outer")
        def measured(x):
            return inner(x) * x

        x = torch.linspace(-1, 1, 32, requires_grad=True)
        rng = torch.get_rng_state().clone()
        expected = x.sin() * x
        grad = torch.autograd.grad(expected.sum(), x)[0]
        actual = measured(x)
        actual.sum().backward()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(x.grad, grad, atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        names = [name for kind, name, _ in self.recorder.events if kind == "start"]
        self.assertTrue(any(name.startswith("comp:inner.bwd:") for name in names))
        self.assertTrue(any(name.startswith("comp:outer.bwd:") for name in names))

    def test_unused_output_and_retained_graph_close_each_backward(self):
        fn = self.nvtx.component("partial")(lambda x: (x.sin(), x.cos()))
        x = torch.ones(4, requires_grad=True)
        used, unused = fn(x)
        used.sum().backward(retain_graph=True)
        self.assertEqual(self.recorder.open, {})
        used.sum().backward()
        self.assertEqual(sum(kind == "start" for kind, *_ in self.recorder.events), 2)
        self.assertIsNotNone(unused.grad_fn)

    def test_backward_runs_on_the_executing_thread(self):
        fn = self.nvtx.component("thread")(lambda x: x.sin())
        y = fn(torch.ones(4, requires_grad=True))
        main_thread = threading.get_ident()
        worker = threading.Thread(target=lambda: y.sum().backward())
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        backward_threads = {thread for kind, _, thread in self.recorder.events if kind == "start"}
        self.assertEqual(len(backward_threads), 1)
        self.assertNotIn(main_thread, backward_threads)

    def test_exception_closes_forward_and_restores_context(self):
        @self.nvtx.component("failure")
        def fn(x):
            raise ValueError("fixture")

        with self.assertRaisesRegex(ValueError, "fixture"):
            fn(torch.ones(4))
        self.assertEqual(self.nvtx._claimed.get(), ())

    def test_no_grad_is_forward_unless_checkpoint_marks_replay(self):
        fn = self.nvtx.component("test")(lambda x: x.sin())
        with torch.no_grad():
            fn(torch.ones(2))
        self.assertIn("phase=forward", self.recorder.events[0][1])

    def test_checkpoint_preserves_original_microbatch_and_layer_metadata(self):
        nvtx = self.nvtx

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(dsv4_version="v4.1")
                self.layer_number = 41
                self.current_microbatch = 7

            @nvtx.layer_scope
            @nvtx.component("attention")
            def forward(self, x):
                return x.sin()

        layer = Layer()
        groups = SimpleNamespace(
            **{key: SimpleNamespace(rank=lambda: 0) for key in ("pp", "cp", "dp")}
        )
        stack = SimpleNamespace(layers=[layer], pg_collection=groups)

        @nvtx.microbatch_scope
        def run(stack, x):
            return checkpoint(nvtx.checkpoint_forward(layer), x, use_reentrant=True)

        nvtx.set_capture_context(True, 12)
        x = torch.ones(4, requires_grad=True)
        y = run(stack, x)
        layer.current_microbatch = 8
        y.sum().backward()
        names = [name for kind, name, _ in self.recorder.events if kind == "push"]
        self.assertTrue(any("phase=forward" in name and "hf=20" in name for name in names))
        self.assertTrue(
            any(
                "phase=recompute" in name and "mb=7" in name and "step=12" in name for name in names
            )
        )
        self.assertFalse(any("mb=8" in name for name in names))

    def test_closed_capture_does_not_emit_components(self):
        fn = self.nvtx.component("test")(lambda x: x.sin())
        self.nvtx.set_capture_context(False)
        fn(torch.ones(4, requires_grad=True)).sum().backward()
        self.assertEqual(self.recorder.events, [])

    def test_host_scope_does_not_scan_or_hook_output_graph(self):
        fn = self.nvtx.host_scope("communication.pp.typed")(lambda value: value)
        value = torch.ones(4, requires_grad=True).sin()
        with patch.object(self.nvtx, "_track_backward", side_effect=AssertionError("host scan")):
            self.assertIs(fn(value), value)
        self.assertFalse(any(kind == "start" for kind, *_ in self.recorder.events))

    def test_actual_hybrid_checkpoint_entry_marks_recompute(self):
        path = ROOT / "megatron/core/models/hybrid/hybrid_stack_adapter.py"
        tree = ast.parse(path.read_text())
        node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "checkpointed_hybrid_forward"
        )
        namespace = {
            "torch": torch,
            "checkpoint_forward": self.nvtx.checkpoint_forward,
            "tensor_parallel": SimpleNamespace(
                checkpoint=lambda fn, distribute, *args: checkpoint(fn, *args, use_reentrant=True)
            ),
        }
        code = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                node,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
        config = SimpleNamespace(
            recompute_method="uniform",
            recompute_num_layers=1,
            fp8=None,
            fp4=None,
            quant_recipe=None,
            distribute_saved_activations=False,
        )

        class Context:
            cross_layer_state = None
            mhc_state = None

            def save_for_recompute(self):
                return (), lambda values: Context()

        layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        layers[0].layer_number = 1
        x = torch.ones(2, 4, requires_grad=True)
        fn = self.nvtx.component("fixture")(lambda layer, hidden, context: layer(hidden).tanh())
        out = namespace["checkpointed_hybrid_forward"](
            config,
            layers,
            x,
            Context(),
            tp_group=None,
            quantization_context=lambda *args: nullcontext(),
            layer_forward=fn,
        )
        out.sum().backward()
        names = [name for kind, name, _ in self.recorder.events if kind == "push"]
        self.assertTrue(any("phase=forward" in name for name in names))
        self.assertTrue(any("phase=recompute" in name for name in names))

    def test_real_megatron_checkpoint_keeps_numerics_with_instrumentation(self):
        path = ROOT / "megatron/core/tensor_parallel/random.py"
        tree = ast.parse(path.read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CheckpointFunction"
        )
        graph_module = SimpleNamespace(is_graph_capturing=lambda: False)
        namespace = {
            "torch": torch,
            "Any": object,
            "Callable": object,
            "_R": object,
            "_set_checkpointing": lambda: None,
            "_unset_checkpointing": lambda: None,
            "_get_all_rng_states": lambda: (None,),
            "_set_all_rng_states": lambda *args: None,
            "_fork_rng": nullcontext,
            "detach_variable": __import__(
                "torch.utils.checkpoint", fromlist=["detach_variable"]
            ).detach_variable,
        }
        code = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                cls,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
        inputs = [torch.linspace(-1, 1, 16, requires_grad=True) for _ in range(2)]
        with patch.dict(sys.modules, {"megatron.core.transformer.cuda_graphs": graph_module}):
            outputs = []
            for measured, x in zip((False, True), inputs):
                fn = lambda x: x.sin() * x
                if measured:
                    fn = self.nvtx.checkpoint_forward(self.nvtx.component("real_checkpoint")(fn))
                y = namespace["CheckpointFunction"].apply(fn, False, x)
                y.sum().backward()
                outputs.append(y)
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        torch.testing.assert_close(inputs[0].grad, inputs[1].grad, atol=0, rtol=0)

    def test_real_nvtx_frontend_accepts_forward_and_backward_ranges(self):
        self.mock.stop()
        fn = self.nvtx.component("real_nvtx")(lambda x: x.sin() * x)
        fn(torch.ones(4, requires_grad=True)).sum().backward()
        self.mock.start()


if __name__ == "__main__":
    unittest.main()
