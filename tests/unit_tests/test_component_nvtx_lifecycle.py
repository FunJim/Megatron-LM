# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Direct CPU checks of capture and autograd failure boundaries."""

import ast
import importlib.util
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import test_component_nvtx as fixtures


def load_mhc(helper):
    path = fixtures.ROOT / "megatron/core/transformer/mhc_chunked.py"
    spec = importlib.util.spec_from_file_location("mhc_chunked_lifecycle_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"megatron.core.component_nvtx": helper}):
        spec.loader.exec_module(module)
    return module


class LifecycleTests(unittest.TestCase):
    setUp = fixtures.ComponentNvtxTests.setUp

    def tearDown(self):
        fixtures.ComponentNvtxTests.tearDown(self)
        self.assertEqual(self.nvtx._node_contexts, {})
        self.assertEqual(self.nvtx._task_boundaries, {})
        self.assertIsNone(self.nvtx._backward_tasks.get())
        self.assertEqual(self.nvtx._execution.get().phase, "forward")

    def test_capabilities_and_managed_import_do_not_initialize_cuda(self):
        with patch.dict(os.environ, {"MCORE_COMPONENT_NVTX_CAPTURE_MANAGED": "1"}):
            helper = fixtures.load_nvtx()
        self.assertFalse(helper.capabilities()["active"])
        self.assertTrue(helper.capabilities()["capture_managed"])
        self.assertEqual(helper.capabilities()["api_version"], 1)
        self.assertTrue(helper.capabilities()["backward_hooks"])
        self.assertFalse(helper.capabilities()["cuda_graph_replay_qualified"])
        self.assertFalse(torch.cuda.is_initialized())
        helper.set_capture_context(True, 17)
        self.assertTrue(helper.capabilities()["active"])

    def test_capture_managed_boolean_validation_is_import_time(self):
        key = "MCORE_COMPONENT_NVTX_CAPTURE_MANAGED"
        for value in ("1", "true", "yes", "on", "0", "false", "no", "off"):
            managed = value in ("1", "true", "yes", "on")
            with self.subTest(value=value):
                with patch.dict(os.environ, {key: f" {value.upper()} "}):
                    helper = fixtures.load_nvtx()
                self.assertIs(helper.CAPTURE_MANAGED, managed)
                self.assertIs(helper.capabilities()["capture_managed"], managed)
                self.assertIs(helper.capabilities()["active"], not managed)
                with patch.dict(os.environ, {key: "0" if managed else "1"}):
                    self.assertIs(helper.capabilities()["capture_managed"], managed)
        for enabled in (False, True):
            for value in ("", "2", "invalid"):
                with self.subTest(enabled=enabled, invalid=value):
                    with patch.dict(os.environ, {key: value}):
                        with self.assertRaisesRegex(ValueError, key):
                            fixtures.load_nvtx(enabled)
        self.assertFalse(torch.cuda.is_initialized())

    def test_backward_exception_closes_ranges_and_restores_context(self):
        helper = self.nvtx

        class Failing(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                raise RuntimeError("backward fixture")

        fn = helper.component("failure")(Failing.apply)
        output = fn(torch.ones(2, requires_grad=True))
        with self.assertRaisesRegex(RuntimeError, "backward fixture"):
            with helper.backward_scope(output):
                output.sum().backward()
        self.assertEqual(helper._node_contexts, {})
        self.assertEqual(helper.capture_execution().phase, "forward")
        self.assertEqual(self.recorder.open, {})
        helper.component("next")(lambda x: x.square())(
            torch.ones(2, requires_grad=True)
        ).sum().backward()

    def test_custom_backward_inner_ranges_receive_original_metadata(self):
        helper = self.nvtx

        class Native(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                with helper.forward_range("native.inner"):
                    return gradient.clone()

        helper.set_capture_context(True, 27)
        with helper.scheduled_microbatch(9):
            output = helper.component("native")(Native.apply)(torch.ones(2, requires_grad=True))
        with helper.backward_scope(output):
            output.sum().backward()
        names = [name for kind, name, _ in self.recorder.events if kind == "push"]
        self.assertTrue(any("comp:native.inner:" in name and "phase=backward" in name
                            and "step=27" in name and "mb=9" in name for name in names))
        self.assertEqual(helper._node_contexts, {})
        self.assertEqual(helper.capture_execution().phase, "forward")

    def test_stop_capture_during_retained_graph_is_safe(self):
        fn = self.nvtx.component("retained")(lambda x: (x.sin(), x.cos()))
        x = torch.ones(2, requires_grad=True)
        left, right = fn(x)
        left.sum().backward(retain_graph=True)
        self.nvtx.set_capture_context(False)
        before = len(self.recorder.events)
        right.sum().backward()
        self.assertEqual(len(self.recorder.events), before)
        self.assertEqual(self.nvtx._node_contexts, {})

    def test_independent_graphs_close_on_concurrent_backward_threads(self):
        fn = self.nvtx.component("concurrent")(lambda x: x.sin() * x)
        outputs = [fn(torch.ones(4, requires_grad=True)) for _ in range(2)]
        errors = []

        def run(output):
            try:
                with self.nvtx.backward_scope(output):
                    output.sum().backward()
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=run, args=(output,)) for output in outputs]
        for worker in threads:
            worker.start()
        for worker in threads:
            worker.join(timeout=10)
        self.assertFalse(any(worker.is_alive() for worker in threads))
        self.assertEqual(errors, [])
        self.assertEqual(self.nvtx._node_contexts, {})

    def test_actual_stable_topk_keeps_ties_nonfinite_and_input_immutable(self):
        path = fixtures.ROOT / "megatron/core/transformer/experimental_attention_variant/csa2.py"
        tree = ast.parse(path.read_text())
        method = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                      for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "select_indices")
        scores = torch.tensor([[2., 2., 1., float("-inf")],
                               [float("nan"), 3., float("inf"), -1.]])
        original = scores.clone()
        outputs = []
        for enabled in (False, True):
            helper = self.nvtx if enabled else fixtures.load_nvtx(False)
            namespace = {"torch": torch, "host_scope": helper.host_scope}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
            holder = SimpleNamespace(config=SimpleNamespace(csa_query_chunk_size=1), topk=3)
            outputs.append(namespace["select_indices"](holder, scores))
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        torch.testing.assert_close(scores, original, equal_nan=True, atol=0, rtol=0)
        self.assertTrue(any("comp:indexer.selection.stable_topk:" in name
                            for kind, name, _ in self.recorder.events if kind == "push"))

    def test_native_mhc_chunk_projection_sinkhorn_and_backward_recompute(self):
        helper = self.nvtx
        namespace = {"torch": torch, "F": torch.nn.functional, "os": os,
                     "once_differentiable": torch.autograd.function.once_differentiable,
                     **{name: getattr(helper, name) for name in
                        ("capture_execution", "execution_phase", "forward_range", "backward_scope")}}
        path = fixtures.ROOT / "megatron/core/transformer/mhc_chunked.py"
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"), namespace)
        path = fixtures.ROOT / "megatron/core/transformer/hyper_connection.py"
        tree = ast.parse(path.read_text())
        sinkhorn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "_sinkhorn_iterations_native")
        method = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                      for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_compute_single_pass_mappings")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, sinkhorn, method], type_ignores=[])), str(path), "exec"), namespace)
        torch.manual_seed(27)
        holder = SimpleNamespace(n=4, hidden_size=3, norm_eps=1e-20, compute_h_eps=1e-6,
                                 sinkhorn_eps=1e-6, sinkhorn_iterations=20,
                                 _proj_rms_compute_h_op=None,
                                 mapping_proj=torch.nn.Linear(12, 24, bias=False),
                                 alpha_pre=torch.nn.Parameter(torch.ones(())),
                                 alpha_post=torch.nn.Parameter(torch.ones(())),
                                 alpha_res=torch.nn.Parameter(torch.ones(())),
                                 bias=torch.nn.Parameter(torch.zeros(24)))
        x = torch.randn(5, 2, 12, requires_grad=True)
        parameters = (x, holder.mapping_proj.weight, holder.alpha_pre,
                      holder.alpha_post, holder.alpha_res, holder.bias)
        results = []
        with patch.dict(os.environ, {"MCORE_MHC_TOKEN_CHUNK_SIZE": "3"}):
            for active in (False, True):
                helper.set_capture_context(active, 15)
                with helper.scheduled_microbatch(7):
                    fn = helper.component("mhc.mappings")(namespace["_compute_single_pass_mappings"])
                    output = fn(holder, x)
                with helper.backward_scope(output):
                    grads = torch.autograd.grad(sum(value.sum() for value in output), parameters)
                results.append((output, grads))
        for left,right in zip(results[0][0] + results[0][1], results[1][0] + results[1][1]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        names = [name for kind,name,_ in self.recorder.events if kind == "push"]
        for component in ("mhc.native_projection_rms", "mhc.native_sinkhorn"):
            self.assertTrue(any(f"comp:{component}:" in name and "phase=forward" in name for name in names))
            self.assertTrue(any(f"comp:{component}:" in name and "phase=backward_recompute" in name
                                and "mb=7" in name and "step=15" in name for name in names))

    def test_stopped_capture_prehook_failure_does_not_poison_next_step(self):
        helper = self.nvtx

        class Failing(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                raise RuntimeError("stopped fixture")

        helper.set_capture_context(True, 31)
        output = helper.component("stopped")(Failing.apply)(torch.ones(2, requires_grad=True))
        helper.set_capture_context(False, 32)
        before = len(self.recorder.events)
        with self.assertRaisesRegex(RuntimeError, "stopped fixture"):
            with helper.backward_scope(output):
                output.sum().backward()
        self.assertEqual(len(self.recorder.events), before)
        self.assertEqual(helper._node_contexts, {})
        self.assertEqual(helper._execution.get().phase, "forward")
        helper.set_capture_context(True, 33)
        x = torch.tensor([2., 3.], requires_grad=True)
        output = helper.component("fresh")(torch.square)(x)
        with helper.backward_scope(output):
            output.sum().backward()
        torch.testing.assert_close(x.grad, 2 * x, atol=0, rtol=0)
        names = [name for kind, name, _ in self.recorder.events if kind == "push"]
        self.assertTrue(any("comp:fresh:" in name and "phase=forward" in name
                            and "step=33" in name for name in names))

    def test_stop_inside_backward_does_not_leave_thread_context(self):
        helper = self.nvtx

        class Stop(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                helper.set_capture_context(False, 42)
                return gradient

        helper.set_capture_context(True, 41)
        x = torch.ones(2, requires_grad=True)
        helper.component("stop_inside")(Stop.apply)(x).sum().backward()
        self.assertEqual(helper._node_contexts, {})
        self.assertEqual(helper._execution.get().phase, "forward")
        helper.set_capture_context(True, 43)
        with helper.forward_range("after_stop"):
            pass
        self.assertIn("phase=forward:step=43", self.recorder.events[-2][1])
        torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)

    def test_reentrant_exception_closes_child_tasks_and_worker_context(self):
        helper = self.nvtx
        observed = []
        caller_thread = threading.get_ident()

        class Reentrant(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, depth, fail, measured):
                ctx.save_for_backward(x)
                ctx.depth, ctx.fail, ctx.measured = depth, fail, measured
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                if ctx.depth == 0:
                    observed.append((threading.get_ident(), helper.capture_execution(),
                                     helper._execution.get()))
                    if ctx.fail:
                        raise RuntimeError("reentrant fixture")
                    return gradient, None, None, None
                with torch.enable_grad():
                    x = ctx.saved_tensors[0].detach().requires_grad_()
                    fn = helper.component("reentrant_child")(Reentrant.apply) if ctx.measured else Reentrant.apply
                    output = fn(x, ctx.depth - 1, ctx.fail, ctx.measured)
                    grad = torch.autograd.grad(output, x, gradient)[0]
                return grad, None, None, None

        helper.set_capture_context(True, 51)
        with helper.scheduled_microbatch(4):
            output = helper.component("reentrant_outer")(Reentrant.apply)(
                torch.ones(2, requires_grad=True), 65, True, True)
        with self.assertRaisesRegex(RuntimeError, "reentrant fixture"):
            with helper.backward_scope(output):
                output.sum().backward()
        self.assertNotEqual(observed[0][0], caller_thread)
        failure_state = observed[0][1]
        self.assertEqual((failure_state.phase, failure_state.step, failure_state.mb),
                         ("backward", 51, 4))
        self.assertEqual(self.recorder.open, {})
        self.assertEqual(helper._node_contexts, {})
        helper.set_capture_context(True, 52)
        x = torch.ones(2, requires_grad=True)
        Reentrant.apply(x, 65, False, False).sum().backward()
        self.assertEqual(len(observed), 2)
        worker_thread, state, raw_state = observed[1]
        self.assertNotEqual(worker_thread, caller_thread)
        self.assertEqual((state.phase, state.step, state.mb), ("forward", 52, None))
        self.assertEqual(raw_state.phase, "forward")
        torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)

    def test_nested_grad_restores_outer_metadata_and_unused_retained_inputs(self):
        helper = self.nvtx
        observed = []

        class Nested(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, gradient):
                observed.append(helper.capture_execution())
                with torch.enable_grad():
                    x = ctx.saved_tensors[0].detach().requires_grad_()
                    unused = torch.ones_like(x, requires_grad=True)
                    outputs = helper.component("nested_inner")(lambda v: (v.sin(), v.cos()))(x)
                    for retain in (True, False):
                        with helper.backward_scope(outputs):
                            grads = torch.autograd.grad(outputs[0], (x, unused), gradient,
                                                        retain_graph=retain, allow_unused=True)
                        self.assertIsNone(grads[1])
                observed.append(helper.capture_execution())
                return grads[0]

        helper.set_capture_context(True, 61)
        x = torch.linspace(-1, 1, 5, requires_grad=True)
        with helper.scheduled_microbatch(8):
            output = helper.component("nested_outer")(Nested.apply)(x)
        with helper.backward_scope(output):
            output.sum().backward()
        torch.testing.assert_close(output, x.sin(), atol=0, rtol=0)
        torch.testing.assert_close(x.grad, x.cos(), atol=0, rtol=0)
        self.assertEqual([(state.phase, state.step, state.mb) for state in observed],
                         [("backward", 61, 8)] * 2)
        self.assertEqual(helper._node_contexts, {})
        self.assertEqual(helper.capture_execution().phase, "forward")
        starts = [name for kind, name, _ in self.recorder.events if kind == "start"]
        self.assertEqual(sum(name.startswith("comp:nested_inner.bwd:") for name in starts), 2)

    def test_mhc_internal_grad_exception_cleans_second_chunk_and_next_backward(self):
        helper = self.nvtx
        mhc = load_mhc(helper)
        recomputes = 0
        boundaries = []

        class Failing(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, fail):
                ctx.fail = fail
                ctx.parent_task = torch._C._current_graph_task_id()
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                boundaries.append((ctx.parent_task, torch._C._current_graph_task_id(),
                                   set(helper._backward_tasks.get())))
                if ctx.fail:
                    raise RuntimeError("mhc internal grad fixture")
                return gradient, None

        @helper.component("mhc.inner_failure")
        def function(x):
            nonlocal recomputes
            if torch.is_grad_enabled():
                recomputes += 1
            return (Failing.apply(x, recomputes == 2),)

        helper.set_capture_context(True, 71)
        output = helper.component("mhc.outer_failure")(
            lambda x: mhc.chunked_mhc(function, 2, 1, x)
        )(torch.ones(3, 2, 4, requires_grad=True))
        with self.assertRaisesRegex(RuntimeError, "mhc internal grad fixture"):
            with helper.backward_scope(output):
                output[0].sum().backward()
        self.assertEqual(recomputes, 2)
        self.assertEqual(len(boundaries), 2)
        for parent_task, child_task, task_ids in boundaries:
            self.assertNotEqual(parent_task, child_task)
            self.assertEqual(task_ids, {child_task})
        self.assertEqual(self.recorder.open, {})
        self.assertEqual(helper._node_contexts, {})
        self.assertEqual(helper.capture_execution().phase, "forward")
        helper.set_capture_context(True, 72)
        x = torch.linspace(-1, 1, 24).reshape(3, 2, 4).requires_grad_()
        output = mhc.chunked_mhc(helper.component("mhc.recovery")(
            lambda value: (value.square(),)), 2, 1, x)
        with helper.backward_scope(output):
            output[0].sum().backward()
        torch.testing.assert_close(x.grad, 2 * x, atol=0, rtol=0)
        self.assertEqual(helper._node_contexts, {})

    def test_actual_mhc_aggregate_bda_capture_toggle_is_bitwise_identical(self):
        helper = self.nvtx
        torch.manual_seed(93)
        values = (
            torch.randn(3, 2, 8), torch.randn(3, 2, 2),
            torch.randn(3, 2, 2, 2), torch.randn(3, 2, 4), torch.randn(4),
        )
        results = []
        for active, step in ((None, 80), (False, 81), (True, 82), (False, 83), (True, 84)):
            current = fixtures.load_nvtx(False) if active is None else helper
            current.set_capture_context(bool(active), step)
            mhc = load_mhc(current)
            hidden, mix, residual_mix, branch, bias = [v.clone().requires_grad_() for v in values]
            with current.scheduled_microbatch(6):
                aggregate = mhc.chunked_mhc(current.component("mhc.aggregate")(mhc.aggregate_chunk),
                                            2, 2, hidden, mix)[0]
                merged = mhc.chunked_mhc(current.component("mhc.bda")(mhc.bda_chunk),
                                        2, 4, residual_mix, hidden, mix, branch, bias)[0]
            with current.backward_scope((aggregate, merged)):
                grads = torch.autograd.grad(aggregate.square().sum() + merged.square().sum(),
                                           (hidden, mix, residual_mix, branch, bias))
            results.append((aggregate, merged, *grads))
            self.assertEqual(current._node_contexts, {})
            self.assertEqual(current._task_boundaries, {})
            self.assertEqual(self.recorder.open, {})
            self.assertEqual(current.capture_execution().phase, "forward")
        for result in results[1:]:
            for expected, actual in zip(results[0], result):
                torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        backward_names = [name for kind, name, _ in self.recorder.events if kind == "start"]
        self.assertTrue(any("step=82" in name and "mb=6" in name for name in backward_names))
        self.assertTrue(any("step=84" in name and "mb=6" in name for name in backward_names))
        self.assertFalse(any("step=81" in name or "step=83" in name for name in backward_names))

    def test_failed_graph_does_not_close_concurrent_live_graph(self):
        helper = self.nvtx
        good_entered = threading.Event()
        failed_cleaned = threading.Event()
        observed = []
        errors = []
        surviving_names = []

        class Good(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                observed.append(helper.capture_execution())
                good_entered.set()
                if not failed_cleaned.wait(timeout=5):
                    raise RuntimeError("concurrent cleanup timed out")
                observed.append(helper.capture_execution())
                return gradient

        class Failing(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gradient):
                if not good_entered.wait(timeout=5):
                    raise RuntimeError("concurrent entry timed out")
                raise RuntimeError("concurrent fixture")

        helper.set_capture_context(True, 91)
        good_x = torch.ones(2, requires_grad=True)
        with helper.scheduled_microbatch(2):
            good = helper.component("concurrent_good")(Good.apply)(good_x)
        with helper.scheduled_microbatch(3):
            bad = helper.component("concurrent_bad")(Failing.apply)(torch.ones(2, requires_grad=True))

        def run_good():
            try:
                with helper.backward_scope(good):
                    good.sum().backward()
            except Exception as error:
                errors.append(error)

        def run_bad():
            try:
                try:
                    with helper.backward_scope(bad):
                        bad.sum().backward()
                except RuntimeError as error:
                    if str(error) != "concurrent fixture":
                        raise
                else:
                    raise AssertionError("expected failing engine invocation")
                surviving_names.extend(self.recorder.open.values())
            except Exception as error:
                errors.append(error)
            finally:
                failed_cleaned.set()

        workers = [threading.Thread(target=run_good), threading.Thread(target=run_bad)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=10)
        self.assertFalse(any(worker.is_alive() for worker in workers))
        self.assertEqual(errors, [])
        self.assertEqual(len(surviving_names), 1)
        self.assertTrue(surviving_names[0].startswith("comp:concurrent_good.bwd:"))
        self.assertEqual([(state.phase, state.step, state.mb) for state in observed],
                         [("backward", 91, 2)] * 2)
        torch.testing.assert_close(good_x.grad, torch.ones_like(good_x), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
