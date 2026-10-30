"""Small CPU numerical checks for the native V4.1 SFT changes.

Run directly with unittest to avoid the distributed GPU fixtures in unit_tests.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.core.optimizer import clip_grads
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.transformer.experimental_attention_variant.csa import (
    _unfused_compressed_sparse_attn,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2Indexer
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_pipeline import (
    CSA2PipelineBoundary,
    CSA2PipelinePayload,
)


class NativeSFTTest(unittest.TestCase):
    def test_pipeline_restore_uses_static_cp_group(self):
        boundary = CSA2PipelineBoundary(
            layer_offset=1,
            last_attention_layer=None,
            kv_source_layer=None,
            index_source_layer=None,
            candidate_source_layer=None,
            needs_indexer_k=False,
            compress_ratio=None,
            qkv_format='thd',
            hidden_size=4,
            activation_dtype=torch.bfloat16,
            num_streams=1,
            single_pass_mhc=False,
            kv_dim=4,
            indexer_dim=4,
            indexer_k_requires_grad=False,
            candidate_block_size=1,
        )
        hidden = torch.randn(4, 1, 4, dtype=torch.bfloat16)
        prefix = torch.tensor([0, 8], dtype=torch.int32)
        group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
        payload = CSA2PipelinePayload(boundary, (hidden, prefix, prefix), 8, 2, 1)
        _, state, _, params = payload.restore(cp_group=group)
        self.assertIsNone(params.local_cp_size)
        self.assertIs(params.cp_group, group)
        self.assertEqual(params.cp_partition_mode, 'contiguous')
        torch.testing.assert_close(state.thd_layout.position_ids, torch.arange(4, 8))
        with self.assertRaisesRegex(ValueError, 'CP shard'):
            payload.restore(cp_group=SimpleNamespace(size=lambda: 2, rank=lambda: 0))

    def test_sparse_attention_outputs_and_gradients(self):
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q = torch.randn(257, 4, 16, dtype=dtype)
                kv = torch.randn(73, 16)
                sink = torch.randn(4)
                indices = torch.randint(0, 73, (257, 11), dtype=torch.int32)
                indices[::9] = -1
                indices[1:10, 0] = 5
                grad = torch.randn(257, 64, dtype=dtype)
                reference_inputs = [x.clone().requires_grad_() for x in (q, kv, sink)]
                actual_inputs = [x.clone().requires_grad_() for x in (q, kv, sink)]
                reference = _unfused_compressed_sparse_attn(*reference_inputs, indices, 0.25)
                actual = unfused_compressed_sparse_attn(
                    *actual_inputs, indices, 0.25, query_chunk_size=128
                )
                torch.testing.assert_close(actual, reference)
                reference_grads = torch.autograd.grad(reference, reference_inputs, grad)
                actual_grads = torch.autograd.grad(actual, actual_inputs, grad)
                for expected, result in zip(reference_grads, actual_grads):
                    torch.testing.assert_close(result, expected, rtol=0.03, atol=0.015625)

    def test_sbhd_chunk_option_preserves_batch_isolation(self):
        q = torch.randn(17, 2, 3, 8, requires_grad=True)
        kv = torch.randn(21, 2, 8, requires_grad=True)
        sink = torch.randn(3, requires_grad=True)
        indices = torch.randint(0, 21, (2, 17, 5))
        expected = _unfused_compressed_sparse_attn(q, kv, sink, indices, 0.2)
        actual = unfused_compressed_sparse_attn(q, kv, sink, indices, 0.2, query_chunk_size=8)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(all(torch.isfinite(x.grad).all() for x in (q, kv, sink)))

    def test_chunked_indexer_mask_and_topk(self):
        torch.manual_seed(8)
        q, k, weights = torch.randn(257, 4, 16), torch.randn(73, 16), torch.randn(257, 4)
        layout = SimpleNamespace(
            valid_tokens=torch.arange(257) % 7 != 0,
            sequence_ids=torch.arange(257) % 2,
            position_ids=torch.arange(257),
        )
        compressed = SimpleNamespace(
            valid_groups=torch.arange(73) % 5 != 0,
            sequence_ids=torch.arange(73) % 2,
            position_ids=2 * torch.arange(73),
        )
        indexer = SimpleNamespace(
            config=SimpleNamespace(csa_query_chunk_size=0),
            head_dim=16,
            n_heads=4,
            compress_ratio=2,
            topk=8,
        )
        for candidates in (None, torch.rand(257, 73) > 0.4):
            kwargs = dict(thd_layout=layout, compressed_layout=compressed, candidates=candidates)
            with torch.no_grad():
                expected = CSA2Indexer._score_projected(indexer, q, k, weights, **kwargs)
                indexer.config.csa_query_chunk_size = 128
                actual = CSA2Indexer._score_projected(indexer, q, k, weights, **kwargs)
                torch.testing.assert_close(actual, expected)
                self.assertTrue(
                    torch.equal(
                        CSA2Indexer.select_indices(indexer, actual),
                        CSA2Indexer.select_indices(indexer, expected),
                    )
                )
            indexer.config.csa_query_chunk_size = 0
        indexer.config.csa_query_chunk_size = 128
        score = CSA2Indexer._score_projected(
            indexer, q.requires_grad_(), k, weights, thd_layout=layout, compressed_layout=compressed
        )
        score.masked_fill(~score.isfinite(), 0).sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_mixed_gradient_kernel_groups_and_norm_order(self):
        grads = [
            torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
            torch.tensor([12.0]),
            torch.tensor([0.0, 5.0], dtype=torch.bfloat16),
        ]
        calls = []

        def kernel(op, overflow, lists, value):
            calls.append([x.dtype for x in lists[0]])
            self.assertEqual(len(set(calls[-1])), 1)
            if op is clip_grads.l2_norm_impl:
                norms = torch.stack([x.float().norm() for x in lists[0]])
                return norms.norm().reshape(1), norms
            for x in lists[0]:
                x.mul_(value)

        with patch.object(clip_grads, 'multi_tensor_applier', side_effect=kernel):
            norm, per_tensor = clip_grads._multi_tensor_applier_by_dtype(
                clip_grads.l2_norm_impl, torch.zeros(1), [grads], True
            )
            torch.testing.assert_close(norm, torch.tensor([194.0]).sqrt())
            torch.testing.assert_close(per_tensor, torch.tensor([5.0, 12.0, 5.0]))
            clip_grads._multi_tensor_applier_by_dtype(
                clip_grads.multi_tensor_scale_impl, torch.zeros(1), [grads, grads], 0.5
            )
        torch.testing.assert_close(grads[1], torch.tensor([6.0]))
        self.assertEqual(len(calls), 4)

    def test_fp32_shard_is_leaf_and_shares_model_storage(self):
        param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
        key = (torch.float32, torch.float32)
        config = SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False)
        ranges = [{key: [{'param_map': {param: {'param': Range(2, 6)}}}]}]
        groups = [{'params': [param], 'orig_group': {'params': [param]}}]
        native_type = torch.Tensor.type

        def cpu_as_cuda(tensor, *args, **kwargs):
            if not args and not kwargs and tensor is param:
                return 'torch.cuda.FloatTensor'
            return native_type(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, 'type', cpu_as_cuda):
            result = DistributedOptimizer._build_model_and_main_param_groups(
                ranges, {param: (0, key, 0)}, groups, config
            )
        shard = result[3][0][0]
        self.assertTrue(shard.is_leaf)
        self.assertIsNone(shard.grad_fn)
        self.assertEqual(shard.untyped_storage().data_ptr(), param.untyped_storage().data_ptr())
        optimizer = torch.optim.AdamW([shard], lr=0.01)
        shard.grad = torch.ones_like(shard)
        optimizer.step()
        self.assertFalse(torch.equal(param.detach()[2:6], torch.arange(2, 6).float()))


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
