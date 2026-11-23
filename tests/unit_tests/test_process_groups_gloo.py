# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional CPU data/context groups at the MPU compatibility boundary."""

from unittest.mock import Mock, patch

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection


def test_mpu_collection_exposes_existing_data_context_gloo_group():
    group = Mock()
    with (
        patch.object(parallel_state, '_DATA_PARALLEL_GROUP_WITH_CP_GLOO', group),
        patch.object(parallel_state, 'get_data_parallel_group_gloo', return_value=group) as getter,
    ):
        collection = ProcessGroupCollection.use_mpu_process_groups(['dp_cp_gloo'])
    assert collection.dp_cp_gloo is group
    getter.assert_called_once_with(with_context_parallel=True)


def test_mpu_collection_allows_uninitialized_data_context_gloo_group():
    with (
        patch.object(parallel_state, '_DATA_PARALLEL_GROUP_WITH_CP_GLOO', None),
        patch.object(parallel_state, 'get_data_parallel_group_gloo') as getter,
    ):
        collection = ProcessGroupCollection.use_mpu_process_groups(['dp_cp_gloo'])
    assert collection.dp_cp_gloo is None
    getter.assert_not_called()


def test_mpu_collection_only_materializes_requested_groups():
    group = Mock()
    with (
        patch.object(parallel_state, 'get_context_parallel_group', return_value=group),
        patch.object(parallel_state, 'get_data_parallel_group_gloo') as getter,
    ):
        collection = ProcessGroupCollection.use_mpu_process_groups(['cp'])
    assert collection.cp is group
    assert not hasattr(collection, 'dp_cp_gloo')
    getter.assert_not_called()
