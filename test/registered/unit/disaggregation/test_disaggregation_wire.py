import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.utils import (
    group_concurrent_contiguous,
    pack_int_list,
    pack_int_lists,
    pack_list_of_buffers,
    pack_nested_transfer_layout,
    pack_transfer_layout,
    unpack_int_list,
    unpack_int_lists,
    unpack_list_of_buffers,
    unpack_nested_transfer_layout,
    unpack_transfer_layout,
)
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    _set_mooncake_transfer_device,
)
from sglang.srt.disaggregation.utils import (
    append_draft_kv_data,
    get_dsv4_c128_state_indices,
    setup_state_kv_args,
    should_transfer_draft_cache,
)
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
    CpCacheLayerSplitDeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.cp_cache_layer_split.pool_base import (
    CpCacheLayerSplitPoolBase,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_TRANSFER_C128_STATE,
    DSV4_TRANSFER_SWA_KV,
    DeepSeekV4TokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDisaggregationWire(unittest.TestCase):
    def test_int_lists_roundtrip(self):
        cases = [
            ("Q", [[1, 2, 3], [4]]),
            ("I", [[10, 20], [30, 40, 50]]),
            ("i", [[-1, 2], [3, -4, 5]]),
        ]
        for fmt, sample in cases:
            packed = pack_int_lists(sample, fmt)
            self.assertEqual(unpack_int_lists(packed, fmt), sample, msg=fmt)

    def test_pack_accepts_ndarray(self):
        arrs = [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([4, 5], dtype=np.int32),
        ]
        packed = pack_int_lists(arrs, "i")
        self.assertEqual(unpack_int_lists(packed, "i"), [[1, 2, 3], [4, 5]])

    def test_flat_int_list_roundtrip(self):
        self.assertEqual(unpack_int_list(pack_int_list([7, 8, 9], "I"), "I"), [7, 8, 9])
        self.assertEqual(pack_int_list([], "I"), b"")
        self.assertEqual(unpack_int_list(b"", "I"), [])

    def test_empty_outer_list(self):
        self.assertEqual(pack_int_lists([], "Q"), b"")
        self.assertEqual(unpack_int_lists(b"", "Q"), [])

    def test_empty_inner_list(self):
        packed = pack_int_lists([[]], "I")
        self.assertEqual(unpack_int_lists(packed, "I"), [[]])

    def test_list_of_buffers_roundtrip(self):
        bufs = [b"abc", b"", b"de", b"x" * 17]
        self.assertEqual(unpack_list_of_buffers(pack_list_of_buffers(bufs)), bufs)

    def test_page_stride_accepts_tensor_and_returns_compact_numpy(self):
        from sglang.srt.mem_cache.common import kv_to_page_indices

        token_indices = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])

        page_indices = kv_to_page_indices(token_indices, page_size=4)

        self.assertIsInstance(page_indices, np.ndarray)
        np.testing.assert_array_equal(page_indices, np.array([0, 2]))


class TestDSparkHiddenReleaseGuard(unittest.TestCase):
    def test_worker_reservation_prevents_scheduler_double_free(self):
        from sglang.srt.disaggregation.common.utils import DSparkHiddenReleaseGuard

        guard = DSparkHiddenReleaseGuard()

        self.assertTrue(guard.reserve_worker())
        self.assertFalse(guard.claim_scheduler())
        self.assertTrue(guard.begin_worker_release())
        guard.mark_worker_finished()

        self.assertTrue(guard.worker_finished())
        self.assertFalse(guard.claim_scheduler())

    def test_scheduler_claim_prevents_late_worker_release(self):
        from sglang.srt.disaggregation.common.utils import DSparkHiddenReleaseGuard

        guard = DSparkHiddenReleaseGuard()

        self.assertTrue(guard.claim_scheduler())
        self.assertFalse(guard.reserve_worker())
        self.assertFalse(guard.begin_worker_release())


class TestMooncakeTransferQueueSharding(unittest.TestCase):
    @staticmethod
    def _manager():
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = object.__new__(MooncakeKVManager)
        manager.transfer_queues = [object(), object(), object(), object()]
        manager._transfer_queue_shard_lock = threading.Lock()
        manager._transfer_queue_shard_by_sessions = {}
        manager._transfer_queue_next_shard = 0
        manager._transfer_queue_affinity_overflow_warned = False
        return manager

    def test_aligned_destination_ports_do_not_collapse_to_one_queue(self):
        manager = self._manager()

        first = manager._get_transfer_queue_shard(("node-a:1000", "node-b:2000"))
        second = manager._get_transfer_queue_shard(("node-c:1004", "node-d:2004"))

        self.assertNotEqual(first, second)

    def test_session_set_keeps_affinity_regardless_of_registration_order(self):
        manager = self._manager()

        first = manager._get_transfer_queue_shard(("node-a:1000", "node-b:2000"))
        second = manager._get_transfer_queue_shard(("node-b:2000", "node-a:1000"))

        self.assertEqual(first, second)


class TestMooncakeTransferWorkerDevice(CustomTestCase):
    @patch("sglang.srt.disaggregation.mooncake.conn.torch.cuda.set_device")
    @patch("sglang.srt.disaggregation.mooncake.conn.torch.cuda.is_available")
    def test_transfer_threads_bind_rank_gpu(self, is_available, set_device):
        is_available.return_value = True

        _set_mooncake_transfer_device(3)

        set_device.assert_called_once_with(3)


class TestMooncakeStateAuxOrdering(unittest.TestCase):
    def test_state_failure_prevents_aux_success_from_masking_it(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        calls = []
        manager = SimpleNamespace(
            maybe_send_extra=lambda *_args: calls.append("state") or 7,
            send_aux=lambda *_args: calls.append("aux") or 0,
        )

        ret = MooncakeKVManager._send_last_chunk_state_and_aux(
            manager,
            SimpleNamespace(),
            [np.array([1], dtype=np.int32)],
            None,
            None,
            3,
            [11],
        )

        self.assertEqual(ret, 7)
        self.assertEqual(calls, ["state"])

    def test_aux_status_is_returned_after_state_succeeds(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        calls = []
        manager = SimpleNamespace(
            maybe_send_extra=lambda *_args: calls.append("state") or 0,
            send_aux=lambda *_args: calls.append("aux") or 5,
        )

        ret = MooncakeKVManager._send_last_chunk_state_and_aux(
            manager,
            SimpleNamespace(),
            [np.array([1], dtype=np.int32)],
            None,
            None,
            3,
            [11],
        )

        self.assertEqual(ret, 5)
        self.assertEqual(calls, ["state", "aux"])


class TestDSACacheTransferSkipFlags(unittest.TestCase):
    def test_non_owner_cp_rank_uses_default_non_sharded_state_policy(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = object.__new__(MooncakeKVManager)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = True
        manager.attn_tp_size = 1
        manager.attn_cp_size = 2
        manager.attn_cp_rank = 1
        manager.kv_args = SimpleNamespace(engine_rank=1)

        self.assertEqual(
            manager._get_dsa_cache_transfer_skip_flags(None),
            (False, True),
        )


class TestCpCacheLayerSplitTransferLayoutWire(CustomTestCase):
    def test_layer_split_transfer_layout_roundtrip_preserves_none_slots(self):
        layout = [("dsv4_c4_kv", 1), None, ("dsv4_c128_kv", 3)]

        self.assertEqual(unpack_transfer_layout(pack_transfer_layout(layout)), layout)
        self.assertEqual(pack_transfer_layout([]), b"")
        self.assertEqual(unpack_transfer_layout(b""), [])

    def test_layer_split_state_layout_roundtrip_preserves_component_boundaries(self):
        layouts = [
            [("dsv4_swa_kv", 0), None],
            [("dsv4_attention_state", 1), ("dsv4_indexer_state", 1)],
        ]

        self.assertEqual(
            unpack_nested_transfer_layout(pack_nested_transfer_layout(layouts)),
            layouts,
        )
        self.assertEqual(pack_nested_transfer_layout([]), b"")
        self.assertEqual(unpack_nested_transfer_layout(b""), [])


class TestCpCacheLayerSplitDescriptorMatching(CustomTestCase):
    def _build_params(self, **kwargs):
        params = dict(
            src_data_ptrs=[100],
            dst_data_ptrs=[200],
            item_lens=[16],
            src_data_layout=[("dsv4_c4_kv", 1)],
            dst_data_layout=[("dsv4_c4_kv", 1)],
            dst_item_lens=[16],
        )
        params.update(kwargs)
        return CommonKVManager.build_descriptor_matched_transfer_params(**params)

    def test_descriptor_matching_checks_destination_item_size(self):
        with self.assertRaisesRegex(RuntimeError, "item size mismatch"):
            self._build_params(dst_item_lens=[32])

    def test_required_descriptor_matching_rejects_missing_layouts(self):
        with self.assertRaisesRegex(RuntimeError, "descriptors on both"):
            self._build_params(
                src_data_layout=[],
                dst_data_layout=[],
            )

    def test_descriptor_matching_returns_pointer_item_len_tuples(self):
        self.assertEqual(
            self._build_params(),
            [(100, 200, 16)],
        )

    def test_dspark_hidden_can_force_positional_transfer(self):
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(require_descriptor_matched_transfer=True)
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.enable_custom_mem_pool = False
        manager.get_mla_kv_ptrs_with_pp = (
            lambda src_ptrs, dst_ptrs, _state_type: (src_ptrs, dst_ptrs, 1)
        )
        manager.build_descriptor_matched_transfer_params = lambda *_args: self.fail(
            "DSpark hidden transfer must not require LayerSplit descriptors"
        )
        transferred = []
        manager._transfer_data = (
            lambda _session_id, blocks: transferred.extend(blocks) or 0
        )

        ret = manager._send_kvcache_generic(
            mooncake_session_id="decode-session",
            src_data_ptrs=[100],
            dst_data_ptrs=[200],
            item_lens=[16],
            prefill_data_indices=np.array([2], dtype=np.int32),
            dst_data_indices=np.array([5], dtype=np.int32),
            executor=None,
            state_type=StateType.DSPARK_HIDDEN,
            force_flat=True,
            force_positional=True,
        )

        self.assertEqual(ret, 0)
        self.assertEqual(transferred, [(132, 280, 16)])


class TestGroupConcurrentContiguous(unittest.TestCase):
    @staticmethod
    def _arr(values):
        return np.array(values, dtype=np.int32)

    def test_single_contiguous_group(self):
        src = self._arr([10, 11, 12])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11, 12]], [[5, 6, 7]]),
        )

    def test_splits_on_discontiguous_indices(self):
        src = self._arr([10, 11, 20])
        dst = self._arr([5, 6, 7])
        self.assertEqual(
            group_concurrent_contiguous(src, dst),
            ([[10, 11], [20]], [[5, 6], [7]]),
        )

    def test_both_empty(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([])), ([], [])
        )

    def test_empty_src_nonempty_dst(self):
        self.assertEqual(
            group_concurrent_contiguous(self._arr([]), self._arr([1, 2])), ([], [])
        )

    def test_nonempty_src_empty_dst(self):
        # Regression: a non-empty source paired with an empty destination must not
        # raise a NumPy broadcast error (observed transferring DSA sparse-attention
        # state on a disaggregated GLM deployment when decode registered zero dst indices).
        self.assertEqual(
            group_concurrent_contiguous(self._arr([1, 2]), self._arr([])), ([], [])
        )

    def test_mismatched_nonempty_lengths_raise(self):
        with self.assertRaises(ValueError):
            group_concurrent_contiguous(self._arr([1, 2, 3]), self._arr([1, 2]))


class TestDSV4C128StateIndices(unittest.TestCase):
    def test_online_aligned_boundary_has_no_partial_state(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 256, online=True, ring_size=1),
            np.empty((0,), dtype=np.int32),
        )

    def test_online_partial_boundary_uses_request_slot(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 257, online=True, ring_size=1),
            np.array([7], dtype=np.int32),
        )

    def test_offline_aligned_boundary_has_no_partial_state(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 256, online=False, ring_size=128),
            np.empty((0,), dtype=np.int32),
        )

    def test_offline_partial_boundary_uses_request_local_page(self):
        np.testing.assert_array_equal(
            get_dsv4_c128_state_indices(7, 129, online=False, ring_size=256),
            np.array([15], dtype=np.int32),
        )


def _buf_infos(*ptrs):
    return list(ptrs), [ptr + 100 for ptr in ptrs], [ptr + 200 for ptr in ptrs]


def _make_dsv4_target(*, unified, mapping=None):
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = unified
    pool.page_size = 256
    pool.sliding_window = 128
    pool.full_to_swa_index_mapping = mapping
    pool.unified_swa_window = 128
    pool.unified_swa_ring_size = 131
    pool.unified_swa_pages = 524
    pool.get_state_buf_infos = lambda: _buf_infos(11)
    pool.get_state_transfer_layout = lambda: []
    pool.get_unified_swa_ring_buf_infos = lambda: (
        _buf_infos(12) if unified else ([], [], [])
    )
    pool.get_c128_state_buf_infos = lambda: ([], [], [])
    return pool


def _make_dsv4_draft(*, unified, mapping=None):
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = unified
    pool.compression_ratios = [0]
    pool.page_size = 256
    pool.sliding_window = 128
    pool.full_to_swa_index_mapping = mapping
    pool.unified_swa_window = 128
    pool.unified_swa_ring_size = 131
    pool.unified_swa_pages = 524
    pool.compress_state_pools = [None]
    pool.indexer_compress_state_pools = [None]
    pool.get_state_transfer_layout = lambda: (
        [] if unified else [(DSV4_TRANSFER_SWA_KV, 0)]
    )
    if unified:
        pool.unified_kv_pool = SimpleNamespace(
            swa_pages=524,
            kv_buffer=[torch.empty((524, 16), dtype=torch.uint8)],
        )
    else:
        pool.swa_kv_pool = SimpleNamespace(
            kv_buffer=[torch.empty((2, 16), dtype=torch.uint8)]
        )
    return pool


class TestDSV4DraftStateRegistration(CustomTestCase):
    def test_draft_state_is_a_separate_component(self):
        mapping = torch.arange(16)
        cases = [
            (
                "paged",
                _make_dsv4_target(unified=False, mapping=mapping),
                _make_dsv4_draft(unified=False, mapping=mapping),
                [StateType.SWA, StateType.SWA],
                [[11]],
                [(DSV4_TRANSFER_SWA_KV, 0)],
            ),
            (
                "unified",
                _make_dsv4_target(unified=True),
                _make_dsv4_draft(unified=True),
                [StateType.SWA, StateType.SWA_RING, StateType.SWA_RING],
                [[11], [12]],
                [],
            ),
        ]

        for name, target, draft, expected_types, target_ptrs, draft_layout in cases:
            with self.subTest(name=name):
                if draft._unified_kv:
                    expected_infos = draft.get_unified_swa_ring_buf_infos()
                else:
                    expected_infos = draft.get_state_buf_infos()
                kv_args = KVArgs()

                setup_state_kv_args(kv_args, target, draft)

                self.assertEqual(kv_args.state_types, expected_types)
                self.assertEqual(kv_args.state_data_ptrs[:-1], target_ptrs)
                self.assertEqual(kv_args.state_data_ptrs[-1], expected_infos[0])
                self.assertEqual(kv_args.state_data_lens[-1], expected_infos[1])
                self.assertEqual(kv_args.state_item_lens[-1], expected_infos[2])
                self.assertEqual(kv_args.state_data_layouts[-1], draft_layout)


class TestDSV4DraftLayerSplitTransfer(CustomTestCase):
    def test_empty_draft_kv_data_preserves_target_descriptors(self):
        ptrs, lens, item_lens = [1], [2], [3]
        layout = [("dsv4_c4_kv", 4)]
        draft = SimpleNamespace(get_contiguous_buf_infos=lambda: ([], [], []))

        added = append_draft_kv_data(ptrs, lens, item_lens, layout, draft)

        self.assertEqual(added, 0)
        self.assertEqual(layout, [("dsv4_c4_kv", 4)])

    def test_only_last_layer_split_rank_transfers_replicated_draft(self):
        pool = object.__new__(CpCacheLayerSplitPoolBase)
        pool.cp_size = 4

        pool.cp_rank = 0
        self.assertFalse(should_transfer_draft_cache(pool))
        pool.cp_rank = 3
        self.assertTrue(should_transfer_draft_cache(pool))
        self.assertTrue(should_transfer_draft_cache(object()))

    def test_empty_layer_split_c128_component_is_not_transferred(self):
        manager = object.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            state_types=[StateType.C128_STATE],
            state_data_ptrs=[[]],
            state_item_lens=[[]],
            require_descriptor_matched_transfer=True,
        )
        req = SimpleNamespace(dst_state_indices=[[0]])

        self.assertEqual(
            manager.maybe_send_extra(req, [[0]], executor=None),
            0,
        )


class TestDSV4C128StateRegistration(CustomTestCase):
    def test_c128_state_layout_is_registered_as_separate_component(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.get_state_buf_infos = lambda: _buf_infos(11)
        pool.get_state_transfer_layout = lambda: [(DSV4_TRANSFER_SWA_KV, 0)]
        pool.get_c128_state_buf_infos = lambda: _buf_infos(12)
        pool.get_c128_state_transfer_layout = lambda: [(DSV4_TRANSFER_C128_STATE, 5)]
        kv_args = KVArgs()

        setup_state_kv_args(kv_args, pool)

        self.assertEqual(kv_args.state_types, [StateType.SWA, StateType.C128_STATE])
        self.assertEqual(
            kv_args.state_data_layouts,
            [
                [(DSV4_TRANSFER_SWA_KV, 0)],
                [(DSV4_TRANSFER_C128_STATE, 5)],
            ],
        )

    def test_layer_split_keeps_empty_c128_slot_before_draft_state(self):
        mapping = torch.arange(16)
        target = object.__new__(CpCacheLayerSplitDeepSeekV4TokenToKVPool)
        target._unified_kv = False
        target.compression_ratios = [0, 128]
        target.page_size = 256
        target.sliding_window = 128
        target.full_to_swa_index_mapping = mapping
        target.get_state_buf_infos = lambda: _buf_infos(11)
        target.get_state_transfer_layout = lambda: [(DSV4_TRANSFER_SWA_KV, 0)]
        target.get_c128_state_buf_infos = lambda: ([], [], [])
        target.get_c128_state_transfer_layout = lambda: []
        draft = _make_dsv4_draft(unified=False, mapping=mapping)
        kv_args = KVArgs()

        setup_state_kv_args(kv_args, target, draft)

        self.assertEqual(
            kv_args.state_types,
            [StateType.SWA, StateType.C128_STATE, StateType.SWA],
        )
        self.assertEqual(kv_args.state_data_ptrs[1], [])
        self.assertEqual(
            kv_args.state_data_layouts,
            [
                [(DSV4_TRANSFER_SWA_KV, 0)],
                [],
                [(DSV4_TRANSFER_SWA_KV, 0)],
            ],
        )


if __name__ == "__main__":
    unittest.main()
