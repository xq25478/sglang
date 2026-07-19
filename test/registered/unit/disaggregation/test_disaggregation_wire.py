import threading
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.common.utils import (
    group_concurrent_contiguous,
    pack_int_lists,
    pack_list_of_buffers,
    unpack_int_lists,
    unpack_list_of_buffers,
)
from sglang.srt.disaggregation.utils import get_dsv4_c128_state_indices
from sglang.test.ci.ci_register import register_cpu_ci

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


if __name__ == "__main__":
    unittest.main()
