import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    FAKE_BOOTSTRAP_HOST,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False

    def abort(self):
        pass

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


class TestDecodeQueueCleanup(CustomTestCase):
    def test_fake_health_transfer_skips_dspark_hidden_metadata(self):
        receiver = SimpleNamespace(send_metadata=MagicMock())
        req = SimpleNamespace(
            rid="HEALTH_CHECK_test",
            origin_input_ids=[0],
            output_ids=[],
            sampling_params=SimpleNamespace(max_new_tokens=1),
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            bootstrap_room=0,
            req_pool_idx=0,
            to_finish=None,
            finished_reason=None,
            return_logprob=False,
            time_stats=MagicMock(),
        )
        decode_req = DecodeRequest(
            req=req,
            kv_receiver=receiver,
            waiting_for_input=True,
        )

        dspark_pool = SimpleNamespace(
            size=8,
            hidden_size=4,
            alloc=MagicMock(return_value=[0]),
        )
        scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            enable_priority_scheduling=False,
            enable_hisparse=False,
            enable_decode_hicache=False,
            server_args=SimpleNamespace(
                disaggregation_decode_enable_radix_cache=False,
                disaggregation_transfer_backend="mooncake",
            ),
            spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
            model_config=SimpleNamespace(num_hidden_layers=1, hidden_size=4),
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    dflash_or_dspark_target_layer_ids=[0],
                    spec_aux_config=None,
                )
            ),
            draft_worker=None,
            output_streamer=MagicMock(),
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.scheduler = scheduler
        queue.transfer_queue = SimpleNamespace(enable_staging=False, queue=[])
        queue.req_to_token_pool = SimpleNamespace(
            available_size=MagicMock(return_value=1),
            req_to_token=torch.tensor([[7]], dtype=torch.int64),
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=MagicMock(return_value=1),
            alloc=MagicMock(return_value=0),
        )
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        queue.token_to_kv_pool = MagicMock()
        queue.metadata_buffers = SimpleNamespace(dspark_hidden_pool=dspark_pool)
        queue.kv_manager = SimpleNamespace(
            kv_args=SimpleNamespace(state_types=[StateType.DSPARK_HIDDEN])
        )
        queue.tree_cache = MagicMock()
        queue.num_reserved_decode_tokens = 0
        queue._last_dspark_hidden_recv_credit_warning_time = 0.0
        queue._resolve_pending_reqs = MagicMock()
        queue._retry_pending_abort_notifications = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=100)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._pre_alloc = MagicMock(
            return_value=torch.tensor([7], dtype=torch.int64)
        )

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [decode_req])
        self.assertEqual(failed, [])
        dspark_pool.alloc.assert_not_called()
        self.assertIsNone(decode_req.dspark_hidden_dst_indices)
        self.assertIsNone(decode_req.dspark_hidden_dst_indices_by_pp)
        self.assertIsNone(receiver.send_metadata.call_args.args[2])
        self.assertIsNone(receiver.send_metadata.call_args.kwargs["spec_metadata"])

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            bootstrap_room=7,
            to_finish=None,
            finished_reason=FINISH_ABORT("aborted"),
            return_logprob=False,
        )
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._retry_pending_abort_notifications = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue.transfer_queue = MagicMock()
        queue.transfer_queue.queue = []

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_clears_receiver_before_removing_request(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
        )
        decode_req = DecodeRequest(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch = None
        scheduler.enable_overlap = False
        scheduler.ps = SimpleNamespace(pp_size=1)
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False

        self.assertFalse(scheduler.is_fully_idle())


if __name__ == "__main__":
    unittest.main()
