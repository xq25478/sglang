import unittest
from http import HTTPStatus
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _OutputCollector:
    def __init__(self):
        self.outputs = []

    def send_output(self, output, req):
        self.outputs.append((output, req))


class TestSchedulerChunkedAbortReason(unittest.TestCase):
    def test_pending_abort_reason_survives_chunk_finalization(self):
        pending_reason = FINISH_ABORT(
            "client cancelled during chunked prefill",
            HTTPStatus.BAD_REQUEST,
            "BadRequestError",
        )
        trace_aborts = []
        req = SimpleNamespace(
            rid="chunked-abort",
            finished_reason=None,
            to_finish=pending_reason,
            return_logprob=False,
            req_pool_idx=None,
            kv_committed_freed=False,
            time_stats=SimpleNamespace(
                trace_ctx=SimpleNamespace(
                    abort=lambda abort_info: trace_aborts.append(abort_info)
                )
            ),
        )
        output_collector = _OutputCollector()
        scheduler = object.__new__(Scheduler)
        scheduler._pending_chunked_abort_req = req
        scheduler.chunked_req = req
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hicache_storage = False
        scheduler.tree_cache = SimpleNamespace(supports_mamba=lambda: False)
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=output_collector
        )

        scheduler.process_pending_chunked_abort()

        self.assertIs(req.finished_reason, pending_reason)
        self.assertIsNone(req.to_finish)
        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(scheduler._pending_chunked_abort_req)
        self.assertEqual(
            trace_aborts,
            [{"reason": "client cancelled during chunked prefill"}],
        )
        self.assertEqual(len(output_collector.outputs), 1)
        output, output_req = output_collector.outputs[0]
        self.assertIs(output_req, req)
        self.assertEqual(output.finished_reason, pending_reason.to_json())
        self.assertEqual(output.abort_message, pending_reason.message)


if __name__ == "__main__":
    unittest.main()
