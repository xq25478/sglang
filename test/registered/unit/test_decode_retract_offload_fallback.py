"""Regression: PD-decode retraction must not crash for KV pools without CPU offload.

Decode-side retraction hibernates a request by snapshotting its KV to CPU
(``Req.offload_kv_cache`` -> pool ``get_cpu_copy``) and restores it on resume
(``load_cpu_copy``). Composite sparse pools (e.g. ``DeepSeekV4TokenToKVPool``)
do not implement these methods; a retract used to raise ``NotImplementedError``
out of ``release_req`` and kill the scheduler process. The fallback keeps the
scheduler alive: pools without CPU-offload support degrade retraction to a
graceful ``FINISH_ABORT`` instead of re-queuing the request as retracted.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.srt.managers.schedule_batch as schedule_batch_mod
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Req,
    ScheduleBatch,
    kv_pool_supports_decode_retract_offload,
    release_req,
)
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.test.test_utils import CustomTestCase


class _BaseFakePool(KVCache):
    """Concrete KVCache that inherits the base (raising) cpu-copy methods."""

    def __init__(self):
        pass

    def get_key_buffer(self, layer_id):
        raise NotImplementedError()

    def get_value_buffer(self, layer_id):
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id):
        raise NotImplementedError()

    def set_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError()


class _OffloadCapablePool(_BaseFakePool):
    def get_cpu_copy(self, indices, mamba_indices=None):
        return b"snapshot"

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        pass


class _RaisingOffloadPool(_OffloadCapablePool):
    """Declares overrides (like HiSparseC4DevicePool) but refuses at runtime."""

    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError("cpu offload not supported in this configuration")

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError("cpu offload not supported in this configuration")


class _FakeAllocator:
    def __init__(self, pool):
        self._pool = pool

    def get_kvcache(self):
        return self._pool


def _decode_server_args(**overrides):
    args = SimpleNamespace(
        disaggregation_mode="decode",
        speculative_algorithm=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestKvPoolSupportsDecodeRetractOffload(CustomTestCase):
    def test_base_pool_without_overrides_is_unsupported(self):
        allocator = _FakeAllocator(_BaseFakePool())
        self.assertFalse(kv_pool_supports_decode_retract_offload(allocator))

    def test_pool_with_overrides_is_supported(self):
        allocator = _FakeAllocator(_OffloadCapablePool())
        self.assertTrue(kv_pool_supports_decode_retract_offload(allocator))

    def test_raise_only_overrides_pass_static_check(self):
        # The static probe cannot detect bodies that raise; the runtime
        # NotImplementedError guard in release_req is the safety net for these.
        allocator = _FakeAllocator(_RaisingOffloadPool())
        self.assertTrue(kv_pool_supports_decode_retract_offload(allocator))


class TestReleaseReqOffloadGating(CustomTestCase):
    def _run(self, allocator, req):
        with (
            mock.patch.object(schedule_batch_mod, "release_kv_cache"),
            mock.patch.object(schedule_batch_mod, "evict_from_tree_cache"),
        ):
            release_req(
                req=req,
                remaing_req_count=1,
                server_args=_decode_server_args(),
                req_to_token_pool=mock.MagicMock(),
                token_to_kv_pool_allocator=allocator,
                tree_cache=mock.MagicMock(),
                hisparse_coordinator=None,
            )

    def test_unsupported_pool_skips_offload_without_raising(self):
        req = mock.MagicMock(spec=Req)
        req.offload_kv_cache.side_effect = AssertionError("must not be called")
        self._run(_FakeAllocator(_BaseFakePool()), req)
        req.offload_kv_cache.assert_not_called()
        req.reset_for_retract.assert_called_once()

    def test_raising_pool_offload_failure_does_not_kill_scheduler(self):
        req = mock.MagicMock(spec=Req)
        req.offload_kv_cache.side_effect = NotImplementedError()
        self._run(_FakeAllocator(_RaisingOffloadPool()), req)
        req.offload_kv_cache.assert_called_once()
        req.reset_for_retract.assert_called_once()

    def test_capable_pool_offloads_normally(self):
        req = mock.MagicMock(spec=Req)
        self._run(_FakeAllocator(_OffloadCapablePool()), req)
        req.offload_kv_cache.assert_called_once()
        req.reset_for_retract.assert_called_once()

    def test_non_decode_mode_never_offloads(self):
        req = mock.MagicMock(spec=Req)
        with (
            mock.patch.object(schedule_batch_mod, "release_kv_cache"),
            mock.patch.object(schedule_batch_mod, "evict_from_tree_cache"),
        ):
            release_req(
                req=req,
                remaing_req_count=1,
                server_args=_decode_server_args(disaggregation_mode="null"),
                req_to_token_pool=mock.MagicMock(),
                token_to_kv_pool_allocator=_FakeAllocator(_OffloadCapablePool()),
                tree_cache=mock.MagicMock(),
                hisparse_coordinator=None,
            )
        req.offload_kv_cache.assert_not_called()


class _FakeReq:
    def __init__(self, rid, num_input_tokens, num_output_tokens):
        self.rid = rid
        self.origin_input_ids = list(range(num_input_tokens))
        self.output_ids = list(range(num_output_tokens))
        self.sampling_params = SimpleNamespace(max_new_tokens=64)
        self.to_finish = None


def _make_retract_batch(num_reqs, *, offload_supported):
    """Minimal ScheduleBatch stand-in that forces exactly one retraction."""
    batch = SimpleNamespace(
        reqs=[
            _FakeReq(rid=f"req-{i}", num_input_tokens=i + 1, num_output_tokens=i + 1)
            for i in range(num_reqs)
        ],
    )
    # Mem fits only after at least one request is retracted.
    batch.check_decode_mem = lambda selected_indices: len(selected_indices) < num_reqs

    def _release_req(idx, remaining_req_count, server_args):
        req = batch.reqs[idx]
        if offload_supported:
            req.kv_cache_cpu = b"snapshot"

    batch.release_req = _release_req

    def _filter_batch(keep_indices, **kwargs):
        batch.reqs = [batch.reqs[i] for i in keep_indices]

    batch.filter_batch = _filter_batch
    return batch


class TestRetractDecodeAbortFallback(CustomTestCase):
    def test_retract_aborts_request_when_pool_cannot_offload(self):
        batch = _make_retract_batch(3, offload_supported=False)
        retracted, _, aborts = ScheduleBatch.retract_decode(
            batch, _decode_server_args()
        )

        self.assertEqual(retracted, [])
        self.assertEqual(len(aborts), 1)
        self.assertIsInstance(aborts[0].to_finish, FINISH_ABORT)
        self.assertFalse(hasattr(aborts[0], "kv_cache_cpu"))
        # The least-progressed request (fewest output tokens) is retracted, and
        # the remaining requests keep decoding.
        self.assertEqual(aborts[0].rid, "req-0")
        self.assertEqual({r.rid for r in batch.reqs}, {"req-1", "req-2"})

    def test_retract_hibernates_request_when_pool_offloads(self):
        batch = _make_retract_batch(3, offload_supported=True)
        retracted, _, aborts = ScheduleBatch.retract_decode(
            batch, _decode_server_args()
        )

        self.assertEqual(aborts, [])
        self.assertEqual(len(retracted), 1)
        self.assertEqual(retracted[0].kv_cache_cpu, b"snapshot")
        self.assertIsNone(retracted[0].to_finish)

    def test_non_decode_mode_never_aborts(self):
        batch = _make_retract_batch(3, offload_supported=False)
        retracted, _, aborts = ScheduleBatch.retract_decode(
            batch, _decode_server_args(disaggregation_mode="null")
        )

        self.assertEqual(aborts, [])
        self.assertEqual(len(retracted), 1)
        self.assertIsNone(retracted[0].to_finish)


if __name__ == "__main__":
    unittest.main()
