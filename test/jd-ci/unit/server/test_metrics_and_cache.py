import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache


class _JDCacheForMetrics(BasePrefixCache):
    def __init__(self):
        self.metrics_collector = MagicMock()

    def reset(self):
        pass

    def match_prefix(self, *args, **kwargs):
        pass

    def cache_finished_req(self, *args, **kwargs):
        pass

    def cache_unfinished_req(self, *args, **kwargs):
        pass

    def evict(self, *args, **kwargs):
        pass

    def inc_lock_ref(self, *args, **kwargs):
        pass

    def dec_lock_ref(self, *args, **kwargs):
        pass


class TestJDCacheResidencyMetrics(unittest.TestCase):
    def setUp(self):
        self.cache = _JDCacheForMetrics()

    def test_l1_residency_is_observed_exactly_once(self):
        node = SimpleNamespace(l1_tier_enter_time=10.0)
        with patch(
            "sglang.srt.mem_cache.base_prefix_cache.time.monotonic",
            return_value=14.5,
        ):
            self.cache.finish_l1_tier_stay(node)
            self.cache.finish_l1_tier_stay(node)

        self.cache.metrics_collector.observe_l1_residency.assert_called_once_with(4.5)
        self.assertEqual(node.l1_tier_enter_time, 0.0)

    def test_l2_enter_timestamp_is_not_overwritten(self):
        node = SimpleNamespace(l2_tier_enter_time=0.0)
        with patch(
            "sglang.srt.mem_cache.base_prefix_cache.time.monotonic",
            side_effect=[20.0, 30.0],
        ):
            self.cache.mark_l2_tier_enter(node)
            self.cache.mark_l2_tier_enter(node)

        self.assertEqual(node.l2_tier_enter_time, 20.0)


class TestJDKVCacheBytesPerToken(unittest.TestCase):
    def test_tuple_kv_sizes_are_summed(self):
        scheduler = object.__new__(Scheduler)
        kv_pool = MagicMock()
        kv_pool.get_kv_size_bytes.return_value = (1024, 2048)
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.token_to_kv_pool_allocator.get_kvcache.return_value = kv_pool
        scheduler.max_total_num_tokens = 128
        scheduler.ps = SimpleNamespace(tp_rank=0)

        self.assertEqual(scheduler._compute_kv_cache_bytes_per_token(), 24.0)

    def test_metric_is_not_emitted_when_metrics_are_disabled(self):
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(enable_metrics=False)
        scheduler.metrics_collector = MagicMock()

        scheduler._emit_kv_cache_bytes_per_token_metric()

        scheduler.metrics_collector.emit_kv_cache_bytes_per_token.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
