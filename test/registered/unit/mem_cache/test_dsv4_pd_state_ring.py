import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
    should_use_speculative_state_ring,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV4PDStateRing(unittest.TestCase):
    def test_dspark_prefill_matches_speculative_decode_ring_policy(self):
        prefill_args = SimpleNamespace(
            speculative_algorithm=None,
            disaggregation_mode="prefill",
        )
        regular_args = SimpleNamespace(
            speculative_algorithm=None,
            disaggregation_mode="null",
        )

        with patch.object(
            envs.SGLANG_DSPARK_PD_TARGET_LAYER_IDS,
            "get",
            return_value=(40, 41, 42),
        ):
            self.assertTrue(should_use_speculative_state_ring(prefill_args))
            self.assertFalse(should_use_speculative_state_ring(regular_args))

    def test_prefill_without_dspark_hidden_capture_keeps_regular_ring(self):
        prefill_args = SimpleNamespace(
            speculative_algorithm=None,
            disaggregation_mode="prefill",
        )

        with patch.object(
            envs.SGLANG_DSPARK_PD_TARGET_LAYER_IDS,
            "get",
            return_value=tuple(),
        ):
            self.assertFalse(should_use_speculative_state_ring(prefill_args))

    def test_explicit_pool_policy_controls_ring_stride(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool.use_speculative_state_ring = True
        self.assertEqual(pool.get_ring_size(4), 16)
        self.assertEqual(pool.get_ring_size(128), 256)

        pool.use_speculative_state_ring = False
        self.assertEqual(pool.get_ring_size(4), 8)
        self.assertEqual(pool.get_ring_size(128), 128)


if __name__ == "__main__":
    unittest.main()
