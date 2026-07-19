"""Correctness guards for DSpark's folded verify epilogue."""

import unittest
from types import SimpleNamespace

from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_accept_can_be_folded,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _sampling_info(*, is_all_greedy: bool, **overrides):
    values = {
        "is_all_greedy": is_all_greedy,
        "has_custom_logit_processor": False,
        "acc_additive_penalties": None,
        "acc_scaling_penalties": None,
        "penalizer_orchestrator": None,
        "vocab_mask": None,
        "logit_bias": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDSparkFoldedAccept(unittest.TestCase):
    def test_missing_sampling_info_can_fold(self):
        self.assertTrue(verify_accept_can_be_folded(None))

    def test_all_greedy_without_logits_adjustments_can_fold(self):
        self.assertTrue(verify_accept_can_be_folded(_sampling_info(is_all_greedy=True)))

    def test_sampling_request_cannot_use_greedy_folded_accept(self):
        self.assertFalse(
            verify_accept_can_be_folded(_sampling_info(is_all_greedy=False))
        )

    def test_mixed_greedy_and_sampling_batch_cannot_fold(self):
        self.assertFalse(
            verify_accept_can_be_folded(_sampling_info(is_all_greedy=False))
        )

    def test_logits_adjustments_still_disable_fold(self):
        self.assertFalse(
            verify_accept_can_be_folded(
                _sampling_info(
                    is_all_greedy=True,
                    has_custom_logit_processor=True,
                )
            )
        )

    def test_grammar_still_disables_fold(self):
        self.assertFalse(
            verify_accept_can_be_folded(
                _sampling_info(is_all_greedy=True), grammar_mask=object()
            )
        )


if __name__ == "__main__":
    unittest.main()
