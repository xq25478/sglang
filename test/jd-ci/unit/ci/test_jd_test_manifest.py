import sys
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    def register_cpu_ci(*args, **kwargs):
        return None

    CustomTestCase = unittest.TestCase


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "test/jd-ci"))

from jd_test_manifest import (  # noqa: E402
    INTERNAL_COMMITS,
    JDCase,
    all_cases,
    validate_cases,
)


register_cpu_ci(est_time=3, suite="base-a-test-cpu")


EXPECTED_V0515_COMMITS = {
    "a7aa64409012d18855f164eebb1c42aaf939ebce",
    "96311e38d3a6b3f9be87e638961e3e290553418e",
    "bc480e4882bb3f0df6a6b88e2d790aa3344888fe",
    "2809c9380b5a2db5b0a1afa36bc95cb2acdb9aef",
    "2f46d777d7ef4f7cd654e6a5d954111a5ccabfd9",
    "864ffbc480895a2c6abed22fcfb85a231af0a189",
    "7c6caf9ed699f22dc992f0e27673f14ffc471cda",
    "1de0e5095ce81f9c7334d87bf5fd4715d62c00cf",
    "10ba5495c26bfb5dea5ba5205cf9ab3b9f17ffea",
    "8238f9171c2b70bf6cd3169acc82a7503549d870",
    "0778f0db576164670aec2c31d58bd01c23f1d333",
    "225d4b2fcc2d95b9f9929266a6c22c2ff3b104b9",
    "2bd369e31410ef5fa735a1dbb4513bb0b9e58b06",
    "8d27c47997722ac61e0a51005716800e4f219f15",
    "a684c2001f669bbe09b865314d67f2d737e139f7",
    "3cd534c9d38682361514ae9f4b8d105c9a6a0e08",
    "77465c63d4bf9d5500132e29a0d8e47f60eead8c",
    "222b102f00f8bb7ddbedc887bc32d33755794f73",
    "fb21094805856bc73899df4d2d46beeece26a352",
    "1625f1cbcd97c6e99acf70904f84112b8dfe713a",
    "1bbc3102c77fd9e6aa6896658d1387a3bf93dba1",
}


class TestJDTestManifest(CustomTestCase):
    def test_manifest_covers_every_v0515_production_commit(self):
        self.assertEqual(set(INTERNAL_COMMITS), EXPECTED_V0515_COMMITS)

        report = validate_cases(all_cases(), INTERNAL_COMMITS, check_paths=False)

        self.assertEqual(report["missing_commits"], [])
        self.assertEqual(report["duplicate_case_ids"], [])
        self.assertEqual(report["upstream_test_commands"], [])
        self.assertEqual(report["untracked_cases"], [])
        self.assertEqual(report["invalid_head_tracking"], [])

    def test_ci_owned_cases_track_the_single_ci_head_without_self_sha(self):
        tracked = {
            case.case_id for case in all_cases() if case.tracks_ci_head
        }
        self.assertEqual(
            tracked,
            {
                "jd-ci-contract",
                "jd-dsv4-multistream-lifetime",
                "jd-dsv4-cp-prefill-correctness",
                "jd-dsv4-cp-prefill-performance",
                "jd-dsv4-norm-rope-correctness",
                "jd-dsv4-norm-rope-performance",
            },
        )
        for case in all_cases():
            if case.tracks_ci_head:
                self.assertEqual(case.commits, ())

    def test_every_operator_has_correctness_and_performance(self):
        cases = all_cases()
        correctness = {
            case.operator
            for case in cases
            if case.category == "operator_correctness"
        }
        performance = {
            case.operator
            for case in cases
            if case.category == "operator_performance"
        }

        self.assertTrue(correctness)
        self.assertEqual(correctness, performance)
        self.assertNotIn(None, correctness)

    def test_upstream_suite_command_is_rejected(self):
        bad_case = JDCase(
            case_id="bad-upstream-suite",
            commits=(next(iter(EXPECTED_V0515_COMMITS)),),
            category="cpu",
            command=("python3", "test/run_suite.py", "--suite", "base-a-test-cpu"),
            assertion="incorrectly runs an upstream suite",
        )

        report = validate_cases(
            [bad_case],
            [bad_case.commits[0]],
            check_paths=False,
        )

        self.assertEqual(report["upstream_test_commands"], ["bad-upstream-suite"])

    def test_unmapped_commit_is_rejected(self):
        report = validate_cases(all_cases(), [*INTERNAL_COMMITS, "f" * 40], check_paths=False)

        self.assertEqual(report["missing_commits"], ["f" * 40])

    def test_case_inventory_is_fixed_not_diff_selected(self):
        first = [case.case_id for case in all_cases()]
        second = [case.case_id for case in all_cases()]

        self.assertEqual(first, second)
        self.assertNotIn("changed_files", all_cases.__code__.co_varnames)

    def test_model_specific_protocol_fixtures_remain_explicit(self):
        test_source = (
            REPO_ROOT / "test/jd-ci/unit/server/test_openai_and_function_call.py"
        ).read_text(encoding="utf-8")
        required_methods = (
            "test_invalid_thinking_list_is_ignored",
            "test_invalid_thinking_dict_without_type_is_ignored",
            "test_invalid_thinking_unknown_string_is_ignored",
            "test_invalid_thinking_integer_is_ignored",
            "test_deepseek_v4_reasoning_switch",
            "test_glm45_non_stream_tool_interruption",
            "test_glm45_stream_tool_interruption",
            "test_reasoning_token_usage",
        )

        for method in required_methods:
            with self.subTest(method=method):
                self.assertIn(f"def {method}(", test_source)


if __name__ == "__main__":
    unittest.main()
