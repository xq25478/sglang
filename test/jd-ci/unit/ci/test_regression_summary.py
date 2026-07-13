import sys
import tempfile
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

from report.generate_regression_summary import build_summary
from report.regression_report import write_regression_report


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRegressionSummary(CustomTestCase):
    def _write_regression(self, root, test_area, status, **metadata):
        case_status = (
            status
            if status in {"passed", "failed", "skipped", "blocked"}
            else "failed"
        )
        write_regression_report(
            Path(root) / test_area / "report.json",
            test_area=test_area,
            status=status,
            cases=[{"name": f"{test_area}-case", "status": case_status}],
            metadata=metadata,
        )

    def test_all_passed_is_green(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for test_area in ("cpu_mock", "server_api", "operator"):
                self._write_regression(tmp_dir, test_area, "passed")

            summary = build_summary(tmp_dir, {"event_type": "note__merge_request"})

            self.assertEqual(summary["status"], "passed")
            self.assertEqual(summary["failed_regressions"], [])
            self.assertEqual(len(summary["regressions"]), 3)

    def test_failed_or_blocked_regression_is_red(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_regression(tmp_dir, "cpu_mock", "passed")
            self._write_regression(tmp_dir, "server_api", "failed")
            self._write_regression(
                tmp_dir,
                "operator",
                "blocked",
                required_gpus=2,
                available_gpus=1,
            )

            summary = build_summary(tmp_dir, {})

            self.assertEqual(summary["status"], "failed")
            self.assertEqual(
                summary["failed_regressions"], ["server_api", "operator"]
            )

    def test_explicitly_configured_skip_is_green(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_regression(
                tmp_dir, "cpu_mock", "skipped", skip_allowed=True
            )
            self._write_regression(tmp_dir, "server_api", "passed")
            self._write_regression(
                tmp_dir, "operator", "skipped", skip_allowed=True
            )

            summary = build_summary(tmp_dir, {})

            self.assertEqual(summary["status"], "passed")

    def test_unapproved_skip_and_missing_report_are_red(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_regression(tmp_dir, "cpu_mock", "passed")
            self._write_regression(
                tmp_dir, "server_api", "skipped", skip_allowed=False
            )

            summary = build_summary(tmp_dir, {})

            self.assertEqual(summary["status"], "failed")
            self.assertEqual(
                summary["failed_regressions"], ["server_api", "operator"]
            )


if __name__ == "__main__":
    unittest.main()
