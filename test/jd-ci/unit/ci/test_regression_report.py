import json
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

from report.regression_report import (
    write_regression_report,
    write_skipped_regression_report,
)


register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRegressionReport(CustomTestCase):
    def test_report_counts_each_case_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "nested" / "report.json"

            write_regression_report(
                report_path,
                test_area="cpu_mock",
                status="failed",
                cases=[
                    {"name": "static", "status": "passed"},
                    {"name": "cpu", "status": "failed"},
                    {"name": "optional", "status": "skipped"},
                ],
                metadata={"gpu_required": False},
                duration_seconds=1.25,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["test_area"], "cpu_mock")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["total"], 3)
            self.assertEqual(report["passed"], 1)
            self.assertEqual(report["failed"], 1)
            self.assertEqual(report["skipped"], 1)
            self.assertEqual(report["duration_seconds"], 1.25)
            self.assertFalse(report["gpu_required"])

    def test_report_rejects_unknown_case_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "unsupported case status"):
                write_regression_report(
                    Path(tmp_dir) / "report.json",
                    test_area="operator",
                    status="failed",
                    cases=[{"name": "hardware", "status": "waiting"}],
                )

    def test_report_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "report.json"
            report_path.write_text('{"old": true}\n', encoding="utf-8")

            write_regression_report(
                report_path,
                test_area="server_api",
                status="passed",
                cases=[{"name": "mock-model", "status": "passed"}],
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertNotIn("old", report)
            self.assertEqual(report["passed"], 1)

    def test_explicit_skip_is_marked_as_allowed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "report.json"

            write_skipped_regression_report(
                report_path,
                test_area="operator",
                reason="JD_CI_OPERATOR_DRY_RUN=1",
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "skipped")
            self.assertTrue(report["skip_allowed"])
            self.assertEqual(report["skip_reason"], "JD_CI_OPERATOR_DRY_RUN=1")

    def test_report_preserves_observable_case_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "report.json"

            write_regression_report(
                report_path,
                test_area="server_api",
                status="passed",
                cases=[
                    {
                        "name": "jd-observable-case",
                        "status": "passed",
                        "assertion": "expected behavior",
                        "duration_seconds": 0.125,
                        "timeout_seconds": 60,
                        "exit_code": 0,
                        "detail": "",
                        "log_file": "/tmp/jd-observable-case.log",
                    }
                ],
            )

            case = json.loads(report_path.read_text(encoding="utf-8"))["cases"][0]
            self.assertEqual(case["assertion"], "expected behavior")
            self.assertEqual(case["timeout_seconds"], 60)
            self.assertGreaterEqual(case["duration_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
