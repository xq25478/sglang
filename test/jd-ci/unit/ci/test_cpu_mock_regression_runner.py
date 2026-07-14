import json
import os
import subprocess
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


register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER = REPO_ROOT / "test/jd-ci/pipeline/run_cpu_mock_regression.sh"


class TestCPUAndMockRegressionRunner(CustomTestCase):
    def test_dry_run_writes_gpu_free_report_with_all_cpu_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = os.environ.copy()
            env["JD_CI_CPU_MOCK_DRY_RUN"] = "1"
            result = subprocess.run(
                ["bash", str(RUNNER), str(REPO_ROOT), "HEAD", tmp_dir],
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(
                (Path(tmp_dir) / "cpu_mock/report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["test_area"], "cpu_mock")
            self.assertEqual(report["status"], "skipped")
            self.assertFalse(report["gpu_required"])
            self.assertTrue(report["gpu_hidden"])
            self.assertEqual(report["failed"], 0)
            self.assertGreater(report["total"], 0)
            self.assertTrue(
                all(case["name"].startswith("jd-") for case in report["cases"])
            )

    def test_main_pipeline_keeps_build_and_packaging_flow(self):
        script = (REPO_ROOT / "test/jd-ci/run_jd_ci.sh").read_text(encoding="utf-8")

        self.assertEqual(script.count("'${MOONCAKE_TE_WORK_DIR}/compile' 'te'"), 1)
        self.assertEqual(
            script.count("'${MOONCAKE_STORE_WORK_DIR}/compile' 'store'"), 1
        )
        self.assertEqual(
            script.count("bash '${SOURCE_PATH}/test/jd-ci/env/build_sgl_kernel.sh'"),
            1,
        )
        self.assertIn("pipeline/run_cpu_mock_regression.sh", script)
        self.assertIn("pipeline/run_server_api_regression.sh", script)
        self.assertIn("pipeline/run_operator_regression.sh", script)
        self.assertNotIn("pipeline/run_p1_accuracy.sh", script)
        self.assertNotIn("pipeline/run_p2_functional.sh", script)
        self.assertIn("固定累积 JD", script)
        publish_condition = (
            'if [[ ${EXIT_CODE} -eq 0 && "${PUBLISH_IMAGES}" == "1" ]]'
        )
        self.assertIn(publish_condition, script)
        self.assertLess(
            script.index("STORE_EXIT_CODE="), script.index(publish_condition)
        )
        self.assertIn('docker commit \\', script)
        self.assertIn('docker push "${CLOUD_IMAGE}"', script)

    def test_cpu_mock_runs_fixed_jd_manifest_not_upstream_suites(self):
        script = RUNNER.read_text(encoding="utf-8")

        self.assertIn("jd_test_manifest.py", script)
        self.assertNotIn("--include-files-from", script)
        self.assertNotIn("test/run_suite.py", script)

    def test_cpu_mock_rejects_new_test_assets_outside_jd_ci(self):
        script = RUNNER.read_text(encoding="utf-8")

        self.assertIn("git diff --diff-filter=A --name-only", script)
        self.assertIn("JD test assets must stay under test/jd-ci/", script)

    def test_cpu_mock_prints_inventory_and_uses_progress_runner(self):
        script = RUNNER.read_text(encoding="utf-8")

        self.assertIn("pipeline/case_progress.py", script)
        for argument in (
            "--index",
            "--total",
            "--assertion",
            "--timeout-seconds",
        ):
            self.assertIn(argument, script)
        self.assertIn("[INVENTORY ", script)

    def test_cpu_mock_records_failure_and_continues_remaining_cases(self):
        script = RUNNER.read_text(encoding="utf-8")
        failure_block = script.split(
            'if [[ ${exit_code} -eq 0 ]]', maxsplit=1
        )[1].split("\n    fi\n", maxsplit=1)[0]

        self.assertNotIn("exit ${", failure_block)
        self.assertTrue(script.rstrip().endswith("exit ${REGRESSION_EXIT_CODE}"))


if __name__ == "__main__":
    unittest.main()
