import base64
import io
import json
import os
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path

from PIL import Image

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    def register_cpu_ci(*args, **kwargs):
        return None

    CustomTestCase = unittest.TestCase


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "test/jd-ci/pipeline"))

from server_api_dummy_model import (
    SERVER_CASES,
    VALID_PNG_DATA_URL,
    build_config_from_env,
    parse_args,
)


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestServerAPIRegressionConfig(CustomTestCase):
    def test_inline_image_fixture_is_strictly_decodable_rgb_png(self):
        prefix = "data:image/png;base64,"
        self.assertTrue(VALID_PNG_DATA_URL.startswith(prefix))

        image_bytes = base64.b64decode(
            VALID_PNG_DATA_URL.removeprefix(prefix), validate=True
        )
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (1, 1))

    def test_defaults_use_minimal_jd_server_without_upstream_canary(self):
        config = build_config_from_env({})

        self.assertEqual(config.visible_gpu_count, 1)
        self.assertEqual(config.timeout_seconds, 600)
        self.assertEqual(
            config.model_path, "/mnt/nas/models/Qwen2.5-VL-7B-Instruct/"
        )
        self.assertEqual(
            config.server_args,
            (
                "--load-format",
                "dummy",
                "--disable-cuda-graph",
                "--tp-size",
                "1",
                "--mem-fraction-static",
                "0.20",
            ),
        )

    def test_main_pipeline_injects_the_same_single_vlm_model(self):
        main_pipeline = (REPO_ROOT / "test/jd-ci/run_jd_ci.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "JD_CI_SERVER_API_MODEL_PATH:-/mnt/nas/models/Qwen2.5-VL-7B-Instruct/",
            main_pipeline,
        )
        self.assertNotIn("/mnt/nas/models/Qwen3-0.6B", main_pipeline)

    def test_environment_overrides_server_bounds(self):
        config = build_config_from_env(
            {
                "JD_CI_SERVER_API_TIMEOUT_SEC": "120",
                "JD_CI_SERVER_API_MODEL_PATH": "/models/mock-config",
            }
        )

        self.assertEqual(config.timeout_seconds, 120)
        self.assertEqual(config.model_path, "/models/mock-config")
        self.assertEqual(config.visible_gpu_count, 1)

    def test_non_positive_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "TIMEOUT"):
            build_config_from_env({"JD_CI_SERVER_API_TIMEOUT_SEC": "0"})

    def test_runner_is_jd_api_case_not_generic_benchmark(self):
        helper = (
            REPO_ROOT / "test/jd-ci/pipeline/server_api_dummy_model.py"
        ).read_text(encoding="utf-8")

        self.assertIn("/v1/chat/completions", helper)
        self.assertNotIn("run_mock_model_bench_serving", helper)
        self.assertEqual(helper.count("popen_launch_server("), 1)
        self.assertIn('"JD_ENABLE_IGNORE_EOS": "true"', helper)
        self.assertIn('"JD_DEFAULT_MAX_TOKENS": "8"', helper)

    def test_runner_does_not_import_upstream_mock_canary_helpers(self):
        helper = (
            REPO_ROOT / "test/jd-ci/pipeline/server_api_dummy_model.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("sglang.test.mock_model.utils", helper)
        self.assertNotIn("mock_model_server_args", helper)
        self.assertNotIn("mock_model_server_env", helper)
        self.assertIn("other_args=list(config.server_args)", helper)

    def test_fixed_server_subcase_inventory_is_complete(self):
        self.assertEqual(
            [case.case_id for case in SERVER_CASES],
            [
                "jd-models-endpoint",
                "jd-text-non-streaming",
                "jd-text-streaming",
                "jd-image-non-streaming",
                "jd-image-streaming",
                "jd-broken-base64-non-streaming",
                "jd-invalid-image-url-non-streaming",
                "jd-broken-base64-streaming",
                "jd-invalid-image-url-streaming",
                "jd-ignore-eos-token-limit",
                "jd-invalid-thinking-list",
                "jd-invalid-thinking-dict",
                "jd-invalid-thinking-string",
                "jd-invalid-thinking-int",
                "jd-tool-choice-none",
            ],
        )

    def test_list_cases_writes_inventory_without_starting_server(self):
        helper = REPO_ROOT / "test/jd-ci/pipeline/server_api_dummy_model.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "server-cases.json"
            result = subprocess.run(
                [sys.executable, str(helper), "--list-cases", "--output", str(output)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            inventory = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(inventory), 15)
            self.assertTrue(all(case["assertion"] for case in inventory))
            self.assertTrue(all(case["timeout_seconds"] > 0 for case in inventory))

    def test_cli_requires_named_jd_case(self):
        args = parse_args(
            [
                "--case",
                "jd-server-api-regressions",
                "--output",
                "/tmp/result.json",
            ]
        )

        self.assertEqual(args.case, "jd-server-api-regressions")

    def test_dry_run_is_reported_as_skipped(self):
        runner = REPO_ROOT / "test/jd-ci/pipeline/run_server_api_regression.sh"
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = os.environ.copy()
            env["JD_CI_SERVER_API_DRY_RUN"] = "1"
            result = subprocess.run(
                ["bash", str(runner), str(REPO_ROOT), tmp_dir],
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(
                (Path(tmp_dir) / "server_api/report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["test_area"], "server_api")
            self.assertEqual(report["status"], "skipped")
            self.assertEqual(report["total"], 15)
            self.assertEqual(report["cases"][0]["name"], "jd-models-endpoint")
            self.assertEqual(report["cases"][-1]["name"], "jd-tool-choice-none")

    def test_runner_lists_and_imports_individual_server_subcases(self):
        runner = (
            REPO_ROOT / "test/jd-ci/pipeline/run_server_api_regression.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("--list-cases", runner)
        self.assertIn('result.get("cases", [])', runner)

    def test_runner_prints_inventory_and_uses_progress_runner(self):
        runner = (
            REPO_ROOT / "test/jd-ci/pipeline/run_server_api_regression.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("pipeline/case_progress.py", runner)
        for argument in (
            "--index",
            "--total",
            "--assertion",
            "--timeout-seconds",
        ):
            self.assertIn(argument, runner)
        self.assertIn("[INVENTORY ", runner)

    def test_runner_records_failure_and_continues_remaining_cases(self):
        runner = (
            REPO_ROOT / "test/jd-ci/pipeline/run_server_api_regression.sh"
        ).read_text(encoding="utf-8")
        failure_block = runner.split(
            'if [[ ${case_exit_code} -ne 0 ]]', maxsplit=1
        )[1].split("\nfi\n", maxsplit=1)[0]

        self.assertNotIn("exit ${", failure_block)
        self.assertIn('result.get("cases", [])', runner)
        self.assertTrue(runner.rstrip().endswith("exit ${REGRESSION_EXIT_CODE}"))


if __name__ == "__main__":
    unittest.main()
