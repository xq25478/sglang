import json
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

from operator_registry import resolve_operator_specs, validate_operator_pairs


register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestOperatorRegistry(CustomTestCase):
    def test_runner_records_failure_and_continues_remaining_cases(self):
        runner = (
            REPO_ROOT / "test/jd-ci/pipeline/run_operator_regression.sh"
        ).read_text(encoding="utf-8")
        failure_block = runner.split(
            'if [[ ${case_exit_code} -ne 0 ]]', maxsplit=1
        )[1].split("\n    fi\n", maxsplit=1)[0]

        self.assertIn("continue", failure_block)
        self.assertNotIn("exit ${", failure_block)
        self.assertTrue(runner.rstrip().endswith("exit ${REGRESSION_EXIT_CODE}"))

    def test_runner_prints_inventory_and_uses_progress_runner(self):
        runner = (
            REPO_ROOT / "test/jd-ci/pipeline/run_operator_regression.sh"
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

    def test_resolves_every_fixed_operator_case_without_groups(self):
        specs = resolve_operator_specs()

        self.assertEqual(len(specs), 8)
        self.assertEqual(
            {spec.role for spec in specs}, {"correctness", "performance"}
        )
        self.assertEqual(validate_operator_pairs(specs), [])

    def test_registry_contains_only_jd_test_paths(self):
        commands = "\n".join(" ".join(spec.command) for spec in resolve_operator_specs())

        self.assertIn("test/jd-ci/operators/", commands)
        self.assertNotIn("test/registered/", commands)
        self.assertNotIn("test/manual/", commands)
        self.assertNotIn("sgl-kernel/tests/", commands)

    def test_multi_gpu_requirement_is_retained(self):
        specs = {spec.name: spec for spec in resolve_operator_specs()}

        self.assertEqual(specs["jd-dp-allgather-correctness"].min_gpus, 2)
        self.assertEqual(specs["jd-dp-allgather-performance"].min_gpus, 2)

    def test_resolution_is_deterministic(self):
        first = [spec.name for spec in resolve_operator_specs()]
        second = [spec.name for spec in resolve_operator_specs()]

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
