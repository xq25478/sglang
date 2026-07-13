import os
import subprocess
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
PARALLELISM_HELPER = REPO_ROOT / "test/jd-ci/env/sgl_kernel_parallelism.sh"


class TestSglKernelParallelism(CustomTestCase):
    def run_helper(self, cpu_count: int, **overrides: str) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env.pop("JD_CI_SGL_KERNEL_BUILD_MAX_JOBS", None)
        env.pop("JD_CI_SGL_KERNEL_NVCC_THREADS", None)
        env.update(overrides)
        return subprocess.run(
            ["bash", str(PARALLELISM_HELPER), str(cpu_count)],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

    def test_uses_all_192_cores_for_ninja(self):
        result = self.run_helper(192)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "192 1")

    def test_never_computes_zero_ninja_jobs(self):
        result = self.run_helper(1)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1 1")

    def test_explicit_overrides_are_preserved(self):
        result = self.run_helper(
            192,
            JD_CI_SGL_KERNEL_BUILD_MAX_JOBS="24",
            JD_CI_SGL_KERNEL_NVCC_THREADS="3",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "24 3")

    def test_rejects_non_positive_overrides(self):
        result = self.run_helper(192, JD_CI_SGL_KERNEL_NVCC_THREADS="0")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("JD_CI_SGL_KERNEL_NVCC_THREADS", result.stderr)


if __name__ == "__main__":
    unittest.main()
