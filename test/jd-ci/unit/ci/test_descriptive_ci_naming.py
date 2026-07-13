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


REPO_ROOT = Path(__file__).resolve().parents[4]
DESIGN_DOC = "docs/superpowers/specs/2026-07-11-jd-ci-cumulative-internal-regression-design.md"
PLAN_DOC = "docs/superpowers/plans/2026-07-11-jd-ci-cumulative-internal-regression.md"

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDescriptiveCINaming(CustomTestCase):
    def test_legacy_ci_assets_are_removed(self):
        legacy_paths = (
            "test/jd-ci/accuracy",
            "test/jd-ci/ci_logs",
            "test/jd-ci/data/eval",
            "test/jd-ci/functional",
            "test/jd-ci/test_func",
            "test/jd-ci/test_serve",
            "test/jd-ci/pipeline/run_p1_accuracy.sh",
            "test/jd-ci/pipeline/run_p2_functional.sh",
            "test/jd-ci/jd_ci_base.py",
            "test/jd-ci/jd_ci_gpu_cleanup.py",
            "test/jd-ci/jd_ci_plugins.py",
            "test/jd-ci/jd_ci_registry.py",
            "test/jd-ci/report/generate_ci_report.py",
            "test/jd-ci/report/print_failure_summary.py",
            "test/jd-ci/tmp.lock",
        )

        tracked = subprocess.check_output(
            ["git", "ls-files", "test/jd-ci"],
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()
        offenders = [
            relative
            for relative in tracked
            if (REPO_ROOT / relative).exists()
            and any(
                relative == legacy or relative.startswith(f"{legacy}/")
                for legacy in legacy_paths
            )
        ]

        self.assertEqual(offenders, [])

    def test_jd_owned_tests_do_not_live_under_registered(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "test/registered/unit/jd"],
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()

        self.assertEqual(tracked, [])

    def test_tracked_jd_ci_files_use_descriptive_names(self):
        forbidden_terms = ("sta" + "ge", "阶" + "段")
        tracked = subprocess.check_output(
            [
                "git",
                "ls-files",
                "test/jd-ci",
                DESIGN_DOC,
                PLAN_DOC,
            ],
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()

        offenders = []
        for relative in tracked:
            path = REPO_ROOT / relative
            if not path.is_file():
                continue
            if any(term in relative.casefold() for term in forbidden_terms):
                offenders.append(relative)
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(term in text.casefold() for term in forbidden_terms):
                offenders.append(relative)

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
