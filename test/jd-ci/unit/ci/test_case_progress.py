import io
import sys
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PIPELINE_DIR = REPO_ROOT / "test/jd-ci/pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from case_progress import ProgressReporter, run_command


class TestProgressReporter(unittest.TestCase):
    def test_reporter_prints_start_heartbeat_and_pass(self):
        output = io.StringIO()
        reporter = ProgressReporter(
            area="CPU and Mock Regression",
            case_id="jd-example",
            index=1,
            total=2,
            assertion="example assertion",
            timeout_seconds=30,
            heartbeat_seconds=0.01,
            stream=output,
        )

        reporter.start("command")
        time.sleep(0.03)
        reporter.finish("PASS")

        text = output.getvalue()
        self.assertIn("[CASE 1/2][START] id=jd-example", text)
        self.assertIn("assertion=example assertion", text)
        self.assertIn("[CASE 1/2][RUNNING] id=jd-example phase=command", text)
        self.assertIn("elapsed=", text)
        self.assertIn("timeout=30s", text)
        self.assertIn("[CASE 1/2][PASS] id=jd-example duration=", text)

    def test_reporter_supports_server_lifecycle_without_counter(self):
        output = io.StringIO()
        reporter = ProgressReporter(
            area="Server and API Regression",
            case_id="jd-server",
            index=None,
            total=None,
            assertion="server becomes ready",
            timeout_seconds=10,
            action="SERVER",
            heartbeat_seconds=0.01,
            stream=output,
        )

        reporter.start("startup")
        reporter.set_phase("health-check")
        reporter.finish("PASS", detail="ready")

        text = output.getvalue()
        self.assertIn("[SERVER][START] id=jd-server", text)
        self.assertNotIn("[SERVER ", text)
        self.assertIn("detail=ready", text)


class TestRunCommand(unittest.TestCase):
    def _run(self, command, timeout_seconds=1.0):
        output = io.StringIO()
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        log_file = Path(temporary_directory.name) / "case.log"
        reporter = ProgressReporter(
            area="CPU and Mock Regression",
            case_id="jd-command",
            index=1,
            total=1,
            assertion="command contract",
            timeout_seconds=timeout_seconds,
            heartbeat_seconds=0.01,
            stream=output,
        )
        reporter.start("command")
        exit_code = run_command(
            command,
            reporter=reporter,
            log_file=log_file,
            timeout_seconds=timeout_seconds,
            kill_after_seconds=0.1,
        )
        reporter.finish("PASS" if exit_code == 0 else "FAIL", exit_code=exit_code)
        return exit_code, output.getvalue(), log_file.read_text(encoding="utf-8")

    def test_command_exit_zero_is_streamed_and_logged(self):
        exit_code, output, log = self._run(
            [sys.executable, "-c", "print('child-success', flush=True)"]
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("child-success", output)
        self.assertIn("child-success", log)

    def test_command_nonzero_exit_is_preserved(self):
        exit_code, output, log = self._run(
            [
                sys.executable,
                "-c",
                "import sys; print('child-failure', flush=True); sys.exit(7)",
            ]
        )

        self.assertEqual(exit_code, 7)
        self.assertIn("child-failure", output)
        self.assertIn("child-failure", log)

    def test_command_timeout_returns_124(self):
        exit_code, output, log = self._run(
            [
                sys.executable,
                "-c",
                "import time; print('child-timeout', flush=True); time.sleep(10)",
            ],
            timeout_seconds=0.05,
        )

        self.assertEqual(exit_code, 124)
        self.assertIn("child-timeout", output)
        self.assertIn("child-timeout", log)


if __name__ == "__main__":
    unittest.main()
