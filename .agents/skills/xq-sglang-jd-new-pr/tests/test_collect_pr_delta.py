import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_pr_delta.py"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


class GitRepo:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        git(self.path, "init", "-q", "-b", "main")
        git(self.path, "config", "user.name", "JD CI Test")
        git(self.path, "config", "user.email", "jd-ci@example.com")
        self.write("README.md", "base\n")
        self.commit("base")

    def cleanup(self):
        self.temporary.cleanup()

    def write(self, relative: str, content: str):
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def commit(self, message: str) -> str:
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "-m", message)
        return git(self.path, "rev-parse", "HEAD")

    def write_manifest(self, *mapped_commits: str):
        commits = "\n    ".join(f"{commit!r}," for commit in mapped_commits)
        case_commits = " ".join(f"{commit!r}," for commit in mapped_commits)
        self.write(
            "test/jd-ci/jd_test_manifest.py",
            "INTERNAL_COMMITS: tuple[str, ...] = (\n"
            f"    {commits}\n"
            ")\n"
            "CASES = (\n"
            f"    JDCase(case_id='case', commits=({case_commits})),\n"
            ")\n",
        )


class TestCollectPRDelta(unittest.TestCase):
    def setUp(self):
        self.repo = GitRepo()

    def tearDown(self):
        self.repo.cleanup()

    def run_collector(self, *args: str, env: dict[str, str] | None = None):
        command = [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(self.repo.path),
            *args,
        ]
        return subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    def test_explicit_base_emits_commits_files_and_uncovered_sha(self):
        base = git(self.repo.path, "rev-parse", "HEAD")
        self.repo.write("python/sglang/jd_feature.py", "ENABLED = True\n")
        feature = self.repo.commit("add JD feature")

        result = self.run_collector("--base", base)

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["requested_base"], base)
        self.assertEqual(data["resolved_base"], base)
        self.assertEqual(data["merge_base"], base)
        self.assertEqual(data["head"], feature)
        self.assertEqual([item["sha"] for item in data["commits"]], [feature])
        self.assertEqual(data["uncovered_commits"], [feature])
        self.assertIn(
            {"status": "A", "path": "python/sglang/jd_feature.py"}, data["files"]
        )

    def test_invalid_explicit_base_is_rejected(self):
        result = self.run_collector("--base", "missing-JD-base")

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot resolve base ref", result.stderr)

    def test_ambiguous_nearest_jd_release_bases_are_rejected(self):
        root = git(self.repo.path, "rev-parse", "HEAD")
        git(self.repo.path, "checkout", "-q", "-b", "JD-v0.5.14")
        self.repo.write("left.txt", "left\n")
        left = self.repo.commit("left")
        git(self.repo.path, "checkout", "-q", "-b", "JD-v0.5.15", root)
        self.repo.write("right.txt", "right\n")
        self.repo.commit("right")
        git(self.repo.path, "checkout", "-q", "-b", "feature", left)
        git(self.repo.path, "merge", "-q", "--no-ff", "JD-v0.5.15", "-m", "merge")

        result = self.run_collector()

        self.assertEqual(result.returncode, 2)
        self.assertIn("ambiguous JD base", result.stderr)

    def test_manifest_mapped_commit_is_not_uncovered(self):
        base = git(self.repo.path, "rev-parse", "HEAD")
        self.repo.write("feature.py", "value = 1\n")
        feature = self.repo.commit("feature")
        self.repo.write_manifest(feature)

        result = self.run_collector("--base", base)

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["manifest"]["mapped_commits"], [feature])
        self.assertEqual(data["uncovered_commits"], [])

    def test_rename_is_preserved_in_file_status(self):
        self.repo.write("old_name.py", "value = 1\n")
        self.repo.commit("add old name")
        base = git(self.repo.path, "rev-parse", "HEAD")
        git(self.repo.path, "mv", "old_name.py", "new_name.py")
        self.repo.commit("rename file")

        result = self.run_collector("--base", base)

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertIn(
            {"status": "R100", "old_path": "old_name.py", "path": "new_name.py"},
            data["files"],
        )

    def test_empty_comparison_range_is_rejected(self):
        head = git(self.repo.path, "rev-parse", "HEAD")

        result = self.run_collector("--base", head)

        self.assertEqual(result.returncode, 2)
        self.assertIn("comparison range is empty", result.stderr)

    def test_merge_commit_lists_changes_against_first_parent(self):
        root = git(self.repo.path, "rev-parse", "HEAD")
        self.repo.write("left.txt", "left\n")
        left = self.repo.commit("left")
        git(self.repo.path, "checkout", "-q", "-b", "topic", root)
        self.repo.write("right.txt", "right\n")
        self.repo.commit("right")
        git(self.repo.path, "checkout", "-q", "main")
        git(self.repo.path, "merge", "-q", "--no-ff", "topic", "-m", "merge topic")

        result = self.run_collector("--base", left)

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        merge_commit = next(
            commit for commit in data["commits"] if commit["subject"] == "merge topic"
        )
        self.assertIn(
            {"status": "A", "path": "right.txt"}, merge_commit["files"]
        )


if __name__ == "__main__":
    unittest.main()
