import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jd_release_rebase.py"


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return result


class ReleaseRepo:
    def __init__(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        git(self.path, "init", "-q", "-b", "main")
        git(self.path, "config", "user.name", "JD Release Test")
        git(self.path, "config", "user.email", "jd-release@example.com")
        self.write("base.txt", "base\n")
        self.old_upstream = self.commit("community v1")
        git(self.path, "tag", "v1.0")

    def cleanup(self):
        listing = git(self.path, "worktree", "list", "--porcelain", check=False)
        paths = [
            Path(line.removeprefix("worktree "))
            for line in listing.stdout.splitlines()
            if line.startswith("worktree ")
        ]
        for path in paths:
            if path.resolve() != self.path.resolve():
                git(self.path, "worktree", "remove", "--force", str(path), check=False)
                shutil.rmtree(path, ignore_errors=True)
        self.temporary.cleanup()

    def write(self, relative: str, content: str):
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def commit(self, message: str) -> str:
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "-m", message)
        return git(self.path, "rev-parse", "HEAD").stdout.strip()

    def checkout(self, *args: str):
        git(self.path, "checkout", "-q", *args)

    def create_standard_history(self) -> dict[str, str]:
        self.checkout("-b", "JD-v1.0", "refs/tags/v1.0")
        self.write("jd_one.py", "JD_ONE = True\n")
        jd_one = self.commit("jd: add first feature")
        self.write("python/pyproject.toml", "[tool.jd]\nenabled = true\n")
        jd_two = self.commit("jd: add second feature")
        old_internal = git(self.path, "rev-parse", "HEAD").stdout.strip()

        self.checkout("main")
        self.write("upstream.py", "UPSTREAM = 2\n")
        new_upstream = self.commit("community v2")
        git(self.path, "tag", "v2.0")
        return {
            "jd_one": jd_one,
            "jd_two": jd_two,
            "old_internal": old_internal,
            "new_upstream": new_upstream,
        }


class TestJDReleaseRebase(unittest.TestCase):
    def setUp(self):
        self.repo = ReleaseRepo()

    def tearDown(self):
        self.repo.cleanup()

    def run_tool(self, *args: str):
        env = os.environ.copy()
        for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            env.pop(name, None)
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    def plan(self, output: Path, **overrides: str):
        values = {
            "old_internal": "refs/heads/JD-v1.0",
            "old_upstream": "refs/tags/v1.0",
            "new_upstream": "refs/tags/v2.0",
            "new_internal": "JD-v2.0",
        }
        values.update(overrides)
        return self.run_tool(
            "plan",
            "--repo",
            str(self.repo.path),
            "--old-internal",
            values["old_internal"],
            "--old-upstream",
            values["old_upstream"],
            "--new-upstream",
            values["new_upstream"],
            "--new-internal",
            values["new_internal"],
            "--output",
            str(output),
        )

    def test_plan_preserves_original_order_and_marks_high_risk_paths(self):
        history = self.repo.create_standard_history()
        output = self.repo.path / "plan.json"

        result = self.plan(output)

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(output.read_text())
        self.assertEqual(data["old_upstream"]["sha"], self.repo.old_upstream)
        self.assertEqual(data["old_internal"]["sha"], history["old_internal"])
        self.assertEqual(data["new_upstream"]["sha"], history["new_upstream"])
        self.assertEqual(
            [item["sha"] for item in data["replay_commits"]],
            [history["jd_one"], history["jd_two"]],
        )
        self.assertEqual(data["absorbed_commits"], [])
        self.assertIn("python/pyproject.toml", data["high_risk_paths"])
        self.assertEqual(data["new_internal"]["name"], "JD-v2.0")

    def test_plan_defers_every_commit_touching_jd_ci_as_one_final_unit(self):
        self.repo.checkout("-b", "JD-v1.0", "refs/tags/v1.0")
        self.repo.write("production.py", "VALUE = 1\n")
        production = self.repo.commit("jd: production repair")
        self.repo.write("test/jd-ci/run_jd_ci.sh", "#!/bin/bash\n")
        self.repo.write("sgl-kernel/CMakeLists.txt", "set(JD_CI_BUILD ON)\n")
        mixed_ci = self.repo.commit("ci: add JD build workflow")
        self.repo.write(
            ".agents/skills/xq-sglang-jd-release-rebase/SKILL.md",
            "release skill\n",
        )
        skill_ci = self.repo.commit("ci: update JD release skill")
        self.repo.checkout("main")
        self.repo.write("upstream.py", "UPSTREAM = 2\n")
        self.repo.commit("community v2")
        git(self.repo.path, "tag", "v2.0")
        output = self.repo.path / "plan.json"

        result = self.plan(output)

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(output.read_text())
        self.assertEqual(
            [item["sha"] for item in data["replay_commits"]], [production]
        )
        self.assertEqual(
            [item["sha"] for item in data["deferred_ci_commits"]],
            [mixed_ci, skill_ci],
        )
        self.assertIn("sgl-kernel/CMakeLists.txt", data["deferred_ci_commits"][0]["files"])

    def test_prepare_and_commit_ci_creates_one_head_commit(self):
        self.repo.checkout("-b", "JD-v1.0", "refs/tags/v1.0")
        self.repo.write("production.py", "VALUE = 1\n")
        production = self.repo.commit("jd: production repair")
        self.repo.write("test/jd-ci/run_jd_ci.sh", "#!/bin/bash\n")
        first_ci = self.repo.commit("ci: add JD runner")
        self.repo.write("test/jd-ci/README.md", "JD CI\n")
        second_ci = self.repo.commit("ci: document JD runner")
        self.repo.checkout("main")
        self.repo.write("upstream.py", "UPSTREAM = 2\n")
        new_upstream = self.repo.commit("community v2")
        git(self.repo.path, "tag", "v2.0")
        plan_file = self.repo.path / "plan.json"
        state_file = self.repo.path / "state.json"
        report_file = self.repo.path / "report.json"
        worktree = self.repo.path / "release-worktree"
        self.assertEqual(self.plan(plan_file).returncode, 0)
        self.assertEqual(
            self.run_tool(
                "execute",
                "--plan",
                str(plan_file),
                "--worktree",
                str(worktree),
                "--state-file",
                str(state_file),
            ).returncode,
            0,
        )

        prepare = self.run_tool("prepare-ci", "--state-file", str(state_file))
        self.assertEqual(prepare.returncode, 0, prepare.stderr)
        state = json.loads(state_file.read_text())
        self.assertEqual(state["status"], "ci-prepared")
        self.assertEqual(git(worktree, "rev-parse", "HEAD").stdout.strip(), state["mappings"][production])
        production_new_sha = state["mappings"][production]
        (worktree / "test/jd-ci/jd_test_manifest.py").write_text(
            f"INTERNAL_COMMITS = ({production_new_sha!r},)\n"
            f"CASES = (JDCase(commits=({production_new_sha!r},)),)\n",
            encoding="utf-8",
        )
        git(worktree, "add", "test/jd-ci/jd_test_manifest.py")

        commit = self.run_tool(
            "commit-ci",
            "--state-file",
            str(state_file),
            "--message",
            "ci: migrate JD CI for v2.0",
        )
        self.assertEqual(commit.returncode, 0, commit.stderr)
        state = json.loads(state_file.read_text())
        head = git(worktree, "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["new_head"], head)
        self.assertEqual(state["mappings"][first_ci], head)
        self.assertEqual(state["mappings"][second_ci], head)
        self.assertEqual(
            git(worktree, "rev-list", "--count", f"{new_upstream}..HEAD").stdout.strip(),
            "2",
        )
        self.assertEqual(
            git(worktree, "log", "-1", "--format=%s").stdout.strip(),
            "ci: migrate JD CI for v2.0",
        )

        checked = self.run_tool(
            "check", "--state-file", str(state_file), "--output", str(report_file)
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        report = json.loads(report_file.read_text())
        self.assertEqual(report["jd_ci_commit_count"], 1)
        self.assertTrue(report["jd_ci_commit_is_head"])

    def test_plan_marks_patch_already_absorbed_by_new_upstream(self):
        self.repo.checkout("-b", "JD-v1.0", "refs/tags/v1.0")
        self.repo.write("absorbed.py", "VALUE = 1\n")
        old_patch = self.repo.commit("jd: patch later accepted upstream")
        self.repo.checkout("main")
        self.repo.write("absorbed.py", "VALUE = 1\n")
        self.repo.commit("community: accept JD patch")
        git(self.repo.path, "tag", "v2.0")
        output = self.repo.path / "plan.json"

        result = self.plan(output)

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(output.read_text())
        self.assertEqual(data["replay_commits"], [])
        self.assertEqual([item["sha"] for item in data["absorbed_commits"]], [old_patch])

    def test_plan_marks_exact_commits_reachable_from_new_upstream_as_absorbed(self):
        self.repo.checkout("-b", "JD-v1.0", "refs/tags/v1.0")
        self.repo.write("included.py", "VALUE = 1\n")
        old_patch = self.repo.commit("jd: commit included exactly")
        self.repo.checkout("main")
        git(self.repo.path, "merge", "-q", "--no-ff", "JD-v1.0", "-m", "include JD")
        git(self.repo.path, "tag", "v2.0")
        output = self.repo.path / "plan.json"

        result = self.plan(output)

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(output.read_text())
        self.assertEqual(data["replay_commits"], [])
        self.assertEqual([item["sha"] for item in data["absorbed_commits"]], [old_patch])

    def test_plan_reports_internal_merge_commits_for_audit(self):
        self.repo.checkout("-b", "topic", "refs/tags/v1.0")
        self.repo.write("topic.py", "TOPIC = True\n")
        self.repo.commit("jd: topic feature")
        self.repo.checkout("-b", "JD-v1.0", "refs/tags/v1.0")
        self.repo.write("direct.py", "DIRECT = True\n")
        self.repo.commit("jd: direct feature")
        git(self.repo.path, "merge", "-q", "--no-ff", "topic", "-m", "merge JD topic")
        merge_sha = git(self.repo.path, "rev-parse", "HEAD").stdout.strip()
        self.repo.checkout("main")
        self.repo.write("upstream.py", "UPSTREAM = 2\n")
        self.repo.commit("community v2")
        git(self.repo.path, "tag", "v2.0")
        output = self.repo.path / "plan.json"

        result = self.plan(output)

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(output.read_text())
        self.assertEqual(
            [item["sha"] for item in data["audit_merge_commits"]], [merge_sha]
        )
        self.assertEqual(data["audit_merge_commits"][0]["subject"], "merge JD topic")

    def test_short_ref_is_rejected_when_branch_and_tag_differ(self):
        history = self.repo.create_standard_history()
        git(self.repo.path, "tag", "JD-v1.0", history["new_upstream"])
        output = self.repo.path / "plan.json"

        result = self.plan(output, old_internal="JD-v1.0")

        self.assertEqual(result.returncode, 2)
        self.assertIn("ambiguous ref", result.stderr)

    def test_plan_rejects_existing_new_internal_branch(self):
        self.repo.create_standard_history()
        git(self.repo.path, "branch", "JD-v2.0", "refs/tags/v2.0")
        output = self.repo.path / "plan.json"

        result = self.plan(output)

        self.assertEqual(result.returncode, 2)
        self.assertIn("new internal ref already exists", result.stderr)

    def test_classify_moves_semantic_upstream_fix_out_of_replay(self):
        history = self.repo.create_standard_history()
        plan_file = self.repo.path / "plan.json"
        self.assertEqual(self.plan(plan_file).returncode, 0)

        result = self.run_tool(
            "classify",
            "--plan",
            str(plan_file),
            "--absorbed",
            history["jd_one"],
            "--reason",
            "upstream v2 implements the same behavior with a different patch",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(plan_file.read_text())
        self.assertEqual(
            [item["sha"] for item in data["replay_commits"]], [history["jd_two"]]
        )
        semantic = next(
            item for item in data["absorbed_commits"] if item["sha"] == history["jd_one"]
        )
        self.assertEqual(semantic["classification"], "absorbed-semantic")
        self.assertEqual(
            semantic["absorption_reason"],
            "upstream v2 implements the same behavior with a different patch",
        )

    def test_annotated_upstream_tag_resolves_to_commit_sha(self):
        history = self.repo.create_standard_history()
        git(self.repo.path, "tag", "-a", "v2.0-annotated", "-m", "annotated v2")
        output = self.repo.path / "plan.json"

        result = self.plan(output, new_upstream="refs/tags/v2.0-annotated")

        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(output.read_text())
        self.assertEqual(data["new_upstream"]["sha"], history["new_upstream"])

    def test_execute_creates_isolated_branch_and_complete_sha_mapping(self):
        history = self.repo.create_standard_history()
        plan_file = self.repo.path / "plan.json"
        state_file = self.repo.path / "state.json"
        worktree = self.repo.path / "release-worktree"
        self.assertEqual(self.plan(plan_file).returncode, 0)

        result = self.run_tool(
            "execute",
            "--plan",
            str(plan_file),
            "--worktree",
            str(worktree),
            "--state-file",
            str(state_file),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads(state_file.read_text())
        self.assertEqual(state["status"], "completed")
        self.assertEqual(set(state["mappings"]), {history["jd_one"], history["jd_two"]})
        new_head = git(worktree, "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(state["new_head"], new_head)
        self.assertEqual(git(worktree, "branch", "--show-current").stdout.strip(), "JD-v2.0")
        self.assertEqual(
            git(self.repo.path, "rev-parse", "refs/heads/JD-v1.0").stdout.strip(),
            history["old_internal"],
        )
        self.assertEqual(
            git(
                worktree,
                "merge-base",
                "--is-ancestor",
                history["new_upstream"],
                "HEAD",
            ).returncode,
            0,
        )

    def test_conflict_is_persisted_and_resume_completes_replay(self):
        self.repo.write("config.txt", "base\n")
        self.repo.commit("add shared config")
        git(self.repo.path, "tag", "-f", "v1.0")
        self.repo.checkout("-b", "JD-v1.0", "refs/tags/v1.0")
        self.repo.write("config.txt", "jd\n")
        old_commit = self.repo.commit("jd: change config")
        self.repo.checkout("main")
        self.repo.write("config.txt", "upstream\n")
        self.repo.commit("community: change config")
        git(self.repo.path, "tag", "v2.0")
        plan_file = self.repo.path / "plan.json"
        state_file = self.repo.path / "state.json"
        worktree = self.repo.path / "conflict-worktree"
        self.assertEqual(self.plan(plan_file).returncode, 0)

        execute = self.run_tool(
            "execute",
            "--plan",
            str(plan_file),
            "--worktree",
            str(worktree),
            "--state-file",
            str(state_file),
        )

        self.assertEqual(execute.returncode, 3, execute.stderr)
        state = json.loads(state_file.read_text())
        self.assertEqual(state["status"], "conflict")
        self.assertEqual(state["pending_commit"], old_commit)
        self.assertEqual(state["conflict_files"], ["config.txt"])

        (worktree / "config.txt").write_text("resolved\n", encoding="utf-8")
        git(worktree, "add", "config.txt")
        resume = self.run_tool("resume", "--state-file", str(state_file))

        self.assertEqual(resume.returncode, 0, resume.stderr)
        state = json.loads(state_file.read_text())
        self.assertEqual(state["status"], "completed")
        self.assertIn(old_commit, state["mappings"])
        self.assertEqual((worktree / "config.txt").read_text(), "resolved\n")

    def test_check_reports_and_clears_manifest_sha_migration(self):
        history = self.repo.create_standard_history()
        plan_file = self.repo.path / "plan.json"
        state_file = self.repo.path / "state.json"
        worktree = self.repo.path / "check-worktree"
        self.assertEqual(self.plan(plan_file).returncode, 0)
        self.assertEqual(
            self.run_tool(
                "execute",
                "--plan",
                str(plan_file),
                "--worktree",
                str(worktree),
                "--state-file",
                str(state_file),
            ).returncode,
            0,
        )
        state = json.loads(state_file.read_text())
        old_shas = list(state["mappings"])
        new_shas = list(state["mappings"].values())
        manifest = worktree / "test/jd-ci/jd_test_manifest.py"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "INTERNAL_COMMITS = (\n"
            + "".join(f"    {sha!r},\n" for sha in old_shas)
            + ")\nCASES = (\n"
            + "".join(f"    JDCase(commits=({sha!r},)),\n" for sha in old_shas)
            + ")\n",
            encoding="utf-8",
        )
        first_report = self.repo.path / "check-old.json"

        first = self.run_tool(
            "check", "--state-file", str(state_file), "--output", str(first_report)
        )

        self.assertEqual(first.returncode, 4, first.stderr)
        report = json.loads(first_report.read_text())
        self.assertEqual(report["manifest_stale_old_shas"], sorted(old_shas))
        self.assertEqual(report["manifest_missing_new_shas"], sorted(new_shas))

        manifest.write_text(
            "INTERNAL_COMMITS = (\n"
            + "".join(f"    {sha!r},\n" for sha in new_shas)
            + ")\nCASES = (\n"
            + "".join(f"    JDCase(commits=({sha!r},)),\n" for sha in new_shas)
            + ")\n",
            encoding="utf-8",
        )
        second_report = self.repo.path / "check-new.json"
        second = self.run_tool(
            "check", "--state-file", str(state_file), "--output", str(second_report)
        )

        self.assertEqual(second.returncode, 0, second.stderr)
        report = json.loads(second_report.read_text())
        self.assertEqual(report["manifest_stale_old_shas"], [])
        self.assertEqual(report["manifest_missing_new_shas"], [])
        self.assertTrue(report["ancestry_ok"])
        self.assertTrue(report["mapping_complete"])

    def test_check_rejects_absorbed_old_sha_left_in_manifest(self):
        self.repo.checkout("-b", "JD-v1.0", "refs/tags/v1.0")
        self.repo.write("absorbed.py", "VALUE = 1\n")
        old_patch = self.repo.commit("jd: upstream-bound patch")
        self.repo.checkout("main")
        self.repo.write("absorbed.py", "VALUE = 1\n")
        self.repo.commit("community: absorb patch")
        git(self.repo.path, "tag", "v2.0")
        plan_file = self.repo.path / "plan.json"
        state_file = self.repo.path / "state.json"
        worktree = self.repo.path / "absorbed-worktree"
        self.assertEqual(self.plan(plan_file).returncode, 0)
        self.assertEqual(
            self.run_tool(
                "execute",
                "--plan",
                str(plan_file),
                "--worktree",
                str(worktree),
                "--state-file",
                str(state_file),
            ).returncode,
            0,
        )
        manifest = worktree / "test/jd-ci/jd_test_manifest.py"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            f"INTERNAL_COMMITS = ({old_patch!r},)\n"
            f"CASES = (JDCase(commits=({old_patch!r},)),)\n",
            encoding="utf-8",
        )
        report_file = self.repo.path / "absorbed-check.json"

        result = self.run_tool(
            "check", "--state-file", str(state_file), "--output", str(report_file)
        )

        self.assertEqual(result.returncode, 4, result.stderr)
        report = json.loads(report_file.read_text())
        self.assertEqual(report["manifest_stale_old_shas"], [old_patch])
        self.assertEqual(report["manifest_missing_new_shas"], [])


if __name__ == "__main__":
    unittest.main()
