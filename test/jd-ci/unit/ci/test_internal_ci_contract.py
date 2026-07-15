import os
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


class TestJDInternalCIContract(unittest.TestCase):
    @staticmethod
    def _script() -> str:
        return (REPO_ROOT / "test/jd-ci/run_jd_ci.sh").read_text(encoding="utf-8")

    def _run_argument_prologue(self, *arguments, **environment):
        prologue = self._script().split("CI_SCRIPT_PATH=", maxsplit=1)[0]
        env = os.environ.copy()
        for name in (
            "JD_CI_SKIP_MOONCAKE_BUILD",
            "JD_CI_SKIP_SGL_KERNEL_BUILD",
            "JD_CI_SKIP_TEST",
        ):
            env.pop(name, None)
        env.update(environment)
        return subprocess.run(
            ["bash", "-s", "--", *arguments],
            input=prologue,
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

    def test_short_long_and_legacy_options_select_the_expected_mode(self):
        cases = (
            ((), "review", "note__merge_request", "1", "0"),
            (("-r",), "review", "note__merge_request", "1", "0"),
            (("--review",), "review", "note__merge_request", "1", "0"),
            (
                ("note__merge_request",),
                "review",
                "note__merge_request",
                "1",
                "0",
            ),
            (("-m",), "merge", "merge_request__merged", "0", "1"),
            (("--merge",), "merge", "merge_request__merged", "0", "1"),
            (
                ("merge_request__merged",),
                "merge",
                "merge_request__merged",
                "0",
                "1",
            ),
            (("-t",), "temp-image", "temp_image", "1", "1"),
            (("--temp-image",), "temp-image", "temp_image", "1", "1"),
        )

        for arguments, mode, event_type, run_tests, publish_images in cases:
            with self.subTest(arguments=arguments):
                result = self._run_argument_prologue(*arguments)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"CI_MODE={mode}", result.stdout)
                self.assertIn(f"EVENT_TYPE={event_type}", result.stdout)
                self.assertIn(f"RUN_CI_TESTS={run_tests}", result.stdout)
                self.assertIn(f"PUBLISH_IMAGES={publish_images}", result.stdout)

    def test_help_lists_modes_compatibility_and_temporary_options(self):
        for option in ("-h", "--help"):
            with self.subTest(option=option):
                result = self._run_argument_prologue(option)

                self.assertEqual(result.returncode, 0, result.stderr)
                for expected in (
                    "-r, --review",
                    "-m, --merge",
                    "-t, --temp-image",
                    "note__merge_request",
                    "merge_request__merged",
                    "JD_CI_SKIP_SGL_KERNEL_BUILD",
                    "JD_CI_SKIP_MOONCAKE_BUILD",
                    "JD_CI_SKIP_TEST",
                ):
                    self.assertIn(expected, result.stdout)

    def test_unknown_or_extra_arguments_exit_with_usage_error(self):
        for arguments in (("--unknown",), ("-r", "unexpected")):
            with self.subTest(arguments=arguments):
                result = self._run_argument_prologue(*arguments)

                self.assertEqual(result.returncode, 2)
                self.assertIn("用法:", result.stderr)

    def test_temporary_options_only_accept_binary_values(self):
        for name in (
            "JD_CI_SKIP_SGL_KERNEL_BUILD",
            "JD_CI_SKIP_MOONCAKE_BUILD",
            "JD_CI_SKIP_TEST",
        ):
            with self.subTest(name=name):
                result = self._run_argument_prologue("-t", **{name: "2"})

                self.assertEqual(result.returncode, 2)
                self.assertIn(name, result.stderr)

    def test_formal_modes_reject_temporary_skip_options(self):
        for mode in ("-r", "note__merge_request", "-m", "merge_request__merged"):
            for name in (
                "JD_CI_SKIP_SGL_KERNEL_BUILD",
                "JD_CI_SKIP_MOONCAKE_BUILD",
                "JD_CI_SKIP_TEST",
            ):
                with self.subTest(mode=mode, name=name):
                    result = self._run_argument_prologue(mode, **{name: "1"})

                    self.assertEqual(result.returncode, 2)
                    self.assertIn("仅允许在 -t", result.stderr)

    def test_temp_mode_allows_explicit_skips(self):
        result = self._run_argument_prologue(
            "-t",
            JD_CI_SKIP_SGL_KERNEL_BUILD="1",
            JD_CI_SKIP_MOONCAKE_BUILD="1",
            JD_CI_SKIP_TEST="1",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RUN_CI_TESTS=0", result.stdout)

    def test_build_components_are_present_once(self):
        script = (REPO_ROOT / "test/jd-ci/run_jd_ci.sh").read_text(encoding="utf-8")

        self.assertEqual(script.count("'${MOONCAKE_TE_WORK_DIR}/compile' 'te'"), 1)
        self.assertEqual(
            script.count("'${MOONCAKE_STORE_WORK_DIR}/compile' 'store'"), 1
        )
        self.assertEqual(
            script.count("bash '${SOURCE_PATH}/test/jd-ci/env/build_sgl_kernel.sh'"),
            1,
        )

    def test_sgl_kernel_build_is_enabled_by_default(self):
        script = (REPO_ROOT / "test/jd-ci/run_jd_ci.sh").read_text(encoding="utf-8")

        self.assertIn(
            'JD_CI_SKIP_SGL_KERNEL_BUILD="${JD_CI_SKIP_SGL_KERNEL_BUILD:-0}"',
            script,
        )

    def test_failure_log_dump_and_artifact_isolation_remain_wired(self):
        script = (REPO_ROOT / "test/jd-ci/run_jd_ci.sh").read_text(encoding="utf-8")

        self.assertIn("dump_ci_logs.py", script)
        self.assertIn("CI_ARTIFACT_ROOT", script)
        self.assertIn("RELEASE_ARTIFACT_BRANCH", script)

    def test_modes_select_formal_or_commit_scoped_artifacts(self):
        script = self._script()

        self.assertIn('case "${CI_MODE}" in', script)
        self.assertIn("review|merge)", script)
        self.assertIn("temp-image)", script)
        self.assertIn(
            'JD_CI_TEMP_ARTIFACT_ROOT="${CI_RUNNER_ROOT}/artifacts"',
            script,
        )
        self.assertNotIn("persistent-reuse", script)
        self.assertIn('if [[ "${CI_MODE}" == "merge" ]]; then', script)
        self.assertIn("MOONCAKE_REQUIRE_CACHE=1", script)

    def test_merge_reuses_release_cache_on_any_branch_and_temp_stays_non_release(self):
        script = self._script()
        readme = (REPO_ROOT / "test/jd-ci/README.md").read_text(encoding="utf-8")

        self.assertNotIn(
            'if [[ "${CI_MODE}" == "merge" '
            '&& "${BRANCH_NAME}" != "${RELEASE_ARTIFACT_BRANCH}" ]]; then',
            script,
        )
        self.assertIn("任意分支都可以复用对应版本主分支的正式缓存", script)
        for required in (
            "任意分支",
            "对应版本主分支的正式缓存",
            "cache miss",
            "不允许回退到源码编译",
            "当前 commit",
        ):
            with self.subTest(required=required):
                self.assertIn(required, readme)
        self.assertIn(
            'if [[ "${CI_MODE}" == "temp-image" '
            '&& "${BRANCH_NAME}" == "${RELEASE_ARTIFACT_BRANCH}" ]]; then',
            script,
        )

    def test_merge_requires_sgl_kernel_cache(self):
        script = (REPO_ROOT / "test/jd-ci/env/build_sgl_kernel.sh").read_text(
            encoding="utf-8"
        )

        merge_condition = 'if [[ "${EVENT_TYPE}" == "merge_request__merged" ]]'
        self.assertIn(merge_condition, script)
        self.assertIn("SGL-Kernel wheel cache miss", script)
        merge_policy = script.split(merge_condition, maxsplit=1)[1]
        self.assertLess(
            merge_policy.index("SGL-Kernel wheel cache miss"),
            merge_policy.index("make -C"),
        )

    def test_temp_sgl_kernel_build_reuses_release_fetchcontent_deps(self):
        script = self._script()

        self.assertIn(
            "'${PERSISTENT_SGL_KERNEL_CACHE_HOST}' 2>&1 | tee '${SGL_KERNEL_BUILD_LOG}'",
            script,
        )
        self.assertIn(
            'WHEEL_CACHE_HOST="${JD_CI_TEMP_ARTIFACT_ROOT}/sgl-kernel/${BASE_IMAGE_TAG}"',
            script,
        )

    def test_review_event_cannot_publish_an_image(self):
        script = self._script()

        self.assertIn("PUBLISH_IMAGES=0", script)
        self.assertIn('docker push "${CLOUD_IMAGE}"', script)

    def test_temp_image_tag_component_is_sanitized_and_bounded(self):
        prologue = self._script().split("CI_SCRIPT_PATH=", maxsplit=1)[0]
        prologue += """
value=$(sanitize_docker_tag_component 'Feature/ABC @@@ 123')
printf 'SANITIZED=%s\\n' "${value}"
printf 'LENGTH=%s\\n' "${#value}"
"""

        result = subprocess.run(
            ["bash", "-s", "--", "-t"],
            input=prologue,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SANITIZED=feature-abc-123", result.stdout)
        length_line = next(
            line for line in result.stdout.splitlines() if line.startswith("LENGTH=")
        )
        self.assertLessEqual(int(length_line.split("=", maxsplit=1)[1]), 48)

    def test_temp_test_skip_writes_explicit_reports(self):
        script = self._script()

        self.assertIn("if [[ '${RUN_CI_TESTS}' == '1' ]]; then", script)
        self.assertIn("elif [[ '${CI_MODE}' == 'temp-image' ]]; then", script)
        self.assertIn("regression_report.py", script)
        self.assertIn("JD_CI_SKIP_TEST=1 in temp-image mode", script)
        self.assertNotRegex(script, r"(?<!JD_CI_)SKIP_CI_TEST")
        self.assertIn("for test_area in cpu_mock server_api operator", script)

    def test_images_publish_only_after_both_containers_succeed(self):
        script = self._script()
        store_status = script.index("STORE_EXIT_CODE=")
        first_commit = script.index("docker commit")
        publish_condition = (
            'if [[ ${EXIT_CODE} -eq 0 && "${PUBLISH_IMAGES}" == "1" ]]'
        )
        self.assertIn(publish_condition, script)
        publish_gate = script.index(publish_condition)

        self.assertLess(store_status, publish_gate)
        self.assertLess(publish_gate, first_commit)
        self.assertEqual(script.count("docker commit"), 2)
        self.assertEqual(script.count("docker push"), 2)

    def test_temp_images_have_branch_and_commit_scoped_tags(self):
        script = self._script()

        self.assertIn(
            "${BASE_IMAGE_TAG}_JD_TMP_${BRANCH_NAME_FOR_DOCKER}_${COMMIT_ID}",
            script,
        )
        self.assertIn(
            "${MSTORE_IMAGE_TAG}_JD_TMP_${BRANCH_NAME_FOR_DOCKER}_${COMMIT_ID}",
            script,
        )

    def test_obsolete_priority_skip_switches_are_removed(self):
        script = (REPO_ROOT / "test/jd-ci/run_jd_ci.sh").read_text(encoding="utf-8")

        for priority in ("P1", "P2"):
            self.assertNotIn(f"JD_CI_SKIP_{priority}", script)

    def test_review_event_always_runs_every_regression_without_skip_switches(self):
        script = (REPO_ROOT / "test/jd-ci/run_jd_ci.sh").read_text(encoding="utf-8")
        tracked_jd_ci_files = subprocess.run(
            ["git", "ls-files", "test/jd-ci"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()

        for area in ("CPU_MOCK", "SERVER_API", "OPERATOR"):
            switch = f"JD_CI_SKIP_{area}_REGRESSION"
            for relative_path in tracked_jd_ci_files:
                path = REPO_ROOT / relative_path
                if path.is_file() and path.suffix in {".md", ".py", ".sh"}:
                    self.assertNotIn(switch, path.read_text(encoding="utf-8"))

        for runner in (
            "run_cpu_mock_regression.sh",
            "run_server_api_regression.sh",
            "run_operator_regression.sh",
        ):
            self.assertEqual(script.count(f"pipeline/{runner}"), 1)
        self.assertNotIn("前一回归项失败", script)

    def test_runner_workspace_uses_nine_character_commit_id(self):
        script = self._script()

        self.assertIn("COMMIT_SHA=$(git rev-parse HEAD)", script)
        self.assertIn('COMMIT_ID="${COMMIT_SHA:0:9}"', script)
        self.assertIn('CI_RUNNER_ID="${COMMIT_ID}"', script)
        self.assertIn(
            'CI_RUNNER_ROOT="${CI_ARTIFACT_ROOT}/runners/${CI_RUNNER_ID}"',
            script,
        )
        self.assertIn('CI_LOGS_DIR="${CI_RUNNER_ROOT}/logs"', script)

    def test_component_logs_and_workspaces_are_isolated(self):
        script = self._script()

        for required in (
            'MAIN_PIPELINE_LOG="${CI_CONTAINER_LOGS_DIR}/sglang.log"',
            'MSTORE_PIPELINE_LOG="${CI_CONTAINER_LOGS_DIR}/mooncake-store.log"',
            'SGL_KERNEL_BUILD_LOG="${CI_BUILD_LOGS_DIR}/sgl-kernel.log"',
            'MOONCAKE_TE_BUILD_LOG="${CI_BUILD_LOGS_DIR}/mooncake-te.log"',
            'MOONCAKE_STORE_BUILD_LOG="${CI_BUILD_LOGS_DIR}/mooncake-store.log"',
            'MAIN_CONTAINER_WORK_DIR="${CI_RUNNER_WORK_DIR}/containers/sglang"',
            'MSTORE_CONTAINER_WORK_DIR="${CI_RUNNER_WORK_DIR}/containers/mooncake-store"',
            'SGL_KERNEL_WORK_DIR="${CI_RUNNER_WORK_DIR}/builds/sgl-kernel"',
            'MOONCAKE_TE_WORK_DIR="${CI_RUNNER_WORK_DIR}/builds/mooncake-te"',
            'MOONCAKE_STORE_WORK_DIR="${CI_RUNNER_WORK_DIR}/builds/mooncake-store"',
            'CPU_MOCK_TEST_WORK_DIR="${CI_RUNNER_WORK_DIR}/tests/cpu-mock"',
            'SERVER_API_TEST_WORK_DIR="${CI_RUNNER_WORK_DIR}/tests/server-api"',
            'OPERATOR_TEST_WORK_DIR="${CI_RUNNER_WORK_DIR}/tests/operator"',
            '-v "${MAIN_CONTAINER_TMP_DIR}:/tmp"',
            '-v "${MSTORE_CONTAINER_TMP_DIR}:/tmp"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, script)

        self.assertNotIn('CI_TMP_DIR="${CI_ARTIFACT_ROOT}/tmp/"', script)
        self.assertNotIn('-v "${CI_TMP_DIR}:/tmp"', script)
        self.assertNotIn("/tmp/* /tmp/.[!.]* /tmp/..?*", script)

    def test_mooncake_clone_roots_exist_before_container_builds(self):
        script = self._script()
        setup = script.split("cleanup_ci_runner_dir \"启动前清理\"", maxsplit=1)[
            1
        ].split("# 构建镜像信息", maxsplit=1)[0]

        self.assertIn('"${MOONCAKE_TE_WORK_DIR}/compile"', setup)
        self.assertIn('"${MOONCAKE_STORE_WORK_DIR}/compile"', setup)

    def test_exit_cleanup_removes_containers_before_runner_workspace(self):
        script = self._script()
        cleanup = script.split("cleanup_on_exit() {", maxsplit=1)[1].split(
            "trap cleanup_on_exit EXIT", maxsplit=1
        )[0]

        main_cleanup = cleanup.index(
            'cleanup_container_by_name "${CONTAINER_NAME}" "主容器"'
        )
        store_cleanup = cleanup.index(
            'cleanup_container_by_name "${MSTORE_CONTAINER}" "mooncake-store 容器"'
        )
        runner_cleanup = cleanup.index('cleanup_ci_runner_dir "收尾清理"')
        self.assertLess(main_cleanup, runner_cleanup)
        self.assertLess(store_cleanup, runner_cleanup)
        self.assertIn('CI_RUNNER_ROOT="${CI_ARTIFACT_ROOT}/runners/', script)
        self.assertIn("rm -rf \"${CI_RUNNER_ROOT}\"", script)

    def test_readme_documents_ephemeral_runner_directory(self):
        readme = (REPO_ROOT / "test/jd-ci/README.md").read_text(encoding="utf-8")

        for required in (
            "runners/${COMMIT_ID:0:9}",
            "主 SGLang 容器",
            "mooncake-store 容器",
            "SGL-Kernel",
            "Mooncake TE/store",
            "无论成功、失败还是中断",
        ):
            with self.subTest(required=required):
                self.assertIn(required, readme)


if __name__ == "__main__":
    unittest.main(verbosity=2)
