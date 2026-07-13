import json
import re
import unittest
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
SKILL_FILE = SKILL_DIR / "SKILL.md"
REPO_ROOT = Path(__file__).resolve().parents[4]


class TestSkillContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SKILL_FILE.read_text(encoding="utf-8")

    def test_frontmatter_is_minimal_and_discoverable(self):
        match = re.match(r"\A---\n(.*?)\n---\n", self.text, re.DOTALL)
        self.assertIsNotNone(match)
        fields = {}
        for line in match.group(1).splitlines():
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
        self.assertEqual(set(fields), {"name", "description"})
        self.assertEqual(fields["name"], "xq-sglang-jd-release-rebase")
        self.assertTrue(fields["description"].startswith("用于"))

    def test_skill_has_no_scaffold_or_obsolete_terminology(self):
        forbidden = (
            "TO" + "DO",
            "T" + "BD",
            "FIX" + "ME",
            "X" * 3,
            "sta" + "ge",
            "阶" + "段",
        )
        for term in forbidden:
            with self.subTest(term=term):
                self.assertNotIn(term.casefold(), self.text.casefold())
        self.assertLess(len(self.text.splitlines()), 500)

    def test_skill_requires_full_ref_and_patch_audit(self):
        for required in (
            "refs/tags/",
            "refs/remotes/",
            "git cherry",
            "git range-diff",
            "old-to-new SHA",
            "isolated worktree",
            "high_risk_paths",
            "audit_merge_commits",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_skill_documents_all_helper_commands(self):
        for command in ("plan", "classify", "execute", "resume", "check"):
            self.assertRegex(self.text, rf"jd_release_rebase\.py\s+{command}\b")

    def test_skill_audits_build_prerequisites_before_execute(self):
        for required in (
            "Build Prerequisite Decision",
            "test/jd-ci/env/build_sgl_kernel.sh",
            "test/jd-ci/env/build_mooncake.sh",
            "test/jd-ci/run_jd_ci.sh",
            "sgl-kernel/pyproject.toml",
            "sgl-kernel/CMakeLists.txt",
            "docker/Dockerfile",
            "scripts/ci/cuda/ci_install_dependency.sh",
            "SGL-Kernel",
            "Mooncake",
            "no change",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

        audit = self.text.index("## 2. 审计编译前依赖")
        execute = self.text.index("jd_release_rebase.py execute")
        self.assertLess(audit, execute)

    def test_build_prerequisite_audit_fails_closed(self):
        for required in (
            "Python/PyTorch ABI",
            "CUDA/NVCC",
            "CMake/Ninja/scikit-build",
            "FlashAttention/FlashMLA/CUTLASS",
            "RDMA/libibverbs",
            "NUMA",
            "protobuf/gRPC",
            "Rust/cargo",
            "Go/etcd",
            "NVLink/MNNVL",
            "用户明确批准",
            "clean rebuild",
            "cache",
            "single final JD CI commit",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_skill_makes_upstream_equivalence_the_core_decision(self):
        self.assertIn("外部已有同类修复：放弃内部修复", self.text)
        self.assertIn("外部没有同类修复：继续使用内部修复", self.text)
        self.assertIn("absorbed-semantic", self.text)

    def test_skill_migrates_jd_tests_and_limits_external_effects(self):
        for required in (
            "test/jd-ci/jd_test_manifest.py",
            "$xq-sglang-jd-new-pr",
            "operator_correctness",
            "operator_performance",
            "不得 push、force-push、删除分支、启动远端 CI 或发布镜像",
            "--force-with-lease",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_skill_requires_one_final_jd_ci_commit_at_head(self):
        for required in (
            "JD CI commit boundary",
            "test/jd-ci/",
            ".agents/skills/xq-sglang-jd-new-pr/",
            ".agents/skills/xq-sglang-jd-release-rebase/",
            "deferred_ci_commits",
            "jd_release_rebase.py prepare-ci",
            "jd_release_rebase.py commit-ci",
            "exactly one commit",
            "must be `HEAD`",
            "mixed production and JD CI commit",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

        migrate = self.text.index("## 6. 迁移 JD CI 映射")
        commit_ci = self.text.index("jd_release_rebase.py commit-ci")
        verify = self.text.index("## 7. 检查与本地验证")
        self.assertLess(migrate, commit_ci)
        self.assertLess(commit_ci, verify)

    def test_skill_allows_networked_exact_dependency_updates_on_controller(self):
        for required in (
            "控制机可以联网",
            "目标版本",
            "immutable tag or commit SHA",
            "dependency declaration",
            "lock files",
            "网络权限不等于 push、启动远端 CI",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_release_verification_uses_jd_ci_owned_tests(self):
        self.assertIn("test/jd-ci/unit/ci", self.text)
        self.assertIn("JD test-only 资产", self.text)
        self.assertNotIn("test/registered/unit/jd", self.text)

    def test_release_revalidates_single_server_observable_ci_design(self):
        for required in (
            "Qwen2.5-VL-7B-Instruct/",
            "恰好一个",
            "SERVER_CASES",
            "15 个固定 HTTP 子 case",
            "--disable-cuda-graph",
            "token_oracle",
            "KV canary",
            "上游 mock-model",
            "deterministic CPU fixtures",
            "case_progress.py",
            "每 5 秒心跳",
            "全部固定 JD case",
            "普通 case 失败后继续",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_jd_readme_documents_release_skill(self):
        readme = (REPO_ROOT / "test/jd-ci/README.md").read_text(encoding="utf-8")
        self.assertIn("$xq-sglang-jd-release-rebase", readme)
        self.assertIn("jd_release_rebase.py plan", readme)
        self.assertIn("--old-internal refs/remotes/origin/JD-", readme)
        self.assertIn("外部已有同类修复，放弃内部修复", readme)
        self.assertIn("外部没有同类修复，继续使用内部修复", readme)

    def test_skill_has_controlled_self_evolution_contract(self):
        for required in (
            "evolution/policy.json",
            "jd_skill_evolution.py check",
            "jd_skill_evolution.py list",
            "jd_skill_evolution.py record",
            "jd_skill_evolution.py evaluate",
            "jd_skill_evolution.py promote",
            "pending-review",
            "进化报告",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)
        for boundary in (
            "不得改写 `SKILL.md`",
            "不得 commit 或 push",
            "不得启动 CI 或发布镜像",
            "不得自动判定内部修复已被吸收",
            "不得自动调用 `classify`",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, self.text)

    def test_release_evolution_never_auto_absorbs_internal_fix(self):
        policy = json.loads(
            (SKILL_DIR / "evolution/policy.json").read_text(encoding="utf-8")
        )
        self.assertEqual(policy["schema_version"], 1)
        self.assertEqual(policy["skill"], "xq-sglang-jd-release-rebase")
        self.assertEqual(policy["minimum_independent_evidence"], 2)
        self.assertEqual(
            set(policy["auto_promote_kinds"]),
            {
                "path-symbol-mapping",
                "validation-addition",
                "failure-signature",
                "api-rename-adaptation",
            },
        )
        for mandatory in (
            "semantic-equivalence",
            "absorbed-semantic",
            "internal-fix-retirement",
            "case-retirement",
            "conflict-behavior",
            "dependency-choice",
            "accuracy-threshold",
            "performance-threshold",
            "model-gpu-scope",
            "hard-boundary-change",
            "publication-policy",
        ):
            self.assertIn(mandatory, policy["mandatory_review_kinds"])

    def test_release_evolution_state_starts_empty(self):
        observations = SKILL_DIR / "evolution/observations.jsonl"
        rules = json.loads(
            (SKILL_DIR / "evolution/rules.json").read_text(encoding="utf-8")
        )
        self.assertEqual(observations.read_text(encoding="utf-8"), "")
        self.assertEqual(
            rules,
            {
                "schema_version": 1,
                "skill": "xq-sglang-jd-release-rebase",
                "rules": {},
            },
        )

    def test_jd_readme_keeps_release_evolution_reviewed(self):
        readme = (REPO_ROOT / "test/jd-ci/README.md").read_text(encoding="utf-8")
        for required in (
            "机械路径/API 迁移",
            "语义等价不能自动判定",
            "删除 JD 内部修复或 case 必须人工确认",
        ):
            with self.subTest(required=required):
                self.assertIn(required, readme)

    def test_skill_requires_remote_pipeline_verification_without_image_output(self):
        for required in (
            "流水线机器真实验证",
            "必须到用户指定的流水线机器",
            "tmux",
            "不得保存、发布或推送任何镜像",
            "/export/zhangyu/ci/sglang/sgl-kernel/v0.5.15/_deps",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_skill_documents_any_branch_merge_cache_policy(self):
        for required in (
            "`-m`",
            "任意分支",
            "对应版本主分支的正式缓存",
            "cache miss",
            "不得回退源码编译",
            "镜像发布仍需用户单独授权",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_skill_requires_location_independent_fetchcontent_cache(self):
        for required in (
            "下载压缩包和 `*-src`",
            "`*-build`",
            "`*-subbuild`",
            "CMake 状态",
            "旧容器绝对路径",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)


if __name__ == "__main__":
    unittest.main()
