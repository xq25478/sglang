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

    def test_frontmatter_is_discoverable_and_minimal(self):
        match = re.match(r"\A---\n(.*?)\n---\n", self.text, re.DOTALL)
        self.assertIsNotNone(match)
        fields = {}
        for line in match.group(1).splitlines():
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
        self.assertEqual(set(fields), {"name", "description"})
        self.assertEqual(fields["name"], "xq-sglang-jd-new-pr")
        self.assertTrue(fields["description"].startswith("用于"))

    def test_skill_has_no_scaffold_placeholders(self):
        forbidden = "|".join(("TO" + "DO", "T" + "BD", "FIX" + "ME", "X" * 3))
        self.assertNotRegex(self.text, rf"(?i)\b(?:{forbidden})\b")
        self.assertLess(len(self.text.splitlines()), 500)

    def test_skill_requires_auditable_delta_before_writes(self):
        for required in (
            "scripts/collect_pr_delta.py",
            "git merge-base",
            "commit-to-case",
            "明确 base",
            "歧义",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_skill_preserves_jd_coverage_integrity(self):
        for required in (
            "test/jd-ci/README.md",
            "test/README.md",
            "jd_test_manifest.py",
            "operator_registry.py",
            "operator_correctness",
            "operator_performance",
            "dummy-weight",
            "real-model",
            "固定累积全量清单",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_skill_preserves_single_server_observable_ci_design(self):
        for required in (
            "Qwen2.5-VL-7B-Instruct/",
            "只能启动一个",
            "SERVER_CASES",
            "15 个固定 HTTP 子 case",
            "--disable-cuda-graph",
            "token_oracle",
            "KV canary",
            "上游 mock-model",
            "模型特定 parser",
            "确定性 CPU fixture",
            "case_progress.py",
            "每 5 秒心跳",
            "全部固定 JD case",
            "普通 case 失败后继续",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_skill_confines_jd_test_assets_to_jd_ci(self):
        for required in (
            "test/jd-ci/unit/",
            "test/jd-ci/operators/",
            "test/jd-ci/pipeline/",
            "所有新增 JD 测试资产",
            "git diff --diff-filter=A",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)
        self.assertNotIn("test/registered/unit/jd", self.text)

        readme = (REPO_ROOT / "test/jd-ci/README.md").read_text(encoding="utf-8")
        self.assertIn("JD CI 新增测试资产只能放在 `test/jd-ci/`", readme)

    def test_skill_requires_tdd_and_safe_mutation_boundaries(self):
        for required in (
            "TDD",
            "RED",
            "不得 commit、push、发布镜像或启动远端 CI",
            "tracked 和 untracked 改动",
            "跳过 GPU 执行只能记录为证据缺失",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_skill_audits_existing_pr_tests_and_real_execution_paths(self):
        for required in (
            "先审计 PR 已有测试资产",
            "真实 dispatch 条件",
            "至少一个 case 必须经过生产调用入口",
            "candidate 与 reference 必须进入不同的底层实现",
            "共享 wrapper",
            "base 公共接口",
            "迁移或删除原位置的重复 JD 测试资产",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.text)

    def test_jd_readme_documents_the_skill_entrypoint(self):
        readme = (REPO_ROOT / "test/jd-ci/README.md").read_text(encoding="utf-8")
        self.assertIn("$xq-sglang-jd-new-pr", readme)
        self.assertIn("scripts/collect_pr_delta.py", readme)
        self.assertIn("--base <JD-target-ref>", readme)

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
            "不得用 mock 覆盖替代 real-server、real-model、多 GPU、算子正确性或算子性能覆盖",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, self.text)

    def test_new_pr_evolution_policy_is_additive_only(self):
        policy = json.loads(
            (SKILL_DIR / "evolution/policy.json").read_text(encoding="utf-8")
        )
        self.assertEqual(policy["schema_version"], 1)
        self.assertEqual(policy["skill"], "xq-sglang-jd-new-pr")
        self.assertEqual(policy["minimum_independent_evidence"], 2)
        self.assertEqual(
            set(policy["auto_promote_kinds"]),
            {
                "path-symbol-mapping",
                "validation-addition",
                "failure-signature",
                "additive-coverage-pattern",
            },
        )
        for mandatory in (
            "semantic-equivalence",
            "coverage-downgrade",
            "case-retirement",
            "accuracy-threshold",
            "performance-threshold",
            "model-gpu-scope",
            "hard-boundary-change",
            "publication-policy",
        ):
            self.assertIn(mandatory, policy["mandatory_review_kinds"])

    def test_new_pr_evolution_state_starts_empty(self):
        observations = SKILL_DIR / "evolution/observations.jsonl"
        rules = json.loads(
            (SKILL_DIR / "evolution/rules.json").read_text(encoding="utf-8")
        )
        self.assertEqual(observations.read_text(encoding="utf-8"), "")
        self.assertEqual(
            rules,
            {
                "schema_version": 1,
                "skill": "xq-sglang-jd-new-pr",
                "rules": {},
            },
        )

    def test_jd_readme_documents_controlled_evolution(self):
        readme = (REPO_ROOT / "test/jd-ci/README.md").read_text(encoding="utf-8")
        for required in (
            "受控自进化",
            "两次独立",
            "稳定核心",
            "不能削弱 JD CI",
            "jd_skill_evolution.py check",
        ):
            with self.subTest(required=required):
                self.assertIn(required, readme)

    def test_skill_requires_real_pipeline_machine_verification_without_images(self):
        for required in (
            "流水线机器真实验证",
            "必须到用户指定的流水线机器",
            "tmux",
            "不得保存、发布或推送镜像",
            "tracks_ci_head=true",
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


if __name__ == "__main__":
    unittest.main()
