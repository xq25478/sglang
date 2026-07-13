import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "test/jd-ci"))

import jd_skill_evolution as evolution  # noqa: E402


class TestJDSkillEvolution(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.skill_dir = Path(self.temporary_directory.name) / "example-skill"
        evolution_dir = self.skill_dir / "evolution"
        evolution_dir.mkdir(parents=True)
        (evolution_dir / "policy.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "skill": "example-skill",
                    "minimum_independent_evidence": 2,
                    "auto_promote_kinds": [
                        "additive-coverage-pattern",
                        "validation-addition",
                    ],
                    "mandatory_review_kinds": [
                        "semantic-equivalence",
                        "coverage-downgrade",
                    ],
                    "forbidden_action_patterns": [
                        "replace real-model coverage with mock coverage",
                        "skip required verification",
                        "relax tolerance",
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (evolution_dir / "observations.jsonl").write_text("", encoding="utf-8")
        (evolution_dir / "rules.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "skill": "example-skill",
                    "rules": {},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def observation(
        self,
        *,
        run_id: str,
        context: str,
        kind: str = "additive-coverage-pattern",
        observation_id: str | None = None,
        action: str = "add a parser regression assertion",
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "skill": "example-skill",
            "observation_id": observation_id or f"observation-{run_id}",
            "run_id": run_id,
            "recorded_at": "2026-07-13T00:00:00Z",
            "source_context": {
                "identity": context,
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
            },
            "rule": {
                "kind": kind,
                "scope": "python/sglang/example.py",
                "condition": "parser option changes",
                "action": action,
            },
            "evidence": ["python/sglang/example.py:12"],
            "verification": [
                {
                    "command": "python3 -m unittest test_example.py -v",
                    "exit_code": 0,
                }
            ],
            "outcome": "success",
        }

    def make_eligible_rule(self) -> str:
        rule_id = evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-1", context="context-a"),
        )
        evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-2", context="context-b"),
        )
        evolution.evaluate_rules(self.skill_dir)
        return rule_id

    def make_pending_review_rule(self, kind: str = "semantic-equivalence") -> str:
        rule_id = evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-1", context="context-a", kind=kind),
        )
        evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-2", context="context-b", kind=kind),
        )
        evolution.evaluate_rules(self.skill_dir)
        return rule_id

    def test_first_observation_remains_candidate(self):
        rule_id = evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-1", context="context-a"),
        )

        report = evolution.evaluate_rules(self.skill_dir)

        self.assertEqual(report["rules"][rule_id]["status"], "candidate")

    def test_two_independent_observations_make_low_risk_rule_eligible(self):
        evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-1", context="context-a"),
        )
        evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-2", context="context-b"),
        )

        report = evolution.evaluate_rules(self.skill_dir)

        self.assertEqual(
            next(iter(report["rules"].values()))["status"], "eligible"
        )

    def test_duplicate_source_context_does_not_count_as_independent(self):
        evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-1", context="same-context"),
        )
        evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-2", context="same-context"),
        )

        report = evolution.evaluate_rules(self.skill_dir)

        self.assertEqual(
            next(iter(report["rules"].values()))["status"], "candidate"
        )

    def test_high_risk_rule_requires_review_even_with_repeated_evidence(self):
        for run_id, context in (("run-1", "context-a"), ("run-2", "context-b")):
            evolution.record_observation(
                self.skill_dir,
                self.observation(
                    run_id=run_id,
                    context=context,
                    kind="semantic-equivalence",
                ),
            )

        report = evolution.evaluate_rules(self.skill_dir)

        self.assertEqual(
            next(iter(report["rules"].values()))["status"], "pending-review"
        )

    def test_unsafe_observations_fail_without_changing_rules(self):
        rules_path = self.skill_dir / "evolution/rules.json"
        original_rules = rules_path.read_bytes()
        unsafe_observations = []

        absolute_evidence = self.observation(run_id="run-1", context="context-a")
        absolute_evidence["evidence"] = ["/tmp/private/source.py:1"]
        unsafe_observations.append(absolute_evidence)

        secret_key = self.observation(run_id="run-2", context="context-b")
        secret_key["api_token"] = "do-not-store"
        unsafe_observations.append(secret_key)

        missing_verification = self.observation(run_id="run-3", context="context-c")
        missing_verification["verification"] = []
        unsafe_observations.append(missing_verification)

        failed_success = self.observation(run_id="run-4", context="context-d")
        failed_success["verification"][0]["exit_code"] = 1
        unsafe_observations.append(failed_success)

        absolute_command = self.observation(run_id="run-5", context="context-e")
        absolute_command["verification"][0]["command"] = (
            "python3 /Users/example/private/test.py"
        )
        unsafe_observations.append(absolute_command)

        for observation in unsafe_observations:
            with self.subTest(observation_id=observation["observation_id"]):
                with self.assertRaises(evolution.EvolutionError):
                    evolution.record_observation(self.skill_dir, observation)
                self.assertEqual(rules_path.read_bytes(), original_rules)

    def test_duplicate_observation_id_with_different_content_is_rejected(self):
        original = self.observation(
            run_id="run-1", context="context-a", observation_id="same-id"
        )
        evolution.record_observation(self.skill_dir, original)
        conflicting = self.observation(
            run_id="run-2", context="context-b", observation_id="same-id"
        )

        with self.assertRaisesRegex(evolution.EvolutionError, "different content"):
            evolution.record_observation(self.skill_dir, conflicting)

    def test_malformed_observation_log_fails_closed(self):
        rules_path = self.skill_dir / "evolution/rules.json"
        original_rules = rules_path.read_bytes()
        (self.skill_dir / "evolution/observations.jsonl").write_text(
            "{not-json}\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(evolution.EvolutionError, "invalid observation JSON"):
            evolution.evaluate_rules(self.skill_dir)

        self.assertEqual(rules_path.read_bytes(), original_rules)

    def test_promote_activates_only_eligible_allowlisted_rule(self):
        rule_id = self.make_eligible_rule()

        rule = evolution.promote_rule(self.skill_dir, rule_id)

        self.assertEqual(rule["status"], "promoted")
        self.assertEqual(evolution.promote_rule(self.skill_dir, rule_id), rule)

    def test_promote_refuses_high_risk_rule(self):
        rule_id = self.make_pending_review_rule()

        with self.assertRaisesRegex(evolution.EvolutionError, "human review"):
            evolution.promote_rule(self.skill_dir, rule_id)

    def test_promote_refuses_candidate_with_contradiction(self):
        rule_id = evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-1", context="context-a"),
        )
        contradiction = self.observation(
            run_id="run-2",
            context="context-b",
            observation_id="contradiction-1",
        )
        contradiction["outcome"] = "contradiction"
        evolution.record_observation(self.skill_dir, contradiction)
        evolution.evaluate_rules(self.skill_dir)

        with self.assertRaisesRegex(evolution.EvolutionError, "eligible"):
            evolution.promote_rule(self.skill_dir, rule_id)

    def test_review_requires_reason_and_records_decision(self):
        rule_id = self.make_pending_review_rule()

        with self.assertRaises(evolution.EvolutionError):
            evolution.review_rule(self.skill_dir, rule_id, "approve", "")
        rule = evolution.review_rule(
            self.skill_dir,
            rule_id,
            "approve",
            "Reviewed the upstream source and behavior",
        )

        self.assertEqual(rule["status"], "promoted")
        self.assertEqual(
            rule["review"]["reason"], "Reviewed the upstream source and behavior"
        )

    def test_review_can_reject_pending_rule(self):
        rule_id = self.make_pending_review_rule()

        rule = evolution.review_rule(
            self.skill_dir,
            rule_id,
            "reject",
            "Evidence does not prove equivalent behavior",
        )

        self.assertEqual(rule["status"], "rejected")

    def test_rollback_blocks_automatic_repromotion(self):
        rule_id = self.make_eligible_rule()
        evolution.promote_rule(self.skill_dir, rule_id)

        rolled_back = evolution.rollback_rule(
            self.skill_dir, rule_id, "Contradicted by a later verified task"
        )
        reevaluated = evolution.evaluate_rules(self.skill_dir)

        self.assertEqual(rolled_back["status"], "rolled-back")
        self.assertEqual(reevaluated["rules"][rule_id]["status"], "rolled-back")
        with self.assertRaises(evolution.EvolutionError):
            evolution.promote_rule(self.skill_dir, rule_id)

    def test_list_rules_filters_and_sorts(self):
        first_rule_id = self.make_eligible_rule()
        evolution.promote_rule(self.skill_dir, first_rule_id)

        rules = evolution.list_rules(self.skill_dir, status="promoted")

        self.assertEqual([rule["rule_id"] for rule in rules], [first_rule_id])

    def test_monotonicity_rejects_coverage_downgrade(self):
        downgrade = self.observation(
            run_id="run-1",
            context="context-a",
            kind="coverage-downgrade",
            action="replace real-model coverage with mock coverage",
        )

        with self.assertRaisesRegex(evolution.EvolutionError, "non-monotonic"):
            evolution.record_observation(self.skill_dir, downgrade)

    def test_cli_exposes_complete_lifecycle(self):
        parser = evolution.build_parser()
        commands = (
            ["promote", "--skill-dir", str(self.skill_dir), "--rule-id", "a" * 64],
            [
                "review",
                "--skill-dir",
                str(self.skill_dir),
                "--rule-id",
                "a" * 64,
                "--decision",
                "approve",
                "--reason",
                "reviewed",
            ],
            [
                "rollback",
                "--skill-dir",
                str(self.skill_dir),
                "--rule-id",
                "a" * 64,
                "--reason",
                "contradicted",
            ],
            ["list", "--skill-dir", str(self.skill_dir), "--status", "promoted"],
        )

        self.assertEqual(
            [parser.parse_args(command).command for command in commands],
            ["promote", "review", "rollback", "list"],
        )

    def test_cli_records_evaluates_promotes_and_lists_deterministically(self):
        for run_id, context in (("run-1", "context-a"), ("run-2", "context-b")):
            input_path = Path(self.temporary_directory.name) / f"{run_id}.json"
            input_path.write_text(
                json.dumps(self.observation(run_id=run_id, context=context)),
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    evolution.main(
                        [
                            "record",
                            "--skill-dir",
                            str(self.skill_dir),
                            "--input",
                            str(input_path),
                        ]
                    ),
                    0,
                )
        evaluation_path = Path(self.temporary_directory.name) / "evaluation.json"
        self.assertEqual(
            evolution.main(
                [
                    "evaluate",
                    "--skill-dir",
                    str(self.skill_dir),
                    "--output",
                    str(evaluation_path),
                ]
            ),
            0,
        )
        rule_id = next(iter(json.loads(evaluation_path.read_text())["rules"]))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                evolution.main(
                    [
                        "promote",
                        "--skill-dir",
                        str(self.skill_dir),
                        "--rule-id",
                        rule_id,
                    ]
                ),
                0,
            )
        list_path = Path(self.temporary_directory.name) / "list.json"
        self.assertEqual(
            evolution.main(
                [
                    "list",
                    "--skill-dir",
                    str(self.skill_dir),
                    "--status",
                    "promoted",
                    "--output",
                    str(list_path),
                ]
            ),
            0,
        )

        self.assertEqual(json.loads(list_path.read_text())[0]["rule_id"], rule_id)

    def test_cli_policy_violation_returns_nonzero_without_traceback(self):
        input_path = Path(self.temporary_directory.name) / "unsafe.json"
        input_path.write_text(
            json.dumps(
                self.observation(
                    run_id="run-1",
                    context="context-a",
                    kind="coverage-downgrade",
                    action="replace real-model coverage with mock coverage",
                )
            ),
            encoding="utf-8",
        )
        errors = io.StringIO()

        with redirect_stderr(errors):
            result = evolution.main(
                [
                    "record",
                    "--skill-dir",
                    str(self.skill_dir),
                    "--input",
                    str(input_path),
                ]
            )

        self.assertEqual(result, 2)
        self.assertIn("non-monotonic", errors.getvalue())

    def test_engine_does_not_import_external_effect_tools(self):
        source = (REPO_ROOT / "test/jd-ci/jd_skill_evolution.py").read_text(
            encoding="utf-8"
        )
        for forbidden in ("import subprocess", "import requests", "import urllib"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_check_rejects_tampered_high_risk_promotion(self):
        rule_id = self.make_pending_review_rule()
        rules_path = self.skill_dir / "evolution/rules.json"
        state = json.loads(rules_path.read_text(encoding="utf-8"))
        state["rules"][rule_id]["status"] = "promoted"
        state["rules"][rule_id].pop("review", None)
        rules_path.write_text(json.dumps(state), encoding="utf-8")

        with self.assertRaisesRegex(evolution.EvolutionError, "review approval"):
            evolution.check_state(self.skill_dir)

    def test_check_rejects_tampered_low_risk_promotion(self):
        rule_id = self.make_eligible_rule()
        rules_path = self.skill_dir / "evolution/rules.json"
        state = json.loads(rules_path.read_text(encoding="utf-8"))
        state["rules"][rule_id]["status"] = "promoted"
        state["rules"][rule_id].pop("promotion", None)
        rules_path.write_text(json.dumps(state), encoding="utf-8")

        with self.assertRaisesRegex(evolution.EvolutionError, "promotion metadata"):
            evolution.check_state(self.skill_dir)

    def test_check_rejects_tampered_observation_rule_hash(self):
        evolution.record_observation(
            self.skill_dir,
            self.observation(run_id="run-1", context="context-a"),
        )
        observations_path = self.skill_dir / "evolution/observations.jsonl"
        observation = json.loads(observations_path.read_text(encoding="utf-8"))
        observation["rule_id"] = "f" * 64
        observations_path.write_text(json.dumps(observation) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(evolution.EvolutionError, "rule_id"):
            evolution.check_state(self.skill_dir)

    def test_check_rejects_promoted_rule_without_observations(self):
        rules_path = self.skill_dir / "evolution/rules.json"
        rule_id = evolution.canonical_rule_id(
            "example-skill",
            "validation-addition",
            "python/sglang/injected.py",
            "injected condition",
            "add injected validation",
        )
        state = json.loads(rules_path.read_text(encoding="utf-8"))
        state["rules"][rule_id] = {
            "rule_id": rule_id,
            "kind": "validation-addition",
            "scope": "python/sglang/injected.py",
            "condition": "injected condition",
            "action": "add injected validation",
            "status": "promoted",
            "supporting_observations": ["invented-observation"],
            "independent_evidence_count": 2,
            "contradictions": [],
            "promotion": {
                "mode": "automatic",
                "reason": "invented evidence",
            },
        }
        rules_path.write_text(json.dumps(state), encoding="utf-8")

        with self.assertRaisesRegex(evolution.EvolutionError, "supporting observations"):
            evolution.check_state(self.skill_dir)

    def test_secret_detection_does_not_reject_tokenizer_metadata(self):
        observation = self.observation(run_id="run-1", context="context-a")
        observation["source_context"]["tokenizer"] = "mock-tokenizer"

        rule_id = evolution.record_observation(self.skill_dir, observation)

        self.assertEqual(len(rule_id), 64)


if __name__ == "__main__":
    unittest.main()
