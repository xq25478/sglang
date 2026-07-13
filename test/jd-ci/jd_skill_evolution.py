#!/usr/bin/env python3
"""Deterministic evidence and rule lifecycle for JD repository skills."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
LOW_RISK_STATUSES = {
    "candidate",
    "eligible",
    "promoted",
    "retired",
    "rolled-back",
}
REVIEW_STATUSES = {
    "pending-review",
    "promoted",
    "rejected",
    "retired",
    "rolled-back",
}
ALL_STATUSES = LOW_RISK_STATUSES | REVIEW_STATUSES
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SECRET_KEY_NAMES = {
    "api_key",
    "access_key",
    "private_key",
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "credentials",
}
SECRET_KEY_SUFFIXES = ("_password", "_secret", "_token", "_credential")
DEVELOPER_PATH_PATTERN = re.compile(r"(?:^|\s)/(?:Users|home|root)/\S+")


class EvolutionError(RuntimeError):
    """Raised when evolution state or a requested transition is unsafe."""


def _evolution_dir(skill_dir: Path) -> Path:
    return Path(skill_dir).resolve() / "evolution"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvolutionError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvolutionError(f"expected a JSON object in {path}")
    return value


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _require_string(container: dict[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvolutionError(f"{key} must be a non-empty string")
    return value.strip()


def _check_for_secret_keys(value: Any, location: str = "observation") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
            if normalized in SECRET_KEY_NAMES or normalized.endswith(
                SECRET_KEY_SUFFIXES
            ):
                raise EvolutionError(f"secret-like key is forbidden at {location}.{key}")
            _check_for_secret_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_for_secret_keys(child, f"{location}[{index}]")


def _validate_policy(skill_dir: Path) -> dict[str, Any]:
    path = _evolution_dir(skill_dir) / "policy.json"
    policy = _read_json(path)
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise EvolutionError(f"unsupported policy schema in {path}")
    expected_skill = Path(skill_dir).resolve().name
    if policy.get("skill") != expected_skill:
        raise EvolutionError(
            f"policy skill {policy.get('skill')!r} does not match {expected_skill!r}"
        )
    threshold = policy.get("minimum_independent_evidence")
    if not isinstance(threshold, int) or threshold < 2:
        raise EvolutionError("minimum_independent_evidence must be an integer >= 2")
    for key in (
        "auto_promote_kinds",
        "mandatory_review_kinds",
        "forbidden_action_patterns",
    ):
        values = policy.get(key)
        if not isinstance(values, list) or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            raise EvolutionError(f"policy {key} must be a list of non-empty strings")
    overlap = set(policy["auto_promote_kinds"]) & set(
        policy["mandatory_review_kinds"]
    )
    if overlap:
        raise EvolutionError(f"rule kinds cannot be both automatic and reviewed: {overlap}")
    return policy


def _validate_rules_state(skill_dir: Path) -> dict[str, Any]:
    path = _evolution_dir(skill_dir) / "rules.json"
    state = _read_json(path)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise EvolutionError(f"unsupported rules schema in {path}")
    expected_skill = Path(skill_dir).resolve().name
    if state.get("skill") != expected_skill:
        raise EvolutionError(
            f"rules skill {state.get('skill')!r} does not match {expected_skill!r}"
        )
    rules = state.get("rules")
    if not isinstance(rules, dict):
        raise EvolutionError("rules must be a JSON object keyed by rule ID")
    for rule_id, rule in rules.items():
        if not isinstance(rule, dict):
            raise EvolutionError(f"rule {rule_id} must be an object")
        if rule.get("rule_id") != rule_id:
            raise EvolutionError(f"rule key does not match rule_id for {rule_id}")
        if rule.get("status") not in ALL_STATUSES:
            raise EvolutionError(f"rule {rule_id} has an invalid status")
    return state


def _read_observations(skill_dir: Path) -> list[dict[str, Any]]:
    path = _evolution_dir(skill_dir) / "observations.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvolutionError(f"cannot read observations from {path}: {error}") from error
    observations: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise EvolutionError(f"blank observation line at {path}:{line_number}")
        try:
            observation = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvolutionError(
                f"invalid observation JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(observation, dict):
            raise EvolutionError(
                f"observation at {path}:{line_number} must be an object"
            )
        observations.append(observation)
    return observations


def canonical_rule_id(
    skill: str, kind: str, scope: str, condition: str, action: str
) -> str:
    payload = json.dumps(
        [skill, kind, scope, condition, action],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_observation(
    skill_dir: Path, observation: dict[str, Any], policy: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    _check_for_secret_keys(observation)
    if observation.get("schema_version") != SCHEMA_VERSION:
        raise EvolutionError("unsupported observation schema")
    expected_skill = Path(skill_dir).resolve().name
    if observation.get("skill") != expected_skill:
        raise EvolutionError("observation skill does not match the target skill directory")
    _require_string(observation, "observation_id")
    _require_string(observation, "run_id")
    _require_string(observation, "recorded_at")

    source_context = observation.get("source_context")
    if not isinstance(source_context, dict):
        raise EvolutionError("source_context must be an object")
    _require_string(source_context, "identity")
    for key, value in source_context.items():
        if key.endswith("_sha") and (
            not isinstance(value, str) or not SHA_PATTERN.fullmatch(value)
        ):
            raise EvolutionError(f"source_context {key} must be a full commit SHA")

    rule = observation.get("rule")
    if not isinstance(rule, dict):
        raise EvolutionError("rule must be an object")
    kind = _require_string(rule, "kind")
    scope = _require_string(rule, "scope")
    condition = _require_string(rule, "condition")
    action = _require_string(rule, "action")
    known_kinds = set(policy["auto_promote_kinds"]) | set(
        policy["mandatory_review_kinds"]
    )
    if kind not in known_kinds:
        raise EvolutionError(f"rule kind {kind!r} is not declared by policy")
    normalized_action = action.casefold()
    for pattern in policy["forbidden_action_patterns"]:
        if pattern.casefold() in normalized_action:
            raise EvolutionError(f"non-monotonic rule action matches {pattern!r}")

    evidence = observation.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise EvolutionError("evidence must be a non-empty list")
    for location in evidence:
        if not isinstance(location, str) or not location.strip():
            raise EvolutionError("evidence locations must be non-empty strings")
        file_part = location.split(":", 1)[0]
        evidence_path = Path(file_part)
        if evidence_path.is_absolute() or ".." in evidence_path.parts:
            raise EvolutionError("evidence paths must be repository-relative")

    verification = observation.get("verification")
    if not isinstance(verification, list) or not verification:
        raise EvolutionError("verification must be a non-empty list")
    for result in verification:
        if not isinstance(result, dict):
            raise EvolutionError("verification entries must be objects")
        command = _require_string(result, "command")
        if DEVELOPER_PATH_PATTERN.search(command):
            raise EvolutionError(
                "verification commands must not contain absolute developer paths"
            )
        if not isinstance(result.get("exit_code"), int):
            raise EvolutionError("verification exit_code must be an integer")
    outcome = _require_string(observation, "outcome")
    if outcome not in {"success", "contradiction", "failure"}:
        raise EvolutionError(f"unsupported observation outcome {outcome!r}")
    if outcome == "success" and any(
        result["exit_code"] != 0 for result in verification
    ):
        raise EvolutionError("successful observations require zero verification exit codes")

    rule_id = canonical_rule_id(expected_skill, kind, scope, condition, action)
    supplied_rule_id = observation.get("rule_id")
    if supplied_rule_id is not None and supplied_rule_id != rule_id:
        raise EvolutionError("observation rule_id does not match its canonical rule")
    normalized = json.loads(
        json.dumps(observation, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    normalized["rule_id"] = rule_id
    return rule_id, normalized


def record_observation(skill_dir: Path, observation: dict[str, Any]) -> str:
    """Validate one bounded observation and append it atomically as canonical JSONL."""

    skill_dir = Path(skill_dir)
    policy = _validate_policy(skill_dir)
    check_state(skill_dir)
    existing = _read_observations(skill_dir)
    rule_id, normalized = _validate_observation(skill_dir, observation, policy)
    observation_id = normalized["observation_id"]
    for item in existing:
        if item.get("observation_id") != observation_id:
            continue
        if item == normalized:
            return rule_id
        raise EvolutionError(
            f"observation_id {observation_id!r} already exists with different content"
        )
    path = _evolution_dir(skill_dir) / "observations.jsonl"
    serialized = [
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for item in existing
    ]
    serialized.append(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    _atomic_write(path, "\n".join(serialized) + "\n")
    return rule_id


def evaluate_rules(skill_dir: Path) -> dict[str, Any]:
    """Rebuild candidate state from observations without activating any rule."""

    skill_dir = Path(skill_dir)
    policy = _validate_policy(skill_dir)
    state = _validate_rules_state(skill_dir)
    observations = _read_observations(skill_dir)
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_observation_ids: dict[str, dict[str, Any]] = {}
    for observation in observations:
        rule_id, normalized = _validate_observation(skill_dir, observation, policy)
        observation_id = normalized["observation_id"]
        previous = seen_observation_ids.get(observation_id)
        if previous is not None and previous != normalized:
            raise EvolutionError(
                f"observation_id {observation_id!r} has conflicting content"
            )
        seen_observation_ids[observation_id] = normalized
        grouped.setdefault(rule_id, []).append(normalized)

    existing_rules = state["rules"]
    evaluated_rules: dict[str, dict[str, Any]] = {}
    terminal_statuses = {"promoted", "rejected", "retired", "rolled-back"}
    for rule_id, items in sorted(grouped.items()):
        first = items[0]
        rule_spec = first["rule"]
        successful = [item for item in items if item["outcome"] == "success"]
        contradictions = sorted(
            item["observation_id"]
            for item in items
            if item["outcome"] == "contradiction"
        )
        run_ids = {item["run_id"] for item in successful}
        source_contexts = {
            item["source_context"]["identity"] for item in successful
        }
        independent_count = min(len(run_ids), len(source_contexts))
        previous_rule = existing_rules.get(rule_id, {})
        previous_status = previous_rule.get("status")
        if previous_status in terminal_statuses:
            status = previous_status
        elif rule_spec["kind"] in policy["mandatory_review_kinds"]:
            status = "pending-review"
        elif contradictions:
            status = "candidate"
        elif independent_count >= policy["minimum_independent_evidence"]:
            status = "eligible"
        else:
            status = "candidate"
        evaluated_rule = {
            "rule_id": rule_id,
            "kind": rule_spec["kind"],
            "scope": rule_spec["scope"],
            "condition": rule_spec["condition"],
            "action": rule_spec["action"],
            "status": status,
            "supporting_observations": sorted(
                item["observation_id"] for item in successful
            ),
            "independent_evidence_count": independent_count,
            "contradictions": contradictions,
        }
        for key in ("promotion", "review", "rollback_history"):
            if key in previous_rule:
                evaluated_rule[key] = previous_rule[key]
        evaluated_rules[rule_id] = evaluated_rule

    for rule_id, rule in existing_rules.items():
        if rule_id not in evaluated_rules:
            evaluated_rules[rule_id] = rule
    result = {
        "schema_version": SCHEMA_VERSION,
        "skill": skill_dir.resolve().name,
        "rules": dict(sorted(evaluated_rules.items())),
    }
    _write_json(_evolution_dir(skill_dir) / "rules.json", result)
    return result


def check_state(skill_dir: Path) -> dict[str, Any]:
    """Fail closed on schema, hash, transition, path, secret, or invariant violations."""

    skill_dir = Path(skill_dir)
    policy = _validate_policy(skill_dir)
    state = _validate_rules_state(skill_dir)
    observations = _read_observations(skill_dir)
    seen_ids: dict[str, dict[str, Any]] = {}
    observations_by_rule: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        rule_id, normalized = _validate_observation(skill_dir, observation, policy)
        observation_id = normalized["observation_id"]
        previous = seen_ids.get(observation_id)
        if previous is not None and previous != normalized:
            raise EvolutionError(f"conflicting observation ID {observation_id!r}")
        seen_ids[observation_id] = normalized
        observations_by_rule.setdefault(rule_id, []).append(normalized)
    for rule_id, rule in state["rules"].items():
        expected = canonical_rule_id(
            state["skill"],
            _require_string(rule, "kind"),
            _require_string(rule, "scope"),
            _require_string(rule, "condition"),
            _require_string(rule, "action"),
        )
        if expected != rule_id:
            raise EvolutionError(f"rule hash mismatch for {rule_id}")
        rule_observations = observations_by_rule.get(rule_id, [])
        successful_observations = {
            item["observation_id"]: item
            for item in rule_observations
            if item["outcome"] == "success"
        }
        supporting_ids = rule.get("supporting_observations")
        if not isinstance(supporting_ids, list) or not supporting_ids:
            raise EvolutionError(
                f"rule {rule_id} has no supporting observations"
            )
        if any(item not in successful_observations for item in supporting_ids):
            raise EvolutionError(
                f"rule {rule_id} references invalid supporting observations"
            )
        supporting_items = [successful_observations[item] for item in supporting_ids]
        independent_count = min(
            len({item["run_id"] for item in supporting_items}),
            len({item["source_context"]["identity"] for item in supporting_items}),
        )
        if rule.get("independent_evidence_count") != independent_count:
            raise EvolutionError(
                f"rule {rule_id} has inconsistent independent evidence count"
            )
        contradiction_ids = sorted(
            item["observation_id"]
            for item in rule_observations
            if item["outcome"] == "contradiction"
        )
        if rule.get("contradictions") != contradiction_ids:
            raise EvolutionError(
                f"rule {rule_id} has inconsistent contradiction evidence"
            )
        kind = rule["kind"]
        status = rule["status"]
        requires_review = kind in policy["mandatory_review_kinds"]
        if requires_review and status in {"candidate", "eligible"}:
            raise EvolutionError(
                f"mandatory-review rule {rule_id} has unsafe status {status}"
            )
        if not requires_review and status == "pending-review":
            raise EvolutionError(
                f"automatic rule {rule_id} cannot be pending-review"
            )
        if status == "promoted":
            if requires_review:
                review = rule.get("review")
                if (
                    not isinstance(review, dict)
                    or review.get("decision") != "approve"
                    or not isinstance(review.get("reason"), str)
                    or not review["reason"].strip()
                ):
                    raise EvolutionError(
                        f"high-risk promoted rule {rule_id} requires review approval"
                    )
            promotion = rule.get("promotion")
            if not isinstance(promotion, dict) or not isinstance(
                promotion.get("reason"), str
            ):
                raise EvolutionError(
                    f"promoted rule {rule_id} requires promotion metadata"
                )
            if requires_review:
                if promotion.get("mode") != "human-review":
                    raise EvolutionError(
                        f"high-risk promoted rule {rule_id} requires review approval"
                    )
            elif promotion.get("mode") != "automatic":
                raise EvolutionError(
                    f"low-risk promoted rule {rule_id} requires automatic promotion metadata"
                )
            elif independent_count < policy["minimum_independent_evidence"]:
                raise EvolutionError(
                    f"low-risk promoted rule {rule_id} lacks independent evidence"
                )
        if status == "rejected":
            review = rule.get("review")
            if not isinstance(review, dict) or review.get("decision") != "reject":
                raise EvolutionError(f"rejected rule {rule_id} requires review rejection")
        if status == "rolled-back" and not rule.get("rollback_history"):
            raise EvolutionError(
                f"rolled-back rule {rule_id} requires rollback history"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "skill": state["skill"],
        "observation_count": len(observations),
        "rule_count": len(state["rules"]),
        "status_counts": {
            status: sum(
                1 for rule in state["rules"].values() if rule["status"] == status
            )
            for status in sorted(ALL_STATUSES)
            if any(rule["status"] == status for rule in state["rules"].values())
        },
    }


def _rule_from_state(state: dict[str, Any], rule_id: str) -> dict[str, Any]:
    rule = state["rules"].get(rule_id)
    if not isinstance(rule, dict):
        raise EvolutionError(f"unknown rule_id {rule_id!r}")
    return rule


def _store_rule(
    skill_dir: Path, state: dict[str, Any], rule_id: str, rule: dict[str, Any]
) -> dict[str, Any]:
    state["rules"][rule_id] = rule
    state["rules"] = dict(sorted(state["rules"].items()))
    _write_json(_evolution_dir(skill_dir) / "rules.json", state)
    return json.loads(json.dumps(rule, sort_keys=True))


def promote_rule(skill_dir: Path, rule_id: str) -> dict[str, Any]:
    """Promote an eligible low-risk rule after rerunning state validation."""

    skill_dir = Path(skill_dir)
    check_state(skill_dir)
    state = evaluate_rules(skill_dir)
    policy = _validate_policy(skill_dir)
    rule = _rule_from_state(state, rule_id)
    if rule["status"] == "promoted":
        return json.loads(json.dumps(rule, sort_keys=True))
    if rule["kind"] in policy["mandatory_review_kinds"]:
        raise EvolutionError(
            f"rule {rule_id} requires explicit human review and cannot be auto-promoted"
        )
    if rule["status"] != "eligible":
        raise EvolutionError(
            f"rule {rule_id} must be eligible before promotion; got {rule['status']}"
        )
    if rule["contradictions"]:
        raise EvolutionError(f"rule {rule_id} has contradiction evidence")
    promoted = dict(rule)
    promoted["status"] = "promoted"
    promoted["promotion"] = {
        "mode": "automatic",
        "reason": (
            f"{promoted['independent_evidence_count']} independent successful "
            "observations passed policy gates"
        ),
    }
    return _store_rule(skill_dir, state, rule_id, promoted)


def review_rule(
    skill_dir: Path, rule_id: str, decision: str, reason: str
) -> dict[str, Any]:
    """Approve or reject a pending-review rule with an explicit human reason."""

    skill_dir = Path(skill_dir)
    if decision not in {"approve", "reject"}:
        raise EvolutionError("review decision must be 'approve' or 'reject'")
    if not reason.strip():
        raise EvolutionError("review reason must be non-empty")
    check_state(skill_dir)
    state = evaluate_rules(skill_dir)
    rule = _rule_from_state(state, rule_id)
    if rule["status"] != "pending-review":
        raise EvolutionError(
            f"rule {rule_id} must be pending-review; got {rule['status']}"
        )
    reviewed = dict(rule)
    reviewed["status"] = "promoted" if decision == "approve" else "rejected"
    reviewed["review"] = {"decision": decision, "reason": reason.strip()}
    if decision == "approve":
        reviewed["promotion"] = {
            "mode": "human-review",
            "reason": reason.strip(),
        }
    return _store_rule(skill_dir, state, rule_id, reviewed)


def rollback_rule(skill_dir: Path, rule_id: str, reason: str) -> dict[str, Any]:
    """Deactivate a promoted rule and block automatic re-promotion."""

    skill_dir = Path(skill_dir)
    if not reason.strip():
        raise EvolutionError("rollback reason must be non-empty")
    check_state(skill_dir)
    state = _validate_rules_state(skill_dir)
    rule = _rule_from_state(state, rule_id)
    if rule["status"] != "promoted":
        raise EvolutionError(
            f"rule {rule_id} must be promoted before rollback; got {rule['status']}"
        )
    rolled_back = dict(rule)
    rolled_back["status"] = "rolled-back"
    history = list(rolled_back.get("rollback_history", []))
    history.append({"reason": reason.strip()})
    rolled_back["rollback_history"] = history
    return _store_rule(skill_dir, state, rule_id, rolled_back)


def list_rules(
    skill_dir: Path, status: str | None = None
) -> list[dict[str, Any]]:
    """Return deterministic rule summaries sorted by rule ID."""

    skill_dir = Path(skill_dir)
    check_state(skill_dir)
    if status is not None and status not in ALL_STATUSES:
        raise EvolutionError(f"unknown rule status {status!r}")
    state = _validate_rules_state(skill_dir)
    return [
        json.loads(json.dumps(rule, sort_keys=True))
        for _, rule in sorted(state["rules"].items())
        if status is None or rule["status"] == status
    ]


def _write_output(value: Any, output: str | None) -> None:
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output:
        _atomic_write(Path(output), content)
    else:
        sys.stdout.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Append one validated observation")
    record.add_argument("--skill-dir", required=True)
    record.add_argument("--input", required=True)

    for command, help_text in (
        ("evaluate", "Recompute candidate eligibility"),
        ("check", "Validate evolution state"),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument("--skill-dir", required=True)
        subparser.add_argument("--output")

    promote = subparsers.add_parser("promote", help="Activate an eligible rule")
    promote.add_argument("--skill-dir", required=True)
    promote.add_argument("--rule-id", required=True)

    review = subparsers.add_parser(
        "review", help="Approve or reject a mandatory-review rule"
    )
    review.add_argument("--skill-dir", required=True)
    review.add_argument("--rule-id", required=True)
    review.add_argument("--decision", choices=("approve", "reject"), required=True)
    review.add_argument("--reason", required=True)

    rollback = subparsers.add_parser("rollback", help="Deactivate a promoted rule")
    rollback.add_argument("--skill-dir", required=True)
    rollback.add_argument("--rule-id", required=True)
    rollback.add_argument("--reason", required=True)

    list_parser = subparsers.add_parser("list", help="List rules deterministically")
    list_parser.add_argument("--skill-dir", required=True)
    list_parser.add_argument("--status", choices=sorted(ALL_STATUSES))
    list_parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "record":
            observation = _read_json(Path(args.input))
            result: Any = {
                "rule_id": record_observation(Path(args.skill_dir), observation)
            }
            _write_output(result, None)
        elif args.command == "evaluate":
            _write_output(evaluate_rules(Path(args.skill_dir)), args.output)
        elif args.command == "check":
            _write_output(check_state(Path(args.skill_dir)), args.output)
        elif args.command == "promote":
            _write_output(
                promote_rule(Path(args.skill_dir), args.rule_id),
                None,
            )
        elif args.command == "review":
            _write_output(
                review_rule(
                    Path(args.skill_dir), args.rule_id, args.decision, args.reason
                ),
                None,
            )
        elif args.command == "rollback":
            _write_output(
                rollback_rule(Path(args.skill_dir), args.rule_id, args.reason),
                None,
            )
        elif args.command == "list":
            _write_output(
                list_rules(Path(args.skill_dir), status=args.status),
                args.output,
            )
        else:
            raise EvolutionError(f"unsupported command {args.command}")
    except EvolutionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
