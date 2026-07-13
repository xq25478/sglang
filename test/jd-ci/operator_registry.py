#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from jd_test_manifest import all_cases


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    operator: str
    role: str
    commits: tuple[str, ...]
    min_gpus: int
    timeout_seconds: int
    command: tuple[str, ...]
    assertion: str


def resolve_operator_specs() -> list[OperatorSpec]:
    specs = []
    for category, role in (
        ("operator_correctness", "correctness"),
        ("operator_performance", "performance"),
    ):
        for case in all_cases(category):
            if not case.operator:
                raise ValueError(f"operator is required for {case.case_id}")
            specs.append(
                OperatorSpec(
                    name=case.case_id,
                    operator=case.operator,
                    role=role,
                    commits=case.commits,
                    min_gpus=case.min_gpus,
                    timeout_seconds=case.timeout_seconds,
                    command=case.command,
                    assertion=case.assertion,
                )
            )
    return sorted(specs, key=lambda spec: (spec.operator, spec.role, spec.name))


def validate_operator_pairs(specs: Sequence[OperatorSpec]) -> list[str]:
    roles: dict[str, set[str]] = {}
    for spec in specs:
        roles.setdefault(spec.operator, set()).add(spec.role)
    return sorted(
        operator
        for operator, operator_roles in roles.items()
        if operator_roles != {"correctness", "performance"}
    )


def write_specs(path: str | Path, specs: Sequence[OperatorSpec]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps([asdict(spec) for spec in specs], indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit every fixed JD operator test")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    specs = resolve_operator_specs()
    missing_pairs = validate_operator_pairs(specs)
    if missing_pairs:
        raise SystemExit(
            "JD operators missing correctness/performance pair: "
            + ", ".join(missing_pairs)
        )
    write_specs(args.output, specs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
