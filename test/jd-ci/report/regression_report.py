#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CASE_STATUSES = ("passed", "failed", "skipped", "blocked")
TEST_AREAS = ("cpu_mock", "server_api", "operator")


def write_regression_report(
    path: str | Path,
    *,
    test_area: str,
    status: str,
    cases: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
    duration_seconds: float = 0.0,
) -> None:
    if test_area not in TEST_AREAS:
        raise ValueError(f"unsupported test area: {test_area}")
    normalized_cases = [dict(case) for case in cases]
    unknown_statuses = sorted(
        {
            str(case.get("status"))
            for case in normalized_cases
            if case.get("status") not in CASE_STATUSES
        }
    )
    if unknown_statuses:
        raise ValueError(
            "unsupported case status: " + ", ".join(unknown_statuses)
        )

    report = {
        "test_area": test_area,
        "status": status,
        "total": len(normalized_cases),
        "passed": sum(case.get("status") == "passed" for case in normalized_cases),
        "failed": sum(case.get("status") == "failed" for case in normalized_cases),
        "skipped": sum(case.get("status") == "skipped" for case in normalized_cases),
        "blocked": sum(case.get("status") == "blocked" for case in normalized_cases),
        "duration_seconds": round(float(duration_seconds), 3),
        "cases": normalized_cases,
    }
    if metadata:
        report.update(dict(metadata))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def write_skipped_regression_report(
    path: str | Path, *, test_area: str, reason: str
) -> None:
    write_regression_report(
        path,
        test_area=test_area,
        status="skipped",
        cases=[
            {
                "name": f"{test_area}_configured_skip",
                "status": "skipped",
                "exit_code": 0,
                "detail": reason,
                "log_file": "",
            }
        ],
        metadata={
            "skip_allowed": True,
            "skip_reason": reason,
        },
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a JD CI regression report")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-area", required=True, choices=TEST_AREAS)
    parser.add_argument("--skip-reason", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    write_skipped_regression_report(
        args.output,
        test_area=args.test_area,
        reason=args.skip_reason,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
