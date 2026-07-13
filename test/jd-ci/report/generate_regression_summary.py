#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


TEST_AREAS = (
    ("cpu_mock", "CPU and Mock Regression"),
    ("server_api", "Server and API Regression"),
    ("operator", "Operator Correctness and Performance Regression"),
)


def _load_regression_report(
    logs_dir: Path, test_area: str, display_name: str
) -> dict[str, object]:
    report_path = logs_dir / test_area / "report.json"
    if not report_path.exists():
        return {
            "test_area": test_area,
            "display_name": display_name,
            "status": "missing",
            "total": 0,
            "passed": 0,
            "failed": 1,
            "skipped": 0,
            "blocked": 0,
            "cases": [],
            "detail": f"missing report: {report_path}",
            "report_file": str(report_path),
        }
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)
    report["display_name"] = display_name
    report["report_file"] = str(report_path)
    return report


def _regression_is_successful(report: Mapping[str, object]) -> bool:
    status = report.get("status")
    if status == "passed":
        return int(report.get("failed", 0)) == 0 and int(
            report.get("blocked", 0)
        ) == 0
    if status == "skipped":
        return report.get("skip_allowed") is True
    return False


def build_summary(
    logs_dir: str | Path, metadata: Mapping[str, object]
) -> dict[str, object]:
    root = Path(logs_dir)
    reports = [
        _load_regression_report(root, test_area, display_name)
        for test_area, display_name in TEST_AREAS
    ]
    failed_regressions = [
        str(report.get("test_area"))
        for report in reports
        if not _regression_is_successful(report)
    ]
    return {
        "status": "failed" if failed_regressions else "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "failed_regressions": failed_regressions,
        "metadata": dict(metadata),
        "regressions": reports,
    }


def _markdown(summary: Mapping[str, object]) -> str:
    lines = [
        "# JD CI Regression Summary",
        "",
        f"Overall: **{str(summary['status']).upper()}**",
        "",
        "| Regression Area | Status | Passed | Failed | Skipped | Blocked | GPU |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for report in summary["regressions"]:
        required = report.get("required_gpus", 0)
        available = report.get("available_gpus")
        gpu_text = str(required)
        if available is not None:
            gpu_text = f"{required} required / {available} available"
        lines.append(
            "| {name} | {status} | {passed} | {failed} | {skipped} | {blocked} | {gpu} |".format(
                name=report.get("display_name", report.get("test_area", "")),
                status=report.get("status", ""),
                passed=report.get("passed", 0),
                failed=report.get("failed", 0),
                skipped=report.get("skipped", 0),
                blocked=report.get("blocked", 0),
                gpu=gpu_text,
            )
        )

    failures = []
    for report in summary["regressions"]:
        name = report.get("display_name", report.get("test_area", ""))
        for case in report.get("cases", []):
            if case.get("status") in {"failed", "blocked"}:
                failures.append(
                    f"- {name} / {case.get('name')}: "
                    f"{case.get('detail') or case.get('log_file') or 'failed'}"
                )
        if report.get("status") == "missing":
            failures.append(f"- {name}: {report.get('detail')}")
    if failures:
        lines.extend(["", "## Failures", "", *failures])
    lines.append("")
    return "\n".join(lines)


def write_summary(logs_dir: str | Path, summary: Mapping[str, object]) -> None:
    root = Path(logs_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "jd_ci_report.json"
    markdown_path = root / "jd_ci_report.md"
    json_tmp = json_path.with_suffix(".json.tmp")
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    json_tmp.write_text(
        json.dumps(dict(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_tmp.write_text(_markdown(summary), encoding="utf-8")
    json_tmp.replace(json_path)
    markdown_tmp.replace(markdown_path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate JD regression summary")
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--event-type", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--base-image-tag", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(
        args.logs_dir,
        {
            "event_type": args.event_type,
            "branch": args.branch,
            "commit": args.commit,
            "base_image_tag": args.base_image_tag,
        },
    )
    write_summary(args.logs_dir, summary)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
