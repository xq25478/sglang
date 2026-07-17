#!/usr/bin/env python3
"""Dump JD CI report/log files to stdout before the CI workspace is cleaned."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Iterable
from typing import Any


REPORT_NAMES = (
    "jd_ci_report.md",
    "jd_ci_report.json",
    "jd_accuracy_report.json",
    "jd_accuracy_results.json",
    "jd_functional_report.json",
    "jd_functional_results.json",
    "test_results.tsv",
)

LOG_SUFFIXES = (".log", ".txt", ".md", ".json", ".tsv")
LOG_PATH_KEYS = {
    "log",
    "log_file",
    "server_log_file",
    "precompile_log_file",
    "error_log_file",
}
FAILURE_KEYWORDS = (
    "ERROR",
    "FAILED",
    "Traceback",
    "ImportError",
    "RuntimeError",
    "ValueError",
    "TimeoutError",
    "CUDA out of memory",
    "Segmentation fault",
    "core dumped",
    "失败",
    "异常",
)
FAILURE_LINE_PATTERN = re.compile(
    r"(?:\b(?:ERROR|FAILED|FAILURE|FATAL)\b|Traceback|"
    r"(?:Exception|Error):|RuntimeError|ValueError|TimeoutError|"
    r"timed?\s*out|CUDA out of memory|Segmentation fault|core dumped|"
    r"No such file|not exist|失败|异常)",
    re.IGNORECASE,
)
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
FAILURE_CONTEXT_MAX_FILES = 4
FAILURE_CONTEXT_MAX_LINE_LENGTH = 800


def load_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), root]) == root
    except ValueError:
        return False


def is_text_file(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
    except OSError:
        return False
    return b"\0" not in chunk


def iter_files(root: str) -> Iterable[str]:
    for current_root, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(current_root, name)
            if os.path.isfile(path):
                yield path


def maybe_add_path(paths: list[str], seen: set[str], path: Any, logs_root: str) -> None:
    if not isinstance(path, str) or not path:
        return
    candidate = os.path.realpath(path)
    if not os.path.isfile(candidate) or not is_under(candidate, logs_root):
        return
    if candidate in seen:
        return
    seen.add(candidate)
    paths.append(candidate)


def collect_json_log_paths(value: Any, paths: list[str], seen: set[str], logs_root: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in LOG_PATH_KEYS or key.endswith("_log_file"):
                maybe_add_path(paths, seen, item, logs_root)
            collect_json_log_paths(item, paths, seen, logs_root)
    elif isinstance(value, list):
        for item in value:
            collect_json_log_paths(item, paths, seen, logs_root)


def file_has_failure_keyword(path: str) -> bool:
    if not path.endswith(LOG_SUFFIXES) or not is_text_file(path):
        return False
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return any(keyword in text for keyword in FAILURE_KEYWORDS)


def collect_paths(logs_dir: str, full: bool) -> list[str]:
    logs_root = os.path.realpath(logs_dir)
    paths: list[str] = []
    seen: set[str] = set()

    for path in iter_files(logs_root):
        name = os.path.basename(path)
        if name in REPORT_NAMES or name.startswith("build_"):
            maybe_add_path(paths, seen, path, logs_root)

    for path in list(paths):
        if path.endswith(".json"):
            collect_json_log_paths(load_json(path), paths, seen, logs_root)

    if full:
        for path in iter_files(logs_root):
            if path.endswith(LOG_SUFFIXES):
                maybe_add_path(paths, seen, path, logs_root)
    else:
        for path in iter_files(logs_root):
            if file_has_failure_keyword(path):
                maybe_add_path(paths, seen, path, logs_root)

    return paths


def _find_report(logs_root: str, name: str) -> str | None:
    for path in iter_files(logs_root):
        if os.path.basename(path) == name:
            return path
    return None


def _display_value(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip().replace("\n", " ")
    return text or default


def _failure_context(path: str) -> list[str]:
    try:
        with open(path, "rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - 512 * 1024))
            text = file.read().decode("utf-8", errors="replace")
    except OSError:
        return []

    lines = [ANSI_ESCAPE_PATTERN.sub("", line).rstrip() for line in text.splitlines()]
    matches = [index for index, line in enumerate(lines) if FAILURE_LINE_PATTERN.search(line)]
    if matches:
        last_match = matches[-1]
        selected = lines[max(0, last_match - 3) : min(len(lines), last_match + 5)]
    else:
        selected = [line for line in lines[-12:] if line.strip()]
    return [line[:FAILURE_CONTEXT_MAX_LINE_LENGTH] for line in selected if line.strip()]


def _append_candidate(
    candidates: list[tuple[str, str]],
    seen: set[str],
    path: Any,
    allowed_root: str,
) -> None:
    if not isinstance(path, str) or not path:
        return
    candidate = os.path.realpath(path)
    if (
        candidate in seen
        or not os.path.isfile(candidate)
        or not is_under(candidate, allowed_root)
    ):
        return
    seen.add(candidate)
    candidates.append((candidate, allowed_root))


def render_failure_summary(
    logs_dir: str,
    *,
    overall_exit_code: str,
    main_exit_code: str = "",
    fallback_logs_dir: str = "",
) -> str:
    logs_root = os.path.realpath(logs_dir)
    lines = [
        f"[JD CI FAILURE] overall_exit_code={_display_value(overall_exit_code, 'unknown')}"
    ]
    if main_exit_code and main_exit_code != "0":
        lines.append(f"[JD CI FAILURE] pipeline=sglang exit_code={main_exit_code}")

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    report_path = _find_report(logs_root, "jd_ci_report.json") if os.path.isdir(logs_root) else None
    report = load_json(report_path) if report_path else None
    failed_case_count = 0
    if isinstance(report, dict):
        for regression in report.get("regressions", []):
            if not isinstance(regression, dict):
                continue
            area = _display_value(
                regression.get("display_name"),
                _display_value(regression.get("test_area"), "unknown regression"),
            )
            for case in regression.get("cases", []):
                if not isinstance(case, dict) or case.get("status") not in {
                    "failed",
                    "blocked",
                }:
                    continue
                failed_case_count += 1
                case_name = _display_value(case.get("name"), "unknown case")
                status = _display_value(case.get("status"), "failed")
                exit_code = _display_value(case.get("exit_code"), "unknown")
                detail = _display_value(
                    case.get("detail") or case.get("assertion"), "no detail"
                )
                lines.append(
                    f"[JD CI FAILURE] case={area}/{case_name} "
                    f"status={status} exit_code={exit_code} detail={detail}"
                )
                _append_candidate(candidates, seen, case.get("log_file"), logs_root)

    builds_dir = os.path.join(logs_root, "builds")
    if os.path.isdir(builds_dir):
        for path in iter_files(builds_dir):
            if file_has_failure_keyword(path):
                _append_candidate(candidates, seen, path, logs_root)

    main_pipeline_log = os.path.join(logs_root, "containers", "sglang.log")
    if main_exit_code and main_exit_code != "0":
        _append_candidate(
            candidates,
            seen,
            main_pipeline_log,
            logs_root,
        )
    fallback_root = os.path.realpath(fallback_logs_dir) if fallback_logs_dir else ""
    if fallback_root and os.path.isdir(fallback_root):
        if (
            main_exit_code
            and main_exit_code != "0"
            and not os.path.isfile(main_pipeline_log)
        ):
            _append_candidate(
                candidates,
                seen,
                os.path.join(fallback_root, "containers", "sglang.log"),
                fallback_root,
            )

    context_count = 0
    for path, display_root in candidates[:FAILURE_CONTEXT_MAX_FILES]:
        context = _failure_context(path)
        if not context:
            continue
        context_count += 1
        relative_path = os.path.relpath(path, display_root)
        if display_root != logs_root:
            relative_path = f"final-output-tail/{relative_path}"
        lines.append(f"[JD CI FAILURE] root_cause_log={relative_path}")
        lines.extend(f"[JD CI FAILURE]   {line}" for line in context)

    if failed_case_count == 0 and context_count == 0:
        lines.append(
            "[JD CI FAILURE] root_cause=未从现有日志提取到明确错误行，请查看上方完整日志转储"
        )
    else:
        lines.append("[JD CI FAILURE] 完整上下文见上方 [JD CI LOG DUMP]")
    return "\n".join(lines)


def print_file(path: str, logs_root: str, max_bytes: int) -> None:
    rel = os.path.relpath(path, logs_root)
    size = os.path.getsize(path)
    print(f"[JD CI LOG DUMP] ===== BEGIN {rel} size={size} bytes =====", flush=True)
    if not is_text_file(path):
        print(f"[JD CI LOG DUMP] skip non-text file: {rel}", flush=True)
    else:
        with open(path, "rb") as f:
            if max_bytes > 0 and size > max_bytes:
                f.seek(size - max_bytes)
                print(
                    f"[JD CI LOG DUMP] file truncated: showing last {max_bytes} of {size} bytes",
                    flush=True,
                )
            data = f.read()
        print(data.decode("utf-8", errors="replace"), end="", flush=True)
        if data and not data.endswith(b"\n"):
            print(flush=True)
    print(f"[JD CI LOG DUMP] ===== END {rel} =====", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump JD CI logs to stdout")
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--full", action="store_true", help="Dump every text log/report file")
    parser.add_argument(
        "--failure-summary",
        action="store_true",
        help="Print a concise final failure summary instead of the full log dump",
    )
    parser.add_argument("--overall-exit-code", default="")
    parser.add_argument("--main-exit-code", default="")
    parser.add_argument("--fallback-logs-dir", default="")
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="Per-file tail limit in bytes. 0 means no limit.",
    )
    args = parser.parse_args()

    logs_dir = os.path.realpath(args.logs_dir)
    if args.failure_summary:
        print(
            render_failure_summary(
                logs_dir,
                overall_exit_code=args.overall_exit_code,
                main_exit_code=args.main_exit_code,
                fallback_logs_dir=args.fallback_logs_dir,
            ),
            flush=True,
        )
        return 0
    if not os.path.isdir(logs_dir):
        print(f"[JD CI LOG DUMP] logs dir not found: {args.logs_dir}", flush=True)
        return 0

    paths = collect_paths(logs_dir, args.full)
    mode = "full" if args.full else "failure-focused"
    print(
        f"[JD CI LOG DUMP] mode={mode} logs_dir={logs_dir} files={len(paths)}",
        flush=True,
    )
    for path in paths:
        print_file(path, logs_dir, args.max_bytes)
    print("[JD CI LOG DUMP] complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
