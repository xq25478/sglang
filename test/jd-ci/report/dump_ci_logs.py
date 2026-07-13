#!/usr/bin/env python3
"""Dump JD CI report/log files to stdout before the CI workspace is cleaned."""

from __future__ import annotations

import argparse
import json
import os
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
        "--max-bytes",
        type=int,
        default=0,
        help="Per-file tail limit in bytes. 0 means no limit.",
    )
    args = parser.parse_args()

    logs_dir = os.path.realpath(args.logs_dir)
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
