#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


INTERNAL_COMMITS: tuple[str, ...] = (
    "a7aa64409012d18855f164eebb1c42aaf939ebce",
    "96311e38d3a6b3f9be87e638961e3e290553418e",
    "bc480e4882bb3f0df6a6b88e2d790aa3344888fe",
    "2809c9380b5a2db5b0a1afa36bc95cb2acdb9aef",
    "2f46d777d7ef4f7cd654e6a5d954111a5ccabfd9",
    "864ffbc480895a2c6abed22fcfb85a231af0a189",
    "7c6caf9ed699f22dc992f0e27673f14ffc471cda",
    "1de0e5095ce81f9c7334d87bf5fd4715d62c00cf",
    "10ba5495c26bfb5dea5ba5205cf9ab3b9f17ffea",
    "8238f9171c2b70bf6cd3169acc82a7503549d870",
    "0778f0db576164670aec2c31d58bd01c23f1d333",
    "225d4b2fcc2d95b9f9929266a6c22c2ff3b104b9",
    "2bd369e31410ef5fa735a1dbb4513bb0b9e58b06",
    "8d27c47997722ac61e0a51005716800e4f219f15",
    "a684c2001f669bbe09b865314d67f2d737e139f7",
    "3cd534c9d38682361514ae9f4b8d105c9a6a0e08",
    "77465c63d4bf9d5500132e29a0d8e47f60eead8c",
    "222b102f00f8bb7ddbedc887bc32d33755794f73",
    "fb21094805856bc73899df4d2d46beeece26a352",
)

VALID_CATEGORIES = frozenset(
    {"cpu", "server", "operator_correctness", "operator_performance"}
)


@dataclass(frozen=True, slots=True)
class JDCase:
    case_id: str
    commits: tuple[str, ...]
    category: str
    command: tuple[str, ...]
    assertion: str
    min_gpus: int = 0
    timeout_seconds: int = 300
    operator: str | None = None
    tracks_ci_head: bool = False


CASES: tuple[JDCase, ...] = (
    JDCase(
        case_id="jd-openai-function-call",
        commits=(
            "96311e38d3a6b3f9be87e638961e3e290553418e",
            "bc480e4882bb3f0df6a6b88e2d790aa3344888fe",
            "fb21094805856bc73899df4d2d46beeece26a352",
        ),
        category="cpu",
        command=(
            "python3",
            "test/jd-ci/unit/server/test_openai_and_function_call.py",
            "-v",
        ),
        assertion="JD invalid-thinking, DeepSeek/GLM reasoning, usage, and Kimi/DSV4 parsing",
    ),
    JDCase(
        case_id="jd-runtime-model-fixes",
        commits=(
            "2f46d777d7ef4f7cd654e6a5d954111a5ccabfd9",
            "7c6caf9ed699f22dc992f0e27673f14ffc471cda",
            "10ba5495c26bfb5dea5ba5205cf9ab3b9f17ffea",
            "8238f9171c2b70bf6cd3169acc82a7503549d870",
            "225d4b2fcc2d95b9f9929266a6c22c2ff3b104b9",
            "8d27c47997722ac61e0a51005716800e4f219f15",
            "a684c2001f669bbe09b865314d67f2d737e139f7",
            "77465c63d4bf9d5500132e29a0d8e47f60eead8c",
        ),
        category="cpu",
        command=(
            "python3",
            "test/jd-ci/unit/server/test_runtime_and_model_fixes.py",
            "-v",
        ),
        assertion="JD runtime, multimodal, EPLB, CUDA-graph, CP, quantization, and OCR branches",
    ),
    JDCase(
        case_id="jd-metrics-cache",
        commits=("3cd534c9d38682361514ae9f4b8d105c9a6a0e08",),
        category="cpu",
        command=(
            "python3",
            "test/jd-ci/unit/server/test_metrics_and_cache.py",
            "-v",
        ),
        assertion="JD L1/L2 cache metrics state and duration accounting",
    ),
    JDCase(
        case_id="jd-deploy-and-tma-configs",
        commits=(
            "2809c9380b5a2db5b0a1afa36bc95cb2acdb9aef",
            "864ffbc480895a2c6abed22fcfb85a231af0a189",
            "1de0e5095ce81f9c7334d87bf5fd4715d62c00cf",
            "0778f0db576164670aec2c31d58bd01c23f1d333",
            "2bd369e31410ef5fa735a1dbb4513bb0b9e58b06",
        ),
        category="cpu",
        command=(
            "python3",
            "test/jd-ci/unit/server/test_deploy_and_tma_configs.py",
            "-v",
        ),
        assertion="JD deploy model mapping and internal H20D/H200 TMA configuration files",
    ),
    JDCase(
        case_id="jd-ci-contract",
        commits=(),
        category="cpu",
        tracks_ci_head=True,
        command=(
            "python3",
            "test/jd-ci/unit/ci/test_internal_ci_contract.py",
            "-v",
        ),
        assertion="JD build, artifact, log-dump, and orchestration contracts",
    ),
    JDCase(
        case_id="jd-server-api-regressions",
        commits=(
            "96311e38d3a6b3f9be87e638961e3e290553418e",
            "7c6caf9ed699f22dc992f0e27673f14ffc471cda",
            "8238f9171c2b70bf6cd3169acc82a7503549d870",
            "77465c63d4bf9d5500132e29a0d8e47f60eead8c",
        ),
        category="server",
        command=(
            "python3",
            "test/jd-ci/pipeline/server_api_dummy_model.py",
            "--case",
            "jd-server-api-regressions",
            "--output",
            "{result}",
        ),
        assertion="One Qwen2.5-VL dummy Server covers all JD HTTP and request-lifecycle fixes",
        min_gpus=1,
        timeout_seconds=600,
    ),
    JDCase(
        case_id="jd-rmsnorm-correctness",
        commits=("a7aa64409012d18855f164eebb1c42aaf939ebce",),
        category="operator_correctness",
        command=("python3", "test/jd-ci/operators/test_optimized_rmsnorm.py"),
        assertion="Optimized RMSNorm matches the reference RMSNorm",
        min_gpus=1,
        operator="optimized_rmsnorm",
    ),
    JDCase(
        case_id="jd-rmsnorm-performance",
        commits=("a7aa64409012d18855f164eebb1c42aaf939ebce",),
        category="operator_performance",
        command=("python3", "test/jd-ci/operators/bench_optimized_rmsnorm.py"),
        assertion="Optimized RMSNorm is not slower than its reference path",
        min_gpus=1,
        operator="optimized_rmsnorm",
    ),
    JDCase(
        case_id="jd-dp-allgather-correctness",
        commits=(
            "2f46d777d7ef4f7cd654e6a5d954111a5ccabfd9",
            "222b102f00f8bb7ddbedc887bc32d33755794f73",
        ),
        category="operator_correctness",
        command=(
            "torchrun",
            "--standalone",
            "--nproc-per-node=2",
            "test/jd-ci/operators/test_dp_attention_allgather.py",
        ),
        assertion="Compressed DP-attention all-gather reconstructs legacy metadata",
        min_gpus=2,
        operator="dp_attention_allgather",
    ),
    JDCase(
        case_id="jd-dp-allgather-performance",
        commits=(
            "2f46d777d7ef4f7cd654e6a5d954111a5ccabfd9",
            "222b102f00f8bb7ddbedc887bc32d33755794f73",
        ),
        category="operator_performance",
        command=(
            "torchrun",
            "--standalone",
            "--nproc-per-node=2",
            "test/jd-ci/operators/bench_dp_attention_allgather.py",
        ),
        assertion="Compressed DP-attention metadata reduces transfer overhead",
        min_gpus=2,
        operator="dp_attention_allgather",
    ),
    JDCase(
        case_id="jd-dsv4-norm-rope-correctness",
        commits=(),
        category="operator_correctness",
        command=("python3", "test/jd-ci/operators/test_dsv4_norm_rope.py"),
        assertion="JD DSV4 norm-rope kernel matches its reference path",
        min_gpus=1,
        operator="dsv4_norm_rope",
        tracks_ci_head=True,
    ),
    JDCase(
        case_id="jd-dsv4-norm-rope-performance",
        commits=(),
        category="operator_performance",
        command=("python3", "test/jd-ci/operators/bench_dsv4_norm_rope.py"),
        assertion="JD DSV4 norm-rope kernel stays within its relative performance gate",
        min_gpus=1,
        operator="dsv4_norm_rope",
        tracks_ci_head=True,
    ),
    JDCase(
        case_id="jd-w4a8-correctness",
        commits=("a684c2001f669bbe09b865314d67f2d737e139f7",),
        category="operator_correctness",
        command=("python3", "test/jd-ci/operators/test_w4a8.py"),
        assertion="JD W4A8 group-size, scale packing, and dynamic quantization are correct",
        min_gpus=1,
        operator="w4a8",
    ),
    JDCase(
        case_id="jd-w4a8-performance",
        commits=("a684c2001f669bbe09b865314d67f2d737e139f7",),
        category="operator_performance",
        command=("python3", "test/jd-ci/operators/bench_w4a8.py"),
        assertion="JD W4A8 optimized path stays within its relative performance gate",
        min_gpus=1,
        operator="w4a8",
    ),
)


def all_cases(category: str | None = None) -> list[JDCase]:
    if category is None:
        return list(CASES)
    if category not in VALID_CATEGORIES:
        raise ValueError(f"unknown JD test category: {category}")
    return [case for case in CASES if case.category == category]


def _uses_upstream_test_command(case: JDCase) -> bool:
    command = " ".join(case.command)
    if "test/run_suite.py" in command:
        return True
    return "test/registered/" in command


def _command_paths(case: JDCase) -> list[str]:
    return [part for part in case.command if part.endswith((".py", ".sh"))]


def validate_cases(
    cases: Sequence[JDCase],
    expected_commits: Sequence[str],
    *,
    repo_root: str | Path | None = None,
    check_paths: bool = True,
) -> dict[str, list[str]]:
    ids = [case.case_id for case in cases]
    mapped_commits = {commit for case in cases for commit in case.commits}
    expected = set(expected_commits)
    duplicate_case_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    invalid_categories = sorted(
        case.case_id for case in cases if case.category not in VALID_CATEGORIES
    )
    invalid_commits = sorted(
        {
            commit
            for case in cases
            for commit in case.commits
            if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit)
        }
    )
    upstream_test_commands = sorted(
        case.case_id for case in cases if _uses_upstream_test_command(case)
    )
    untracked_cases = sorted(
        case.case_id for case in cases if not case.commits and not case.tracks_ci_head
    )
    invalid_head_tracking = sorted(
        case.case_id for case in cases if case.commits and case.tracks_ci_head
    )
    missing_paths: list[str] = []
    if check_paths:
        root = Path(repo_root or Path(__file__).resolve().parents[2])
        for case in cases:
            for path in _command_paths(case):
                if not (root / path).is_file():
                    missing_paths.append(f"{case.case_id}:{path}")

    return {
        "missing_commits": sorted(expected - mapped_commits),
        "unexpected_commits": sorted(mapped_commits - expected),
        "duplicate_case_ids": duplicate_case_ids,
        "invalid_categories": invalid_categories,
        "invalid_commits": invalid_commits,
        "upstream_test_commands": upstream_test_commands,
        "untracked_cases": untracked_cases,
        "invalid_head_tracking": invalid_head_tracking,
        "missing_paths": sorted(missing_paths),
    }


def validate_manifest(repo_root: str | Path) -> dict[str, list[str]]:
    report = validate_cases(CASES, INTERNAL_COMMITS, repo_root=repo_root)
    failures = {name: values for name, values in report.items() if values}
    if failures:
        raise ValueError("invalid JD test manifest: " + json.dumps(failures, sort_keys=True))
    return report


def write_cases(path: str | Path, cases: Sequence[JDCase]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps([asdict(case) for case in cases], indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit the fixed cumulative JD test inventory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--category", choices=sorted(VALID_CATEGORIES))
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--skip-path-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_cases(
        CASES,
        INTERNAL_COMMITS,
        repo_root=args.source,
        check_paths=not args.skip_path_check,
    )
    failures = {name: values for name, values in report.items() if values}
    if failures:
        raise SystemExit("invalid JD test manifest: " + json.dumps(failures, sort_keys=True))
    write_cases(args.output, all_cases(args.category))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
