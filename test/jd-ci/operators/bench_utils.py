from __future__ import annotations

import json
import statistics
from typing import Callable

import torch


def cuda_samples(
    function: Callable[[], None], *, warmup: int = 10, repeats: int = 30
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def assert_relative_performance(
    *,
    operator: str,
    optimized: list[float],
    reference: list[float],
    max_ratio: float,
) -> dict[str, object]:
    optimized_median = statistics.median(optimized)
    reference_median = statistics.median(reference)
    ratio = optimized_median / reference_median
    result = {
        "operator": operator,
        "optimized_ms": optimized,
        "reference_ms": reference,
        "optimized_median_ms": optimized_median,
        "reference_median_ms": reference_median,
        "ratio": ratio,
        "max_ratio": max_ratio,
        "samples": len(optimized),
    }
    print(json.dumps(result, sort_keys=True))
    if ratio > max_ratio:
        raise AssertionError(
            f"{operator} performance ratio {ratio:.3f} exceeds {max_ratio:.3f}"
        )
    return result
