"""Benchmark ungrouped MoE fused gate implementations."""

from __future__ import annotations

import itertools
from typing import Tuple

import torch
import triton
import triton.testing

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    DEFAULT_QUANTILES,
    get_benchmark_range,
)
from sglang.jit_kernel.moe_fused_gate import moe_fused_gate as moe_fused_gate_jit
from sglang.jit_kernel.triton.moe_fused_gate import moe_fused_gate_triton
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, suite="stage-b-kernel-benchmark-1-gpu-large")


TOKEN_RANGE = get_benchmark_range(
    full_range=[1, 512, 2048, 8192, 65536],
    ci_range=[1, 512, 2048],
)
NUM_EXPERTS_RANGE = get_benchmark_range(
    full_range=[256, 384, 512],
    ci_range=[256, 512],
)
TOPK_RANGE = get_benchmark_range(
    full_range=[1, 2, 5, 6],
    ci_range=[1, 6],
)
SHARED_RANGE = [0]


def _configs():
    configs = []
    for num_tokens, num_experts, topk, shared in itertools.product(
        TOKEN_RANGE,
        NUM_EXPERTS_RANGE,
        TOPK_RANGE,
        SHARED_RANGE,
    ):
        if topk <= shared:
            continue
        configs.append((num_tokens, num_experts, topk, shared))
    return configs


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk", "shared"],
        x_vals=_configs(),
        line_arg="provider",
        line_vals=["jit_cuda", "triton", "speedup"],
        line_names=["JIT CUDA", "Triton", "Speedup "],
        styles=[("blue", "-"), ("red", "-"), ("green", "--")],
        ylabel="us / speedup",
        plot_name="moe-fused-gate-triton-performance",
        args={
            "scoring_func": "sigmoid",
            "renormalize": True,
            "routed_scaling_factor": 2.5,
            "apply_routed_scaling_factor_on_output": False,
        },
    )
)
def benchmark(
    num_tokens: int,
    num_experts: int,
    topk: int,
    shared: int,
    provider: str,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[float, float, float]:
    torch.manual_seed(num_tokens + num_experts * 17 + topk * 31 + shared)
    scores = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device=DEFAULT_DEVICE
    )
    bias = torch.randn((num_experts,), dtype=torch.float32, device=DEFAULT_DEVICE) * 0.1
    torch.cuda.synchronize()

    kwargs = dict(
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=shared,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    def bench_impl(impl):
        # Trigger JIT/Triton compilation before timing.
        impl(scores, bias, **kwargs)
        torch.cuda.synchronize()

        def fn():
            impl(scores, bias, **kwargs)

        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            fn, quantiles=DEFAULT_QUANTILES
        )
        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    if provider == "jit_cuda":
        return bench_impl(moe_fused_gate_jit)
    if provider == "triton":
        return bench_impl(moe_fused_gate_triton)

    jit_us, _, _ = bench_impl(moe_fused_gate_jit)
    triton_us, _, _ = bench_impl(moe_fused_gate_triton)
    speedup = jit_us / triton_us
    return speedup, speedup, speedup


if __name__ == "__main__":
    benchmark.run(print_data=True)
