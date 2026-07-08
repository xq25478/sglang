#!/usr/bin/env python3
"""Compare complete CUTLASS, FlashInfer, Triton, and Humming W4A8 MoE calls."""

from __future__ import annotations

import argparse
import atexit
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import torch


_PARALLEL_OVERRIDE = None
ALL_BACKENDS = ("cutlass", "flashinfer", "triton", "humming")
BACKEND_LABELS = {
    "cutlass": "CUTLASS",
    "flashinfer": "FlashInfer",
    "triton": "Triton",
    "humming": "Humming",
}


@dataclass(frozen=True)
class BenchmarkShape:
    hidden_size: int = 4096
    intermediate_size: int = 256
    num_experts: int = 256
    top_k: int = 6
    group_size: int = 128
    tp_size: int = 8


def resolve_local_intermediate_size(
    *,
    model_intermediate_size: int,
    tp_size: int,
    override: int | None,
) -> int:
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if override is not None:
        return override
    if model_intermediate_size <= 0:
        raise ValueError("model_intermediate_size must be positive")
    if model_intermediate_size % tp_size:
        raise ValueError(
            "model_intermediate_size must be divisible by tp_size when "
            "--intermediate-size is not provided"
        )
    return model_intermediate_size // tp_size


def scale_interleave_factor(dim: int) -> int:
    return 4 if dim % 512 == 0 else (2 if dim % 256 == 0 else 1)


def validate_shape(shape: BenchmarkShape) -> None:
    if shape.tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if shape.group_size != 128:
        raise ValueError("group_size must be 128 for this FlashInfer W4A8 experiment")
    if shape.hidden_size % (2 * shape.group_size):
        raise ValueError("hidden_size must be divisible by 2 * group_size")
    if shape.intermediate_size % (2 * shape.group_size):
        raise ValueError("intermediate_size must be divisible by 2 * group_size")
    if not 0 < shape.top_k <= shape.num_experts:
        raise ValueError("top_k must be in [1, num_experts]")


def interleave_group_scales(scales: torch.Tensor, dim: int) -> torch.Tensor:
    if scales.ndim != 3:
        raise ValueError("scales must be a 3D tensor")
    factor = scale_interleave_factor(dim)
    experts, rows, groups = scales.shape
    if groups % factor:
        raise ValueError(
            f"scale group count {groups} must be divisible by factor {factor}"
        )
    return (
        scales.reshape(experts, rows, groups // factor, factor)
        .permute(0, 2, 1, 3)
        .reshape(experts, groups // factor, rows * factor)
        .contiguous()
    )


def compare_outputs(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    if lhs.shape != rhs.shape:
        raise ValueError(f"shape mismatch: {tuple(lhs.shape)} != {tuple(rhs.shape)}")
    lhs32, rhs32 = lhs.float(), rhs.float()
    if not torch.isfinite(lhs32).all() or not torch.isfinite(rhs32).all():
        raise ValueError("non-finite value found in backend output")
    diff = (lhs32 - rhs32).abs()
    cosine = torch.nn.functional.cosine_similarity(
        lhs32.reshape(1, -1), rhs32.reshape(1, -1)
    ).item()
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "cosine_similarity": cosine,
    }


def compare_backends(
    baseline_name: str,
    outputs: dict[str, torch.Tensor],
    *,
    threshold: float,
) -> dict[str, dict[str, float]]:
    if baseline_name not in outputs:
        raise ValueError(f"baseline backend {baseline_name!r} is missing")
    baseline = outputs[baseline_name]
    results = {}
    for name, output in outputs.items():
        if name == baseline_name:
            continue
        metrics = compare_outputs(baseline, output)
        if metrics["cosine_similarity"] < threshold:
            raise RuntimeError(
                f"{name} correctness failed against {baseline_name}: {metrics}"
            )
        results[f"{name}_vs_{baseline_name}"] = metrics
    return results


def validate_backends(backends: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(backends)
    unknown = [name for name in selected if name not in ALL_BACKENDS]
    if unknown:
        raise ValueError(f"unknown backend(s): {', '.join(unknown)}")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate backends are not allowed")
    if "cutlass" not in selected:
        raise ValueError("cutlass is required as the correctness baseline")
    return selected


def format_markdown(
    rows: Iterable[dict[str, float]],
    backends: Iterable[str] = ALL_BACKENDS,
) -> str:
    selected = validate_backends(backends)
    headers = ["M"]
    for backend in selected:
        label = BACKEND_LABELS[backend]
        headers.extend(
            [
                f"{label} mean (us)",
                f"{label} P50 (us)",
                f"{label} tokens/s",
            ]
        )
        if backend != "cutlass":
            headers.append(f"{label}/CUTLASS")
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" for _ in headers) + "|",
    ]
    for row in rows:
        cells = [f"{int(row['m'])}"]
        for backend in selected:
            cells.extend(
                [
                    f"{row[f'{backend}_mean_us']:.2f}",
                    f"{row[f'{backend}_p50_us']:.2f}",
                    f"{row[f'{backend}_tokens_per_s']:.0f}",
                ]
            )
            if backend != "cutlass":
                cells.append(
                    f"{row[f'{backend}_speedup_vs_cutlass']:.3f}x"
                )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def backend_order(
    index: int,
    backends: Iterable[str] = ALL_BACKENDS,
) -> tuple[str, ...]:
    selected = validate_backends(backends)
    offset = index % len(selected)
    return selected[offset:] + selected[:offset]


def _make_weights(
    shape: BenchmarkShape, *, seed: int, device: torch.device
) -> dict[str, torch.Tensor]:
    validate_shape(shape)
    generator = torch.Generator(device=device).manual_seed(seed)
    e, k, n, group = (
        shape.num_experts,
        shape.hidden_size,
        shape.intermediate_size,
        shape.group_size,
    )
    w13 = torch.randint(
        0, 256, (e, 2 * n, k // 2), dtype=torch.uint8, device=device, generator=generator
    )
    w2 = torch.randint(
        0, 256, (e, k, n // 2), dtype=torch.uint8, device=device, generator=generator
    )
    # Positive, narrow scales keep the synthetic output finite.
    w13_scale = (
        torch.rand(
            (e, 2 * n, k // group),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.01
        + 0.005
    ).to(torch.bfloat16)
    w2_scale = (
        torch.rand(
            (e, k, n // group),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.01
        + 0.005
    ).to(torch.bfloat16)
    return {
        "w13": w13,
        "w2": w2,
        "w13_scale": w13_scale,
        "w2_scale": w2_scale,
    }


def _prepare_weights(
    shape: BenchmarkShape,
    raw: dict[str, torch.Tensor],
    backends: Iterable[str] = ALL_BACKENDS,
) -> dict[str, Any]:
    selected = validate_backends(backends)
    from sglang.srt.layers.quantization.w4afp8 import (
        get_cutlass_w4a8_scale_pack,
        interleave_scales,
    )

    device = raw["w13"].device
    e, k, n = shape.num_experts, shape.hidden_size, shape.intermediate_size
    a1_scale = torch.tensor([0.125], dtype=torch.float32, device=device)
    a2_scale = torch.tensor([0.125], dtype=torch.float32, device=device)

    cutlass = {
        "w13": raw["w13"].view(torch.int8),
        "w2": raw["w2"].view(torch.int8),
        "w13_scale": interleave_scales(
            raw["w13_scale"],
            get_cutlass_w4a8_scale_pack(k, shape.group_size),
        ),
        "w2_scale": interleave_scales(
            raw["w2_scale"],
            get_cutlass_w4a8_scale_pack(n, shape.group_size),
        ),
        "a1_scale": a1_scale,
        "a2_scale": a2_scale,
        "a_strides1": torch.full((e, 3), k, dtype=torch.int64, device=device),
        "c_strides1": torch.full((e, 3), 2 * n, dtype=torch.int64, device=device),
        "a_strides2": torch.full((e, 3), n, dtype=torch.int64, device=device),
        "c_strides2": torch.full((e, 3), k, dtype=torch.int64, device=device),
        "expert_offsets": torch.empty(e + 1, dtype=torch.int32, device=device),
        "problem_sizes1": torch.empty((e, 3), dtype=torch.int32, device=device),
        "problem_sizes2": torch.empty((e, 3), dtype=torch.int32, device=device),
    }
    cutlass["b_strides1"] = cutlass["a_strides1"]
    cutlass["s_strides13"] = cutlass["c_strides1"]
    cutlass["b_strides2"] = cutlass["a_strides2"]
    cutlass["s_strides2"] = cutlass["c_strides2"]
    prepared: dict[str, Any] = {"cutlass": cutlass}

    if "flashinfer" in selected:
        from flashinfer.fused_moe import (
            interleave_moe_weights_for_sm90_mixed_gemm,
        )
        from sglang.srt.layers.quantization.w4afp8 import (
            interleave_flashinfer_w4a8_scales,
        )

        dtype = torch.bfloat16
        empty = torch.empty(0, dtype=dtype, device=device)
        fc1_prequant = torch.full(
            (k,), 1 / a1_scale.item(), dtype=dtype, device=device
        )
        fc2_prequant = torch.full(
            (n,), 1 / a2_scale.item(), dtype=dtype, device=device
        )
        # SGLang stores [gate, up], while FlashInfer's fused SwiGLU contract
        # is [up, gate].
        w13_gate, w13_up = raw["w13"].chunk(2, dim=1)
        w13_scale_gate, w13_scale_up = raw["w13_scale"].chunk(2, dim=1)
        flashinfer_w13 = torch.cat((w13_up, w13_gate), dim=1).contiguous()
        flashinfer_w13_scale = torch.cat(
            (w13_scale_up, w13_scale_gate), dim=1
        ).contiguous()
        prepared["flashinfer"] = {
            "w13": interleave_moe_weights_for_sm90_mixed_gemm(
                flashinfer_w13, "int4"
            ),
            "w2": interleave_moe_weights_for_sm90_mixed_gemm(
                raw["w2"], "int4"
            ),
            "quant_scales": (
                interleave_flashinfer_w4a8_scales(
                    flashinfer_w13_scale,
                    k=k,
                    group_size=shape.group_size,
                ),
                interleave_flashinfer_w4a8_scales(
                    raw["w2_scale"],
                    k=n,
                    group_size=shape.group_size,
                ),
                fc1_prequant,
                fc2_prequant,
                empty,
                empty,
                torch.full(
                    (e,),
                    a1_scale.item(),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.full(
                    (e,),
                    a2_scale.item(),
                    dtype=torch.float32,
                    device=device,
                ),
            ),
        }

    if "humming" in selected:
        from sglang.srt.layers.moe.moe_runner.humming_w4a8 import (
            prepare_humming_w4a8_layer,
        )

        humming = torch.nn.Module()
        humming.params_dtype = torch.bfloat16
        humming.w13_weight = torch.nn.Parameter(
            raw["w13"].view(torch.int8),
            requires_grad=False,
        )
        humming.w2_weight = torch.nn.Parameter(
            raw["w2"].view(torch.int8),
            requires_grad=False,
        )
        humming.w13_weight_scale_inv = torch.nn.Parameter(
            raw["w13_scale"],
            requires_grad=False,
        )
        humming.w2_weight_scale_inv = torch.nn.Parameter(
            raw["w2_scale"],
            requires_grad=False,
        )
        prepare_humming_w4a8_layer(humming, group_size=shape.group_size)
        prepared["humming"] = humming

    return prepared


def _make_tokens(
    shape: BenchmarkShape, m: int, *, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = (
        torch.randn(
            (m, shape.hidden_size),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.1
    ).to(torch.bfloat16)
    logits = torch.randn(
        (m, shape.num_experts),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), shape.top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return x, weights.float(), ids.int()


def _dequant_int4(
    packed: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    unpacked = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    return (
        unpacked.float()
        * scales.float().repeat_interleave(group_size, dim=-1)
    ).to(torch.bfloat16)


def _reference_call(
    shape: BenchmarkShape,
    raw: dict[str, torch.Tensor],
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    x, topk_weights, topk_ids = tokens
    w13 = _dequant_int4(raw["w13"], raw["w13_scale"], shape.group_size)
    w2 = _dequant_int4(raw["w2"], raw["w2_scale"], shape.group_size)
    gate, up = w13.chunk(2, dim=1)
    output = torch.zeros_like(x)
    a1_scale = 0.125
    a2_scale = 0.125
    for expert_id in range(shape.num_experts):
        mask = topk_ids == expert_id
        if not mask.any():
            continue
        batch_idx, nth_expert = torch.where(mask)
        expert_input = (
            (x[batch_idx].float() / a1_scale)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(torch.bfloat16)
        )
        gate_output = (expert_input @ gate[expert_id].T) * a1_scale
        up_output = (expert_input @ up[expert_id].T) * a1_scale
        intermediate = torch.nn.functional.silu(gate_output) * up_output
        intermediate_q = (
            (intermediate.float() / a2_scale)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(torch.bfloat16)
        )
        expert_output = (intermediate_q @ w2[expert_id].T) * a2_scale
        output[batch_idx] += (
            topk_weights[batch_idx, nth_expert, None] * expert_output
        )
    return output


def _cutlass_call(
    shape: BenchmarkShape,
    prepared: dict[str, Any],
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Callable[[], torch.Tensor]:
    from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe

    p = prepared["cutlass"]
    x, topk_weights, topk_ids = tokens

    def call() -> torch.Tensor:
        return cutlass_w4a8_moe(
            x,
            p["w13"],
            p["w2"],
            p["w13_scale"],
            p["w2_scale"],
            topk_weights,
            topk_ids,
            p["a_strides1"],
            p["b_strides1"],
            p["c_strides1"],
            p["a_strides2"],
            p["b_strides2"],
            p["c_strides2"],
            p["s_strides13"],
            p["s_strides2"],
            p["expert_offsets"],
            p["problem_sizes1"],
            p["problem_sizes2"],
            p["a1_scale"],
            p["a2_scale"],
            group_size=shape.group_size,
        )

    return call


def _ensure_single_rank_runtime() -> None:
    global _PARALLEL_OVERRIDE
    if _PARALLEL_OVERRIDE is not None:
        return
    from sglang.srt.runtime_context import get_parallel

    _PARALLEL_OVERRIDE = get_parallel().override(moe_ep_size=1)
    _PARALLEL_OVERRIDE.__enter__()
    atexit.register(_PARALLEL_OVERRIDE.__exit__, None, None, None)


def _flashinfer_call(
    shape: BenchmarkShape,
    prepared: dict[str, Any],
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Callable[[], torch.Tensor]:
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    p = prepared["flashinfer"]
    x, topk_weights, topk_ids = tokens
    output = torch.empty_like(x)
    tune_max_tokens = 1 << max(0, (x.shape[0] - 1).bit_length())

    def call() -> torch.Tensor:
        cutlass_fused_moe(
            input=x,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=p["w13"],
            fc2_expert_weights=p["w2"],
            output_dtype=torch.bfloat16,
            quant_scales=p["quant_scales"],
            output=output,
            use_w4_group_scaling=True,
            use_packed_weights=True,
            tune_max_num_tokens=tune_max_tokens,
            activation_type=ActivationType.Swiglu,
            tp_size=shape.tp_size,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
        )
        return output

    return call


def _ensure_sglang_runtime() -> None:
    from sglang.srt.runtime_context import get_context

    context = get_context()
    try:
        context.server_args
    except ValueError:
        context.set_server_args(
            SimpleNamespace(
                enable_deterministic_inference=False,
                enable_fused_moe_sum_all_reduce=False,
            )
        )


def _triton_call(
    shape: BenchmarkShape,
    raw: dict[str, torch.Tensor],
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Callable[[], torch.Tensor]:
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_experts,
    )

    _ensure_sglang_runtime()
    x, topk_weights, topk_ids = tokens
    runner_config = MoeRunnerConfig(
        num_experts=shape.num_experts,
        num_local_experts=shape.num_experts,
        hidden_size=shape.hidden_size,
        intermediate_size_per_partition=shape.intermediate_size,
        top_k=shape.top_k,
        inplace=False,
        routed_scaling_factor=1.0,
    )

    def call() -> torch.Tensor:
        return fused_experts(
            x,
            raw["w13"].view(torch.int8),
            raw["w2"].view(torch.int8),
            (topk_weights, topk_ids, None),
            moe_runner_config=runner_config,
            use_int4_w4a8=True,
            w1_scale=raw["w13_scale"],
            w2_scale=raw["w2_scale"],
            block_shape=[0, shape.group_size],
        )

    return call


def _humming_call(
    shape: BenchmarkShape,
    prepared: dict[str, Any],
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> Callable[[], torch.Tensor]:
    from sglang.srt.layers.moe.moe_runner.humming_w4a8 import (
        humming_w4a8_moe,
    )

    layer = prepared["humming"]
    x, topk_weights, topk_ids = tokens

    def call() -> torch.Tensor:
        return humming_w4a8_moe(
            layer,
            x,
            topk_weights,
            topk_ids,
        )

    return call


def _make_backend_calls(
    backends: Iterable[str],
    shape: BenchmarkShape,
    raw: dict[str, torch.Tensor],
    prepared: dict[str, Any],
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Callable[[], torch.Tensor]]:
    selected = validate_backends(backends)
    factories = {
        "cutlass": lambda: _cutlass_call(shape, prepared, tokens),
        "flashinfer": lambda: _flashinfer_call(shape, prepared, tokens),
        "triton": lambda: _triton_call(shape, raw, tokens),
        "humming": lambda: _humming_call(shape, prepared, tokens),
    }
    return {backend: factories[backend]() for backend in selected}


def _autotune(call: Callable[[], torch.Tensor]) -> None:
    try:
        from flashinfer.autotuner import autotune
    except ImportError:
        context = nullcontext()
    else:
        context = autotune(True)
    with context:
        call()
    torch.cuda.synchronize()


def _time_call(
    call: Callable[[], torch.Tensor],
    *,
    warmup: int,
    samples: int,
    target_sample_ms: float,
) -> dict[str, float | int]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    end.record()
    torch.cuda.synchronize()
    pilot_us = max(start.elapsed_time(end) * 1000, 1.0)
    iterations = max(1, min(2000, math.ceil(target_sample_ms * 1000 / pilot_us)))
    measurements = []
    for _ in range(samples):
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        torch.cuda.synchronize()
        measurements.append(start.elapsed_time(end) * 1000 / iterations)
    return {
        "mean_us": statistics.mean(measurements),
        "p50_us": statistics.median(measurements),
        "iterations_per_sample": iterations,
    }


def build_result_row(
    m: int,
    timings: dict[str, dict[str, float | int]],
    backends: Iterable[str] = ALL_BACKENDS,
) -> dict[str, float | int]:
    selected = validate_backends(backends)
    row: dict[str, float | int] = {"m": m}
    for backend in selected:
        timing = timings[backend]
        mean_us = float(timing["mean_us"])
        row[f"{backend}_mean_us"] = mean_us
        row[f"{backend}_p50_us"] = float(timing["p50_us"])
        row[f"{backend}_tokens_per_s"] = m * 1_000_000 / mean_us
        row[f"{backend}_iterations_per_sample"] = int(
            timing["iterations_per_sample"]
        )
    cutlass_mean = float(row["cutlass_mean_us"])
    for backend in selected:
        if backend != "cutlass":
            row[f"{backend}_speedup_vs_cutlass"] = cutlass_mean / float(
                row[f"{backend}_mean_us"]
            )
    return row


def run_correctness(
    threshold: float,
    backends: Iterable[str] = ALL_BACKENDS,
) -> dict[str, Any]:
    selected = validate_backends(backends)
    shape = BenchmarkShape(
        hidden_size=512,
        intermediate_size=512,
        num_experts=2,
        top_k=2,
        tp_size=1,
    )
    raw = _make_weights(shape, seed=7, device=torch.device("cuda"))
    prepared = _prepare_weights(shape, raw, selected)
    tokens = _make_tokens(shape, 4, seed=11, device=torch.device("cuda"))
    calls = _make_backend_calls(selected, shape, raw, prepared, tokens)
    outputs = {name: call() for name, call in calls.items()}
    torch.cuda.synchronize()
    metrics = compare_backends("cutlass", outputs, threshold=threshold)
    return {"shape": asdict(shape), "comparisons": metrics}


def run_shape_correctness(
    shape: BenchmarkShape,
    raw: dict[str, torch.Tensor],
    prepared: dict[str, Any],
    *,
    m: int,
    threshold: float,
    backends: Iterable[str] = ALL_BACKENDS,
) -> dict[str, Any]:
    selected = validate_backends(backends)
    tokens = _make_tokens(shape, m, seed=29 + m, device=torch.device("cuda"))
    calls = _make_backend_calls(selected, shape, raw, prepared, tokens)
    return compare_backend_calls(
        shape,
        m,
        calls,
        threshold=threshold,
    )


def compare_backend_calls(
    shape: BenchmarkShape,
    m: int,
    calls: dict[str, Callable[[], torch.Tensor]],
    *,
    threshold: float,
) -> dict[str, Any]:
    outputs = {name: call() for name, call in calls.items()}
    torch.cuda.synchronize()
    return {
        "shape": asdict(shape),
        "m": m,
        "comparisons": compare_backends(
            "cutlass",
            outputs,
            threshold=threshold,
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (9, 0):
        raise RuntimeError(f"SM90 is required, found SM{capability[0]}{capability[1]}")

    selected = validate_backends(args.backends)
    _ensure_single_rank_runtime()
    correctness = run_correctness(args.correctness_threshold, selected)
    print("correctness:", json.dumps(correctness, sort_keys=True), flush=True)

    intermediate_size = resolve_local_intermediate_size(
        model_intermediate_size=args.model_intermediate_size,
        tp_size=args.tp_size,
        override=args.intermediate_size,
    )
    shape = BenchmarkShape(
        hidden_size=args.hidden_size,
        intermediate_size=intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        group_size=args.group_size,
        tp_size=args.tp_size,
    )
    validate_shape(shape)
    raw = _make_weights(shape, seed=args.seed, device=torch.device("cuda"))
    prepared = _prepare_weights(shape, raw, selected)
    benchmark_shape_correctness = []
    if args.correctness_only:
        for m in args.m_values:
            result = run_shape_correctness(
                shape,
                raw,
                prepared,
                m=m,
                threshold=args.correctness_threshold,
                backends=selected,
            )
            benchmark_shape_correctness.append(result)
            print(
                "benchmark shape correctness:",
                json.dumps(result, sort_keys=True),
                flush=True,
            )
        return {
            "correctness": correctness,
            "benchmark_shape_correctness": benchmark_shape_correctness,
            "backends": selected,
        }

    versions = {}
    if "flashinfer" in selected:
        import flashinfer

        versions["flashinfer_version"] = getattr(
            flashinfer, "__version__", "unknown"
        )
    if "humming" in selected:
        import humming

        versions["humming_version"] = getattr(humming, "__version__", "unknown")
    if "triton" in selected:
        import triton

        versions["triton_version"] = triton.__version__

    torch.cuda.empty_cache()

    rows = []
    for index, m in enumerate(args.m_values):
        tokens = _make_tokens(shape, m, seed=args.seed + m, device=torch.device("cuda"))
        calls = _make_backend_calls(selected, shape, raw, prepared, tokens)
        shape_correctness = compare_backend_calls(
            shape,
            m,
            calls,
            threshold=args.correctness_threshold,
        )
        benchmark_shape_correctness.append(shape_correctness)
        print(
            "benchmark shape correctness:",
            json.dumps(shape_correctness, sort_keys=True),
            flush=True,
        )
        if "flashinfer" in calls:
            _autotune(calls["flashinfer"])
        if "triton" in calls:
            calls["triton"]()
        if "humming" in calls:
            calls["humming"]()
        torch.cuda.synchronize()
        timings: dict[str, dict[str, float | int]] = {}
        for name in backend_order(index, selected):
            timings[name] = _time_call(
                calls[name],
                warmup=args.warmup,
                samples=args.samples,
                target_sample_ms=args.target_sample_ms,
            )
        row = build_result_row(m, timings, selected)
        rows.append(row)
        print(format_markdown([row], selected), flush=True)

    result = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": capability,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "backends": selected,
        "shape": asdict(shape),
        "correctness": correctness,
        "benchmark_shape_correctness": benchmark_shape_correctness,
        "settings": {
            "warmup": args.warmup,
            "samples": args.samples,
            "target_sample_ms": args.target_sample_ms,
            "seed": args.seed,
            "correctness_threshold": args.correctness_threshold,
            "model_intermediate_size": args.model_intermediate_size,
        },
        "rows": rows,
    }
    result.update(versions)
    print("\n" + format_markdown(rows, selected), flush=True)
    return result


def _write_json_atomic(path: str, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        os.unlink(temporary)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[1, 8, 32, 64, 128, 256, 1024, 4096, 8192],
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--model-intermediate-size",
        type=int,
        default=2048,
        help="Unsharded model MoE intermediate size.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=8,
        help=(
            "Simulated tensor-parallel size; for the default model, "
            "TP2/TP4/TP8 map local N to 1024/512/256."
        ),
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=None,
        help="Per-rank intermediate size override; defaults to model size / TP size.",
    )
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--target-sample-ms", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(ALL_BACKENDS),
        help="Ordered backend subset; cutlass is required as baseline.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--correctness-threshold", type=float, default=0.98)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    if args.output_json:
        _write_json_atomic(args.output_json, result)


if __name__ == "__main__":
    main()
