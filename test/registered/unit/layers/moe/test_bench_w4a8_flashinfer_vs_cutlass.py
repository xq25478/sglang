from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import pytest
import torch


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[4]
    / "manual"
    / "layers"
    / "moe"
    / "bench_w4a8_flashinfer_vs_cutlass.py"
)
SPEC = importlib.util.spec_from_file_location("bench_w4a8", BENCHMARK_PATH)
assert SPEC is not None and SPEC.loader is not None
bench_w4a8 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_w4a8
SPEC.loader.exec_module(bench_w4a8)


def test_scale_interleave_factor():
    assert bench_w4a8.scale_interleave_factor(4096) == 4
    assert bench_w4a8.scale_interleave_factor(256) == 2
    assert bench_w4a8.scale_interleave_factor(128) == 1


def test_validate_real_shape():
    shape = bench_w4a8.BenchmarkShape()
    assert shape == bench_w4a8.BenchmarkShape(
        hidden_size=4096,
        intermediate_size=256,
        num_experts=256,
        top_k=6,
        group_size=128,
        tp_size=8,
    )
    bench_w4a8.validate_shape(shape)


@pytest.mark.parametrize(
    "tp_size,expected",
    [
        (2, 1024),
        (4, 512),
        (8, 256),
    ],
)
def test_resolve_tp_local_intermediate_size(tp_size, expected):
    assert (
        bench_w4a8.resolve_local_intermediate_size(
            model_intermediate_size=2048,
            tp_size=tp_size,
            override=None,
        )
        == expected
    )


def test_resolve_tp_local_intermediate_size_honors_override():
    assert (
        bench_w4a8.resolve_local_intermediate_size(
            model_intermediate_size=2048,
            tp_size=4,
            override=768,
        )
        == 768
    )


@pytest.mark.parametrize(
    "model_intermediate_size,tp_size,message",
    [
        (2048, 0, "tp_size"),
        (2050, 4, "divisible"),
    ],
)
def test_resolve_tp_local_intermediate_size_rejects_invalid_inputs(
    model_intermediate_size, tp_size, message
):
    with pytest.raises(ValueError, match=message):
        bench_w4a8.resolve_local_intermediate_size(
            model_intermediate_size=model_intermediate_size,
            tp_size=tp_size,
            override=None,
        )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("hidden_size", 4100, "hidden_size"),
        ("intermediate_size", 2050, "intermediate_size"),
        ("top_k", 257, "top_k"),
        ("group_size", 64, "group_size"),
        ("tp_size", 0, "tp_size"),
    ],
)
def test_validate_shape_rejects_invalid_contract(field, value, message):
    kwargs = {
        "hidden_size": 4096,
        "intermediate_size": 2048,
        "num_experts": 256,
        "top_k": 6,
        "group_size": 128,
        "tp_size": 8,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        bench_w4a8.validate_shape(bench_w4a8.BenchmarkShape(**kwargs))


def test_compare_outputs_reports_metrics():
    metrics = bench_w4a8.compare_outputs(
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[1.0, 2.5]]),
    )
    assert metrics["max_abs_error"] == pytest.approx(0.5)
    assert metrics["mean_abs_error"] == pytest.approx(0.25)
    assert metrics["cosine_similarity"] == pytest.approx(0.99654576)


def test_compare_outputs_rejects_nonfinite():
    with pytest.raises(ValueError, match="non-finite"):
        bench_w4a8.compare_outputs(
            torch.tensor([float("nan")]),
            torch.tensor([0.0]),
        )


def test_group_128_scale_interleave_matches_trtllm_layout():
    scales = torch.arange(2 * 4 * 8).reshape(2, 4, 8)
    actual = bench_w4a8.interleave_group_scales(scales, dim=4096)
    expected = (
        scales.reshape(2, 4, 2, 4)
        .permute(0, 2, 1, 3)
        .reshape(2, 2, 16)
        .contiguous()
    )
    torch.testing.assert_close(actual, expected)


def test_group_scale_interleave_rejects_invalid_rank():
    with pytest.raises(ValueError, match="3D"):
        bench_w4a8.interleave_group_scales(torch.ones(2, 8), dim=4096)


def test_group_scale_interleave_rejects_incompatible_groups():
    with pytest.raises(ValueError, match="divisible"):
        bench_w4a8.interleave_group_scales(torch.ones(2, 4, 6), dim=4096)


def test_markdown_contains_required_performance_columns():
    table = bench_w4a8.format_markdown(
        [
            {
                "m": 8,
                "cutlass_mean_us": 20.0,
                "cutlass_p50_us": 19.0,
                "cutlass_tokens_per_s": 400_000.0,
                "flashinfer_mean_us": 10.0,
                "flashinfer_p50_us": 9.0,
                "flashinfer_tokens_per_s": 800_000.0,
                "flashinfer_speedup_vs_cutlass": 2.0,
                "triton_mean_us": 12.0,
                "triton_p50_us": 11.0,
                "triton_tokens_per_s": 666_666.0,
                "triton_speedup_vs_cutlass": 1.667,
                "humming_mean_us": 11.0,
                "humming_p50_us": 10.0,
                "humming_tokens_per_s": 727_273.0,
                "humming_speedup_vs_cutlass": 1.818,
            }
        ]
    )
    assert "CUTLASS mean (us)" in table
    assert "FlashInfer P50 (us)" in table
    assert "Triton mean (us)" in table
    assert "Triton/CUTLASS" in table
    assert "Humming mean (us)" in table
    assert "Humming/CUTLASS" in table
    assert "tokens/s" in table


def test_backend_order_rotates_all_four_backends():
    assert bench_w4a8.backend_order(0) == (
        "cutlass",
        "flashinfer",
        "triton",
        "humming",
    )
    assert bench_w4a8.backend_order(1) == (
        "flashinfer",
        "triton",
        "humming",
        "cutlass",
    )
    assert bench_w4a8.backend_order(2) == (
        "triton",
        "humming",
        "cutlass",
        "flashinfer",
    )
    assert bench_w4a8.backend_order(3) == (
        "humming",
        "cutlass",
        "flashinfer",
        "triton",
    )
    assert bench_w4a8.backend_order(4) == bench_w4a8.backend_order(0)


def test_cutlass_humming_subset_is_valid_and_rotates():
    backends = bench_w4a8.validate_backends(["cutlass", "humming"])
    assert backends == ("cutlass", "humming")
    assert bench_w4a8.backend_order(0, backends) == ("cutlass", "humming")
    assert bench_w4a8.backend_order(1, backends) == ("humming", "cutlass")


@pytest.mark.parametrize(
    "backends,message",
    [
        (["humming"], "cutlass"),
        (["cutlass", "unknown"], "unknown"),
        (["cutlass", "humming", "humming"], "duplicate"),
    ],
)
def test_backend_selection_rejects_invalid_subsets(backends, message):
    with pytest.raises(ValueError, match=message):
        bench_w4a8.validate_backends(backends)


def test_cutlass_humming_markdown_omits_unselected_backends():
    timings = {
        "cutlass": {
            "mean_us": 20.0,
            "p50_us": 19.0,
            "iterations_per_sample": 10,
        },
        "humming": {
            "mean_us": 8.0,
            "p50_us": 7.5,
            "iterations_per_sample": 24,
        },
    }
    backends = ("cutlass", "humming")
    row = bench_w4a8.build_result_row(8, timings, backends)
    table = bench_w4a8.format_markdown([row], backends)
    assert "CUTLASS mean (us)" in table
    assert "Humming/CUTLASS" in table
    assert "FlashInfer" not in table
    assert "Triton" not in table
    assert row["humming_speedup_vs_cutlass"] == pytest.approx(2.5)


def test_backend_subset_controls_preparation_and_calls():
    prepare_source = inspect.getsource(bench_w4a8._prepare_weights)
    calls_source = inspect.getsource(bench_w4a8._make_backend_calls)
    assert 'if "flashinfer" in selected' in prepare_source
    assert 'if "humming" in selected' in prepare_source
    assert "for backend in selected" in calls_source


def test_triton_call_uses_production_w4a8_contract():
    source = inspect.getsource(bench_w4a8._triton_call)
    assert "fused_experts" in source
    assert "use_int4_w4a8=True" in source
    assert 'raw["w13_scale"]' in source
    assert 'raw["w2_scale"]' in source
    assert "block_shape=[0, shape.group_size]" in source


def test_humming_call_uses_production_w4a8_contract():
    prepare_source = inspect.getsource(bench_w4a8._prepare_weights)
    call_source = inspect.getsource(bench_w4a8._humming_call)
    assert "prepare_humming_w4a8_layer" in prepare_source
    assert "humming_w4a8_moe" in call_source
    assert 'prepared["humming"]' in call_source


def test_flashinfer_call_receives_simulated_tp_size():
    call_source = inspect.getsource(bench_w4a8._flashinfer_call)
    assert "tp_size=shape.tp_size" in call_source
    assert "tp_rank=0" in call_source


def test_benchmark_checks_the_measured_shape_before_timing():
    source = inspect.getsource(bench_w4a8.run_benchmark)
    assert "run_shape_correctness(" in source
    assert "compare_backend_calls(" in source
    assert "benchmark_shape_correctness.append(" in source


def test_compare_backends_reports_each_nonbaseline_backend():
    outputs = {
        "cutlass": torch.tensor([[1.0, 0.0]]),
        "flashinfer": torch.tensor([[1.0, 0.01]]),
        "triton": torch.tensor([[1.0, -0.01]]),
        "humming": torch.tensor([[1.0, 0.02]]),
    }
    metrics = bench_w4a8.compare_backends("cutlass", outputs, threshold=0.99)
    assert set(metrics) == {
        "flashinfer_vs_cutlass",
        "triton_vs_cutlass",
        "humming_vs_cutlass",
    }
    assert metrics["triton_vs_cutlass"]["cosine_similarity"] > 0.99


def test_compare_backends_rejects_low_cosine():
    outputs = {
        "cutlass": torch.tensor([[1.0, 0.0]]),
        "triton": torch.tensor([[0.0, 1.0]]),
    }
    with pytest.raises(RuntimeError, match="triton.*correctness"):
        bench_w4a8.compare_backends("cutlass", outputs, threshold=0.98)


def test_build_result_row_contains_all_four_backends():
    timings = {
        "cutlass": {
            "mean_us": 20.0,
            "p50_us": 19.0,
            "iterations_per_sample": 10,
        },
        "flashinfer": {
            "mean_us": 10.0,
            "p50_us": 9.0,
            "iterations_per_sample": 20,
        },
        "triton": {
            "mean_us": 12.5,
            "p50_us": 12.0,
            "iterations_per_sample": 16,
        },
        "humming": {
            "mean_us": 8.0,
            "p50_us": 7.5,
            "iterations_per_sample": 24,
        },
    }
    row = bench_w4a8.build_result_row(8, timings)
    assert row["cutlass_tokens_per_s"] == pytest.approx(400_000)
    assert row["flashinfer_speedup_vs_cutlass"] == pytest.approx(2.0)
    assert row["triton_speedup_vs_cutlass"] == pytest.approx(1.6)
    assert row["humming_speedup_vs_cutlass"] == pytest.approx(2.5)
    for backend in ("cutlass", "flashinfer", "triton", "humming"):
        assert f"{backend}_mean_us" in row
        assert f"{backend}_p50_us" in row
        assert f"{backend}_tokens_per_s" in row
        assert f"{backend}_iterations_per_sample" in row
