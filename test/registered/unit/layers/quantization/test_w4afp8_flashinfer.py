from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.flashinfer_w4a8_autotune import (
    RouteAwareProfile,
    RouteRecorder,
    route_recording,
    set_active_profile,
)
from sglang.srt.layers.quantization.w4afp8 import (
    W4AFp8Config,
    W4AFp8MoEMethod,
    build_flashinfer_w4a8_quant_scales,
    interleave_flashinfer_w4a8_scales,
    swap_gate_up,
)
import sglang.srt.layers.quantization.w4afp8 as w4afp8
from sglang.srt.server_args import FLASHINFER_CUTLASS_MOE_QUANTIZATIONS


def test_swap_gate_up_reorders_expert_rows():
    weights = torch.tensor([[[1], [2], [3], [4]]], dtype=torch.int8)

    actual = swap_gate_up(weights)

    torch.testing.assert_close(
        actual,
        torch.tensor([[[3], [4], [1], [2]]], dtype=torch.int8),
    )
    assert actual.is_contiguous()


def test_build_flashinfer_w4a8_quant_scales_matches_kernel_contract():
    w13_scale = torch.ones((2, 4, 3), dtype=torch.bfloat16)
    w2_scale = torch.full((2, 5, 2), 2.0, dtype=torch.bfloat16)
    w13_input_scale = torch.tensor([0.125], dtype=torch.float32)
    w2_input_scale = torch.tensor([0.25], dtype=torch.float32)

    scales = build_flashinfer_w4a8_quant_scales(
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_input_scale=w13_input_scale,
        w2_input_scale=w2_input_scale,
        hidden_size=8,
        intermediate_size=4,
        num_experts=2,
    )

    assert len(scales) == 8
    assert scales[0] is w13_scale
    assert scales[1] is w2_scale
    torch.testing.assert_close(
        scales[2], torch.full((8,), 8.0, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(
        scales[3], torch.full((4,), 4.0, dtype=torch.bfloat16)
    )
    assert scales[4].numel() == 0
    assert scales[5].numel() == 0
    torch.testing.assert_close(scales[6], torch.full((2,), 0.125))
    torch.testing.assert_close(scales[7], torch.full((2,), 0.25))


def test_interleave_flashinfer_w4a8_scales_uses_legacy_layout(monkeypatch):
    scales = torch.arange(2 * 4 * 2, dtype=torch.bfloat16).reshape(2, 4, 2)
    monkeypatch.setattr(
        w4afp8,
        "_interleave_w4a8_scales_with_flashinfer",
        lambda _scales, _group_size: None,
    )

    actual = interleave_flashinfer_w4a8_scales(
        scales, k=256, group_size=128
    )

    expected = (
        scales.reshape(2, 4, 1, 2)
        .permute(0, 2, 1, 3)
        .reshape(2, 1, 8)
        .contiguous()
    )
    torch.testing.assert_close(actual, expected)


def test_interleave_flashinfer_w4a8_scales_prefers_native_layout(monkeypatch):
    scales = torch.ones((2, 64, 2), dtype=torch.bfloat16)
    expected = torch.empty((2, 1, 1, 64, 8), dtype=torch.bfloat16)
    captured = {}

    def fake_native(actual_scales, actual_group_size):
        captured["scales"] = actual_scales
        captured["group_size"] = actual_group_size
        return expected

    monkeypatch.setattr(
        w4afp8,
        "_interleave_w4a8_scales_with_flashinfer",
        fake_native,
        raising=False,
    )

    actual = interleave_flashinfer_w4a8_scales(
        scales, k=256, group_size=128
    )

    assert actual is expected
    assert captured == {"scales": scales, "group_size": 128}


def test_flashinfer_cutlass_runner_forwards_w4a8_contract(monkeypatch):
    captured = {}

    def fake_fused_moe(**kwargs):
        captured.update(kwargs)
        return (kwargs["output"],)

    monkeypatch.setattr(
        flashinfer_cutlass,
        "_flashinfer_cutlass_fused_moe",
        lambda: (fake_fused_moe, object()),
    )
    monkeypatch.setattr(
        flashinfer_cutlass,
        "_activation_type",
        lambda _runner_config: "swiglu",
    )

    x = torch.ones((2, 8), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
    output = torch.empty_like(x)
    w13 = torch.zeros((2, 8, 4), dtype=torch.uint8)
    w2 = torch.zeros((2, 8, 2), dtype=torch.uint8)
    quant_scales = [torch.empty(0) for _ in range(8)]
    dispatch_output = SimpleNamespace(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=SimpleNamespace(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        ),
    )
    quant_info = FlashInferCutlassMoeQuantInfo(
        quant_type="w4a8",
        w13_weight=w13,
        w2_weight=w2,
        quant_scales=quant_scales,
        output_dtype=torch.bfloat16,
    )

    actual = flashinfer_cutlass._run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=MoeRunnerConfig(activation="silu", is_gated=True),
        output=output,
    )

    assert actual is output
    assert captured["fc1_expert_weights"] is w13
    assert captured["fc2_expert_weights"] is w2
    assert captured["quant_scales"] is quant_scales
    assert captured["use_w4_group_scaling"] is True
    assert captured["use_packed_weights"] is True
    assert captured["tune_max_num_tokens"] == 8192


def test_flashinfer_cutlass_runner_uses_route_aware_profile(monkeypatch):
    captured = {}

    def fake_fused_moe(**kwargs):
        captured.update(kwargs)
        return (kwargs["output"],)

    monkeypatch.setattr(
        flashinfer_cutlass,
        "_flashinfer_cutlass_fused_moe",
        lambda: (fake_fused_moe, object()),
    )
    monkeypatch.setattr(
        flashinfer_cutlass,
        "_activation_type",
        lambda _runner_config: "swiglu",
    )

    x = torch.ones((12, 8), dtype=torch.bfloat16)
    topk_ids = torch.zeros((12, 2), dtype=torch.int32)
    output = torch.empty_like(x)
    dispatch_output = SimpleNamespace(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=SimpleNamespace(
            topk_ids=topk_ids,
            topk_weights=torch.full((12, 2), 0.5),
        ),
    )
    quant_info = FlashInferCutlassMoeQuantInfo(
        quant_type="w4a8",
        w13_weight=torch.zeros((2, 8, 4), dtype=torch.uint8),
        w2_weight=torch.zeros((2, 8, 2), dtype=torch.uint8),
        quant_scales=[torch.empty(0) for _ in range(8)],
    )
    set_active_profile(
        RouteAwareProfile(metadata={"schema": 1}, tactics={16: (3, 101)})
    )
    try:
        flashinfer_cutlass._run_flashinfer_cutlass(
            dispatch_output=dispatch_output,
            quant_info=quant_info,
            runner_config=MoeRunnerConfig(activation="silu", is_gated=True),
            output=output,
        )
    finally:
        set_active_profile(None)

    assert captured["profile_ids"] == [3, 101]


def test_flashinfer_cutlass_recording_uses_fallback_and_records_routes(monkeypatch):
    captured = {}

    def fake_fused_moe(**kwargs):
        captured.update(kwargs)
        return (kwargs["output"],)

    monkeypatch.setattr(
        flashinfer_cutlass,
        "_flashinfer_cutlass_fused_moe",
        lambda: (fake_fused_moe, object()),
    )
    monkeypatch.setattr(
        flashinfer_cutlass,
        "_activation_type",
        lambda _runner_config: "swiglu",
    )

    topk_ids = torch.tensor([[0, 1], [0, 2]], dtype=torch.int32)
    dispatch_output = SimpleNamespace(
        hidden_states=torch.ones((2, 8), dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_output=SimpleNamespace(
            topk_ids=topk_ids,
            topk_weights=torch.full((2, 2), 0.5),
        ),
    )
    quant_info = FlashInferCutlassMoeQuantInfo(
        quant_type="w4a8",
        w13_weight=torch.zeros((4, 8, 4), dtype=torch.uint8),
        w2_weight=torch.zeros((4, 8, 2), dtype=torch.uint8),
        quant_scales=[torch.empty(0) for _ in range(8)],
    )
    recorder = RouteRecorder(num_experts=4)

    with route_recording(recorder, "decode"):
        flashinfer_cutlass._run_flashinfer_cutlass(
            dispatch_output=dispatch_output,
            quant_info=quant_info,
            runner_config=MoeRunnerConfig(activation="silu", is_gated=True),
            output=torch.empty((2, 8), dtype=torch.bfloat16),
        )

    assert captured["profile_ids"] == [-1, -1]
    assert recorder.histograms["decode"][0].tolist() == [2, 1, 1, 0]


def test_w4afp8_method_creates_flashinfer_cutlass_runner(monkeypatch):
    backend = SimpleNamespace(is_flashinfer_cutlass=lambda: True)
    runner = object()
    monkeypatch.setattr(
        w4afp8, "get_moe_runner_backend", lambda: backend, raising=False
    )
    monkeypatch.setattr(
        w4afp8,
        "MoeRunner",
        lambda actual_backend, actual_config: (
            runner
            if (actual_backend, actual_config) == (backend, "runner-config")
            else None
        ),
        raising=False,
    )
    method = W4AFp8MoEMethod(W4AFp8Config())

    method.create_moe_runner(SimpleNamespace(), "runner-config")

    assert method.runner is runner
    assert method.fuse_routed_scaling_factor_in_topk is True


def test_w4afp8_apply_routes_prepared_weights_through_flashinfer():
    captured = {}
    expected = object()

    class FakeRunner:
        def run(self, dispatch_output, quant_info):
            captured["dispatch_output"] = dispatch_output
            captured["quant_info"] = quant_info
            return expected

    method = W4AFp8MoEMethod(W4AFp8Config())
    method.runner = FakeRunner()
    method.flashinfer_quant_scales = [torch.empty(0) for _ in range(8)]
    dispatch_output = object()
    layer = SimpleNamespace(
        w13_weight=torch.empty((2, 4, 2), dtype=torch.uint8),
        w2_weight=torch.empty((2, 4, 1), dtype=torch.uint8),
        moe_tp_size=2,
        moe_tp_rank=1,
        moe_ep_size=1,
        moe_ep_rank=0,
    )

    actual = method.apply(layer, dispatch_output)

    assert actual is expected
    assert captured["dispatch_output"] is dispatch_output
    quant_info = captured["quant_info"]
    assert isinstance(quant_info, FlashInferCutlassMoeQuantInfo)
    assert quant_info.quant_type == "w4a8"
    assert quant_info.quant_scales is method.flashinfer_quant_scales
    assert quant_info.moe_tp_size == 2
    assert quant_info.moe_tp_rank == 1
    assert quant_info.group_size == 128
    assert quant_info.apply_routed_scaling_factor is False


def test_server_args_allow_w4afp8_with_flashinfer_cutlass():
    assert "w4afp8" in FLASHINFER_CUTLASS_MOE_QUANTIZATIONS
