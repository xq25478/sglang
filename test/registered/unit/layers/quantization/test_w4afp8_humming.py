from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[5]
W4AFP8_PATH = ROOT / "python/sglang/srt/layers/quantization/w4afp8.py"
MOE_UTILS_PATH = ROOT / "python/sglang/srt/layers/moe/utils.py"
SERVER_ARGS_PATH = ROOT / "python/sglang/srt/server_args.py"
HUMMING_PATH = ROOT / "python/sglang/srt/layers/moe/moe_runner/humming_w4a8.py"


def test_humming_is_registered_as_moe_runner_backend():
    moe_utils = MOE_UTILS_PATH.read_text()
    server_args = SERVER_ARGS_PATH.read_text()
    assert 'HUMMING = "humming"' in moe_utils
    assert "def is_humming(self)" in moe_utils
    assert '"humming",' in server_args
    assert 'view.moe_runner_backend == "humming"' in server_args
    assert 'view.quantization == "w4afp8"' in server_args
    assert "Humming W4A8 supports only moe_a2a_backend='none'" in server_args


def test_humming_f16_accum_env_is_registered(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.delenv("SGLANG_HUMMING_USE_F16_ACCUM", raising=False)
    assert envs.SGLANG_HUMMING_USE_F16_ACCUM.get() is False

    monkeypatch.setenv("SGLANG_HUMMING_USE_F16_ACCUM", "1")
    assert envs.SGLANG_HUMMING_USE_F16_ACCUM.get() is True


def _load_humming_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("humming_w4a8", HUMMING_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_signed_int4_storage_conversion_preserves_all_nibbles():
    assert HUMMING_PATH.exists()
    humming_w4a8 = _load_humming_module()
    packed = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        dtype=torch.uint8,
    )
    converted = humming_w4a8.signed_int4_to_humming_storage(packed)
    low = (converted & 0x0F).to(torch.int16) - 8
    high = ((converted >> 4) & 0x0F).to(torch.int16) - 8
    actual = torch.stack((low, high), dim=-1).reshape(-1)
    expected = torch.arange(0, 16, dtype=torch.int16)
    expected = torch.where(expected >= 8, expected - 16, expected)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(packed, torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        dtype=torch.uint8,
    ))


def test_humming_packed_weight_uses_int32_contract():
    humming_w4a8 = _load_humming_module()
    packed = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    converted = humming_w4a8.pack_signed_int4_for_humming(packed)
    assert converted.dtype == torch.int32
    assert converted.shape == (2, 2)


def test_humming_w4a8_accepts_only_group_128():
    humming_w4a8 = _load_humming_module()
    humming_w4a8.validate_humming_w4a8_group_size(128)
    with pytest.raises(ValueError, match="group_size=128"):
        humming_w4a8.validate_humming_w4a8_group_size(32)


def test_humming_runtime_version_is_fail_closed():
    humming_w4a8 = _load_humming_module()
    humming_w4a8.validate_humming_runtime_version("0.1.11")
    with pytest.raises(RuntimeError, match="requires humming-kernels==0.1.11"):
        humming_w4a8.validate_humming_runtime_version("0.1.10")


def test_humming_padding_maps_negative_expert_ids_to_zero_weight():
    humming_w4a8 = _load_humming_module()
    topk_ids = torch.tensor([[7, -1, 3], [-1, -1, 9]], dtype=torch.int32)
    topk_weights = torch.tensor(
        [[0.5, 0.3, 0.2], [0.6, 0.4, 1.0]], dtype=torch.float32
    )

    actual_ids, actual_weights = humming_w4a8.sanitize_humming_topk_padding_(
        topk_ids, topk_weights
    )

    torch.testing.assert_close(
        actual_ids,
        torch.tensor([[7, 0, 3], [0, 0, 9]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        actual_weights,
        torch.tensor([[0.5, 0.0, 0.2], [0.0, 0.0, 1.0]], dtype=torch.float32),
    )


def test_humming_packed_k_uses_at_least_two_n_fragments_per_warp():
    humming_w4a8 = _load_humming_module()
    configs = [
        [
            168,
            2**30,
            {
                "block_shape": (64, 128, 128),
                "warp_shape": (64, 16, 128),
            },
        ]
    ]

    humming_w4a8.ensure_humming_packed_k_warp_compatibility(configs)

    assert configs[0][2]["warp_shape"] == (64, 32, 128)


def test_prepare_uses_real_humming_schema_fields_and_weight_device(monkeypatch):
    humming_w4a8 = _load_humming_module()
    captured = {}

    class FakeWeightSchema:
        def __init__(
            self,
            *,
            b_dtype,
            bs_dtype,
            weight_scale_group_size,
            weight_scale_type,
            has_zero_point,
        ):
            captured["weight_group_size"] = weight_scale_group_size

    class FakeInputSchema:
        def __init__(self, *, a_dtype, input_scale_group_size):
            captured["input_group_size"] = input_scale_group_size

    class FakeHummingMethod:
        @classmethod
        def prepare_layer_meta(cls, **kwargs):
            captured.setdefault("meta_calls", []).append(kwargs)
            kwargs["layer"].humming_metas = {
                **getattr(kwargs["layer"], "humming_metas", {}),
                kwargs["sublayer_name"]: SimpleNamespace(),
            }

        @classmethod
        def transform_humming_layer(cls, layer, sublayer_name):
            captured.setdefault("transform_calls", []).append(sublayer_name)

    monkeypatch.setattr(
        humming_w4a8,
        "_humming_imports",
        lambda: (object(), FakeHummingMethod, FakeInputSchema, FakeWeightSchema),
    )
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros((2, 512, 256), dtype=torch.int8),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros((2, 512, 128), dtype=torch.int8),
        requires_grad=False,
    )
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones((2, 512, 4), dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.ones((2, 512, 2), dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.params_dtype = torch.bfloat16

    humming_w4a8.prepare_humming_w4a8_layer(layer, group_size=128)

    assert captured["weight_group_size"] == 128
    assert captured["input_group_size"] == 128
    assert captured["transform_calls"] == ["w13", "w2"]
    assert layer.locks.device == layer.w13_weight.device


def test_runtime_uses_group_quantization_and_independent_gemm_alignment():
    source = HUMMING_PATH.read_text()
    assert "may_hadamard_quant_input" in source
    assert source.count("moe_align_block_size(") >= 2
    assert "moe_fused_mul_sum" in source
    assert "envs.SGLANG_HUMMING_USE_F16_ACCUM.get()" in source
    assert "swiglu_limit: float | None = None" in source
    assert "silu_and_mul_clamp" in source


def test_w4afp8_dispatches_to_humming_without_generic_moe_runner():
    source = W4AFP8_PATH.read_text()
    assert "runner_backend.is_humming()" in source
    assert "prepare_humming_w4a8_layer" in source
    assert "humming_w4a8_moe" in source
    assert "swiglu_limit=self.moe_runner_config.swiglu_limit" in source


def test_w4afp8_humming_rejects_deepep_dispatch_explicitly():
    source = W4AFP8_PATH.read_text()
    assert source.count("_reject_humming_nonstandard_dispatch(") >= 3
    assert "Standard/TP MoE dispatch." in source
