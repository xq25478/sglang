from __future__ import annotations

import json
import math
from functools import lru_cache
from importlib.metadata import version as package_version
from typing import Any

import torch
import triton
import triton.language as tl
from torch.nn import Parameter


def signed_int4_to_humming_storage(packed: torch.Tensor) -> torch.Tensor:
    """Convert packed two's-complement INT4 to Humming offset-binary storage."""
    if packed.dtype not in (torch.int8, torch.uint8):
        raise TypeError(f"packed INT4 storage must be int8/uint8, got {packed.dtype}")
    return packed.view(torch.uint8).bitwise_xor(0x88).contiguous()


def pack_signed_int4_for_humming(packed: torch.Tensor) -> torch.Tensor:
    converted = signed_int4_to_humming_storage(packed)
    if converted.shape[-1] % 4:
        raise ValueError("packed INT4 K dimension must be divisible by four bytes")
    return converted.view(torch.int32)


def validate_humming_w4a8_group_size(group_size: int) -> None:
    if group_size != 128:
        raise ValueError(
            "Humming signed-INT4 W4A8 currently requires group_size=128, "
            f"got {group_size}."
        )


def validate_humming_runtime_version(actual_version: str) -> None:
    if actual_version != "0.1.11":
        raise RuntimeError(
            "Humming W4A8 requires humming-kernels==0.1.11, "
            f"got {actual_version}."
        )


def ensure_humming_packed_k_warp_compatibility(
    tuning_configs: list[Any],
) -> None:
    """Satisfy Humming packed-K's two-fragment minimum on the warp N axis."""
    for _, _, config in tuning_configs:
        warp_m, warp_n, warp_k = config["warp_shape"]
        if warp_n < 32:
            config["warp_shape"] = (warp_m, 32, warp_k)


@triton.jit
def _sanitize_humming_topk_padding_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    topk_ids = tl.load(topk_ids_ptr + offsets, mask=mask)
    is_padding = topk_ids < 0
    topk_weights = tl.load(topk_weights_ptr + offsets, mask=mask)
    tl.store(topk_ids_ptr + offsets, tl.where(is_padding, 0, topk_ids), mask=mask)
    tl.store(
        topk_weights_ptr + offsets,
        tl.where(is_padding, 0.0, topk_weights),
        mask=mask,
    )


def sanitize_humming_topk_padding_(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make CUDA padding safe for Humming's indexed GEMM, in place."""
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            "topk_ids and topk_weights must have the same shape, got "
            f"{tuple(topk_ids.shape)} and {tuple(topk_weights.shape)}"
        )
    if topk_ids.numel() == 0:
        return topk_ids, topk_weights
    if topk_ids.is_cuda:
        block_size = 256
        _sanitize_humming_topk_padding_kernel[
            (triton.cdiv(topk_ids.numel(), block_size),)
        ](
            topk_ids,
            topk_weights,
            topk_ids.numel(),
            BLOCK_SIZE=block_size,
        )
    else:
        is_padding = topk_ids < 0
        topk_weights.masked_fill_(is_padding, 0)
        topk_ids.masked_fill_(is_padding, 0)
    return topk_ids, topk_weights


@lru_cache(maxsize=1)
def _humming_imports():
    try:
        validate_humming_runtime_version(package_version("humming-kernels"))
        from humming.config import GemmType
        from humming.layer import HummingMethod
        from humming.schema import HummingInputSchema, HummingWeightSchema
    except ImportError as error:
        raise ImportError(
            "The Humming W4A8 backend requires humming-kernels==0.1.11."
        ) from error
    return GemmType, HummingMethod, HummingInputSchema, HummingWeightSchema


def _replace_parameter(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    setattr(layer, name, Parameter(value, requires_grad=False))


def prepare_humming_w4a8_layer(
    layer: torch.nn.Module,
    *,
    group_size: int,
) -> None:
    """Destructively replace native W4A8 tensors with Humming-formatted ones."""
    validate_humming_w4a8_group_size(group_size)
    _, HummingMethod, HummingInputSchema, HummingWeightSchema = _humming_imports()

    if getattr(layer, "_humming_w4a8_prepared", False):
        return

    for prefix in ("w13", "w2"):
        packed = getattr(layer, f"{prefix}_weight")
        _replace_parameter(
            layer,
            f"{prefix}_weight",
            pack_signed_int4_for_humming(packed),
        )

        scale_name = f"{prefix}_weight_scale_inv"
        scale = getattr(layer, scale_name).to(torch.bfloat16).contiguous()
        delattr(layer, scale_name)
        _replace_parameter(layer, f"{prefix}_weight_scale", scale)

    if not hasattr(layer, "locks"):
        layer.register_buffer(
            "locks",
            torch.zeros(
                1024,
                dtype=torch.int32,
                device=layer.w13_weight.device,
            ),
        )

    num_experts = layer.w13_weight.shape[0]
    hidden_size = layer.w2_weight.shape[1]
    intermediate_size = layer.w13_weight.shape[1] // 2
    layer.num_experts = num_experts
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = intermediate_size
    if not hasattr(layer, "params_dtype"):
        layer.params_dtype = getattr(
            getattr(layer, "moe_runner_config", None),
            "params_dtype",
            torch.bfloat16,
        )

    weight_schema = HummingWeightSchema(
        b_dtype="int4",
        bs_dtype="bfloat16",
        weight_scale_group_size=group_size,
        weight_scale_type="group",
        has_zero_point=False,
    )
    input_schema = HummingInputSchema(
        a_dtype="float8e4m3",
        input_scale_group_size=group_size,
    )

    shapes = {
        "w13": (intermediate_size * 2, hidden_size),
        "w2": (hidden_size, intermediate_size),
    }
    for prefix, (shape_n, shape_k) in shapes.items():
        HummingMethod.prepare_layer_meta(
            layer=layer,
            shape_n=shape_n,
            shape_k=shape_k,
            weight_schema=weight_schema,
            input_schema=input_schema,
            num_experts=num_experts,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            torch_dtype=layer.params_dtype,
            sublayer_name=prefix,
        )
        HummingMethod.transform_humming_layer(layer, sublayer_name=prefix)

    layer._humming_w4a8_prepared = True


def _get_humming_configs(layer: torch.nn.Module) -> dict[str, Any]:
    cached = getattr(layer, "_humming_w4a8_configs", None)
    if cached is not None:
        return cached

    GemmType, HummingMethod, _, _ = _humming_imports()
    from sglang.srt.environ import envs

    use_f16_accum = envs.SGLANG_HUMMING_USE_F16_ACCUM.get()
    compute = {
        "use_f16_accum": use_f16_accum,
        "gemm_type": GemmType.INDEXED.value,
    }
    cached = {
        "compute": json.dumps(compute),
        "w13": HummingMethod.get_default_tuning_configs(
            layer=layer,
            use_f16_accum=use_f16_accum,
            gemm_type=GemmType.INDEXED,
            sublayer_name="w13",
        ),
        "w2": HummingMethod.get_default_tuning_configs(
            layer=layer,
            use_f16_accum=use_f16_accum,
            gemm_type=GemmType.INDEXED,
            sublayer_name="w2",
        ),
    }
    if getattr(layer.humming_metas["w13"], "use_packed_k_layout", False):
        ensure_humming_packed_k_warp_compatibility(cached["w13"])
    layer._humming_w4a8_configs = cached
    return cached


def _select_moe_block_size(tuning_configs: list[Any], valid_shape_m: int) -> int:
    for min_shape_m, max_shape_m, config in tuning_configs:
        if min_shape_m < valid_shape_m <= max_shape_m:
            return int(config["block_shape"][0])
    raise ValueError(f"No Humming tuning range covers shape M={valid_shape_m}")


def _prepare_buffers(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    from humming import dtypes

    m = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    real_m = m * top_k
    hidden_size = layer.hidden_size
    intermediate_size = layer.intermediate_size_per_partition
    a_dtype = layer.humming_metas["w13"].a_dtype
    c_dtype = layer.humming_metas["w13"].c_dtype
    dtype_map = {
        dtypes.float16: torch.float16,
        dtypes.bfloat16: torch.bfloat16,
        dtypes.float8e4m3: torch.float8_e4m3fn,
        dtypes.int8: torch.int8,
        dtypes.int4: torch.uint8,
    }
    metas = {
        "quanted_gate_up_input": ((m, hidden_size), dtype_map[a_dtype]),
        "gate_up_output": ((real_m, intermediate_size * 2), dtype_map[c_dtype]),
        "activation_output": ((real_m, intermediate_size), dtype_map[c_dtype]),
        "quanted_down_input": ((real_m, intermediate_size), dtype_map[a_dtype]),
        "down_output": ((real_m, hidden_size), dtype_map[c_dtype]),
        "output": ((m, hidden_size), dtype_map[c_dtype]),
    }
    required = (
        "quanted_gate_up_input",
        "gate_up_output",
        "activation_output",
        "quanted_down_input",
        "down_output",
        "output",
    )
    workspace_nbytes = [0, 0]
    for index, name in enumerate(reversed(required)):
        shape, dtype = metas[name]
        workspace_nbytes[index % 2] = max(
            workspace_nbytes[index % 2],
            math.prod(shape) * dtype.itemsize,
        )
    workspaces = [
        torch.empty(
            (
                math.ceil(
                    nbytes
                    / torch.empty((), dtype=layer.params_dtype).element_size()
                ),
            ),
            dtype=layer.params_dtype,
            device=hidden_states.device,
        )
        for nbytes in workspace_nbytes
    ]
    buffers: dict[str, torch.Tensor] = {}
    for index, name in enumerate(reversed(required)):
        shape, dtype = metas[name]
        workspace = workspaces[index % 2].view(dtype)
        buffers[name] = workspace[: math.prod(shape)].view(shape)
    return buffers


def humming_w4a8_moe(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    routed_scaling_factor: float = 1.0,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Run Standard/TP W4A8 MoE through Humming indexed GEMMs."""
    if not getattr(layer, "_humming_w4a8_prepared", False):
        raise RuntimeError("Humming W4A8 weights were not prepared")
    if hidden_states.shape[0] == 0:
        return torch.empty_like(hidden_states)

    _, HummingMethod, _, _ = _humming_imports()
    from humming.ops import moe_fused_mul_sum
    from sgl_kernel import silu_and_mul
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

    hidden_states = hidden_states.contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids, topk_weights = sanitize_humming_topk_padding_(
        topk_ids, topk_weights
    )
    valid_shape_m = topk_ids.numel()
    configs = _get_humming_configs(layer)
    w13_block_size = _select_moe_block_size(configs["w13"], valid_shape_m)
    w13_sorted_ids, w13_expert_ids, w13_num_tokens_padded = moe_align_block_size(
        topk_ids,
        w13_block_size,
        layer.num_experts,
    )
    w13_common = {
        "sorted_ids": w13_sorted_ids,
        "expert_ids": w13_expert_ids,
        "num_tokens_padded": w13_num_tokens_padded,
        "compute_config": configs["compute"],
        "valid_shape_m": valid_shape_m,
    }
    buffers = _prepare_buffers(layer, hidden_states, topk_ids)

    inputs, input_scale = HummingMethod.may_hadamard_quant_input(
        layer=layer,
        inputs=hidden_states,
        quanted_input=buffers["quanted_gate_up_input"],
        sublayer_name="w13",
    )
    HummingMethod.forward_layer(
        layer=layer,
        inputs=inputs,
        input_scale=input_scale,
        outputs=buffers["gate_up_output"],
        top_k=topk_ids.shape[1],
        tuning_config=configs["w13"],
        sublayer_name="w13",
        **w13_common,
    )
    if swiglu_limit is None:
        silu_and_mul(buffers["gate_up_output"], buffers["activation_output"])
    else:
        if swiglu_limit != 10:
            raise ValueError(
                "Humming W4A8 supports only DeepSeek V4 swiglu_limit=10, "
                f"got {swiglu_limit}."
            )
        from sglang.jit_kernel.dsv4 import silu_and_mul_clamp

        silu_and_mul_clamp(
            buffers["gate_up_output"],
            buffers["activation_output"],
            swiglu_limit,
        )

    inputs, input_scale = HummingMethod.may_hadamard_quant_input(
        layer=layer,
        inputs=buffers["activation_output"],
        quanted_input=buffers["quanted_down_input"],
        sublayer_name="w2",
    )
    w2_block_size = _select_moe_block_size(configs["w2"], valid_shape_m)
    w2_sorted_ids, w2_expert_ids, w2_num_tokens_padded = moe_align_block_size(
        topk_ids.view(-1, 1),
        w2_block_size,
        layer.num_experts,
    )
    HummingMethod.forward_layer(
        layer=layer,
        inputs=inputs,
        input_scale=input_scale,
        outputs=buffers["down_output"],
        top_k=1,
        tuning_config=configs["w2"],
        sublayer_name="w2",
        sorted_ids=w2_sorted_ids,
        expert_ids=w2_expert_ids,
        num_tokens_padded=w2_num_tokens_padded,
        compute_config=configs["compute"],
        valid_shape_m=valid_shape_m,
    )

    factors = topk_weights.to(buffers["output"].dtype)
    if routed_scaling_factor != 1.0:
        factors = factors * routed_scaling_factor
    moe_fused_mul_sum(
        buffers["down_output"].view(
            hidden_states.shape[0],
            topk_ids.shape[1],
            layer.hidden_size,
        ),
        factors,
        outputs=buffers["output"],
    )
    return buffers["output"]
