from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.jit_kernel.triton.moe_fused_gate import (
    moe_fused_gate_triton,
    moe_fused_gate_triton_dsv4,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "sqrtsoftplus": 1,
}


@cache_once
def _jit_moe_fused_gate_module() -> Module:
    return load_jit(
        "moe_fused_gate",
        cuda_files=["moe/moe_fused_gate.cuh"],
        cuda_wrappers=[("moe_fused_gate", "MoEFusedGateKernel::run")],
    )


@cache_once
def can_use_moe_fused_gate() -> bool:
    logger = logging.getLogger(__name__)
    try:
        _jit_moe_fused_gate_module()
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT MoE fused gate kernel: {e}")
        return False


def _use_triton_moe_fused_gate(num_rows: int) -> bool:
    if os.getenv("SGLANG_USE_TRITON_MOE_FUSED_GATE", "0") != "1":
        return False

    max_m = int(os.getenv("SGLANG_TRITON_MOE_FUSED_GATE_MAX_M", "0"))
    return max_m <= 0 or num_rows <= max_m


def moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert (
        scoring_func_int is not None
    ), f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"

    assert input.dtype == torch.float32, "input must be float32"
    assert bias.dtype == torch.float32, "bias must be float32"
    assert input.ndim == 2, "input must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert input.size(1) == bias.size(0), "input and bias must have same num_experts"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"
    num_rows, _ = input.shape

    if (
        topk == 6
        and num_fused_shared_experts == 0
        and scoring_func == "sqrtsoftplus"
        and input.size(1) == 256
        and input.is_contiguous()
        and bias.is_contiguous()
    ):
        return moe_fused_gate_triton_dsv4(
            input,
            bias,
            routed_scaling_factor,
            renormalize,
            apply_routed_scaling_factor_on_output,
        )
    return moe_fused_gate_triton(
        input,
        bias,
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
