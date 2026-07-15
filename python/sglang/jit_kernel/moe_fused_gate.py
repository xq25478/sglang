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
_SUPPORTED_SCORING_FUNCS = (*_SCORING_FUNC_MAP.keys(), "softmax")


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


def _moe_fused_gate_torch_fallback(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
    num_expert_group: int,
    topk_group: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "sigmoid":
        activated = torch.sigmoid(input)
        ranked_scores = activated + bias
    elif scoring_func == "sqrtsoftplus":
        activated = torch.sqrt(torch.nn.functional.softplus(input))
        ranked_scores = activated + bias
    elif scoring_func == "softmax":
        ranked_scores = input + bias
        activated = torch.softmax(ranked_scores, dim=-1)
    else:
        raise AssertionError(
            f"Unknown scoring_func '{scoring_func}', must be one of "
            f"{list(_SUPPORTED_SCORING_FUNCS)}"
        )

    num_rows, num_experts = input.shape
    topk_routed = topk - num_fused_shared_experts
    if num_expert_group > 1:
        assert (
            num_experts % num_expert_group == 0
        ), "num_experts must be divisible by num_expert_group"
        assert 0 < topk_group <= num_expert_group, (
            "topk_group must be positive and <= num_expert_group"
        )
        experts_per_group = num_experts // num_expert_group
        grouped_scores = ranked_scores.reshape(
            num_rows, num_expert_group, experts_per_group
        )
        group_scores = torch.topk(
            grouped_scores,
            k=min(2, experts_per_group),
            dim=-1,
            sorted=False,
        ).values.sum(dim=-1)
        selected_groups = torch.topk(
            group_scores, k=topk_group, dim=-1, sorted=False
        ).indices
        kept_groups = torch.zeros_like(group_scores, dtype=torch.bool)
        kept_groups.scatter_(1, selected_groups, True)
        kept_experts = (
            kept_groups.unsqueeze(-1)
            .expand(-1, -1, experts_per_group)
            .reshape(num_rows, num_experts)
        )
        ranked_scores = ranked_scores.masked_fill(~kept_experts, -torch.inf)

    routed_ids = torch.topk(ranked_scores, k=topk_routed, dim=-1, sorted=True).indices
    routed_weights = activated.gather(1, routed_ids)
    routed_sum = routed_weights.sum(dim=-1, keepdim=True)

    if num_fused_shared_experts:
        shared_ids = torch.arange(
            num_experts,
            num_experts + num_fused_shared_experts,
            dtype=routed_ids.dtype,
            device=input.device,
        ).expand(num_rows, -1)
        shared_weights = routed_sum.expand(-1, num_fused_shared_experts) / float(
            routed_scaling_factor
        )
        topk_ids = torch.cat((routed_ids, shared_ids), dim=-1)
        topk_weights = torch.cat((routed_weights, shared_weights), dim=-1)
    else:
        topk_ids = routed_ids
        topk_weights = routed_weights

    if renormalize:
        topk_weights = topk_weights / torch.where(
            routed_sum > 0.0, routed_sum, torch.ones_like(routed_sum)
        )
    if apply_routed_scaling_factor_on_output:
        topk_weights = topk_weights * float(routed_scaling_factor)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _moe_fused_gate_jit(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = torch.empty((input.shape[0], topk), dtype=torch.float32, device=input.device)
    indices = torch.empty((input.shape[0], topk), dtype=torch.int32, device=input.device)
    if input.shape[0] == 0:
        return weights, indices

    _jit_moe_fused_gate_module().moe_fused_gate(
        input,
        bias,
        weights,
        indices,
        topk,
        _SCORING_FUNC_MAP[scoring_func],
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
    return weights, indices


def moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    num_expert_group: int = 1,
    topk_group: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scoring_func = scoring_func.lower()
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func)
    assert (
        scoring_func_int is not None or scoring_func == "softmax"
    ), f"Unknown scoring_func '{scoring_func}', must be one of {list(_SUPPORTED_SCORING_FUNCS)}"

    assert input.dtype == torch.float32, "input must be float32"
    assert bias.dtype == torch.float32, "bias must be float32"
    assert input.ndim == 2, "input must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert input.size(1) == bias.size(0), "input and bias must have same num_experts"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"
    assert num_expert_group >= 1, "num_expert_group must be >= 1"
    assert topk_group >= 1, "topk_group must be >= 1"
    assert (
        num_fused_shared_experts == 0 or routed_scaling_factor != 0.0
    ), "routed_scaling_factor must be non-zero when shared experts are fused"
    num_rows, _ = input.shape

    if scoring_func == "softmax" or num_expert_group != 1 or topk_group != 1:
        return _moe_fused_gate_torch_fallback(
            input,
            bias,
            topk,
            scoring_func,
            num_fused_shared_experts,
            renormalize,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
            num_expert_group,
            topk_group,
        )

    if (
        topk == 6
        and num_fused_shared_experts == 0
        and scoring_func == "sqrtsoftplus"
        and input.size(1) == 256
        and input.is_contiguous()
        and bias.is_contiguous()
    ):
        if _use_triton_moe_fused_gate(num_rows):
            return moe_fused_gate_triton_dsv4(
                input,
                bias,
                routed_scaling_factor,
                renormalize,
                apply_routed_scaling_factor_on_output,
            )
        return _moe_fused_gate_jit(
            input,
            bias,
            topk,
            scoring_func,
            num_fused_shared_experts,
            renormalize,
            routed_scaling_factor,
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
