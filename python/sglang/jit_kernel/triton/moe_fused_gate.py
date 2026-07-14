from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.utils import is_arch_support_pdl

_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "sqrtsoftplus": 1,
}


@triton.jit
def _argmax_combine(v1, i1, v2, i2):
    gt = v1 > v2
    eq = v1 == v2
    v = tl.where(gt, v1, v2)
    i = tl.where(gt | (eq & (i1 < i2)), i1, i2)
    return v, i


@triton.jit
def _moe_fused_gate_triton_k1_kernel(
    scores_ptr,
    bias_ptr,
    out_weights_ptr,
    out_indices_ptr,
    routed_scaling_factor,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCORING_FUNC: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    USE_GDC: tl.constexpr,
    INPUT_CONTIGUOUS: tl.constexpr,
    stride_sm,
    stride_sn,
) -> None:
    pid = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    if INPUT_CONTIGUOUS:
        scores_ptrs = scores_ptr + pid * N + offs_n
    else:
        scores_ptrs = scores_ptr + pid * stride_sm + offs_n * stride_sn
    scores = tl.load(scores_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    if SCORING_FUNC == 0:
        activated = tl.sigmoid(scores)
    else:
        softplus = tl.where(
            scores > 20.0,
            scores,
            tl.log(1.0 + tl.exp(scores)),
        )
        activated = tl.sqrt(softplus)

    # Non-finite ranks are zero-weight fallbacks with real expert IDs.
    ranked_scores = activated + bias
    finite_rank = (ranked_scores > -float("inf")) & (
        ranked_scores < float("inf")
    )
    ranked_scores = tl.where(mask_n & finite_rank, ranked_scores, -float("inf"))
    lane_id = tl.where(mask_n, offs_n, N + 1).to(tl.int32)
    max_val, win_lane = tl.reduce((ranked_scores, lane_id), 0, _argmax_combine)
    win_bias = tl.load(bias_ptr + win_lane).to(tl.float32)
    routed_weight = tl.where(
        max_val > -float("inf"), max_val - win_bias, 0.0
    )

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

    routed_out = routed_weight
    shared_out = routed_weight / routed_scaling_factor
    if RENORMALIZE:
        norm = tl.where(routed_weight > 0.0, routed_weight, 1.0)
        routed_out = routed_out / norm
        shared_out = shared_out / norm
    if APPLY_SCALE:
        routed_out = routed_out * routed_scaling_factor
        shared_out = shared_out * routed_scaling_factor

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    out_vals = tl.where(offs_k == 0, routed_out, shared_out)
    out_idxs = tl.where(offs_k == 0, win_lane, N + (offs_k - 1)).to(tl.int32)

    tl.store(out_weights_ptr + pid * K + offs_k, out_vals, mask=mask_k)
    tl.store(out_indices_ptr + pid * K + offs_k, out_idxs, mask=mask_k)


@triton.jit
def _moe_fused_gate_triton_k2_kernel(
    scores_ptr,
    bias_ptr,
    out_weights_ptr,
    out_indices_ptr,
    routed_scaling_factor,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCORING_FUNC: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    USE_GDC: tl.constexpr,
    INPUT_CONTIGUOUS: tl.constexpr,
    stride_sm,
    stride_sn,
) -> None:
    pid = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    if INPUT_CONTIGUOUS:
        scores_ptrs = scores_ptr + pid * N + offs_n
    else:
        scores_ptrs = scores_ptr + pid * stride_sm + offs_n * stride_sn
    scores = tl.load(scores_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    if SCORING_FUNC == 0:
        activated = tl.sigmoid(scores)
    else:
        softplus = tl.where(
            scores > 20.0,
            scores,
            tl.log(1.0 + tl.exp(scores)),
        )
        activated = tl.sqrt(softplus)

    # Non-finite ranks are zero-weight fallbacks with real expert IDs.
    m0 = activated + bias
    finite_rank = (m0 > -float("inf")) & (m0 < float("inf"))
    m0 = tl.where(mask_n & finite_rank, m0, -float("inf"))
    m1 = tl.full([BLOCK_N], -float("inf"), tl.float32)
    j0 = tl.where(mask_n, offs_n, N + 1).to(tl.int32)
    j1 = tl.full([BLOCK_N], N + 1, tl.int32)

    max0, idx0 = tl.reduce((m0, j0), 0, _argmax_combine)
    is_winner = j0 == idx0
    m0 = tl.where(is_winner, m1, m0)
    j0 = tl.where(is_winner, j1, j0)
    max1, idx1 = tl.reduce((m0, j0), 0, _argmax_combine)

    bias0 = tl.load(bias_ptr + idx0).to(tl.float32)
    bias1 = tl.load(bias_ptr + idx1).to(tl.float32)
    weight0 = tl.where(max0 > -float("inf"), max0 - bias0, 0.0)
    weight1 = tl.where(max1 > -float("inf"), max1 - bias1, 0.0)

    routed_sum = weight0 + weight1
    shared_weight = routed_sum / routed_scaling_factor

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)
        weight0 = weight0 / norm
        weight1 = weight1 / norm
        shared_weight = shared_weight / norm
    if APPLY_SCALE:
        weight0 = weight0 * routed_scaling_factor
        weight1 = weight1 * routed_scaling_factor
        shared_weight = shared_weight * routed_scaling_factor

    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    out_vals = tl.where(offs_k == 0, weight0, weight1)
    out_vals = tl.where(offs_k >= 2, shared_weight, out_vals)
    out_idxs = tl.where(offs_k == 0, idx0, idx1)
    out_idxs = tl.where(offs_k >= 2, N + (offs_k - 2), out_idxs).to(tl.int32)

    tl.store(out_weights_ptr + pid * K + offs_k, out_vals, mask=mask_k)
    tl.store(out_indices_ptr + pid * K + offs_k, out_idxs, mask=mask_k)


@triton.jit
def _moe_fused_gate_triton_kernel(
    scores_ptr,
    bias_ptr,
    out_weights_ptr,
    out_indices_ptr,
    routed_scaling_factor,
    N: tl.constexpr,
    K: tl.constexpr,
    K_ROUTED: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCORING_FUNC: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    USE_GDC: tl.constexpr,
    INPUT_CONTIGUOUS: tl.constexpr,
    stride_sm,
    stride_sn,
) -> None:
    pid = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    if USE_GDC:
        tl.extra.cuda.gdc_wait()

    if INPUT_CONTIGUOUS:
        scores_ptrs = scores_ptr + pid * N + offs_n
    else:
        scores_ptrs = scores_ptr + pid * stride_sm + offs_n * stride_sn
    scores = tl.load(scores_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    if SCORING_FUNC == 0:
        activated = tl.sigmoid(scores)
    else:
        softplus = tl.where(
            scores > 20.0,
            scores,
            tl.log(1.0 + tl.exp(scores)),
        )
        activated = tl.sqrt(softplus)

    # Non-finite ranks are zero-weight fallbacks with real expert IDs.
    ranked_scores = activated + bias
    finite_rank = (ranked_scores > -float("inf")) & (
        ranked_scores < float("inf")
    )
    ranked_scores = tl.where(mask_n & finite_rank, ranked_scores, -float("inf"))
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    mask_k_routed = offs_k < K_ROUTED
    selected_vals = tl.zeros([BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_K], dtype=tl.int32)

    available = mask_n
    for k in tl.static_range(K_ROUTED):
        cur = tl.where(available, ranked_scores, -float("inf"))
        max_val = tl.max(cur, axis=0)
        is_max = available & (cur == max_val)
        lane_id = tl.where(is_max, offs_n, N + 1)
        win_lane = tl.min(lane_id, axis=0).to(tl.int32)
        win_bias = tl.load(bias_ptr + win_lane).to(tl.float32)
        win_activated = tl.where(
            max_val > -float("inf"), max_val - win_bias, 0.0
        )
        slot = offs_k == k
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        available = available & (offs_n != win_lane)

    routed_sum = tl.sum(tl.where(mask_k_routed, selected_vals, 0.0), axis=0)

    if K_ROUTED < K:
        is_shared = (offs_k >= K_ROUTED) & mask_k
        selected_vals = tl.where(
            is_shared, routed_sum / routed_scaling_factor, selected_vals
        )
        selected_idx = tl.where(is_shared, N + (offs_k - K_ROUTED), selected_idx)

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)
        selected_vals = selected_vals / norm
    if APPLY_SCALE:
        selected_vals = selected_vals * routed_scaling_factor

    tl.store(
        out_weights_ptr + pid * K + offs_k,
        selected_vals,
        mask=mask_k,
    )
    tl.store(
        out_indices_ptr + pid * K + offs_k,
        selected_idx,
        mask=mask_k,
    )


def moe_fused_gate_triton_dsv4(
    scores: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    renormalize: bool,
    apply_routed_scaling_factor_on_output: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 ungrouped gate: N=256, topk=6, sqrtsoftplus."""

    M = scores.shape[0]
    weights = torch.empty((M, 6), dtype=torch.float32, device=scores.device)
    indices = torch.empty((M, 6), dtype=torch.int32, device=scores.device)
    if M == 0:
        return weights, indices

    use_gdc = is_arch_support_pdl()
    if use_gdc:
        _moe_fused_gate_triton_kernel[(M,)](
            scores,
            bias,
            weights,
            indices,
            routed_scaling_factor=float(routed_scaling_factor),
            N=256,
            K=6,
            K_ROUTED=6,
            BLOCK_N=256,
            BLOCK_K=8,
            SCORING_FUNC=1,
            RENORMALIZE=bool(renormalize),
            APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
            USE_GDC=True,
            INPUT_CONTIGUOUS=True,
            stride_sm=scores.stride(0),
            stride_sn=scores.stride(1),
            num_warps=1,
            launch_pdl=True,
        )
    else:
        _moe_fused_gate_triton_kernel[(M,)](
            scores,
            bias,
            weights,
            indices,
            routed_scaling_factor=float(routed_scaling_factor),
            N=256,
            K=6,
            K_ROUTED=6,
            BLOCK_N=256,
            BLOCK_K=8,
            SCORING_FUNC=1,
            RENORMALIZE=bool(renormalize),
            APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
            USE_GDC=False,
            INPUT_CONTIGUOUS=True,
            stride_sm=scores.stride(0),
            stride_sn=scores.stride(1),
            num_warps=1,
        )
    return weights, indices


def moe_fused_gate_triton(
    scores: torch.Tensor,
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
    assert scores.dtype == torch.float32, "scores must be float32"
    assert bias.dtype == torch.float32, "bias must be float32"
    assert scores.ndim == 2, "scores must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert scores.size(1) == bias.size(0), "scores and bias must have same num_experts"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"
    assert (
        num_fused_shared_experts == 0 or routed_scaling_factor != 0.0
    ), "routed_scaling_factor must be non-zero when shared experts are fused"

    M, N = scores.shape
    K = topk
    K_routed = topk - num_fused_shared_experts
    assert K_routed <= N, "routed topk must be <= num_experts"

    weights = torch.empty((M, K), dtype=torch.float32, device=scores.device)
    indices = torch.empty((M, K), dtype=torch.int32, device=scores.device)
    if M == 0:
        return weights, indices

    bias = bias.contiguous()
    block_n = triton.next_power_of_2(N)
    block_k = triton.next_power_of_2(K)
    use_gdc = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_gdc else {}

    if K_routed == 1:
        _moe_fused_gate_triton_k1_kernel[(M,)](
            scores,
            bias,
            weights,
            indices,
            routed_scaling_factor=float(routed_scaling_factor),
            N=N,
            K=K,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SCORING_FUNC=scoring_func_int,
            RENORMALIZE=bool(renormalize),
            APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
            USE_GDC=use_gdc,
            INPUT_CONTIGUOUS=scores.is_contiguous(),
            stride_sm=scores.stride(0),
            stride_sn=scores.stride(1),
            num_warps=1,
            **pdl_kwargs,
        )
    elif K_routed == 2:
        _moe_fused_gate_triton_k2_kernel[(M,)](
            scores,
            bias,
            weights,
            indices,
            routed_scaling_factor=float(routed_scaling_factor),
            N=N,
            K=K,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SCORING_FUNC=scoring_func_int,
            RENORMALIZE=bool(renormalize),
            APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
            USE_GDC=use_gdc,
            INPUT_CONTIGUOUS=scores.is_contiguous(),
            stride_sm=scores.stride(0),
            stride_sn=scores.stride(1),
            num_warps=1,
            **pdl_kwargs,
        )
    else:
        _moe_fused_gate_triton_kernel[(M,)](
            scores,
            bias,
            weights,
            indices,
            routed_scaling_factor=float(routed_scaling_factor),
            N=N,
            K=K,
            K_ROUTED=K_routed,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SCORING_FUNC=scoring_func_int,
            RENORMALIZE=bool(renormalize),
            APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
            USE_GDC=use_gdc,
            INPUT_CONTIGUOUS=scores.is_contiguous(),
            stride_sm=scores.stride(0),
            stride_sn=scores.stride(1),
            num_warps=1,
            **pdl_kwargs,
        )
    return weights, indices
