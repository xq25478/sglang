import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.moe_fused_gate import moe_fused_gate as moe_fused_gate_jit
from sglang.jit_kernel.triton.moe_fused_gate import moe_fused_gate_triton
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=180, suite="nightly-kernel-1-gpu", nightly=True)


CORRECTNESS_CASES = get_ci_test_range(
    full_range=[
        case
        for case in itertools.product(
            [1, 128],
            [256, 512],
            [1, 2, 4, 8],
            [0, 1],
            ["sigmoid", "sqrtsoftplus"],
            [True, False],
            [False],
        )
        if case[2] > case[3]
    ]
    + [
        (17, 384, 8, 0, "sqrtsoftplus", True, False),
        (512, 384, 8, 1, "sigmoid", False, False),
        (512, 512, 16, 2, "sqrtsoftplus", True, True),
        (513, 512, 8, 0, "sqrtsoftplus", True, False),
    ],
    ci_range=[
        (1, 256, 1, 0, "sigmoid", True, False),
        (17, 256, 2, 1, "sigmoid", True, False),
        (128, 384, 8, 0, "sqrtsoftplus", True, False),
        (512, 512, 16, 2, "sqrtsoftplus", True, True),
        (2048, 384, 8, 1, "sigmoid", False, False),
        (513, 512, 8, 0, "sqrtsoftplus", True, False),
    ],
)

NON_CONTIGUOUS_CASES = get_ci_test_range(
    full_range=list(
        itertools.product(
            [3, 257],
            [256, 512],
            [4, 8],
            [0, 1],
            ["sigmoid", "sqrtsoftplus"],
            [True],
        )
    ),
    ci_range=[
        (3, 256, 4, 0, "sigmoid", False),
        (3, 384, 8, 1, "sqrtsoftplus", True),
        (257, 512, 8, 0, "sqrtsoftplus", False),
    ],
)


def _moe_fused_gate_torch_ref(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
):
    if scoring_func == "sigmoid":
        activated = torch.sigmoid(scores)
    elif scoring_func == "sqrtsoftplus":
        activated = torch.sqrt(torch.nn.functional.softplus(scores))
    else:
        raise AssertionError(f"Unexpected scoring_func={scoring_func}")

    topk_routed = topk - num_fused_shared_experts
    _, routed_ids = torch.topk(activated + bias, k=topk_routed, dim=-1, sorted=True)
    routed_weights = activated.gather(1, routed_ids)
    routed_sum = routed_weights.sum(dim=-1, keepdim=True)

    if num_fused_shared_experts:
        num_tokens, num_experts = scores.shape
        shared_ids = torch.arange(
            num_experts,
            num_experts + num_fused_shared_experts,
            dtype=routed_ids.dtype,
            device=scores.device,
        ).expand(num_tokens, -1)
        shared_weights = routed_sum.expand(-1, num_fused_shared_experts) / float(
            routed_scaling_factor
        )
        topk_ids = torch.cat([routed_ids, shared_ids], dim=-1)
        topk_weights = torch.cat([routed_weights, shared_weights], dim=-1)
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


def _scatter_by_expert(
    weights: torch.Tensor, ids: torch.Tensor, num_experts: int
) -> torch.Tensor:
    dense = torch.zeros(
        (weights.shape[0], num_experts), dtype=torch.float32, device=weights.device
    )
    dense.scatter_(1, ids.long(), weights)
    return dense


def _assert_same_topk_set(
    weights: torch.Tensor,
    ids: torch.Tensor,
    ref_weights: torch.Tensor,
    ref_ids: torch.Tensor,
    num_experts_with_shared: int,
) -> None:
    torch.testing.assert_close(
        _scatter_by_expert(weights, ids, num_experts_with_shared),
        _scatter_by_expert(ref_weights, ref_ids, num_experts_with_shared),
        rtol=1e-4,
        atol=1e-5,
    )


def _make_inputs(num_tokens: int, num_experts: int, seed: int):
    torch.manual_seed(seed)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda") * 0.1
    return scores, bias


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk,num_fused_shared_experts,scoring_func,renormalize,apply_scale",
    CORRECTNESS_CASES,
)
def test_moe_fused_gate_triton_matches_reference_and_jit_cuda(
    num_tokens: int,
    num_experts: int,
    topk: int,
    num_fused_shared_experts: int,
    scoring_func: str,
    renormalize: bool,
    apply_scale: bool,
) -> None:
    scores, bias = _make_inputs(
        num_tokens,
        num_experts,
        seed=1000
        + num_tokens
        + num_experts * 17
        + topk * 31
        + num_fused_shared_experts,
    )
    routed_scaling_factor = 2.5
    kwargs = dict(
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_scale,
    )

    triton_weights, triton_ids = moe_fused_gate_triton(scores, bias, **kwargs)
    ref_weights, ref_ids = _moe_fused_gate_torch_ref(scores, bias, **kwargs)
    jit_weights, jit_ids = moe_fused_gate_jit(scores, bias, **kwargs)
    torch.cuda.synchronize()

    num_experts_with_shared = num_experts + num_fused_shared_experts
    _assert_same_topk_set(
        triton_weights, triton_ids, ref_weights, ref_ids, num_experts_with_shared
    )
    _assert_same_topk_set(
        triton_weights, triton_ids, jit_weights, jit_ids, num_experts_with_shared
    )


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk,num_fused_shared_experts,scoring_func,renormalize",
    NON_CONTIGUOUS_CASES,
)
def test_moe_fused_gate_triton_non_contiguous_scores(
    num_tokens: int,
    num_experts: int,
    topk: int,
    num_fused_shared_experts: int,
    scoring_func: str,
    renormalize: bool,
) -> None:
    torch.manual_seed(2000 + num_tokens * 17 + num_experts + topk)
    base_scores = torch.randn(
        (num_experts, num_tokens), dtype=torch.float32, device="cuda"
    )
    bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda") * 0.1
    scores = base_scores.t()
    assert not scores.is_contiguous()

    kwargs = dict(
        topk=topk,
        scoring_func=scoring_func,
        num_fused_shared_experts=num_fused_shared_experts,
        renormalize=renormalize,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=False,
    )
    triton_weights, triton_ids = moe_fused_gate_triton(scores, bias, **kwargs)
    ref_weights, ref_ids = _moe_fused_gate_torch_ref(scores, bias, **kwargs)
    torch.cuda.synchronize()

    _assert_same_topk_set(
        triton_weights,
        triton_ids,
        ref_weights,
        ref_ids,
        num_experts + num_fused_shared_experts,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
