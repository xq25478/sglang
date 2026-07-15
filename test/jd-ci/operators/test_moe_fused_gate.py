from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import torch

from sglang.jit_kernel.moe_fused_gate import moe_fused_gate
from sglang.jit_kernel.triton.moe_fused_gate import (
    moe_fused_gate_triton,
    moe_fused_gate_triton_dsv4,
)


ROUTED_SCALING_FACTOR = 1.5


def _moe_fused_gate_torch_reference(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
    num_expert_group: int = 1,
    topk_group: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "sigmoid":
        activated = torch.sigmoid(scores)
        ranked_scores = activated + bias
    elif scoring_func == "sqrtsoftplus":
        activated = torch.sqrt(torch.nn.functional.softplus(scores))
        ranked_scores = activated + bias
    elif scoring_func == "softmax":
        ranked_scores = scores + bias
        activated = torch.softmax(ranked_scores, dim=-1)
    else:
        raise AssertionError(f"Unexpected scoring_func={scoring_func}")

    num_tokens, num_experts = scores.shape
    topk_routed = topk - num_fused_shared_experts
    if num_expert_group > 1:
        if num_experts % num_expert_group != 0:
            raise AssertionError("num_experts must be divisible by num_expert_group")
        experts_per_group = num_experts // num_expert_group
        grouped_scores = ranked_scores.reshape(
            num_tokens, num_expert_group, experts_per_group
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
            .reshape(num_tokens, num_experts)
        )
        ranked_scores = ranked_scores.masked_fill(~kept_experts, -torch.inf)

    routed_ids = torch.topk(
        ranked_scores, k=topk_routed, dim=-1, sorted=True
    ).indices
    routed_weights = activated.gather(1, routed_ids)
    routed_sum = routed_weights.sum(dim=-1, keepdim=True)

    if num_fused_shared_experts:
        shared_ids = torch.arange(
            num_experts,
            num_experts + num_fused_shared_experts,
            dtype=routed_ids.dtype,
            device=scores.device,
        ).expand(num_tokens, -1)
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


def _scatter_by_expert(
    weights: torch.Tensor, ids: torch.Tensor, num_experts: int
) -> torch.Tensor:
    dense = torch.zeros(
        (weights.shape[0], num_experts), dtype=torch.float32, device=weights.device
    )
    dense.scatter_(1, ids.long(), weights)
    return dense


def _assert_same_topk_set(
    label: str,
    weights: torch.Tensor,
    ids: torch.Tensor,
    reference_weights: torch.Tensor,
    reference_ids: torch.Tensor,
    num_experts_with_shared: int,
) -> None:
    expected_shape = reference_weights.shape
    if weights.shape != expected_shape or ids.shape != expected_shape:
        raise AssertionError(
            f"{label}: output shapes {(weights.shape, ids.shape)} != {expected_shape}"
        )
    if weights.dtype != torch.float32 or ids.dtype != torch.int32:
        raise AssertionError(
            f"{label}: output dtypes {(weights.dtype, ids.dtype)} != "
            "(torch.float32, torch.int32)"
        )
    if ids.numel():
        if int(ids.min()) < 0 or int(ids.max()) >= num_experts_with_shared:
            raise AssertionError(f"{label}: expert IDs are outside the valid range")
        sorted_ids = ids.sort(dim=-1).values
        if sorted_ids.shape[1] > 1 and not torch.all(
            sorted_ids[:, 1:] != sorted_ids[:, :-1]
        ):
            raise AssertionError(f"{label}: duplicate expert IDs in a top-k row")

    torch.testing.assert_close(
        _scatter_by_expert(weights, ids, num_experts_with_shared),
        _scatter_by_expert(
            reference_weights, reference_ids, num_experts_with_shared
        ),
        rtol=1e-4,
        atol=1e-5,
        msg=lambda message: f"{label}: {message}",
    )


def _make_inputs(
    num_tokens: int, num_experts: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    bias = torch.randn(
        (num_experts,),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    ) * 0.1
    return scores, bias


def _assert_matches_reference(
    label: str,
    implementation: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    scores: torch.Tensor,
    bias: torch.Tensor,
    *,
    call_kwargs: dict[str, object],
    reference_kwargs: dict[str, object],
) -> None:
    weights, ids = implementation(scores, bias, **call_kwargs)
    reference_weights, reference_ids = _moe_fused_gate_torch_reference(
        scores, bias, **reference_kwargs
    )
    torch.cuda.synchronize()
    _assert_same_topk_set(
        label,
        weights,
        ids,
        reference_weights,
        reference_ids,
        scores.shape[1]
        + int(reference_kwargs.get("num_fused_shared_experts", 0)),
    )


def _test_dsv4_specialization() -> None:
    for num_tokens in (0, 1, 17, 128, 513):
        for renormalize in (False, True):
            for apply_scale in (False, True):
                scores, bias = _make_inputs(
                    num_tokens,
                    256,
                    seed=(
                        20260714
                        + num_tokens * 17
                        + int(renormalize) * 3
                        + int(apply_scale)
                    ),
                )
                common_kwargs: dict[str, object] = {
                    "routed_scaling_factor": ROUTED_SCALING_FACTOR,
                    "renormalize": renormalize,
                    "apply_routed_scaling_factor_on_output": apply_scale,
                }
                reference_kwargs: dict[str, object] = {
                    "topk": 6,
                    "scoring_func": "sqrtsoftplus",
                    "num_fused_shared_experts": 0,
                    **common_kwargs,
                }
                case = (
                    f"dsv4 M={num_tokens} renormalize={renormalize} "
                    f"apply_scale={apply_scale}"
                )
                _assert_matches_reference(
                    f"{case} direct",
                    moe_fused_gate_triton_dsv4,
                    scores,
                    bias,
                    call_kwargs=common_kwargs,
                    reference_kwargs=reference_kwargs,
                )
                with (
                    patch.dict(
                        "os.environ", {"SGLANG_USE_TRITON_MOE_FUSED_GATE": "1"}
                    ),
                    patch(
                        "sglang.jit_kernel.moe_fused_gate."
                        "moe_fused_gate_triton_dsv4",
                        wraps=moe_fused_gate_triton_dsv4,
                    ) as specialized_spy,
                    patch(
                        "sglang.jit_kernel.moe_fused_gate.moe_fused_gate_triton",
                        wraps=moe_fused_gate_triton,
                    ) as generic_spy,
                ):
                    _assert_matches_reference(
                        f"{case} wrapper",
                        moe_fused_gate,
                        scores,
                        bias,
                        call_kwargs=reference_kwargs,
                        reference_kwargs=reference_kwargs,
                    )
                specialized_spy.assert_called_once()
                generic_spy.assert_not_called()


def _test_generic_branches() -> None:
    cases = (
        ("k1", 3, 64, 1, 0, "sigmoid", False, False),
        ("k2", 17, 96, 2, 0, "sqrtsoftplus", True, True),
        ("k_gt_2", 33, 384, 8, 0, "sqrtsoftplus", True, False),
        ("shared", 19, 128, 6, 2, "sigmoid", True, True),
    )
    for (
        name,
        num_tokens,
        num_experts,
        topk,
        num_fused_shared_experts,
        scoring_func,
        renormalize,
        apply_scale,
    ) in cases:
        scores, bias = _make_inputs(
            num_tokens,
            num_experts,
            seed=20260800 + num_tokens + num_experts + topk,
        )
        kwargs: dict[str, object] = {
            "topk": topk,
            "scoring_func": scoring_func,
            "num_fused_shared_experts": num_fused_shared_experts,
            "renormalize": renormalize,
            "routed_scaling_factor": ROUTED_SCALING_FACTOR,
            "apply_routed_scaling_factor_on_output": apply_scale,
        }
        for implementation_name, implementation in (
            ("triton", moe_fused_gate_triton),
            ("wrapper", moe_fused_gate),
        ):
            _assert_matches_reference(
                f"generic {name} {implementation_name}",
                implementation,
                scores,
                bias,
                call_kwargs=kwargs,
                reference_kwargs=kwargs,
            )

    base_scores, bias = _make_inputs(256, 256, seed=20260901)
    non_contiguous_scores = base_scores.t()
    if non_contiguous_scores.is_contiguous():
        raise AssertionError(
            "non-contiguous case did not create a strided score tensor"
        )
    kwargs = {
        "topk": 6,
        "scoring_func": "sqrtsoftplus",
        "num_fused_shared_experts": 0,
        "renormalize": True,
        "routed_scaling_factor": ROUTED_SCALING_FACTOR,
        "apply_routed_scaling_factor_on_output": False,
    }
    _assert_matches_reference(
        "generic non_contiguous triton",
        moe_fused_gate_triton,
        non_contiguous_scores,
        bias,
        call_kwargs=kwargs,
        reference_kwargs=kwargs,
    )
    with (
        patch(
            "sglang.jit_kernel.moe_fused_gate.moe_fused_gate_triton_dsv4",
            wraps=moe_fused_gate_triton_dsv4,
        ) as specialized_spy,
        patch(
            "sglang.jit_kernel.moe_fused_gate.moe_fused_gate_triton",
            wraps=moe_fused_gate_triton,
        ) as generic_spy,
    ):
        _assert_matches_reference(
            "generic non_contiguous wrapper_fallback",
            moe_fused_gate,
            non_contiguous_scores,
            bias,
            call_kwargs=kwargs,
            reference_kwargs=kwargs,
        )
    specialized_spy.assert_not_called()
    generic_spy.assert_called_once()


def _test_softmax_interface_compatibility() -> None:
    scores, bias = _make_inputs(11, 64, seed=20261001)
    kwargs: dict[str, object] = {
        "topk": 4,
        "scoring_func": "softmax",
        "num_fused_shared_experts": 0,
        "renormalize": True,
        "routed_scaling_factor": 1.0,
        "apply_routed_scaling_factor_on_output": False,
    }
    _assert_matches_reference(
        "softmax wrapper compatibility",
        moe_fused_gate,
        scores,
        bias,
        call_kwargs=kwargs,
        reference_kwargs=kwargs,
    )


def _test_grouped_interface_compatibility() -> None:
    scores, bias = _make_inputs(13, 128, seed=20261002)
    kwargs: dict[str, object] = {
        "topk": 4,
        "scoring_func": "sigmoid",
        "num_fused_shared_experts": 0,
        "renormalize": True,
        "routed_scaling_factor": ROUTED_SCALING_FACTOR,
        "apply_routed_scaling_factor_on_output": True,
        "num_expert_group": 8,
        "topk_group": 3,
    }
    _assert_matches_reference(
        "grouped wrapper compatibility",
        moe_fused_gate,
        scores,
        bias,
        call_kwargs=kwargs,
        reference_kwargs=kwargs,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for JD moe_fused_gate correctness")

    failures: list[str] = []
    for section_name, section in (
        ("dsv4_specialization", _test_dsv4_specialization),
        ("generic_branches", _test_generic_branches),
        ("softmax_interface", _test_softmax_interface_compatibility),
        ("grouped_interface", _test_grouped_interface_compatibility),
    ):
        try:
            section()
        except Exception as error:
            failures.append(f"{section_name}: {type(error).__name__}: {error}")

    if failures:
        raise AssertionError("JD moe_fused_gate failures:\n" + "\n".join(failures))
    print("JD moe_fused_gate correctness passed")


if __name__ == "__main__":
    main()
