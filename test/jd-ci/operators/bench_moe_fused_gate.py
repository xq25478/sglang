import torch

from bench_utils import assert_relative_performance, cuda_samples
from sglang.jit_kernel.moe_fused_gate import (
    _jit_moe_fused_gate_module,
    moe_fused_gate,
)


def main() -> None:
    torch.manual_seed(20260714)
    scores = torch.randn(16384, 256, dtype=torch.float32, device="cuda")
    bias = torch.randn(256, dtype=torch.float32, device="cuda")
    legacy_module = _jit_moe_fused_gate_module()

    def candidate():
        return moe_fused_gate(
            scores,
            bias,
            topk=6,
            scoring_func="sqrtsoftplus",
            num_fused_shared_experts=0,
            routed_scaling_factor=1.5,
            renormalize=True,
            apply_routed_scaling_factor_on_output=False,
        )

    def reference():
        weights = torch.empty((16384, 6), dtype=torch.float32, device="cuda")
        indices = torch.empty((16384, 6), dtype=torch.int32, device="cuda")
        legacy_module.moe_fused_gate(
            scores,
            bias,
            weights,
            indices,
            6,
            1,
            0,
            True,
            1.5,
            False,
        )
        return weights, indices

    # Compile both independent device implementations before warmup and timing.
    candidate()
    reference()
    torch.cuda.synchronize()

    assert_relative_performance(
        operator="moe_fused_gate",
        optimized=cuda_samples(candidate),
        reference=cuda_samples(reference),
        max_ratio=1.10,
    )


if __name__ == "__main__":
    main()
