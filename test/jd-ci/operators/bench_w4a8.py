import torch

from bench_utils import assert_relative_performance, cuda_samples
from sglang.srt.layers.moe.ep_moe.kernels import (
    silu_and_mul_masked_post_per_tensor_quant_dynamic_fwd,
)


def main() -> None:
    torch.manual_seed(20260711)
    experts, tokens, inner = 8, 128, 2048
    input_tensor = torch.randn(
        experts, tokens, inner * 2, dtype=torch.bfloat16, device="cuda"
    )
    masked_m = torch.full((experts,), tokens, dtype=torch.int32, device="cuda")
    output = torch.empty(
        experts, tokens, inner, dtype=torch.float8_e4m3fn, device="cuda"
    )
    scale = torch.empty(1, dtype=torch.float32, device="cuda")

    def optimized():
        silu_and_mul_masked_post_per_tensor_quant_dynamic_fwd(
            input_tensor, output, masked_m, scale
        )

    def reference():
        gate, up = input_tensor.float().chunk(2, dim=-1)
        value = torch.nn.functional.silu(gate) * up
        ref_scale = value.abs().max() / torch.finfo(torch.float8_e4m3fn).max
        value.div(ref_scale).to(torch.float8_e4m3fn)

    assert_relative_performance(
        operator="w4a8_dynamic_quant",
        optimized=cuda_samples(optimized),
        reference=cuda_samples(reference),
        max_ratio=0.90,
    )


if __name__ == "__main__":
    main()
