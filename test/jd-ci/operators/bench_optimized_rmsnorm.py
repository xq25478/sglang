import sgl_kernel
import torch

from bench_utils import assert_relative_performance, cuda_samples


def main() -> None:
    torch.manual_seed(20260711)
    x = torch.randn(4096, 128, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    optimized_input = torch.empty_like(x)

    def optimized():
        optimized_input.copy_(x)
        sgl_kernel.optimized_rms_norm(optimized_input, weight)

    def reference():
        sgl_kernel.rmsnorm(x, weight)

    assert_relative_performance(
        operator="optimized_rmsnorm",
        optimized=cuda_samples(optimized),
        reference=cuda_samples(reference),
        max_ratio=1.10,
    )


if __name__ == "__main__":
    main()
