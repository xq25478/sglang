import sgl_kernel
import torch

from bench_utils import assert_relative_performance, cuda_samples
from test_dsv4_norm_rope import reference_rmsnorm, reference_rope


def main() -> None:
    torch.manual_seed(20260711)
    batch, heads, head_dim, rope_dim = 128, 8, 192, 64
    q = torch.randn(batch, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    freqs = torch.randn(4096, rope_dim, dtype=torch.float32, device="cuda")
    positions = torch.randint(0, 4096, (batch,), dtype=torch.int32, device="cuda")

    def optimized():
        sgl_kernel.dsv4_fused_q_norm_rope(q, freqs, positions, 1e-6)

    def reference():
        reference_rope(reference_rmsnorm(q, 1e-6), freqs, positions, rope_dim)

    assert_relative_performance(
        operator="dsv4_norm_rope",
        optimized=cuda_samples(optimized),
        reference=cuda_samples(reference),
        max_ratio=0.80,
    )


if __name__ == "__main__":
    main()
