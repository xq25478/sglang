import sgl_kernel
import torch


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def main() -> None:
    torch.manual_seed(20260711)
    for token_num in (1, 19, 989):
        for head_num in (8, 48):
            for head_dim in (64, 128):
                x = torch.randn(
                    token_num * head_num,
                    head_dim,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                weight = torch.randn(
                    head_dim, dtype=torch.bfloat16, device="cuda"
                )
                expected = reference_rmsnorm(x, weight)
                actual = x.clone()
                sgl_kernel.optimized_rms_norm(actual, weight)
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
    print("JD optimized RMSNorm correctness passed")


if __name__ == "__main__":
    main()
