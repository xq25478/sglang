import sgl_kernel
import torch


def reference_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(
        x.dtype
    )


def reference_rope(
    x: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    out = x.clone()
    nope_dim = x.shape[-1] - rope_dim
    for index, position in enumerate(positions.tolist()):
        pairs = out[index, ..., nope_dim:].float().reshape(
            *out[index, ..., nope_dim:].shape[:-1], rope_dim // 2, 2
        )
        freq = freqs[position].reshape(rope_dim // 2, 2)
        real = pairs[..., 0] * freq[:, 0] - pairs[..., 1] * freq[:, 1]
        imag = pairs[..., 0] * freq[:, 1] + pairs[..., 1] * freq[:, 0]
        out[index, ..., nope_dim:] = torch.stack((real, imag), -1).reshape(
            out[index, ..., nope_dim:].shape
        )
    return out


def main() -> None:
    torch.manual_seed(20260711)
    eps = 1e-6
    for batch, heads, head_dim in ((1, 1, 128), (4, 8, 192), (16, 8, 192)):
        rope_dim = 64
        q = torch.randn(
            batch, heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        freqs = torch.randn(512, rope_dim, dtype=torch.float32, device="cuda")
        positions = torch.randint(0, 512, (batch,), dtype=torch.int32, device="cuda")
        actual = sgl_kernel.dsv4_fused_q_norm_rope(q, freqs, positions, eps)
        expected = reference_rope(reference_rmsnorm(q, eps), freqs, positions, rope_dim)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-2)
    print("JD DSV4 norm-rope correctness passed")


if __name__ == "__main__":
    main()
