import torch

from sglang.srt.layers.moe.ep_moe.kernels import (
    silu_and_mul_masked_post_per_tensor_quant_dynamic_fwd,
)


def main() -> None:
    torch.manual_seed(20260711)
    experts, tokens, inner = 4, 32, 1024
    input_tensor = torch.randn(
        experts, tokens, inner * 2, dtype=torch.bfloat16, device="cuda"
    )
    masked_m = torch.tensor([32, 17, 1, 0], dtype=torch.int32, device="cuda")
    output = torch.empty(
        experts, tokens, inner, dtype=torch.float8_e4m3fn, device="cuda"
    )
    scale = torch.empty(1, dtype=torch.float32, device="cuda")

    silu_and_mul_masked_post_per_tensor_quant_dynamic_fwd(
        input_tensor, output, masked_m, scale
    )

    gate, up = input_tensor.float().chunk(2, dim=-1)
    reference = torch.nn.functional.silu(gate) * up
    valid = torch.arange(tokens, device="cuda")[None, :] < masked_m[:, None]
    expected_scale = reference[valid].abs().max() / torch.finfo(torch.float8_e4m3fn).max
    torch.testing.assert_close(scale, expected_scale.reshape_as(scale), rtol=1e-3, atol=1e-6)
    dequantized = output.float() * scale
    torch.testing.assert_close(
        dequantized[valid], reference[valid], rtol=0.15, atol=0.15
    )
    print("JD W4A8 dynamic quantization correctness passed")


if __name__ == "__main__":
    main()
