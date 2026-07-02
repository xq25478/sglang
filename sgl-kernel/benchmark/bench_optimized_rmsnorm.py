import itertools

import torch
import triton
from sgl_kernel import rmsnorm, optimized_rms_norm


def rmsnorm_sgl_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])

    output = rmsnorm(x, weight, eps)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def optimized_rmsnorm_sgl_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    optimized_rms_norm(x, weight, eps=eps)
    return x


def calculate_diff(token_num, head_num, head_dim):
    dtype = torch.bfloat16
    x = torch.randn(token_num, head_num * head_dim, dtype=dtype, device="cuda")
    weight = torch.ones(head_dim, dtype=dtype, device="cuda")
    output_sgl_kernel = rmsnorm_sgl_kernel(x.reshape(-1, head_dim).clone(), weight)
    output_optimized_sgl_kernel = optimized_rmsnorm_sgl_kernel(
        x.reshape(-1, head_dim).clone(), weight
    )

    output_sgl_kernel = output_sgl_kernel.reshape(token_num, head_num, head_dim)
    output_optimized_sgl_kernel = output_optimized_sgl_kernel.reshape(
        token_num, head_num, head_dim
    )
    print(f"SGL Kernel output={output_sgl_kernel}")
    print(f"optimized SGL Kernel output={output_optimized_sgl_kernel}")
    if torch.allclose(
        output_sgl_kernel, output_optimized_sgl_kernel, atol=1e-2, rtol=1e-2
    ):
        print("✅ All implementations match")
    else:
        ref_diff = torch.abs(
            output_sgl_kernel - output_optimized_sgl_kernel
        ) / torch.abs(output_sgl_kernel)
        print(f"Ref diff={ref_diff}")
        print("❌ Implementations differ")


token_num_range = [2**i for i in range(0, 15, 1)]
head_num_range = [16, 32, 48]
configs = list(itertools.product(head_num_range, token_num_range))


def get_benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["head_num", "token_num"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["sglang", "optimized"],
            line_names=["SGLang", "My opt"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name=f"rmsnorm-performance(head_dim=128)",
            args={},
        )
    )
    def benchmark(head_num, token_num, provider):
        dtype = torch.bfloat16
        head_dim = 128
        x = torch.randn(token_num, head_num * head_dim, dtype=dtype, device="cuda")
        weight = torch.ones(head_dim, dtype=dtype, device="cuda")

        quantiles = [0.5, 0.2, 0.8]

        if provider == "sglang":
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: rmsnorm_sgl_kernel(
                    x.reshape(-1, head_dim).clone(),
                    weight,
                ),
                quantiles=quantiles,
            )
        else:  # optimized_rmsnorm
            ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
                lambda: optimized_rmsnorm_sgl_kernel(
                    x.reshape(-1, head_dim).clone(),
                    weight,
                ),
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results/",
        help="Path to save optimized_rmsnorm benchmark results",
    )
    args = parser.parse_args()

    # Run correctness test
    calculate_diff(token_num=1024, head_num=16, head_dim=128)

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark()
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)
