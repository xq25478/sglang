import torch

from bench_utils import assert_relative_performance, cuda_samples
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    combine_topk_swa_indices_for_query_seq_lens,
    combined_topk_width,
)


def _benchmark(compress_ratio: int) -> None:
    num_tokens = 4096
    topk = 128
    window_size = 128
    device = torch.device("cuda")
    topk_indices = torch.arange(
        num_tokens * topk, dtype=torch.int32, device=device
    ).reshape(num_tokens, topk)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    query_seq_lens = torch.arange(
        compress_ratio,
        (num_tokens + 1) * compress_ratio,
        compress_ratio,
        dtype=torch.int32,
        device=device,
    )
    swa_first_pos = torch.tensor([0], dtype=torch.int32, device=device)
    compressed_base = torch.tensor([1000], dtype=torch.int32, device=device)
    swa_base = torch.tensor([2000], dtype=torch.int32, device=device)
    width = combined_topk_width(topk, window_size)
    offsets = torch.arange(width, dtype=torch.int32, device=device)[None, :]

    def candidate():
        return combine_topk_swa_indices_for_query_seq_lens(
            topk_indices=topk_indices,
            query_start_loc=query_start_loc,
            query_seq_lens=query_seq_lens,
            swa_first_pos=swa_first_pos,
            compressed_base=compressed_base,
            swa_base=swa_base,
            window_size=window_size,
            compress_ratio=compress_ratio,
            topk=topk,
        )

    def reference():
        topk_lens = torch.minimum(
            query_seq_lens // compress_ratio,
            torch.tensor(topk, dtype=torch.int32, device=device),
        )
        swa_lens = torch.minimum(
            query_seq_lens,
            torch.tensor(window_size, dtype=torch.int32, device=device),
        )
        safe_topk_offsets = torch.minimum(
            offsets, torch.tensor(topk - 1, dtype=torch.int32, device=device)
        ).long()
        topk_values = (
            torch.gather(topk_indices, 1, safe_topk_offsets.expand(num_tokens, -1))
            + compressed_base[0]
        )
        swa_offsets = offsets - topk_lens[:, None]
        swa_values = (
            swa_base[0]
            + swa_offsets
            + query_seq_lens[:, None]
            - swa_lens[:, None]
            - swa_first_pos[0]
        )
        output = torch.full((num_tokens, width), -1, dtype=torch.int32, device=device)
        output = torch.where(offsets < topk_lens[:, None], topk_values, output)
        output = torch.where(
            (swa_offsets >= 0) & (swa_offsets < swa_lens[:, None]),
            swa_values,
            output,
        )
        return output, topk_lens + swa_lens

    assert_relative_performance(
        operator=f"dsv4_cp_sparse_prefill_c{compress_ratio}",
        optimized=cuda_samples(candidate),
        reference=cuda_samples(reference),
        max_ratio=1.10,
    )


def main() -> None:
    for compress_ratio in (4, 128):
        _benchmark(compress_ratio)


if __name__ == "__main__":
    main()
