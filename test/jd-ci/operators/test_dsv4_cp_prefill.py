from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    combine_topk_swa_indices_for_query_seq_lens,
    combined_topk_width,
)
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import MQALayer


def _reference_combine(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    query_seq_lens: torch.Tensor,
    swa_first_pos: torch.Tensor,
    compressed_base: torch.Tensor,
    swa_base: torch.Tensor,
    *,
    window_size: int,
    compress_ratio: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.full(
        (
            query_seq_lens.numel(),
            combined_topk_width(topk, window_size),
        ),
        -1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    lengths = torch.zeros_like(query_seq_lens)
    for req_idx in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[req_idx])
        end = int(query_start_loc[req_idx + 1])
        for token_idx in range(start, end):
            causal_len = int(query_seq_lens[token_idx])
            topk_len = min(causal_len // compress_ratio, topk)
            swa_len = min(causal_len, window_size)
            window_start = causal_len - swa_len
            output[token_idx, :topk_len] = (
                topk_indices[token_idx, :topk_len] + compressed_base[req_idx]
            )
            output[token_idx, topk_len : topk_len + swa_len] = (
                swa_base[req_idx]
                + torch.arange(swa_len, device=output.device, dtype=torch.int32)
                + window_start
                - swa_first_pos[req_idx]
            )
            lengths[token_idx] = topk_len + swa_len
    return output, lengths


def _test_sparse_cp_combine() -> None:
    device = torch.device("cuda")
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    swa_first_pos = torch.tensor([0, 2], dtype=torch.int32, device=device)
    compressed_base = torch.tensor([10, 20], dtype=torch.int32, device=device)
    swa_base = torch.tensor([100, 200], dtype=torch.int32, device=device)
    topk_indices = torch.arange(5 * 8, dtype=torch.int32, device=device).reshape(5, 8)

    cases = (
        (4, 4, 4, [4, 8, 12, 6, 10]),
        (128, 8, 16, [128, 256, 512, 192, 384]),
    )
    for compress_ratio, topk, window_size, causal_lens in cases:
        query_seq_lens = torch.tensor(causal_lens, dtype=torch.int32, device=device)
        actual = combine_topk_swa_indices_for_query_seq_lens(
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
        expected = _reference_combine(
            topk_indices,
            query_start_loc,
            query_seq_lens,
            swa_first_pos,
            compressed_base,
            swa_base,
            window_size=window_size,
            compress_ratio=compress_ratio,
            topk=topk,
        )
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


def _test_cp_multi_stream_prepare() -> None:
    layer = MQALayer.__new__(MQALayer)
    torch.nn.Module.__init__(layer)
    layer.alt_streams = [torch.cuda.Stream() for _ in range(3)]
    layer.fuse_wqa_wkv = False
    layer.cp_size = 2
    layer.layer_id = 3
    layer.indexer = MagicMock()
    layer.compressor = object()
    layer._compute_q_a = MagicMock(side_effect=lambda value, qkv_a=None: value + 1)
    layer._compute_kv_bf16 = MagicMock(
        side_effect=lambda value, positions, qkv_a=None: value + 2
    )
    layer._compute_q_b = MagicMock(
        side_effect=lambda q_lora, positions, q_out=None: q_lora + 3
    )

    backend = MagicMock()
    x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    positions = torch.arange(4, dtype=torch.int64, device="cuda")
    forward_batch = object()

    with patch.object(
        deepseek_v4,
        "cp_all_gather_rerange_output",
        side_effect=lambda kv, cp_size, batch, stream: kv,
    ) as all_gather:
        q, kv = layer._forward_prepare_multi_stream_cp(
            x, positions, forward_batch, backend
        )
    torch.cuda.synchronize()

    torch.testing.assert_close(q, x + 4)
    torch.testing.assert_close(kv, x + 2)
    all_gather.assert_called_once()
    self_indexer_call = layer.indexer.call_args.kwargs
    if not self_indexer_call["enable_multi_stream"]:
        raise AssertionError("CP indexer did not enter its multi-stream path")
    if not isinstance(self_indexer_call["q_lora_ready"], torch.cuda.Event):
        raise AssertionError("CP indexer did not receive the Q readiness event")
    backend.store_cache.assert_called_once()
    backend.forward_core_compressor.assert_called_once()


def _test_cp_fused_wqkv_allocator_lifetime() -> None:
    """Stress the production cross-stream ownership pattern from 3P1D prefill."""
    layer = MQALayer.__new__(MQALayer)
    torch.nn.Module.__init__(layer)
    layer.alt_streams = [torch.cuda.Stream() for _ in range(3)]
    layer.fuse_wqa_wkv = True
    layer.cp_size = 2
    layer.layer_id = 3
    layer.indexer = None
    layer.compressor = None

    width = 64
    delay_input = torch.randn(
        (4096, 4096), dtype=torch.float16, device="cuda"
    )
    backend = MagicMock()
    forward_batch = object()

    for num_tokens in (768, 32608, 65280):
        for iteration in range(3):
            torch.manual_seed(num_tokens + iteration)
            x = torch.randn(
                (num_tokens, width), dtype=torch.float16, device="cuda"
            )
            positions = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
            allocator_churn = []

            def wqkv_a(value):
                return torch.cat((value + 11, value + 29), dim=-1), None

            def compute_q_a(value, qkv_a=None):
                if qkv_a is None:
                    raise AssertionError("fused WQKV path was not exercised")
                # Production q_norm materializes a new tensor, so q_lora does
                # not keep the fused WQKV allocation alive by aliasing it.
                return qkv_a[..., :width].clone()

            def compute_kv_bf16(value, positions, qkv_a=None):
                if qkv_a is None:
                    raise AssertionError("fused WQKV path was not exercised")
                # Keep stream_kv busy before it reads qkv_a.  The old code
                # released qkv_a immediately and the main stream reused the
                # same-size block below while this work was still in flight.
                torch.mm(delay_input, delay_input)
                return qkv_a[..., width:].contiguous()

            def compute_q_b(q_lora, positions, q_out=None):
                for _ in range(4):
                    allocator_churn.append(
                        torch.full(
                            (num_tokens, width * 2),
                            -123,
                            dtype=x.dtype,
                            device=x.device,
                        )
                    )
                return q_lora + 3

            layer.wqkv_a = wqkv_a
            layer._compute_q_a = compute_q_a
            layer._compute_kv_bf16 = compute_kv_bf16
            layer._compute_q_b = compute_q_b

            with patch.object(
                deepseek_v4,
                "cp_all_gather_rerange_output",
                side_effect=lambda kv, cp_size, batch, stream: kv,
            ):
                q, kv = layer._forward_prepare_multi_stream_cp(
                    x, positions, forward_batch, backend
                )
            torch.cuda.synchronize()

            torch.testing.assert_close(q, (x + 11) + 3)
            torch.testing.assert_close(kv, x + 29)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for JD DSV4 CP prefill correctness")
    _test_sparse_cp_combine()
    _test_cp_multi_stream_prepare()
    _test_cp_fused_wqkv_allocator_lifetime()
    print("JD DSV4 CP sparse-prefill and multi-stream correctness passed")


if __name__ == "__main__":
    main()
