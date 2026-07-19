import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import FP8_DTYPE, C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    NonPagedIndexerPlan,
    PagedIndexerMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsv4.indexer"


class TestDSV4NonPagedIndexer(CustomTestCase):
    def test_logits_chunk_rows_follow_memory_budget(self):
        get_chunk_rows = C4IndexerBackendMixin._get_mqa_logits_chunk_rows

        self.assertEqual(
            get_chunk_rows(num_q=1024, num_k=1024, logits_budget_bytes=1),
            1024,
        )

        num_q = 65_280
        num_k = 86_016
        budget = 4 * 1024**3
        expected = budget // (num_k * 4)
        self.assertEqual(
            get_chunk_rows(
                num_q=num_q,
                num_k=num_k,
                logits_budget_bytes=budget,
            ),
            expected,
        )
        self.assertEqual(
            get_chunk_rows(
                num_q=num_q,
                num_k=num_k,
                logits_budget_bytes=num_q * num_k * 4,
            ),
            num_q,
        )

    def test_logits_budget_tracks_each_batch_free_memory(self):
        mem_get_info = MagicMock(
            side_effect=[(500, 1000), (200, 1000), (500, 1000)]
        )
        batch_budgets = [{}, {}, {}]
        with (
            envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.override(0.2),
            patch(
                f"{_INDEXER}.get_global_server_args",
                return_value=SimpleNamespace(mem_fraction_static=0.7),
            ),
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=False),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=1000),
            ),
            patch("torch.cuda.mem_get_info", mem_get_info),
        ):
            first = C4IndexerBackendMixin._get_mqa_logits_budget_bytes(
                0, batch_budgets[0]
            )
            first_reused = C4IndexerBackendMixin._get_mqa_logits_budget_bytes(
                0, batch_budgets[0]
            )
            lower = C4IndexerBackendMixin._get_mqa_logits_budget_bytes(
                0, batch_budgets[1]
            )
            recovered = C4IndexerBackendMixin._get_mqa_logits_budget_bytes(
                0, batch_budgets[2]
            )

        self.assertEqual(first, 60)
        self.assertEqual(first_reused, first)
        self.assertEqual(lower, 40)
        self.assertEqual(recovered, 60)
        self.assertEqual(mem_get_info.call_count, 3)
        mem_get_info.assert_called_with(0)

    def test_chunked_topk_matches_full_topk(self):
        batch_size = 7
        max_seq_len = 128
        topk = 8
        scores = torch.arange(
            batch_size * max_seq_len, dtype=torch.float32
        ).reshape(batch_size, max_seq_len)
        seq_lens = torch.tensor([128, 127, 96, 65, 64, 63, 32], dtype=torch.int32)
        page_tables = torch.arange(batch_size * 2, dtype=torch.int32).reshape(
            batch_size, 2
        )
        full_pages = torch.full((batch_size, topk), -1, dtype=torch.int32)
        full_raw = torch.full_like(full_pages, -1)
        chunked_pages = torch.full_like(full_pages, -1)
        chunked_raw = torch.full_like(full_pages, -1)

        with envs.SGLANG_TOPK_TRANSFORM_512_TORCH.override(True):
            C4IndexerBackendMixin._transform_indexer_topk(
                logits=scores,
                c4_seq_lens=seq_lens,
                page_table=page_tables,
                out_page_indices=full_pages,
                c4_page_size=64,
                raw_indices=full_raw,
                topk_metadata=None,
            )
            for start, end in ((0, 3), (3, 5), (5, 7)):
                C4IndexerBackendMixin._transform_indexer_topk(
                    logits=scores[start:end],
                    c4_seq_lens=seq_lens[start:end],
                    page_table=page_tables[start:end],
                    out_page_indices=chunked_pages[start:end],
                    c4_page_size=64,
                    raw_indices=chunked_raw[start:end],
                    topk_metadata=None,
                )

        torch.testing.assert_close(chunked_pages, full_pages)
        torch.testing.assert_close(chunked_raw, full_raw)

    def test_deep_gemm_metadata_is_replanned_for_exact_chunk(self):
        metadata = PagedIndexerMetadata.__new__(PagedIndexerMetadata)
        metadata.page_size = 256
        planned = torch.tensor([17], dtype=torch.int32)
        get_metadata = MagicMock(return_value=planned)
        deep_gemm = SimpleNamespace(
            get_num_sms=MagicMock(return_value=132),
            get_paged_mqa_logits_metadata=get_metadata,
        )
        c4_seq_lens = torch.tensor([63, 64, 65], dtype=torch.int64)

        with (
            envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.override(False),
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
        ):
            actual = metadata.build_deep_gemm_metadata(c4_seq_lens)

        self.assertIs(actual, planned)
        call = get_metadata.call_args
        torch.testing.assert_close(
            call.args[0], c4_seq_lens.to(torch.int32).unsqueeze(-1)
        )
        self.assertEqual(call.args[1:], (64, 132))

    def test_chunk_metadata_is_cached_per_query_range(self):
        metadata = PagedIndexerMetadata.__new__(PagedIndexerMetadata)
        metadata._deep_gemm_chunk_metadata = {}
        metadata._topk_chunk_metadata = {}
        deep_gemm_plan = MagicMock(
            side_effect=lambda seq_lens: seq_lens.new_tensor([seq_lens.numel()])
        )
        metadata.build_deep_gemm_metadata = deep_gemm_plan
        topk_plan = MagicMock(
            side_effect=lambda seq_lens: seq_lens.new_tensor([seq_lens.numel()])
        )
        c4_seq_lens = torch.tensor([60, 61, 62, 63], dtype=torch.int32)

        with patch("sglang.jit_kernel.dsv4.plan_topk_v2", topk_plan):
            first_deep_gemm = metadata.get_deep_gemm_chunk_metadata(
                0, 2, c4_seq_lens[:2]
            )
            second_deep_gemm = metadata.get_deep_gemm_chunk_metadata(
                0, 2, c4_seq_lens[:2]
            )
            first_topk = metadata.get_topk_chunk_metadata(0, 2, c4_seq_lens[:2])
            second_topk = metadata.get_topk_chunk_metadata(0, 2, c4_seq_lens[:2])

        self.assertIs(first_deep_gemm, second_deep_gemm)
        self.assertIs(first_topk, second_topk)
        deep_gemm_plan.assert_called_once()
        topk_plan.assert_called_once()

    def test_chunk_metadata_cache_is_cleared_when_batch_is_copied(self):
        metadata = PagedIndexerMetadata.__new__(PagedIndexerMetadata)
        metadata.nonpaged_plan = MagicMock()
        metadata._deep_gemm_chunk_metadata = {(0, 2): MagicMock()}
        metadata._topk_chunk_metadata = {(0, 2): MagicMock()}
        metadata._mqa_logits_batch_budget_bytes = {0: 60}

        with (
            patch("sglang.srt.layers.attention.dsv4.metadata.is_hip", return_value=False),
            patch("sglang.srt.layers.attention.dsv4.metadata.copy_metadata"),
        ):
            metadata.copy_(MagicMock())

        self.assertIsNone(metadata.nonpaged_plan)
        self.assertEqual(metadata._deep_gemm_chunk_metadata, {})
        self.assertEqual(metadata._topk_chunk_metadata, {})
        self.assertEqual(metadata._mqa_logits_batch_budget_bytes, {})

    def _is_eligible(self, **overrides):
        backend = SimpleNamespace(hisparse_coordinator=None)
        c4_indexer = SimpleNamespace(use_fp4_indexer=overrides.get("fp4", False))
        forward_batch = SimpleNamespace(
            forward_mode=overrides.get("mode", ForwardMode.EXTEND),
            _original_forward_mode=overrides.get("original_mode"),
            tbo_parent_token_range=overrides.get("tbo"),
            batch_size=overrides.get("batch_size", 1),
        )
        metadata = SimpleNamespace(
            use_prefill_cuda_graph=overrides.get("prefill_graph", False)
        )
        with (
            envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.override(
                overrides.get("enabled", True)
            ),
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(False),
            envs.SGLANG_OPT_USE_AITER_INDEXER.override(False),
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            patch(f"{_INDEXER}.is_cuda", return_value=True),
            patch(f"{_INDEXER}.is_hip", return_value=False),
            patch(f"{_INDEXER}.get_attention_cp_size", return_value=1),
            patch(
                f"{_INDEXER}.is_in_tc_piecewise_cuda_graph",
                return_value=overrides.get("piecewise_graph", False),
            ),
            patch(f"{_INDEXER}.is_in_breakable_cuda_graph", return_value=False),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            return C4IndexerBackendMixin._can_use_nonpaged_indexer(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=forward_batch,
                indexer_metadata=metadata,
            )

    def test_eligibility_is_fail_closed(self):
        self.assertIs(envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.default, True)
        self.assertEqual(
            envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS.default, 8192
        )
        self.assertTrue(self._is_eligible())
        for case in (
            {"enabled": False},
            {"mode": ForwardMode.DECODE},
            {"original_mode": ForwardMode.DECODE},
            {"batch_size": 2},
            {"batch_size": 20_000},
            {"tbo": (1, 2)},
            {"prefill_graph": True},
            {"piecewise_graph": True},
            {"fp4": True},
        ):
            with self.subTest(case=case):
                self.assertFalse(self._is_eligible(**case))

    def test_single_request_plan_contract(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        query_rows = 4
        batch = SimpleNamespace(
            seq_lens=torch.tensor([262], dtype=torch.int32),
            seq_lens_cpu=[262],
            extend_seq_lens_cpu=[query_rows],
            extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_num_tokens=query_rows,
        )
        metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)
        page_table = torch.tensor([[3, 1]], dtype=torch.int32).repeat(query_rows, 1)
        c4_seq_lens = torch.tensor([62, 63, 64, 65], dtype=torch.int32)

        def build_plan():
            return C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=batch,
                indexer_metadata=metadata,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                query_rows=query_rows,
            )

        threshold = envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS
        with threshold.override(threshold.default):
            self.assertIsNone(build_plan())
        with threshold.override(query_rows):
            plan = build_plan()
        self.assertEqual(
            (plan.seq_len_sum, plan.max_seqlen_k, plan.query_rows),
            (65, 128, query_rows),
        )
        torch.testing.assert_close(plan.page_table, page_table[:1])
        torch.testing.assert_close(plan.ke, c4_seq_lens)
        torch.testing.assert_close(plan.gather_seq_lens, c4_seq_lens[-1:])

        metadata.nonpaged_plan = None
        batch.extend_seq_lens_cpu = [2, 2]
        with threshold.override(0):
            self.assertIsNone(build_plan())

    def test_extreme_plan_metadata_is_bounded_and_fail_closed(self):
        backend = SimpleNamespace(_can_use_nonpaged_indexer=lambda **_: True)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        query_rows = 4
        batch = SimpleNamespace(
            seq_lens=torch.tensor([500_000], dtype=torch.int32),
            seq_lens_cpu=[500_000],
            extend_seq_lens_cpu=[query_rows],
            extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
            extend_start_loc=torch.tensor([0], dtype=torch.int32),
            extend_num_tokens=query_rows,
        )
        metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)
        page_table = torch.zeros((query_rows, 1), dtype=torch.int32)
        c4_seq_lens = torch.tensor(
            [124_997, 124_998, 124_999, 125_000], dtype=torch.int32
        )

        def build_plan():
            return C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=batch,
                indexer_metadata=metadata,
                page_table=page_table,
                c4_seq_lens=c4_seq_lens,
                query_rows=query_rows,
            )

        threshold = envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS
        with threshold.override(query_rows):
            plan = build_plan()
        self.assertEqual(plan.seq_len_sum, 125_000)
        self.assertEqual(plan.max_seq_len, 125_000)
        self.assertEqual(plan.max_seqlen_k, 125_056)

        metadata.nonpaged_plan = None
        batch.seq_lens = torch.tensor([500_000, 200], dtype=torch.int32)
        batch.seq_lens_cpu = [500_000, 200]
        batch.extend_seq_lens_cpu = [2, 2]
        batch.extend_seq_lens = torch.tensor([2, 2], dtype=torch.int32)
        batch.extend_start_loc = torch.tensor([0, 2], dtype=torch.int32)
        with threshold.override(query_rows):
            self.assertIsNone(build_plan())

    def test_query_threshold_boundary(self):
        can_use_nonpaged_indexer = MagicMock(return_value=True)
        backend = SimpleNamespace(_can_use_nonpaged_indexer=can_use_nonpaged_indexer)
        c4_indexer = SimpleNamespace(use_fp4_indexer=False)
        metadata = SimpleNamespace(nonpaged_plan=None, c4_page_size=64)

        def build_plan(query_rows):
            batch = SimpleNamespace(
                seq_lens=torch.tensor([query_rows], dtype=torch.int32),
                seq_lens_cpu=[query_rows],
                extend_seq_lens_cpu=[query_rows],
                extend_seq_lens=torch.tensor([query_rows], dtype=torch.int32),
                extend_start_loc=torch.tensor([0], dtype=torch.int32),
                extend_num_tokens=query_rows,
            )
            c4_seq_lens = torch.div(
                torch.arange(1, query_rows + 1, dtype=torch.int32),
                4,
                rounding_mode="floor",
            ).clamp_min_(1)
            return C4IndexerBackendMixin._get_nonpaged_indexer_plan(
                backend,
                c4_indexer=c4_indexer,
                forward_batch=batch,
                indexer_metadata=metadata,
                page_table=torch.zeros((query_rows, 1), dtype=torch.int32),
                c4_seq_lens=c4_seq_lens,
                query_rows=query_rows,
            )

        for query_rows, expected in ((8191, False), (8192, True), (8193, True)):
            with self.subTest(query_rows=query_rows):
                metadata.nonpaged_plan = None
                can_use_nonpaged_indexer.reset_mock()
                self.assertIs(build_plan(query_rows) is not None, expected)
                if expected:
                    can_use_nonpaged_indexer.assert_called_once()
                else:
                    can_use_nonpaged_indexer.assert_not_called()

        metadata.nonpaged_plan = None
        threshold = envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS
        with threshold.override(8193):
            self.assertIsNone(build_plan(8192))

    def test_nonpaged_dispatch_uses_gathered_kv_contract(self):
        query_rows = 4
        plan = NonPagedIndexerPlan(
            page_table=torch.tensor([[3, 1]], dtype=torch.int32),
            gather_seq_lens=torch.tensor([65], dtype=torch.int32),
            ks=torch.zeros(query_rows, dtype=torch.int32),
            ke=torch.tensor([62, 63, 64, 65], dtype=torch.int32),
            seq_len_sum=65,
            max_seq_len=65,
            max_seqlen_k=128,
            query_rows=query_rows,
        )
        q_indexer = torch.zeros((6, 2, 128), dtype=torch.uint8).view(FP8_DTYPE)
        weights = torch.ones((6, 2), dtype=torch.float32)
        k_u8 = torch.zeros((65, 128), dtype=torch.uint8)
        scale_u8 = torch.zeros((65, 4), dtype=torch.uint8)
        token_to_kv_pool = MagicMock()
        token_to_kv_pool.get_index_k_scale_buffer.return_value = (k_u8, scale_u8)
        c4_indexer = SimpleNamespace(layer_id=17)
        expected = MagicMock(name="logits")
        deep_gemm = SimpleNamespace(fp8_mqa_logits=MagicMock(return_value=expected))

        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            actual = C4IndexerBackendMixin._forward_nonpaged_indexer(
                q_indexer=q_indexer,
                weights=weights,
                c4_indexer=c4_indexer,
                token_to_kv_pool=token_to_kv_pool,
                plan=plan,
            )

        self.assertIs(actual, expected)
        token_to_kv_pool.get_index_k_scale_buffer.assert_called_once_with(
            layer_id=17,
            seq_len_tensor=plan.gather_seq_lens,
            page_indices=plan.page_table,
            seq_len_sum=65,
            max_seq_len=65,
        )
        call = deep_gemm.fp8_mqa_logits.call_args
        torch.testing.assert_close(call.args[0], q_indexer[:query_rows])
        torch.testing.assert_close(call.args[1][0], k_u8.view(FP8_DTYPE))
        torch.testing.assert_close(
            call.args[1][1], scale_u8.view(torch.float32).squeeze(-1)
        )
        torch.testing.assert_close(call.args[2], weights[:query_rows])
        torch.testing.assert_close(call.args[3], plan.ks)
        torch.testing.assert_close(call.args[4], plan.ke)
        self.assertEqual(call.kwargs, {"clean_logits": False, "max_seqlen_k": 128})


if __name__ == "__main__":
    unittest.main()
