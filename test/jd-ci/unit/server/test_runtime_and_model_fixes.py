import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.quantization.w4afp8 import (
    get_cutlass_w4a8_scale_pack,
    interleave_scales,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiHttpWorkerDetokenizerMixin,
)


class TestJDRequestAndMultimodalRuntime(unittest.TestCase):
    def test_worker_ids_accept_single_and_multiple_request_ids(self):
        mixin = object.__new__(MultiHttpWorkerDetokenizerMixin)

        self.assertEqual(mixin.get_worker_ids_from_req_rids("3_request"), [3])
        self.assertEqual(
            mixin.get_worker_ids_from_req_rids(["1_a", "7_b"]), [1, 7]
        )
        self.assertEqual(mixin.get_worker_ids_from_req_rids("invalid"), [])

    def test_cuda_graph_metadata_lookup_is_safe_before_warmup(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.forward_metadata = object()

        self.assertIsNone(backend._lookup_full_metadata_buffer())

    def test_cuda_graph_metadata_lookup_reuses_matching_buffer(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        raw = object()
        full = object()
        backend.forward_metadata = raw
        backend.cuda_graph_metadata_of_bucket_and_bs = {"decode": {4: raw}}
        backend.cuda_graph_full_metadata_of_bucket_and_bs = {"decode": {4: full}}

        self.assertIs(backend._lookup_full_metadata_buffer(), full)


class TestJDW4A8Configuration(unittest.TestCase):
    def test_scale_pack_supports_jd_group_sizes(self):
        self.assertEqual(get_cutlass_w4a8_scale_pack(7168, 32), 4)
        self.assertEqual(get_cutlass_w4a8_scale_pack(7168, 128), 4)
        self.assertEqual(get_cutlass_w4a8_scale_pack(7040, 128), 1)
        self.assertEqual(get_cutlass_w4a8_scale_pack(4096, 128), 4)
        with self.assertRaisesRegex(ValueError, "Unsupported W4A8 group_size"):
            get_cutlass_w4a8_scale_pack(4096, 64)

    def test_scale_interleave_matches_jd_packed_layout(self):
        scales = torch.arange(1 * 2 * 4, dtype=torch.float32).reshape(1, 2, 4)

        packed = interleave_scales(scales, scale_pack=2)

        expected = torch.tensor([[[0.0, 1.0, 4.0, 5.0], [2.0, 3.0, 6.0, 7.0]]])
        torch.testing.assert_close(packed, expected)


class TestJDEPLBAutoDispatch(unittest.TestCase):
    def test_extend_and_decode_batches_route_to_different_recorders(self):
        from sglang.srt.eplb.expert_distribution import _DeepepAutoSinglePassGatherer

        gatherer = object.__new__(_DeepepAutoSinglePassGatherer)
        gatherer._normal_gatherer = MagicMock()
        gatherer._low_latency_gatherer = MagicMock()
        gatherer._is_extend_in_batch = False

        gatherer.on_forward_pass_start(SimpleNamespace(is_extend_in_batch=True))
        gatherer.on_select_experts(0, MagicMock())
        gatherer.on_deepep_dispatch_low_latency(0, MagicMock())
        gatherer._normal_gatherer.on_select_experts.assert_called_once()
        gatherer._low_latency_gatherer.on_deepep_dispatch_low_latency.assert_not_called()

        gatherer.on_forward_pass_start(SimpleNamespace(is_extend_in_batch=False))
        gatherer.on_deepep_dispatch_low_latency(0, MagicMock())
        gatherer._low_latency_gatherer.on_deepep_dispatch_low_latency.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
