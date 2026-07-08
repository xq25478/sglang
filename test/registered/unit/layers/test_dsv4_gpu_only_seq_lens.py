"""CPU contract tests for host-free DSV4 speculative metadata preparation."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4GpuOnlySeqLens(unittest.TestCase):
    def test_backend_exposes_model_context_len_to_speculative_workers(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        model_runner = SimpleNamespace(
            device="cpu",
            model_config=SimpleNamespace(context_len=131072, head_dim=0),
        )

        with self.assertRaisesRegex(AssertionError, "DSV4 MQA head_dim"):
            DeepseekV4AttnBackend.__init__(backend, model_runner)

        self.assertEqual(backend.max_context_len, 131072)

    def test_target_verify_build_accepts_gpu_only_seq_lens(self):
        backend = object.__new__(DeepseekV4AttnBackend)
        req_to_token = torch.zeros((4, 32), dtype=torch.int32)
        backend.req_to_token = req_to_token
        backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        backend.swa_page_size = 128
        backend.page_size = 256
        backend.MAX_SEQ_LEN_FOR_CAPTURE = 4096
        backend.is_dspark_draft = False
        backend.online_c128_mtp = SimpleNamespace(
            prepare_forward=lambda *args, **kwargs: 0
        )
        backend._resolve_verify_layout = lambda forward_batch, bs: None

        captured = {}

        def init_target_verify(**kwargs):
            captured.update(kwargs)
            return "metadata"

        backend.init_forward_metadata_target_verify = init_target_verify
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([128, 256], dtype=torch.int32),
            seq_lens_cpu=None,
            out_cache_loc=torch.arange(8, dtype=torch.int32),
            spec_info=SimpleNamespace(draft_token=None, draft_token_num=4),
        )

        metadata = backend._build_forward_metadata(batch)

        self.assertEqual(metadata, "metadata")
        self.assertEqual(captured["max_seq_len"], 4096)
        self.assertIsNone(captured["seq_lens_cpu"])


if __name__ == "__main__":
    unittest.main()
