import math
import unittest

import torch

from sglang.jit_kernel.dsv4 import fused_k_norm_rope_flashmla


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestFusedKNormRopeFlashMLASentinel(unittest.TestCase):
    def test_negative_cache_location_is_a_noop(self) -> None:
        torch.manual_seed(0)
        page_size = 256
        page_bytes = math.ceil(584 * page_size / 576) * 576
        kv = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
        kv_weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        freqs_cis = torch.polar(
            torch.ones(8, 32, device="cuda", dtype=torch.float32),
            torch.zeros(8, 32, device="cuda", dtype=torch.float32),
        )
        positions = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
        out_loc = torch.tensor([-1, 0], device="cuda", dtype=torch.int32)
        cache = torch.zeros(2, page_bytes, device="cuda", dtype=torch.uint8)

        fused_k_norm_rope_flashmla(
            kv=kv,
            kv_weight=kv_weight,
            eps=1e-6,
            freqs_cis=freqs_cis,
            positions=positions,
            out_loc=out_loc,
            kvcache=cache,
            page_size=page_size,
        )
        torch.cuda.synchronize()

        self.assertTrue(cache[0].any().item())
        self.assertFalse(cache[1].any().item())


if __name__ == "__main__":
    unittest.main()
