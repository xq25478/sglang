import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DSV4RawDecodeMetadata,
    DSV4RawVerifyMetadata,
    DeepseekV4AttnBackend,
)


class TestDSV4InferenceBufferCopy(unittest.TestCase):
    def test_full_metadata_copy_supports_inference_tensors(self):
        class TensorBackedFullMetadata:
            def __init__(self):
                with torch.inference_mode():
                    self.value = torch.zeros(1)

            def copy_(self, other):
                self.value.copy_(other.value)

        raw_metadata_cases = (
            (
                DSV4RawDecodeMetadata(
                    req_pool_indices=torch.zeros(1, dtype=torch.int32),
                    seq_lens=torch.ones(1, dtype=torch.int32),
                    out_cache_loc=torch.zeros(1, dtype=torch.int32),
                ),
                "make_forward_metadata_from_raw_decode",
            ),
            (
                DSV4RawVerifyMetadata(
                    req_pool_indices=torch.zeros(1, dtype=torch.int32),
                    seq_lens=torch.ones(1, dtype=torch.int32),
                    out_cache_loc=torch.zeros(1, dtype=torch.int32),
                ),
                "make_forward_metadata_from_raw_verify",
            ),
        )

        for raw_metadata, factory_name in raw_metadata_cases:
            with self.subTest(raw_metadata=type(raw_metadata).__name__):
                backend = object.__new__(DeepseekV4AttnBackend)
                backend.forward_metadata = raw_metadata
                backend.online_c128_mtp = SimpleNamespace(
                    state_slot_offset=MagicMock(return_value=0)
                )
                full_metadata = TensorBackedFullMetadata()
                source_metadata = SimpleNamespace(value=torch.ones(1))
                backend._lookup_full_metadata_buffer = MagicMock(
                    return_value=full_metadata
                )
                setattr(
                    backend,
                    factory_name,
                    MagicMock(return_value=source_metadata),
                )

                backend.init_forward_metadata_in_graph(
                    SimpleNamespace(out_cache_loc=None)
                )

                self.assertIs(backend.forward_metadata, full_metadata)
                self.assertTrue(torch.is_inference(full_metadata.value))
                torch.testing.assert_close(full_metadata.value, source_metadata.value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
