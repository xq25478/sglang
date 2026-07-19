import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkPDMetadataConfig(unittest.TestCase):
    def _new_scheduler(self, mode, *, is_dspark=False):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = mode
        scheduler.transfer_backend = TransferBackend.MOONCAKE
        scheduler.spec_algorithm = SimpleNamespace(is_dspark=lambda: is_dspark)
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                spec_aux_config=SimpleNamespace(dflash_target_layer_ids=(40, 41, 42))
            )
        )
        scheduler.model_config = SimpleNamespace(
            hidden_size=7168,
            window_size=4096,
            num_hidden_layers=61,
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        )
        scheduler.server_args = SimpleNamespace(
            max_prefill_buffer_tokens=lambda: 16384
        )
        scheduler.max_prefill_tokens = 16384
        scheduler.max_running_requests = 128
        scheduler.ps = SimpleNamespace(pp_size=1, pp_rank=0, gpu_id=0)
        return scheduler

    def test_base_eagle_prefill_does_not_register_dspark_hidden(self):
        scheduler = self._new_scheduler(DisaggregationMode.PREFILL)

        with patch.object(
            envs.SGLANG_DSPARK_PD_TARGET_LAYER_IDS, "get", return_value=()
        ), patch("torch.cuda.is_available", return_value=False):
            config = scheduler.init_dspark_disaggregation_metadata_config()

        self.assertEqual(config, (0, 0, 0, "cpu"))
        self.assertEqual(scheduler.dspark_pd_local_capture_layer_ids, [])

    def test_explicit_dspark_prefill_registers_hidden_pool(self):
        scheduler = self._new_scheduler(DisaggregationMode.PREFILL)

        with patch.object(
            envs.SGLANG_DSPARK_PD_TARGET_LAYER_IDS,
            "get",
            return_value=(40, 41, 42),
        ), patch("torch.cuda.is_available", return_value=False):
            _, pool_rows, hidden_size, device = (
                scheduler.init_dspark_disaggregation_metadata_config()
            )

        self.assertGreater(pool_rows, 0)
        self.assertEqual(hidden_size, 3 * 7168)
        self.assertEqual(device, "cpu")
        self.assertEqual(scheduler.dspark_pd_local_capture_layer_ids, [40, 41, 42])


if __name__ == "__main__":
    unittest.main()
