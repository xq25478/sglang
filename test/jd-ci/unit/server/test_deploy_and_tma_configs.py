import json
import re
import unittest
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = (
    REPO_ROOT
    / "python/sglang/srt/layers/moe/moe_runner/triton_utils/configs"
)


class TestJDDeployContract(unittest.TestCase):
    def test_internal_deploy_scripts_are_shell_validated_and_map_models(self):
        start = (REPO_ROOT / "deploy/infer/start.sh").read_text(encoding="utf-8")
        show_gids = (REPO_ROOT / "deploy/infer/show_gids").read_text(encoding="utf-8")

        self.assertIn('ORIGINAL_ARGS[$i]="--model-path"', start)
        self.assertIn("MC_GID_INDEX", start)
        self.assertIn("gid_attrs/types", show_gids)
        self.assertIn("INDEX", show_gids)
        self.assertNotRegex(start + show_gids, r"^(<<<<<<< |>>>>>>> )")


class TestJDTMAConfigs(unittest.TestCase):
    EXPECTED_FILES = (
        "triton_3_5_1/E=256,N=384,device_name=NVIDIA_H20D,dtype=fp8_w8a8,block_shape=[128, 128].json",
        "triton_3_5_1/E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json",
        "triton_3_4_0/E=128,N=192,device_name=NVIDIA_H20D,dtype=fp8_w8a8.json",
        "triton_3_6_0/E=512,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128, 128].json",
    )

    def test_internal_tma_configs_are_nonempty_numeric_objects(self):
        for relative in self.EXPECTED_FILES:
            with self.subTest(relative=relative):
                path = CONFIG_ROOT / relative
                self.assertTrue(path.is_file(), path)
                value = json.loads(path.read_text(encoding="utf-8"))
                self.assertIsInstance(value, dict)
                self.assertTrue(value)
                self.assertTrue(
                    all(
                        isinstance(config, dict)
                        and config
                        and all(
                            isinstance(item, (int, float))
                            for item in config.values()
                        )
                        for config in value.values()
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
