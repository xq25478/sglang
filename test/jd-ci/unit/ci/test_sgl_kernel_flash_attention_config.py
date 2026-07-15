import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:

    def register_cpu_ci(*args, **kwargs):
        return None

    CustomTestCase = unittest.TestCase


register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
CMAKE_FILE = REPO_ROOT / "sgl-kernel/CMakeLists.txt"
BUILD_SCRIPT = REPO_ROOT / "test/jd-ci/env/build_sgl_kernel.sh"


class TestSglKernelFlashAttentionConfig(CustomTestCase):
    def test_fa3_sparse_mask_is_disabled_by_default(self):
        cmake = CMAKE_FILE.read_text(encoding="utf-8")

        self.assertIn(
            'option(SGL_KERNEL_ENABLE_FA3_SPARSE_MASK  "Enable FA3 sparse mask kernels" OFF)',
            cmake,
        )

    def test_disabled_sparse_mask_prunes_flash_attention_templates(self):
        cmake = CMAKE_FILE.read_text(encoding="utf-8")

        self.assertIn("if(NOT SGL_KERNEL_ENABLE_FA3_SPARSE_MASK)", cmake)
        self.assertIn(
            "list(APPEND FLASH_OPS_COMPILE_DEFS FLASHATTENTION_DISABLE_SPARSE_MASK)",
            cmake,
        )

    def test_hdimdiff_aggregates_are_not_compiled_twice(self):
        cmake = CMAKE_FILE.read_text(encoding="utf-8")

        self.assertNotIn("flash_fwd_hdimdiff_", cmake)

    def test_fetchcontent_uses_release_scoped_dependency_directory(self):
        script = BUILD_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'FETCHCONTENT_CACHE_ROOT="${6:-${WHEEL_CACHE_ROOT}}"',
            script,
        )
        self.assertIn(
            'SGL_KERNEL_FETCHCONTENT_BASE_DIR="${FETCHCONTENT_CACHE_ROOT%/}/_deps"',
            script,
        )
        self.assertIn('mkdir -p "${SGL_KERNEL_FETCHCONTENT_BASE_DIR}"', script)
        self.assertIn(
            "-DFETCHCONTENT_BASE_DIR=${SGL_KERNEL_FETCHCONTENT_BASE_DIR}", script
        )

    def test_fetchcontent_cache_sanitizer_preserves_downloads_and_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            subbuild = cache_root / "repo-cutlass-subbuild"
            archive = (
                subbuild
                / "repo-cutlass-populate-prefix"
                / "src"
                / "cutlass.tar.gz"
            )
            archive.parent.mkdir(parents=True)
            archive.write_text("download", encoding="utf-8")
            (subbuild / "CMakeCache.txt").write_text(
                "/old/container/path", encoding="utf-8"
            )
            (subbuild / "CMakeFiles").mkdir()
            (subbuild / "CMakeFiles" / "state").write_text(
                "generated", encoding="utf-8"
            )

            build_dir = cache_root / "repo-cutlass-build"
            build_dir.mkdir()
            (build_dir / "object.o").write_text("generated", encoding="utf-8")

            source_dir = cache_root / "repo-cutlass-src"
            source_dir.mkdir()
            source_fixture = source_dir / "CMakeCache.txt"
            source_fixture.write_text("source fixture", encoding="utf-8")

            subprocess.run(
                [
                    "bash",
                    str(BUILD_SCRIPT),
                    "--sanitize-fetchcontent-cache",
                    str(cache_root),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(archive.is_file())
            self.assertTrue(source_fixture.is_file())
            self.assertFalse((subbuild / "CMakeCache.txt").exists())
            self.assertFalse((subbuild / "CMakeFiles").exists())
            self.assertFalse(build_dir.exists())

    def test_infllm_extension_architectures_are_verified(self):
        script = BUILD_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'INFLLM_OPS=$(find_installed_sgl_kernel_so "infllm_ops*.so")', script
        )
        self.assertIn('require_cubin_arch "${INFLLM_OPS}" "sm_90"', script)
        self.assertIn('require_cubin_arch "${INFLLM_OPS}" "sm_120a"', script)
        self.assertIn(
            'version_ge "${CUDA_TOOLKIT_VERSION}" "12.8"', script
        )


if __name__ == "__main__":
    unittest.main()
