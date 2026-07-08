import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
TRITON_RUNNER = REPO_ROOT / "python/sglang/srt/layers/moe/moe_runner/triton.py"
TRITON_KERNELS = (
    REPO_ROOT
    / "python/sglang/srt/layers/moe/moe_runner/triton_utils"
    / "fused_moe_triton_kernels.py"
)
TRITON_CONFIG = (
    REPO_ROOT
    / "python/sglang/srt/layers/moe/moe_runner/triton_utils"
    / "fused_moe_triton_config.py"
)
TRITON_FUSED_MOE = (
    REPO_ROOT
    / "python/sglang/srt/layers/moe/moe_runner/triton_utils"
    / "fused_moe.py"
)
TUNER = REPO_ROOT / "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py"
SEP_TUNER = (
    REPO_ROOT / "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton_sep.py"
)


def _source(path: Path) -> str:
    return path.read_text()


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(_source(path))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _args(path: Path, name: str) -> set[str]:
    arguments = _function(path, name).args
    return {arg.arg for arg in (*arguments.args, *arguments.kwonlyargs)}


def _load_function(path: Path, name: str, globals_dict: dict):
    function = _function(path, name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    namespace = dict(globals_dict)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def test_triton_quant_info_exposes_w4a8_flag():
    tree = ast.parse(_source(TRITON_RUNNER))
    quant_info = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TritonMoeQuantInfo"
    )
    fields = {
        node.target.id
        for node in quant_info.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert "use_int4_w4a8" in fields


def test_w4a8_flag_is_threaded_through_fused_moe_entrypoints():
    for name in (
        "inplace_fused_experts",
        "outplace_fused_experts",
        "fused_experts",
        "_prepare_fused_moe_run",
        "_fused_moe_kernel_sequence",
        "fused_experts_impl",
        "fused_moe",
    ):
        assert "use_int4_w4a8" in _args(TRITON_FUSED_MOE, name), name
    assert "use_int4_w4a8" in _args(TRITON_KERNELS, "invoke_fused_moe_kernel")


def test_config_dtype_has_independent_int4_w4a8_value():
    fake_torch = type("FakeTorch", (), {"float": object()})
    get_dtype = _load_function(
        TRITON_CONFIG, "get_config_dtype_str", {"torch": fake_torch}
    )
    assert get_dtype(object(), use_int4_w4a8=True) == "int4_w4a8"


def test_w4a8_config_filename_keeps_zero_block_n_selector():
    get_filename = _load_function(
        TRITON_CONFIG,
        "get_config_file_name",
        {"get_device_name": lambda: "NVIDIA H20"},
    )
    assert get_filename(256, 128, "int4_w4a8", [0, 128]) == (
        "E=256,N=128,device_name=NVIDIA_H20,dtype=int4_w4a8,"
        "block_shape=[0, 128].json"
    )


def test_w4a8_kernel_quantizes_activation_and_uses_scaled_fp8_dot():
    source = _source(TRITON_KERNELS)
    invoke_source = ast.unparse(_function(TRITON_KERNELS, "invoke_fused_moe_kernel"))
    kernel_source = ast.unparse(_function(TRITON_KERNELS, "fused_moe_kernel_gptq_awq"))

    assert "use_int4_w4a8: tl.constexpr" in source
    assert "sglang_per_token_group_quant_fp8(A, block_k)" in invoke_source
    assert "use_int4_w4a8" in kernel_source
    assert "a_scale" in kernel_source
    assert "b_scale" in kernel_source
    assert "tl.dot" in kernel_source


def test_w4a8_kernel_sign_extends_signed_int4_nibbles():
    kernel_source = ast.unparse(_function(TRITON_KERNELS, "fused_moe_kernel_gptq_awq"))

    assert "tl.where(b >= 8, b - 16, b)" in kernel_source
    assert "tl.where(b >= 8, b - 16, b).to(tl.float32).to(a.dtype)" in kernel_source


def test_tuners_accept_int4_w4a8_dtype():
    for path in (TUNER, SEP_TUNER):
        source = _source(path)
        assert '"int4_w4a8"' in source
        assert "use_int4_w4a8" in source
