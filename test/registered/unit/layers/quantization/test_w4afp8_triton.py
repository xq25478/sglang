import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
W4AFP8 = REPO_ROOT / "python/sglang/srt/layers/quantization/w4afp8.py"


def _source() -> str:
    return W4AFP8.read_text()


def _method(class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(_source())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_triton_preserves_natural_group_scale_layout():
    source = ast.unparse(_method("W4AFp8MoEMethod", "process_weights_after_loading"))
    assert "use_triton" in source
    assert "interleave_scales" in source
    assert "elif use_triton" in source


def test_triton_runner_uses_native_w4a8_quant_info():
    create_source = ast.unparse(_method("W4AFp8MoEMethod", "create_moe_runner"))
    apply_source = ast.unparse(_method("W4AFp8MoEMethod", "_apply_triton"))

    assert "moe_runner_backend.is_triton()" in create_source
    assert "use_int4_w4a8=True" in apply_source
    assert "use_int4_w4a16" not in apply_source
    assert "block_shape=[0, group_size]" in apply_source


def test_triton_w4a8_rejects_non_128_group_size():
    apply_source = ast.unparse(_method("W4AFp8MoEMethod", "_apply_triton"))
    assert "group_size != 128" in apply_source
    assert "ValueError" in apply_source
