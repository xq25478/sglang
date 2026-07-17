import __future__
import ast
import copy
import gc
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

try:
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase


REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_SOURCE = REPO_ROOT / "python/sglang/srt/models/deepseek_v4.py"


def _load_mqa_method(name: str, torch_stub):
    """Compile one production method without importing the CUDA runtime."""
    tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    mqa_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MQALayer"
    )
    method = copy.deepcopy(
        next(
            node
            for node in mqa_class.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "torch": torch_stub,
        "cp_all_gather_rerange_output": (
            lambda kv, cp_size, forward_batch, stream: kv
        ),
    }
    exec(
        compile(
            module,
            str(MODEL_SOURCE),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


class _QKVAStorage:
    pass


class _KVStorage:
    def contiguous(self):
        return self


class _Stream:
    def __init__(self, name: str, state: dict):
        self.name = name
        self.state = state

    def wait_stream(self, producer):
        if self.name != "current" or producer.name != "kv":
            return
        if not self.state["q_computed"]:
            raise AssertionError(
                "Q projection must remain overlapped before the KV-stream join"
            )
        if self.state["qkv_ref"]() is None:
            raise AssertionError(
                "fused WQKV storage was released before the KV-stream join"
            )
        self.state["kv_joined"] = True

    def wait_event(self, event):
        self.state["waited_events"].append(event)

    def record_event(self):
        event = object()
        self.state["recorded_events"].append(event)
        return event


class _Cuda:
    Event = object

    def __init__(self, current_stream):
        self._current_stream = current_stream

    def current_stream(self):
        return self._current_stream

    @staticmethod
    def stream(stream):
        return nullcontext()


class TestDSV4MultiStreamLifetime(CustomTestCase):
    def _run_prepare(self, method_name: str):
        state = {
            "q_computed": False,
            "kv_joined": False,
            "qkv_ref": None,
            "recorded_events": [],
            "waited_events": [],
        }
        current_stream = _Stream("current", state)
        streams = [
            _Stream("kv", state),
            _Stream("compressor", state),
            _Stream("indexer", state),
        ]
        torch_stub = SimpleNamespace(cuda=_Cuda(current_stream))
        prepare = _load_mqa_method(method_name, torch_stub)

        def wqkv_a(value):
            qkv_a = _QKVAStorage()
            state["qkv_ref"] = weakref.ref(qkv_a)
            return qkv_a, None

        def compute_q_a(value, qkv_a=None):
            if qkv_a is None:
                raise AssertionError("fused WQKV path was not exercised")
            return object()

        def compute_q_b(q_lora, positions, q_out=None):
            state["q_computed"] = True
            return "q"

        layer = SimpleNamespace(
            alt_streams=streams,
            fuse_wqa_wkv=True,
            wqkv_a=wqkv_a,
            _compute_q_a=compute_q_a,
            _compute_q_b=compute_q_b,
            _compute_kv_to_cache=lambda *args, **kwargs: None,
            _compute_kv_bf16=lambda *args, **kwargs: _KVStorage(),
            indexer=None,
            compressor=None,
            cp_size=2,
            layer_id=3,
        )
        backend = SimpleNamespace(store_cache=lambda **kwargs: None)

        result = prepare(layer, object(), object(), object(), backend)

        self.assertTrue(state["kv_joined"])
        self.assertEqual(len(state["recorded_events"]), 2)
        self.assertEqual(len(state["waited_events"]), 1)
        if method_name.endswith("_cp"):
            self.assertEqual(result[0], "q")
            self.assertIsInstance(result[1], _KVStorage)
        else:
            self.assertEqual(result, "q")

        gc.collect()
        self.assertIsNone(state["qkv_ref"]())

    def test_fused_wqkv_lives_through_standard_multistream_join(self):
        self._run_prepare("_forward_prepare_multi_stream")

    def test_fused_wqkv_lives_through_cp_multistream_join(self):
        self._run_prepare("_forward_prepare_multi_stream_cp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
