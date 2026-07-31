from pathlib import Path
from types import SimpleNamespace

import torch

import sglang.srt.model_executor.runner.base_runner as base_runner_module
import sglang.srt.model_executor.runner.flashinfer_autotune as flashinfer_autotune
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMoeQuantInfo,
)
from sglang.srt.layers.moe.moe_runner.flashinfer_w4a8_autotune import (
    RouteAwareProfile,
    observe_route,
    save_profile,
    set_active_profile,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.base_runner import (
    BaseRunner,
    _allocate_decode_buffers,
)


def _runner(backend, chunked_prefill_size):
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            server_args=SimpleNamespace(
                moe_runner_backend=backend,
                chunked_prefill_size=chunked_prefill_size,
            )
        )
    )


def test_prefill_autotune_uses_configured_chunk_size():
    runner = _runner("flashinfer_cutlass", 4096)

    assert BaseRunner._flashinfer_autotune_prefill_batch_size(runner) == 4096


def test_prefill_autotune_caps_shape_at_tuning_ceiling():
    runner = _runner("flashinfer_cutlass", 16384)

    assert BaseRunner._flashinfer_autotune_prefill_batch_size(runner) == 8192


def test_prefill_autotune_skips_other_moe_backends():
    runner = _runner("cutlass", 8192)

    assert BaseRunner._flashinfer_autotune_prefill_batch_size(runner) is None


def test_dummy_buffer_allocation_can_skip_large_logits_tensor():
    buffers = _allocate_decode_buffers(
        device=torch.device("cpu"),
        max_bs=1,
        max_num_token=8192,
        hidden_size=8,
        vocab_size=100_000,
        dtype=torch.bfloat16,
        dp_size=1,
        pp_size=1,
        is_encoder_decoder=False,
        require_mlp_tp_gather=False,
        seq_len_fill_value=1,
        encoder_len_fill_value=0,
        num_tokens_per_bs=8192,
        cache_loc_dtype=torch.int64,
        enable_mamba_track=False,
        allocate_logits_buffer=False,
    )

    assert buffers.next_token_logits_buffer is None


def test_explicit_dummy_forward_mode_wins_over_speculative_default():
    model_runner = SimpleNamespace(
        is_generation=True,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(
            is_speculative=lambda: True,
            supports_target_verify_for_draft=lambda: True,
        ),
        decode_num_tokens_per_bs=lambda: 5,
    )

    assert base_runner_module._resolve_dummy_forward_shape(
        model_runner, ForwardMode.EXTEND
    ) == (ForwardMode.EXTEND, 1)
    assert base_runner_module._resolve_dummy_forward_shape(
        model_runner, ForwardMode.DECODE
    ) == (ForwardMode.DECODE, 1)
    assert base_runner_module._resolve_dummy_forward_shape(
        model_runner, ForwardMode.TARGET_VERIFY
    ) == (ForwardMode.TARGET_VERIFY, 5)
    assert base_runner_module._resolve_dummy_forward_shape(model_runner, None) == (
        ForwardMode.TARGET_VERIFY,
        5,
    )


def _calibration_sample():
    quant_info = FlashInferCutlassMoeQuantInfo(
        quant_type="w4a8",
        w13_weight=torch.empty((4, 16, 4), dtype=torch.uint8),
        w2_weight=torch.empty((4, 8, 8), dtype=torch.uint8),
        quant_scales=[torch.empty(0) for _ in range(8)],
        group_size=128,
    )
    runner_config = MoeRunnerConfig(
        num_experts=4,
        num_local_experts=4,
        hidden_size=8,
        intermediate_size_per_partition=8,
        top_k=2,
        activation="silu",
        is_gated=True,
    )
    return quant_info, runner_config


def test_route_profile_metadata_uses_parallel_state(monkeypatch):
    import flashinfer

    runner = SimpleNamespace(
        device="cuda",
        dtype=torch.bfloat16,
        ps=SimpleNamespace(
            tp_size=8,
            pp_size=2,
            attn_dp_size=3,
            moe_ep_size=4,
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        server_args=SimpleNamespace(model_path="/model"),
    )
    monkeypatch.setattr(flashinfer, "__version__", "test-version")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _device: (9, 0),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _device: "NVIDIA H20",
    )

    metadata = flashinfer_autotune._flashinfer_w4a8_profile_metadata(
        runner,
        _calibration_sample(),
    )

    assert metadata["tp_size"] == 8
    assert metadata["pp_size"] == 2
    assert metadata["dp_size"] == 3
    assert metadata["moe_ep_size"] == 4


def test_route_profile_metadata_uses_legacy_model_runner_fields(monkeypatch):
    import flashinfer

    runner = SimpleNamespace(
        device="cuda",
        dtype=torch.bfloat16,
        tp_size=8,
        pp_size=2,
        dp_size=3,
        moe_ep_size=4,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        server_args=SimpleNamespace(model_path="/model"),
    )
    monkeypatch.setattr(flashinfer, "__version__", "test-version")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA H20")

    metadata = flashinfer_autotune._flashinfer_w4a8_profile_metadata(
        runner,
        _calibration_sample(),
    )

    assert metadata["tp_size"] == 8
    assert metadata["pp_size"] == 2
    assert metadata["dp_size"] == 3
    assert metadata["moe_ep_size"] == 4


def test_route_cache_path_uses_parallel_state_ranks(monkeypatch, tmp_path: Path):
    runner = SimpleNamespace(
        ps=SimpleNamespace(pp_rank=2, dp_rank=1),
    )
    monkeypatch.setattr(
        flashinfer_autotune,
        "flashinfer_autotune_cache_path",
        lambda _runner: tmp_path / "rank_tp3_pp2_dp1.json",
    )

    assert flashinfer_autotune.flashinfer_w4a8_route_cache_path(runner) == (
        tmp_path / "w4a8_route_pp2_dp1.json"
    )


def test_route_cache_path_uses_legacy_model_runner_ranks(monkeypatch, tmp_path: Path):
    runner = SimpleNamespace(pp_rank=2, dp_rank=1)
    monkeypatch.setattr(
        flashinfer_autotune,
        "flashinfer_autotune_cache_path",
        lambda _runner: tmp_path / "rank_tp3_pp2_dp1.json",
    )

    assert flashinfer_autotune.flashinfer_w4a8_route_cache_path(runner) == (
        tmp_path / "w4a8_route_pp2_dp1.json"
    )


class _CalibrationRunner:
    def __init__(self, tmp_path: Path):
        self.calls = []
        self.allocations = []
        self.sample = _calibration_sample()
        self.model_runner = SimpleNamespace(
            device="cuda",
            dtype=torch.bfloat16,
            tp_size=1,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            dp_rank=0,
            moe_ep_size=1,
            model=SimpleNamespace(modules=lambda: []),
            model_config=SimpleNamespace(
                quantization="w4afp8",
                hf_config=SimpleNamespace(),
            ),
            server_args=SimpleNamespace(
                model_path="/model",
                quantization="w4afp8",
                moe_runner_backend="flashinfer_cutlass",
                moe_a2a_backend="none",
                disable_flashinfer_autotune=False,
            ),
            tp_group=SimpleNamespace(
                rank_in_group=0,
                world_size=1,
                broadcast_object=lambda value, src=0: value,
            ),
            decode_num_tokens_per_bs=lambda: 5,
        )
        self.cache_path = tmp_path / "route-aware.json"

    def _alloc_dummy_decode_buffers(
        self,
        max_bs: int,
        *,
        num_tokens_per_bs: int = 1,
        allocate_logits_buffer: bool = True,
    ):
        self.allocations.append(
            (max_bs, num_tokens_per_bs, allocate_logits_buffer)
        )
        return SimpleNamespace(
            max_bs=max_bs,
            max_num_tokens=max_bs * num_tokens_per_bs,
        )

    def _dummy_run(
        self,
        batch_size,
        *,
        buffers,
        num_tokens_override=None,
        forward_mode_override=None,
    ):
        self.calls.append((batch_size, num_tokens_override, forward_mode_override))
        natural_num_tokens = batch_size
        if forward_mode_override != ForwardMode.EXTEND:
            natural_num_tokens *= self.model_runner.decode_num_tokens_per_bs()
        if (
            num_tokens_override is not None
            and forward_mode_override != ForwardMode.EXTEND
            and num_tokens_override != natural_num_tokens
        ):
            raise ValueError(
                "num_tokens_override may change the natural token count "
                "only for EXTEND forwards"
            )
        num_tokens = num_tokens_override or natural_num_tokens
        assert num_tokens <= buffers.max_num_tokens
        topk_ids = (
            torch.arange(num_tokens * 2, dtype=torch.int32).reshape(num_tokens, 2) % 4
        )
        observe_route(topk_ids, sample=self.sample)


def test_route_calibration_records_decode_and_prefill_shapes(
    monkeypatch, tmp_path: Path
):
    runner = _CalibrationRunner(tmp_path)
    expected = RouteAwareProfile(
        metadata={"schema": 1},
        tactics={1: (3, 101), 8192: (5, 103)},
    )
    monkeypatch.setattr(
        flashinfer_autotune,
        "_get_flashinfer_w4a8_api",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        flashinfer_autotune,
        "_find_w4a8_tuning_sample",
        lambda _mr: runner.sample,
    )
    monkeypatch.setattr(
        flashinfer_autotune,
        "_flashinfer_w4a8_profile_metadata",
        lambda _mr, _sample: expected.metadata,
    )
    monkeypatch.setattr(
        flashinfer_autotune,
        "flashinfer_w4a8_route_cache_path",
        lambda _mr: runner.cache_path,
    )
    monkeypatch.setattr(
        flashinfer_autotune,
        "_tune_flashinfer_w4a8_profile",
        lambda **_kwargs: expected,
    )

    set_active_profile(None)
    assert flashinfer_autotune.maybe_calibrate_flashinfer_w4a8(runner)
    assert runner.calls == [
        (64, None, None),
        (1, 8192, ForwardMode.EXTEND),
    ]
    assert runner.allocations == [
        (64, 5, False),
        (1, 8192, False),
    ]


def test_route_calibration_cache_hit_skips_recording(monkeypatch, tmp_path: Path):
    runner = _CalibrationRunner(tmp_path)
    metadata = {"schema": 1}
    cached = RouteAwareProfile(metadata=metadata, tactics={64: (7, 107)})
    save_profile(runner.cache_path, cached)
    monkeypatch.setattr(
        flashinfer_autotune,
        "_get_flashinfer_w4a8_api",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        flashinfer_autotune,
        "_find_w4a8_tuning_sample",
        lambda _mr: runner.sample,
    )
    monkeypatch.setattr(
        flashinfer_autotune,
        "_flashinfer_w4a8_profile_metadata",
        lambda _mr, _sample: metadata,
    )
    monkeypatch.setattr(
        flashinfer_autotune,
        "flashinfer_w4a8_route_cache_path",
        lambda _mr: runner.cache_path,
    )

    set_active_profile(None)
    assert flashinfer_autotune.maybe_calibrate_flashinfer_w4a8(runner)
    assert runner.calls == []


def test_route_calibration_missing_api_and_failure_are_nonfatal(
    monkeypatch, tmp_path: Path
):
    runner = _CalibrationRunner(tmp_path)
    monkeypatch.setattr(
        flashinfer_autotune,
        "_get_flashinfer_w4a8_api",
        lambda: None,
    )
    assert not flashinfer_autotune.maybe_calibrate_flashinfer_w4a8(runner)
    assert runner.calls == []

    monkeypatch.setattr(
        flashinfer_autotune,
        "_get_flashinfer_w4a8_api",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        flashinfer_autotune,
        "_find_w4a8_tuning_sample",
        lambda _mr: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    assert not flashinfer_autotune.maybe_calibrate_flashinfer_w4a8(runner)


def test_warmup_calibrates_before_shape_autotune(monkeypatch):
    events = []
    mr = SimpleNamespace(
        device="cuda",
        _kernel_warmed_up=False,
        pp_size=1,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
    )
    runner = SimpleNamespace(
        model_runner=mr,
        _pre_initialize_flashinfer_allreduce_workspace=lambda: None,
        _autotune_buffers=lambda: ("buffers", 64),
        _flashinfer_autotune=lambda **_kwargs: events.append("shape-autotune"),
    )
    monkeypatch.setattr(
        base_runner_module,
        "should_run_flashinfer_autotune",
        lambda _mr: True,
    )
    monkeypatch.setattr(
        base_runner_module,
        "maybe_calibrate_flashinfer_w4a8",
        lambda _runner: events.append("route-calibration"),
    )
    monkeypatch.setattr(
        base_runner_module.envs.SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP,
        "get",
        lambda: False,
    )

    BaseRunner.warmup(runner)

    assert events == ["route-calibration", "shape-autotune"]
