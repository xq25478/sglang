import sys
from types import SimpleNamespace
from unittest import mock

import torch


class _ForwardBatchProbe:
    def __init__(self):
        self.global_num_tokens_cpu = None
        self.global_num_tokens_gpu = object()
        self.num_token_non_padded = torch.tensor(9)
        self.scatter_prepared = False
        self.adjusted_round_robin_with = None

    def prepare_attn_tp_scatter_input(self, _runner):
        self.scatter_prepared = True

    def adjust_num_token_non_padded_for_dsa_cp_round_robin(self, *, server_args):
        self.adjusted_round_robin_with = server_args


def test_dsa_cp_megamoe_localizes_non_padding_token_count():
    """Catch MegaMoE consuming a CP-global TopK padding count on every rank."""
    from sglang.srt.model_executor import model_runner

    server_args = SimpleNamespace(moe_a2a_backend="megamoe")
    runner = SimpleNamespace(server_args=server_args, hisparse_coordinator=None)
    forward_batch = _ForwardBatchProbe()

    with (
        mock.patch.object(model_runner, "require_gathered_buffer", return_value=True),
        mock.patch.object(model_runner, "is_dsa_enable_prefill_cp", return_value=True),
        mock.patch.object(
            model_runner,
            "can_dsa_prefill_cp_round_robin_split",
            return_value=True,
        ),
        mock.patch.object(model_runner, "is_mla_prefill_cp_enabled", return_value=False),
    ):
        model_runner.ModelRunner._prepare_eager_forward_batch(runner, forward_batch)

    assert forward_batch.scatter_prepared
    assert forward_batch.adjusted_round_robin_with is server_args


def test_dsa_cp_megamoe_keeps_global_count_when_forward_falls_back_from_cp():
    from sglang.srt.model_executor import model_runner

    server_args = SimpleNamespace(moe_a2a_backend="megamoe")
    runner = SimpleNamespace(server_args=server_args, hisparse_coordinator=None)
    forward_batch = _ForwardBatchProbe()

    with (
        mock.patch.object(model_runner, "require_gathered_buffer", return_value=True),
        mock.patch.object(model_runner, "is_dsa_enable_prefill_cp", return_value=True),
        mock.patch.object(
            model_runner,
            "can_dsa_prefill_cp_round_robin_split",
            return_value=False,
        ),
        mock.patch.object(model_runner, "is_mla_prefill_cp_enabled", return_value=False),
    ):
        model_runner.ModelRunner._prepare_eager_forward_batch(runner, forward_batch)

    assert forward_batch.scatter_prepared
    assert forward_batch.adjusted_round_robin_with is None


def test_round_robin_cp4_localizes_nine_real_tokens_in_twelve_slots():
    from sglang.srt.model_executor.forward_batch_info import (
        compute_round_robin_local_num_token_non_padded,
    )

    counts = [
        compute_round_robin_local_num_token_non_padded(
            torch.tensor(9), num_tokens_per_dp=12, cp_rank=rank, cp_size=4
        ).item()
        for rank in range(4)
    ]

    assert counts == [3, 2, 2, 2]


def test_dflash_verify_exposes_request_token_count_for_dp_padding():
    """DSpark reuses DFlash verify metadata in the DP idle-participation path."""
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    spec_info = DFlashVerifyInput(
        draft_token=torch.zeros(5, dtype=torch.int64),
        positions=torch.arange(5, dtype=torch.int64),
        draft_token_num=5,
    )

    assert spec_info.num_tokens_per_req == 5


def test_dp_idle_padding_accepts_absent_lora_metadata():
    """Synthetic DSpark idle batches have no scheduler-provided LoRA list."""
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    probe = SimpleNamespace(
        input_ids=torch.empty(0, dtype=torch.int64),
        req_pool_indices=torch.empty(0, dtype=torch.int64),
        lora_ids=None,
        seq_lens_sum=0,
        seq_lens=torch.empty(0, dtype=torch.int64),
        seq_lens_cpu=torch.empty(0, dtype=torch.int64),
        out_cache_loc=torch.empty(0, dtype=torch.int64),
        encoder_lens=None,
        positions=torch.empty(0, dtype=torch.int64),
        mamba_track_indices=None,
        mamba_track_mask=None,
        mamba_track_seqlens=None,
        mrope_positions=None,
        extend_seq_lens=None,
        rids_int=None,
        sampling_info=None,
        bootstrap_room_ids_int=None,
        spec_info=None,
    )
    probe._pad_tensor_to_size = lambda tensor, size, value=0: torch.cat(
        [tensor, tensor.new_full((size - tensor.shape[0],), value)]
    )
    runner = SimpleNamespace(
        attn_backend=SimpleNamespace(
            get_cuda_graph_seq_len_fill_value=lambda: 0,
        )
    )

    ForwardBatch._pad_inputs_to_size(probe, runner, num_tokens=1, bs=1)

    assert probe.lora_ids is None
    assert probe.input_ids.shape == (1,)


def test_dp_idle_postprocess_accepts_hidden_only_dspark_output():
    """The draft idle forward may intentionally skip logits computation."""
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    probe = SimpleNamespace(
        _original_forward_mode=None,
        _original_batch_size=None,
        forward_mode=ForwardMode.IDLE,
        batch_size=1,
        spec_info=SimpleNamespace(),
    )
    output = SimpleNamespace(
        next_token_logits=None,
        hidden_states=torch.ones((1, 4)),
    )

    ForwardBatch.post_forward_mlp_sync_batch(probe, output)

    assert output.next_token_logits is None
    assert output.hidden_states.shape == (1, 4)


def test_sm90_megamoe_requires_dedicated_backend_symbols_and_weights():
    from sglang.srt.layers.moe import mega_moe_sm90

    backend = SimpleNamespace(
        fp8_mega_moe=lambda: None,
        mega_moe_pre_dispatch_sm90=lambda: None,
    )
    experts = SimpleNamespace(_mega_moe_sm90_fp8_weights=True)

    with (
        mock.patch.object(mega_moe_sm90, "_device_sm", 90),
        mock.patch.dict(sys.modules, {"deep_gemm": backend}),
    ):
        assert mega_moe_sm90.is_sm90_fp8_mega_moe_available(experts)
        experts._mega_moe_sm90_fp8_weights = False
        assert not mega_moe_sm90.is_sm90_fp8_mega_moe_available(experts)
        experts._mega_moe_sm90_fp8_weights = True
        del backend.fp8_mega_moe
        assert not mega_moe_sm90.is_sm90_fp8_mega_moe_available(experts)


def test_sm90_megamoe_calls_hopper_backend_contract():
    from sglang.srt.layers.moe import mega_moe_sm90

    calls = {}

    def pre_dispatch(*args, **kwargs):
        calls["pre_dispatch"] = (args, kwargs)

    def fp8_mega_moe(output, *args, **kwargs):
        calls["fp8_mega_moe"] = (args, kwargs)
        output.zero_()

    backend = SimpleNamespace(
        mega_moe_pre_dispatch_sm90=pre_dispatch,
        fp8_mega_moe=fp8_mega_moe,
    )
    experts = SimpleNamespace(
        should_fuse_routed_scaling_factor_in_topk=False,
        mega_l1_weights=object(),
        mega_l2_weights=object(),
    )
    moe = SimpleNamespace(
        experts=experts,
        routed_scaling_factor=2.5,
        config=SimpleNamespace(hidden_size=4, swiglu_limit=7.0),
    )
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    buf = SimpleNamespace(
        x=torch.empty((2, 4)),
        x_sf=torch.empty((2, 1)),
        topk_idx=torch.empty((2, 1), dtype=torch.int32),
        topk_weights=torch.empty((2, 1)),
    )

    with mock.patch.dict(sys.modules, {"deep_gemm": backend}):
        output = mega_moe_sm90.run_sm90_mega_routed(
            moe, hidden_states, topk_ids, topk_weights, buf, num_tokens=2
        )

    assert output.shape == (2, 4)
    assert calls["pre_dispatch"][1] == {
        "num_tokens": 2,
        "group_size": 128,
        "routed_scaling_factor": 2.5,
    }
    assert calls["fp8_mega_moe"][1] == {
        "recipe": (128, 128, 128),
        "activation": "swiglu",
        "activation_clamp": 7.0,
        "fast_math": True,
    }
