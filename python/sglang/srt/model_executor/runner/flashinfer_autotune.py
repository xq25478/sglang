# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import contextlib
import datetime
import hashlib
import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.flashinfer_w4a8_autotune import (
    FLASHINFER_W4A8_TUNING_BUCKETS,
    RouteAwareProfile,
    RouteRecorder,
    aggregate_route_probabilities,
    build_topk_ids,
    load_profile,
    route_recording,
    save_profile,
    select_best_tactic,
    set_active_profile,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)

_W4A8_ROUTE_PROFILE_SCHEMA = 1
_W4A8_ROUTE_CALIBRATION_TIMEOUT_SECONDS = 180.0
_W4A8_ROUTE_DECODE_TOKENS = 64
_W4A8_ROUTE_PREFILL_TOKENS = 8192
_W4A8_ROUTE_WARMUPS = 2
_W4A8_ROUTE_SAMPLES = 5


def _get_flashinfer_w4a8_api():
    try:
        from flashinfer.fused_moe import (
            cutlass_fused_moe,
            get_cutlass_fused_moe_valid_profile_ids,
        )

        return get_cutlass_fused_moe_valid_profile_ids, cutlass_fused_moe
    except (ImportError, AttributeError):
        return None


def _is_flashinfer_w4a8_route_calibration_enabled(model_runner) -> bool:
    mr = model_runner
    server_args = mr.server_args
    quantization = (
        getattr(mr.model_config, "quantization", None)
        or getattr(server_args, "quantization", None)
        or ""
    )
    return (
        mr.device == "cuda"
        and server_args.moe_runner_backend == "flashinfer_cutlass"
        and str(quantization).lower() in ("w4afp8", "w4a8")
        and not server_args.disable_flashinfer_autotune
        and getattr(server_args, "moe_a2a_backend", None) in (None, "none")
    )


def _find_w4a8_tuning_sample(model_runner):
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        FlashInferCutlassMoeQuantInfo,
    )
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8MoEMethod

    for layer in model_runner.model.modules():
        method = getattr(layer, "quant_method", None)
        if not isinstance(method, W4AFp8MoEMethod):
            continue
        if not hasattr(method, "flashinfer_quant_scales"):
            continue
        runner_config = getattr(layer, "moe_runner_config", None) or getattr(
            method, "moe_runner_config", None
        )
        if runner_config is None:
            continue
        quant_info = FlashInferCutlassMoeQuantInfo(
            quant_type="w4a8",
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            quant_scales=method.flashinfer_quant_scales,
            output_dtype=torch.bfloat16,
            moe_tp_size=getattr(layer, "moe_tp_size", 1),
            moe_tp_rank=getattr(layer, "moe_tp_rank", 0),
            moe_ep_size=getattr(layer, "moe_ep_size", 1),
            moe_ep_rank=getattr(layer, "moe_ep_rank", 0),
            group_size=method.quant_config.group_size,
            apply_routed_scaling_factor=False,
        )
        return quant_info, runner_config
    raise RuntimeError("No loaded W4A8 FlashInfer-CUTLASS MoE layer was found")


def _flashinfer_w4a8_profile_metadata(model_runner, sample) -> dict[str, Any]:
    import flashinfer

    quant_info, runner_config = sample[:2]
    major, minor = torch.cuda.get_device_capability(model_runner.device)
    scales = quant_info.quant_scales or []
    return {
        "schema": _W4A8_ROUTE_PROFILE_SCHEMA,
        "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
        "compute_capability": f"{major}.{minor}",
        "gpu_name": torch.cuda.get_device_name(model_runner.device),
        "model_path": str(model_runner.server_args.model_path),
        "model_config_class": (model_runner.model_config.hf_config.__class__.__name__),
        "dtype": str(model_runner.dtype),
        "tp_size": int(model_runner.ps.tp_size),
        "pp_size": int(model_runner.ps.pp_size),
        "dp_size": int(model_runner.ps.attn_dp_size),
        "moe_ep_size": int(model_runner.ps.moe_ep_size),
        "w13_shape": list(quant_info.w13_weight.shape),
        "w2_shape": list(quant_info.w2_weight.shape),
        "scale_shapes": [list(scale.shape) for scale in scales],
        "top_k": int(runner_config.top_k),
        "num_experts": int(runner_config.num_experts),
        "num_local_experts": int(runner_config.num_local_experts),
        "group_size": int(quant_info.group_size),
        "activation": str(runner_config.activation),
        "is_gated": bool(runner_config.is_gated),
        "buckets": list(FLASHINFER_W4A8_TUNING_BUCKETS),
    }


def flashinfer_w4a8_route_cache_path(model_runner) -> Path:
    base_path = flashinfer_autotune_cache_path(model_runner)
    pp_rank = int(model_runner.ps.pp_rank)
    dp_rank = int(model_runner.ps.dp_rank or 0)
    return base_path.parent / f"w4a8_route_pp{pp_rank}_dp{dp_rank}.json"


def _tp_broadcast_object(model_runner, value):
    return model_runner.tp_group.broadcast_object(value, src=0)


def _tp_rank(model_runner) -> int:
    return int(getattr(model_runner.tp_group, "rank_in_group", 0))


def _measure_flashinfer_w4a8_pair(
    *,
    model_runner,
    fused_moe,
    call_kwargs: dict[str, Any],
    profile_ids: tuple[int, int],
) -> Optional[float]:
    local_ms = math.inf
    try:
        kwargs = dict(call_kwargs)
        kwargs["profile_ids"] = [int(profile_ids[0]), int(profile_ids[1])]
        for _ in range(_W4A8_ROUTE_WARMUPS):
            fused_moe(**kwargs)
        torch.cuda.synchronize(call_kwargs["input"].device)

        timings = []
        for _ in range(_W4A8_ROUTE_SAMPLES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fused_moe(**kwargs)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        local_ms = sum(timings) / len(timings)
    except Exception:
        logger.debug(
            "FlashInfer W4A8 tactic %s failed",
            profile_ids,
            exc_info=True,
        )

    timing = torch.tensor(
        [local_ms],
        dtype=torch.float64,
        device=call_kwargs["input"].device,
    )
    if getattr(model_runner.tp_group, "world_size", 1) > 1:
        torch.distributed.all_reduce(
            timing,
            op=torch.distributed.ReduceOp.MAX,
            group=model_runner.tp_group.device_group,
        )
    reduced_ms = float(timing.item())
    return reduced_ms if math.isfinite(reduced_ms) else None


def _tune_flashinfer_w4a8_profile(
    *,
    model_runner,
    recorder: RouteRecorder,
    sample,
    api,
    metadata: dict[str, Any],
    deadline: float,
) -> RouteAwareProfile:
    enumerate_profiles, fused_moe = api
    quant_info, runner_config = sample[:2]
    top_k = int(runner_config.top_k)
    hidden_size = int(runner_config.hidden_size)
    device = quant_info.w13_weight.device
    output_dtype = quant_info.output_dtype or torch.bfloat16

    probabilities = {
        phase: aggregate_route_probabilities(histograms)
        for phase, histograms in recorder.histograms.items()
    }
    tactics: dict[int, tuple[int, int]] = {}

    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        _activation_type,
    )

    activation_type = _activation_type(runner_config)
    stop_flag = torch.zeros(1, dtype=torch.int32, device=device)

    def deadline_expired() -> bool:
        stop_flag.fill_(int(time.monotonic() >= deadline))
        if getattr(model_runner.tp_group, "world_size", 1) > 1:
            torch.distributed.all_reduce(
                stop_flag,
                op=torch.distributed.ReduceOp.MAX,
                group=model_runner.tp_group.device_group,
            )
        return bool(stop_flag.item())

    for bucket in FLASHINFER_W4A8_TUNING_BUCKETS:
        if deadline_expired():
            break
        phase = "decode" if bucket <= _W4A8_ROUTE_DECODE_TOKENS else "prefill"
        x = torch.randn(
            (bucket, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        topk_ids = build_topk_ids(
            probabilities[phase],
            bucket,
            top_k,
            device=device,
        )
        topk_weights = torch.full(
            (bucket, top_k),
            1.0 / top_k,
            dtype=torch.float32,
            device=device,
        )
        output = torch.empty(
            (bucket, hidden_size),
            dtype=output_dtype,
            device=device,
        )

        enumeration = None
        if _tp_rank(model_runner) == 0:
            try:
                enumeration = enumerate_profiles(
                    x,
                    quant_info.w13_weight,
                    quant_info.w2_weight,
                    output_dtype,
                    top_k=top_k,
                    tp_size=quant_info.moe_tp_size,
                    tp_rank=quant_info.moe_tp_rank,
                    ep_size=quant_info.moe_ep_size,
                    ep_rank=quant_info.moe_ep_rank,
                    use_w4_group_scaling=True,
                    use_packed_weights=True,
                    activation_type=activation_type,
                )
            except Exception:
                logger.warning(
                    "FlashInfer W4A8 tactic enumeration failed for M=%d",
                    bucket,
                    exc_info=True,
                )
        enumeration = _tp_broadcast_object(model_runner, enumeration)
        if not enumeration:
            continue
        gemm1_ids, gemm2_ids = enumeration

        call_kwargs = {
            "output": output,
            "input": x,
            "token_selected_experts": topk_ids,
            "token_final_scales": topk_weights,
            "fc1_expert_weights": quant_info.w13_weight,
            "fc2_expert_weights": quant_info.w2_weight,
            "output_dtype": output_dtype,
            "input_sf": None,
            "quant_scales": quant_info.quant_scales,
            "ep_size": quant_info.moe_ep_size,
            "ep_rank": quant_info.moe_ep_rank,
            "tp_size": quant_info.moe_tp_size,
            "tp_rank": quant_info.moe_tp_rank,
            "tune_max_num_tokens": _W4A8_ROUTE_PREFILL_TOKENS,
            "activation_type": activation_type,
            "enable_alltoall": False,
            "use_w4_group_scaling": True,
            "use_packed_weights": True,
        }

        def measure_pair(pair):
            return _measure_flashinfer_w4a8_pair(
                model_runner=model_runner,
                fused_moe=fused_moe,
                call_kwargs=call_kwargs,
                profile_ids=pair,
            )

        gemm1 = select_best_tactic(
            gemm1_ids,
            stage=1,
            fixed_tactic=-1,
            measure_pair=measure_pair,
            should_stop=deadline_expired,
        )
        gemm2 = select_best_tactic(
            gemm2_ids,
            stage=2,
            fixed_tactic=gemm1,
            measure_pair=measure_pair,
            should_stop=deadline_expired,
        )
        if gemm1 != -1 or gemm2 != -1:
            tactics[bucket] = (gemm1, gemm2)
            logger.info(
                "FlashInfer W4A8 route profile M=%d selected [%d, %d]",
                bucket,
                gemm1,
                gemm2,
            )
        del x, topk_ids, topk_weights, output

    if not tactics:
        raise RuntimeError("No FlashInfer W4A8 route-aware tactics were measured")
    return RouteAwareProfile(metadata=metadata, tactics=tactics)


def maybe_calibrate_flashinfer_w4a8(runner) -> bool:
    """Load or create the W4A8 route profile without blocking service startup."""
    mr = runner.model_runner
    if not _is_flashinfer_w4a8_route_calibration_enabled(mr):
        return False

    api = _get_flashinfer_w4a8_api()
    if api is None:
        logger.warning(
            "FlashInfer does not expose W4A8 valid-profile enumeration; "
            "using shape-only autotune"
        )
        return False

    try:
        sample = _find_w4a8_tuning_sample(mr)
        metadata = _flashinfer_w4a8_profile_metadata(mr, sample)
        cache_path = flashinfer_w4a8_route_cache_path(mr)

        cached = None
        if _tp_rank(mr) == 0:
            cached = load_profile(cache_path, expected_metadata=metadata)
        cached = _tp_broadcast_object(mr, cached)
        if cached is not None:
            set_active_profile(cached)
            logger.info(
                "Loaded FlashInfer W4A8 route profile from %s",
                cache_path,
            )
            return True

        deadline = (
            time.monotonic() + _W4A8_ROUTE_CALIBRATION_TIMEOUT_SECONDS
            if _tp_rank(mr) == 0
            else None
        )
        deadline = float(_tp_broadcast_object(mr, deadline))
        recorder = RouteRecorder(num_experts=int(sample[1].num_experts))
        from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

        decode_buffers = runner._alloc_dummy_decode_buffers(
            max_bs=_W4A8_ROUTE_DECODE_TOKENS,
            allocate_logits_buffer=False,
        )
        with route_recording(recorder, "decode"), autotune_dummy_run_mode():
            runner._dummy_run(
                batch_size=_W4A8_ROUTE_DECODE_TOKENS,
                buffers=decode_buffers,
                num_tokens_override=_W4A8_ROUTE_DECODE_TOKENS,
                forward_mode_override=ForwardMode.DECODE,
            )
        del decode_buffers

        prefill_buffers = runner._alloc_dummy_decode_buffers(
            max_bs=1,
            num_tokens_per_bs=_W4A8_ROUTE_PREFILL_TOKENS,
            allocate_logits_buffer=False,
        )
        with route_recording(recorder, "prefill"), autotune_dummy_run_mode():
            runner._dummy_run(
                batch_size=1,
                buffers=prefill_buffers,
                num_tokens_override=_W4A8_ROUTE_PREFILL_TOKENS,
                forward_mode_override=ForwardMode.EXTEND,
            )
        del prefill_buffers

        profile = _tune_flashinfer_w4a8_profile(
            model_runner=mr,
            recorder=recorder,
            sample=sample,
            api=api,
            metadata=metadata,
            deadline=deadline,
        )
        if _tp_rank(mr) == 0:
            save_profile(cache_path, profile)
        profile = _tp_broadcast_object(mr, profile)
        set_active_profile(profile)
        logger.info(
            "Saved FlashInfer W4A8 route profile with %d buckets to %s",
            len(profile.tactics),
            cache_path,
        )
        return True
    except Exception:
        set_active_profile(None)
        logger.warning(
            "FlashInfer W4A8 route calibration failed; using shape-only autotune",
            exc_info=True,
        )
        return False


def should_run_flashinfer_autotune(
    model_runner: ModelRunner, *, for_speculative_draft: bool = False
) -> bool:
    """Check if flashinfer autotune should be run."""
    mr = model_runner
    if mr.device != "cuda":
        return False
    if mr.server_args.disable_flashinfer_autotune:
        return False

    # CuteDSL v1 (cutedsl runner + deepep a2a) bypasses MoeRunner and must not
    # be autotuned -- its _dummy_run would dispatch more tokens per rank than
    # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, tripping a DeepEP assert.
    # Read server_args directly to avoid depending on initialize_moe_config()
    # having already populated the MoE backend globals.
    if (
        mr.server_args.moe_runner_backend == "flashinfer_cutedsl"
        and mr.server_args.moe_a2a_backend == "deepep"
    ):
        return False

    backend_str = mr.server_args.moe_runner_backend

    # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

    moe_needs_autotune = backend_str in [
        "flashinfer_trtllm",
        "flashinfer_trtllm_routed",
        "flashinfer_mxfp4",
        "flashinfer_cutedsl",
        "flashinfer_cutlass",
    ]

    from sglang.srt.layers.quantization.fp4_utils import (
        get_fp4_gemm_runner_backend,
    )

    model_quantization = mr.model_config.quantization
    model_uses_fp4 = model_quantization in (
        "modelopt_fp4",
        "modelopt_mixed",
    )
    fp4_gemm_needs_autotune = model_uses_fp4 and (
        get_fp4_gemm_runner_backend().is_flashinfer_cutlass()
        or get_fp4_gemm_runner_backend().is_flashinfer_cutedsl()
    )

    from sglang.srt.layers.quantization.fp8_utils import (
        get_fp8_gemm_runner_backend,
    )
    from sglang.srt.utils import is_sm100_supported

    model_uses_modelopt_fp8 = model_quantization in (
        "modelopt",
        "modelopt_fp8",
        "modelopt_mixed",
    )
    # Online MXFP8 (microscaling) linears dispatch to flashinfer's
    # ``mm_mxfp8``, which the flashinfer fp8 autotune dummy run does not
    # exercise correctly -- it triggers an illegal memory access inside the
    # mxfp8 cutlass cubin. The mxfp8 gemm is fixed-config and needs no
    # tuning, so skip autotune for these models.
    model_uses_mxfp8 = "mxfp8" in (model_quantization or "")
    fp8_gemm_needs_autotune = not model_uses_mxfp8 and (
        get_fp8_gemm_runner_backend().is_flashinfer_cutlass()
        or (model_uses_modelopt_fp8 and is_sm100_supported())
    )

    if not (moe_needs_autotune or fp4_gemm_needs_autotune or fp8_gemm_needs_autotune):
        return False

    if torch.cuda.get_device_capability()[0] < 9:
        return False

    if mr.spec_algorithm.is_speculative():
        return mr.is_draft_worker if for_speculative_draft else not mr.is_draft_worker

    return True


def flashinfer_autotune_cache_path(model_runner: ModelRunner) -> Path:
    import flashinfer

    mr = model_runner
    major, minor = torch.cuda.get_device_capability(mr.device)
    arch = f"sm{major}{minor}"
    flashinfer_version = getattr(flashinfer, "__version__", "unknown")

    server_args = mr.server_args
    model_key_parts = [
        str(server_args.model_path),
        str(mr.dtype),
        str(server_args.quantization),
        str(server_args.moe_runner_backend),
        str(mr.tp_size),
        str(mr.pp_size),
        str(mr.dp_size),
        str(mr.moe_ep_size),
        str(mr.model_config.hf_config.__class__.__name__),
    ]
    if mr.is_draft_worker:
        model_key_parts.append(f"draft_quant={mr.model_config.quantization}")
    model_key = "|".join(model_key_parts)
    cache_key = hashlib.sha256(model_key.encode()).hexdigest()[:16]
    cache_dir = (
        Path(envs.SGLANG_CACHE_DIR.get())
        / "flashinfer"
        / "autotune"
        / flashinfer_version
        / arch
        / cache_key
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"rank_tp{mr.tp_rank}_pp{mr.pp_rank}_dp{mr.dp_rank or 0}.json"


@contextlib.contextmanager
def flashinfer_autotune_context(model_runner: ModelRunner, *, skip_logits: bool):
    from flashinfer.autotuner import autotune

    mr = model_runner
    cache_path = flashinfer_autotune_cache_path(mr)
    if envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get():
        autotune_cache = cache_path
        logger.info("Running FlashInfer autotune with cache: %s", autotune_cache)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_dir = cache_path.parent / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        autotune_cache = runs_dir / f"{cache_path.stem}.{timestamp}{cache_path.suffix}"
        logger.info(
            "Running FlashInfer autotune (cache reuse DISABLED via "
            "SGLANG_FLASHINFER_AUTOTUNE_CACHE=0); writing fresh result to: %s",
            autotune_cache,
        )

    # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
    # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
    mr.forward_stream.wait_stream(torch.cuda.current_stream())
    with torch.get_device_module(mr.device).stream(mr.forward_stream):
        maybe_skip_logits = contextlib.nullcontext()
        if skip_logits:
            from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

            maybe_skip_logits = autotune_dummy_run_mode()
        with torch.inference_mode(), autotune(
            True, cache=str(autotune_cache)
        ), maybe_skip_logits:
            yield
    torch.cuda.current_stream().wait_stream(mr.forward_stream)
    logger.info("FlashInfer autotune completed.")


def run_flashinfer_autotune_forward(
    model_runner: ModelRunner, forward_fn: Callable[[], None], *, skip_logits: bool
) -> None:
    """Run flashinfer autotune forward."""
    with flashinfer_autotune_context(model_runner, skip_logits=skip_logits):
        forward_fn()


def maybe_flashinfer_autotune_speculative_draft(
    runner: BaseRunner,
    forward_fn: Callable[[], None],
    *,
    post_warmup_hook: Optional[Callable[[], None]] = None,
    skip_logits: bool = False,
) -> None:
    """Run speculative draft flashinfer autotune."""
    mr = runner.model_runner
    phase_key = f"{runner.__class__.__module__}.{runner.__class__.__qualname__}"
    tuned_phases = getattr(mr, "_flashinfer_spec_draft_autotuned_phases", None)
    if tuned_phases is None:
        tuned_phases = set()
        mr._flashinfer_spec_draft_autotuned_phases = tuned_phases
    if phase_key in tuned_phases:
        return
    if (
        not mr.spec_algorithm.is_speculative()
        or not mr.is_draft_worker
        or not should_run_flashinfer_autotune(mr, for_speculative_draft=True)
    ):
        return

    def run_and_reset():
        forward_fn()
        if post_warmup_hook is not None:
            post_warmup_hook()

    run_flashinfer_autotune_forward(mr, run_and_reset, skip_logits=skip_logits)
    tuned_phases.add(phase_key)
