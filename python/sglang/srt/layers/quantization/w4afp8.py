from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.layers.moe import MoeRunner, get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        StandardDispatchOutput,
    )

ACTIVATION_SCHEMES = ["static", "dynamic"]
SUPPORTED_GROUP_SIZES = (32, 128)

logger = logging.getLogger(__name__)


def get_cutlass_w4a8_scale_pack(k: int, group_size: int) -> int:
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"Unsupported W4A8 group_size {group_size}. "
            f"Supported values are {SUPPORTED_GROUP_SIZES}."
        )
    tile_k = 128 if group_size == 32 else (512 if k % 512 == 0 else 128)
    if tile_k % group_size != 0:
        raise ValueError(
            f"W4A8 group_size {group_size} must divide Cutlass tile_k {tile_k}."
        )
    return tile_k // group_size


class W4AFp8Config(QuantizationConfig):
    """Config class for MIXED_PRECISION W4AFp8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = True,
        is_checkpoint_w4afp8_serialized: bool = True,
        linear_activation_scheme: str = "dynamic",
        moe_activation_scheme: str = "static",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: Optional[List[int]] = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.is_checkpoint_w4afp8_serialized = is_checkpoint_w4afp8_serialized
        if is_checkpoint_w4afp8_serialized:
            logger.warning("Detected w4afp8 checkpoint. Please note that")
        if moe_activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {moe_activation_scheme}")
        self.linear_activation_scheme = linear_activation_scheme
        self.moe_activation_scheme = moe_activation_scheme
        self.ignored_layers = ignored_layers or []
        self.weight_block_size = [128, 128]
        if group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(
                f"Unsupported W4A8 group_size {group_size}. "
                f"Supported values are {SUPPORTED_GROUP_SIZES}."
            )
        self.group_size = group_size

    @classmethod
    def get_name(cls) -> str:
        return "w4afp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> W4AFp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        is_checkpoint_w4afp8_serialized = "w4afp8" in quant_method
        linear_activation_scheme = "dynamic"
        moe_activation_scheme = "static"
        weight_block_size = [128, 128]
        group_size = cls.get_from_keys_or(config, ["group_size"], 128)
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            is_checkpoint_w4afp8_serialized=is_checkpoint_w4afp8_serialized,
            linear_activation_scheme=linear_activation_scheme,
            moe_activation_scheme=moe_activation_scheme,
            weight_block_size=weight_block_size,
            group_size=group_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return W4AFp8MoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


def interleave_scales(scales: torch.Tensor, scale_pack: int) -> torch.Tensor:
    """Interleave scales to match the Cutlass packed scale layout."""
    s_shape = scales.shape
    if s_shape[2] % scale_pack != 0:
        raise ValueError(
            f"Scale K groups {s_shape[2]} must be divisible by scale_pack {scale_pack}."
        )
    scales_interleaved = scales.reshape(
        s_shape[0], s_shape[1], (s_shape[2] // scale_pack), scale_pack
    )
    # Permute dimensions to interleave
    scales_interleaved = scales_interleaved.permute(0, 2, 1, 3)
    # Reshape back to original dimensions but with interleaved values
    scales_interleaved = scales_interleaved.reshape(
        s_shape[0], s_shape[2] // scale_pack, s_shape[1] * scale_pack
    )
    return scales_interleaved.contiguous()


def _interleave_w4a8_scales_with_flashinfer(
    scales: torch.Tensor, group_size: int
) -> Optional[torch.Tensor]:
    """Use FlashInfer's optimized W4A8 scale layout when it supports BF16."""
    from flashinfer.fused_moe import (
        interleave_moe_scales_for_sm90_mixed_gemm,
    )

    try:
        return interleave_moe_scales_for_sm90_mixed_gemm(
            scales, group_size
        )
    except ValueError as error:
        # FlashInfer 0.6.12 exposes this helper for MXFP4 E8M0 scales only.
        # Keep its legacy W4A8 layout until the optimized BF16 implementation
        # available in newer FlashInfer builds is installed.
        if scales.dtype != torch.uint8 and "must be uint8" in str(error):
            return None
        raise


def interleave_flashinfer_w4a8_scales(
    scales: torch.Tensor, *, k: int, group_size: int
) -> torch.Tensor:
    """Convert natural group scales to FlashInfer's SM90 INT4 layout."""
    if group_size != 128:
        raise ValueError(
            "FlashInfer SM90 W4A8 supports only group_size=128, "
            f"got group_size={group_size}."
        )
    if scales.shape[2] != k // group_size:
        raise ValueError(
            f"Expected {k // group_size} scale groups for K={k}, "
            f"got {scales.shape[2]}."
        )
    native_layout = _interleave_w4a8_scales_with_flashinfer(
        scales, group_size
    )
    if native_layout is not None:
        return native_layout
    scale_pack = 4 if k % 512 == 0 else (2 if k % 256 == 0 else 1)
    return interleave_scales(scales, scale_pack=scale_pack)


def swap_gate_up(tensor: torch.Tensor) -> torch.Tensor:
    """Convert SGLang's [gate, up] rows to FlashInfer's [up, gate] order."""
    if tensor.ndim < 2 or tensor.shape[1] % 2:
        raise ValueError(
            "gate/up tensor must have at least two dimensions and an even row count"
        )
    gate, up = tensor.chunk(2, dim=1)
    return torch.cat((up, gate), dim=1).contiguous()


def build_flashinfer_w4a8_quant_scales(
    *,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
) -> list[torch.Tensor]:
    """Build FlashInfer's eight-tensor group-scaled INT4 quant contract."""
    device = w13_scale.device
    dtype = torch.bfloat16
    a1_scale = w13_input_scale.float().reshape(-1).max()
    a2_scale = w2_input_scale.float().reshape(-1).max()
    empty = torch.empty(0, dtype=dtype, device=device)
    return [
        w13_scale,
        w2_scale,
        torch.full(
            (hidden_size,),
            1.0 / a1_scale.item(),
            dtype=dtype,
            device=device,
        ),
        torch.full(
            (intermediate_size,),
            1.0 / a2_scale.item(),
            dtype=dtype,
            device=device,
        ),
        empty,
        empty,
        torch.full(
            (num_experts,),
            a1_scale.item(),
            dtype=torch.float32,
            device=device,
        ),
        torch.full(
            (num_experts,),
            a2_scale.item(),
            dtype=torch.float32,
            device=device,
        ),
    ]


class W4AFp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: W4AFp8Config):
        self.quant_config = quant_config
        self.runner = None
        self.fuse_routed_scaling_factor_in_topk = False

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        assert "weight_loader" in extra_weight_attrs
        if hidden_size % self.quant_config.group_size != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by "
                f"group_size {self.quant_config.group_size}."
            )
        if intermediate_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                f"intermediate_size_per_partition {intermediate_size_per_partition} "
                f"must be divisible by group_size {self.quant_config.group_size}."
            )

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition * 2,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.GROUP.value}
        )
        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.quant_config.group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # Input scales
        w13_input_scale = torch.nn.Parameter(
            torch.ones((num_experts, 2), dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        # Pre-populate the strides
        device = layer.w13_weight.device

        self.a_strides1 = torch.full(
            (num_experts, 3),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides1 = torch.full(
            (num_experts, 3),
            2 * intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.a_strides2 = torch.full(
            (num_experts, 3),
            intermediate_size_per_partition,
            device=device,
            dtype=torch.int64,
        )
        self.c_strides2 = torch.full(
            (num_experts, 3),
            hidden_size,
            device=device,
            dtype=torch.int64,
        )
        self.b_strides1 = self.a_strides1
        self.s_strides13 = self.c_strides1
        self.b_strides2 = self.a_strides2
        self.s_strides2 = self.c_strides2

        self.expert_offsets = torch.empty(
            (num_experts + 1), dtype=torch.int32, device=device
        )
        self.problem_sizes1 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )
        self.problem_sizes2 = torch.empty(
            (num_experts, 3), dtype=torch.int32, device=device
        )

        return

    def process_weights_after_loading(self, layer: Module) -> None:
        dtype = torch.bfloat16
        device = layer.w2_weight.device
        runner_backend = get_moe_runner_backend()
        use_flashinfer = runner_backend.is_flashinfer_cutlass()
        use_triton = runner_backend.is_triton()
        use_humming = runner_backend.is_humming()

        # FlashInfer requires [up, gate] packed rows and its SM90 mixed-GEMM
        # weight interleave. SGLang checkpoints and the native CUTLASS kernel
        # use [gate, up].
        if use_flashinfer:
            from flashinfer.fused_moe import (
                interleave_moe_weights_for_sm90_mixed_gemm,
            )

            w13_weight = swap_gate_up(layer.w13_weight).view(torch.uint8)
            w2_weight = layer.w2_weight.view(torch.uint8)
            layer.w13_weight = Parameter(
                interleave_moe_weights_for_sm90_mixed_gemm(w13_weight, "int4"),
                requires_grad=False,
            )
            layer.w2_weight = Parameter(
                interleave_moe_weights_for_sm90_mixed_gemm(w2_weight, "int4"),
                requires_grad=False,
            )

        # Interleave w13_weight_scale (gate_up_proj).
        w13_weight_scale = layer.w13_weight_scale_inv.to(dtype)
        if use_flashinfer:
            w13_weight_scale = swap_gate_up(w13_weight_scale)
        w13_k = layer.w13_weight.shape[2] * 2
        if use_flashinfer:
            w13_weight_scale = interleave_flashinfer_w4a8_scales(
                w13_weight_scale,
                k=w13_k,
                group_size=self.quant_config.group_size,
            )
        elif use_triton or use_humming:
            # The Triton kernel indexes checkpoint-native [E, N, K/group]
            # scales directly. Humming consumes this natural layout before
            # its own destructive weight transform below.
            pass
        else:
            w13_scale_pack = get_cutlass_w4a8_scale_pack(
                w13_k, self.quant_config.group_size
            )
            w13_weight_scale = interleave_scales(
                w13_weight_scale, w13_scale_pack
            )
        layer.w13_weight_scale_inv = Parameter(w13_weight_scale, requires_grad=False)

        # Interleave w2_weight_scale (down_proj)
        w2_weight_scale = layer.w2_weight_scale_inv.to(dtype)
        w2_k = layer.w2_weight.shape[2] * 2
        if use_flashinfer:
            w2_weight_scale = interleave_flashinfer_w4a8_scales(
                w2_weight_scale,
                k=w2_k,
                group_size=self.quant_config.group_size,
            )
        elif use_triton or use_humming:
            pass
        else:
            w2_scale_pack = get_cutlass_w4a8_scale_pack(
                w2_k, self.quant_config.group_size
            )
            w2_weight_scale = interleave_scales(
                w2_weight_scale, w2_scale_pack
            )
        layer.w2_weight_scale_inv = Parameter(w2_weight_scale, requires_grad=False)

        # Process input scales
        w13_input_scale_max = layer.w13_input_scale.max().to(torch.float32).item()
        new_w13_input_scale = torch.tensor(
            [w13_input_scale_max],
            dtype=torch.float32,
            device=device,
        )
        layer.w13_input_scale = Parameter(new_w13_input_scale, requires_grad=False)

        w2_input_scale_max = layer.w2_input_scale.max().to(torch.float32).item()
        new_w2_input_scale = torch.tensor(
            [w2_input_scale_max], dtype=torch.float32, device=device
        )
        layer.w2_input_scale = Parameter(new_w2_input_scale, requires_grad=False)

        # W4AFp8 cutlass G4A8 GEMM2 outputs BF16, and DeepEP's normal combine
        # kernel is dtype-coupled to the dispatch dtype: FP8 dispatch makes
        # combine reject the BF16 expert output ('Unsupported type' in
        # intranode_combine). So dispatch must stay BF16. The per-tensor
        # dynamic quant then runs on the BF16 a_states in the MoE core.
        if hasattr(layer, 'dispatcher') and layer.dispatcher is not None:
            layer.dispatcher.set_quant_config({'dispatcher_output_dtype': 'bf16'})

        # Pre-compute whether input_scale is valid (not default 1.0)
        # 1.0 means uncalibrated → use dynamic quantization
        w13_val = layer.w13_input_scale.detach().cpu().item()
        w2_val = layer.w2_input_scale.detach().cpu().item()
        layer._has_static_input_scale = (w13_val != 1.0 and w2_val != 1.0)

        if use_flashinfer:
            self.flashinfer_quant_scales = build_flashinfer_w4a8_quant_scales(
                w13_scale=layer.w13_weight_scale_inv,
                w2_scale=layer.w2_weight_scale_inv,
                w13_input_scale=layer.w13_input_scale,
                w2_input_scale=layer.w2_input_scale,
                hidden_size=w13_k,
                intermediate_size=w2_k,
                num_experts=layer.w13_weight.shape[0],
            )
        elif use_humming:
            from sglang.srt.layers.moe.moe_runner.humming_w4a8 import (
                prepare_humming_w4a8_layer,
            )

            prepare_humming_w4a8_layer(
                layer,
                group_size=self.quant_config.group_size,
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_flashinfer_cutlass():
            # Fuse routed scaling into TopK weights instead of launching an
            # output-sized mul kernel after every FlashInfer MoE call.
            self.fuse_routed_scaling_factor_in_topk = True
            import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401

            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        elif moe_runner_backend.is_triton():
            import sglang.srt.layers.moe.moe_runner.triton  # noqa: F401

            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        elif moe_runner_backend.is_humming():
            # The experimental W4AFp8 Humming path is a focused Standard/TP
            # fused path and does not instantiate the generic MoeRunner.
            self.runner = None

    def apply(
        self,
        layer: Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        if get_moe_runner_backend().is_humming():
            return self._apply_humming(layer, dispatch_output)

        if self.runner is not None:
            if get_moe_runner_backend().is_triton():
                return self._apply_triton(layer, dispatch_output)

            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                FlashInferCutlassMoeQuantInfo,
            )

            quant_info = FlashInferCutlassMoeQuantInfo(
                quant_type="w4a8",
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                quant_scales=self.flashinfer_quant_scales,
                output_dtype=torch.bfloat16,
                moe_tp_size=layer.moe_tp_size,
                moe_tp_rank=layer.moe_tp_rank,
                moe_ep_size=layer.moe_ep_size,
                moe_ep_rank=layer.moe_ep_rank,
                apply_routed_scaling_factor=False,
            )
            return self.runner.run(dispatch_output, quant_info)

        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        # Access named fields instead of relying on the legacy three-item tuple
        # shape. Internal fused-gate paths may attach packed routing metadata to
        # the standard top-k carrier, while W4A8 only consumes weights and ids.
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        output = cutlass_w4a8_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_weights,
            topk_ids,
            self.a_strides1,
            self.b_strides1,
            self.c_strides1,
            self.a_strides2,
            self.b_strides2,
            self.c_strides2,
            self.s_strides13,
            self.s_strides2,
            self.expert_offsets,
            self.problem_sizes1,
            self.problem_sizes2,
            layer.w13_input_scale,
            layer.w2_input_scale,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,
            group_size=self.quant_config.group_size,
        )
        return StandardCombineInput(hidden_states=output)

    def _apply_humming(
        self,
        layer: Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.humming_w4a8 import (
            humming_w4a8_moe,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        topk_output = dispatch_output.topk_output
        output = humming_w4a8_moe(
            layer,
            dispatch_output.hidden_states,
            topk_output.topk_weights,
            topk_output.topk_ids,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor or 1.0,
            swiglu_limit=self.moe_runner_config.swiglu_limit,
        )
        return StandardCombineInput(hidden_states=output)

    def _apply_triton(
        self,
        layer: Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """Run packed INT4 weights with dynamic per-token-group FP8 activations."""
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        group_size = self.quant_config.group_size
        if group_size != 128:
            raise ValueError(
                "Triton W4A8 supports only group_size=128, "
                f"got group_size={group_size}."
            )
        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            w13_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            use_int4_w4a8=True,
            block_shape=[0, group_size],
        )
        return self.runner.run(dispatch_output, quant_info)

    @staticmethod
    def _reject_humming_nonstandard_dispatch() -> None:
        if get_moe_runner_backend().is_humming():
            raise NotImplementedError(
                "The experimental W4A8 Humming backend supports only "
                "Standard/TP MoE dispatch."
            )

    def apply_deepep_ll(
        self,
        layer: DeepEPMoE,
        dispatch_output: DeepEPLLDispatchOutput,
    ) -> torch.Tensor:
        self._reject_humming_nonstandard_dispatch()

        from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe_deepep_ll

        hidden_states, hidden_scales, topk_ids, _, masked_m, _ = dispatch_output

        output = cutlass_w4a8_moe_deepep_ll(
            hidden_states,
            hidden_scales,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_ids,
            masked_m,
            layer.quant_method.a_strides1,
            layer.quant_method.b_strides1,
            layer.quant_method.c_strides1,
            layer.quant_method.a_strides2,
            layer.quant_method.b_strides2,
            layer.quant_method.c_strides2,
            layer.quant_method.s_strides13,
            layer.quant_method.s_strides2,
            layer.quant_method.expert_offsets,
            layer.quant_method.problem_sizes1,
            layer.quant_method.problem_sizes2,
            # 有 input_scale 走静态, 没有走动态
            layer.w13_input_scale if getattr(layer, '_has_static_input_scale', False) else None,
            layer.w2_input_scale if getattr(layer, '_has_static_input_scale', False) else None,
            group_size=self.quant_config.group_size,
        )

        return output

    def apply_deepep_normal(
        self,
        layer: DeepEPMoE,
        dispatch_output: DeepEPNormalDispatchOutput,
    ) -> torch.Tensor:
        self._reject_humming_nonstandard_dispatch()

        from sglang.srt.layers.moe.cutlass_w4a8_moe import (
            cutlass_w4a8_moe_deepep_normal,
        )

        hidden_states, topk_idx, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_ids,
            dispatch_output.topk_weights,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        num_tokens = hidden_states.shape[0]
        if num_tokens > 0:
            return cutlass_w4a8_moe_deepep_normal(
                hidden_states,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                topk_weights,
                topk_idx,
                self.a_strides1,
                self.b_strides1,
                self.c_strides1,
                self.a_strides2,
                self.b_strides2,
                self.c_strides2,
                self.s_strides13,
                self.s_strides2,
                self.expert_offsets,
                self.problem_sizes1,
                self.problem_sizes2,
                layer.w13_input_scale,
                layer.w2_input_scale,
                group_size=self.quant_config.group_size,
            )
        else:
            return hidden_states
