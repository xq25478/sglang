"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""

from __future__ import annotations

import hashlib
import logging
import time
from array import array
from collections import deque
from http import HTTPStatus
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.utils import DSparkHiddenReleaseGuard
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVClassType,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    all_reduce_attn_cp_tp_keyed_values,
    append_draft_kv_data,
    get_disagg_poll_cpu_groups,
    get_dsv4_c128_state_indices,
    get_kv_class,
    is_aborted,
    is_dsv4_c128_online_enabled,
    is_mla_backend,
    poll_and_all_reduce_attn_cp_tp_group,
    prepare_abort,
    setup_state_kv_args,
    should_transfer_draft_cache,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    Req,
    ScheduleBatch,
)
from sglang.srt.mem_cache.common import (
    kv_to_page_indices,
    kv_to_page_num,
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.mem_cache.cp_cache_layer_split import is_cp_cache_layer_split_pool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    if retry_prob <= 0 or req.time_stats.prefill_retry_count > 0 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64


def maybe_release_metadata_buffer(
    req: Req,
    allocator: ReqToMetadataIdxAllocator,
    dspark_hidden_pool=None,
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if req.metadata_buffer_index >= 0:
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1
    indices = getattr(req, "dspark_hidden_src_indices", None)
    if indices and dspark_hidden_pool is not None:
        guard = getattr(req, "dspark_hidden_release_guard", None)
        if guard is not None:
            if guard.claim_scheduler():
                dspark_hidden_pool.free(indices)
        else:
            sender = getattr(req, "disagg_kv_sender", None)
            kv_mgr = getattr(sender, "kv_mgr", None)
            pop_hidden_done = getattr(kv_mgr, "pop_dspark_hidden_done", None)
            worker_released = pop_hidden_done is not None and pop_hidden_done(
                getattr(sender, "bootstrap_room", req.bootstrap_room)
            )
            if not worker_released:
                dspark_hidden_pool.free(indices)
        req.dspark_hidden_src_indices = None
        req.dspark_hidden_written = None
        req.dspark_hidden_release_guard = None
    elif not indices:
        req.dspark_hidden_src_indices = None
        req.dspark_hidden_release_guard = None
    req.dspark_hidden_capture_layer_ids = None


def maybe_release_dspark_hidden_rows(req: Req, dspark_hidden_pool) -> None:
    """Release source hidden rows once the local RDMA transfer is complete."""
    if dspark_hidden_pool is None:
        return
    indices = getattr(req, "dspark_hidden_src_indices", None)
    if indices:
        guard = getattr(req, "dspark_hidden_release_guard", None)
        if guard is None or guard.claim_scheduler():
            dspark_hidden_pool.free(indices)
        req.dspark_hidden_src_indices = None
        req.dspark_hidden_written = None
        req.dspark_hidden_release_guard = None


def maybe_release_dspark_hidden_rows_on_hidden_done(
    req: Req, dspark_hidden_pool
) -> bool:
    """Release source hidden rows after DSPARK_HIDDEN finishes, before KV success."""
    indices = getattr(req, "dspark_hidden_src_indices", None)
    if not indices or dspark_hidden_pool is None:
        return False
    guard = getattr(req, "dspark_hidden_release_guard", None)
    if guard is not None:
        if not guard.worker_finished():
            return False
        req.dspark_hidden_src_indices = None
        req.dspark_hidden_written = None
        req.dspark_hidden_release_guard = None
        return True

    sender = getattr(req, "disagg_kv_sender", None)
    kv_mgr = getattr(sender, "kv_mgr", None)
    pop_hidden_done = getattr(kv_mgr, "pop_dspark_hidden_done", None)
    if pop_hidden_done is None or not pop_hidden_done(
        getattr(sender, "bootstrap_room", req.bootstrap_room)
    ):
        return False

    req.dspark_hidden_src_indices = None
    req.dspark_hidden_written = None
    return True


def dspark_hidden_capture_intersects_extend(req: Req) -> bool:
    """Whether the current prefill chunk contains required DSpark hidden rows."""
    if not getattr(req, "dspark_hidden_src_indices", None) and not getattr(
        req, "dspark_hidden_capture_layer_ids", None
    ):
        return False

    meta = getattr(req, "dspark_hidden_meta", None) or {}
    hidden_start = int(
        getattr(req, "dspark_prefill_recompute_start", None)
        if getattr(req, "dspark_prefill_recompute_start", None) is not None
        else meta.get("hidden_start", 0)
    )
    origin_input_ids = getattr(req, "origin_input_ids", None)
    if origin_input_ids is not None:
        hidden_end = len(origin_input_ids)
    else:
        hidden_end = hidden_start + int(
            meta.get("hidden_len", len(getattr(req, "dspark_hidden_src_indices", [])))
        )
    extend_range = getattr(req, "extend_range", None)
    if extend_range is None:
        # Conservatively capture when scheduling metadata is unavailable.
        return True
    return int(extend_range.start) < hidden_end and int(extend_range.end) > hidden_start


def get_dspark_hidden_payload_error(req: Req) -> Optional[str]:
    """Return a request-scoped error when DSpark hidden rows are not sendable."""
    src_indices = getattr(req, "dspark_hidden_src_indices", None)
    capture_layer_ids = getattr(req, "dspark_hidden_capture_layer_ids", None)
    sender = getattr(req, "disagg_kv_sender", None)
    kv_mgr = getattr(sender, "kv_mgr", None)
    if getattr(kv_mgr, "is_dummy_cp_rank", False):
        # Dummy CP ranks run the same hidden capture so attention collectives
        # have identical shapes, but only the authoritative CP rank owns and
        # transfers hidden rows.
        return None
    if src_indices is None and capture_layer_ids:
        return (
            "DSpark hidden row pool was not materialized before PD transfer: "
            f"rid={req.rid}"
        )
    if not src_indices:
        return None

    written = getattr(req, "dspark_hidden_written", None)
    if written is not None:
        if len(written) != len(src_indices):
            return (
                "DSpark hidden row bookkeeping length mismatch before PD transfer: "
                f"rid={req.rid}, written={len(written)}, rows={len(src_indices)}"
            )
        if not all(written):
            missing = [i for i, ok in enumerate(written) if not ok][:8]
            return (
                "DSpark hidden rows are incomplete before PD transfer: "
                f"rid={req.rid}, missing_offsets={missing}"
            )
    return None


class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
        scheduler: Scheduler,
        pp_rank: int,
        pp_size: int,
        transfer_backend: TransferBackend,
    ):
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.is_mla_backend = is_mla_backend(token_to_kv_pool)
        self.metadata_buffers = metadata_buffers
        self.req_to_metadata_buffer_idx_allocator = req_to_metadata_buffer_idx_allocator
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_id = gpu_id
        self.bootstrap_port = bootstrap_port
        self.queue: List[Req] = []
        self.gloo_group = gloo_group
        self.scheduler = scheduler
        self.max_total_num_tokens = (
            self.scheduler.tp_worker.model_runner.max_token_pool_size
        )
        self._last_dspark_hidden_credit_warning_time = 0.0
        self.transfer_backend = transfer_backend
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get() and self.is_mla_backend:
            raise RuntimeError(
                "SGLANG_DISAGG_STAGING_BUFFER is designed for non-MLA models "
                "(e.g. GQA, MHA). MLA models should not set this flag."
            )
        self.kv_manager = self._init_kv_manager()

        if self.scheduler.tp_worker.is_hybrid_swa:
            chunked_prefill_size = self.scheduler.chunked_prefill_size
            if chunked_prefill_size is not None and chunked_prefill_size > 0:
                self.max_total_num_tokens = min(
                    self.max_total_num_tokens,
                    self.scheduler.tp_worker.model_runner.full_max_total_num_tokens,
                )
            else:
                # Without chunked prefill, a single forward needs all SWA KV live;
                # keep the conservative SWA cap to avoid OOM in the forward path.
                self.max_total_num_tokens = min(
                    self.max_total_num_tokens,
                    self.scheduler.tp_worker.model_runner.swa_max_total_num_tokens,
                )

    def _init_kv_manager(self) -> CommonKVManager:
        kv_args_class = get_kv_class(self.transfer_backend, KVClassType.KVARGS)
        kv_args = kv_args_class()
        kv_args.engine_rank = self.tp_rank
        kv_args.pp_rank = self.pp_rank
        kv_args.system_dp_rank = self.scheduler.ps.dp_rank
        # layer_shard_enabled is the DSA contiguous-range contract used
        # to derive prefill_start/end_layer. It is not the generic LayerSplit flag.
        layer_shard_enabled = getattr(
            self.token_to_kv_pool, "layer_shard_enabled", False
        )
        transfer_draft_cache = should_transfer_draft_cache(self.token_to_kv_pool)
        kv_args.prefill_start_layer = (
            getattr(
                self.token_to_kv_pool,
                "layer_shard_start",
                self.token_to_kv_pool.start_layer,
            )
            if layer_shard_enabled
            else self.token_to_kv_pool.start_layer
        )
        kv_args.mla_compression_ratios = None
        kv_data_ptrs, kv_data_lens, kv_item_lens = (
            self.token_to_kv_pool.get_contiguous_buf_infos()
        )
        kv_args.prefill_end_layer = (
            kv_args.prefill_start_layer + len(kv_data_ptrs)
            if layer_shard_enabled
            else getattr(self.token_to_kv_pool, "end_layer", None)
        )
        kv_data_layout = (
            self.token_to_kv_pool.get_kv_transfer_layout()
            if hasattr(self.token_to_kv_pool, "get_kv_transfer_layout")
            else []
        )

        if self.draft_token_to_kv_pool is not None and transfer_draft_cache:
            # We should also transfer draft model kv cache. The indices are
            # always shared with a target model.
            append_draft_kv_data(
                kv_data_ptrs,
                kv_data_lens,
                kv_item_lens,
                kv_data_layout,
                self.draft_token_to_kv_pool,
            )

        kv_args.kv_data_ptrs = kv_data_ptrs
        kv_args.kv_data_lens = kv_data_lens
        kv_args.kv_item_lens = kv_item_lens
        kv_args.kv_data_layout = kv_data_layout

        # Generic PD wire metadata: decode must account for cache layers split
        # across prefill CP ranks, regardless of whether ownership is contiguous.
        kv_args.cp_cache_layer_split = is_cp_cache_layer_split_pool(
            self.token_to_kv_pool
        )
        kv_args.require_descriptor_matched_transfer = bool(
            kv_args.cp_cache_layer_split
            and self.token_to_kv_pool.requires_descriptor_matched_transfer
        )
        if not self.is_mla_backend:
            kv_args.kv_head_num = self.token_to_kv_pool.head_num
            kv_args.total_kv_head_num = (
                self.scheduler.model_config.get_total_num_kv_heads()
            )
        kv_args.page_size = self.token_to_kv_pool.page_size

        kv_args.aux_data_ptrs, kv_args.aux_data_lens, kv_args.aux_item_lens = (
            self.metadata_buffers.get_buf_infos()
        )
        kv_args.ib_device = self.scheduler.server_args.disaggregation_ib_device
        kv_args.gpu_id = self.scheduler.ps.gpu_id

        req_to_token_pool = getattr(self.scheduler, "req_to_token_pool", None)
        setup_state_kv_args(
            kv_args,
            self.token_to_kv_pool,
            self.draft_token_to_kv_pool if transfer_draft_cache else None,
            self.scheduler.model_config.num_hidden_layers,
            req_to_token_pool=req_to_token_pool,
            dspark_hidden_pool=getattr(
                self.metadata_buffers, "dspark_hidden_pool", None
            ),
        )

        if isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool):
            # V4's KVCache is organized by compression-ratio
            # buckets rather than by layer.
            kv_args.mla_compression_ratios = list(
                self.token_to_kv_pool.compression_ratios
            )

        kv_manager_class = get_kv_class(self.transfer_backend, KVClassType.MANAGER)
        kv_manager = kv_manager_class(
            kv_args,
            DisaggregationMode.PREFILL,
            self.scheduler.server_args,
            self.is_mla_backend,
        )
        kv_manager.dspark_hidden_pool = getattr(
            self.metadata_buffers, "dspark_hidden_pool", None
        )
        # Pass KV pool tensor refs to the manager for GPU gather (staging mode)
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and hasattr(kv_manager, "set_kv_buffer_tensors")
            and not self.is_mla_backend
        ):
            kv_pool = self.token_to_kv_pool
            if hasattr(kv_pool, "full_kv_pool"):
                kv_pool = kv_pool.full_kv_pool
            if hasattr(kv_pool, "k_buffer") and hasattr(kv_pool, "v_buffer"):
                kv_manager.set_kv_buffer_tensors(
                    kv_pool.k_buffer,
                    kv_pool.v_buffer,
                    kv_pool.page_size,
                )
        return kv_manager

    def create_sender(self, req: Req, num_kv_heads: int) -> bool:
        """Create a KV sender for the request without enqueuing it.
        Returns False if the request exceeds KV capacity."""
        if self._check_if_req_exceed_kv_capacity(req):
            return False

        backend = (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )
        kv_sender_class = get_kv_class(backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
        )
        self._process_req(req)
        req.pending_bootstrap = True
        return True

    def ensure_metadata_buffer(self, req: Req) -> bool:
        if req.metadata_buffer_index >= 0:
            return True

        if self.req_to_metadata_buffer_idx_allocator.available_size() == 0:
            return False
        req.metadata_buffer_index = self.req_to_metadata_buffer_idx_allocator.alloc()
        if req.metadata_buffer_index is None:
            req.metadata_buffer_index = -1
            logger.error(
                "Metadata buffer allocator reported capacity but returned no slot: "
                "rid=%s bootstrap_room=%s",
                req.rid,
                req.bootstrap_room,
            )
            return False
        return True

    def _requires_dspark_hidden_transfer(self, req: Req) -> bool:
        return bool(self.kv_manager.req_to_dspark_hidden_meta.get(req.bootstrap_room))

    @staticmethod
    def _validate_dspark_hidden_range(
        req: Req, dspark_meta: dict, decode_prefix_len: int
    ) -> Optional[str]:
        prompt_len = len(req.origin_input_ids)
        decode_prefix_len = int(decode_prefix_len)
        hidden_start = int(dspark_meta.get("hidden_start", 0))
        hidden_len = int(dspark_meta.get("hidden_len", prompt_len))
        draft_context_window = dspark_meta.get("draft_context_window")

        if decode_prefix_len < 0 or decode_prefix_len > prompt_len:
            return (
                "Invalid DSpark decode prefix length: "
                f"decode_prefix_len={decode_prefix_len}, prompt_len={prompt_len}, "
                f"rid={req.rid}"
            )

        if draft_context_window is None:
            expected_start = decode_prefix_len
        else:
            draft_context_window = int(draft_context_window)
            if draft_context_window <= 0:
                return (
                    "Invalid DSpark draft context window: "
                    f"draft_context_window={draft_context_window}, rid={req.rid}"
                )
            expected_start = max(decode_prefix_len, prompt_len - draft_context_window)
        expected_len = prompt_len - expected_start

        if hidden_start != expected_start or hidden_len != expected_len:
            return (
                "DSpark hidden metadata does not cover the required draft suffix: "
                f"hidden_start={hidden_start}, hidden_len={hidden_len}, "
                f"expected_start={expected_start}, expected_len={expected_len}, "
                f"decode_prefix_len={decode_prefix_len}, prompt_len={prompt_len}, "
                f"draft_context_window={draft_context_window}, rid={req.rid}"
            )
        return None

    def finalize_bootstrap(self, req: Req) -> bool:
        """Initialize the sender after bootstrap completes.
        Returns False if no metadata buffer is available (non-terminal)."""
        if not req.pending_bootstrap:
            return True
        metadata_buffer_was_unallocated = req.metadata_buffer_index < 0
        if not self.ensure_metadata_buffer(req):
            return False

        decode_prefix_len = getattr(req, "disagg_decode_prefix_len", None)
        if decode_prefix_len is None:
            decode_prefix_len = req.disagg_kv_sender.pop_decode_prefix_len()
            req.disagg_decode_prefix_len = decode_prefix_len
        dspark_meta = self.kv_manager.req_to_dspark_hidden_meta.get(req.bootstrap_room)
        if dspark_meta and not self._finalize_dspark_hidden_bootstrap(
            req, dspark_meta, decode_prefix_len
        ):
            if metadata_buffer_was_unallocated and req.metadata_buffer_index >= 0:
                self.req_to_metadata_buffer_idx_allocator.free(
                    req.metadata_buffer_index
                )
                req.metadata_buffer_index = -1
            return False

        req.time_stats.set_bootstrap_done_time()
        num_kv_indices = len(req.origin_input_ids)
        req.start_send_idx = decode_prefix_len
        num_kv_indices_to_send = num_kv_indices - decode_prefix_len
        num_pages = kv_to_page_num(
            num_kv_indices_to_send, self.token_to_kv_pool.page_size
        )
        req.disagg_kv_sender.init(num_pages, req.metadata_buffer_index)
        req.pending_bootstrap = False
        return True

    def _probe_bootstrap_ready(
        self,
        req: Req,
        metadata_credits: int,
        hidden_row_credits: int,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        """Validate metadata readiness without reserving transfer resources."""
        metadata_cost = 1 if req.metadata_buffer_index < 0 else 0
        if metadata_cost > metadata_credits:
            return None, None

        dspark_meta = self.kv_manager.req_to_dspark_hidden_meta.get(req.bootstrap_room)
        if not dspark_meta:
            return (metadata_cost, 0), None

        decode_prefix_len = getattr(req, "disagg_decode_prefix_len", None)
        if decode_prefix_len is None:
            decode_prefix_len = self.kv_manager.req_to_decode_prefix_len.get(
                req.bootstrap_room
            )
        if decode_prefix_len is None:
            return None, None

        range_error = self._validate_dspark_hidden_range(
            req, dspark_meta, decode_prefix_len
        )
        if range_error is not None:
            return None, range_error

        hidden_start = int(dspark_meta.get("hidden_start", 0))

        pp_slices = dspark_meta.get("pp_slices") or {}
        local_pp_slice = pp_slices.get(str(self.pp_rank)) if pp_slices else None
        if pp_slices and local_pp_slice is None:
            return None, (
                "DSpark hidden metadata is missing PP slice for prefill rank: "
                f"pp_rank={self.pp_rank}, "
                f"available_pp_slices={sorted(pp_slices.keys())}"
            )
        local_layer_ids = (
            [int(x) for x in local_pp_slice.get("layer_ids", [])]
            if local_pp_slice
            else (
                []
                if pp_slices
                else [int(x) for x in dspark_meta.get("target_layer_ids", [])]
            )
        )
        if not local_layer_ids:
            return (metadata_cost, 0), None

        local_slice_len = (
            int(local_pp_slice.get("slice_len", 0))
            if local_pp_slice
            else len(local_layer_ids) * int(self.scheduler.model_config.hidden_size)
        )
        expected_hidden_size = len(local_layer_ids) * int(
            self.scheduler.model_config.hidden_size
        )
        if local_slice_len != expected_hidden_size:
            return None, (
                "DSpark hidden size mismatch on prefill: "
                f"layers={local_layer_ids}, expected={expected_hidden_size}, "
                f"metadata={local_slice_len}, pp_rank={self.pp_rank}"
            )
        model_runner = self.scheduler.tp_worker.model_runner
        spec_aux_config = getattr(model_runner, "spec_aux_config", None)
        configured_layer_ids = getattr(spec_aux_config, "dflash_target_layer_ids", None)
        all_target_layer_ids = [
            int(x) for x in dspark_meta.get("target_layer_ids", local_layer_ids)
        ]
        if (
            configured_layer_ids is not None
            and list(configured_layer_ids) != all_target_layer_ids
        ):
            return None, (
                "DSpark target layer mismatch between prefill config and decode "
                f"metadata: prefill={configured_layer_ids}, "
                f"decode={all_target_layer_ids}"
            )

        hidden_len = int(dspark_meta.get("hidden_len", len(req.origin_input_ids)))
        dst_indices = [
            int(x)
            for x in (
                local_pp_slice.get("dst_indices", [])
                if local_pp_slice
                else dspark_meta.get("dst_indices", [])
            )
        ]
        if (
            hidden_len != len(dst_indices)
            or hidden_start < 0
            or hidden_len < 0
            or hidden_start + hidden_len > len(req.origin_input_ids)
        ):
            return None, (
                "Invalid DSpark hidden metadata from decode: "
                f"hidden_start={hidden_start}, hidden_len={hidden_len}, "
                f"dst_indices={len(dst_indices)}, "
                f"prompt_len={len(req.origin_input_ids)}, pp_rank={self.pp_rank}"
            )

        pool = getattr(self.metadata_buffers, "dspark_hidden_pool", None)
        if pool is None:
            return None, (
                "DSpark hidden metadata targets a prefill PP rank without a "
                f"hidden row pool: pp_rank={self.pp_rank}, "
                f"local_layer_ids={local_layer_ids}"
            )
        if hidden_len > pool.size:
            return None, (
                "DSpark hidden rows exceed prefill hidden pool capacity: "
                f"rid={req.rid}, hidden_len={hidden_len}, pool_size={pool.size}"
            )

        hidden_cost = (
            0
            if getattr(req, "dspark_hidden_src_indices", None) is not None
            else hidden_len
        )
        if hidden_cost > hidden_row_credits:
            now = time.monotonic()
            if now - self._last_dspark_hidden_credit_warning_time > 30:
                logger.warning(
                    "DSpark hidden pool blocked prefill bootstrap: "
                    "rid=%s hidden_len=%d required_rows=%d free_rows=%d "
                    "pool_rows=%d bootstrap_queue=%d",
                    req.rid,
                    hidden_len,
                    hidden_cost,
                    hidden_row_credits,
                    pool.size,
                    len(self.queue),
                )
                self._last_dspark_hidden_credit_warning_time = now
            return None, None

        return (metadata_cost, hidden_cost), None

    def _is_dspark_hidden_credit_blocked(
        self, req: Req, metadata_credits: int, hidden_row_credits: int
    ) -> bool:
        metadata_cost = 1 if req.metadata_buffer_index < 0 else 0
        if metadata_cost > metadata_credits:
            return False
        dspark_meta = self.kv_manager.req_to_dspark_hidden_meta.get(req.bootstrap_room)
        if not dspark_meta:
            return False

        pp_slices = dspark_meta.get("pp_slices") or {}
        local_pp_slice = pp_slices.get(str(self.pp_rank)) if pp_slices else None
        local_layer_ids = (
            [int(x) for x in local_pp_slice.get("layer_ids", [])]
            if local_pp_slice
            else (
                []
                if pp_slices
                else [int(x) for x in dspark_meta.get("target_layer_ids", [])]
            )
        )
        if not local_layer_ids or getattr(req, "dspark_hidden_src_indices", None):
            return False

        pool = getattr(self.metadata_buffers, "dspark_hidden_pool", None)
        if pool is None:
            return False
        hidden_len = int(dspark_meta.get("hidden_len", len(req.origin_input_ids)))
        return hidden_len <= pool.size and hidden_len > hidden_row_credits

    def stage_pp_bootstrap_consensus(self, rids: List[str]) -> List[str]:
        """Enter the resource-commit phase after metadata consensus."""
        rid_set = set(rids)
        committed = []
        for req in self.queue:
            if req.rid not in rid_set:
                continue
            req.dspark_pp_bootstrap_consensus = True
            if req.pending_bootstrap and not should_force_retry(req):
                if self.finalize_bootstrap(req):
                    committed.append(req.rid)
        return committed

    def _abort_dspark_hidden_bootstrap(self, req: Req, message: str) -> None:
        logger.error(message)
        prepare_abort(req, message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        sender = getattr(req, "disagg_kv_sender", None)
        kv_mgr = getattr(sender, "kv_mgr", None)
        if sender is not None and kv_mgr is not None:
            kv_mgr.record_failure(
                getattr(sender, "bootstrap_room", req.bootstrap_room), message
            )
            kv_mgr.update_status(
                getattr(sender, "bootstrap_room", req.bootstrap_room), KVPoll.Failed
            )
            sender.conclude_state = KVPoll.Failed

    def _finalize_dspark_hidden_bootstrap(
        self, req: Req, dspark_meta: dict, decode_prefix_len: int
    ) -> bool:
        hidden_start = int(dspark_meta.get("hidden_start", 0))
        hidden_len = int(dspark_meta.get("hidden_len", len(req.origin_input_ids)))
        range_error = self._validate_dspark_hidden_range(
            req, dspark_meta, decode_prefix_len
        )
        if range_error is not None:
            self._abort_dspark_hidden_bootstrap(req, range_error)
            return False

        # Prefill HiCache and decode radix caching are independent.  In
        # particular, DSV4 decode uses a specialized SWA pool that cannot use
        # decode radix caching, while prefill can still restore target KV from
        # HiCache and send it from decode_prefix_len.  The recompute boundary
        # below guarantees that the hidden suffix is produced even on a cache
        # hit, so requiring identical cache policies is both unnecessary and
        # prevents the useful P-HiCache/D-no-radix topology.
        req.dspark_prefill_recompute_start = hidden_start

        pp_slices = dspark_meta.get("pp_slices") or {}
        local_pp_slice = pp_slices.get(str(self.pp_rank)) if pp_slices else None
        if pp_slices and local_pp_slice is None:
            message = (
                "DSpark hidden metadata is missing PP slice for prefill rank: "
                f"pp_rank={self.pp_rank}, available_pp_slices={sorted(pp_slices.keys())}"
            )
            self._abort_dspark_hidden_bootstrap(req, message)
            return False
        dst_indices = [
            int(x)
            for x in (
                local_pp_slice.get("dst_indices", [])
                if local_pp_slice
                else ([] if pp_slices else dspark_meta.get("dst_indices", []))
            )
        ]
        local_layer_ids = (
            [int(x) for x in local_pp_slice.get("layer_ids", [])]
            if local_pp_slice
            else (
                []
                if pp_slices
                else [int(x) for x in dspark_meta.get("target_layer_ids", [])]
            )
        )
        local_slice_len = (
            int(local_pp_slice.get("slice_len", 0))
            if local_pp_slice
            else len(local_layer_ids) * int(self.scheduler.model_config.hidden_size)
        )
        if not local_layer_ids:
            req.dspark_hidden_meta = dict(dspark_meta)
            req.dspark_hidden_src_indices = []
            req.dspark_hidden_dst_indices = []
            req.dspark_hidden_written = []
            req.dspark_hidden_release_guard = None
            return True

        pool = getattr(self.metadata_buffers, "dspark_hidden_pool", None)
        if pool is None:
            message = (
                "Decode requested DSpark hidden metadata on a prefill PP rank "
                "that owns target layers, but no hidden row pool was initialized: "
                f"pp_rank={self.pp_rank}, local_layer_ids={local_layer_ids}"
            )
            self._abort_dspark_hidden_bootstrap(req, message)
            return False

        if hidden_len != len(dst_indices):
            message = (
                "Invalid DSpark hidden metadata from decode: "
                f"hidden_len={hidden_len}, dst_indices={len(dst_indices)}, "
                f"pp_rank={self.pp_rank}"
            )
            self._abort_dspark_hidden_bootstrap(req, message)
            return False
        if (
            hidden_start < 0
            or hidden_len < 0
            or hidden_start + hidden_len > len(req.origin_input_ids)
        ):
            message = (
                "Invalid DSpark hidden metadata from decode: "
                f"hidden_start={hidden_start}, hidden_len={hidden_len}, "
                f"prompt_len={len(req.origin_input_ids)}"
            )
            self._abort_dspark_hidden_bootstrap(req, message)
            return False

        src_indices = pool.alloc(hidden_len)
        if src_indices is None:
            if hidden_len > pool.size:
                message = (
                    "DSpark hidden rows exceed prefill hidden pool capacity: "
                    f"rid={req.rid}, hidden_len={hidden_len}, pool_size={pool.size}. "
                    "Increase SGLANG_DSPARK_PD_HIDDEN_POOL_TOKENS or reduce the "
                    "maximum prompt/hidden transfer length."
                )
                self._abort_dspark_hidden_bootstrap(req, message)
            return False

        try:
            self._configure_dspark_hidden_capture(
                req, local_layer_ids, local_slice_len, dspark_meta
            )
        except Exception as exc:
            pool.free(src_indices)
            message = f"Failed to configure DSpark hidden capture: {exc}"
            self._abort_dspark_hidden_bootstrap(req, message)
            return False
        req.dspark_hidden_meta = dict(dspark_meta)
        req.dspark_hidden_release_guard = DSparkHiddenReleaseGuard()
        req.dspark_hidden_src_indices = src_indices
        req.dspark_hidden_dst_indices = dst_indices
        req.dspark_hidden_written = [False] * hidden_len
        return True

    def _sync_dspark_hicache_recompute_starts(self, reqs: List[Req]) -> None:
        """Derive a conservative DSpark recompute suffix on every CP/TP rank.

        Request metadata is intentionally not broadcast here.  This method runs
        in the scheduler loop where ranks can temporarily be in different queue
        phases; using a collective here can cross-match another phase.  Prompt
        length and the model window are static on every participant, so deriving
        the same suffix locally is both deterministic and deadlock-free.
        """
        if not reqs:
            return

        capture_layer_ids = list(
            getattr(self.scheduler, "dspark_pd_local_capture_layer_ids", [])
        )
        window_size = getattr(
            self.scheduler, "dspark_pd_hidden_transfer_window_size", None
        )
        for req in reqs:
            if getattr(req, "bootstrap_host", None) == FAKE_BOOTSTRAP_HOST:
                continue
            if capture_layer_ids:
                req.dspark_hidden_capture_layer_ids = capture_layer_ids.copy()
                prompt_len = len(req.origin_input_ids)
                req.dspark_prefill_recompute_start = (
                    max(0, prompt_len - int(window_size))
                    if window_size is not None and int(window_size) > 0
                    else 0
                )

    def _configure_dspark_hidden_capture(
        self,
        req: Req,
        target_layer_ids: List[int],
        hidden_size: int,
        dspark_meta: dict,
    ) -> None:
        if not target_layer_ids:
            raise RuntimeError(
                f"DSpark hidden metadata for req {req.rid} has empty target_layer_ids."
            )
        expected_hidden_size = len(target_layer_ids) * int(
            self.scheduler.model_config.hidden_size
        )
        if expected_hidden_size != int(hidden_size):
            raise RuntimeError(
                "DSpark hidden size mismatch on prefill: "
                f"metadata layers={target_layer_ids}, expected={expected_hidden_size}, "
                f"pool={hidden_size}"
            )

        model_runner = self.scheduler.tp_worker.model_runner
        spec_aux_config = getattr(model_runner, "spec_aux_config", None)
        configured = getattr(spec_aux_config, "dflash_target_layer_ids", None)
        all_target_layer_ids = [
            int(x) for x in dspark_meta.get("target_layer_ids", target_layer_ids)
        ]
        if configured is not None and list(configured) != all_target_layer_ids:
            raise RuntimeError(
                "DSpark target layer mismatch between prefill config and decode "
                f"metadata: prefill={configured}, decode={all_target_layer_ids}"
            )
        req.dspark_hidden_capture_layer_ids = [int(x) for x in target_layer_ids]

    def add(self, req: Req, num_kv_heads: int) -> None:
        if not self.create_sender(req, num_kv_heads):
            return
        self.queue.append(req)

    def extend(self, reqs: List[Req], num_kv_heads: int) -> None:
        for req in reqs:
            self.add(req, num_kv_heads)

    def _check_if_req_exceed_kv_capacity(self, req: Req) -> bool:
        if len(req.origin_input_ids) > self.max_total_num_tokens:
            message = f"Request {req.rid} exceeds the maximum number of tokens: {len(req.origin_input_ids)} > {self.max_total_num_tokens}"
            logger.error(message)
            req.time_stats.trace_ctx.abort(abort_info={"reason": message})
            prepare_abort(req, message, status_code=HTTPStatus.BAD_REQUEST)
            self.scheduler.output_streamer.stream_output([req], req.return_logprob)
            return True
        return False

    def _process_req(self, req: Req) -> None:
        """
        Set max_new_tokens = 1, so PrefillAdder memory estimation is accurate
        """
        req.sampling_params.max_new_tokens = 1

    def pop_bootstrapped(
        self,
        return_failed_reqs: bool = False,
        rids_to_check: Optional[List[str]] = None,
    ) -> List[Req]:
        """
        pop the reqs which has finished bootstrapping

        return_failed_reqs: For PP, on rank 0, also return the failed reqs to notify the next rank
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """

        bootstrapped_reqs = []
        failed_reqs = []
        indices_to_remove = set()
        attn_cp_cpu_group, attn_tp_cpu_group = get_disagg_poll_cpu_groups(
            self.scheduler, "bootstrap"
        )

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.queue],
            attn_cp_cpu_group,
            attn_tp_cpu_group,
            keys=[req.rid for req in self.queue],
            phase="bootstrap_poll",
        )

        metadata_credits = self.req_to_metadata_buffer_idx_allocator.available_size()
        pool = getattr(self.metadata_buffers, "dspark_hidden_pool", None)
        hidden_row_credits = pool.available_size() if pool is not None else 0
        local_admission_states = []

        for req, poll in zip(self.queue, polls):
            if (
                rids_to_check is not None
                and req.rid not in rids_to_check
                and poll != KVPoll.Failed
            ):
                # In PP mode, successful bootstrap still requires cross-rank
                # consensus. Local failures are terminal and must be drained
                # even if an earlier PP rank has already removed the request.
                local_admission_states.append(KVPoll.Bootstrapping)
                continue

            if poll == KVPoll.Failed:
                local_admission_states.append(KVPoll.Failed)
            elif poll == KVPoll.Bootstrapping:
                if self._requires_dspark_hidden_transfer(req):
                    # DSpark hidden must be captured for every prefill chunk.
                    # Do not run optimistic forward before hidden rows and
                    # capture metadata are materialized.
                    local_admission_states.append(KVPoll.Bootstrapping)
                    continue
                if (
                    req.time_stats.prefill_retry_count
                    < self.scheduler.server_args.optimistic_prefill_retries
                    and not req.is_retracted  # engine paused
                ):
                    metadata_cost = 1 if req.metadata_buffer_index < 0 else 0
                    if metadata_cost <= metadata_credits:
                        metadata_credits -= metadata_cost
                        local_admission_states.append(KVPoll.WaitingForInput)
                    else:
                        local_admission_states.append(KVPoll.Bootstrapping)
                else:
                    local_admission_states.append(KVPoll.Bootstrapping)
            elif poll == KVPoll.WaitingForInput:
                if should_force_retry(req):  # skip checking for testing
                    metadata_cost = 1 if req.metadata_buffer_index < 0 else 0
                    if metadata_cost <= metadata_credits:
                        metadata_credits -= metadata_cost
                        local_admission_states.append(KVPoll.WaitingForInput)
                    else:
                        local_admission_states.append(KVPoll.Bootstrapping)
                elif req.pending_bootstrap:
                    costs, error = self._probe_bootstrap_ready(
                        req, metadata_credits, hidden_row_credits
                    )
                    if error is not None:
                        self._abort_dspark_hidden_bootstrap(req, error)
                        local_admission_states.append(KVPoll.Failed)
                    elif costs is None:
                        local_admission_states.append(KVPoll.Bootstrapping)
                    else:
                        metadata_cost, hidden_cost = costs
                        metadata_credits -= metadata_cost
                        hidden_row_credits -= hidden_cost
                        local_admission_states.append(KVPoll.WaitingForInput)
                else:
                    local_admission_states.append(KVPoll.WaitingForInput)
            else:
                message = (
                    f"Unexpected poll state {poll} for req {req.rid} "
                    "in pop_bootstrapped"
                )
                self._abort_dspark_hidden_bootstrap(req, message)
                local_admission_states.append(KVPoll.Failed)

        keys = [req.rid for req in self.queue]
        admission_states = all_reduce_attn_cp_tp_keyed_values(
            local_admission_states,
            keys,
            attn_cp_cpu_group,
            attn_tp_cpu_group,
            phase="bootstrap_admission",
        )

        local_commit_states = []
        for req, poll, admission_state in zip(self.queue, polls, admission_states):
            if admission_state != KVPoll.WaitingForInput:
                local_commit_states.append(admission_state)
                continue

            if poll == KVPoll.Bootstrapping or should_force_retry(req):
                committed = self.ensure_metadata_buffer(req)
            elif req.pending_bootstrap:
                committed = self.finalize_bootstrap(req)
            else:
                committed = True
            local_commit_states.append(
                KVPoll.WaitingForInput
                if committed
                else (KVPoll.Failed if is_aborted(req) else KVPoll.Bootstrapping)
            )

        commit_states = all_reduce_attn_cp_tp_keyed_values(
            local_commit_states,
            keys,
            attn_cp_cpu_group,
            attn_tp_cpu_group,
            phase="bootstrap_commit",
        )

        for i, (req, commit_state) in enumerate(zip(self.queue, commit_states)):
            if commit_state == KVPoll.Failed:
                self.scheduler.handle_bootstrap_failure(req)
                indices_to_remove.add(i)
                failed_reqs.append(req)
            elif commit_state == KVPoll.WaitingForInput:
                bootstrapped_reqs.append(req)
                indices_to_remove.add(i)
                req.time_stats.set_wait_queue_entry_time()
            elif commit_state != KVPoll.Bootstrapping:
                message = (
                    f"Unexpected bootstrap commit state {commit_state} for "
                    f"req {req.rid}"
                )
                self._abort_dspark_hidden_bootstrap(req, message)
                self.scheduler.handle_bootstrap_failure(req)
                indices_to_remove.add(i)
                failed_reqs.append(req)

        self._sync_dspark_hicache_recompute_starts(bootstrapped_reqs)

        self.queue = [
            entry for i, entry in enumerate(self.queue) if i not in indices_to_remove
        ]

        if return_failed_reqs is False:
            return bootstrapped_reqs
        else:
            return bootstrapped_reqs, failed_reqs

    def get_ready_bootstrapped_rids_for_pp(self) -> Tuple[List[str], List[str]]:
        """Return ordered PP candidates using a side-effect-free credit probe."""
        attn_cp_cpu_group, attn_tp_cpu_group = get_disagg_poll_cpu_groups(
            self.scheduler, "pp_bootstrap"
        )
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.queue],
            attn_cp_cpu_group,
            attn_tp_cpu_group,
            keys=[req.rid for req in self.queue],
            phase="pp_bootstrap_poll",
        )

        metadata_credits = self.req_to_metadata_buffer_idx_allocator.available_size()
        pool = getattr(self.metadata_buffers, "dspark_hidden_pool", None)
        hidden_row_credits = pool.available_size() if pool is not None else 0
        local_readiness_states = []
        admission_blocked = False

        for req, poll in zip(self.queue, polls):
            if poll == KVPoll.Failed:
                local_readiness_states.append(KVPoll.Failed)
            elif admission_blocked:
                # Preserve the original FCFS admission contract while still
                # retaining one collective slot for every queued request.
                local_readiness_states.append(KVPoll.Bootstrapping)
            elif poll == KVPoll.WaitingForInput:
                if should_force_retry(req):
                    metadata_cost = 1 if req.metadata_buffer_index < 0 else 0
                    if metadata_cost > metadata_credits:
                        local_readiness_states.append(KVPoll.Bootstrapping)
                        admission_blocked = True
                    else:
                        metadata_credits -= metadata_cost
                        local_readiness_states.append(KVPoll.WaitingForInput)
                elif req.pending_bootstrap:
                    costs, error = self._probe_bootstrap_ready(
                        req, metadata_credits, hidden_row_credits
                    )
                    if error is not None:
                        self._abort_dspark_hidden_bootstrap(req, error)
                        local_readiness_states.append(KVPoll.Failed)
                    elif costs is None:
                        local_readiness_states.append(KVPoll.Bootstrapping)
                        admission_blocked = True
                    else:
                        metadata_cost, hidden_cost = costs
                        metadata_credits -= metadata_cost
                        hidden_row_credits -= hidden_cost
                        local_readiness_states.append(KVPoll.WaitingForInput)
                else:
                    local_readiness_states.append(KVPoll.WaitingForInput)
            elif poll == KVPoll.Bootstrapping:
                local_readiness_states.append(KVPoll.Bootstrapping)
            else:
                message = (
                    f"Unexpected poll state {poll} for req {req.rid} "
                    "in get_ready_bootstrapped_rids_for_pp"
                )
                self._abort_dspark_hidden_bootstrap(req, message)
                local_readiness_states.append(KVPoll.Failed)

        keys = [req.rid for req in self.queue]
        readiness_states = all_reduce_attn_cp_tp_keyed_values(
            local_readiness_states,
            keys,
            attn_cp_cpu_group,
            attn_tp_cpu_group,
            phase="pp_bootstrap_admission",
        )
        good_rids = [
            rid
            for rid, state in zip(keys, readiness_states)
            if state == KVPoll.WaitingForInput
        ]
        failed_rids = [
            rid for rid, state in zip(keys, readiness_states) if state == KVPoll.Failed
        ]
        return good_rids, failed_rids

    def release_memory_occupation(self):
        self.queue.clear()
        if hasattr(self.kv_manager, "deregister_buffer_to_engine"):
            self.kv_manager.deregister_buffer_to_engine()

    def resume_memory_occupation(self):
        if hasattr(self.kv_manager, "register_buffer_to_engine"):
            self.kv_manager.register_buffer_to_engine()


class SchedulerDisaggregationPrefillMixin:
    """
    Mixin for Scheduler to handle disaggregation prefill
    """

    def maybe_prefetch_staging_for_batch(self: Scheduler, batch: ScheduleBatch) -> None:
        """Pre-send STAGING_REQ so decode allocates staging during GPU forward."""
        kv_mgr = self.disagg_prefill_bootstrap_queue.kv_manager
        prefetch = getattr(kv_mgr, "_prefetch_staging_reqs", None)
        if prefetch is None:
            return
        for req in batch.reqs:
            room = getattr(req, "bootstrap_room", None)
            if room is not None and room in kv_mgr.transfer_infos:
                prefetch(room)

    def resolve_waiting_queue_bootstrap(self: Scheduler) -> None:
        """Resolve bootstrap status for waiting prefill requests before admission.

        Covers the window between leaving the bootstrap queue and being admitted
        into a running batch: aborts requests whose decode peer died, and
        finalizes optimistic requests whose bootstrap completed so they skip
        the post-forward bootstrap check.
        """
        # Keep the collective shape identical across CP ranks even when an abort
        # control message becomes visible on one rank one scheduler tick earlier.
        # The local failure mask is reduced with MIN, so one rank's abort is
        # propagated to every CP/TP participant without dropping a poll slot.
        candidates = list(self.waiting_queue)
        attn_cp_cpu_group, attn_tp_cpu_group = get_disagg_poll_cpu_groups(
            self, "waiting"
        )
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in candidates],
            attn_cp_cpu_group,
            attn_tp_cpu_group,
            local_failed_mask=[is_aborted(req) for req in candidates],
            keys=[req.rid for req in candidates],
            phase="waiting",
        )
        failed = set()
        for req, poll in zip(candidates, polls):
            if poll == KVPoll.Failed:
                self.handle_bootstrap_failure(req)
                failed.add(req)
            elif (
                poll == KVPoll.WaitingForInput
                and req.pending_bootstrap
                and not should_force_retry(req)
            ):
                # Optimistic requests reserved a metadata buffer when popped, so
                # finalize cannot fail here; if it ever does, the request stays
                # pending and the post-forward check resolves it.
                self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req)
        if failed:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in failed
            ]

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def get_next_disagg_prefill_batch_to_run(
        self: Scheduler,
    ) -> Optional[ScheduleBatch]:
        self.process_pending_chunked_abort()

        # HACK (byronhsu): reset the batch_is_full flag because we never enter update_running_batch which resets it
        # Otherwise, it hangs under high concurrency
        self.running_batch.batch_is_full = False

        self.process_prefill_chunk()

        self.resolve_waiting_queue_bootstrap()

        # A hybrid-SWA continuation can need pages released by the previous
        # chunk.  Under overlap scheduling that chunk's forward/result is still
        # in flight here, so trying to allocate the continuation may wait on
        # the same CUDA stream that the scheduler must drain.  This forms a
        # livelock (all ranks sit in allocator merge/sort and heartbeats stop).
        # Keep overlap for independent batches, but insert one result boundary
        # before reusing SWA pages for the same long request.
        wait_for_previous_swa_chunk = (
            self.enable_overlap
            and self.chunked_req is not None
            and self.chunked_req.inflight_middle_chunks > 0
            and self.tree_cache.supports_swa()
        )
        batch = None if wait_for_previous_swa_chunk else self.get_new_batch_prefill()
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
        self._prepare_dspark_hidden_capture_for_batch(batch)

        if batch:
            set_schedule_time_batch(batch)

        return batch

    def _prepare_dspark_hidden_capture_for_batch(
        self: Scheduler, batch: Optional[ScheduleBatch]
    ) -> None:
        if not batch:
            return

        configured_capture_layers = getattr(
            self, "dspark_pd_local_capture_layer_ids", None
        )
        if configured_capture_layers is None:
            configured_capture_layers = next(
                (
                    req.dspark_hidden_capture_layer_ids
                    for req in batch.reqs
                    if getattr(req, "dspark_hidden_capture_layer_ids", None)
                ),
                [],
            )
        dspark_capture_layers = list(configured_capture_layers)
        for req in batch.reqs:
            req_layers = getattr(req, "dspark_hidden_capture_layer_ids", None)
            if req_layers and list(req_layers) != dspark_capture_layers:
                self._fail_dspark_hidden_request(
                    req,
                    "DSpark capture layers differ from the local model config: "
                    f"rid={req.rid}, request={list(req_layers)}, "
                    f"local={dspark_capture_layers}",
                )
        if not any(dspark_hidden_capture_intersects_extend(req) for req in batch.reqs):
            return
        if dspark_capture_layers:
            batch.dspark_hidden_capture_layer_ids = [
                int(x) for x in dspark_capture_layers
            ]
            batch.capture_hidden_mode = CaptureHiddenMode.FULL

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                result = self.run_batch(batch)
                self._disagg_optimistic_polls_by_rid = (
                    self._poll_optimistic_prefill_batch(batch)
                )
                self.process_batch_result(batch, result)
            else:
                self._disagg_optimistic_polls_by_rid = (
                    self._poll_optimistic_prefill_batch(None)
                )
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()

            # Update last_batch
            self.last_batch = batch

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self: Scheduler) -> None:
        self.result_queue = deque()

        while True:
            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            self.waiting_queue.extend(
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
            )
            if self._engine_paused:
                continue

            self._apply_war_barrier()

            # Get the next batch to run
            batch = self.get_next_disagg_prefill_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                if self.enable_staging:
                    self.maybe_prefetch_staging_for_batch(batch)
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # Process the last batch
            result_batch = self.result_queue[0][0] if self.last_batch else None
            self._disagg_optimistic_polls_by_rid = self._poll_optimistic_prefill_batch(
                result_batch
            )
            if self.last_batch:
                tmp_batch, tmp_result = self.result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            self.process_disagg_prefill_inflight_queue()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            self.launch_batch_sample_if_needed(batch_result)

            # Update last_batch
            self.last_batch = batch

    def _write_dspark_hidden_rows_for_batch(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        pool = getattr(self.disagg_metadata_buffers, "dspark_hidden_pool", None)
        logits_output = result.logits_output
        hidden_states = getattr(logits_output, "hidden_states", None)
        if hidden_states is None and result.pp_hidden_states_proxy_tensors is not None:
            proxy_tensors = result.pp_hidden_states_proxy_tensors.tensors
            aux_keys = sorted(
                key
                for key in proxy_tensors
                if key.startswith("dspark_aux_hidden_states_")
            )
            if aux_keys:
                hidden_states = torch.cat(
                    [proxy_tensors[key] for key in aux_keys], dim=-1
                )
        needs_dspark_hidden = any(
            dspark_hidden_capture_intersects_extend(req) for req in batch.reqs
        )
        if needs_dspark_hidden and (pool is None or hidden_states is None):
            reason = (
                "DSpark hidden row pool is unavailable"
                if pool is None
                else "forward output has no hidden states"
            )
            for req in batch.reqs:
                if not dspark_hidden_capture_intersects_extend(req):
                    continue
                self._fail_dspark_hidden_request(
                    req,
                    "DSpark hidden capture failed because "
                    f"{reason}: rid={req.rid}, "
                    f"batch_capture_layers={batch.dspark_hidden_capture_layer_ids}",
                )
            return
        if pool is None or hidden_states is None or batch.extend_lens is None:
            return

        expected_hidden_rows = sum(int(extend_len) for extend_len in batch.extend_lens)
        if hidden_states.ndim < 2 or hidden_states.shape[0] < expected_hidden_rows:
            for req in batch.reqs:
                if not dspark_hidden_capture_intersects_extend(req):
                    continue
                self._fail_dspark_hidden_request(
                    req,
                    "DSpark hidden capture returned an invalid batch shape: "
                    f"rid={req.rid}, shape={tuple(hidden_states.shape)}, "
                    f"expected_rows={expected_hidden_rows}",
                )
            return

        if batch.seq_lens_cpu is not None:
            chunk_ends = [int(x) for x in batch.seq_lens_cpu.tolist()]
        else:
            assert batch.prefix_lens is not None
            chunk_ends = [
                int(prefix_len) + int(extend_len)
                for prefix_len, extend_len in zip(
                    batch.prefix_lens, batch.extend_lens, strict=True
                )
            ]

        hidden_offset = 0
        for req, extend_len, chunk_end in zip(
            batch.reqs, batch.extend_lens, chunk_ends, strict=True
        ):
            extend_len = int(extend_len)
            req_hidden = hidden_states[hidden_offset : hidden_offset + extend_len]
            hidden_offset += extend_len

            src_indices = getattr(req, "dspark_hidden_src_indices", None)
            if not src_indices:
                continue

            meta = getattr(req, "dspark_hidden_meta", None) or {}
            hidden_start = int(meta.get("hidden_start", 0))
            hidden_len = int(meta.get("hidden_len", len(src_indices)))
            chunk_start = chunk_end - extend_len
            write_start = max(chunk_start, hidden_start)
            write_end = min(chunk_end, hidden_start + hidden_len)
            if write_end <= write_start:
                continue

            local_start = write_start - hidden_start
            local_end = write_end - hidden_start
            chunk_local_start = write_start - chunk_start
            chunk_local_end = write_end - chunk_start
            req_hidden_to_write = req_hidden
            pp_slices = meta.get("pp_slices") or {}
            pp_rank = int(self.ps.pp_rank)
            local_pp_slice = pp_slices.get(str(pp_rank)) if pp_slices else None
            local_slice_len = (
                int(local_pp_slice.get("slice_len", 0))
                if local_pp_slice
                else pool.hidden_size
            )
            if local_slice_len > 0 and req_hidden_to_write.shape[-1] != local_slice_len:
                local_slice_start = (
                    int(local_pp_slice.get("slice_start", 0)) if local_pp_slice else 0
                )
                local_slice_end = local_slice_start + local_slice_len
                if req_hidden_to_write.shape[-1] < local_slice_end:
                    self._fail_dspark_hidden_request(
                        req,
                        "DSpark hidden width does not match prefill PP slice: "
                        f"rid={req.rid}, pp_rank={pp_rank}, "
                        f"hidden_width={req_hidden_to_write.shape[-1]}, "
                        f"slice_start={local_slice_start}, "
                        f"slice_len={local_slice_len}",
                    )
                    continue
                req_hidden_to_write = req_hidden_to_write[
                    :, local_slice_start:local_slice_end
                ]
            try:
                pool.write(
                    src_indices[local_start:local_end],
                    req_hidden_to_write[chunk_local_start:chunk_local_end],
                )
            except ValueError as error:
                # Shape and row-index mismatches are isolated to this request.
                # Do not catch CUDA/runtime errors: the device context may no
                # longer be safe for subsequent requests.
                self._fail_dspark_hidden_request(
                    req,
                    f"DSpark hidden row write validation failed: rid={req.rid}, "
                    f"error={error}",
                )
                continue
            rows = local_end - local_start
            written = getattr(req, "dspark_hidden_written", None)
            if written is not None:
                written[local_start:local_end] = [True] * rows

    def _fail_dspark_hidden_request(self: Scheduler, req: Req, message: str) -> None:
        """Fail one malformed DSpark PD request without killing the scheduler."""
        if is_aborted(req):
            return

        logger.error(message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": message})
        prepare_abort(req, message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        req.pending_bootstrap = False

        sender = getattr(req, "disagg_kv_sender", None)
        if sender is None:
            return
        try:
            sender.abort()
        except Exception:
            logger.exception(
                "Failed to abort DSpark PD sender after hidden-state failure: "
                "rid=%s bootstrap_room=%s",
                req.rid,
                req.bootstrap_room,
            )
        kv_mgr = getattr(sender, "kv_mgr", None)
        if kv_mgr is not None:
            kv_mgr.record_failure(
                getattr(sender, "bootstrap_room", req.bootstrap_room), message
            )

    def process_batch_result_disagg_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """
        Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        Adapted from process_batch_result_prefill
        """
        (
            logits_output,
            next_token_ids,
            extend_input_len_per_req,
            extend_logprob_start_len_per_req,
            copy_done,
        ) = (
            result.logits_output,
            result.next_token_ids,
            result.extend_input_len_per_req,
            result.extend_logprob_start_len_per_req,
            result.copy_done,
        )

        if copy_done is not None:
            copy_done.synchronize()
        if result.routed_experts_output is not None:
            result.routed_experts_output.finalize()
            result.routed_experts_output = None
        if result.indexer_topk_output is not None:
            result.indexer_topk_output.finalize()
            result.indexer_topk_output = None

        logprob_pt = 0
        # Transfer kv for prefill completed requests and add it into disagg_prefill_inflight_queue
        next_token_ids = result.next_token_ids.tolist()
        self.batch_result_processor.move_logprobs_to_cpu(
            batch=batch,
            logits_output=logits_output,
        )
        self._write_dspark_hidden_rows_for_batch(batch, result)

        def advance_logprob_pt(i: int, req: Req) -> None:
            nonlocal logprob_pt
            if not req.return_logprob or extend_input_len_per_req is None:
                return
            extend_logprob_start_len = extend_logprob_start_len_per_req[i]
            extend_input_len = extend_input_len_per_req[i]
            if extend_logprob_start_len < extend_input_len:
                logprob_pt += extend_input_len - extend_logprob_start_len

        # The event loop enters the optimistic collective unconditionally once
        # per scheduler tick.  Consuming the precomputed result here prevents a
        # result-dependent branch from changing collective order across CP.
        optimistic_polls_by_rid = getattr(self, "_disagg_optimistic_polls_by_rid", {})

        for i, (req, next_token_id) in enumerate(
            zip(batch.reqs, next_token_ids, strict=True)
        ):
            if req.inflight_middle_chunks <= 0:
                req.time_stats.set_prefill_finished_time()

                # For optimistic requests, check bootstrap before side effects
                if req.rid in optimistic_polls_by_rid:
                    if not self.handle_pending_bootstrap(
                        req,
                        optimistic_polls_by_rid[req.rid],
                        defer_release=False,
                    ):
                        advance_logprob_pt(i, req)
                        continue

                if is_aborted(req):
                    # Keep the request in the normal inflight failure path so
                    # KV, hidden rows, metadata, metrics, and client output are
                    # finalized exactly once.
                    advance_logprob_pt(i, req)
                    self.disagg_prefill_inflight_queue.append(req)
                    req.time_stats.set_prefill_transfer_queue_entry_time()
                    continue

                req.output_ids.append(next_token_id)
                maybe_cache_unfinished_req(req, self.tree_cache)
                self.disagg_prefill_inflight_queue.append(req)
                if self.spec_algorithm.is_eagle() and batch.spec_info is not None:
                    req.output_topk_p = batch.spec_info.topk_p[i]
                    req.output_topk_index = batch.spec_info.topk_index[i]
                    req.hidden_states_tensor = (
                        batch.spec_info.hidden_states[i].cpu().clone()
                    )
                else:
                    req.hidden_states_tensor = None
                if req.return_logprob:
                    assert extend_logprob_start_len_per_req is not None
                    assert extend_input_len_per_req is not None
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    num_input_logprobs = extend_input_len - extend_logprob_start_len
                    self.batch_result_processor.logprob_result_processor.add_logprob_return_values(
                        i,
                        req,
                        logprob_pt,
                        next_token_ids,
                        num_input_logprobs,
                        logits_output,
                    )
                    logprob_pt += num_input_logprobs
                self.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()

                if req.grammar is not None:
                    try:
                        req.grammar.accept_token(next_token_id)
                    except ValueError as e:
                        error_message = f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
                        release_kv_cache(req, self.tree_cache)
                        prepare_abort(
                            req,
                            error_message,
                            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    req.grammar.finished = req.finished()
            else:
                # being chunked reqs' prefill is not finished
                req.inflight_middle_chunks -= 1

                # Overlap deferred release for optimistic requests stopped in process_prefill_chunk
                if req.pending_bootstrap:
                    advance_logprob_pt(i, req)
                    self.optimistic_release_and_requeue(req)
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                # Optimistic bootstrap can fail while this overlapped chunk is
                # already running. Drop aborted chunks instead of sending KV.
                if is_aborted(req):
                    advance_logprob_pt(i, req)
                    if self.chunked_req is req:
                        # Stop launching new chunks. Already-launched overlap
                        # chunks remain represented by inflight_middle_chunks
                        # and are drained before the request is finalized.
                        self.chunked_req = None
                    if req.inflight_middle_chunks <= 0:
                        self.disagg_prefill_inflight_queue.append(req)
                        req.time_stats.set_prefill_transfer_queue_entry_time()
                    req.time_stats.set_last_chunked_prefill_finish_time()
                    continue

                if req.return_logprob:
                    extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                    extend_input_len = extend_input_len_per_req[i]
                    if extend_logprob_start_len < extend_input_len:
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.batch_result_processor.logprob_result_processor.add_input_logprob_return_values(
                            i,
                            req,
                            logits_output,
                            logprob_pt,
                            num_input_logprobs,
                            last_prefill_chunk=False,
                        )
                        logprob_pt += num_input_logprobs

                if self.enable_overlap:
                    if req.metadata_buffer_index < 0:
                        self._fail_dspark_hidden_request(
                            req,
                            "Chunked PD prefill lost its metadata buffer: "
                            f"rid={req.rid}, bootstrap_room={req.bootstrap_room}",
                        )
                        if self.chunked_req is req:
                            self.chunked_req = None
                        if req.inflight_middle_chunks <= 0:
                            self.disagg_prefill_inflight_queue.append(req)
                            req.time_stats.set_prefill_transfer_queue_entry_time()
                        req.time_stats.set_last_chunked_prefill_finish_time()
                        continue
                    self.send_kv_chunk(req, last_chunk=False, end_idx=req.tmp_end_idx)
                req.time_stats.set_last_chunked_prefill_finish_time()

        can_run_cuda_graph = result.can_run_cuda_graph
        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def process_disagg_prefill_inflight_queue(
        self: Scheduler, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        """
        Poll the requests in the middle of transfer. If done, return the request.
        rids_to_check: For PP, on rank > 0, check the rids from the previous rank has consensus with the current rank.
        """
        done_reqs = []
        attn_cp_cpu_group, attn_tp_cpu_group = get_disagg_poll_cpu_groups(
            self, "inflight"
        )

        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            attn_cp_cpu_group,
            attn_tp_cpu_group,
            keys=[req.rid for req in self.disagg_prefill_inflight_queue],
            phase="inflight",
        )

        undone_reqs: List[Req] = []
        terminal_rids_to_check = (
            set(rids_to_check) if rids_to_check is not None else None
        )
        # Check .poll() for the reqs in disagg_prefill_inflight_queue. If Success, respond to the client and remove it from the queue
        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            if terminal_rids_to_check is not None:
                if req.rid not in terminal_rids_to_check:
                    undone_reqs.append(req)
                    continue

                # In PP mode, the previous rank may have reached a terminal
                # state (Success/Failed) while this rank's local poll is still
                # in a transient state due to clock skew or propagation delay.
                # Treat non-terminal states as undone instead of crashing.
                if poll not in (
                    KVPoll.Success,
                    KVPoll.Failed,
                ):
                    logger.warning_once(
                        f"PP rank {self.ps.pp_rank}: unexpected poll state {poll} for rid {req.rid} "
                        f"from consensus; treating as undone",
                    )
                    undone_reqs.append(req)
                    continue

            maybe_release_dspark_hidden_rows_on_hidden_done(
                req,
                getattr(self.disagg_metadata_buffers, "dspark_hidden_pool", None),
            )

            if req.pending_bootstrap:
                # Parked: prefill finished before bootstrap completed.
                if self.handle_pending_bootstrap(req, poll):
                    self.send_kv_chunk(req, last_chunk=True)
                    undone_reqs.append(req)
                elif poll != KVPoll.Failed:
                    undone_reqs.append(req)
                continue

            if poll in [KVPoll.WaitingForInput, KVPoll.Transferring]:
                undone_reqs.append(req)
            elif poll == KVPoll.Success:  # transfer done
                if req.req_pool_idx is not None or self.tree_cache.supports_mamba():
                    release_kv_cache(req, self.tree_cache)  # unlock the tree
                req.finished_reason = FINISH_LENGTH(length=0)
                # FIXME: clean up req's data in transfer engine
                if hasattr(req.disagg_kv_sender, "clear"):
                    req.disagg_kv_sender.clear()
                done_reqs.append(req)
                req.time_stats.set_prefill_kv_transfer_finish_time()
            elif poll == KVPoll.Failed:
                error_message = f"Prefill transfer failed for request rank={self.ps.tp_rank} {req.rid=} {req.bootstrap_room=}"
                is_propagated = False
                try:
                    req.disagg_kv_sender.failure_exception()
                except Exception as e:
                    error_message += f" with exception {e}"
                    is_propagated = getattr(e, "is_from_another_rank", False)
                # Mute error message for propagated exceptions to avoid duplicate logging
                if is_propagated:
                    logger.debug(error_message)
                else:
                    logger.warning(error_message)
                req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
                if req.req_pool_idx is not None or self.tree_cache.supports_mamba():
                    release_kv_cache(req, self.tree_cache)  # unlock the tree
                prepare_abort(
                    req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
                )
                done_reqs.append(req)
                if self.metrics_reporter.enable_metrics:
                    self.metrics_collector.increment_transfer_failed_reqs()
            else:
                logger.warning_once(
                    f"Unexpected polling state {poll} for rid {req.rid} in inflight queue; "
                    f"treating as undone",
                )
                undone_reqs.append(req)

        for req in done_reqs:
            req.time_stats.set_completion_time()

        for req in done_reqs:
            if isinstance(req.finished_reason, FINISH_ABORT):
                continue
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST:
                continue
            kv_mgr = getattr(req.disagg_kv_sender, "kv_mgr", None)
            if kv_mgr and getattr(kv_mgr, "is_dummy_cp_rank", False):
                continue
            metrics = req.time_stats.compute_and_observe_kv_transfer_metrics(
                req.disagg_kv_sender.get_transfer_metric()
            )
            if metrics:
                # Update last-value for REST API
                if "latency_ms" in metrics:
                    self.metrics_reporter.kv_transfer_latency_ms = metrics["latency_ms"]
                if "speed_gb_s" in metrics:
                    self.metrics_reporter.kv_transfer_speed_gb_s = metrics["speed_gb_s"]

        # Stream requests which have finished transfer
        self.output_streamer.stream_output(
            done_reqs,
            any(req.return_logprob for req in done_reqs),
            None,
        )
        for req in done_reqs:
            req: Req

            maybe_release_metadata_buffer(
                req,
                self.req_to_metadata_buffer_idx_allocator,
                getattr(self.disagg_metadata_buffers, "dspark_hidden_pool", None),
            )

        self.disagg_prefill_inflight_queue = undone_reqs

        return done_reqs

    def get_transferred_rids(self: Scheduler) -> List[str]:
        """
        Used by PP to inspect local terminal transfers without popping requests.
        """
        attn_cp_cpu_group, attn_tp_cpu_group = get_disagg_poll_cpu_groups(
            self, "transferred"
        )
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in self.disagg_prefill_inflight_queue],
            attn_cp_cpu_group,
            attn_tp_cpu_group,
            keys=[req.rid for req in self.disagg_prefill_inflight_queue],
            phase="transferred",
        )

        transferred_rids: List[str] = []
        dspark_hidden_pool = getattr(
            self.disagg_metadata_buffers, "dspark_hidden_pool", None
        )

        for req, poll in zip(self.disagg_prefill_inflight_queue, polls):
            maybe_release_dspark_hidden_rows_on_hidden_done(req, dspark_hidden_pool)
            if poll == KVPoll.Success or poll == KVPoll.Failed:
                transferred_rids.append(req.rid)

        return transferred_rids

    def handle_bootstrap_failure(self: Scheduler, req: Req) -> None:
        error_message = (
            f"Prefill bootstrap failed for request rank={self.ps.tp_rank} "
            f"{req.rid=} {req.bootstrap_room=}"
        )
        is_propagated = False
        try:
            req.disagg_kv_sender.failure_exception()
        except Exception as e:
            error_message += f" with exception {e}"
            is_propagated = getattr(e, "is_from_another_rank", False)
        # Mute error message for propagated exceptions to avoid duplicate logging
        if is_propagated:
            logger.debug(error_message)
        else:
            logger.warning(error_message)
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_message})
        if req.req_pool_idx is not None or self.tree_cache.supports_mamba():
            release_kv_cache(req, self.tree_cache)
        maybe_release_metadata_buffer(
            req,
            self.req_to_metadata_buffer_idx_allocator,
            getattr(self.disagg_metadata_buffers, "dspark_hidden_pool", None),
        )
        req.pending_bootstrap = False
        prepare_abort(req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        self.output_streamer.stream_output([req], req.return_logprob)
        if self.metrics_reporter.enable_metrics:
            self.metrics_collector.increment_bootstrap_failed_reqs()
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)

    def handle_pending_bootstrap(
        self: Scheduler, req: Req, poll: KVPoll, defer_release: bool
    ) -> bool:
        """Return True when bootstrap is finalized and KV transfer can proceed."""
        if poll == KVPoll.Failed:
            self.handle_bootstrap_failure(req)
            return False
        elif poll == KVPoll.Bootstrapping:
            if not defer_release:
                self.optimistic_release_and_requeue(req)
            return False
        elif poll == KVPoll.WaitingForInput:
            force_retry = should_force_retry(req)  # test hook
            if force_retry:
                if not defer_release:
                    self.optimistic_release_and_requeue(req)
                return False
            if self.disagg_prefill_bootstrap_queue.finalize_bootstrap(req):
                return True
            if is_aborted(req):
                self.handle_bootstrap_failure(req)
            return False
        else:
            message = f"Unexpected poll state {poll} for req {req.rid} in handle_pending_bootstrap"
            self.disagg_prefill_bootstrap_queue._abort_dspark_hidden_bootstrap(
                req, message
            )
            self.handle_bootstrap_failure(req)
            return False

    def _poll_optimistic_prefill_batch(
        self: Scheduler, batch: Optional[ScheduleBatch]
    ) -> dict:
        """Enter the optimistic-poll collective once per scheduler tick."""
        candidates = (
            [
                req
                for req in batch.reqs
                if req.pending_bootstrap and req.inflight_middle_chunks <= 0
            ]
            if batch is not None
            else []
        )
        attn_cp_cpu_group, attn_tp_cpu_group = get_disagg_poll_cpu_groups(
            self, "optimistic"
        )
        polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in candidates],
            attn_cp_cpu_group,
            attn_tp_cpu_group,
            keys=[req.rid for req in candidates],
            phase="optimistic",
        )
        return {req.rid: poll for req, poll in zip(candidates, polls)}

    def check_bootstrap(self: Scheduler, req: Req, poll: Optional[KVPoll]) -> bool:
        """Check bootstrap status for an optimistic prefilled request.
        Returns True if bootstrap is finished."""
        if not req.pending_bootstrap:
            return True
        if poll is None:
            raise RuntimeError(
                "Missing chunked bootstrap poll for pending request: " f"rid={req.rid}"
            )
        return self.handle_pending_bootstrap(
            req, poll, defer_release=self.enable_overlap
        )

    def process_prefill_chunk(self: Scheduler) -> None:
        chunked_req_to_exclude = set()
        pending_chunked_req = (
            self.chunked_req
            if self.chunked_req is not None and self.chunked_req.pending_bootstrap
            else None
        )
        chunked_candidates = [pending_chunked_req] if pending_chunked_req else []
        attn_cp_cpu_group, attn_tp_cpu_group = get_disagg_poll_cpu_groups(
            self, "chunked"
        )
        chunked_polls = poll_and_all_reduce_attn_cp_tp_group(
            [req.disagg_kv_sender for req in chunked_candidates],
            attn_cp_cpu_group,
            attn_tp_cpu_group,
            keys=[req.rid for req in chunked_candidates],
            phase="chunked",
        )
        chunked_poll = chunked_polls[0] if chunked_polls else None
        if self.chunked_req:
            chunked_req_to_exclude.add(self.chunked_req)
            maybe_cache_unfinished_req(self.chunked_req, self.tree_cache, chunked=True)

            if not self.check_bootstrap(self.chunked_req, chunked_poll):
                self.chunked_req = None  # stop the current chunked prefill
            elif self.enable_overlap:
                # Delay KV transfer to process_batch_result_disagg_prefill when overlap is enabled to ensure results are resolved
                self.chunked_req.tmp_end_idx = min(
                    self.chunked_req.extend_range.end,
                    len(self.chunked_req.origin_input_ids),
                )
            else:
                self.send_kv_chunk(self.chunked_req)

            if self.chunked_req is not None:
                self.running_batch.batch_is_full = False

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.last_batch.chunked_req:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(self.last_batch.chunked_req)

            last_bs = self.last_batch.batch_size()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            if self.last_batch.batch_size() < last_bs:
                self.running_batch.batch_is_full = False

    def maybe_send_cached_prefix_chunk(self: Scheduler, req: Req) -> None:
        # Only bootstrap-finalized requests; staging excluded.
        if (
            not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get()
            or self.enable_staging
            or req.pending_bootstrap
        ):
            return

        # Device-resident prefix only; page-aligned so start_send_idx stays exact.
        cached_end = len(req.prefix_indices) - req.host_hit_length
        if cached_end <= req.start_send_idx:
            return
        page_size = self.token_to_kv_pool_allocator.page_size
        cached_end -= cached_end % page_size
        if cached_end <= req.start_send_idx:
            return
        self.send_kv_chunk(req, last_chunk=False, end_idx=cached_end)

    def send_kv_chunk(
        self: Scheduler,
        req: Req,
        last_chunk: bool = False,
        end_idx: Optional[int] = None,
    ) -> bool:
        """
        Send a prefilled chunk to the decode server
        """
        if is_aborted(req):
            return False

        page_size = self.token_to_kv_pool_allocator.page_size
        start_idx = req.start_send_idx
        transfer_input_len = len(req.origin_input_ids)
        end_idx = (
            end_idx
            if end_idx is not None
            else min(req.extend_range.end, transfer_input_len)
        )

        if not last_chunk:
            # if not the last chunk and the last page is partial, delay the last partial page to the next send
            end_idx = end_idx - end_idx % page_size

        if end_idx < start_idx:
            logger.debug(
                "send_kv_chunk skip: rid=%s start_send_idx=%s end_idx=%s",
                req.rid,
                start_idx,
                end_idx,
            )
            return True

        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, start_idx:end_idx
        ]
        state_indices: Optional[List] = None
        if last_chunk:
            self.disagg_metadata_buffers.set_buf(req)

            # Most state payloads read token-pool rows and should match the KV
            # range actually materialized on prefill. C128 state is request
            # scoped, so its transfer index must use the logical input length
            # that decode used to register the destination row.
            seq_len = min(req.extend_range.end, transfer_input_len)
            c128_seq_len = transfer_input_len

            def _mamba_payload():
                return [
                    self.req_to_token_pool.req_index_to_mamba_index_mapping[
                        req.req_pool_idx
                    ]
                    .cpu()
                    .numpy()
                ]

            def _swa_payload():
                window_size = self.sliding_window_size
                window_start = max(0, seq_len - window_size)
                window_start = (window_start // page_size) * page_size
                window_kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, window_start:seq_len
                ]
                window_kv_indices_swa = (
                    self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                        window_kv_indices_full
                    )
                )
                return kv_to_page_indices(window_kv_indices_swa, page_size)

            def _dsa_payload():
                kv_indices_full = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :seq_len
                ]
                return kv_to_page_indices(kv_indices_full, page_size)

            def _swa_ring_payload():
                # Unified_kv SWA ring rows (req_pool_idx*ring_stride + pos%ring_stride)
                # for the last `window` positions, in ascending position order so
                # decode (its own req_pool_idx) matches positionally.
                _pool = self.token_to_kv_pool_allocator.get_kvcache()
                ring_stride = _pool.unified_swa_ring_size
                window_size = _pool.unified_swa_window
                window_start = max(0, seq_len - window_size)
                positions = np.arange(window_start, seq_len, dtype=np.int64)
                state_slot = int(req.req_pool_idx)
                ring_rows = state_slot * ring_stride + (positions % ring_stride)
                return ring_rows.astype(np.int32)

            def _c128_state_payload():
                online = is_dsv4_c128_online_enabled()
                ring_size = (
                    1
                    if online
                    else self.token_to_kv_pool_allocator.get_kvcache().get_ring_size(
                        128
                    )
                )
                return get_dsv4_c128_state_indices(
                    int(req.req_pool_idx),
                    c128_seq_len,
                    online=online,
                    ring_size=ring_size,
                )

            def _dspark_hidden_payload():
                src_indices = getattr(req, "dspark_hidden_src_indices", None)
                if not src_indices:
                    return []
                return np.asarray(src_indices, dtype=np.int32)

            dspark_payload_error = get_dspark_hidden_payload_error(req)
            if dspark_payload_error is not None:
                self._fail_dspark_hidden_request(req, dspark_payload_error)
                return False

            state_types = (
                self.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
            )
            state_indices = []
            for st in state_types:
                if st == StateType.MAMBA:
                    state_indices.append(_mamba_payload())
                elif st == StateType.SWA:
                    state_indices.append(_swa_payload())
                elif st == StateType.DSA:
                    state_indices.append(_dsa_payload())
                elif st == StateType.MINIMAX_INDEX_K:
                    # Index rows live at the same loc as main KV on the same
                    # page_size, so reuse the full-seq page-ids.
                    state_indices.append(_dsa_payload())
                elif st == StateType.SWA_RING:
                    state_indices.append(_swa_ring_payload())
                elif st == StateType.C128_STATE:
                    state_indices.append(_c128_state_payload())
                elif st == StateType.DSPARK_HIDDEN:
                    state_indices.append(_dspark_hidden_payload())
                else:
                    state_indices.append(None)

        page_indices = kv_to_page_indices(kv_indices, page_size)
        if not req.disagg_kv_sender.should_send_kv_chunk(len(page_indices), last_chunk):
            return True
        if (
            last_chunk
            and getattr(req, "dspark_hidden_src_indices", None)
            and hasattr(req.disagg_kv_sender, "set_dspark_hidden_release_guard")
        ):
            req.disagg_kv_sender.set_dspark_hidden_release_guard(
                getattr(req, "dspark_hidden_release_guard", None)
            )
        if (
            last_chunk
            and getattr(req, "dspark_hidden_src_indices", None)
            and hasattr(req.disagg_kv_sender, "set_source_event")
        ):
            source_event = self.device_module.Event()
            source_event.record()
            req.disagg_kv_sender.set_source_event(source_event)
        req.disagg_kv_sender.send(page_indices, state_indices)
        req.start_send_idx = end_idx
        return True

    def optimistic_release_and_requeue(self: Scheduler, req: Req) -> None:
        """Release KV cache and requeue an optimistic prefill request."""
        max_retries = self.server_args.optimistic_prefill_retries
        maybe_cache_unfinished_req(req, self.tree_cache)
        release_kv_cache(req, self.tree_cache)
        req.reset_for_retract()
        req.output_ids = array("q")
        req.start_send_idx = 0
        req.tmp_end_idx = -1
        req.hidden_states_tensor = None
        req.pending_bootstrap = True
        req.time_stats.reset_prefill_retry_time()
        if req.time_stats.prefill_retry_count >= max_retries:
            logger.info(
                f"Req {req.rid} exhausted optimistic prefill retries "
                "falling back to bootstrap queue"
            )
            # Reset it so the next real bootstrap done can be recorded.
            req.time_stats.bootstrap_done_time = 0.0
            self.disagg_prefill_bootstrap_queue.queue.append(req)
        else:
            req.time_stats.prefill_retry_count += 1
            logger.info(
                f"Req {req.rid} optimistic prefill retry "
                f"{req.time_stats.prefill_retry_count}/{max_retries}"
            )
            if self.metrics_reporter.enable_metrics:
                self.metrics_collector.increment_prefill_retries(1)
            req.time_stats.set_wait_queue_entry_time()
            self.waiting_queue.insert(0, req)
