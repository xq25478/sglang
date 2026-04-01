"""
Draft-side Scheduler Mixin v2 for Remote Speculative Decoding.

Draft Batch 分离 + Draft 优先调度 (draft_batch_design.md v0.5)
=================================================================

v2 相较 v1 的核心改进:
  1. 管理平面分离: Draft 请求独立使用 draft_waiting_queue / draft_batch / draft_paused_reqs，
     不再与 Normal 请求混用 waiting_queue / running_batch / paused_reqs。
  2. DraftReqLocation 枚举: 替代字符串 location，消除 _get_req_actual_location() 遍历。
  3. 两种执行策略（由开关控制）:
     - 混合模式（默认）: draft_batch 合入 running_batch，一起 decode → GPU 利用率最高。
     - Draft 优先模式: draft_batch 独立 decode N 步，再跑 Normal → Draft 延迟最低。
  4. _prefill_draft_reqs() 复用 PrefillAdder 的 admission 语义（替代手写预算）。
  5. _add_req_to_draft_batch() / _build_decode_batch_from_reqs() 消除 v1 中 ~80 行手工
     tensor 构建代码（_extend_batch_tensors / _init_batch_tensors）。
  6. _check_and_pause_draft_req() 增加幂等性守卫，防止双重 _send_draft_response。
  7. save/restore last_batch 防止 Draft phase 干扰 Normal 的 get_next_batch_to_run()。
  8. _extract_paused_drafts_from_running() 仅做搬迁，不重复触发 pause check。

架构:
    waiting_queue         = [Normal reqs]      ← Normal 专用
    running_batch         = [Normal reqs]      ← Normal 专用（混合模式时临时含 Draft）

    draft_waiting_queue   = [Draft reqs]       ← Draft 专用（待 prefill）
    draft_batch           = [Draft reqs]       ← Draft 专用（decode 中）
    draft_paused_reqs     = [Draft reqs]       ← Draft 专用（已 pause，等待 Target）

新增 server_args 参数（如未定义则使用默认值）:
    remote_speculative_draft_priority: bool = False
        True  → Draft 优先模式（独立 decode）
        False → 混合模式（合并 decode，默认）
    remote_speculative_max_draft_priority_steps: int = 0
        0 → 自动计算（Draft 批次中剩余步数最大值）
        >0 → 上限步数（防止 Normal 饥饿）
"""
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import PrefillAdder, AddReqResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.layers.sampler import SamplingBatchInfo
from sglang.srt.speculative.remote_spec.remote_spec_protocol import (
    RemoteSpecRequest,
    RemoteSpecAction,
    SpecType,
)
from sglang.srt.speculative.remote_spec.draft_end.remote_spec_state_mamager import (
    RemoteSpecDraftState,
    RemoteSpecDraftStateManager,
)
from sglang.srt.speculative.remote_spec.draft_end.remote_spec_kv_rollbacker import (
    RemoteSpecKVRollbacker,
)
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Draft request location enum
# ---------------------------------------------------------------------------

class DraftReqLocation(str, Enum):
    """Explicit location states for Draft requests (replaces string-based location)."""
    DRAFT_WAITING = "draft_waiting"  # in draft_waiting_queue, waiting for prefill
    DRAFT_BATCH   = "draft_batch"    # in draft_batch, being decoded
    PAUSED        = "paused"         # in draft_paused_reqs, waiting for Target's next round


# ---------------------------------------------------------------------------
# Helper: fix SamplingParams stop_strs / stop_regex_strs
# ---------------------------------------------------------------------------

def _fix_sampling_params_stop_strs(sp) -> None:
    """Ensure stop_strs and stop_regex_strs are lists (not None or str)."""
    if not hasattr(sp, 'stop_strs') or sp.stop_strs is None:
        sp.stop_strs = []
    elif isinstance(sp.stop_strs, str):
        sp.stop_strs = [sp.stop_strs]

    if not hasattr(sp, 'stop_regex_strs') or sp.stop_regex_strs is None:
        sp.stop_regex_strs = []
    elif isinstance(sp.stop_regex_strs, str):
        sp.stop_regex_strs = [sp.stop_regex_strs]


# ---------------------------------------------------------------------------
# Main mixin class
# ---------------------------------------------------------------------------

class RemoteSpecDraftSchedulerMixinV2:
    """
    Scheduler mixin v2 for Draft-side remote speculative decoding.

    Implements draft_batch_design.md v0.5 (Draft Batch 分离 + Draft 优先调度).

    Required scheduler attributes (provided by the concrete Scheduler class):
        zmq_communicator           ZMQ communication worker (rank 0 only)
        server_args                ServerArgs instance
        running_batch              Normal-request decode batch (ScheduleBatch)
        waiting_queue              Normal-request prefill queue (List[Req])
        tree_cache                 RadixCache instance
        token_to_kv_pool_allocator KV pool allocator
        req_to_token_pool          Request → token pool mapping
        model_config               ModelConfig
        tokenizer                  Tokenizer
        tp_rank, tp_size           Tensor-parallel rank and world size
        tp_group, tp_cpu_group     TP process groups
        policy                     SchedulePolicy (for calc_priority)
        new_token_ratio            Current new-token ratio
        max_prefill_tokens         Per-batch prefill token budget
        decode_mem_cache_buf_multiplier  Decode memory buffer multiplier
        enable_overlap             Whether overlap scheduling is on
        spec_algorithm             SpeculativeAlgorithm enum value
        page_size                  KV cache page size
        pp_size                    Pipeline-parallel world size
    """

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_draft_components(self) -> None:
        """Initialize all Draft-specific components and data structures."""
        # State manager (single source of truth for per-request metadata)
        self.draft_state_manager = RemoteSpecDraftStateManager(timeout_threshold=60.0)

        # KV rollbacker (local rollback + delayed cache strategy)
        self.draft_kv_manager = RemoteSpecKVRollbacker(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
            page_size=self.server_args.page_size or 1,
            tp_rank=self.tp_rank,
        )

        # ── NEW v2: separate Draft management-plane data structures ──────────
        # Requests waiting for prefill (equivalent to waiting_queue for Normal)
        self.draft_waiting_queue: List[Req] = []
        # Requests currently being decoded (equivalent to running_batch for Normal)
        self.draft_batch: ScheduleBatch = ScheduleBatch(reqs=[])
        # Requests that have been paused (sent draft tokens, waiting for Target)
        self.draft_paused_reqs: List[Req] = []
        # Last draft batch (reserved for future use / compatibility)
        self.last_draft_batch: Optional[ScheduleBatch] = None
        # ─────────────────────────────────────────────────────────────────────

        # Pending-batch: reqs to be added to draft_batch in bulk at the end of
        # recv_and_process_draft_requests().  Using a deferred accumulation
        # avoids calling _build_decode_batch_from_reqs([req]) + SamplingBatchInfo
        # per-request (K×GPU-tensor-alloc), replacing it with ONE bulk build.
        self._draft_batch_pending_adds: List[Req] = []

        self.draft_forward_cycle: int = 0

    # =========================================================================
    # State Management helpers (thin wrappers around draft_state_manager)
    # =========================================================================

    def _get_draft_state(self, req_id: str) -> Optional[RemoteSpecDraftState]:
        return self.draft_state_manager.get_state(req_id)

    def _set_draft_state(self, req_id: str, state: RemoteSpecDraftState) -> None:
        self.draft_state_manager.set_state(req_id, state)

    def _delete_draft_state(self, req_id: str) -> bool:
        return self.draft_state_manager.delete(req_id)

    def _exists_draft_state(self, req_id: str) -> bool:
        return self.draft_state_manager.exists(req_id)

    # =========================================================================
    # Main Event Loop
    # =========================================================================

    @DynamicGradMode()
    def event_loop_normal_remote_spec_draft(self) -> None:
        """Main event loop for Draft server (v2).

        Replaces v1's event_loop_normal_remote_spec_draft.
        Supports two execution strategies controlled by
        server_args.remote_speculative_draft_priority.
        """
        self.last_batch = None
        self._init_draft_components()

        # Read switch once to avoid repeated attribute lookup
        draft_priority: bool = self.server_args.remote_speculative_draft_priority

        while True:
            # ── Shared: receive Normal + Draft requests ───────────────────────
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            self.recv_and_process_draft_requests()

            # ── Shared: Draft Prefill ─────────────────────────────────────────
            # v0.5: save/restore last_batch so Draft prefill batch does NOT
            # contaminate Normal's get_next_batch_to_run() logic.
            saved_last_batch = self.last_batch
            if self.draft_waiting_queue:
                self._prefill_draft_reqs()
            self.last_batch = saved_last_batch

            # ── Branch: execution strategy ────────────────────────────────────
            if draft_priority:
                # Phase 1: draft_batch decodes independently until all paused
                # start_time = time.perf_counter()
                self._run_draft_priority_phase()
                # end_time = time.perf_counter()
                # logger.debug(f"\033[31m [DraftV2] Draft priority phase took {(end_time - start_time) * 1000} ms \033[0m")
            else:
                # Mixed: merge draft_batch into running_batch for combined decode
                self._merge_draft_into_running()

            # ── Shared: Normal scheduling ─────────────────────────────────────
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
                self.draft_forward_cycle += 1
            else:
                self.self_check_during_idle()

            self.last_batch = batch

            # ── Mixed mode: extract hook-marked paused draft reqs ─────────────
            # v0.5: process_batch_result_decode hook is the ONLY pause-check
            # entry.  Here we only migrate already-marked reqs.
            if not draft_priority:
                self._extract_paused_drafts_from_running()

            # Periodic cleanup of timed-out draft states
            if self.draft_forward_cycle % 500 == 0:
                self._cleanup_stale_draft_states()

    # =========================================================================
    # Draft Priority Phase (Phase 1)
    # =========================================================================

    def _run_draft_priority_phase(self) -> None:
        """Phase 1: Draft batch decodes independently until all reqs are paused.

        v0.5 protections:
        - save/restore last_batch: Draft phase must not affect Normal's
          last_batch semantics (prevents Draft prefill batch being merged
          into running_batch by a subsequent get_next_batch_to_run call).
        - max_draft_priority_steps upper limit: prevents Normal starvation.
        - mid-recv after each decode step: prevents socket buffer buildup
          and new Draft messages timing out.
        """
        saved_last_batch = self.last_batch

        # Filter already-paused reqs before starting
        self._filter_draft_batch()

        if self.draft_batch.is_empty():
            self.last_batch = saved_last_batch
            return

        # Compute decode steps upper limit
        remaining_steps = max(
            (r.draft_tokens_target - (len(r.output_ids) - r.draft_generation_start_len))
            for r in self.draft_batch.reqs
            if not getattr(r, 'draft_is_paused', False)
        )
        max_steps = self.server_args.remote_speculative_max_draft_priority_steps
        if max_steps <= 0:
            max_steps = remaining_steps
        steps_taken = max(1, min(remaining_steps, max_steps))

        for _step in range(steps_taken):
            self._filter_draft_batch()
            if self.draft_batch.is_empty():
                break

            # Memory check before decode
            if not self.draft_batch.check_decode_mem(self.decode_mem_cache_buf_multiplier):
                self._handle_draft_batch_oom()
                break

            # Decode one step
            self.draft_batch.prepare_for_decode()
            result = self.run_batch(self.draft_batch)
            self._process_draft_decode_result(self.draft_batch, result)

            # Memory feedback: sync new_token_ratio and filter stale reqs
            self._update_draft_batch_after_decode()

            # ── Mid-recv ─────────────────────────────────────────────────────
            # Accept new Normal + Draft messages during Phase 1 (CPU only,
            # no GPU trigger).  This prevents socket buffer overflow and
            # ensures Phase 2 starts with a populated waiting_queue.
            # mid_recv_reqs = self.recv_requests()
            # self.process_input_requests(mid_recv_reqs)
            # self.recv_and_process_draft_requests()

            # # Mid-recv may produce new draft_waiting_queue entries → prefill now
            # if self.draft_waiting_queue:
            #     self._prefill_draft_reqs()
            # ─────────────────────────────────────────────────────────────────

        # v0.5: restore last_batch so Phase 2 Normal scheduling is unaffected
        self.last_batch = saved_last_batch

    def _process_draft_decode_result(self, batch: ScheduleBatch, result) -> None:
        """v0.5: reuse generic decode result processing.

        Pause check is handled by the hook inside process_batch_result_decode.
        After processing, filter out paused reqs from draft_batch.
        """
        self.process_batch_result_decode(batch, result)
        # Draft priority mode: remove hook-marked paused reqs from draft_batch
        self._filter_draft_batch()

    def _update_draft_batch_after_decode(self) -> None:
        """Memory feedback after a draft decode step.

        Simplified version: just filter finished/paused reqs.
        TODO (stage B): add retract_decode for draft_batch if OOM pressure.
        """
        self._filter_draft_batch()

    def _handle_draft_batch_oom(self) -> None:
        """Handle out-of-memory in draft batch by aborting all draft reqs."""
        if self.tp_rank == 0:
            logger.warning("[DraftV2] Draft batch OOM, aborting all draft reqs in batch")
        for req in list(self.draft_batch.reqs):
            try:
                self._finish_draft_request(req.rid)
            except Exception as e:
                if self.tp_rank == 0:
                    logger.error(f"[DraftV2] Failed to finish {req.rid} during OOM: {e}")
        self.draft_batch = ScheduleBatch(reqs=[])

    def _filter_draft_batch(self) -> None:
        """Remove finished and paused reqs from draft_batch.

        Paused reqs are moved to draft_paused_reqs (if not already there).
        This is called after each decode step and before each new step.
        """
        if self.draft_batch.is_empty():
            return

        keep_indices: List[int] = []
        for i, req in enumerate(self.draft_batch.reqs):
            if getattr(req, 'draft_is_paused', False):
                # Ensure paused req is in draft_paused_reqs
                if req not in self.draft_paused_reqs:
                    self.draft_paused_reqs.append(req)
                # Exclude from keep_indices (will be filtered out)
            elif req.finished():
                # Finished reqs are dropped; lifecycle cleanup already done elsewhere
                pass
            else:
                keep_indices.append(i)

        if len(keep_indices) < len(self.draft_batch.reqs):
            if keep_indices:
                self.draft_batch.filter_batch(keep_indices=keep_indices)
            else:
                self.draft_batch = ScheduleBatch(reqs=[])

    # =========================================================================
    # Mixed Mode: Merge / Extract
    # =========================================================================

    def _merge_draft_into_running(self) -> None:
        """Mixed mode: merge active draft_batch reqs into running_batch.

        Reuses standard ScheduleBatch.merge_batch() to handle all tensor
        concatenation correctly (eliminates ~53-line _extend_batch_tensors).

        After merge, draft_batch is reset to empty.
        """
        self._filter_draft_batch()
        if self.draft_batch.is_empty():
            return

        if self.running_batch.is_empty():
            # Replace empty running_batch with draft_batch directly
            self.running_batch = self.draft_batch
        else:
            self.running_batch.merge_batch(self.draft_batch)

        self.draft_batch = ScheduleBatch(reqs=[])

    def _extract_paused_drafts_from_running(self) -> None:
        """Mixed mode: move hook-marked paused draft reqs out of running_batch.

        v0.5 key: this method ONLY migrates reqs (does NOT check pause conditions
        and does NOT call _send_draft_response).  Pause check is done exclusively
        by the process_batch_result_decode hook in scheduler_output_processor_mixin.

        Flow:
            1. Scan running_batch for reqs with draft_is_paused=True
            2. Add to draft_paused_reqs, update state.location
            3. Remove from running_batch via filter_batch(keep_indices=)
        """
        if self.running_batch.is_empty():
            return

        paused_indices: List[int] = []
        for i, req in enumerate(self.running_batch.reqs):
            if not getattr(req, 'draft_is_paused', False):
                continue
            if req not in self.draft_paused_reqs:
                self.draft_paused_reqs.append(req)
            state = self._get_draft_state(req.rid)
            if state:
                state.location = DraftReqLocation.PAUSED
            paused_indices.append(i)

        if paused_indices:
            paused_set = set(paused_indices)
            keep = [i for i in range(len(self.running_batch.reqs)) if i not in paused_set]
            self.running_batch.filter_batch(keep_indices=keep)

    # =========================================================================
    # Draft Prefill
    # =========================================================================

    def _prefill_draft_reqs(self) -> None:
        """Prefill draft requests from draft_waiting_queue.

        Uses PrefillAdder for admission control (consistent with main scheduler,
        avoids hand-written budget formulas that can drift from main scheduling).

        v0.5 parameter rationale for _build_prefill_adder_for_draft():
        - running_batch = self.running_batch (Normal batch, for eviction calc)
        - mixed_with_decode_tokens = 0 (draft prefill is a standalone batch)
        - priority_scheduling_preemption_threshold = 0 (no preemption)
        """
        if not self.draft_waiting_queue:
            return

        # Initialize next-round inputs via match_prefix (consistent with main scheduler)
        for req in self.draft_waiting_queue:
            req.init_next_round_input(self.tree_cache)

        # Admission control via PrefillAdder
        adder = self._build_prefill_adder_for_draft()
        for req in self.draft_waiting_queue:
            res = adder.add_one_req(
                req,
                has_chunked_req=False,
                truncation_align_size=getattr(self, 'truncation_align_size', None),
            )
            if res != AddReqResult.CONTINUE:
                break

        admitted: List[Req] = adder.can_run_list
        admitted_set = set(id(r) for r in admitted)
        self.draft_waiting_queue = [
            r for r in self.draft_waiting_queue if id(r) not in admitted_set
        ]
        if not admitted:
            return

        # Build + run prefill batch (reuses standard ScheduleBatch.init_new path)
        draft_prefill_batch = ScheduleBatch.init_new(
            admitted,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        draft_prefill_batch.prepare_for_extend()
        result = self.run_batch(draft_prefill_batch)
        self._process_draft_prefill_result(draft_prefill_batch, result)

        # Filter finished reqs (shouldn't happen for draft; max_new_tokens is large)
        # filter_batch() also removes draft_is_paused reqs via the special case in
        # schedule_batch.py, but freshly prefilled draft reqs have draft_is_paused=False.
        draft_prefill_batch.filter_batch()

        if not draft_prefill_batch.is_empty():
            if self.draft_batch.is_empty():
                self.draft_batch = draft_prefill_batch
            else:
                self.draft_batch.merge_batch(draft_prefill_batch)

            # Update location for newly admitted reqs
            for req in draft_prefill_batch.reqs:
                state = self._get_draft_state(req.rid)
                if state:
                    state.location = DraftReqLocation.DRAFT_BATCH

    def _process_draft_prefill_result(
        self, batch: ScheduleBatch, result
    ) -> None:
        """Process prefill result for draft batch.

        Reuses the generic process_batch_result_prefill path.
        Draft reqs have stream=False and grammar=None, so stream_output and
        grammar checks are no-ops.  The req won't finish (max_new_tokens is
        large), so finish handling is also a no-op.
        """
        self.process_batch_result_prefill(batch, result)

    def _build_prefill_adder_for_draft(self) -> PrefillAdder:
        """Build a PrefillAdder for Draft request admission control.

        v0.5 parameters:
        - running_batch = self.running_batch:
            Covers Normal decode headroom.  Draft batch headroom is small
            (draft_tokens_target = 5-8) and is acceptable to omit here.
        - mixed_with_decode_tokens = 0:
            Draft prefill runs as a standalone batch (not mixed-chunk).
        - priority_scheduling_preemption_threshold = 0:
            Draft prefill must NOT evict Normal decode reqs.
        - rem_chunk_tokens = None:
            No chunked prefill for draft (inputs are usually short).
        """
        return PrefillAdder(
            page_size=self.page_size,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            running_batch=self.running_batch,
            new_token_ratio=self.new_token_ratio,
            rem_input_tokens=self.max_prefill_tokens,
            rem_chunk_tokens=None,
            mixed_with_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )

    # =========================================================================
    # Batch Building Helpers (replace ~80-line _extend_batch_tensors/_init_batch_tensors)
    # =========================================================================

    def _add_req_to_draft_batch(self, req: Req) -> None:
        """Add a KV-cached req to draft_batch for decode.

        Builds a temporary single-req ScheduleBatch and merges it into
        draft_batch.  Reuses merge_batch() for all tensor concatenation.

        v0.5 design note (section 6.7): this approach reduces the batch-building
        code from ~80 lines to ~20 lines while maintaining correctness.
        """
        tmp_batch = self._build_decode_batch_from_reqs([req])
        if self.draft_batch.is_empty():
            self.draft_batch = tmp_batch
        else:
            self.draft_batch.merge_batch(tmp_batch)

    def _build_decode_batch_from_reqs(self, reqs: List[Req]) -> ScheduleBatch:
        """Build a decode-ready ScheduleBatch from reqs that already have KV cache.

        These reqs have already been through at least one prefill, so they have:
        - req_pool_idx set
        - origin_input_ids and output_ids populated
        - kv_committed_len tracking their KV state

        The resulting batch can be passed to merge_batch() or used as draft_batch.
        """
        try:
            device = self.tp_group.device
        except Exception:
            device = "cuda"

        def _seq_len(r: Req) -> int:
            return len(r.origin_input_ids) + len(r.output_ids) - 1

        seq_lens_list = [_seq_len(r) for r in reqs]

        # Detect SWA allocator for is_hybrid_swa flag
        try:
            from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
            is_hybrid_swa = isinstance(self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator)
        except ImportError:
            is_hybrid_swa = False

        batch = ScheduleBatch(
            reqs=list(reqs),
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            is_hybrid_swa=is_hybrid_swa,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
            forward_mode=ForwardMode.DECODE,
            device=device,
        )

        # Core decode tensors
        batch.req_pool_indices = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int64, device=device
        )
        batch.seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        batch.out_cache_loc = None
        batch.seq_lens_sum = sum(seq_lens_list)
        # output_ids: last generated token (input for next decode step)
        batch.output_ids = torch.tensor(
            [r.output_ids[-1] if r.output_ids else r.origin_input_ids[-1] for r in reqs],
            dtype=torch.int64,
            device=device,
        )

        # Logprob fields
        batch.return_logprob = any(r.return_logprob for r in reqs)
        batch.top_logprobs_nums = [
            r.top_logprobs_num if r.return_logprob else 0 for r in reqs
        ]
        batch.token_ids_logprobs = [
            r.token_ids_logprob if r.return_logprob else None for r in reqs
        ]

        # Multimodal / flag fields
        batch.multimodal_inputs = [r.multimodal_inputs for r in reqs]
        batch.has_stream = any(r.stream for r in reqs)
        batch.has_grammar = any(r.grammar for r in reqs)
        batch.return_hidden_states = any(r.return_hidden_states for r in reqs)

        # SamplingBatchInfo (required by merge_batch and filter_batch)
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )

        return batch

    # =========================================================================
    # Message Processing
    # =========================================================================

    def recv_and_process_draft_requests(self) -> None:
        """Receive and process draft requests from Target server.

        TP-aware: only rank 0 receives ZMQ messages; messages are broadcast
        to other ranks before processing.
        """
        # Guard: check ZMQ availability for single-TP case
        if self.tp_size == 1:
            if not hasattr(self, 'zmq_communicator') or self.zmq_communicator is None:
                return

        # Reject new requests if server is under high load
        if self._is_self_high_overhead_draft():
            if self.tp_size == 1 or self.tp_rank == 0:
                self._send_reject_message()
            return

        # Rank 0: receive messages from ZMQ
        if self.tp_size == 1 or self.tp_rank == 0:
            messages = self._recv_draft_requests()
            if messages:
                logger.debug(
                    f"\033[32m [DraftV2][Recv] {len(messages)} messages from Target \033[0m"
                )
        else:
            messages = None

        # TP: broadcast messages to all ranks
        if self.tp_size > 1:
            messages = broadcast_pyobj(
                messages if messages else [],
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )

        if not messages:
            return

        control_msgs, latest_msgs = self.deduplicate_draft_requests(messages)

        if not control_msgs and not latest_msgs:
            return

        self.token_to_kv_pool_allocator.free_group_begin()
        self._process_control_message(control_msgs)
        self._process_draft_requests(latest_msgs)
        self.token_to_kv_pool_allocator.free_group_end()

        # Bulk-add all reqs collected during _process_draft_requests into
        # draft_batch in ONE _build_decode_batch_from_reqs call to avoid
        # K×SamplingBatchInfo.from_schedule_batch overhead.
        self._flush_draft_batch_pending_adds()

    def _recv_draft_requests(self) -> List[RemoteSpecRequest]:
        """Receive all pending draft requests from ZMQ communicator."""
        try:
            msgs: List[RemoteSpecRequest] = []
            if hasattr(self, 'zmq_communicator') and self.zmq_communicator is not None:
                msgs = self.zmq_communicator.recv_all_objs()
                if msgs:
                    # Poll once more to collect messages that arrived in the same
                    # batch but were split across TCP segments / IO thread timing.
                    more = self.zmq_communicator.recv_all_objs()
                    if more:
                        msgs.extend(more)
            return msgs
        except (ConnectionError, OSError) as e:
            if self.tp_rank == 0:
                logger.error(f"[DraftV2] Network error in recv: {e}", exc_info=True)
            return []
        except Exception as e:
            if self.tp_rank == 0:
                logger.error(f"[DraftV2] Unexpected recv error: {e}", exc_info=True)
            return []

    def deduplicate_draft_requests(
        self,
        messages: List[RemoteSpecRequest],
    ) -> Tuple[List[RemoteSpecRequest], Dict[str, RemoteSpecRequest]]:
        """Deduplicate messages: keep only the latest spec_cnt per request.

        Returns:
            (control_msgs, latest_draft_msgs)
            control_msgs       FINISH / ABORT messages (all kept, order-preserved)
            latest_draft_msgs  Dict[req_id → latest RemoteSpecRequest]
        """
        latest_msgs: Dict[str, RemoteSpecRequest] = {}
        control_msgs: List[RemoteSpecRequest] = []

        for draft_req in messages:
            req_id = draft_req.request_id
            action = getattr(draft_req, 'action', RemoteSpecAction.DRAFT)

            if action in (RemoteSpecAction.FINISH, RemoteSpecAction.ABORT):
                control_msgs.append(draft_req)
                continue

            # Keep the message with the highest spec_cnt.
            # If the newer message has no input_ids, carry them over from the
            # older message so that _create_new_draft_req can still initialize.
            if req_id not in latest_msgs or draft_req.spec_cnt > latest_msgs[req_id].spec_cnt:
                if req_id in latest_msgs and draft_req.input_ids is None:
                    draft_req.input_ids = latest_msgs[req_id].input_ids
                    draft_req.sampling_params = (
                        draft_req.sampling_params or latest_msgs[req_id].sampling_params
                    )
                latest_msgs[req_id] = draft_req

        total = len(messages)
        kept = len(latest_msgs) + len(control_msgs)
        if total > kept and self.tp_rank == 0:
            logger.debug(f"\033[36m [DraftV2][Dedup] {total} → {kept} \033[0m")

        return control_msgs, latest_msgs

    def _process_control_message(self, control_msgs: List[RemoteSpecRequest]) -> None:
        """Process FINISH / ABORT control messages from Target."""
        for draft_req in control_msgs:
            action = draft_req.action
            if action in (RemoteSpecAction.FINISH, RemoteSpecAction.ABORT):
                if self.tp_rank == 0:
                    logger.debug(f"[DraftV2] Received {action} for {draft_req.request_id}")
                self._finish_draft_request(draft_req.request_id)

    def _process_draft_requests(
        self, latest_msgs: Dict[str, RemoteSpecRequest]
    ) -> None:
        """Process draft update messages from Target.

        For each message:
        - If no existing state → create new request (goes to draft_waiting_queue)
        - If existing state → compare tokens, handle divergence or resume
        """
        for req_id, draft_req in latest_msgs.items():
            try:
                state = self._get_draft_state(req_id)

                if state is None:
                    if draft_req.input_ids is None:
                        if self.tp_rank == 0:
                            logger.warning(
                                f"[DraftV2] {req_id}: no state and no input_ids, skipping"
                            )
                        continue
                    self._create_new_draft_req(draft_req)
                    continue

                state.last_updated_time = time.time()
                req = state.req_object

                if req is None:
                    if self.tp_rank == 0:
                        logger.warning(f"[DraftV2] {req_id} has None req_object")
                    self._finish_draft_request(req_id)
                    continue

                req.target_send_time = draft_req.target_send_time
                req.draft_recv_time = draft_req.draft_recv_time

                # Resolve target's token sequence
                if draft_req.input_ids is not None:
                    target_input = draft_req.input_ids
                    state.target_origin_input_ids = list(target_input)
                else:
                    target_input = state.target_origin_input_ids or []

                target_fill_ids: List[int] = (
                    target_input
                    + (draft_req.output_ids or [])
                    + (draft_req.draft_token_ids or [])
                )
                draft_fill_ids: List[int] = (
                    (req.origin_input_ids or []) + (req.output_ids or [])
                )

                if not target_fill_ids:
                    continue

                # Skip the shared prompt prefix in fork-point comparison:
                # target_origin_input_ids never changes, so the first
                # len(target_input) tokens are guaranteed identical.
                skip = len(target_input)
                if skip > 0 and len(draft_fill_ids) >= skip and len(target_fill_ids) >= skip:
                    is_identical, fork_offset = self._find_fork_point(
                        draft_fill_ids[skip:], target_fill_ids[skip:]
                    )
                    fork_point = skip + fork_offset
                else:
                    is_identical, fork_point = self._find_fork_point(
                        draft_fill_ids, target_fill_ids
                    )

                if is_identical:
                    self._handle_identical_tokens(req, draft_req, state)
                else:
                    self._handle_divergence(req, target_fill_ids, fork_point, draft_req, state)

            except Exception as e:
                if self.tp_rank == 0:
                    logger.error(
                        f"\033[31m [DraftV2] Error processing {req_id}: {e} \033[0m",
                        exc_info=True,
                    )
                try:
                    self._finish_draft_request(req_id)
                except Exception:
                    pass

    # =========================================================================
    # Divergence Handling (same logic as v1; updated for new data structures)
    # =========================================================================

    def _find_fork_point(
        self,
        draft_ids: List[int],
        target_ids: List[int],
    ) -> Tuple[bool, int]:
        """Find the first position where the two sequences differ.

        Returns:
            (is_identical, fork_point)
            is_identical   True iff both sequences are equal in content and length
            fork_point     Index of first differing element (or min length if one
                           is a prefix of the other)
        """
        min_len = min(len(draft_ids), len(target_ids))
        for i in range(min_len):
            if draft_ids[i] != target_ids[i]:
                return (False, i)
        return (len(draft_ids) == len(target_ids), min_len)

    def _handle_identical_tokens(
        self,
        req: Req,
        draft_req: RemoteSpecRequest,
        state: RemoteSpecDraftState,
    ) -> None:
        """Handle Case 0: token sequences are identical → resume / continue decode."""
        if self.tp_rank == 0:
            logger.debug(
                f"\033[34m [DraftV2][NoChange] {req.rid}, "
                f"spec_cnt={draft_req.spec_cnt}, location={state.location} \033[0m"
            )
        self._update_req_state(req, draft_req, state)
        # tokens_changed=False: skip _rebuild_req_in_draft_batch for DRAFT_BATCH
        # location — the req is already in draft_batch with correct tensors; only
        # the state counters (spec_cnt, draft_tokens_target, etc.) were updated
        # in-place above.  The rebuild overhead is O(batch_size) tensor ops per
        # call, so skipping it for the no-change path is a meaningful speedup.
        self._resume_or_update(req, state, tokens_changed=False)

    def _handle_divergence(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        draft_req: RemoteSpecRequest,
        state: RemoteSpecDraftState,
    ) -> None:
        """Dispatch to the appropriate divergence handler based on length comparison."""
        state.last_updated_time = time.time()
        req.skip_radix_lookup = False

        current_len = len(req.origin_input_ids) + len(req.output_ids)
        target_len = len(target_fill_ids)
        current_kv_len = current_len - 1
        needs_kv_release = fork_point < current_kv_len

        if current_len == target_len:
            self._handle_equal_length(
                req, target_fill_ids, fork_point, current_len, current_kv_len,
                needs_kv_release, draft_req, state,
            )
        elif current_len > target_len:
            self._handle_draft_ahead(
                req, target_fill_ids, fork_point, current_kv_len,
                needs_kv_release, draft_req, state,
            )
        else:
            self._handle_target_ahead(
                req, target_fill_ids, fork_point, current_len, current_kv_len,
                needs_kv_release, draft_req, state,
            )

    def _handle_equal_length(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_len: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: RemoteSpecRequest,
        state: RemoteSpecDraftState,
    ) -> None:
        """Case 1: equal-length sequences."""
        if fork_point == current_len - 1:
            # Case 1.1: Only last token differs → replace it and decode
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 1.1] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"replace last token \033[0m"
                )
            self._update_tokens(req, fork_point, target_fill_ids[fork_point:])
            self._update_req_state(req, draft_req, state)
            self._resume_or_update(req, state)
        else:
            # Case 1.2: Multiple tokens differ
            self._handle_multi_token_divergence(
                req, target_fill_ids, fork_point, current_kv_len,
                needs_kv_release, draft_req, state, "1.2",
            )

    def _handle_draft_ahead(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: RemoteSpecRequest,
        state: RemoteSpecDraftState,
    ) -> None:
        """Case 2: Draft has more tokens than Target."""
        target_len = len(target_fill_ids)

        if fork_point == target_len:
            # Case 2.1: Target is a strict prefix of Draft → Draft is ahead
            target_output_len = target_len - len(req.origin_input_ids)
            draft_output_len = len(req.output_ids)
            tokens_ahead = draft_output_len - target_output_len

            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 2.1] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"draft ahead by {tokens_ahead} token(s) \033[0m"
                )

            req.draft_generation_start_len = target_output_len
            req.spec_cnt = draft_req.spec_cnt
            req.draft_tokens_target = draft_req.num_draft_tokens
            req.len_output_ids = draft_output_len
            state.last_updated_time = time.time()

            if tokens_ahead >= draft_req.num_draft_tokens:
                # Already have enough tokens → send immediately and pause
                if self.tp_rank == 0:
                    logger.debug(
                        f"\033[33m [Case 2.1] {req.rid=}: already {tokens_ahead} ahead, "
                        f"sending immediately \033[0m"
                    )
                self._send_draft_response(req)
                self._pause_req(req, state)
            else:
                # Need more tokens → resume / continue decode
                req.draft_is_paused = False
                self._resume_or_update(req, state)
        else:
            # Case 2.2: Tokens differ (not a clean prefix relationship)
            self._handle_multi_token_divergence(
                req, target_fill_ids, fork_point, current_kv_len,
                needs_kv_release, draft_req, state, "2.2",
            )

    def _handle_target_ahead(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_len: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: RemoteSpecRequest,
        state: RemoteSpecDraftState,
    ) -> None:
        """Case 3: Target has more tokens than Draft → re-prefill for extend."""
        if fork_point == current_len:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 3.1] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"re-prefill for extend \033[0m"
                )
        else:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case 3.2] {req.rid=}, {draft_req.spec_cnt=}, "
                    f"re-prefill for extend+rollback \033[0m"
                )
        self._prepare_for_reprefill(req, target_fill_ids, draft_req, state)

    def _handle_multi_token_divergence(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: RemoteSpecRequest,
        state: RemoteSpecDraftState,
        case_name: str,
    ) -> None:
        """Handle multi-token divergence (Cases 1.2 and 2.2).

        Strategy:
        - If local rollback is possible AND decoding can start right after
          fork_point, do local rollback + decode (fast path).
        - Otherwise fall back to re-prefill (handles RadixCache-region
          divergence and extend scenarios).
        """
        new_len = len(target_fill_ids)
        can_decode_after_rollback = (new_len - 1 <= fork_point)

        if (
            can_decode_after_rollback
            and self.draft_kv_manager.can_local_rollback(req, fork_point)
        ):
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[35m [Case {case_name}] {req.rid=}, {draft_req.spec_cnt=} "
                    f"→ local rollback + decode \033[0m"
                )
            if needs_kv_release:
                self.draft_kv_manager.local_rollback(req, fork_point, current_kv_len)
            self._update_tokens(req, fork_point, target_fill_ids[fork_point:])
            self._update_req_state(req, draft_req, state)
            self._resume_or_update(req, state)
        else:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[36m [Case {case_name}] {req.rid=}, {draft_req.spec_cnt=} "
                    f"→ re-prefill \033[0m"
                )
            self._prepare_for_reprefill(req, target_fill_ids, draft_req, state)

    # =========================================================================
    # Request State Updates
    # =========================================================================

    def _update_tokens(
        self, req: Req, fork_point: int, delta_tokens: List[int]
    ) -> None:
        """Truncate req.output_ids at fork_point and append delta_tokens."""
        truncate_point = fork_point - len(req.origin_input_ids)
        req.output_ids = req.output_ids[:max(0, truncate_point)]
        req.output_ids.extend(delta_tokens)
        req.fill_ids = req.origin_input_ids + req.output_ids

    def _update_req_state(
        self,
        req: Req,
        draft_req: RemoteSpecRequest,
        state: RemoteSpecDraftState,
    ) -> None:
        """Update request decode state fields after receiving a new Target message."""
        req.spec_cnt = draft_req.spec_cnt
        req.draft_tokens_target = draft_req.num_draft_tokens
        req.draft_generation_start_len = len(req.output_ids)
        req.draft_is_paused = False
        req.len_output_ids = len(req.output_ids)
        state.last_updated_time = time.time()

    def _resume_or_update(
        self,
        req: Req,
        state: RemoteSpecDraftState,
        tokens_changed: bool = True,
    ) -> None:
        """Resume a paused request or update an in-batch request after token update.

        v2 uses state.location (DraftReqLocation enum) instead of scanning all
        data structures (eliminates _get_req_actual_location() traversal).

        Args:
            tokens_changed: True when req.output_ids was modified (rollback /
                token replacement).  False for identical-token updates where
                only state counters changed.  When False and location is
                DRAFT_BATCH, we skip _rebuild_req_in_draft_batch because
                seq_lens / output_ids tensors are still correct.
        """
        location = state.location

        if location == DraftReqLocation.PAUSED:
            # Move from draft_paused_reqs back to draft_batch (via pending list)
            self._resume_draft_req(req, state)
        elif location == DraftReqLocation.DRAFT_BATCH:
            if tokens_changed:
                # Tokens changed → seq_lens tensor is stale; must rebuild.
                self._rebuild_req_in_draft_batch(req, state)
            # else: counters only changed; seq_lens still valid → no tensor work
        # DraftReqLocation.DRAFT_WAITING: req will be prefilled soon → no action

    def _resume_draft_req(self, req: Req, state: RemoteSpecDraftState) -> None:
        """Resume a paused request: remove from draft_paused_reqs, enqueue for draft_batch.

        Does NOT call _add_req_to_draft_batch immediately.  Instead appends to
        _draft_batch_pending_adds so that all resumes in one recv cycle are
        bulk-added in a single _build_decode_batch_from_reqs call (one
        SamplingBatchInfo build for all K reqs vs K individual builds).
        """
        if req in self.draft_paused_reqs:
            self.draft_paused_reqs.remove(req)

        if req.req_pool_idx is None:
            # Edge case: KV was lost while paused (shouldn't happen normally)
            if self.tp_rank == 0:
                logger.warning(
                    f"[DraftV2][Resume] {req.rid} has no KV pool slot, "
                    f"falling back to re-prefill"
                )
            req.draft_is_paused = False
            state.location = DraftReqLocation.DRAFT_WAITING
            if req not in self.draft_waiting_queue:
                self.draft_waiting_queue.append(req)
            return

        req.draft_is_paused = False
        state.location = DraftReqLocation.DRAFT_BATCH
        if req not in self._draft_batch_pending_adds:
            self._draft_batch_pending_adds.append(req)

        if self.tp_rank == 0:
            logger.debug(f"[DraftV2][Resume] {req.rid} → draft_batch (pending)")

    def _rebuild_req_in_draft_batch(
        self, req: Req, state: RemoteSpecDraftState
    ) -> None:
        """Re-add a req that is already in draft_batch but whose tokens changed.

        Removes req from draft_batch (filter_batch), then enqueues in
        _draft_batch_pending_adds for bulk re-addition by
        _flush_draft_batch_pending_adds().
        """
        if not self.draft_batch.is_empty() and req in self.draft_batch.reqs:
            self.draft_batch.filter_batch(chunked_req_to_exclude=[req])

        if req.req_pool_idx is not None:
            req.draft_is_paused = False
            state.location = DraftReqLocation.DRAFT_BATCH
            if req not in self._draft_batch_pending_adds:
                self._draft_batch_pending_adds.append(req)
        else:
            state.location = DraftReqLocation.DRAFT_WAITING
            if req not in self.draft_waiting_queue:
                self.draft_waiting_queue.append(req)

    def _flush_draft_batch_pending_adds(self) -> None:
        """Bulk-add all pending reqs to draft_batch in one operation.

        This is the performance-critical flush point that replaces the old
        per-request _add_req_to_draft_batch() pattern.

        Performance rationale:
          Old (v2 naive): K requests × (_build_decode_batch_from_reqs([req]) +
              SamplingBatchInfo.from_schedule_batch(1-req, vocab) + merge_batch)
              → K × 5 CUDA tensor allocs + K × SamplingBatchInfo.from_schedule_batch
              → ~K × 5+ CUDA kernel launches per scheduler iteration

          New (batched): one _build_decode_batch_from_reqs(all_K_reqs) +
              one SamplingBatchInfo.from_schedule_batch(K-req, vocab) +
              at most one merge_batch
              → 5 CUDA tensor allocs + 1 SamplingBatchInfo.from_schedule_batch
              → ~5 CUDA kernel launches per scheduler iteration (K×speedup)
        """
        if not self._draft_batch_pending_adds:
            return

        new_batch = self._build_decode_batch_from_reqs(self._draft_batch_pending_adds)
        self._draft_batch_pending_adds.clear()

        if self.draft_batch.is_empty():
            self.draft_batch = new_batch
        else:
            self.draft_batch.merge_batch(new_batch)

    def _pause_req(self, req: Req, state: RemoteSpecDraftState) -> None:
        """Explicitly pause a request (Case 2.1: Draft already ahead enough).

        Marks draft_is_paused, removes from draft_batch if present,
        and moves to draft_paused_reqs.
        """
        # Remove from draft_batch if the req is currently there
        if not self.draft_batch.is_empty() and req in self.draft_batch.reqs:
            self.draft_batch.filter_batch(chunked_req_to_exclude=[req])

        req.draft_is_paused = True
        if req not in self.draft_paused_reqs:
            self.draft_paused_reqs.append(req)
        state.location = DraftReqLocation.PAUSED
        state.last_updated_time = time.time()

    def _reset_req_logprob_fields(self, req: Req) -> None:
        """Reset all logprob-related fields on req (used before re-prefill)."""
        req.input_token_logprobs_val = None
        req.input_token_logprobs_idx = None
        req.input_top_logprobs_val = None
        req.input_top_logprobs_idx = None
        req.input_token_ids_logprobs_val = None
        req.input_token_ids_logprobs_idx = None
        req.input_token_logprobs = None
        req.temp_input_top_logprobs_val = None
        req.temp_input_top_logprobs_idx = None
        req.temp_input_token_ids_logprobs_val = None
        req.temp_input_token_ids_logprobs_idx = None
        req.input_logprob_sent = False

        req.output_token_logprobs_val = []
        req.output_token_logprobs_idx = []
        req.output_top_logprobs_val = []
        req.output_top_logprobs_idx = []
        req.output_token_ids_logprobs_val = []
        req.output_token_ids_logprobs_idx = []

    # =========================================================================
    # KV Operations
    # =========================================================================

    def _remove_draft_req(self, req: Req) -> None:
        """Remove a request from all Draft data structures.

        v2 simplification: only 3 locations to check (vs 4+ in v1).
        No lock needed (draft_paused_reqs is only accessed on the main thread).
        """
        if req in self.draft_paused_reqs:
            self.draft_paused_reqs.remove(req)

        if not self.draft_batch.is_empty() and req in self.draft_batch.reqs:
            self.draft_batch.filter_batch(chunked_req_to_exclude=[req])

        if req in self.draft_waiting_queue:
            self.draft_waiting_queue.remove(req)

    def _prepare_for_reprefill(
        self,
        req: Req,
        target_fill_ids: List[int],
        draft_req: RemoteSpecRequest,
        state: RemoteSpecDraftState,
    ) -> None:
        """Prepare for re-prefill: release KV, reset req, enqueue in draft_waiting_queue.

        This handles all divergence cases that require a full new prefill:
        - Divergence in the RadixCache-managed region (fork_point < prefix_len)
        - Extend scenarios (target_len > current_len)

        v0.5 RadixCache consistency note (section 6.8): tree_cache only contains
        KV data that was inserted at the last explicit finish/prefill.
        match_prefix will match the longest valid prefix, which is safe even
        with Delayed Cache strategy.  skip_radix_lookup=False is intentional.
        """
        if self.tp_rank == 0:
            logger.debug(
                f"[DraftV2][RePrefill] {req.rid=}, {draft_req.spec_cnt=}, "
                f"new_len={len(target_fill_ids)}"
            )

        # Remove from all Draft locations first
        self._remove_draft_req(req)

        # Release KV cache (Delayed Cache: only at finish / re-prefill)
        if req.req_pool_idx is not None:
            self.draft_kv_manager.release_all_kv_for_reprefill_req(req)

        # Reset req for a fresh prefill
        req.fill_ids = target_fill_ids
        req.origin_input_ids = list(target_fill_ids)
        req.output_ids = []
        req.prefix_indices = []
        req.extend_input_len = len(req.fill_ids)
        req.skip_radix_lookup = False  # allow match_prefix (see section 6.8)

        req.spec_cnt = draft_req.spec_cnt
        req.draft_tokens_target = draft_req.num_draft_tokens
        req.draft_generation_start_len = 0
        req.draft_is_paused = False
        req.len_output_ids = 0

        req.last_node = None
        req.kv_committed_len = 0
        req.kv_committed_freed = False
        req.kv_overallocated_freed = False

        state.location = DraftReqLocation.DRAFT_WAITING
        req.logprob_start_len = len(req.origin_input_ids) - 1
        self._reset_req_logprob_fields(req)

        # Add to draft_waiting_queue (NOT the shared waiting_queue)
        self.draft_waiting_queue.append(req)

    # =========================================================================
    # Pause and Response
    # =========================================================================

    def _check_and_pause_draft_req(self, req: Req) -> bool:
        """Check if req has generated enough draft tokens and should be paused.

        Called by the hook in process_batch_result_decode
        (scheduler_output_processor_mixin.py).  This is the ONLY place where
        _send_draft_response is triggered.

        v0.5 idempotency guard: if already paused, return True immediately
        without calling _send_draft_response again.  This prevents duplicate
        responses even if the hook fires multiple times (defensive programming).
        """
        if getattr(req, 'spec_type', None) != SpecType.DRAFT_REQUEST:
            return False

        # v0.5 idempotency guard
        if req.draft_is_paused:
            return True

        tokens_generated = len(req.output_ids) - req.draft_generation_start_len

        if tokens_generated >= req.draft_tokens_target:
            self._send_draft_response(req)

            req.draft_is_paused = True

            # Move to draft_paused_reqs (if not already there)
            if req not in self.draft_paused_reqs:
                self.draft_paused_reqs.append(req)

            state = self._get_draft_state(req.rid)
            if state:
                state.location = DraftReqLocation.PAUSED
                state.last_updated_time = time.time()

            return True

        return False

    def _send_draft_response(self, req: Req) -> None:
        """Send draft tokens back to Target server via ZMQ.

        Only rank 0 sends; advances req.draft_generation_start_len afterwards
        to mark these tokens as "sent".
        """
        draft_tokens = req.output_ids[req.draft_generation_start_len:]

        draft_logits: List[float] = []
        if hasattr(req, 'output_token_logprobs_val') and req.output_token_logprobs_val:
            start = req.draft_generation_start_len
            end = start + len(draft_tokens)
            if len(req.output_token_logprobs_val) >= end:
                draft_logits = req.output_token_logprobs_val[start:end]

        response = RemoteSpecRequest(
            request_id=req.rid,
            spec_cnt=req.spec_cnt,
            action=RemoteSpecAction.DRAFT,
            spec_type=SpecType.DRAFT_RESPONSE,
            draft_token_ids=draft_tokens,
            draft_logprobs=draft_logits or [],
            target_send_time=req.target_send_time,
            draft_recv_time=req.draft_recv_time,
        )

        if self.tp_size == 1 or self.tp_rank == 0:
            if hasattr(self, 'zmq_communicator') and self.zmq_communicator is not None:
                self.zmq_communicator.send_objs([response])

        # Advance start pointer so duplicate calls don't re-send the same tokens
        req.draft_generation_start_len = len(req.output_ids)

    # =========================================================================
    # Request Lifecycle
    # =========================================================================

    def _create_new_draft_req(self, draft_req: RemoteSpecRequest) -> None:
        """Create a new draft Req from a RemoteSpecRequest and enqueue in draft_waiting_queue.

        Key v2 change: adds to self.draft_waiting_queue (not shared waiting_queue).
        """
        req_id = draft_req.request_id

        # Clean up stale state if present
        if self._exists_draft_state(req_id):
            self._finish_draft_request(req_id)

        input_ids: List[int] = (
            (draft_req.input_ids or [])
            + (draft_req.output_ids or [])
            + (draft_req.draft_token_ids or [])
        )

        # Prepare and normalize sampling_params
        if draft_req.sampling_params is None:
            from sglang.srt.sampling.sampling_params import SamplingParams
            sampling_params = SamplingParams()
        else:
            sampling_params = draft_req.sampling_params

        if hasattr(sampling_params, 'normalize'):
            try:
                sampling_params.normalize(self.tokenizer)
            except Exception as e:
                if self.tp_rank == 0:
                    logger.warning(
                        f"[DraftV2] Failed to normalize SamplingParams for {req_id}: {e}, "
                        f"applying manual fix"
                    )
                _fix_sampling_params_stop_strs(sampling_params)
        else:
            _fix_sampling_params_stop_strs(sampling_params)

        req = Req(
            rid=req_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            spec_cnt=draft_req.spec_cnt,
            spec_type=SpecType.DRAFT_REQUEST,
            return_logprob=True,
            top_logprobs_num=1,
            token_ids_logprob=None,
            stream=False,
            lora_id=None,
            input_embeds=None,
            custom_logit_processor=None,
            return_hidden_states=False,
            eos_token_ids=self.model_config.hf_eos_token_id,
            bootstrap_host=None,
            bootstrap_port=8998,
            bootstrap_room=None,
            vocab_size=self.model_config.vocab_size,
        )

        req.tokenizer = self.tokenizer
        req.logprob_start_len = len(req.origin_input_ids) - 1
        req.draft_tokens_target = draft_req.num_draft_tokens
        req.draft_generation_start_len = 0
        req.draft_is_paused = False
        req.len_output_ids = 0
        req.skip_radix_lookup = False
        req.target_send_time = draft_req.target_send_time
        req.draft_recv_time = draft_req.draft_recv_time

        # v2: use draft_waiting_queue, NOT shared waiting_queue
        self.draft_waiting_queue.append(req)

        self._set_draft_state(
            req_id,
            RemoteSpecDraftState(
                req_id=req_id,
                spec_cnt=draft_req.spec_cnt,
                req_object=req,
                location=DraftReqLocation.DRAFT_WAITING,
                target_origin_input_ids=(
                    list(draft_req.input_ids) if draft_req.input_ids else []
                ),
                last_prefix_length=len(input_ids),
                last_output_length=0,
            ),
        )

        if self.tp_rank == 0:
            logger.debug(
                f"[DraftV2][New] {req_id=}, {req.spec_cnt=}, input_len={len(input_ids)}"
            )

    def _finish_draft_request(self, req_id: str) -> None:
        """Clean up a finished/aborted draft request.

        This is the ONLY place where RadixCache is updated (Delayed Cache strategy):
        KV is flushed to RadixCache via cache_finished_req at finish time.
        """
        state = self._get_draft_state(req_id)
        if state is None:
            return

        req = state.req_object
        self._remove_draft_req(req)

        if not req.finished():
            req.to_abort = True
            req.finished_reason = FINISH_ABORT("Target request finished")

        # Delayed Cache: update RadixCache now (only at finish time)
        if req.req_pool_idx is not None and not getattr(req, 'kv_committed_freed', False):
            self.draft_kv_manager.release_all_kv_for_finished_req(req)

        self._delete_draft_state(req_id)
        if self.tp_rank == 0:
            logger.debug(f"[DraftV2][Finish] {req_id=}")

    def _cleanup_stale_draft_states(self) -> None:
        """Periodic cleanup of timed-out draft states."""
        for req_id in self.draft_state_manager.cleanup_stale_states():
            try:
                self._finish_draft_request(req_id)
            except Exception as e:
                if self.tp_rank == 0:
                    logger.warning(f"[DraftV2] Cleanup failed for {req_id=}: {e}")
            finally:
                try:
                    self._delete_draft_state(req_id)
                except Exception:
                    pass

    # =========================================================================
    # High-load detection and rejection
    # =========================================================================

    def _is_self_high_overhead_draft(self) -> bool:
        """Return True if the server is under high load and should reject new requests."""
        if not hasattr(self, 'running_batch') or self.running_batch is None:
            return False
        return self.running_batch.batch_size() > self.server_args.remote_speculative_max_batch_size

    def _send_reject_message(self) -> None:
        """Send a REJECT message to Target to indicate high load."""
        if self.tp_size > 1 and self.tp_rank != 0:
            return
        if not hasattr(self, 'zmq_communicator') or self.zmq_communicator is None:
            return

        reject_msg = RemoteSpecRequest(
            request_id="system",
            spec_cnt=0,
            action=RemoteSpecAction.REJECT,
            spec_type=SpecType.DRAFT_RESPONSE,
            draft_token_ids=[],
            draft_logprobs=[],
        )
        self.zmq_communicator.send_objs([reject_msg])
        if self.tp_rank == 0:
            logger.debug("[DraftV2] Sent REJECT to Target (high load)")

    # =========================================================================
    # get_num_allocatable_reqs override
    # =========================================================================

    def get_num_allocatable_reqs(self, running_bs: int) -> int:
        """Override to account for draft_paused_reqs in slot counting.

        v0.5 (section 8.3): use set-union to prevent double-counting during
        the migration period when both paused_reqs (v1 bridge) and
        draft_paused_reqs (v2) may coexist.
        """
        paused_id_set: set = set()

        # Bridge compatibility: include legacy paused_reqs if present
        paused_reqs_lock = getattr(self, 'paused_reqs_lock', None)
        if paused_reqs_lock is not None and hasattr(self, 'paused_reqs'):
            try:
                with paused_reqs_lock:
                    paused_id_set.update(id(r) for r in self.paused_reqs)
            except Exception:
                pass

        # v2 primary source
        if hasattr(self, 'draft_paused_reqs'):
            paused_id_set.update(id(r) for r in self.draft_paused_reqs)

        paused_count = len(paused_id_set)
        total_occupied = running_bs + paused_count

        from sglang.srt.server_args import get_global_server_args
        res = get_global_server_args().pp_max_micro_batch_size - total_occupied
        if self.pp_size > 1:
            res = min(res, self.req_to_token_pool.available_size())
        return res
