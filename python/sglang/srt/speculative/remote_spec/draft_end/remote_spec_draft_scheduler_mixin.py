"""
Draft-side Scheduler Mixin for Remote Speculative Decoding.

Strategy 5: Local Rollback + Delayed Cache
==========================================
This mixin implements an optimized KV management strategy for Draft server:

KV Memory Layout:
    [0, prefix_len)         → RadixCache-managed indices (from match_prefix)
    [prefix_len, kv_len)    → Draft-allocated indices (can be freed directly)

Key Behaviors:
    1. Local Rollback: When divergence in Draft-allocated region (fork_point >= prefix_len),
       directly free [fork_point, kv_len) without touching RadixCache.
    
    2. Delayed Cache: RadixCache is ONLY updated at request finish time.
       No RadixCache operations during decode/pause/rollback cycles.
    
    3. Re-prefill: Only when divergence in RadixCache region (fork_point < prefix_len).
       This is rare as Target's verified tokens usually stay within Draft's generation.

Divergence Cases:
    Case 1: len(draft) == len(target) (Equal length)
        1.1: Only last token differs → Replace token, decode
        1.2: Multiple tokens differ → Local rollback or re-prefill
    
    Case 2: len(draft) > len(target) (Draft ahead)
        2.1: Target is prefix → Continue decode
        2.2: Tokens differ → Local rollback or re-prefill
    
    Case 3: len(draft) < len(target) (Target ahead)
        3.1: Draft is prefix → Direct extend (optimal!)
        3.2: Tokens differ → Local rollback + extend or re-prefill
"""
import logging
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.speculative.remote_spec.remote_spec_protocol import (
    RemoteSpecRequestFromTargetToDraft as DraftRequest,
    RemoteSpecResponseFromDraftToTarget as DraftResponse,
    RemoteSpecAction,
    SpecType
)
from sglang.srt.speculative.remote_spec.draft_end.remote_spec_state_mamager import (
    RemoteSpecDraftState,
    RemoteSpecDraftStateManager,
)
from sglang.srt.speculative.remote_spec.draft_end.remote_spec_kv_rollbacker import (
    RemoteSpecKVRollbacker,
)
from sglang.srt.utils import DynamicGradMode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RemoteSpecDraftSchedulerMixin:
    """
    Scheduler mixin for Draft-side remote speculative decoding.
    
    RemoteSpecDraftSchedulerMixin is used in the draft end of remote spec.
    It is responsible for scheduling the draft requests and normal requests.
    
    Provides:
        - event_loop_normal_remote_spec_draft: Main event loop
        - Local rollback KV management (Strategy 5)
        - Draft request state management
        - Token divergence handling
    
    Required scheduler attributes:
        - smart_comm_worker: ZMQ communication worker
        - server_args: Server configuration
        - running_batch, waiting_queue, paused_reqs: Request queues
        - tree_cache: RadixCache instance
        - token_to_kv_pool_allocator, req_to_token_pool: Memory pools
        - model_config, tokenizer: Model configuration
    """
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def _init_draft_components(self):
        """Initialize Draft-specific components."""
        # Initialize RemoteSpecDraftStateManager (the single source of truth for draft states)
        self.draft_state_manager = RemoteSpecDraftStateManager(timeout_threshold=60.0)
        
        # Initialize RemoteSpecKVRollbacker
        self.draft_kv_manager = RemoteSpecKVRollbacker(
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            tree_cache=self.tree_cache,
            page_size=getattr(self.server_args, 'page_size', 1),
        )
        
        self.draft_forward_cycle = 0
    
    # =========================================================================
    # State Management
    # =========================================================================
    
    def _get_draft_state(self, req_id: str) -> Optional[RemoteSpecDraftState]:
        """Get draft state by request ID."""
        return self.draft_state_manager.get_state(req_id)
    
    def _set_draft_state(self, req_id: str, state: RemoteSpecDraftState) -> None:
        """Set draft state for request ID."""
        self.draft_state_manager.set_state(req_id, state)
    
    def _delete_draft_state(self, req_id: str) -> bool:
        """Delete draft state by request ID."""
        return self.draft_state_manager.delete(req_id)
    
    def _exists_draft_state(self, req_id: str) -> bool:
        """Check if draft state exists for request ID."""
        return self.draft_state_manager.exists(req_id)
    
    def _get_req_actual_location(self, req: Req) -> str:
        """
        Get actual location of request by checking data structures.
        
        Returns: "paused", "running_batch", "last_batch", "waiting_queue", or "unknown"
        """
        with self.paused_reqs_lock:
            if req in self.paused_reqs:
                return "paused"
        
        if not self.running_batch.is_empty() and req in self.running_batch.reqs:
            return "running_batch"
        
        if self.last_batch and not self.last_batch.is_empty() and req in self.last_batch.reqs:
            return "last_batch"
        
        if req in self.waiting_queue:
            return "waiting_queue"
        
        return "unknown"
    
    # =========================================================================
    # Main Event Loop
    # =========================================================================
    
    @DynamicGradMode()
    def event_loop_normal_remote_spec_draft(self):
        """Main event loop for Draft server."""
        self.last_batch = None
        self._init_draft_components()
        
        while True:
            # Process normal and draft requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            # Process draft requests from Target server
            self.recv_and_process_draft_requests()
            # BREAKPOINT: Uncomment to force debugger stop
            # breakpoint()  # Force breakpoint here for debugging

            # Run batch
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
                self.draft_forward_cycle += 1
            else:
                self.self_check_during_idle()
            
            self.last_batch = batch

            # Periodic cleanup
            if self.draft_forward_cycle % 1000000000000 == 0:
                self._cleanup_stale_draft_states()
    
    # =========================================================================
    # Message Processing
    # =========================================================================
    
    def recv_and_process_draft_requests(self):
        """Process draft requests from Target server."""
        if not hasattr(self, 'zmq_communicator') or self.zmq_communicator is None:
            return
        
        # Check for high load before processing requests
        if self._is_self_high_overhead_draft():
            self._send_reject_message()
            return
        
        messages = self._recv_draft_requests()
        # BREAKPOINT: Uncomment to force debugger stop when messages received
        # if messages:
        #     breakpoint()  # Force breakpoint when messages are received
        
        draft_recv_time = time.perf_counter()
        
        control_msgs, latest_msgs = self.deduplicate_draft_requests(messages, draft_recv_time)
        
        self.token_to_kv_pool_allocator.free_group_begin()
        
        self._process_control_message(control_msgs)
        reqs_to_merge = self._process_draft_requests(latest_msgs)
        
        if reqs_to_merge:
            self._merge_requests_to_batch(reqs_to_merge)
        
        self.token_to_kv_pool_allocator.free_group_end()
    
    def _recv_draft_requests(self) -> List[DraftRequest]:
        """Receive draft requests from communication worker."""
        try:
            msgs = []
            _msgs = self.zmq_communicator.recv_all_objs()
            for _msg in _msgs:
                msgs.append(DraftRequest.from_dict(_msg))
            return msgs
        except (ConnectionError, OSError) as e:
            logger.error(f"[Draft] Network error in _recv_draft_requests: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"[Draft] Unexpected error in _recv_draft_requests: {e}", exc_info=True)
            return []
    
    def deduplicate_draft_requests(
        self,
        messages: List[DraftRequest],
        draft_recv_time: float,
    ) -> Tuple[List[DraftRequest], Dict[str, DraftRequest]]:
        """
        Deduplicate messages, keeping only latest spec_cnt per request.
        
        Returns:
            Tuple of (control_messages, latest_draft_messages_dict)
        """
        latest_msgs = {}
        control_msgs = []
        
        for draft_req in messages:
            req_id = draft_req.request_id
            action = getattr(draft_req, 'action', RemoteSpecAction.DRAFT)
            
            # Classify control vs draft messages
            if action in [RemoteSpecAction.FINISH, RemoteSpecAction.ABORT]:
                control_msgs.append(draft_req)
                continue
            
            draft_req.draft_recv_time = draft_recv_time
            
            # Keep latest spec_cnt
            if req_id not in latest_msgs or draft_req.spec_cnt > latest_msgs[req_id].spec_cnt:
                latest_msgs[req_id] = draft_req
        
        # Log deduplication
        total, kept = len(messages), len(latest_msgs) + len(control_msgs)
        if total > kept:
            logger.info(f"\033[36m [Draft][Recv] Deduplicated: {total} -> {kept} \033[0m")
        
        return control_msgs, latest_msgs
    
    def _process_control_message(self, control_msgs: List[DraftRequest]) -> None:
        """Process finish/abort messages."""
        for draft_req in control_msgs:
            req_id = draft_req.request_id
            action = draft_req.action
            
            if action in [RemoteSpecAction.FINISH, RemoteSpecAction.ABORT]:
                logger.info(f"[Draft] Received {action} for {req_id}")
                self._finish_draft_request(req_id)
    
    def _process_draft_requests(self, latest_msgs: Dict[str, DraftRequest]) -> List[Req]:
        """
        Process draft messages, return requests to merge to batch.
        
        Args:
            latest_msgs: Dictionary of request_id to latest DraftRequest
            
        Returns:
            List of requests that need to be merged back to running batch
        """
        reqs_to_merge = []
        
        for req_id, draft_req in latest_msgs.items():
            try:
                state = self._get_draft_state(req_id)
                
                if state is None:
                    self._create_new_draft_req(draft_req)
                    continue
                
                state.last_updated_time = time.time()
                req = state.req_object
                
                if req is None:
                    logger.warning(f"[Draft] {req_id} has None req_object")
                    self._finish_draft_request(req_id)
                    continue
                
                # Update timestamps
                req.target_send_time = draft_req.target_send_time
                req.draft_recv_time = draft_req.draft_recv_time
                
                # Build sequences for comparison
                target_fill_ids = (
                    (draft_req.input_ids or []) + 
                    (draft_req.output_ids or []) + 
                    (draft_req.draft_token_ids or [])
                )
                draft_fill_ids = (req.origin_input_ids or []) + (req.output_ids or [])
                
                if not target_fill_ids:
                    continue
                
                # Find divergence point
                is_identical, fork_point = self._find_fork_point(draft_fill_ids, target_fill_ids)
                
                # logger.info(
                #     f"\033[34m [find fork point] {req_id=} {is_identical=}, {fork_point=}, "
                #     f"{len(draft_fill_ids)=}, {len(target_fill_ids)=} \033[0m"
                # )
                
                if is_identical:
                    self._handle_identical_tokens(req, draft_req, state, reqs_to_merge)
                else:
                    self._handle_divergence(
                        req, target_fill_ids, fork_point, draft_req, state, reqs_to_merge
                    )
                
            except Exception as e:
                logger.error(f"\033[31m [Draft] Error processing {req_id}: {e} \033[0m")
                try:
                    self._finish_draft_request(req_id)
                except:
                    pass
        
        return reqs_to_merge
    
    # =========================================================================
    # Divergence Handling
    # =========================================================================
    
    def _find_fork_point(
        self,
        draft_ids: List[int],
        target_ids: List[int],
    ) -> Tuple[bool, int]:
        """Find first difference between sequences."""
        min_len = min(len(draft_ids), len(target_ids))
        
        for i in range(min_len):
            if draft_ids[i] != target_ids[i]:
                return (False, i)
        
        return (len(draft_ids) == len(target_ids), min_len)
    
    def _handle_identical_tokens(
        self,
        req: Req,
        draft_req: DraftRequest,
        state: RemoteSpecDraftState,
        reqs_to_merge: List[Req],
    ) -> None:
        """Handle case where tokens are identical."""
        logger.info(
            f"\033[34m [Draft][No Change] {req.rid}, "
            f"spec_cnt={draft_req.spec_cnt}, location={state.location} \033[0m"
        )
        
        self._update_req_state(req, draft_req, state)
        self._resume_or_update(req, state, reqs_to_merge)
    
    def _handle_divergence(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        draft_req: DraftRequest,
        state: RemoteSpecDraftState,
        reqs_to_merge: List[Req],
    ) -> None:
        """Handle token divergence with local rollback or re-prefill."""
        state.last_updated_time = time.time()
        req.skip_radix_lookup = False
        
        current_len = len(req.origin_input_ids) + len(req.output_ids)
        target_len = len(target_fill_ids)
        current_kv_len = current_len - 1
        needs_kv_release = fork_point < current_kv_len
        
        if current_len == target_len:
            self._handle_equal_length(
                req, target_fill_ids, fork_point, current_len, current_kv_len,
                needs_kv_release, draft_req, state, reqs_to_merge
            )
        elif current_len > target_len:
            self._handle_draft_ahead(
                req, target_fill_ids, fork_point, current_kv_len,
                needs_kv_release, draft_req, state, reqs_to_merge
            )
        else:
            self._handle_target_ahead(
                req, target_fill_ids, fork_point, current_len, current_kv_len,
                needs_kv_release, draft_req, state, reqs_to_merge
            )
    
    def _handle_equal_length(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_len: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: DraftRequest,
        state: RemoteSpecDraftState,
        reqs_to_merge: List[Req],
    ) -> None:
        """Handle Case 1: Equal length sequences."""
        if fork_point == current_len - 1:
            # Case 1.1: Only last token differs
            logger.info(f"\033[36m [Case 1.1] {req.rid}, replacing last token \033[0m")
            self._update_tokens(req, fork_point, target_fill_ids[fork_point:])
            self._update_req_state(req, draft_req, state)
            self._resume_or_update(req, state, reqs_to_merge)
        else:
            # Case 1.2: Multiple tokens differ
            self._handle_multi_token_divergence(
                req, target_fill_ids, fork_point, current_kv_len,
                needs_kv_release, draft_req, state, reqs_to_merge,
                case_name="1.2"
            )
    
    def _handle_draft_ahead(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: DraftRequest,
        state: RemoteSpecDraftState,
        reqs_to_merge: List[Req],
    ) -> None:
        """Handle Case 2: Draft ahead of Target."""
        target_len = len(target_fill_ids)
        
        if fork_point == target_len:
            # Case 2.1: Target is prefix of Draft
            logger.info(f"\033[36m [Case 2.1] {req.rid}, Draft ahead, continuing \033[0m")
            self._update_req_state(req, draft_req, state)
            self._resume_or_update(req, state, reqs_to_merge)
        else:
            # Case 2.2: Tokens differ
            self._handle_multi_token_divergence(
                req, target_fill_ids, fork_point, current_kv_len,
                needs_kv_release, draft_req, state, reqs_to_merge,
                case_name="2.2"
            )
    
    def _handle_target_ahead(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_len: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: DraftRequest,
        state: RemoteSpecDraftState,
        reqs_to_merge: List[Req],
    ) -> None:
        """
        Handle Case 3: Target ahead of Draft.
        
        Both 3.1 and 3.2 need extend, so use re-prefill for now.
        TODO: Optimize with local extend in future.
        """
        if fork_point == current_len:
            # Case 3.1: Draft is prefix of Target - needs extend
            logger.info(f"\033[36m [Case 3.1] {req.rid}, re-prefill for extend \033[0m")
        else:
            # Case 3.2: Tokens differ - needs extend
            logger.info(f"\033[36m [Case 3.2] {req.rid}, re-prefill for extend \033[0m")
        
        # All Case 3 scenarios need extend, use re-prefill
        self._prepare_for_reprefill(req, target_fill_ids, draft_req, state)
    
    def _handle_multi_token_divergence(
        self,
        req: Req,
        target_fill_ids: List[int],
        fork_point: int,
        current_kv_len: int,
        needs_kv_release: bool,
        draft_req: DraftRequest,
        state: RemoteSpecDraftState,
        reqs_to_merge: List[Req],
        case_name: str,
    ) -> None:
        """
        Handle multi-token divergence (Case 1.2, 2.2).
        
        Strategy: Only use local rollback if we can decode directly after.
        If extend is needed, use re-prefill for simplicity.
        """
        # Check if local rollback + decode is possible
        new_len = len(target_fill_ids)
        can_decode_after_rollback = (new_len - 1 <= fork_point)
        
        if can_decode_after_rollback and self.draft_kv_manager.can_local_rollback(req, fork_point):
            # Local rollback + decode
            logger.info(f"\033[35m [Case {case_name}] {req.rid}, local rollback + decode \033[0m")
            if needs_kv_release:
                self.draft_kv_manager.local_rollback(req, fork_point, current_kv_len)
            self._update_tokens(req, fork_point, target_fill_ids[fork_point:])
            self._update_req_state(req, draft_req, state)
            self._resume_or_update(req, state, reqs_to_merge)
        else:
            # Need extend or divergence in RadixCache region - use re-prefill
            logger.info(f"\033[36m [Case {case_name}] {req.rid}, re-prefill \033[0m")
            self._prepare_for_reprefill(req, target_fill_ids, draft_req, state)
    
    # =========================================================================
    # Request State Updates
    # =========================================================================
    
    def _update_tokens(self, req: Req, fork_point: int, delta_tokens: List[int]) -> None:
        """Update request tokens from fork_point."""
        truncate_point = fork_point - len(req.origin_input_ids)
        req.output_ids = req.output_ids[:max(0, truncate_point)]
        req.output_ids.extend(delta_tokens)
        req.fill_ids = req.origin_input_ids + req.output_ids
    
    def _update_req_state(self, req: Req, draft_req: DraftRequest, state: RemoteSpecDraftState) -> None:
        """Update request state for decode continuation."""
        req.spec_cnt = draft_req.spec_cnt
        req.draft_tokens_target = draft_req.num_draft_tokens
        # Only reset draft_generation_start_len if it hasn't been set yet (initial state)
        # or if the request was paused and is now resuming
        # Don't reset during active decode to avoid breaking token counting
        if not hasattr(req, 'draft_generation_start_len') or req.draft_generation_start_len == 0 or req.draft_is_paused:
            req.draft_generation_start_len = len(req.output_ids)
        req.draft_is_paused = False
        req.len_output_ids = len(req.output_ids)
        state.last_updated_time = time.time()
    
    def _reset_req_logprob_fields(self, req: Req):
        """Reset all logprob-related fields."""
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
    
    def _resume_or_update(
        self,
        req: Req,
        state: RemoteSpecDraftState,
        reqs_to_merge: List[Req],
    ) -> None:
        """Resume paused request or update running request."""
        location = self._get_req_actual_location(req)
        
        if location == "paused":
            self._resume_from_paused(req, state, reqs_to_merge)
        elif location == "running_batch":
            self._update_in_running_batch(req, state, reqs_to_merge)
        # last_batch/waiting_queue: no action needed
    
    def _resume_from_paused(
        self,
        req: Req,
        state: RemoteSpecDraftState,
        reqs_to_merge: List[Req],
    ) -> None:
        """
        Resume a paused request back to running_batch.
        
        Paused requests have valid KV but were removed from batch to wait
        for Target's next message. Now we need to merge them back.
        """
        with self.paused_reqs_lock:
            if req in self.paused_reqs:
                self.paused_reqs.remove(req)
        
        if req.req_pool_idx is None:
            # Edge case: lost KV while paused (shouldn't happen normally)
            logger.warning(f"[Draft][Resume] {req.rid} has no KV, moving to waiting_queue")
            state.location = "waiting_queue"
            if req not in self.waiting_queue:
                self._add_request_to_queue(req)
            return
        
        # Add to merge list so it will be merged back to running_batch
        if req not in self.running_batch.reqs:
            reqs_to_merge.append(req)
        
        state.location = "running_batch"
        logger.debug(f"[Draft][Resume] {req.rid} resuming to running_batch")
    
    def _update_in_running_batch(
        self,
        req: Req,
        state: RemoteSpecDraftState,
        reqs_to_merge: List[Req],
    ) -> None:
        """
        Update request that is already in running batch.
        
        This is called when tokens are identical and req is already decoding.
        We just need to verify state consistency - no action needed since
        req will continue decoding in the normal scheduler flow.
        """
        if req.req_pool_idx is None:
            # Edge case: req in batch but lost KV (shouldn't happen)
            logger.warning(f"[Draft][Update] {req.rid} has no KV, moving to waiting_queue")
            state.location = "waiting_queue"
            if req not in self.waiting_queue:
                self._add_request_to_queue(req)
            return
        
        # Req is already in running_batch with valid KV - just continue decoding
        # No action needed here, scheduler will process it normally
        logger.debug(f"[Draft][Update] {req.rid} continuing in running_batch")
    
    # =========================================================================
    # KV Operations (Strategy 5)
    # =========================================================================
    
    def _remove_from_all_locations(self, req: Req) -> None:
        """Remove request from all queues/batches."""
        with self.paused_reqs_lock:
            if req in self.paused_reqs:
                self.paused_reqs.remove(req)
        
        if not self.running_batch.is_empty() and req in self.running_batch.reqs:
            self.running_batch.filter_batch(chunked_req_to_exclude=[req])
        
        if self.last_batch and req in self.last_batch.reqs:
            self.last_batch.filter_batch(chunked_req_to_exclude=[req])
        
        if req in self.waiting_queue:
            self.waiting_queue.remove(req)
    
    def _prepare_for_reprefill(
        self,
        req: Req,
        target_fill_ids: List[int],
        draft_req: DraftRequest,
        state: RemoteSpecDraftState,
    ) -> None:
        """Prepare for re-prefill when divergence in RadixCache region."""
        logger.info(f"[Draft][RePrefill] {req.rid}, new_len={len(target_fill_ids)}")
        
        self._remove_from_all_locations(req)
        
        if req.req_pool_idx is not None:
            self.draft_kv_manager.release_all_kv_for_reprefill_req(req)
        
        # Reset request for fresh prefill
        req.fill_ids = target_fill_ids
        req.origin_input_ids = list(target_fill_ids)
        req.output_ids = []
        req.prefix_indices = []
        req.extend_input_len = len(req.fill_ids)
        req.skip_radix_lookup = False
        
        req.spec_cnt = draft_req.spec_cnt
        req.draft_tokens_target = draft_req.num_draft_tokens
        req.draft_generation_start_len = 0
        req.draft_is_paused = False
        req.len_output_ids = 0
        
        req.last_node = None
        req.kv_committed_len = 0
        req.kv_committed_freed = False
        req.kv_overallocated_freed = False
        
        state.location = "waiting_queue"
        req.logprob_start_len = len(req.origin_input_ids) - 1
        self._reset_req_logprob_fields(req)
        
        self._add_request_to_queue(req)
    
    # TODO: Add _prepare_for_local_extend when optimizing extend scenarios
    
    # =========================================================================
    # Request Lifecycle
    # =========================================================================
    
    def _create_new_draft_req(self, draft_req: DraftRequest) -> None:
        """Create new draft request."""
        req_id = draft_req.request_id
        
        if self._exists_draft_state(req_id):
            self._finish_draft_request(req_id)
        
        input_ids = (
            (draft_req.input_ids or []) + 
            (draft_req.output_ids or []) + 
            (draft_req.draft_token_ids or [])
        )
        
        # Create sampling_params if not provided
        if draft_req.sampling_params is None:
            from sglang.srt.sampling.sampling_params import SamplingParams
            sampling_params = SamplingParams()
        else:
            sampling_params = draft_req.sampling_params
        
        # Normalize sampling_params to ensure stop_strs and stop_regex_strs are lists
        # This is important because SamplingParams.__init__ can set stop_strs = None
        # and the normalize() method converts None to [] and str to [str]
        if hasattr(sampling_params, 'normalize'):
            try:
                sampling_params.normalize(self.tokenizer)
            except Exception as e:
                logger.warning(f"[Draft] Failed to normalize SamplingParams for {req_id}: {e}, fixing manually")
                # Manual fix if normalize fails
                if not hasattr(sampling_params, 'stop_strs') or sampling_params.stop_strs is None:
                    sampling_params.stop_strs = []
                elif isinstance(sampling_params.stop_strs, str):
                    sampling_params.stop_strs = [sampling_params.stop_strs]
                
                if not hasattr(sampling_params, 'stop_regex_strs') or sampling_params.stop_regex_strs is None:
                    sampling_params.stop_regex_strs = []
                elif isinstance(sampling_params.stop_regex_strs, str):
                    sampling_params.stop_regex_strs = [sampling_params.stop_regex_strs]
        else:
            # Manual fix if normalize method doesn't exist
            if not hasattr(sampling_params, 'stop_strs') or sampling_params.stop_strs is None:
                sampling_params.stop_strs = []
            elif isinstance(sampling_params.stop_strs, str):
                sampling_params.stop_strs = [sampling_params.stop_strs]
            
            if not hasattr(sampling_params, 'stop_regex_strs') or sampling_params.stop_regex_strs is None:
                sampling_params.stop_regex_strs = []
            elif isinstance(sampling_params.stop_regex_strs, str):
                sampling_params.stop_regex_strs = [sampling_params.stop_regex_strs]
        
        req = Req(
            rid=req_id,
            origin_input_text="", # draft_req.input_text or "",
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
            data_parallel_rank=None,
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
        
        self._add_request_to_queue(req)
        
        self._set_draft_state(req_id, RemoteSpecDraftState(
            req_id=req_id,
            spec_cnt=draft_req.spec_cnt,
            req_object=req,
            location="waiting_queue",
            last_prefix_length=len(input_ids),
            last_output_length=0,
        ))
        
        logger.debug(f"[Draft][New] {req_id}, len={len(input_ids)}")
    
    def _finish_draft_request(self, req_id: str) -> None:
        """Clean up finished request (ONLY place RadixCache is updated)."""
        state = self._get_draft_state(req_id)
        if state is None:
            return
        
        req = state.req_object
        self._remove_from_all_locations(req)
        
        #! TODO: req 不会被finished，直到收到Target的finish信号。因为跳过了 tokenizer，因此也不需要再去 Detokenizer
        if not req.finished():
            req.to_abort = True
            req.finished_reason = FINISH_ABORT("Target request finished")
        
        # Strategy 5: Only update RadixCache at finish time
        if req.req_pool_idx is not None:
            self.draft_kv_manager.release_all_kv_for_finished_req(req)
        
        self._delete_draft_state(req_id)
        logger.debug(f"[Draft][Finish] {req_id}")
    
    def _cleanup_stale_draft_states(self) -> None:
        """Clean up timed-out draft states."""
        for req_id in self.draft_state_manager.cleanup_stale_states():
            try:
                self._finish_draft_request(req_id)
            except Exception as e:
                logger.warning(f"[Draft] Cleanup failed for {req_id}: {e}")
            finally:
                # Ensure state is deleted even if finish fails
                try:
                    self._delete_draft_state(req_id)
                except Exception as cleanup_error:
                    logger.error(f"[Draft] Failed to delete state for {req_id}: {cleanup_error}")
    
    # =========================================================================
    # Pause and Response
    # =========================================================================
    
    def _check_and_pause_draft_req(self, req: Req) -> bool:
        """Check if request should be paused and send response."""
        if getattr(req, 'spec_type', None) != SpecType.DRAFT_REQUEST:
            return False
        
        tokens_generated = len(req.output_ids) - req.draft_generation_start_len
        
        # Debug logging
        logger.info(
            f"[Draft][CheckPause] {req.rid}: "
            f"output_len={len(req.output_ids)}, "
            f"start_len={req.draft_generation_start_len}, "
            f"tokens_generated={tokens_generated}, "
            f"target={req.draft_tokens_target}, "
            f"paused={req.draft_is_paused}"
        )
        
        if tokens_generated >= req.draft_tokens_target:
            self._send_draft_response(req)
            
            req.draft_is_paused = True
            with self.paused_reqs_lock:
                if req not in self.paused_reqs:
                    self.paused_reqs.append(req)
            
            state = self._get_draft_state(req.rid)
            if state:
                state.location = "paused"
                state.last_updated_time = time.time()
            
            return True
        
        return False
    
    def _send_draft_response(self, req: Req) -> None:
        """Send draft tokens to Target server."""
        draft_tokens = req.output_ids[req.draft_generation_start_len:]
        
        draft_logits = []
        if hasattr(req, 'output_token_logprobs_val') and req.output_token_logprobs_val:
            start = req.draft_generation_start_len
            end = start + len(draft_tokens)
            if len(req.output_token_logprobs_val) >= end:
                draft_logits = req.output_token_logprobs_val[start:end]
        
        from sglang.srt.speculative.remote_spec.remote_spec_protocol import SpecType
        response = DraftResponse(
            request_id=req.rid,
            spec_cnt=req.spec_cnt,
            action=RemoteSpecAction.DRAFT,
            spec_type=SpecType.DRAFT_RESPONSE,
            draft_token_ids=draft_tokens,
            draft_logprobs=draft_logits or [],
            target_send_time=req.target_send_time,
            draft_receive_time=req.draft_recv_time,
            draft_send_time=time.perf_counter(),
        )
        
        self.zmq_communicator.send_obj(response)
        req.draft_generation_start_len = len(req.output_ids)
        
        # logger.info(
        #     f"\033[32m [Draft][Send] {req.rid} spec_cnt={req.spec_cnt}, "
        #     f"tokens={len(draft_tokens)}, fill_len={response.fill_len} \033[0m"
        # )
    
    # =========================================================================
    # Batch Merging
    # =========================================================================
    
    def _merge_requests_to_batch(self, reqs_to_merge: List[Req]) -> None:
        """Merge resumed requests into running batch."""
        if not reqs_to_merge:
            return
        
        # Filter invalid requests
        valid_reqs = [r for r in reqs_to_merge if r.req_pool_idx is not None]
        if not valid_reqs:
            return
        
        old_bs = len(self.running_batch.reqs)
        self.running_batch.reqs.extend(valid_reqs)
        
        if old_bs > 0:
            self._extend_batch_tensors(valid_reqs, old_bs)
        else:
            self._init_batch_tensors(valid_reqs)
        
        from sglang.srt.layers.sampler import SamplingBatchInfo
        self.running_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self.running_batch,
            self.model_config.vocab_size,
        )
    
    def _extend_batch_tensors(self, reqs: List[Req], old_bs: int) -> None:
        """Extend batch tensors with new requests."""
        batch = self.running_batch
        device = batch.device
        
        def tensor(data, dtype):
            return torch.tensor(data, dtype=dtype, device=device)
        
        def seq_len(r):
            return len(r.origin_input_ids) + len(r.output_ids) - 1
        
        # Core tensors
        new_idx = tensor([r.req_pool_idx for r in reqs], torch.int64)
        batch.req_pool_indices = torch.cat([batch.req_pool_indices, new_idx]) if batch.req_pool_indices is not None else new_idx
        
        new_lens = tensor([seq_len(r) for r in reqs], torch.int64)
        batch.seq_lens = torch.cat([batch.seq_lens, new_lens]) if batch.seq_lens is not None else new_lens
        
        new_orig = tensor([seq_len(r) for r in reqs], torch.int32)
        batch.orig_seq_lens = torch.cat([batch.orig_seq_lens, new_orig]) if batch.orig_seq_lens is not None else new_orig
        
        batch.out_cache_loc = None
        batch.seq_lens_sum = batch.seq_lens.sum().item()
        
        new_out = tensor([r.output_ids[-1] if r.output_ids else r.origin_input_ids[-1] for r in reqs], torch.int64)
        batch.output_ids = torch.cat([batch.output_ids, new_out]) if batch.output_ids is not None else new_out
        
        # Logprob handling
        new_logprob = any(r.return_logprob for r in reqs)
        if batch.return_logprob or new_logprob:
            top_nums = [r.top_logprobs_num if r.return_logprob else 0 for r in reqs]
            token_ids = [r.token_ids_logprob if r.return_logprob else None for r in reqs]
            
            if batch.return_logprob:
                batch.top_logprobs_nums.extend(top_nums)
                batch.token_ids_logprobs.extend(token_ids)
            else:
                batch.top_logprobs_nums = [0] * old_bs + top_nums
                batch.token_ids_logprobs = [None] * old_bs + token_ids
        
        # Multimodal
        mm_inputs = [r.multimodal_inputs for r in reqs]
        if batch.multimodal_inputs is not None:
            batch.multimodal_inputs.extend(mm_inputs)
        else:
            batch.multimodal_inputs = [None] * old_bs + mm_inputs
        
        # Flags
        batch.return_logprob = batch.return_logprob or new_logprob
        batch.has_stream = batch.has_stream or any(r.stream for r in reqs)
        batch.has_grammar = batch.has_grammar or any(r.grammar for r in reqs)
        batch.return_hidden_states = batch.return_hidden_states or any(r.return_hidden_states for r in reqs)
    
    def _init_batch_tensors(self, reqs: List[Req]) -> None:
        """Initialize batch tensors from requests."""
        batch = self.running_batch
        device = self.tp_group.device
        
        def tensor(data, dtype):
            return torch.tensor(data, dtype=dtype, device=device)
        
        def seq_len(r):
            return len(r.origin_input_ids) + len(r.output_ids) - 1
        
        batch.req_pool_indices = tensor([r.req_pool_idx for r in reqs], torch.int64)
        batch.seq_lens = tensor([seq_len(r) for r in reqs], torch.int64)
        batch.orig_seq_lens = tensor([seq_len(r) for r in reqs], torch.int32)
        batch.out_cache_loc = None
        batch.seq_lens_sum = batch.seq_lens.sum().item()
        batch.output_ids = tensor([r.output_ids[-1] if r.output_ids else r.origin_input_ids[-1] for r in reqs], torch.int64)
        
        batch.top_logprobs_nums = [r.top_logprobs_num if r.return_logprob else 0 for r in reqs]
        batch.token_ids_logprobs = [r.token_ids_logprob if r.return_logprob else None for r in reqs]
        batch.multimodal_inputs = [r.multimodal_inputs for r in reqs]
        batch.return_logprob = any(r.return_logprob for r in reqs)
        batch.has_stream = any(r.stream for r in reqs)
        batch.has_grammar = any(r.grammar for r in reqs)
        batch.return_hidden_states = any(r.return_hidden_states for r in reqs)
    
    # =========================================================================
    # Placeholder stubs for compatibility
    # =========================================================================

    def _is_self_high_overhead_draft(self) -> bool:
        """Check if server is under high load based on batch size."""
        if not hasattr(self, 'running_batch') or self.running_batch is None:
            return False
        
        current_bsz = self.running_batch.batch_size()
        if current_bsz > self.server_args.remote_speculative_max_batch_size:
            return True
        return False

    def _send_reject_message(self):
        """Send rejection message to Target."""
        from sglang.srt.speculative.remote_spec.remote_spec_protocol import (
            RemoteSpecResponseFromDraftToTarget,
            RemoteSpecAction,
        )
        
        # Send a reject message to indicate high load
        reject_msg = RemoteSpecResponseFromDraftToTarget(
            request_id="system",
            spec_cnt=0,
            action=RemoteSpecAction.REJECT,
            spec_type=None,
            draft_tokens=[],
            draft_logits=[],
            target_send_time=None,
            draft_receive_time=None,
            draft_send_time=time.perf_counter(),
        )
        self.zmq_communicator.send_obj(reject_msg)
        logger.info("[Draft] Sent reject message to Target due to high load")
