"""
KV Cache Management for Draft Requests in Remote Speculative Decoding.

Strategy 5: Local Rollback + Delayed Cache
==========================================
This module implements an optimized KV management strategy:

1. **Local Rollback** (page_size == 1 only):
   When divergence occurs in the "unstable" region (positions >= prefix_len),
   we directly free KV slots without touching RadixCache. This is fast and
   doesn't require tree operations.
   
   NOTE: When page_size > 1, local rollback is DISABLED because:
   - PagedTokenToKVPoolAllocator.free() converts indices to page indices
   - Partial page frees followed by cache_finished_req can cause double-free
   - All divergence cases fall back to re-prefill via cache_finished_req

2. **Delayed Cache**: RadixCache is only updated at request finish time
   via cache_finished_req. This reduces RadixCache contention during
   the request lifecycle.

3. **Memory Layout**:
   - [0, prefix_len): RadixCache-managed indices (from match_prefix)
   - [prefix_len, current_kv_len): Draft-allocated indices (can be freed directly)

Benefits:
- Minimal latency for divergence handling (when page_size == 1)
- No RadixCache operations during decode/rollback
- Only finish triggers RadixCache update
- Simple and reliable memory management
- Safe handling of page_size > 1 via re-prefill fallback
"""
import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.speculative.remote_spec.remote_spec_protocol import SpecType

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

logger = logging.getLogger(__name__)


class RemoteSpecKVRollbacker:
    """
    KV cache manager for draft requests using Strategy 5.
    
    RemoteSpecKVRollbacker is used in the draft end of remote spec.
    It is responsible for rolling back the KV cache for the draft requests.
    
    Strategy 5: Local Rollback + Delayed Cache
    - Local rollback for divergence in unstable region (fast)
    - RadixCache only updated at finish time
    - Minimal tree operations during request lifecycle
    """
    
    def __init__(
        self,
        token_to_kv_pool_allocator: "TokenToKVPoolAllocator",
        req_to_token_pool: "ReqToTokenPool",
        tree_cache: "BasePrefixCache",
        page_size: int = 1,
        promote_interval: int = 50,  # Unused, kept for API compatibility
        num_draft_tokens: int = 5,   # Unused, kept for API compatibility
        tp_rank: int = 0,  # TP适配：用于控制日志输出
    ) -> None:
        """
        Initialize the KV manager.
        
        Args:
            token_to_kv_pool_allocator: KV pool allocator
            req_to_token_pool: Request to token pool mapping
            tree_cache: RadixCache instance
            page_size: KV cache page size
            promote_interval: Unused (kept for compatibility)
            num_draft_tokens: Unused (kept for compatibility)
            tp_rank: TP rank for controlling log output (default: 0)
        """
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.token_to_kv_pool = token_to_kv_pool_allocator  # Alias for compatibility
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache
        self.page_size = page_size
        self.tp_rank = tp_rank
    
    def get_prefix_len(self, req: "Req") -> int:
        """
        Get the length of RadixCache-managed prefix.
        
        This is the boundary between RadixCache indices and Draft-allocated indices.
        
        Args:
            req: The request object
            
        Returns:
            Length of prefix_indices (0 if not set)
        """
        if hasattr(req, 'prefix_indices') and req.prefix_indices is not None:
            if isinstance(req.prefix_indices, torch.Tensor):
                return len(req.prefix_indices)
            elif isinstance(req.prefix_indices, list):
                return len(req.prefix_indices)
        return 0
    
    def can_local_rollback(self, req: "Req", fork_point: int) -> bool:
        """
        Check if divergence can be handled with local rollback.
        
        Local rollback is possible when:
        1. fork_point >= prefix_len (divergence in Draft-allocated region)
        2. page_size == 1 (no page alignment issues)
        
        When page_size > 1, local_rollback is disabled because freeing partial
        pages can cause double-free issues with the PagedTokenToKVPoolAllocator.
        
        Args:
            req: The request object
            fork_point: Point of divergence
            
        Returns:
            True if local rollback is possible, False if re-prefill is needed
        """
        # Disable local rollback when page_size > 1 to avoid double-free issues
        if self.page_size > 1:
            return False
        
        prefix_len = self.get_prefix_len(req)
        return fork_point >= prefix_len
    
    def rollback(self, req: "Req", fork_point: int, current_kv_len: Optional[int] = None) -> bool:
        """
        Perform rollback: free KV slots from fork_point onwards.
        
        This is the main entry point for handling divergence. It will:
        1. Check if local rollback is possible (fork_point >= prefix_len and page_size == 1)
        2. If yes, perform local rollback (fast path)
        3. If no, caller should use release_all_kv_for_reprefill_req (slow path)
        
        Args:
            req: The request object
            fork_point: Start of the range to free (first position to discard)
            current_kv_len: End of the range to free (exclusive). If None, computed from req.
            
        Returns:
            True if rollback was performed, False if re-prefill is needed
        """
        if current_kv_len is None:
            # Compute current KV length from request with null safety
            input_ids = getattr(req, 'origin_input_ids', [])
            output_ids = getattr(req, 'output_ids', [])
            input_len = len(input_ids) if input_ids is not None else 0
            output_len = len(output_ids) if output_ids is not None else 0
            total_len = input_len + output_len
            current_kv_len = max(0, total_len - 1)  # Last token has no KV
        
        if self.can_local_rollback(req, fork_point):
            return self.local_rollback(req, fork_point, current_kv_len)
        else:
            # Cannot do local rollback, need re-prefill
            return False
    
    def local_rollback(
        self,
        req: "Req",
        fork_point: int,
        current_kv_len: int,
    ) -> bool:
        """
        Perform local rollback: free KV slots from fork_point to current_kv_len.
        
        This directly frees Draft-allocated KV slots without touching RadixCache.
        ONLY call this when:
        1. fork_point >= prefix_len
        2. page_size == 1
        
        Args:
            req: The request object
            fork_point: Start of the range to free (first position to discard)
            current_kv_len: End of the range to free (exclusive)
            
        Returns:
            True if rollback was performed, False if skipped
        """
        # Safety check: local_rollback is not safe when page_size > 1
        if self.page_size > 1:
            if self.tp_rank == 0:
                logger.warning(
                    f"[RemoteSpecKVRollbacker] local_rollback called with page_size={self.page_size}, "
                    "this is not safe! Use re-prefill instead."
                )
            return False
        
        if req.req_pool_idx is None:
            return False
        
        if fork_point >= current_kv_len:
            # Nothing to free
            return False
        
        # Safety check: ensure we're not freeing RadixCache indices
        prefix_len = self.get_prefix_len(req)
        if fork_point < prefix_len:
            if self.tp_rank == 0:
                logger.error(
                    f"[RemoteSpecKVRollbacker] local_rollback called with fork_point={fork_point} < "
                    f"prefix_len={prefix_len}, this would free RadixCache indices! Aborting."
                )
            return False
        
        try:
            # Get indices to free from req_to_token_pool with bounds checking
            max_len = self.req_to_token_pool.req_to_token.shape[1]
            start = min(fork_point, max_len)
            end = min(current_kv_len, max_len)
            
            if start >= end:
                if self.tp_rank == 0:
                    logger.warning(
                        f"[RemoteSpecKVRollbacker] Invalid rollback range for {req.rid}: "
                        f"start={start}, end={end}, max_len={max_len}"
                    )
                return False
            
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, start:end
            ]
            
            # Free the indices back to the pool
            self.token_to_kv_pool_allocator.free(kv_indices)
            
            # CRITICAL: Update kv_committed_len and kv_allocated_len after rollback
            # This fixes the memory leak where kv_committed_len keeps accumulating
            # through generate-rollback cycles without being decremented
            old_committed = req.kv_committed_len
            old_allocated = req.kv_allocated_len
            
            # After rollback to fork_point, the new KV length should be fork_point
            req.kv_committed_len = fork_point
            req.kv_allocated_len = fork_point
            
            if self.tp_rank == 0:
                logger.debug(
                    f"[RemoteSpecKVRollbacker] Local rollback for {req.rid}: "
                    f"freed [{fork_point}, {current_kv_len}), "
                    f"kv_committed: {old_committed} -> {req.kv_committed_len}, "
                    f"kv_allocated: {old_allocated} -> {req.kv_allocated_len}, "
                    f"prefix_len={prefix_len}"
                )
            return True
        except Exception as e:
            if self.tp_rank == 0:
                logger.warning(
                    f"[RemoteSpecKVRollbacker] Failed to local_rollback for {req.rid}: {e}"
                )
            return False
    
    def release_all_kv_for_finished_req(self, req: "Req") -> None:
        """
        Release all KV cache for a FINISHED request via cache_finished_req.
        
        This is the ONLY place where RadixCache is updated during a request's
        lifecycle. Called when the request is finished (Target sends finish signal).
        
        cache_finished_req will:
        1. Insert valid tokens into RadixCache (for future prefix sharing)
        2. Free duplicate KV indices
        3. Release the req_pool slot
        4. Release the lock on last_node
        
        Args:
            req: The request object
        """
        if req.req_pool_idx is None:
            return
        
        kv_len = req.kv_committed_len
        req.fill_ids = (req.origin_input_ids + req.output_ids)[:kv_len]
        
        self.tree_cache.cache_finished_req(req)
        req.req_pool_idx = None
        
        if self.tp_rank == 0:
            logger.debug(
                f"[RemoteSpecKVRollbacker] Released all KV for finished request {req.rid}"
            )
    
    def release_all_kv_for_reprefill_req(self, req: "Req") -> None:
        """
        Release all KV cache for RE-PREFILL (when divergence is in RadixCache region).
        
        This is called when fork_point < prefix_len, meaning the divergence
        is in the RadixCache-managed region and we need a complete re-prefill.
        
        Args:
            req: The request object
        """
        if req.req_pool_idx is None:
            return
        
        kv_len = req.kv_committed_len
        req.fill_ids = (req.origin_input_ids + req.output_ids)[:kv_len]
        
        self.tree_cache.cache_finished_req(req)
        req.req_pool_idx = None
        
        if self.tp_rank == 0:
            logger.debug(
                f"[RemoteSpecKVRollbacker] Released all KV for re-prefill {req.rid}"
            )
    
    # =========================================================================
    # Legacy and compatibility API methods
    # =========================================================================
    
    def release_all_kv_for_finish(self, req: "Req") -> None:
        """Legacy API - use release_all_kv_for_finished_req instead."""
        self.release_all_kv_for_finished_req(req)
    
    def release_all_kv_for_reprefill(self, req: "Req") -> None:
        """Legacy API - use release_all_kv_for_reprefill_req instead."""
        self.release_all_kv_for_reprefill_req(req)
    
    def release_all_kv(self, req: "Req") -> None:
        """Legacy API - use release_all_kv_for_finished_req instead."""
        self.release_all_kv_for_finished_req(req)
    
    def compute_unstable_tokens(self, req: "Req") -> int:
        """
        Compute unstable tokens for memory accounting.
        
        In Strategy 5, unstable tokens = current_kv_len - prefix_len
        These are the Draft-allocated tokens that can be rolled back.
        
        Args:
            req: The request object
            
        Returns:
            Number of unstable tokens
        """
        if req is None or getattr(req, 'req_pool_idx', None) is None:
            return 0
        
        if not hasattr(req, 'spec_type') or req.spec_type != SpecType.DRAFT_REQUEST:
            return 0
        
        total_len = len(getattr(req, 'origin_input_ids', [])) + len(getattr(req, 'output_ids', []))
        current_kv_len = max(0, total_len - 1)  # Last token has no KV
        prefix_len = self.get_prefix_len(req)
        
        return max(0, current_kv_len - prefix_len)
    
    def compute_total_unstable_tokens(
        self,
        paused_reqs: list,
        running_batch_reqs: list,
        waiting_queue: list,
    ) -> int:
        """
        Compute total unstable tokens across all draft requests.
        
        Args:
            paused_reqs: List of paused requests
            running_batch_reqs: List of running batch requests
            waiting_queue: List of waiting requests
            
        Returns:
            Total number of unstable tokens
        """
        total = 0
        seen_rids = set()
        
        for req in paused_reqs:
            if req.rid not in seen_rids:
                total += self.compute_unstable_tokens(req)
                seen_rids.add(req.rid)
        
        for req in running_batch_reqs:
            if req.rid not in seen_rids:
                total += self.compute_unstable_tokens(req)
                seen_rids.add(req.rid)
        
        for req in waiting_queue:
            if req.rid not in seen_rids:
                total += self.compute_unstable_tokens(req)
                seen_rids.add(req.rid)
        
        return total
    
    def get_memory_stats(self, req: "Req") -> dict:
        """
        Get memory stats for debugging.
        
        Args:
            req: The request object
            
        Returns:
            Dictionary with memory statistics
        """
        if req is None:
            return {'rid': 'unknown', 'status': 'no_request'}
        
        total_len = len(getattr(req, 'origin_input_ids', [])) + len(getattr(req, 'output_ids', []))
        current_kv_len = max(0, total_len - 1)
        prefix_len = self.get_prefix_len(req)
        has_kv = getattr(req, 'req_pool_idx', None) is not None
        
        return {
            'rid': req.rid,
            'total_len': total_len,
            'current_kv_len': current_kv_len,
            'prefix_len': prefix_len,
            'unstable': max(0, current_kv_len - prefix_len),
            'has_kv': has_kv,
            'strategy': 'local_rollback',
        }
