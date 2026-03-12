"""
Draft Request State Management for Remote Speculative Decoding.
This module manages the state of draft requests on the Draft server, including:
- Tracking request location (waiting_queue, running_batch, paused)
- Managing request lifecycle with version tracking (spec_cnt)
- Thread-safe state operations
- Timeout-based cleanup
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass
class RemoteSpecDraftState:
    """
    Persistent state for a draft request (for incremental updates).
    
    Attributes:
        req_id: Request ID
        spec_cnt: Current version number
        req_object: Reference to the Req object
        location: Current location ("waiting_queue" | "running_batch" | "paused" | "finished")
        target_origin_input_ids: Target's original prompt token ids (never changes for a
            given request). Stored from the first full message so that subsequent
            incremental messages (input_ids=None) can reconstruct target_fill_ids
            without retransmitting the prompt.
        last_prefix_length: Previous prefix length
        last_output_length: Previous output length
        last_updated_time: Last update timestamp
        created_time: Creation timestamp
        timeout_threshold: Timeout threshold in seconds
    """
    req_id: str
    spec_cnt: int
    req_object: Optional["Req"]
    location: str = "waiting_queue"  # "waiting_queue" | "running_batch" | "paused" | "finished"
    
    target_origin_input_ids: Optional[List[int]] = None
    
    # For verifying continuity
    last_prefix_length: int = 0
    last_output_length: int = 0
    
    # For timeout cleanup
    last_updated_time: float = field(default_factory=time.time)
    created_time: float = field(default_factory=time.time)
    timeout_threshold: float = 30.0  # seconds


class RemoteSpecDraftStateManager:
    """
    Manages the state of all draft requests on the Draft server.
    
    This class provides thread-safe operations for:
    - Creating, updating, and deleting draft request states
    - Looking up request states by ID
    - Cleaning up stale/timed-out states
    
    Attributes:
        active_draft_states: Dictionary mapping req_id to RemoteSpecDraftState
        _lock: Threading lock for thread-safe operations
        time_out_cycle: Default timeout for cleanup
    """
    
    def __init__(self, timeout_threshold: float = 60.0):
        """
        Initialize the state manager.
        
        Args:
            timeout_threshold: Default timeout in seconds for stale state cleanup
        """
        self.active_draft_states: Dict[str, RemoteSpecDraftState] = {}
        self._lock = threading.Lock()
        self.time_out_cycle = 200
        self.timeout_threshold = timeout_threshold

    def get_state(self, req_id: str) -> Optional[RemoteSpecDraftState]:
        """
        Thread-safe lookup of draft state by request ID.
        
        Args:
            req_id: Request ID to look up
            
        Returns:
            RemoteSpecDraftState if found, None otherwise
        """
        with self._lock:
            return self.active_draft_states.get(req_id)
        
    def set_state(self, req_id: str, state: RemoteSpecDraftState):
        """
        Thread-safe setting of draft state.
        
        Args:
            req_id: Request ID
            state: RemoteSpecDraftState to store
        """
        with self._lock:
            self.active_draft_states[req_id] = state
    
    def delete_state(self, req_id: str):
        """
        Thread-safe deletion of draft state.
        
        Args:
            req_id: Request ID to delete
        """
        with self._lock:
            if req_id in self.active_draft_states:
                del self.active_draft_states[req_id]
    
    def exists(self, req_id: str) -> bool:
        """
        Thread-safe check if request ID exists.
        
        Args:
            req_id: Request ID to check
            
        Returns:
            True if exists, False otherwise
        """
        with self._lock:
            return req_id in self.active_draft_states
        
    def get(self, req_id: str) -> Optional[RemoteSpecDraftState]:
        """Alias for get_state for consistency with 0.5.1 API."""
        return self.get_state(req_id)
    
    def set(self, req_id: str, state: RemoteSpecDraftState) -> None:
        """Alias for set_state for consistency with 0.5.1 API."""
        self.set_state(req_id, state)
    
    def delete(self, req_id: str) -> bool:
        """
        Thread-safe deletion of draft state with return value.
        
        Args:
            req_id: Request ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if req_id in self.active_draft_states:
                del self.active_draft_states[req_id]
                return True
            return False
    
    def update_location(self, req_id: str, location: str) -> bool:
        """
        Thread-safe update of request location.
        
        Args:
            req_id: Request ID
            location: New location
            
        Returns:
            True if updated, False if not found
        """
        with self._lock:
            if req_id in self.active_draft_states:
                self.active_draft_states[req_id].location = location
                self.active_draft_states[req_id].last_updated_time = time.time()
                return True
            return False
    
    def update_spec_cnt(self, req_id: str, spec_cnt: int) -> bool:
        """
        Thread-safe update of spec_cnt.
        
        Args:
            req_id: Request ID
            spec_cnt: New spec_cnt value
            
        Returns:
            True if updated, False if not found
        """
        with self._lock:
            if req_id in self.active_draft_states:
                self.active_draft_states[req_id].spec_cnt = spec_cnt
                self.active_draft_states[req_id].last_updated_time = time.time()
                return True
            return False
    
    def touch(self, req_id: str) -> bool:
        """
        Update the last_updated_time for heartbeat.
        
        Args:
            req_id: Request ID
            
        Returns:
            True if updated, False if not found
        """
        with self._lock:
            if req_id in self.active_draft_states:
                self.active_draft_states[req_id].last_updated_time = time.time()
                return True
            return False
    
    def create_state(
        self,
        req_id: str,
        spec_cnt: int,
        req_object: "Req",
        location: str = "waiting_queue",
        prefix_length: int = 0,
        output_length: int = 0,
    ) -> RemoteSpecDraftState:
        """
        Create and register a new draft state.
        
        Args:
            req_id: Request ID
            spec_cnt: Initial spec_cnt
            req_object: The Req object
            location: Initial location
            prefix_length: Initial prefix length
            output_length: Initial output length
            
        Returns:
            The created RemoteSpecDraftState
        """
        now = time.time()
        state = RemoteSpecDraftState(
            req_id=req_id,
            spec_cnt=spec_cnt,
            req_object=req_object,
            location=location,
            last_prefix_length=prefix_length,
            last_output_length=output_length,
            last_updated_time=now,
            created_time=now,
            timeout_threshold=self.timeout_threshold,
        )
        
        with self._lock:
            self.active_draft_states[req_id] = state
        
        return state
    
    def cleanup_stale_states(self, timeout: Optional[float] = None) -> List[str]:
        """
        Remove states that haven't been updated within the timeout.
        
        Args:
            timeout: Override timeout in seconds (uses default if not provided)
            
        Returns:
            List of removed request IDs
        """
        if timeout is None:
            timeout = self.timeout_threshold
        
        current_time = time.time()
        to_remove = []
        
        with self._lock:
            for req_id, state in list(self.active_draft_states.items()):
                idle_time = current_time - state.last_updated_time
                if idle_time > timeout:
                    to_remove.append(req_id)
        
        # Delete outside the lock to avoid holding it too long
        for req_id in to_remove:
            state = self.get_state(req_id)
            # Cleanup will be handled by _cleanup_stale_draft_states which calls _finish_draft_request
            # This method only marks states for cleanup
            logger.debug(f"[RemoteSpecDraftStateManager] Marked stale state for cleanup: {req_id}")
        
        return to_remove
    
    def get_all_rids(self) -> List[str]:
        """
        Get all active request IDs.
        
        Returns:
            List of request IDs
        """
        with self._lock:
            return list(self.active_draft_states.keys())
    
    def size(self) -> int:
        """
        Get the number of active states.
        
        Returns:
            Number of active states
        """
        with self._lock:
            return len(self.active_draft_states)
    
    def clear(self) -> int:
        """
        Clear all states.
        
        Returns:
            Number of states cleared
        """
        with self._lock:
            count = len(self.active_draft_states)
            self.active_draft_states.clear()
            return count
