import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.remote_spec.remote_spec_protocol import (
    RemoteSpecAction,
    RemoteSpecRequest,
    SpecType,
)
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj

logger = logging.getLogger(__name__)


# =========================================================================
# Draft Circuit Breaker
# =========================================================================

class DraftCircuitBreaker:
    """Circuit breaker that stops sending to Draft after consecutive failures.

    State machine:
        CLOSED  -- normal, send + wait
          |  (N consecutive timeouts)
          v
        OPEN    -- broken, skip send + wait, pure decode
          |  (after cooldown_rounds)
          v
        HALF_OPEN -- probe: try one round
          |          success -> CLOSED
          |          failure -> OPEN (reset cooldown)
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 30,
        cooldown_rounds: int = 100,
        tp_rank: int = 0,
    ):
        self.state = self.CLOSED
        self.consecutive_failures = 0
        self.failure_threshold = failure_threshold
        self.cooldown_rounds = cooldown_rounds
        self.rounds_in_open = 0
        self.tp_rank = tp_rank

    def should_send(self) -> bool:
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            self.rounds_in_open += 1
            if self.rounds_in_open >= self.cooldown_rounds:
                self.state = self.HALF_OPEN
                if self.tp_rank == 0:
                    logger.debug(
                        "\033[34m [CircuitBreaker] OPEN -> HALF_OPEN, probing draft... \033[0m"
                    )
                return True
            return False
        if self.state == self.HALF_OPEN:
            return True
        return False

    def record_success(self):
        if self.state != self.CLOSED and self.tp_rank == 0:
            logger.debug(
                f"\033[34m [CircuitBreaker] {self.state} -> CLOSED, draft recovered \033[0m"
            )
        self.consecutive_failures = 0
        self.state = self.CLOSED
        self.rounds_in_open = 0

    def record_failure(self):
        self.consecutive_failures += 1
        if self.state == self.HALF_OPEN:
            self.state = self.OPEN
            self.rounds_in_open = 0
            if self.tp_rank == 0:
                logger.debug(
                    "\033[34m [CircuitBreaker] HALF_OPEN -> OPEN, probe failed \033[0m"
                )
        elif self.consecutive_failures >= self.failure_threshold:
            if self.state != self.OPEN and self.tp_rank == 0:
                logger.debug(
                    f"\033[34m [CircuitBreaker] CLOSED -> OPEN after "
                    f"{self.consecutive_failures} consecutive timeouts, "
                    f"cooldown {self.cooldown_rounds} rounds \033[0m"
                )
            self.state = self.OPEN
            self.rounds_in_open = 0


# =========================================================================
# Target Scheduler Mixin
# =========================================================================

class RemoteSpecTargetSchedulerMixin:
    """Target-side scheduler mixin for remote speculative decoding.

    Provides:
    - Background recv thread + threading.Event for efficient draft waiting
    - Circuit breaker to handle Draft unavailability
    - Main event loop for target-side scheduling

    spec_cnt contract
    -----------------
    spec_cnt counts the total number of draft generation requests sent to Draft
    for a given request, including retries.  Every call to send_batch_draft_requests
    and every call to retry_drafts_for_reqs increments spec_cnt by 1.  This keeps
    spec_cnt values unique and prevents stale responses from being mistaken for
    fresh ones.
    """

    # =====================================================================
    # Initialization
    # =====================================================================

    def _init_draft_recv_infra(self):
        """Initialize background recv thread, Event, and circuit breaker."""
        self._recv_timeout_s = (
            float(os.environ.get("REMOTE_SPEC_RECV_TIMEOUT_MS", "200")) / 1000.0
        )

        # Message buffer: background thread writes, main thread reads
        self._msg_buffer: List[RemoteSpecRequest] = []
        self._msg_lock = threading.Lock()
        self._data_ready = threading.Event()
        self._bg_running = True

        failure_threshold = int(os.environ.get("REMOTE_SPEC_FAILURE_THRESHOLD", "30"))
        cooldown_rounds = int(os.environ.get("REMOTE_SPEC_COOLDOWN_ROUNDS", "100"))
        self.draft_circuit_breaker = DraftCircuitBreaker(
            failure_threshold=failure_threshold,
            cooldown_rounds=cooldown_rounds,
            tp_rank=self.tp_rank,
        )

        if self.tp_size == 1 or self.tp_rank == 0:
            self._bg_recv_thread = threading.Thread(
                target=self._bg_recv_loop, daemon=True, name="draft_recv_bg"
            )
            self._bg_recv_thread.start()
            logger.debug("\033[34m [Target] Background draft recv thread started \033[0m")

    def _bg_recv_loop(self):
        """Background thread: poll ZMQ, buffer messages, signal Event.

        Runs only on rank 0.  The main thread never calls recv_all_objs()
        directly, avoiding any race condition on the ZMQ endpoint.
        """
        while self._bg_running:
            try:
                if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                    msgs = self.zmq_communicator.recv_all_objs()
                    if msgs:
                        with self._msg_lock:
                            self._msg_buffer.extend(msgs)
                        self._data_ready.set()
                    else:
                        time.sleep(0.0005)
                else:
                    time.sleep(0.0005)
            except Exception as e:
                logger.error(f"\033[34m [Target][BgRecv] Error: {e} \033[0m")
                time.sleep(0.0005)

    def _drain_msg_buffer(self) -> List[RemoteSpecRequest]:
        """Drain all buffered messages and clear Event."""
        with self._msg_lock:
            msgs = list(self._msg_buffer)
            self._msg_buffer.clear()
        self._data_ready.clear()
        return msgs

    # =====================================================================
    # Main Event Loop
    # =====================================================================

    @DynamicGradMode()
    def event_loop_normal_remote_spec_target(self):
        self.req_to_draft_token: Dict[
            str, Dict[int, Optional[Tuple[List[int], List[float]]]]
        ] = defaultdict(dict)
        self.is_rejected: bool = False
        self.rejected_forward_ct: int = 0

        self._init_draft_recv_infra()

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                # if self is overhead, set draft_num_tokens to 1 and recv_draft_fn to None, and run as normal decode
                if self._is_self_high_overhead_target(batch):
                    batch.draft_num_tokens = 1
                    batch.recv_draft_fn = None
                # if circuit breaker is OPEN, set draft_num_tokens to 1 and recv_draft_fn to None, and run as normal decode
                elif not self.draft_circuit_breaker.should_send():
                    batch.draft_num_tokens = 1
                    batch.recv_draft_fn = None
                else:
                    # if circuit breaker is CLOSED, send draft requests and run as speculative decode
                    draft_num_tokens = self._decide_speculative_num_draft_tokens(batch)
                    self.send_batch_draft_requests(batch, draft_num_tokens)
                    batch.draft_num_tokens = self._decide_verify_num_draft_tokens(batch)
                    # Use bound method references rather than closures so that
                    # state is accessed through self (the scheduler) explicitly.
                    batch.recv_draft_fn = self.recv_drafts_for_batch
                    batch.retry_fn = self.retry_drafts_for_reqs
                    batch.retry_fail_ratio = self.server_args.remote_speculative_retry_fail_ratio
                    # Minimum absolute number of failed reqs required to trigger retry.
                    # When too few reqs fail it is not worth the extra RTT.
                    batch.retry_min_count = self.server_args.remote_speculative_retry_min_count

                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.self_check_during_idle()

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    # =====================================================================
    # Low-level Recv Primitives (shared by recv_drafts_for_batch and
    # retry_drafts_for_reqs to avoid code duplication)
    # =====================================================================

    def _collect_draft_messages(
        self,
        pending_rids: Set[str],
        pending_spec_cnts: Dict[str, int],
        timeout_s: float,
    ) -> List[RemoteSpecRequest]:
        """Wait on the bg buffer until all pending_rids are resolved or timeout.

        Args:
            pending_rids: Mutable set of rids still awaited. Cleared in-place
                as matching messages arrive.
            pending_spec_cnts: If non-empty, only resolve a rid when the received
                message's spec_cnt matches the expected value.  Pass an empty dict
                to accept any spec_cnt for a rid (used by recv_drafts_for_batch
                because Draft always echoes back the spec_cnt we sent).
            timeout_s: Maximum wall-clock wait time in seconds.

        Returns:
            All messages collected during the wait window, including messages for
            rids that were not in pending_rids (they will be stored in the cache
            for future rounds by the caller).
        """
        all_messages: List[RemoteSpecRequest] = []
        deadline = time.perf_counter() + timeout_s

        while pending_rids:
            msgs = self._drain_msg_buffer()
            if msgs:
                all_messages.extend(msgs)
                for msg in msgs:
                    if msg.action != RemoteSpecAction.DRAFT:
                        continue
                    if msg.request_id not in pending_rids:
                        continue
                    expected_sc = pending_spec_cnts.get(msg.request_id)
                    if expected_sc is None or msg.spec_cnt == expected_sc:
                        pending_rids.discard(msg.request_id)
                if not pending_rids:
                    break

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                if pending_rids and self.tp_rank == 0:
                    logger.debug(
                        f"\033[32m [Target] Recv timeout, {len(pending_rids)} rids still pending: "
                        f"{list(pending_rids)}"
                    )
                break
            self._data_ready.wait(timeout=remaining)

        return all_messages

    def _tp_broadcast_messages(
        self, messages: Optional[List[RemoteSpecRequest]]
    ) -> List[RemoteSpecRequest]:
        """Broadcast messages from rank 0 to all TP ranks (single call site)."""
        if self.tp_size > 1:
            return broadcast_pyobj(
                messages if messages else [],
                self.tp_group.rank,
                self.tp_cpu_group,
                src=self.tp_group.ranks[0],
            )
        return messages or []

    def _store_messages(self, messages: List[RemoteSpecRequest]) -> bool:
        """Store DRAFT responses in req_to_draft_token cache.

        Returns:
            True if at least one DRAFT response was stored (used by caller to
            update the circuit breaker).
        """
        has_draft = False
        for msg in messages:
            assert isinstance(msg, RemoteSpecRequest), (
                f"Expected RemoteSpecRequest, got {type(msg)}"
            )
            if msg.action == RemoteSpecAction.REJECT:
                self.process_reject_action()
                if self.tp_rank == 0:
                    logger.debug(
                        "\033[32m [Target] Received REJECT from draft \033[0m"
                    )
            elif msg.action == RemoteSpecAction.DRAFT:
                self.req_to_draft_token[msg.request_id][msg.spec_cnt] = (
                    msg.draft_token_ids,
                    msg.draft_logprobs,
                )
                has_draft = True
        return has_draft

    def _build_result_from_cache(
        self, reqs: List[Req]
    ) -> Dict[str, Tuple[List[int], List[float]]]:
        """Consume draft cache entries for the given reqs.

        Side effect: cleans up stale entries (spec_cnt < req.spec_cnt).
        Stale entries arise when a retry response arrives late—after the request
        has already advanced its spec_cnt past the retry round—and is stored in
        the cache by a subsequent _store_messages call but never consumed.
        Clearing them here prevents unbounded growth of req_to_draft_token.
        """
        result = {}
        for req in reqs:
            if _is_health_check(req):
                continue
            rid, sc = req.rid, req.spec_cnt
            rid_cache = self.req_to_draft_token.get(rid)
            if rid_cache is None:
                continue

            entry = rid_cache.get(sc)
            if entry is not None:
                result[rid] = entry
                del rid_cache[sc]

            # Clean stale entries from earlier spec_cnts
            stale_keys = [k for k in list(rid_cache.keys()) if k < sc]
            for k in stale_keys:
                del rid_cache[k]

        return result

    # =====================================================================
    # High-level Draft Recv API
    # Registered as bound method references on batch (not closures):
    #   batch.recv_draft_fn = self.recv_drafts_for_batch
    #   batch.retry_fn      = self.retry_drafts_for_reqs
    # Worker calls them as: batch.recv_draft_fn(batch) / batch.retry_fn(reqs)
    # =====================================================================

    def recv_drafts_for_batch(self, batch: ScheduleBatch) -> dict:
        """Collect draft responses for a batch after target GPU forward.

        Architecture:
            bg thread: ZMQ recv -> buffer -> Event.set()
            main thread: _collect_draft_messages -> TP broadcast
                         -> _store_messages -> _build_result_from_cache

        Draft sends responses per-request asynchronously.  Different requests
        may finish at different times (e.g., one needing 2 forward passes vs one
        needing 10).  The loop collects progressively until the full batch is
        gathered or the deadline is hit.
        """
        if logger.isEnabledFor(logging.DEBUG):
            start_time = time.perf_counter()

        # Phase 1: Collect (rank 0 loops; other ranks skip directly to broadcast)
        if self.tp_size == 1 or self.tp_rank == 0:
            pending_rids = {
                req.rid
                for req in batch.reqs
                if not _is_health_check(req)
                and req.spec_cnt in self.req_to_draft_token.get(req.rid, {})
                and self.req_to_draft_token[req.rid][req.spec_cnt] is None
            }
            messages = self._collect_draft_messages(
                pending_rids=pending_rids,
                # Empty: Draft always echoes our spec_cnt, accept any.
                pending_spec_cnts={},
                timeout_s=self._recv_timeout_s,
            )
        else:
            messages = None

        # Phase 2: TP broadcast (single call, always matches across ranks)
        messages = self._tp_broadcast_messages(messages)

        # Phase 3: Store + circuit breaker update
        has_draft = self._store_messages(messages)
        if has_draft:
            self.draft_circuit_breaker.record_success()
        else:
            self.draft_circuit_breaker.record_failure()

        # Phase 4: Build result (consume entries from cache)
        result = self._build_result_from_cache(
            [req for req in batch.reqs if not _is_health_check(req)]
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"\033[32m [Target][RecvDrafts] recv_drafts_for_batch recv {len(messages)} reqs, "
                f"rids={[r.request_id for r in messages]}, "
                f"took {(time.perf_counter() - start_time) * 1000:.3f} ms \033[0m"
            )
        return result

    def retry_drafts_for_reqs(self, failed_reqs: List[Req]) -> dict:
        """Re-request fresh draft tokens for diverged requests.

        Called from RemoteSpecWorker._post_verify_update_drafts when the
        fork-point comparison fails (is_matched=False or no draft received).

        spec_cnt at call time has already been incremented once by the regular
        verify round.  This method sends another draft generation request using
        that spec_cnt, so the caller (worker) must increment spec_cnt again after
        this call returns.  Both increments together count the two separate draft
        generation requests issued within a single decode step.
        """
        if logger.isEnabledFor(logging.DEBUG):
            start_time = time.perf_counter()
        if not failed_reqs:
            return {}

        num_draft_tokens = self.server_args.speculative_num_steps + 1

        # Step 1: Register pending sentinels on all ranks so that
        # _build_result_from_cache can correctly find the entries.
        for req in failed_reqs:
            if _is_health_check(req):
                continue
            self.req_to_draft_token[req.rid][req.spec_cnt] = None

        # Step 2: Send retry requests (rank 0 only)
        if self.tp_size == 1 or self.tp_rank == 0:
            self._send_retry_requests(failed_reqs, max(1, num_draft_tokens - 1))

        # Step 3: Collect with strict spec_cnt matching to distinguish retry
        # responses from any in-flight responses for other rounds.
        if self.tp_size == 1 or self.tp_rank == 0:
            pending_rids = {
                req.rid for req in failed_reqs if not _is_health_check(req)
            }
            pending_spec_cnts = {
                req.rid: req.spec_cnt
                for req in failed_reqs
                if not _is_health_check(req)
            }
            messages = self._collect_draft_messages(
                pending_rids=pending_rids,
                pending_spec_cnts=pending_spec_cnts,
                timeout_s=self._recv_timeout_s * 0.5,
            )
        else:
            messages = None

        # Step 4: TP broadcast
        messages = self._tp_broadcast_messages(messages)

        # Step 5: Store (no circuit breaker update — retry is an internal
        # recovery step, not an indicator of draft server health)
        self._store_messages(messages)

        # Step 6: Build result
        result = self._build_result_from_cache(failed_reqs)
        if self.tp_rank == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"\033[32m [Target][Retry] Got drafts for "
                    f"{len(result)}/{len(failed_reqs)} reqs, rids={[r for r in result.keys()]}, "
                    f"took {(time.perf_counter() - start_time) * 1000:.3f} ms \033[0m"
                )
        return result

    # =====================================================================
    # Send
    # =====================================================================

    def send_batch_draft_requests(
        self, batch: ScheduleBatch, speculative_num_draft_tokens: int
    ) -> None:
        """Send draft requests to Draft server and register pending sentinels."""
        if (
            self.is_rejected
            and self.server_args.remote_speculative_reject_interval > 0
            and (
                (self.forward_ct - self.rejected_forward_ct + 1)
                % self.server_args.remote_speculative_reject_interval
                != 0
            )
        ):
            return

        self.is_rejected = False

        # Register pending sentinels on ALL ranks so recv_drafts_for_batch
        # (which runs on every rank) can correctly build pending_rids.
        for req in batch.reqs:
            if _is_health_check(req):
                continue
            self.req_to_draft_token[req.rid][req.spec_cnt] = None

        # ZMQ I/O only on rank 0.  Building RemoteSpecRequest objects on other
        # ranks is pure waste since those objects are never used.
        if self.tp_size == 1 or self.tp_rank == 0:
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                draft_reqs = []
                for req in batch.reqs:
                    if _is_health_check(req):
                        continue
                    draft_reqs.append(
                        RemoteSpecRequest(
                            request_id=req.rid,
                            spec_cnt=req.spec_cnt,
                            action=RemoteSpecAction.DRAFT,
                            spec_type=SpecType.DRAFT_REQUEST,
                            # Only transmit full input_ids on spec_cnt==0 to reduce
                            # serialization overhead on subsequent rounds.
                            input_ids=req.origin_input_ids if req.spec_cnt == 0 else None,
                            output_ids=req.output_ids,
                            draft_token_ids=req.cur_drafts,
                            num_draft_tokens=speculative_num_draft_tokens,
                            sampling_params=req.sampling_params if req.spec_cnt == 0 else None,
                            grammar=None,
                        )
                    )
                self._zmq_send(draft_reqs)
                logger.debug(
                    f"\033[32m [Target][Send] {len(draft_reqs)} draft requests \033[0m"
                )

    def _send_retry_requests(self, failed_reqs: List[Req], num_draft_tokens: int) -> None:
        """Send retry draft requests to Draft. Rank 0 only."""
        if not (hasattr(self, "zmq_communicator") and self.zmq_communicator is not None):
            return
        retry_send_time = time.perf_counter()
        reqs_to_send = [
            RemoteSpecRequest(
                request_id=req.rid,
                spec_cnt=req.spec_cnt,
                action=RemoteSpecAction.DRAFT,
                spec_type=SpecType.DRAFT_REQUEST,
                output_ids=req.output_ids,
                # Empty cur_drafts: tell Draft to generate fresh from output_ids[-1]
                draft_token_ids=[],
                num_draft_tokens=num_draft_tokens,
                target_send_time=retry_send_time,
            )
            for req in failed_reqs
            if not _is_health_check(req)
        ]
        if reqs_to_send:
            self._zmq_send(reqs_to_send)
            logger.debug(
                f"\033[32m [Target][Retry] Sent {len(reqs_to_send)} retry requests "
                f"rids={[r.request_id for r in reqs_to_send]} \033[0m"
            )

    def _zmq_send(self, reqs: List[RemoteSpecRequest]) -> None:
        """Send requests to the first available Draft. Rank 0 only."""
        all_drafts_identity = self.zmq_communicator.get_all_drafts_identity()
        if not all_drafts_identity:
            logger.warning("\033[32m [Target] No draft available, check draft status! \033[0m")
            return
        self.zmq_communicator.send_objs(reqs, all_drafts_identity[0])

    # =====================================================================
    # Notifications & Helpers
    # =====================================================================

    def notify_draft_request_finished_or_aborted(
        self, req: Req, action: RemoteSpecAction
    ) -> None:
        if _is_health_check(req):
            return

        msg = RemoteSpecRequest(
            request_id=req.rid,
            spec_cnt=req.spec_cnt,
            action=action,
            spec_type=SpecType.DRAFT_REQUEST,
            input_ids=[],
            output_ids=[],
            draft_token_ids=[],
            num_draft_tokens=0,
        )

        if self.tp_size == 1 or self.tp_rank == 0:
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                self._zmq_send([msg])

        try:
            if req.rid in self.req_to_draft_token:
                del self.req_to_draft_token[req.rid]
        except Exception as e:
            if self.tp_rank == 0:
                logger.error(
                    f"\033[34m [Target][Notify] Failed to cleanup req_to_draft_token "
                    f"for {req.rid}: {e}"
                )

    def _is_self_high_overhead_target(self, batch: ScheduleBatch) -> bool:
        current_bsz = max(batch.batch_size(), self.running_batch.batch_size())
        if current_bsz > self.server_args.remote_speculative_max_batch_size:
            batch.is_high_overhead = True
            return True
        batch.is_high_overhead = False
        return False

    def _decide_speculative_num_draft_tokens(self, batch: ScheduleBatch) -> int:
        return self.server_args.speculative_num_steps + 1

    def process_reject_action(self) -> None:
        self.is_rejected = True
        self.rejected_forward_ct = self.forward_ct

    def _decide_verify_num_draft_tokens(self, batch: ScheduleBatch) -> int:
        if batch.forward_mode == ForwardMode.EXTEND:
            return self.server_args.speculative_num_draft_tokens

        if self.is_rejected:
            if self.tp_rank == 0:
                logger.debug("\033[34m [Target] draft_num_tokens=1 (rejected) \033[0m")
            return 1

        no_draft_reqs = sum(
            1 for req in batch.reqs
            if not _is_health_check(req) and not req.cur_drafts
        )
        bs = batch.batch_size()
        if bs > 0 and no_draft_reqs / bs > self.server_args.remote_speculative_no_draft_ratio:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[34m [Target] draft_num_tokens=1 "
                    f"(no_draft_ratio {no_draft_reqs/bs:.2f} >= "
                    f"{self.server_args.remote_speculative_no_draft_ratio}) \033[0m"
                )
            return 1
        return self.server_args.speculative_num_draft_tokens


# =========================================================================
# Module-level helper
# =========================================================================

def _is_health_check(req) -> bool:
    return getattr(req, "rid", "").startswith("HEALTH_CHECK")
