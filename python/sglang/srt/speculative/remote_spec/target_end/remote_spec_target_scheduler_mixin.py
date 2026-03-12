import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

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
                        "\033[33m [CircuitBreaker] OPEN -> HALF_OPEN, probing draft... \033[0m"
                    )
                return True
            return False
        if self.state == self.HALF_OPEN:
            return True
        return False

    def record_success(self):
        if self.state != self.CLOSED:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[32m [CircuitBreaker] {self.state} -> CLOSED, "
                    f"draft recovered \033[0m"
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
                    "\033[31m [CircuitBreaker] HALF_OPEN -> OPEN, "
                    "probe failed, resetting cooldown \033[0m"
                )
        elif self.consecutive_failures >= self.failure_threshold:
            if self.state != self.OPEN:
                if self.tp_rank == 0:
                    logger.debug(
                        f"\033[31m [CircuitBreaker] CLOSED -> OPEN after "
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

        # Circuit breaker
        failure_threshold = int(os.environ.get("REMOTE_SPEC_FAILURE_THRESHOLD", "30"))
        cooldown_rounds = int(os.environ.get("REMOTE_SPEC_COOLDOWN_ROUNDS", "100"))
        self.draft_circuit_breaker = DraftCircuitBreaker(
            failure_threshold=failure_threshold,
            cooldown_rounds=cooldown_rounds,
            tp_rank=self.tp_rank,
        )

        # Start background recv thread (only rank 0 does ZMQ IO)
        if self.tp_size == 1 or self.tp_rank == 0:
            self._bg_recv_thread = threading.Thread(
                target=self._bg_recv_loop, daemon=True, name="draft_recv_bg"
            )
            self._bg_recv_thread.start()
            logger.debug("[Target] Background draft recv thread started")

    def _bg_recv_loop(self):
        """Background thread: poll ZMQ, buffer messages, signal Event.

        Runs only on rank 0. The main thread never calls recv_all_objs()
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
                    time.sleep(0.005)
            except Exception as e:
                logger.error(f"[Target][BgRecv] Error: {e}")
                time.sleep(0.005)

    def _drain_msg_buffer(self) -> List[RemoteSpecRequest]:
        """Drain all buffered messages and reset Event."""
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
                if self._is_self_high_overhead_target(batch):
                    batch.draft_num_tokens = 1
                    batch.recv_draft_fn = None
                elif not self.draft_circuit_breaker.should_send():
                    # Circuit breaker OPEN: skip draft, run as normal decode
                    batch.draft_num_tokens = 1
                    batch.recv_draft_fn = None
                else:
                    draft_num_tokens = self._decide_speculative_num_draft_tokens(batch)
                    self.send_batch_draft_requests(batch, draft_num_tokens)
                    batch.draft_num_tokens = self._decide_verify_num_draft_tokens(batch)
                    batch.recv_draft_fn = self._make_recv_draft_fn()
                    batch.retry_fn = self._make_retry_fn()
                    batch.retry_fail_ratio = self.server_args.remote_speculative_retry_fail_ratio

                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.self_check_during_idle()

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()

    # =====================================================================
    # Draft Recv (Event-based)
    # =====================================================================

    def _make_recv_draft_fn(self):
        """Create a callback that waits for draft tokens via Event.

        Architecture:
            bg thread: ZMQ recv -> buffer -> Event.set()
            main thread: loop { Event.wait() -> drain } until all reqs
                         fulfilled or timeout -> ONE broadcast -> process

        Draft sends responses per-request, and different requests may
        finish at different forward passes (e.g., req needing 2 passes
        vs req needing 10 passes → 40ms gap). This loop collects
        progressively until the full batch is gathered or deadline hit.
        """
        timeout_s = self._recv_timeout_s

        def recv_draft_fn(batch: ScheduleBatch) -> dict:
            start_time = time.perf_counter()
            # --- Phase 1: Collect messages (rank 0 loops, others wait at broadcast) ---
            if self.tp_size == 1 or self.tp_rank == 0:
                # Build the set of rids we expect responses for.
                # send_batch_draft_requests marks pending entries as None.
                pending_rids = set()
                for req in batch.reqs:
                    if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                        continue
                    rid = req.rid
                    sc = req.spec_cnt
                    if (
                        rid in self.req_to_draft_token
                        and sc in self.req_to_draft_token[rid]
                        and self.req_to_draft_token[rid][sc] is None
                    ):
                        pending_rids.add(rid)
                # logger.debug(f"\033[32m [Target][Recv] Still waiting for {len(pending_rids)} requests, rids: {list(pending_rids)} \033[0m")
                all_messages: List[RemoteSpecRequest] = []
                deadline = time.perf_counter() + timeout_s

                while pending_rids:
                    msgs = self._drain_msg_buffer()
                    if msgs:
                        all_messages.extend(msgs)
                        for msg in msgs:
                            if msg.action == RemoteSpecAction.DRAFT:
                                logger.debug(f"\033[32m [Target][Recv] Received draft for {msg.request_id}, spec_cnt: {msg.spec_cnt}, "
                                            f"T2D transfer time: {(msg.draft_recv_time - msg.target_send_time) / 1000} ms, "
                                            f"D processing time: {(msg.draft_send_time - msg.draft_recv_time) / 1000} ms, "
                                            f"D2T transfer time: {(msg.target_recv_time - msg.draft_send_time) / 1000} ms\033[0m")
                                pending_rids.discard(msg.request_id)
                        if not pending_rids:
                            break

                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        logger.debug(f"\033[32m [Target][Recv] Timeout, {len(pending_rids)} requests still pending, rids: {list(pending_rids)} \033[0m")
                        break

                    self._data_ready.wait(timeout=remaining)

                messages = all_messages
            else:
                messages = None

            # TP broadcast (single call, always matches across ranks)
            if self.tp_size > 1:
                messages = broadcast_pyobj(
                    messages if messages else [],
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )

            # --- Phase 2: Process messages into req_to_draft_token cache ---
            has_draft_response = False
            if messages:
                for msg in messages:
                    assert isinstance(msg, RemoteSpecRequest), "Invalid message type"
                    if msg.action == RemoteSpecAction.REJECT:
                        self.process_reject_action()
                        if self.tp_rank == 0:
                            logger.debug(
                                "\033[35m [Target][Recv] Received reject from draft \033[0m"
                            )
                        continue
                    if msg.action == RemoteSpecAction.DRAFT:
                        self.req_to_draft_token[msg.request_id][msg.spec_cnt] = (
                            msg.draft_token_ids,
                            msg.draft_logprobs,
                        )
                        has_draft_response = True

            # --- Phase 3: Circuit breaker update ---
            if has_draft_response:
                self.draft_circuit_breaker.record_success()
            else:
                self.draft_circuit_breaker.record_failure()

            # --- Phase 4: Build result from cache ---
            result = {}
            for req in batch.reqs:
                if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                    continue
                rid = req.rid
                spec_cnt = req.spec_cnt
                if rid in self.req_to_draft_token and spec_cnt in self.req_to_draft_token[rid]:
                    drafts = self.req_to_draft_token[rid][spec_cnt]
                    if drafts is not None:
                        result[rid] = drafts
                        try:
                            del self.req_to_draft_token[rid][spec_cnt]
                        except Exception:
                            pass
            logger.debug(f"\033[31m recv_fn took {(time.perf_counter() - start_time) * 1000} ms \033[0m")
            return result

        return recv_draft_fn

    def _make_retry_fn(self):
        """Create a retry callback for re-requesting drafts after verification failure.

        Called from _post_verify_update_drafts when is_matched=False or no draft
        was received.  At that point spec_cnt has already been incremented, so the
        retry uses the next-round spec_cnt key, keeping it disjoint from the
        already-consumed current round.

        The retry request sends cur_drafts=[] (diverged state). The draft returns
        [d0, d1, ...] rooted directly at len(output_ids), so the caller stores all
        tokens without skipping d0 (unlike the normal matched path).

        To avoid a duplicate send in the next regular round, each retried request
        is stamped with skip_draft_send_for_spec_cnt = req.spec_cnt so that
        send_batch_draft_requests can skip it.
        """
        timeout_s = self._recv_timeout_s * 0.5
        num_draft_tokens = self.server_args.speculative_num_steps + 1

        def retry_fn(failed_reqs) -> dict:
            if not failed_reqs:
                return {}

            # --- Step 1: Register pending entries (all ranks need this) ---
            for req in failed_reqs:
                if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                    continue
                self.req_to_draft_token[req.rid][req.spec_cnt] = None
                # req.skip_draft_send_for_spec_cnt = req.spec_cnt

            # --- Step 2: Send retry requests (rank 0 only) ---
            if self.tp_size == 1 or self.tp_rank == 0:
                if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                    retry_send_time = time.perf_counter()
                    reqs_to_send = []
                    for req in failed_reqs:
                        if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                            continue
                        reqs_to_send.append(
                            RemoteSpecRequest(
                                request_id=req.rid,
                                spec_cnt=req.spec_cnt,
                                action=RemoteSpecAction.DRAFT,
                                spec_type=SpecType.DRAFT_REQUEST,
                                output_ids=req.output_ids,
                                draft_token_ids=[],
                                num_draft_tokens=max(1, num_draft_tokens-1),
                                target_send_time=retry_send_time,
                            )
                        )
                    all_drafts_identity = self.zmq_communicator.get_all_drafts_identity()
                    if all_drafts_identity and reqs_to_send:
                        self.zmq_communicator.send_objs(reqs_to_send, all_drafts_identity[0])
                        logger.debug(
                            f"\033[33m [Target][Retry] Sent {len(reqs_to_send)} retry requests "
                            f"for spec_cnts={[r.spec_cnt for r in failed_reqs]} \033[0m"
                        )

            # --- Step 3: Collect responses (rank 0 waits; others skip to broadcast) ---
            if self.tp_size == 1 or self.tp_rank == 0:
                pending_rids = {
                    req.rid
                    for req in failed_reqs
                    if not getattr(req, "rid", "").startswith("HEALTH_CHECK")
                }
                pending_spec_cnt = {
                    req.rid: req.spec_cnt
                    for req in failed_reqs
                    if not getattr(req, "rid", "").startswith("HEALTH_CHECK")
                }
                all_messages = []
                deadline = time.perf_counter() + timeout_s

                while pending_rids:
                    msgs = self._drain_msg_buffer()
                    if msgs:
                        all_messages.extend(msgs)
                        for msg in msgs:
                            if (
                                msg.action == RemoteSpecAction.DRAFT
                                and msg.request_id in pending_rids
                                and msg.spec_cnt == pending_spec_cnt.get(msg.request_id)
                            ):
                                pending_rids.discard(msg.request_id)
                        if not pending_rids:
                            break

                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        logger.debug(
                            f"\033[33m [Target][Retry] Timeout, "
                            f"{len(pending_rids)} requests still pending \033[0m"
                        )
                        break

                    self._data_ready.wait(timeout=remaining)

                messages = all_messages
            else:
                messages = None

            # --- Step 4: TP broadcast ---
            if self.tp_size > 1:
                messages = broadcast_pyobj(
                    messages if messages else [],
                    self.tp_group.rank,
                    self.tp_cpu_group,
                    src=self.tp_group.ranks[0],
                )

            # --- Step 5: Store all messages in req_to_draft_token ---
            # Non-retry messages (for other reqs/spec_cnts) are preserved here
            # so that future recv_draft_fn calls can still retrieve them.
            if messages:
                for msg in messages:
                    if msg.action == RemoteSpecAction.DRAFT:
                        self.req_to_draft_token[msg.request_id][msg.spec_cnt] = (
                            msg.draft_token_ids,
                            msg.draft_logprobs,
                        )

            # --- Step 6: Build result for failed reqs ---
            result = {}
            for req in failed_reqs:
                if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                    continue
                rid = req.rid
                sc = req.spec_cnt
                if rid in self.req_to_draft_token and sc in self.req_to_draft_token[rid]:
                    drafts = self.req_to_draft_token[rid][sc]
                    if drafts is not None:
                        result[rid] = drafts
                        try:
                            del self.req_to_draft_token[rid][sc]
                        except Exception:
                            pass

            logger.debug(
                f"\033[33m [Target][Retry] Got drafts for {len(result)}/{len(failed_reqs)} "
                f"failed reqs \033[0m"
            )
            return result

        return retry_fn

    # =====================================================================
    # Notifications & Helpers
    # =====================================================================

    def notify_draft_request_finished_or_aborted(
        self, req: Req, action: RemoteSpecAction
    ) -> None:
        if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
            return

        finished_or_aborted_req = RemoteSpecRequest(
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
                all_drafts_identity = self.zmq_communicator.get_all_drafts_identity()
                if not all_drafts_identity:
                    logger.warning("当前无 Draft 可用，检查 Draft 状态！")
                    to_send_identity = "NO DRAFT"
                else:
                    to_send_identity = all_drafts_identity[0]
                self.zmq_communicator.send_obj(
                    finished_or_aborted_req, to_send_identity
                )

        try:
            if req.rid in self.req_to_draft_token:
                del self.req_to_draft_token[req.rid]
        except Exception as e:
            if self.tp_rank == 0:
                logger.error(
                    f"[Target][Notify] Failed to delete req_to_draft_token "
                    f"for {req.rid}: {e}"
                )

    def _get_default_draft_tokens_and_logprobs(self) -> Dict[str, torch.Tensor]:
        if self.tokenizer:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 0
        return {
            "draft_tokens": torch.tensor([pad_token_id], dtype=torch.int64, device="cpu"),
            "draft_logprobs": torch.tensor([0.0], dtype=torch.float32, device="cpu"),
        }

    def _is_self_high_overhead_target(self, batch: ScheduleBatch) -> bool:
        current_bsz = max(batch.batch_size(), self.running_batch.batch_size())
        if current_bsz > self.server_args.remote_speculative_max_batch_size:
            batch.is_high_overhead = True
            return True
        batch.is_high_overhead = False
        return False

    def _decide_speculative_num_draft_tokens(
        self, batch: ScheduleBatch
    ) -> int:
        return self.server_args.speculative_num_steps + 1

    def send_batch_draft_requests(
        self, batch: ScheduleBatch, speculative_num_draft_tokens: int
    ) -> None:
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

        # All ranks need to register the pending sentinel so that recv_draft_fn
        # (which runs on every rank) can correctly build pending_rids.
        for req in batch.reqs:
            if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                continue
            self.req_to_draft_token[req.rid][req.spec_cnt] = None

        # Only rank 0 constructs the RemoteSpecRequest objects and does ZMQ I/O.
        # Building these objects on non-rank-0 processes is pure waste because
        # the lists (origin_input_ids, output_ids, cur_drafts) are copied
        # unnecessarily and the objects are never used.
        if self.tp_size == 1 or self.tp_rank == 0:
            if hasattr(self, "zmq_communicator") and self.zmq_communicator is not None:
                # target_send_time = time.perf_counter()
                draft_reqs_to_send = []
                for req in batch.reqs:
                    if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                        continue
                    # Only send sampling_params on the first speculative round
                    # (spec_cnt == 0) to reduce serialization and transmission
                    # overhead on subsequent rounds. The draft side falls back to
                    # a default SamplingParams() when the value is None.
                    draft_reqs_to_send.append(
                        RemoteSpecRequest(
                            request_id=req.rid,
                            spec_cnt=req.spec_cnt,
                            action=RemoteSpecAction.DRAFT,
                            spec_type=SpecType.DRAFT_REQUEST,
                            input_ids=req.origin_input_ids if req.spec_cnt == 0 else None,
                            output_ids=req.output_ids,
                            draft_token_ids=req.cur_drafts,
                            num_draft_tokens=speculative_num_draft_tokens,
                            sampling_params=req.sampling_params if req.spec_cnt == 0 else None,
                            grammar=None,
                            # target_send_time=target_send_time,
                        )
                    )
                all_drafts_identity = self.zmq_communicator.get_all_drafts_identity()
                if not all_drafts_identity:
                    logger.warning("当前无 Draft 可用，检查 Draft 状态！")
                    to_send_identity = "NO DRAFT"
                else:
                    to_send_identity = all_drafts_identity[0]
                self.zmq_communicator.send_objs(draft_reqs_to_send, to_send_identity)
                logger.debug(f"\033[32m [Target][SendBatchDraftRequests] Sent {len(draft_reqs_to_send)} draft requests to draft {to_send_identity} \033[0m")

    def process_reject_action(self) -> None:
        self.is_rejected = True
        self.rejected_forward_ct = self.forward_ct

    def _decide_verify_num_draft_tokens(self, batch: ScheduleBatch) -> int:
        if batch.forward_mode == ForwardMode.EXTEND:
            return self.server_args.speculative_num_draft_tokens

        if self.is_rejected:
            if self.tp_rank == 0:
                logger.debug(
                    "\033[35m [Target] draft_num_tokens=1 (rejected) \033[0m"
                )
            return 1

        no_draft_reqs = 0
        bs = batch.batch_size()
        for req in batch.reqs:
            if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                continue
            if not req.cur_drafts:
                no_draft_reqs += 1
        if no_draft_reqs / bs >= self.server_args.remote_speculative_no_draft_ratio:
            if self.tp_rank == 0:
                logger.debug(
                    f"\033[35m [Target] draft_num_tokens=1 "
                    f"(no_draft_ratio {no_draft_reqs/bs:.2f} >= "
                    f"{self.server_args.remote_speculative_no_draft_ratio}) \033[0m"
                )
            return 1
        return self.server_args.speculative_num_draft_tokens