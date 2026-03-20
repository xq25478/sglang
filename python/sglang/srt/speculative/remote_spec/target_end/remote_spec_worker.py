"""
Remote Speculative Decoding Worker

This module implements a speculative decoding worker that obtains draft tokens
from a remote source (e.g., another server) instead of running a local draft model.

The draft tokens are expected to be stored in `Req.draft_tokens_and_logits` with
the following format:

    draft_tokens_and_logits = {
        # Required fields for tree construction:
        "draft_tokens": torch.Tensor,      # shape: (num_draft_tokens - 1,) - draft tokens without verified_id
        # Optional field (reserved for future use):
        "draft_probs": torch.Tensor,       # shape: (num_draft_tokens - 1,) - probability of each draft token
                                           # Currently unused. Reserved for future rejection sampling support.
    }

Current verification mode: Target-only (like EAGLE)
- Only uses target model probabilities for verification
- draft_probs is not used (set to zeros internally)
- Simpler and requires less data to transfer over network

The tree structure is built using `build_tree_kernel_efficient` which generates:
- tree_mask: attention mask for the tree structure
- positions: position indices for each draft token
- retrive_index, retrive_next_token, retrive_next_sibling: for tree traversal during verification

Draft state update uses post-verify token comparison (fork-point), NOT kernel
extension.  After the standard EAGLE verify kernel produces accepted tokens +
bonus, we compare [cur_drafts + d0_new] against [verified_tokens] to decide
whether the new draft path is still valid.

Pipeline semantics
------------------
Each decode round:
  1. send_batch_draft_requests: send (output_ids, cur_drafts) to Draft,
     cur_drafts = tree for THIS round (positions pos+1 .. pos+k).
  2. GPU verify with cur_drafts as tree.
  3. recv_drafts_for_batch: Draft returns [d0_new, d1_new, ...] where
     d0_new predicts the bonus position (pos+k+1).
  4. Fork-point: if [cur_drafts + d0_new] fully matches verified_tokens,
     carry forward [d1_new, ...] for next round (pipelined path).
     Otherwise clear cur_drafts and call retry_drafts_for_reqs.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_for_decode
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import detect_nan

logger = logging.getLogger(__name__)

_DEFAULT_DRAFT = {
    "draft_tokens": torch.tensor([0], dtype=torch.int64, device="cpu"),
    "draft_logprobs": torch.tensor([0.0], dtype=torch.float32, device="cpu"),
}


class RemoteSpecWorker:
    """Remote Speculative Decoding Worker.

    Receives draft tokens from a remote source and uses them for speculative
    decoding verification with the local target model.  Unlike EAGLEWorker
    this worker does not run a local draft model.

    Draft state update uses *post-verify fork-point comparison* rather than
    kernel extension, matching the original proven-correct design.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.tp_rank = tp_rank
        self.tp_group = target_worker.get_tp_group()
        self.tp_size = self.tp_group.world_size if self.tp_group else 1

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        self._cached_tree_structures: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}


    @property
    def draft_model_runner(self):
        return None

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker
        self._cached_tree_structures.clear()

    
    # =========================================================================
    # forward
    # =========================================================================
    def forward_batch_generation(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if logger.isEnabledFor(logging.DEBUG):
                start_time = time.perf_counter()

            logits_output, next_token_ids, _ = self.forward_target_extend(batch)

            if logger.isEnabledFor(logging.DEBUG):
                torch.cuda.synchronize()
                logger.debug(
                    f"\033[36m [Target][Extend] forward took "
                    f"{(time.perf_counter() - start_time) * 1000:.1f} ms \033[0m"
                )
            
            self._recv_drafts_after_extend(batch, next_token_ids)
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            draft_num_tokens = getattr(
                batch, "draft_num_tokens", self.speculative_num_draft_tokens
            )

            # When draft_num_tokens=1, skip speculative verify and use normal decode.
            # TARGET_VERIFY with a single token cannot use CUDA graphs and runs slower
            # than a standard DECODE forward pass.
            if draft_num_tokens == 1 and not batch.forward_mode.is_idle():
                logger.debug("\033[36m [Target][forward] use normal decode \033[0m")
                
                if logger.isEnabledFor(logging.DEBUG):
                    start_time = time.perf_counter()

                batch_result = self._forward_normal_decode(batch)

                if logger.isEnabledFor(logging.DEBUG):
                    torch.cuda.synchronize()
                    logger.debug(
                        f"\033[36m [Target][forward] normal decode took "
                        f"{(time.perf_counter() - start_time) * 1000:.1f} ms \033[0m"
                    )
                
                return batch_result

            logger.debug(
                f"\033[36m [Target][forward] speculative decode, "
                f"draft_num_tokens={draft_num_tokens} \033[0m"
            )
            spec_steps = max(draft_num_tokens - 1, 1)
            spec_info = self.construct_draft_input(batch, draft_num_tokens, spec_steps)

            recv_draft_fn = getattr(batch, "recv_draft_fn", None)
            retry_fn = getattr(batch, "retry_fn", None)
            retry_fail_ratio = getattr(batch, "retry_fail_ratio", 0.0)
            retry_min_count = getattr(batch, "retry_min_count", 4)

            if logger.isEnabledFor(logging.DEBUG):
                start_time = time.perf_counter()

            logits_output, verify_output, _, can_run_cuda_graph = self.verify(
                batch,
                spec_info,
                recv_draft_fn=recv_draft_fn,
                retry_fn=retry_fn,
                retry_fail_ratio=retry_fail_ratio,
                retry_min_count=retry_min_count,
            )

            if logger.isEnabledFor(logging.DEBUG):
                torch.cuda.synchronize()
                logger.debug(
                    f"\033[36m [Target][forward] verify took "
                    f"{(time.perf_counter() - start_time) * 1000:.1f} ms \033[0m"
                )
             
                accept_stats = []
                for req, accept_len in zip(
                    batch.reqs, verify_output.accept_length_per_req_cpu
                ):
                    if _is_health_check(req):
                        continue
                    accept_stats.append(f"{req.rid} (accepted_tokens={accept_len})")
                logger.debug(
                    f"\033[36m [Target][Verify] batch_accept_stats="
                    f"[{', '.join(accept_stats)}] \033[0m"
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    # =========================================================================
    # Extend (prefill)
    # =========================================================================

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor]]:
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        return (
            batch_result.logits_output,
            batch_result.next_token_ids,
            model_worker_batch.seq_lens_cpu,
        )

    def _recv_drafts_after_extend(
        self, batch: ScheduleBatch, next_token_ids: torch.Tensor
    ):
        """Receive draft responses after prefill and update per-req draft state.

        Compare target's T0 vs draft's d0:
          - Match  -> store [d1, d2, ...] as cur_drafts for the first decode round.
          - Mismatch -> discard all (draft path diverged at the very first token).
        """
        recv_draft_fn = getattr(batch, "recv_draft_fn", None)
        if recv_draft_fn is None:
            return

        new_drafts = recv_draft_fn(batch)
        target_tokens = next_token_ids.tolist()
        decoding_reqs = getattr(batch, "decoding_reqs", None) or []
        decoding_req_ids = {id(req) for req in decoding_reqs if not _is_health_check(req)}

        for i, req in enumerate(batch.reqs):
            if _is_health_check(req):
                continue
            if (
                id(req) not in decoding_req_ids
                and getattr(req, "is_chunked", 0) > 0
            ):
                # Chunked prefill only uses Draft as an asynchronous warm-up.
                # There is no real T0 yet, so we cannot align d0 or advance the
                # speculative state until the last chunk finishes.
                continue
            token = target_tokens[i] if i < len(target_tokens) else None
            if token is not None:
                _apply_drafts_to_req(req, token, new_drafts.get(req.rid), skip_d0=True)
            else:
                req.cur_drafts = []
                req.draft_tokens_and_logits = _default_draft()

            req.spec_cnt += 1
            # +1 because process_batch_result_prefill has not yet appended T0 to
            # req.output_ids when this function runs.  _post_verify_update_drafts
            # slices output_ids[len_output_ids:] to get "tokens added by the verify
            # kernel", so we must account for that one pending token here.
            req.len_output_ids = len(req.output_ids) + 1

    # =========================================================================
    # Decode: construct -> target forward -> verify -> fork-point
    # =========================================================================

    def construct_draft_input(
        self,
        batch: ScheduleBatch,
        draft_num_tokens: Optional[int] = None,
        spec_steps: Optional[int] = None,
    ) -> EagleVerifyInput:
        """Build ``EagleVerifyInput`` from draft tokens stored in each request."""
        num_draft_tokens = (
            draft_num_tokens
            if draft_num_tokens is not None
            else self.speculative_num_draft_tokens
        )
        spec_steps = (
            spec_steps if spec_steps is not None else self.speculative_num_steps
        )

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk, spec_steps, num_draft_tokens
            )

        bs = batch.batch_size()
        device = batch.device
        topk = self.topk

        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        if (
            batch.seq_lens_cpu is None
            or batch.seq_lens_cpu.sum().item() != batch.seq_lens_sum
        ):
            batch.seq_lens_cpu = batch.seq_lens.cpu()

        verified_id_buf = torch.empty(bs, dtype=torch.int64)
        draft_tokens_buf = torch.zeros(bs, spec_steps, dtype=torch.int64)

        for i, req in enumerate(batch.reqs):
            verified_id_buf[i] = (
                req.output_ids[-1]
                if len(req.output_ids) > 0
                else req.origin_input_ids[-1]
            )
            dtl = req.draft_tokens_and_logits
            if dtl is not None:
                dt = dtl.get("draft_tokens")
                if dt is not None:
                    if isinstance(dt, torch.Tensor):
                        n = min(dt.numel(), spec_steps)
                        draft_tokens_buf[i, :n] = dt[:n].cpu() if dt.is_cuda else dt[:n]
                    else:
                        n = min(len(dt), spec_steps)
                        draft_tokens_buf[i, :n] = torch.tensor(
                            dt[:n], dtype=torch.int64
                        )

        verified_id = verified_id_buf.to(device=device, non_blocking=True)
        draft_tokens = draft_tokens_buf.to(device=device, non_blocking=True)

        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                verified_id
            )

        if topk == 1:
            cached_parent, cached_index = self._get_cached_tree_structure(
                num_draft_tokens, spec_steps
            )
            parent_list = cached_parent.unsqueeze(0).expand(bs, -1).contiguous()
            top_scores_index = cached_index.unsqueeze(0).expand(bs, -1).contiguous()
            draft_tokens = draft_tokens[:, : num_draft_tokens - 1].contiguous()
        else:
            parent_list, top_scores_index, draft_tokens = (
                self._construct_tree_structure_general(
                    draft_tokens, bs, device, topk, spec_steps, num_draft_tokens
                )
            )

        (
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            final_draft_tokens,
        ) = build_tree_kernel_efficient(
            verified_id=verified_id,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=batch.seq_lens,
            seq_lens_sum=batch.seq_lens_sum,
            topk=topk,
            spec_steps=spec_steps,
            num_verify_tokens=num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=final_draft_tokens,
            custom_mask=tree_mask,
            positions=positions,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=spec_steps,
            topk=topk,
            draft_token_num=num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=batch.seq_lens_sum,
            seq_lens_cpu=batch.seq_lens_cpu,
        )

    def verify(
        self,
        batch: ScheduleBatch,
        spec_info: EagleVerifyInput,
        recv_draft_fn=None,
        retry_fn=None,
        retry_fail_ratio: float = 0.0,
        retry_min_count: int = 4,
    ):
        """Target forward -> recv new drafts -> standard verify -> fork-point update.

        The verify kernel runs with the PREVIOUS round's drafts (no d3 extension).
        After verify appends accepted tokens to output_ids, we use fork-point
        comparison to decide whether the new draft path is still valid.
        """

        # ---------- prepare & target forward ----------
        # seq_lens_pre_verify = batch.seq_lens.clone()
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_batch = spec_info.spec_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        if self.enable_nan_detection:
            detect_nan(logits_output)

        # ---------- receive new drafts (overlaps with GPU forward) ----------
        torch.cuda.synchronize()
        new_drafts_per_req: dict = {}
        if recv_draft_fn is not None and not batch.forward_mode.is_idle():
            new_drafts_per_req = recv_draft_fn(batch)

        spec_info.hidden_states = logits_output.hidden_states

        # ---------- standard EAGLE verify kernel (no d3 extension) ----------
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask=None,
        )

        # ---------- post-verify: fork-point draft state update ----------
        self._post_verify_update_drafts(
            batch,
            res,
            new_drafts_per_req,
            retry_fn=retry_fn,
            retry_fail_ratio=retry_fail_ratio,
            retry_min_count=retry_min_count,
        )

        # ---------- post-process ----------
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[
            res.accepted_indices
        ]
        batch.forward_mode = (
            ForwardMode.DECODE
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    # =========================================================================
    # Post-verify draft state update (fork-point comparison)
    # =========================================================================

    def _post_verify_update_drafts(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        new_drafts_per_req: dict,
        retry_fn=None,
        retry_fail_ratio: float = 0.0,
        retry_min_count: int = 4,
    ):
        """Compare verified output tokens with [cur_drafts + d0_new] to update draft state.

        For each request:
          1. verified_tokens = output_ids[len_output_ids:]   (tokens added by verify kernel)
          2. cur_draft_tokens = req.cur_drafts + [d0_new]    (old tree + Draft's bonus prediction)
          3. Fork-point: if all verified_tokens match cur_draft_tokens -> pipelined path
                         otherwise -> retry path

        Pipelined path (is_matched=True):
          d0_new was Draft's prediction for the bonus position and it was correct.
          Carry forward d1_new, d2_new, ... as cur_drafts for the next verify round.

        Retry path (is_matched=False or no draft received):
          Clear cur_drafts; if retry_fn is set and failed ratio exceeds threshold,
          re-request fresh drafts from Draft.  Those drafts are rooted at
          output_ids[-1] so they are stored directly (skip_d0=False).

        spec_cnt contract:
          - First  += 1: regular verify round (counts send_batch_draft_requests call)
          - Second += 1: retry round (counts retry_drafts_for_reqs call)
          Both are intentional: spec_cnt counts total draft generation requests sent.
        """
        failed_reqs: List = []

        for i, req in enumerate(batch.reqs):
            if _is_health_check(req):
                continue

            drafts = new_drafts_per_req.get(req.rid)

            if drafts is not None:
                draft_token_ids, draft_logprobs = drafts
                verified_tokens = req.output_ids[req.len_output_ids:]
                cur_draft_tokens = list(getattr(req, "cur_drafts", []))
                cur_draft_tokens.append(draft_token_ids[0])

                logger.debug(f"\033[36m [Worker][Verify] req-{req.rid}: "
                            f"verified={verified_tokens}, cur_drafts={cur_draft_tokens} \033[0m")

                is_matched, matched_idx = _find_fork_point(
                    verified_tokens, cur_draft_tokens
                )
                req.draft_cnt += len(cur_draft_tokens)
                req.accept_cnt += matched_idx

                if is_matched:
                    # Pipelined path: carry forward d1_new, d2_new, ...
                    req.cur_drafts = list(draft_token_ids[1:])
                    req.draft_tokens_and_logits = _make_draft_dict(
                        draft_token_ids[1:], draft_logprobs[1:]
                    )
                else:
                    req.cur_drafts = []
                    req.draft_tokens_and_logits = _default_draft()
                    failed_reqs.append(req)
            else:
                req.cur_drafts = []
                req.draft_tokens_and_logits = _default_draft()
                failed_reqs.append(req)

            # First spec_cnt increment: counts the regular send_batch_draft_requests call.
            req.spec_cnt += 1
            req.len_output_ids = len(req.output_ids)

        # Retry: re-request fresh drafts for diverged / no-draft requests.
        # spec_cnt has already been incremented to N+1 above.  retry_drafts_for_reqs
        # sends another draft generation request using N+1, so we increment again
        # to N+2 after it returns.  Both increments together count the two separate
        # draft generation requests issued within this decode step.
        bsz = sum(1 for req in batch.reqs if not _is_health_check(req))
        should_retry = (
            retry_fn is not None
            and failed_reqs
            and bsz > 0
            and bsz > retry_min_count
            and len(failed_reqs) / bsz > retry_fail_ratio
        )
        if should_retry:
            retry_drafts = retry_fn(failed_reqs)
            for req in failed_reqs:
                # Retry path: Draft generated from output_ids[-1], store all directly.
                _apply_drafts_to_req(
                    req,
                    verified_token=-1,          # unused when skip_d0=False
                    drafts=retry_drafts.get(req.rid),
                    skip_d0=False,
                )
                logger.debug(
                            f"\033[36m [Worker][Retry] req-{req.rid}: "
                            f"got {retry_drafts.get(req.rid)[0] if retry_drafts.get(req.rid) else []} tokens after retry \033[0m"
                            )
                # Second spec_cnt increment: counts the retry_drafts_for_reqs call.
                req.spec_cnt += 1

    # =========================================================================
    # Normal decode fallback (draft_num_tokens == 1)
    # =========================================================================

    def _forward_normal_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run a standard decode forward when draft_num_tokens==1.

        prepare_for_decode() returns early for speculative algorithms, so we
        replicate the relevant steps here to set up input_ids, KV-cache
        allocation, and seq_lens before calling the target worker in plain
        DECODE mode (no is_verify=True, CUDA-graph eligible).

        There are two reasons draft_num_tokens==1 may be set while recv_draft_fn
        is still present:
          (a) recv_draft_fn is None  – target overloaded / circuit-breaker open;
              no draft requests were sent, nothing to receive.
          (b) recv_draft_fn is set   – requests were already sent via
              send_batch_draft_requests but _decide_verify_num_draft_tokens
              returned 1 (rejected / no-draft-ratio).  We must drain the
              responses to avoid ZMQ buffer build-up and to carry forward
              valid drafts for the next spec round.
        """
        bs = batch.batch_size()

        # ---- Replicate the non-spec parts of prepare_for_decode ----
        last_token_ids_cpu = [
            req.output_ids[-1] if req.output_ids else req.origin_input_ids[-1]
            for req in batch.reqs
        ]
        device = batch.seq_lens.device

        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                torch.tensor(last_token_ids_cpu, dtype=torch.int64, device=device)
            )

        batch.input_ids = torch.tensor(
            last_token_ids_cpu, dtype=torch.int32, device=device
        )
        batch.output_ids = None
        batch.forward_mode = ForwardMode.DECODE
        # Clear stale spec_info from the previous spec round.  If left in place
        # it propagates through get_model_worker_batch -> ForwardBatch and causes
        # _get_actual_ntpb to return N (wrong) instead of 1.
        batch.spec_info = None

        if batch.global_num_tokens is not None:
            batch.global_num_tokens = [bs]
        if batch.global_num_tokens_for_logprob is not None:
            batch.global_num_tokens_for_logprob = [bs]

        batch.out_cache_loc = alloc_for_decode(batch, token_per_req=1)

        for req in batch.reqs:
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        batch.seq_lens.add_(1)
        batch.seq_lens_cpu.add_(1)
        if batch.orig_seq_lens is not None:
            batch.orig_seq_lens.add_(1)
        batch.seq_lens_sum += bs

        # ---- Target forward (normal DECODE, CUDA-graph eligible) ----
        model_worker_batch = batch.get_model_worker_batch()
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)

        # ---- Receive drafts if any were requested (case b above) ----
        recv_draft_fn = getattr(batch, "recv_draft_fn", None)
        new_drafts_per_req: dict = recv_draft_fn(batch) if recv_draft_fn is not None else {}

        # ---- Update per-request state ----
        next_token_ids_list = batch_result.next_token_ids.tolist()
        for i, req in enumerate(batch.reqs):
            if _is_health_check(req):
                continue
            token = next_token_ids_list[i] if i < len(next_token_ids_list) else None
            if token is not None:
                req.output_ids.append(token)
                if req.grammar is not None and not req.finished():
                    try:
                        req.grammar.accept_token(token)
                    except ValueError:
                        logger.error(
                            f"\033[36m [NormalDecode] grammar.accept_token failed "
                            f"for req {req.rid} token {token} \033[0m"
                        )

            # Align received drafts with this decode token (same logic as
            # _recv_drafts_after_extend: skip_d0=True, check d0 == token).
            _apply_drafts_to_req(
                req,
                verified_token=token if token is not None else -1,
                drafts=new_drafts_per_req.get(req.rid) if new_drafts_per_req else None,
                skip_d0=True,
            )
            req.spec_cnt += 1
            req.len_output_ids = len(req.output_ids)

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=batch_result.next_token_ids,
            num_accepted_tokens=0,
            accept_length_per_req_cpu=[1] * bs,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_cached_tree_structure(
        self, num_draft_tokens: int, spec_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._cached_tree_structures.get(num_draft_tokens)
        if cached is None:
            device = self.device
            parent_list = torch.arange(
                -1, spec_steps - 1, dtype=torch.int64, device=device
            )
            top_scores_index = torch.arange(
                num_draft_tokens - 1, dtype=torch.int64, device=device
            )
            cached = (parent_list, top_scores_index)
            self._cached_tree_structures[num_draft_tokens] = cached
        return cached

    def _construct_tree_structure_general(
        self,
        all_draft_tokens: torch.Tensor,
        bs: int,
        device: torch.device,
        topk: int,
        spec_steps: int,
        num_draft_tokens: int,
    ):
        """Build tree structure for topk > 1 via ``organize_draft_results``."""
        score_list, token_list, parents_list = [], [], []
        for i in range(spec_steps):
            if i == 0:
                scores = torch.ones(
                    (bs, 1, topk), dtype=torch.float32, device=device
                )
                tokens = all_draft_tokens[:, 0:1].repeat(1, topk)
                parents = (
                    torch.arange(-1, topk, dtype=torch.int64, device=device)
                    .unsqueeze(0)
                    .repeat(bs, 1)
                )
            else:
                scores = torch.ones(
                    (bs, topk, topk), dtype=torch.float32, device=device
                )
                tokens = all_draft_tokens[:, i : i + 1].repeat(1, topk * topk)
                topk_cs_index = torch.zeros(
                    (bs, topk), dtype=torch.int64, device=device
                )
                parents = topk_cs_index + (topk * topk * (i - 1) + topk)
            score_list.append(scores)
            token_list.append(tokens)
            parents_list.append(parents)

        return organize_draft_results(
            score_list=score_list,
            token_list=token_list,
            parents_list=parents_list,
            num_draft_token=num_draft_tokens,
        )


# ======================================================================
# Module-level helpers (stateless)
# ======================================================================


def _apply_drafts_to_req(
    req,
    verified_token: int,
    drafts: Optional[Tuple[List[int], List[float]]],
    *,
    skip_d0: bool = True,
) -> None:
    """Update req.cur_drafts and req.draft_tokens_and_logits after a decode step.

    Single entry point for all three draft-alignment call sites:
      - _recv_drafts_after_extend   (after prefill, skip_d0=True)
      - _forward_normal_decode      (normal decode fallback, skip_d0=True)
      - _post_verify_update_drafts  (retry path, skip_d0=False)

    Args:
        req: The request to update.
        verified_token: Token produced in this step (bonus or normal decode token).
            Only used when skip_d0=True to check if d0 matches.
        drafts: (token_ids, logprobs) from Draft, or None if nothing received.
        skip_d0:
            True  – normal / pipelined path: draft_token_ids[0] (d0) is Draft's
                    prediction for the bonus position.  Carry forward [1:] only
                    if d0 == verified_token (fork-point held).
            False – retry path: Draft generated starting from output_ids[-1],
                    so all token_ids are valid future tokens; store directly.
    """
    if drafts is not None:
        token_ids, logprobs = drafts
        if token_ids:
            if not skip_d0:
                # Retry path: all tokens are rooted at current output[-1]
                req.cur_drafts = list(token_ids)
                req.draft_tokens_and_logits = _make_draft_dict(token_ids, logprobs)
                return
            if token_ids[0] == verified_token:
                # Normal/pipelined path: d0 matched → carry forward d1, d2, ...
                req.cur_drafts = list(token_ids[1:])
                req.draft_tokens_and_logits = _make_draft_dict(
                    token_ids[1:], logprobs[1:]
                )
                return
    # No draft received, or d0 mismatch (diverged): clear draft state
    req.cur_drafts = []
    req.draft_tokens_and_logits = _default_draft()


def _is_health_check(req) -> bool:
    return getattr(req, "rid", "").startswith("HEALTH_CHECK")


def _default_draft() -> dict:
    return {
        "draft_tokens": _DEFAULT_DRAFT["draft_tokens"].clone(),
        "draft_logprobs": _DEFAULT_DRAFT["draft_logprobs"].clone(),
    }


def _make_draft_dict(token_ids, logprobs) -> dict:
    return {
        "draft_tokens": torch.tensor(token_ids, dtype=torch.int64, device="cpu"),
        "draft_logprobs": torch.tensor(logprobs, dtype=torch.float32, device="cpu"),
    }


def _find_fork_point(
    verified_tokens: List[int], draft_tokens: List[int]
) -> Tuple[bool, int]:
    """Find the first divergence between verified output and draft prediction.

    Returns (is_fully_matched, matched_count).

    is_fully_matched=True means every verified token (including the bonus) was
    correctly predicted by the draft, AND len(draft) <= len(verified).  This is
    the pipelined-carry-forward condition.
    """
    if not verified_tokens or not draft_tokens:
        return False, 0

    min_len = min(len(verified_tokens), len(draft_tokens))
    for i in range(min_len):
        if verified_tokens[i] != draft_tokens[i]:
            return False, i

    if len(draft_tokens) > len(verified_tokens):
        return False, len(verified_tokens)

    return True, len(draft_tokens)
