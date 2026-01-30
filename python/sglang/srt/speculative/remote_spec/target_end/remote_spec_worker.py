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
"""

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
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


class RemoteSpecWorker:
    """
    Remote Speculative Decoding Worker.
    
    This worker receives draft tokens from a remote source and uses them
    for speculative decoding verification with the local target model.
    Unlike EAGLEWorker, this worker does not run a local draft model.
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
        # Parse arguments
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
        
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        
    @property
    def draft_model_runner(self):
        return None
    
    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run remote speculative decoding forward.
        
        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(
                batch
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            draft_num_tokens = getattr(batch, 'draft_num_tokens', self.speculative_num_draft_tokens)
            # #! just for testing dynamic draft_num_tokens ===================
            # import random
            # draft_num_tokens = random.randint(1, 4)
            # batch.draft_num_tokens = draft_num_tokens
            # # ==============================================================
            spec_steps = draft_num_tokens - 1 if draft_num_tokens > 1 else 1
            
            logger.info(f"\033[36m[RemoteSpec] draft_num_tokens={draft_num_tokens}, "
                       f"spec_steps={spec_steps}\033[0m")
            
            spec_info = self.construct_draft_input(batch, draft_num_tokens, spec_steps)
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )
            
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )
    
    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor]]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            seq_lens_cpu: Sequence lengths on CPU.
        """
        # Forward with the target model and get hidden states.
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        return (
            logits_output,
            next_token_ids,
            model_worker_batch.seq_lens_cpu,
        )

    def construct_draft_input(
        self, 
        batch: ScheduleBatch, 
        draft_num_tokens: Optional[int] = None,
        spec_steps: Optional[int] = None,
    ) -> EagleVerifyInput:
        """
        Construct EagleVerifyInput from remote draft tokens stored in each request.
        
        This method reads draft tokens from `req.draft_tokens_and_logits` for each
        request in the batch and constructs the tree structure needed for verification.
        
        For topk=1 (linear chain), we directly construct the tree structure without
        going through the complex score_list/token_list/parents_list path.
        
        Args:
            batch: The schedule batch containing requests with draft tokens.
            draft_num_tokens: Optional dynamic draft token count (defaults to self.speculative_num_draft_tokens)
            spec_steps: Optional dynamic speculative steps (defaults to self.speculative_num_steps)
            
        Returns:
            EagleVerifyInput: The constructed verification input.
        """
        # Use provided values or fall back to instance defaults
        num_draft_tokens = draft_num_tokens if draft_num_tokens is not None else self.speculative_num_draft_tokens
        spec_steps = spec_steps if spec_steps is not None else self.speculative_num_steps
        
        # Handle idle mode
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                spec_steps,
                num_draft_tokens,
            )
        
        bs = batch.batch_size()
        device = batch.device
        topk = self.topk
        
        # Update batch.seq_lens_sum and ensure correct dtypes
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        if batch.seq_lens_cpu is None or batch.seq_lens_cpu.sum().item() != batch.seq_lens_sum:
            batch.seq_lens_cpu = batch.seq_lens.cpu()
        
        # Single pass: collect verified_id and draft tokens from each request
        verified_id_list = []
        all_draft_tokens_list = []
        pad_token_id = batch.reqs[0].tokenizer.pad_token_id if batch.reqs[0].tokenizer else 0
        
        for req in batch.reqs:
            # Get verified_id (last output token or last input token)
            verified_id_list.append(
                req.output_ids[-1] if len(req.output_ids) > 0 else req.origin_input_ids[-1]
            )
            
            # Get draft tokens from req.draft_tokens_and_logits
            if req.draft_tokens_and_logits is not None and "draft_tokens" in req.draft_tokens_and_logits:
                req_draft_tokens = req.draft_tokens_and_logits["draft_tokens"]
                if not isinstance(req_draft_tokens, torch.Tensor):
                    req_draft_tokens = torch.tensor(req_draft_tokens, dtype=torch.int64, device=device)
                else:
                    req_draft_tokens = req_draft_tokens.to(device=device, dtype=torch.int64)
            else:
                req_draft_tokens = torch.zeros(spec_steps, dtype=torch.int64, device=device)
            
            # Pad or truncate to spec_steps length
            current_len = req_draft_tokens.numel()
            if current_len < spec_steps:
                req_draft_tokens = torch.nn.functional.pad(
                    req_draft_tokens, (0, spec_steps - current_len), value=pad_token_id
                )
            elif current_len > spec_steps:
                req_draft_tokens = req_draft_tokens[:spec_steps]
            
            all_draft_tokens_list.append(req_draft_tokens)
        
        # Create tensors
        verified_id = torch.tensor(verified_id_list, dtype=torch.int64, device=device)
        draft_tokens = torch.stack(all_draft_tokens_list, dim=0)  # (bs, spec_steps)
        
        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(verified_id)
        
        # For topk=1 (linear chain), directly construct tree structure
        # For topk>1, use the general path via organize_draft_results
        if topk == 1:
            # Linear chain: parent_list = [-1, 0, 1, ..., spec_steps-2] for each batch
            # This represents: token 0 has no parent (-1), token 1's parent is token 0, etc.
            parent_list = torch.arange(-1, spec_steps - 1, dtype=torch.int64, device=device)
            parent_list = parent_list.unsqueeze(0).expand(bs, -1).contiguous()
            
            # top_scores_index: indices [0, 1, ..., num_draft_tokens-2] for each batch
            top_scores_index = torch.arange(num_draft_tokens - 1, dtype=torch.int64, device=device)
            top_scores_index = top_scores_index.unsqueeze(0).expand(bs, -1).contiguous()
            
            # draft_tokens is already (bs, spec_steps), take first num_draft_tokens-1
            draft_tokens = draft_tokens[:, :num_draft_tokens - 1].contiguous()
        else:
            # General case: use organize_draft_results for topk > 1
            parent_list, top_scores_index, draft_tokens = self._construct_tree_structure_general(
                draft_tokens, bs, device, topk, spec_steps, num_draft_tokens
            )
        
        # Build the tree structure using the same kernel as EAGLE
        (
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            final_draft_tokens,  # This includes verified_id prepended
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
        
        # Synchronize to ensure kernel execution is complete
        torch.cuda.synchronize()
        
        logger.debug(
            f"\033[36m[RemoteSpec] build_tree result: \n"
            f"tree_mask shape={tree_mask.shape}, \n"
            f"positions={positions}, \n"
            f"retrive_index={retrive_index}, \n"
            f"retrive_next_token={retrive_next_token}, \n"
            f"retrive_next_sibling={retrive_next_sibling}, \n"
            f"final_draft_tokens={final_draft_tokens} \033[0m"
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

    def _construct_tree_structure_general(
        self,
        all_draft_tokens: torch.Tensor,
        bs: int,
        device: torch.device,
        topk: int,
        spec_steps: int,
        num_draft_tokens: int,
    ):
        """
        Construct tree structure for topk > 1 using organize_draft_results.
        
        This follows EagleWorker's approach of constructing score_list, token_list,
        parents_list to simulate select_top_k_tokens output.
        """
        score_list = []
        token_list = []
        parents_list = []
        
        for i in range(spec_steps):
            if i == 0:
                # Step 0: shape (b, 1, topk), (b, topk), (b, topk+1)
                scores = torch.ones((bs, 1, topk), dtype=torch.float32, device=device)
                tokens = all_draft_tokens[:, 0:1].repeat(1, topk)
                parents = torch.arange(-1, topk, dtype=torch.int64, device=device).unsqueeze(0).repeat(bs, 1)
            else:
                # Step i > 0: shape (b, topk, topk), (b, topk*topk), (b, topk)
                scores = torch.ones((bs, topk, topk), dtype=torch.float32, device=device)
                tokens = all_draft_tokens[:, i:i+1].repeat(1, topk * topk)
                topk_cs_index = torch.zeros((bs, topk), dtype=torch.int64, device=device)
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

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        """
        Run verification with the target model.
        
        Args:
            batch: The schedule batch.
            spec_info: The EagleVerifyInput containing draft tokens and tree structure.
            
        Returns:
            Tuple of (logits_output, verify_output, model_worker_batch, can_run_cuda_graph)
        """
        seq_lens_pre_verify = batch.seq_lens.clone()
        spec_info.prepare_for_verify(batch, self.page_size)
        # Use spec_info.spec_steps for dynamic draft_num_tokens support
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
        
        # Forward with target model
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )
        
        if self.enable_nan_detection:
            detect_nan(logits_output)
        
        spec_info.hidden_states = logits_output.hidden_states
        
        # Run verification
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask=None,  # Grammar not supported in remote spec yet
        )
        
        # Post process based on verified outputs
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]
        
        # Prepare the batch for the next iteration
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input
        
        return logits_output, res, model_worker_batch, can_run_cuda_graph
