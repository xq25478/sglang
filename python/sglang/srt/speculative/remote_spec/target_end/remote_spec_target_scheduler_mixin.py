from sglang.srt.utils import DynamicGradMode
from typing import Optional, Union, Tuple
from sglang.srt.managers.schedule_batch import ScheduleBatch, Req
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors, ForwardMode
from sglang.srt.environ import envs
import time
from sglang.srt.speculative.remote_spec.remote_spec_protocol import RemoteSpecRequestFromTargetToDraft, RemoteSpecAction, SpecType, RemoteSpecResponseFromDraftToTarget
from collections import defaultdict
from typing import Dict, List
import logging
import torch
# from sglang.srt.speculative.remote_spec.mock_communicator import MockRemoteSpecWorker

logger = logging.getLogger(__name__)


class RemoteSpecTargetSchedulerMixin:
    '''
    RemoteSpecTargetSchedulerMixin is used in the target end of remote spec.
    It is responsible for scheduling the normal requests and draft responses.
    '''
    @DynamicGradMode()
    def event_loop_normal_remote_spec_target(self):
        #! TODO: move to scheduler.py
        self.req_to_draft_token: Dict[str, Dict[int, Optional[Tuple[List[int], List[float]]]]] = defaultdict(dict)
        self.is_rejected: bool = False
        self.rejected_forward_ct: int = 0
        # self.zmq_worker = MockRemoteSpecWorker()
        while True:
            # Receive requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                if self._is_self_high_overhead(batch):
                    batch.draft_num_tokens = 1
                else:
                    draft_num_tokens = self._decide_speculative_num_draft_tokens(batch)
                    self.send_batch_draft_requests(batch, draft_num_tokens)
                    batch.draft_num_tokens = self._decide_verify_num_draft_tokens(batch)
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()
            
            if self.last_batch is not None and not self._is_self_high_overhead(self.last_batch):
                self._recv_and_process_draft_responses()


    def _recv_and_process_draft_responses(self) -> None:
        messages = self.zmq_worker.recv_all_objs()
        if messages:
            messages = [RemoteSpecResponseFromDraftToTarget.from_dict(msg) for msg in messages]

            target_recv_time = time.perf_counter()

            for msg in messages:
                assert isinstance(msg, RemoteSpecResponseFromDraftToTarget), "Invalid message type"
                if msg.action == RemoteSpecAction.REJECT:
                    self.process_reject_action()
                    continue
                if msg.action == RemoteSpecAction.DRAFT:
                    self.req_to_draft_token[msg.request_id][msg.spec_cnt] = (msg.draft_token_ids, msg.draft_logprobs)
                    # logger.info(f"\033[34m [Target][Recv] Received draft tokens from draft server {msg.request_id}, spec_cnt {msg.spec_cnt},"
                    #             f"TD trans time: {msg.draft_receive_time - msg.target_send_time}," 
                    #             f"D processing time: {msg.draft_send_time - msg.draft_receive_time},"
                    #             f"DT trans time: {target_recv_time - msg.draft_send_time} \033[0m")
            
        if self.last_batch is None:
            return
        
        for req in self.last_batch.reqs:
            if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                continue

            self.update_req_draft_token(req)

            req.spec_cnt += 1
            req.len_output_ids = len(req.output_ids)

    
    def update_req_draft_token(self, req: Req) -> bool:
        rid = req.rid
        spec_cnt = req.spec_cnt

        if rid not in self.req_to_draft_token:
            req.cur_drafts = []
            req.draft_tokens_and_logits = self._get_default_draft_tokens_and_logprobs()
            return
        
        if spec_cnt not in self.req_to_draft_token[rid]:
            req.cur_drafts = []
            req.draft_tokens_and_logits = self._get_default_draft_tokens_and_logprobs()
            return
        
        if not self.req_to_draft_token[rid][spec_cnt]:
            req.cur_drafts = []
            req.draft_tokens_and_logits = self._get_default_draft_tokens_and_logprobs()
            logger.info(f"\033[34m [Target] Still waiting for draft tokens from draft server {rid=} {spec_cnt=}, or send failed \033[0m")
            return
        else:
            draft_tokens, draft_logprobs = self.req_to_draft_token[rid][spec_cnt]
            verified_tokens = req.output_ids[req.len_output_ids:]
            cur_draft_tokens = req.cur_drafts
            cur_draft_tokens.append(draft_tokens[0]) # add first draft token to identify pre/post-verify

            is_matched, matched_idx = self._find_fork_point(verified_tokens, cur_draft_tokens)
            num_draft = len(cur_draft_tokens)
            num_accepted = matched_idx
            req.draft_cnt += num_draft
            req.accept_cnt += num_accepted

            if is_matched:
                req.draft_tokens_and_logits = {
                    "draft_tokens": torch.tensor(cur_draft_tokens, dtype=torch.int64, device="cpu"),
                    "draft_logprobs": torch.tensor(draft_logprobs, dtype=torch.float32, device="cpu"),
                }
                req.cur_drafts = draft_tokens[1:]
                logger.info(f"\033[34m [Target][Verify] Request {rid=} all matched, "
                f"spec_cnt={spec_cnt}, accepted={num_accepted}/{num_draft} \033[0m")
            
            else:
                req.draft_tokens_and_logits = self._get_default_draft_tokens_and_logprobs()
                req.cur_drafts = []
                # logger.info(
                # f"\033[34m [Target][Verify] Request {rid} partial match, "
                # f"spec_cnt={spec_cnt}, accepted={num_accepted}/{num_draft} \033[0m")

            try:
                del self.req_to_draft_token[rid][spec_cnt]
            except Exception as e:
                logger.error(f"\033[34m [Target][Verify] Failed to delete req_to_draft_token for request {rid} spec_cnt {spec_cnt}: {e} \033[0m")


    def notify_draft_request_finished_or_aborted(self, req: Req, action: RemoteSpecAction) -> None:
        if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
            return
        
        finished_or_aborted_req = RemoteSpecRequestFromTargetToDraft(
            request_id=req.rid,
            spec_cnt=req.spec_cnt,
            action=action,
            spec_type=SpecType.DRAFT_REQUEST,
            input_ids=[],
            output_ids=[],
            draft_token_ids=[],
            num_draft_tokens=0,
        )

        success = self.zmq_worker.send_obj(finished_or_aborted_req)
        if not success:
            logger.error(f"\033[34m [Target][Notify] Failed to send finished or aborted request {req.rid} spec_cnt {req.spec_cnt} to draft server, action: {action} \033[0m")

        try:
            if req.rid in self.req_to_draft_token:
                del self.req_to_draft_token[req.rid]
        except Exception as e:
            logger.error(f"\033[34m [Target][Notify] Failed to delete req_to_draft_token for request {req.rid} spec_cnt {req.spec_cnt}: {e} \033[0m")

    def _get_default_draft_tokens_and_logprobs(self) -> Dict[str, torch.Tensor]:
        if self.tokenizer:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 0
        default_draft_tokens = {
            "draft_tokens": torch.tensor([pad_token_id], dtype=torch.int64, device="cpu"),
            "draft_logprobs": torch.tensor([0.0], dtype=torch.float32, device="cpu"),
        }
        return default_draft_tokens


    def _is_self_high_overhead(self, batch: ScheduleBatch) -> bool:
        current_bsz = max(batch.batch_size(), self.running_batch.batch_size())
        if current_bsz > self.server_args.remote_speculative_max_batch_size:
            batch.is_high_overhead = True
            return True
        batch.is_high_overhead = False
        return False

    def _decide_speculative_num_draft_tokens(self, batch: ScheduleBatch) -> int:
        return self.server_args.speculative_num_draft_tokens

    def send_batch_draft_requests(self, batch: ScheduleBatch, speculative_num_draft_tokens: int) -> None:
        if self.is_rejected and (self.rejected_forward_ct + 1) % self.server_args.remote_speculative_reject_interval != 0:
            return

        self.is_rejected = False
        target_send_time = time.perf_counter()
        draft_reqs_to_send = []

        for req in batch.reqs:
            if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                continue

            draft_req = RemoteSpecRequestFromTargetToDraft(
                request_id=req.rid,
                spec_cnt=req.spec_cnt,
                action=RemoteSpecAction.DRAFT,
                spec_type=SpecType.DRAFT_REQUEST,
                input_ids=req.origin_input_ids,
                output_ids=req.output_ids,
                draft_token_ids=req.cur_drafts,
                num_draft_tokens=speculative_num_draft_tokens,
                sampling_params=req.sampling_params,
                grammar=None,
                target_send_time=target_send_time,
            )

            self.req_to_draft_token[req.rid][req.spec_cnt] = None

            draft_reqs_to_send.append(draft_req)

        success = self.zmq_worker.send_objs(draft_reqs_to_send)
        if not success:
            self.req_to_draft_token[req.rid][req.spec_cnt] = None

    
    def process_reject_action(self) -> None:
        self.is_rejected = True
        self.rejected_forward_ct = self.forward_ct


    def _decide_verify_num_draft_tokens(self, batch: ScheduleBatch) -> int:
        if batch.forward_mode == ForwardMode.EXTEND:
            return self.server_args.speculative_num_draft_tokens
        
        no_draft_reqs = 0
        bs = batch.batch_size()
        for req in batch.reqs:
            if getattr(req, "rid", "").startswith("HEALTH_CHECK"):
                continue
            if req.cur_drafts is None:
                no_draft_reqs += 1
        if no_draft_reqs / bs >= self.server_args.remote_speculative_no_draft_ratio:
            return 1
        return self.server_args.speculative_num_draft_tokens


    def _find_fork_point(self, verified_tokens: List[int], draft_tokens: List[int]) -> Tuple[bool, int]:
        if not verified_tokens or not draft_tokens:
            return False, 0
        
        min_len = min(len(verified_tokens), len(draft_tokens))
        
        # Find first mismatch
        for i in range(min_len):
            if verified_tokens[i] != draft_tokens[i]:
                return False, i
        
        # Common prefix matches, check length
        if len(draft_tokens) > len(verified_tokens):
            return False, len(verified_tokens)
        
        # Fully matched
        return True, len(draft_tokens)
