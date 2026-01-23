from sglang.srt.utils import DynamicGradMode
from typing import Optional, Union, Tuple
from sglang.srt.managers.scheduler import EmbeddingBatchResult
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


@DynamicGradMode()
class RemoteSpecTargetSchedulerMixin:
    '''
    RemoteSpecTargetSchedulerMixin is used in the target end of remote spec.
    It is responsible for scheduling the normal requests and draft responses.
    '''
    def event_loop_normal_remote_spec_target(self):
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            if batch:
                self.run_batch(batch)
            else:
                self.self_check_during_idle()
            self.last_batch = batch
            self._recv_and_process_draft_responses()

    
    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""

        if self.spec_algorithm.is_none():
            pass
        else:
            if self._is_self_high_overhead():
                speculative_num_draft_tokens = 1
            else:
                bsz = self.running_batch.batch_size()
                kwargs = {
                    'total_tokens': 0
                }
                speculative_num_draft_tokens = self._decide_speculative_num_draft_tokens(bsz, **kwargs)
                self.send_batch_draft_requests(batch, speculative_num_draft_tokens)
                batch_result = self.model_worker.forward_batch_generation(
                        batch, **kwargs
                    )
                
            ret = GenerationBatchResult()

            return ret

    def _recv_and_process_draft_responses(self) -> None:
        control_msg, draft_reqs = self._recv_draft_responses()
        self._process_control_message(control_msg)
        self._process_draft_requests(draft_reqs)

    def _recv_draft_responses(self) -> None:
        pass

    def _process_control_message(self, control_msg) -> None:
        pass

    def _process_draft_requests(self, draft_reqs) -> None:
        # ...
        # ...
        for req in draft_reqs:
            req.draft_token = {
                "draft_token_ids": draft_token_ids, # [topk, num_draft_tokens]
                "draft_logprobs": draft_logprobs, # [topk, num_draft_tokens]
            }

    def _find_fork_point(self, draft_ids, target_ids) -> Tuple[bool, int]:
        pass

    def _is_self_high_overhead(self) -> bool:
        pass

    def _decide_speculative_num_draft_tokens(self, bsz: int, **kwargs) -> int:
        speculative_num_draft_tokens = [2, 3, 4]
        pass

    def send_batch_draft_requests(self, batch: ScheduleBatch, speculative_num_draft_tokens: int) -> None:
        pass