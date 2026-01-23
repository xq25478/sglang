from sglang.srt.utils import DynamicGradMode
from typing import Tuple


class RemoteSpecDraftSchedulerMixin:
    '''
    RemoteSpecDraftSchedulerMixin is used in the draft end of remote spec.
    It is responsible for scheduling the draft requests and normal requests.
    '''
    
    @DynamicGradMode()
    def event_loop_normal_remote_spec_draft(self):
        while True:
            # Process normal and draft requests
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            # process draft requests
            self.recv_and_process_draft_requests()

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                self.self_check_during_idle()
            
            self.last_batch = batch

            if self.draft_forward_cycle % 100 == 0:
                self._cleanup_stale_draft_states()
    
    def recv_and_process_draft_requests(self):
        if self._is_self_high_overhead():
            self._send_reject_message()
            return
        draft_req = self._recv_draft_requests()
        control_msg, draft_reqs = self.deduplicate_draft_requests(draft_req)
        self._process_control_message(control_msg)
        self._process_draft_requests(draft_reqs)
    

    def _recv_draft_requests(self):
        pass

    def deduplicate_draft_requests(self, draft_req):
        pass

    def _process_control_message(self, control_msg):
        pass

    def _process_draft_requests(self, draft_reqs):
        pass

    def _find_fork_point(self, draft_ids, target_ids) -> Tuple[bool, int]:
        pass

    def _is_self_high_overhead(self) -> bool:
        pass

    def _send_reject_message(self):
        pass

    def _cleanup_stale_draft_states(self):
        pass
