from dataclasses import dataclass
from typing import Optional, Dict
from sglang.srt.managers.schedule_batch import Req
import threading


@dataclass
class RemoteSpecDraftState:
    req_id: str
    spec_cnt: int
    req_object: Optional[Req]

    last_updated_time: float
    created_time: float

class RemoteSpecDraftStateManager:
    '''
    RemoteSpecDraftStateManager is used in the draft end of remote spec.
    It is responsible for managing the draft states.
    '''
    def __init__(self):
        self.active_draft_states: Dict[str, RemoteSpecDraftState] = {}
        self._lock = threading.Lock()
        self.time_out_cycle = 200

    def get_state(self, req_id: str) -> Optional[RemoteSpecDraftState]:
        with self._lock:
            return self.active_draft_states.get(req_id)
        
    def set_state(self, req_id: str, state: RemoteSpecDraftState):
        with self._lock:
            self.active_draft_states[req_id] = state
    
    def delete_state(self, req_id: str):
        with self._lock:
            del self.active_draft_states[req_id]
    
    def exists(self, req_id: str) -> bool:
        with self._lock:
            return req_id in self.active_draft_states
        

