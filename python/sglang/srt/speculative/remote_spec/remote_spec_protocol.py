from dataclasses import dataclass, fields
from typing import Optional, List, Dict, Any
from enum import Enum
from sglang.srt.sampling.sampling_params import SamplingParams


class RemoteSpecAction(Enum):
    '''
    RemoteSpecAction is the action to take for the remote spec request.
    draft: normal draft request (D->T & T->D)
    finish: when req is finished in target (T->D)
    abort: when req is aborted in target (T->D)
    reject: when draft is high overhead (D->T)
    '''
    DRAFT = 'draft' # normal draft request 
    FINISH = 'finish' # when req is finished in target
    ABORT = 'abort' # when req is aborted in target
    REJECT = 'reject' # when draft is high overhead

class SpecType(Enum):
    '''
    SpecType is the type of the remote spec request. It is used to distinguish the type of the request.
    normal: normal reques
    draft_request: draft request (D->T)
    draft_response: draft response (T->D)
    '''
    NORMAL = 'normal'
    DRAFT_REQUEST = 'draft_request'
    DRAFT_RESPONSE = 'draft_response'

@dataclass
class RemoteSpecRequestFromTargetToDraft:
    '''
    RemoteSpecRequestFromTargetToDraft is the request for the remote spec.
    It is used to send the request to the draft end from the target end. (T->D)
    '''
    request_id: str # req id
    spec_cnt: int # spec cnt with this req id
    action: RemoteSpecAction # action to take
    spec_type: SpecType # spec type

    input_ids: List[int] # input ids
    output_ids: List[int] # output ids
    draft_token_ids: List[int] # draft token ids
    num_draft_tokens: int # number of draft tokens to generate
    sampling_params: Optional[SamplingParams] = None # sampling params 
    grammar: Optional[str] = None # grammar

    # time record for debug
    target_send_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
            # action: enum / str 统一归一化
            if hasattr(self.action, "value"):
                action = self.action.value
            else:
                action = self.action
            # spec_type: enum / str 统一归一化
            if hasattr(self.spec_type, "value"):
                spec_type = self.spec_type.value
            else:
                spec_type = self.spec_type
            # sampling_params: 显式展开
            if self.sampling_params is not None:
                if hasattr(self.sampling_params, "to_dict"):
                    sampling_params = self.sampling_params.to_dict()
                else:
                    raise TypeError(
                        f"sampling_params must have to_dict(), got {type(self.sampling_params)}"
                    )
            else:
                sampling_params = None
            return {
                "request_id": self.request_id,
                "action": action,
                "spec_cnt": self.spec_cnt,
                "spec_type": spec_type,
                "input_ids": self.input_ids,
                "output_ids": self.output_ids,
                "draft_token_ids": self.draft_token_ids,
                "num_draft_tokens": self.num_draft_tokens,
                "sampling_params": sampling_params,
                "grammar": self.grammar,
                "target_send_time": self.target_send_time,
            }
        
    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        # Only keep fields that are defined in the dataclass
        field_names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in field_names}
        
        d["action"] = RemoteSpecAction(d["action"])
        # Convert spec_type string back to enum
        if "spec_type" in d and not isinstance(d["spec_type"], SpecType):
            d["spec_type"] = SpecType(d["spec_type"])
        
        sp = d.get("sampling_params")
        if sp is not None:
            stop_strs = sp["stop_strs"]
            stop_regex_strs = sp["stop_regex_strs"]
            sp.pop("stop_strs")
            sp.pop("stop_regex_strs")
            sampling_params= SamplingParams(**sp)
            sampling_params.stop_strs = stop_strs
            sampling_params.stop_regex_strs = stop_regex_strs
            d["sampling_params"] = sampling_params
            
        return cls(**d)
    
    
@dataclass
class RemoteSpecBatchRequestFromTargetToDraft:
    '''
    RemoteSpecBatchRequestFromTargetToDraft is the batch request for the remote spec.
    It is used to send the batch request to the draft end from the target end. (T->D)
    '''
    requests: List[RemoteSpecRequestFromTargetToDraft] # requests


@dataclass
class RemoteSpecResponseFromDraftToTarget:
    '''
    RemoteSpecResponseFromDraftToTarget is the response for the remote spec.
    It is used to receive the response from the draft end to the target end. (D->T)
    '''
    request_id: str # req id
    spec_cnt: int # spec cnt with this req id
    action: RemoteSpecAction # action to take
    spec_type: SpecType # spec type

    draft_token_ids: Optional[List[int]] = None # draft token ids
    draft_logprobs: Optional[List[float]] = None # draft logprobs

    # time record for debug
    target_send_time: Optional[float] = None
    draft_receive_time: Optional[float] = None
    draft_send_time: Optional[float] = None


    def to_dict(self) -> Dict[str, Any]:
        # action: enum / str 统一归一化
        if hasattr(self.action, "value"):
            action = self.action.value
        else:
            action = self.action
        # spec_type: enum / str 统一归一化
        if hasattr(self.spec_type, "value"):
            spec_type = self.spec_type.value
        else:
            spec_type = self.spec_type
        return {
            "request_id": self.request_id,
            "action": action,
            "spec_cnt": self.spec_cnt,
            "spec_type": spec_type,
            "draft_token_ids": self.draft_token_ids,
            "draft_logprobs": self.draft_logprobs,
            "target_send_time": self.target_send_time,
            "draft_receive_time": self.draft_receive_time,
            "draft_send_time": self.draft_send_time,
        }
        
    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        # Only keep fields that are defined in the dataclass
        field_names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in field_names}
        
        d["action"] = RemoteSpecAction(d["action"])
        # Convert spec_type string back to enum
        if "spec_type" in d and not isinstance(d["spec_type"], SpecType):
            d["spec_type"] = SpecType(d["spec_type"])
        
        return cls(**d)
    
@dataclass
class RemoteSpecBatchResponseFromDraftToTarget:
    '''
    RemoteSpecBatchResponseFromDraftToTarget is the batch response for the remote spec.
    It is used to receive the batch response from the draft end to the target end. (D->T)
    '''
    responses: List[RemoteSpecResponseFromDraftToTarget] # responses