from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


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

@dataclass
class RemoteSpecBatchResponseFromDraftToTarget:
    '''
    RemoteSpecBatchResponseFromDraftToTarget is the batch response for the remote spec.
    It is used to receive the batch response from the draft end to the target end. (D->T)
    '''
    responses: List[RemoteSpecResponseFromDraftToTarget] # responses