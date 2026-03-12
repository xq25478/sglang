from dataclasses import dataclass, fields
from typing import Optional, List, Dict, Any, get_origin, get_args
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
class RemoteSpecRequest:
    # T 和 D 都有的属性
    request_id: Optional[str] = None
    spec_cnt: Optional[int] = None
    action: RemoteSpecAction = RemoteSpecAction.FINISH
    spec_type: SpecType = SpecType.NORMAL
    draft_token_ids: Optional[List[int]] = None # draft token ids
    target_send_time : float = -1.0
    target_recv_time : float = -1.0
    
            
    # T 独有的属性
    input_ids: Optional[List[int]] = None
    output_ids: Optional[List[int]] = None
    num_draft_tokens: Optional[int] = None
    sampling_params: Optional[SamplingParams] = None # sampling params 
    grammar: Optional[str] = None # grammar
        
    # D 独有的属性
    draft_logprobs: Optional[List[float]] = None # draft logprobs
    draft_recv_time : float = -1.0
    draft_send_time : float = -1.0

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            # 空数据不传递
            if val is not None:
                # 枚举类型统一取 value
                if isinstance(val, Enum):
                    result[f.name] = val.value
                # SamplingParams 类型显式展开
                elif isinstance(val, SamplingParams):
                    if hasattr(val, "to_dict"):
                        result[f.name] = {k: v for k, v in val.to_dict().items() if v is not None}
                    else:
                        raise TypeError(f"{f.name} must have to_dict(), got {type(val)}")
                else:
                    result[f.name] = val
        return result
            
    @classmethod
    def from_dict(cls, d: dict):
        d = dict(d)  # 确保是字典副本

        # 只保留 dataclass 中定义的字段
        field_names = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in field_names}

        init_kwargs = {}
        for f in fields(cls):
            val = d.get(f.name)
            if val is None:
                init_kwargs[f.name] = None
                continue
            # 如果字段是枚举类型，自动转换
            if isinstance(f.type, type) and issubclass(f.type, Enum):
                if not isinstance(val, f.type):
                    init_kwargs[f.name] = f.type(val)
                else:
                    init_kwargs[f.name] = val
            # 如果字段是 SamplingParams，自动调用 from_dict 或手动处理
            elif isinstance(val,dict) and "max_new_tokens" in val:
                stop_strs = val.pop("stop_strs", None)
                stop_regex_strs = val.pop("stop_regex_strs", None)
                sp = SamplingParams(**val)
                if stop_strs:
                    sp.stop_strs = stop_strs
                if stop_regex_strs:
                    sp.stop_regex_strs = stop_regex_strs
                init_kwargs[f.name] = sp
            else:
                init_kwargs[f.name] = val
        return cls(**init_kwargs)
