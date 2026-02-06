import time
import threading
import uuid
import remote_spec_zmq as rsz

from dataclasses import dataclass, asdict, field
from typing import Optional, List
from enum import Enum
from random import randint

ADDR = "ipc:///tmp/zmq_msgpack.ipc"
NUM_MSG = 100  # 测试量可调
PAYLOAD_SIZE = 100  # 可用来生成 dummy draft_token_ids

# =====================================================
# 枚举定义
# =====================================================
class RemoteSpecAction(Enum):
    DRAFT = 'draft'
    FINISH = 'finish'
    ABORT = 'abort'
    REJECT = 'reject'

class SpecType(Enum):
    NORMAL = 'normal'
    DRAFT_REQUEST = 'draft_request'
    DRAFT_RESPONSE = 'draft_response'

# =====================================================
# 测试用 RemoteSpecRequestFromTargetToDraft
# =====================================================
@dataclass
class RemoteSpecRequestFromTargetToDraft:
    request_id: str
    spec_cnt: int
    action: str
    spec_type: str
    input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)
    draft_token_ids: List[int] = field(default_factory=list)
    num_draft_tokens: int = 0
    sampling_params: Optional[dict] = None
    grammar: Optional[str] = None
    target_send_time: Optional[float] = None

    def to_dict(self):
        return asdict(self)  # dataclass -> dict

# =====================================================
# 初始化两个 DEALER
# =====================================================
a = rsz.DealerEndpoint("B", ADDR, bind=False)
a.start()

req = RemoteSpecRequestFromTargetToDraft(
    request_id=str(uuid.uuid4()),
    spec_cnt = 0,
    action=RemoteSpecAction.DRAFT.value,
    spec_type=SpecType.DRAFT_REQUEST.value,
    input_ids=[randint(0, 1000) for _ in range(10)],
    output_ids=[randint(0, 1000) for _ in range(10)],
    draft_token_ids=[randint(0, 1000) for _ in range(20)],
    num_draft_tokens=20,
    sampling_params={"top_p": 0.9, "temperature": 1.0},
    grammar="example_grammar",
    target_send_time=time.time()
)
        
while 1:
    time.sleep(0.2)
    a.send_obj(req.to_dict())
    print("Send")
    msgs = a.get_received_objs()
    if msgs:
        print("Recv")
