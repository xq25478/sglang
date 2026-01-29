import time
import threading
import uuid
import remote_spec_zmq as rsz

from dataclasses import dataclass, asdict, field
from typing import Optional, List
from enum import Enum
from random import randint

ADDR = "ipc:///tmp/zmq_msgpack.ipc"
NUM_MSG = 1  # 测试量可调
PAYLOAD_SIZE = 1024  # 可用来生成 dummy draft_token_ids

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
a = rsz.DealerEndpoint("A", ADDR, bind=True)
b = rsz.DealerEndpoint("B", ADDR, bind=False)

a.start()
b.start()

# =====================================================
# sender 函数
# =====================================================
def sender():
    t0 = time.time()
    for i in range(NUM_MSG):
        req = RemoteSpecRequestFromTargetToDraft(
            request_id=str(uuid.uuid4()),
            spec_cnt=i % 10,
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
        a.send_obj(req.to_dict())
    print("Send done:", time.time() - t0)

# =====================================================
# receiver 函数
# =====================================================
def receiver():
    cnt = 0
    t0 = time.time()
    while cnt < NUM_MSG:
        msgs = b.get_received_objs()
        cnt += len(msgs)
        print(msgs)
    print("Recv done:", time.time() - t0, "msgs:", cnt)

# =====================================================
# 启动线程
# =====================================================
ts = threading.Thread(target=sender)
# tr = threading.Thread(target=receiver)

ts.start()
# tr.start()
ts.join()
# tr.join()

a.stop()
b.stop()
