import time
import threading
import uuid
import argparse
import remote_spec_zmq as rsz
from dataclasses import dataclass, asdict, field
from typing import Optional, List
from enum import Enum
from random import randint

ADDR = "tcp://127.0.0.1:9122"

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
# 测试用结构体
# =====================================================
@dataclass
class RemoteSpecRequest:
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
        return asdict(self)

# =====================================================
# Dealer 逻辑
# =====================================================
def run_dealer(identity: str, infinite_loop: bool = False):
    dealer = rsz.DealerEndpoint(ADDR, identity,False)
    dealer.start()
    def sender():
        i = 0
        while True:
            req = RemoteSpecRequest(
                request_id=str(uuid.uuid4()),
                spec_cnt=i,
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
            dealer.send_objs([req.to_dict()])
            dealer.send_obj(req.to_dict())
            # dealer.send_heartbeat()
            # dealer.reconnect(
            # dealer.stop()
            # dealer.start()
            # time.sleep(1)
            # dealer = rsz.DealerEndpoint(identity, ADDR, identity=identity, bind=False)
            # dealer.start()
            # dealer.send_heartbeat()
    
            print(f"DEALER SEND:{i=}")
            i += 1
            time.sleep(0.01)
            if not infinite_loop and i >= 20:  # 默认固定发送 20 条
                break

    def receiver():
        i = 0
        while True:
            msgs = dealer.get_received_objs()
            if msgs:
                print(f"DEALER RECV {i=}")
                i+=1
            for msg in msgs:
                # print(f"[{identity}] recv {msg}")
                pass
            time.sleep(0.01)
            if not infinite_loop:
                break

    t_send = threading.Thread(target=sender)
    t_recv = threading.Thread(target=receiver)
    t_send.start()
    t_recv.start()
    t_send.join()
    t_recv.join()

    dealer.stop()
    print(f"[Dealer {identity}] finished.")

# =====================================================
# Router 逻辑
# =====================================================
def run_router(infinite_loop: bool = False):
    router = rsz.RouterEndpoint(ADDR,True)
    router.start()

    def router_loop():
        i = 0
        while True:
            msgs = router.get_received_objs()
            # msgs = []
            if msgs:
                i +=1
                for identity, msg in msgs:
                    router.send_objs(identity, [msg])
                    router.send_obj(identity, [msg])
                    print(f"ROUTER LOOP {i=}")
            else:
                time.sleep(0.001)
            if not infinite_loop:
                break

    t_router = threading.Thread(target=router_loop)
    t_router.start()
    t_router.join()
    router.stop()
    print("[Router] finished.")

# =====================================================
# CLI 入口
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZMQ Router/Dealer test")
    parser.add_argument("--role", choices=["d", "r"], required=True, help="Role: d=dealer, r=router")
    parser.add_argument("--infinite", action="store_true", help="Run in infinite loop mode")
    args = parser.parse_args()

    if args.role == "r":
        run_router(infinite_loop=args.infinite)
    elif args.role == "d":
        identity = f"draft-123445"
        run_dealer(identity, infinite_loop=args.infinite)
