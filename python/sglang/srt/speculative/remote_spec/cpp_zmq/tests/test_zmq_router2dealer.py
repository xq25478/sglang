import argparse
import random
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from random import randint
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import remote_spec_zmq as rsz

ADDR = "tcp://127.0.0.1:9122"
POLL_IDLE_SLEEP_S = 0.0002


# =====================================================
# Enum definitions
# =====================================================
class RemoteSpecAction(Enum):
    DRAFT = "draft"
    FINISH = "finish"
    ABORT = "abort"
    REJECT = "reject"


class SpecType(Enum):
    NORMAL = "normal"
    DRAFT_REQUEST = "draft_request"
    DRAFT_RESPONSE = "draft_response"


# =====================================================
# Test-only data structure
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
    target_send_time: float = -1.0
    target_recv_time: float = -1.0
    draft_recv_time: float = -1.0
    draft_send_time: float = -1.0

    def to_dict(self):
        # Build the dict manually to avoid recursive deep copies on large lists.
        return {
            "request_id": self.request_id,
            "spec_cnt": self.spec_cnt,
            "action": self.action,
            "spec_type": self.spec_type,
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
            "draft_token_ids": self.draft_token_ids,
            "num_draft_tokens": self.num_draft_tokens,
            "sampling_params": self.sampling_params,
            "grammar": self.grammar,
            "target_send_time": self.target_send_time,
            "target_recv_time": self.target_recv_time,
            "draft_recv_time": self.draft_recv_time,
            "draft_send_time": self.draft_send_time,
        }


# =====================================================
# Dealer logic
# =====================================================
def run_dealer(identity: str, infinite_loop: bool = False):
    dealer = rsz.DealerEndpoint(ADDR, identity, False)
    dealer.start()

    def sender():
        i = 0
        for i in range(300):
            req = RemoteSpecRequest(
                request_id=str(uuid.uuid4()),
                spec_cnt=i,
                action=RemoteSpecAction.DRAFT.value,
                spec_type=SpecType.DRAFT_REQUEST.value,
                input_ids=[randint(0, 1000) for _ in range(200)],
                output_ids=[randint(0, 1000) for _ in range(200)],
                draft_token_ids=[randint(0, 1000) for _ in range(200)],
                num_draft_tokens=20,
                sampling_params={"top_p": 0.9, "temperature": 1.0},
                grammar="example_grammar",
            )
            payload = req.to_dict()
            batch_size = random.randint(32, 33)
            dealer.send_objs([payload] * batch_size)
            print(f"DEALER SEND:{i=}-{batch_size=}")
            i += 1
            # time.sleep(1)
            if not infinite_loop and i >= 20:  # Send a fixed 20 batches by default.
                break

    def receiver():
        i = 0
        while True:
            msgs = dealer.get_received_objs()
            if msgs:
                print(f"DEALER RECV {i=}-{len(msgs)=}")
                i += 1
                msg = msgs[0]
                d2t_time = (msg["target_recv_time"] - msg["draft_send_time"]) / 1000.0
                t2d_time = (msg["draft_recv_time"] - msg["target_send_time"]) / 1000.0
                print(
                    f"d2t_time={d2t_time:.3f} ms-t2d_time={t2d_time:.3f} ms-count={len(msgs)}"
                )
            else:
                time.sleep(POLL_IDLE_SLEEP_S)
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
# Router logic
# =====================================================
def run_router(infinite_loop: bool = False):
    router = rsz.RouterEndpoint(ADDR, True)
    router.start()

    def router_loop():
        i = 0
        while True:
            msgs = router.get_received_objs()
            if msgs:
                i += 1
                print(f"ROUTER RECV {len(msgs)=} msgs-{i=}")
                router.send_objs(msgs[0][0], [msg for _, msg in msgs])
            else:
                time.sleep(POLL_IDLE_SLEEP_S)
            if not infinite_loop:
                break

    t_router = threading.Thread(target=router_loop)
    t_router.start()
    t_router.join()
    router.stop()
    print("[Router] finished.")


# =====================================================
# CLI entrypoint
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZMQ Router/Dealer test")
    parser.add_argument(
        "--role", choices=["d", "r"], required=True, help="Role: d=dealer, r=router"
    )
    parser.add_argument("--infinite", action="store_true", help="Run in infinite loop mode")
    args = parser.parse_args()

    if args.role == "r":
        run_router(infinite_loop=args.infinite)
    elif args.role == "d":
        identity = "draft-123445"
        run_dealer(identity, infinite_loop=args.infinite)
