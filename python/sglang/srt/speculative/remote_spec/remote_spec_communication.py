from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict, Tuple, Union

from sglang.srt.speculative.remote_spec.remote_spec_protocol import RemoteSpecRequestFromTargetToDraft, RemoteSpecResponseFromDraftToTarget
from sglang.srt.speculative.remote_spec.remote_spec_protocol import RemoteSpecBatchResponseFromDraftToTarget, RemoteSpecBatchRequestFromTargetToDraft
from sglang.srt.environ import envs, EnvStr

from dataclasses import dataclass
import os
import time
import logging
from sglang.srt.server_args import ServerArgs
    
logger = logging.getLogger(__name__)


class RemoteSpecBaseCommunicator(ABC):
    '''
    RemoteSpecBaseCommunicator is the base class for all remote spec communicators.
    It is responsible for sending and receiving RemoteSpecRequestFromTargetToDraft and RemoteSpecResponseFromDraftToTarget.
    '''
    def __init__(self) -> None:
        self.start()

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def send_objs(self, requests: List[Union[RemoteSpecRequestFromTargetToDraft, RemoteSpecResponseFromDraftToTarget]]) -> None:
        pass

    @abstractmethod
    def send_obj(self, request: Union[RemoteSpecRequestFromTargetToDraft, RemoteSpecResponseFromDraftToTarget]) -> None:
        pass

    @abstractmethod
    def recv_all_objs(self) -> List[Union[RemoteSpecBatchResponseFromDraftToTarget, RemoteSpecBatchRequestFromTargetToDraft]]:
        pass


@dataclass
class RemoteSpecConfig:
    """
    Configuration for Remote Speculative Decoding.
    
    This class encapsulates all configuration needed for remote speculative
    decoding, including role, network settings, algorithm parameters, and
    hardware configuration.
    
    Attributes:
        role: Server role ("Target" or "Draft")
        draft_as_router: IP address of the Draft server
        zmq_port: Draft and Target Port for send and recv messages.
        zmq_timeout: ZMQ communication timeout in seconds
        num_draft_tokens: Number of draft tokens per round
        topk: Tree width for speculative decoding
        promote_interval: Interval for promoting stable boundary
        page_size: KV cache page size
        tp_size: Tensor parallel size
        enable_cuda_graph: Whether CUDA graph is enabled
    """
    
    # Role configuration
    role: str = "Target" # "Target" or "Draft"
    
    # Network configuration
    zmq_addr: str = "127.0.0.1"
    zmq_port: str = "30009"
    
    # Algorithm configuration
    num_draft_tokens: int = 5
    topk: int = 1
    promote_interval: int = 50
    
    # Hardware configuration
    page_size: int = 1
    tp_size: int = 1
    enable_cuda_graph: bool = False
    
    # transport
    zmq_transport: str = "tcp" # 跨机 tcp / 同机 ipc 跳过网卡 性能最优   
    
    @classmethod
    def from_server_args(cls, server_args: ServerArgs) -> "RemoteSpecConfig":
        """
        Create a RemoteSpecConfig from ServerArgs.
        
        Args:
            server_args: Server arguments from command line
            
        Returns:
            Configured RemoteSpecConfig instance
        """
        zmq_addr = server_args.remote_speculative_zmq_addr or "127.0.0.1"
        
        # 单机最佳 ipc 多机选择 tcp
        if zmq_addr in ["127.0.0.1", "0.0.0.0"]:
            zmq_transport = "ipc"
        else:
            zmq_transport = "tcp"
        
        return cls(
            role=server_args.remote_speculative_role,
            zmq_addr=zmq_addr,
            zmq_port=server_args.remote_speculative_zmq_port or "30009",
            num_draft_tokens=server_args.speculative_num_steps or 5,
            topk=server_args.speculative_eagle_topk or 1,
            page_size=server_args.page_size,
            tp_size=server_args.tp_size,
            enable_cuda_graph=not server_args.disable_cuda_graph,
            zmq_transport=zmq_transport,
        )
    
    @property
    def is_target(self) -> bool:
        """Check if this is the Target server."""
        return self.role == "Target"
    
    @property
    def is_draft(self) -> bool:
        """Check if this is the Draft server."""
        return self.role == "Draft"
    
    @property
    def supports_tree_draft(self) -> bool:
        """Check if tree-structured draft is supported (topk > 1)."""
        return self.topk > 1
    
    @property
    def supports_paged_kv(self) -> bool:
        """Check if paged KV cache is in use (page_size > 1)."""
        return self.page_size > 1
    
    def get_addr(self) -> str:            
        return f"{self.zmq_transport}://{self.zmq_addr}:{self.zmq_port}"
                
    def validate(self) -> None:
        """
        Validate configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.role not in ("Target", "Draft"):
            raise ValueError(f"Invalid role: {self.role}. Must be 'Target' or 'Draft'")
        
        if self.num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {self.num_draft_tokens}")
        
        if self.topk < 1:
            raise ValueError(f"topk must be >= 1, got {self.topk}")
        
        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {self.page_size}")
        
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {self.tp_size}")
    
    def __repr__(self) -> str:
        return (
            f"RemoteSpecConfig("
            f"role={self.role}, "
            f"zmq_addr={self.zmq_addr}, "
            f"num_draft_tokens={self.num_draft_tokens}, "
            f"topk={self.topk}, "
            f"page_size={self.page_size}, "
            f"tp_size={self.tp_size})"
        )
               
class RemoteSpecZMQCommunicator(RemoteSpecBaseCommunicator):
    """
    ZMQ-based communication backend.
    
    This implementation:
    - Uses C++ ZMQ backend (DealerEndpoint) for efficient async communication
    - Supports smart deduplication (keeps only latest spec_cnt per request)
    - Uses batch serialization for efficiency
    - Runs receive in a separate process to avoid GIL
    
    Args:
        config: Remote speculative decoding configuration
        context: Optional ZMQ context (created if not provided)
    """
    # _instance = None
    
    # # 单例模式
    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super().__new__(cls,*args,**kwargs)
    #     return cls._instance
    
    def __init__(
        self,
        config: RemoteSpecConfig,
    ):
        from sglang.srt.speculative.remote_spec.cpp_zmq import DealerEndpoint
        
        self.config = config
        # 引入 debug 选项，用于耗时分析。
        self.debug = int(os.getenv("REMOTE_SPEC_DEBUG","0"))
        # Determine addresses based on role
        self.zmq_endpoint = config.get_addr()
        self.bind = not config.is_target
        self._running = False
        self.zmq_communicator = DealerEndpoint(  
                                self.config.role, 
                                self.zmq_endpoint, 
                                bind = self.config.role == "Target")
        
    def get_endpoint(self) -> str:
        return self.zmq_endpoint
        
    def start(self) -> None:
        """Start the ZMQ communication workers."""
        if self._running:
            logger.warning("ZMQCommunicator already started")
            return
        
        if not self._running:
            self.zmq_communicator.start()
            self._running = True
            print(f"ZMQ Communicator Started for {self.config.role}")

    def stop(self) -> None:
        if self._running:
            self.zmq_communicator.stop()
            self._running = False
            print(f"ZMQ Communicator Stoped for {self.config.role}")
        
    def _process_data(self, data:Any):
        if isinstance(data, RemoteSpecRequestFromTargetToDraft) or isinstance(data, RemoteSpecResponseFromDraftToTarget):
            return data.to_dict()
        return data
            
    def send_obj(self, request: Union[RemoteSpecRequestFromTargetToDraft, RemoteSpecResponseFromDraftToTarget]) -> None:
        if not self._running:
            logger.warning(f"Cannot send: communicator not running")
        try:
            if self.debug:
                t1 = time.perf_counter()
                
            # ZMQ C++ 端采用 msgpack python 进行打包发送，要求数据格式为 Python 原生数据类型，因此需要将Response 和 Request 转换为 Dict
            self.zmq_communicator.send_obj(self._process_data(request))
            
            if self.debug:
                t2 = time.perf_counter()
                print(f"[REMOTE_SPEC_DEBUG Python][SEND] msgs num:1, time_us:{(t2-t1)*1e6:.1f}")
        except Exception as e:
            logger.error(f"Failed to send: {e}")
        
    def send_objs(self, requests: List[Union[RemoteSpecRequestFromTargetToDraft, RemoteSpecResponseFromDraftToTarget]]) -> None:
        if not self._running:
            logger.warning(f"Cannot send: communicator not running")
        try:
            if self.debug:
                t1 = time.perf_counter()
                
            msgs = [self._process_data(request) for request in requests]
            self.zmq_communicator.send_objs(msgs)
            
            if self.debug:
                t2 = time.perf_counter()
                print(f"[REMOTE_SPEC_DEBUG Python][SEND] msgs nums:{len(msgs)}, time_us:{(t2-t1)*1e6:.1f}")    
        except Exception as e:
            logger.error(f"Failed to send: {e}")
        
    def recv_all_objs(self) -> List[Any]:
        if not self._running:
            return []
        try:
            if self.debug:
                t1 = time.perf_counter()
                
            _msgs = self.zmq_communicator.get_received_objs()

            if self.debug and _msgs:
                t2 = time.perf_counter()
                print(f"[REMOTE_SPEC_DEBUG Python][RECV] msgs nums:{len(_msgs)}, time_us:{(t2-t1)*1e6:.1f}")
                
            if not _msgs:
                return []
            else:
                msgs = []
                for _msg in _msgs:
                    if isinstance(_msg, RemoteSpecRequestFromTargetToDraft):
                        msgs.append(RemoteSpecRequestFromTargetToDraft.from_dict(_msg))
                        
                    elif isinstance(_msg, RemoteSpecResponseFromDraftToTarget):
                        msgs.append(RemoteSpecResponseFromDraftToTarget.from_dict(_msg))
                        
                    else:
                        msgs.append(_msg)
                return msgs
                        
        except Exception as e:
            logger.error(f"Failed to receive: {e}")
            return []
    
    def is_running(self) -> bool:
        """Check if the communicator is running."""
        return self._running
    
    
if __name__ == "__main__":
    import threading
    import uuid
    import time
    from random import randint
    from sglang.srt.sampling.sampling_params import SamplingParams

    PAYLOAD_TOKENS = 20_000

    remote_config_t = RemoteSpecConfig(role="Target")
    remote_config_d = RemoteSpecConfig(role="Draft")

    zmq_comm_t = RemoteSpecZMQCommunicator(remote_config_t)
    zmq_comm_d = RemoteSpecZMQCommunicator(remote_config_d)

    zmq_comm_t.start()
    zmq_comm_d.start()

    def make_req(i: int):
        return RemoteSpecRequestFromTargetToDraft(
            request_id=str(uuid.uuid4()),
            spec_cnt=i,
            action="draft",
            spec_type="draft_request",
            input_ids=[randint(0, 32000) for _ in range(PAYLOAD_TOKENS)],
            output_ids=[],
            draft_token_ids=[randint(0, 32000) for _ in range(PAYLOAD_TOKENS)],
            num_draft_tokens=5,
            sampling_params=SamplingParams(temperature=1.0),
            grammar=None,
            target_send_time=time.time(),
        )
        
    for i in range(10):    
        zmq_comm_t.send_obj(make_req(0))
        zmq_comm_t.send_objs([make_req(i) for i in range(30)])
        zmq_comm_d.send_obj(make_req(4))
        zmq_comm_t.send_objs([make_req(i) for i in range(30)])
        msgs = zmq_comm_t.recv_all_objs()
        # print(f"{msgs=}")
        msgs = zmq_comm_d.recv_all_objs()
        # print(f"{msgs=}")

    zmq_comm_t.stop()
    zmq_comm_d.stop()
    