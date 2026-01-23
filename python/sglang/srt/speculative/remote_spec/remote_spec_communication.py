from abc import ABC, abstractmethod
from typing import List, Any, Optional

from python.sglang.srt.speculative.remote_spec.remote_spec_protocol import RemoteSpecRequestFromTargetToDraft, RemoteSpecResponseFromDraftToTarget
from python.sglang.srt.speculative.remote_spec.remote_spec_protocol import RemoteSpecBatchResponseFromDraftToTarget, RemoteSpecBatchRequestFromTargetToDraft
class RemoteSpecBaseCommunicator(ABC):
    '''
    RemoteSpecBaseCommunicator is the base class for all remote spec communicators.
    It is responsible for sending and receiving RemoteSpecRequestFromTargetToDraft and RemoteSpecResponseFromDraftToTarget.
    '''
    def __init__(self) -> None:
        self._start()

    @abstractmethod
    def _start(self) -> None:
        pass

    @abstractmethod
    def _stop(self) -> None:
        pass

    @abstractmethod
    def send_batch(self, requests: List[RemoteSpecBatchResponseFromDraftToTarget, RemoteSpecBatchRequestFromTargetToDraft]) -> None:
        pass

    @abstractmethod
    def send_single_request(self, request: [RemoteSpecRequestFromTargetToDraft, RemoteSpecResponseFromDraftToTarget]) -> None:
        pass

    @abstractmethod
    def receive_batch(self) -> List[RemoteSpecBatchResponseFromDraftToTarget, RemoteSpecBatchRequestFromTargetToDraft]:
        pass

    @abstractmethod
    def receive_single_response(self) -> [RemoteSpecRequestFromTargetToDraft, RemoteSpecResponseFromDraftToTarget]:
        pass
