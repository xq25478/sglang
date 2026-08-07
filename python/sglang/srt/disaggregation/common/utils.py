import ctypes
import dataclasses
import json
import struct
import threading
from collections import deque
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from sglang.srt.observability.trace import (
    TraceNullContext,
    TraceReqContext,
)


class DSparkHiddenReleaseGuard:
    """Coordinate exactly-once hidden-row release across scheduler and worker.

    The scheduler owns request cleanup while the Mooncake worker owns the
    asynchronous transfer.  Either side may observe a terminal request first,
    so both share this guard and compete for a single release claim.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._owner: Optional[str] = None
        self._worker_finished = False

    def reserve_worker(self) -> bool:
        """Reserve rows before a transfer chunk becomes scheduler-visible."""
        with self._lock:
            if self._owner is not None:
                return False
            self._owner = "worker_reserved"
            return True

    def begin_worker_release(self) -> bool:
        """Claim the one worker-side free after transfer source use is done."""
        with self._lock:
            if self._owner not in (None, "worker_reserved"):
                return False
            self._owner = "worker_releasing"
            return True

    def claim_scheduler(self) -> bool:
        with self._lock:
            if self._owner is not None:
                return False
            self._owner = "scheduler"
            return True

    def mark_worker_finished(self) -> None:
        with self._lock:
            if self._owner != "worker_releasing":
                raise RuntimeError(
                    "DSpark hidden release can finish on the worker only after "
                    "the worker has begun releasing rows"
                )
            self._owner = "worker_finished"
            self._worker_finished = True

    def worker_finished(self) -> bool:
        with self._lock:
            return self._worker_finished


@dataclasses.dataclass
class TransferKVChunk:
    """Work unit for KV cache transfer from prefill to decode."""

    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last_chunk: bool
    prefill_aux_index: Optional[int]
    state_indices: Optional[List]
    chunk_id: Optional[int] = None
    kv_sent: bool = False
    dspark_hidden_packet_idx: int = 0
    dspark_hidden_sent: bool = False
    dspark_hidden_released: bool = False
    dspark_hidden_release_guard: Optional[DSparkHiddenReleaseGuard] = None
    enqueue_time: float = 0.0
    source_event: Optional[Any] = None
    trace_ctx: Union[TraceReqContext, TraceNullContext] = dataclasses.field(
        default_factory=TraceNullContext
    )


def pack_list_of_buffers(buffers: List[bytes]) -> bytes:
    if not buffers:
        return b""
    n = len(buffers)
    header = struct.pack(f"<{n+1}I", n, *(len(b) for b in buffers))
    return header + b"".join(buffers)


def unpack_list_of_buffers(buf: bytes) -> List[bytes]:
    if buf == b"":
        return []
    (n,) = struct.unpack("<I", buf[:4])
    lens = struct.unpack(f"<{n}I", buf[4 : 4 + 4 * n])
    out = []
    offset = 4 + 4 * n
    for length in lens:
        out.append(buf[offset : offset + length])
        offset += length
    return out


def pack_int_lists(lists, fmt: str) -> bytes:
    return pack_list_of_buffers([struct.pack(f"<{len(a)}{fmt}", *a) for a in lists])


def unpack_int_lists(buf: bytes, fmt: str) -> List[List[int]]:
    width = struct.calcsize(fmt)
    return [
        list(struct.unpack(f"<{len(b)//width}{fmt}", b))
        for b in unpack_list_of_buffers(buf)
    ]


def pack_int_list(values, fmt: str) -> bytes:
    if len(values) == 0:
        return b""
    return struct.pack(f"<{len(values)}{fmt}", *values)


def unpack_int_list(buf: bytes, fmt: str) -> List[int]:
    if not buf:
        return []
    width = struct.calcsize(fmt)
    return list(struct.unpack(f"<{len(buf)//width}{fmt}", buf))


def pack_transfer_layout(layout) -> bytes:
    """Pack optional transfer descriptors for ZMQ registration messages."""
    if not layout:
        return b""
    return json.dumps(layout, separators=(",", ":")).encode("utf-8")


def unpack_transfer_layout(buf: bytes):
    if not buf:
        return []
    return [None if x is None else tuple(x) for x in json.loads(buf.decode("utf-8"))]


def pack_nested_transfer_layout(layouts) -> bytes:
    if not layouts:
        return b""
    return json.dumps(layouts, separators=(",", ":")).encode("utf-8")


def unpack_nested_transfer_layout(buf: bytes):
    if not buf:
        return []
    return [
        [None if x is None else tuple(x) for x in component_layout]
        for component_layout in json.loads(buf.decode("utf-8"))
    ]


class FastQueue:
    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()


class AuxDataCodec:
    """Handles serialization and deserialization of auxiliary data buffers."""

    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        """Serialize data from memory buffer to bytes."""
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        """Deserialize bytes into target memory buffer."""
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.int32]]]:
    """Vectorised NumPy implementation."""
    # src/dst indices are transferred pairwise, so an empty side means there is
    # nothing to transfer. Guarding both sides (not just src) avoids a cryptic
    # NumPy broadcast error from np.diff() below when only one side is empty, e.g.
    # a non-empty prefill DSA/SWA state list paired with an empty decode registration.
    if src_indices.size == 0 or dst_indices.size == 0:
        return [], []

    if src_indices.size != dst_indices.size:
        raise ValueError(
            "group_concurrent_contiguous requires equal-length src/dst index arrays, "
            f"got {src_indices.size} and {dst_indices.size}"
        )

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups
