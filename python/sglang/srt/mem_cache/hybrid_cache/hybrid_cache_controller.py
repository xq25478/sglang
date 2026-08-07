from __future__ import annotations

import json
import logging
import os
import threading
import time
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import CacheOperation as BaseCacheOperation
from sglang.srt.managers.cache_controller import (
    HiCacheAck,
)
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.managers.cache_controller import (
    LayerDoneCounter,
)
from sglang.srt.managers.cache_controller import (
    StorageOperation as BaseStorageOperation,
)
from sglang.srt.mem_cache.cp_cache_layer_split import is_cp_cache_layer_split_pool
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.memory_pool_host import PoolEntry
from sglang.srt.utils import get_device_module

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)
device_module = get_device_module()


class CacheOperation(BaseCacheOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        node_id: int,
        priority: Optional[int] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, device_indices, node_id, priority)
        self.pool_transfers = pool_transfers

    @staticmethod
    def merge_pool_transfers(
        ops: List[CacheOperation],
    ) -> Optional[list[PoolTransfer]]:
        grouped: dict[tuple[PoolName, Optional[PoolName]], list[PoolTransfer]] = {}
        for op in ops:
            for t in op.pool_transfers or []:
                grouped.setdefault((t.name, t.indices_from_pool), []).append(t)
        if not grouped:
            return None

        def cat_or_none(tensors):
            parts = [x for x in tensors if x is not None]
            return torch.cat(parts) if parts else None

        return [
            PoolTransfer(
                name=ts[0].name,
                host_indices=cat_or_none(t.host_indices for t in ts),
                device_indices=cat_or_none(t.device_indices for t in ts),
                keys=[k for t in ts if t.keys for k in t.keys] or None,
                hit_policy=ts[0].hit_policy,
                indices_from_pool=ts[0].indices_from_pool,
            )
            for ts in grouped.values()
        ]

    @staticmethod
    def merge_ops(ops: List[CacheOperation]) -> CacheOperation:
        if len(ops) == 1:
            return ops[0]
        host_indices = torch.cat([op.host_indices for op in ops])
        device_indices = torch.cat([op.device_indices for op in ops])
        node_ids = []
        priority = min(op.priority for op in ops)
        for op in ops:
            node_ids.extend(op.node_ids)
        merged = CacheOperation(
            host_indices,
            device_indices,
            -1,
            priority,
            pool_transfers=CacheOperation.merge_pool_transfers(ops),
        )
        merged.node_ids = node_ids
        return merged


class StorageOperation(BaseStorageOperation):
    def __init__(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        super().__init__(host_indices, token_ids, last_hash, hash_value, prefix_keys)
        self.pool_transfers = pool_transfers
        self.pool_storage_result = PoolTransferResult.empty()


class PrefetchOperation(StorageOperation):
    def __init__(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        token_ids: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ):
        self.request_id = request_id
        self._lock = threading.Lock()
        self._terminated_flag = False
        self.start_time = time.monotonic()
        super().__init__(
            host_indices,
            token_ids,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=pool_transfers,
        )
        self.pool_transfers_done = not bool(pool_transfers)

    def increment(self, num_tokens: int):
        with self._lock:
            if self._terminated_flag:
                return False
            self.completed_tokens += num_tokens
            return True

    def mark_terminate(self):
        with self._lock:
            self._terminated_flag = True

    def is_terminated(self) -> bool:
        return self._terminated_flag


class HybridCacheController(BaseHiCacheController):
    def __init__(
        self,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        mem_pool_host: Any,
        page_size: int,
        tp_group: torch.distributed.ProcessGroup,
        load_cache_event: threading.Event,
        attn_cp_group: Optional[torch.distributed.ProcessGroup] = None,
        attn_tp_group: Optional[torch.distributed.ProcessGroup] = None,
        pp_group: Optional[torch.distributed.ProcessGroup] = None,
        write_policy: str = "write_through_selective",
        io_backend: str = "",
        storage_backend: Optional[str] = None,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        transfer_layer_num: Optional[int] = None,
        enable_storage_metrics: bool = False,
    ):
        startup_storage_backend = storage_backend
        self.extra_host_mem_release_queues: dict[PoolName, Queue[torch.Tensor]] = {}
        super().__init__(
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            mem_pool_host=mem_pool_host,
            page_size=page_size,
            tp_group=tp_group,
            load_cache_event=load_cache_event,
            attn_cp_group=attn_cp_group,
            attn_tp_group=attn_tp_group,
            pp_group=pp_group,
            write_policy=write_policy,
            io_backend=io_backend,
            storage_backend=None,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
            enable_storage_metrics=enable_storage_metrics,
        )
        self._init_layer_split_l2_write_stats()
        # Override layer_num: hybrid models transfer all layers (For example, Linear Model (KV + Mamba)),
        # not just the full attention layers reported by full_kv_pool.
        if transfer_layer_num is not None and transfer_layer_num != self.layer_num:
            self.layer_num = transfer_layer_num
            self.layer_done_counter = LayerDoneCounter(self.layer_num)

        if startup_storage_backend is not None:
            self.attach_storage_backend(
                storage_backend=startup_storage_backend,
                prefetch_threshold=prefetch_threshold,
                model_name=model_name,
                storage_backend_extra_config=storage_backend_extra_config,
                host_pools=getattr(mem_pool_host, "entries", None),
            )

    @staticmethod
    def _new_layer_split_l2_write_stats() -> dict[str, int | float]:
        return {
            "attempted_write_pages": 0,
            "successful_write_pages": 0,
            "host_pool_allocation_failures": 0,
            "side_pool_allocation_failures": 0,
            "eviction_shortfall_pages": 0,
            "abandoned_write_pages": 0,
            "d2h_submitted_batches": 0,
            "d2h_completed_batches": 0,
            "d2h_anchor_pages": 0,
            "d2h_pool_page_copies": 0,
            "d2h_gpu_ms": 0.0,
            "d2h_max_gpu_ms": 0.0,
            "d2h_enqueue_cpu_ms": 0.0,
        }

    @staticmethod
    def _new_layer_split_l2_pool_allocation_stats() -> dict[str, int | float]:
        return {
            "initial_allocation_misses": 0,
            "requested_slots": 0,
            "eviction_calls": 0,
            "evicted_slots": 0,
            "retry_successes": 0,
            "retry_failures": 0,
            "evict_retry_cpu_ms": 0.0,
            "evict_retry_max_cpu_ms": 0.0,
        }

    def _init_layer_split_l2_write_stats(self) -> None:
        self._layer_split_l2_write_stats_enabled = (
            self.write_policy in ("write_through", "write_through_selective")
            and is_cp_cache_layer_split_pool(self.mem_pool_device)
            and envs.SGLANG_LOG_HICACHE_LAYER_SPLIT_WRITE_STATS.get()
        )
        self._layer_split_l2_write_stats_phase = "cold_fill"
        self._layer_split_l2_write_stats = {
            "cold_fill": self._new_layer_split_l2_write_stats(),
            "saturated": self._new_layer_split_l2_write_stats(),
        }
        self._layer_split_l2_pool_allocation_stats: dict[
            str, dict[str, dict[str, int | float]]
        ] = {"cold_fill": {}, "saturated": {}}
        self._layer_split_l2_d2h_pool_page_copies: dict[str, dict[str, int]] = {
            "cold_fill": {},
            "saturated": {},
        }
        self._layer_split_l2_pending_d2h_timings: list[dict[str, Any]] = []
        self._layer_split_l2_write_stats_interval_seconds = 10.0
        self._layer_split_l2_write_stats_last_log_time = time.monotonic()
        self._layer_split_l2_last_write_failure_reason: Optional[str] = None
        if not self._layer_split_l2_write_stats_enabled:
            return

        self._layer_split_l2_write_stats_interval_seconds = (
            envs.SGLANG_HICACHE_LAYER_SPLIT_WRITE_STATS_INTERVAL_SECONDS.get()
        )
        if self._layer_split_l2_write_stats_interval_seconds <= 0:
            raise ValueError(
                "SGLANG_HICACHE_LAYER_SPLIT_WRITE_STATS_INTERVAL_SECONDS "
                "must be positive"
            )
        logger.info(
            "HiCache LayerSplit L2 write statistics enabled: page_size=%s, "
            "interval_seconds=%s, allocation_retry_timing=True, "
            "d2h_device_timing=True",
            self.page_size,
            self._layer_split_l2_write_stats_interval_seconds,
        )

    def _record_layer_split_l2_pool_allocation_pressure(
        self,
        *,
        phase: str,
        pool_name: PoolName,
        requested_slots: int,
        evicted_slots: int,
        retry_success: bool,
        elapsed_ms: float,
    ) -> None:
        if not getattr(self, "_layer_split_l2_write_stats_enabled", False):
            return
        phase_stats = self._layer_split_l2_pool_allocation_stats[phase]
        stats = phase_stats.setdefault(
            str(pool_name), self._new_layer_split_l2_pool_allocation_stats()
        )
        stats["initial_allocation_misses"] += 1
        stats["requested_slots"] += requested_slots
        stats["eviction_calls"] += 1
        stats["evicted_slots"] += evicted_slots
        stats["retry_successes" if retry_success else "retry_failures"] += 1
        stats["evict_retry_cpu_ms"] += elapsed_ms
        stats["evict_retry_max_cpu_ms"] = max(
            stats["evict_retry_max_cpu_ms"], elapsed_ms
        )

    @staticmethod
    def _normalize_evicted_slots(evicted: Any) -> int:
        if evicted is None:
            return 0
        if isinstance(evicted, torch.Tensor):
            return int(evicted.numel())
        try:
            return int(evicted)
        except (TypeError, ValueError):
            return 0

    def _record_layer_split_l2_d2h_submit(
        self,
        *,
        phase: str,
        start_event: Any,
        finish_event: Any,
        anchor_pages: int,
        pool_pages: dict[str, int],
        enqueue_cpu_ms: float,
    ) -> None:
        if not getattr(self, "_layer_split_l2_write_stats_enabled", False):
            return
        stats = self._layer_split_l2_write_stats[phase]
        stats["d2h_submitted_batches"] += 1
        stats["d2h_anchor_pages"] += anchor_pages
        stats["d2h_pool_page_copies"] += sum(pool_pages.values())
        stats["d2h_enqueue_cpu_ms"] += enqueue_cpu_ms
        phase_pool_pages = self._layer_split_l2_d2h_pool_page_copies[phase]
        for pool_name, pages in pool_pages.items():
            phase_pool_pages[pool_name] = phase_pool_pages.get(pool_name, 0) + pages
        self._layer_split_l2_pending_d2h_timings.append(
            {
                "phase": phase,
                "start_event": start_event,
                "finish_event": finish_event,
            }
        )

    def _poll_layer_split_l2_d2h_timings(self) -> None:
        if not getattr(self, "_layer_split_l2_write_stats_enabled", False):
            return
        pending = []
        for timing in self._layer_split_l2_pending_d2h_timings:
            finish_event = timing["finish_event"]
            query = getattr(finish_event, "query", None)
            if query is None or not query():
                pending.append(timing)
                continue
            try:
                elapsed_ms = float(
                    timing["start_event"].elapsed_time(finish_event)
                )
            except (RuntimeError, TypeError, ValueError):
                logger.debug(
                    "Failed to read HiCache LayerSplit D2H device timing",
                    exc_info=True,
                )
                continue
            stats = self._layer_split_l2_write_stats[timing["phase"]]
            stats["d2h_completed_batches"] += 1
            stats["d2h_gpu_ms"] += elapsed_ms
            stats["d2h_max_gpu_ms"] = max(stats["d2h_max_gpu_ms"], elapsed_ms)
        self._layer_split_l2_pending_d2h_timings = pending

    def begin_layer_split_l2_write(
        self, num_tokens: int
    ) -> Optional[tuple[str, int]]:
        if not getattr(self, "_layer_split_l2_write_stats_enabled", False):
            return None
        pages = (num_tokens + self.page_size - 1) // self.page_size
        phase = self._layer_split_l2_write_stats_phase
        self._layer_split_l2_write_stats[phase]["attempted_write_pages"] += pages
        return phase, pages

    def finish_layer_split_l2_write(
        self,
        ticket: Optional[tuple[str, int]],
        *,
        success: bool,
        failure_reason: Optional[str] = None,
    ) -> None:
        if ticket is None:
            return
        phase, pages = ticket
        stats = self._layer_split_l2_write_stats[phase]
        if success:
            stats["successful_write_pages"] += pages
        else:
            stats["abandoned_write_pages"] += pages
            if failure_reason == "host_allocation":
                stats["host_pool_allocation_failures"] += 1
            elif failure_reason == "side_pool_allocation":
                stats["side_pool_allocation_failures"] += 1
        self._maybe_log_layer_split_l2_write_stats()

    def mark_layer_split_l2_capacity_pressure(self, reason: str) -> None:
        if not getattr(self, "_layer_split_l2_write_stats_enabled", False):
            return
        if self._layer_split_l2_write_stats_phase == "saturated":
            return
        self._layer_split_l2_write_stats_phase = "saturated"
        logger.info(
            "HiCache LayerSplit L2 capacity pressure observed: reason=%s; "
            "subsequent write attempts are recorded in phase=saturated",
            reason,
        )

    def record_layer_split_l2_eviction_shortfall(
        self,
        ticket: Optional[tuple[str, int]],
        *,
        required_tokens: int,
        evicted_tokens: int,
    ) -> None:
        if ticket is None:
            return
        shortfall_tokens = max(0, required_tokens - evicted_tokens)
        shortfall_pages = (shortfall_tokens + self.page_size - 1) // self.page_size
        phase, _ = ticket
        self._layer_split_l2_write_stats[phase][
            "eviction_shortfall_pages"
        ] += shortfall_pages

    def consume_layer_split_l2_write_failure_reason(self) -> Optional[str]:
        reason = self._layer_split_l2_last_write_failure_reason
        self._layer_split_l2_last_write_failure_reason = None
        return reason

    def _maybe_log_layer_split_l2_write_stats(self, *, force: bool = False) -> None:
        if not getattr(self, "_layer_split_l2_write_stats_enabled", False):
            return
        self._poll_layer_split_l2_d2h_timings()
        now = time.monotonic()
        if (
            not force
            and now - self._layer_split_l2_write_stats_last_log_time
            < self._layer_split_l2_write_stats_interval_seconds
        ):
            return
        self._layer_split_l2_write_stats_last_log_time = now
        for phase, stats in self._layer_split_l2_write_stats.items():
            attempted = stats["attempted_write_pages"]
            if attempted == 0:
                continue
            success_rate = 100.0 * stats["successful_write_pages"] / attempted
            completed_d2h = stats["d2h_completed_batches"]
            avg_d2h_gpu_ms = (
                stats["d2h_gpu_ms"] / completed_d2h if completed_d2h else 0.0
            )
            d2h_pool_pages = ";".join(
                f"{pool_name}:{pages}"
                for pool_name, pages in sorted(
                    self._layer_split_l2_d2h_pool_page_copies[phase].items()
                )
            )
            logger.info(
                "HiCache LayerSplit L2 write stats: phase=%s, "
                "attempted_write_pages=%s, successful_write_pages=%s, "
                "success_rate=%.2f%%, host_pool_allocation_failures=%s, "
                "side_pool_allocation_failures=%s, "
                "eviction_shortfall_pages=%s, abandoned_write_pages=%s, "
                "d2h_submitted_batches=%s, d2h_completed_batches=%s, "
                "d2h_anchor_pages=%s, d2h_pool_page_copies=%s, "
                "d2h_gpu_ms=%.3f, d2h_avg_gpu_ms=%.3f, "
                "d2h_max_gpu_ms=%.3f, d2h_enqueue_cpu_ms=%.3f, "
                "d2h_pending_batches=%s, d2h_pool_pages=%s",
                phase,
                attempted,
                stats["successful_write_pages"],
                success_rate,
                stats["host_pool_allocation_failures"],
                stats["side_pool_allocation_failures"],
                stats["eviction_shortfall_pages"],
                stats["abandoned_write_pages"],
                stats["d2h_submitted_batches"],
                completed_d2h,
                stats["d2h_anchor_pages"],
                stats["d2h_pool_page_copies"],
                stats["d2h_gpu_ms"],
                avg_d2h_gpu_ms,
                stats["d2h_max_gpu_ms"],
                stats["d2h_enqueue_cpu_ms"],
                sum(
                    timing["phase"] == phase
                    for timing in self._layer_split_l2_pending_d2h_timings
                ),
                d2h_pool_pages,
            )
            for pool_name, pool_stats in self._layer_split_l2_pool_allocation_stats[
                phase
            ].items():
                events = pool_stats["initial_allocation_misses"]
                avg_retry_ms = (
                    pool_stats["evict_retry_cpu_ms"] / events if events else 0.0
                )
                logger.info(
                    "HiCache LayerSplit L2 allocation stats: phase=%s, pool=%s, "
                    "initial_allocation_misses=%s, requested_slots=%s, "
                    "eviction_calls=%s, evicted_slots=%s, retry_successes=%s, "
                    "retry_failures=%s, evict_retry_cpu_ms=%.3f, "
                    "evict_retry_avg_cpu_ms=%.3f, "
                    "evict_retry_max_cpu_ms=%.3f",
                    phase,
                    pool_name,
                    events,
                    pool_stats["requested_slots"],
                    pool_stats["eviction_calls"],
                    pool_stats["evicted_slots"],
                    pool_stats["retry_successes"],
                    pool_stats["retry_failures"],
                    pool_stats["evict_retry_cpu_ms"],
                    avg_retry_ms,
                    pool_stats["evict_retry_max_cpu_ms"],
                )

    def _start_storage_threads(self):
        super()._start_storage_threads()
        self._init_extra_host_mem_release_queues()

    def attach_storage_backend(
        self,
        storage_backend: str,
        prefetch_threshold: int = 256,
        model_name: Optional[str] = None,
        storage_backend_extra_config: Optional[dict] = None,
        host_pools: Optional[list[PoolEntry]] = None,
    ):
        super().attach_storage_backend(
            storage_backend=storage_backend,
            prefetch_threshold=prefetch_threshold,
            model_name=model_name,
            storage_backend_extra_config=storage_backend_extra_config,
        )

        for entry in host_pools or []:
            self.storage_backend.register_mem_host_pool_v2(entry.host_pool, entry.name)

    @staticmethod
    def parse_storage_backend_extra_config(
        storage_backend_extra_config: Optional[str],
    ) -> tuple[dict, int, float, float, bool]:
        extra_config = {}
        if storage_backend_extra_config:
            if storage_backend_extra_config.startswith("@"):
                path = storage_backend_extra_config[1:]
                ext = os.path.splitext(path)[1].lower()
                with open(path, "rb" if ext == ".toml" else "r") as f:
                    if ext == ".json":
                        extra_config = json.load(f)
                    elif ext == ".toml":
                        import tomllib

                        extra_config = tomllib.load(f)
                    elif ext in (".yaml", ".yml"):
                        import yaml

                        extra_config = yaml.safe_load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file {path} (config format: {ext})"
                        )
            else:
                extra_config = json.loads(storage_backend_extra_config)

        prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
        prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
        prefetch_timeout_per_ki_token = extra_config.pop(
            "prefetch_timeout_per_ki_token", 0.25
        )
        hicache_storage_pass_prefix_keys = extra_config.pop(
            "hicache_storage_pass_prefix_keys", False
        )

        if not isinstance(prefetch_threshold, int):
            raise ValueError(
                f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
            )
        if not isinstance(prefetch_timeout_base, (int, float)):
            raise ValueError(
                f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
            )
        if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
            raise ValueError(
                "prefetch_timeout_per_ki_token must be number, got "
                f"{type(prefetch_timeout_per_ki_token).__name__}"
            )
        if not isinstance(hicache_storage_pass_prefix_keys, bool):
            raise ValueError(
                "hicache_storage_pass_prefix_keys must be bool, got "
                f"{type(hicache_storage_pass_prefix_keys).__name__}"
            )

        return (
            extra_config,
            prefetch_threshold,
            float(prefetch_timeout_base),
            float(prefetch_timeout_per_ki_token),
            hicache_storage_pass_prefix_keys,
        )

    def clear_storage_backend(self) -> bool:
        if not self.enable_storage:
            logger.warning("Hierarchical cache storage backend is not enabled.")
            return False
        if not hasattr(self.storage_backend, "clear"):
            logger.warning(
                "Storage backend %s does not support clear operation.",
                type(self.storage_backend).__name__,
            )
            return False
        self.storage_backend.clear()
        return True

    def _init_extra_host_mem_release_queues(self) -> None:
        self.extra_host_mem_release_queues = {}
        entries = getattr(self.mem_pool_host, "entries", None) or []
        anchor_entry = getattr(self.mem_pool_host, "anchor_entry", None)
        for entry in entries:
            if entry is anchor_entry or entry.is_primary_index_anchor:
                continue
            self.extra_host_mem_release_queues[entry.name] = Queue()

    def _append_host_mem_release_pages(
        self, release_queue: Queue, host_indices: torch.Tensor, page_size: int
    ) -> None:
        if host_indices.numel() == 0:
            return
        for page in host_indices.split(page_size):
            release_queue.put(page)

    def append_host_mem_release(
        self,
        host_indices: Optional[torch.Tensor] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ):
        if host_indices is not None:
            self._append_host_mem_release_pages(
                self.host_mem_release_queue,
                host_indices,
                self.mem_pool_host.page_size,
            )
        for transfer in extra_pools or []:
            if transfer.host_indices is None or transfer.host_indices.numel() == 0:
                continue
            entry = self.mem_pool_host.entry_map.get(transfer.name)
            if (
                entry is None
                or entry.is_primary_index_anchor
                or transfer.indices_from_pool is not None
            ):
                continue
            release_queue = self.extra_host_mem_release_queues.get(transfer.name)
            if release_queue is None:
                continue
            self._append_host_mem_release_pages(
                release_queue, transfer.host_indices, entry.host_pool.page_size
            )

    def reset(self):
        super().reset()
        if hasattr(self, "_layer_split_l2_write_stats"):
            self._layer_split_l2_write_stats_phase = "cold_fill"
            self._layer_split_l2_write_stats = {
                "cold_fill": self._new_layer_split_l2_write_stats(),
                "saturated": self._new_layer_split_l2_write_stats(),
            }
            self._layer_split_l2_pool_allocation_stats = {
                "cold_fill": {},
                "saturated": {},
            }
            self._layer_split_l2_d2h_pool_page_copies = {
                "cold_fill": {},
                "saturated": {},
            }
            self._layer_split_l2_pending_d2h_timings = []
            self._layer_split_l2_write_stats_last_log_time = time.monotonic()
            self._layer_split_l2_last_write_failure_reason = None
        if self.enable_storage:
            self.host_mem_release_queue.queue.clear()
            for release_queue in self.extra_host_mem_release_queues.values():
                release_queue.queue.clear()
            self.prefetch_tokens_occupied = 0

    def write(
        self,
        device_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        self._layer_split_l2_last_write_failure_reason = None
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        if host_indices is None:
            self.mark_layer_split_l2_capacity_pressure(
                reason="anchor_host_pool_allocation"
            )
            self._layer_split_l2_last_write_failure_reason = "host_allocation"
            return None
        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=True,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            self.mem_pool_host.free(host_indices)
            if self._layer_split_l2_last_write_failure_reason is None:
                self._layer_split_l2_last_write_failure_reason = (
                    "side_pool_allocation"
                )
            return None

        self.write_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        self._poll_layer_split_l2_d2h_timings()
        op = CacheOperation.merge_ops(self.write_queue)
        # Page-first write-back JIT kernels can keep destination host indices on CPU.
        if (
            self.io_backend == "kernel"
            and self.mem_pool_host.layout == "page_first"
            and getattr(self.mem_pool_host, "can_use_write_back_jit", False)
        ):
            host_indices = op.host_indices
            device_indices = op.device_indices
            resolved_pool_transfers = op.pool_transfers
        else:
            host_indices, device_indices, resolved_pool_transfers = (
                self.move_hybrid_indices(op)
            )
        self.write_queue.clear()
        start_event = device_module.Event()
        finish_event = device_module.Event()
        timing_enabled = getattr(
            self, "_layer_split_l2_write_stats_enabled", False
        )
        timing_start_event = (
            device_module.Event(enable_timing=True) if timing_enabled else None
        )
        timing_finish_event = (
            device_module.Event(enable_timing=True) if timing_enabled else None
        )
        timing_phase = self._layer_split_l2_write_stats_phase
        anchor_pages = int(op.device_indices.numel())
        # The DSV4 LayerSplit KV anchor is logical-only; count actual side-pool
        # copies separately from its page-address bookkeeping.
        pool_pages: dict[str, int] = {}
        for transfer in resolved_pool_transfers or []:
            if transfer.device_indices is None:
                continue
            pool_name = str(transfer.name)
            pool_pages[pool_name] = pool_pages.get(pool_name, 0) + int(
                transfer.device_indices.numel()
            )
        start_event.record()
        enqueue_start = time.perf_counter()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            if timing_start_event is not None:
                timing_start_event.record()
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                self.io_backend,
                pool_transfers=resolved_pool_transfers,
            )
            if self.has_draft and host_indices.numel() > 0:
                self.mem_pool_host_draft.backup_from_device_all_layer(
                    self.mem_pool_device_draft,
                    host_indices,
                    device_indices,
                    self.io_backend,
                )
            if timing_finish_event is not None:
                timing_finish_event.record()
            finish_event.record()
            self._record_transfer_indices_on_stream(
                self.write_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        enqueue_cpu_ms = (time.perf_counter() - enqueue_start) * 1000.0
        if timing_start_event is not None and timing_finish_event is not None:
            self._record_layer_split_l2_d2h_submit(
                phase=timing_phase,
                start_event=timing_start_event,
                finish_event=timing_finish_event,
                anchor_pages=anchor_pages,
                pool_pages=pool_pages,
                enqueue_cpu_ms=enqueue_cpu_ms,
            )
        self.ack_write_queue.append(HiCacheAck(start_event, finish_event, op.node_ids))

    def load(
        self,
        host_indices: torch.Tensor,
        priority: Optional[int] = None,
        node_id: int = -1,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> Optional[torch.Tensor]:
        need_load_kv = host_indices.numel() > 0

        full_allocator = getattr(
            self.mem_pool_device_allocator,
            "full_attn_allocator",
            self.mem_pool_device_allocator,
        )
        if not need_load_kv:
            device_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
        else:
            device_indices = full_allocator.alloc(len(host_indices))
            if device_indices is None:
                return None

        pool_transfers = self._resolve_pool_transfers_allocation(
            extra_pools,
            alloc_host=False,
            kv_device_indices=device_indices,
            kv_host_indices=host_indices,
        )
        if pool_transfers is None and extra_pools:
            if need_load_kv:
                full_allocator.free(device_indices)
            return None

        self.load_queue.append(
            CacheOperation(
                host_indices,
                device_indices,
                node_id,
                priority,
                pool_transfers=pool_transfers or None,
            )
        )
        return device_indices

    def start_loading(self) -> int:
        if not self.load_queue:
            return -1
        producer_id = self.layer_done_counter.update_producer()
        op = CacheOperation.merge_ops(self.load_queue)
        host_indices, device_indices, resolved_pool_transfers = (
            self.move_hybrid_indices(op)
        )
        self.load_queue.clear()
        producer_event = self.layer_done_counter.events[producer_id]
        producer_event.start_event.record()
        with device_module.stream(self.load_stream):
            producer_event.start_event.wait(self.load_stream)
            for i in range(self.layer_num):
                self.mem_pool_host.load_to_device_per_layer(
                    self.mem_pool_device,
                    host_indices,
                    device_indices,
                    i,
                    self.io_backend,
                    pool_transfers=resolved_pool_transfers,
                )
                if (
                    self.has_draft
                    and host_indices.numel() > 0
                    and i < self.mem_pool_host_draft.layer_num
                ):
                    self.mem_pool_host_draft.load_to_device_per_layer(
                        self.mem_pool_device_draft,
                        host_indices,
                        device_indices,
                        i,
                        self.io_backend,
                    )
                producer_event.complete(i)
            self._record_transfer_indices_on_stream(
                self.load_stream,
                host_indices,
                device_indices,
                resolved_pool_transfers,
            )
        self.ack_load_queue.append(
            HiCacheAck(
                producer_event.start_event,
                producer_event.finish_event,
                op.node_ids,
            )
        )
        return producer_id

    def _record_transfer_indices_on_stream(
        self,
        stream: torch.Stream,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        pool_transfers: Optional[list[PoolTransfer]] = None,
    ) -> None:
        if host_indices.is_cuda:
            host_indices.record_stream(stream)
        if device_indices.is_cuda:
            device_indices.record_stream(stream)
        for transfer in pool_transfers or []:
            if transfer.host_indices is not None and transfer.host_indices.is_cuda:
                transfer.host_indices.record_stream(stream)
            if transfer.device_indices is not None and transfer.device_indices.is_cuda:
                transfer.device_indices.record_stream(stream)

    def prefetch(
        self,
        request_id: str,
        host_indices: torch.Tensor,
        new_input_tokens: List[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> PrefetchOperation:
        operation = PrefetchOperation(
            request_id,
            host_indices,
            new_input_tokens,
            last_hash,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.prefetch_queue.put(operation)
        return operation

    def write_storage(
        self,
        host_indices: torch.Tensor,
        token_ids: List[int],
        hash_value: Optional[List[str]] = None,
        prefix_keys: Optional[List[str]] = None,
        extra_pools: Optional[list[PoolTransfer]] = None,
    ) -> int:
        operation = StorageOperation(
            host_indices,
            token_ids,
            hash_value=hash_value,
            prefix_keys=prefix_keys,
            pool_transfers=extra_pools,
        )
        self.backup_queue.put(operation)
        return operation.id

    def _storage_hit_query(self, operation) -> tuple[list[str], int]:
        hash_value = self.get_hash_str(
            operation.token_ids, operation.last_hash, page_size=self.page_size
        )

        extra_info = HiCacheStorageExtraInfo(
            prefix_keys=operation.prefix_keys.copy() if operation.prefix_keys else None
        )
        if operation.pool_transfers:
            hit_result = self.storage_backend.batch_exists_v2(
                hash_value, operation.pool_transfers, extra_info
            )
        else:
            kv_hit_count = self.storage_backend.batch_exists(hash_value, extra_info)
            hit_result = PoolTransferResult(
                kv_hit_pages=kv_hit_count, extra_pool_hit_pages={}
            )

        kv_hit_pages = hit_result.kv_hit_pages
        operation.pool_storage_result.update_kv_hit_pages(kv_hit_pages)

        return (
            hash_value[:kv_hit_pages],
            kv_hit_pages * self.page_size,
        )

    def move_hybrid_indices(
        self, operation: CacheOperation
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[list[PoolTransfer]]]:
        host_indices, device_indices = self.move_indices(
            operation.host_indices, operation.device_indices
        )
        resolved_pool_transfers = None
        if operation.pool_transfers:
            resolved_pool_transfers = []
            for transfer in operation.pool_transfers:
                transfer_host_indices, transfer_device_indices = self.move_indices(
                    transfer.host_indices, transfer.device_indices
                )
                # Keep the original PoolTransfer unchanged because tree-owned
                # transfers may still reference radix-tree host state. The
                # controller only needs a normalized execution-time copy.
                resolved_pool_transfers.append(
                    PoolTransfer(
                        name=transfer.name,
                        host_indices=transfer_host_indices,
                        device_indices=transfer_device_indices,
                        keys=transfer.keys,
                        hit_policy=transfer.hit_policy,
                        indices_from_pool=transfer.indices_from_pool,
                    )
                )
        return host_indices, device_indices, resolved_pool_transfers

    def _page_transfer(self, operation):
        # KV pools first — determines actual completed page count
        super()._page_transfer(operation)

        # Extra pools only after KV fully completes. If KV terminated early
        # (IO failure, timeout, TP mismatch), skip extra IO entirely to avoid
        # data misalignment.
        kv_completed_pages = operation.completed_tokens // self.page_size
        if operation.pool_transfers and kv_completed_pages == len(operation.hash_value):
            self._sync_trailing_keys(
                operation.pool_transfers, operation.hash_value, kv_completed_pages
            )
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_get_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)
        operation.pool_transfers_done = True

    def _page_backup(self, operation):
        # Backup extra pools
        if operation.pool_transfers:
            self._resolve_sidecar_derived_pool_transfers(operation)
            results = self.storage_backend.batch_set_v2(operation.pool_transfers)
            operation.pool_storage_result.update_extra_pool_hit_pages(results)

        # Backup kv pools
        super()._page_backup(operation)

    def _resolve_sidecar_derived_pool_transfers(self, operation):
        for transfer in operation.pool_transfers:
            if transfer.indices_from_pool is None:
                continue
            if transfer.indices_from_pool != PoolName.KV:
                source = next(
                    (
                        t
                        for t in operation.pool_transfers
                        if t.indices_from_pool is None
                        and t.name == transfer.indices_from_pool
                    ),
                    None,
                )
                if source is None:
                    raise AssertionError(
                        "Storage sidecar derived pool source missing: "
                        f"{transfer.name} from {transfer.indices_from_pool}."
                    )
                transfer.host_indices = source.host_indices
                if transfer.keys is None:
                    transfer.keys = source.keys
            else:
                transfer.host_indices = operation.host_indices
                if transfer.keys is None:
                    transfer.keys = operation.hash_value

    def _sync_trailing_keys(
        self,
        pool_transfers: list[PoolTransfer],
        all_hashes: list[str],
        kv_hit_pages: int,
    ) -> None:
        """Re-align trailing-page sidecar keys after KV hit truncation.

        When the storage hit is shorter than the original target prefix, each
        pool transfer's keys must be updated to the last N hashes of the actual
        hit range instead of the last N hashes of the original target range.
        For mamba (N=1) this is just the last hit page hash; for SWA (N>1) it
        is a sliding window of the last N hit pages.
        """
        for transfer in pool_transfers:
            if transfer.hit_policy != PoolHitPolicy.TRAILING_PAGES:
                continue
            trailing_n = len(transfer.keys) if transfer.keys else 1
            transfer.keys = all_hashes[max(0, kv_hit_pages - trailing_n) : kv_hit_pages]

    def _resolve_pool_transfers_allocation(
        self,
        extra_pools: Optional[list[PoolTransfer]],
        alloc_host: bool,
        kv_device_indices: Optional[torch.Tensor] = None,
        kv_host_indices: Optional[torch.Tensor] = None,
    ) -> Optional[list[PoolTransfer]]:
        """Auto-alloc host or device indices for PoolTransfers where they are None."""
        if not extra_pools:
            return None
        # (pool, free_fn, indices) for atomic rollback on failure.
        newly_allocated: list[tuple[PoolTransfer, Callable, torch.Tensor]] = []
        derived_transfers: list[PoolTransfer] = []

        def rollback_allocated() -> None:
            for prev_pool, prev_free_fn, prev_indices in newly_allocated:
                prev_free_fn(prev_indices)
                if alloc_host:
                    prev_pool.host_indices = None
                else:
                    prev_pool.device_indices = None

        for pool in extra_pools:
            if pool.indices_from_pool is not None:
                derived_transfers.append(pool)
                continue
            entry = self.mem_pool_host.entry_map.get(pool.name)
            if entry is None:
                continue
            if alloc_host:
                if pool.host_indices is not None or pool.device_indices is None:
                    continue
                alloc_fn = entry.host_pool.alloc
                free_fn = entry.host_pool.free
                evict_fn = entry.host_evict_fn
                size = len(pool.device_indices)
            else:
                if pool.device_indices is not None or pool.host_indices is None:
                    continue
                # device_alloc_fn / device_free_fn override entry.device_pool's
                # methods for pools whose device_pool is a raw KV pool (layout)
                # rather than an allocator (e.g. SWA).
                alloc_fn = entry.device_alloc_fn or entry.device_pool.alloc
                free_fn = entry.device_free_fn or entry.device_pool.free
                evict_fn = entry.device_evict_fn
                size = len(pool.host_indices)
            indices = alloc_fn(size)
            if indices is None and evict_fn:
                pressure_phase = self._layer_split_l2_write_stats_phase
                if alloc_host:
                    self.mark_layer_split_l2_capacity_pressure(
                        reason=f"side_pool_allocation:{pool.name}"
                    )
                retry_start = time.perf_counter()
                evicted = evict_fn(size)
                indices = alloc_fn(size)
                retry_elapsed_ms = (time.perf_counter() - retry_start) * 1000.0
                if alloc_host:
                    self._record_layer_split_l2_pool_allocation_pressure(
                        phase=pressure_phase,
                        pool_name=pool.name,
                        requested_slots=size,
                        evicted_slots=self._normalize_evicted_slots(evicted),
                        retry_success=indices is not None,
                        elapsed_ms=retry_elapsed_ms,
                    )
            if indices is None:
                if alloc_host:
                    self.mark_layer_split_l2_capacity_pressure(
                        reason=f"side_pool_allocation:{pool.name}"
                    )
                    self._layer_split_l2_last_write_failure_reason = (
                        "side_pool_allocation"
                    )
                # Atomic rollback: free everything we successfully allocated.
                rollback_allocated()
                return None
            if alloc_host:
                pool.host_indices = indices
            else:
                pool.device_indices = indices
            newly_allocated.append((pool, free_fn, indices))

        # Assign indices to deferred pools from their source.
        for pool in derived_transfers:
            if pool.indices_from_pool == PoolName.KV:
                pool.host_indices = kv_host_indices
                pool.device_indices = kv_device_indices
                continue

            source = next(
                (
                    transfer
                    for transfer in extra_pools
                    if transfer.indices_from_pool is None
                    and transfer.name == pool.indices_from_pool
                ),
                None,
            )
            if source is None:
                rollback_allocated()
                return None
            pool.host_indices = source.host_indices
            pool.device_indices = source.device_indices
        return extra_pools
