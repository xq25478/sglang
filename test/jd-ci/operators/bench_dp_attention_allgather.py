import json
import os
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist


os.environ["SGLANG_DP_ATTN_COMPRESSED_ALLGATHER"] = "true"
os.environ["SGLANG_ENABLE_METRICS_DP_ATTENTION"] = "false"

from sglang.srt.managers.scheduler_components import dp_attn


MLPSyncBatchInfo = dp_attn.MLPSyncBatchInfo


def samples(function, warmup=20, repeats=100):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise AssertionError(f"JD compressed all-gather benchmark requires 2 ranks, got {world}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    info = MLPSyncBatchInfo(
        dp_size=world,
        tp_size=1,
        cp_size=1,
        num_tokens=1024 + rank,
        num_tokens_for_logprob=128,
        can_cuda_graph=True,
        is_extend_in_batch=False,
        local_can_run_tbo=True,
        local_forward_mode=3,
        can_run_breakable_cuda_graph=True,
    )
    active = torch.ones(world, dtype=torch.int32, device=device)
    tp_group = SimpleNamespace(active_ranks=active, active_ranks_cpu=active.cpu())

    def compressed():
        info.all_gather(device=device, group=dist.group.WORLD)

    legacy_info = MLPSyncBatchInfo(
        dp_size=world,
        tp_size=1,
        cp_size=1,
        num_tokens=1024 + rank,
        num_tokens_for_logprob=128,
        can_cuda_graph=True,
        is_extend_in_batch=False,
        local_can_run_tbo=True,
        local_forward_mode=3,
        can_run_breakable_cuda_graph=True,
    )

    def legacy():
        legacy_info.all_gather(device=device, group=dist.group.WORLD)

    with patch(
        "sglang.srt.managers.scheduler_components.dp_attn.get_tp_group",
        return_value=tp_group,
    ):
        with patch.object(dp_attn, "_ENABLE_COMPRESSED_ALLGATHER", True):
            compressed_ms = samples(compressed)
        with patch.object(dp_attn, "_ENABLE_COMPRESSED_ALLGATHER", False):
            legacy_ms = samples(legacy)
    compressed_bytes = 5 * torch.tensor([], dtype=torch.int32).element_size()
    legacy_bytes = 7 * torch.tensor([], dtype=torch.int64).element_size()
    ratio = median(compressed_ms) / median(legacy_ms)
    result = {
        "operator": "dp_attention_allgather",
        "compressed_bytes_per_rank": compressed_bytes,
        "legacy_bytes_per_rank": legacy_bytes,
        "compressed_median_ms": median(compressed_ms),
        "legacy_median_ms": median(legacy_ms),
        "latency_ratio": ratio,
        "max_latency_ratio": 1.5,
        "samples": len(compressed_ms),
    }
    if compressed_bytes >= legacy_bytes:
        raise AssertionError(result)
    if ratio > 1.5:
        raise AssertionError(result)
    dist.barrier()
    if rank == 0:
        print(json.dumps(result, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
