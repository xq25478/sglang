import os
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist


os.environ["SGLANG_DP_ATTN_COMPRESSED_ALLGATHER"] = "true"
os.environ["SGLANG_ENABLE_METRICS_DP_ATTENTION"] = "false"

from sglang.srt.managers.scheduler_components.dp_attn import MLPSyncBatchInfo


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise AssertionError(f"JD compressed all-gather test requires 2 ranks, got {world}")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    info = MLPSyncBatchInfo(
        dp_size=world,
        tp_size=1,
        cp_size=1,
        num_tokens=10 + rank,
        num_tokens_for_logprob=2 + rank,
        can_cuda_graph=rank == 0,
        is_extend_in_batch=rank == 1,
        local_can_run_tbo=True,
        local_forward_mode=3 + rank,
        can_run_breakable_cuda_graph=rank == 0,
    )
    active = torch.ones(world, dtype=torch.int32, device=device)
    tp_group = SimpleNamespace(active_ranks=active, active_ranks_cpu=active.cpu())
    with patch(
        "sglang.srt.managers.scheduler_components.dp_attn.get_tp_group",
        return_value=tp_group,
    ):
        info.all_gather(device=device, group=dist.group.WORLD)

    expected = torch.tensor(
        [
            [10, 2, 0b101, 3, 1],
            [11, 3, 0b110, 4, 0],
        ],
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(info.tp0_info, expected)
    expected_tbo = torch.tensor(
        [[1, 3], [1, 4]], dtype=torch.int32, device=device
    )
    torch.testing.assert_close(info.tbo_info, expected_tbo)
    if info.global_num_tokens != [10, 11]:
        raise AssertionError(info.global_num_tokens)
    if not info.is_extend_in_batch or info.can_cuda_graph:
        raise AssertionError("compressed flags were not reconstructed correctly")
    dist.barrier()
    if rank == 0:
        print("JD compressed DP-attention all-gather correctness passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
