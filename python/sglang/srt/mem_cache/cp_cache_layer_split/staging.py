"""Staging buffers and index remap kernels for CP Cache LayerSplit."""

from __future__ import annotations

from typing import Callable, Optional

import torch

# The remap kernels receive index tensors of varying shapes (1D token-level
# and 2D page-table). Even with flatten() inside the functions, dynamo's
# guards fire on the input rank before the function body runs, exhausting
# the default recompile_limit of 8. Raise the limit so the compiled kernels
# stay active instead of falling back to slow eager mode.
try:
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, 32
    )
    torch._dynamo.config.accumulated_cache_size_limit = max(
        torch._dynamo.config.accumulated_cache_size_limit, 32
    )
except Exception:
    pass


@torch.compile(dynamic=True)
def build_active_pages_mask(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
) -> torch.Tensor:
    local_mask = torch.zeros(max_pages, dtype=torch.int32, device=indices.device)
    indices_flat = indices.flatten()
    valid = indices_flat >= 0
    safe_indices = torch.clamp(indices_flat, min=0)
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")
    # Clamp page_ids to the owner pool range. Under CP layer-split the page
    # table may carry global logical page ids that exceed the per-rank owner
    # pool size; those entries are not owned by this rank and must not set
    # the mask. Mapping them to page 0 (dummy) keeps index_put_ in bounds.
    page_ids = torch.clamp(page_ids, min=0, max=max_pages - 1)
    local_mask.index_put_(
        (page_ids.to(torch.long),),
        valid.to(torch.int32),
        accumulate=True,
    )
    # Compact staging slot 0 is the fallback for invalid/unselected mappings.
    local_mask[0] = 1
    return local_mask


def all_reduce_active_pages_mask(local_mask: torch.Tensor, pynccl_comm) -> torch.Tensor:
    """Sum the per-rank active-page mask across the attention-CP group."""
    with pynccl_comm.change_state(enable=True):
        pynccl_comm.all_reduce(local_mask)
    return local_mask


@torch.compile(dynamic=True)
def remap_indices_to_staging(
    indices: torch.Tensor,
    selected_pages: torch.Tensor,
    page_size: int,
    max_pages: int,
) -> torch.Tensor:
    indices_flat = indices.flatten()
    page_map = torch.full((max_pages,), -1, dtype=torch.int32, device=indices_flat.device)
    page_map[selected_pages.to(torch.long)] = torch.arange(
        selected_pages.numel(), dtype=torch.int32, device=indices_flat.device
    )
    # Fixed-size selection pads with duplicate zeros; canonical page 0 must
    # always resolve to staging slot 0, not the final padding slot.
    page_map[0] = 0

    valid = indices_flat >= 0
    safe_indices = torch.clamp(indices_flat, min=0)
    page_ids = torch.div(safe_indices, page_size, rounding_mode="floor")
    in_range = valid & (page_ids < max_pages)
    safe_page_ids = torch.where(in_range, page_ids, torch.zeros_like(page_ids))
    offsets = safe_indices - page_ids * page_size
    new_pages = page_map[safe_page_ids.to(torch.long)].to(indices_flat.dtype)
    # Pages not in selected_pages map to -1 in page_map; clamp them to page 0
    # (dummy) so dequantize_k_cache_paged reads a valid staging slot instead
    # of a negative index that causes illegal memory access.
    new_pages = torch.clamp(new_pages, min=0)
    remapped = new_pages * page_size + offsets
    result = torch.where(valid, remapped, indices_flat)
    return result.reshape(indices.shape)


@torch.compile(dynamic=True)
def remap_page_table_to_staging(
    page_table: torch.Tensor,
    selected_pages: torch.Tensor,
    max_pages: int,
) -> torch.Tensor:
    page_table_flat = page_table.flatten()
    page_map = torch.full((max_pages,), -1, dtype=torch.int32, device=page_table_flat.device)
    page_map[selected_pages.to(torch.long)] = torch.arange(
        selected_pages.numel(), dtype=torch.int32, device=page_table_flat.device
    )
    page_map[0] = 0

    valid = page_table_flat >= 0
    in_range = valid & (page_table_flat < max_pages)
    safe_pages = torch.where(in_range, page_table_flat, torch.zeros_like(page_table_flat))
    remapped = page_map[safe_pages.to(torch.long)].to(page_table_flat.dtype)
    # Clamp invalid mappings (-1) to page 0 (dummy) to avoid negative indices.
    remapped = torch.clamp(remapped, min=0)
    result = torch.where(valid, remapped, page_table_flat)
    return result.reshape(page_table.shape)


def active_pages_for_indices(
    indices: torch.Tensor,
    page_size: int,
    max_pages: int,
    pynccl_comm,
) -> torch.Tensor:
    """Select pages touched by any CP rank; all ranks must call in the same order."""
    local_mask = build_active_pages_mask(indices, page_size, max_pages)
    local_mask = all_reduce_active_pages_mask(local_mask, pynccl_comm)

    # Async safety guard: if active pages overflow max_pages (should never happen
    # with the page_ids clamp in build_active_pages_mask), fail fast via an async
    # CUDA assertion. This runs in the stream without host sync and does NOT
    # affect the compact nonzero result used for broadcast.
    if hasattr(torch, "nonzero_static") and hasattr(torch, "_assert_async"):
        # Exclude page 0 (canonical/dummy) from the bound check so that the
        # capacity guard covers real data pages 1..max_pages-1.
        probe = torch.nonzero_static(local_mask[1:], size=max_pages, fill_value=-1)
        torch._assert_async(
            probe[max_pages - 1].eq(-1),
            "CP Cache LayerSplit: active-page bound overflow (max_pages={})".format(
                max_pages
            ),
        )

    return torch.nonzero(local_mask, as_tuple=False).flatten()


class StagingBufferManager:
    """Family-keyed staging buffers allocated before serving."""

    def __init__(self) -> None:
        self._buffers: dict[str, Optional[torch.Tensor]] = {}

    def allocate(
        self,
        family: str,
        num_pages: int,
        allocate_fn: Callable[[int], torch.Tensor],
    ) -> torch.Tensor:
        if family in self._buffers:
            raise RuntimeError(f"Staging buffer is already allocated: {family}")
        buffer = allocate_fn(num_pages)
        self._buffers[family] = buffer
        return buffer

    def get_existing(self, family: str) -> Optional[torch.Tensor]:
        return self._buffers.get(family)
