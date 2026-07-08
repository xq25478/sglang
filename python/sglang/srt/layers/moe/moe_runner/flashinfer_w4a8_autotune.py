"""Route-aware tactic profiles for FlashInfer-CUTLASS W4A8 MoE."""

from __future__ import annotations

import json
import math
import os
import tempfile
from bisect import bisect_left
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import torch


FLASHINFER_W4A8_TUNING_BUCKETS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    3584,
    4096,
    8192,
)


def hybrid_bucket(num_tokens: int) -> Optional[int]:
    """Map a runtime token count to FlashInfer's round-up tuning bucket."""
    if num_tokens <= 0:
        return None
    index = bisect_left(FLASHINFER_W4A8_TUNING_BUCKETS, int(num_tokens))
    if index == len(FLASHINFER_W4A8_TUNING_BUCKETS):
        return None
    return FLASHINFER_W4A8_TUNING_BUCKETS[index]


@dataclass(eq=True)
class RouteAwareProfile:
    metadata: dict
    tactics: dict[int, tuple[int, int]]

    def lookup(self, num_tokens: int) -> Optional[tuple[int, int]]:
        bucket = hybrid_bucket(num_tokens)
        return None if bucket is None else self.tactics.get(bucket)


class RouteRecorder:
    """Collect per-layer routed-expert histograms during calibration forwards."""

    def __init__(self, num_experts: int):
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        self.num_experts = int(num_experts)
        self.histograms: dict[str, list[torch.Tensor]] = {
            "decode": [],
            "prefill": [],
        }
        self.samples: dict[str, Any] = {}

    def observe(
        self, phase: str, topk_ids: torch.Tensor, *, sample: Any = None
    ) -> None:
        if phase not in self.histograms:
            raise ValueError(f"Unknown route calibration phase: {phase!r}")
        ids = topk_ids.detach().to(device="cpu", dtype=torch.int64).flatten()
        ids = ids[(ids >= 0) & (ids < self.num_experts)]
        histogram = torch.bincount(ids, minlength=self.num_experts)
        self.histograms[phase].append(histogram)
        if sample is not None and phase not in self.samples:
            self.samples[phase] = sample


_active_profile: Optional[RouteAwareProfile] = None
_recording_state: ContextVar[Optional[tuple[RouteRecorder, str]]] = ContextVar(
    "flashinfer_w4a8_recording_state", default=None
)


def set_active_profile(profile: Optional[RouteAwareProfile]) -> None:
    global _active_profile
    _active_profile = profile


def get_runtime_profile_ids(num_tokens: int) -> Optional[list[int]]:
    if _recording_state.get() is not None:
        return [-1, -1]
    if _active_profile is None:
        return None
    pair = _active_profile.lookup(num_tokens)
    return None if pair is None else [int(pair[0]), int(pair[1])]


@contextmanager
def route_recording(
    recorder: RouteRecorder, phase: str
) -> Iterator[RouteRecorder]:
    if phase not in recorder.histograms:
        raise ValueError(f"Unknown route calibration phase: {phase!r}")
    token = _recording_state.set((recorder, phase))
    try:
        yield recorder
    finally:
        _recording_state.reset(token)


def observe_route(topk_ids: torch.Tensor, *, sample: Any = None) -> bool:
    state = _recording_state.get()
    if state is None:
        return False
    recorder, phase = state
    recorder.observe(phase, topk_ids, sample=sample)
    return True


def select_best_tactic(
    candidate_ids: Sequence[int],
    *,
    stage: int,
    fixed_tactic: int,
    measure_pair: Callable[[tuple[int, int]], Optional[float]],
    should_stop: Optional[Callable[[], bool]] = None,
) -> int:
    """Select the fastest valid tactic while keeping the other GEMM fixed."""
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")
    best_tactic = -1
    best_time = math.inf
    for candidate in sorted({int(candidate) for candidate in candidate_ids}):
        if should_stop is not None and should_stop():
            break
        pair = (
            (candidate, int(fixed_tactic))
            if stage == 1
            else (int(fixed_tactic), candidate)
        )
        elapsed_ms = measure_pair(pair)
        if elapsed_ms is not None and math.isfinite(elapsed_ms):
            if elapsed_ms < best_time:
                best_tactic = candidate
                best_time = elapsed_ms
    return best_tactic


def aggregate_route_probabilities(
    histograms: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Average normalized per-layer expert histograms."""
    if not histograms:
        raise ValueError("At least one route histogram is required")
    normalized = []
    expected_size = int(histograms[0].numel())
    for histogram in histograms:
        values = histogram.detach().to(device="cpu", dtype=torch.float64).flatten()
        if values.numel() != expected_size:
            raise ValueError("All route histograms must have the same expert count")
        total = values.sum()
        if total <= 0:
            raise ValueError("Route histograms must contain at least one assignment")
        normalized.append(values / total)
    probabilities = torch.stack(normalized).mean(dim=0)
    return probabilities / probabilities.sum()


def _largest_remainder_counts(
    probabilities: torch.Tensor, total_assignments: int
) -> torch.Tensor:
    raw = probabilities * total_assignments
    counts = torch.floor(raw).to(torch.int64)
    remainder = total_assignments - int(counts.sum())
    if remainder:
        order = torch.argsort(raw - counts, descending=True, stable=True)
        counts[order[:remainder]] += 1
    return counts


def build_topk_ids(
    probabilities: torch.Tensor,
    num_tokens: int,
    top_k: int,
    *,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """Synthesize deterministic expert IDs with the requested load histogram."""
    if num_tokens <= 0 or top_k <= 0:
        raise ValueError(
            f"num_tokens and top_k must be positive, got {num_tokens}, {top_k}"
        )
    probs = probabilities.detach().to(device="cpu", dtype=torch.float64).flatten()
    if probs.numel() == 0 or torch.any(probs < 0) or probs.sum() <= 0:
        raise ValueError("probabilities must be a non-empty non-negative vector")
    probs = probs / probs.sum()
    total = int(num_tokens) * int(top_k)
    counts = _largest_remainder_counts(probs, total)
    expert_ids = torch.repeat_interleave(
        torch.arange(probs.numel(), dtype=torch.int64), counts
    )

    stride = total // 2 + 1
    while math.gcd(stride, total) != 1:
        stride += 1
    permutation = (torch.arange(total, dtype=torch.int64) * stride) % total
    result = expert_ids[permutation].reshape(num_tokens, top_k).to(torch.int32)
    return result.to(device=device) if device is not None else result


def save_profile(path: Path, profile: RouteAwareProfile) -> None:
    """Atomically persist a route-aware tactic profile."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": profile.metadata,
        "tactics": {
            str(bucket): [int(ids[0]), int(ids[1])]
            for bucket, ids in sorted(profile.tactics.items())
        },
    }
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_profile(
    path: Path, *, expected_metadata: Mapping
) -> Optional[RouteAwareProfile]:
    """Load a profile only when all versioned metadata matches exactly."""
    try:
        with Path(path).open(encoding="utf-8") as source:
            payload = json.load(source)
        metadata = payload["metadata"]
        if metadata != dict(expected_metadata):
            return None
        tactics = {
            int(bucket): (int(ids[0]), int(ids[1]))
            for bucket, ids in payload["tactics"].items()
        }
        if any(bucket not in FLASHINFER_W4A8_TUNING_BUCKETS for bucket in tactics):
            return None
        return RouteAwareProfile(metadata=metadata, tactics=tactics)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None
