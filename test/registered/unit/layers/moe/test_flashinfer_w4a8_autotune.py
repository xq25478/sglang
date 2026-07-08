from pathlib import Path

import torch

from sglang.srt.layers.moe.moe_runner.flashinfer_w4a8_autotune import (
    RouteAwareProfile,
    RouteRecorder,
    aggregate_route_probabilities,
    build_topk_ids,
    hybrid_bucket,
    load_profile,
    get_runtime_profile_ids,
    route_recording,
    save_profile,
    select_best_tactic,
    set_active_profile,
)


def test_hybrid_bucket_maps_decode_and_prefill_shapes():
    assert hybrid_bucket(1) == 1
    assert hybrid_bucket(12) == 16
    assert hybrid_bucket(64) == 64
    assert hybrid_bucket(4097) == 8192
    assert hybrid_bucket(8192) == 8192
    assert hybrid_bucket(8193) is None


def test_route_profile_lookup_uses_hybrid_bucket():
    profile = RouteAwareProfile(
        metadata={"schema": 1, "model": "test"},
        tactics={16: (3, 101), 8192: (5, 103)},
    )

    assert profile.lookup(12) == (3, 101)
    assert profile.lookup(4097) == (5, 103)
    assert profile.lookup(8193) is None


def test_route_recorder_keeps_decode_and_prefill_separate():
    recorder = RouteRecorder(num_experts=4)
    recorder.observe("decode", torch.tensor([[0, 1], [0, 2]], dtype=torch.int32))
    recorder.observe("prefill", torch.tensor([[3, 3]], dtype=torch.int32))

    assert recorder.histograms["decode"][0].tolist() == [2, 1, 1, 0]
    assert recorder.histograms["prefill"][0].tolist() == [0, 0, 0, 2]


def test_aggregate_route_probabilities_normalizes_each_layer():
    probs = aggregate_route_probabilities(
        [
            torch.tensor([8, 2, 0], dtype=torch.int64),
            torch.tensor([1, 1, 8], dtype=torch.int64),
        ]
    )

    torch.testing.assert_close(
        probs,
        torch.tensor([0.45, 0.15, 0.40], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    assert probs.sum().item() == 1.0


def test_build_topk_ids_preserves_shape_range_and_counts():
    probs = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float64)
    ids = build_topk_ids(probs, num_tokens=10, top_k=2)

    assert ids.shape == (10, 2)
    assert ids.dtype == torch.int32
    assert int(ids.min()) == 0
    assert int(ids.max()) == 2
    assert torch.bincount(ids.flatten().to(torch.int64), minlength=3).tolist() == [
        10,
        6,
        4,
    ]


def test_profile_cache_round_trip_and_metadata_validation(tmp_path: Path):
    path = tmp_path / "route-aware.json"
    profile = RouteAwareProfile(
        metadata={"schema": 1, "model": "m", "flashinfer": "v"},
        tactics={1: (2, 100), 8192: (85, 200)},
    )

    save_profile(path, profile)

    assert load_profile(path, expected_metadata=profile.metadata) == profile
    assert (
        load_profile(
            path,
            expected_metadata={"schema": 1, "model": "different", "flashinfer": "v"},
        )
        is None
    )


def test_runtime_profile_ids_and_recording_override():
    profile = RouteAwareProfile(
        metadata={"schema": 1},
        tactics={16: (3, 101)},
    )
    recorder = RouteRecorder(num_experts=4)
    set_active_profile(profile)
    try:
        assert get_runtime_profile_ids(12) == [3, 101]
        with route_recording(recorder, "decode"):
            assert get_runtime_profile_ids(12) == [-1, -1]
    finally:
        set_active_profile(None)

    assert get_runtime_profile_ids(12) is None


def test_select_best_tactic_uses_measured_pair_and_skips_failures():
    timings = {
        (3, -1): 2.0,
        (5, -1): None,
        (7, -1): 1.0,
        (7, 101): 4.0,
        (7, 103): 3.0,
    }

    gemm1 = select_best_tactic(
        [3, 5, 7],
        stage=1,
        fixed_tactic=-1,
        measure_pair=lambda pair: timings[pair],
    )
    gemm2 = select_best_tactic(
        [101, 103],
        stage=2,
        fixed_tactic=gemm1,
        measure_pair=lambda pair: timings[pair],
    )

    assert (gemm1, gemm2) == (7, 103)
