from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components.dspark_draft import sample_draft_block
from sglang.srt.speculative.dspark_components.kernels.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.srt.speculative.dspark_components.kernels.dspark_verify_window import (
    compact_row_index_triton,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fast_sampling_clamps_all_non_finite_row_to_token_zero():
    logits = torch.tensor(
        [
            [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
            [0.0, 1.0, 3.0, 2.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    temperatures = torch.ones(2, dtype=torch.float32, device="cuda")
    greedy_mask = torch.ones(2, dtype=torch.bool, device="cuda")
    exp_noise = torch.ones_like(logits)

    output = SampleStepTokens.triton(
        step_logits=logits,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
        exp_noise=exp_noise,
    )

    assert output.tolist() == [0, 2]


def test_slow_sampling_clamps_all_non_finite_row_to_token_zero():
    class _MarkovHead:
        @staticmethod
        def sample_block(
            base_logits,
            *,
            first_prev_tokens,
            hidden_states,
            sampler,
        ):
            del first_prev_tokens, hidden_states
            return sampler(base_logits, 0), None

    sampling_info = SimpleNamespace(
        top_ks=torch.tensor([2]),
        temperatures=torch.ones((1, 1), dtype=torch.float32),
        is_all_greedy=False,
    )
    logits = torch.full((1, 4), float("-inf"), dtype=torch.float32)

    with envs.SGLANG_DSPARK_FAST_SAMPLING.override(False):
        result = sample_draft_block(
            base_logits=logits,
            anchor_tokens=torch.zeros(1, dtype=torch.int64),
            draft_hidden=torch.zeros((1, 1), dtype=torch.float32),
            sampling_info=sampling_info,
            markov_head=_MarkovHead(),
            device=torch.device("cpu"),
        )

    assert result.draft_tokens.tolist() == [0]


def test_compact_row_index_rejects_batch_beyond_search_capacity():
    with pytest.raises(AssertionError, match="exceeds row-index search capacity"):
        compact_row_index_triton(
            verify_lens=torch.ones(1025, dtype=torch.int64),
            padded_total=1,
            device=torch.device("cpu"),
        )
