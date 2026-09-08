import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from wllm.engine.sampling import SamplingParams, apply_repetition_penalty, sample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_greedy_matches_argmax_regardless_of_randomness():
    torch.manual_seed(0)
    logits = torch.randn(4, 100, device=DEVICE)
    params = [SamplingParams(temperature=0.0) for _ in range(4)]

    tokens_a, _ = sample(logits, params, generated_ids=[[] for _ in range(4)])
    torch.manual_seed(999)  # different RNG state must not change greedy output
    tokens_b, _ = sample(logits, params, generated_ids=[[] for _ in range(4)])

    assert torch.equal(tokens_a, logits.argmax(dim=-1))
    assert torch.equal(tokens_a, tokens_b)


def test_top_k_1_always_matches_argmax_even_when_sampling():
    torch.manual_seed(0)
    logits = torch.randn(8, 50, device=DEVICE)
    params = [SamplingParams(temperature=1.0, top_k=1) for _ in range(8)]

    tokens, _ = sample(logits, params, generated_ids=[[] for _ in range(8)])
    assert torch.equal(tokens, logits.argmax(dim=-1))


def test_mixed_greedy_and_sampling_rows_are_independent():
    """The core per-row property: one sequence being greedy must not affect
    another sequence's sampling in the same batched call.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, 50, device=DEVICE)
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=1.0, top_k=1)]

    tokens, _ = sample(logits, params, generated_ids=[[], []])
    assert tokens[0].item() == logits[0].argmax().item()
    assert tokens[1].item() == logits[1].argmax().item()  # top_k=1 also forces argmax


def test_repetition_penalty_suppresses_positive_and_negative_logits_correctly():
    logits = torch.tensor([[2.0, -2.0, 0.5]], device=DEVICE)
    penalized = apply_repetition_penalty(logits, generated_ids=[[0, 1]], penalties=[2.0])

    assert penalized[0, 0].item() == pytest.approx(1.0)  # positive: divided by penalty
    assert penalized[0, 1].item() == pytest.approx(-4.0)  # negative: multiplied by penalty (more negative)
    assert penalized[0, 2].item() == pytest.approx(0.5)  # untouched (never generated)


def test_repetition_penalty_noop_at_1_0():
    logits = torch.randn(3, 20, device=DEVICE)
    penalized = apply_repetition_penalty(logits, generated_ids=[[1, 2], [], [5]], penalties=[1.0, 1.0, 1.0])
    assert torch.equal(logits, penalized)


def test_sample_reproducible_with_seeded_generator():
    logits = torch.randn(4, 100, device=DEVICE)
    params = [SamplingParams(temperature=1.0) for _ in range(4)]

    gen_a = torch.Generator(device=DEVICE).manual_seed(42)
    tokens_a, _ = sample(logits, params, generated_ids=[[] for _ in range(4)], generator=gen_a)

    gen_b = torch.Generator(device=DEVICE).manual_seed(42)
    tokens_b, _ = sample(logits, params, generated_ids=[[] for _ in range(4)], generator=gen_b)

    assert torch.equal(tokens_a, tokens_b)


def test_logprobs_are_valid_and_match_manual_computation_for_greedy():
    logits = torch.randn(3, 30, device=DEVICE)
    params = [SamplingParams(temperature=0.0) for _ in range(3)]

    tokens, logprobs = sample(logits, params, generated_ids=[[] for _ in range(3)])

    assert (logprobs <= 0).all()
    assert torch.isfinite(logprobs).all()

    expected = torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(logprobs, expected, atol=1e-4)


def test_top_p_never_drops_the_top_token_even_if_it_alone_exceeds_p():
    # one token completely dominates the distribution
    logits = torch.full((1, 10), -100.0, device=DEVICE)
    logits[0, 3] = 100.0
    params = [SamplingParams(temperature=1.0, top_p=0.1)]

    tokens, _ = sample(logits, params, generated_ids=[[]])
    assert tokens.item() == 3
