"""BeamGroup correctness: a fast, CPU-only, fake-model unit test verifying
the exact candidate-scoring/selection/branching math (the tricky new part),
plus real-model tests exercising the actual mechanism against an
independently-computed reference and proving isolation from concurrent
regular/LoRA decoding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.beam_search import BeamGroup, BeamSearchParams
from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.lora import LoRARegistry
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import Sequence
from wllm.models.qwen2 import load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class _FakeCache:
    def free_sequence(self, seq_id):
        pass


class _FakeModel:
    """Returns queued logits tensors in call order -- BeamGroup.step()
    iterates its beams in a fixed, deterministic order (enumerate(self.beams)),
    so the i-th call corresponds to the i-th currently-active beam.
    """

    def __init__(self, logits_queue: list[torch.Tensor]):
        self._queue = list(logits_queue)
        self.calls = []

    def prefill_with_cache_and_prefix_reuse(self, cache, seq_id, token_ids, lora_id=None, lora_registry=None):
        self.calls.append((seq_id, list(token_ids)))
        logits = self._queue.pop(0)
        return logits.view(1, 1, -1)  # (batch=1, seq=1, vocab) -- step() reads [0, -1, :]


def test_first_step_expands_one_beam_into_top_k_by_manual_computation():
    """Single active (root) beam, vocab size 4, num_beams=2 -- the new beams
    must be exactly the top-2 tokens by log_softmax(logits), with cum_logprob
    exactly equal to that token's logprob (root's cum_logprob is 0).
    """
    logits = torch.tensor([1.0, 3.0, 0.5, 2.0])
    expected_log_probs = logits.log_softmax(dim=-1)

    model = _FakeModel([logits])
    group = BeamGroup(
        group_id=1,
        prompt_token_ids=[10, 11],
        max_new_tokens=5,
        params=BeamSearchParams(num_beams=2, length_penalty=1.0),
    )
    group.step(model, _FakeCache())

    assert len(group.beams) == 2
    got = {(tuple(b.token_ids), round(b.cum_logprob, 6)) for b in group.beams}
    top2 = expected_log_probs.topk(2)
    expected = {
        (( 10, 11, tok), round(lp, 6))
        for lp, tok in zip(top2.values.tolist(), top2.indices.tolist())
    }
    assert got == expected


def test_second_step_global_top_k_can_come_from_one_parent():
    """Two active beams; the fake logits are rigged so BOTH global top-2
    candidates come from beam 0 alone (beam 1's best candidate scores lower
    than either of beam 0's top-2) -- proving a single parent can spawn every
    surviving child, not just at most one each.
    """
    # beam 0: cum_logprob=-1.0, distribution peaked enough that its top-2
    # log_probs (~-0.60, ~-0.80) both still beat beam 1's best possible
    # candidate once each is added to its own cum_logprob below.
    logits0 = torch.tensor([5.0, 4.8, -5.0, -5.0])
    # beam 1: cum_logprob=-0.5, near-uniform distribution -> log_probs close
    # to -log(4) = -1.386 each, so its best candidate scores -0.5-1.386 =
    # -1.886 -- worse than either of beam 0's top-2 (-1.60, -1.80 below).
    logits1 = torch.tensor([0.0, 0.0, 0.0, 0.0])

    model = _FakeModel([logits0, logits1])
    group = BeamGroup(
        group_id=2,
        prompt_token_ids=[5],
        max_new_tokens=5,
        params=BeamSearchParams(num_beams=2, length_penalty=1.0),
    )
    # Seed two active beams directly (bypassing a real first step) to test
    # the multi-beam expansion in isolation.
    from wllm.engine.beam_search import _Beam

    group.beams = [
        _Beam(token_ids=[5, 20], cum_logprob=-1.0),
        _Beam(token_ids=[5, 21], cum_logprob=-0.5),
    ]

    group.step(model, _FakeCache())

    log_probs0 = logits0.log_softmax(dim=-1)
    top2_0 = log_probs0.topk(2)
    expected_scores = sorted((-1.0 + lp for lp in top2_0.values.tolist()), reverse=True)

    assert len(group.beams) == 2
    got_scores = sorted(b.cum_logprob for b in group.beams)
    for got, exp in zip(got_scores, sorted(expected_scores)):
        assert got == pytest.approx(exp, abs=1e-5)
    # Both survivors must be children of beam 0 (token_ids[1] == 20), since
    # beam 1's best candidate is far worse than either of beam 0's top-2.
    assert all(b.token_ids[1] == 20 for b in group.beams)


def test_stop_token_moves_candidate_to_finished_and_finalizes():
    eos_id = 2
    logits = torch.tensor([0.0, 0.0, 10.0, 0.0])  # token 2 (eos) is overwhelmingly likely
    model = _FakeModel([logits])
    group = BeamGroup(
        group_id=3,
        prompt_token_ids=[1],
        max_new_tokens=5,
        params=BeamSearchParams(num_beams=1, length_penalty=1.0),
        eos_token_id=eos_id,
    )
    group.step(model, _FakeCache())
    assert group.is_done()
    assert group.result == [eos_id]


def make_cache(model, num_blocks, block_size=16, enable_prefix_caching=False):
    cfg = model.cfg
    return KVCacheManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        device="cuda",
        dtype=torch.float32,
        enable_prefix_caching=enable_prefix_caching,
    )


def tokenize(tokenizer, question: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0].tolist()


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_native(MODEL_ID, dtype=torch.float32)
    return model, tokenizer


def test_real_model_first_step_matches_independent_topk_reference(model_and_tokenizer):
    """Cross-checks BeamGroup's own log_softmax+topk against a logits tensor
    computed via a completely separate prefill call on the same model --
    exact (fp32, no sampling randomness), and specific to the first step
    where there's no earlier-pruning history to complicate the comparison.
    """
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")
    num_beams = 4

    ref_cache = make_cache(model, num_blocks=16)
    ref_logits = model.prefill_with_cache_and_prefix_reuse(ref_cache, 0, prompt_ids)
    ref_log_probs = ref_logits[0, -1, :].float().log_softmax(dim=-1)
    ref_top = ref_log_probs.topk(num_beams)

    cache = make_cache(model, num_blocks=16)
    group = BeamGroup(
        group_id=42,
        prompt_token_ids=prompt_ids,
        max_new_tokens=5,
        params=BeamSearchParams(num_beams=num_beams),
    )
    group.step(model, cache)

    assert len(group.beams) == num_beams
    got = {(b.token_ids[-1], round(b.cum_logprob, 4)) for b in group.beams}
    expected = {(tok, round(lp, 4)) for tok, lp in zip(ref_top.indices.tolist(), ref_top.values.tolist())}
    assert got == expected


def test_num_beams_one_matches_ordinary_greedy_decoding_exactly(model_and_tokenizer):
    """Beam search with num_beams=1 is mathematically identical to greedy
    decoding (the single top-1 candidate at every step IS the argmax) --
    an exact-match check against this codebase's existing, separately-tested
    greedy Sequence/Scheduler path.
    """
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")
    max_new_tokens = 12

    greedy_cache = make_cache(model, num_blocks=16)
    scheduler = Scheduler(model, greedy_cache)
    scheduler.add_request(Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id))
    finished = scheduler.run_to_completion()
    greedy_tokens = finished[0].output_token_ids

    beam_cache = make_cache(model, num_blocks=16)
    beam_scheduler = Scheduler(model, beam_cache)
    group = BeamGroup(
        group_id=7,
        prompt_token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        params=BeamSearchParams(num_beams=1),
        eos_token_id=tokenizer.eos_token_id,
    )
    beam_scheduler.add_beam_request(group)
    while not group.is_done():
        beam_scheduler.step()

    assert group.result == greedy_tokens


def test_beam_group_isolated_from_concurrent_regular_sequence(model_and_tokenizer):
    """A beam-search request and a regular greedy request run concurrently
    through the same Scheduler/cache -- the regular sequence's output must
    exactly match its own standalone baseline, proving the beam lane's
    scratch-sequence create/free cycle doesn't corrupt or steal blocks from
    unrelated sequences sharing the same step.
    """
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenize(tokenizer, "What is the capital of Germany? Answer in one sentence.")
    max_new_tokens = 12

    standalone_cache = make_cache(model, num_blocks=16)
    standalone_scheduler = Scheduler(model, standalone_cache)
    standalone_scheduler.add_request(
        Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    )
    standalone_text = standalone_scheduler.run_to_completion()[0].output_token_ids

    shared_cache = make_cache(model, num_blocks=32)
    scheduler = Scheduler(model, shared_cache)
    scheduler.add_request(
        Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
    )
    beam_prompt = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")
    group = BeamGroup(
        group_id=1,
        prompt_token_ids=beam_prompt,
        max_new_tokens=max_new_tokens,
        params=BeamSearchParams(num_beams=3),
        eos_token_id=tokenizer.eos_token_id,
    )
    scheduler.add_beam_request(group)

    finished_regular = []
    while scheduler.has_unfinished_requests():
        finished_regular.extend(s for s in scheduler.step() if s.is_finished())

    assert finished_regular[0].output_token_ids == standalone_text
    assert group.is_done()
    assert "Paris" in tokenizer.decode(group.result, skip_special_tokens=True)


def test_beam_search_with_all_zero_lora_adapter_is_a_noop(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenize(tokenizer, "What is the capital of Japan? Answer in one sentence.")
    max_new_tokens = 10
    cfg = model.cfg

    registry = LoRARegistry()
    rank = 4
    A = torch.randn(rank, cfg.hidden_size, device="cuda")
    B = torch.zeros(cfg.num_attention_heads * cfg.head_dim, rank, device="cuda")
    registry._adapters["zero_adapter"] = {(0, "q_proj"): (A, B, 2.0)}

    baseline_cache = make_cache(model, num_blocks=16)
    baseline_group = BeamGroup(
        group_id=1, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens,
        params=BeamSearchParams(num_beams=2), eos_token_id=tokenizer.eos_token_id,
    )
    while not baseline_group.is_done():
        baseline_group.step(model, baseline_cache)

    lora_cache = make_cache(model, num_blocks=16)
    lora_group = BeamGroup(
        group_id=2, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens,
        params=BeamSearchParams(num_beams=2), eos_token_id=tokenizer.eos_token_id,
        lora_id="zero_adapter",
    )
    while not lora_group.is_done():
        lora_group.step(model, lora_cache, lora_registry=registry)

    assert lora_group.result == baseline_group.result
