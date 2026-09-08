import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import SeqStatus, Sequence
from wllm.models.qwen2 import generate_with_kv_cache, load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def make_cache(model, num_blocks, block_size=16, dtype=torch.float32):
    cfg = model.cfg
    return KVCacheManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        device="cuda",
        dtype=dtype,
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


QUESTIONS = [
    "What is the capital of France? Answer in one sentence.",
    "What is the capital of Germany? Answer in one sentence.",
    "What is the capital of Japan? Answer in one sentence.",
]


def test_batched_scheduling_matches_standalone_single_sequence_generation(model_and_tokenizer):
    """The whole point of continuous batching: running N sequences together
    must produce exactly the same result each would produce alone. fp32
    removes precision drift so this is an exact-match check, not a fuzzy one.
    """
    model, tokenizer = model_and_tokenizer
    max_new_tokens = 12

    # Ground truth: each prompt generated alone, one at a time, own fresh cache.
    standalone_texts = []
    for q in QUESTIONS:
        prompt_ids = tokenize(tokenizer, q)
        input_ids = torch.tensor([prompt_ids], device="cuda")
        cache = make_cache(model, num_blocks=16)
        tokens = generate_with_kv_cache(model, cache, seq_id=0, input_ids=input_ids, max_new_tokens=max_new_tokens)
        standalone_texts.append(tokenizer.decode(tokens[0, len(prompt_ids) :], skip_special_tokens=True))

    # Now run all three together through the scheduler.
    shared_cache = make_cache(model, num_blocks=64)
    scheduler = Scheduler(model, shared_cache)
    for i, q in enumerate(QUESTIONS):
        prompt_ids = tokenize(tokenizer, q)
        # eos_token_id=None here to match the standalone path above, which
        # has no EOS-stopping logic and always runs the full max_new_tokens --
        # otherwise an early real EOS would make the two legitimately differ
        # in length without that being a batching bug.
        scheduler.add_request(
            Sequence(seq_id=i, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=None)
        )

    finished = scheduler.run_to_completion()
    assert len(finished) == 3
    finished_by_id = {s.seq_id: s for s in finished}

    for i, q in enumerate(QUESTIONS):
        seq = finished_by_id[i]
        batched_text = tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
        print(f"\n[{q}]\n  standalone: {standalone_texts[i]!r}\n  batched:    {batched_text!r}")
        assert batched_text == standalone_texts[i], f"batched generation diverged for: {q!r}"


def test_admission_control_defers_requests_that_dont_fit(model_and_tokenizer):
    model, _ = model_and_tokenizer
    # 2 blocks * 16 tokens/block = 32 total token slots.
    cache = make_cache(model, num_blocks=2, block_size=16)
    scheduler = Scheduler(model, cache)

    # Each sequence's estimate (prompt + max_new_tokens) needs 2 blocks alone,
    # so both together can't fit -- the second must be deferred.
    long_prompt = list(range(10))  # dummy token ids, never actually generated from
    seq_a = Sequence(seq_id=0, prompt_token_ids=long_prompt, max_new_tokens=20, eos_token_id=None)
    seq_b = Sequence(seq_id=1, prompt_token_ids=long_prompt, max_new_tokens=20, eos_token_id=None)

    scheduler.add_request(seq_a)
    scheduler.add_request(seq_b)
    scheduler._admit()

    assert seq_a in scheduler.running
    assert seq_b in scheduler.waiting
    assert seq_b not in scheduler.running


def test_preemption_frees_victim_and_requeues_it(model_and_tokenizer):
    model, _ = model_and_tokenizer
    cache = make_cache(model, num_blocks=8, block_size=16)
    scheduler = Scheduler(model, cache)

    # Exactly one block's worth of prompt tokens, so after prefill the cache
    # is at a block boundary -- the *next* decode step's reserve() must
    # allocate a fresh block, which is the scenario we want to force OOM on.
    prompt_ids = [100] * 16
    seq = Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_new_tokens=5, eos_token_id=None)
    scheduler.add_request(seq)
    scheduler._admit()
    assert seq in scheduler.running
    assert cache.context_len(seq.seq_id) == 16

    # Artificially drain the allocator to force the next decode step's
    # cache.reserve() to hit "out of free blocks" -- this is the same
    # RuntimeError a real multi-sequence cache exhaustion would raise.
    drained = []
    while cache.allocator.num_free:
        drained.append(cache.allocator.allocate())

    scheduler._decode_active([seq])

    assert seq.status == SeqStatus.WAITING
    assert seq in scheduler.waiting
    assert seq.output_token_ids == []
    # The victim held exactly 1 block (16-token prompt == 1 block); freeing it
    # returns just that 1 -- the rest are still deliberately drained above.
    assert cache.allocator.num_free == 1, "victim's block was not returned to the allocator"

    for b in drained:
        cache.allocator.free(b)


def test_cancel_running_sequence_frees_its_cache(model_and_tokenizer):
    model, _ = model_and_tokenizer
    cache = make_cache(model, num_blocks=8, block_size=16)
    scheduler = Scheduler(model, cache)

    seq = Sequence(seq_id=0, prompt_token_ids=[100] * 16, max_new_tokens=5, eos_token_id=None)
    scheduler.add_request(seq)
    scheduler._admit()
    assert seq in scheduler.running
    assert cache.allocator.num_free == 7  # 1 block used by the 16-token prompt

    scheduler.cancel(0)
    assert seq not in scheduler.running
    assert cache.allocator.num_free == 8


def test_cancel_waiting_sequence_just_removes_it(model_and_tokenizer):
    model, _ = model_and_tokenizer
    cache = make_cache(model, num_blocks=1, block_size=16)  # too small to admit
    scheduler = Scheduler(model, cache)

    seq = Sequence(seq_id=0, prompt_token_ids=[100] * 16, max_new_tokens=50, eos_token_id=None)
    scheduler.add_request(seq)
    scheduler._admit()
    assert seq in scheduler.waiting  # doesn't fit, stays queued

    scheduler.cancel(0)
    assert seq not in scheduler.waiting


def test_cancel_unknown_seq_id_is_a_noop(model_and_tokenizer):
    model, _ = model_and_tokenizer
    cache = make_cache(model, num_blocks=4, block_size=16)
    scheduler = Scheduler(model, cache)
    scheduler.cancel(999)  # must not raise
