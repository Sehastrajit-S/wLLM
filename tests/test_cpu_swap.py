import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import SeqStatus, Sequence
from wllm.models.qwen2 import load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def make_cache(model, num_blocks, block_size=16):
    cfg = model.cfg
    return KVCacheManager(
        num_layers=cfg.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        device="cuda",
        dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_native(MODEL_ID, dtype=torch.float32)
    return model, tokenizer


def tokenize(tokenizer, question: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0].tolist()


def test_forced_preemption_swaps_out_and_preserves_progress(model_and_tokenizer):
    """Direct analogue of test_preemption_frees_victim_and_requeues_it, but
    with swap enabled: the victim's already-generated tokens must survive
    (recompute-preemption resets them; swap must not).
    """
    model, tokenizer = model_and_tokenizer
    cache = make_cache(model, num_blocks=8, block_size=16)
    scheduler = Scheduler(model, cache, enable_cpu_swap=True)

    # 15 (block_size - 1) prompt tokens: after prefill context_len=15 (still
    # within block 0), one decode step advances it to exactly 16 -- a block
    # boundary, so the *next* decode step's reserve() must allocate a fresh
    # block. That's the scenario we want to force OOM on (same reasoning as
    # P1.4's original preemption test).
    prompt_ids = [100] * 15
    seq = Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_new_tokens=10, eos_token_id=None)
    scheduler.add_request(seq)
    scheduler._admit()
    assert seq in scheduler.running
    assert cache.context_len(seq.seq_id) == 15

    scheduler._decode_active([seq])
    assert cache.context_len(seq.seq_id) == 16
    progress_before = list(seq.output_token_ids)
    assert len(progress_before) == 2  # 1 from prefill + 1 decode step

    drained = []
    while cache.allocator.num_free:
        drained.append(cache.allocator.allocate())

    scheduler._decode_active([seq])

    assert seq.status == SeqStatus.SWAPPED
    assert seq in scheduler.swapped
    assert seq not in scheduler.running
    assert cache.is_swapped(seq.seq_id)
    assert seq.output_token_ids == progress_before, "swap must not reset already-generated tokens"

    for b in drained:
        cache.allocator.free(b)


def test_swap_out_then_resume_produces_identical_generation_to_never_preempted(model_and_tokenizer):
    """The real correctness bar: forcing a swap-out mid-generation, then
    letting it resume, must produce exactly the same final text as a run
    that was never interrupted at all.
    """
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")
    max_new_tokens = 12

    # Baseline: no interruption at all.
    cache_baseline = make_cache(model, num_blocks=32)
    scheduler_baseline = Scheduler(model, cache_baseline)
    seq_baseline = Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens)
    scheduler_baseline.add_request(seq_baseline)
    scheduler_baseline.run_to_completion()

    # Swap path: force a swap-out partway through, then let it resume naturally.
    cache_swap = make_cache(model, num_blocks=32)
    scheduler_swap = Scheduler(model, cache_swap, enable_cpu_swap=True)
    seq_swap = Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens)
    scheduler_swap.add_request(seq_swap)
    scheduler_swap._admit()
    scheduler_swap._decode_active([seq_swap])
    scheduler_swap._decode_active([seq_swap])

    scheduler_swap._preempt(seq_swap)
    assert seq_swap.status == SeqStatus.SWAPPED

    # capacity is free again (nothing else is using it) -- resume and finish.
    while seq_swap.status != SeqStatus.FINISHED:
        scheduler_swap.step()

    text_baseline = tokenizer.decode(seq_baseline.output_token_ids, skip_special_tokens=True)
    text_swap = tokenizer.decode(seq_swap.output_token_ids, skip_special_tokens=True)
    print(f"\nbaseline: {text_baseline!r}\nswap:     {text_swap!r}")
    assert seq_baseline.output_token_ids == seq_swap.output_token_ids


def test_two_sequences_competing_for_capacity_with_swap_enabled(model_and_tokenizer):
    """Natural (not manually forced) preemption pressure: a tight cache with
    two concurrent requests should still produce correct answers for both
    once swap-based preemption kicks in.
    """
    model, tokenizer = model_and_tokenizer
    questions = [
        "What is the capital of France? Answer in one sentence.",
        "What is the capital of Germany? Answer in one sentence.",
    ]
    expected = ["Paris", "Berlin"]

    cache = make_cache(model, num_blocks=10, block_size=16)  # tight: forces real contention
    scheduler = Scheduler(model, cache, enable_cpu_swap=True)
    for i, q in enumerate(questions):
        scheduler.add_request(Sequence(seq_id=i, prompt_token_ids=tokenize(tokenizer, q), max_new_tokens=20))

    finished = scheduler.run_to_completion()
    finished_by_id = {s.seq_id: s for s in finished}
    assert len(finished) == 2

    for i, (q, ans) in enumerate(zip(questions, expected)):
        text = tokenizer.decode(finished_by_id[i].output_token_ids, skip_special_tokens=True)
        print(f"\n[{q}] -> {text!r}")
        assert ans in text
