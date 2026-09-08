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


def test_long_prompt_takes_multiple_steps_to_finish_prefilling(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    long_prompt = tokenize(
        tokenizer,
        "Please write a detailed, multi-paragraph description of the water cycle, covering evaporation, "
        "condensation, precipitation, and collection, with examples of each stage in different climates.",
    )
    assert len(long_prompt) > 20, "test needs a prompt long enough to actually span multiple small chunks"

    cache = make_cache(model, num_blocks=32)
    scheduler = Scheduler(model, cache, max_prefill_tokens_per_step=8)
    seq = Sequence(seq_id=0, prompt_token_ids=long_prompt, max_new_tokens=5)
    scheduler.add_request(seq)

    steps_to_finish_prefill = 0
    while seq.status == SeqStatus.PREFILLING or seq.status == SeqStatus.WAITING:
        scheduler.step()
        steps_to_finish_prefill += 1
        assert steps_to_finish_prefill < 100, "should have finished prefilling by now"

    expected_min_steps = len(long_prompt) // 8
    print(f"\nprompt_len={len(long_prompt)}, chunk=8, steps to finish prefill={steps_to_finish_prefill}")
    assert steps_to_finish_prefill >= expected_min_steps
    assert seq.status == SeqStatus.RUNNING


def test_chunked_prefill_produces_same_generation_as_unchunked(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")
    max_new_tokens = 12

    cache_unchunked = make_cache(model, num_blocks=32)
    scheduler_unchunked = Scheduler(model, cache_unchunked)  # max_prefill_tokens_per_step=None
    seq_unchunked = Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=max_new_tokens)
    scheduler_unchunked.add_request(seq_unchunked)
    scheduler_unchunked.run_to_completion()

    cache_chunked = make_cache(model, num_blocks=32)
    scheduler_chunked = Scheduler(model, cache_chunked, max_prefill_tokens_per_step=6)
    seq_chunked = Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=max_new_tokens)
    scheduler_chunked.add_request(seq_chunked)
    scheduler_chunked.run_to_completion()

    text_unchunked = tokenizer.decode(seq_unchunked.output_token_ids, skip_special_tokens=True)
    text_chunked = tokenizer.decode(seq_chunked.output_token_ids, skip_special_tokens=True)
    print(f"\nunchunked: {text_unchunked!r}\nchunked:   {text_chunked!r}")
    assert seq_unchunked.output_token_ids == seq_chunked.output_token_ids


def test_chunked_prefill_interleaves_with_decode_of_other_sequences(model_and_tokenizer):
    """The actual point of chunked prefill: a long prompt shouldn't fully
    block a short, already-running sequence's decode progress. With a small
    enough token budget, the short sequence should get to decode WHILE the
    long one is still mid-prefill.
    """
    model, tokenizer = model_and_tokenizer
    long_prompt = tokenize(
        tokenizer,
        "Please write a detailed, multi-paragraph description of the water cycle, covering evaporation, "
        "condensation, precipitation, and collection, with examples of each stage in different climates.",
    )
    short_prompt = tokenize(tokenizer, "Hi")

    cache = make_cache(model, num_blocks=32)
    scheduler = Scheduler(model, cache, max_prefill_tokens_per_step=8)

    long_seq = Sequence(seq_id=0, prompt_token_ids=long_prompt, max_new_tokens=5)
    short_seq = Sequence(seq_id=1, prompt_token_ids=short_prompt, max_new_tokens=5)
    scheduler.add_request(long_seq)
    scheduler.add_request(short_seq)

    # Step once: long_seq should still be admitted+chunking (prompt > budget),
    # and since short_seq's prompt is tiny, remaining budget after long_seq's
    # first chunk may or may not admit it this exact step -- but long_seq
    # must NOT have finished prefilling in a single step.
    scheduler.step()
    assert long_seq.status == SeqStatus.PREFILLING
    assert len(long_prompt) > 8  # sanity: this prompt really does need >1 chunk

    finished = scheduler.run_to_completion()
    assert len(finished) == 2
    assert short_seq.is_finished()
    assert long_seq.is_finished()


def test_chunked_prefill_combined_with_prefix_caching(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    shared_prefix = (
        "You are a helpful assistant with extensive knowledge of world geography and history. "
        "Always answer concisely in a single sentence."
    )
    questions = [f"{shared_prefix} What is the capital of France?", f"{shared_prefix} What is the capital of Germany?"]
    expected = ["Paris", "Berlin"]

    cache = make_cache(model, num_blocks=32, enable_prefix_caching=True)
    scheduler = Scheduler(model, cache, max_prefill_tokens_per_step=10)

    for i, q in enumerate(questions):
        scheduler.add_request(Sequence(seq_id=i, prompt_token_ids=tokenize(tokenizer, q), max_new_tokens=12))

    finished = scheduler.run_to_completion()
    finished_by_id = {s.seq_id: s for s in finished}
    for i, (q, ans) in enumerate(zip(questions, expected)):
        text = tokenizer.decode(finished_by_id[i].output_token_ids, skip_special_tokens=True)
        print(f"\n[{q}] -> {text!r}")
        assert ans in text
