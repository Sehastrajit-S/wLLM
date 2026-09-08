import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import Sequence
from wllm.models.qwen2 import load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def make_cache(model, num_blocks=32, block_size=16):
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


REPETITIVE_PROMPT = "Repeat the following sentence exactly three times: The quick brown fox jumps over the lazy dog."


def test_speculative_decoding_matches_non_speculative_greedy_exactly(model_and_tokenizer):
    """The real correctness bar: with a prompt likely to induce genuine
    repetition (so the n-gram drafter actually has something to find, not
    just falling back to plain decode every step), speculative decoding
    must produce token-for-token identical output to greedy decode without it.
    """
    model, tokenizer = model_and_tokenizer
    prompt = tokenize(tokenizer, REPETITIVE_PROMPT)
    max_new_tokens = 40

    cache_plain = make_cache(model)
    scheduler_plain = Scheduler(model, cache_plain)
    seq_plain = Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=max_new_tokens)
    scheduler_plain.add_request(seq_plain)
    scheduler_plain.run_to_completion()

    cache_spec = make_cache(model)
    scheduler_spec = Scheduler(model, cache_spec, enable_speculative_decoding=True, speculative_ngram_size=3, speculative_max_draft_len=4)
    seq_spec = Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=max_new_tokens)
    scheduler_spec.add_request(seq_spec)
    scheduler_spec.run_to_completion()

    text_plain = tokenizer.decode(seq_plain.output_token_ids, skip_special_tokens=True)
    text_spec = tokenizer.decode(seq_spec.output_token_ids, skip_special_tokens=True)
    print(f"\nplain: {text_plain!r}\nspec:  {text_spec!r}")
    assert seq_plain.output_token_ids == seq_spec.output_token_ids


def test_speculative_decoding_actually_accepts_multiple_tokens_in_some_step(model_and_tokenizer):
    """Guards against the exact-match test above passing for the wrong
    reason (the drafter never finding anything and silently falling back to
    plain decode every step, which would trivially match). Confirms multi-
    token acceptance genuinely happens somewhere during a real run.
    """
    model, tokenizer = model_and_tokenizer
    prompt = tokenize(tokenizer, REPETITIVE_PROMPT)

    cache = make_cache(model)
    scheduler = Scheduler(model, cache, enable_speculative_decoding=True, speculative_ngram_size=3, speculative_max_draft_len=4)
    seq = Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=50)
    scheduler.add_request(seq)

    max_gain_in_one_step = 0
    while scheduler.has_unfinished_requests():
        before = len(seq.output_token_ids)
        scheduler.step()
        gained = len(seq.output_token_ids) - before
        max_gain_in_one_step = max(max_gain_in_one_step, gained)

    print(f"\nmax tokens accepted in a single step: {max_gain_in_one_step}")
    assert max_gain_in_one_step > 1, "speculative decoding never actually accepted more than 1 token in any step"


def test_speculative_decode_one_rejects_wrong_draft_and_rolls_back_cleanly(model_and_tokenizer):
    """Direct, low-level check: feed a deliberately wrong draft and confirm
    exactly one (corrected) token gets appended, matching what plain decode
    would have produced from the same state, and the cache is rolled back
    to the right length (not left with 4 tentatively-written garbage slots).
    """
    model, tokenizer = model_and_tokenizer
    prompt = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")

    cache_ref = make_cache(model)
    model.prefill_with_cache(cache_ref, seq_id=0, input_ids=torch.tensor([prompt], device="cuda"))

    cache = make_cache(model)
    scheduler = Scheduler(model, cache)
    seq = Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=20)
    scheduler.add_request(seq)
    scheduler._admit()
    context_len_before = cache.context_len(0)

    # what plain decode would produce from here, for comparison
    ref_logits = model.decode_step_with_cache(cache_ref, 0, torch.tensor([[seq.last_token_id]], device="cuda"))
    expected_next_token = ref_logits[:, -1, :].argmax(-1).item()

    bogus_draft = [1, 2, 3, 4]  # essentially guaranteed to mismatch immediately
    ok = scheduler._speculative_decode_one(seq, bogus_draft)

    assert ok
    assert len(seq.output_token_ids) == 2  # the 1 prefill token + exactly 1 more (the correction)
    assert seq.output_token_ids[-1] == expected_next_token
    assert cache.context_len(0) == context_len_before + 1, "cache was not rolled back to just the accepted token"
