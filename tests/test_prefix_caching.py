import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.kv_cache import KVCacheManager
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


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0].tolist()


def test_no_match_path_is_identical_to_original_prefill(model_and_tokenizer):
    """With prefix caching disabled (or nothing registered yet), the new
    unified prefill path must produce bit-identical output to the original,
    already-proven prefill_with_cache -- it's the same math (prefix_kv=None
    exercises the same code shape as forward_prefill).
    """
    model, tokenizer = model_and_tokenizer
    prompt_ids = encode(tokenizer, "What is the capital of France? Answer in one sentence.")
    input_ids = torch.tensor([prompt_ids], device="cuda")

    cache_a = make_cache(model, num_blocks=32)
    cache_b = make_cache(model, num_blocks=32, enable_prefix_caching=True)

    logits_original = model.prefill_with_cache(cache_a, seq_id=0, input_ids=input_ids)
    logits_new = model.prefill_with_cache_and_prefix_reuse(cache_b, seq_id=0, prompt_token_ids=prompt_ids)

    assert torch.equal(logits_original, logits_new)


def test_shared_prefix_is_actually_reused_not_recomputed(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    shared_prefix = (
        "You are a helpful assistant with extensive knowledge of world geography and history. "
        "Always answer concisely in a single sentence."
    )
    question_a = f"{shared_prefix} What is the capital of France?"
    question_b = f"{shared_prefix} What is the capital of Germany?"
    ids_a = encode(tokenizer, question_a)
    ids_b = encode(tokenizer, question_b)

    cache = make_cache(model, num_blocks=32, enable_prefix_caching=True)
    model.prefill_with_cache_and_prefix_reuse(cache, seq_id=0, prompt_token_ids=ids_a)

    free_before = cache.allocator.num_free
    model.prefill_with_cache_and_prefix_reuse(cache, seq_id=1, prompt_token_ids=ids_b)
    free_after = cache.allocator.num_free

    blocks_b_used = len(cache.sequences[1].block_table)
    blocks_newly_allocated = free_before - free_after
    print(f"\nseq1 total blocks: {blocks_b_used}, newly allocated: {blocks_newly_allocated}")
    assert blocks_newly_allocated < blocks_b_used, "no blocks were actually reused"

    # some of seq1's block table entries must be shared (refcount 2) with seq0
    shared = set(cache.sequences[0].block_table) & set(cache.sequences[1].block_table)
    assert len(shared) > 0
    for b in shared:
        assert cache.allocator.ref_counts[b] == 2


def test_prefix_reuse_produces_mathematically_identical_logits_to_full_recompute(model_and_tokenizer):
    """The real correctness bar: reusing cached blocks must be numerically
    transparent -- computing seq1 with prefix reuse must match computing it
    completely fresh (no cache sharing at all), not just "look plausible."
    """
    model, tokenizer = model_and_tokenizer
    shared_prefix = (
        "You are a helpful assistant with extensive knowledge of world geography and history. "
        "Always answer concisely in a single sentence."
    )
    question_a = f"{shared_prefix} What is the capital of France?"
    question_b = f"{shared_prefix} What is the capital of Germany?"
    ids_a = encode(tokenizer, question_a)
    ids_b = encode(tokenizer, question_b)

    # Path 1: seq0 primes the cache, seq1 reuses its shared prefix.
    cache_shared = make_cache(model, num_blocks=32, enable_prefix_caching=True)
    model.prefill_with_cache_and_prefix_reuse(cache_shared, seq_id=0, prompt_token_ids=ids_a)
    logits_reused = model.prefill_with_cache_and_prefix_reuse(cache_shared, seq_id=1, prompt_token_ids=ids_b)

    # Path 2: seq1 computed completely fresh, no sharing at all. Its logits
    # cover the whole prompt; logits_reused only ever computed the suffix
    # (prefix reuse skips the shared portion entirely), so compare just the
    # last suffix_len positions -- causal attention means position i's
    # hidden state only depends on positions 0..i, so these must match.
    cache_fresh = make_cache(model, num_blocks=32)
    logits_fresh = model.prefill_with_cache(cache_fresh, seq_id=0, input_ids=torch.tensor([ids_b], device="cuda"))

    suffix_len = logits_reused.shape[1]
    assert torch.allclose(logits_reused, logits_fresh[:, -suffix_len:], atol=1e-3)


def test_end_to_end_generation_with_prefix_reuse_is_correct(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    shared_prefix = (
        "You are a helpful assistant with extensive knowledge of world geography and history. "
        "Always answer concisely in a single sentence."
    )
    questions = [f"{shared_prefix} What is the capital of France?", f"{shared_prefix} What is the capital of Germany?"]
    expected = ["Paris", "Berlin"]

    cache = make_cache(model, num_blocks=32, enable_prefix_caching=True)

    for seq_id, (q, ans) in enumerate(zip(questions, expected)):
        prompt_ids = encode(tokenizer, q)
        logits = model.prefill_with_cache_and_prefix_reuse(cache, seq_id, prompt_ids)
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)
        tokens = [next_token]
        for _ in range(15):
            logits = model.decode_step_with_cache(cache, seq_id, next_token)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(next_token)
        output_ids = torch.cat(tokens, dim=1)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n[{q}] -> {text!r}")
        assert ans in text
