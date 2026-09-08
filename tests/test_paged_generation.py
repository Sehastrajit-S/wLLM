import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.kv_cache import KVCacheManager
from wllm.models.qwen2 import (
    generate_greedy_naive,
    generate_with_kv_cache,
    load_native,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
MAX_NEW_TOKENS = 16


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


def test_paged_generation_matches_naive_generation_fp32():
    """Removes bf16 precision drift as a variable (same reasoning as P1.2's
    fp32 check): if the paged-cache decode path is mathematically equivalent
    to full causal attention, fp32 token IDs must match exactly.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = tokenizer.apply_chat_template(
        PROMPT, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"].to("cuda")

    model = load_native(MODEL_ID, dtype=torch.float32)

    naive_tokens = generate_greedy_naive(model, input_ids, MAX_NEW_TOKENS)

    cache = make_cache(model)
    paged_tokens = generate_with_kv_cache(model, cache, seq_id=0, input_ids=input_ids, max_new_tokens=MAX_NEW_TOKENS)

    naive_text = tokenizer.decode(naive_tokens[0, input_ids.shape[1] :], skip_special_tokens=True)
    paged_text = tokenizer.decode(paged_tokens[0, input_ids.shape[1] :], skip_special_tokens=True)
    print(f"\nnaive (no cache): {naive_text!r}")
    print(f"paged (KV cache): {paged_text!r}")

    assert torch.equal(naive_tokens, paged_tokens), "paged-cache generation diverged from naive full-recompute generation"


def test_paged_generation_bf16_produces_correct_text():
    """Functional smoke test at the model's real inference dtype."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    input_ids = tokenizer.apply_chat_template(
        PROMPT, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"].to("cuda")

    model = load_native(MODEL_ID, dtype=torch.bfloat16)
    cache = make_cache(model)
    tokens = generate_with_kv_cache(model, cache, seq_id=0, input_ids=input_ids, max_new_tokens=MAX_NEW_TOKENS)
    text = tokenizer.decode(tokens[0, input_ids.shape[1] :], skip_special_tokens=True)

    print(f"\nbf16 paged output: {text!r}")
    assert "Paris" in text
