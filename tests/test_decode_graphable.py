import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.kv_cache import KVCacheManager
from wllm.models.qwen2 import load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]


def make_cache(model, num_blocks=32, block_size=16, dtype=torch.float32):
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


def test_graphable_decode_matches_eager_decode_step_batch():
    """Eager call (no graph capture yet) -- isolates whether the
    index_copy_-based cache write is mathematically equivalent to the
    Python-int-indexed write in decode_step_batch, before adding the
    complexity of actual CUDA graph capture/replay on top.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompt_ids = tokenizer.apply_chat_template(
        PROMPT, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"].to("cuda")

    model = load_native(MODEL_ID, dtype=torch.float32)

    # Two independent caches/sequences, prefilled identically, so their
    # cache state is byte-for-byte the same going into the decode step.
    cache_a = make_cache(model)
    cache_b = make_cache(model)
    model.prefill_with_cache(cache_a, seq_id=0, input_ids=prompt_ids)
    model.prefill_with_cache(cache_b, seq_id=0, input_ids=prompt_ids)

    next_token = torch.tensor([[100]], device="cuda")  # arbitrary valid token id

    logits_eager = model.decode_step_batch(cache_a, [0], next_token)

    block_size = cache_b.block_size
    old_len = cache_b.context_len(0)
    (block_id, slot), = cache_b.reserve(0, 1)
    write_idx = torch.tensor([block_id * block_size + slot], dtype=torch.long, device="cuda")
    position_ids = torch.tensor([[old_len]], device="cuda")
    block_table = cache_b.block_tables_tensor([0], device="cuda")
    context_len = torch.tensor([old_len + 1], dtype=torch.int32, device="cuda")

    logits_graphable = model.decode_step_graphable(cache_b, next_token, position_ids, block_table, context_len, write_idx)
    cache_b.advance(0, 1)

    assert torch.equal(logits_eager, logits_graphable), "graphable decode path diverged from the proven eager path"

    # cache state itself (not just logits) must match too, since later
    # decode steps read back whatever got written this step.
    for i in range(model.cfg.num_hidden_layers):
        assert torch.equal(cache_a.k_caches[i], cache_b.k_caches[i]), f"layer {i} k_cache diverged"
        assert torch.equal(cache_a.v_caches[i], cache_b.v_caches[i]), f"layer {i} v_cache diverged"
