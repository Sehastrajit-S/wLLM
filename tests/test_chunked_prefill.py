import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.kv_cache import KVCacheManager
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


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0].tolist()


def test_multi_chunk_prefill_matches_single_shot_prefill(model_and_tokenizer):
    """The core primitive check: splitting a prompt into several chunks and
    calling continue_prefill repeatedly must produce logits identical to
    processing the whole thing in one prefill_with_cache call.
    """
    model, tokenizer = model_and_tokenizer
    prompt_ids = encode(
        tokenizer,
        "Please write a detailed, multi-paragraph description of the water cycle, covering evaporation, "
        "condensation, precipitation, and collection, with examples of each stage.",
    )

    cache_single = make_cache(model, num_blocks=32)
    logits_single = model.prefill_with_cache(cache_single, seq_id=0, input_ids=torch.tensor([prompt_ids], device="cuda"))

    cache_chunked = make_cache(model, num_blocks=32)
    cache_chunked.create_sequence(0)
    chunk_size = 12
    logits_chunked = None
    for i in range(0, len(prompt_ids), chunk_size):
        chunk = prompt_ids[i : i + chunk_size]
        logits_chunked = model.continue_prefill(cache_chunked, 0, chunk)

    assert cache_chunked.context_len(0) == len(prompt_ids)
    assert torch.allclose(logits_single[:, -1, :], logits_chunked[:, -1, :], atol=1e-3)

    # cache content itself must match too, not just the final logits, since
    # decode steps afterward read back whatever got written.
    for i in range(model.cfg.num_hidden_layers):
        idx_single = torch.tensor(cache_single.sequences[0].block_table, device="cuda")
        idx_chunked = torch.tensor(cache_chunked.sequences[0].block_table, device="cuda")
        k_single = cache_single.k_caches[i][idx_single].reshape(-1, model.cfg.num_key_value_heads, model.cfg.head_dim)[
            : len(prompt_ids)
        ]
        k_chunked = cache_chunked.k_caches[i][idx_chunked].reshape(-1, model.cfg.num_key_value_heads, model.cfg.head_dim)[
            : len(prompt_ids)
        ]
        assert torch.allclose(k_single, k_chunked, atol=1e-3), f"layer {i} k_cache diverged"


def test_multi_chunk_prefill_with_prefix_reuse_combined(model_and_tokenizer):
    """Chunked prefill on top of a reused prefix: the first chunk should
    attend to the reused prefix, later chunks to prefix+earlier-chunks.
    """
    model, tokenizer = model_and_tokenizer
    shared_prefix = (
        "You are a helpful assistant with extensive knowledge of world geography and history. "
        "Always answer concisely in a single sentence."
    )
    ids_a = encode(tokenizer, f"{shared_prefix} What is the capital of France?")
    ids_b = encode(tokenizer, f"{shared_prefix} What is the capital of Germany?")

    cache = make_cache(model, num_blocks=32)
    cache.enable_prefix_caching = True
    model.prefill_with_cache_and_prefix_reuse(cache, seq_id=0, prompt_token_ids=ids_a)

    matched_blocks, num_matched = cache.match_prefix(ids_b)
    assert num_matched > 0, "test setup expects a real prefix match"
    cache.create_sequence_from_prefix(1, matched_blocks)

    suffix = ids_b[num_matched:]
    chunk_size = 5
    logits = None
    for i in range(0, len(suffix), chunk_size):
        logits = model.continue_prefill(cache, 1, suffix[i : i + chunk_size])

    # ground truth: seq1 computed completely fresh (no reuse, no chunking)
    cache_fresh = make_cache(model, num_blocks=32)
    logits_fresh = model.prefill_with_cache(cache_fresh, seq_id=0, input_ids=torch.tensor([ids_b], device="cuda"))

    assert torch.allclose(logits[:, -1, :], logits_fresh[:, -1, :], atol=1e-3)
