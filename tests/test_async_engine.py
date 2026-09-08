import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.async_engine import AsyncEngine
from wllm.engine.kv_cache import KVCacheManager
from wllm.models.qwen2 import load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_cache(model, num_blocks=64, block_size=16):
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


@pytest.mark.anyio
async def test_single_request_streams_correct_text(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cache = make_cache(model)
    engine = AsyncEngine(model, cache, tokenizer)
    engine.start()

    prompt_ids = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")
    chunks = [chunk async for chunk in engine.generate(prompt_ids, max_new_tokens=16)]

    text = "".join(chunks)
    print(f"\nchunks: {chunks}")
    print(f"text: {text!r}")
    assert "Paris" in text
    assert len(chunks) > 1, "expected genuinely incremental streaming, not one giant chunk"

    await engine.stop()


@pytest.mark.anyio
async def test_concurrent_requests_dont_cross_wire(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cache = make_cache(model)
    engine = AsyncEngine(model, cache, tokenizer)
    engine.start()

    questions = [
        "What is the capital of France? Answer in one sentence.",
        "What is the capital of Germany? Answer in one sentence.",
        "What is the capital of Japan? Answer in one sentence.",
    ]
    expected_answers = ["Paris", "Berlin", "Tokyo"]

    async def run(question: str) -> str:
        prompt_ids = tokenize(tokenizer, question)
        chunks = [chunk async for chunk in engine.generate(prompt_ids, max_new_tokens=16)]
        return "".join(chunks)

    results = await asyncio.gather(*(run(q) for q in questions))

    for question, text, expected in zip(questions, results, expected_answers):
        print(f"\n[{question}] -> {text!r}")
        assert expected in text

    await engine.stop()


@pytest.mark.anyio
async def test_breaking_early_cancels_and_frees_cache(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    # Must be large enough for prompt_len + max_new_tokens(100) below, or
    # admission control correctly refuses to ever admit it (working as
    # designed) and the request starves forever instead of hitting the
    # early-break path this test means to exercise.
    cache = make_cache(model, num_blocks=16, block_size=16)
    engine = AsyncEngine(model, cache, tokenizer)
    engine.start()

    prompt_ids = tokenize(tokenizer, "Tell me a long story about a dragon.")
    seen = 0
    agen = engine.generate(prompt_ids, max_new_tokens=100)
    try:
        async for _ in agen:
            seen += 1
            if seen == 2:
                break  # simulates a client disconnect partway through
    finally:
        # `break` alone does NOT call aclose() on an async generator -- it
        # just stops iterating and leaves it suspended, with cleanup left to
        # GC on some unspecified schedule. Real callers (and this test) must
        # close it explicitly to run generate()'s finally block (which
        # cancels the sequence and frees its cache) promptly and reliably.
        await agen.aclose()

    assert cache.allocator.num_free == cache.allocator.num_blocks, "cancelled request's blocks were not freed"
    assert not engine.scheduler.has_unfinished_requests()

    await engine.stop()


@pytest.mark.anyio
async def test_beam_search_returns_full_text_and_can_run_alongside_regular_generate(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    cache = make_cache(model, num_blocks=64)
    engine = AsyncEngine(model, cache, tokenizer)
    engine.start()

    beam_prompt = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")
    regular_prompt = tokenize(tokenizer, "What is the capital of Germany? Answer in one sentence.")

    async def run_regular():
        chunks = [c async for c in engine.generate(regular_prompt, max_new_tokens=16)]
        return "".join(chunks)

    (beam_result, finish_reason), regular_result = await asyncio.gather(
        engine.generate_beam_search(beam_prompt, max_new_tokens=16, num_beams=3, eos_token_id=tokenizer.eos_token_id),
        run_regular(),
    )

    print(f"\nbeam: {beam_result!r} ({finish_reason})\nregular: {regular_result!r}")
    assert "Paris" in beam_result
    assert finish_reason in ("stop", "length")
    assert "Berlin" in regular_result

    await engine.stop()
