import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.cuda_graph_decoder import CUDAGraphDecoder
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


def tokenize(tokenizer, question: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0].tolist()


@torch.inference_mode()
def test_single_sequence_graph_replay_matches_eager_across_many_steps(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")
    input_ids = torch.tensor([prompt_ids], device="cuda")

    cache_eager = make_cache(model, num_blocks=32)
    cache_graph = make_cache(model, num_blocks=32 + 40)  # extra room for scratch sequences

    model.prefill_with_cache(cache_eager, seq_id=0, input_ids=input_ids)
    model.prefill_with_cache(cache_graph, seq_id=0, input_ids=input_ids)

    decoder = CUDAGraphDecoder(model, cache_graph, bucket_sizes=(1, 2, 4), max_blocks_per_seq=32)
    decoder.capture()

    token = 100  # arbitrary shared starting token for both paths
    for step in range(20):
        eager_logits = model.decode_step_batch(cache_eager, [0], torch.tensor([[token]], device="cuda"))
        eager_next = eager_logits[:, -1, :].argmax(-1).item()

        graph_logits = decoder.decode([0], [token])
        graph_next = graph_logits[:, -1, :].argmax(-1).item()

        assert torch.allclose(eager_logits, graph_logits, atol=1e-3), f"step {step}: logits diverged"
        assert eager_next == graph_next, f"step {step}: argmax token diverged"
        token = eager_next


@torch.inference_mode()
def test_batched_graph_replay_with_padding_matches_eager(model_and_tokenizer):
    """3 real sequences -> must be padded up to bucket size 4. Verifies the
    scratch-sequence padding doesn't corrupt or bleed into the real rows.
    """
    model, tokenizer = model_and_tokenizer
    questions = [
        "What is the capital of France? Answer in one sentence.",
        "What is the capital of Germany? Answer in one sentence.",
        "What is the capital of Japan? Answer in one sentence.",
    ]

    cache_eager = make_cache(model, num_blocks=64)
    cache_graph = make_cache(model, num_blocks=64 + 40)
    decoder = CUDAGraphDecoder(model, cache_graph, bucket_sizes=(1, 2, 4, 8), max_blocks_per_seq=32)
    decoder.capture()

    tokens = []
    for i, q in enumerate(questions):
        prompt_ids = tokenize(tokenizer, q)
        input_ids = torch.tensor([prompt_ids], device="cuda")
        model.prefill_with_cache(cache_eager, seq_id=i, input_ids=input_ids)
        model.prefill_with_cache(cache_graph, seq_id=i, input_ids=input_ids)
        tokens.append(100 + i)  # distinct arbitrary starting tokens

    seq_ids = [0, 1, 2]
    for step in range(15):
        eager_logits = model.decode_step_batch(cache_eager, seq_ids, torch.tensor([[t] for t in tokens], device="cuda"))
        eager_next = eager_logits[:, -1, :].argmax(-1).tolist()

        graph_logits = decoder.decode(seq_ids, tokens)
        graph_next = graph_logits[:, -1, :].argmax(-1).tolist()

        assert torch.allclose(eager_logits, graph_logits, atol=1e-3), f"step {step}: logits diverged"
        assert eager_next == graph_next, f"step {step}: tokens diverged"
        tokens = eager_next


@torch.inference_mode()
def test_end_to_end_generation_via_cuda_graph_decoder_is_correct(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")
    input_ids = torch.tensor([prompt_ids], device="cuda")

    cache = make_cache(model, num_blocks=32 + 40)
    decoder = CUDAGraphDecoder(model, cache, bucket_sizes=(1, 2, 4), max_blocks_per_seq=32)
    decoder.capture()

    logits = model.prefill_with_cache(cache, seq_id=0, input_ids=input_ids)
    token = logits[:, -1, :].argmax(-1).item()
    output_tokens = [token]

    for _ in range(15):
        logits = decoder.decode([0], [token])
        token = logits[:, -1, :].argmax(-1).item()
        output_tokens.append(token)

    text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    print(f"\nCUDA-graph-decoded output: {text!r}")
    assert "Paris" in text


@torch.inference_mode()
def test_batch_exceeding_largest_bucket_falls_back_to_eager(model_and_tokenizer):
    model, _ = model_and_tokenizer
    cache = make_cache(model, num_blocks=32 + 10)
    decoder = CUDAGraphDecoder(model, cache, bucket_sizes=(1, 2), max_blocks_per_seq=16)
    decoder.capture()

    input_ids = torch.tensor([[100] * 8], device="cuda")
    model.prefill_with_cache(cache, seq_id=0, input_ids=input_ids)
    model.prefill_with_cache(cache, seq_id=1, input_ids=input_ids)
    model.prefill_with_cache(cache, seq_id=2, input_ids=input_ids)

    # 3 sequences > largest bucket (2) -- must not raise, must fall back cleanly.
    logits = decoder.decode([0, 1, 2], [100, 100, 100])
    assert logits.shape[0] == 3
