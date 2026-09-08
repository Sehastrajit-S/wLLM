"""Proves the native model implementation genuinely supports more than
Qwen2: TinyLlama-1.1B-Chat is a real LlamaForCausalLM checkpoint (ungated,
small enough for a rigorous fp32 comparison), and the only structural
difference this codebase's Attention/MLP/RMSNorm/RoPE/HF-weight-naming
already didn't share with Llama was q/k/v projection bias (Qwen2 has it,
Llama doesn't) -- see ModelConfig.attention_bias. Nothing else needed to
change: the same load_native(), the same PagedAttention kernel, the same
Scheduler, the same CUDAGraphDecoder all work unmodified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.baseline.generate import capture_reference_logits
from wllm.baseline.model import load as load_baseline
from wllm.engine.cuda_graph_decoder import CUDAGraphDecoder
from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import Sequence
from wllm.models.qwen2 import load_native

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]


def make_cache(model, num_blocks, block_size=16, dtype=torch.float32):
    cfg = model.cfg
    return KVCacheManager(
        num_layers=cfg.num_hidden_layers, num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim, device="cuda", dtype=dtype,
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


def test_llama_config_has_no_attention_bias(model_and_tokenizer):
    model, _ = model_and_tokenizer
    assert model.cfg.attention_bias is False
    assert model.cfg.num_attention_heads != model.cfg.num_key_value_heads, "expected this checkpoint to use GQA"


def test_native_llama_matches_hf_baseline_in_fp32(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    hf_model, hf_tokenizer = load_baseline(MODEL_ID, dtype=torch.float32)
    ref = capture_reference_logits(hf_model, hf_tokenizer, PROMPT)
    del hf_model
    torch.cuda.empty_cache()

    with torch.inference_mode():
        logits = model(ref["input_ids"].to("cuda"))

    diff = (logits.float().cpu() - ref["logits"]).abs()
    mismatches = (logits.argmax(-1).cpu() != ref["logits"].argmax(-1)).sum().item()

    assert diff.max().item() < 1e-2, f"max diff {diff.max().item()}"
    assert mismatches == 0, f"{mismatches} argmax mismatches vs HF baseline"


def test_llama_serves_correctly_through_scheduler_and_kernel(model_and_tokenizer):
    """The real point of this codebase's architecture: PagedAttention kernel,
    continuous batching scheduler, and KV cache are config-driven, not
    Qwen2-specific -- this proves it with an actual batched generation
    through the identical code path Qwen2 uses.
    """
    model, tokenizer = model_and_tokenizer
    questions = [
        "What is the capital of France? Answer in one sentence.",
        "What is the capital of Germany? Answer in one sentence.",
    ]
    cache = make_cache(model, num_blocks=64)
    scheduler = Scheduler(model, cache)
    for i, q in enumerate(questions):
        scheduler.add_request(
            Sequence(seq_id=i, prompt_token_ids=tokenize(tokenizer, q), max_new_tokens=24, eos_token_id=tokenizer.eos_token_id)
        )
    finished = {s.seq_id: s for s in scheduler.run_to_completion()}

    assert "Paris" in tokenizer.decode(finished[0].output_token_ids, skip_special_tokens=True)
    assert "Berlin" in tokenizer.decode(finished[1].output_token_ids, skip_special_tokens=True)


def test_llama_cuda_graph_decode_matches_eager(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    q = "What is the capital of France? Answer in one sentence."

    eager_cache = make_cache(model, num_blocks=32)
    eager_scheduler = Scheduler(model, eager_cache)
    eager_scheduler.add_request(
        Sequence(seq_id=0, prompt_token_ids=tokenize(tokenizer, q), max_new_tokens=24, eos_token_id=tokenizer.eos_token_id)
    )
    eager_text = tokenizer.decode(eager_scheduler.run_to_completion()[0].output_token_ids, skip_special_tokens=True)

    graph_cache = make_cache(model, num_blocks=32 + 4)
    graph_decoder = CUDAGraphDecoder(model, graph_cache, bucket_sizes=(1, 2), max_blocks_per_seq=16)
    graph_decoder.capture()
    graph_scheduler = Scheduler(model, graph_cache, graph_decoder=graph_decoder)
    graph_scheduler.add_request(
        Sequence(seq_id=0, prompt_token_ids=tokenize(tokenizer, q), max_new_tokens=24, eos_token_id=tokenizer.eos_token_id)
    )
    graph_text = tokenizer.decode(graph_scheduler.run_to_completion()[0].output_token_ids, skip_special_tokens=True)

    assert graph_text == eager_text
    assert "Paris" in graph_text
