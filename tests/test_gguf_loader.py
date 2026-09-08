import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from wllm.engine.kv_cache import KVCacheManager
from wllm.models.qwen2 import generate_greedy_naive, generate_with_kv_cache
from wllm.quant.gguf_loader import load_gguf
from wllm.quant.quant_linear import QuantizedLinear

REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.fixture(scope="module")
def prompt_ids(tokenizer):
    # Skip must live here, not in the test bodies below: pytest resolves
    # fixtures during test setup, before the test function itself runs, so a
    # `pytest.skip()` inside the test body never gets a chance to fire if
    # this fixture raises first (exactly what happened in CI once the
    # .gitignore bug was fixed and this file was actually collected/run for
    # the first time -- `.to("cuda")` here errored at setup instead of
    # skipping cleanly like every other test in this file already does).
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    inputs = tokenizer.apply_chat_template(PROMPT, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    return inputs["input_ids"].to("cuda")


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


@pytest.mark.parametrize("gguf_file", ["qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2.5-0.5b-instruct-q4_0.gguf"])
def test_gguf_model_uses_quantized_linear_layers(gguf_file):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    path = hf_hub_download(REPO, gguf_file)
    model = load_gguf(path, device="cuda")
    assert isinstance(model.model.layers[0].self_attn.q_proj, QuantizedLinear)
    assert isinstance(model.lm_head, QuantizedLinear)
    # embed_tokens is eagerly dequantized (see gguf_loader's rationale), so it
    # stays a regular nn.Embedding rather than becoming a quantized module.
    assert not isinstance(model.model.embed_tokens, QuantizedLinear)


@pytest.mark.parametrize(
    "gguf_file,expected_dtype_name",
    [("qwen2.5-0.5b-instruct-q8_0.gguf", "Q8_0"), ("qwen2.5-0.5b-instruct-q4_0.gguf", "Q4_0")],
)
def test_gguf_model_naive_generation_is_correct(gguf_file, expected_dtype_name, prompt_ids, tokenizer):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    path = hf_hub_download(REPO, gguf_file)
    model = load_gguf(path, device="cuda")

    out = generate_greedy_naive(model, prompt_ids, max_new_tokens=16)
    text = tokenizer.decode(out[0, prompt_ids.shape[1] :], skip_special_tokens=True)
    print(f"\n{expected_dtype_name} naive: {text!r}")
    assert "Paris" in text


@pytest.mark.parametrize(
    "gguf_file,expected_dtype_name",
    [("qwen2.5-0.5b-instruct-q8_0.gguf", "Q8_0"), ("qwen2.5-0.5b-instruct-q4_0.gguf", "Q4_0")],
)
def test_gguf_model_paged_kv_cache_generation_is_correct(gguf_file, expected_dtype_name, prompt_ids, tokenizer):
    """The real production path -- QuantizedLinear must work as a drop-in
    inside the actual PagedAttention prefill/decode pipeline, not just a
    naive full-recompute forward pass.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    path = hf_hub_download(REPO, gguf_file)
    model = load_gguf(path, device="cuda")
    cache = make_cache(model)

    out = generate_with_kv_cache(model, cache, seq_id=0, input_ids=prompt_ids, max_new_tokens=16)
    text = tokenizer.decode(out[0, prompt_ids.shape[1] :], skip_special_tokens=True)
    print(f"\n{expected_dtype_name} paged: {text!r}")
    assert "Paris" in text


def test_gguf_model_uses_meaningfully_less_gpu_memory_than_bf16(prompt_ids):
    """Confirms quantized weights actually stay compact in GPU memory rather
    than being expanded to bf16 at load time (which would defeat the point).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from wllm.models.qwen2 import load_native

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    bf16_model = load_native("Qwen/Qwen2.5-0.5B-Instruct", device="cuda", dtype=torch.bfloat16)
    bf16_mem = torch.cuda.memory_allocated()
    del bf16_model
    torch.cuda.empty_cache()

    path = hf_hub_download(REPO, "qwen2.5-0.5b-instruct-q4_0.gguf")
    q4_model = load_gguf(path, device="cuda")
    q4_mem = torch.cuda.memory_allocated()
    del q4_model

    print(f"\nbf16: {bf16_mem / 1e9:.3f}GB, Q4_0: {q4_mem / 1e9:.3f}GB")
    # Not the full ~4x a Q4 quantization ratio would suggest: embed_tokens
    # (eagerly dequantized to bf16 for lookup efficiency, see gguf_loader)
    # is ~136M of this model's 494M params -- over a quarter -- and doesn't
    # shrink with quantization at all. That's specific to small, vocab-heavy
    # models like this one; a 7B+ model would show a much better ratio since
    # the embedding table is a far smaller fraction of total params there.
    assert q4_mem < bf16_mem * 0.7
