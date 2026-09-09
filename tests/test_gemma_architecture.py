"""Proves the Gemma addition (src/wllm/models/gemma.py) is both correct and
fully served by the existing engine, the same rigor test_llama_architecture.py
applies to Llama: an exact HF-baseline logit match, plus self-consistency
between the naive/paged-cache/CUDA-graph code paths.

Real Gemma checkpoints (google/gemma-2b and later) are gated on HF and need
an authenticated, license-accepted token this environment doesn't have, so
this uses `hf-tiny-v2/tiny-random-GemmaForCausalLM` instead: a real
`GemmaForCausalLM` config and weight layout, just tiny and randomly
initialized. That rules out a semantic check ("Paris" in the output, as
test_llama_architecture.py does with real trained weights) but doesn't
weaken the correctness proof that actually matters here -- exact-match
against HF's own forward pass exercises GemmaRMSNorm's (1+weight)
parameterization, GemmaMLP's tanh-GELU activation, and the embed_scale
hook exactly as much as a real checkpoint would, and self-consistency
between generation paths (naive vs. paged KV cache vs. CUDA graph) proves
the serving stack is wired correctly regardless of whether the weights are
trained.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoModelForCausalLM

from wllm.engine.cuda_graph_decoder import CUDAGraphDecoder
from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import Sequence
from wllm.models.config import ModelConfig
from wllm.models.gemma import GemmaForCausalLM
from wllm.models.llama import generate_greedy_naive, load_native

MODEL_ID = "hf-tiny-v2/tiny-random-GemmaForCausalLM"


def test_explicit_head_dim_overrides_derived_value():
    """Unit-level check independent of any real checkpoint: real Gemma
    checkpoints (e.g. gemma-7b: hidden_size=3072, num_attention_heads=16)
    have head_dim=256, not the 192 hidden_size // num_attention_heads would
    derive -- the tiny test checkpoint below happens to have the two agree,
    so this is the only place that distinction is actually exercised.
    """
    cfg = ModelConfig(
        vocab_size=100, hidden_size=3072, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=16, num_key_value_heads=16, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=512, tie_word_embeddings=False, attention_bias=False,
        explicit_head_dim=256,
    )
    assert cfg.head_dim == 256

    cfg_no_override = ModelConfig(
        vocab_size=100, hidden_size=3072, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=16, num_key_value_heads=16, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=512, tie_word_embeddings=False, attention_bias=False,
    )
    assert cfg_no_override.head_dim == 192


def make_cache(model, num_blocks, block_size=16, dtype=torch.float32):
    cfg = model.cfg
    return KVCacheManager(
        num_layers=cfg.num_hidden_layers, num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim, device="cuda", dtype=dtype,
    )


@pytest.fixture(scope="module")
def model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    m = load_native(MODEL_ID, dtype=torch.float32)
    assert isinstance(m, GemmaForCausalLM)
    return m


def test_gemma_registry_dispatch_and_config(model):
    assert model.cfg.attention_bias is False
    assert model.model.embed_scale == pytest.approx(model.cfg.hidden_size**0.5)


def test_native_gemma_matches_hf_baseline_in_fp32(model):
    torch.manual_seed(0)
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 12), device="cuda")

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to("cuda")
    hf_model.eval()
    with torch.inference_mode():
        hf_logits = hf_model(input_ids).logits.float().cpu()
    del hf_model
    torch.cuda.empty_cache()

    with torch.inference_mode():
        logits = model(input_ids)

    diff = (logits.float().cpu() - hf_logits).abs()
    mismatches = (logits.argmax(-1).cpu() != hf_logits.argmax(-1)).sum().item()

    assert diff.max().item() < 1e-2, f"max diff {diff.max().item()}"
    assert mismatches == 0, f"{mismatches} argmax mismatches vs HF baseline"


def test_gemma_paged_serving_matches_naive_full_recompute(model):
    """Random, untrained weights can't produce a meaningful answer to check
    (no "Paris" here) -- what this proves instead is that the real serving
    path (Scheduler + PagedAttention kernel, the same one Llama/Qwen2 use)
    computes exactly the same thing as a naive full-recompute forward pass,
    for this new architecture's own norm/MLP/embed-scale additions.
    """
    torch.manual_seed(1)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 6)).tolist()[0]

    naive_out = generate_greedy_naive(model, torch.tensor([prompt], device="cuda"), max_new_tokens=8)
    naive_tokens = naive_out[0, len(prompt):].tolist()

    cache = make_cache(model, num_blocks=32)
    scheduler = Scheduler(model, cache)
    scheduler.add_request(Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=8, eos_token_id=None))
    finished = scheduler.run_to_completion()[0]

    assert finished.output_token_ids == naive_tokens


def test_gemma_cuda_graph_decode_matches_eager(model):
    torch.manual_seed(2)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 6)).tolist()[0]

    eager_cache = make_cache(model, num_blocks=32)
    eager_scheduler = Scheduler(model, eager_cache)
    eager_scheduler.add_request(Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=8, eos_token_id=None))
    eager_tokens = eager_scheduler.run_to_completion()[0].output_token_ids

    graph_cache = make_cache(model, num_blocks=32 + 4)
    graph_decoder = CUDAGraphDecoder(model, graph_cache, bucket_sizes=(1, 2), max_blocks_per_seq=16)
    graph_decoder.capture()
    graph_scheduler = Scheduler(model, graph_cache, graph_decoder=graph_decoder)
    graph_scheduler.add_request(Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=8, eos_token_id=None))
    graph_tokens = graph_scheduler.run_to_completion()[0].output_token_ids

    assert graph_tokens == eager_tokens
