"""Proves the engine genuinely runs on CPU, not just that CUDA-only pieces
fail gracefully: the real Scheduler + paged KV cache path (the same one
every GPU test in this suite exercises) produces exactly the same tokens as
a naive full-recompute forward pass, entirely without CUDA -- no GPU, no
CUDA Toolkit, no MSVC. See tests/test_paged_attention_kernel.py for the
lower-level proof that the CPU fallback decode kernel matches the reference
math; this is the proof that the rest of the engine (Scheduler's device
auto-detection, KVCacheManager, block allocation, prefill/decode) is wired
up correctly around it.

Deliberately does not exercise CUDAGraphDecoder here -- it has no CPU
equivalent by design (see cuda_graph_decoder.py's guard clause) and always
requires an actual CUDA device regardless of test intent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import Sequence
from wllm.models.config import ModelConfig
from wllm.models.llama import LlamaForCausalLM, generate_greedy_naive


def make_tiny_model() -> LlamaForCausalLM:
    cfg = ModelConfig(
        vocab_size=64, hidden_size=32, intermediate_size=37, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, rope_theta=10000.0,
        max_position_embeddings=128, tie_word_embeddings=False, attention_bias=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(device="cpu", dtype=torch.float32)
    model.eval()
    return model


def test_scheduler_derives_cpu_device_from_model():
    """The real bug this fixes: Scheduler used to default device="cuda"
    unconditionally, so a CPU model + CPU cache handed to a bare
    Scheduler(model, cache) would build its own tensors on the wrong device
    and crash with a device-mismatch error, not a clear "no CUDA" message.
    """
    model = make_tiny_model()
    cache = KVCacheManager(
        num_layers=model.cfg.num_hidden_layers, num_blocks=16, block_size=8,
        num_kv_heads=model.cfg.num_key_value_heads, head_dim=model.cfg.head_dim, device="cpu",
    )
    scheduler = Scheduler(model, cache)
    assert scheduler.device.type == "cpu"


def test_cuda_graph_decoder_rejects_cpu_with_a_clear_error():
    from wllm.engine.cuda_graph_decoder import CUDAGraphDecoder

    model = make_tiny_model()
    cache = KVCacheManager(
        num_layers=model.cfg.num_hidden_layers, num_blocks=16, block_size=8,
        num_kv_heads=model.cfg.num_key_value_heads, head_dim=model.cfg.head_dim, device="cpu",
    )
    with pytest.raises(ValueError, match="CUDA"):
        CUDAGraphDecoder(model, cache, device="cpu")


@pytest.mark.parametrize("prompt_len,max_new_tokens", [(4, 6), (1, 10)])
def test_cpu_paged_serving_matches_naive_full_recompute(prompt_len, max_new_tokens):
    model = make_tiny_model()
    torch.manual_seed(1)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_len)).tolist()[0]

    naive_out = generate_greedy_naive(model, torch.tensor([prompt]), max_new_tokens=max_new_tokens)
    naive_tokens = naive_out[0, prompt_len:].tolist()

    cache = KVCacheManager(
        num_layers=model.cfg.num_hidden_layers, num_blocks=16, block_size=8,
        num_kv_heads=model.cfg.num_key_value_heads, head_dim=model.cfg.head_dim, device="cpu",
    )
    scheduler = Scheduler(model, cache)
    scheduler.add_request(Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=max_new_tokens, eos_token_id=None))
    finished = scheduler.run_to_completion()[0]

    assert finished.output_token_ids == naive_tokens


def test_cpu_paged_serving_handles_a_batch():
    """Multiple concurrent sequences with different lengths through one
    scheduler -- decode_step_batch's block_tables_tensor/context_lens_tensor
    padding logic, not just the single-sequence path above.
    """
    model = make_tiny_model()
    torch.manual_seed(2)
    prompts = [
        torch.randint(0, model.cfg.vocab_size, (1, 3)).tolist()[0],
        torch.randint(0, model.cfg.vocab_size, (1, 7)).tolist()[0],
        torch.randint(0, model.cfg.vocab_size, (1, 5)).tolist()[0],
    ]

    cache = KVCacheManager(
        num_layers=model.cfg.num_hidden_layers, num_blocks=32, block_size=8,
        num_kv_heads=model.cfg.num_key_value_heads, head_dim=model.cfg.head_dim, device="cpu",
    )
    scheduler = Scheduler(model, cache)
    for i, prompt in enumerate(prompts):
        scheduler.add_request(Sequence(seq_id=i, prompt_token_ids=prompt, max_new_tokens=5, eos_token_id=None))
    finished = {s.seq_id: s for s in scheduler.run_to_completion()}

    for i, prompt in enumerate(prompts):
        naive_out = generate_greedy_naive(model, torch.tensor([prompt]), max_new_tokens=5)
        assert finished[i].output_token_ids == naive_out[0, len(prompt):].tolist()
