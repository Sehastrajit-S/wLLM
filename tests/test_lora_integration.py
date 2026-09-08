"""End-to-end LoRA correctness: real Qwen2.5-0.5B model, real Scheduler,
proving the plumbing added across Attention/DecoderLayer/Qwen2ForCausalLM/
Scheduler actually applies (and correctly isolates) adapters, rather than
just the standalone tensor-math unit tests in test_lora.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.cuda_graph_decoder import CUDAGraphDecoder
from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.lora import LoRARegistry
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import Sequence
from wllm.models.qwen2 import load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
QUESTIONS = [
    "What is the capital of France? Answer in one sentence.",
    "What is the capital of Germany? Answer in one sentence.",
    "What is the capital of Japan? Answer in one sentence.",
]


def make_cache(model, num_blocks, block_size=16, dtype=torch.float32):
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


def _zero_adapter(model) -> LoRARegistry:
    """An adapter with real (nonzero A, but all-zero B) weights on every
    projection of every layer -- mathematically a guaranteed no-op delta
    (A @ x then @ 0 == 0) while still exercising every lookup/gather/scatter
    the real code path performs. If the wiring were buggy in a way that
    accidentally scaled or corrupted the base output instead of purely
    adding a (here, zero) delta, this would catch it.
    """
    cfg = model.cfg
    rank = 4
    registry = LoRARegistry()
    weights = {}
    dims = {
        "q_proj": (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
        "k_proj": (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
        "v_proj": (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
        "o_proj": (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
    }
    for layer_idx in range(cfg.num_hidden_layers):
        for module_name, (out_f, in_f) in dims.items():
            A = torch.randn(rank, in_f, device="cuda")
            B = torch.zeros(out_f, rank, device="cuda")
            weights[(layer_idx, module_name)] = (A, B, 2.0)
    registry._adapters["zero_adapter"] = weights
    return registry


def _random_adapter(model, seed: int, scale: float = 50.0) -> LoRARegistry:
    """A deliberately large-effect adapter on layer 0's o_proj only -- enough
    to guarantee a visible change in greedy output, used as a sanity check
    that LoRA actually does something (the complementary failure mode to the
    zero-adapter test above).
    """
    cfg = model.cfg
    rank = 4
    g = torch.Generator(device="cuda").manual_seed(seed)
    out_f, in_f = cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim
    A = torch.randn(rank, in_f, device="cuda", generator=g)
    B = torch.randn(out_f, rank, device="cuda", generator=g)
    registry = LoRARegistry()
    registry._adapters[f"adapter_{seed}"] = {(0, "o_proj"): (A, B, scale)}
    return registry, f"adapter_{seed}"


def _run_single(model, tokenizer, question, max_new_tokens, lora_registry=None, lora_id=None, graph_decoder=None):
    cache = make_cache(model, num_blocks=64 + 40)
    scheduler = Scheduler(model, cache, graph_decoder=graph_decoder, lora_registry=lora_registry)
    prompt_ids = tokenize(tokenizer, question)
    scheduler.add_request(
        Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens, eos_token_id=None, lora_id=lora_id)
    )
    finished = scheduler.run_to_completion()
    return tokenizer.decode(finished[0].output_token_ids, skip_special_tokens=True)


def test_all_zero_lora_adapter_is_a_byte_identical_noop(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    registry = _zero_adapter(model)
    q = QUESTIONS[0]

    baseline = _run_single(model, tokenizer, q, max_new_tokens=12)
    with_zero_adapter = _run_single(model, tokenizer, q, max_new_tokens=12, lora_registry=registry, lora_id="zero_adapter")

    assert with_zero_adapter == baseline


def test_nonzero_lora_adapter_actually_changes_output(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    registry, adapter_id = _random_adapter(model, seed=1)
    q = QUESTIONS[0]

    baseline = _run_single(model, tokenizer, q, max_new_tokens=12)
    with_adapter = _run_single(model, tokenizer, q, max_new_tokens=12, lora_registry=registry, lora_id=adapter_id)

    assert with_adapter != baseline


def test_mixed_batch_different_adapters_and_none_do_not_cross_contaminate(model_and_tokenizer):
    """Three sequences sharing one scheduler step: two different adapters and
    one with none. Each must match exactly what it produces standalone --
    proving _decode_active's per-adapter grouping and continue_prefill's
    lora_id threading don't leak one sequence's delta onto another's rows.
    """
    model, tokenizer = model_and_tokenizer
    registry, adapter_a = _random_adapter(model, seed=2)
    registry_b, adapter_b = _random_adapter(model, seed=3)
    registry._adapters[adapter_b] = registry_b._adapters[adapter_b]

    standalone_none = _run_single(model, tokenizer, QUESTIONS[0], max_new_tokens=12)
    standalone_a = _run_single(model, tokenizer, QUESTIONS[1], max_new_tokens=12, lora_registry=registry, lora_id=adapter_a)
    standalone_b = _run_single(model, tokenizer, QUESTIONS[2], max_new_tokens=12, lora_registry=registry, lora_id=adapter_b)

    cache = make_cache(model, num_blocks=64 + 40)
    scheduler = Scheduler(model, cache, lora_registry=registry)
    lora_ids = [None, adapter_a, adapter_b]
    for i, (q, lid) in enumerate(zip(QUESTIONS, lora_ids)):
        prompt_ids = tokenize(tokenizer, q)
        scheduler.add_request(
            Sequence(seq_id=i, prompt_token_ids=prompt_ids, max_new_tokens=12, eos_token_id=None, lora_id=lid)
        )
    finished = {s.seq_id: s for s in scheduler.run_to_completion()}

    batched_none = tokenizer.decode(finished[0].output_token_ids, skip_special_tokens=True)
    batched_a = tokenizer.decode(finished[1].output_token_ids, skip_special_tokens=True)
    batched_b = tokenizer.decode(finished[2].output_token_ids, skip_special_tokens=True)

    assert batched_none == standalone_none
    assert batched_a == standalone_a
    assert batched_b == standalone_b


def test_cuda_graph_decoder_falls_back_to_eager_for_lora_sequences(model_and_tokenizer):
    """A sequence with lora_id set must decode correctly (matching its own
    eager-path, no-graph-decoder baseline) even when a graph_decoder is
    configured for the scheduler -- exercising the graph/eager split added to
    _decode_active. A concurrently running no-adapter sequence must still get
    the CUDA-graph-accelerated path and match its own baseline too.
    """
    model, tokenizer = model_and_tokenizer
    registry, adapter_id = _random_adapter(model, seed=4)

    eager_baseline_lora = _run_single(model, tokenizer, QUESTIONS[0], max_new_tokens=12, lora_registry=registry, lora_id=adapter_id)
    eager_baseline_none = _run_single(model, tokenizer, QUESTIONS[1], max_new_tokens=12)

    cache = make_cache(model, num_blocks=64 + 40)
    graph_decoder = CUDAGraphDecoder(model, cache, bucket_sizes=(1, 2, 4), max_blocks_per_seq=32)
    graph_decoder.capture()

    scheduler = Scheduler(model, cache, graph_decoder=graph_decoder, lora_registry=registry)
    scheduler.add_request(
        Sequence(seq_id=0, prompt_token_ids=tokenize(tokenizer, QUESTIONS[0]), max_new_tokens=12, eos_token_id=None, lora_id=adapter_id)
    )
    scheduler.add_request(
        Sequence(seq_id=1, prompt_token_ids=tokenize(tokenizer, QUESTIONS[1]), max_new_tokens=12, eos_token_id=None)
    )
    finished = {s.seq_id: s for s in scheduler.run_to_completion()}

    text_lora = tokenizer.decode(finished[0].output_token_ids, skip_special_tokens=True)
    text_none = tokenizer.decode(finished[1].output_token_ids, skip_special_tokens=True)

    assert text_lora == eager_baseline_lora
    assert text_none == eager_baseline_none
