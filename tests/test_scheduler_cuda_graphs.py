import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.cuda_graph_decoder import CUDAGraphDecoder
from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import Sequence
from wllm.models.qwen2 import load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
QUESTIONS = [
    "What is the capital of France? Answer in one sentence.",
    "What is the capital of Germany? Answer in one sentence.",
    "What is the capital of Japan? Answer in one sentence.",
]


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


def tokenize(tokenizer, question: str) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )["input_ids"][0].tolist()


@pytest.mark.parametrize("use_cuda_graphs", [False, True])
def test_scheduler_with_and_without_cuda_graphs_agree(use_cuda_graphs):
    """The whole point of CUDAGraphDecoder being an optimization layer: the
    Scheduler must produce identical output whether or not it's plugged in.
    Parametrized rather than a single side-by-side test so each run is
    exactly what a real deployment would do (not an artificial dual-cache
    rig) -- correctness is checked against the known-good eager text.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_native(MODEL_ID, dtype=torch.float32)
    max_new_tokens = 12

    graph_decoder = None
    if use_cuda_graphs:
        cache = make_cache(model, num_blocks=64 + 40)  # +headroom for scratch sequences
        graph_decoder = CUDAGraphDecoder(model, cache, bucket_sizes=(1, 2, 4), max_blocks_per_seq=32)
        graph_decoder.capture()
    else:
        cache = make_cache(model, num_blocks=64)

    scheduler = Scheduler(model, cache, graph_decoder=graph_decoder)
    for i, q in enumerate(QUESTIONS):
        prompt_ids = tokenize(tokenizer, q)
        scheduler.add_request(Sequence(seq_id=i, prompt_token_ids=prompt_ids, max_new_tokens=max_new_tokens))

    finished = scheduler.run_to_completion()
    finished_by_id = {s.seq_id: s for s in finished}

    expected_answers = ["Paris", "Berlin", "Tokyo"]
    for i, (q, expected) in enumerate(zip(QUESTIONS, expected_answers)):
        text = tokenizer.decode(finished_by_id[i].output_token_ids, skip_special_tokens=True)
        print(f"\n[cuda_graphs={use_cuda_graphs}] [{q}] -> {text!r}")
        assert expected in text
