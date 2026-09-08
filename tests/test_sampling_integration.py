import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.kv_cache import KVCacheManager
from wllm.engine.sampling import SamplingParams
from wllm.engine.scheduler import Scheduler
from wllm.engine.sequence import Sequence
from wllm.models.qwen2 import load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


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


def test_stop_token_ids_halts_generation_through_the_real_scheduler(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenize(tokenizer, "What is the capital of France? Answer in one sentence.")

    # Find the token id for "." (or a close single-token punctuation) and stop on it.
    period_id = tokenizer.encode(".", add_special_tokens=False)[-1]

    cache = make_cache(model)
    scheduler = Scheduler(model, cache)
    seq = Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_new_tokens=64, stop_token_ids=frozenset({period_id}))
    scheduler.add_request(seq)
    finished = scheduler.run_to_completion()

    assert len(finished) == 1
    assert finished[0].output_token_ids[-1] == period_id
    assert len(finished[0].output_token_ids) < 64, "should have stopped well before max_new_tokens"

    text = tokenizer.decode(finished[0].output_token_ids, skip_special_tokens=True)
    print(f"\nstopped output: {text!r}")
    assert "Paris" in text


def test_repetition_penalty_changes_real_generation_output(model_and_tokenizer):
    """Not a statistical test -- picks a prompt/setup where an unpenalized
    greedy-ish decode tends to repeat, and checks the penalty actually
    changes what gets generated (proving it's live in the real forward path,
    not just correct in isolation).
    """
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenize(tokenizer, "Say the word 'hello' five times in a row.")

    def run(sampling_params: SamplingParams) -> list[int]:
        cache = make_cache(model)
        scheduler = Scheduler(model, cache)
        seq = Sequence(seq_id=0, prompt_token_ids=prompt_ids, max_new_tokens=40, sampling_params=sampling_params)
        scheduler.add_request(seq)
        finished = scheduler.run_to_completion()
        return finished[0].output_token_ids

    unpenalized = run(SamplingParams(temperature=0.0, repetition_penalty=1.0))
    penalized = run(SamplingParams(temperature=0.0, repetition_penalty=1.3))

    print(f"\nunpenalized: {tokenizer.decode(unpenalized)!r}")
    print(f"penalized:   {tokenizer.decode(penalized)!r}")

    assert unpenalized != penalized, "repetition_penalty had no effect on real generation"
