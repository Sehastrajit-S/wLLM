import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.guided_decoding import JSONSchemaGuide
from wllm.engine.kv_cache import KVCacheManager
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


def test_scheduler_produces_valid_json_for_guided_sequence(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
    }
    guide = JSONSchemaGuide(MODEL_ID, schema, vocab_size=model.cfg.vocab_size)

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Generate a JSON object for the city of Paris, France."}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"][0].tolist()

    cache = make_cache(model)
    scheduler = Scheduler(model, cache)
    seq = Sequence(seq_id=0, prompt_token_ids=prompt, max_new_tokens=40, guide=guide)
    scheduler.add_request(seq)
    scheduler.run_to_completion()

    text = tokenizer.decode(seq.output_token_ids, skip_special_tokens=True)
    print(f"\nscheduler guided output: {text!r}")
    parsed = json.loads(text)
    assert set(parsed.keys()) == {"city", "country"}


def test_guided_and_unguided_sequences_coexist_in_the_same_batch(model_and_tokenizer):
    """Per-row masking must not bleed between sequences batched together."""
    model, tokenizer = model_and_tokenizer
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    guide = JSONSchemaGuide(MODEL_ID, schema, vocab_size=model.cfg.vocab_size)

    guided_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Generate a JSON object with an 'answer' field containing the word hello."}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"][0].tolist()
    plain_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"][0].tolist()

    cache = make_cache(model)
    scheduler = Scheduler(model, cache)
    guided_seq = Sequence(seq_id=0, prompt_token_ids=guided_prompt, max_new_tokens=30, guide=guide)
    plain_seq = Sequence(seq_id=1, prompt_token_ids=plain_prompt, max_new_tokens=16)
    scheduler.add_request(guided_seq)
    scheduler.add_request(plain_seq)
    scheduler.run_to_completion()

    guided_text = tokenizer.decode(guided_seq.output_token_ids, skip_special_tokens=True)
    plain_text = tokenizer.decode(plain_seq.output_token_ids, skip_special_tokens=True)
    print(f"\nguided: {guided_text!r}\nplain:  {plain_text!r}")

    parsed = json.loads(guided_text)
    assert "answer" in parsed
    assert "Paris" in plain_text
