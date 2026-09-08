import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.engine.guided_decoding import JSONSchemaGuide
from wllm.engine.sampling import SamplingParams, apply_token_mask, sample
from wllm.models.qwen2 import load_native

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def test_apply_token_mask_blocks_disallowed_tokens():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    mask = torch.tensor([[True, False, True, False]])
    out = apply_token_mask(logits, [mask[0]])
    assert out[0, 1].item() == float("-inf")
    assert out[0, 3].item() == float("-inf")
    assert out[0, 0].item() == 5.0
    assert out[0, 2].item() == 3.0


def test_apply_token_mask_none_row_is_unconstrained():
    logits = torch.randn(2, 10)
    out = apply_token_mask(logits, [None, torch.zeros(10, dtype=torch.bool)])
    assert torch.equal(out[0], logits[0])
    assert (out[1] == float("-inf")).all()


def test_sample_with_mask_never_selects_a_disallowed_token():
    torch.manual_seed(0)
    logits = torch.randn(4, 100)
    # only tokens 0-9 allowed
    mask = torch.zeros(100, dtype=torch.bool)
    mask[:10] = True
    masks = [mask] * 4
    params = [SamplingParams(temperature=1.0) for _ in range(4)]

    tokens, _ = sample(logits, params, generated_ids=[[] for _ in range(4)], token_masks=masks)
    assert (tokens < 10).all()


def test_json_schema_guide_produces_only_valid_continuations_end_to_end():
    """Real correctness bar: drive an actual model's greedy decode loop with
    the guide's mask applied every step, and verify the final text is valid
    JSON matching the schema -- not just "the mask looked right in isolation".
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_native(MODEL_ID, dtype=torch.float32)

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    guide = JSONSchemaGuide(MODEL_ID, schema, vocab_size=model.cfg.vocab_size)

    prompt = [{"role": "user", "content": "Generate a JSON object describing a person named Alice, age 30."}]
    inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")


    # Naive loop (correctness-first, matches how P1.2 first validated
    # generation) but with the guide's mask applied at each step manually,
    # since generate_greedy_naive itself knows nothing about guides.
    tokens = input_ids
    generated = []
    for _ in range(40):
        logits = model(tokens)
        step_logits = logits[:, -1, :]
        mask = guide.allowed_token_mask(device="cuda")
        step_logits = step_logits.masked_fill(~mask, float("-inf"))
        next_token = step_logits.argmax(-1)
        guide.advance(next_token.item())
        generated.append(next_token.item())
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        if guide.is_finished():
            break

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\nguided output: {text!r}")

    parsed = json.loads(text)
    assert set(parsed.keys()) == {"name", "age"}
    assert isinstance(parsed["name"], str)
    assert isinstance(parsed["age"], int)
