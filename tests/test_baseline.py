import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from wllm.baseline.generate import capture_reference_logits, generate_text
from wllm.baseline.model import DEFAULT_MODEL_ID, load

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REFERENCE_PROMPT = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]


@pytest.fixture(scope="module")
def model_and_tokenizer():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return load(DEFAULT_MODEL_ID)


def test_greedy_generation_is_correct(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text = generate_text(model, tokenizer, REFERENCE_PROMPT, max_new_tokens=32)
    assert "Paris" in text


def test_greedy_generation_is_deterministic(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    text_a = generate_text(model, tokenizer, REFERENCE_PROMPT, max_new_tokens=32)
    text_b = generate_text(model, tokenizer, REFERENCE_PROMPT, max_new_tokens=32)
    assert text_a == text_b


def test_capture_reference_logits(model_and_tokenizer):
    """Saves the ground-truth logits fixture that P1.2's PagedAttention kernel
    output will be diffed against for correctness.
    """
    model, tokenizer = model_and_tokenizer
    ref = capture_reference_logits(model, tokenizer, REFERENCE_PROMPT)

    assert ref["logits"].shape[0] == 1
    assert ref["logits"].shape[1] == ref["input_ids"].shape[1]
    assert torch.isfinite(ref["logits"]).all()

    FIXTURES_DIR.mkdir(exist_ok=True)
    torch.save(
        {"model_id": DEFAULT_MODEL_ID, **ref},
        FIXTURES_DIR / "reference_logits.pt",
    )
