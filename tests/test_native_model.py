import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer

from wllm.baseline.generate import capture_reference_logits
from wllm.baseline.model import DEFAULT_MODEL_ID
from wllm.baseline.model import load as load_baseline
from wllm.models.qwen2 import generate_greedy_naive, load_native

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REFERENCE_PROMPT = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]


@pytest.fixture(scope="module")
def reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    fixture_path = FIXTURES_DIR / "reference_logits.pt"
    if fixture_path.exists():
        return torch.load(fixture_path)

    model, tokenizer = load_baseline(DEFAULT_MODEL_ID)
    ref = capture_reference_logits(model, tokenizer, REFERENCE_PROMPT)
    del model
    torch.cuda.empty_cache()
    return {"model_id": DEFAULT_MODEL_ID, **ref}


def test_native_forward_matches_reference_logits(reference):
    tokenizer = AutoTokenizer.from_pretrained(reference["model_id"])
    native_model = load_native(reference["model_id"])

    with torch.inference_mode():
        logits = native_model(reference["input_ids"].to("cuda"))

    ref_logits = reference["logits"].to("cuda")
    diff = (logits.float() - ref_logits).abs()

    print(f"\nmax abs diff: {diff.max().item():.4f}")
    print(f"mean abs diff: {diff.mean().item():.6f}")

    # bf16 native forward pass vs bf16 HF forward pass: verified in fp32 (see
    # scripts/check_native_model.py) that this implementation matches HF to
    # ~3e-4 -- the larger bf16 gap here is accumulated rounding drift across
    # 24 layers from differing op-fusion order, not a bug. Loose bound just
    # catches gross regressions.
    assert diff.max().item() < 2.0

    ref_argmax = ref_logits.argmax(dim=-1)[0]
    native_argmax = logits.argmax(dim=-1)[0]
    seq_len = ref_argmax.shape[0]
    mismatch_pos = (ref_argmax != native_argmax).nonzero().flatten().tolist()

    # A real bug produces confidently-wrong predictions scattered everywhere;
    # bf16 rounding only flips an argmax when the top-2 logits were already a
    # near-tie. Verify every mismatch is such a near-tie (under the *reference's
    # own* scoring, the token the native run picked was almost as good as the
    # "correct" one), and cap how many of these can occur at all.
    near_tie_threshold = 0.75
    for pos in mismatch_pos:
        gap = (ref_logits[0, pos, ref_argmax[pos]] - ref_logits[0, pos, native_argmax[pos]]).abs().item()
        assert gap < near_tie_threshold, (
            f"position {pos}: argmax mismatch is NOT a near-tie (ref logit gap {gap:.3f}) -- likely a real bug"
        )

    assert len(mismatch_pos) / seq_len < 0.1, (
        f"{len(mismatch_pos)}/{seq_len} positions mismatched -- too many even for bf16 near-ties"
    )

    del tokenizer, native_model
    torch.cuda.empty_cache()


def test_native_model_generates_correct_text(reference):
    tokenizer = AutoTokenizer.from_pretrained(reference["model_id"])
    native_model = load_native(reference["model_id"])

    output_ids = generate_greedy_naive(native_model, reference["input_ids"].to("cuda"), max_new_tokens=16)
    prompt_len = reference["input_ids"].shape[1]
    text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

    print(f"\nnative model output: {text!r}")
    assert "Paris" in text

    del tokenizer, native_model
    torch.cuda.empty_cache()
