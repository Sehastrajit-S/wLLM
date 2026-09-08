"""P1.2 correctness check: compares the native model runner against the
transformers baseline in fp32 (removes bf16 rounding drift as a variable) so
a real implementation bug can't hide behind "it's just precision".

Run: python scripts/check_native_model.py [model_id]
"""
import sys

sys.path.insert(0, "src")

import torch

from wllm.baseline.generate import capture_reference_logits
from wllm.baseline.model import DEFAULT_MODEL_ID
from wllm.baseline.model import load as load_baseline
from wllm.models.qwen2 import load_native

PROMPT = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_ID

    print(f"Loading HF baseline ({model_id}, fp32)...")
    hf_model, tokenizer = load_baseline(model_id, dtype=torch.float32)
    ref = capture_reference_logits(hf_model, tokenizer, PROMPT)
    del hf_model
    torch.cuda.empty_cache()

    print(f"Loading native model runner ({model_id}, fp32)...")
    native_model = load_native(model_id, dtype=torch.float32)

    with torch.inference_mode():
        logits = native_model(ref["input_ids"].to("cuda"))

    diff = (logits.float().cpu() - ref["logits"]).abs()
    mismatches = (logits.argmax(-1).cpu() != ref["logits"].argmax(-1)).sum().item()

    print(f"\nfp32 max abs diff:  {diff.max().item():.6f}")
    print(f"fp32 mean abs diff: {diff.mean().item():.8f}")
    print(f"argmax mismatches:  {mismatches}")

    if diff.max().item() > 1e-2 or mismatches > 0:
        print("\nFAIL: native model runner does not match HF baseline in fp32.")
        return 1

    print("\nSUCCESS: native model runner matches HF baseline in fp32.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
